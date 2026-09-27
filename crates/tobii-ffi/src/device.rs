//! The API and device handles: a device is a connection to the `tobiid`
//! daemon with a reader thread sorting decoded messages into two channels
//! (the answers to requests and subscription changes, and the samples) and
//! ringing the device's doorbell for `wait` after each sample, the
//! registered callbacks, and a synchronous request/reply helper. Threads may
//! share a device: its state is split by concern, each part behind a lock of
//! its own (see [`Device`]).
//!
//! Invariant: every stored callback was registered through the matching
//! `tobii_*_subscribe` entry point, whose safety contract makes it sound to
//! invoke with the stored `user_data`, on any thread that processes the
//! device, until it is unsubscribed or the device is destroyed.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt;
use std::io;
use std::mem;
use std::net::Shutdown;
use std::ops::{Deref, DerefMut};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tobii_ipc::request::{DeviceInfo as DeviceInfoMsg, decode_device_info, encode_request, kind};
use tobii_ipc::{
    self, NotificationValue as WireValue, STREAM_EYE_POSITION, STREAM_GAZE, STREAM_GAZE_DATA,
    STREAM_GAZE_ORIGIN, STREAM_GAZE_RAW, STREAM_HEAD, STREAM_IMAGE, STREAM_NOTIFICATIONS,
    STREAM_PRESENCE, ServerMsg, decode_server, encode_subscribe, read_frame, write_frame,
};

use crate::logger::{self, Level, Logger, SharedLogger};
use crate::status::{
    Status, TOBII_ERROR_ALREADY_SUBSCRIBED, TOBII_ERROR_CALLBACK_IN_PROGRESS,
    TOBII_ERROR_CONFLICTING_API_INSTANCES, TOBII_ERROR_CONNECTION_FAILED, TOBII_ERROR_INTERNAL,
    TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR, TOBII_ERROR_NOT_SUBSCRIBED,
    TOBII_ERROR_TIMED_OUT,
};
use crate::timeouts;
use crate::types::{
    DisplayArea, EyePair, EyePairFn, FieldOfUse, FieldOfUseFn, GazeData, GazeDataEye, GazeDataFn,
    GazePoint, GazePointFn, GazeRaw, GazeRawEye, GazeRawFn, HeadPose, HeadPoseFn, Image, ImageFn,
    Notification, NotificationValue, NotificationsFn, PresenceFn, PresenceStatus,
    TOBII_VALIDITY_INVALID, TOBII_VALIDITY_VALID, Validity, copy_c_string,
};

/// How long a subscription change waits for the daemon's acknowledgement.
const SUBSCRIBE_ACK_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a reconnect waits for the daemon to acknowledge the restored
/// subscriptions. A running daemon's reader for the connection queues the
/// ack once it has scanned the USB bus (when no engine runs) and taken the
/// state lock, and the pump writes it on its next 8 ms tick. The lock is
/// the slow part: it is held while an engine is dropped (its thread
/// joined) and while the pump writes to clients, so the ack usually comes
/// well within this, but not always. One that is not running yet (systemd
/// holds the socket while the service restarts) is left to the caller's
/// next attempt, rather than holding a call an application may make from
/// its frame loop.
///
/// An attempt that gives up closes a connection that already carries its
/// subscription, and the daemon still reads it (under socket activation,
/// once it starts and accepts it from systemd's backlog): it starts the
/// engine for it, then sees the hang-up and drops the engine again unless
/// another client wants it, under the state lock a later attempt's ack
/// waits on. Skipping a subscription whose peer has hung up is the
/// daemon's to do.
const RECONNECT_ACK_TIMEOUT: Duration = Duration::from_millis(500);

thread_local! {
    /// Set while this thread runs application code: a user callback, the
    /// application's logger or `tobii_calibration_retrieve`'s receiver. The
    /// entry points the crate documentation lists refuse a call made from
    /// inside one, as the Stream Engine does for a callback, before they
    /// take any lock: a callback runs under its device's dispatch and
    /// callbacks locks, which are not reentrant, so a call into that device
    /// would deadlock this thread, and destroying it would free it under the
    /// dispatch loop. A call into any other device is refused too, so that
    /// no thread holds two devices' locks: two callbacks each calling into
    /// the other's device would deadlock. The receiver runs with no lock
    /// held, and is guarded as the DLL guards it (see
    /// `tobii_calibration_retrieve`). The flag is this thread's own; other
    /// threads' calls go on.
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is inside a user callback, the logger or
/// `tobii_calibration_retrieve`'s receiver.
pub(crate) fn in_callback() -> bool {
    IN_CALLBACK.get()
}

/// Opaque API handle: the application's logger, if it gave one. Each device
/// created from the handle copies it.
#[derive(Debug)]
pub struct Api {
    pub(crate) logger: Option<Logger>,
}

impl Api {
    pub(crate) fn new(logger: Option<Logger>) -> Self {
        Self { logger }
    }
}

/// Opens a connection to the daemon (a socket pair in tests).
pub(crate) type Connector = Box<dyn FnMut() -> io::Result<UnixStream> + Send>;

/// A connector that makes its first connection with `first` and every later
/// one with `later`: a device's first connection is the one
/// `tobii_device_create` makes, and each later one a reconnect.
fn first_then(
    first: impl FnOnce() -> io::Result<UnixStream> + Send + 'static,
    mut later: impl FnMut() -> io::Result<UnixStream> + Send + 'static,
) -> Connector {
    let mut first = Some(first);
    Box::new(move || match first.take() {
        Some(connect) => connect(),
        None => later(),
    })
}

/// A device's first connection: `connect`, and when nothing listens,
/// `connect_or_spawn` (which connects again before it spawns tobiid) with
/// `spawning` held, so that threads creating devices at once while no
/// daemon runs start one between them. Two would both run: the second binds
/// the socket path over the first's, which keeps the clients it has, and
/// the clients are split between them. A thread that finds another's
/// attempt under way waits for it and then only connects, to the daemon
/// that attempt started or failing as it did, rather than making an attempt
/// of its own after it: in turn, the last of them would wait out all the
/// others' (some 3 s each when tobiid cannot be started). A daemon that
/// listens is reached without the lock, so devices created while one runs
/// never wait for each other. Other processes' clients are not covered: two
/// applications that find no daemon at once can still start one each.
fn connect_or_spawn_alone(
    spawning: &Mutex<()>,
    connect: impl Fn() -> io::Result<UnixStream>,
    connect_or_spawn: impl FnOnce() -> io::Result<UnixStream>,
) -> io::Result<UnixStream> {
    if let Ok(stream) = connect() {
        return Ok(stream);
    }
    if let Some(_spawning) = try_lock(spawning) {
        return connect_or_spawn();
    }
    // Only the wait matters: holding the lock across this connect would
    // hold up the other waiters' connects, and a thread arriving later
    // would wait for it as if for an attempt.
    drop(lock(spawning));
    connect()
}

/// Whether a daemon connection is up, and whether its loss has been reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinkState {
    /// Nothing has seen the connection end.
    Up,
    /// The connection has ended and is closed; `process` has not said so yet.
    Lost,
    /// `process` has returned `TOBII_ERROR_CONNECTION_FAILED` for the loss.
    Reported,
}

/// What tobiid answers a request or a subscription change with: the
/// messages a call waits for, as against the samples `process` delivers.
#[derive(Debug)]
enum Answer {
    /// A subscription change's ack: whether the daemon took the streams.
    Subscribed(bool),
    /// A request's reply.
    Reply {
        /// The id the request carried.
        request_id: u32,
        /// A Stream Engine `tobii_error_t` value; `0` is success.
        status: u8,
        /// Kind-specific payload.
        payload: Vec<u8>,
    },
}

impl TryFrom<ServerMsg> for Answer {
    /// Anything else: a sample, or a message kind a newer daemon may add.
    type Error = ServerMsg;

    fn try_from(msg: ServerMsg) -> Result<Self, ServerMsg> {
        match msg {
            ServerMsg::Subscribed { ok } => Ok(Self::Subscribed(ok)),
            ServerMsg::Reply {
                request_id,
                status,
                payload,
            } => Ok(Self::Reply {
                request_id,
                status,
                payload,
            }),
            other => Err(other),
        }
    }
}

/// Lock `m`, taking a poisoned lock's data as it is. A lock is poisoned
/// only by a panic while it is held. On an application's thread that never
/// goes on to use the data: the panic cannot unwind out of an `extern "C"`
/// entry point, which aborts the process instead, so no thread takes the
/// lock again. A lock another thread takes (the reader's, which takes the
/// doorbell's alone) must keep its data whole through a panic itself (see
/// [`Doorbell`]).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Lock `m` if no other thread holds it, taking a poisoned lock's data as
/// [`lock`] does; `None` while another thread holds it.
fn try_lock<T>(m: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match m.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

/// A lock that lets its waiters in in the order they asked, first come,
/// first served: each takes a ticket and waits for its turn. It is the
/// command lock (see [`Device`]), for which a std `Mutex` will not do: it
/// promises no order, and on Linux a thread that lets it go and asks again
/// at once, as one making requests back to back does, takes it back ahead
/// of a thread already waiting, time after time (on hardware, subscribes and
/// unsubscribes on another thread waited up to 5 s so). The data sits
/// behind a `Mutex` of its own, which only the thread whose turn it is
/// takes, so no thread waits for it. A poisoned lock is taken as it is (see
/// [`lock`]): the tickets' lock is held only to count, which cannot panic,
/// and a holder that panics passes its turn on as it unwinds.
struct TicketLock<T> {
    tickets: Tickets,
    data: Mutex<T>,
}

impl<T> TicketLock<T> {
    fn new(data: T) -> Self {
        Self {
            tickets: Tickets::default(),
            data: Mutex::new(data),
        }
    }

    /// Lock it, once every thread that asked before this one has had it and
    /// let it go.
    fn lock(&self) -> TicketGuard<'_, T> {
        let turn = self.tickets.wait_turn();
        TicketGuard {
            data: lock(&self.data),
            _turn: turn,
        }
    }
}

/// A [`TicketLock`]'s queue.
#[derive(Debug, Default)]
struct Tickets {
    /// The next ticket to hand out, and the one whose turn it is. They
    /// cannot wrap in practice, and would stay in step if they did.
    counts: Mutex<(u64, u64)>,
    /// Notified each time a turn is passed on.
    passed: Condvar,
}

impl Tickets {
    /// Take the next ticket and wait for its turn.
    fn wait_turn(&self) -> Turn<'_> {
        let mut counts = lock(&self.counts);
        let ticket = counts.0;
        counts.0 = ticket.wrapping_add(1);
        drop(
            self.passed
                .wait_while(counts, |counts| counts.1 != ticket)
                .unwrap_or_else(PoisonError::into_inner),
        );
        Turn(self)
    }
}

/// The turn of a ticket; dropping it passes the turn on to the next.
struct Turn<'a>(&'a Tickets);

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let mut counts = lock(&self.0.counts);
        counts.1 = counts.1.wrapping_add(1);
        drop(counts);
        self.0.passed.notify_all();
    }
}

/// A locked [`TicketLock`]: the data, and the turn.
#[must_use = "if unused the lock is let go at once"]
struct TicketGuard<'a, T> {
    // Declared first, so dropped first: the data's lock is let go before the
    // turn is passed on.
    data: MutexGuard<'a, T>,
    _turn: Turn<'a>,
}

impl<T> Deref for TicketGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T> DerefMut for TicketGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

/// What `wait` sleeps on between its looks in the samples channel, rather
/// than blocking on the channel itself: the number of times it has rung,
/// and a condvar to sleep on until that moves. The reader rings it after
/// each sample it queues and once when it stops, having hung up its
/// channels first; an answer rings nothing, as it is nothing to process.
/// A reconnect rings it once it has put its new link in place, and a holder
/// of the dispatch lock may ring it once it lets the lock go (see
/// [`Doorbell::ring_after_hold`]). The reader holds the lock only to bump
/// the count, which cannot panic, so the count is whole even if the lock is
/// poisoned. No other lock is taken while it is held.
///
/// One per device, kept across reconnects and rung by every link's reader,
/// a failed reconnect's included: while a reconnect runs, the old link's
/// reader and the new one's both ring it. The old one rings as the old
/// link goes, which is inside the reconnect's hold of the dispatch lock,
/// so a wait it wakes finds the lock taken and sleeps again, and the new
/// one's first rings may come during the hold too. What wakes a wait
/// sleeping through a reconnect on another thread, on a lost link or a
/// live one, for the samples of the link put in its place is the ring the
/// reconnect makes once that link is in place.
#[derive(Debug, Default)]
struct Doorbell {
    rings: Mutex<u64>,
    rung: Condvar,
}

impl Doorbell {
    /// Count a ring and wake every waiter.
    fn ring(&self) {
        let mut rings = lock(&self.rings);
        *rings = rings.wrapping_add(1);
        drop(rings);
        self.rung.notify_all();
    }

    /// How many times it has rung.
    fn rings(&self) -> u64 {
        *lock(&self.rings)
    }

    /// Look with `look` until it finds something, sleeping between looks
    /// until a ring, for up to `timeout` in all; whether it found something.
    /// The count is read before each look, so a ring that comes after a
    /// look, however soon, ends the sleep that follows it. A ring that
    /// brings nothing (for a sample an earlier look took already) sends it
    /// back to sleep until the same deadline.
    fn wait_for(&self, timeout: Duration, mut look: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            let seen = self.rings();
            if look() {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !self.wait_past(seen, left) {
                return false;
            }
        }
    }

    /// Wait up to `timeout` for it to ring past `seen`, a count read
    /// earlier; whether it has. A ring since `seen` answers at once.
    fn wait_past(&self, seen: u64, timeout: Duration) -> bool {
        let rings = lock(&self.rings);
        let (rings, _) = self
            .rung
            .wait_timeout_while(rings, timeout, |rings| *rings == seen)
            .unwrap_or_else(PoisonError::into_inner);
        *rings != seen
    }

    /// Ring once the dispatch lock is let go, for a wait whose look came
    /// while it was held: such a look finds nothing, and the wait sleeps
    /// until the count moves past what it read before it looked. It is
    /// woken if the holder leaves something to process (`left`), or if the
    /// doorbell rang during the hold, past `seen`, a count read once the
    /// lock was taken: the wait may have read the count after that ring,
    /// and the sample it rang for came after the holder's last look in the
    /// channel, so it is still queued. Otherwise nothing: a wait turned away
    /// from a device with nothing to process sleeps on, as it would have.
    fn ring_after_hold(&self, seen: u64, left: bool) {
        let mut rings = lock(&self.rings);
        if *rings == seen && !left {
            return;
        }
        *rings = rings.wrapping_add(1);
        drop(rings);
        self.rung.notify_all();
    }
}

/// One daemon connection and the thread reading it, which sorts what the
/// daemon sends into two channels, each in the order it came: the answers,
/// read only by a request or subscription change waiting for its own, under
/// the command lock with the rest of the link, and the samples, whose
/// receiver goes to the dispatch lock's [`Dispatch`], read only by `wait`,
/// `process` and `clear_buffers`. A call waiting for its answer leaves the
/// samples where they are, and an answer nobody waits for any more never
/// wakes `wait`. The reader rings the device's doorbell after each sample,
/// and when it stops shuts the connection down, hangs up both channels and
/// rings once more (see [`ReaderEnd`]): a loss found under the command lock
/// reaches `process` that way.
struct Link {
    stream: UnixStream,
    answers: Receiver<Answer>,
    reader: Option<JoinHandle<()>>,
    /// The acks still to come for subscription changes that gave up waiting.
    /// tobiid acks every change on a connection, in order, so the next this
    /// many acks are theirs, and a later change reads past them to its own
    /// rather than take a late one for it. A reply needs no count: it names
    /// its request. Kept with the link, so a new one owes none.
    acks_owed: u32,
}

impl Link {
    /// Connect through `connect` and start the reader, which rings `bell`.
    /// The samples channel's receiver comes back beside the link.
    fn open(
        connect: &mut Connector,
        bell: &Arc<Doorbell>,
    ) -> io::Result<(Self, Receiver<ServerMsg>)> {
        let stream = connect()?;
        let (answers_tx, answers) = mpsc::channel();
        let (samples_tx, samples) = mpsc::channel();
        let end = ReaderEnd {
            stream: stream.try_clone()?,
            channels: Some((answers_tx, samples_tx)),
            bell: Arc::clone(bell),
        };
        let reader = thread::Builder::new()
            .name("tobii-ffi-reader".into())
            .spawn(move || end.run())?;
        let link = Self {
            stream,
            answers,
            reader: Some(reader),
            acks_owed: 0,
        };
        Ok((link, samples))
    }

    /// Close the connection and join the reader. Everything it read is then
    /// in the channels, each followed by a disconnect, so the next `process`
    /// or `wait` finds the loss whether or not the reader had seen one.
    fn close(&mut self) {
        // Shutting down the socket makes the reader's blocking read return, so
        // the join below cannot hang; a shutdown error only means it is already
        // closed.
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(h) = self.reader.take()
            && h.join().is_err()
        {
            tracing::warn!("ffi reader thread panicked");
        }
    }

    /// Write one frame to the daemon. A failed write loses the connection
    /// (the daemon has closed it, or the frame is cut short mid-stream) and
    /// closes it. A body too long for a frame fails before a byte is
    /// written, and leaves the connection as it was.
    fn send(&mut self, body: &[u8]) -> Result<(), Status> {
        if u32::try_from(body.len()).is_err() {
            tracing::debug!(len = body.len(), "frame body too long for tobiid");
            return Err(TOBII_ERROR_CONNECTION_FAILED);
        }
        write_frame(&mut self.stream, body).map_err(|e| {
            tracing::debug!(error = %e, "could not write to tobiid");
            self.close();
            TOBII_ERROR_CONNECTION_FAILED
        })
    }

    /// Wait up to `timeout` for the daemon's next answer. The samples that
    /// arrive meanwhile stay queued for `process`. A hung-up channel closes
    /// the connection.
    fn recv_answer(&mut self, timeout: Duration) -> Result<Answer, Status> {
        match self.answers.recv_timeout(timeout) {
            Ok(answer) => Ok(answer),
            Err(RecvTimeoutError::Timeout) => Err(TOBII_ERROR_TIMED_OUT),
            Err(RecvTimeoutError::Disconnected) => {
                self.close();
                Err(TOBII_ERROR_CONNECTION_FAILED)
            }
        }
    }

    /// Drop an ack that no subscription change waits for: the late one of a
    /// change that gave up, which is then owed no more.
    fn late_ack(&mut self) {
        if let Some(owed) = self.acks_owed.checked_sub(1) {
            self.acks_owed = owed;
            tracing::debug!(owed, "late subscription ack dropped");
        } else {
            tracing::debug!("subscription ack nobody asked for dropped");
        }
    }

    /// Subscribe to the streams in `mask` and wait up to `timeout` for the
    /// daemon's ack, reading past the late ones still owed (see
    /// [`Link::acks_owed`]). Returns the ack's `ok` flag. Giving up leaves
    /// its ack owed.
    fn send_subscription(&mut self, mask: u32, timeout: Duration) -> Result<bool, Status> {
        self.send(&encode_subscribe(mask))?;
        let deadline = Instant::now() + timeout;
        loop {
            match self.recv_answer(deadline.saturating_duration_since(Instant::now())) {
                Ok(Answer::Subscribed(ok)) if self.acks_owed == 0 => return Ok(ok),
                Ok(Answer::Subscribed(_)) => self.late_ack(),
                Ok(Answer::Reply { request_id, .. }) => {
                    tracing::debug!(request_id, "stale reply dropped");
                }
                Err(TOBII_ERROR_TIMED_OUT) => {
                    self.acks_owed = self.acks_owed.saturating_add(1);
                    return Err(TOBII_ERROR_TIMED_OUT);
                }
                Err(status) => return Err(status),
            }
        }
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        self.close();
    }
}

/// What the reader thread owns: its clone of the connection, the senders of
/// both channels, and the device's doorbell. However the reader stops (the
/// daemon hangs up, a read fails, the `Link` goes away, or a panic unwinds),
/// dropping this shuts the connection down, hangs up both channels and only
/// then rings, so the wait that ring wakes finds the loss.
struct ReaderEnd {
    stream: UnixStream,
    /// `Some` until dropped, when taking them hangs up before the ring.
    channels: Option<(Sender<Answer>, Sender<ServerMsg>)>,
    bell: Arc<Doorbell>,
}

impl ReaderEnd {
    /// Pump frames from the daemon until EOF, a read error, or the receiving
    /// `Link` going away: answers into the answers channel, and everything
    /// else (the samples, and message kinds a newer daemon may add) into the
    /// samples channel, ringing the doorbell after each of those.
    fn run(mut self) {
        let Some((answers, samples)) = &self.channels else {
            return;
        };
        while let Ok(Some(body)) = read_frame(&mut self.stream) {
            let Some(msg) = decode_server(&body) else {
                continue;
            };
            let sent = match Answer::try_from(msg) {
                Ok(answer) => answers.send(answer).is_ok(),
                Err(sample) => {
                    let sent = samples.send(sample).is_ok();
                    if sent {
                        self.bell.ring();
                    }
                    sent
                }
            };
            if !sent {
                break;
            }
        }
    }
}

impl Drop for ReaderEnd {
    fn drop(&mut self) {
        // The connection closes however its loss is found, not only once a
        // `process` or `wait` notices it; a shutdown error only means it is
        // closed already.
        let _ = self.stream.shutdown(Shutdown::Both);
        // Hang up before ringing: a wait woken by a ring made with the
        // channels still up would find the samples channel empty rather
        // than disconnected, and sleep again with nothing left to ring.
        self.channels = None;
        self.bell.ring();
    }
}

/// What the command lock guards: the connection's write half and answers,
/// how to open another, the request ids, and the identity fetched over the
/// connection. A request, a subscription change, a recenter and a reconnect
/// each hold it for their whole round trip, so they run one at a time on a
/// device, in the order they asked (see [`TicketLock`]).
struct Command {
    link: Link,
    connect: Connector,
    next_request_id: u32,
    /// Identity, fetched once per connection: a reconnect clears it, since a
    /// restarted daemon may serve another tracker.
    device_info: Option<DeviceInfoMsg>,
}

impl Command {
    /// Send a request and wait up to `timeout` for its reply (see
    /// [`Self::reply`]). A reply with a non-zero status is that status.
    fn request(&mut self, kind: u8, payload: &[u8], timeout: Duration) -> Result<Vec<u8>, Status> {
        match self.reply(kind, payload, timeout)? {
            (TOBII_ERROR_NO_ERROR, payload) => Ok(payload),
            (status, _) => Err(status),
        }
    }

    /// Send a request and wait up to `timeout` for its reply, dropping the
    /// stale replies of requests that gave up, and the late acks it reads
    /// past, which are then owed no more (see [`Link::acks_owed`]; no
    /// subscription change waits at the same time, as both hold the command
    /// lock). The samples that arrive meanwhile stay queued for `process`.
    /// The reply's status and payload, whatever the status; `Err` only when
    /// no reply comes.
    fn reply(
        &mut self,
        kind: u8,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<(Status, Vec<u8>), Status> {
        #[cfg(test)]
        tests::note_request(timeout);
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        let id = self.next_request_id;
        self.link.send(&encode_request(id, kind, payload))?;
        let deadline = Instant::now() + timeout;
        loop {
            match self
                .link
                .recv_answer(deadline.saturating_duration_since(Instant::now()))?
            {
                Answer::Reply {
                    request_id,
                    status,
                    payload,
                } if request_id == id => return Ok((Status::from(status), payload)),
                Answer::Reply { request_id, .. } => {
                    tracing::debug!(request_id, "stale reply dropped");
                }
                Answer::Subscribed(_) => self.link.late_ack(),
            }
        }
    }
}

/// What the dispatch lock guards: the samples channel, the samples a look
/// of `wait` took from it ahead of `process` (and, after a reconnect, the
/// old link's notifications), and whether the connection is up. A reconnect
/// swaps it for its new link's (see [`Dispatch::swap_samples`]).
struct Dispatch {
    samples: Receiver<ServerMsg>,
    pending: VecDeque<ServerMsg>,
    /// Kept with the samples, so a new link (a reconnect) starts `Up` and a
    /// later loss is reported again.
    state: LinkState,
}

impl Dispatch {
    fn new(samples: Receiver<ServerMsg>) -> Self {
        Self {
            samples,
            pending: VecDeque::new(),
            state: LinkState::Up,
        }
    }

    /// Move the next queued sample into `pending`; whether there was one.
    /// The channel empty and hung up is the loss of the connection: the
    /// reader has stopped, having read everything, or a call under the
    /// command lock has closed the link.
    fn take_one(&mut self) -> bool {
        if self.state != LinkState::Up {
            return false;
        }
        match self.samples.try_recv() {
            Ok(msg) => {
                self.pending.push_back(msg);
                true
            }
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => {
                self.state = LinkState::Lost;
                false
            }
        }
    }

    /// Move every queued sample into `pending`.
    fn take_all(&mut self) {
        while self.take_one() {}
    }

    /// Take `samples`, a new link's channel, in place of the old link's,
    /// and start up. Of what the old link brought and nothing delivered
    /// (what `pending` holds, then what its channel still does), the
    /// notifications stay queued, in the order they came, ahead of what the
    /// new link brings; the samples are dropped.
    ///
    /// tobiid writes each tick to every connection subscribed to its
    /// streams, so from the tick that acks the new link on, the old one
    /// brings the samples the new one does, and delivering both would
    /// repeat samples, their stamps stepping back; those tobiid sent the old
    /// link alone before go with them. A notification cannot go: tobiid
    /// sends one to the connections subscribed when it comes, and a new
    /// subscriber is not told the state again, so the application would
    /// hold a stale state until the next change. Kept, those tobiid sent
    /// both links (from the tick that acks the new one until the old one is
    /// closed) come twice, the old link's copies ahead of all the new
    /// link's: a copy may come again after later ones, so a state may be
    /// seen to step back before it settles, and the last delivered is
    /// current. Telling the copies apart would take a stamp that
    /// notifications do not carry.
    ///
    /// Only what the old channel holds now is taken, so the old link's
    /// reader must have been joined first, as the reconnect does by closing
    /// the old link: then the reader has put in everything it was sent, and
    /// hung up. A debug build checks that the channel is hung up.
    fn swap_samples(&mut self, samples: Receiver<ServerMsg>) {
        let old = mem::replace(&mut self.samples, samples);
        let notification = |msg: &ServerMsg| matches!(msg, ServerMsg::Notification(_));
        self.pending.retain(notification);
        self.pending.extend(old.try_iter().filter(notification));
        debug_assert!(
            matches!(old.try_recv(), Err(TryRecvError::Disconnected)),
            "the old link's reader must be joined before the swap"
        );
        self.state = LinkState::Up;
    }

    /// Whether `process` has something to do among what has been taken: a
    /// sample, or a loss it has not reported yet.
    fn anything_to_process(&self) -> bool {
        !self.pending.is_empty() || self.state == LinkState::Lost
    }
}

/// A registered callback and its user data.
pub(crate) type Slot<F> = Option<(F, *mut c_void)>;

/// Every callback a device can have registered.
#[derive(Default)]
pub(crate) struct Callbacks {
    pub(crate) head: Slot<HeadPoseFn>,
    pub(crate) gaze: Slot<GazePointFn>,
    pub(crate) presence: Slot<PresenceFn>,
    pub(crate) gaze_origin: Slot<EyePairFn>,
    pub(crate) eye_position: Slot<EyePairFn>,
    pub(crate) user_position_guide: Slot<EyePairFn>,
    pub(crate) gaze_data: Slot<GazeDataFn>,
    pub(crate) gaze_raw: Slot<GazeRawFn>,
    pub(crate) image: Slot<ImageFn>,
    pub(crate) notifications: Slot<NotificationsFn>,
    /// Registered but never called: the field of use cannot change.
    pub(crate) field_of_use: Slot<FieldOfUseFn>,
}

// SAFETY: the `user_data` pointers are the application's, and libtobii never
// reads through them: it hands each to its callback. The subscribe contract
// (see `streams`) makes each callback sound to invoke with its `user_data` on
// any thread that calls `tobii_device_process_callbacks` on the device, so the
// table may move to, and be used on, whichever thread holds its lock; the
// device keeps it behind one (`Device::callbacks`), which runs the callbacks
// one at a time.
unsafe impl Send for Callbacks {}

impl Callbacks {
    /// The daemon streams these callbacks need; the single source of truth
    /// for the subscription mask.
    pub(crate) fn mask(&self) -> u32 {
        let mut mask = 0;
        let mut need = |on: bool, bit: u32| {
            if on {
                mask |= bit;
            }
        };
        need(self.head.is_some(), STREAM_HEAD);
        need(self.gaze.is_some(), STREAM_GAZE);
        need(self.presence.is_some(), STREAM_PRESENCE);
        need(self.gaze_origin.is_some(), STREAM_GAZE_ORIGIN);
        need(
            self.eye_position.is_some() || self.user_position_guide.is_some(),
            STREAM_EYE_POSITION,
        );
        need(self.gaze_data.is_some(), STREAM_GAZE_DATA);
        need(self.gaze_raw.is_some(), STREAM_GAZE_RAW);
        need(self.image.is_some(), STREAM_IMAGE);
        need(self.notifications.is_some(), STREAM_NOTIFICATIONS);
        mask
    }
}

/// Why a reconnect failed, as its ERROR line says.
#[derive(Debug)]
enum ReconnectError {
    /// Nothing listens, or the connection could not be set up.
    Connect(io::Error),
    /// The daemon did not take back the streams in `mask`: `why` says how.
    Subscribe { mask: u32, why: &'static str },
}

impl fmt::Display for ReconnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect(e) => write!(f, "{e}"),
            Self::Subscribe { mask, why } => write!(
                f,
                "asked for the subscriptions back (streams {mask:#x}), it {why}"
            ),
        }
    }
}

/// Opaque device handle, which threads may share, as the Stream Engine
/// promises. Its state is split by concern, each part behind a lock of its
/// own, as the DLL splits a device's among critical sections of its own:
///
/// - `command` (`Command`; the DLL's API mutex, dev+0x4e0, which its
///   requests hold for the whole tracker round trip: device info at
///   0x180142ee1, subscribe 0x18015ce3d..0x18015ceaf, reconnect
///   0x180143873..0x180143a0f): held for a request's, subscription
///   change's, recenter's or reconnect's whole round trip. It is a
///   `TicketLock`: its waiters take it in the order they asked, so a call
///   waits for the round trip under way and those asked for before it, and
///   no more, however soon the thread ahead asks again. A critical section
///   promises no order among its waiters (Windows semantics, not read from
///   the DLL), nor does a std `Mutex`, which on Linux kept subscribes out
///   for up to 5 s behind requests made back to back.
/// - `dispatch` (`Dispatch`; its platform module's process mutex, +0x4628,
///   which process only try-enters, 0x18000e9d9, and the notification
///   queue's dev+0x9818): held by one `process` while it drains the samples
///   and runs the callbacks, by `clear_buffers`, by each look of a `wait`,
///   and by a reconnect from before its new link asks for the streams until
///   it has swapped the samples channel (see `Device::reconnect`).
/// - `callbacks` (`Callbacks`; dev+0x4d8, held around each user callback,
///   0x1801546a0..0x180155091): held around each callback, and while a
///   subscription change reads or sets a slot.
///
/// Lock order: `command`, then `dispatch`, then `callbacks`; the doorbell's
/// lock is a leaf, and the reader thread takes no other. A thread holds one
/// device's locks at a time: `tobii_wait_for_callbacks` takes its devices
/// one after another, and a callback, which runs under its device's
/// `dispatch` and `callbacks`, may call into no device (`in_callback`
/// refuses it before any lock is taken, so these locks, which are not
/// reentrant, never deadlock on their own thread). Nothing is held while a
/// `wait` sleeps or `tobii_calibration_retrieve`'s receiver runs, nor while
/// the logger runs, but for a line a call made from inside a callback logs
/// (a refused `field_of_use`), which runs under that callback's locks. A
/// subscribe lets `callbacks` go for its round trip, where the DLL holds
/// dev+0x4d8 (0x18015371b..0x180153845). A poisoned lock is taken as it is
/// (see `lock` and `TicketLock`).
///
/// What can still deadlock is the Stream Engine's own: a callback that
/// blocks on another thread's call into any device (or on a thread that
/// waits for one). Of the calls that take a lock, only `process` and `wait`
/// return promptly while a callback runs. A subscribe, an unsubscribe, a
/// clear or a reconnect waits for it, on `callbacks` or `dispatch`; a
/// request or a recenter may wait, on `command`, behind a subscription
/// change or reconnect under way or queued ahead of it, and a clear, on
/// `dispatch`, behind a reconnect's round trip. Across devices the same makes a cycle: X's
/// callback waits for an unsubscribe of Y, which waits for Y's callback,
/// which waits for an unsubscribe of X, which waits for X's.
///
/// Destroying the device takes no lock, as in the DLL: no other thread may
/// be inside a call on it, or use it afterwards.
pub struct Device {
    /// Address of the API handle this device was created from.
    pub(crate) api: usize,
    pub(crate) field_of_use: FieldOfUse,
    /// The logger of the API this device was created from, copied, so the
    /// device keeps it after `tobii_api_destroy`.
    logger: Option<SharedLogger>,
    command: TicketLock<Command>,
    dispatch: Mutex<Dispatch>,
    callbacks: Mutex<Callbacks>,
    /// Rung by every link's reader, kept across reconnects.
    doorbell: Arc<Doorbell>,
    /// Whether `process` has reported the loss of the current connection:
    /// what a `process` that finds another thread dispatching answers with.
    /// Set and cleared under the dispatch lock, read without it.
    reported: AtomicBool,
}

// Threads may share a device: every field is `Send` and `Sync` by type, the
// application's pointers inside kept in the two types that say why they may
// be (`Callbacks`, `SharedLogger`), so a field added later that is not fails
// the build here rather than making the handle unsound to share.
const _: () = {
    const fn shareable<T: Send + Sync>() {}
    shareable::<Device>();
};

impl fmt::Debug for Device {
    // What can be read without waiting: a slot table another thread holds
    // shows as locked. The dispatch state is left out, since holding its
    // lock could turn a `wait` away (see `Device::in_dispatch`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Device");
        match try_lock(&self.callbacks) {
            Some(callbacks) => s.field("streams", &callbacks.mask()),
            None => s.field("streams", &format_args!("<locked>")),
        };
        s.field("reported", &self.reported.load(Ordering::Relaxed))
            .field("field_of_use", &self.field_of_use)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Connect through `connect` and start the reader thread.
    pub(crate) fn new(
        mut connect: Connector,
        api: usize,
        field_of_use: FieldOfUse,
    ) -> io::Result<Self> {
        let doorbell = Arc::new(Doorbell::default());
        let (link, samples) = Link::open(&mut connect, &doorbell)?;
        Ok(Self {
            api,
            field_of_use,
            logger: None,
            command: TicketLock::new(Command {
                link,
                connect,
                next_request_id: 0,
                device_info: None,
            }),
            dispatch: Mutex::new(Dispatch::new(samples)),
            callbacks: Mutex::new(Callbacks::default()),
            doorbell,
            reported: AtomicBool::new(false),
        })
    }

    /// Take the logger of the API this device was created from, and say it
    /// has connected, as the DLL says on each connect ("Connected to
    /// platform module", INFO, 0x180153c19).
    pub(crate) fn adopt(&mut self, logger: Option<Logger>) {
        self.set_logger(logger);
        self.log(Level::Info, format_args!("connected to tobiid"));
    }

    /// Log to `logger` from now on, before the handle is handed out.
    pub(crate) fn set_logger(&mut self, logger: Option<Logger>) {
        self.logger = logger.map(SharedLogger);
    }

    /// Log a line to the device's logger (see [`crate::logger`]). Never
    /// called with one of the device's locks held.
    pub(crate) fn log(&self, level: Level, args: fmt::Arguments<'_>) {
        logger::emit(self.logger.map(|SharedLogger(l)| l), level, args);
    }

    /// A reply from the daemon that does not decode, `what` naming it:
    /// logged, and `TOBII_ERROR_INTERNAL`.
    pub(crate) fn malformed(&self, what: &str) -> Status {
        self.log(
            Level::Error,
            format_args!("tobiid sent a {what} reply that does not decode"),
        );
        TOBII_ERROR_INTERNAL
    }

    /// Connect to the daemon, spawning it if needed, one spawn at a time in
    /// this process (see [`connect_or_spawn_alone`]). A reconnect only
    /// connects: it fails at once when no daemon listens, and never spawns
    /// one, which could start a daemon outside systemd or race other clients
    /// into starting two.
    #[cfg(not(test))]
    pub(crate) fn connect_daemon(api: usize, field_of_use: FieldOfUse) -> io::Result<Self> {
        /// Held while a device's first connection may spawn tobiid.
        static SPAWNING: Mutex<()> = Mutex::new(());
        let first =
            || connect_or_spawn_alone(&SPAWNING, tobii_ipc::connect, tobii_ipc::connect_or_spawn);
        let connect = first_then(first, tobii_ipc::connect);
        Self::new(connect, api, field_of_use)
    }

    /// Unit tests never reach the real daemon, which would open the tracker:
    /// their devices talk to `tests::fake_daemon`, and a constructor that
    /// gets this far connects through the stand-in a test left in
    /// `tests::DAEMON` on this thread, or fails as if no daemon could be
    /// reached.
    #[cfg(test)]
    pub(crate) fn connect_daemon(api: usize, field_of_use: FieldOfUse) -> io::Result<Self> {
        let connect = tests::DAEMON.take().ok_or(io::ErrorKind::NotConnected)?;
        Self::new(connect, api, field_of_use)
    }

    /// Write one frame to the daemon (see [`Link::send`]), after the round
    /// trips other threads have under way or queued ahead of it.
    pub(crate) fn send(&self, body: &[u8]) -> Result<(), Status> {
        self.command.lock().link.send(body)
    }

    /// Register `callback` in `slot` and subscribe its stream. The Stream
    /// Engine's rules: a missing callback is invalid, an occupied slot is
    /// already subscribed.
    ///
    /// The slot is set before the daemon is asked, and the callbacks lock
    /// let go for the round trip, so another thread's process keeps
    /// delivering the streams already subscribed meanwhile, and a sample of
    /// the new stream that comes ahead of the ack (a presence) is not lost.
    /// So a callback may run on another thread before this returns, even
    /// if the change then fails, where the DLL stores the callback only
    /// once the tracker has taken the subscription (0x1801537cc..0x1801537de).
    /// A change that fails is rolled back under that lock, so once this
    /// returns, a callback another thread ran for such a sample has
    /// returned too, and none runs later. Taking the lock, to check and set
    /// the slot and to roll it back, waits for a callback another thread is
    /// running.
    pub(crate) fn subscribe<F: Copy>(
        &self,
        slot: fn(&mut Callbacks) -> &mut Slot<F>,
        callback: Option<F>,
        user_data: *mut c_void,
    ) -> Status {
        let Some(callback) = callback else {
            return TOBII_ERROR_INVALID_PARAMETER;
        };
        // Held throughout: the slots change only under it, so the mask sent
        // is the table's until the change is done or undone.
        let mut command = self.command.lock();
        let mask = {
            let mut callbacks = lock(&self.callbacks);
            if slot(&mut callbacks).is_some() {
                return TOBII_ERROR_ALREADY_SUBSCRIBED;
            }
            let before = callbacks.mask();
            *slot(&mut callbacks) = Some((callback, user_data));
            let after = callbacks.mask();
            if after == before {
                return TOBII_ERROR_NO_ERROR;
            }
            after
        };
        let status = match command.link.send_subscription(mask, SUBSCRIBE_ACK_TIMEOUT) {
            Ok(true) => TOBII_ERROR_NO_ERROR,
            Ok(false) => TOBII_ERROR_CONFLICTING_API_INSTANCES,
            Err(status) => status,
        };
        if status != TOBII_ERROR_NO_ERROR {
            *slot(&mut lock(&self.callbacks)) = None;
        }
        status
    }

    /// Drop the callback in `slot` and, if no other callback needs its
    /// stream, tell the daemon. Taking the slot waits for its callback to
    /// return if another thread's process is running it, as the DLL's
    /// unsubscribe waits on dev+0x4d8 (0x180153580): once this returns, the
    /// callback is not running on any thread, and none calls it again.
    pub(crate) fn unsubscribe<F: Copy>(&self, slot: fn(&mut Callbacks) -> &mut Slot<F>) -> Status {
        let mut command = self.command.lock();
        let mask = {
            let mut callbacks = lock(&self.callbacks);
            let before = callbacks.mask();
            if slot(&mut callbacks).take().is_none() {
                return TOBII_ERROR_NOT_SUBSCRIBED;
            }
            let after = callbacks.mask();
            if after == before {
                return TOBII_ERROR_NO_ERROR;
            }
            after
        };
        match command.link.send_subscription(mask, SUBSCRIBE_ACK_TIMEOUT) {
            Ok(_) => TOBII_ERROR_NO_ERROR,
            Err(status) => status,
        }
    }

    /// Send a request and wait up to `timeout` for its reply (see
    /// [`Command::request`]), after any round trip another thread has under
    /// way: requests on a device run one at a time.
    pub(crate) fn request(
        &self,
        kind: u8,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<Vec<u8>, Status> {
        self.command.lock().request(kind, payload, timeout)
    }

    /// As [`Self::request`], for a reply whose payload says something
    /// whatever its status: the status and the payload, `Err` only when no
    /// reply comes.
    pub(crate) fn request_reply(
        &self,
        kind: u8,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<(Status, Vec<u8>), Status> {
        self.command.lock().reply(kind, payload, timeout)
    }

    /// The device's identity, fetched from the daemon once per connection.
    /// The check, the fetch and the store make one round trip under the
    /// command lock, so a fetch racing a reconnect on another thread cannot
    /// keep the old connection's identity for the new one; a read of the
    /// kept one, too, waits for the round trips other threads have under way
    /// or queued ahead of it.
    pub(crate) fn device_info(&self) -> Result<DeviceInfoMsg, Status> {
        let mut command = self.command.lock();
        if let Some(info) = &command.device_info {
            return Ok(info.clone());
        }
        let payload = command.request(kind::DEVICE_INFO, &[], timeouts::FACTS)?;
        let Some(info) = decode_device_info(&payload) else {
            drop(command);
            return Err(self.malformed("device info"));
        };
        command.device_info = Some(info.clone());
        Ok(info)
    }

    /// Open a fresh connection and subscribe every registered stream on it,
    /// then drop the old one. As in the DLL, any failure (nothing listens,
    /// the daemon hangs up, refuses the subscriptions, or does not ack
    /// within [`RECONNECT_ACK_TIMEOUT`]) is `TOBII_ERROR_CONNECTION_FAILED`,
    /// and it changes nothing: the old link, lost or not, stays as it was,
    /// so a lost device stays lost and a loss already reported is not
    /// reported again.
    ///
    /// The old link goes only once the new one is subscribed, so reconnecting
    /// a live connection never leaves the daemon a moment with no client
    /// wanting the subscribed streams, in which it would drop its engine and
    /// start the tracker cold again. A device with no subscription has
    /// nothing to send first, so the daemon may still drop an engine that
    /// only the old connection's requests kept, until the next request.
    ///
    /// Of what the old link brought and nothing has delivered, the samples
    /// are dropped and the notifications kept, ahead of the new link's (see
    /// [`Dispatch::swap_samples`]). tobiid writes each tick to every
    /// connection subscribed to its streams, so from the tick that acks the
    /// new link on, the old one brings the samples the new one does:
    /// delivering from both would repeat samples, their stamps stepping
    /// back, so no sample is delivered from the old link once the new one
    /// has asked for the streams (and nothing at all is delivered from then
    /// until the swap, below), and the samples tobiid sent the old link
    /// alone before are lost. Its notifications are not: tobiid sends each
    /// once, and does not repeat it to a new subscriber, so they are kept,
    /// and those it sent both links come twice, the old link's copies ahead
    /// of the new link's. For them the old link is closed before its
    /// channel is read out, which lets its reader read to the end what
    /// tobiid wrote it. That is all tobiid sent it alone: tobiid's pump
    /// writes its connections in the order they were made, each its queued
    /// frames first, so what it sent the old one before it took the new
    /// one's subscriptions was in the old socket before the ack left for
    /// the new one.
    ///
    /// The new link starts up, so its loss is reported again, and owes no
    /// acks. Once it is in place this rings the device's doorbell (see
    /// [`Doorbell`]): the new link's reader rings it too, but its first
    /// samples may come while this still waits for the ack, and a wait they
    /// woke then was turned away by the dispatch lock and slept again. The
    /// device info is fetched again: a restarted daemon may serve another
    /// tracker.
    ///
    /// It runs under the command lock, after the round trips other threads
    /// have under way or queued ahead of it. Before it asks for the
    /// subscriptions back it takes the dispatch lock, once a process another
    /// thread runs has finished its callbacks, as the DLL's waits for its
    /// process mutex (0x18000eb88), and it holds that lock until it has
    /// closed the old link and swapped the samples channel, or failed: for
    /// up to [`RECONNECT_ACK_TIMEOUT`], and then the moment its old link's
    /// reader takes to read what that link still holds, a process on
    /// another thread returns at once, delivering nothing, a wait sleeps
    /// until this rings or its own timeout ends, and a clear waits. It reads
    /// the subscriptions under the callbacks lock, which is free by then.
    /// Each failure is logged at ERROR, a success at INFO, once the locks
    /// are let go.
    pub(crate) fn reconnect(&self) -> Status {
        match self.replace_link() {
            Ok(()) => {
                self.log(Level::Info, format_args!("reconnected to tobiid"));
                TOBII_ERROR_NO_ERROR
            }
            Err(e) => {
                self.log(
                    Level::Error,
                    format_args!("could not reconnect to tobiid: {e}"),
                );
                TOBII_ERROR_CONNECTION_FAILED
            }
        }
    }

    /// What [`Device::reconnect`] does under the command lock.
    fn replace_link(&self) -> Result<(), ReconnectError> {
        let mut command = self.command.lock();
        let (mut link, samples) =
            Link::open(&mut command.connect, &self.doorbell).map_err(ReconnectError::Connect)?;
        // Taken before the new link asks for anything, and held until the
        // swap: what the old link brings from then on, the new one may bring
        // too, and no process may deliver it from the old one meanwhile.
        let mut dispatch = lock(&self.dispatch);
        let seen = self.doorbell.rings();
        let mask = lock(&self.callbacks).mask();
        if mask != 0 {
            let why = match link.send_subscription(mask, RECONNECT_ACK_TIMEOUT) {
                Ok(true) => None,
                Ok(false) => Some("refused them"),
                Err(TOBII_ERROR_TIMED_OUT) => Some("did not acknowledge them in time"),
                Err(_) => Some("hung up"),
            };
            if let Some(why) = why {
                // As `in_dispatch` lets the lock go: the old link stays, and
                // a wait the hold turned away may have something on it.
                let left = dispatch.anything_to_process();
                drop(dispatch);
                self.doorbell.ring_after_hold(seen, left);
                return Err(ReconnectError::Subscribe { mask, why });
            }
        }
        // Closed before the swap reads out its channel: a socket shut down
        // still reads what it holds, and tobiid can write it no more, so the
        // join leaves all the old link was sent in its channel, and none of
        // the notifications among it is lost. Its reader takes no lock but
        // the doorbell's, and has at most a socket buffer left to read.
        let mut old = mem::replace(&mut command.link, link);
        old.close();
        dispatch.swap_samples(samples);
        self.reported.store(false, Ordering::Relaxed);
        drop(dispatch);
        // Also wakes a wait whose look the hold turned away.
        self.doorbell.ring();
        command.device_info = None;
        drop(command);
        drop(old);
        Ok(())
    }

    /// Drop every queued sample, once a process another thread runs has
    /// finished its callbacks, or a reconnect another thread runs has put
    /// its new link in place or failed. A lost connection stays lost, and a
    /// loss not reported yet is still reported. The answers are left alone:
    /// a late ack dropped here would still be counted as owed.
    ///
    /// The DLL's differs on both counts: it holds its API mutex (dev+0x4e0,
    /// 0x180143a85), so it queues behind requests and reconnects, and it
    /// clears by running its process with the callbacks swapped out
    /// (0x180158a20), which empties the device's notification queue but,
    /// while another thread processes, fails its try-enter (0x18000e9d9)
    /// and leaves the rest queued (read from the code, not observed).
    /// libtobii's waits for the dispatch instead: never for a request, but
    /// for a reconnect too, which holds the dispatch lock while it waits for
    /// its new link's ack and closes the old link.
    pub(crate) fn clear_buffers(&self) {
        self.in_dispatch(lock(&self.dispatch), |dispatch| {
            dispatch.take_all();
            dispatch.pending.clear();
        });
    }

    /// Whether `process` has something to do, waiting up to `timeout` for
    /// it: a queued sample, or a lost connection it has not reported yet.
    /// Between looks it sleeps on the doorbell (see [`Doorbell::wait_for`]),
    /// which the reader rings after each sample and when it stops, so either
    /// wakes it at once. An answer (a stale reply, a late ack) is nothing to
    /// process and rings nothing. Once the loss is reported, the lost link
    /// rings no more, so this sleeps out `timeout` and says no, as for a
    /// quiet link; answering at once would spin a wait-and-process loop. A
    /// ring that brings nothing (from an old or a failed link, or for a
    /// sample an earlier look took) sends it back to sleep until the same
    /// deadline. A reconnect on another thread wakes it.
    ///
    /// It holds no lock while it sleeps. A look that finds another thread
    /// holding the dispatch lock (a process running the callbacks, another
    /// look, a clear, a reconnect waiting for its new link's ack) finds
    /// nothing and sleeps on, where the DLL answers at once with nothing to
    /// wait on; that thread rings again once it lets the lock go if it
    /// leaves something to process, or a sample came meanwhile (see
    /// [`Device::in_dispatch`]), and a reconnect that has swapped links
    /// rings anyway.
    pub(crate) fn wait(&self, timeout: Duration) -> bool {
        self.doorbell.wait_for(timeout, || self.look())
    }

    /// One look of `wait`: whether `process` has something to do now,
    /// moving a queued sample into `pending`. Nothing while another thread
    /// holds the dispatch lock.
    fn look(&self) -> bool {
        try_lock(&self.dispatch).is_some_and(|dispatch| {
            self.in_dispatch(dispatch, |dispatch| {
                if !dispatch.anything_to_process() {
                    dispatch.take_one();
                }
                dispatch.anything_to_process()
            })
        })
    }

    /// Deliver every queued sample to its callbacks, on this thread, then
    /// say whether the daemon connection is up. Once it is lost, what had
    /// arrived before is still delivered, and then this call and every later
    /// one returns `TOBII_ERROR_CONNECTION_FAILED` until a reconnect connects
    /// again.
    ///
    /// One thread dispatches a device at a time. A call that finds another
    /// thread holding the dispatch lock (a process running the callbacks, a
    /// look of `wait`, a clear, or a reconnect, from its new link's
    /// subscription to its swap) returns at once, delivering nothing, and
    /// what is queued stays for the next call:
    /// `TOBII_ERROR_NO_ERROR`, or `TOBII_ERROR_CONNECTION_FAILED` once the
    /// loss has been reported. The DLL's returns at once too, once it has
    /// delivered the notifications queued for the device
    /// (0x180159515..0x180159566): it try-enters its process mutex only
    /// then, and answers 0 when that fails (0x18000e9d9..0x18000e9ea), even
    /// after a loss. libtobii has one queue, which the thread dispatching
    /// delivers.
    #[must_use]
    pub(crate) fn process(&self) -> Status {
        let Some(dispatch) = try_lock(&self.dispatch) else {
            return if self.reported.load(Ordering::Relaxed) {
                TOBII_ERROR_CONNECTION_FAILED
            } else {
                TOBII_ERROR_NO_ERROR
            };
        };
        let (status, newly_lost) = self.in_dispatch(dispatch, |dispatch| {
            dispatch.take_all();
            while let Some(msg) = dispatch.pending.pop_front() {
                self.deliver(&msg);
            }
            match dispatch.state {
                LinkState::Up => (TOBII_ERROR_NO_ERROR, false),
                LinkState::Lost => {
                    dispatch.state = LinkState::Reported;
                    self.reported.store(true, Ordering::Relaxed);
                    (TOBII_ERROR_CONNECTION_FAILED, true)
                }
                LinkState::Reported => (TOBII_ERROR_CONNECTION_FAILED, false),
            }
        });
        if newly_lost {
            // Once per loss, by the call that reported it: a host keeps
            // calling at its frame rate.
            self.log(
                Level::Error,
                format_args!("lost the connection to tobiid; tobii_device_reconnect restores it"),
            );
        }
        status
    }

    /// Run `f` on the dispatch state under `dispatch`, a guard of its lock,
    /// then let the lock go and ring the doorbell again if a `wait` on
    /// another thread may have looked meanwhile, found the lock held, and
    /// gone back to sleep with something to process (see
    /// [`Doorbell::ring_after_hold`]). Every hold but a reconnect's goes
    /// through here; a reconnect rings once it has swapped links, and as
    /// this does when it fails.
    fn in_dispatch<R>(
        &self,
        mut dispatch: MutexGuard<'_, Dispatch>,
        f: impl FnOnce(&mut Dispatch) -> R,
    ) -> R {
        let seen = self.doorbell.rings();
        let out = f(&mut dispatch);
        let left = dispatch.anything_to_process();
        drop(dispatch);
        self.doorbell.ring_after_hold(seen, left);
        out
    }

    /// Deliver one daemon message to the matching callbacks, if any, under
    /// the callbacks lock. The timestamps go through as the daemon sent
    /// them, on the host clock already (gaze data's tracker time and raw
    /// gaze's aside): nothing is converted here.
    fn deliver(&self, msg: &ServerMsg) {
        let cb = lock(&self.callbacks);
        match msg {
            ServerMsg::Head {
                ts_us,
                pos_mm,
                rot_rad,
            } => {
                let hp = HeadPose {
                    timestamp_us: *ts_us,
                    position_validity: TOBII_VALIDITY_VALID,
                    position_xyz: *pos_mm,
                    rotation_validity_xyz: [TOBII_VALIDITY_VALID; 3],
                    rotation_xyz: *rot_rad,
                };
                if let Some((f, ud)) = cb.head {
                    // SAFETY: registered through `tobii_head_pose_subscribe`
                    // (see the module invariant), called on the processing
                    // thread under the callbacks lock; `hp` outlives the call.
                    call(|| unsafe { f(&raw const hp, ud) });
                }
            }
            ServerMsg::Gaze {
                ts_us, valid, xy, ..
            } => {
                let gp = GazePoint {
                    timestamp_us: *ts_us,
                    validity: validity(*valid),
                    position_xy: *xy,
                };
                if let Some((f, ud)) = cb.gaze {
                    // SAFETY: registered through `tobii_gaze_point_subscribe`;
                    // `gp` outlives the call.
                    call(|| unsafe { f(&raw const gp, ud) });
                }
            }
            ServerMsg::Presence { ts_us, status } => {
                if let Some((f, ud)) = cb.presence {
                    // SAFETY: registered through `tobii_user_presence_subscribe`,
                    // callable with any status, timestamp and `ud`.
                    call(|| unsafe { f(PresenceStatus::from(*status), *ts_us, ud) });
                }
            }
            ServerMsg::GazeOrigin(pair) => {
                let c = eye_pair(pair);
                if let Some((f, ud)) = cb.gaze_origin {
                    // SAFETY: registered through `tobii_gaze_origin_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::EyePosition(pair) => {
                let c = eye_pair(pair);
                for (f, ud) in [cb.eye_position, cb.user_position_guide]
                    .into_iter()
                    .flatten()
                {
                    // SAFETY: registered through
                    // `tobii_eye_position_normalized_subscribe` or
                    // `tobii_user_position_guide_subscribe`; `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::GazeData(data) => {
                let c = GazeData {
                    timestamp_tracker_us: data.timestamp_tracker_us,
                    timestamp_system_us: data.timestamp_system_us,
                    left: gaze_data_eye(&data.left),
                    right: gaze_data_eye(&data.right),
                };
                if let Some((f, ud)) = cb.gaze_data {
                    // SAFETY: registered through `tobii_gaze_data_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::GazeRaw(record) => {
                let c = gaze_raw(record);
                if let Some((f, ud)) = cb.gaze_raw {
                    // SAFETY: registered through `tobii_gaze_raw_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::Image(image) => {
                let (Ok(width), Ok(height)) =
                    (i32::try_from(image.width), i32::try_from(image.height))
                else {
                    return;
                };
                let c = Image {
                    timestamp_us: image.ts_us,
                    width,
                    padding_per_row: 0,
                    height,
                    bits_per_pixel: i32::from(image.bits_per_pixel),
                    data: image.pixels.as_ptr().cast(),
                };
                if let Some((f, ud)) = cb.image {
                    // SAFETY: registered through `tobii_image_subscribe`; `c`
                    // and the pixels it points to outlive the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::Notification(n) => {
                let c = notification(n);
                if let Some((f, ud)) = cb.notifications {
                    // SAFETY: registered through `tobii_notifications_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            // Message kinds a newer daemon may add: nothing to deliver. The
            // answers never get here, the reader sorts them out.
            _ => {}
        }
    }
}

/// Run application code (a user callback, the logger or
/// `tobii_calibration_retrieve`'s receiver) with the re-entry guard set, then
/// leave the guard as it was: the logger can run inside a callback or the
/// receiver, which still need the guard once the line is logged.
pub(crate) fn call(f: impl FnOnce()) {
    let was = IN_CALLBACK.replace(true);
    f();
    IN_CALLBACK.set(was);
}

fn validity(valid: bool) -> Validity {
    if valid {
        TOBII_VALIDITY_VALID
    } else {
        TOBII_VALIDITY_INVALID
    }
}

fn eye_pair(p: &tobii_ipc::EyePair) -> EyePair {
    EyePair {
        timestamp_us: p.ts_us,
        left_validity: validity(p.left.valid),
        left_xyz: p.left.xyz,
        right_validity: validity(p.right.valid),
        right_xyz: p.right.xyz,
    }
}

fn gaze_data_eye(e: &tobii_ipc::GazeDataEye) -> GazeDataEye {
    GazeDataEye {
        gaze_origin_validity: validity(e.gaze_origin_valid),
        gaze_origin_from_eye_tracker_mm_xyz: e.gaze_origin_mm,
        gaze_origin_in_track_box_normalized_xyz: e.gaze_origin_in_track_box,
        gaze_point_validity: validity(e.gaze_point_valid),
        gaze_point_from_eye_tracker_mm_xyz: e.gaze_point_mm,
        gaze_point_on_display_normalized_xy: e.gaze_point_on_display,
        eyeball_center_validity: validity(e.eyeball_center_valid),
        eyeball_center_from_eye_tracker_mm_xyz: e.eyeball_center_mm,
        pupil_validity: validity(e.pupil_valid),
        pupil_diameter_mm: e.pupil_diameter_mm,
    }
}

fn gaze_raw_eye(e: &tobii_ipc::GazeRawEye) -> GazeRawEye {
    GazeRawEye {
        gaze_origin_from_eye_tracker_mm_xyz: e.gaze_origin_mm,
        gaze_origin_in_track_box_normalized_xyz: e.gaze_origin_in_track_box,
        gaze_point_from_eye_tracker_mm_xyz: e.gaze_point_mm,
        gaze_point_on_display_normalized_xy: e.gaze_point_on_display,
        pupil_diameter_mm: e.pupil_diameter_mm,
        status: e.status,
    }
}

/// A raw gaze sample in the 232-byte C layout, as the DLL's dispatch copies
/// its record (0x180171b81..0x180171db8): every value as the daemon sent it,
/// a flagged key's validity 1 when the frame had the key, and 0 (with a
/// value of 0) when it did not; the slots the record never fills, 0.
fn gaze_raw(r: &tobii_ipc::GazeRaw) -> GazeRaw {
    let sent = |word: Option<u32>| (validity(word.is_some()), word.unwrap_or(0));
    let (key_0e_validity, key_0e) = sent(r.key_0e);
    let (key_11_validity, key_11) = sent(r.key_11);
    let (frame_counter_validity, frame_counter) = sent(r.frame_counter);
    let (left_origin_flag_validity, left_origin_flag) = sent(r.left_origin_flag);
    let (right_origin_flag_validity, right_origin_flag) = sent(r.right_origin_flag);
    GazeRaw {
        timestamp_tracker_us: r.timestamp_tracker_us,
        left: gaze_raw_eye(&r.left),
        right: gaze_raw_eye(&r.right),
        combined_gaze_point_on_display_normalized_xy: r.combined_gaze_point_on_display,
        combined_gaze_validity: r.combined_gaze_validity,
        key_0e_validity,
        key_0e,
        reserved_84: TOBII_VALIDITY_INVALID,
        reserved_88: 0.0,
        reserved_8c: TOBII_VALIDITY_INVALID,
        reserved_90: 0.0,
        key_11_validity,
        key_11,
        reserved_9c: TOBII_VALIDITY_INVALID,
        reserved_a0: 0.0,
        reserved_a4: TOBII_VALIDITY_INVALID,
        reserved_a8: 0.0,
        frame_counter_validity,
        frame_counter,
        left_origin_flag_validity,
        left_origin_flag,
        right_origin_flag_validity,
        right_origin_flag,
        left_eyeball_center_validity: validity(r.left_eyeball_center_mm.is_some()),
        left_eyeball_center_from_eye_tracker_mm_xyz: r.left_eyeball_center_mm.unwrap_or_default(),
        right_eyeball_center_validity: validity(r.right_eyeball_center_mm.is_some()),
        right_eyeball_center_from_eye_tracker_mm_xyz: r.right_eyeball_center_mm.unwrap_or_default(),
        reserved_e4: 0,
    }
}

/// A display area in the C layout.
#[allow(clippy::cast_possible_truncation)] // reason: the C ABI carries float
pub(crate) fn display_area(a: &tobii_ipc::geometry::DisplayArea) -> DisplayArea {
    let f = |v: [f64; 3]| v.map(|c| c as f32);
    DisplayArea {
        top_left_mm_xyz: f(a.top_left_mm),
        top_right_mm_xyz: f(a.top_right_mm),
        bottom_left_mm_xyz: f(a.bottom_left_mm),
    }
}

/// A wire notification in the 520-byte C layout.
fn notification(n: &tobii_ipc::Notification) -> Notification {
    let mut c = Notification {
        type_: u32::from(n.kind),
        value_type: crate::types::TOBII_NOTIFICATION_VALUE_TYPE_NONE,
        value: NotificationValue { string_: [0; 512] },
    };
    match &n.value {
        WireValue::Float(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_FLOAT;
            c.value.float_ = *v;
        }
        WireValue::State(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_STATE;
            c.value.state = u32::from(*v);
        }
        WireValue::DisplayArea(a) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_DISPLAY_AREA;
            c.value.display_area = display_area(a);
        }
        WireValue::Uint(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_UINT;
            c.value.uint_ = *v;
        }
        WireValue::EnabledEye(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_ENABLED_EYE;
            c.value.enabled_eye = u32::from(*v);
        }
        WireValue::String(s) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_STRING;
            let mut buf = [0; 512];
            copy_c_string(&mut buf, s);
            c.value.string_ = buf;
        }
        _ => {}
    }
    c
}

/// Borrow a device handle from C, shared: threads may each borrow one at
/// once, its locks keeping them apart.
///
/// Refuses every call made from inside a callback, the logger or
/// `tobii_calibration_retrieve`'s receiver (as the Stream Engine does for a
/// callback), before any lock is taken, then a null handle.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that is
/// not destroyed before the borrow ends.
pub(crate) unsafe fn device_ref<'a>(device: *mut Device) -> Result<&'a Device, Status> {
    if in_callback() {
        return Err(TOBII_ERROR_CALLBACK_IN_PROGRESS);
    }
    // SAFETY: the caller guarantees `device` is null or live for the borrow;
    // other threads may borrow it meanwhile, as `Device` is `Sync`.
    unsafe { device.as_ref() }.ok_or(TOBII_ERROR_INVALID_PARAMETER)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::logger::tests::{Recorder, SyncRecorder};
    use crate::types::{
        LogLevel, TOBII_LOG_LEVEL_ERROR, TOBII_LOG_LEVEL_INFO, TOBII_STATE_CALIBRATION_ACTIVE,
    };
    use std::cell::RefCell;
    use std::ffi::c_char;
    use std::io::Write as _;
    use std::ptr;
    use std::sync::atomic::{AtomicI64, AtomicU32, AtomicUsize};
    use tobii_ipc::{
        STREAM_GAZE_ORIGIN, encode_gaze, encode_gaze_origin, encode_reply, encode_subscribed,
    };

    impl Device {
        /// The streams its callbacks need.
        fn mask(&self) -> u32 {
            lock(&self.callbacks).mask()
        }

        /// Subscribe the streams in `mask` on the current connection, as a
        /// subscription change does, waiting up to `timeout` for the ack.
        fn send_subscription(&self, mask: u32, timeout: Duration) -> Result<bool, Status> {
            self.command.lock().link.send_subscription(mask, timeout)
        }

        /// Queue `msg` for the next `process`, as a look of `wait` does.
        fn queue(&self, msg: ServerMsg) {
            lock(&self.dispatch).pending.push_back(msg);
        }

        /// Whether a look of `wait` has taken a sample for `process`.
        fn has_pending(&self) -> bool {
            !lock(&self.dispatch).pending.is_empty()
        }
    }

    impl<T> TicketLock<T> {
        /// Whether a thread holds it, rather than only waits for it: only
        /// the thread whose turn it is takes the data's lock. A holder
        /// shows as such once it has taken that lock, a moment after its
        /// turn came.
        fn held(&self) -> bool {
            try_lock(&self.data).is_none()
        }

        /// The tickets handed out whose turn has not been passed on: the
        /// holder's, if any, and the waiters'.
        fn out(&self) -> u64 {
            let counts = lock(&self.tickets.counts);
            counts.0.wrapping_sub(counts.1)
        }
    }

    thread_local! {
        /// The daemon stand-in the next device constructor on this thread
        /// connects to, taken by the test build of `Device::connect_daemon`;
        /// with none there, the constructor fails to connect.
        pub(crate) static DAEMON: Cell<Option<Connector>> = const { Cell::new(None) };

        /// The timeout of each request made on this thread, noted by the
        /// test build of `Command::request`.
        static REQUEST_TIMEOUTS: RefCell<Vec<Duration>> = const { RefCell::new(Vec::new()) };
    }

    /// Note that a request on this thread waits up to `timeout`.
    pub(super) fn note_request(timeout: Duration) {
        REQUEST_TIMEOUTS.with_borrow_mut(|timeouts| timeouts.push(timeout));
    }

    /// The timeouts of the requests made on this thread since the last take.
    pub(crate) fn take_request_timeouts() -> Vec<Duration> {
        REQUEST_TIMEOUTS.take()
    }

    /// A daemon stand-in: answers each client frame with whatever `handler`
    /// returns (whole frame bodies), on a socket pair.
    pub(crate) fn fake_daemon(
        handler: impl FnMut(&[u8]) -> Vec<Vec<u8>> + Send + 'static,
    ) -> Connector {
        let handler = std::sync::Arc::new(std::sync::Mutex::new(handler));
        Box::new(move || {
            let (client, mut server) = UnixStream::pair()?;
            let handler = std::sync::Arc::clone(&handler);
            thread::spawn(move || {
                while let Ok(Some(body)) = read_frame(&mut server) {
                    let replies = (handler.lock().expect("handler"))(&body);
                    for r in replies {
                        if write_frame(&mut server, &r).is_err() {
                            return;
                        }
                    }
                }
            });
            Ok(client)
        })
    }

    /// Acks every subscription and answers every request with `status`/`payload`.
    pub(crate) fn device_with(status: u8, payload: Vec<u8>) -> Device {
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![encode_subscribed(true)],
            Some(&tobii_ipc::TAG_REQUEST) => {
                let req = tobii_ipc::request::decode_request(body).expect("request");
                vec![encode_reply(req.id, status, &payload)]
            }
            _ => vec![],
        });
        Device::new(connect, 1, 1).expect("device")
    }

    /// The request reads past a stale reply to its own, and leaves the
    /// sample that came first where it was, in the samples channel, for
    /// `process` to deliver. What a caller sees, the sample delivered, held
    /// when the request queued it in `pending` too; that the waiting call
    /// moves no sample is a structural check.
    #[test]
    fn a_request_gets_its_reply_and_keeps_samples_that_came_first() {
        let connect = fake_daemon(|body| {
            let req = tobii_ipc::request::decode_request(body).expect("request");
            vec![
                encode_gaze(1, true, [0.5, 0.5], [f32::NAN; 2]),
                encode_reply(req.id.wrapping_add(7), 0, b"stale"),
                encode_reply(req.id, 0, b"answer"),
            ]
        });
        let d = Device::new(connect, 1, 1).expect("device");
        let mut stamps = Stamps::default();
        lock(&d.callbacks).gaze = Some((stamp_gaze as GazePointFn, (&raw mut stamps).cast()));

        let got = d.request(1, &[], Duration::from_secs(2));

        assert_eq!(got, Ok(b"answer".to_vec()));
        assert!(!d.has_pending(), "the request moved no sample");
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(d);
        assert_eq!(stamps.gaze, 1, "the gaze sample waited for process()");
    }

    /// A subscription change, too, leaves the samples that arrive ahead of
    /// its ack for `process`. As above, the delivery held before; that the
    /// change moves no sample is a structural check.
    #[test]
    fn a_subscription_change_keeps_samples_that_came_first() {
        let connect = fake_daemon(|body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![
                encode_gaze_origin(&tobii_ipc::EyePair::default()),
                encode_subscribed(true),
            ],
            _ => vec![],
        });
        let d = Device::new(connect, 1, 1).expect("device");
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );

        assert!(!d.has_pending(), "the change moved no sample");
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(d);
        assert_eq!(hits, 1, "the sample waited for process()");
    }

    /// A subscription change that gave up waiting still has its ack coming:
    /// the next change reads past that late ack to its own answer, here a
    /// refusal, rather than take the late one for it.
    #[test]
    fn a_late_subscription_ack_is_not_taken_for_the_next_change() {
        // Leaves the first change unacked, then acks it late, ahead of its
        // refusal of the second.
        let mut changes = 0u32;
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                changes += 1;
                if changes == 1 {
                    vec![]
                } else {
                    vec![encode_subscribed(true), encode_subscribed(false)]
                }
            }
            _ => vec![],
        });
        let d = Device::new(connect, 1, 1).expect("device");
        assert_eq!(
            d.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
            Err(TOBII_ERROR_TIMED_OUT)
        );
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.eye_position, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_CONFLICTING_API_INSTANCES
        );

        assert_eq!(d.mask(), 0, "rolled back");
    }

    /// A daemon that leaves the first subscription change unacked and
    /// refuses every later one. It answers a client frame tagged `late` with
    /// the first change's ack, late, followed by what `late_frames` gives for
    /// that frame.
    fn late_acking_daemon(
        late: u8,
        mut late_frames: impl FnMut(&[u8]) -> Vec<Vec<u8>> + Send + 'static,
    ) -> Connector {
        let mut changes = 0u32;
        fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                changes += 1;
                if changes == 1 {
                    vec![]
                } else {
                    vec![encode_subscribed(false)]
                }
            }
            Some(&tag) if tag == late => {
                let mut frames = vec![encode_subscribed(true)];
                frames.extend(late_frames(body));
                frames
            }
            _ => vec![],
        })
    }

    /// A late ack that a request reads past on its way to its reply is owed
    /// no more: the next subscription change takes its own answer at once,
    /// rather than read past it too and give up.
    #[test]
    fn a_late_ack_a_request_reads_past_is_owed_no_more() {
        let connect = late_acking_daemon(tobii_ipc::TAG_REQUEST, |body| {
            let req = tobii_ipc::request::decode_request(body).expect("request");
            vec![encode_reply(req.id, 0, b"answer")]
        });
        let d = Device::new(connect, 1, 1).expect("device");
        assert_eq!(
            d.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
            Err(TOBII_ERROR_TIMED_OUT)
        );
        assert_eq!(d.request(1, &[], LONG_WAIT), Ok(b"answer".to_vec()));
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        let t = Instant::now();
        assert_eq!(
            d.subscribe(|c| &mut c.eye_position, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_CONFLICTING_API_INSTANCES
        );
        assert!(t.elapsed() < PROMPT);
    }

    /// Clearing the buffers drops the samples and leaves the answers: a late
    /// ack cleared away would still be counted as owed, and the next change
    /// would read past its own answer and give up.
    #[test]
    fn clearing_the_buffers_leaves_a_late_ack_to_be_read_past() {
        // The late ack comes when the client recenters, a sample behind it.
        let connect = late_acking_daemon(tobii_ipc::TAG_RECENTER, |_| {
            vec![encode_gaze_origin(&tobii_ipc::EyePair::default())]
        });
        let d = Device::new(connect, 1, 1).expect("device");
        assert_eq!(
            d.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
            Err(TOBII_ERROR_TIMED_OUT)
        );
        assert_eq!(d.send(&tobii_ipc::encode_recenter()), Ok(()));
        assert!(
            d.wait(LONG_WAIT),
            "the sample, so the ack ahead of it is in"
        );
        // A callback for the sample's stream, so `process` would deliver it
        // had the clear left it.
        let mut hits = 0u32;
        lock(&d.callbacks).gaze_origin = Some((count_pair as EyePairFn, (&raw mut hits).cast()));

        d.clear_buffers();

        let t = Instant::now();
        assert_eq!(
            d.subscribe(
                |c| &mut c.eye_position,
                Some(ignore_pair as EyePairFn),
                ptr::null_mut()
            ),
            TOBII_ERROR_CONFLICTING_API_INSTANCES
        );
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(d);
        assert_eq!(hits, 0, "the sample was cleared");
    }

    /// Only a sample wakes `wait`: a reply that comes after its request gave
    /// up is nothing to process. The next subscription change reads past
    /// that reply to its own ack, which also shows the reply was read; that
    /// half is a guard, as it held when the reply was queued for `process`.
    #[test]
    fn a_stale_reply_wakes_no_wait_and_a_change_reads_past_it() {
        let (connect, daemons) = scripted_daemon(vec![vec![
            encode_reply(7, 0, b"stale"),
            encode_subscribed(true),
        ]]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut daemon = daemons.recv().expect("daemon end");

        let t = Instant::now();
        assert!(!d.wait(SHORT_WAIT), "nothing to process");
        assert!(t.elapsed() >= SHORT_WAIT, "slept out the timeout");
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);

        let t = Instant::now();
        assert_eq!(
            d.subscribe(
                |c| &mut c.gaze_origin,
                Some(ignore_pair as EyePairFn),
                ptr::null_mut()
            ),
            TOBII_ERROR_NO_ERROR
        );
        assert!(t.elapsed() < PROMPT);
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
    }

    #[test]
    fn a_failed_reply_is_its_status_and_silence_times_out() {
        let d = device_with(15, vec![]);
        assert_eq!(d.request(0x10, &[2], Duration::from_secs(2)), Err(15));

        let quiet = Device::new(fake_daemon(|_| vec![]), 1, 1).expect("device");
        assert_eq!(
            quiet.request(1, &[], Duration::from_millis(50)),
            Err(TOBII_ERROR_TIMED_OUT)
        );
    }

    pub(crate) unsafe extern "C" fn count_pair(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the tests pass `&raw mut u32` as `ud`.
        unsafe { *ud.cast::<u32>() += 1 };
    }

    #[test]
    fn subscription_rules_follow_the_stream_engine() {
        let d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, None::<EyePairFn>, ud),
            TOBII_ERROR_INVALID_PARAMETER
        );
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_ALREADY_SUBSCRIBED
        );
        assert_eq!(d.mask(), STREAM_GAZE_ORIGIN);
        assert_eq!(d.unsubscribe(|c| &mut c.gaze_origin), 0);
        assert_eq!(
            d.unsubscribe(|c| &mut c.gaze_origin),
            TOBII_ERROR_NOT_SUBSCRIBED
        );
    }

    /// Eye position and the user position guide share one daemon stream: the
    /// bit stays while either is subscribed, and each sample reaches both.
    #[test]
    fn a_shared_stream_feeds_both_callbacks() {
        let d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.eye_position, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(
            d.subscribe(
                |c| &mut c.user_position_guide,
                Some(count_pair as EyePairFn),
                ud
            ),
            0
        );

        d.deliver(&ServerMsg::EyePosition(tobii_ipc::EyePair::default()));

        assert_eq!(hits, 2);
        assert_eq!(d.unsubscribe(|c| &mut c.eye_position), 0);
        assert_eq!(d.mask(), STREAM_EYE_POSITION);
    }

    #[test]
    fn samples_are_dispatched_in_the_c_layout() {
        let d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        let body = encode_gaze_origin(&tobii_ipc::EyePair::default());
        d.queue(decode_server(&body).expect("decodes"));

        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);

        assert_eq!(hits, 1);
    }

    /// The timestamps each callback got last.
    #[derive(Debug, Default, PartialEq, Eq)]
    pub(crate) struct Stamps {
        pub(crate) head: i64,
        pub(crate) gaze: i64,
        pub(crate) presence: i64,
        pub(crate) gaze_origin: i64,
        pub(crate) eye_position: i64,
        pub(crate) user_position_guide: i64,
        /// `(timestamp_tracker_us, timestamp_system_us)`.
        pub(crate) gaze_data: (i64, i64),
        /// `timestamp_tracker_us`.
        pub(crate) gaze_raw: i64,
        pub(crate) image: i64,
    }

    // The `stamp_*` callbacks note their sample's timestamps in the `Stamps`
    // that the tests pass as `ud`.

    unsafe extern "C" fn stamp_head(p: *const HeadPose, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).head = (*p).timestamp_us };
    }

    pub(crate) unsafe extern "C" fn stamp_gaze(p: *const GazePoint, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).gaze = (*p).timestamp_us };
    }

    unsafe extern "C" fn stamp_presence(_s: PresenceStatus, ts: i64, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps` (see above).
        unsafe { (*ud.cast::<Stamps>()).presence = ts };
    }

    unsafe extern "C" fn stamp_gaze_origin(p: *const EyePair, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).gaze_origin = (*p).timestamp_us };
    }

    unsafe extern "C" fn stamp_eye_position(p: *const EyePair, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).eye_position = (*p).timestamp_us };
    }

    unsafe extern "C" fn stamp_user_position_guide(p: *const EyePair, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).user_position_guide = (*p).timestamp_us };
    }

    pub(crate) unsafe extern "C" fn stamp_gaze_data(p: *const GazeData, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe {
            (*ud.cast::<Stamps>()).gaze_data =
                ((*p).timestamp_tracker_us, (*p).timestamp_system_us);
        }
    }

    unsafe extern "C" fn stamp_gaze_raw(p: *const GazeRaw, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).gaze_raw = (*p).timestamp_tracker_us };
    }

    unsafe extern "C" fn stamp_image(p: *const Image, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).image = (*p).timestamp_us };
    }

    /// libtobii hands each callback the timestamp its frame carried, which
    /// the daemon sends on the host clock, and gaze data's tracker time
    /// beside it; raw gaze carries the tracker time alone: nothing is
    /// converted on this side.
    #[test]
    fn every_callback_gets_the_timestamps_its_frame_carried() {
        let pair = |ts_us| tobii_ipc::EyePair {
            ts_us,
            ..tobii_ipc::EyePair::default()
        };
        let frames = vec![
            tobii_ipc::encode_head(11, [0.0; 3], [0.0; 3]),
            encode_gaze(12, true, [0.5; 2], [f32::NAN; 2]),
            tobii_ipc::encode_presence(13, tobii_ipc::PRESENCE_PRESENT),
            encode_gaze_origin(&pair(14)),
            tobii_ipc::encode_eye_position(&pair(15)),
            tobii_ipc::encode_gaze_data(&tobii_ipc::GazeData {
                timestamp_tracker_us: 7,
                timestamp_system_us: 16,
                ..tobii_ipc::GazeData::default()
            }),
            tobii_ipc::encode_image(17, 1, 1, 8, &[0]),
            tobii_ipc::encode_gaze_raw(&tobii_ipc::GazeRaw {
                timestamp_tracker_us: 8,
                ..tobii_ipc::GazeRaw::default()
            }),
        ];
        // Sends them all ahead of every subscription ack, so that they wait
        // for `process`.
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                let mut out = frames.clone();
                out.push(encode_subscribed(true));
                out
            }
            _ => vec![],
        });
        let d = Device::new(connect, 1, 1).expect("device");
        let mut stamps = Stamps::default();
        let ud = (&raw mut stamps).cast::<c_void>();
        *lock(&d.callbacks) = Callbacks {
            head: Some((stamp_head as HeadPoseFn, ud)),
            gaze: Some((stamp_gaze as GazePointFn, ud)),
            presence: Some((stamp_presence as PresenceFn, ud)),
            gaze_origin: Some((stamp_gaze_origin as EyePairFn, ud)),
            eye_position: Some((stamp_eye_position as EyePairFn, ud)),
            user_position_guide: Some((stamp_user_position_guide as EyePairFn, ud)),
            gaze_data: Some((stamp_gaze_data as GazeDataFn, ud)),
            gaze_raw: Some((stamp_gaze_raw as GazeRawFn, ud)),
            image: Some((stamp_image as ImageFn, ud)),
            ..Callbacks::default()
        };

        assert_eq!(d.send_subscription(d.mask(), LONG_WAIT), Ok(true));
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(d);

        assert_eq!(
            stamps,
            Stamps {
                head: 11,
                gaze: 12,
                presence: 13,
                gaze_origin: 14,
                eye_position: 15,
                user_position_guide: 15,
                gaze_data: (7, 16),
                gaze_raw: 8,
                image: 17,
            }
        );
    }

    unsafe extern "C" fn keep_gaze_data(p: *const GazeData, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Option<GazeData>` as `ud`, which
        // outlives the device; `p` is a live sample for the call.
        unsafe { *ud.cast::<Option<GazeData>>() = Some(*p) };
    }

    /// Gaze data's pupil diameters reach the application as the daemon sent
    /// them: a valid one, and an invalid one still passed on, as the Stream
    /// Engine does.
    #[test]
    #[allow(clippy::float_cmp)] // reason: the values go through unchanged
    fn gaze_data_hands_on_the_pupil_diameters() {
        let eye = |pupil_valid, pupil_diameter_mm| tobii_ipc::GazeDataEye {
            pupil_valid,
            pupil_diameter_mm,
            ..tobii_ipc::GazeDataEye::default()
        };
        let frame = tobii_ipc::encode_gaze_data(&tobii_ipc::GazeData {
            left: eye(true, 6.25),
            right: eye(false, 6.0),
            ..tobii_ipc::GazeData::default()
        });
        // Sent ahead of the ack, so that it waits for `process`.
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![frame.clone(), encode_subscribed(true)],
            _ => vec![],
        });
        let d = Device::new(connect, 1, 1).expect("device");
        let mut seen: Option<GazeData> = None;
        let ud = (&raw mut seen).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.gaze_data, Some(keep_gaze_data as GazeDataFn), ud),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(d);

        let data = seen.expect("gaze data");
        assert_eq!(
            (data.left.pupil_validity, data.left.pupil_diameter_mm),
            (TOBII_VALIDITY_VALID, 6.25)
        );
        assert_eq!(
            (data.right.pupil_validity, data.right.pupil_diameter_mm),
            (TOBII_VALIDITY_INVALID, 6.0)
        );
    }

    unsafe extern "C" fn reenter(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Status` as `ud`.
        let out = unsafe { &mut *ud.cast::<Status>() };
        // SAFETY: a null handle is never dereferenced; the guard answers first.
        *out = match unsafe { device_ref(ptr::null_mut()) } {
            Err(s) => s,
            Ok(_) => 0,
        };
    }

    #[test]
    fn calls_from_inside_a_callback_are_refused() {
        let d = device_with(0, vec![]);
        let mut seen: Status = -1;
        let ud = (&raw mut seen).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(reenter as EyePairFn), ud),
            0
        );

        d.deliver(&ServerMsg::GazeOrigin(tobii_ipc::EyePair::default()));

        assert_eq!(seen, TOBII_ERROR_CALLBACK_IN_PROGRESS);
        assert!(!in_callback());
    }

    #[test]
    fn notifications_fill_the_520_byte_union() {
        let n = notification(&tobii_ipc::Notification {
            kind: tobii_ipc::notification::CALIBRATION_ID_CHANGED,
            value: WireValue::Uint(0x7186_ba7d),
        });
        assert_eq!(
            (n.type_, n.value_type),
            (8, crate::types::TOBII_NOTIFICATION_VALUE_TYPE_UINT)
        );
        // SAFETY: `value_type` says `uint_` is the active field.
        assert_eq!(unsafe { n.value.uint_ }, 0x7186_ba7d);

        let s = notification(&tobii_ipc::Notification {
            kind: 10,
            value: WireValue::String("x".repeat(600)),
        });
        // SAFETY: `value_type` says `string_` is the active field.
        let bytes = unsafe { s.value.string_ };
        assert_eq!(bytes[510], c_char_of(b'x'));
        assert_eq!(bytes[511], 0, "truncated and terminated");
    }

    unsafe extern "C" fn keep_notification(p: *const Notification, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Vec<Notification>` as `ud`,
        // which outlives the device; `p` is live for the call.
        unsafe { (*ud.cast::<Vec<Notification>>()).push(*p) };
    }

    /// The tracker's fault and warning lists, as the daemon sends them,
    /// reach the notifications callback through `process` as strings of
    /// types 10 (`FAULTS_CHANGED`) and 11 (`WARNINGS_CHANGED`). A regression
    /// check of the dispatch path, which passes any kind through unchanged;
    /// the cut to 511 bytes is `notifications_fill_the_520_byte_union`'s.
    #[test]
    fn fault_and_warning_lists_reach_the_notifications_callback() {
        let d = device_with(0, vec![]);
        let mut seen: Vec<Notification> = Vec::new();
        let ud = (&raw mut seen).cast::<c_void>();
        assert_eq!(
            d.subscribe(
                |c| &mut c.notifications,
                Some(keep_notification as NotificationsFn),
                ud
            ),
            TOBII_ERROR_NO_ERROR
        );
        for (kind, text) in [
            (tobii_ipc::notification::FAULTS_CHANGED, "FAULT_A"),
            (tobii_ipc::notification::WARNINGS_CHANGED, "ok"),
        ] {
            let body = tobii_ipc::encode_notification(&tobii_ipc::Notification {
                kind,
                value: WireValue::String(text.to_owned()),
            });
            d.queue(decode_server(&body).expect("decodes"));
        }

        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(d);

        let [faults, warnings] = &seen[..] else {
            panic!("{} notifications", seen.len());
        };
        let string = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_STRING;
        assert_eq!((faults.type_, faults.value_type), (10, string));
        assert_eq!((warnings.type_, warnings.value_type), (11, string));
        // SAFETY: `value_type` says `string_` is the active field.
        let (faults, warnings) = unsafe { (faults.value.string_, warnings.value.string_) };
        assert_eq!(faults[..8], b"FAULT_A\0".map(c_char_of));
        assert_eq!(warnings[..3], b"ok\0".map(c_char_of));
    }

    fn c_char_of(b: u8) -> std::ffi::c_char {
        std::ffi::c_char::from_ne_bytes([b])
    }

    /// A `CALIBRATION_ID_CHANGED` notification frame naming calibration `id`.
    fn calibration_id(id: u32) -> Vec<u8> {
        tobii_ipc::encode_notification(&tobii_ipc::Notification {
            kind: tobii_ipc::notification::CALIBRATION_ID_CHANGED,
            value: WireValue::Uint(id),
        })
    }

    /// The calibration ids `seen`, gathered by [`keep_notification`], names,
    /// in the order they came; every one must be a `CALIBRATION_ID_CHANGED`.
    fn calibration_ids(seen: &[Notification]) -> Vec<u32> {
        let kind = u32::from(tobii_ipc::notification::CALIBRATION_ID_CHANGED);
        let uint = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_UINT;
        seen.iter()
            .map(|n| {
                assert_eq!((n.type_, n.value_type), (kind, uint));
                // SAFETY: `value_type` says `uint_` is the active field.
                unsafe { n.value.uint_ }
            })
            .collect()
    }

    /// A reconnect's swap keeps, of what the old link brought, the
    /// notifications, in the order they came (those a look of `wait` took,
    /// then those still in the channel), ahead of the new link's, and drops
    /// its samples. It fails should the swap drop the old link's
    /// notifications, as a swap that clears does, as every swap did before
    /// (3bbcd3e only widened what it dropped to all the old link brought
    /// during the new one's subscription), or keep a sample. The old link's
    /// sender is dropped first, as its reader's is once the reconnect has
    /// closed that link.
    #[test]
    fn a_swap_keeps_the_old_links_notifications_ahead_of_the_new_links() {
        let msg = |body: Vec<u8>| decode_server(&body).expect("decodes");
        let (old, old_samples) = mpsc::channel();
        let mut dispatch = Dispatch::new(old_samples);
        dispatch
            .pending
            .extend([msg(calibration_id(1)), gaze_origin_sample()]);
        for body in [
            sample(STREAM_GAZE_ORIGIN, 1),
            calibration_id(2),
            sample(STREAM_PRESENCE, 2),
        ] {
            old.send(msg(body)).expect("the old link's");
        }
        drop(old);
        let (new, new_samples) = mpsc::channel();
        for body in [calibration_id(3), sample(STREAM_GAZE_ORIGIN, 3)] {
            new.send(msg(body)).expect("the new link's");
        }

        dispatch.swap_samples(new_samples);
        dispatch.take_all();

        assert_eq!(
            dispatch.pending,
            [
                msg(calibration_id(1)),
                msg(calibration_id(2)),
                msg(calibration_id(3)),
                msg(sample(STREAM_GAZE_ORIGIN, 3)),
            ]
        );
        assert_eq!(dispatch.state, LinkState::Up);
        drop(new);
    }

    #[test]
    fn reconnect_restores_the_subscription() {
        let d = device_with(0, vec![]);
        let ud = ptr::null_mut();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(d.reconnect(), 0);
        assert_eq!(d.mask(), STREAM_GAZE_ORIGIN);
        d.clear_buffers();
        assert!(!d.wait(Duration::from_millis(10)));
    }

    /// Far longer than any wait answered at once takes.
    const LONG_WAIT: Duration = Duration::from_secs(5);
    /// What "at once" means here: well under `LONG_WAIT`.
    pub(crate) const PROMPT: Duration = Duration::from_secs(1);
    /// A wait that is meant to run out.
    const SHORT_WAIT: Duration = Duration::from_millis(50);

    /// A daemon stand-in the test drives by hand. Each connect makes a socket
    /// pair, writes the next entry of `greetings` (whole frame bodies) into
    /// the daemon's end and hands that end to the test, which reads what the
    /// client sent and hangs up by dropping it. A greeting can hold the ack
    /// of the subscription the client is about to send: it waits in the
    /// socket until the client asks. Once the test drops the receiver, a
    /// connect fails as if nothing listened. A read of a frame the client
    /// never sends fails after `LONG_WAIT` rather than hanging the suite.
    pub(crate) fn scripted_daemon(
        greetings: Vec<Vec<Vec<u8>>>,
    ) -> (Connector, Receiver<UnixStream>) {
        let (tx, daemons) = mpsc::channel();
        let mut greetings = VecDeque::from(greetings);
        let connect: Connector = Box::new(move || {
            let (client, mut daemon) = UnixStream::pair()?;
            daemon.set_read_timeout(Some(LONG_WAIT))?;
            for body in greetings.pop_front().unwrap_or_default() {
                write_frame(&mut daemon, &body)?;
            }
            tx.send(daemon)
                .map_err(|_| io::Error::from(io::ErrorKind::ConnectionRefused))?;
            Ok(client)
        });
        (connect, daemons)
    }

    /// A subscription ack followed by `samples` gaze-origin samples.
    fn ack_then_gaze_origin(samples: usize) -> Vec<Vec<u8>> {
        let sample = encode_gaze_origin(&tobii_ipc::EyePair::default());
        let mut frames = vec![encode_subscribed(true)];
        frames.extend(std::iter::repeat_n(sample, samples));
        frames
    }

    /// The stream mask of the next frame the client sent, if it is a
    /// subscription.
    fn subscription(daemon: &mut UnixStream) -> Option<u32> {
        let body = read_frame(daemon).expect("read").expect("a frame");
        tobii_ipc::decode_subscribe(&body)
    }

    /// Wait for the reader to see the daemon hang up, so everything the
    /// daemon sent is in the channels before the test looks.
    fn hung_up(d: &Device) {
        let reader = d.command.lock().link.reader.take();
        if let Some(h) = reader {
            h.join().expect("reader");
        }
    }

    /// A device subscribed to gaze origin, `ud` counting the deliveries,
    /// whose daemon acked, sent `samples` samples and hung up. Later connects
    /// get `later`, one greeting each; their daemon ends come out of the
    /// receiver.
    pub(crate) fn lost_device(
        samples: usize,
        later: Vec<Vec<Vec<u8>>>,
        ud: *mut c_void,
    ) -> (Device, Receiver<UnixStream>) {
        let mut greetings = vec![ack_then_gaze_origin(samples)];
        greetings.extend(later);
        let (connect, daemons) = scripted_daemon(greetings);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut daemon = daemons.recv().expect("daemon end");
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        drop(daemon);
        hung_up(&d);
        (d, daemons)
    }

    #[test]
    fn process_delivers_what_arrived_then_reports_the_loss() {
        let mut hits = 0u32;
        let (d, _daemons) = lost_device(2, vec![], (&raw mut hits).cast());

        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(hits, 2, "both samples came before the hang-up");
        assert_eq!(
            d.process(),
            TOBII_ERROR_CONNECTION_FAILED,
            "until a reconnect"
        );
        assert_eq!(hits, 2);
    }

    #[test]
    fn wait_wakes_for_a_loss_until_process_reports_it_then_sleeps() {
        let mut hits = 0u32;
        let (d, _daemons) = lost_device(0, vec![], (&raw mut hits).cast());

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT), "the loss is something to process");
        assert!(d.wait(LONG_WAIT), "until process has reported it");
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        for _ in 0..2 {
            let t = Instant::now();
            assert!(!d.wait(SHORT_WAIT), "reported: nothing left to process");
            assert!(t.elapsed() >= SHORT_WAIT, "slept out the timeout");
        }
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    #[test]
    fn clearing_the_buffers_does_not_hide_the_loss() {
        let mut hits = 0u32;
        let (d, _daemons) = lost_device(1, vec![], (&raw mut hits).cast());

        d.clear_buffers();

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT));
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(hits, 0, "cleared, not delivered");
    }

    /// A device subscribed to gaze origin, `ud` counting the deliveries,
    /// whose daemon acked, sent one sample and stopped reading. The daemon's
    /// end, returned, stays open, so the reader sees nothing, but a write
    /// fails.
    pub(crate) fn deaf_daemon_device(ud: *mut c_void) -> (Device, UnixStream) {
        let (connect, daemons) = scripted_daemon(vec![ack_then_gaze_origin(1)]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut daemon = daemons.recv().expect("daemon end");
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        daemon.shutdown(Shutdown::Read).expect("shutdown");
        (d, daemon)
    }

    /// A write fails once the daemon stops reading, although its end is
    /// still open and the reader has seen nothing: the loss is reported all
    /// the same, after the sample that came first.
    #[test]
    fn a_failed_write_loses_the_connection_while_the_reader_still_runs() {
        let mut hits = 0u32;
        let (d, daemon) = deaf_daemon_device((&raw mut hits).cast());

        assert_eq!(
            d.request(tobii_ipc::request::kind::TRACK_BOX, &[], LONG_WAIT),
            Err(TOBII_ERROR_CONNECTION_FAILED)
        );

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT));
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(hits, 1);
        drop(daemon);
    }

    /// The requests, the recenter frame and the subscription failing at once
    /// held before losses were reported, and are kept as a regression guard.
    /// New is that they leave the loss for `process` to report.
    #[test]
    fn requests_and_new_subscriptions_on_a_lost_connection_fail_at_once() {
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();
        let (d, _daemons) = lost_device(0, vec![], ud);

        let t = Instant::now();
        assert_eq!(
            d.request(tobii_ipc::request::kind::TRACK_BOX, &[], LONG_WAIT),
            Err(TOBII_ERROR_CONNECTION_FAILED)
        );
        assert_eq!(
            d.send(&tobii_ipc::encode_recenter()),
            Err(TOBII_ERROR_CONNECTION_FAILED),
            "what tobii_recenter sends"
        );
        assert_eq!(
            d.subscribe(|c| &mut c.eye_position, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_CONNECTION_FAILED
        );
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.mask(), STREAM_GAZE_ORIGIN, "rolled back");
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    #[test]
    fn reconnect_restores_the_subscription_and_samples_flow_again() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(1)], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        let t = Instant::now();
        assert!(d.wait(LONG_WAIT));
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        assert_eq!(hits, 1);
        let t = Instant::now();
        assert!(!d.wait(SHORT_WAIT), "a quiet connection that is up");
        assert!(t.elapsed() >= SHORT_WAIT);
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(daemon);
    }

    #[test]
    fn a_second_loss_after_a_reconnect_wakes_wait_again() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(0)], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);

        drop(daemon);
        hung_up(&d);

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT), "the new loss is something to process");
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    /// A ring that comes after the waiter read the count, but before it
    /// sleeps, still wakes it: the count has moved past what it read. A
    /// condvar waited on bare would have missed that notify and slept out
    /// its timeout.
    #[test]
    fn a_ring_before_the_sleep_is_not_missed() {
        let bell = Doorbell::default();
        let seen = bell.rings();
        bell.ring();

        let t = Instant::now();
        assert!(bell.wait_past(seen, LONG_WAIT));
        assert!(t.elapsed() < PROMPT);
        let t = Instant::now();
        assert!(!bell.wait_past(bell.rings(), SHORT_WAIT), "no ring since");
        assert!(t.elapsed() >= SHORT_WAIT);
    }

    /// A sample queued right after a look found nothing ends the sleep that
    /// follows at once: the count is read before each look, not after. Here
    /// the first look rings, as the reader would just after it; were the
    /// count read after the look, that ring would be slept through.
    #[test]
    fn a_ring_right_after_a_look_ends_the_sleep_that_follows() {
        let bell = Doorbell::default();
        let mut looks = 0u32;

        let t = Instant::now();
        let found = bell.wait_for(LONG_WAIT, || {
            looks += 1;
            if looks == 1 {
                bell.ring();
            }
            looks > 1
        });

        assert!(found);
        assert!(t.elapsed() < PROMPT);
        assert_eq!(looks, 2);
    }

    /// Long enough for a wait to have gone to sleep.
    const ASLEEP: Duration = Duration::from_millis(100);

    /// A device subscribed to gaze origin, `ud` counting the deliveries, and
    /// its daemon's end, which has read the subscription and sent nothing
    /// since the ack.
    fn subscribed_device(ud: *mut c_void) -> (Device, UnixStream) {
        let (connect, daemons) = scripted_daemon(vec![ack_then_gaze_origin(0)]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut daemon = daemons.recv().expect("daemon end");
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        (d, daemon)
    }

    /// Hand the daemon's end to a thread that, for each delay sent to it,
    /// sleeps that long and writes one gaze-origin sample. It gives the end
    /// back once the sender is dropped.
    fn sampler(mut daemon: UnixStream) -> (Sender<Duration>, JoinHandle<UnixStream>) {
        let (go, delays) = mpsc::channel();
        let sampler = thread::spawn(move || {
            let sample = encode_gaze_origin(&tobii_ipc::EyePair::default());
            for delay in delays {
                thread::sleep(delay);
                write_frame(&mut daemon, &sample).expect("sample");
            }
            daemon
        });
        (go, sampler)
    }

    /// A sample that comes while a wait sleeps wakes it at once, and
    /// process delivers it. A guard: the wait blocked on the samples channel
    /// before, which woke it as promptly.
    #[test]
    fn a_sample_wakes_a_wait_asleep() {
        let mut hits = 0u32;
        let (d, daemon) = subscribed_device((&raw mut hits).cast());
        let (go, sampler) = sampler(daemon);
        go.send(ASLEEP).expect("go");

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT));
        assert!(t.elapsed() < PROMPT);

        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(go);
        let daemon = sampler.join().expect("sampler");
        drop((d, daemon));
        assert_eq!(hits, 1);
    }

    /// However soon after the wait starts a sample comes (before its look,
    /// between the look and the sleep, or during the sleep), the wait wakes
    /// for it: no ring is lost. A guard, a thousand rounds with the delay
    /// varied, from the reader's ring to the wait's wake; the window a wrong
    /// order of reading the count and looking would open is too narrow for
    /// it, and `a_ring_right_after_a_look_ends_the_sleep_that_follows` pins
    /// that order.
    #[test]
    fn no_sample_is_slept_through_whenever_it_comes() {
        const ROUNDS: u32 = 1000;
        let delays = [0, 0, 20, 50, 100, 200, 400].map(Duration::from_micros);
        let mut hits = 0u32;
        let (d, daemon) = subscribed_device((&raw mut hits).cast());
        let (go, sampler) = sampler(daemon);

        for (round, delay) in (0..ROUNDS).zip(delays.iter().cycle()) {
            go.send(*delay).expect("go");
            let t = Instant::now();
            assert!(d.wait(LONG_WAIT), "round {round}");
            assert!(t.elapsed() < PROMPT, "round {round}");
            assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        }

        drop(go);
        let daemon = sampler.join().expect("sampler");
        drop((d, daemon));
        assert_eq!(hits, ROUNDS, "one sample a round");
    }

    /// A daemon that hangs up while a wait sleeps wakes it at once: the
    /// reader hangs up its channels, then rings. Every wait answers at once
    /// until process reports the loss, and then one sleeps out its timeout,
    /// so a wait-and-process loop wakes once. A guard: the wait blocked on
    /// the samples channel before, whose disconnect woke it as promptly.
    #[test]
    fn a_loss_wakes_a_wait_asleep_once() {
        let mut hits = 0u32;
        let (d, daemon) = subscribed_device((&raw mut hits).cast());
        let hang_up = thread::spawn(move || {
            thread::sleep(ASLEEP);
            drop(daemon);
        });

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT), "woken by the loss");
        assert!(d.wait(LONG_WAIT), "until process has reported it");
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);

        let t = Instant::now();
        assert!(!d.wait(SHORT_WAIT), "reported: nothing left to process");
        assert!(t.elapsed() >= SHORT_WAIT, "slept out the timeout");
        hang_up.join().expect("daemon");
    }

    /// Rings that bring nothing to process (rung here by hand, as the ring
    /// of a sample an earlier look took already would be) neither end a
    /// wait nor stretch it: it looks, sleeps again, and gives up at its own
    /// deadline. So too on a reported loss, where answering at once would
    /// spin a wait-and-process loop.
    #[test]
    fn a_wait_keeps_its_deadline_through_rings_that_bring_nothing() {
        let mut hits = 0u32;
        let quiet = Device::new(fake_daemon(|_| vec![]), 1, 1).expect("device");
        let (lost, _daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        assert_eq!(lost.process(), TOBII_ERROR_CONNECTION_FAILED);

        for d in [quiet, lost] {
            // Rings every few ms until told to stop, or for `LONG_WAIT`
            // should the wait never end.
            let bell = Arc::clone(&d.doorbell);
            let (stop, stopped) = mpsc::channel::<()>();
            let ringer = thread::spawn(move || {
                let until = Instant::now() + LONG_WAIT;
                while Instant::now() < until
                    && stopped.recv_timeout(Duration::from_millis(5))
                        == Err(RecvTimeoutError::Timeout)
                {
                    bell.ring();
                }
            });

            let t = Instant::now();
            let woke = d.wait(SHORT_WAIT);
            let took = t.elapsed();

            drop(stop);
            ringer.join().expect("ringer");
            assert!(!woke, "{d:?}");
            assert!((SHORT_WAIT..PROMPT).contains(&took), "{d:?} {took:?}");
        }
    }

    /// A reconnect's new link rings the device's own doorbell, the one a
    /// wait on the reported loss sleeps on: when another thread reconnects,
    /// that ring is what wakes such a wait for the new link's samples. The
    /// count moves by two, the sample's ring and the one the reconnect makes
    /// once the link is in place. Structural: it fails should a link ring a
    /// doorbell of its own, which would leave the reconnect's ring alone on
    /// the device's. The two-thread case is
    /// `a_wait_on_a_reported_loss_wakes_for_a_reconnect_on_another_thread`.
    #[test]
    fn a_reconnects_samples_ring_the_doorbell_a_lost_wait_sleeps_on() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(1)], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        let bell = Arc::clone(&d.doorbell);
        let seen = bell.rings();

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let t = Instant::now();
        assert!(bell.wait_past(seen + 1, LONG_WAIT), "the new link's sample");
        assert!(t.elapsed() < PROMPT);
        assert_eq!(bell.rings(), seen + 2, "the sample's ring and the swap's");
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert!(d.wait(LONG_WAIT));
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop((d, daemon));
        assert_eq!(hits, 1);
    }

    /// A reconnect rings the doorbell once it has put its new link in place,
    /// though that link brings no sample: a wait on another thread that the
    /// new link's first samples woke while the reconnect still waited for
    /// its ack was turned away by the dispatch lock the reconnect holds, and
    /// slept again; this ring sends it back to look at the new link. The
    /// greeting here is the ack alone, and the old link's reader stopped
    /// before the count was read, so the reconnect's ring is the only one.
    #[test]
    fn a_reconnect_rings_the_doorbell_once_its_new_link_is_in_place() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(0)], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        let seen = d.doorbell.rings();

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        assert_eq!(d.doorbell.rings(), seen + 1);
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        drop((d, daemon));
    }

    /// The reader hangs up its channels before its last ring, so the wait
    /// that ring wakes finds the samples channel disconnected, not empty.
    /// The test holds the doorbell's lock, so once the daemon hangs up the
    /// reader blocks in that ring, and the channel is disconnected all the
    /// same. Were the ring first, the reader would block in it with its
    /// senders still up, and the channel would stay open until the lock is
    /// let go.
    #[test]
    fn the_reader_hangs_up_before_its_last_ring() {
        let mut hits = 0u32;
        let (d, daemon) = subscribed_device((&raw mut hits).cast());
        let held = lock(&d.doorbell.rings);

        drop(daemon);

        assert_eq!(
            lock(&d.dispatch).samples.recv_timeout(PROMPT).err(),
            Some(RecvTimeoutError::Disconnected),
            "hung up while the ring waits for the lock"
        );
        drop(held);
        let t = Instant::now();
        assert!(d.wait(LONG_WAIT), "the loss");
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    /// The reader shuts the connection down when it stops, so it closes
    /// however its loss is found: here the daemon only half-closes, and no
    /// process or wait comes to notice, yet the daemon's read ends.
    #[test]
    fn the_reader_closes_the_connection_when_it_stops() {
        let mut hits = 0u32;
        let (d, mut daemon) = subscribed_device((&raw mut hits).cast());
        daemon.set_read_timeout(Some(PROMPT)).expect("read timeout");

        daemon.shutdown(Shutdown::Write).expect("half-close");

        assert!(
            matches!(read_frame(&mut daemon), Ok(None)),
            "the client closed its end"
        );
        drop(d);
    }

    /// Only samples ring: an answer is nothing for `wait`, and a ring for
    /// one would only wake it to look and sleep again. The reader handles
    /// frames in order, so the ack's ring, the sample's and the stale
    /// reply's would all come before the reply the request returns with.
    #[test]
    fn answers_ring_no_doorbell() {
        let connect = fake_daemon(|body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![encode_subscribed(true)],
            Some(&tobii_ipc::TAG_REQUEST) => {
                let req = tobii_ipc::request::decode_request(body).expect("request");
                vec![
                    encode_gaze(1, true, [0.5; 2], [f32::NAN; 2]),
                    encode_reply(req.id.wrapping_add(7), 0, b"stale"),
                    encode_reply(req.id, 0, b"answer"),
                ]
            }
            _ => vec![],
        });
        let d = Device::new(connect, 1, 1).expect("device");
        let before = d.doorbell.rings();

        assert_eq!(d.send_subscription(STREAM_GAZE, LONG_WAIT), Ok(true));
        assert_eq!(d.request(1, &[], LONG_WAIT), Ok(b"answer".to_vec()));

        assert_eq!(d.doorbell.rings(), before + 1, "the sample's ring alone");
    }

    /// Wait out `SHORT_WAIT` and say no, as for a loss already reported, then
    /// report the loss still: the device is lost, and was not woken again.
    fn assert_still_lost(d: &Device) {
        let t = Instant::now();
        assert!(!d.wait(SHORT_WAIT), "no second wake for the same loss");
        assert!(t.elapsed() >= SHORT_WAIT, "slept out the timeout");
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    /// A regression guard: the scripted connector fails at once, as
    /// `tobii_ipc::connect` does, and this held before. `first_then`'s test
    /// below shows the order it calls its connectors in; that
    /// `connect_daemon` passes `tobii_ipc::connect` as `later` is by
    /// inspection, since no test may reach the real socket.
    #[test]
    fn reconnecting_while_nothing_listens_fails_at_once_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        drop(daemons);

        let t = Instant::now();
        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        assert!(t.elapsed() < PROMPT);

        assert_still_lost(&d);
    }

    /// The daemon takes the connection and the subscription but never acks:
    /// the reconnect gives up after its own short timeout, not the 2 s a
    /// subscription change waits, with `TOBII_ERROR_CONNECTION_FAILED`, never
    /// `TOBII_ERROR_TIMED_OUT`, and hangs up on that daemon.
    #[test]
    fn a_reconnect_the_daemon_does_not_ack_fails_soon_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        // The second connection's daemon is sent nothing to answer with.
        let (d, daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);

        let t = Instant::now();
        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        let took = t.elapsed();

        assert!(
            (RECONNECT_ACK_TIMEOUT..SUBSCRIBE_ACK_TIMEOUT).contains(&took),
            "{took:?}"
        );
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert!(
            matches!(read_frame(&mut daemon), Ok(None)),
            "the attempt's connection is closed"
        );
        assert_still_lost(&d);
    }

    #[test]
    fn a_reconnect_the_daemon_hangs_up_on_fails_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        // Reads the subscription, then hangs up without acking it.
        let daemon = thread::spawn(move || {
            let mut daemon = daemons.recv().expect("second daemon end");
            subscription(&mut daemon)
        });

        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);

        assert_eq!(daemon.join().expect("daemon"), Some(STREAM_GAZE_ORIGIN));
        assert_still_lost(&d);
    }

    /// No daemon refuses a subscription today, but only an ack that takes
    /// them counts: a reconnect never reports success for streams the
    /// daemon will not serve.
    #[test]
    fn a_reconnect_the_daemon_refuses_fails_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        let refusal = vec![encode_subscribed(false)];
        let (d, daemons) = lost_device(0, vec![refusal], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);

        let t = Instant::now();
        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        assert!(t.elapsed() < PROMPT);

        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert!(
            matches!(read_frame(&mut daemon), Ok(None)),
            "the attempt's connection is closed"
        );
        assert_eq!(d.mask(), STREAM_GAZE_ORIGIN, "kept for a retry");
        assert_still_lost(&d);
    }

    /// Were the old connection closed first, the daemon could see no client
    /// wanting the streams between the two and stop the tracker.
    #[test]
    fn reconnecting_a_live_connection_subscribes_the_new_one_before_closing_the_old() {
        let (connect, daemons) = scripted_daemon(vec![ack_then_gaze_origin(0)]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut old = daemons.recv().expect("daemon end");
        let ud = ptr::null_mut();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(ignore_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(subscription(&mut old), Some(STREAM_GAZE_ORIGIN));
        // The new daemon answers from its own thread, so it can look at the
        // old connection while the client waits for the ack.
        let daemon = thread::spawn(move || {
            let mut new = daemons.recv().expect("new daemon end");
            let mask = subscription(&mut new);
            old.set_read_timeout(Some(SHORT_WAIT))
                .expect("read timeout");
            let old_open = matches!(
                read_frame(&mut old),
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut)
            );
            write_frame(&mut new, &encode_subscribed(true)).expect("ack");
            (mask, old_open, old, new)
        });

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let (mask, old_open, mut old, new) = daemon.join().expect("daemon");
        assert_eq!(mask, Some(STREAM_GAZE_ORIGIN));
        assert!(
            old_open,
            "the old connection was up while the new subscribed"
        );
        assert!(
            matches!(read_frame(&mut old), Ok(None)),
            "and closed once it had"
        );
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(new);
    }

    /// The acks owed are the old connection's: a reconnect's new connection
    /// owes none, and takes the first ack it gets for its own. A guard: the
    /// count is kept with the link, so a new one starts at none by
    /// construction, and this keeps it so should the count move off it.
    #[test]
    fn a_reconnect_owes_none_of_the_old_connections_acks() {
        let (connect, daemons) = scripted_daemon(vec![vec![], ack_then_gaze_origin(0)]);
        let d = Device::new(connect, 1, 1).expect("device");
        let old = daemons.recv().expect("daemon end");
        lock(&d.callbacks).gaze_origin = Some((ignore_pair as EyePairFn, ptr::null_mut()));
        assert_eq!(
            d.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
            Err(TOBII_ERROR_TIMED_OUT),
            "the old connection owes its ack"
        );

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let mut new = daemons.recv().expect("new daemon end");
        assert_eq!(subscription(&mut new), Some(STREAM_GAZE_ORIGIN));
        drop((old, new));
    }

    unsafe extern "C" fn ignore_pair(_p: *const EyePair, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_head(_p: *const HeadPose, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_gaze(_p: *const GazePoint, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_presence(_s: PresenceStatus, _ts: i64, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_gaze_data(_p: *const GazeData, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_gaze_raw(_p: *const GazeRaw, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_image(_p: *const Image, _ud: *mut c_void) {}
    unsafe extern "C" fn ignore_notification(_p: *const Notification, _ud: *mut c_void) {}

    #[test]
    fn a_reconnect_restores_every_stream_and_fetches_the_device_info_again() {
        use std::sync::{Arc, Mutex};
        use tobii_ipc::request::{DeviceInfo, decode_request, encode_device_info};
        // Acks every subscription, recording its mask, and gives each device
        // info request the next serial number.
        let masks = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&masks);
        let mut serials = 0u32;
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                log.lock()
                    .expect("log")
                    .push(tobii_ipc::decode_subscribe(body));
                vec![encode_subscribed(true)]
            }
            Some(&tobii_ipc::TAG_REQUEST) => {
                let req = decode_request(body).expect("request");
                serials += 1;
                let info = DeviceInfo {
                    serial_number: serials.to_string(),
                    ..DeviceInfo::default()
                };
                vec![encode_reply(req.id, 0, &encode_device_info(&info))]
            }
            _ => vec![],
        });
        let d = Device::new(connect, 1, 1).expect("device");
        let ud = ptr::null_mut();
        *lock(&d.callbacks) = Callbacks {
            head: Some((ignore_head as HeadPoseFn, ud)),
            gaze: Some((ignore_gaze as GazePointFn, ud)),
            presence: Some((ignore_presence as PresenceFn, ud)),
            gaze_origin: Some((ignore_pair as EyePairFn, ud)),
            user_position_guide: Some((ignore_pair as EyePairFn, ud)),
            gaze_data: Some((ignore_gaze_data as GazeDataFn, ud)),
            gaze_raw: Some((ignore_gaze_raw as GazeRawFn, ud)),
            image: Some((ignore_image as ImageFn, ud)),
            notifications: Some((ignore_notification as NotificationsFn, ud)),
            ..Callbacks::default()
        };
        let serial = |d: &Device| d.device_info().map(|info| info.serial_number);
        assert_eq!(serial(&d), Ok("1".into()));
        assert_eq!(serial(&d), Ok("1".into()), "fetched once");

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let every = STREAM_HEAD
            | STREAM_GAZE
            | STREAM_PRESENCE
            | STREAM_GAZE_ORIGIN
            | STREAM_EYE_POSITION
            | STREAM_GAZE_DATA
            | STREAM_GAZE_RAW
            | STREAM_IMAGE
            | STREAM_NOTIFICATIONS;
        assert_eq!(*masks.lock().expect("log"), [Some(every)]);
        assert_eq!(serial(&d), Ok("2".into()), "asked the daemon again");
    }

    /// What `process` logs for a loss.
    const LOST: &str = "lost the connection to tobiid; tobii_device_reconnect restores it";

    /// `tobii_device_create` hands a connected device its API's logger
    /// through `adopt`, which says so at INFO, once.
    #[test]
    fn adopting_the_apis_logger_logs_the_connect() {
        let recorder = Recorder::default();
        let mut d = device_with(0, vec![]);

        d.adopt(recorder.logger());
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);

        assert_eq!(
            recorder.lines(),
            [(TOBII_LOG_LEVEL_INFO, "connected to tobiid".to_owned())]
        );
    }

    #[test]
    fn a_lost_connection_is_logged_once() {
        let (recorder, mut hits) = (Recorder::default(), 0u32);
        let (mut d, _daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        d.set_logger(recorder.logger());

        for _ in 0..3 {
            assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        }

        assert_eq!(recorder.lines(), [(TOBII_LOG_LEVEL_ERROR, LOST.to_owned())]);
    }

    #[test]
    fn each_failed_reconnect_is_logged() {
        let (recorder, mut hits) = (Recorder::default(), 0u32);
        let refusal = vec![encode_subscribed(false)];
        let (mut d, daemons) = lost_device(0, vec![refusal], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        d.set_logger(recorder.logger());

        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        drop(daemons);
        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);

        let lines = recorder.lines();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(
            lines
                .iter()
                .all(|(level, text)| *level == TOBII_LOG_LEVEL_ERROR
                    && text.starts_with("could not reconnect to tobiid: ")),
            "{lines:?}"
        );
        assert!(lines[0].1.ends_with("it refused them"), "{lines:?}");
    }

    /// One INFO line for the reconnect, and nothing for the samples that
    /// flow again after it.
    #[test]
    fn a_reconnect_is_logged_at_info() {
        let (recorder, mut hits) = (Recorder::default(), 0u32);
        let (mut d, daemons) =
            lost_device(0, vec![ack_then_gaze_origin(1)], (&raw mut hits).cast());
        d.set_logger(recorder.logger());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert!(d.wait(LONG_WAIT));
        for _ in 0..2 {
            assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(hits, 1);
        assert_eq!(
            recorder.lines(),
            [
                (TOBII_LOG_LEVEL_ERROR, LOST.to_owned()),
                (TOBII_LOG_LEVEL_INFO, "reconnected to tobiid".to_owned()),
            ]
        );
        drop(daemon);
    }

    /// A logger's context: the device it calls back into, and what that call
    /// returned.
    struct Reentry {
        device: *mut Device,
        got: Cell<Status>,
    }

    unsafe extern "C" fn process_from_the_logger(
        context: *mut c_void,
        _level: LogLevel,
        _text: *const c_char,
    ) {
        // SAFETY: the test passes a live `Reentry` as the context.
        let r = unsafe { &*context.cast::<Reentry>() };
        // SAFETY: the guard refuses the call before the device is read.
        r.got
            .set(unsafe { crate::api::tobii_device_process_callbacks(r.device) });
    }

    /// The logger runs under the callback guard, so a call from inside it
    /// into the device that logs is refused, as from a callback. Kept as
    /// the Stream Engine's rule for a callback, though no lock of the
    /// device's is held while the logger runs (see
    /// `a_logger_may_wait_for_another_thread_using_the_device`).
    #[test]
    fn a_logger_that_calls_back_in_is_refused() {
        let mut hits = 0u32;
        let (d, _daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        let d = Box::into_raw(Box::new(d));
        let reentry = Reentry {
            device: d,
            got: Cell::new(-1),
        };
        let context = std::ptr::from_ref(&reentry).cast_mut().cast();
        // SAFETY: `d` is live and destroyed once; `reentry` and `hits`
        // outlive it.
        unsafe {
            (*d).set_logger(crate::logger::tests::logger(
                process_from_the_logger,
                context,
            ));
            assert_eq!(
                crate::api::tobii_device_process_callbacks(d),
                TOBII_ERROR_CONNECTION_FAILED
            );
            assert_eq!(crate::api::tobii_device_destroy(d), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(reentry.got.get(), TOBII_ERROR_CALLBACK_IN_PROGRESS);
        assert!(!in_callback());
    }

    /// A device's first connection goes through `first` (`connect_daemon`
    /// passes `tobii_ipc::connect_or_spawn`, through
    /// [`connect_or_spawn_alone`]), and every reconnect through `later`
    /// (`tobii_ipc::connect`, which never spawns).
    #[test]
    fn first_then_uses_first_once_then_later() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (firsts, laters) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (f, l) = (Arc::clone(&firsts), Arc::clone(&laters));
        let connect = first_then(
            move || {
                f.fetch_add(1, Ordering::Relaxed);
                UnixStream::pair().map(|(client, _daemon)| client)
            },
            move || {
                l.fetch_add(1, Ordering::Relaxed);
                Err(io::ErrorKind::ConnectionRefused.into())
            },
        );
        let d = Device::new(connect, 1, 1).expect("device");

        for _ in 0..2 {
            assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        }

        let calls = (
            firsts.load(Ordering::Relaxed),
            laters.load(Ordering::Relaxed),
        );
        assert_eq!(calls, (1, 2));
    }

    /// How many threads create a device at once in the spawn tests below.
    const CREATORS: usize = 4;

    /// A stand-in for tobiid and for `tobii_ipc`'s connect and spawn, which
    /// counts the spawns: a daemon that listens from `STARTUP` after the
    /// first spawn on, if it `starts` at all.
    #[derive(Default)]
    struct Spawnable {
        starts: bool,
        first_spawn: std::sync::OnceLock<Instant>,
        spawns: AtomicUsize,
        /// The connects [`Spawnable::create`] has tried before looking at
        /// the lock, and after waiting on it.
        tried: AtomicUsize,
    }

    impl Spawnable {
        /// How long a spawned daemon takes to listen: far longer than the
        /// creating threads take to try their first connects.
        const STARTUP: Duration = Duration::from_millis(200);
        /// How long a spawn waits for its daemon to listen
        /// (`tobii_ipc::connect_or_spawn` waits about 3 s).
        const GIVE_UP: Duration = PROMPT;

        fn new(starts: bool) -> Arc<Self> {
            Arc::new(Self {
                starts,
                ..Self::default()
            })
        }

        /// As `tobii_ipc::connect`.
        fn connect(&self) -> io::Result<UnixStream> {
            let up = self
                .first_spawn
                .get()
                .is_some_and(|t| self.starts && t.elapsed() >= Self::STARTUP);
            if up {
                UnixStream::pair().map(|(client, _daemon)| client)
            } else {
                Err(io::ErrorKind::ConnectionRefused.into())
            }
        }

        /// As `tobii_ipc::connect_or_spawn`: connect, else spawn and retry
        /// until the daemon listens or `GIVE_UP` has passed. Past `GIVE_UP`
        /// it also waits, for up to a `PROMPT` more, until `CREATORS`
        /// connects have been tried, which, while it holds the lock, are the
        /// creating threads' first ones: a thread held up past `GIVE_UP`
        /// would otherwise find the lock free, and its attempt would count
        /// against the lock.
        fn connect_or_spawn(&self) -> io::Result<UnixStream> {
            if let Ok(stream) = self.connect() {
                return Ok(stream);
            }
            self.spawns.fetch_add(1, Ordering::Relaxed);
            let spawned = Instant::now();
            self.first_spawn.get_or_init(|| spawned);
            let trying = |waited: Duration| {
                waited < Self::GIVE_UP
                    || (self.tried.load(Ordering::Relaxed) < CREATORS
                        && waited < Self::GIVE_UP + PROMPT)
            };
            while trying(spawned.elapsed()) {
                thread::sleep(Duration::from_millis(5));
                if let Ok(stream) = self.connect() {
                    return Ok(stream);
                }
            }
            self.connect()
        }

        /// A device's first connection through `spawning`, as
        /// `connect_daemon` makes it.
        fn create(&self, spawning: &Mutex<()>) -> io::Result<UnixStream> {
            let connect = || {
                self.tried.fetch_add(1, Ordering::Relaxed);
                self.connect()
            };
            connect_or_spawn_alone(spawning, connect, || self.connect_or_spawn())
        }

        /// Have `CREATORS` threads make their first connection at once
        /// through `spawning`, and hand back what each got.
        fn create_at_once(&self, spawning: &Mutex<()>) -> Vec<io::Result<UnixStream>> {
            let start = std::sync::Barrier::new(CREATORS);
            thread::scope(|s| {
                let creators: Vec<_> = (0..CREATORS)
                    .map(|_| {
                        s.spawn(|| {
                            start.wait();
                            self.create(spawning)
                        })
                    })
                    .collect();
                creators.into_iter().map(joined).collect()
            })
        }
    }

    /// Threads creating devices at once while no daemon listens spawn one
    /// between them, and all connect to it. Fails without the lock: each
    /// thread finds nothing listening and spawns a daemon of its own.
    #[test]
    fn devices_created_together_while_no_daemon_runs_spawn_one() {
        let tobiid = Spawnable::new(true);

        let daemon = Arc::clone(&tobiid);
        let got = within(WATCHDOG, move || daemon.create_at_once(&Mutex::new(())));

        assert!(got.iter().all(Result::is_ok), "{got:?}");
        assert_eq!(tobiid.spawns.load(Ordering::Relaxed), 1);
    }

    /// When the spawned daemon never listens, the threads that waited for
    /// that spawn fail with it rather than each spawning in turn, and a
    /// device created once they have failed makes an attempt of its own: the
    /// lock serialises the attempts, it does not make one for good. Fails
    /// without the lock (a spawn per thread), with a lock that makes each
    /// waiting thread spawn again once it has it (a spawn per thread, each
    /// after the last has given up), and with a spawn made at most once per
    /// process (none for the later device, which then could never start
    /// tobiid again once it had stopped).
    #[test]
    fn a_failed_spawn_fails_the_threads_that_waited_but_not_a_later_device() {
        let tobiid = Spawnable::new(false);

        let daemon = Arc::clone(&tobiid);
        let (together, took, spawned, later) = within(WATCHDOG, move || {
            let spawning = Mutex::new(());
            let (together, took) = timed(|| daemon.create_at_once(&spawning));
            let spawned = daemon.spawns.load(Ordering::Relaxed);
            (together, took, spawned, daemon.create(&spawning))
        });

        assert!(together.iter().all(Result::is_err), "{together:?}");
        assert!(later.is_err(), "{later:?}");
        let spawns = (spawned, tobiid.spawns.load(Ordering::Relaxed));
        assert_eq!(spawns, (1, 2), "one for the threads, one for the later");
        assert!(
            took < Spawnable::GIVE_UP + PROMPT,
            "one spawn's wait: {took:?}"
        );
    }

    /// A device created while a daemon listens connects without the lock,
    /// even while another thread holds it for a spawn. A guard against
    /// locking too much: it passes without the lock too.
    #[test]
    fn a_device_created_while_a_daemon_listens_waits_for_no_spawn() {
        let tobiid = Spawnable::new(true);
        tobiid.connect_or_spawn().expect("the daemon listens");
        let spawning = Arc::new(Mutex::new(()));
        let _held = lock(&spawning);

        let (daemon, busy) = (Arc::clone(&tobiid), Arc::clone(&spawning));
        let got = within(PROMPT, move || daemon.create(&busy));

        assert!(got.is_ok(), "{got:?}");
        assert_eq!(tobiid.spawns.load(Ordering::Relaxed), 1, "the first alone");
    }

    // Threads sharing a device. Each test says whether it fails without the
    // lock split it pins, or is a guard against locking too much.

    /// What `f` gave, and how long it took.
    fn timed<R>(f: impl FnOnce() -> R) -> (R, Duration) {
        let t = Instant::now();
        let got = f();
        (got, t.elapsed())
    }

    /// A gaze-origin sample, as the reader queues it.
    fn gaze_origin_sample() -> ServerMsg {
        decode_server(&encode_gaze_origin(&tobii_ipc::EyePair::default())).expect("decodes")
    }

    /// The kind of request the daemon stand-ins below echo, answering with
    /// the payload it carried. They do not decode it, so any kind would do
    /// but a state request, which [`streaming_daemon`] answers as tobiid.
    const ECHOED: u8 = kind::DEVICE_INFO;

    /// The id of the next frame the client sent `daemon`, a request of
    /// `request_kind`.
    fn asked(daemon: &mut UnixStream, request_kind: u8) -> u32 {
        let body = read_frame(daemon).expect("read").expect("a frame");
        let req = tobii_ipc::request::decode_request(&body).expect("request");
        assert_eq!(req.kind, request_kind);
        req.id
    }

    /// `d` as a handle for the C entry points, which borrow it shared only.
    fn handle(d: &Device) -> *mut Device {
        ptr::from_ref(d).cast_mut()
    }

    /// Ask `d` through `tobii_get_state_bool` whether a calibration is
    /// active: the status, and the value written (7 if none was).
    fn calibration_active(d: &Device) -> (Status, u32) {
        let mut active = 7u32;
        // SAFETY: `d` is live for the call, which forms only a shared borrow
        // of it; `active` is a live local.
        let status = unsafe {
            crate::api::tobii_get_state_bool(
                handle(d),
                TOBII_STATE_CALIBRATION_ACTIVE,
                &raw mut active,
            )
        };
        (status, active)
    }

    /// A callback's user data that holds it inside until let go:
    /// [`Gate::hold`], which [`held_pair`] calls, counts each call, then
    /// waits for `open`, or `LONG_WAIT` at most, so a test that fails before
    /// letting it go cannot hang.
    #[derive(Debug, Default)]
    pub(crate) struct Gate {
        /// The calls so far, and whether the gate is open.
        state: Mutex<(u32, bool)>,
        changed: Condvar,
    }

    impl Gate {
        pub(crate) fn ud(&self) -> *mut c_void {
            ptr::from_ref(self).cast_mut().cast()
        }

        /// Whether the callback has been called `n` times, waiting up to
        /// `PROMPT` for it.
        pub(crate) fn entered(&self, n: u32) -> bool {
            let state = lock(&self.state);
            let (state, _) = self
                .changed
                .wait_timeout_while(state, PROMPT, |(calls, _)| *calls < n)
                .unwrap_or_else(PoisonError::into_inner);
            state.0 >= n
        }

        fn calls(&self) -> u32 {
            lock(&self.state).0
        }

        /// Let every call through, now and later.
        pub(crate) fn open(&self) {
            lock(&self.state).1 = true;
            self.changed.notify_all();
        }

        /// Count a call, then wait for `open`, or `LONG_WAIT` at most.
        pub(crate) fn hold(&self) {
            let mut state = lock(&self.state);
            state.0 += 1;
            self.changed.notify_all();
            drop(
                self.changed
                    .wait_timeout_while(state, LONG_WAIT, |(_, open)| !*open),
            );
        }
    }

    unsafe extern "C" fn held_pair(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the tests pass a live `Gate` as `ud`.
        unsafe { &*ud.cast::<Gate>() }.hold();
    }

    /// Run `f` on this thread while another thread's `process` on `d` is
    /// held inside the gaze-origin callback, `gate`'s (set here), for a
    /// sample queued for it; then let it go. What `f` gave, and what that
    /// process returned. `gate` must outlive `d`.
    fn while_a_callback_runs<R>(d: &Device, gate: &Gate, f: impl FnOnce() -> R) -> (R, Status) {
        lock(&d.callbacks).gaze_origin = Some((held_pair as EyePairFn, gate.ud()));
        d.queue(gaze_origin_sample());
        thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(gate.entered(1), "the callback runs on the other thread");
            let got = f();
            gate.open();
            (got, processing.join().expect("process"))
        })
    }

    /// A process call made while another thread dispatches the device
    /// returns at once, as the DLL's does when its try-enter fails
    /// (0x18000e9d9..0x18000e9ea): no error while the connection is up or
    /// its loss is still to be reported (by the call dispatching), and
    /// `TOBII_ERROR_CONNECTION_FAILED` once it has been, until a reconnect.
    /// It fails should process wait for the dispatch lock (the callback's
    /// `LONG_WAIT`), or answer from anything but the last report.
    #[test]
    fn a_busy_process_returns_at_once_with_what_the_last_report_said() {
        let gates: [Gate; 3] = Default::default();
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(0)], (&raw mut hits).cast());
        let busy = || timed(|| d.process());

        let ((unreported, took), dispatching) = while_a_callback_runs(&d, &gates[0], busy);
        assert_eq!(
            unreported, TOBII_ERROR_NO_ERROR,
            "the other call reports it"
        );
        assert!(took < PROMPT, "{took:?}");
        assert_eq!(dispatching, TOBII_ERROR_CONNECTION_FAILED);

        let ((reported, took), dispatching) = while_a_callback_runs(&d, &gates[1], busy);
        assert_eq!(reported, TOBII_ERROR_CONNECTION_FAILED);
        assert!(took < PROMPT, "{took:?}");
        assert_eq!(dispatching, TOBII_ERROR_CONNECTION_FAILED);

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        let ((mended, took), dispatching) = while_a_callback_runs(&d, &gates[2], busy);
        assert_eq!(mended, TOBII_ERROR_NO_ERROR, "reconnected");
        assert!(took < PROMPT, "{took:?}");
        assert_eq!(dispatching, TOBII_ERROR_NO_ERROR);
        assert_eq!(gates.each_ref().map(Gate::calls), [1; 3]);
        drop((d, daemon));
    }

    /// An unsubscribe waits for its callback to return when another
    /// thread's process is running it, as the DLL's waits on dev+0x4d8
    /// (0x180153580), and once it has returned that callback is not called
    /// again. It fails should unsubscribe take the slot without the
    /// callbacks lock.
    #[test]
    fn an_unsubscribe_waits_for_its_callback_running_on_another_thread() {
        let gate = Gate::default();
        let d = device_with(0, vec![]);
        assert_eq!(
            d.subscribe(
                |c| &mut c.gaze_origin,
                Some(held_pair as EyePairFn),
                gate.ud()
            ),
            TOBII_ERROR_NO_ERROR
        );
        d.queue(gaze_origin_sample());
        let d = &d;

        thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(gate.entered(1), "the callback runs on the other thread");
            let (tx, done) = mpsc::channel();
            s.spawn(move || tx.send(d.unsubscribe(|c| &mut c.gaze_origin)));

            assert_eq!(
                done.recv_timeout(ASLEEP),
                Err(RecvTimeoutError::Timeout),
                "it waits for the callback"
            );
            gate.open();
            assert_eq!(done.recv_timeout(PROMPT), Ok(TOBII_ERROR_NO_ERROR));
            assert_eq!(processing.join().expect("process"), TOBII_ERROR_NO_ERROR);
        });

        d.queue(gaze_origin_sample());
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        assert_eq!(gate.calls(), 1, "not called once the unsubscribe returned");
        assert_eq!(d.mask(), 0);
    }

    /// A guard against locking too much: while another thread's process
    /// runs a callback, a request on this thread goes through, as it needs
    /// the command lock alone, and the callback guard refuses only the
    /// callback's own thread.
    #[test]
    fn a_request_goes_through_while_another_thread_runs_a_callback() {
        let gate = Gate::default();
        let d = device_with(0, vec![1]);

        let ((got, took), processed) =
            while_a_callback_runs(&d, &gate, || timed(|| calibration_active(&d)));

        assert_eq!(got, (TOBII_ERROR_NO_ERROR, 1));
        assert!(took < PROMPT, "{took:?}");
        assert_eq!(processed, TOBII_ERROR_NO_ERROR);
    }

    /// A wait on a device another thread is dispatching neither blocks
    /// behind that dispatch nor answers at once, where the DLL answers at
    /// once (see `tobii_wait_for_callbacks`): it sleeps out its timeout on
    /// the doorbell. It fails should a look wait for the dispatch lock (it
    /// would take the callback's `LONG_WAIT`), or should a busy device count
    /// as something to process.
    #[test]
    fn a_wait_on_a_device_another_thread_dispatches_sleeps_out_its_timeout() {
        let gate = Gate::default();
        let d = device_with(0, vec![]);

        let ((woke, took), processed) =
            while_a_callback_runs(&d, &gate, || timed(|| d.wait(SHORT_WAIT)));

        assert!(!woke);
        assert!((SHORT_WAIT..PROMPT).contains(&took), "{took:?}");
        assert_eq!(processed, TOBII_ERROR_NO_ERROR);
    }

    /// A subscribe lets the callbacks lock go for its round trip, so
    /// another thread's process keeps delivering the streams already
    /// subscribed while it waits for the ack. The DLL holds dev+0x4d8 across
    /// it (0x18015371b..0x180153845), stalling every callback of the device
    /// for as long as the ack takes. It fails should subscribe hold the lock
    /// across the round trip: the process then waits for the 2 s the ack is
    /// given.
    #[test]
    fn callbacks_keep_flowing_while_another_thread_subscribes() {
        let mut hits = 0u32;
        let (d, mut daemon) = subscribed_device((&raw mut hits).cast());
        let sample = encode_gaze_origin(&tobii_ipc::EyePair::default());

        thread::scope(|s| {
            let subscribing = s.spawn(|| {
                d.subscribe(
                    |c| &mut c.eye_position,
                    Some(ignore_pair as EyePairFn),
                    ptr::null_mut(),
                )
            });
            assert_eq!(
                subscription(&mut daemon),
                Some(STREAM_GAZE_ORIGIN | STREAM_EYE_POSITION),
                "the other thread waits for its ack"
            );
            write_frame(&mut daemon, &sample).expect("sample");

            let (status, took) = timed(|| {
                assert!(d.wait(LONG_WAIT));
                d.process()
            });

            assert_eq!(status, TOBII_ERROR_NO_ERROR);
            assert!(took < PROMPT, "{took:?}");
            write_frame(&mut daemon, &encode_subscribed(true)).expect("ack");
            assert_eq!(subscribing.join().expect("subscribe"), TOBII_ERROR_NO_ERROR);
        });

        drop((d, daemon));
        assert_eq!(hits, 1);
    }

    /// Clearing the buffers waits for a dispatch another thread runs, then
    /// drops what came meanwhile: the next process delivers nothing from
    /// before the clear. It fails should clear drain the channel without the
    /// dispatch lock: it returns at once.
    #[test]
    fn clearing_waits_for_a_dispatch_on_another_thread_then_leaves_nothing() {
        let gate = Gate::default();
        let mut hits = 0u32;
        let (d, mut daemon) = subscribed_device((&raw mut hits).cast());
        lock(&d.callbacks).gaze_origin = Some((held_pair as EyePairFn, gate.ud()));
        d.queue(gaze_origin_sample());
        let d = &d;

        thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(gate.entered(1), "the callback runs on the other thread");
            let seen = d.doorbell.rings();
            write_frame(
                &mut daemon,
                &encode_gaze_origin(&tobii_ipc::EyePair::default()),
            )
            .expect("sample");
            assert!(
                d.doorbell.wait_past(seen, LONG_WAIT),
                "the sample is queued"
            );
            let (tx, done) = mpsc::channel();
            s.spawn(move || {
                d.clear_buffers();
                tx.send(())
            });

            assert_eq!(
                done.recv_timeout(ASLEEP),
                Err(RecvTimeoutError::Timeout),
                "it waits for the dispatch"
            );
            gate.open();
            assert_eq!(done.recv_timeout(PROMPT), Ok(()));
            assert_eq!(processing.join().expect("process"), TOBII_ERROR_NO_ERROR);
        });

        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        assert_eq!(gate.calls(), 1, "what came during the dispatch was cleared");
        drop(daemon);
    }

    /// Requests from several threads at once run one at a time under the
    /// command lock, each getting its own reply, and the samples the daemon
    /// sends ahead of every reply stay queued for process. It fails should
    /// the lock cover the write but not the wait: a thread would read
    /// another's reply, drop it as stale, and time out.
    #[test]
    fn requests_from_several_threads_each_get_their_own_reply() {
        const THREADS: u8 = 4;
        const EACH: u8 = 100;
        // Echoes each request's payload, a sample ahead of the reply.
        let connect = fake_daemon(|body| {
            let req = tobii_ipc::request::decode_request(body).expect("request");
            vec![
                encode_gaze(1, true, [0.5; 2], [f32::NAN; 2]),
                encode_reply(req.id, 0, req.payload),
            ]
        });
        let d = Device::new(connect, 1, 1).expect("device");

        thread::scope(|s| {
            for t in 0..THREADS {
                let d = &d;
                s.spawn(move || {
                    for i in 0..EACH {
                        let payload = [t, i];
                        let echo = d.request(ECHOED, &payload, LONG_WAIT);
                        assert_eq!(echo, Ok(payload.to_vec()));
                    }
                });
            }
        });

        let queued = lock(&d.dispatch).samples.try_iter().count();
        assert_eq!(queued, usize::from(THREADS) * usize::from(EACH));
    }

    /// Whether `turns` has `n` tickets out, waiting up to `PROMPT` for it.
    fn tickets_out<T>(turns: &TicketLock<T>, n: u64) -> bool {
        let deadline = Instant::now() + PROMPT;
        while turns.out() != n {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// The command lock's order: the threads waiting for a ticket lock get
    /// it in the order they asked, and a thread that lets it go and asks
    /// again at once waits behind them, where a std `Mutex` would usually
    /// let it straight back in. It fails should a ticket not wait for its
    /// turn (tried: the first thread then goes straight back in).
    #[test]
    fn a_ticket_lock_lets_its_waiters_in_in_the_order_they_asked() {
        within(WATCHDOG, || {
            let turns = TicketLock::new(Vec::new());
            let mut held = turns.lock();
            held.push('a');
            thread::scope(|s| {
                for (waiter, out) in [('b', 2), ('c', 3)] {
                    let turns = &turns;
                    s.spawn(move || turns.lock().push(waiter));
                    assert!(tickets_out(turns, out), "{waiter} waits its turn");
                }
                drop(held);
                turns.lock().push('a');
            });
            assert_eq!(*turns.lock(), ['a', 'b', 'c', 'a']);
        });
    }

    /// A ticket lock held by a thread that panics passes the turn on as it
    /// unwinds, and the next holder takes the data as it was left, as
    /// [`lock`] takes a poisoned `Mutex`'s. It fails at the watchdog should
    /// the panic leave the turn with the thread that panicked.
    #[test]
    fn a_ticket_lock_passes_the_turn_on_through_a_panic() {
        within(WATCHDOG, || {
            let turns = TicketLock::new(0u32);
            let panicked = thread::scope(|s| {
                s.spawn(|| {
                    let mut held = turns.lock();
                    *held = 1;
                    panic!("panicking with the lock held");
                })
                .join()
                .is_err()
            });
            assert!(panicked);
            assert_eq!(*turns.lock(), 1);
            assert_eq!(turns.out(), 0, "no ticket is left out");
        });
    }

    /// While another thread makes requests back to back, a thread's
    /// subscription changes each wait for the request under way when they
    /// ask, and no more: the command lock lets its waiters in in the order
    /// they asked. For each change the daemon stand-in holds a request until
    /// the change has asked (its ticket is out, behind the requester's),
    /// then lets it go, and notes how many more requests it reads before the
    /// change: none, however soon the requester asks again. It fails, in
    /// every run tried, with a lock that lets a thread that has just let it
    /// go take it again ahead of one already waiting, as a std `Mutex` does
    /// on Linux (tried: tickets that do not wait, which leaves the order to
    /// the data's `Mutex`): the requester then goes ahead of a change, for
    /// up to thousands of requests (on hardware, subscribes and unsubscribes
    /// waited up to 5 s so).
    #[test]
    fn subscription_changes_are_not_starved_by_requests_back_to_back() {
        /// The subscribes made, each followed by an unsubscribe.
        const CHANGES: usize = 20;

        /// What the daemon stand-in shares with the test.
        #[derive(Debug, Default)]
        struct Held {
            /// The gate the next request it reads waits at.
            next: Option<Arc<Gate>>,
            /// The requests it has read since the last one held.
            since: usize,
            /// For each change it has read, the requests it read between the
            /// last one held and the change.
            ahead: Vec<usize>,
        }

        within(WATCHDOG, || {
            let held = Arc::new(Mutex::new(Held::default()));
            let noted = Arc::clone(&held);
            let connect = fake_daemon(move |body| {
                if body.first() == Some(&tobii_ipc::TAG_SUBSCRIBE) {
                    let mut noted = lock(&noted);
                    let since = noted.since;
                    noted.ahead.push(since);
                    return vec![encode_subscribed(true)];
                }
                let gate = {
                    let mut noted = lock(&noted);
                    let gate = noted.next.take();
                    noted.since = if gate.is_some() { 0 } else { noted.since + 1 };
                    gate
                };
                if let Some(gate) = gate {
                    gate.hold();
                }
                let req = tobii_ipc::request::decode_request(body).expect("request");
                vec![encode_reply(req.id, 0, req.payload)]
            });
            let device = Device::new(connect, 1, 1).expect("device");
            let d = &device;
            let stop = AtomicBool::new(false);

            let ahead = thread::scope(|s| {
                let requester = s.spawn(|| {
                    let deadline = Instant::now() + LONG_WAIT;
                    while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
                        assert_eq!(d.request(ECHOED, &[], LONG_WAIT), Ok(vec![]));
                    }
                });
                let callback = Some(ignore_pair as EyePairFn);
                for on in [true, false].into_iter().cycle().take(2 * CHANGES) {
                    let gate = Arc::new(Gate::default());
                    lock(&held).next = Some(Arc::clone(&gate));
                    assert!(gate.entered(1), "a request is under way");
                    let change = s.spawn(move || {
                        if on {
                            d.subscribe(|c| &mut c.gaze_origin, callback, ptr::null_mut())
                        } else {
                            d.unsubscribe(|c| &mut c.gaze_origin)
                        }
                    });
                    assert!(
                        tickets_out(&d.command, 2),
                        "the change waits behind the request"
                    );
                    gate.open();
                    assert_eq!(joined(change), TOBII_ERROR_NO_ERROR);
                    // Stop at the first change kept out, rather than wait for
                    // the requester to stop.
                    if lock(&held).ahead.last() != Some(&0) {
                        break;
                    }
                }
                stop.store(true, Ordering::Relaxed);
                joined(requester);
                lock(&held).ahead.clone()
            });

            assert_eq!(
                ahead,
                [0; 2 * CHANGES],
                "the requests read between the one held and each change"
            );
        });
    }

    /// A wait whose look comes while another thread's process holds the
    /// dispatch lock finds nothing and sleeps. A sample that came during
    /// that dispatch is still queued when it ends, so the process rings the
    /// doorbell once more as it lets the lock go. Here the sample's own ring
    /// comes before the wait starts, so only that second ring can wake it.
    /// It fails without it: the wait sleeps out its timeout with a sample
    /// queued.
    #[test]
    fn a_wait_turned_away_by_a_dispatch_wakes_for_what_came_meanwhile() {
        let gate = Gate::default();
        let mut hits = 0u32;
        let (d, mut daemon) = subscribed_device((&raw mut hits).cast());
        lock(&d.callbacks).gaze_origin = Some((held_pair as EyePairFn, gate.ud()));
        d.queue(gaze_origin_sample());
        let d = &d;

        let (woke, took) = thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(gate.entered(1), "the callback runs on the other thread");
            let seen = d.doorbell.rings();
            write_frame(
                &mut daemon,
                &encode_gaze_origin(&tobii_ipc::EyePair::default()),
            )
            .expect("sample");
            assert!(d.doorbell.wait_past(seen, LONG_WAIT), "the sample's ring");
            let waiting = s.spawn(|| timed(|| d.wait(LONG_WAIT)));
            // Long enough for its look to have been turned away.
            thread::sleep(ASLEEP);
            gate.open();
            assert_eq!(processing.join().expect("process"), TOBII_ERROR_NO_ERROR);
            waiting.join().expect("wait")
        });

        assert!(woke, "{took:?}");
        assert!(took < ASLEEP + PROMPT, "{took:?}");
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        assert_eq!(gate.calls(), 2);
        drop(daemon);
    }

    /// Two threads waiting on one device: a look on one takes a sample into
    /// `pending` while the other's look is turned away. The sample rang
    /// before either looked, so nothing rings during the hold, and the
    /// holder rings once more as it lets the lock go because it leaves
    /// something to process. It fails should the holder ring only for a
    /// ring during its hold: the turned-away wait sleeps out its timeout
    /// with the sample in `pending`.
    #[test]
    fn a_wait_turned_away_by_a_look_that_takes_a_sample_wakes_for_it() {
        let mut hits = 0u32;
        let (d, mut daemon) = subscribed_device((&raw mut hits).cast());
        let seen = d.doorbell.rings();
        write_frame(
            &mut daemon,
            &encode_gaze_origin(&tobii_ipc::EyePair::default()),
        )
        .expect("sample");
        assert!(d.doorbell.wait_past(seen, LONG_WAIT), "the sample's ring");

        let (woke, took) = thread::scope(|s| {
            // The other thread's look, held open until this one's has come.
            let waiting = d.in_dispatch(lock(&d.dispatch), |dispatch| {
                let waiting = s.spawn(|| timed(|| d.wait(LONG_WAIT)));
                // Long enough for its look to have been turned away.
                thread::sleep(ASLEEP);
                assert!(dispatch.take_one(), "the look takes the sample");
                waiting
            });
            waiting.join().expect("wait")
        });

        assert!(woke, "{took:?}");
        assert!(took < ASLEEP + PROMPT, "{took:?}");
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop((d, daemon));
        assert_eq!(hits, 1);
    }

    /// A reconnect on one thread waits for a callback another thread's
    /// process is running: it takes the dispatch lock, and reads the
    /// subscriptions under the callbacks lock, before it asks for them back,
    /// as the DLL's reconnect waits for its process mutex (0x18000eb88). The
    /// next process then delivers the new link's sample. It fails should the
    /// reconnect take neither lock.
    #[test]
    fn a_reconnect_waits_for_a_callback_running_on_another_thread() {
        let gate = Gate::default();
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(1)], (&raw mut hits).cast());
        lock(&d.callbacks).gaze_origin = Some((held_pair as EyePairFn, gate.ud()));
        d.queue(gaze_origin_sample());
        let d = &d;

        thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(gate.entered(1), "the callback runs on the other thread");
            let (tx, done) = mpsc::channel();
            s.spawn(move || tx.send(d.reconnect()));

            assert_eq!(
                done.recv_timeout(ASLEEP),
                Err(RecvTimeoutError::Timeout),
                "it waits for the callback"
            );
            gate.open();
            assert_eq!(done.recv_timeout(PROMPT), Ok(TOBII_ERROR_NO_ERROR));
            assert_eq!(
                processing.join().expect("process"),
                TOBII_ERROR_CONNECTION_FAILED,
                "the old connection's loss"
            );
        });

        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert!(d.wait(LONG_WAIT));
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        assert_eq!(gate.calls(), 2, "the new connection's sample");
        drop(daemon);
    }

    /// A reconnect of a live link while another thread processes the device
    /// delivers no sample twice. tobiid writes each tick to every client
    /// subscribed to its streams, so the tick that acks the new link's
    /// subscription reaches the old link too, and first; the reconnect holds
    /// the dispatch lock from before it subscribes the new link until it has
    /// swapped the samples channel, so the other thread delivers none of
    /// that tick from the old link, and the new link's copy is delivered
    /// alone. Here the stand-in writes the old link's copy once the new link
    /// has subscribed, gives the other thread time to deliver it, then acks
    /// the new link in one write with its copy. It fails should the
    /// reconnect take the dispatch lock only for the swap: the other thread
    /// delivers the old link's copy while the reconnect waits for its ack,
    /// and then the new link's, stamped alike.
    #[test]
    fn a_reconnect_of_a_live_link_delivers_no_sample_twice() {
        let deliveries = Deliveries::default();
        let (connect, daemons) = scripted_daemon(vec![ack_then_gaze_origin(0)]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut old = daemons.recv().expect("daemon end");
        let callback = Some(note_gaze_origin as EyePairFn);
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, callback, deliveries.ud()),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(subscription(&mut old), Some(STREAM_GAZE_ORIGIN));
        let stop = AtomicBool::new(false);

        let (reconnected, mut new) = thread::scope(|s| {
            let processing = s.spawn(|| {
                let until = Instant::now() + LONG_WAIT;
                while !stop.load(Ordering::Relaxed) && Instant::now() < until {
                    d.wait(SHORT_WAIT);
                    assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
                }
            });
            let (old, delivered) = (&mut old, &deliveries);
            let daemon = s.spawn(move || {
                let mut new = daemons.recv_timeout(LONG_WAIT).expect("new daemon end");
                assert_eq!(subscription(&mut new), Some(STREAM_GAZE_ORIGIN));
                // The tick that acks it, which reaches the old link first.
                let tick = sample(STREAM_GAZE_ORIGIN, 1);
                write_frame(old, &tick).expect("the old link's copy");
                // Time for the other thread to deliver it, were it free to.
                let until = Instant::now() + ASLEEP;
                while delivered.calls() == 0 && Instant::now() < until {
                    thread::sleep(TICK);
                }
                new.write_all(&frames([encode_subscribed(true), tick]))
                    .expect("the ack and the new link's copy");
                new
            });
            let reconnected = d.reconnect();
            let new = joined(daemon);
            stop.store(true, Ordering::Relaxed);
            joined(processing);
            (reconnected, new)
        });
        // The next tick, the new link's alone: once it is delivered, so is
        // everything before it.
        write_frame(&mut new, &sample(STREAM_GAZE_ORIGIN, 2)).expect("the next tick");
        let until = Instant::now() + LONG_WAIT;
        while deliveries.last[GAZE_ORIGIN].load(Ordering::Relaxed) < 2 && Instant::now() < until {
            d.wait(SHORT_WAIT);
            assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        }

        assert_eq!(reconnected, TOBII_ERROR_NO_ERROR);
        assert_eq!(deliveries.calls(), 2, "each tick once");
        deliveries.assert_in_turn();
        drop((d, old, new));
    }

    /// The samples the old link of
    /// [`a_reconnect_of_a_live_link_keeps_the_notifications_the_old_link_alone_got`]
    /// is sent ahead of its last notifications, some 80 KB of frames. They
    /// fit in the socket's buffer, so their write returns at once and the
    /// new link's ack follows it, however slowly the old link's reader
    /// reads them (a backlog the buffer cannot hold would keep the ack back
    /// until the reader had read the excess, inside the ack's timeout). The
    /// reader then has most of them still to read when the ack comes, far
    /// more than it reads in the moment the reconnect takes to swap.
    const BACKLOG: i64 = 2_000;

    /// A reconnect of a live link keeps the notifications tobiid sent the
    /// old link alone, before it took the new link's subscriptions, which
    /// tobiid does not repeat to the new one: one a look of `wait` had
    /// taken, one the old link's reader had queued, and one it still had to
    /// read, behind a backlog of samples, when the new link's ack came. The
    /// next process delivers them in the order they came, ahead of the new
    /// link's, the acking tick's twice, as that tick reaches both links;
    /// of the samples, the new link's alone. It fails should the swap drop
    /// the old link's notifications, as a swap that clears does, as every
    /// swap did before (only the new link's come; 3bbcd3e only widened what
    /// it dropped to all the old link brought during the new one's
    /// subscription), and should the swap read the old link's channel out
    /// before closing it: the swap's check fails in a debug build, and in a
    /// release build, but for a run whose old reader catches up in that
    /// moment, the last is lost.
    #[test]
    fn a_reconnect_of_a_live_link_keeps_the_notifications_the_old_link_alone_got() {
        let (connect, daemons) = scripted_daemon(vec![vec![]]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut old = daemons.recv().expect("daemon end");
        let mut hits = 0u32;
        let mut seen: Vec<Notification> = Vec::new();
        {
            let mut callbacks = lock(&d.callbacks);
            callbacks.gaze_origin = Some((count_pair as EyePairFn, (&raw mut hits).cast()));
            callbacks.notifications =
                Some((keep_notification as NotificationsFn, (&raw mut seen).cast()));
        }
        d.queue(decode_server(&calibration_id(1)).expect("decodes"));
        d.queue(gaze_origin_sample());
        let device = &d;

        let (reconnected, (old, new)) = thread::scope(|s| {
            let daemon = s.spawn(move || {
                let mut new = daemons.recv_timeout(LONG_WAIT).expect("new daemon end");
                assert_eq!(
                    subscription(&mut new),
                    Some(STREAM_GAZE_ORIGIN | STREAM_NOTIFICATIONS)
                );
                // Sent the old link alone, and queued by its reader.
                let rung = device.doorbell.rings();
                write_frame(&mut old, &calibration_id(2)).expect("the old link's");
                let until = Instant::now() + LONG_WAIT;
                while device.doorbell.rings() == rung && Instant::now() < until {
                    thread::sleep(TICK);
                }
                // Sent the old link alone too, behind samples its reader
                // has yet to read; then the acking tick, which reaches the
                // old link first.
                let behind = (0..BACKLOG).map(|ts_us| sample(STREAM_GAZE_ORIGIN, ts_us));
                old.write_all(&frames(
                    behind.chain([calibration_id(3), calibration_id(4)]),
                ))
                .expect("the old link's backlog and its copy of the tick");
                new.write_all(&frames([
                    encode_subscribed(true),
                    calibration_id(4),
                    sample(STREAM_GAZE_ORIGIN, BACKLOG),
                ]))
                .expect("the ack and the new link's copy of the tick");
                (old, new)
            });
            let reconnected = device.reconnect();
            (reconnected, joined(daemon))
        });
        let until = Instant::now() + LONG_WAIT;
        while hits == 0 && Instant::now() < until {
            d.wait(SHORT_WAIT);
            assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        }
        drop(d);

        assert_eq!(reconnected, TOBII_ERROR_NO_ERROR);
        assert_eq!(hits, 1, "the new link's sample alone");
        assert_eq!(calibration_ids(&seen), [1, 2, 3, 4, 4]);
        drop((old, new));
    }

    /// A guard: a reconnect on one thread waits for a device-info fetch
    /// another thread has under way, which gets its answer from the
    /// connection it asked, and the next fetch asks the new connection. Both
    /// follow from the connection and the kept identity being the command
    /// lock's. That the fetch checks, asks and keeps under one hold of it,
    /// so that a reconnect cannot come between the ask and the keep and
    /// leave the old identity kept for the new connection, this pins only
    /// when the reconnect would win the race for the lock.
    #[test]
    fn a_reconnect_waits_for_a_device_info_fetch_on_another_thread() {
        use tobii_ipc::request::{DeviceInfo, encode_device_info};
        let answer = |daemon: &mut UnixStream, id: u32, serial: &str| {
            let info = DeviceInfo {
                serial_number: serial.into(),
                ..DeviceInfo::default()
            };
            let reply = encode_reply(id, 0, &encode_device_info(&info));
            write_frame(daemon, &reply).expect("reply");
        };
        let serial = |d: &Device| d.device_info().map(|info| info.serial_number);
        let (connect, daemons) = scripted_daemon(vec![]);
        let d = Device::new(connect, 1, 1).expect("device");
        let mut first = daemons.recv().expect("daemon end");

        thread::scope(|s| {
            let fetching = s.spawn(|| serial(&d));
            let id = asked(&mut first, kind::DEVICE_INFO);
            let (tx, done) = mpsc::channel();
            let reconnecting = &d;
            s.spawn(move || tx.send(reconnecting.reconnect()));

            assert!(
                daemons.recv_timeout(ASLEEP).is_err(),
                "it waits for the fetch before it connects"
            );
            answer(&mut first, id, "old");
            assert_eq!(fetching.join().expect("fetch"), Ok("old".into()));
            assert_eq!(done.recv_timeout(PROMPT), Ok(TOBII_ERROR_NO_ERROR));
        });

        let mut second = daemons.recv().expect("second daemon end");
        thread::scope(|s| {
            let fetching = s.spawn(|| serial(&d));
            let id = asked(&mut second, kind::DEVICE_INFO);
            answer(&mut second, id, "new");
            assert_eq!(fetching.join().expect("fetch"), Ok("new".into()));
        });
        drop((d, first, second));
    }

    /// An end-to-end guard: a wait sleeping on a reported loss wakes for the
    /// sample of the link a reconnect on another thread puts in place. What
    /// it depends on is pinned by
    /// `a_reconnects_samples_ring_the_doorbell_a_lost_wait_sleeps_on`, which
    /// fails should a link ring a doorbell of its own.
    #[test]
    fn a_wait_on_a_reported_loss_wakes_for_a_reconnect_on_another_thread() {
        let mut hits = 0u32;
        let (d, daemons) = lost_device(0, vec![ack_then_gaze_origin(1)], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);

        let (woke, took) = thread::scope(|s| {
            let waiting = s.spawn(|| timed(|| d.wait(LONG_WAIT)));
            thread::sleep(ASLEEP);
            assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);
            waiting.join().expect("wait")
        });

        assert!(woke, "{took:?}");
        assert!(took < ASLEEP + PROMPT, "{took:?}");
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop((d, daemon));
        assert_eq!(hits, 1);
    }

    /// A guard: however many threads process a lost device at once, one
    /// reports the loss and logs it, once; a call that finds another
    /// dispatching answers from the last report. It held with one thread.
    #[test]
    fn a_loss_is_logged_once_however_many_threads_process() {
        const THREADS: usize = 4;
        let recorder = SyncRecorder::default();
        let mut hits = 0u32;
        let (mut d, _daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        d.set_logger(recorder.logger());
        let d = &d;
        let barrier = std::sync::Barrier::new(THREADS);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    barrier.wait();
                    for _ in 0..50 {
                        let status = d.process();
                        assert!(
                            matches!(status, TOBII_ERROR_NO_ERROR | TOBII_ERROR_CONNECTION_FAILED),
                            "{status}"
                        );
                    }
                });
            }
        });

        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(recorder.lines(), [(TOBII_LOG_LEVEL_ERROR, LOST.to_owned())]);
    }

    /// A logger's context that hands the device that logs to another
    /// thread, which makes `call` on it, and waits `PROMPT` for it to come
    /// back: whether it did, one per line.
    struct Handoff {
        /// The device that logs, set once its logger is.
        device: Cell<*const Device>,
        call: fn(&Device),
        came_back: Mutex<Vec<bool>>,
        threads: Mutex<Vec<JoinHandle<()>>>,
    }

    impl Handoff {
        fn new(call: fn(&Device)) -> Self {
            Self {
                device: Cell::new(ptr::null()),
                call,
                came_back: Mutex::default(),
                threads: Mutex::default(),
            }
        }

        fn context(&self) -> *mut c_void {
            ptr::from_ref(self).cast_mut().cast()
        }

        /// Join the threads it started, then say whether each call came back
        /// while the logger waited.
        fn came_back(&self) -> Vec<bool> {
            for thread in lock(&self.threads).drain(..) {
                thread.join().expect("call");
            }
            lock(&self.came_back).clone()
        }
    }

    unsafe extern "C" fn hand_off(context: *mut c_void, _level: LogLevel, _text: *const c_char) {
        // SAFETY: the test passes a live `Handoff` as the context, logging
        // on the thread that set its device.
        let h = unsafe { &*context.cast::<Handoff>() };
        // SAFETY: the device is live, and the test joins every thread this
        // starts (`Handoff::came_back`) before it drops the device.
        let device: &'static Device = unsafe { &*h.device.get() };
        let call = h.call;
        let (tx, rx) = mpsc::channel();
        let thread = thread::spawn(move || {
            call(device);
            let _ = tx.send(());
        });
        lock(&h.came_back).push(rx.recv_timeout(PROMPT).is_ok());
        lock(&h.threads).push(thread);
    }

    /// A call that logs lets every lock it took go first, so a logger may
    /// wait for another thread's call into the device that logs: a clear,
    /// which needs the dispatch lock, on the loss a process reports, and a
    /// recenter, which needs the command lock, on a reconnect. It fails
    /// should either line be logged under the lock its call needs: the call
    /// then comes back only once the logger has given up on it.
    #[test]
    fn a_logger_may_wait_for_another_thread_using_the_device() {
        let loss = Handoff::new(Device::clear_buffers);
        let reconnect = Handoff::new(|d| {
            assert_eq!(d.send(&tobii_ipc::encode_recenter()), Ok(()));
        });
        let mut hits = 0u32;
        let (mut d, daemons) =
            lost_device(0, vec![ack_then_gaze_origin(0)], (&raw mut hits).cast());

        d.set_logger(crate::logger::tests::logger(hand_off, loss.context()));
        loss.device.set(ptr::from_ref(&d));
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(loss.came_back(), [true], "the clear, on the loss line");

        d.set_logger(crate::logger::tests::logger(hand_off, reconnect.context()));
        reconnect.device.set(ptr::from_ref(&d));
        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);
        assert_eq!(
            reconnect.came_back(),
            [true],
            "the recenter, on the reconnect line"
        );

        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        let recenter = read_frame(&mut daemon).expect("read").expect("a frame");
        assert_eq!(recenter.first(), Some(&tobii_ipc::TAG_RECENTER));
        drop((d, daemon));
    }

    /// While another thread's request waits for its reply, holding the
    /// command lock, a wait on this thread wakes for a sample that comes
    /// meanwhile, and a process delivers it; the request then gets its
    /// reply. A guard against locking too much, which would pass with no
    /// lock at all: it fails should a request hold the dispatch lock, or
    /// the device be one lock. The wait and the process then get nothing
    /// done until the request has given up on its reply, which comes only
    /// once they have returned (seen with both).
    #[test]
    fn wait_and_process_go_on_while_another_thread_waits_for_a_reply() {
        let mut hits = 0u32;
        let (device, mut daemon) = subscribed_device((&raw mut hits).cast());
        let d = &device;

        let ((woke, took), processed) = thread::scope(|s| {
            let asking = s.spawn(|| calibration_active(d));
            let id = asked(&mut daemon, kind::STATE);
            write_frame(
                &mut daemon,
                &encode_gaze_origin(&tobii_ipc::EyePair::default()),
            )
            .expect("sample");

            let waited = timed(|| d.wait(LONG_WAIT));
            let processed = d.process();

            write_frame(&mut daemon, &encode_reply(id, 0, &[1])).expect("reply");
            assert_eq!(asking.join().expect("request"), (TOBII_ERROR_NO_ERROR, 1));
            (waited, processed)
        });

        assert!(woke, "{took:?}");
        assert!(took < PROMPT, "{took:?}");
        assert_eq!(processed, TOBII_ERROR_NO_ERROR);
        drop((device, daemon));
        assert_eq!(hits, 1, "delivered while the request waited");
    }

    /// A callback's user data for [`reenter_held`]: the device it calls
    /// back into, what that call gave, and the gate that then holds it.
    struct Reentrant<'a> {
        device: &'a Device,
        got: Mutex<Option<(Status, u32)>>,
        gate: Gate,
    }

    unsafe extern "C" fn reenter_held(p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the test passes a live `Reentrant` as `ud`, whose device is
        // the one being processed.
        let r = unsafe { &*ud.cast::<Reentrant<'_>>() };
        *lock(&r.got) = Some(calibration_active(r.device));
        // SAFETY: `p` is the sample, live for this call, and `r.gate` a live
        // `Gate`.
        unsafe { held_pair(p, r.gate.ud()) };
    }

    /// The callback guard is its own thread's: a callback that calls into
    /// its device is refused, while another thread's call into the same
    /// device goes through as the callback runs. It fails should the guard
    /// be dropped (the callback's call goes through too: it needs only the
    /// command lock, which the callback does not hold), or be one flag for
    /// the whole process (this thread's call is refused too).
    #[test]
    fn a_callback_is_refused_on_its_own_thread_while_another_threads_call_goes_through() {
        let d = device_with(0, vec![1]);
        let reentrant = Reentrant {
            device: &d,
            got: Mutex::default(),
            gate: Gate::default(),
        };
        let ud = ptr::from_ref(&reentrant).cast_mut().cast();
        lock(&d.callbacks).gaze_origin = Some((reenter_held as EyePairFn, ud));
        d.queue(gaze_origin_sample());

        let (ours, processed) = thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(
                reentrant.gate.entered(1),
                "the callback runs on the other thread"
            );
            let ours = calibration_active(&d);
            reentrant.gate.open();
            (ours, processing.join().expect("process"))
        });

        assert_eq!(
            ours,
            (TOBII_ERROR_NO_ERROR, 1),
            "this thread's goes through"
        );
        assert_eq!(
            *lock(&reentrant.got),
            Some((TOBII_ERROR_CALLBACK_IN_PROGRESS, 7)),
            "the callback's is refused, writing nothing"
        );
        assert_eq!(processed, TOBII_ERROR_NO_ERROR);
    }

    /// Whether another thread holds `d`'s command lock, rather than only
    /// waits for it, looking for up to `PROMPT`: a call that holds it while
    /// it waits for another lock shows as held however often this looks.
    fn command_held(d: &Device) -> bool {
        let deadline = Instant::now() + PROMPT;
        while !d.command.held() {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// Pins today's behaviour, which the deadlock rule in [`Device`]'s docs
    /// assumes, not a guarantee: once an unsubscribe on a third thread is
    /// queued ahead of it, a request made while another thread runs a
    /// callback waits for that callback, as the unsubscribe holds the
    /// command lock while it waits for the callbacks lock. So a callback
    /// that waited for such a request would deadlock, as in the DLL, whose
    /// unsubscribe takes its API mutex (dev+0x4e0, 0x18015d0b4) and then
    /// waits on dev+0x4d8 (0x180153580), and a callback must not block on
    /// any other thread's call into a device, its own included. With
    /// nothing queued ahead, the request goes through
    /// (`a_request_goes_through_while_another_thread_runs_a_callback`).
    ///
    /// Should unsubscribe stop holding the command lock while it waits for
    /// the callback (taking the slot first, then telling the daemon under
    /// the command lock), this fails at `command_held`, though nothing got
    /// worse: delete it then, and narrow the rule to match.
    #[test]
    fn a_request_behind_an_unsubscribe_waits_for_the_callback_as_the_rule_assumes() {
        let gate = Gate::default();
        let d = device_with(0, vec![1]);
        assert_eq!(
            d.subscribe(
                |c| &mut c.gaze_origin,
                Some(held_pair as EyePairFn),
                gate.ud()
            ),
            TOBII_ERROR_NO_ERROR
        );
        d.queue(gaze_origin_sample());
        let d = &d;

        thread::scope(|s| {
            let processing = s.spawn(|| d.process());
            assert!(gate.entered(1), "the callback runs on the other thread");
            let (tx, unsubscribed) = mpsc::channel();
            s.spawn(move || tx.send(d.unsubscribe(|c| &mut c.gaze_origin)));
            assert!(
                command_held(d),
                "the unsubscribe holds the command lock while it waits for the callback"
            );
            let (tx, answered) = mpsc::channel();
            s.spawn(move || tx.send(calibration_active(d)));

            assert_eq!(
                answered.recv_timeout(ASLEEP),
                Err(RecvTimeoutError::Timeout),
                "the request waits behind the unsubscribe"
            );
            gate.open();
            assert_eq!(unsubscribed.recv_timeout(PROMPT), Ok(TOBII_ERROR_NO_ERROR));
            assert_eq!(answered.recv_timeout(PROMPT), Ok((TOBII_ERROR_NO_ERROR, 1)));
            assert_eq!(processing.join().expect("process"), TOBII_ERROR_NO_ERROR);
        });
    }

    /// A guard against locking too much: a wait asleep on one thread
    /// neither holds up a subscribe on another nor misses the sample the
    /// daemon sends ahead of the subscribe's ack. It fails should a wait
    /// sleep holding a lock a subscribe needs (the command or the callbacks
    /// lock): the subscribe then waits out the wait's `LONG_WAIT`.
    #[test]
    fn a_wait_wakes_for_what_a_subscribe_on_another_thread_brings() {
        let connect = fake_daemon(|body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![
                encode_gaze_origin(&tobii_ipc::EyePair::default()),
                encode_subscribed(true),
            ],
            _ => vec![],
        });
        let device = Device::new(connect, 1, 1).expect("device");
        let d = &device;
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        let ((subscribed, subscribing), (woke, waiting)) = thread::scope(|s| {
            let asleep = s.spawn(|| timed(|| d.wait(LONG_WAIT)));
            // Long enough for it to have gone to sleep.
            thread::sleep(ASLEEP);
            let subscribed =
                timed(|| d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud));
            (subscribed, asleep.join().expect("wait"))
        });

        assert_eq!(subscribed, TOBII_ERROR_NO_ERROR);
        assert!(subscribing < PROMPT, "{subscribing:?}");
        assert!(woke, "{waiting:?}");
        assert!(waiting < ASLEEP + PROMPT, "{waiting:?}");
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);
        drop(device);
        assert_eq!(hits, 1);
    }

    // Under load: a daemon stand-in that streams, and a watchdog, so that a
    // deadlock fails the test rather than hanging the suite.

    /// How long a test run under [`within`] may take before it counts as
    /// deadlocked: far longer than any takes, the round trips' own timeouts
    /// included.
    const WATCHDOG: Duration = Duration::from_secs(30);

    /// Run `f` on a thread of its own and hand back what it returned, or
    /// fail the test once `limit` has passed: a deadlock then fails it
    /// rather than hanging the suite. A thread that hangs is left behind
    /// with what it owns, which is why `f` must own everything it uses:
    /// nothing is freed under a thread that still runs.
    fn within<R: Send + 'static>(limit: Duration, f: impl FnOnce() -> R + Send + 'static) -> R {
        let (tx, done) = mpsc::channel();
        let runner = thread::spawn(move || {
            // The receiver is gone only once the test has failed.
            let _ = tx.send(f());
        });
        match done.recv_timeout(limit) {
            Ok(got) => {
                runner.join().expect("runner");
                got
            }
            Err(RecvTimeoutError::Timeout) => {
                panic!("still running after {limit:?}: deadlocked?")
            }
            Err(RecvTimeoutError::Disconnected) => match runner.join() {
                Err(panic) => std::panic::resume_unwind(panic),
                Ok(()) => panic!("the runner stopped without an answer"),
            },
        }
    }

    /// What a scoped thread returned, or its panic, resumed on this thread,
    /// so that a thread that failed fails the test with its own message.
    fn joined<T>(thread: thread::ScopedJoinHandle<'_, T>) -> T {
        thread
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }

    /// How often a [`streaming_daemon`] sends each stream subscribed.
    const TICK: Duration = Duration::from_millis(1);

    /// What a [`streaming_daemon`] noted.
    #[derive(Debug, Default)]
    struct Served {
        /// For each connection, in the order they were made, the streams
        /// last subscribed on it (none until a subscription).
        subscribed: Mutex<Vec<u32>>,
        /// The connections it hung up on.
        hang_ups: AtomicUsize,
    }

    /// A [`streaming_daemon`] connection, as its pump sees it.
    struct Client {
        /// The connection's write half.
        stream: UnixStream,
        /// When it was made.
        born: Instant,
        /// The streams last subscribed on it.
        streams: u32,
        /// The subscription changes read from it and not acked yet.
        acks: usize,
        /// Whether it has ended: the client hung up, a write failed, or the
        /// pump hung up on it.
        ended: bool,
    }

    /// What a [`streaming_daemon`]'s threads share, under one lock as
    /// tobiid's share its state: the clock every sample is stamped from,
    /// whichever connection it goes to, and the connections, in the order
    /// they were made.
    #[derive(Default)]
    struct Pump {
        /// The stamp of the last sample written.
        ts_us: i64,
        clients: Vec<Client>,
    }

    impl Pump {
        /// The next stamp: one microsecond after the last.
        fn stamp(&mut self) -> i64 {
            self.ts_us += 1;
            self.ts_us
        }

        /// One tick, as tobiid's pump writes it: to each connection in the
        /// order they were made, the acks it is owed and then a sample of
        /// each stream subscribed on it, in one write, the samples stamped
        /// alike on every connection. So the tick that acks a connection's
        /// subscription reaches those made before it first, bringing them
        /// byte for byte what it brings the new one. A connection that has
        /// lasted `life`, if given, is hung up on instead.
        fn tick(&mut self, life: Option<Duration>, served: &Served) {
            let ts_us = self.stamp();
            for client in self.clients.iter_mut().filter(|c| !c.ended) {
                if life.is_some_and(|life| client.born.elapsed() >= life) {
                    served.hang_ups.fetch_add(1, Ordering::Relaxed);
                    // Ends its reader's read too.
                    let _ = client.stream.shutdown(Shutdown::Both);
                    client.ended = true;
                    continue;
                }
                let acks =
                    std::iter::repeat_n(encode_subscribed(true), mem::take(&mut client.acks));
                let samples = [STREAM_GAZE_ORIGIN, STREAM_PRESENCE]
                    .into_iter()
                    .filter(|&stream| client.streams & stream != 0)
                    .map(|stream| sample(stream, ts_us));
                let burst = frames(acks.chain(samples));
                client.ended = client.stream.write_all(&burst).is_err();
            }
        }
    }

    /// A sample of `stream`, gaze origin or presence, stamped `ts_us`.
    fn sample(stream: u32, ts_us: i64) -> Vec<u8> {
        if stream == STREAM_PRESENCE {
            tobii_ipc::encode_presence(ts_us, tobii_ipc::PRESENCE_PRESENT)
        } else {
            encode_gaze_origin(&tobii_ipc::EyePair {
                ts_us,
                ..tobii_ipc::EyePair::default()
            })
        }
    }

    /// `bodies` framed, one after another, for a single write.
    fn frames(bodies: impl IntoIterator<Item = Vec<u8>>) -> Vec<u8> {
        let mut burst = Vec::new();
        for body in bodies {
            write_frame(&mut burst, &body).expect("a frame in memory");
        }
        burst
    }

    /// A daemon stand-in that streams as tobiid does: a pump thread writes
    /// to every connection each `TICK` (see [`Pump::tick`]), and a thread
    /// per connection reads what the client sends. A subscription change is
    /// acked by the next tick, in one write with that tick's samples, and
    /// from then on each tick brings a gaze-origin sample while gaze origin
    /// is subscribed, and a presence sample while presence is. It answers a
    /// state request with true (`[1]`) and any other request with the
    /// payload that request carried, each behind a gaze-origin sample. It
    /// hangs up on a connection once it has lasted `life`, if given. A
    /// connection's reader stops once the client hangs up, or the stand-in
    /// does, and the pump once the connector and every reader are gone.
    ///
    /// Every sample is stamped from one clock, whichever connection it goes
    /// to, as tobiid stamps a tick once for all its clients: each tick's and
    /// each answered request's one microsecond after the last. So the
    /// samples a client is delivered come in the order of their stamps
    /// however it reconnects, unless it delivers one twice, or one from
    /// before another it delivered already.
    fn streaming_daemon(life: Option<Duration>) -> (Connector, Arc<Served>) {
        let served = Arc::new(Served::default());
        let shared = Arc::new(Mutex::new(Pump::default()));
        let (pump, noted) = (Arc::downgrade(&shared), Arc::clone(&served));
        thread::spawn(move || {
            loop {
                thread::sleep(TICK);
                let Some(shared) = pump.upgrade() else {
                    return;
                };
                lock(&shared).tick(life, &noted);
            }
        });
        let noted = Arc::clone(&served);
        let connect: Connector = Box::new(move || {
            let (client, mut daemon) = UnixStream::pair()?;
            let stream = daemon.try_clone()?;
            let connection = {
                let mut pump = lock(&shared);
                pump.clients.push(Client {
                    stream,
                    born: Instant::now(),
                    streams: 0,
                    acks: 0,
                    ended: false,
                });
                lock(&noted.subscribed).push(0);
                pump.clients.len() - 1
            };
            let (shared, notes) = (Arc::clone(&shared), Arc::clone(&noted));
            thread::spawn(move || {
                while let Ok(Some(body)) = read_frame(&mut daemon) {
                    let mut pump = lock(&shared);
                    let written = match body.first() {
                        Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                            let mask = tobii_ipc::decode_subscribe(&body).expect("streams");
                            lock(&notes.subscribed)[connection] = mask;
                            let client = &mut pump.clients[connection];
                            client.streams = mask;
                            client.acks += 1;
                            Ok(())
                        }
                        Some(&tobii_ipc::TAG_REQUEST) => {
                            let req = tobii_ipc::request::decode_request(&body).expect("request");
                            let payload = if req.kind == kind::STATE {
                                &[1][..]
                            } else {
                                req.payload
                            };
                            let ts_us = pump.stamp();
                            let burst = frames([
                                sample(STREAM_GAZE_ORIGIN, ts_us),
                                encode_reply(req.id, 0, payload),
                            ]);
                            pump.clients[connection].stream.write_all(&burst)
                        }
                        _ => Ok(()),
                    };
                    if written.is_err() {
                        break;
                    }
                }
                lock(&shared).clients[connection].ended = true;
            });
            Ok(client)
        });
        (connect, served)
    }

    /// Which of [`Deliveries`]' streams.
    const GAZE_ORIGIN: usize = 0;
    /// See [`GAZE_ORIGIN`].
    const PRESENCE: usize = 1;

    /// How long a call to [`note_gaze_origin`] or [`note_presence`] lasts,
    /// at least. It widens the window in which a callback that outlives its
    /// unsubscribe, or one that runs beside another, is caught, without
    /// making sure of it: an unsubscribe's round trip to a
    /// [`streaming_daemon`] usually fits in it.
    const DWELL: Duration = Duration::from_micros(200);

    /// What [`note_gaze_origin`] and [`note_presence`] count, shared with
    /// whichever thread runs them.
    #[derive(Debug, Default)]
    struct Deliveries {
        /// The calls, per stream.
        calls: [AtomicU32; 2],
        /// Per stream, the stamp of the sample the last call got.
        last: [AtomicI64; 2],
        /// The calls that got a sample stamped no later than the one the
        /// call before on that stream got: delivered out of order.
        disordered: AtomicU32,
        /// The calls running now.
        running: AtomicU32,
        /// The calls begun while another ran.
        overlapped: AtomicU32,
        /// Per stream, whether its last unsubscribe, or a subscribe that
        /// failed, has returned, with no subscribe begun since. The
        /// callbacks lock orders these stores and the callbacks' loads: a
        /// subscribe sets the slot under it after the store, and an
        /// unsubscribe, or a failed subscribe's rollback, has taken the slot
        /// under it before.
        off: [AtomicBool; 2],
        /// The calls made while `off` said so: a callback run after its
        /// unsubscribe returned.
        late: AtomicU32,
    }

    impl Deliveries {
        fn ud(&self) -> *mut c_void {
            ptr::from_ref(self).cast_mut().cast()
        }

        /// Count a call for `stream` with a sample stamped `ts_us`, dwelling
        /// in it for `DWELL`: an unsubscribe that did not wait for a
        /// running callback would return, and the stream show as off,
        /// meanwhile, and a callback run on another thread without waiting
        /// for this one would begin.
        fn note(&self, stream: usize, ts_us: i64) {
            if self.running.fetch_add(1, Ordering::Relaxed) > 0 {
                self.overlapped.fetch_add(1, Ordering::Relaxed);
            }
            self.calls[stream].fetch_add(1, Ordering::Relaxed);
            if self.last[stream].swap(ts_us, Ordering::Relaxed) >= ts_us {
                self.disordered.fetch_add(1, Ordering::Relaxed);
            }
            thread::sleep(DWELL);
            if self.off[stream].load(Ordering::Relaxed) {
                self.late.fetch_add(1, Ordering::Relaxed);
            }
            self.running.fetch_sub(1, Ordering::Relaxed);
        }

        /// The calls so far, all streams together.
        fn calls(&self) -> u32 {
            self.calls.iter().map(|c| c.load(Ordering::Relaxed)).sum()
        }

        /// Check that the callbacks ran, one at a time, each stream's
        /// samples in the order they were sent, none twice.
        fn assert_in_turn(&self) {
            assert!(self.calls() > 0, "the samples were delivered");
            let overlapped = self.overlapped.load(Ordering::Relaxed);
            assert_eq!(overlapped, 0, "callbacks ran beside each other");
            let disordered = self.disordered.load(Ordering::Relaxed);
            assert_eq!(disordered, 0, "samples were delivered out of order");
        }
    }

    unsafe extern "C" fn note_gaze_origin(p: *const EyePair, ud: *mut c_void) {
        // SAFETY: libtobii passes the sample, live for the call, and the
        // tests a live `Deliveries` as `ud`.
        let (ts_us, deliveries) = unsafe { ((*p).timestamp_us, &*ud.cast::<Deliveries>()) };
        deliveries.note(GAZE_ORIGIN, ts_us);
    }

    unsafe extern "C" fn note_presence(_s: PresenceStatus, ts_us: i64, ud: *mut c_void) {
        // SAFETY: the tests pass a live `Deliveries` as `ud`.
        unsafe { &*ud.cast::<Deliveries>() }.note(PRESENCE, ts_us);
    }

    /// A deadlock guard: two threads wait on the same two streaming devices
    /// in opposite orders, and process both, for a while. A thread holds
    /// one device's locks at a time, and none while it sleeps, so this
    /// cannot deadlock; it would should a wait hold one device's lock while
    /// it takes the next one's. It checks, too, that each device's callback
    /// ran one call at a time, its samples in order, but the two threads
    /// seldom dispatch one device at once here: the hammers are the tests
    /// that fail should a process deliver outside the dispatch lock.
    #[test]
    fn waits_on_two_devices_in_opposite_orders_do_not_deadlock() {
        within(WATCHDOG, || {
            let deliveries: [Deliveries; 2] = Default::default();
            let devices: [Device; 2] = std::array::from_fn(|i| {
                let d = Device::new(streaming_daemon(None).0, 1, 1).expect("device");
                let callback = Some(note_gaze_origin as EyePairFn);
                assert_eq!(
                    d.subscribe(|c| &mut c.gaze_origin, callback, deliveries[i].ud()),
                    TOBII_ERROR_NO_ERROR
                );
                d
            });
            let stop = AtomicBool::new(false);

            thread::scope(|s| {
                for order in [[0, 1], [1, 0]] {
                    let (devices, stop) = (&devices, &stop);
                    s.spawn(move || {
                        let handles = order.map(|i| handle(&devices[i]));
                        while !stop.load(Ordering::Relaxed) {
                            // SAFETY: both devices are live until the scope
                            // ends, and the call forms only shared borrows of
                            // them; `handles` is a live local of the length
                            // passed.
                            let waited = unsafe {
                                crate::api::tobii_wait_for_callbacks(2, handles.as_ptr())
                            };
                            assert!(
                                matches!(waited, TOBII_ERROR_NO_ERROR | TOBII_ERROR_TIMED_OUT),
                                "{waited}"
                            );
                            for h in handles {
                                // SAFETY: as for the wait.
                                let processed =
                                    unsafe { crate::api::tobii_device_process_callbacks(h) };
                                assert_eq!(processed, TOBII_ERROR_NO_ERROR);
                            }
                        }
                    });
                }
                thread::sleep(3 * ASLEEP);
                stop.store(true, Ordering::Relaxed);
            });

            for delivered in &deliveries {
                delivered.assert_in_turn();
            }
        });
    }

    /// An API and a device made through the C entry points, destroyed
    /// through them, the device first, at the latest when this is dropped:
    /// a test that fails then leaves no device behind whose reader and
    /// daemon stand-in go on streaming for the rest of the run, pointing at
    /// locals of the test that are gone. Every thread that used them must
    /// have been joined by then, as a thread scope does before it unwinds.
    struct Handles {
        api: *mut Api,
        device: *mut Device,
    }

    impl Default for Handles {
        fn default() -> Self {
            Self {
                api: ptr::null_mut(),
                device: ptr::null_mut(),
            }
        }
    }

    impl Handles {
        /// Destroy both now, if made: what each destroy answered.
        fn destroy(&mut self) -> [Status; 2] {
            use crate::api::{tobii_api_destroy, tobii_device_destroy};
            let device = mem::replace(&mut self.device, ptr::null_mut());
            let api = mem::replace(&mut self.api, ptr::null_mut());
            // SAFETY: each is null or a live handle, destroyed only here, once
            // (the fields are null from now on), with no other thread inside
            // a call on it or using it later (see `Handles`); the device goes
            // before its API.
            unsafe { [tobii_device_destroy(device), tobii_api_destroy(api)] }
        }
    }

    impl Drop for Handles {
        fn drop(&mut self) {
            self.destroy();
        }
    }

    /// How long the hammer runs.
    const HAMMER_FOR: Duration = Duration::from_secs(1);
    /// How often the hammer clears the buffers.
    const CLEAR_EVERY: Duration = Duration::from_millis(10);
    /// How often the hammer reconnects.
    const RECONNECT_EVERY: Duration = Duration::from_millis(100);
    /// How long each connection lasts before the daemon stand-in hangs up
    /// on it, when the hammer loses connections: a third of
    /// `RECONNECT_EVERY`, so each is lost, and the loss reported, long
    /// before a reconnect replaces it.
    const CONNECTION_LIFE: Duration = Duration::from_millis(30);

    /// Six threads use one device at once, for `HAMMER_FOR`, over a daemon
    /// stand-in that streams, mostly through the C entry points (the echoed
    /// requests call `Device::request`). They
    /// 1. and 2. both wait for callbacks and process them;
    /// 3. switch gaze origin and presence on and off, in turn;
    /// 4. ask for a state, and make requests the stand-in echoes;
    /// 5. clear the buffers every `CLEAR_EVERY`;
    /// 6. reconnect every `RECONNECT_EVERY`.
    ///
    /// Every call must answer as it would on one thread (none fails: the
    /// connection is never lost; see
    /// `hammer_a_device_that_keeps_losing_its_connection`), each request
    /// with its own reply. The callbacks must run one at a time, each
    /// stream's samples in the order the stand-in sent them, none twice, and
    /// none once its unsubscribe has returned. At the end the daemon was
    /// last asked, on the connection in use, for the streams the callbacks
    /// need, and the log holds the connect and a line per reconnect. The
    /// watchdog fails it should it deadlock.
    ///
    /// It fails, in every run tried, should a callback run outside the
    /// callbacks lock (an unsubscribe then returns while it runs), a request
    /// let the command lock go between its send and its reply (another
    /// thread's subscription change reads the reply and drops it), a
    /// process let the dispatch lock go before it delivers what it took
    /// (the other processing thread then delivers a later sample first), or
    /// a reconnect take the dispatch lock only once its new link has its ack
    /// (a processing thread delivers the tick that acks it from the old link
    /// meanwhile, and then again from the new one). The last ask on the
    /// connection in use is only checked once all is done, so a race early
    /// in the run that a later change mends goes unseen there.
    ///
    /// Only a stress test: a race it does not happen to hit goes unseen. For
    /// data races, run the crate's tests by hand under the thread sanitizer,
    /// which needs nightly and the standard library rebuilt instrumented:
    ///
    /// ```text
    /// RUSTFLAGS=-Zsanitizer=thread cargo +nightly test -Zbuild-std \
    ///     --target x86_64-unknown-linux-gnu -p tobii-ffi --lib
    /// ```
    #[test]
    fn hammer_one_device_from_six_threads() {
        hammer(None);
    }

    /// The hammer, over a daemon stand-in that hangs up on each connection
    /// once it has lasted `CONNECTION_LIFE`, so that the device keeps losing
    /// its connection and the reconnects keep restoring it. The calls may
    /// fail too, then, but only with `TOBII_ERROR_CONNECTION_FAILED`, and
    /// the other checks hold, but for the daemon's last ask, which a change
    /// the loss cut short leaves behind the callbacks. Each loss comes long
    /// before the reconnect that mends it, so a process reports it first,
    /// and it is logged at ERROR once: as many loss lines as hang-ups, the
    /// last connection's included, which the end waits for.
    ///
    /// It fails, in every run tried, should a callback run outside the
    /// callbacks lock, a process let the dispatch lock go before it
    /// delivers, a reconnect leave the loss of the old connection standing
    /// for the new one (no later loss is reported), or a process log a loss
    /// again once it has been reported (the end's last process would). A
    /// request that loses its reply to another thread fails here only with
    /// the lost connection, so the hammer above is the one for that.
    #[test]
    fn hammer_a_device_that_keeps_losing_its_connection() {
        hammer(Some(CONNECTION_LIFE));
    }

    /// Hammer a device over a [`streaming_daemon`] with the connection life
    /// given, if any (see the tests that call it).
    fn hammer(life: Option<Duration>) {
        use crate::api::{
            tobii_api_create, tobii_device_clear_callback_buffers, tobii_device_create,
            tobii_device_process_callbacks, tobii_device_reconnect, tobii_wait_for_callbacks,
        };
        use crate::streams::{
            tobii_gaze_origin_subscribe, tobii_gaze_origin_unsubscribe,
            tobii_user_presence_subscribe, tobii_user_presence_unsubscribe,
        };
        // Whether `status` may answer a call: a success, or a lost
        // connection when the stand-in hangs up.
        let settled = move |status: Status| {
            status == TOBII_ERROR_NO_ERROR
                || life.is_some() && status == TOBII_ERROR_CONNECTION_FAILED
        };
        within(WATCHDOG, move || {
            let (recorder, deliveries) = (SyncRecorder::default(), Deliveries::default());
            let (connect, served) = streaming_daemon(life);
            let log = recorder.custom_log();
            // Dropped before `recorder` and `deliveries`, which the device
            // points to.
            let mut handles = Handles::default();
            DAEMON.set(Some(connect));
            // SAFETY: the out-parameters and `log` are live locals, and
            // `recorder` outlives both handles, which `handles` destroys, on
            // a failure too.
            unsafe {
                assert_eq!(
                    tobii_api_create(&raw mut handles.api, ptr::null(), &raw const log),
                    TOBII_ERROR_NO_ERROR
                );
                assert_eq!(
                    tobii_device_create(
                        handles.api,
                        ptr::null(),
                        crate::types::TOBII_FIELD_OF_USE_INTERACTIVE,
                        &raw mut handles.device
                    ),
                    TOBII_ERROR_NO_ERROR
                );
            }
            // SAFETY: `handles` destroys the device only once every thread
            // using it has been joined, by the scope below, which joins them
            // before it unwinds on a failure too.
            let d = unsafe { &*handles.device };
            let stop = AtomicBool::new(false);
            let running = || !stop.load(Ordering::Relaxed);
            // Switch `stream` on or off.
            let switch = |stream: usize, on: bool| {
                let (h, ud) = (handle(d), deliveries.ud());
                // SAFETY: `d` is live until the threads are joined, and each
                // call forms only a shared borrow of it; each callback
                // matches its slot, and `ud` is `deliveries`, which outlives
                // the device and which any thread may share.
                unsafe {
                    match (stream, on) {
                        (GAZE_ORIGIN, true) => {
                            tobii_gaze_origin_subscribe(h, Some(note_gaze_origin as EyePairFn), ud)
                        }
                        (GAZE_ORIGIN, false) => tobii_gaze_origin_unsubscribe(h),
                        (_, true) => {
                            tobii_user_presence_subscribe(h, Some(note_presence as PresenceFn), ud)
                        }
                        (_, false) => tobii_user_presence_unsubscribe(h),
                    }
                }
            };

            let (changes, requests, clears, [reconnected, unreconnected]) = thread::scope(|s| {
                for _ in 0..2 {
                    s.spawn(|| {
                        let devices = [handle(d)];
                        while running() {
                            // SAFETY: as for `switch`; `devices` is a live
                            // local of the length passed.
                            let waited = unsafe { tobii_wait_for_callbacks(1, devices.as_ptr()) };
                            assert!(
                                matches!(waited, TOBII_ERROR_NO_ERROR | TOBII_ERROR_TIMED_OUT),
                                "wait: {waited}"
                            );
                            // SAFETY: as for `switch`.
                            let processed = unsafe { tobii_device_process_callbacks(devices[0]) };
                            assert!(settled(processed), "process: {processed}");
                        }
                    });
                }
                let changes = s.spawn(|| {
                    let mut on = [false; 2];
                    let mut changes = 0usize;
                    for stream in [GAZE_ORIGIN, PRESENCE].into_iter().cycle() {
                        if !running() {
                            break;
                        }
                        if on[stream] {
                            let status = switch(stream, false);
                            assert!(settled(status), "unsubscribe: {status}");
                            // The slot is emptied even when the daemon cannot
                            // be told.
                            on[stream] = false;
                        } else {
                            deliveries.off[stream].store(false, Ordering::Relaxed);
                            let status = switch(stream, true);
                            assert!(settled(status), "subscribe: {status}");
                            // One that fails empties its slot again.
                            on[stream] = status == TOBII_ERROR_NO_ERROR;
                        }
                        deliveries.off[stream].store(!on[stream], Ordering::Relaxed);
                        changes += 1;
                        thread::sleep(TICK);
                    }
                    changes
                });
                let requests = s.spawn(|| {
                    let mut requests = 0usize;
                    while running() {
                        let (status, active) = calibration_active(d);
                        assert!(settled(status), "state: {status}");
                        if status == TOBII_ERROR_NO_ERROR {
                            assert_eq!(active, 1, "state");
                        }
                        let payload = requests.to_le_bytes();
                        match d.request(ECHOED, &payload, LONG_WAIT) {
                            Ok(echo) => assert_eq!(echo, payload, "a reply of its own"),
                            Err(status) => assert!(settled(status), "request: {status}"),
                        }
                        requests += 1;
                        thread::sleep(TICK);
                    }
                    requests
                });
                let clears = s.spawn(|| {
                    let h = handle(d);
                    let mut clears = 0usize;
                    while running() {
                        // SAFETY: as for `switch`.
                        let cleared = unsafe { tobii_device_clear_callback_buffers(h) };
                        assert_eq!(cleared, TOBII_ERROR_NO_ERROR, "clear");
                        clears += 1;
                        thread::sleep(CLEAR_EVERY);
                    }
                    clears
                });
                let reconnects = s.spawn(|| {
                    let h = handle(d);
                    // Those that succeeded, and those that failed.
                    let mut reconnects = [0usize; 2];
                    while running() {
                        thread::sleep(RECONNECT_EVERY);
                        // SAFETY: as for `switch`.
                        let reconnected = unsafe { tobii_device_reconnect(h) };
                        assert!(settled(reconnected), "reconnect: {reconnected}");
                        reconnects[usize::from(reconnected != TOBII_ERROR_NO_ERROR)] += 1;
                    }
                    reconnects
                });
                thread::sleep(HAMMER_FOR);
                stop.store(true, Ordering::Relaxed);
                (
                    joined(changes),
                    joined(requests),
                    joined(clears),
                    joined(reconnects),
                )
            });

            // SAFETY: as for `switch`.
            let process = || unsafe { tobii_device_process_callbacks(handles.device) };
            let mut processed = process();
            if life.is_some() {
                // Until the stand-in has hung up on the last connection too:
                // then every hang-up has been reported. Then once more, which
                // answers the same and logs nothing.
                while processed == TOBII_ERROR_NO_ERROR {
                    d.wait(LONG_WAIT);
                    processed = process();
                }
                processed = process();
            }
            let last = if life.is_some() {
                TOBII_ERROR_CONNECTION_FAILED
            } else {
                TOBII_ERROR_NO_ERROR
            };
            assert_eq!(processed, last, "the last process");
            let counts = [changes, requests, clears, reconnected];
            assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
            let streams = lock(&served.subscribed).clone();
            assert_eq!(
                streams.len(),
                1 + reconnected + unreconnected,
                "a connection per reconnect"
            );
            if life.is_none() {
                assert_eq!(
                    streams.last(),
                    Some(&d.mask()),
                    "the daemon was last asked for the callbacks' streams"
                );
            }
            let late = deliveries.late.load(Ordering::Relaxed);
            assert_eq!(late, 0, "a callback ran after its unsubscribe returned");
            deliveries.assert_in_turn();

            let lines = recorder.lines();
            let count = |level: LogLevel, text: &str| {
                let is = |(l, t): &&(LogLevel, String)| *l == level && t.starts_with(text);
                lines.iter().filter(is).count()
            };
            let (hang_ups, losses) = (
                served.hang_ups.load(Ordering::Relaxed),
                count(TOBII_LOG_LEVEL_ERROR, LOST),
            );
            let connected = (TOBII_LOG_LEVEL_INFO, "connected to tobiid".to_owned());
            assert_eq!(lines.first(), Some(&connected), "{lines:?}");
            assert_eq!(
                count(TOBII_LOG_LEVEL_INFO, "reconnected to tobiid"),
                reconnected
            );
            assert_eq!(
                count(TOBII_LOG_LEVEL_ERROR, "could not reconnect to tobiid"),
                unreconnected
            );
            assert_eq!(losses, hang_ups, "a loss logged for each hang-up, once");
            assert_eq!(hang_ups > 0, life.is_some(), "{hang_ups} hang-ups");
            assert_eq!(
                lines.len(),
                1 + reconnected + unreconnected + losses,
                "nothing else: {lines:?}"
            );
            assert_eq!(handles.destroy(), [TOBII_ERROR_NO_ERROR; 2]);
        });
    }
}
