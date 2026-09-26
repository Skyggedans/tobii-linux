//! The API and device handles: a device is a connection to the `tobiid`
//! daemon with a reader thread sorting decoded messages into two channels
//! (the answers to requests and subscription changes, and the samples), the
//! registered callbacks, and a synchronous request/reply helper.
//!
//! Invariant: every stored callback was registered through the matching
//! `tobii_*_subscribe` entry point, whose safety contract makes it sound to
//! invoke with the stored `user_data` until it is unsubscribed or the device
//! is destroyed.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt;
use std::io;
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tobii_ipc::request::{DeviceInfo as DeviceInfoMsg, encode_request};
use tobii_ipc::{
    self, NotificationValue as WireValue, STREAM_EYE_POSITION, STREAM_GAZE, STREAM_GAZE_DATA,
    STREAM_GAZE_ORIGIN, STREAM_HEAD, STREAM_IMAGE, STREAM_NOTIFICATIONS, STREAM_PRESENCE,
    ServerMsg, decode_server, encode_subscribe, read_frame, write_frame,
};

use crate::logger::{self, Level, Logger};
use crate::status::{
    Status, TOBII_ERROR_ALREADY_SUBSCRIBED, TOBII_ERROR_CALLBACK_IN_PROGRESS,
    TOBII_ERROR_CONFLICTING_API_INSTANCES, TOBII_ERROR_CONNECTION_FAILED, TOBII_ERROR_INTERNAL,
    TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR, TOBII_ERROR_NOT_SUBSCRIBED,
    TOBII_ERROR_TIMED_OUT,
};
use crate::types::{
    DisplayArea, EyePair, EyePairFn, FieldOfUse, FieldOfUseFn, GazeData, GazeDataEye, GazeDataFn,
    GazePoint, GazePointFn, HeadPose, HeadPoseFn, Image, ImageFn, Notification, NotificationValue,
    NotificationsFn, PresenceFn, PresenceStatus, TOBII_VALIDITY_INVALID, TOBII_VALIDITY_VALID,
    Validity, copy_c_string,
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
    /// Set while this thread runs application code: a user callback or the
    /// application's logger. The entry points the crate documentation lists
    /// refuse a call made from inside one, as the Stream Engine does for a
    /// callback: re-entering with the device being dispatched (or logging)
    /// would alias its `&mut`, and destroying it would free it under the
    /// dispatch loop.
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is inside a user callback or the logger.
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

/// One daemon connection and the thread reading it, which sorts what the
/// daemon sends into two channels, each in the order it came: the answers,
/// read only by a request or subscription change waiting for its own, and
/// the samples, read only by `wait`, `process` and `clear_buffers`. A call
/// waiting for its answer leaves the samples where they are, and an answer
/// nobody waits for any more never wakes `wait`.
struct Link {
    stream: UnixStream,
    answers: Receiver<Answer>,
    samples: Receiver<ServerMsg>,
    reader: Option<JoinHandle<()>>,
    /// Kept with the link, so a new one (a reconnect) starts `Up` and a later
    /// loss is reported again.
    state: LinkState,
    /// The acks still to come for subscription changes that gave up waiting.
    /// tobiid acks every change on a connection, in order, so the next this
    /// many acks are theirs, and a later change reads past them to its own
    /// rather than take a late one for it. A reply needs no count: it names
    /// its request. Kept with the link, so a new one owes none.
    acks_owed: u32,
}

impl Link {
    fn open(connect: &mut Connector) -> io::Result<Self> {
        let stream = connect()?;
        let reader_stream = stream.try_clone()?;
        let (answers_tx, answers) = mpsc::channel();
        let (samples_tx, samples) = mpsc::channel();
        let reader = thread::Builder::new()
            .name("tobii-ffi-reader".into())
            .spawn(move || reader_loop(reader_stream, &answers_tx, &samples_tx))?;
        Ok(Self {
            stream,
            answers,
            samples,
            reader: Some(reader),
            state: LinkState::Up,
            acks_owed: 0,
        })
    }

    /// Close the connection and join the reader. Everything it read is then
    /// in the channels, each followed by a disconnect.
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

    /// Note that the connection has ended, and close it: the next `process`
    /// reports the loss whether or not the reader has seen it.
    fn lose(&mut self) {
        if self.state == LinkState::Up {
            self.close();
            self.state = LinkState::Lost;
        }
    }

    /// Write one frame to the daemon. A failed write loses the connection:
    /// the daemon has closed it, or the frame is cut short mid-stream. A body
    /// too long for a frame fails before a byte is written, and leaves the
    /// connection as it was.
    fn send(&mut self, body: &[u8]) -> Result<(), Status> {
        if u32::try_from(body.len()).is_err() {
            tracing::debug!(len = body.len(), "frame body too long for tobiid");
            return Err(TOBII_ERROR_CONNECTION_FAILED);
        }
        write_frame(&mut self.stream, body).map_err(|e| {
            tracing::debug!(error = %e, "could not write to tobiid");
            self.lose();
            TOBII_ERROR_CONNECTION_FAILED
        })
    }

    /// Wait up to `timeout` for the daemon's next answer. The samples that
    /// arrive meanwhile stay queued for `process`.
    fn recv_answer(&mut self, timeout: Duration) -> Result<Answer, Status> {
        match self.answers.recv_timeout(timeout) {
            Ok(answer) => Ok(answer),
            Err(RecvTimeoutError::Timeout) => Err(TOBII_ERROR_TIMED_OUT),
            Err(RecvTimeoutError::Disconnected) => {
                self.lose();
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

/// Pump frames from the daemon until EOF, a read error, or the receiving
/// `Link` going away: answers into `answers`, and everything else (the
/// samples, and message kinds a newer daemon may add) into `samples`.
fn reader_loop(mut stream: UnixStream, answers: &Sender<Answer>, samples: &Sender<ServerMsg>) {
    while let Ok(Some(body)) = read_frame(&mut stream) {
        let Some(msg) = decode_server(&body) else {
            continue;
        };
        let sent = match Answer::try_from(msg) {
            Ok(answer) => answers.send(answer).is_ok(),
            Err(sample) => samples.send(sample).is_ok(),
        };
        if !sent {
            break;
        }
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
    pub(crate) image: Slot<ImageFn>,
    pub(crate) notifications: Slot<NotificationsFn>,
    /// Registered but never called: the field of use cannot change.
    pub(crate) field_of_use: Slot<FieldOfUseFn>,
}

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
        need(self.image.is_some(), STREAM_IMAGE);
        need(self.notifications.is_some(), STREAM_NOTIFICATIONS);
        mask
    }
}

/// Opaque device handle.
pub struct Device {
    link: Link,
    connect: Connector,
    pending: VecDeque<ServerMsg>,
    pub(crate) callbacks: Callbacks,
    pub(crate) field_of_use: FieldOfUse,
    /// Address of the API handle this device was created from.
    pub(crate) api: usize,
    next_request_id: u32,
    /// Identity, fetched once per connection: a reconnect clears it, since a
    /// restarted daemon may serve another tracker.
    pub(crate) device_info: Option<DeviceInfoMsg>,
    /// The logger of the API this device was created from, copied, so the
    /// device keeps it after `tobii_api_destroy`.
    pub(crate) logger: Option<Logger>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("link", &self.link.state)
            .field("streams", &self.callbacks.mask())
            .field("pending", &self.pending.len())
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
        let link = Link::open(&mut connect)?;
        Ok(Self {
            link,
            connect,
            pending: VecDeque::new(),
            callbacks: Callbacks::default(),
            field_of_use,
            api,
            next_request_id: 0,
            device_info: None,
            logger: None,
        })
    }

    /// Take the logger of the API this device was created from, and say it
    /// has connected, as the DLL says on each connect ("Connected to
    /// platform module", INFO, 0x180153c19).
    pub(crate) fn adopt(&mut self, logger: Option<Logger>) {
        self.logger = logger;
        self.log(Level::Info, format_args!("connected to tobiid"));
    }

    /// Log a line to the device's logger (see [`crate::logger`]).
    pub(crate) fn log(&self, level: Level, args: fmt::Arguments<'_>) {
        logger::emit(self.logger, level, args);
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

    /// Connect to the daemon, spawning it if needed. A reconnect only
    /// connects: it fails at once when no daemon listens, and never spawns
    /// one, which could start a daemon outside systemd or race other clients
    /// into starting two.
    #[cfg(not(test))]
    pub(crate) fn connect_daemon(api: usize, field_of_use: FieldOfUse) -> io::Result<Self> {
        let connect = first_then(tobii_ipc::connect_or_spawn, tobii_ipc::connect);
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

    /// Write one frame to the daemon (see [`Link::send`]).
    pub(crate) fn send(&mut self, body: &[u8]) -> Result<(), Status> {
        self.link.send(body)
    }

    /// Resend the subscription mask and wait for the daemon's ack (see
    /// [`Link::send_subscription`]). Returns the ack's `ok` flag.
    fn resend_subscription(&mut self) -> Result<bool, Status> {
        self.link
            .send_subscription(self.callbacks.mask(), SUBSCRIBE_ACK_TIMEOUT)
    }

    /// Register `callback` in `slot` and subscribe its stream. The Stream
    /// Engine's rules: a missing callback is invalid, an occupied slot is
    /// already subscribed.
    pub(crate) fn subscribe<F: Copy>(
        &mut self,
        slot: fn(&mut Callbacks) -> &mut Slot<F>,
        callback: Option<F>,
        user_data: *mut c_void,
    ) -> Status {
        let Some(callback) = callback else {
            return TOBII_ERROR_INVALID_PARAMETER;
        };
        if slot(&mut self.callbacks).is_some() {
            return TOBII_ERROR_ALREADY_SUBSCRIBED;
        }
        let before = self.callbacks.mask();
        *slot(&mut self.callbacks) = Some((callback, user_data));
        if self.callbacks.mask() == before {
            return TOBII_ERROR_NO_ERROR;
        }
        let status = match self.resend_subscription() {
            Ok(true) => TOBII_ERROR_NO_ERROR,
            Ok(false) => TOBII_ERROR_CONFLICTING_API_INSTANCES,
            Err(status) => status,
        };
        if status != TOBII_ERROR_NO_ERROR {
            *slot(&mut self.callbacks) = None;
        }
        status
    }

    /// Drop the callback in `slot` and, if no other callback needs its
    /// stream, tell the daemon.
    pub(crate) fn unsubscribe<F: Copy>(
        &mut self,
        slot: fn(&mut Callbacks) -> &mut Slot<F>,
    ) -> Status {
        let before = self.callbacks.mask();
        if slot(&mut self.callbacks).take().is_none() {
            return TOBII_ERROR_NOT_SUBSCRIBED;
        }
        if self.callbacks.mask() == before {
            return TOBII_ERROR_NO_ERROR;
        }
        match self.resend_subscription() {
            Ok(_) => TOBII_ERROR_NO_ERROR,
            Err(status) => status,
        }
    }

    /// Send a request and wait up to `timeout` for its reply, dropping the
    /// stale replies of requests that gave up, and the late acks it reads
    /// past, which are then owed no more (see [`Link::acks_owed`]; no
    /// subscription change waits at the same time). The samples that arrive
    /// meanwhile stay queued for `process`. A reply with a non-zero status is
    /// that status.
    pub(crate) fn request(
        &mut self,
        kind: u8,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<Vec<u8>, Status> {
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        let id = self.next_request_id;
        self.send(&encode_request(id, kind, payload))?;
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
                } if request_id == id => {
                    return if status == 0 {
                        Ok(payload)
                    } else {
                        Err(Status::from(status))
                    };
                }
                Answer::Reply { request_id, .. } => {
                    tracing::debug!(request_id, "stale reply dropped");
                }
                Answer::Subscribed(_) => self.link.late_ack(),
            }
        }
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
    /// Samples queued from the old link are dropped. The new link starts up,
    /// so its loss is reported again, and owes no acks. The device info is
    /// fetched again: a restarted daemon may serve another tracker. Each
    /// failure is logged at ERROR, a success at INFO.
    pub(crate) fn reconnect(&mut self) -> Status {
        let mut link = match Link::open(&mut self.connect) {
            Ok(link) => link,
            Err(e) => {
                self.log(
                    Level::Error,
                    format_args!("could not reconnect to tobiid: {e}"),
                );
                return TOBII_ERROR_CONNECTION_FAILED;
            }
        };
        let mask = self.callbacks.mask();
        if mask != 0 {
            let why = match link.send_subscription(mask, RECONNECT_ACK_TIMEOUT) {
                Ok(true) => None,
                Ok(false) => Some("refused them"),
                Err(TOBII_ERROR_TIMED_OUT) => Some("did not acknowledge them in time"),
                Err(_) => Some("hung up"),
            };
            if let Some(why) = why {
                self.log(
                    Level::Error,
                    format_args!(
                        "could not reconnect to tobiid: asked for the subscriptions back \
                         (streams {mask:#x}), it {why}"
                    ),
                );
                return TOBII_ERROR_CONNECTION_FAILED;
            }
        }
        self.link = link;
        self.pending.clear();
        self.device_info = None;
        self.log(Level::Info, format_args!("reconnected to tobiid"));
        TOBII_ERROR_NO_ERROR
    }

    /// Drop every queued sample. A lost connection stays lost, and a loss not
    /// reported yet is still reported. The answers are left alone: a late ack
    /// dropped here would still be counted as owed.
    pub(crate) fn clear_buffers(&mut self) {
        while self.link.samples.try_recv().is_ok() {}
        self.pending.clear();
    }

    /// Whether `process` has something to do, waiting up to `timeout` for
    /// it: a queued sample, or a lost connection it has not reported yet.
    /// Once it has, nothing arrives until a reconnect, so this sleeps out
    /// `timeout` and says no, as for a quiet link; answering at once would
    /// spin a wait-and-process loop. An answer (a stale reply, a late ack)
    /// is nothing to process and does not wake it.
    pub(crate) fn wait(&mut self, timeout: Duration) -> bool {
        if !self.pending.is_empty() {
            return true;
        }
        if self.link.state == LinkState::Up {
            match self.link.samples.recv_timeout(timeout) {
                Ok(msg) => {
                    self.pending.push_back(msg);
                    return true;
                }
                Err(RecvTimeoutError::Timeout) => return false,
                Err(RecvTimeoutError::Disconnected) => self.link.lose(),
            }
        }
        if self.link.state == LinkState::Reported {
            thread::sleep(timeout);
            return false;
        }
        true
    }

    /// Deliver every queued sample to its callbacks, on this thread, then
    /// say whether the daemon connection is up. Once it is lost, what had
    /// arrived before is still delivered, and then this call and every later
    /// one returns `TOBII_ERROR_CONNECTION_FAILED` until a reconnect connects
    /// again.
    #[must_use]
    pub(crate) fn process(&mut self) -> Status {
        loop {
            match self.link.samples.try_recv() {
                Ok(msg) => self.pending.push_back(msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.link.lose();
                    break;
                }
            }
        }
        while let Some(msg) = self.pending.pop_front() {
            self.dispatch(&msg);
        }
        match self.link.state {
            LinkState::Up => TOBII_ERROR_NO_ERROR,
            LinkState::Lost => {
                // Once per loss: a host keeps calling at its frame rate.
                self.link.state = LinkState::Reported;
                self.log(
                    Level::Error,
                    format_args!(
                        "lost the connection to tobiid; tobii_device_reconnect restores it"
                    ),
                );
                TOBII_ERROR_CONNECTION_FAILED
            }
            LinkState::Reported => TOBII_ERROR_CONNECTION_FAILED,
        }
    }

    /// Deliver one daemon message to the matching callbacks, if any. The
    /// timestamps go through as the daemon sent them, on the host clock
    /// already (gaze data's tracker time aside): nothing is converted here.
    fn dispatch(&self, msg: &ServerMsg) {
        let cb = &self.callbacks;
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
                    // (see the module invariant); `hp` outlives the call.
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

/// Run application code (a user callback or the logger) with the re-entry
/// guard set, then leave the guard as it was: the logger can run inside a
/// callback, whose dispatch still needs the guard once the line is logged.
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

/// Borrow a device handle from C.
///
/// Refuses every call made from inside a callback (as the Stream Engine
/// does), then a null handle.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the borrow.
pub(crate) unsafe fn device_mut<'a>(device: *mut Device) -> Result<&'a mut Device, Status> {
    if in_callback() {
        return Err(TOBII_ERROR_CALLBACK_IN_PROGRESS);
    }
    // SAFETY: the caller guarantees `device` is null or live and unaliased.
    unsafe { device.as_mut() }.ok_or(TOBII_ERROR_INVALID_PARAMETER)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::logger::tests::Recorder;
    use crate::types::{LogLevel, TOBII_LOG_LEVEL_ERROR, TOBII_LOG_LEVEL_INFO};
    use std::ffi::c_char;
    use tobii_ipc::{
        STREAM_GAZE_ORIGIN, encode_gaze, encode_gaze_origin, encode_reply, encode_subscribed,
    };

    thread_local! {
        /// The daemon stand-in the next device constructor on this thread
        /// connects to, taken by the test build of `Device::connect_daemon`;
        /// with none there, the constructor fails to connect.
        pub(crate) static DAEMON: Cell<Option<Connector>> = const { Cell::new(None) };
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        let mut stamps = Stamps::default();
        d.callbacks.gaze = Some((stamp_gaze as GazePointFn, (&raw mut stamps).cast()));

        let got = d.request(1, &[], Duration::from_secs(2));

        assert_eq!(got, Ok(b"answer".to_vec()));
        assert!(d.pending.is_empty(), "the request moved no sample");
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );

        assert!(d.pending.is_empty(), "the change moved no sample");
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        assert_eq!(
            d.link.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
            Err(TOBII_ERROR_TIMED_OUT)
        );
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.eye_position, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_CONFLICTING_API_INSTANCES
        );

        assert_eq!(d.callbacks.mask(), 0, "rolled back");
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        assert_eq!(
            d.link.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        assert_eq!(
            d.link.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
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
        d.callbacks.gaze_origin = Some((count_pair as EyePairFn, (&raw mut hits).cast()));

        d.clear_buffers();

        let t = Instant::now();
        assert_eq!(
            d.subscribe(
                |c| &mut c.eye_position,
                Some(ignore_pair as EyePairFn),
                std::ptr::null_mut()
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
        let mut d = Device::new(connect, 1, 1).expect("device");
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
                std::ptr::null_mut()
            ),
            TOBII_ERROR_NO_ERROR
        );
        assert!(t.elapsed() < PROMPT);
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
    }

    #[test]
    fn a_failed_reply_is_its_status_and_silence_times_out() {
        let mut d = device_with(15, vec![]);
        assert_eq!(d.request(0x10, &[2], Duration::from_secs(2)), Err(15));

        let mut quiet = Device::new(fake_daemon(|_| vec![]), 1, 1).expect("device");
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
        let mut d = device_with(0, vec![]);
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
        assert_eq!(d.callbacks.mask(), STREAM_GAZE_ORIGIN);
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
        let mut d = device_with(0, vec![]);
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

        d.dispatch(&ServerMsg::EyePosition(tobii_ipc::EyePair::default()));

        assert_eq!(hits, 2);
        assert_eq!(d.unsubscribe(|c| &mut c.eye_position), 0);
        assert_eq!(d.callbacks.mask(), STREAM_EYE_POSITION);
    }

    #[test]
    fn samples_are_dispatched_in_the_c_layout() {
        let mut d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        let body = encode_gaze_origin(&tobii_ipc::EyePair::default());
        d.pending.push_back(decode_server(&body).expect("decodes"));

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

    unsafe extern "C" fn stamp_image(p: *const Image, ud: *mut c_void) {
        // SAFETY: `ud` is a live `Stamps`, `p` a live sample (see above).
        unsafe { (*ud.cast::<Stamps>()).image = (*p).timestamp_us };
    }

    /// libtobii hands each callback the timestamp its frame carried, which
    /// the daemon sends on the host clock, and gaze data's tracker time
    /// beside it: nothing is converted on this side.
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        let mut stamps = Stamps::default();
        let ud = (&raw mut stamps).cast::<c_void>();
        d.callbacks = Callbacks {
            head: Some((stamp_head as HeadPoseFn, ud)),
            gaze: Some((stamp_gaze as GazePointFn, ud)),
            presence: Some((stamp_presence as PresenceFn, ud)),
            gaze_origin: Some((stamp_gaze_origin as EyePairFn, ud)),
            eye_position: Some((stamp_eye_position as EyePairFn, ud)),
            user_position_guide: Some((stamp_user_position_guide as EyePairFn, ud)),
            gaze_data: Some((stamp_gaze_data as GazeDataFn, ud)),
            image: Some((stamp_image as ImageFn, ud)),
            ..Callbacks::default()
        };

        assert_eq!(d.resend_subscription(), Ok(true));
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
        let mut d = Device::new(connect, 1, 1).expect("device");
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
        *out = match unsafe { device_mut(std::ptr::null_mut()) } {
            Err(s) => s,
            Ok(_) => 0,
        };
    }

    #[test]
    fn calls_from_inside_a_callback_are_refused() {
        let mut d = device_with(0, vec![]);
        let mut seen: Status = -1;
        let ud = (&raw mut seen).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(reenter as EyePairFn), ud),
            0
        );

        d.dispatch(&ServerMsg::GazeOrigin(tobii_ipc::EyePair::default()));

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
        let mut d = device_with(0, vec![]);
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
            d.pending.push_back(decode_server(&body).expect("decodes"));
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

    #[test]
    fn reconnect_restores_the_subscription() {
        let mut d = device_with(0, vec![]);
        let ud = std::ptr::null_mut();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(d.reconnect(), 0);
        assert_eq!(d.callbacks.mask(), STREAM_GAZE_ORIGIN);
        d.clear_buffers();
        assert!(!d.wait(Duration::from_millis(10)));
    }

    /// Far longer than any wait answered at once takes.
    const LONG_WAIT: Duration = Duration::from_secs(5);
    /// What "at once" means here: well under `LONG_WAIT`.
    const PROMPT: Duration = Duration::from_secs(1);
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
    fn hung_up(d: &mut Device) {
        if let Some(h) = d.link.reader.take() {
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        let mut daemon = daemons.recv().expect("daemon end");
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_NO_ERROR
        );
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        drop(daemon);
        hung_up(&mut d);
        (d, daemons)
    }

    #[test]
    fn process_delivers_what_arrived_then_reports_the_loss() {
        let mut hits = 0u32;
        let (mut d, _daemons) = lost_device(2, vec![], (&raw mut hits).cast());

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
        let (mut d, _daemons) = lost_device(0, vec![], (&raw mut hits).cast());

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
        let (mut d, _daemons) = lost_device(1, vec![], (&raw mut hits).cast());

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
        let mut d = Device::new(connect, 1, 1).expect("device");
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
        let (mut d, daemon) = deaf_daemon_device((&raw mut hits).cast());

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
        let (mut d, _daemons) = lost_device(0, vec![], ud);

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
        assert_eq!(d.callbacks.mask(), STREAM_GAZE_ORIGIN, "rolled back");
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    #[test]
    fn reconnect_restores_the_subscription_and_samples_flow_again() {
        let mut hits = 0u32;
        let (mut d, daemons) =
            lost_device(0, vec![ack_then_gaze_origin(1)], (&raw mut hits).cast());
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
        let (mut d, daemons) =
            lost_device(0, vec![ack_then_gaze_origin(0)], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);
        let mut daemon = daemons.recv().expect("second daemon end");
        assert_eq!(subscription(&mut daemon), Some(STREAM_GAZE_ORIGIN));
        assert_eq!(d.process(), TOBII_ERROR_NO_ERROR);

        drop(daemon);
        hung_up(&mut d);

        let t = Instant::now();
        assert!(d.wait(LONG_WAIT), "the new loss is something to process");
        assert!(t.elapsed() < PROMPT);
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
    }

    /// Wait out `SHORT_WAIT` and say no, as for a loss already reported, then
    /// report the loss still: the device is lost, and was not woken again.
    fn assert_still_lost(d: &mut Device) {
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
        let (mut d, daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        drop(daemons);

        let t = Instant::now();
        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        assert!(t.elapsed() < PROMPT);

        assert_still_lost(&mut d);
    }

    /// The daemon takes the connection and the subscription but never acks:
    /// the reconnect gives up after its own short timeout, not the 2 s a
    /// subscription change waits, with `TOBII_ERROR_CONNECTION_FAILED`, never
    /// `TOBII_ERROR_TIMED_OUT`, and hangs up on that daemon.
    #[test]
    fn a_reconnect_the_daemon_does_not_ack_fails_soon_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        // The second connection's daemon is sent nothing to answer with.
        let (mut d, daemons) = lost_device(0, vec![], (&raw mut hits).cast());
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
        assert_still_lost(&mut d);
    }

    #[test]
    fn a_reconnect_the_daemon_hangs_up_on_fails_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        let (mut d, daemons) = lost_device(0, vec![], (&raw mut hits).cast());
        assert_eq!(d.process(), TOBII_ERROR_CONNECTION_FAILED);
        // Reads the subscription, then hangs up without acking it.
        let daemon = thread::spawn(move || {
            let mut daemon = daemons.recv().expect("second daemon end");
            subscription(&mut daemon)
        });

        assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);

        assert_eq!(daemon.join().expect("daemon"), Some(STREAM_GAZE_ORIGIN));
        assert_still_lost(&mut d);
    }

    /// No daemon refuses a subscription today, but only an ack that takes
    /// them counts: a reconnect never reports success for streams the
    /// daemon will not serve.
    #[test]
    fn a_reconnect_the_daemon_refuses_fails_and_leaves_the_device_lost() {
        let mut hits = 0u32;
        let refusal = vec![encode_subscribed(false)];
        let (mut d, daemons) = lost_device(0, vec![refusal], (&raw mut hits).cast());
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
        assert_eq!(d.callbacks.mask(), STREAM_GAZE_ORIGIN, "kept for a retry");
        assert_still_lost(&mut d);
    }

    /// Were the old connection closed first, the daemon could see no client
    /// wanting the streams between the two and stop the tracker.
    #[test]
    fn reconnecting_a_live_connection_subscribes_the_new_one_before_closing_the_old() {
        let (connect, daemons) = scripted_daemon(vec![ack_then_gaze_origin(0)]);
        let mut d = Device::new(connect, 1, 1).expect("device");
        let mut old = daemons.recv().expect("daemon end");
        let ud = std::ptr::null_mut();
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        let old = daemons.recv().expect("daemon end");
        d.callbacks.gaze_origin = Some((ignore_pair as EyePairFn, std::ptr::null_mut()));
        assert_eq!(
            d.link.send_subscription(STREAM_GAZE_ORIGIN, SHORT_WAIT),
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
        let mut d = Device::new(connect, 1, 1).expect("device");
        let ud = std::ptr::null_mut();
        d.callbacks = Callbacks {
            head: Some((ignore_head as HeadPoseFn, ud)),
            gaze: Some((ignore_gaze as GazePointFn, ud)),
            presence: Some((ignore_presence as PresenceFn, ud)),
            gaze_origin: Some((ignore_pair as EyePairFn, ud)),
            user_position_guide: Some((ignore_pair as EyePairFn, ud)),
            gaze_data: Some((ignore_gaze_data as GazeDataFn, ud)),
            image: Some((ignore_image as ImageFn, ud)),
            notifications: Some((ignore_notification as NotificationsFn, ud)),
            ..Callbacks::default()
        };
        let serial =
            |d: &mut Device| crate::api::fetch_device_info(d).map(|info| info.serial_number);
        assert_eq!(serial(&mut d), Ok("1".into()));
        assert_eq!(serial(&mut d), Ok("1".into()), "fetched once");

        assert_eq!(d.reconnect(), TOBII_ERROR_NO_ERROR);

        let every = STREAM_HEAD
            | STREAM_GAZE
            | STREAM_PRESENCE
            | STREAM_GAZE_ORIGIN
            | STREAM_EYE_POSITION
            | STREAM_GAZE_DATA
            | STREAM_IMAGE
            | STREAM_NOTIFICATIONS;
        assert_eq!(*masks.lock().expect("log"), [Some(every)]);
        assert_eq!(serial(&mut d), Ok("2".into()), "asked the daemon again");
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
        d.logger = recorder.logger();

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
        d.logger = recorder.logger();

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
        d.logger = recorder.logger();
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

    /// The device that logs is borrowed while its logger runs, so the
    /// logger may not use it: the guard refuses the call, as from a callback.
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
            (*d).logger = crate::logger::tests::logger(process_from_the_logger, context);
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
    /// passes `tobii_ipc::connect_or_spawn`), and every reconnect through
    /// `later` (`tobii_ipc::connect`, which never spawns).
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
        let mut d = Device::new(connect, 1, 1).expect("device");

        for _ in 0..2 {
            assert_eq!(d.reconnect(), TOBII_ERROR_CONNECTION_FAILED);
        }

        let calls = (
            firsts.load(Ordering::Relaxed),
            laters.load(Ordering::Relaxed),
        );
        assert_eq!(calls, (1, 2));
    }
}
