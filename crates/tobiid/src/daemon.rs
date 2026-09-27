//! `tobiid`: the single process that claims the Tobii device. It owns one
//! `Engine`, accepts client connections over a Unix socket, fans out samples
//! to subscribed clients and answers their requests. Every stream is served
//! by that one engine (the device's 0x50e IR image stream gives head pose
//! concurrently with gaze — see `engine`); head-pose inference and image
//! fan-out are switched on only while some client subscribes to them.
//!
//! Only the pump thread writes to client sockets: replies are queued in the
//! client's outbox and sent ahead of the next samples, so a reply can never
//! interleave with a sample frame.
//!
//! Each client has a reader thread, which handles its subscription changes
//! and recenter requests, and from its first request on a request worker,
//! which runs its requests one at a time in the order they came. A request can wait on the device for
//! long (a cold engine takes ~12 s to arm, and a command queued meanwhile
//! may wait 30 s for it), and the client's subscription changes do not wait
//! behind it: libtobii gives up on a subscription's ack after 2 s.
//!
//! Log lines go through `tracing` (the `tobiid` binary installs the
//! subscriber; under systemd stderr lands in the journal).

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io;
use std::net::Shutdown;
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use tobii_ipc::geometry::DisplayArea;
use tobii_ipc::request::decode_request;
use tobii_ipc::{
    self, STREAM_HEAD, STREAM_IMAGE, STREAM_PRESENCE, decode_subscribe, encode_reply,
    encode_subscribed, read_frame, write_frame,
};
use tobii_proto::facts::{DeviceFacts, DeviceNotification};
use tobii_usb::device::{BusAddress, OpenRefusal};
use tobii_usb::engine::{Engine, PresenceSample, Sample};

use crate::calibration::Calibration;
use crate::device::DeviceCommands;
use crate::frames::{presence_frame, push_sample_frames};
use crate::restart::Backoff;

/// Set by the SIGUSR1 handler; a poller thread turns it into a recenter request.
static RECENTER_SIGNAL: AtomicBool = AtomicBool::new(false);

/// Set by the SIGTERM/SIGINT handler; a poller thread turns it into a graceful
/// shutdown that drops the engine (running the device teardown) before exiting.
static SHUTDOWN_SIGNAL: AtomicBool = AtomicBool::new(false);

// Both signal flags are pure signals (no data is published with them), so
// every access uses `Ordering::Relaxed`.

extern "C" fn on_sigusr1(_sig: libc::c_int) {
    // Async-signal-safe: only an atomic store.
    RECENTER_SIGNAL.store(true, Ordering::Relaxed);
}

extern "C" fn on_shutdown(_sig: libc::c_int) {
    // Async-signal-safe: only an atomic store. The poller thread does the real
    // work (Rust does not run Drop on a signal-terminated process, so we must
    // shut down cooperatively to get the device teardown to run).
    SHUTDOWN_SIGNAL.store(true, Ordering::Relaxed);
}

pub(crate) struct Client {
    pub(crate) id: u64,
    pub(crate) streams: u32,
    out: UnixStream,
    /// Frames for this client only (replies, a cached presence), written by
    /// the pump ahead of the next samples.
    outbox: VecDeque<Vec<u8>>,
    /// Made a request: keep the device up while it is connected, as an open
    /// `tobii_device_t` does.
    pub(crate) holds_device: bool,
    /// The pump found the connection dead and hung up on it (see
    /// [`Self::hang_up`]): nothing more is written or queued for it.
    hung_up: bool,
}

impl Client {
    /// Client `id`, connected over `out`, as the accept loop registers it.
    fn new(id: u64, out: UnixStream) -> Self {
        Self {
            id,
            streams: 0,
            out,
            outbox: VecDeque::new(),
            holds_device: false,
            hung_up: false,
        }
    }

    /// Stop serving a connection a write failed on: shut its socket down,
    /// which ends its reader's reads once it has read what the client sent,
    /// even if the peer keeps the socket open without reading, and drop
    /// what waits in its outbox. The client stays listed, with the streams
    /// and the device it holds: its reader cleans up after it once its
    /// requests have run (see [`client_reader`]).
    pub(crate) fn hang_up(&mut self) {
        // Nothing to do if it fails: the socket is being given up on.
        let _ = self.out.shutdown(Shutdown::Both);
        self.outbox = VecDeque::new();
        self.hung_up = true;
    }

    /// Write the outbox, then those of `frames` (bodies tagged with the
    /// stream they carry) this client subscribes to. False once a write
    /// fails: what is left is not written.
    fn write_out(&mut self, frames: &[(u32, Vec<u8>)]) -> bool {
        while let Some(body) = self.outbox.pop_front() {
            if write_frame(&mut self.out, &body).is_err() {
                return false;
            }
        }
        frames
            .iter()
            .filter(|(stream, _)| self.streams & stream != 0)
            .all(|(_, body)| write_frame(&mut self.out, body).is_ok())
    }
}

pub(crate) struct State {
    pub(crate) engine: Option<Engine>,
    /// Every connection, from its accept until its reader has cleaned up
    /// after it; one the pump hung up on stays listed until then.
    pub(crate) clients: Vec<Client>,
    /// Keep the engine running even with no clients, so the device stays warm
    /// and client connects are instant (`TOBII_PREWARM`).
    prewarm: bool,
    /// What the device reported at its last init (kept across restarts).
    pub(crate) facts: Option<Arc<DeviceFacts>>,
    /// The last presence report (the device sends one at stream start and
    /// on change), replayed to each new presence subscriber with the host
    /// time it was stamped with.
    last_presence: Option<PresenceSample>,
    /// The newest `(device_us, host_rx_us)` pair seen on the gaze stream: a
    /// frame's device timestamp and the host time it was read at.
    /// Cleared whenever the engine is dropped or replaced and at every
    /// device init: the device clock restarts with the device, so a pair
    /// holds for one init only.
    pub(crate) clock: Option<(i64, i64)>,
    /// Gaze frames seen so far. A TIMESYNC waits for this to move rather
    /// than for a host time past its own (see `requests::timesync`).
    pub(crate) gaze_frames: u64,
    /// The display area a client set (or the saved one), re-applied at every
    /// device init.
    pub(crate) display_override: Option<DisplayArea>,
    /// `TOBII_DISPLAY_MM`: the monitor to configure once the mounting is known.
    pub(crate) display_request: Option<crate::requests::DisplaySize>,
    /// Where a display area a client sets is saved (see [`crate::display`]);
    /// `None` keeps it in memory only.
    pub(crate) display_file: Option<PathBuf>,
    /// The name a client gave the device (or the saved one); `None` until
    /// one is set, when the model stands in (see [`crate::name`]).
    pub(crate) device_name: Option<Vec<u8>>,
    /// Where a device name a client sets is saved; `None` keeps it in memory
    /// only.
    pub(crate) name_file: Option<PathBuf>,
    /// Held while a name is saved and recorded, so that the file and
    /// `device_name` agree. Taken before the state lock, never under it.
    pub(crate) name_lock: Arc<Mutex<()>>,
    /// The calibration session, if any, and the active calibration id.
    pub(crate) calibration: Calibration,
    /// Whether the device is paused, as clients are told (see
    /// [`crate::pause`]).
    pub(crate) paused: bool,
    /// The client whose pause is in effect; the device resumes when it goes
    /// away.
    pub(crate) pause_holder: Option<u64>,
    /// A pause is on its way to the device: no calibration may start.
    pub(crate) pausing: bool,
    /// Held while a pause or resume runs, so that the device and `paused`
    /// agree. Taken before the state lock, never under it.
    pub(crate) pause_lock: Arc<Mutex<()>>,
    /// How many times the engine was lost. A pause the device accepted
    /// counts only if no loss came in between: the next init resumes it.
    /// Likewise a calibration start: one the device took counts only if no
    /// loss came in between (see [`crate::calibration`]).
    pub(crate) engine_losses: u64,
    /// How many times the device finished an init. A calibration start
    /// counts only if none came after the device answered its 1010: the
    /// init ended the session (see [`crate::calibration`]).
    pub(crate) device_inits: u64,
    /// The log said the tracker is off the bus since an engine last
    /// started: it says so once per absence, not at every request.
    tracker_absence_logged: bool,
    /// The engine running, or the last one, got the tracker ready: its
    /// init went through (`DeviceReady`) or a gaze frame came.
    engine_armed: bool,
    /// When the engine running, or the last one, started.
    engine_started_at: Option<Instant>,
    /// When the watchdog may start an engine again after engines that
    /// ended before the tracker was ready (see [`crate::restart`]).
    restarts: Backoff,
    /// Stand-in for the engine's command queue in tests.
    #[cfg(test)]
    pub(crate) fake_device: Option<Arc<dyn DeviceCommands>>,
    /// The pause hint last given to the engine, in tests.
    #[cfg(test)]
    pub(crate) pause_hint: Option<bool>,
    /// The next device fetch in tests finds the engine dead: it goes
    /// through [`Self::ensure_engine`], which drops it and, for a tracker
    /// on the bus, records a start in place of another (no engine starts in
    /// tests); the stand-in stands for that one, and without a tracker it
    /// goes too.
    #[cfg(test)]
    pub(crate) lose_engine_on_fetch: bool,
    /// Whether a tracker is on the bus, in tests: none unless a test says
    /// so.
    #[cfg(test)]
    pub(crate) fake_present: bool,
    /// The address of that tracker on bus 1, in tests: another one is a
    /// re-plug.
    #[cfg(test)]
    pub(crate) fake_address: u8,
    /// Stand-in for an engine that has ended, in tests, where none really
    /// starts: what [`Engine::finish`] says of it. It is dropped as a dead
    /// engine is, and its end noted (see [`State::drop_engine_unless_alive`]).
    #[cfg(test)]
    pub(crate) dead_engine: Option<Option<OpenRefusal>>,
    /// How many times the bus was looked at for the tracker, in tests.
    #[cfg(test)]
    pub(crate) presence_probes: u32,
    /// The display area each engine was started with, in tests, where none
    /// really starts: an engine would open the tracker.
    #[cfg(test)]
    pub(crate) engines_started: Vec<Option<DisplayArea>>,
}

impl State {
    pub(crate) fn new(prewarm: bool) -> Self {
        Self {
            engine: None,
            clients: Vec::new(),
            prewarm,
            facts: None,
            last_presence: None,
            clock: None,
            gaze_frames: 0,
            display_override: None,
            display_request: crate::requests::DisplaySize::from_env(),
            display_file: None,
            device_name: None,
            name_file: None,
            name_lock: Arc::default(),
            calibration: Calibration::default(),
            paused: false,
            pause_holder: None,
            pausing: false,
            pause_lock: Arc::default(),
            engine_losses: 0,
            device_inits: 0,
            tracker_absence_logged: false,
            engine_armed: false,
            engine_started_at: None,
            restarts: Backoff::default(),
            #[cfg(test)]
            fake_device: None,
            #[cfg(test)]
            pause_hint: None,
            #[cfg(test)]
            lose_engine_on_fetch: false,
            #[cfg(test)]
            fake_present: false,
            #[cfg(test)]
            fake_address: 1,
            #[cfg(test)]
            dead_engine: None,
            #[cfg(test)]
            presence_probes: 0,
            #[cfg(test)]
            engines_started: Vec::new(),
        }
    }

    /// Every stream some client subscribes to, but those only clients the
    /// pump hung up on subscribe to: nothing is written to them, so nothing
    /// is encoded or computed for them.
    fn wanted_mask(&self) -> u32 {
        self.clients
            .iter()
            .filter(|c| !c.hung_up)
            .fold(0, |m, c| m | c.streams)
    }

    /// True if some client consumes a stream or holds the device, or
    /// pre-warm is set. A client the pump hung up on counts until its
    /// reader has cleaned up after it: its requests may still need the
    /// engine (see [`client_reader`]).
    fn is_engine_wanted(&self) -> bool {
        self.prewarm
            || self
                .clients
                .iter()
                .any(|c| c.streams != 0 || c.holds_device)
    }

    /// Stop the engine once nobody needs it, unless pre-warm is configured,
    /// and end the watchdog's backoff then (see [`crate::restart`]): it
    /// restarts no engine that is not wanted, and a client that wants one
    /// later starts it itself. Also drops a dead engine. Starting another,
    /// for pre-warm too, is left to the watchdog, which looks for the
    /// tracker without the state lock (the pump runs this under it).
    fn reconcile(&mut self) {
        if !self.is_engine_wanted() {
            self.drop_engine();
            if self.restarts.reset() {
                info!("no engine is wanted any more; the restart backoff is over");
            }
        } else if self.has_engine() {
            self.drop_engine_unless_alive(false);
        }
        self.sync_wanted();
    }

    /// Start the engine if it is not running and the tracker is on the bus,
    /// as `tracker_on_bus` says (see [`look_for_tracker`]): without one, an
    /// engine would only fail its opens and stop, and the watchdog starts
    /// one once the tracker is plugged in while a client wants it. A dead
    /// engine is dropped either way, and the calibration session and the
    /// pause it took with it end before a new engine starts, so that its
    /// first init writes the display area the session started from; a
    /// session whose stop saves is left to that stop, and the init writes
    /// the area it saves (see [`crate::calibration::on_engine_lost`]).
    ///
    /// The watchdog's backoff (see [`crate::restart`]) does not hold this
    /// start back: a client's subscription or request gets an engine at
    /// once even while the watchdog waits. An engine the tracker refuses
    /// gives up within about a second (0.7 s without permission, 1.4 s on
    /// an interface another process holds), and a tracker fixed meanwhile
    /// (the udev rule installed, the other process gone) is taken at once
    /// rather than after up to a minute. An engine started so that ends
    /// before the tracker is ready adds a step to the backoff all the same.
    pub(crate) fn ensure_engine(&mut self, tracker_on_bus: bool) {
        if !self.drop_engine_unless_alive(tracker_on_bus) {
            return;
        }
        if tracker_on_bus {
            self.start_engine();
        } else {
            // Only a look at the bus says `false`; one that says `true`
            // starts an engine, whose end forgets where the tracker was.
            self.restarts.saw_bus(None);
            if !std::mem::replace(&mut self.tracker_absence_logged, true) {
                info!("no tracker on the bus; the engine starts once it is plugged in");
            }
        }
    }

    /// Start an engine now (see [`Self::start_engine_at`]).
    fn start_engine(&mut self) {
        self.start_engine_at(Instant::now());
    }

    /// Start an engine at `now`, from when the watchdog's backoff counts
    /// should it end before the tracker is ready. In tests none starts,
    /// whatever the bus says: the display area it would start with is
    /// recorded instead.
    fn start_engine_at(&mut self, now: Instant) {
        self.tracker_absence_logged = false;
        self.engine_armed = false;
        self.engine_started_at = Some(now);
        #[cfg(not(test))]
        {
            self.engine = Some(Engine::start_with(self.display_override));
            self.sync_wanted();
        }
        #[cfg(test)]
        {
            self.engines_started.push(self.display_override);
        }
    }

    /// Whether an engine runs to send gaze frames. In tests the stand-in
    /// device stands for one, unless the next fetch is to find it dead.
    pub(crate) fn is_engine_running(&self) -> bool {
        #[cfg(test)]
        if self.fake_device.is_some() {
            return !self.lose_engine_on_fetch;
        }
        self.engine.as_ref().is_some_and(Engine::is_alive)
    }

    /// Whether there is an engine, running or dead but not dropped yet. In
    /// tests, the stand-in for a dead one counts.
    fn has_engine(&self) -> bool {
        #[cfg(test)]
        if self.dead_engine.is_some() {
            return true;
        }
        self.engine.is_some()
    }

    /// Drop the engine unless it is running (see [`Self::drop_engine`]).
    /// Whether none runs now: the step [`Self::ensure_engine`] takes before
    /// it starts one, and the one [`Self::reconcile`] and the watchdog take
    /// for a dead engine, whose end they note (see
    /// [`Self::note_engine_end`]); `restart_now` if the caller starts
    /// another at once.
    fn drop_engine_unless_alive(&mut self, restart_now: bool) -> bool {
        if self.engine.as_ref().is_some_and(Engine::is_alive) {
            return false;
        }
        if let Some(refusal) = self.finish_engine() {
            self.note_engine_end(refusal, Instant::now(), restart_now);
        }
        self.drop_engine();
        true
    }

    /// Take the engine, which has ended, and say why it gave up if the
    /// tracker refused to be opened (see [`Engine::finish`]); `None` without
    /// one. In tests, the stand-in for a dead engine is taken.
    fn finish_engine(&mut self) -> Option<Option<OpenRefusal>> {
        #[cfg(test)]
        if let Some(end) = self.dead_engine.take() {
            return Some(end);
        }
        self.engine.take().map(Engine::finish)
    }

    /// The engine ended by itself, found at `now`, having given up on the
    /// tracker for `refusal` if it refused to be opened. One that never got
    /// the tracker ready backs the watchdog off a step (see
    /// [`crate::restart`]), logged once with the reason, and when the
    /// watchdog may start the next unless `restart_now` (the caller starts
    /// one at once); one that did had the backoff reset when it got it
    /// ready (see [`Self::note_engine_armed`]).
    fn note_engine_end(&mut self, refusal: Option<OpenRefusal>, now: Instant, restart_now: bool) {
        let started = self.engine_started_at.take().unwrap_or(now);
        if std::mem::take(&mut self.engine_armed) {
            return;
        }
        self.restarts.engine_failed(started);
        let in_a_row = self.restarts.in_a_row();
        let reason = refusal.as_ref().map(tracing::field::display);
        if restart_now {
            warn!(
                reason,
                in_a_row, "engine ended before the tracker was ready; a client starts another"
            );
        } else {
            let retry_in = self.restarts.wait_left(now).unwrap_or_default();
            warn!(
                reason,
                in_a_row,
                retry_in = %format_args!("{retry_in:.1?}"),
                "engine ended before the tracker was ready; backing off its restarts"
            );
        }
    }

    /// The engine got the tracker ready: its init went through, or a gaze
    /// frame came. The watchdog's backoff starts afresh.
    fn note_engine_armed(&mut self) {
        if !std::mem::replace(&mut self.engine_armed, true) && self.restarts.reset() {
            info!("the tracker is ready; the restart backoff is over");
        }
    }

    /// Drop the engine, with what lasts only as long as it runs: a started
    /// calibration session unless its stop saves (see
    /// [`crate::calibration::on_engine_lost`]), the pause (see
    /// [`crate::pause::on_engine_lost`]) and the clock pair.
    pub(crate) fn drop_engine(&mut self) {
        self.engine = None;
        #[cfg(test)]
        {
            self.dead_engine = None;
        }
        self.clock = None;
        crate::calibration::on_engine_lost(self);
        crate::pause::on_engine_lost(self);
    }

    /// Tell the engine which optional work anyone consumes: head-pose
    /// inference, and IR frames for image subscribers.
    fn sync_wanted(&self) {
        if let Some(engine) = self.engine.as_ref() {
            let wanted = self.wanted_mask();
            engine.set_head_wanted(wanted & STREAM_HEAD != 0);
            engine.set_image_wanted(wanted & STREAM_IMAGE != 0);
        }
    }

    /// The device's command queue, starting the engine if needed (see
    /// [`Self::ensure_engine`], which `tracker_on_bus` is for) and pinning
    /// it to `client`. `None`, never a dead engine's queue, when no engine
    /// runs: the tracker is off the bus. `client` holds the device all the
    /// same, so the watchdog starts an engine for it once the tracker is
    /// plugged in.
    pub(crate) fn device_for(
        &mut self,
        client: u64,
        tracker_on_bus: bool,
    ) -> Option<Arc<dyn DeviceCommands>> {
        if let Some(c) = self.clients.iter_mut().find(|c| c.id == client) {
            c.holds_device = true;
        }
        #[cfg(test)]
        if let Some(fake) = self.fake_device.clone() {
            if std::mem::take(&mut self.lose_engine_on_fetch) {
                let started = self.engines_started.len();
                self.ensure_engine(tracker_on_bus);
                // The stand-in stands for the engine started in place of
                // the dead one: without a tracker, none is.
                if self.engines_started.len() == started {
                    self.fake_device = None;
                    return None;
                }
            }
            return Some(fake);
        }
        self.ensure_engine(tracker_on_bus);
        self.commands()
    }

    /// The running engine's command queue, if there is one. Unlike
    /// [`Self::device_for`], it neither starts an engine nor pins one.
    pub(crate) fn commands(&self) -> Option<Arc<dyn DeviceCommands>> {
        #[cfg(test)]
        if let Some(fake) = &self.fake_device {
            return Some(Arc::clone(fake));
        }
        self.engine
            .as_ref()
            .filter(|e| e.is_alive())
            .map(|e| Arc::new(e.commands()) as Arc<dyn DeviceCommands>)
    }

    /// Tell the engine whether the device is paused (see
    /// [`Engine::set_paused`]).
    pub(crate) fn set_pause_hint(&mut self, paused: bool) {
        if let Some(engine) = self.engine.as_ref() {
            engine.set_paused(paused);
        }
        #[cfg(test)]
        {
            self.pause_hint = Some(paused);
        }
    }

    /// Queue `body` for one client, unless the pump hung up on it.
    pub(crate) fn send_to(&mut self, client: u64, body: Vec<u8>) {
        if let Some(c) = self
            .clients
            .iter_mut()
            .find(|c| c.id == client && !c.hung_up)
        {
            c.outbox.push_back(body);
        }
    }

    /// Queue `body` for every client subscribed to `stream`, but those the
    /// pump hung up on.
    pub(crate) fn broadcast(&mut self, stream: u32, body: &[u8]) {
        for c in self
            .clients
            .iter_mut()
            .filter(|c| c.streams & stream != 0 && !c.hung_up)
        {
            c.outbox.push_back(body.to_vec());
        }
    }

    /// Note the `(device_us, host_rx_us)` pair of a gaze frame: its device
    /// timestamp and the host time it was read at.
    pub(crate) fn note_clock(&mut self, device_us: i64, host_rx_us: i64) {
        self.clock = Some((device_us, host_rx_us));
        self.gaze_frames = self.gaze_frames.wrapping_add(1);
    }

    /// Fold a sample into the daemon's own state. While the device is
    /// paused, a frame that still arrives neither dates the clock pair nor
    /// replaces the presence a new subscriber is told.
    pub(crate) fn observe(&mut self, sample: &Sample) {
        match sample {
            Sample::DeviceReady(facts) => {
                self.note_engine_armed();
                let mut facts = (**facts).clone();
                keep_unreported(&mut facts, self.facts.as_deref());
                if let Some(area) = self.display_override {
                    facts.display_area = Some(area);
                }
                if let Some(id) = facts.calibration_id {
                    self.calibration.id = Some(id);
                }
                self.facts = Some(Arc::new(facts));
                // A pair from before the init may not hold after it.
                self.clock = None;
                self.device_inits = self.device_inits.wrapping_add(1);
                crate::calibration::on_device_ready(self);
                crate::requests::apply_display_request(self);
                crate::pause::on_device_ready(self);
            }
            Sample::Presence(p) if !self.paused => self.last_presence = Some(*p),
            Sample::Gaze(g) => {
                self.note_engine_armed();
                if !self.paused {
                    let device = i64::try_from(g.frame.device_ts_us).unwrap_or(i64::MAX);
                    self.note_clock(device, g.host_rx_us);
                }
            }
            Sample::Notification(DeviceNotification::DisplayAreaChanged(area)) => {
                if let Some(facts) = &self.facts {
                    let mut facts = (**facts).clone();
                    facts.display_area = Some(*area);
                    self.facts = Some(Arc::new(facts));
                }
            }
            Sample::Notification(DeviceNotification::CalibrationIdChanged(id)) => {
                self.calibration.id = Some(*id);
            }
            Sample::Notification(DeviceNotification::DevicePausedChanged(paused)) => {
                crate::pause::on_notification(*paused);
            }
            Sample::Notification(n @ DeviceNotification::FaultsChanged(text)) => {
                self.note_status_list("faults", text, n);
            }
            Sample::Notification(n @ DeviceNotification::WarningsChanged(text)) => {
                self.note_status_list("warnings", text, n);
            }
            _ => {}
        }
    }

    /// A fault or warning list the tracker announced (`n`, a 3200 or 3210
    /// carrying `text` as the new `list`): it replaces the one the last init
    /// reported, which the FAULT and WARNING states answer (see
    /// [`DeviceFacts::apply_notification`]), as the DLL's handler updates its
    /// cache (0x18016f0e9, 0x18016f108). A list that init left out stays left
    /// out, and before the first init there is nothing to update; subscribers
    /// are told either way (see [`crate::frames::notification_of`]). Unlike
    /// the DLL, which takes messages strictly in arrival order, a list read
    /// during an init arrives after its `DeviceReady` and replaces what that
    /// init's 1490 said, even one the tracker sent before answering the 1490.
    /// Logged, since neither was ever captured: at info, or at warn for
    /// anything but "ok".
    fn note_status_list(&mut self, list: &'static str, text: &str, n: &DeviceNotification) {
        let state_updated = self
            .facts
            .as_mut()
            .is_some_and(|facts| Arc::make_mut(facts).apply_notification(n));
        if text == "ok" {
            info!(list, text = ?text, state_updated, "tracker status list changed");
        } else {
            warn!(list, text = ?text, state_updated, "tracker status list changed");
        }
    }
}

/// Fill in, from the previous init's facts, what a new init did not report
/// because a response was lost: the stream catalogue, the hardware
/// configuration and the property strings (1330, which give the integration
/// type) are the firmware's, so the old ones still hold, and a lost 1200,
/// 2120 or 1330 must not blank them. The DLL, too, leaves its copy of the
/// integration type as it was when a 1330 fails (0x18016e328; whether that
/// copy outlives a reconnect is not traced). The status strings (1490) are
/// not kept: the DLL's reconnect empties its copy of the fault and warning
/// lists too.
fn keep_unreported(facts: &mut DeviceFacts, previous: Option<&DeviceFacts>) {
    let Some(previous) = previous else {
        return;
    };
    if facts.streams.is_empty() {
        facts.streams.clone_from(&previous.streams);
    }
    if facts.hardware.is_none() {
        facts.hardware.clone_from(&previous.hardware);
    }
    if facts.properties.is_empty() {
        facts.properties.clone_from(&previous.properties);
    }
}

/// Whether the tracker is on the bus (see [`find_tracker`]), for a caller
/// that does not hold the state lock.
pub(crate) fn is_tracker_present(state: &Mutex<State>) -> bool {
    find_tracker(state).is_some()
}

/// Where the tracker is on the bus, if on it (see
/// [`tobii_usb::device::find_device`]), for a caller that does not hold the
/// state lock: libusb scans the bus (a few milliseconds) without it.
#[cfg(not(test))]
fn find_tracker(_state: &Mutex<State>) -> Option<BusAddress> {
    tobii_usb::device::find_device()
}

/// Where the tracker is on the bus: in tests, what the state's stand-in
/// says, counted.
#[cfg(test)]
fn find_tracker(state: &Mutex<State>) -> Option<BusAddress> {
    let mut st = lock_state(state);
    st.presence_probes += 1;
    st.fake_present.then_some(BusAddress {
        bus: 1,
        address: st.fake_address,
    })
}

/// Whether the tracker is on the bus, for [`State::ensure_engine`] and
/// [`State::device_for`], looked for before the caller takes the state
/// lock (see [`is_tracker_present`]). `true` without a look while an engine
/// runs, since none is started then. The answer may be stale once the lock
/// is taken, which does no harm: an engine started for a tracker just
/// unplugged stops as any engine does, and the watchdog starts one for a
/// tracker just plugged in.
pub(crate) fn look_for_tracker(state: &Mutex<State>) -> bool {
    // Its own statement, so that the guard goes before the bus is scanned.
    let running = lock_state(state).is_engine_running();
    running || is_tracker_present(state)
}

/// Lock the shared state, recovering from poisoning. Every critical section
/// here leaves `State` consistent at each statement (the engine is an
/// `Option`, clients a plain list), so a panic while holding the lock cannot
/// leave it half-updated; continuing beats taking the whole daemon down.
pub(crate) fn lock_state(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `TOBII_PREWARM` (any of `1`, `head`, `gaze`; the historical mode names are
/// accepted since one engine now serves everything): start the engine at
/// daemon start and keep it warm.
fn is_prewarm_enabled() -> bool {
    matches!(
        std::env::var("TOBII_PREWARM").ok().as_deref(),
        Some("1" | "head" | "head-camera" | "camera" | "gaze" | "yes" | "true")
    )
}

/// A file's identity on disk: its device and inode numbers. While a daemon
/// runs, its listening socket holds the file it bound, so no file made at the
/// same path since can have the same numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileId {
    dev: u64,
    ino: u64,
}

impl FileId {
    /// The file `path` names now (a symlink itself, not what it points to).
    fn of(path: &Path) -> std::io::Result<Self> {
        let meta = std::fs::symlink_metadata(path)?;
        Ok(Self {
            dev: meta.dev(),
            ino: meta.ino(),
        })
    }
}

/// Who made the file of the socket the daemon listens on, which decides
/// whether the daemon removes it at shutdown.
#[derive(Debug)]
enum SocketFile {
    /// systemd, which passed the socket in (socket activation). The file is
    /// the socket unit's and outlives the daemon: `systemctl --user restart
    /// tobiid` hands the same socket to the next daemon, which clients reach
    /// only through that file.
    Systemd,
    /// This daemon, which bound `path` and made the file `bound` there.
    Bound { path: PathBuf, bound: FileId },
}

impl SocketFile {
    /// Whether the daemon removes the socket file at shutdown, given the file
    /// its path names by then (`None`: none). Only a file it bound itself, and
    /// only while the path still names that file: a daemon started since has
    /// replaced it (`bind_listener` replaces a file it finds there), and
    /// removing that daemon's file would leave it where no client can connect.
    fn is_ours_to_remove(&self, now: Option<FileId>) -> bool {
        match self {
            Self::Systemd => false,
            Self::Bound { bound, .. } => now == Some(*bound),
        }
    }

    /// Remove the socket file at shutdown if it is ours to remove. A file
    /// that is not, or one that cannot be looked at, is left. (A daemon that
    /// takes the path between the check and the removal still loses its
    /// file; only a lock would close that window.)
    fn remove_at_shutdown(&self) {
        let Self::Bound { path, .. } = self else {
            return;
        };
        let now = match FileId::of(path) {
            Ok(now) => Some(now),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "could not check the socket file; leaving it");
                return;
            }
        };
        if self.is_ours_to_remove(now) {
            if let Err(e) = std::fs::remove_file(path) {
                warn!(path = %path.display(), error = %e, "failed to remove the socket file");
            }
        } else if now.is_some() {
            // Someone re-bound the path while this daemon ran: a daemon a
            // client spawned, or a socket unit started since.
            warn!(
                path = %path.display(),
                "the socket path names a file this daemon did not bind; leaving it"
            );
        }
    }
}

/// Use the socket passed by systemd socket activation (fd 3) if present,
/// otherwise bind our own; say which, for the shutdown. The
/// `sd_listen_fds(3)` protocol: `LISTEN_PID` must equal our pid and
/// `LISTEN_FDS` >= 1, with the first socket at fd 3.
fn obtain_listener() -> Result<(UnixListener, SocketFile)> {
    let activated = std::env::var("LISTEN_PID")
        .ok()
        .and_then(|p| p.parse::<u32>().ok())
        .is_some_and(|p| p == std::process::id())
        && std::env::var("LISTEN_FDS")
            .ok()
            .and_then(|n| n.parse::<i32>().ok())
            .is_some_and(|n| n >= 1);

    if activated {
        // SAFETY: `LISTEN_PID`/`LISTEN_FDS` say systemd handed us a listening
        // Unix socket at fd 3 (SD_LISTEN_FDS_START); nothing else in this
        // process owns or has touched that fd, so taking ownership once is
        // sound.
        let listener = unsafe { UnixListener::from_raw_fd(3) };
        info!(fd = 3, "using systemd socket activation");
        return Ok((listener, SocketFile::Systemd));
    }

    bind_listener(tobii_ipc::socket_path())
}

/// Bind a listening socket at `path` and note the file it makes there.
///
/// The socket is bound at `path` plus `.<pid>`, a name no other daemon uses,
/// its file noted there and then renamed onto `path`. Noted at `path` itself,
/// the file of a daemon that re-bound `path` in between would pass for ours.
/// The rename replaces a stale file from a previous run (or a running
/// daemon's, whose clients stay with it) in one step, so a client never finds
/// the path missing, nor a file there that does not listen yet.
fn bind_listener(path: PathBuf) -> Result<(UnixListener, SocketFile)> {
    let mut own = path.clone().into_os_string();
    own.push(format!(".{}", std::process::id()));
    let own = PathBuf::from(own);
    // Only a daemon that had this pid and died binding could have left one.
    let _ = std::fs::remove_file(&own);
    let listener =
        UnixListener::bind(&own).with_context(|| format!("failed to bind {}", own.display()))?;
    let placed = FileId::of(&own)
        .with_context(|| format!("failed to read back {} after binding it", own.display()))
        .and_then(|bound| {
            std::fs::rename(&own, &path).with_context(|| {
                format!("failed to move {} onto {}", own.display(), path.display())
            })?;
            Ok(bound)
        });
    if placed.is_err() {
        // Nothing would ever remove it.
        let _ = std::fs::remove_file(&own);
    }
    let bound = placed?;
    info!(path = %path.display(), "listening");
    Ok((listener, SocketFile::Bound { path, bound }))
}

/// Run the daemon: bind (or adopt) the listening socket, install the signal
/// handlers, start the pump/watchdog threads and serve clients until a
/// shutdown signal exits the process.
///
/// # Errors
///
/// Returns an error if the Unix socket cannot be bound, or its file cannot be
/// read back or moved onto the socket path right after binding.
pub fn run() -> Result<()> {
    let (listener, socket_file) = obtain_listener()?;

    let prewarm = is_prewarm_enabled();
    let state = Arc::new(Mutex::new(State::new(prewarm)));
    {
        let mut st = lock_state(&state);
        st.display_file = crate::display::default_path();
        crate::requests::restore_saved_display_area(&mut st);
        st.name_file = crate::name::default_path();
        crate::name::restore(&mut st);
    }

    // Pre-warm: bring the device up now (pays the cold-start lottery once) and
    // keep it streaming so later client connects are instant.
    if prewarm {
        info!("pre-warming device");
        let on_bus = is_tracker_present(&state);
        lock_state(&state).ensure_engine(on_bus);
    }

    // Watchdog: if the device thread died (a cold start that exhausted its
    // internal retries, an unplug, etc.), or none was started while the
    // tracker was unplugged, but it's still wanted, restart it, backing off
    // while engines end before the tracker is ready (see `restart`).
    {
        let state = Arc::clone(&state);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(3));
                watch_engine(&state);
            }
        });
    }

    // SIGUSR1 -> recenter; SIGTERM/SIGINT -> graceful shutdown with device teardown.
    // SAFETY: both handlers are `extern "C" fn(c_int)` matching `sighandler_t`
    // and only perform an atomic store, so they are async-signal-safe. The
    // previous handlers (the defaults) need no restoring.
    unsafe {
        libc::signal(
            libc::SIGUSR1,
            on_sigusr1 as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
        let h = on_shutdown as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGTERM, h);
        libc::signal(libc::SIGINT, h);
    }
    {
        let state = Arc::clone(&state);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_millis(150));
                if SHUTDOWN_SIGNAL.load(Ordering::Relaxed) {
                    // Drop the engine so its Drop runs the device teardown (stop the
                    // 0x83 stream), then exit.
                    let mut st = lock_state(&state);
                    st.prewarm = false; // don't let reconcile/watchdog respawn it
                    st.engine = None; // blocks until the engine thread + teardown finish
                    drop(st);
                    // The socket file goes only if this daemon bound it, never
                    // when it is systemd's (see `SocketFile`). It stays the
                    // only file action here: `SocketFile` is tested, this
                    // thread is not.
                    socket_file.remove_at_shutdown();
                    info!("shutdown signal: device teardown done, exiting");
                    std::process::exit(0);
                }
                if RECENTER_SIGNAL.swap(false, Ordering::Relaxed)
                    && let Some(engine) = lock_state(&state).engine.as_ref()
                {
                    engine.request_recenter();
                }
            }
        });
    }

    // Accept loop on its own thread; the pump stays on the main thread so a
    // panic in fan-out — the one path nothing else supervises — still ends
    // the process and lets systemd's Restart=on-failure bring the daemon
    // back (engine-thread panics stay recoverable through the watchdog).
    {
        let state = Arc::clone(&state);
        thread::spawn(move || accept_loop(&listener, &state));
    }
    pump(&state);
    Ok(())
}

/// One watchdog pass: drop a dead engine, and start one if it is wanted and
/// the tracker is on the bus, whether an engine died or none was started
/// because the tracker was unplugged. The bus is scanned only when an
/// engine is wanted, so that nothing spins (reloading the model) on an
/// unplugged tracker, and once a pass, without the state lock.
fn watch_engine(state: &Mutex<State>) {
    watch_engine_at(state, Instant::now());
}

/// [`watch_engine`] at `now`: an engine is started only once the backoff
/// after engines that ended before the tracker was ready has passed (see
/// [`crate::restart`]), or at once when the tracker was plugged back in
/// (another address than the last pass found, or back after a pass found
/// it gone), which starts the backoff afresh.
fn watch_engine_at(state: &Mutex<State>, now: Instant) {
    let wanted = {
        let mut st = lock_state(state);
        // Dropped even while unplugged, so that clients hear of it.
        if st.has_engine() {
            st.drop_engine_unless_alive(false);
        }
        !st.has_engine() && st.is_engine_wanted()
    };
    if !wanted {
        return;
    }
    let found = find_tracker(state);
    let mut st = lock_state(state);
    if st.restarts.saw_bus(found) {
        info!("the tracker was plugged back in; the restart backoff is over");
    }
    // A request may have started one meanwhile.
    if found.is_none() || st.has_engine() || !st.is_engine_wanted() {
        return;
    }
    if let Some(wait) = st.restarts.wait_left(now) {
        // The step was logged once, at the engine's end (`note_engine_end`).
        debug!(?wait, "engine wanted; backing off before restarting it");
        return;
    }
    match st.restarts.in_a_row() {
        0 => warn!("engine not running but wanted; restarting"),
        // Warned of at the engine's end already.
        in_a_row => info!(in_a_row, "restarting the engine after backing off"),
    }
    st.start_engine_at(now);
}

/// How long the pump may sit in a write to a client, with `state` locked,
/// before it hangs up on the client (see [`Client::hang_up`]).
const WRITE_TIMEOUT: Duration = Duration::from_millis(250);

/// Accept client connections forever, registering each with `state`.
fn accept_loop(listener: &UnixListener, state: &Arc<Mutex<State>>) {
    // Only the accept loop hands out ids, so a plain counter suffices.
    let mut next_id: u64 = 1;
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let Ok(out) = stream.try_clone() else {
            continue;
        };
        // Bound how long pump() can sit in write_frame() with `state` locked:
        // a client that stops reading is hung up on instead of wedging
        // fan-out (and the SIGTERM teardown, which needs the same lock).
        let _ = out.set_write_timeout(Some(WRITE_TIMEOUT));
        let id = next_id;
        next_id = next_id.wrapping_add(1);
        lock_state(state).clients.push(Client::new(id, out));
        let state = Arc::clone(state);
        thread::spawn(move || client_reader(&state, id, stream));
    }
}

/// How many of a client's requests may wait behind the one its worker runs.
/// A client with that many waiting is read no further until the worker
/// takes the next: its frames wait in the socket, SUBSCRIBEs included.
/// libtobii has one request in flight per device and waits for its answer
/// longer than the daemon may take over it (see [`tobii_ipc::deadline`]), so
/// it leaves none waiting; only a client that gives up sooner and asks again
/// fills the queue.
const REQUEST_QUEUE: usize = 32;

/// A client's request worker: a thread that runs the client's requests one
/// at a time, in the order its reader queued them.
struct RequestWorker {
    /// REQUEST frame bodies, as read.
    queue: SyncSender<Vec<u8>>,
    thread: JoinHandle<()>,
}

impl RequestWorker {
    /// Start the worker for client `id`, connected over `stream`.
    fn spawn(state: &Arc<Mutex<State>>, id: u64, stream: &UnixStream) -> io::Result<Self> {
        let hang_up = HangUpOnPanic(stream.try_clone()?);
        let (queue, requests) = mpsc::sync_channel(REQUEST_QUEUE);
        let state = Arc::clone(state);
        let thread = thread::Builder::new()
            // At most 15 bytes are kept (the kernel's limit).
            .name(format!("tobiid-rq-{id}"))
            .spawn(move || {
                let _hang_up = hang_up;
                run_requests(&state, id, &requests);
            })?;
        Ok(Self { queue, thread })
    }

    /// Take no more requests, and wait for those queued to run.
    fn finish(self, id: u64) {
        let Self { queue, thread } = self;
        drop(queue);
        if thread.join().is_err() {
            warn!(client = id, "a request worker panicked");
        }
    }
}

/// Hangs up on a client when its request worker panics: nothing would
/// answer its requests any more. Its reader then stops at once, and the
/// client learns the connection is lost before it is cleaned up after, not
/// at its next request.
struct HangUpOnPanic(UnixStream);

impl Drop for HangUpOnPanic {
    fn drop(&mut self) {
        if thread::panicking() {
            let _ = self.0.shutdown(Shutdown::Both);
        }
    }
}

/// Run client `id`'s requests as they come, one at a time and in order,
/// until its reader drops the queue; those queued by then still run. Each
/// reply goes to the client's outbox.
fn run_requests(state: &Mutex<State>, id: u64, requests: &Receiver<Vec<u8>>) {
    for body in requests {
        if let Some(request) = decode_request(&body) {
            // Runs without the state lock held while the device works.
            let reply = crate::requests::handle(state, id, &request);
            lock_state(state).send_to(id, encode_reply(request.id, reply.status, &reply.payload));
        }
    }
}

/// One per connection: handle SUBSCRIBE and RECENTER frames, hand REQUEST
/// frames to the client's request worker, and detect disconnect. Replies go
/// through the client's outbox. The requests a client sent before it hung
/// up still run (a calibration stop that keeps the result, sent without
/// waiting for the answer, still keeps it); only then is the client
/// cleaned up after, here and once. The pump, which may find the hang-up
/// first (at its next write to the client, a reply or a sample), only
/// hangs up on the client in turn (see [`Client::hang_up`]): it stays
/// listed with the streams and the device it holds, so that no engine
/// stops under its requests. A subscription change takes effect at once,
/// ahead of the client's requests still to run: an unsubscribe before one
/// of them pins the device (see [`State::device_for`]) may stop an engine
/// that it then starts again. A subscription change is skipped if the
/// client has hung up by the time it would take effect (see
/// [`handle_subscribe`]): nobody reads its ack, and an engine it started
/// would only stop again at the cleanup. That is a libtobii reconnect that
/// gave up on its ack: while systemd held the socket for a daemon restart
/// (the daemon that comes up finds the connection in the backlog, the
/// subscription in it and the hang-up behind), or while its reader waited
/// for the state lock.
fn client_reader(state: &Arc<Mutex<State>>, id: u64, mut stream: UnixStream) {
    if let Some(worker) = read_frames(state, id, &mut stream) {
        worker.finish(id);
    }
    // No request of the client's runs any more: its worker has returned or
    // panicked (see `pause::release`). Resumed before the engine may stop
    // below.
    crate::pause::release(state, id);
    crate::calibration::on_client_gone(state, id);
    let mut st = lock_state(state);
    st.clients.retain(|c| c.id != id);
    st.reconcile();
}

/// Read client `id`'s frames until it hangs up (EOF or a read error), or
/// its request worker dies, which hangs up on it. The worker starts at the
/// client's first request (a client that only streams needs none) and is
/// handed back to be waited for. A subscription change is skipped once the
/// client has hung up, and the frames after it are still read: requests
/// the client sent before it hung up still run.
fn read_frames(
    state: &Arc<Mutex<State>>,
    id: u64,
    stream: &mut UnixStream,
) -> Option<RequestWorker> {
    let mut worker: Option<RequestWorker> = None;
    // Whether a subscription change was skipped yet: said once at info.
    let mut skipped = false;
    while let Ok(Some(body)) = read_frame(stream) {
        match body.first().copied() {
            Some(tobii_ipc::TAG_SUBSCRIBE) => {
                if let Some(streams) = decode_subscribe(&body)
                    && !handle_subscribe(state, id, streams, || has_peer_hung_up(id, stream))
                {
                    if std::mem::replace(&mut skipped, true) {
                        debug!(
                            client = id,
                            streams = %format_args!("{streams:#x}"),
                            "skipping another subscription from a client that has hung up"
                        );
                    } else {
                        info!(
                            client = id,
                            streams = %format_args!("{streams:#x}"),
                            "skipping a subscription from a client that has hung up"
                        );
                    }
                }
            }
            Some(tobii_ipc::TAG_RECENTER) => {
                if let Some(engine) = lock_state(state).engine.as_ref() {
                    engine.request_recenter();
                }
            }
            Some(tobii_ipc::TAG_REQUEST) => {
                if worker.is_none() {
                    match RequestWorker::spawn(state, id, stream) {
                        Ok(started) => worker = Some(started),
                        Err(e) => {
                            warn!(client = id, error = %e, "could not start a request worker; closing the connection");
                            break;
                        }
                    }
                }
                // Waits while the queue is full (see `REQUEST_QUEUE`).
                if !worker.as_ref().is_some_and(|w| w.queue.send(body).is_ok()) {
                    // It panicked, and has hung up on the client; frames
                    // the client sent before that may still be read.
                    warn!(
                        client = id,
                        "the request worker stopped; closing the connection"
                    );
                    break;
                }
            }
            _ => {}
        }
    }
    worker
}

/// Whether client `id`, on `stream`, has hung up: closed its end, or shut
/// down its sending, or the pump hung up on it (see [`Client::hang_up`]).
/// What the client sent before may still wait to be read. Asked of the
/// kernel with a `poll` for `POLLRDHUP`, which a Unix socket reports as
/// soon as its peer is gone, however much of what it sent is still unread;
/// a peek would see the hang-up only once all of that was read. The poll
/// does not block, so it may be taken under the state lock. `false` should
/// the socket not be polled: the client is served as if it were still
/// there.
fn has_peer_hung_up(id: u64, stream: &UnixStream) -> bool {
    let mut fd = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLRDHUP,
        revents: 0,
    };
    loop {
        // SAFETY: `fd` is a live local, valid for the kernel to read and
        // write for the one entry the count gives, and its descriptor is
        // `stream`'s, open while `stream` is borrowed. With a zero timeout
        // the call does not block.
        let ready = unsafe { libc::poll(&raw mut fd, 1, 0) };
        if ready >= 0 {
            return ready > 0
                && fd.revents & (libc::POLLRDHUP | libc::POLLHUP | libc::POLLERR) != 0;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            debug!(client = id, error = %e, "could not poll a client's socket for its hang-up");
            return false;
        }
    }
}

/// Register the client's streams, starting the engine if it isn't running
/// (and the tracker is on the bus; otherwise the watchdog starts it once the
/// tracker is plugged in), and acknowledge. Always succeeds (the one engine
/// serves every stream); `streams == 0` unsubscribes. `true` once done.
///
/// Skipped, with nothing changed and nothing acked (`false`), if the client
/// has hung up (`gone`, see [`has_peer_hung_up`]). That is asked before the
/// bus is scanned and again once the state lock is held, just before
/// anything changes: the scan and the wait for the lock (held while another
/// client's engine is dropped and its thread joined) are where a libtobii
/// reconnect's 500 ms go, and one that gave up there would get an engine
/// started for nobody, dropped again at its cleanup under the lock.
fn handle_subscribe(state: &Mutex<State>, id: u64, streams: u32, gone: impl Fn() -> bool) -> bool {
    if gone() {
        return false;
    }
    let on_bus = streams != 0 && look_for_tracker(state);
    let mut st = lock_state(state);
    if gone() {
        return false;
    }
    let before = st
        .clients
        .iter()
        .find(|c| c.id == id)
        .map_or(0, |c| c.streams);
    if let Some(c) = st.clients.iter_mut().find(|c| c.id == id) {
        c.streams = streams;
    }
    if streams == 0 {
        st.reconcile();
    } else {
        st.ensure_engine(on_bus);
        st.sync_wanted();
    }
    st.send_to(id, encode_subscribed(true));
    replay_presence(&mut st, id, before, streams);
    true
}

/// A new presence subscriber is told the current presence straight away,
/// since the device reports it only on change; not while the device is
/// paused, when what it last reported may no longer hold.
fn replay_presence(st: &mut State, id: u64, before: u32, streams: u32) {
    if streams & !before & STREAM_PRESENCE != 0
        && !st.paused
        && let Some(p) = st.last_presence
    {
        st.send_to(id, presence_frame(&p));
    }
}

/// Fan-out loop: every 8 ms, a pass of [`pump_once`]. The buffers live
/// across ticks so the per-frame path does not reallocate.
fn pump(state: &Mutex<State>) {
    let mut samples: Vec<Sample> = Vec::new();
    let mut frames: Vec<(u32, Vec<u8>)> = Vec::new();
    loop {
        pump_once(&mut lock_state(state), &mut samples, &mut frames);
        thread::sleep(Duration::from_millis(8));
    }
}

/// One pass of the pump, under the state lock: drain the engine into
/// `samples`, fold them into the state, encode each once into `frames`, and
/// write them out (see [`write_clients`]).
fn pump_once(st: &mut State, samples: &mut Vec<Sample>, frames: &mut Vec<(u32, Vec<u8>)>) {
    samples.clear();
    frames.clear();
    if let Some(engine) = st.engine.as_mut() {
        engine.drain_into(samples);
    }
    let wanted = st.wanted_mask();
    for s in samples.iter() {
        st.observe(s);
    }
    for s in samples.iter() {
        push_sample_frames(s, wanted, frames);
    }
    write_clients(st, frames);
}

/// Write each client its outbox, then those of `frames` (bodies tagged
/// with the stream they carry) it subscribes to. A client a write fails for
/// (it hung up, or stopped reading for longer than [`WRITE_TIMEOUT`]) is
/// hung up on and written nothing more (see [`Client::hang_up`]), and the
/// engine is told the work only it wanted is no longer (see
/// [`State::wanted_mask`]). It stays listed: its reader cleans up after it
/// once the requests it sent before have run, and only then may the engine
/// stop (see [`client_reader`]).
fn write_clients(st: &mut State, frames: &[(u32, Vec<u8>)]) {
    let mut hung_up = false;
    for client in st.clients.iter_mut().filter(|c| !c.hung_up) {
        if !client.write_out(frames) {
            client.hang_up();
            hung_up = true;
        }
    }
    if hung_up {
        st.sync_wanted();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::calibration::tests::device_blob;

    /// A state with one connected client (its socket's peer is dropped, so
    /// a pump pass finds the connection dead).
    pub(crate) fn state_with_client(id: u64) -> State {
        let (out, _peer) = UnixStream::pair().expect("socket pair");
        let mut st = State::new(false);
        st.clients.push(Client::new(id, out));
        st
    }

    pub(crate) fn outbox(st: &State, id: u64) -> Vec<Vec<u8>> {
        st.clients
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.outbox.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Client `id` changes its subscription to `streams` and stays
    /// connected: the change is acted on.
    fn subscribe(state: &Mutex<State>, id: u64, streams: u32) {
        assert!(handle_subscribe(state, id, streams, || false), "acted on");
    }

    #[test]
    fn a_stopped_engine_takes_its_clock_pair_along() {
        let mut st = state_with_client(1);
        st.clock = Some((5_000_000, 12_000_000_000));
        // Nobody wants the engine (and prewarm is off): no engine starts.
        st.reconcile();
        assert_eq!(st.clock, None);
    }

    /// A display area as a client sends it: whole millimetres, so that it
    /// comes back equal from the wire (f32).
    fn area(width_mm: f64) -> DisplayArea {
        let half = width_mm / 2.0;
        DisplayArea {
            top_left_mm: [-half, 300.0, 0.0],
            top_right_mm: [half, 300.0, 0.0],
            bottom_left_mm: [-half, 0.0, 0.0],
        }
    }

    #[test]
    fn replacing_the_engine_ends_the_session_before_the_new_one_starts() {
        let old = area(600.0);
        let mut st = state_with_client(1);
        st.fake_device = Some(Arc::new(crate::requests::tests::Answering(1)));
        st.calibration.location = tobii_calib::store::Location::Embedded;
        st.display_override = Some(old);
        let state = Mutex::new(st);
        let ask = |kind, payload: &[u8]| {
            crate::requests::handle(
                &state,
                1,
                &tobii_ipc::request::Request {
                    id: 1,
                    kind,
                    payload,
                },
            )
        };
        let start = ask(tobii_ipc::request::kind::CALIBRATION_START, &[2]);
        assert_eq!(start.status, tobii_ipc::request::status::OK);
        let set = ask(
            tobii_ipc::request::kind::DISPLAY_AREA_SET,
            &tobii_ipc::request::encode_display_area(&area(520.0)),
        );
        assert_eq!(set.status, tobii_ipc::request::status::OK);
        let mut st = lock_state(&state);
        st.note_clock(5_000_000, 1);

        // No engine runs: one starts, with the display area configured by
        // then.
        st.ensure_engine(true);

        assert_eq!(
            st.engines_started,
            [Some(old)],
            "what the new engine's first init writes"
        );
        assert_eq!(st.display_override, Some(old));
        assert!(!st.calibration.is_active());
        assert_eq!(
            st.calibration.display_access(1),
            Err(tobii_ipc::request::status::CALIBRATION_NOT_STARTED),
            "its owner is told on its next request"
        );
        assert_eq!(st.clock, None);
    }

    #[test]
    fn no_engine_is_started_for_an_absent_tracker() {
        let mut st = state_with_client(1);

        st.ensure_engine(false);
        assert!(
            st.device_for(1, false).is_none(),
            "requests that need the tracker fail with CONNECTION_FAILED"
        );

        assert!(st.engines_started.is_empty());
        assert!(
            st.clients[0].holds_device,
            "the watchdog starts it once it is plugged in"
        );
    }

    #[test]
    fn an_engine_is_started_once_the_tracker_is_on_the_bus() {
        let mut st = state_with_client(1);
        st.ensure_engine(false);
        assert!(st.tracker_absence_logged);

        let _ = st.device_for(1, true);

        assert_eq!(st.engines_started.len(), 1);
        assert!(st.clients[0].holds_device);
        assert!(
            !st.tracker_absence_logged,
            "the next absence is logged again"
        );
    }

    #[test]
    fn the_bus_is_looked_at_only_when_no_engine_runs() {
        let mut st = state_with_client(1);
        st.fake_device = Some(Arc::new(crate::requests::tests::Answering(1)));
        let state = Mutex::new(st);

        assert!(look_for_tracker(&state), "nothing would be started");
        assert_eq!(lock_state(&state).presence_probes, 0);

        lock_state(&state).fake_device = None;
        assert!(!look_for_tracker(&state));
        assert_eq!(lock_state(&state).presence_probes, 1);
    }

    #[test]
    fn a_subscription_starts_an_engine_only_for_a_tracker_on_the_bus() {
        let state = Mutex::new(state_with_client(1));

        subscribe(&state, 1, STREAM_PRESENCE);
        {
            let st = lock_state(&state);
            assert_eq!(outbox(&st, 1), [encode_subscribed(true)]);
            assert!(st.engines_started.is_empty());
            assert!(
                st.is_engine_wanted(),
                "the watchdog starts it once the tracker is plugged in"
            );
        }

        lock_state(&state).fake_present = true;
        subscribe(&state, 1, STREAM_PRESENCE);

        let st = lock_state(&state);
        assert_eq!(st.engines_started.len(), 1);
        assert_eq!(st.presence_probes, 2, "one look for each subscription");
    }

    #[test]
    fn the_watchdog_starts_an_engine_once_the_tracker_is_plugged_in() {
        let state = Mutex::new(state_with_client(1));
        lock_state(&state).fake_present = true;
        watch_engine(&state);
        assert_eq!(
            lock_state(&state).presence_probes,
            0,
            "nobody wants one: the bus is not even scanned"
        );
        {
            let mut st = lock_state(&state);
            st.fake_present = false;
            // A request found the tracker unplugged; its client stays.
            assert!(st.device_for(1, false).is_none());
        }
        watch_engine(&state);
        assert!(
            lock_state(&state).engines_started.is_empty(),
            "not while it is unplugged"
        );

        let probes = {
            let mut st = lock_state(&state);
            st.fake_present = true;
            st.presence_probes
        };
        watch_engine(&state);

        let st = lock_state(&state);
        assert_eq!(st.engines_started.len(), 1);
        assert_eq!(st.presence_probes, probes + 1, "one scan a pass");
    }

    #[test]
    fn a_pre_warmed_engine_is_left_to_the_watchdog_to_start() {
        let state = Mutex::new(State::new(true));
        lock_state(&state).fake_present = true;

        lock_state(&state).reconcile();
        {
            let st = lock_state(&state);
            assert!(st.engines_started.is_empty());
            assert_eq!(st.presence_probes, 0, "no scan under the state lock");
        }
        watch_engine(&state);

        assert_eq!(lock_state(&state).engines_started.len(), 1);
    }

    const SECOND: Duration = Duration::from_secs(1);

    /// A pre-warmed daemon (it wants an engine with no client) whose
    /// tracker is on the bus.
    fn pre_warmed_with_tracker() -> Mutex<State> {
        let mut st = State::new(true);
        st.fake_present = true;
        Mutex::new(st)
    }

    fn engines_started(state: &Mutex<State>) -> usize {
        lock_state(state).engines_started.len()
    }

    fn in_a_row(state: &Mutex<State>) -> u32 {
        lock_state(state).restarts.in_a_row()
    }

    /// The engine last started ends by itself, giving up on the tracker for
    /// `refusal` if it refused to be opened: the next look finds it dead.
    fn end_engine(state: &Mutex<State>, refusal: Option<OpenRefusal>) {
        lock_state(state).dead_engine = Some(refusal);
    }

    #[test]
    fn engines_that_end_before_the_tracker_is_ready_space_the_watchdogs_starts_out() {
        let state = pre_warmed_with_tracker();
        let t0 = Instant::now();
        watch_engine_at(&state, t0);
        assert_eq!(engines_started(&state), 1);

        // It may not open the tracker: the next pass, 3 s after it started,
        // drops it and starts another.
        end_engine(&state, Some(OpenRefusal::NoPermission));
        watch_engine_at(&state, t0 + 3 * SECOND);
        assert_eq!(engines_started(&state), 2);
        assert_eq!(in_a_row(&state), 1);

        // That one ends the same way: the next starts 6 s after it did.
        end_engine(&state, Some(OpenRefusal::NoPermission));
        watch_engine_at(&state, t0 + 6 * SECOND);
        assert_eq!(engines_started(&state), 2, "dropped, not replaced");
        watch_engine_at(&state, t0 + 8 * SECOND);
        assert_eq!(engines_started(&state), 2);
        watch_engine_at(&state, t0 + 9 * SECOND);
        assert_eq!(engines_started(&state), 3);

        // One that stopped without a refusal counts the same.
        end_engine(&state, None);
        watch_engine_at(&state, t0 + 20 * SECOND);
        assert_eq!(engines_started(&state), 3);
        watch_engine_at(&state, t0 + 21 * SECOND);
        assert_eq!(engines_started(&state), 4);
        assert_eq!(in_a_row(&state), 3);
    }

    #[test]
    fn an_engine_that_got_the_tracker_ready_leaves_no_backoff() {
        let state = pre_warmed_with_tracker();
        let t0 = Instant::now();
        watch_engine_at(&state, t0);
        end_engine(&state, Some(OpenRefusal::InUse));
        watch_engine_at(&state, t0 + 3 * SECOND);
        assert_eq!(in_a_row(&state), 1);

        // The next engine's init goes through, then it ends.
        lock_state(&state).observe(&Sample::DeviceReady(Arc::new(DeviceFacts::default())));
        assert_eq!(in_a_row(&state), 0, "reset as it got ready");
        end_engine(&state, None);
        watch_engine_at(&state, t0 + 4 * SECOND);

        assert_eq!(engines_started(&state), 3, "restarted at once");
        assert_eq!(in_a_row(&state), 0);
    }

    #[test]
    fn a_gaze_frame_starts_the_backoff_afresh() {
        let state = pre_warmed_with_tracker();
        let t0 = Instant::now();
        watch_engine_at(&state, t0);
        end_engine(&state, None);
        watch_engine_at(&state, t0 + 3 * SECOND);
        {
            let mut st = lock_state(&state);
            st.paused = true;
            let frame = tobii_proto::gaze83::GazeFrame::default();
            st.observe(&Sample::Gaze(Box::new(tobii_usb::engine::GazeSample::new(
                frame, 100, 90,
            ))));
            assert_eq!(st.restarts.in_a_row(), 0, "even while paused");
        }
        end_engine(&state, None);

        watch_engine_at(&state, t0 + 4 * SECOND);

        assert_eq!(engines_started(&state), 3);
    }

    #[test]
    fn an_armed_engine_dropped_as_unwanted_does_not_hide_the_next_ones_failure() {
        let state = Mutex::new(state_with_client(1));
        lock_state(&state).fake_present = true;
        subscribe(&state, 1, STREAM_PRESENCE);
        lock_state(&state).observe(&Sample::DeviceReady(Arc::new(DeviceFacts::default())));
        // The client unsubscribes: the armed engine stops.
        subscribe(&state, 1, 0);
        subscribe(&state, 1, STREAM_PRESENCE);
        assert_eq!(engines_started(&state), 2);

        end_engine(&state, Some(OpenRefusal::NoPermission));
        watch_engine_at(&state, Instant::now());

        assert_eq!(in_a_row(&state), 1);
    }

    #[test]
    fn a_dead_engine_a_request_finds_adds_a_step_and_is_replaced_at_once() {
        let state = Mutex::new(state_with_client(1));
        lock_state(&state).fake_present = true;
        subscribe(&state, 1, STREAM_PRESENCE);
        end_engine(&state, Some(OpenRefusal::InUse));

        let _ = lock_state(&state).device_for(1, true);

        assert_eq!(engines_started(&state), 2);
        assert_eq!(in_a_row(&state), 1);
        assert!(lock_state(&state).dead_engine.is_none(), "dropped");
    }

    #[test]
    fn a_dead_engine_found_as_another_client_leaves_adds_a_step() {
        let state = Mutex::new(state_with_client(1));
        {
            let mut st = lock_state(&state);
            let (out, _peer) = UnixStream::pair().expect("socket pair");
            st.clients.push(Client::new(2, out));
            st.fake_present = true;
        }
        subscribe(&state, 1, STREAM_PRESENCE);
        subscribe(&state, 2, STREAM_PRESENCE);
        end_engine(&state, Some(OpenRefusal::InUse));
        let started = engines_started(&state);

        subscribe(&state, 2, 0);

        let st = lock_state(&state);
        assert!(st.dead_engine.is_none(), "dropped");
        assert_eq!(st.restarts.in_a_row(), 1);
        assert_eq!(st.engines_started.len(), started, "left to the watchdog");
    }

    #[test]
    fn no_engine_wanted_any_more_ends_the_backoff() {
        let state = Mutex::new(state_with_client(1));
        lock_state(&state).fake_present = true;
        subscribe(&state, 1, STREAM_PRESENCE);
        let t0 = Instant::now();
        for _ in 0..5 {
            let _ = lock_state(&state).restarts.engine_failed(t0);
        }
        end_engine(&state, Some(OpenRefusal::NoPermission));

        // The last client leaves.
        subscribe(&state, 1, 0);

        let st = lock_state(&state);
        assert!(st.dead_engine.is_none(), "dropped");
        assert_eq!(st.restarts.in_a_row(), 0);
        assert_eq!(st.restarts.wait_left(t0), None);
    }

    #[test]
    fn the_tracker_back_on_the_bus_restarts_the_engine_without_the_backoff() {
        let state = pre_warmed_with_tracker();
        let t0 = Instant::now();
        for _ in 0..3 {
            let _ = lock_state(&state).restarts.engine_failed(t0);
        }
        watch_engine_at(&state, t0);
        assert_eq!(engines_started(&state), 0, "held off for 12 s");

        lock_state(&state).fake_present = false;
        watch_engine_at(&state, t0 + SECOND);
        lock_state(&state).fake_present = true;
        watch_engine_at(&state, t0 + 2 * SECOND);

        assert_eq!(engines_started(&state), 1);
        assert_eq!(in_a_row(&state), 0);
    }

    #[test]
    fn a_replug_between_two_passes_restarts_the_engine_without_the_backoff() {
        let state = pre_warmed_with_tracker();
        let t0 = Instant::now();
        for _ in 0..3 {
            let _ = lock_state(&state).restarts.engine_failed(t0);
        }
        watch_engine_at(&state, t0);
        watch_engine_at(&state, t0 + SECOND);
        assert_eq!(engines_started(&state), 0, "held off for 12 s");

        // Unplugged and plugged back in within a pass: another address.
        lock_state(&state).fake_address = 2;
        watch_engine_at(&state, t0 + 2 * SECOND);

        assert_eq!(engines_started(&state), 1);
        assert_eq!(in_a_row(&state), 0);
    }

    #[test]
    fn a_request_that_found_the_tracker_gone_counts_as_an_absence() {
        let state = Mutex::new(state_with_client(1));
        let t0 = Instant::now();
        let _ = lock_state(&state).restarts.engine_failed(t0);
        // The request finds no tracker: its client holds the device.
        assert!(lock_state(&state).device_for(1, false).is_none());

        lock_state(&state).fake_present = true;
        watch_engine_at(&state, t0);

        assert_eq!(engines_started(&state), 1, "plugged back in");
    }

    #[test]
    fn a_client_gets_an_engine_at_once_while_the_watchdog_backs_off() {
        let state = Mutex::new(state_with_client(1));
        let t0 = Instant::now();
        {
            let mut st = lock_state(&state);
            st.fake_present = true;
            let _ = st.restarts.engine_failed(t0);
        }

        subscribe(&state, 1, STREAM_PRESENCE);
        assert_eq!(engines_started(&state), 1, "a subscription");
        let _ = lock_state(&state).device_for(1, true);
        assert_eq!(engines_started(&state), 2, "a request");

        watch_engine_at(&state, t0);
        assert_eq!(engines_started(&state), 2, "not the watchdog");
    }

    #[test]
    fn a_device_init_takes_the_clock_pair_along() {
        let mut st = state_with_client(1);
        st.note_clock(5_000_000, 12_000_000_000);

        st.observe(&Sample::DeviceReady(Arc::new(DeviceFacts::default())));

        assert_eq!(st.clock, None);
    }

    /// The clock pair takes a gaze frame's read time, which `tobii_timesync`
    /// brackets, not the host time the frame is sent with.
    #[test]
    fn the_clock_pair_takes_a_gaze_frames_read_time() {
        let mut st = state_with_client(1);
        let frame = tobii_proto::gaze83::GazeFrame {
            device_ts_us: 5_000_000,
            ..tobii_proto::gaze83::GazeFrame::default()
        };
        // Read at 100, taken at 90 on the host clock.
        let gaze = Sample::Gaze(Box::new(tobii_usb::engine::GazeSample::new(frame, 100, 90)));

        st.observe(&gaze);

        assert_eq!((st.clock, st.gaze_frames), (Some((5_000_000, 100)), 1));
        let mut sent = Vec::new();
        push_sample_frames(&gaze, tobii_ipc::STREAM_GAZE_DATA, &mut sent);
        let [(_, body)] = &sent[..] else {
            panic!("no gaze data frame: {sent:?}");
        };
        let Some(tobii_ipc::ServerMsg::GazeData(data)) = tobii_ipc::decode_server(body) else {
            panic!("not gaze data: {body:?}");
        };
        assert_eq!(
            (data.timestamp_tracker_us, data.timestamp_system_us),
            (5_000_000, 90)
        );
    }

    #[test]
    fn an_init_without_a_catalogue_keeps_the_previous_one() {
        let mut st = state_with_client(1);
        let gaze = tobii_ipc::request::StreamType {
            id: 0x500,
            name: "gaze".into(),
            ..Default::default()
        };
        let ready = |streams: Vec<_>| {
            Sample::DeviceReady(Arc::new(DeviceFacts {
                streams,
                ..DeviceFacts::default()
            }))
        };
        st.observe(&ready(vec![gaze.clone()]));

        st.observe(&ready(vec![]));
        assert_eq!(
            st.facts.as_ref().map(|f| f.streams.clone()),
            Some(vec![gaze])
        );

        let image = tobii_ipc::request::StreamType {
            id: 0x501,
            name: "image".into(),
            ..Default::default()
        };
        st.observe(&ready(vec![image.clone()]));
        assert_eq!(
            st.facts.as_ref().map(|f| f.streams.clone()),
            Some(vec![image]),
            "a reported catalogue replaces the old one"
        );
    }

    #[test]
    fn an_init_without_a_hardware_configuration_keeps_the_previous_one() {
        let mut st = state_with_client(1);
        let ready = |hardware| {
            Sample::DeviceReady(Arc::new(DeviceFacts {
                hardware,
                ..DeviceFacts::default()
            }))
        };
        let hardware = tobii_ipc::request::HardwareConfiguration {
            mode: 1,
            ..Default::default()
        };
        st.observe(&ready(Some(hardware.clone())));

        st.observe(&ready(None));

        assert_eq!(
            st.facts.as_ref().and_then(|f| f.hardware.clone()),
            Some(hardware)
        );
    }

    #[test]
    fn an_init_without_properties_keeps_the_integration_type() {
        use crate::requests::{handle, tests::Answering};
        use tobii_ipc::request::{Request, decode_device_info, kind, status};
        let mut st = state_with_client(1);
        st.fake_device = Some(Arc::new(Answering(1)));
        let state = Mutex::new(st);
        let ready = |properties: Vec<(u32, String)>| {
            Sample::DeviceReady(Arc::new(DeviceFacts {
                properties,
                ..DeviceFacts::default()
            }))
        };
        let integration_type = || {
            let reply = handle(
                &state,
                1,
                &Request {
                    id: 1,
                    kind: kind::DEVICE_INFO,
                    payload: &[],
                },
            );
            assert_eq!(reply.status, status::OK);
            decode_device_info(&reply.payload).map(|i| i.integration_type)
        };
        lock_state(&state).observe(&ready(vec![(0, "Peripheral".into())]));
        assert_eq!(integration_type().as_deref(), Some("Peripheral"));

        lock_state(&state).observe(&ready(vec![]));

        assert_eq!(integration_type().as_deref(), Some("Peripheral"));

        lock_state(&state).observe(&ready(vec![(0, "HMD".into())]));
        assert_eq!(
            integration_type().as_deref(),
            Some("HMD"),
            "reported properties replace the old ones"
        );
    }

    #[test]
    fn an_init_without_a_status_drops_the_fault_and_warning_lists() {
        use crate::requests::{Reply, handle, tests::Answering};
        use tobii_ipc::request::{Request, encode_u32, kind, state, status};
        let mut st = state_with_client(1);
        st.fake_device = Some(Arc::new(Answering(1)));
        let state = Mutex::new(st);
        let ready = |status: Vec<(u32, String)>| {
            Sample::DeviceReady(Arc::new(DeviceFacts {
                status,
                ..DeviceFacts::default()
            }))
        };
        let ask = |id| {
            handle(
                &state,
                1,
                &Request {
                    id: 1,
                    kind: kind::STATE,
                    payload: &encode_u32(id),
                },
            )
        };
        lock_state(&state).observe(&ready(vec![(5, "ok".into()), (6, "ok".into())]));
        assert_eq!(ask(state::FAULT), Reply::ok(b"ok".to_vec()));
        assert_eq!(ask(state::WARNING), Reply::ok(b"ok".to_vec()));

        lock_state(&state).observe(&ready(vec![]));

        assert_eq!(ask(state::FAULT), Reply::err(status::NOT_SUPPORTED));
        assert_eq!(ask(state::WARNING), Reply::err(status::NOT_SUPPORTED));
    }

    /// A fault list announced before the first init has no facts to go
    /// into, and makes none (a guard: it held before lists were followed).
    /// That subscribers still get it is the frames tests'.
    #[test]
    fn a_fault_list_before_the_first_init_makes_no_facts() {
        let mut st = state_with_client(1);

        st.observe(&Sample::Notification(DeviceNotification::FaultsChanged(
            "FAULT_A".into(),
        )));

        assert_eq!(st.facts, None);
    }

    /// Samples are folded in in the order they are observed (the pump folds
    /// a whole drained batch in before any of it goes out): a list observed
    /// before an init's `DeviceReady` gives way to that init's 1490; one
    /// observed after it replaces it, including one the tracker sent during
    /// that init, which the engine delivers after its `DeviceReady`.
    #[test]
    fn a_fault_list_and_an_init_count_in_the_order_observed() {
        use tobii_proto::facts::STATUS_FAULTS;
        let faults = Sample::Notification(DeviceNotification::FaultsChanged("FAULT_A".into()));
        let ready = Sample::DeviceReady(Arc::new(DeviceFacts {
            status: vec![(STATUS_FAULTS, "ok".into())],
            ..DeviceFacts::default()
        }));
        let fault_list = |batch: [&Sample; 2]| {
            let mut st = state_with_client(1);
            st.observe(&Sample::DeviceReady(Arc::new(DeviceFacts {
                status: vec![(STATUS_FAULTS, "FAULT_OLD".into())],
                ..DeviceFacts::default()
            })));
            for s in batch {
                st.observe(s);
            }
            st.facts
                .as_ref()
                .and_then(|f| f.status_string(STATUS_FAULTS))
                .map(str::to_owned)
        };

        assert_eq!(fault_list([&faults, &ready]).as_deref(), Some("ok"));
        assert_eq!(fault_list([&ready, &faults]).as_deref(), Some("FAULT_A"));
    }

    #[test]
    fn a_new_presence_subscriber_gets_the_last_state() {
        let state = Mutex::new(state_with_client(1));
        // Device time 77, which is 70 on the host clock.
        let presence = Sample::Presence(PresenceSample::new(77, 70, true));
        {
            let mut st = lock_state(&state);
            st.observe(&presence);
            // No engine in tests: keep reconcile/ensure_engine from starting one.
            st.prewarm = false;
        }
        // Subscribe without touching the engine: set the mask directly and
        // run the presence replay handle_subscribe runs.
        let mut st = lock_state(&state);
        if let Some(c) = st.clients.iter_mut().find(|c| c.id == 1) {
            c.streams = STREAM_PRESENCE;
        }
        replay_presence(&mut st, 1, 0, STREAM_PRESENCE);
        let sent = outbox(&st, 1);
        assert_eq!(
            tobii_ipc::decode_server(&sent[0]),
            Some(tobii_ipc::ServerMsg::Presence {
                ts_us: 70,
                status: tobii_ipc::PRESENCE_PRESENT
            })
        );
        let mut live = Vec::new();
        push_sample_frames(&presence, STREAM_PRESENCE, &mut live);
        assert_eq!(
            live,
            [(STREAM_PRESENCE, sent[0].clone())],
            "the frame the subscribers of the time got, host time and all"
        );
    }

    #[test]
    fn a_paused_device_replays_no_presence() {
        let mut st = state_with_client(1);
        st.observe(&Sample::Presence(PresenceSample::new(77, 70, true)));
        st.paused = true;

        st.observe(&Sample::Presence(PresenceSample::new(88, 80, false)));
        replay_presence(&mut st, 1, 0, STREAM_PRESENCE);

        assert!(outbox(&st, 1).is_empty());
        st.paused = false;
        replay_presence(&mut st, 1, 0, STREAM_PRESENCE);
        assert_eq!(
            tobii_ipc::decode_server(&outbox(&st, 1)[0]),
            Some(tobii_ipc::ServerMsg::Presence {
                ts_us: 70,
                status: tobii_ipc::PRESENCE_PRESENT
            }),
            "the presence from before the pause is kept"
        );
    }

    /// The pump hangs up on a client it cannot write to (here at a sample
    /// of the stream it subscribes to), and neither writes nor queues
    /// anything for it any more, nor encodes what only it subscribes to,
    /// but leaves it listed with the streams and the device it holds, for
    /// its reader to clean up after: no engine stops meanwhile. The clients
    /// after it are still written their outbox, then the samples of the
    /// streams they subscribe to, and only those.
    #[test]
    fn the_pump_hangs_up_on_a_client_it_cannot_write_to_and_leaves_it_listed() {
        use tobii_ipc::{STREAM_GAZE, STREAM_NOTIFICATIONS};
        // Client 1's peer is gone.
        let mut st = state_with_client(1);
        st.clients[0].streams = STREAM_NOTIFICATIONS;
        st.clients[0].holds_device = true;
        let (out, mut peer) = UnixStream::pair().expect("socket pair");
        st.clients.push(Client::new(2, out));
        st.clients[1].streams = STREAM_GAZE;
        st.send_to(2, encode_subscribed(true));
        let frames = [
            (STREAM_GAZE, b"gaze".to_vec()),
            (STREAM_NOTIFICATIONS, b"notification".to_vec()),
        ];
        let losses = st.engine_losses;

        write_clients(&mut st, &frames);

        assert_eq!(
            st.clients
                .iter()
                .map(|c| (c.id, c.hung_up))
                .collect::<Vec<_>>(),
            [(1, true), (2, false)]
        );
        assert!(st.is_engine_wanted(), "client 1 still counts");
        assert_eq!(st.engine_losses, losses, "no engine dropped");
        assert_eq!(
            st.wanted_mask(),
            STREAM_GAZE,
            "client 1's stream is not encoded"
        );
        st.send_to(1, encode_subscribed(true));
        st.broadcast(STREAM_NOTIFICATIONS, b"notification");
        assert!(outbox(&st, 1).is_empty(), "nothing queued for client 1");
        // Client 2's end closes with the state: what it was written is
        // followed by the end of the stream.
        drop(st);
        peer.set_read_timeout(Some(WAIT)).expect("read timeout");
        let mut written = Vec::new();
        while let Some(body) = read_frame(&mut peer).expect("a frame or the end") {
            written.push(body);
        }
        assert_eq!(
            written,
            [encode_subscribed(true), b"gaze".to_vec()],
            "client 2's outbox, then its stream's sample"
        );
    }

    /// How long a test waits for what must happen.
    const WAIT: Duration = Duration::from_secs(5);
    /// How long a test gives what must not happen the time to.
    const GRACE: Duration = Duration::from_millis(100);

    /// Holds the first command it gets (the first `held`, when set) until
    /// the test lets it go (or drops its keys), answers the rest at once,
    /// and logs every command. A calibration read gets the calibration the
    /// device computes (see [`device_blob`]).
    struct Gate {
        log: Mutex<Vec<(u32, Vec<u8>)>>,
        held: Option<u32>,
        armed: AtomicBool,
        entered: Mutex<mpsc::Sender<()>>,
        release: Mutex<Receiver<()>>,
    }

    /// The test's side of a [`Gate`].
    struct GateKeys {
        /// Rung once the command is held.
        entered: Receiver<()>,
        /// Lets it go.
        release: mpsc::Sender<()>,
    }

    fn gate() -> (Arc<Gate>, GateKeys) {
        gate_holding(None)
    }

    /// A [`Gate`] that holds the first `held` command (`None`: whichever
    /// comes first).
    fn gate_holding(held: Option<u32>) -> (Arc<Gate>, GateKeys) {
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let gate = Gate {
            log: Mutex::new(Vec::new()),
            held,
            armed: AtomicBool::new(true),
            entered: Mutex::new(entered_tx),
            release: Mutex::new(release_rx),
        };
        (Arc::new(gate), GateKeys { entered, release })
    }

    impl Gate {
        fn log(&self) -> Vec<(u32, Vec<u8>)> {
            self.log.lock().expect("log").clone()
        }
    }

    impl DeviceCommands for Gate {
        fn run(
            &self,
            cmd: u32,
            payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<tobii_usb::engine::CommandResponse, tobii_usb::engine::CommandError> {
            self.log.lock().expect("log").push((cmd, payload));
            if self.held.is_none_or(|held| held == cmd) && self.armed.swap(false, Ordering::Relaxed)
            {
                let _ = self.entered.lock().expect("entered").send(());
                // Bounded, should a test forget to let it go.
                let _ = self
                    .release
                    .lock()
                    .expect("release")
                    .recv_timeout(Duration::from_secs(10));
            }
            let answer = if cmd == tobii_proto::calibration::cmd::READ {
                tobii_proto::calibration::write_payload(&device_blob())
            } else {
                Vec::new()
            };
            Ok(tobii_usb::engine::CommandResponse::ok(answer))
        }
    }

    /// Client 1's connection, registered (with the pump's write timeout)
    /// and served by [`client_reader`] as the accept loop does, with
    /// `device` standing in for the engine. The pump does not run: what the
    /// daemon sends the client stays in its outbox, unless a test runs a
    /// pass of it ([`pump_once`]) or hangs up on the client as one does
    /// ([`Client::hang_up`]), on the socket the reader reads.
    struct Connection {
        state: Arc<Mutex<State>>,
        /// The client's end, until it hangs up.
        peer: Option<UnixStream>,
        /// Hung up once the reader has returned.
        served: Receiver<()>,
    }

    impl Connection {
        fn open(device: Arc<dyn DeviceCommands>) -> Self {
            let (peer, stream) = UnixStream::pair().expect("socket pair");
            let out = stream.try_clone().expect("socket clone");
            out.set_write_timeout(Some(WRITE_TIMEOUT))
                .expect("write timeout");
            let mut st = State::new(false);
            st.clients.push(Client::new(1, out));
            st.fake_device = Some(device);
            // No calibration file is read or saved.
            st.calibration.location = tobii_calib::store::Location::Embedded;
            let state = Arc::new(Mutex::new(st));
            let (served_tx, served) = mpsc::channel::<()>();
            let reader_state = Arc::clone(&state);
            thread::spawn(move || {
                client_reader(&reader_state, 1, stream);
                drop(served_tx);
            });
            Self {
                state,
                peer: Some(peer),
                served,
            }
        }

        fn send(&mut self, body: &[u8]) {
            let peer = self.peer.as_mut().expect("still connected");
            write_frame(peer, body).expect("frame written");
        }

        fn request(&mut self, id: u32, kind: u8, payload: &[u8]) {
            self.send(&tobii_ipc::request::encode_request(id, kind, payload));
        }

        fn hang_up(&mut self) {
            self.peer = None;
        }

        /// What the daemon has sent the client so far, decoded.
        fn sent(&self) -> Vec<tobii_ipc::ServerMsg> {
            outbox(&lock_state(&self.state), 1)
                .iter()
                .map(|body| tobii_ipc::decode_server(body).expect("a server frame"))
                .collect()
        }

        /// Wait until the daemon has sent the client `n` frames.
        fn wait_for_frames(&self, n: usize) -> Vec<tobii_ipc::ServerMsg> {
            let deadline = std::time::Instant::now() + WAIT;
            loop {
                let sent = self.sent();
                if sent.len() >= n {
                    return sent;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "{n} frames expected, {} sent: {sent:?}",
                    sent.len()
                );
                thread::sleep(Duration::from_millis(5));
            }
        }

        /// Wait until the reader has returned, having cleaned up after the
        /// client.
        fn wait_until_served(&self) {
            assert_eq!(
                self.served.recv_timeout(WAIT),
                Err(mpsc::RecvTimeoutError::Disconnected),
                "the reader returned in time"
            );
        }
    }

    fn reply(request_id: u32, status: u8, payload: &[u8]) -> tobii_ipc::ServerMsg {
        tobii_ipc::ServerMsg::Reply {
            request_id,
            status,
            payload: payload.to_vec(),
        }
    }

    const ACK: tobii_ipc::ServerMsg = tobii_ipc::ServerMsg::Subscribed { ok: true };

    /// Found on hardware: a calibration retrieve waiting for a cold engine
    /// held the client's subscribe up past libtobii's 2 s. Neither do the
    /// requests behind it (libtobii's that timed out meanwhile), as many as
    /// the queue holds.
    #[test]
    fn a_request_on_the_device_holds_no_subscription_ack_up() {
        use tobii_ipc::request::{encode_display_area, encode_u32, kind, state, status};
        let (device, keys) = gate();
        let mut c = Connection::open(device);
        c.request(
            1,
            kind::DISPLAY_AREA_SET,
            &encode_display_area(&area(600.0)),
        );
        keys.entered
            .recv_timeout(WAIT)
            .expect("the request is held");
        let paused = encode_u32(state::DEVICE_PAUSED);
        // A full queue.
        let last = u32::try_from(REQUEST_QUEUE).expect("queue length") + 1;
        for id in 2..=last {
            c.request(id, kind::STATE, &paused);
        }

        c.send(&tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE));

        assert_eq!(c.wait_for_frames(1), [ACK], "acked while the requests wait");
        keys.release.send(()).expect("let go");
        let mut expected = vec![ACK, reply(1, status::OK, &[])];
        expected.extend((2..=last).map(|id| reply(id, status::OK, &[0])));
        assert_eq!(c.wait_for_frames(expected.len()), expected);
        c.hang_up();
        c.wait_until_served();
        assert!(lock_state(&c.state).clients.is_empty());
    }

    #[test]
    fn a_clients_requests_run_one_at_a_time_in_order() {
        use tobii_ipc::request::{encode_display_area, kind, status};
        let (device, keys) = gate();
        let mut c = Connection::open(device);
        c.request(
            1,
            kind::DISPLAY_AREA_SET,
            &encode_display_area(&area(600.0)),
        );
        keys.entered
            .recv_timeout(WAIT)
            .expect("the request is held");
        c.request(2, kind::DEVICE_NAME_SET, b"desk");
        // Answers the name only if it runs after the set.
        c.request(3, kind::DEVICE_NAME_GET, &[]);

        thread::sleep(GRACE);
        assert!(c.sent().is_empty(), "none overtakes the one on the device");
        keys.release.send(()).expect("let go");

        assert_eq!(
            c.wait_for_frames(3),
            [
                reply(1, status::OK, &[]),
                reply(2, status::OK, &[]),
                reply(3, status::OK, b"desk"),
            ]
        );
    }

    /// The pause the client asked for is taken before its hang-up is acted
    /// on, and then resumed once: cleaning up before the request ran would
    /// leave the device paused for a client that is gone.
    #[test]
    fn a_client_that_hangs_up_during_a_request_is_cleaned_up_after_it_once() {
        use tobii_ipc::request::kind;
        use tobii_proto::facts::device_pause_payload;
        use tobii_proto::protocol::cmd;
        let (device, keys) = gate();
        let mut c = Connection::open(Arc::clone(&device) as Arc<dyn DeviceCommands>);
        c.request(1, kind::DEVICE_PAUSE, &[1]);
        keys.entered.recv_timeout(WAIT).expect("the pause is held");
        c.request(2, kind::DEVICE_NAME_SET, b"desk");

        c.hang_up();

        thread::sleep(GRACE);
        assert_eq!(
            lock_state(&c.state).clients.len(),
            1,
            "not cleaned up while its request runs"
        );
        keys.release.send(()).expect("let go");
        c.wait_until_served();
        assert_eq!(
            device.log(),
            [
                (cmd::DEVICE_PAUSE, device_pause_payload(true)),
                (cmd::DEVICE_PAUSE, device_pause_payload(false)),
            ],
            "paused, then resumed once"
        );
        let st = lock_state(&c.state);
        // No pause landed after the cleanup. The log above shows the
        // resume: the engine dropped once the client went clears these
        // either way.
        assert!(!st.paused);
        assert_eq!(st.pause_holder, None);
        assert_eq!(
            st.device_name.as_deref(),
            Some(&b"desk"[..]),
            "a request sent before the hang-up still ran"
        );
        assert!(st.clients.is_empty());
    }

    /// A connection whose client started a calibration session and
    /// computed a calibration, and whose collect (request 3) is held on the
    /// device.
    fn calibrating_with_a_collect_held() -> (Arc<Gate>, GateKeys, Connection) {
        use tobii_ipc::request::{encode_point_2d, encode_u32, kind, status};
        use tobii_proto::calibration::cmd;
        let (device, keys) = gate_holding(Some(cmd::COLLECT_2D));
        let mut c = Connection::open(Arc::clone(&device) as Arc<dyn DeviceCommands>);
        // Both eyes.
        c.request(1, kind::CALIBRATION_START, &[2]);
        c.request(2, kind::CALIBRATION_COMPUTE, &[]);
        let computed = tobii_calib::blob::calibration_id(&device_blob()).expect("an id");
        assert_eq!(
            c.wait_for_frames(2),
            [
                reply(1, status::OK, &[]),
                reply(2, status::OK, &encode_u32(computed)),
            ]
        );
        c.request(3, kind::CALIBRATION_COLLECT_2D, &encode_point_2d(0.5, 0.1));
        keys.entered
            .recv_timeout(WAIT)
            .expect("the collect is held");
        (device, keys, c)
    }

    /// What the device got after the collect.
    fn after_the_collect(device: &Gate) -> Vec<(u32, Vec<u8>)> {
        let log = device.log();
        let collect = log
            .iter()
            .position(|(command, _)| *command == tobii_proto::calibration::cmd::COLLECT_2D)
            .expect("the collect");
        log[collect + 1..].to_vec()
    }

    /// A stop that keeps the session, sent without waiting for the answer
    /// just before the hang-up, commits once the request before it has run;
    /// the cleanup then finds no session to discard.
    #[test]
    fn a_kept_stop_sent_before_a_hang_up_during_a_request_keeps_the_session() {
        use tobii_ipc::request::{STOP_KEEP, kind};
        use tobii_proto::calibration::{cmd, write_payload};
        let (device, keys, mut c) = calibrating_with_a_collect_held();
        c.request(4, kind::CALIBRATION_STOP, STOP_KEEP);

        c.hang_up();

        thread::sleep(GRACE);
        assert_eq!(
            after_the_collect(&device),
            [],
            "nothing overtakes the collect"
        );
        keys.release.send(()).expect("let go");
        c.wait_until_served();
        assert_eq!(
            after_the_collect(&device),
            [
                (cmd::STOP, Vec::new()),
                (cmd::WRITE, write_payload(&device_blob())),
            ],
            "stopped once, keeping what the session computed"
        );
        let st = lock_state(&c.state);
        assert!(!st.calibration.is_active());
        assert_eq!(
            st.calibration.id,
            tobii_calib::blob::calibration_id(&device_blob())
        );
        assert!(st.clients.is_empty());
    }

    /// As above for a client that streams gaze, with the pump finding the
    /// hang-up first: it writes to the client while the collect runs. The
    /// client stays listed until its requests have run, so no engine stops
    /// under them and the stop still commits; the engine stops once the
    /// client is cleaned up after.
    #[test]
    fn a_kept_stop_sent_before_a_hang_up_the_pump_finds_first_keeps_the_session() {
        use tobii_ipc::request::{STOP_KEEP, kind};
        use tobii_proto::calibration::{cmd, write_payload};
        let (device, keys, mut c) = calibrating_with_a_collect_held();
        // A client the pump writes to at every pass. Incidental to what is
        // shown: the device the client holds keeps it counted either way.
        lock_state(&c.state).clients[0].streams = tobii_ipc::STREAM_GAZE;
        c.request(4, kind::CALIBRATION_STOP, STOP_KEEP);
        c.hang_up();
        let losses = lock_state(&c.state).engine_losses;

        // It writes the replies waiting in the outbox to a peer that is gone.
        pump_once(&mut lock_state(&c.state), &mut Vec::new(), &mut Vec::new());

        {
            let st = lock_state(&c.state);
            assert!(
                st.clients.iter().any(|c| c.id == 1 && c.hung_up),
                "hung up on, and still listed"
            );
            assert_eq!(st.engine_losses, losses, "no engine stopped");
            assert!(st.calibration.is_active(), "the session is not discarded");
        }
        keys.release.send(()).expect("let go");
        c.wait_until_served();
        assert_eq!(
            after_the_collect(&device),
            [
                (cmd::STOP, Vec::new()),
                (cmd::WRITE, write_payload(&device_blob())),
            ],
            "stopped once, keeping what the session computed"
        );
        let st = lock_state(&c.state);
        assert!(!st.calibration.is_active());
        assert_eq!(
            st.calibration.id,
            tobii_calib::blob::calibration_id(&device_blob())
        );
        assert_eq!(
            st.engine_losses,
            losses.wrapping_add(1),
            "stopped once, after the requests"
        );
        assert!(st.clients.is_empty());
    }

    /// A client the pump hangs up on while its peer keeps the connection
    /// open without reading (a write that timed out) is cleaned up after
    /// all the same: the hang-up ends its reader's reads.
    #[test]
    fn a_client_the_pump_hangs_up_on_is_cleaned_up_after_while_its_peer_stays_open() {
        let (device, _keys) = gate();
        let mut c = Connection::open(device);
        c.send(&tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE));
        assert_eq!(c.wait_for_frames(1), [ACK]);
        let losses = lock_state(&c.state).engine_losses;

        lock_state(&c.state).clients[0].hang_up();

        c.wait_until_served();
        {
            let st = lock_state(&c.state);
            assert!(st.clients.is_empty());
            assert_eq!(
                st.engine_losses,
                losses.wrapping_add(1),
                "the engine stopped once"
            );
        }
        let peer = c.peer.as_mut().expect("still connected");
        peer.set_read_timeout(Some(WAIT)).expect("read timeout");
        assert!(
            matches!(read_frame(peer), Ok(None)),
            "the connection is closed"
        );
    }

    /// The session of a client that hangs up during a request of it is
    /// discarded once the request has run, and once.
    #[test]
    fn a_calibrating_client_that_hangs_up_during_a_request_has_its_session_discarded_after_it() {
        use tobii_proto::calibration::{cmd, write_payload};
        let (device, keys, mut c) = calibrating_with_a_collect_held();

        c.hang_up();

        thread::sleep(GRACE);
        assert_eq!(
            after_the_collect(&device),
            [],
            "not discarded while the collect runs"
        );
        keys.release.send(()).expect("let go");
        c.wait_until_served();
        let previous = tobii_usb::calibration::embedded_blob().expect("blob");
        assert_eq!(
            after_the_collect(&device),
            [
                (cmd::STOP, Vec::new()),
                (cmd::WRITE, write_payload(&previous))
            ],
            "stopped once, putting back the calibration it started from"
        );
        let st = lock_state(&c.state);
        assert!(!st.calibration.is_active());
        assert_eq!(
            st.calibration.id,
            tobii_calib::blob::calibration_id(&previous)
        );
        assert!(st.clients.is_empty());
    }

    #[test]
    fn a_client_with_a_full_request_queue_is_read_no_further() {
        use tobii_ipc::request::{encode_display_area, encode_u32, kind, state, status};
        let (device, keys) = gate();
        let mut c = Connection::open(device);
        c.request(
            1,
            kind::DISPLAY_AREA_SET,
            &encode_display_area(&area(600.0)),
        );
        keys.entered
            .recv_timeout(WAIT)
            .expect("the request is held");
        let paused = encode_u32(state::DEVICE_PAUSED);
        // A full queue, and one more that the reader waits to queue.
        let last = u32::try_from(REQUEST_QUEUE).expect("queue length") + 2;
        for id in 2..=last {
            c.request(id, kind::STATE, &paused);
        }

        c.send(&tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE));

        thread::sleep(GRACE);
        assert!(c.sent().is_empty(), "the subscription waits in the socket");
        keys.release.send(()).expect("let go");
        let count = usize::try_from(last).expect("frame count") + 1;
        let sent = c.wait_for_frames(count);
        let replies: Vec<_> = sent.iter().filter(|m| **m != ACK).cloned().collect();
        let mut expected = vec![reply(1, status::OK, &[])];
        expected.extend((2..=last).map(|id| reply(id, status::OK, &[0])));
        assert_eq!(replies, expected, "every request answered, in order");
        assert_eq!(sent.len() - replies.len(), 1, "and the subscription acked");
    }

    /// Answers nothing: panics.
    struct Panicking;

    impl DeviceCommands for Panicking {
        fn run(
            &self,
            _cmd: u32,
            _payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<tobii_usb::engine::CommandResponse, tobii_usb::engine::CommandError> {
            panic!("the stand-in device panics");
        }
    }

    /// Nothing answers the client's requests once its worker panicked: the
    /// connection is closed then, not at the client's next request.
    #[test]
    fn a_connection_whose_request_worker_panicked_is_closed() {
        use tobii_ipc::request::{encode_display_area, kind};
        let mut c = Connection::open(Arc::new(Panicking));

        c.request(
            1,
            kind::DISPLAY_AREA_SET,
            &encode_display_area(&area(600.0)),
        );

        c.wait_until_served();
        assert!(lock_state(&c.state).clients.is_empty());
        let peer = c.peer.as_mut().expect("still connected");
        peer.set_read_timeout(Some(WAIT)).expect("read timeout");
        assert!(
            matches!(read_frame(peer), Ok(None)),
            "the connection is closed"
        );
    }

    /// The client is found gone once it closes its end or shuts down its
    /// sending, or once the pump hangs up on it, with the frame it sent
    /// still there to be read; not while it is connected, frame or none.
    #[test]
    fn a_client_is_found_hung_up_with_its_frames_still_unread() {
        let subscribe = tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE);
        let (mut ours, mut peer) = UnixStream::pair().expect("socket pair");
        assert!(!has_peer_hung_up(1, &ours), "connected");
        write_frame(&mut peer, &subscribe).expect("frame written");
        assert!(
            !has_peer_hung_up(1, &ours),
            "connected, with a frame unread"
        );
        drop(peer);
        assert!(has_peer_hung_up(1, &ours), "closed, with a frame unread");
        assert_eq!(read_frame(&mut ours).expect("a frame"), Some(subscribe));

        let (ours, peer) = UnixStream::pair().expect("socket pair");
        peer.shutdown(Shutdown::Write).expect("half-close");
        assert!(has_peer_hung_up(1, &ours), "done sending");

        let (ours, _peer) = UnixStream::pair().expect("socket pair");
        let mut client = Client::new(1, ours.try_clone().expect("socket clone"));
        client.hang_up();
        assert!(has_peer_hung_up(1, &ours), "hung up on by the pump");
    }

    /// Serve, as the accept loop does, client 1's connection over which it
    /// sent `frames` and hung up before the daemon read any of them. The
    /// reader runs here, and has cleaned up after the client on return.
    fn serve_abandoned(state: &Arc<Mutex<State>>, frames: &[Vec<u8>]) {
        let (mut peer, stream) = UnixStream::pair().expect("socket pair");
        for body in frames {
            write_frame(&mut peer, body).expect("frame written");
        }
        drop(peer);
        let out = stream.try_clone().expect("socket clone");
        lock_state(state).clients.push(Client::new(1, out));
        client_reader(state, 1, stream);
    }

    /// A daemon with its tracker on the bus: a subscription it acts on
    /// starts an engine.
    fn with_tracker() -> Arc<Mutex<State>> {
        let mut st = State::new(false);
        st.fake_present = true;
        Arc::new(Mutex::new(st))
    }

    /// Noted in d8c6a35: a libtobii reconnect that gave up on its ack while
    /// systemd held the socket for a restart leaves its subscription in the
    /// backlog, and the hang-up behind it. The daemon that comes up neither
    /// looks for the tracker nor starts an engine for it, only to drop the
    /// engine again under the state lock a live reconnect's ack waits on.
    #[test]
    fn a_subscription_from_a_client_that_has_hung_up_starts_no_engine() {
        let state = with_tracker();

        serve_abandoned(
            &state,
            &[tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE)],
        );

        let st = lock_state(&state);
        assert!(st.engines_started.is_empty(), "no engine started");
        assert_eq!(st.presence_probes, 0, "nor the tracker looked for");
        assert!(st.clients.is_empty(), "cleaned up after");
    }

    /// A client that hangs up while its subscription waits, for the bus
    /// scan or for the state lock another reader holds while it drops an
    /// engine, is looked at again under the lock, and nothing changes: no
    /// engine starts for it and no ack is queued.
    #[test]
    fn a_client_that_hangs_up_while_its_subscription_waits_gets_no_engine() {
        use std::cell::RefCell;
        use tobii_ipc::STREAM_GAZE;
        let state = Mutex::new(state_with_client(1));
        lock_state(&state).fake_present = true;
        // Each look at the client: whether the state lock was held then.
        let looks = RefCell::new(Vec::new());
        let gone = || {
            let mut looks = looks.borrow_mut();
            looks.push(state.try_lock().is_err());
            // Connected at the first look, gone by the next.
            looks.len() > 1
        };

        assert!(!handle_subscribe(&state, 1, STREAM_GAZE, gone), "skipped");

        assert_eq!(
            *looks.borrow(),
            [false, true],
            "looked at again under the lock"
        );
        let st = lock_state(&state);
        assert!(st.engines_started.is_empty(), "no engine started");
        assert!(outbox(&st, 1).is_empty(), "nor an ack queued");
        assert_eq!(st.clients[0].streams, 0, "nor the streams registered");
        assert_eq!(
            st.presence_probes, 1,
            "the bus was looked at while it was there"
        );
    }

    /// The frames after a skipped subscription are still read: a request
    /// the client sent before it hung up still runs.
    #[test]
    fn a_request_behind_a_skipped_subscription_still_runs() {
        use tobii_ipc::request::{encode_request, kind};
        let state = with_tracker();

        serve_abandoned(
            &state,
            &[
                tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE),
                encode_request(1, kind::DEVICE_NAME_SET, b"desk"),
            ],
        );

        let st = lock_state(&state);
        assert_eq!(st.device_name.as_deref(), Some(&b"desk"[..]));
        assert!(st.engines_started.is_empty(), "the subscription skipped");
        assert!(st.clients.is_empty());
    }

    /// Only the subscription is skipped: a request behind it that needs the
    /// device (a calibration stop that keeps the result, sent just before
    /// the client hung up, among them) still gets an engine started for it.
    #[test]
    fn a_device_request_behind_a_skipped_subscription_still_starts_its_engine() {
        use tobii_ipc::request::{encode_display_area, encode_request, kind};
        let state = with_tracker();

        serve_abandoned(
            &state,
            &[
                tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE),
                encode_request(
                    1,
                    kind::DISPLAY_AREA_SET,
                    &encode_display_area(&area(600.0)),
                ),
            ],
        );

        let st = lock_state(&state);
        assert_eq!(st.engines_started.len(), 1, "one engine, the request's");
        assert_eq!(st.presence_probes, 1, "one look at the bus, the request's");
        assert!(st.clients.is_empty());
    }

    /// A client that subscribes and stays connected is served: acked, with
    /// an engine started for it.
    #[test]
    fn a_subscription_from_a_connected_client_is_acked() {
        let mut c = Connection::open(Arc::new(crate::requests::tests::Answering(0)));

        c.send(&tobii_ipc::encode_subscribe(tobii_ipc::STREAM_GAZE));

        assert_eq!(c.wait_for_frames(1), [ACK]);
        assert_eq!(engines_started(&c.state), 1);
        c.hang_up();
        c.wait_until_served();
        assert!(lock_state(&c.state).clients.is_empty());
    }

    fn file_id(dev: u64, ino: u64) -> FileId {
        FileId { dev, ino }
    }

    #[test]
    fn a_socket_file_from_systemd_is_left_at_shutdown() {
        let file = SocketFile::Systemd;

        assert!(
            !file.is_ours_to_remove(Some(file_id(1, 7))),
            "the next daemon systemd starts is reached through it"
        );
        assert!(!file.is_ours_to_remove(None));
    }

    #[test]
    fn a_bound_socket_file_is_removed_only_while_its_path_names_it() {
        let file = SocketFile::Bound {
            path: PathBuf::from("/run/user/1000/tobiid.sock"),
            bound: file_id(1, 7),
        };

        assert!(file.is_ours_to_remove(Some(file_id(1, 7))));
        assert!(
            !file.is_ours_to_remove(Some(file_id(1, 8))),
            "another daemon has bound the path since"
        );
        assert!(
            !file.is_ours_to_remove(Some(file_id(2, 7))),
            "the same inode on another file system is another file"
        );
        assert!(!file.is_ours_to_remove(None), "nothing left to remove");
    }

    /// A fresh directory for one test's sockets, removed on drop. Under
    /// `/tmp` rather than `TMPDIR`, which can be too long for `sun_path`'s
    /// 108 bytes.
    struct SocketDir(PathBuf);

    impl SocketDir {
        fn new(test: &str) -> Self {
            let root = Path::new("/tmp");
            let root = if root.is_dir() {
                root.to_path_buf()
            } else {
                std::env::temp_dir()
            };
            let dir = root.join(format!("tobiid-sock-{test}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("socket dir");
            Self(dir)
        }

        fn socket(&self) -> PathBuf {
            self.0.join("tobiid.sock")
        }

        /// The names in the directory, sorted.
        fn names(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .expect("read socket dir")
                .map(|entry| {
                    entry
                        .expect("socket dir entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for SocketDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_daemon_removes_the_socket_file_it_bound_at_shutdown() {
        let dir = SocketDir::new("own");
        let path = dir.socket();
        let (_listener, file) = bind_listener(path.clone()).expect("bind");

        file.remove_at_shutdown();

        assert!(!path.exists());
    }

    #[test]
    fn a_daemon_replaces_a_stale_socket_file_and_leaves_nothing_else() {
        let dir = SocketDir::new("stale");
        let path = dir.socket();
        std::fs::write(&path, b"left by a previous run").expect("stale file");

        let (_listener, file) = bind_listener(path.clone()).expect("bind");

        assert_eq!(dir.names(), ["tobiid.sock"], "no file of its own is left");
        assert!(file.is_ours_to_remove(FileId::of(&path).ok()));
        UnixStream::connect(&path).expect("clients reach the daemon at the path");
    }

    #[test]
    fn a_daemon_that_cannot_take_the_socket_path_leaves_no_file_behind() {
        let dir = SocketDir::new("taken");
        let path = dir.socket();
        // A directory full of files, which a rename does not replace.
        std::fs::create_dir(&path).expect("directory at the path");
        std::fs::write(path.join("keep"), b"").expect("file in it");

        assert!(bind_listener(path.clone()).is_err());

        assert_eq!(dir.names(), ["tobiid.sock"], "no file of its own is left");
        assert!(path.join("keep").exists());
    }

    #[test]
    fn a_daemon_leaves_the_socket_file_of_a_daemon_started_since() {
        let dir = SocketDir::new("since");
        let path = dir.socket();
        let (_first_listener, first) = bind_listener(path.clone()).expect("bind");
        // A second daemon (spawned by a client, say) replaces the file while
        // the first still runs.
        let (_second_listener, second) = bind_listener(path.clone()).expect("bind again");

        first.remove_at_shutdown();

        assert!(
            second.is_ours_to_remove(FileId::of(&path).ok()),
            "the path still names the second daemon's socket"
        );
        UnixStream::connect(&path).expect("clients still reach the second daemon");
        second.remove_at_shutdown();
        assert!(!path.exists());
    }
}
