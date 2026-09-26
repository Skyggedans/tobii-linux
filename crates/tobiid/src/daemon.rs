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
//! Log lines go through `tracing` (the `tobiid` binary installs the
//! subscriber; under systemd stderr lands in the journal).

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::net::Shutdown;
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use tracing::{info, warn};

use tobii_ipc::geometry::DisplayArea;
use tobii_ipc::request::decode_request;
use tobii_ipc::{
    self, STREAM_HEAD, STREAM_IMAGE, STREAM_PRESENCE, decode_subscribe, encode_reply,
    encode_subscribed, read_frame, write_frame,
};
use tobii_proto::facts::{DeviceFacts, DeviceNotification};
use tobii_usb::engine::{Engine, PresenceSample, Sample};

use crate::calibration::Calibration;
use crate::device::DeviceCommands;
use crate::frames::{presence_frame, push_sample_frames};

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
}

pub(crate) struct State {
    pub(crate) engine: Option<Engine>,
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
            #[cfg(test)]
            fake_device: None,
            #[cfg(test)]
            pause_hint: None,
            #[cfg(test)]
            lose_engine_on_fetch: false,
            #[cfg(test)]
            fake_present: false,
            #[cfg(test)]
            presence_probes: 0,
            #[cfg(test)]
            engines_started: Vec::new(),
        }
    }

    /// Every stream some client subscribes to.
    fn wanted_mask(&self) -> u32 {
        self.clients.iter().fold(0, |m, c| m | c.streams)
    }

    /// True if some client consumes a stream or holds the device, or
    /// pre-warm is set.
    fn is_engine_wanted(&self) -> bool {
        self.prewarm
            || self
                .clients
                .iter()
                .any(|c| c.streams != 0 || c.holds_device)
    }

    /// Stop the engine once nobody needs it, unless pre-warm is configured.
    /// Also drops a dead engine. Starting another, for pre-warm too, is left
    /// to the watchdog, which looks for the tracker without the state lock
    /// (the pump runs this under it).
    fn reconcile(&mut self) {
        if self.engine.is_some() {
            self.drop_engine_unless_alive();
        }
        if !self.is_engine_wanted() {
            self.drop_engine();
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
    pub(crate) fn ensure_engine(&mut self, tracker_on_bus: bool) {
        if !self.drop_engine_unless_alive() {
            return;
        }
        if tracker_on_bus {
            self.start_engine();
        } else if !std::mem::replace(&mut self.tracker_absence_logged, true) {
            info!("no tracker on the bus; the engine starts once it is plugged in");
        }
    }

    /// Start an engine. In tests none starts, whatever the bus says: the
    /// display area it would start with is recorded instead.
    fn start_engine(&mut self) {
        self.tracker_absence_logged = false;
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

    /// Drop the engine unless it is running (see [`Self::drop_engine`]).
    /// Whether none runs now: the step [`Self::ensure_engine`] takes before
    /// it starts one, and the one [`Self::reconcile`] and the watchdog take
    /// for a dead engine.
    fn drop_engine_unless_alive(&mut self) -> bool {
        if self.engine.as_ref().is_some_and(Engine::is_alive) {
            return false;
        }
        self.drop_engine();
        true
    }

    /// Drop the engine, with what lasts only as long as it runs: a started
    /// calibration session unless its stop saves (see
    /// [`crate::calibration::on_engine_lost`]), the pause (see
    /// [`crate::pause::on_engine_lost`]) and the clock pair.
    pub(crate) fn drop_engine(&mut self) {
        self.engine = None;
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

    /// Queue `body` for one client.
    pub(crate) fn send_to(&mut self, client: u64, body: Vec<u8>) {
        if let Some(c) = self.clients.iter_mut().find(|c| c.id == client) {
            c.outbox.push_back(body);
        }
    }

    /// Queue `body` for every client subscribed to `stream`.
    pub(crate) fn broadcast(&mut self, stream: u32, body: &[u8]) {
        for c in self.clients.iter_mut().filter(|c| c.streams & stream != 0) {
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
            Sample::Gaze(g) if !self.paused => {
                let device = i64::try_from(g.frame.device_ts_us).unwrap_or(i64::MAX);
                self.note_clock(device, g.host_rx_us);
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

/// Whether the tracker is on the bus (see
/// [`tobii_usb::device::is_device_present`]), for a caller that does not
/// hold the state lock: libusb scans the bus (a few milliseconds) without
/// it.
#[cfg(not(test))]
pub(crate) fn is_tracker_present(_state: &Mutex<State>) -> bool {
    tobii_usb::device::is_device_present()
}

/// Whether the tracker is on the bus: in tests, what the state's stand-in
/// says, counted.
#[cfg(test)]
pub(crate) fn is_tracker_present(state: &Mutex<State>) -> bool {
    let mut st = lock_state(state);
    st.presence_probes += 1;
    st.fake_present
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
    // tracker was unplugged, but it's still wanted, restart it.
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
    let wanted = {
        let mut st = lock_state(state);
        // Dropped even while unplugged, so that clients hear of it.
        if st.engine.is_some() {
            st.drop_engine_unless_alive();
        }
        st.engine.is_none() && st.is_engine_wanted()
    };
    if !wanted || !is_tracker_present(state) {
        return;
    }
    let mut st = lock_state(state);
    // A request may have started one meanwhile.
    if st.engine.is_none() && st.is_engine_wanted() {
        warn!("engine not running but wanted; restarting");
        st.start_engine();
    }
}

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
        // a client that stops reading is dropped instead of wedging fan-out
        // (and the SIGTERM teardown, which needs the same lock).
        let _ = out.set_write_timeout(Some(Duration::from_millis(250)));
        let id = next_id;
        next_id = next_id.wrapping_add(1);
        lock_state(state).clients.push(Client {
            id,
            streams: 0,
            out,
            outbox: VecDeque::new(),
            holds_device: false,
        });
        let state = Arc::clone(state);
        thread::spawn(move || client_reader(&state, id, stream));
    }
}

/// One per connection: handle SUBSCRIBE, RECENTER and REQUEST frames and
/// detect disconnect. Replies go through the client's outbox.
fn client_reader(state: &Mutex<State>, id: u64, mut stream: UnixStream) {
    // Loop ends on EOF or a read error.
    while let Ok(Some(body)) = read_frame(&mut stream) {
        match body.first().copied() {
            Some(tobii_ipc::TAG_SUBSCRIBE) => {
                if let Some(streams) = decode_subscribe(&body) {
                    handle_subscribe(state, id, streams);
                }
            }
            Some(tobii_ipc::TAG_RECENTER) => {
                if let Some(engine) = lock_state(state).engine.as_ref() {
                    engine.request_recenter();
                }
            }
            Some(tobii_ipc::TAG_REQUEST) => {
                if let Some(request) = decode_request(&body) {
                    // Runs without the state lock held while the device works.
                    let reply = crate::requests::handle(state, id, &request);
                    lock_state(state)
                        .send_to(id, encode_reply(request.id, reply.status, &reply.payload));
                }
            }
            _ => {}
        }
    }
    // Resumed before the engine may stop below.
    crate::pause::release(state, id);
    crate::calibration::on_client_gone(state, id);
    let mut st = lock_state(state);
    st.clients.retain(|c| c.id != id);
    st.reconcile();
}

/// Register the client's streams, starting the engine if it isn't running
/// (and the tracker is on the bus; otherwise the watchdog starts it once the
/// tracker is plugged in), and acknowledge. Always succeeds (the one engine
/// serves every stream); `streams == 0` unsubscribes.
fn handle_subscribe(state: &Mutex<State>, id: u64, streams: u32) {
    let on_bus = streams != 0 && look_for_tracker(state);
    let mut st = lock_state(state);
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

/// Fan-out loop: every 8 ms flush each client's outbox, then drain the
/// engine, encode each sample once and write it to every client subscribed
/// to that stream. The buffers live across ticks so the per-frame path does
/// not reallocate.
fn pump(state: &Mutex<State>) {
    let mut samples: Vec<Sample> = Vec::new();
    let mut frames: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut dead: Vec<u64> = Vec::new();
    loop {
        samples.clear();
        frames.clear();
        dead.clear();
        {
            let mut st = lock_state(state);
            if let Some(engine) = st.engine.as_mut() {
                engine.drain_into(&mut samples);
            }
            let wanted = st.wanted_mask();
            for s in &samples {
                st.observe(s);
            }
            for s in &samples {
                push_sample_frames(s, wanted, &mut frames);
            }
            for client in &mut st.clients {
                let mut ok = true;
                while ok && let Some(body) = client.outbox.pop_front() {
                    ok = write_frame(&mut client.out, &body).is_ok();
                }
                for (need, body) in &frames {
                    if !ok {
                        break;
                    }
                    if client.streams & need != 0 {
                        ok = write_frame(&mut client.out, body).is_ok();
                    }
                }
                if !ok {
                    dead.push(client.id);
                }
            }
            if !dead.is_empty() {
                // Its reader sees the end of the stream and cleans up after
                // it (a pause it holds, its calibration session), even if
                // the peer keeps the socket open without reading.
                for c in st.clients.iter().filter(|c| dead.contains(&c.id)) {
                    let _ = c.out.shutdown(Shutdown::Both);
                }
                st.clients.retain(|c| !dead.contains(&c.id));
                st.reconcile();
            }
        }
        thread::sleep(Duration::from_millis(8));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A state with one connected client (its socket's peer is dropped; the
    /// pump never runs in these tests).
    pub(crate) fn state_with_client(id: u64) -> State {
        let (out, _peer) = UnixStream::pair().expect("socket pair");
        let mut st = State::new(false);
        st.clients.push(Client {
            id,
            streams: 0,
            out,
            outbox: VecDeque::new(),
            holds_device: false,
        });
        st
    }

    pub(crate) fn outbox(st: &State, id: u64) -> Vec<Vec<u8>> {
        st.clients
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.outbox.iter().cloned().collect())
            .unwrap_or_default()
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

        handle_subscribe(&state, 1, STREAM_PRESENCE);
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
        handle_subscribe(&state, 1, STREAM_PRESENCE);

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
