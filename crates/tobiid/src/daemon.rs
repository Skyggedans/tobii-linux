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
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
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
    /// The last presence change, replayed to each new presence subscriber
    /// (the device reports presence only when it changes).
    last_presence: Option<PresenceSample>,
    /// The newest `(device_us, host_us)` pair seen on the gaze stream.
    /// Cleared whenever the engine stops or starts: the device clock
    /// restarts with the device, so a pair is good for its session only.
    pub(crate) clock: Option<(i64, i64)>,
    /// Gaze frames seen so far. A TIMESYNC waits for this to move, which,
    /// unlike the wall clock, never steps back.
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
    pub(crate) engine_losses: u64,
    /// Stand-in for the engine's command queue in tests.
    #[cfg(test)]
    pub(crate) fake_device: Option<Arc<dyn DeviceCommands>>,
    /// The pause hint last given to the engine, in tests.
    #[cfg(test)]
    pub(crate) pause_hint: Option<bool>,
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
            #[cfg(test)]
            fake_device: None,
            #[cfg(test)]
            pause_hint: None,
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

    /// Stop the engine once nobody needs it — unless pre-warm is configured,
    /// in which case keep (or restart) it. Also drops a dead engine so it can
    /// be restarted.
    fn reconcile(&mut self) {
        if self.engine.as_ref().is_some_and(|e| !e.is_alive()) {
            self.engine = None;
            self.clock = None;
            crate::calibration::on_engine_lost(self);
            crate::pause::on_engine_lost(self);
        }
        if self.prewarm {
            self.ensure_engine();
        } else if !self.is_engine_wanted() {
            self.engine = None;
            self.clock = None;
            crate::calibration::on_engine_lost(self);
            crate::pause::on_engine_lost(self);
        }
        self.sync_wanted();
    }

    /// Start the engine if it is not running. A dead engine it replaces
    /// took the pause with it.
    pub(crate) fn ensure_engine(&mut self) {
        if self.engine.as_ref().is_none_or(|e| !e.is_alive()) {
            self.clock = None;
            crate::pause::on_engine_lost(self);
            self.engine = Some(Engine::start_with(self.display_override));
            self.sync_wanted();
        }
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

    /// The device's command queue, starting the engine if needed and pinning
    /// it to `client`.
    pub(crate) fn device_for(&mut self, client: u64) -> Option<Arc<dyn DeviceCommands>> {
        if let Some(c) = self.clients.iter_mut().find(|c| c.id == client) {
            c.holds_device = true;
        }
        #[cfg(test)]
        if let Some(fake) = &self.fake_device {
            return Some(Arc::clone(fake));
        }
        self.ensure_engine();
        self.engine
            .as_ref()
            .map(|e| Arc::new(e.commands()) as Arc<dyn DeviceCommands>)
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

    /// Note the `(device_us, host_us)` pair of a gaze frame.
    pub(crate) fn note_clock(&mut self, device_us: i64, host_us: i64) {
        self.clock = Some((device_us, host_us));
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
            _ => {}
        }
    }
}

/// Fill in, from the previous init's facts, what a new init did not report
/// because a response was lost: the stream catalogue and the hardware
/// configuration are the firmware's, so the old ones still hold, and a lost
/// 1200 or 2120 must not blank them.
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

/// Use the socket passed by systemd socket activation (fd 3) if present,
/// otherwise bind our own. The `sd_listen_fds(3)` protocol: `LISTEN_PID` must
/// equal our pid and `LISTEN_FDS` >= 1, with the first socket at fd 3.
fn obtain_listener() -> Result<UnixListener> {
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
        return Ok(listener);
    }

    let path = tobii_ipc::socket_path();
    // A stale socket file from a previous run is expected; a missing one is fine.
    let _ = std::fs::remove_file(&path);
    let listener =
        UnixListener::bind(&path).with_context(|| format!("failed to bind {}", path.display()))?;
    info!(path = %path.display(), "listening");
    Ok(listener)
}

/// Run the daemon: bind (or adopt) the listening socket, install the signal
/// handlers, start the pump/watchdog threads and serve clients until a
/// shutdown signal exits the process.
///
/// # Errors
///
/// Returns an error if the Unix socket cannot be bound.
pub fn run() -> Result<()> {
    let listener = obtain_listener()?;

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
        lock_state(&state).ensure_engine();
    }

    // Watchdog: if the device thread died (a cold start that exhausted its
    // internal retries, an unplug, etc.) but it's still wanted, restart it.
    {
        let state = Arc::clone(&state);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(3));
                let mut st = lock_state(&state);
                // Dropped even while unplugged, so that clients hear of it.
                if st.engine.as_ref().is_some_and(|e| !e.is_alive()) {
                    crate::calibration::on_engine_lost(&mut st);
                    crate::pause::on_engine_lost(&mut st);
                    st.engine = None;
                    st.clock = None;
                }
                // Don't spin (reloading the model) on an unplugged device.
                if st.engine.is_none()
                    && st.is_engine_wanted()
                    && tobii_usb::device::is_device_present()
                {
                    warn!("engine not running but wanted; restarting");
                    st.ensure_engine();
                }
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
                    // 0x83 stream), then leave a clean socket and exit.
                    let mut st = lock_state(&state);
                    st.prewarm = false; // don't let reconcile/watchdog respawn it
                    st.engine = None; // blocks until the engine thread + teardown finish
                    drop(st);
                    // Under socket activation the file is systemd's; elsewhere it
                    // may already be gone. Either way there is nothing to do.
                    let _ = std::fs::remove_file(tobii_ipc::socket_path());
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

/// Register the client's streams, starting the engine if it isn't running,
/// and acknowledge. Always succeeds (the one engine serves every stream);
/// `streams == 0` unsubscribes.
fn handle_subscribe(state: &Mutex<State>, id: u64, streams: u32) {
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
        st.ensure_engine();
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
            let display = st.facts.as_ref().and_then(|f| f.display_area);
            for s in &samples {
                push_sample_frames(s, wanted, display.as_ref(), &mut frames);
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
        st.clock = Some((5_000_000, 1_700_000_000_000_000));
        // Nobody wants the engine (and prewarm is off): no engine starts.
        st.reconcile();
        assert_eq!(st.clock, None);
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
    fn a_new_presence_subscriber_gets_the_last_state() {
        let state = Mutex::new(state_with_client(1));
        {
            let mut st = lock_state(&state);
            st.observe(&Sample::Presence(PresenceSample::new(77, true)));
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
                ts_us: 77,
                status: tobii_ipc::PRESENCE_PRESENT
            })
        );
    }

    #[test]
    fn a_paused_device_replays_no_presence() {
        let mut st = state_with_client(1);
        st.observe(&Sample::Presence(PresenceSample::new(77, true)));
        st.paused = true;

        st.observe(&Sample::Presence(PresenceSample::new(88, false)));
        replay_presence(&mut st, 1, 0, STREAM_PRESENCE);

        assert!(outbox(&st, 1).is_empty());
        st.paused = false;
        replay_presence(&mut st, 1, 0, STREAM_PRESENCE);
        assert_eq!(
            tobii_ipc::decode_server(&outbox(&st, 1)[0]),
            Some(tobii_ipc::ServerMsg::Presence {
                ts_us: 77,
                status: tobii_ipc::PRESENCE_PRESENT
            }),
            "the presence from before the pause is kept"
        );
    }
}
