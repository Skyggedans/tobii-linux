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
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
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
    pub(crate) clock: Option<(i64, i64)>,
    /// The display area a client set, re-applied at every device init.
    pub(crate) display_override: Option<DisplayArea>,
    /// `TOBII_DISPLAY_MM`: the monitor to configure once the mounting is known.
    pub(crate) display_request: Option<crate::requests::DisplaySize>,
    /// The calibration session, if any, and the active calibration id.
    pub(crate) calibration: Calibration,
    /// Stand-in for the engine's command queue in tests.
    #[cfg(test)]
    pub(crate) fake_device: Option<Arc<dyn DeviceCommands>>,
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
            display_override: None,
            display_request: crate::requests::DisplaySize::from_env(),
            calibration: Calibration::default(),
            #[cfg(test)]
            fake_device: None,
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
            crate::calibration::on_engine_lost(self);
        }
        if self.prewarm {
            self.ensure_engine();
        } else if !self.is_engine_wanted() {
            self.engine = None;
            crate::calibration::on_engine_lost(self);
        }
        self.sync_wanted();
    }

    /// Start the engine if it is not running.
    pub(crate) fn ensure_engine(&mut self) {
        if self.engine.as_ref().is_none_or(|e| !e.is_alive()) {
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

    /// Fold a sample into the daemon's own state.
    fn observe(&mut self, sample: &Sample) {
        match sample {
            Sample::DeviceReady(facts) => {
                let mut facts = (**facts).clone();
                if let Some(area) = self.display_override {
                    facts.display_area = Some(area);
                }
                if let Some(id) = facts.calibration_id {
                    self.calibration.id = Some(id);
                }
                self.facts = Some(Arc::new(facts));
                crate::requests::apply_display_request(self);
            }
            Sample::Presence(p) => self.last_presence = Some(*p),
            Sample::Gaze(g) => {
                let device = i64::try_from(g.frame.device_ts_us).unwrap_or(i64::MAX);
                self.clock = Some((device, g.host_rx_us));
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
            _ => {}
        }
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
                let running = st.engine.as_ref().is_some_and(Engine::is_alive);
                // Don't spin (reloading the model) on an unplugged device.
                if !running && st.is_engine_wanted() && tobii_usb::device::is_device_present() {
                    warn!("engine not running but wanted; restarting");
                    crate::calibration::on_engine_lost(&mut st);
                    st.engine = None;
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
    crate::calibration::on_client_gone(state, id);
    let mut st = lock_state(state);
    st.clients.retain(|c| c.id != id);
    st.reconcile();
}

/// Register the client's streams, starting the engine if it isn't running,
/// and acknowledge. Always succeeds (the one engine serves every stream);
/// `streams == 0` unsubscribes. A new presence subscriber is told the current
/// presence straight away, since the device reports it only on change.
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
    if streams & !before & STREAM_PRESENCE != 0
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
    fn a_new_presence_subscriber_gets_the_last_state() {
        let state = Mutex::new(state_with_client(1));
        {
            let mut st = lock_state(&state);
            st.observe(&Sample::Presence(PresenceSample::new(77, true)));
            // No engine in tests: keep reconcile/ensure_engine from starting one.
            st.prewarm = false;
        }
        // Subscribe without touching the engine: set the mask directly and
        // run the presence replay the way handle_subscribe does.
        let mut st = lock_state(&state);
        if let Some(c) = st.clients.iter_mut().find(|c| c.id == 1) {
            c.streams = STREAM_PRESENCE;
        }
        let p = st.last_presence.expect("cached");
        st.send_to(1, presence_frame(&p));
        let sent = outbox(&st, 1);
        assert_eq!(
            tobii_ipc::decode_server(&sent[0]),
            Some(tobii_ipc::ServerMsg::Presence {
                ts_us: 77,
                status: tobii_ipc::PRESENCE_PRESENT
            })
        );
    }
}
