//! `tobiid`: the single process that claims the Tobii device. It owns one
//! `Engine`, accepts client connections over a Unix socket, and fans out
//! samples to subscribed clients. Head pose, gaze and presence are all served
//! by that one engine (the device's 0x50e IR image stream gives head pose
//! concurrently with gaze — see `engine`); head-pose inference is switched on
//! only while some client subscribes to it.
//!
//! Log lines go through `tracing` (the `tobiid` binary installs the
//! subscriber; under systemd stderr lands in the journal).

use anyhow::{Context, Result};
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use tracing::{info, warn};

use tobii_ipc::{
    self, PRESENCE_AWAY, PRESENCE_PRESENT, STREAM_GAZE, STREAM_HEAD, STREAM_PRESENCE,
    decode_subscribe, encode_gaze, encode_head, encode_presence, encode_subscribed, read_frame,
    write_frame,
};
use tobii_usb::engine::{Engine, Sample};

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

struct Client {
    id: u64,
    streams: u8,
    out: UnixStream,
}

struct State {
    engine: Option<Engine>,
    clients: Vec<Client>,
    /// Keep the engine running even with no clients, so the device stays warm
    /// and client connects are instant (`TOBII_PREWARM`).
    prewarm: bool,
}

impl State {
    /// True if some client consumes a stream, or pre-warm is set.
    fn is_engine_wanted(&self) -> bool {
        self.prewarm || self.clients.iter().any(|c| c.streams != 0)
    }

    /// Stop the engine once no client consumes any stream — unless pre-warm is
    /// configured, in which case keep (or restart) it. Also drops a dead engine
    /// so it can be restarted.
    fn reconcile(&mut self) {
        if self.engine.as_ref().is_some_and(|e| !e.is_alive()) {
            self.engine = None;
        }
        if self.prewarm {
            if self.engine.is_none() {
                self.engine = Some(Engine::start());
            }
        } else if self.clients.iter().all(|c| c.streams == 0) {
            self.engine = None;
        }
        self.sync_head_wanted();
    }

    /// Tell the engine whether anyone consumes head pose, so the gaze engine
    /// runs (or skips) the per-frame head-pose inference accordingly.
    fn sync_head_wanted(&self) {
        if let Some(engine) = self.engine.as_ref() {
            let wanted = self.clients.iter().any(|c| c.streams & STREAM_HEAD != 0);
            engine.set_head_wanted(wanted);
        }
    }
}

/// Lock the shared state, recovering from poisoning. Every critical section
/// here leaves `State` consistent at each statement (the engine is an
/// `Option`, clients a plain list), so a panic while holding the lock cannot
/// leave it half-updated; continuing beats taking the whole daemon down.
fn lock_state(state: &Mutex<State>) -> MutexGuard<'_, State> {
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
    let state = Arc::new(Mutex::new(State {
        engine: None,
        clients: Vec::new(),
        prewarm,
    }));

    // Pre-warm: bring the device up now (pays the cold-start lottery once) and
    // keep it streaming so later client connects are instant.
    if prewarm {
        info!("pre-warming device");
        lock_state(&state).engine = Some(Engine::start());
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
                    st.engine = Some(Engine::start());
                    st.sync_head_wanted();
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
        });
        let state = Arc::clone(state);
        thread::spawn(move || client_reader(&state, id, stream));
    }
}

/// One per connection: handle SUBSCRIBE frames and detect disconnect.
fn client_reader(state: &Mutex<State>, id: u64, mut stream: UnixStream) {
    // Loop ends on EOF or a read error.
    while let Ok(Some(body)) = read_frame(&mut stream) {
        match body.first().copied() {
            Some(tobii_ipc::TAG_SUBSCRIBE) => {
                if let Some(streams) = decode_subscribe(&body) {
                    let ok = handle_subscribe(state, id, streams);
                    // A failed reply means the client is gone; the next read
                    // ends the loop.
                    let _ = write_frame(&mut stream, &encode_subscribed(ok));
                }
            }
            Some(tobii_ipc::TAG_RECENTER) => {
                if let Some(engine) = lock_state(state).engine.as_ref() {
                    engine.request_recenter();
                }
            }
            _ => {}
        }
    }
    let mut st = lock_state(state);
    st.clients.retain(|c| c.id != id);
    st.reconcile();
}

/// Register the client's streams, starting the engine if it isn't running.
/// Always succeeds (the one engine serves every stream); `streams == 0`
/// unsubscribes.
fn handle_subscribe(state: &Mutex<State>, id: u64, streams: u8) -> bool {
    if streams == 0 {
        // Unsubscribed from everything: release the client's streams (and the
        // engine / head inference if nobody else needs them).
        let mut st = lock_state(state);
        if let Some(c) = st.clients.iter_mut().find(|c| c.id == id) {
            c.streams = 0;
        }
        st.reconcile();
        return true;
    }
    let mut st = lock_state(state);
    if st.engine.as_ref().is_none_or(|e| !e.is_alive()) {
        st.engine = Some(Engine::start());
    }
    if let Some(c) = st.clients.iter_mut().find(|c| c.id == id) {
        c.streams = streams;
    }
    st.sync_head_wanted();
    true
}

/// Fan-out loop: every 8 ms drain the engine, encode each sample once and
/// write it to every client subscribed to that stream. The buffers live
/// across ticks so the per-frame path does not reallocate.
fn pump(state: &Mutex<State>) {
    let mut samples: Vec<Sample> = Vec::new();
    let mut frames: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut dead: Vec<u64> = Vec::new();
    loop {
        samples.clear();
        frames.clear();
        dead.clear();
        {
            let mut st = lock_state(state);
            if let Some(engine) = st.engine.as_mut() {
                engine.drain_into(&mut samples);
                if !samples.is_empty() {
                    // Pre-encode frames, then write to matching clients.
                    for s in &samples {
                        push_sample_frames(s, &mut frames);
                    }
                    for client in &mut st.clients {
                        for (need, body) in &frames {
                            if client.streams & need != 0
                                && write_frame(&mut client.out, body).is_err()
                            {
                                dead.push(client.id);
                                break;
                            }
                        }
                    }
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

/// Append an engine sample's (required-stream-bit, frame-body) pairs to `out`.
// reason: the wire format is f32; the f64 -> f32 narrowing is the intended
// precision of the IPC protocol.
#[allow(clippy::cast_possible_truncation)]
fn push_sample_frames(s: &Sample, out: &mut Vec<(u8, Vec<u8>)>) {
    match s {
        Sample::Pose(p) => {
            // cm -> mm; rotation about x=pitch, y=yaw, z=roll in radians.
            let pos = [
                (p.pos_cm[0] * 10.0) as f32,
                (p.pos_cm[1] * 10.0) as f32,
                (p.pos_cm[2] * 10.0) as f32,
            ];
            let rot = [
                p.rot_deg[1].to_radians() as f32,
                p.rot_deg[0].to_radians() as f32,
                p.rot_deg[2].to_radians() as f32,
            ];
            out.push((STREAM_HEAD, encode_head(p.timestamp_us, pos, rot)));
        }
        Sample::Gaze(g) => {
            let xy = [
                ((g.gaze_norm[0] + 1.0) * 0.5).clamp(0.0, 1.0) as f32,
                ((g.gaze_norm[1] + 1.0) * 0.5).clamp(0.0, 1.0) as f32,
            ];
            let status = if g.present {
                PRESENCE_PRESENT
            } else {
                PRESENCE_AWAY
            };
            let pupil = [g.pupil_mm[0] as f32, g.pupil_mm[1] as f32];
            out.push((
                STREAM_GAZE,
                encode_gaze(g.timestamp_us, g.gaze_valid, xy, pupil),
            ));
            out.push((STREAM_PRESENCE, encode_presence(g.timestamp_us, status)));
        }
        // `Sample` is #[non_exhaustive]: a kind added by a newer engine is
        // dropped rather than breaking the wire protocol.
        _ => {}
    }
}
