//! `tobiid`: the single process that claims the Tobii device. It owns one
//! `Engine`, accepts client connections over a Unix socket, and fans out
//! samples to subscribed clients. Head pose, gaze and presence are all served
//! by that one engine (the device's 0x50e IR image stream gives head pose
//! concurrently with gaze — see `engine`); head-pose inference is switched on
//! only while some client subscribes to it.

use anyhow::{Context, Result};
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::engine::{Engine, Sample};

/// Set by the SIGUSR1 handler; a poller thread turns it into a recenter request.
static RECENTER_SIGNAL: AtomicBool = AtomicBool::new(false);

/// Set by the SIGTERM/SIGINT handler; a poller thread turns it into a graceful
/// shutdown that drops the engine (running the device teardown) before exiting.
static SHUTDOWN_SIGNAL: AtomicBool = AtomicBool::new(false);

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
use crate::ipc::{
    self, decode_subscribe, encode_gaze, encode_head, encode_presence, encode_subscribed,
    read_frame, write_frame, STREAM_GAZE, STREAM_HEAD, STREAM_PRESENCE,
};

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
    fn engine_wanted(&self) -> bool {
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

/// `TOBII_PREWARM` (any of `1`, `head`, `gaze`; the historical mode names are
/// accepted since one engine now serves everything): start the engine at
/// daemon start and keep it warm.
fn prewarm_enabled() -> bool {
    matches!(
        std::env::var("TOBII_PREWARM").ok().as_deref(),
        Some("1" | "head" | "head-camera" | "camera" | "gaze" | "yes" | "true")
    )
}

/// Use the socket passed by systemd socket activation (fd 3) if present,
/// otherwise bind our own. The sd_listen_fds(3) protocol: `LISTEN_PID` must
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
        // SD_LISTEN_FDS_START = 3.
        let listener = unsafe { UnixListener::from_raw_fd(3) };
        println!("tobiid: using systemd socket activation (fd 3)");
        return Ok(listener);
    }

    let path = ipc::socket_path();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)
        .with_context(|| format!("failed to bind {}", path.display()))?;
    println!("tobiid listening on {}", path.display());
    Ok(listener)
}

pub fn run() -> Result<()> {
    let listener = obtain_listener()?;

    let prewarm = prewarm_enabled();
    let state = Arc::new(Mutex::new(State {
        engine: None,
        clients: Vec::new(),
        prewarm,
    }));

    // Pre-warm: bring the device up now (pays the cold-start lottery once) and
    // keep it streaming so later client connects are instant.
    if prewarm {
        println!("pre-warming device");
        let mut st = state.lock().unwrap();
        st.engine = Some(Engine::start());
    }

    // Pump thread: drain the active engine and fan samples out to clients.
    {
        let state = state.clone();
        thread::spawn(move || pump(state));
    }

    // Watchdog: if the device thread died (a cold start that exhausted its
    // internal retries, an unplug, etc.) but it's still wanted, restart it.
    {
        let state = state.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(3));
            let mut st = state.lock().unwrap();
            let dead = st.engine.as_ref().is_some_and(|e| !e.is_alive());
            if (dead || st.engine.is_none()) && st.engine_wanted() {
                // Don't spin (reloading the model) on an unplugged device.
                if crate::device::device_present() {
                    println!("engine not running but wanted; restarting");
                    st.engine = Some(Engine::start());
                    st.sync_head_wanted();
                }
            }
        });
    }

    // SIGUSR1 -> recenter; SIGTERM/SIGINT -> graceful shutdown with device teardown.
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
        let state = state.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(150));
            if SHUTDOWN_SIGNAL.load(Ordering::Relaxed) {
                // Drop the engine so its Drop runs the device teardown (stop the
                // 0x83 stream), then leave a clean socket and exit.
                let mut st = state.lock().unwrap();
                st.prewarm = false; // don't let reconcile/watchdog respawn it
                st.engine = None; // blocks until the engine thread + teardown finish
                drop(st);
                let _ = std::fs::remove_file(ipc::socket_path());
                println!("tobiid: shutdown signal -> device teardown done, exiting");
                std::process::exit(0);
            }
            if RECENTER_SIGNAL.swap(false, Ordering::Relaxed) {
                if let Some(engine) = state.lock().unwrap().engine.as_ref() {
                    engine.request_recenter();
                }
            }
        });
    }

    let ids = AtomicU64::new(1);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let out = match stream.try_clone() {
            Ok(o) => o,
            Err(_) => continue,
        };
        // Bound how long pump() can sit in write_frame() with `state` locked:
        // a client that stops reading is dropped instead of wedging fan-out
        // (and the SIGTERM teardown, which needs the same lock).
        let _ = out.set_write_timeout(Some(Duration::from_millis(250)));
        let id = ids.fetch_add(1, Ordering::Relaxed);
        state.lock().unwrap().clients.push(Client {
            id,
            streams: 0,
            out,
        });
        let state = state.clone();
        thread::spawn(move || client_reader(state, id, stream));
    }
    Ok(())
}

/// One per connection: handle SUBSCRIBE frames and detect disconnect.
fn client_reader(state: Arc<Mutex<State>>, id: u64, mut stream: UnixStream) {
    loop {
        match read_frame(&mut stream) {
            Ok(Some(body)) => match body.first().copied() {
                Some(ipc::TAG_SUBSCRIBE) => {
                    if let Some(streams) = decode_subscribe(&body) {
                        let ok = handle_subscribe(&state, id, streams);
                        let _ = write_frame(&mut stream, &encode_subscribed(ok));
                    }
                }
                Some(ipc::TAG_RECENTER) => {
                    if let Some(engine) = state.lock().unwrap().engine.as_ref() {
                        engine.request_recenter();
                    }
                }
                _ => {}
            },
            _ => break, // EOF or error
        }
    }
    let mut st = state.lock().unwrap();
    st.clients.retain(|c| c.id != id);
    st.reconcile();
}

/// Register the client's streams, starting the engine if it isn't running.
/// Always succeeds (the one engine serves every stream); `streams == 0`
/// unsubscribes.
fn handle_subscribe(state: &Arc<Mutex<State>>, id: u64, streams: u8) -> bool {
    if streams == 0 {
        // Unsubscribed from everything: release the client's streams (and the
        // engine / head inference if nobody else needs them).
        let mut st = state.lock().unwrap();
        if let Some(c) = st.clients.iter_mut().find(|c| c.id == id) {
            c.streams = 0;
        }
        st.reconcile();
        return true;
    }
    let mut st = state.lock().unwrap();
    if st.engine.as_ref().is_none_or(|e| !e.is_alive()) {
        st.engine = Some(Engine::start());
    }
    if let Some(c) = st.clients.iter_mut().find(|c| c.id == id) {
        c.streams = streams;
    }
    st.sync_head_wanted();
    true
}

fn pump(state: Arc<Mutex<State>>) {
    loop {
        let mut dead = Vec::new();
        {
            let mut st = state.lock().unwrap();
            if let Some(engine) = st.engine.as_mut() {
                let samples = engine.drain();
                if !samples.is_empty() {
                    // Pre-encode frames, then write to matching clients.
                    let frames: Vec<(u8, Vec<u8>)> = samples
                        .iter()
                        .flat_map(|s| sample_frames(s))
                        .collect();
                    for client in st.clients.iter_mut() {
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

/// Convert an engine sample into (required-stream-bit, frame-body) pairs.
fn sample_frames(s: &Sample) -> Vec<(u8, Vec<u8>)> {
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
            vec![(STREAM_HEAD, encode_head(p.timestamp_us, pos, rot))]
        }
        Sample::Gaze(g) => {
            let xy = [
                ((g.gaze_norm[0] + 1.0) * 0.5).clamp(0.0, 1.0) as f32,
                ((g.gaze_norm[1] + 1.0) * 0.5).clamp(0.0, 1.0) as f32,
            ];
            let status = if g.present { 2u8 } else { 1u8 };
            let pupil = [g.pupil_mm[0] as f32, g.pupil_mm[1] as f32];
            vec![
                (STREAM_GAZE, encode_gaze(g.timestamp_us, g.gaze_valid, xy, pupil)),
                (STREAM_PRESENCE, encode_presence(g.timestamp_us, status)),
            ]
        }
    }
}
