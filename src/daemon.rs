//! `tobiid`: the single process that claims the Tobii device. It owns one
//! `Engine` at a time, accepts client connections over a Unix socket, arbitrates
//! the device mode (head-camera vs gaze/0x83 are mutually exclusive — see
//! `engine`), and fans out samples to all clients subscribed to the active mode.

use anyhow::{Context, Result};
use std::os::unix::io::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::engine::{Engine, EngineMode, Sample};

/// Set by the SIGUSR1 handler; a poller thread turns it into a recenter request.
static RECENTER_SIGNAL: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigusr1(_sig: libc::c_int) {
    // Async-signal-safe: only an atomic store.
    RECENTER_SIGNAL.store(true, Ordering::Relaxed);
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
    mode: Option<EngineMode>,
    clients: Vec<Client>,
    /// If set, keep an engine of this mode running even with no clients, so the
    /// device stays warm and client connects are instant (`TOBII_PREWARM`).
    prewarm: Option<EngineMode>,
}

impl State {
    /// True if some client consumes a stream, or a pre-warm mode is set.
    fn engine_wanted(&self) -> bool {
        self.prewarm.is_some() || self.clients.iter().any(|c| c.streams != 0)
    }

    /// Stop the engine once no client consumes any stream — unless a pre-warm
    /// mode is configured, in which case keep (or restart) it. Also drops a dead
    /// engine so it can be restarted.
    fn reconcile(&mut self) {
        if self.engine.as_ref().is_some_and(|e| !e.is_alive()) {
            self.engine = None;
        }
        if let Some(mode) = self.prewarm {
            if self.engine.is_none() {
                self.engine = Some(Engine::start(mode));
                self.mode = Some(mode);
            }
            return;
        }
        if self.clients.iter().all(|c| c.streams == 0) {
            self.engine = None;
            self.mode = None;
        }
    }
}

/// Pre-warm mode from `TOBII_PREWARM` (head | gaze), else none.
fn prewarm_mode() -> Option<EngineMode> {
    match std::env::var("TOBII_PREWARM").ok().as_deref() {
        Some("head" | "head-camera" | "camera") => Some(EngineMode::HeadCamera),
        Some("gaze") => Some(EngineMode::Gaze),
        _ => None,
    }
}

fn desired_mode(streams: u8) -> Option<EngineMode> {
    let head = streams & STREAM_HEAD != 0;
    let gaze = streams & (STREAM_GAZE | STREAM_PRESENCE) != 0;
    match (head, gaze) {
        (true, false) => Some(EngineMode::HeadCamera),
        (false, true) => Some(EngineMode::Gaze),
        _ => None, // nothing, or a head+gaze conflict in one client
    }
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

    let prewarm = prewarm_mode();
    let state = Arc::new(Mutex::new(State {
        engine: None,
        mode: None,
        clients: Vec::new(),
        prewarm,
    }));

    // Pre-warm: bring the device up now (pays the cold-start lottery once) and
    // keep it streaming so later client connects are instant.
    if let Some(mode) = prewarm {
        println!("pre-warming device in {mode:?} mode");
        let mut st = state.lock().unwrap();
        st.engine = Some(Engine::start(mode));
        st.mode = Some(mode);
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
                if let Some(mode) = st.prewarm.or(st.mode) {
                    if crate::device::device_present() {
                        println!("engine ({mode:?}) not running but wanted; restarting");
                        st.engine = Some(Engine::start(mode));
                        st.mode = Some(mode);
                    }
                }
            }
        });
    }

    // SIGUSR1 -> recenter the head rest pose (alternative to a client command).
    unsafe {
        libc::signal(
            libc::SIGUSR1,
            on_sigusr1 as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }
    {
        let state = state.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(150));
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

/// Arbitrate the device mode and register the client's streams. Returns false
/// if the requested mode conflicts with the one already running.
fn handle_subscribe(state: &Arc<Mutex<State>>, id: u64, streams: u8) -> bool {
    let Some(mode) = desired_mode(streams) else {
        return false;
    };
    let mut st = state.lock().unwrap();
    match st.mode {
        Some(active) if active != mode => return false, // busy with the other mode
        Some(_) => {}
        None => {
            st.engine = Some(Engine::start(mode));
            st.mode = Some(mode);
        }
    }
    if let Some(c) = st.clients.iter_mut().find(|c| c.id == id) {
        c.streams = streams;
    }
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
            vec![
                (STREAM_GAZE, encode_gaze(g.timestamp_us, g.gaze_valid, xy)),
                (STREAM_PRESENCE, encode_presence(g.timestamp_us, status)),
            ]
        }
    }
}
