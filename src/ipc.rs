//! Tiny dependency-free IPC between the daemon (`tobiid`), thin executables,
//! and the `libtobii.so` client. Length-prefixed binary frames over a Unix
//! domain socket. One frame = `u32 LE length` + `1 byte tag` + body.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

// Stream subscription bits (client -> daemon).
pub const STREAM_HEAD: u8 = 1 << 0;
pub const STREAM_GAZE: u8 = 1 << 1;
pub const STREAM_PRESENCE: u8 = 1 << 2;

// Frame tags.
pub const TAG_SUBSCRIBE: u8 = 0x01; // client -> daemon: u8 streams
pub const TAG_SUBSCRIBED: u8 = 0x10; // daemon -> client: u8 ok(1)/busy(0)
pub const TAG_HEAD: u8 = 0x20; // i64 ts, 3xf32 pos(mm), 3xf32 rot(rad)
pub const TAG_GAZE: u8 = 0x21; // i64 ts, u8 valid, 2xf32 xy(0..1)
pub const TAG_PRESENCE: u8 = 0x22; // i64 ts, u8 status

/// Where the daemon listens. Per-user under the XDG runtime dir, else /tmp.
pub fn socket_path() -> PathBuf {
    match std::env::var("XDG_RUNTIME_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir).join("tobiid.sock"),
        _ => PathBuf::from("/tmp/tobiid.sock"),
    }
}

/// Connect to a running daemon.
pub fn connect() -> io::Result<UnixStream> {
    UnixStream::connect(socket_path())
}

/// Connect, spawning `tobiid` (sibling of the current exe, else from PATH) if no
/// daemon is listening yet, then retrying for a few seconds.
pub fn connect_or_spawn() -> io::Result<UnixStream> {
    if let Ok(s) = connect() {
        return Ok(s);
    }
    spawn_daemon();
    for _ in 0..60 {
        std::thread::sleep(Duration::from_millis(50));
        if let Ok(s) = connect() {
            return Ok(s);
        }
    }
    connect()
}

fn spawn_daemon() {
    use std::process::{Command, Stdio};
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("tobiid")));
    let candidates = sibling
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .chain(std::iter::once("tobiid".to_string()));
    for cmd in candidates {
        // Detach stdio so the daemon never holds the client's pipes open.
        if Command::new(&cmd)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok()
        {
            return;
        }
    }
}

/// A decoded sample as delivered by the daemon to a client.
pub enum ServerMsg {
    Subscribed { ok: bool },
    Head { ts_us: i64, pos_mm: [f32; 3], rot_rad: [f32; 3] },
    Gaze { ts_us: i64, valid: bool, xy: [f32; 2] },
    Presence { ts_us: i64, status: u8 },
}

pub fn write_frame(w: &mut impl Write, body: &[u8]) -> io::Result<()> {
    w.write_all(&(body.len() as u32).to_le_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// Read one length-prefixed frame, or `None` on clean EOF.
pub fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_le_bytes(len) as usize;
    let mut body = vec![0u8; n];
    r.read_exact(&mut body)?;
    Ok(Some(body))
}

pub fn encode_subscribe(streams: u8) -> Vec<u8> {
    vec![TAG_SUBSCRIBE, streams]
}

pub fn encode_subscribed(ok: bool) -> Vec<u8> {
    vec![TAG_SUBSCRIBED, ok as u8]
}

pub fn encode_head(ts_us: i64, pos_mm: [f32; 3], rot_rad: [f32; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(33);
    v.push(TAG_HEAD);
    v.extend_from_slice(&ts_us.to_le_bytes());
    for x in pos_mm.iter().chain(rot_rad.iter()) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

pub fn encode_gaze(ts_us: i64, valid: bool, xy: [f32; 2]) -> Vec<u8> {
    let mut v = Vec::with_capacity(18);
    v.push(TAG_GAZE);
    v.extend_from_slice(&ts_us.to_le_bytes());
    v.push(valid as u8);
    for x in &xy {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

pub fn encode_presence(ts_us: i64, status: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(10);
    v.push(TAG_PRESENCE);
    v.extend_from_slice(&ts_us.to_le_bytes());
    v.push(status);
    v
}

fn rd_i64(b: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Decode a daemon -> client frame body.
pub fn decode_server(body: &[u8]) -> Option<ServerMsg> {
    match *body.first()? {
        TAG_SUBSCRIBED if body.len() >= 2 => Some(ServerMsg::Subscribed { ok: body[1] != 0 }),
        TAG_HEAD if body.len() >= 33 => Some(ServerMsg::Head {
            ts_us: rd_i64(body, 1),
            pos_mm: [rd_f32(body, 9), rd_f32(body, 13), rd_f32(body, 17)],
            rot_rad: [rd_f32(body, 21), rd_f32(body, 25), rd_f32(body, 29)],
        }),
        TAG_GAZE if body.len() >= 18 => Some(ServerMsg::Gaze {
            ts_us: rd_i64(body, 1),
            valid: body[9] != 0,
            xy: [rd_f32(body, 10), rd_f32(body, 14)],
        }),
        TAG_PRESENCE if body.len() >= 10 => Some(ServerMsg::Presence {
            ts_us: rd_i64(body, 1),
            status: body[9],
        }),
        _ => None,
    }
}

/// Decode the streams bitmask from a client SUBSCRIBE frame body.
pub fn decode_subscribe(body: &[u8]) -> Option<u8> {
    if body.len() >= 2 && body[0] == TAG_SUBSCRIBE {
        Some(body[1])
    } else {
        None
    }
}
