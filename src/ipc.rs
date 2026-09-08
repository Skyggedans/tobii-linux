//! Tiny dependency-free IPC between the daemon (`tobiid`), thin executables,
//! and the `libtobii.so` client. Length-prefixed binary frames over a Unix
//! domain socket. One frame = `u32 LE length` + `1 byte tag` + body.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

// Stream subscription bits (client -> daemon), OR-ed into the SUBSCRIBE body.

/// Subscribe to head pose ([`TAG_HEAD`] frames).
pub const STREAM_HEAD: u8 = 1 << 0;
/// Subscribe to gaze points ([`TAG_GAZE`] frames).
pub const STREAM_GAZE: u8 = 1 << 1;
/// Subscribe to user presence ([`TAG_PRESENCE`] frames).
pub const STREAM_PRESENCE: u8 = 1 << 2;

// Frame tags (first body byte).

/// Client -> daemon: `u8 streams` bitmask (`STREAM_*`); `0` unsubscribes.
pub const TAG_SUBSCRIBE: u8 = 0x01;
/// Client -> daemon: reset the head rest pose. No payload.
pub const TAG_RECENTER: u8 = 0x02;
/// Daemon -> client: `u8` ok(1) / busy(0), in reply to a SUBSCRIBE.
pub const TAG_SUBSCRIBED: u8 = 0x10;
/// Daemon -> client: `i64 ts_us`, `3 x f32` position (mm), `3 x f32` rotation (rad).
pub const TAG_HEAD: u8 = 0x20;
/// Daemon -> client: `i64 ts_us`, `u8 valid`, `2 x f32` xy (0..1), then an
/// optional `2 x f32` pupil-diameter tail (mm, left/right).
pub const TAG_GAZE: u8 = 0x21;
/// Daemon -> client: `i64 ts_us`, `u8 status` (`PRESENCE_*`).
pub const TAG_PRESENCE: u8 = 0x22;

// Presence status values carried by TAG_PRESENCE (Stream-Engine numbering).

/// No user in front of the tracker.
pub const PRESENCE_AWAY: u8 = 1;
/// A user is in front of the tracker.
pub const PRESENCE_PRESENT: u8 = 2;

/// Where the daemon listens. Per-user under the XDG runtime dir, else /tmp.
#[must_use]
pub fn socket_path() -> PathBuf {
    match std::env::var("XDG_RUNTIME_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir).join("tobiid.sock"),
        _ => PathBuf::from("/tmp/tobiid.sock"),
    }
}

/// Connect to a running daemon.
///
/// # Errors
///
/// Returns the socket error if no daemon is listening at [`socket_path`].
pub fn connect() -> io::Result<UnixStream> {
    UnixStream::connect(socket_path())
}

/// Connect, spawning `tobiid` (sibling of the current exe, else from PATH) if no
/// daemon is listening yet, then retrying for a few seconds.
///
/// # Errors
///
/// Returns the last connect error if the daemon could not be reached within
/// the retry window (about three seconds).
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
        .chain(std::iter::once(PathBuf::from("tobiid")));
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ServerMsg {
    /// Reply to a SUBSCRIBE frame.
    Subscribed {
        /// `true` if the subscription was accepted; `false` if the daemon is busy.
        ok: bool,
    },
    /// Head pose.
    Head {
        /// Device timestamp, microseconds.
        ts_us: i64,
        /// Translation `[x, y, z]` in millimetres.
        pos_mm: [f32; 3],
        /// Rotation `[pitch, yaw, roll]` in radians (Stream-Engine axis order).
        rot_rad: [f32; 3],
    },
    /// Gaze point.
    Gaze {
        /// Device timestamp, microseconds.
        ts_us: i64,
        /// Whether `xy` holds a usable gaze point.
        valid: bool,
        /// Gaze point in normalised screen coordinates, `0..1` per axis.
        xy: [f32; 2],
        /// Pupil diameter `[left, right]` in millimetres; `NaN` when the daemon
        /// did not send the optional tail.
        pupil_mm: [f32; 2],
    },
    /// User presence.
    Presence {
        /// Device timestamp, microseconds.
        ts_us: i64,
        /// One of [`PRESENCE_AWAY`] / [`PRESENCE_PRESENT`].
        status: u8,
    },
}

/// Write one length-prefixed frame (`u32 LE length` + `body`) and flush.
///
/// # Errors
///
/// Returns `InvalidInput` if `body` exceeds `u32::MAX` bytes, else any write
/// or flush error from `w`.
pub fn write_frame(w: &mut impl Write, body: &[u8]) -> io::Result<()> {
    let len = u32::try_from(body.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "frame body too large"))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// Read one length-prefixed frame, or `None` on clean EOF.
///
/// # Errors
///
/// Returns any read error from `r`, including an `UnexpectedEof` inside a
/// partially received frame.
pub fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = usize::try_from(u32::from_le_bytes(len))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame length exceeds usize"))?;
    let mut body = vec![0u8; n];
    r.read_exact(&mut body)?;
    Ok(Some(body))
}

/// Client -> daemon SUBSCRIBE body for the given `STREAM_*` bitmask.
#[must_use]
pub fn encode_subscribe(streams: u8) -> Vec<u8> {
    vec![TAG_SUBSCRIBE, streams]
}

/// Client -> daemon RECENTER body.
#[must_use]
pub fn encode_recenter() -> Vec<u8> {
    vec![TAG_RECENTER]
}

/// Daemon -> client SUBSCRIBED reply body.
#[must_use]
pub fn encode_subscribed(ok: bool) -> Vec<u8> {
    vec![TAG_SUBSCRIBED, u8::from(ok)]
}

/// Daemon -> client HEAD body: position in mm, rotation in radians.
#[must_use]
pub fn encode_head(ts_us: i64, pos_mm: [f32; 3], rot_rad: [f32; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(33);
    v.push(TAG_HEAD);
    v.extend_from_slice(&ts_us.to_le_bytes());
    for x in pos_mm.iter().chain(rot_rad.iter()) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// Daemon -> client GAZE body, including the pupil tail.
#[must_use]
pub fn encode_gaze(ts_us: i64, valid: bool, xy: [f32; 2], pupil_mm: [f32; 2]) -> Vec<u8> {
    let mut v = Vec::with_capacity(26);
    v.push(TAG_GAZE);
    v.extend_from_slice(&ts_us.to_le_bytes());
    v.push(u8::from(valid));
    for x in xy.iter().chain(pupil_mm.iter()) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// Daemon -> client PRESENCE body (`status` is a `PRESENCE_*` value).
#[must_use]
pub fn encode_presence(ts_us: i64, status: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(10);
    v.push(TAG_PRESENCE);
    v.extend_from_slice(&ts_us.to_le_bytes());
    v.push(status);
    v
}

/// Little-endian `i64` at byte offset `o`, or `None` if `b` is too short.
fn rd_i64(b: &[u8], o: usize) -> Option<i64> {
    let bytes: [u8; 8] = b.get(o..o.checked_add(8)?)?.try_into().ok()?;
    Some(i64::from_le_bytes(bytes))
}

/// Little-endian `f32` at byte offset `o`, or `None` if `b` is too short.
fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    let bytes: [u8; 4] = b.get(o..o.checked_add(4)?)?.try_into().ok()?;
    Some(f32::from_le_bytes(bytes))
}

/// Decode a daemon -> client frame body. `None` for an unknown tag or a body
/// too short for its tag.
#[must_use]
pub fn decode_server(body: &[u8]) -> Option<ServerMsg> {
    match *body.first()? {
        TAG_SUBSCRIBED => Some(ServerMsg::Subscribed {
            ok: *body.get(1)? != 0,
        }),
        TAG_HEAD => Some(ServerMsg::Head {
            ts_us: rd_i64(body, 1)?,
            pos_mm: [rd_f32(body, 9)?, rd_f32(body, 13)?, rd_f32(body, 17)?],
            rot_rad: [rd_f32(body, 21)?, rd_f32(body, 25)?, rd_f32(body, 29)?],
        }),
        TAG_GAZE => Some(ServerMsg::Gaze {
            ts_us: rd_i64(body, 1)?,
            valid: *body.get(9)? != 0,
            xy: [rd_f32(body, 10)?, rd_f32(body, 14)?],
            // Pupil is an optional tail: older daemons omit it (18-byte frame).
            pupil_mm: match (rd_f32(body, 18), rd_f32(body, 22)) {
                (Some(l), Some(r)) => [l, r],
                _ => [f32::NAN, f32::NAN],
            },
        }),
        TAG_PRESENCE => Some(ServerMsg::Presence {
            ts_us: rd_i64(body, 1)?,
            status: *body.get(9)?,
        }),
        _ => None,
    }
}

/// Decode the streams bitmask from a client SUBSCRIBE frame body.
#[must_use]
pub fn decode_subscribe(body: &[u8]) -> Option<u8> {
    match body {
        [TAG_SUBSCRIBE, streams, ..] => Some(*streams),
        _ => None,
    }
}

#[cfg(test)]
// reason: err-no-unwrap-prod exempts test code; a failed unwrap here is the test failing.
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // reason: encode/decode is a bit-exact round trip
    fn gaze_pupil_round_trips() {
        let body = encode_gaze(123, true, [0.25, -0.5], [3.42, 3.35]);
        assert_eq!(body.len(), 26);
        match decode_server(&body) {
            Some(ServerMsg::Gaze {
                ts_us,
                valid,
                xy,
                pupil_mm,
            }) => {
                assert_eq!(ts_us, 123);
                assert!(valid);
                assert_eq!(xy, [0.25, -0.5]);
                assert_eq!(pupil_mm, [3.42, 3.35]);
            }
            _ => panic!("expected Gaze"),
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: encode/decode is a bit-exact round trip
    fn legacy_gaze_frame_without_pupil_decodes_to_nan() {
        // An 18-byte frame from an older daemon: base fields still decode, pupil absent.
        let mut body = encode_gaze(7, false, [0.1, 0.2], [f32::NAN, f32::NAN]);
        body.truncate(18);
        match decode_server(&body) {
            Some(ServerMsg::Gaze {
                ts_us,
                xy,
                pupil_mm,
                ..
            }) => {
                assert_eq!(ts_us, 7);
                assert_eq!(xy, [0.1, 0.2]);
                assert!(pupil_mm[0].is_nan() && pupil_mm[1].is_nan());
            }
            _ => panic!("expected Gaze"),
        }
    }

    #[test]
    fn head_round_trips_and_short_body_is_rejected() {
        let body = encode_head(9, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
        assert_eq!(body.len(), 33);
        assert_eq!(
            decode_server(&body),
            Some(ServerMsg::Head {
                ts_us: 9,
                pos_mm: [1.0, 2.0, 3.0],
                rot_rad: [0.1, 0.2, 0.3],
            })
        );
        assert_eq!(decode_server(&body[..32]), None);
    }

    #[test]
    fn presence_and_subscribed_round_trip() {
        assert_eq!(
            decode_server(&encode_presence(5, PRESENCE_PRESENT)),
            Some(ServerMsg::Presence {
                ts_us: 5,
                status: PRESENCE_PRESENT
            })
        );
        assert_eq!(
            decode_server(&encode_subscribed(true)),
            Some(ServerMsg::Subscribed { ok: true })
        );
        assert_eq!(decode_server(&[TAG_SUBSCRIBED]), None);
        assert_eq!(decode_server(&[]), None);
        assert_eq!(decode_server(&[0xff, 1, 2]), None);
    }

    #[test]
    fn subscribe_frame_round_trips() {
        assert_eq!(
            decode_subscribe(&encode_subscribe(STREAM_HEAD | STREAM_GAZE)),
            Some(3)
        );
        assert_eq!(decode_subscribe(&[TAG_SUBSCRIBE]), None);
        assert_eq!(decode_subscribe(&[TAG_RECENTER, 1]), None);
    }

    #[test]
    fn frames_round_trip_over_a_buffer() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &encode_recenter()).unwrap();
        write_frame(&mut buf, &encode_subscribe(STREAM_PRESENCE)).unwrap();
        let mut cursor = io::Cursor::new(buf);
        assert_eq!(read_frame(&mut cursor).unwrap(), Some(vec![TAG_RECENTER]));
        assert_eq!(
            read_frame(&mut cursor).unwrap(),
            Some(vec![TAG_SUBSCRIBE, STREAM_PRESENCE])
        );
        assert_eq!(read_frame(&mut cursor).unwrap(), None);
    }
}
