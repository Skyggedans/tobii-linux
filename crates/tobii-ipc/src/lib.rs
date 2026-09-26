//! Tiny IPC between the daemon (`tobiid`), thin executables, and the
//! `libtobii.so` client, with no dependency but libc. Length-prefixed binary
//! frames over a Unix domain socket. One frame = `u32 LE length` + `1 byte
//! tag` + body.
//!
//! | tag | direction | body |
//! |---|---|---|
//! | `0x01` SUBSCRIBE | client -> daemon | `u32 LE` stream mask (a legacy 1-byte `u8` mask is accepted) |
//! | `0x02` RECENTER | client -> daemon | none |
//! | `0x03` REQUEST | client -> daemon | `u32 id`, `u8 kind`, payload ([`request`]) |
//! | `0x10` SUBSCRIBED | daemon -> client | `u8 ok` |
//! | `0x11` REPLY | daemon -> client | `u32 id`, `u8 status`, payload |
//! | `0x20` HEAD .. `0x28` `GAZE_RAW` | daemon -> client | samples ([`ServerMsg`]) |
//!
//! The daemon runs a connection's REQUESTs one at a time, in the order they
//! came, and answers a SUBSCRIBE without waiting for the requests before it
//! to finish (only a backlog of dozens of them holds it up): a SUBSCRIBED
//! may arrive ahead of the REPLYs to earlier requests, so a client matches
//! a REPLY to its REQUEST by id. How long the daemon may take to answer a
//! REQUEST follows from the deadlines in [`deadline`].
//!
//! Sample timestamps are the host clock, [`host_clock_us`] in microseconds,
//! which is also what `tobii_system_clock` returns: the daemon maps the
//! device time each sample was taken at onto it. Its USB engine estimates
//! the offset from the smallest `receipt - device` time of the gaze frames
//! and IR images of the last 120 s, afresh at every device init. Within one
//! open the stamps of a stream strictly increase (a PRESENCE replayed to a
//! new subscriber keeps the stamp it was last sent with); the device clock
//! may restart at an init, and the stamps then run on with the host's
//! instead of jumping back. A head pose has the time of the image it was
//! made from. The device clock is left only in gaze data's tracker time, raw
//! gaze's time and the TIMESYNC pair's device time: a raw gaze sample has no
//! host time, as the Stream Engine's raw gaze record has none. These device
//! times are passed on as they are, so they go back when the device clock
//! restarts.
//!
//! The SUBSCRIBE mask is written as `u32 LE`, whose first byte is the low
//! byte of the mask: a daemon that reads only one byte still sees every
//! stream below bit 8, so old and new peers interoperate either way. Raw
//! gaze, [`STREAM_GAZE_RAW`], is bit 8, the first past that byte: a legacy
//! one-byte SUBSCRIBE (paperwm-gaze sends one) cannot ask for it. A daemon
//! from before raw gaze that reads the `u32` mask keeps the bit but never
//! sends the frame, so the stream stays silent: SUBSCRIBED says ok all the
//! same, and a mask of raw gaze alone still has it start and hold the
//! tracker. One that reads a single byte sees such a mask as 0 and
//! unsubscribes. A client from before raw gaze drops the frame as a tag it
//! does not know ([`decode_server`] returns `None`).

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

mod clock;
pub mod deadline;
pub mod geometry;
pub mod request;
mod sample;
mod wire;

pub use clock::host_clock_us;
pub use sample::{
    EyePair, EyePoint, GazeData, GazeDataEye, GazeRaw, GazeRawEye, Image, Notification,
    NotificationValue, ServerMsg, decode_server, encode_eye_position, encode_gaze,
    encode_gaze_data, encode_gaze_origin, encode_gaze_raw, encode_head, encode_image,
    encode_notification, encode_presence, encode_reply, encode_subscribed, notification,
};

// Stream subscription bits (client -> daemon), OR-ed into the SUBSCRIBE mask.

/// Subscribe to head pose ([`TAG_HEAD`] frames).
pub const STREAM_HEAD: u32 = 1 << 0;
/// Subscribe to gaze points ([`TAG_GAZE`] frames).
pub const STREAM_GAZE: u32 = 1 << 1;
/// Subscribe to user presence ([`TAG_PRESENCE`] frames, on change).
pub const STREAM_PRESENCE: u32 = 1 << 2;
/// Subscribe to gaze origins ([`TAG_GAZE_ORIGIN`] frames).
pub const STREAM_GAZE_ORIGIN: u32 = 1 << 3;
/// Subscribe to track-box-normalised eye positions ([`TAG_EYE_POSITION`]).
pub const STREAM_EYE_POSITION: u32 = 1 << 4;
/// Subscribe to per-eye gaze data ([`TAG_GAZE_DATA`] frames).
pub const STREAM_GAZE_DATA: u32 = 1 << 5;
/// Subscribe to IR camera frames ([`TAG_IMAGE`], ~2.6 MB/s).
pub const STREAM_IMAGE: u32 = 1 << 6;
/// Subscribe to device notifications ([`TAG_NOTIFICATION`] frames).
pub const STREAM_NOTIFICATIONS: u32 = 1 << 7;
/// Subscribe to raw gaze ([`TAG_GAZE_RAW`] frames), the Stream Engine's own
/// record of each gaze frame. Past the low byte: a legacy one-byte
/// SUBSCRIBE cannot carry it.
pub const STREAM_GAZE_RAW: u32 = 1 << 8;

// Frame tags (first body byte).

/// Client -> daemon: `u32 LE` stream mask (`STREAM_*`); `0` unsubscribes.
pub const TAG_SUBSCRIBE: u8 = 0x01;
/// Client -> daemon: reset the head rest pose. No payload.
pub const TAG_RECENTER: u8 = 0x02;
/// Client -> daemon: `u32 id`, `u8 kind`, payload (see [`request`]).
pub const TAG_REQUEST: u8 = 0x03;
/// Daemon -> client: `u8` ok(1) / busy(0), in reply to a SUBSCRIBE.
pub const TAG_SUBSCRIBED: u8 = 0x10;
/// Daemon -> client: `u32 id`, `u8 status`, payload, in reply to a REQUEST.
pub const TAG_REPLY: u8 = 0x11;
/// Daemon -> client: `i64 ts_us`, `3 x f32` position (mm), `3 x f32` rotation (rad).
pub const TAG_HEAD: u8 = 0x20;
/// Daemon -> client: `i64 ts_us`, `u8 valid`, `2 x f32` xy, then an optional
/// `2 x f32` pupil-diameter tail (mm, left/right, `NaN` when not valid).
pub const TAG_GAZE: u8 = 0x21;
/// Daemon -> client: `i64 ts_us`, `u8 status` (`PRESENCE_*`).
pub const TAG_PRESENCE: u8 = 0x22;
/// Daemon -> client: `i64 ts_us`, then per eye `u8 valid` + `3 x f32` (mm,
/// display frame).
pub const TAG_GAZE_ORIGIN: u8 = 0x23;
/// Daemon -> client: as [`TAG_GAZE_ORIGIN`], track-box-normalised.
pub const TAG_EYE_POSITION: u8 = 0x24;
/// Daemon -> client: a [`GazeData`] sample.
pub const TAG_GAZE_DATA: u8 = 0x25;
/// Daemon -> client: `i64 ts_us`, `u32 width`, `u32 height`, `u8 bpp`, pixels.
pub const TAG_IMAGE: u8 = 0x26;
/// Daemon -> client: `u8 type`, `u8 value_type`, value (a [`Notification`]).
pub const TAG_NOTIFICATION: u8 = 0x27;
/// Daemon -> client: a [`GazeRaw`] sample, stamped with the device clock
/// rather than the host's.
pub const TAG_GAZE_RAW: u8 = 0x28;

// Presence status values carried by TAG_PRESENCE (Stream-Engine numbering).

/// Presence has not been reported yet.
pub const PRESENCE_UNKNOWN: u8 = 0;
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
pub fn encode_subscribe(streams: u32) -> Vec<u8> {
    let mut body = Vec::with_capacity(5);
    body.push(TAG_SUBSCRIBE);
    body.extend_from_slice(&streams.to_le_bytes());
    body
}

/// Client -> daemon RECENTER body.
#[must_use]
pub fn encode_recenter() -> Vec<u8> {
    vec![TAG_RECENTER]
}

/// Decode the streams bitmask from a client SUBSCRIBE frame body: a `u32 LE`
/// mask, or the legacy single `u8`.
#[must_use]
pub fn decode_subscribe(body: &[u8]) -> Option<u32> {
    match body {
        [TAG_SUBSCRIBE, a, b, c, d, ..] => Some(u32::from_le_bytes([*a, *b, *c, *d])),
        [TAG_SUBSCRIBE, streams] => Some(u32::from(*streams)),
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
        // Plausible diameters (keys 0x06/0x0c; the session1 frame has
        // 6.247/5.997 mm), exact in f32.
        let pupil = [6.25, 6.0];
        let body = encode_gaze(123, true, [0.25, -0.5], pupil);
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
                assert_eq!(pupil_mm, pupil);
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
        let body = encode_subscribe(STREAM_HEAD | STREAM_GAZE | STREAM_NOTIFICATIONS);
        assert_eq!(body.len(), 5);
        assert_eq!(decode_subscribe(&body), Some(0x83));
        assert_eq!(decode_subscribe(&[TAG_SUBSCRIBE]), None);
        assert_eq!(decode_subscribe(&[TAG_RECENTER, 1]), None);
    }

    /// paperwm-gaze (GJS) and older clients send a single mask byte.
    #[test]
    fn legacy_one_byte_subscribe_is_accepted() {
        assert_eq!(decode_subscribe(&[TAG_SUBSCRIBE, 0x05]), Some(0x05));
    }

    /// An older daemon reads only the byte after the tag: the low byte of the
    /// new little-endian mask, which carries every stream it knows.
    #[test]
    fn new_subscribe_is_readable_by_a_one_byte_decoder() {
        let body = encode_subscribe(STREAM_HEAD | STREAM_PRESENCE);
        assert_eq!(body.get(1), Some(&0x05));
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
            Some(encode_subscribe(STREAM_PRESENCE))
        );
        assert_eq!(read_frame(&mut cursor).unwrap(), None);
    }

    fn pair() -> EyePair {
        EyePair {
            ts_us: 9_613_320_391,
            left: EyePoint {
                valid: true,
                xyz: [-57.9, 112.5, 618.3],
            },
            right: EyePoint {
                valid: false,
                xyz: [0.0; 3],
            },
        }
    }

    #[test]
    fn eye_pair_frames_round_trip() {
        assert_eq!(
            decode_server(&encode_gaze_origin(&pair())),
            Some(ServerMsg::GazeOrigin(pair()))
        );
        assert_eq!(
            decode_server(&encode_eye_position(&pair())),
            Some(ServerMsg::EyePosition(pair()))
        );
        let body = encode_gaze_origin(&pair());
        assert_eq!(decode_server(&body[..body.len() - 1]), None);
    }

    #[test]
    fn gaze_data_round_trips() {
        let eye = GazeDataEye {
            gaze_origin_valid: true,
            gaze_origin_mm: [1.0, 2.0, 3.0],
            gaze_origin_in_track_box: [0.4, 0.5, 0.6],
            gaze_point_valid: true,
            gaze_point_mm: [4.0, 5.0, 6.0],
            gaze_point_on_display: [0.25, 0.75],
            eyeball_center_valid: false,
            eyeball_center_mm: [7.0, 8.0, 9.0],
            pupil_valid: true,
            pupil_diameter_mm: 3.5,
        };
        let data = GazeData {
            timestamp_tracker_us: 1,
            timestamp_system_us: 2,
            left: eye,
            right: GazeDataEye {
                pupil_valid: false,
                ..eye
            },
        };
        assert_eq!(
            decode_server(&encode_gaze_data(&data)),
            Some(ServerMsg::GazeData(Box::new(data)))
        );
    }

    /// A raw gaze sample whose fields all differ, so a field the codec moves
    /// or swaps shows up.
    fn gaze_raw() -> GazeRaw {
        GazeRaw {
            timestamp_tracker_us: 9_613_320_391,
            left: GazeRawEye {
                gaze_origin_mm: [-28.25, 30.5, 598.75],
                gaze_origin_in_track_box: [0.5625, 0.375, 0.4375],
                gaze_point_mm: [40.5, 292.5, 99.625],
                gaze_point_on_display: [0.5623, 0.1015],
                pupil_diameter_mm: 6.247_36,
                status: 0,
            },
            right: GazeRawEye {
                gaze_origin_mm: [33.75, 31.25, 601.5],
                gaze_origin_in_track_box: [0.4375, 0.3125, 0.5],
                gaze_point_mm: [41.25, 291.75, 99.5],
                gaze_point_on_display: [0.5625, 0.103],
                pupil_diameter_mm: 5.996_61,
                status: 4,
            },
            combined_gaze_point_on_display: [0.5624, 0.1022],
            combined_gaze_validity: 1,
            key_0e: Some(9),
            key_11: Some(4),
            frame_counter: Some(43_780),
            left_origin_flag: Some(1),
            right_origin_flag: Some(3),
            left_eyeball_center_mm: Some([-30.5, 29.75, 610.25]),
            right_eyeball_center_mm: Some([31.5, 30.25, 612.0]),
        }
    }

    /// Every field of `raw` as bits, so `NaN` and `-0.0` compare by what was
    /// sent; a field the frame lacks is a clear presence word and zeros.
    fn gaze_raw_bits(raw: &GazeRaw) -> (i64, Vec<u32>) {
        let mut bits = Vec::new();
        for eye in [&raw.left, &raw.right] {
            let floats = eye
                .gaze_origin_mm
                .iter()
                .chain(&eye.gaze_origin_in_track_box)
                .chain(&eye.gaze_point_mm)
                .chain(&eye.gaze_point_on_display)
                .chain([&eye.pupil_diameter_mm]);
            bits.extend(floats.map(|v| v.to_bits()));
            bits.push(eye.status);
        }
        bits.extend(raw.combined_gaze_point_on_display.map(f32::to_bits));
        bits.push(raw.combined_gaze_validity);
        for word in [
            raw.key_0e,
            raw.key_11,
            raw.frame_counter,
            raw.left_origin_flag,
            raw.right_origin_flag,
        ] {
            bits.extend([u32::from(word.is_some()), word.unwrap_or(0)]);
        }
        for point in [raw.left_eyeball_center_mm, raw.right_eyeball_center_mm] {
            bits.push(u32::from(point.is_some()));
            bits.extend(point.unwrap_or_default().map(f32::to_bits));
        }
        (raw.timestamp_tracker_us, bits)
    }

    fn decode_gaze_raw(body: &[u8]) -> Option<GazeRaw> {
        match decode_server(body)? {
            ServerMsg::GazeRaw(raw) => Some(*raw),
            other => panic!("expected GazeRaw, got {other:?}"),
        }
    }

    /// A raw gaze sample round-trips in 176 bytes, the tag included, with
    /// each field the frame may lack told apart from one sent as 0.
    #[test]
    fn gaze_raw_round_trips() {
        let sent = gaze_raw();
        let lacking = GazeRaw {
            key_0e: None,
            key_11: None,
            frame_counter: None,
            left_origin_flag: None,
            right_origin_flag: None,
            left_eyeball_center_mm: None,
            right_eyeball_center_mm: None,
            ..sent
        };
        let zero = GazeRaw {
            key_0e: Some(0),
            key_11: Some(0),
            frame_counter: Some(0),
            left_origin_flag: Some(0),
            right_origin_flag: Some(0),
            left_eyeball_center_mm: Some([0.0; 3]),
            right_eyeball_center_mm: Some([0.0; 3]),
            ..sent
        };

        for raw in [sent, lacking, zero, GazeRaw::default()] {
            let body = encode_gaze_raw(&raw);

            assert_eq!((body.len(), body.first()), (176, Some(&TAG_GAZE_RAW)));
            assert_eq!(
                decode_server(&body),
                Some(ServerMsg::GazeRaw(Box::new(raw)))
            );
        }
        assert_ne!(encode_gaze_raw(&lacking), encode_gaze_raw(&zero));
    }

    /// The fields sit where [`encode_gaze_raw`]'s doc puts them, so a daemon
    /// and a libtobii built from different commits read one another: the
    /// round trips pass for a change made to both codecs alike, such as two
    /// flagged words swapped or a presence byte moved; this does not.
    #[test]
    fn gaze_raw_fields_sit_where_the_doc_puts_them() {
        let raw = gaze_raw();
        let body = encode_gaze_raw(&raw);
        let lacking = encode_gaze_raw(&GazeRaw {
            key_0e: None,
            right_eyeball_center_mm: None,
            ..raw
        });
        let at = |body: &[u8], from: usize, to: usize| body.get(from..to).map(<[u8]>::to_vec);
        let word = |w: u32| Some(w.to_le_bytes().to_vec());
        let floats = |vs: &[f32]| -> Option<Vec<u8>> {
            Some(vs.iter().flat_map(|v| v.to_le_bytes()).collect())
        };
        let present = |w: u32| Some([vec![1], w.to_le_bytes().to_vec()].concat());
        let (left, right) = (&raw.left, &raw.right);

        assert_eq!(
            at(&body, 1, 9),
            Some(raw.timestamp_tracker_us.to_le_bytes().to_vec())
        );
        assert_eq!(at(&body, 9, 21), floats(&left.gaze_origin_mm));
        assert_eq!(at(&body, 53, 57), floats(&[left.pupil_diameter_mm]));
        assert_eq!(at(&body, 61, 73), floats(&right.gaze_origin_mm));
        assert_eq!(at(&body, 109, 113), word(right.status));
        assert_eq!(
            at(&body, 113, 121),
            floats(&raw.combined_gaze_point_on_display)
        );
        assert_eq!(at(&body, 121, 125), word(raw.combined_gaze_validity));
        assert_eq!(at(&body, 125, 130), present(raw.key_0e.unwrap()));
        assert_eq!(at(&body, 135, 140), present(raw.frame_counter.unwrap()));
        assert_eq!(at(&body, 140, 145), present(raw.left_origin_flag.unwrap()));
        assert_eq!(body.get(150), Some(&1));
        assert_eq!(
            at(&body, 151, 163),
            floats(&raw.left_eyeball_center_mm.unwrap())
        );
        assert_eq!(body.get(163), Some(&1));
        assert_eq!(
            at(&body, 164, 176),
            floats(&raw.right_eyeball_center_mm.unwrap())
        );
        assert_eq!(at(&lacking, 125, 130), Some(vec![0; 5]));
        assert_eq!(at(&lacking, 163, 176), Some(vec![0; 13]));
    }

    /// The record filters nothing, so neither does the frame: `NaN` (with a
    /// payload, and negative), `-0.0`, the infinities, a subnormal, and the
    /// ±1 mm (±1024 wire units) the device sends for an invalid component
    /// all arrive with the bits they left with, as do the extreme words and
    /// device times. Compared by bits: `NaN != NaN`.
    #[test]
    fn gaze_raw_values_round_trip_bit_exactly() {
        let odd = [
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_0000),
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
            1.0,
            -1.0,
            f32::MAX,
        ];
        let mut next = odd.iter().copied().cycle();
        let mut eye = |status| GazeRawEye {
            gaze_origin_mm: std::array::from_fn(|_| next.next().unwrap()),
            gaze_origin_in_track_box: std::array::from_fn(|_| next.next().unwrap()),
            gaze_point_mm: std::array::from_fn(|_| next.next().unwrap()),
            gaze_point_on_display: std::array::from_fn(|_| next.next().unwrap()),
            pupil_diameter_mm: next.next().unwrap(),
            status,
        };
        let (left, right) = (eye(u32::MAX), eye(2));
        let odd_raw = |timestamp_tracker_us| GazeRaw {
            timestamp_tracker_us,
            left,
            right,
            combined_gaze_point_on_display: [f32::NAN, -0.0],
            combined_gaze_validity: u32::MAX,
            key_0e: Some(u32::MAX),
            left_eyeball_center_mm: Some([f32::NAN, -1.0, f32::from_bits(0x7fc0_1234)]),
            right_eyeball_center_mm: Some([1.0, -0.0, f32::NEG_INFINITY]),
            ..gaze_raw()
        };

        for raw in [odd_raw(i64::MIN), odd_raw(i64::MAX), odd_raw(-1)] {
            let got = decode_gaze_raw(&encode_gaze_raw(&raw)).expect("decodes");

            assert_eq!(gaze_raw_bits(&got), gaze_raw_bits(&raw));
        }
    }

    /// A body cut anywhere is dropped; one with more after it decodes, which
    /// leaves room for a tail a later daemon may add (a host time, say).
    #[test]
    fn a_short_gaze_raw_body_is_rejected_and_a_tail_is_ignored() {
        let raw = gaze_raw();
        let body = encode_gaze_raw(&raw);

        for len in 0..body.len() {
            assert_eq!(decode_server(&body[..len]), None, "cut at {len}");
        }
        let mut longer = body.clone();
        longer.extend_from_slice(&[0xa5; 9]);
        assert_eq!(decode_gaze_raw(&longer), Some(raw));
    }

    /// Raw gaze takes the first bit past the mask's low byte and the first
    /// free tag, so no older peer reads either as something else. A `u32`
    /// SUBSCRIBE carries the bit; the byte a one-byte decoder reads keeps the
    /// other streams, and a legacy one-byte SUBSCRIBE cannot ask for it.
    #[test]
    fn raw_gaze_takes_a_new_bit_and_tag() {
        let older = [
            STREAM_HEAD,
            STREAM_GAZE,
            STREAM_PRESENCE,
            STREAM_GAZE_ORIGIN,
            STREAM_EYE_POSITION,
            STREAM_GAZE_DATA,
            STREAM_IMAGE,
            STREAM_NOTIFICATIONS,
        ];
        assert_eq!(older.iter().fold(0, |mask, bit| mask | bit), 0xff);
        assert_eq!(STREAM_GAZE_RAW, 0x100);
        let tags = [
            TAG_SUBSCRIBE,
            TAG_RECENTER,
            TAG_REQUEST,
            TAG_SUBSCRIBED,
            TAG_REPLY,
            TAG_HEAD,
            TAG_GAZE,
            TAG_PRESENCE,
            TAG_GAZE_ORIGIN,
            TAG_EYE_POSITION,
            TAG_GAZE_DATA,
            TAG_IMAGE,
            TAG_NOTIFICATION,
        ];
        assert!(!tags.contains(&TAG_GAZE_RAW));

        let body = encode_subscribe(STREAM_GAZE_RAW | STREAM_GAZE);

        assert_eq!(decode_subscribe(&body), Some(0x102));
        assert_eq!(body.get(1), Some(&0x02));
        assert_eq!(
            decode_subscribe(&[TAG_SUBSCRIBE, 0xff]).map(|mask| mask & STREAM_GAZE_RAW),
            Some(0)
        );
    }

    #[test]
    fn image_round_trips() {
        let pixels: Vec<u8> = (0..=255).collect();
        let body = encode_image(42, 16, 16, 8, &pixels);
        match decode_server(&body) {
            Some(ServerMsg::Image(image)) => {
                assert_eq!((image.ts_us, image.width, image.height), (42, 16, 16));
                assert_eq!(image.bits_per_pixel, 8);
                assert_eq!(image.pixels, pixels);
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn notifications_round_trip_every_value_type() {
        let area = geometry::DisplayArea {
            top_left_mm: [-298.5, 336.0, 115.0],
            top_right_mm: [298.5, 336.0, 115.0],
            bottom_left_mm: [-298.5, 10.25, -3.0],
        };
        let values = [
            NotificationValue::None,
            NotificationValue::Float(33.0),
            NotificationValue::State(true),
            NotificationValue::DisplayArea(area),
            NotificationValue::Uint(0x7186_ba7d),
            NotificationValue::EnabledEye(2),
            NotificationValue::String("fault".into()),
        ];
        for value in values {
            let n = Notification {
                kind: notification::CALIBRATION_ID_CHANGED,
                value,
            };
            assert_eq!(
                decode_server(&encode_notification(&n)),
                Some(ServerMsg::Notification(n))
            );
        }
    }

    #[test]
    fn requests_and_replies_round_trip() {
        let body = request::encode_request(7, request::kind::STATE, &request::encode_u32(6));
        let req = request::decode_request(&body).unwrap();
        assert_eq!((req.id, req.kind), (7, request::kind::STATE));
        assert_eq!(request::decode_u32(req.payload), Some(6));
        assert_eq!(request::decode_request(&[TAG_REQUEST, 1, 0]), None);

        assert_eq!(
            decode_server(&encode_reply(7, request::status::OK, &[1, 2, 3])),
            Some(ServerMsg::Reply {
                request_id: 7,
                status: 0,
                payload: vec![1, 2, 3],
            })
        );
    }

    #[test]
    fn request_payloads_round_trip() {
        use request::*;
        let info = DeviceInfo {
            serial_number: "SERIAL-0001".into(),
            model: "IS5_Large_Eyetracker_5".into(),
            generation: "IS5".into(),
            firmware_version: "02a1a6a977".into(),
            integration_type: "Peripheral".into(),
        };
        assert_eq!(decode_device_info(&encode_device_info(&info)), Some(info));

        let mut track_box = geometry::TrackBox::default();
        track_box.corners_mm[0] = [125.0, 100.0, 450.0];
        track_box.corners_mm[7] = [250.0, -200.0, 900.0];
        assert_eq!(
            decode_track_box(&encode_track_box(&track_box)),
            Some(track_box)
        );

        let mounting = geometry::GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 20.0,
            external_offset_mm: [0.0, -0.5, 13.5],
            internal_offset_mm: [0.0, 5.5, 9.5],
        };
        assert_eq!(
            decode_geometry_mounting(&encode_geometry_mounting(&mounting)),
            Some(mounting)
        );

        let sync = Timesync {
            host_start_us: 10,
            device_us: 20,
            host_end_us: 30,
        };
        assert_eq!(decode_timesync(&encode_timesync(&sync)), Some(sync));
        assert_eq!(
            decode_point_2d(&encode_point_2d(0.1, 0.9)),
            Some((0.1, 0.9))
        );
        assert_eq!(decode_display_area(&[0; 35]), None);
    }

    #[test]
    fn a_device_info_reply_without_the_integration_type_still_decodes() {
        use request::*;
        let info = DeviceInfo {
            serial_number: "SERIAL-0001".into(),
            model: "IS5_Large_Eyetracker_5".into(),
            generation: "IS5".into(),
            firmware_version: "02a1a6a977".into(),
            integration_type: String::new(),
        };
        // What a daemon from before the integration type sends.
        let four = wire::Writer::default()
            .str(&info.serial_number)
            .str(&info.model)
            .str(&info.generation)
            .str(&info.firmware_version)
            .finish();
        assert_eq!(decode_device_info(&four), Some(info.clone()));

        // A client from before it reads the four strings and ignores the
        // rest, so a new daemon's reply must start with them.
        let with_type = DeviceInfo {
            integration_type: "Peripheral".into(),
            ..info.clone()
        };
        assert!(encode_device_info(&with_type).starts_with(&four));

        // A tail that claims more bytes than follow is dropped, not the reply.
        let mut cut = four.clone();
        cut.extend_from_slice(&[10, 0, b'P', b'e']);
        assert_eq!(decode_device_info(&cut), Some(info));

        assert_eq!(decode_device_info(&four[..four.len() - 1]), None);
    }

    #[test]
    fn stream_types_round_trip_and_a_count_past_the_payload_is_rejected() {
        use request::*;
        let types = vec![
            StreamType {
                id: 0x500,
                name: "gaze".into(),
                ..StreamType::default()
            },
            StreamType {
                id: 0x508,
                name: "image_collection".into(),
                text: "Bildsammlung ü".into(),
                value: 1000,
            },
        ];
        let body = encode_stream_types(&types);
        assert_eq!(decode_stream_types(&body), Some(types));
        assert_eq!(decode_stream_types(&encode_stream_types(&[])), Some(vec![]));

        for len in 0..body.len() {
            assert_eq!(decode_stream_types(&body[..len]), None, "cut at {len}");
        }
        // A huge count with a short payload fails before allocating.
        let mut huge = u32::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(&[0; 12]);
        assert_eq!(decode_stream_types(&huge), None);
        // Exactly as many minimal entries as the payload holds.
        let mut two = 2u32.to_le_bytes().to_vec();
        two.extend_from_slice(&[0; 24]);
        assert_eq!(
            decode_stream_types(&two),
            Some(vec![StreamType::default(), StreamType::default()])
        );
    }

    fn hardware() -> request::HardwareConfiguration {
        use request::*;
        let mut values = [0.0; 15];
        values[3] = 5.22;
        values[13] = 0.001_75;
        HardwareConfiguration {
            entries: vec![
                HardwareEntry {
                    param_a: 16.0,
                    param_b: 100.0,
                    position_mm: [0.0, 0.0, 4.14],
                    values,
                    width: 2240,
                    height: 2240,
                    coefficients: vec![0.1, -0.2],
                    param_d: 1.0 / 3.0,
                    ..HardwareEntry::default()
                },
                HardwareEntry {
                    id: u32::MAX,
                    param_a: 62.0,
                    point_b_mm: [1.0, 2.0, 3.0],
                    ..HardwareEntry::default()
                },
            ],
            points_mm: vec![[0.0; 3], [130.0, 0.76, 1.62], [-130.0, 0.76, 1.62]],
            mode: 1,
        }
    }

    #[test]
    fn a_hardware_configuration_round_trips_at_full_precision() {
        use request::*;
        let h = hardware();
        let body = encode_hardware_configuration(&h);
        assert_eq!(decode_hardware_configuration(&body), Some(h));
        assert_eq!(
            decode_hardware_configuration(&encode_hardware_configuration(
                &HardwareConfiguration::default()
            )),
            Some(HardwareConfiguration::default())
        );
        for len in 0..body.len() {
            assert_eq!(
                decode_hardware_configuration(&body[..len]),
                None,
                "cut at {len}"
            );
        }
    }

    #[test]
    fn hardware_configuration_lists_are_bounded() {
        use request::*;
        let mut big = hardware();
        big.entries.push(HardwareEntry::default());
        big.entries[0].coefficients = vec![0.5; HARDWARE_COEFFICIENTS_MAX + 1];
        big.points_mm = vec![[1.0; 3]; HARDWARE_POINTS_MAX + 1];

        let got = decode_hardware_configuration(&encode_hardware_configuration(&big))
            .expect("the encoder cuts every list to its limit");
        assert_eq!(got.entries.len(), HARDWARE_ENTRIES_MAX);
        assert_eq!(got.entries[0].coefficients.len(), HARDWARE_COEFFICIENTS_MAX);
        assert_eq!(got.points_mm.len(), HARDWARE_POINTS_MAX);

        // Counts past the limits, however much follows them.
        let mut entries = vec![3];
        entries.extend_from_slice(&[0; 4096]);
        assert_eq!(decode_hardware_configuration(&entries), None);
        let mut points = vec![0, 41];
        points.extend_from_slice(&[0; 41 * 24 + 4]);
        assert_eq!(decode_hardware_configuration(&points), None);
        let one = HardwareConfiguration {
            entries: vec![HardwareEntry::default()],
            ..HardwareConfiguration::default()
        };
        let mut coefficients = encode_hardware_configuration(&one);
        // The coefficient count: after the entry count, id, two values, the
        // position, the values and three words.
        let at = 1 + 4 + 2 * 8 + 3 * 8 + 15 * 8 + 3 * 4;
        assert_eq!(coefficients[at], 0);
        coefficients[at] = 65;
        coefficients.splice(at + 1..at + 1, [0; 65 * 8]);
        assert_eq!(decode_hardware_configuration(&coefficients), None);
    }
}
