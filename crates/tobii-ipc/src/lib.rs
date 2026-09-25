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
//! | `0x20` HEAD .. `0x27` NOTIFICATION | daemon -> client | samples ([`ServerMsg`]) |
//!
//! Sample timestamps are the device clock in microseconds. Host timestamps
//! (gaze data's system time, the TIMESYNC pair) are [`host_clock_us`], which
//! is also what `tobii_system_clock` returns.
//!
//! The SUBSCRIBE mask is written as `u32 LE`, whose first byte is the low
//! byte of the mask: a daemon that reads only one byte still sees every
//! stream below bit 8, so old and new peers interoperate either way.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

mod clock;
pub mod geometry;
pub mod request;
mod sample;
mod wire;

pub use clock::host_clock_us;
pub use sample::{
    EyePair, EyePoint, GazeData, GazeDataEye, Image, Notification, NotificationValue, ServerMsg,
    decode_server, encode_eye_position, encode_gaze, encode_gaze_data, encode_gaze_origin,
    encode_head, encode_image, encode_notification, encode_presence, encode_reply,
    encode_subscribed, notification,
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
/// `2 x f32` pupil-diameter tail (mm, left/right).
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
