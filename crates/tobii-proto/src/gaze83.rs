//! The 0x500 gaze stream and the 0x504 presence stream, decoded by field key.
//!
//! Each 0x500 message carries 39 keyed fields (see [`crate::tlv`]). Which key
//! is what was established from labelled captures, and the parts the Stream
//! Engine forwards were matched bit-for-bit against its own output: its
//! `gazePoint` is key `0x1c` / 1024 with validity `0x1b`, its `gazeOrigin` is
//! keys `0x22`/`0x24` / 1024 (display frame) with validity `0x21`/`0x23`, and
//! its timestamps are key `0x01` rebased. Like the Stream Engine, values are
//! passed through even when their validity flag is clear; the flag is
//! reported next to them.
//!
//! The other per-eye 3-D positions, in mm, are in the tracker frame: the
//! cornea centres `0x02`/`0x08`, the eyeball centres `0x17`/`0x18` and the
//! 3-D gaze points `0x04`/`0x0a`, which on the display area of the capture
//! lie on the screen plane, where the 2-D gaze points `0x05`/`0x0b` put them.
//! The track-box position (`0x03`/`0x09`) and those 2-D points are
//! normalised, not in mm.
//!
//! The pupil diameters, in mm, are keys `0x06`/`0x0c` (16.16): the Stream
//! Engine hands them out as gaze data's `pupil_diameter_mm`, valid while the
//! eye's status (`0x07`/`0x0d`) is below 2 (0x18017199a..0x1801719af). They
//! do not shrink with the eye's distance, as a size in camera pixels would.
//! Keys `0x25`/`0x27`'s third component, which decode.rs reports as
//! `secondary_range_*`, is not the pupil: it tracks the eye's range.
//!
//! The Stream Engine's own record of a frame, which `process_gaze`
//! (0x18018d9a0) fills and the raw gaze callback receives, takes keys
//! `0x01`..`0x0d`, `0x1b` and `0x1c` and, each behind a flag set when the
//! frame has it, keys `0x0e`, `0x11` and `0x14`..`0x18`; it reads no key
//! above `0x1c`. A [`GazeFrame`] holds all of it: the keys its other fields
//! reduce to a validity, or leave out, are kept as sent in
//! [`EyeFrame::status`] and [`GazeFrame::record`]. The unfiltered combined
//! gaze, [`GazeFrame::gaze_raw`] (key `0x20`), is not part of that record.

use crate::protocol::{Message, STREAM_ID_GAZE, STREAM_ID_PRESENCE};
use crate::tlv::{UNITS_PER_MM, keyed_fields};

/// Field keys of the 0x500 message.
pub mod key {
    /// `u64` device timestamp, µs.
    pub const TIMESTAMP: u32 = 0x01;
    /// `u32` frame counter (132 Hz, steps by 4 per output frame).
    pub const FRAME_COUNTER: u32 = 0x14;
    /// Validity of [`COMBINED_GAZE`].
    pub const COMBINED_GAZE_VALID: u32 = 0x1b;
    /// Filtered combined gaze point, display-normalised x1024.
    pub const COMBINED_GAZE: u32 = 0x1c;
    /// Validity of [`RAW_COMBINED_GAZE`].
    pub const RAW_COMBINED_GAZE_VALID: u32 = 0x1f;
    /// Unfiltered combined gaze, `(left + right) / 2`.
    pub const RAW_COMBINED_GAZE: u32 = 0x20;
    /// `u32` of unknown meaning, which the ET5 never sends; the Stream
    /// Engine's gaze record keeps it.
    pub const KEY_0E: u32 = 0x0e;
    /// `u32` of unknown meaning; the ET5 sends 4 in every frame.
    pub const KEY_11: u32 = 0x11;

    /// The keys of one eye.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Eye {
        /// Tracking status: 0 tracked, 4 not tracked.
        pub status: u32,
        /// Pupil diameter, mm, 16.16.
        pub pupil: u32,
        /// Cornea centre (gaze origin), tracker frame.
        pub origin_tracker: u32,
        /// Validity of `origin_tracker`.
        pub origin_tracker_valid: u32,
        /// Gaze origin, display frame.
        pub origin_display: u32,
        /// Validity of `origin_display`.
        pub origin_display_valid: u32,
        /// Eye position normalised to the track box, x1024.
        pub track_box: u32,
        /// 3-D gaze point on the screen plane, tracker frame.
        pub gaze_point_3d: u32,
        /// 2-D gaze point, display-normalised x1024.
        pub gaze_point_2d: u32,
        /// Validity of both gaze points.
        pub gaze_point_valid: u32,
        /// Eyeball rotation centre, tracker frame.
        pub eyeball: u32,
    }

    /// The user's left eye.
    pub const LEFT: Eye = Eye {
        status: 0x07,
        pupil: 0x06,
        origin_tracker: 0x02,
        origin_tracker_valid: 0x16,
        origin_display: 0x22,
        origin_display_valid: 0x21,
        track_box: 0x03,
        gaze_point_3d: 0x04,
        gaze_point_2d: 0x05,
        gaze_point_valid: 0x1d,
        eyeball: 0x17,
    };

    /// The user's right eye.
    pub const RIGHT: Eye = Eye {
        status: 0x0d,
        pupil: 0x0c,
        origin_tracker: 0x08,
        origin_tracker_valid: 0x15,
        origin_display: 0x24,
        origin_display_valid: 0x23,
        track_box: 0x09,
        gaze_point_3d: 0x0a,
        gaze_point_2d: 0x0b,
        gaze_point_valid: 0x1e,
        eyeball: 0x18,
    };
}

/// A value and whether the device vouches for it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Valued<T> {
    /// Whether `value` is a measurement for this frame.
    pub valid: bool,
    /// The value, passed through even when invalid.
    pub value: T,
}

/// One eye of a [`GazeFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EyeFrame {
    /// The device tracks this eye in this frame: its status is 0.
    pub tracked: bool,
    /// Tracking status as sent: the ET5 sends 0 for a tracked eye, 4 for a
    /// lost one. `None` when the frame lacks the key, which the Stream
    /// Engine's zeroed record would read as 0.
    pub status: Option<u32>,
    /// Cornea centre (gaze origin), tracker frame, mm.
    pub origin_tracker_mm: Valued<[f64; 3]>,
    /// Gaze origin, display frame, mm (the Stream Engine's `gazeOrigin`).
    pub origin_display_mm: Valued<[f64; 3]>,
    /// Eye position normalised to the track box, `0..1` per axis.
    pub track_box: Valued<[f64; 3]>,
    /// Gaze point on the screen plane, tracker frame, mm.
    pub gaze_point_tracker_mm: Valued<[f64; 3]>,
    /// Gaze point, normalised display coordinates.
    pub gaze_point_norm: Valued<[f64; 2]>,
    /// Eyeball rotation centre, tracker frame, mm.
    pub eyeball_center_mm: Valued<[f64; 3]>,
    /// Pupil diameter, mm; valid, as the Stream Engine judges it, while the
    /// eye's status is below 2. Unlike it, not valid when the frame lacks
    /// either key (see [`decode_gaze_frame`]).
    pub pupil_diameter_mm: Valued<f64>,
}

/// One decoded 0x500 message.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GazeFrame {
    /// Device timestamp, µs.
    pub device_ts_us: u64,
    /// Device frame counter.
    pub frame_counter: u32,
    /// Filtered combined gaze point, normalised display coordinates,
    /// unclamped (the Stream Engine's `gazePoint`).
    pub gaze: Valued<[f64; 2]>,
    /// Unfiltered combined gaze point. Not part of the Stream Engine's gaze
    /// record, which is what its raw gaze hands out.
    pub gaze_raw: Valued<[f64; 2]>,
    /// The user's left eye.
    pub left: EyeFrame,
    /// The user's right eye.
    pub right: EyeFrame,
    /// The rest of what the Stream Engine's gaze record keeps of the frame.
    pub record: RecordKeys,
}

/// Keys of a 0x500 message that the Stream Engine's gaze record keeps as
/// sent and the rest of a [`GazeFrame`] does not: the combined gaze
/// validity, and the keys the record flags as present or not. `None`, or
/// `false`, for a key the frame lacks.
///
/// Where the record puts each (`process_gaze`, 0x18018d9a0, whose key
/// switch jumps through the table at RVA 0x18f5a8; the flagged keys' cases
/// span 0x18018dcf5..0x18018de01): the combined gaze point (key `0x1c`,
/// [`GazeFrame::gaze`]) at `+0x70`, its validity at `+0x78`, then flag and
/// value of `0x0e` at `+0x7c`, `0x11` at `+0x94`, `0x14` at `+0xac`, `0x16`
/// at `+0xb4`, `0x15` at `+0xbc`, and the eyeball centres `0x17` at `+0xc4`
/// and `0x18` at `+0xd4`, whose values are [`EyeFrame::eyeball_center_mm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecordKeys {
    /// Key `0x1b`, the combined gaze validity; [`GazeFrame::gaze`] is valid
    /// when it is 1. The record keeps it without a flag, 0 when absent.
    pub combined_gaze_validity: Option<u32>,
    /// Key `0x0e`, of unknown meaning; the ET5 never sends it.
    pub key_0e: Option<u32>,
    /// Key `0x11`, of unknown meaning; the ET5 sends 4.
    pub key_11: Option<u32>,
    /// Key `0x14`, the frame counter; [`GazeFrame::frame_counter`] is it,
    /// or 0.
    pub frame_counter: Option<u32>,
    /// Key `0x16`, the left eye's gaze-origin validity;
    /// [`EyeFrame::origin_tracker_mm`] is valid when it is 1.
    pub left_origin_flag: Option<u32>,
    /// Key `0x15`, the right eye's gaze-origin validity.
    pub right_origin_flag: Option<u32>,
    /// The frame has key `0x17`, the left eyeball centre, as a 3-D point.
    pub left_eyeball_sent: bool,
    /// The frame has key `0x18`, the right eyeball centre, as a 3-D point.
    pub right_eyeball_sent: bool,
}

/// Wire value of an invalid component: exactly 0 or ±1024 (±1 mm-unit scale
/// or ±1 normalised, depending on the field).
fn is_sentinel(v: f64) -> bool {
    v == 0.0 || v.abs() == UNITS_PER_MM
}

/// Decode a 0x500 message; `None` for any other message or one without a
/// device timestamp.
#[must_use]
pub fn decode_gaze_frame(msg: &Message<'_>) -> Option<GazeFrame> {
    if msg.id != STREAM_ID_GAZE {
        return None;
    }
    let fields = keyed_fields(msg.payload.get(2..)?);
    let device_ts_us = fields.scalar(key::TIMESTAMP)?;
    let word = |k: u32| fields.scalar(k).and_then(|v| u32::try_from(v).ok());
    let flag = |k: u32| word(k) == Some(1);
    let scaled = |v: [f64; 3]| v.map(|c| c / UNITS_PER_MM);

    let eye = |k: key::Eye| {
        let status = word(k.status);
        let tracked = status == Some(0);
        let point3 = |key: u32| fields.point::<3>(key);
        let flagged3 = |key: u32, valid: u32| {
            point3(key).map_or_else(Valued::default, |v| Valued {
                valid: flag(valid),
                value: scaled(v),
            })
        };
        // Fields without a flag of their own: trusted when the eye is
        // tracked and no component is a sentinel.
        let unflagged3 = |key: u32| {
            point3(key).map_or_else(Valued::default, |v| Valued {
                valid: tracked && !v.iter().copied().any(is_sentinel),
                value: scaled(v),
            })
        };
        EyeFrame {
            tracked,
            status,
            origin_tracker_mm: flagged3(k.origin_tracker, k.origin_tracker_valid),
            origin_display_mm: flagged3(k.origin_display, k.origin_display_valid),
            track_box: unflagged3(k.track_box),
            gaze_point_tracker_mm: flagged3(k.gaze_point_3d, k.gaze_point_valid),
            gaze_point_norm: fields
                .point::<2>(k.gaze_point_2d)
                .map_or_else(Valued::default, |v| Valued {
                    valid: flag(k.gaze_point_valid),
                    value: v.map(|c| c / UNITS_PER_MM),
                }),
            eyeball_center_mm: unflagged3(k.eyeball),
            // The Stream Engine's test (0x180171818, 0x180171844). A frame
            // without the status or the diameter, which the ET5 never sends,
            // gives none. The Stream Engine, reading its zeroed record
            // (0x18018dadd), would take a missing status as 0 and a missing
            // diameter as a valid 0.0 mm; libtobii reports neither as valid.
            pupil_diameter_mm: fields
                .fixed16(k.pupil)
                .map_or_else(Valued::default, |v| Valued {
                    valid: status.is_some_and(|s| s < 2),
                    value: v,
                }),
        }
    };

    let gaze2 = |point: u32, valid: u32| {
        fields
            .point::<2>(point)
            .map_or_else(Valued::default, |v| Valued {
                valid: flag(valid),
                value: v.map(|c| c / UNITS_PER_MM),
            })
    };

    let record = RecordKeys {
        combined_gaze_validity: word(key::COMBINED_GAZE_VALID),
        key_0e: word(key::KEY_0E),
        key_11: word(key::KEY_11),
        frame_counter: word(key::FRAME_COUNTER),
        left_origin_flag: word(key::LEFT.origin_tracker_valid),
        right_origin_flag: word(key::RIGHT.origin_tracker_valid),
        left_eyeball_sent: fields.point::<3>(key::LEFT.eyeball).is_some(),
        right_eyeball_sent: fields.point::<3>(key::RIGHT.eyeball).is_some(),
    };

    Some(GazeFrame {
        device_ts_us,
        frame_counter: record.frame_counter.unwrap_or(0),
        gaze: gaze2(key::COMBINED_GAZE, key::COMBINED_GAZE_VALID),
        gaze_raw: gaze2(key::RAW_COMBINED_GAZE, key::RAW_COMBINED_GAZE_VALID),
        left: eye(key::LEFT),
        right: eye(key::RIGHT),
        record,
    })
}

/// Presence as the 0x504 stream reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresenceFrame {
    /// Device timestamp, µs.
    pub device_ts_us: u64,
    /// 1 away, 2 present (the Stream Engine's numbering).
    pub state: u32,
}

/// Decode a 0x504 message (key 1 = timestamp, key 2 = state). The device
/// sends one when the stream starts and then only on change.
#[must_use]
pub fn decode_presence_frame(msg: &Message<'_>) -> Option<PresenceFrame> {
    if msg.id != STREAM_ID_PRESENCE {
        return None;
    }
    let fields = keyed_fields(msg.payload.get(2..)?);
    Some(PresenceFrame {
        device_ts_us: fields.scalar(1)?,
        state: u32::try_from(fields.scalar(2)?).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::parse_message;
    use crate::tlv::{KEY_FIELD_ID, TYPE_FIELD_ID, TYPE_U32};
    use tobii_ipc::geometry::tracker_to_display;

    /// Frame 43780 of session1: what the Windows Stream Engine delivered for
    /// it is in session1.jsonl (gazePoint and gazeOrigin at 323209472 µs).
    #[test]
    #[allow(clippy::float_cmp)] // reason: the DLL's values are reproduced bit for bit
    fn decodes_what_the_stream_engine_reported() {
        let bytes = crate::fixture!("session1-gaze-frame");
        let msg = parse_message(&bytes).expect("message");

        let frame = decode_gaze_frame(&msg).expect("gaze frame");

        assert_eq!(frame.device_ts_us, 9_613_320_391);
        assert_eq!(frame.frame_counter, 43_780);
        assert!(frame.gaze.valid);
        assert_eq!(
            frame.gaze.value,
            [0.587_291_359_901_428_2, 0.099_273_845_553_398_13]
        );
        let (left, right) = (frame.left.origin_display_mm, frame.right.origin_display_mm);
        assert!(left.valid && right.valid);
        assert_eq!(
            left.value,
            [
                -57.900_634_765_625,
                112.468_688_964_843_75,
                618.279_418_945_312_5
            ]
        );
        assert_eq!(
            right.value,
            [
                16.091_180_801_391_6,
                108.908_447_265_625,
                621.471_313_476_562_5
            ]
        );
        assert!(frame.left.tracked && frame.right.tracked);
        assert!(frame.left.track_box.valid);
        assert!(
            frame
                .left
                .track_box
                .value
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        );
    }

    /// Keys `0x04`/`0x0a` are in the tracker frame, like `0x02`/`0x08`: on
    /// the display area of the capture (the init's 1430), key `0x02` maps
    /// onto the display-frame origin the Stream Engine reported (`0x22`), and
    /// key `0x04` onto the screen plane, where the 2-D gaze point (`0x05`)
    /// puts it.
    #[test]
    fn the_3d_gaze_point_is_in_the_tracker_frame() {
        let frame = decode_gaze_frame(
            &parse_message(&crate::fixture!("session1-gaze-frame")).expect("message"),
        )
        .expect("gaze frame");
        let (area, _) = crate::facts::parse_display_area(
            &parse_message(&crate::fixture!("init-rsp-1430")).expect("message"),
        )
        .expect("display area");
        let length = |a: [f64; 3], b: [f64; 3]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        };
        let width = length(area.top_right_mm, area.top_left_mm);
        let height = length(area.top_left_mm, area.bottom_left_mm);

        for eye in [frame.left, frame.right] {
            let origin =
                tracker_to_display(&area, eye.origin_tracker_mm.value).expect("a display frame");
            assert!(
                length(origin, eye.origin_display_mm.value) < 1e-3,
                "{origin:?}"
            );
            let [x, y, z] = tracker_to_display(&area, eye.gaze_point_tracker_mm.value)
                .expect("a display frame");
            let [u, v] = eye.gaze_point_norm.value;
            assert!(z.abs() < 1e-3, "{z} mm off the screen");
            assert!((x / width + 0.5 - u).abs() < 1e-4, "{x} vs {u}");
            assert!((0.5 - y / height - v).abs() < 1e-4, "{y} vs {v}");
        }
    }

    /// The 0x500 fixture frame.
    fn session1() -> Vec<u8> {
        crate::fixture!("session1-gaze-frame")
    }

    /// Where the key word of keyed field `key` sits in a message, after
    /// `05 KEY_FIELD_ID 02`; the field's value follows it.
    fn key_at(bytes: &[u8], key: u32) -> usize {
        let mut announce = vec![TYPE_FIELD_ID, 0, 0, 0, 4];
        announce.extend_from_slice(&KEY_FIELD_ID.to_be_bytes());
        announce.extend_from_slice(&[TYPE_U32, 0, 0, 0, 4]);
        let start = announce.len();
        announce.extend_from_slice(&key.to_be_bytes());
        bytes
            .windows(announce.len())
            .position(|w| w == announce)
            .expect("the key")
            + start
    }

    /// `bytes` with the 4-byte scalar of keyed field `key` set to `value`.
    fn with_scalar(mut bytes: Vec<u8>, key: u32, value: u32) -> Vec<u8> {
        // The key word, then the value's type byte and u32 length.
        let at = key_at(&bytes, key) + 4 + 5;
        bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
        bytes
    }

    /// `bytes` with keyed field `from` announced as key `to` instead.
    fn renamed(mut bytes: Vec<u8>, from: u32, to: u32) -> Vec<u8> {
        let at = key_at(&bytes, from);
        bytes[at..at + 4].copy_from_slice(&to.to_be_bytes());
        bytes
    }

    /// `bytes` without keyed field `key`: it is announced as `0x0f`, a key
    /// the ET5 never sends and neither this decoder nor the Stream Engine's
    /// record reads.
    fn without(bytes: Vec<u8>, key: u32) -> Vec<u8> {
        renamed(bytes, key, 0x0f)
    }

    fn decode(bytes: &[u8]) -> GazeFrame {
        decode_gaze_frame(&parse_message(bytes).expect("message")).expect("gaze frame")
    }

    /// Keys `0x06`/`0x0c`, 16.16, are what the Stream Engine hands out as gaze
    /// data's `pupil_diameter_mm`.
    #[test]
    #[allow(clippy::float_cmp)] // reason: 16.16 values decode exactly
    fn the_pupil_diameters_are_keys_06_and_0c() {
        let frame = decode(&session1());

        assert_eq!(
            frame.left.pupil_diameter_mm,
            Valued {
                valid: true,
                value: 6.247_360_229_492_187_5
            }
        );
        assert_eq!(
            frame.right.pupil_diameter_mm,
            Valued {
                valid: true,
                value: 5.996_612_548_828_125
            }
        );
    }

    /// The Stream Engine's pupil validity is the eye's status below 2
    /// (0x180171818, 0x180171844): the ET5's 4 (not tracked) clears it. The
    /// diameter is passed through either way, as the Stream Engine does.
    #[test]
    #[allow(clippy::float_cmp)] // reason: 16.16 values decode exactly
    fn a_pupil_is_valid_while_the_eye_status_is_below_2() {
        for (status, valid) in [(0, true), (1, true), (2, false), (4, false)] {
            let frame = decode(&with_scalar(session1(), key::RIGHT.status, status));

            assert_eq!(frame.right.pupil_diameter_mm.valid, valid, "{status}");
            assert_eq!(frame.right.pupil_diameter_mm.value, 5.996_612_548_828_125);
            assert!(frame.left.pupil_diameter_mm.valid, "the other eye");
        }
    }

    /// A frame without the eye's status or pupil key reports no pupil for
    /// that eye; the ET5 sends both in every frame. This departs from the
    /// Stream Engine on purpose: its zeroed record would read a missing
    /// status as 0 and a missing diameter as a valid 0.0 mm.
    #[test]
    fn a_missing_status_or_pupil_gives_no_pupil() {
        for missing in [key::LEFT.status, key::LEFT.pupil] {
            let frame = decode(&without(session1(), missing));

            assert!(!frame.left.pupil_diameter_mm.valid, "{missing:#x}");
            assert!(frame.right.pupil_diameter_mm.valid);
        }
    }

    /// The Stream Engine's gaze record (`process_gaze`, 0x18018d9a0) takes
    /// keys `0x02`..`0x07` for the left eye and `0x08`..`0x0d` for the
    /// right, the combined gaze `0x1c`/`0x1b`, and flags `0x0e`, `0x11`,
    /// `0x14`..`0x18`. A decoded frame holds every one of them as the record
    /// does: points / 1024, the pupil as 16.16, the scalars as sent.
    #[test]
    #[allow(clippy::float_cmp)] // reason: the same wire values, scaled the same way
    fn a_frame_holds_what_the_stream_engine_record_takes() {
        let bytes = session1();
        let msg = parse_message(&bytes).expect("message");
        let keys = keyed_fields(msg.payload.get(2..).expect("keys"));
        let point3 = |k: u32| {
            keys.point::<3>(k)
                .expect("the key")
                .map(|c| c / UNITS_PER_MM)
        };
        let point2 = |k: u32| {
            keys.point::<2>(k)
                .expect("the key")
                .map(|c| c / UNITS_PER_MM)
        };

        let frame = decode_gaze_frame(&msg).expect("gaze frame");

        for (eye, first, eyeball) in [(frame.left, 0x02, 0x17), (frame.right, 0x08, 0x18)] {
            assert_eq!(eye.origin_tracker_mm.value, point3(first));
            assert_eq!(eye.track_box.value, point3(first + 1));
            assert_eq!(eye.gaze_point_tracker_mm.value, point3(first + 2));
            assert_eq!(eye.gaze_point_norm.value, point2(first + 3));
            assert_eq!(Some(eye.pupil_diameter_mm.value), keys.fixed16(first + 4));
            assert_eq!(eye.status.map(u64::from), keys.scalar(first + 5));
            assert_eq!(eye.eyeball_center_mm.value, point3(eyeball));
        }
        assert_eq!(frame.gaze.value, point2(0x1c));
        assert_eq!(
            frame.record,
            RecordKeys {
                combined_gaze_validity: Some(1),
                key_0e: None,
                key_11: Some(4),
                frame_counter: Some(43_780),
                left_origin_flag: Some(1),
                right_origin_flag: Some(1),
                left_eyeball_sent: true,
                right_eyeball_sent: true,
            }
        );
    }

    /// Each scalar the record keeps lands in its own field as sent: the
    /// statuses `0x07`/`0x0d`, the combined gaze validity `0x1b`, `0x0e`
    /// (on a frame that has it), `0x11`, the frame counter `0x14`, and the
    /// origin flags `0x16` (left, record `+0xb8`) and `0x15` (right,
    /// `+0xc0`). The fields derived from them follow.
    #[test]
    fn the_record_keeps_each_scalar_as_sent() {
        // Key 0x2a, a u32 the ET5 always sends as 0, stands in for 0x0e.
        let mut bytes = renamed(session1(), 0x2a, 0x0e);
        for (k, value) in [
            (0x07, 4),
            (0x0d, 1),
            (0x1b, 2),
            (0x0e, 9),
            (0x11, 5),
            (0x14, 8),
            (0x16, 7),
            (0x15, 3),
        ] {
            bytes = with_scalar(bytes, k, value);
        }

        let frame = decode(&bytes);

        assert_eq!((frame.left.status, frame.right.status), (Some(4), Some(1)));
        assert!(!frame.left.tracked && !frame.right.tracked);
        assert_eq!(
            frame.record,
            RecordKeys {
                combined_gaze_validity: Some(2),
                key_0e: Some(9),
                key_11: Some(5),
                frame_counter: Some(8),
                left_origin_flag: Some(7),
                right_origin_flag: Some(3),
                left_eyeball_sent: true,
                right_eyeball_sent: true,
            }
        );
        assert_eq!(frame.frame_counter, 8);
        assert!(!frame.gaze.valid);
        assert!(!frame.left.origin_tracker_mm.valid && !frame.right.origin_tracker_mm.valid);
    }

    /// A key the frame lacks is `None` (or not sent), where the Stream
    /// Engine's zeroed record would hold 0 and a clear flag; the other
    /// eye's keys are untouched.
    #[test]
    fn keys_the_frame_lacks_are_none() {
        let bytes = [0x07, 0x1b, 0x11, 0x14, 0x16, 0x17]
            .into_iter()
            .fold(session1(), without);

        let frame = decode(&bytes);

        assert_eq!((frame.left.status, frame.right.status), (None, Some(0)));
        assert!(!frame.left.tracked && frame.right.tracked);
        assert_eq!(
            frame.record,
            RecordKeys {
                combined_gaze_validity: None,
                key_0e: None,
                key_11: None,
                frame_counter: None,
                left_origin_flag: None,
                right_origin_flag: Some(1),
                left_eyeball_sent: false,
                right_eyeball_sent: true,
            }
        );
        assert_eq!(frame.frame_counter, 0);
        assert_eq!(frame.left.eyeball_center_mm, Valued::default());
    }

    #[test]
    fn a_non_gaze_message_is_not_a_gaze_frame() {
        let bytes = crate::fixture!("init-presence");
        let msg = parse_message(&bytes).expect("message");
        assert_eq!(decode_gaze_frame(&msg), None);
    }

    #[test]
    fn decodes_presence_at_stream_start_and_on_change() {
        let init = crate::fixture!("init-presence");
        let present = decode_presence_frame(&parse_message(&init).expect("msg")).expect("presence");
        assert_eq!(present.state, 2);
        assert_eq!(present.device_ts_us, 0x18_91b5_3908);

        let away = crate::fixture!("session2-presence-away");
        let away = decode_presence_frame(&parse_message(&away).expect("msg")).expect("presence");
        assert_eq!(away.state, 1);
    }
}
