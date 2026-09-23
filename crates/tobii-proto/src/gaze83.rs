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
//! There is no pupil diameter on this wire. The fields that decode.rs labels
//! `pupil_*` (keys `0x25`/`0x27`, third component) track the eye's range, not
//! its pupil, and keys `0x06`/`0x0c` are an unidentified quantity.

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

    /// The keys of one eye.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Eye {
        /// Tracking status: 0 tracked, 4 not tracked.
        pub status: u32,
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
        /// 3-D gaze point on the screen plane, display frame.
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
    /// The device tracks this eye in this frame.
    pub tracked: bool,
    /// Cornea centre (gaze origin), tracker frame, mm.
    pub origin_tracker_mm: Valued<[f64; 3]>,
    /// Gaze origin, display frame, mm (the Stream Engine's `gazeOrigin`).
    pub origin_display_mm: Valued<[f64; 3]>,
    /// Eye position normalised to the track box, `0..1` per axis.
    pub track_box: Valued<[f64; 3]>,
    /// Gaze point on the screen plane, display frame, mm.
    pub gaze_point_display_mm: Valued<[f64; 3]>,
    /// Gaze point, normalised display coordinates.
    pub gaze_point_norm: Valued<[f64; 2]>,
    /// Eyeball rotation centre, tracker frame, mm.
    pub eyeball_center_mm: Valued<[f64; 3]>,
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
    /// Unfiltered combined gaze point.
    pub gaze_raw: Valued<[f64; 2]>,
    /// The user's left eye.
    pub left: EyeFrame,
    /// The user's right eye.
    pub right: EyeFrame,
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
    let flag = |k: u32| fields.scalar(k) == Some(1);
    let scaled = |v: [f64; 3]| v.map(|c| c / UNITS_PER_MM);

    let eye = |k: key::Eye| {
        let tracked = fields.scalar(k.status) == Some(0);
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
            origin_tracker_mm: flagged3(k.origin_tracker, k.origin_tracker_valid),
            origin_display_mm: flagged3(k.origin_display, k.origin_display_valid),
            track_box: unflagged3(k.track_box),
            gaze_point_display_mm: flagged3(k.gaze_point_3d, k.gaze_point_valid),
            gaze_point_norm: fields
                .point::<2>(k.gaze_point_2d)
                .map_or_else(Valued::default, |v| Valued {
                    valid: flag(k.gaze_point_valid),
                    value: v.map(|c| c / UNITS_PER_MM),
                }),
            eyeball_center_mm: unflagged3(k.eyeball),
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

    Some(GazeFrame {
        device_ts_us,
        frame_counter: fields
            .scalar(key::FRAME_COUNTER)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        gaze: gaze2(key::COMBINED_GAZE, key::COMBINED_GAZE_VALID),
        gaze_raw: gaze2(key::RAW_COMBINED_GAZE, key::RAW_COMBINED_GAZE_VALID),
        left: eye(key::LEFT),
        right: eye(key::RIGHT),
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
