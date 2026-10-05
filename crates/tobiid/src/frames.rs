//! Engine samples -> IPC frames.
//!
//! Each sample is encoded once, and only into the streams some client wants;
//! the pump then writes the bodies to every client subscribed to each. What
//! goes out mirrors the Stream Engine: the gaze point is the device's
//! filtered combined point, unclamped; gaze origins are in the display frame
//! and gaze data in the tracker frame, both as the device sends them; values
//! are passed through with their validity flags (the GAZE frame's pupil
//! tail, which has none, sends `NaN` for an eye whose diameter is not
//! valid); raw gaze is the Stream Engine's own record of the gaze frame,
//! every value as sent; the head pose goes out for every image the engine
//! makes one of, valid or not, its four validities all set from the pose's
//! one; every timestamp is the sample's host time (the engine's `host_us`,
//! on [`tobii_ipc::host_clock_us`]), bar gaze data's tracker time and raw
//! gaze's time, which stay the device clock. Gaze data's system time is
//! that tracker time on the host clock, as in the Stream Engine, not the
//! time the frame was read. HEAD, the daemon's own, legacy head pose, is
//! retired: it is never sent (see [`engine_wanted`]).

use tobii_ipc::{
    EyePair, EyePoint, GazeData, GazeDataEye, GazeRaw, GazeRawEye, HeadPose, Notification,
    NotificationValue, PRESENCE_AWAY, PRESENCE_PRESENT, STREAM_EYE_POSITION, STREAM_GAZE,
    STREAM_GAZE_DATA, STREAM_GAZE_ORIGIN, STREAM_GAZE_RAW, STREAM_HEAD_POSE, STREAM_IMAGE,
    STREAM_NOTIFICATIONS, STREAM_PRESENCE, encode_eye_position, encode_gaze, encode_gaze_data,
    encode_gaze_origin, encode_gaze_raw, encode_head_pose, encode_image, encode_notification,
    encode_presence, notification,
};
use tobii_proto::facts::DeviceNotification;
use tobii_proto::gaze83::{EyeFrame, GazeFrame, Valued};
use tobii_usb::engine::{GazeSample, PoseSample, PresenceSample, Sample, Wanted};

/// Narrow to the wire's `f32`: the C ABI this protocol feeds carries `float`.
#[allow(clippy::cast_possible_truncation)] // reason: f32 is the wire type
fn f32s<const N: usize>(v: [f64; N]) -> [f32; N] {
    v.map(|c| c as f32)
}

/// Device microseconds as the `i64` the wire carries.
fn ts(us: u64) -> i64 {
    i64::try_from(us).unwrap_or(i64::MAX)
}

fn point(v: Valued<[f64; 3]>) -> EyePoint {
    EyePoint {
        valid: v.valid,
        xyz: f32s(v.value),
    }
}

/// The per-eye fields of `tobii_gaze_data_t`. The device reports the 3-D
/// points in the tracker frame, the Stream Engine's frame for them, so they
/// pass through as they are, whatever the display area. The pupil diameter
/// is valid while the eye's status is below 2, as in the Stream Engine.
fn gaze_data_eye(eye: &EyeFrame) -> GazeDataEye {
    let [pupil] = f32s([eye.pupil_diameter_mm.value]);
    GazeDataEye {
        gaze_origin_valid: eye.origin_tracker_mm.valid,
        gaze_origin_mm: f32s(eye.origin_tracker_mm.value),
        gaze_origin_in_track_box: f32s(eye.track_box.value),
        gaze_point_valid: eye.gaze_point_tracker_mm.valid,
        gaze_point_mm: f32s(eye.gaze_point_tracker_mm.value),
        gaze_point_on_display: f32s(eye.gaze_point_norm.value),
        eyeball_center_valid: eye.eyeball_center_mm.valid,
        eyeball_center_mm: f32s(eye.eyeball_center_mm.value),
        pupil_valid: eye.pupil_diameter_mm.valid,
        pupil_diameter_mm: pupil,
    }
}

/// One eye's block of the raw gaze record: its values as sent, whatever
/// their validity, and 0 for a key the frame lacks, as in the record's
/// zeroed memory (0x18018dadd).
fn gaze_raw_eye(eye: &EyeFrame) -> GazeRawEye {
    let [pupil] = f32s([eye.pupil_diameter_mm.value]);
    GazeRawEye {
        gaze_origin_mm: f32s(eye.origin_tracker_mm.value),
        gaze_origin_in_track_box: f32s(eye.track_box.value),
        gaze_point_mm: f32s(eye.gaze_point_tracker_mm.value),
        gaze_point_on_display: f32s(eye.gaze_point_norm.value),
        pupil_diameter_mm: pupil,
        status: eye.status.unwrap_or(0),
    }
}

/// The Stream Engine's raw gaze record of `frame` (`tobii_gaze_raw_t`, which
/// `process_gaze`, 0x18018d9a0, fills): both eyes, the combined gaze point
/// and what [`GazeFrame::record`] keeps, each value as sent, stamped with
/// the device clock, as the record is. An eyeball centre is there only when
/// the frame sent it, as the record flags it.
fn gaze_raw(frame: &GazeFrame) -> GazeRaw {
    let record = &frame.record;
    let eyeball = |sent: bool, eye: &EyeFrame| sent.then(|| f32s(eye.eyeball_center_mm.value));
    GazeRaw {
        timestamp_tracker_us: ts(frame.device_ts_us),
        left: gaze_raw_eye(&frame.left),
        right: gaze_raw_eye(&frame.right),
        combined_gaze_point_on_display: f32s(frame.gaze.value),
        combined_gaze_validity: record.combined_gaze_validity.unwrap_or(0),
        key_0e: record.key_0e,
        key_11: record.key_11,
        frame_counter: record.frame_counter,
        left_origin_flag: record.left_origin_flag,
        right_origin_flag: record.right_origin_flag,
        left_eyeball_center_mm: eyeball(record.left_eyeball_sent, &frame.left),
        right_eyeball_center_mm: eyeball(record.right_eyeball_sent, &frame.right),
    }
}

/// The GAZE frame's pupil tail: each eye's diameter, `NaN` when not valid.
fn pupil_tail(frame: &GazeFrame) -> [f32; 2] {
    let mm = |p: Valued<f64>| if p.valid { p.value } else { f64::NAN };
    f32s([
        mm(frame.left.pupil_diameter_mm),
        mm(frame.right.pupil_diameter_mm),
    ])
}

fn gaze_frames(g: &GazeSample, wanted: u32, out: &mut Vec<(u32, Vec<u8>)>) {
    let frame: &GazeFrame = &g.frame;
    if wanted & STREAM_GAZE != 0 {
        out.push((
            STREAM_GAZE,
            encode_gaze(
                g.host_us,
                frame.gaze.valid,
                f32s(frame.gaze.value),
                pupil_tail(frame),
            ),
        ));
    }
    if wanted & STREAM_GAZE_ORIGIN != 0 {
        out.push((
            STREAM_GAZE_ORIGIN,
            encode_gaze_origin(&EyePair {
                ts_us: g.host_us,
                left: point(frame.left.origin_display_mm),
                right: point(frame.right.origin_display_mm),
            }),
        ));
    }
    if wanted & STREAM_EYE_POSITION != 0 {
        out.push((
            STREAM_EYE_POSITION,
            encode_eye_position(&EyePair {
                ts_us: g.host_us,
                left: point(frame.left.track_box),
                right: point(frame.right.track_box),
            }),
        ));
    }
    if wanted & STREAM_GAZE_DATA != 0 {
        out.push((
            STREAM_GAZE_DATA,
            encode_gaze_data(&GazeData {
                timestamp_tracker_us: ts(frame.device_ts_us),
                timestamp_system_us: g.host_us,
                left: gaze_data_eye(&frame.left),
                right: gaze_data_eye(&frame.right),
            }),
        ));
    }
    if wanted & STREAM_GAZE_RAW != 0 {
        out.push((STREAM_GAZE_RAW, encode_gaze_raw(&gaze_raw(frame))));
    }
}

/// The work the engine is to do for the streams in `wanted` (see
/// [`Engine::set_wanted`](tobii_usb::engine::Engine::set_wanted)): its
/// head-pose inference, for the Stream Engine's head pose of every image
/// ([`STREAM_HEAD_POSE`]), and the IR images for [`STREAM_IMAGE`]. Nothing for
/// HEAD ([`crate::daemon::RETIRED_STREAM_HEAD`], bit 0), the daemon's own,
/// legacy head pose, which is retired: a client from before that still
/// subscribes to it is acked and sent nothing for it, as for a stream bit
/// the daemon does not know.
#[must_use]
pub(crate) fn engine_wanted(wanted: u32) -> Wanted {
    Wanted {
        head: wanted & STREAM_HEAD_POSE != 0,
        image: wanted & STREAM_IMAGE != 0,
    }
}

/// A pose sample's `HEAD_POSE` frame, the Stream Engine's head pose, while
/// `wanted` has it: one for every sample, stamped with the host time of the
/// image the pose was made of, its four validities all set from the pose's
/// one.
fn pose_frame(p: &PoseSample, wanted: u32, out: &mut Vec<(u32, Vec<u8>)>) {
    if wanted & STREAM_HEAD_POSE == 0 {
        return;
    }
    let head = &p.head;
    out.push((
        STREAM_HEAD_POSE,
        encode_head_pose(&HeadPose {
            ts_us: p.host_us,
            position_valid: head.valid,
            position_mm: f32s(head.position_mm),
            rotation_valid: [head.valid; 3],
            rotation_rad: f32s(head.rotation_rad),
        }),
    ));
}

/// A presence sample as its PRESENCE frame body, stamped with its host time:
/// the frame a new subscriber is replayed is the one the others got.
pub(crate) fn presence_frame(p: &PresenceSample) -> Vec<u8> {
    let status = if p.present {
        PRESENCE_PRESENT
    } else {
        PRESENCE_AWAY
    };
    encode_presence(p.host_us, status)
}

/// A device notification in the Stream Engine's terms; `None` for the ones it
/// has no name for. The fault and warning lists (3200, 3210) go out as
/// strings on every one, changed or not, as the DLL's dispatcher passes
/// each on (0x18016c9a3, 0x18016c9b7) without comparing it with the last;
/// libtobii cuts them to 511 bytes, as the DLL does (0x18016c9ef).
pub(crate) fn notification_of(n: &DeviceNotification) -> Option<Notification> {
    match n {
        DeviceNotification::DisplayAreaChanged(area) => Some(Notification {
            kind: notification::DISPLAY_AREA_CHANGED,
            value: NotificationValue::DisplayArea(*area),
        }),
        DeviceNotification::CalibrationIdChanged(id) => Some(Notification {
            kind: notification::CALIBRATION_ID_CHANGED,
            value: NotificationValue::Uint(*id),
        }),
        DeviceNotification::FaultsChanged(list) => Some(Notification {
            kind: notification::FAULTS_CHANGED,
            value: NotificationValue::String(list.clone()),
        }),
        DeviceNotification::WarningsChanged(list) => Some(Notification {
            kind: notification::WARNINGS_CHANGED,
            value: NotificationValue::String(list.clone()),
        }),
        // The daemon reports the pause itself (see `pause`).
        DeviceNotification::DevicePausedChanged(_) => None,
        _ => None,
    }
}

/// Append a sample's `(stream bit, frame body)` pairs to `out`, encoding only
/// the streams in `wanted`.
pub(crate) fn push_sample_frames(s: &Sample, wanted: u32, out: &mut Vec<(u32, Vec<u8>)>) {
    match s {
        Sample::Pose(p) => pose_frame(p, wanted, out),
        Sample::Gaze(g) => gaze_frames(g, wanted, out),
        Sample::Presence(p) if wanted & STREAM_PRESENCE != 0 => {
            out.push((STREAM_PRESENCE, presence_frame(p)));
        }
        Sample::Image(image) if wanted & STREAM_IMAGE != 0 => {
            let frame = &image.frame;
            let (Ok(width), Ok(height)) = (u32::try_from(frame.width), u32::try_from(frame.height))
            else {
                return;
            };
            out.push((
                STREAM_IMAGE,
                encode_image(image.host_us, width, height, 8, &frame.pixels),
            ));
        }
        Sample::Notification(n) if wanted & STREAM_NOTIFICATIONS != 0 => {
            if let Some(n) = notification_of(n) {
                out.push((STREAM_NOTIFICATIONS, encode_notification(&n)));
            }
        }
        // Filtered-out streams, DeviceReady (daemon state, not a client
        // frame), and kinds a newer engine may add.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tobii_ipc::geometry::tracker_to_display;
    use tobii_ipc::{ServerMsg, TAG_HEAD_POSE, decode_server};
    use tobii_proto::facts::parse_display_area;
    use tobii_proto::gaze83::{RecordKeys, decode_gaze_frame};
    use tobii_proto::image83::ImageFrame;
    use tobii_proto::protocol::{hex_to_bytes, parse_message};
    use tobii_proto::tlv::{KeyedFields, UNITS_PER_MM, keyed_fields};
    use tobii_usb::engine::{HeadPose as EngineHeadPose, ImageSample};

    use crate::daemon::RETIRED_STREAM_HEAD;

    /// The session1 fixture frame's device timestamp.
    const DEVICE_US: i64 = 9_613_320_391;
    /// When the host read that frame.
    const READ_US: i64 = 12_000_008_000;
    /// The frame's device timestamp on the host clock: 6 ms before the read,
    /// as for a read that far above the latency floor.
    const HOST_US: i64 = 12_000_002_000;

    /// A capture from tobii-proto's fixtures.
    fn fixture(name: &str) -> Vec<u8> {
        let path = format!(
            "{}/../tobii-proto/fixtures/{name}.hex",
            env!("CARGO_MANIFEST_DIR")
        );
        hex_to_bytes(&std::fs::read_to_string(path).expect("fixture")).expect("hex")
    }

    fn session1_frame() -> GazeFrame {
        let bytes = fixture("session1-gaze-frame");
        decode_gaze_frame(&parse_message(&bytes).expect("msg")).expect("frame")
    }

    fn session1_sample() -> Sample {
        Sample::Gaze(Box::new(GazeSample::new(
            session1_frame(),
            READ_US,
            HOST_US,
        )))
    }

    /// The raw gaze of `frame`, the one frame its sample gives when raw gaze
    /// alone is wanted.
    fn raw_gaze_of(frame: GazeFrame) -> GazeRaw {
        let sample = Sample::Gaze(Box::new(GazeSample::new(frame, READ_US, HOST_US)));
        match &frames(&sample, STREAM_GAZE_RAW)[..] {
            [(STREAM_GAZE_RAW, ServerMsg::GazeRaw(raw))] => **raw,
            other => panic!("expected raw gaze alone, got {other:?}"),
        }
    }

    /// The session1 frame's keyed fields, as sent.
    fn session1_keys() -> KeyedFields {
        let gaze = fixture("session1-gaze-frame");
        keyed_fields(
            parse_message(&gaze)
                .expect("msg")
                .payload
                .get(2..)
                .expect("keys"),
        )
    }

    /// Keyed 3-D point `k` in mm, as the Stream Engine's f32.
    fn key_point(keys: &KeyedFields, k: u32) -> [f32; 3] {
        f32s(keys.point::<3>(k).expect("key").map(|c| c / UNITS_PER_MM))
    }

    fn frames(sample: &Sample, wanted: u32) -> Vec<(u32, ServerMsg)> {
        let mut out = Vec::new();
        push_sample_frames(sample, wanted, &mut out);
        out.iter()
            .map(|(bit, body)| (*bit, decode_server(body).expect("decodes")))
            .collect()
    }

    /// The gaze point clients get is the Stream Engine's, not a rescaled one
    /// (the old `(x + 1) / 2` squeezed it into the right/bottom half).
    #[test]
    #[allow(clippy::float_cmp)] // reason: the DLL's f32 values are reproduced exactly
    fn gaze_is_what_the_stream_engine_reports() {
        match &frames(&session1_sample(), STREAM_GAZE)[..] {
            [
                (
                    STREAM_GAZE,
                    ServerMsg::Gaze {
                        ts_us, valid, xy, ..
                    },
                ),
            ] => {
                assert_eq!(*ts_us, HOST_US);
                assert!(*valid);
                // The f64 values session1.jsonl recorded, narrowed like the DLL's f32.
                #[allow(clippy::cast_possible_truncation)] // reason: the DLL hands out f32
                let expected = [
                    0.587_291_359_901_428_2_f64 as f32,
                    0.099_273_845_553_398_13_f64 as f32,
                ];
                assert_eq!(*xy, expected);
            }
            other => panic!("unexpected frames {other:?}"),
        }
    }

    #[test]
    fn only_wanted_streams_are_encoded() {
        let sample = session1_sample();
        let bits_of = |wanted| -> Vec<u32> {
            frames(&sample, wanted)
                .iter()
                .map(|(bit, _)| *bit)
                .collect()
        };
        assert!(frames(&sample, STREAM_HEAD_POSE | STREAM_PRESENCE).is_empty());
        let all = frames(
            &sample,
            STREAM_GAZE | STREAM_GAZE_ORIGIN | STREAM_EYE_POSITION | STREAM_GAZE_DATA,
        );
        let bits: Vec<u32> = all.iter().map(|(b, _)| *b).collect();
        assert_eq!(
            bits,
            vec![
                STREAM_GAZE,
                STREAM_GAZE_ORIGIN,
                STREAM_EYE_POSITION,
                STREAM_GAZE_DATA
            ]
        );
        assert_eq!(bits_of(!STREAM_GAZE_RAW), bits, "raw gaze not wanted");
        assert_eq!(bits_of(STREAM_GAZE_RAW), [STREAM_GAZE_RAW]);
        match &all[1].1 {
            ServerMsg::GazeOrigin(pair) => {
                assert!(pair.left.valid && pair.right.valid);
                assert!((pair.left.xyz[0] + 57.900_635).abs() < 1e-4);
            }
            other => panic!("expected gaze origin, got {other:?}"),
        }
        match &all[3].1 {
            ServerMsg::GazeData(data) => {
                assert!(data.left.gaze_point_valid && data.right.gaze_point_valid);
                assert!(data.left.pupil_valid && data.right.pupil_valid);
            }
            other => panic!("expected gaze data, got {other:?}"),
        }
    }

    /// The session1 frame's keys `0x06`/`0x0c`, 16.16, as the DLL's f32.
    #[allow(clippy::cast_possible_truncation)] // reason: exact in f32
    fn session1_pupils() -> [f32; 2] {
        [0x0006_3f53, 0x0005_ff22].map(|raw: i32| (f64::from(raw) / 65_536.0) as f32)
    }

    /// Gaze data's pupil diameters are keys `0x06`/`0x0c` narrowed to f32, as
    /// the Stream Engine hands them out, and the GAZE frame's pupil tail
    /// carries the same values.
    #[test]
    #[allow(clippy::float_cmp)] // reason: 16.16 values narrow to f32 exactly
    fn gaze_data_and_the_gaze_tail_carry_the_pupil_diameters() {
        let wanted = STREAM_GAZE | STREAM_GAZE_DATA;
        let [
            (_, ServerMsg::Gaze { pupil_mm, .. }),
            (_, ServerMsg::GazeData(data)),
        ] = &frames(&session1_sample(), wanted)[..]
        else {
            panic!("the gaze sample did not give its gaze and gaze data");
        };

        let expected = session1_pupils();
        assert_eq!(*pupil_mm, expected);
        assert!(data.left.pupil_valid && data.right.pupil_valid);
        assert_eq!(
            [data.left.pupil_diameter_mm, data.right.pupil_diameter_mm],
            expected
        );
    }

    /// An eye whose pupil is not valid (its status 2 or more) keeps its
    /// diameter in gaze data, marked invalid, as the Stream Engine passes it,
    /// and is `NaN` in the GAZE frame's tail, which has no validity.
    #[test]
    #[allow(clippy::float_cmp)] // reason: 16.16 values narrow to f32 exactly
    fn an_invalid_pupil_is_flagged_in_gaze_data_and_nan_in_the_gaze_tail() {
        let mut frame = session1_frame();
        frame.left.pupil_diameter_mm.valid = false;
        let sample = Sample::Gaze(Box::new(GazeSample::new(frame, READ_US, HOST_US)));

        let [
            (_, ServerMsg::Gaze { pupil_mm, .. }),
            (_, ServerMsg::GazeData(data)),
        ] = &frames(&sample, STREAM_GAZE | STREAM_GAZE_DATA)[..]
        else {
            panic!("the gaze sample did not give its gaze and gaze data");
        };

        let [left, right] = session1_pupils();
        assert!(pupil_mm[0].is_nan());
        assert_eq!(pupil_mm[1], right);
        assert!(!data.left.pupil_valid && data.right.pupil_valid);
        assert_eq!(data.left.pupil_diameter_mm, left);
    }

    /// Gaze data's 3-D gaze point is keys `0x04`/`0x0a` as the device sends
    /// them, valid, with no display area involved: they are already in the
    /// tracker frame. On the display area of the capture, the point sent lies
    /// on the screen where the 2-D gaze point sent with it says; converting it
    /// as if it were in the display frame put it about 190 mm off.
    #[test]
    #[allow(clippy::float_cmp)] // reason: the key's value is passed through, narrowed once
    fn gaze_data_sends_the_3d_gaze_point_as_the_device_does() {
        let keys = session1_keys();
        let (area, _) = parse_display_area(&parse_message(&fixture("init-rsp-1430")).expect("msg"))
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

        let [(STREAM_GAZE_DATA, ServerMsg::GazeData(data))] =
            &frames(&session1_sample(), STREAM_GAZE_DATA)[..]
        else {
            panic!("the gaze sample did not give its gaze data");
        };

        for (eye, k) in [(data.left, 0x04), (data.right, 0x0a)] {
            assert!(eye.gaze_point_valid);
            assert_eq!(eye.gaze_point_mm, key_point(&keys, k));
            let [x, y, z] = tracker_to_display(&area, eye.gaze_point_mm.map(f64::from))
                .expect("a display frame");
            let [u, v] = eye.gaze_point_on_display.map(f64::from);
            assert!(z.abs() < 0.01, "{z} mm off the screen");
            assert!((x / width + 0.5 - u).abs() < 1e-4, "{x} vs {u}");
            assert!((0.5 - y / height - v).abs() < 1e-4, "{y} vs {v}");
        }
    }

    /// Raw gaze is the Stream Engine's record of the frame: each key the
    /// record takes (`process_gaze`, 0x18018d9a0) as the session1 frame sent
    /// it, narrowed to f32 once, in its own field. The 3-D points are in the
    /// tracker frame, as sent, with no display area involved; the stamp is
    /// the device time.
    #[test]
    #[allow(clippy::float_cmp)] // reason: each key's value is passed through, narrowed once
    fn raw_gaze_is_the_stream_engines_record_of_the_frame() {
        let keys = session1_keys();
        let point = |k| key_point(&keys, k);
        let point2 = |k| f32s(keys.point::<2>(k).expect("key").map(|c| c / UNITS_PER_MM));
        let word = |k| keys.scalar(k).and_then(|v| u32::try_from(v).ok());
        let raw_eye = |[origin, track_box, gaze_3d, gaze_2d, pupil, status]: [u32; 6]| {
            let [pupil] = f32s([keys.fixed16(pupil).expect("pupil")]);
            GazeRawEye {
                gaze_origin_mm: point(origin),
                gaze_origin_in_track_box: point(track_box),
                gaze_point_mm: point(gaze_3d),
                gaze_point_on_display: point2(gaze_2d),
                pupil_diameter_mm: pupil,
                status: word(status).expect("status"),
            }
        };

        let raw = raw_gaze_of(session1_frame());

        assert_eq!(raw.timestamp_tracker_us, DEVICE_US);
        assert_eq!(raw.left, raw_eye([0x02, 0x03, 0x04, 0x05, 0x06, 0x07]));
        assert_eq!(raw.right, raw_eye([0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d]));
        assert_eq!(raw.combined_gaze_point_on_display, point2(0x1c));
        assert_eq!(Some(raw.combined_gaze_validity), word(0x1b));
        // The ET5 sends no 0x0e, and 4 for 0x11.
        assert_eq!(
            [
                raw.key_0e,
                raw.key_11,
                raw.frame_counter,
                raw.left_origin_flag,
                raw.right_origin_flag,
            ],
            [0x0e, 0x11, 0x14, 0x16, 0x15].map(word)
        );
        assert_eq!([raw.key_0e, raw.key_11], [None, Some(4)]);
        assert_eq!(
            [raw.left_eyeball_center_mm, raw.right_eyeball_center_mm],
            [Some(point(0x17)), Some(point(0x18))]
        );
    }

    /// Raw gaze passes a value whatever its validity, a word as sent rather
    /// than the validity derived from it, and a key the frame lacks as the
    /// record holds it: 0 where the record has no flag for it, `None` where
    /// it has one, told apart from a 0 that was sent. An eyeball centre is
    /// there when the frame sent it, lost eye or not.
    #[test]
    #[allow(clippy::float_cmp)] // reason: values are passed through, narrowed once
    fn raw_gaze_passes_what_was_sent_and_zeroes_what_was_not() {
        let mut frame = session1_frame();
        // A lost left eye, whose values the device still sends.
        frame.left.tracked = false;
        frame.left.status = Some(4);
        frame.left.origin_tracker_mm.valid = false;
        frame.left.track_box.valid = false;
        frame.left.gaze_point_tracker_mm.valid = false;
        frame.left.gaze_point_norm.valid = false;
        frame.left.eyeball_center_mm.valid = false;
        frame.left.pupil_diameter_mm.valid = false;
        frame.gaze.valid = false;
        // The right eye's status and pupil, which the ET5 always sends,
        // missing, and 0x0e, which it never sends, present.
        frame.right.status = None;
        frame.right.pupil_diameter_mm = Valued::default();
        frame.record = RecordKeys {
            // Not 1, so the point is not valid.
            combined_gaze_validity: Some(2),
            key_0e: Some(9),
            key_11: None,
            frame_counter: None,
            // Told apart, so the eyes' flags (0x16 left, 0x15 right) cannot
            // swap unseen.
            left_origin_flag: Some(0),
            right_origin_flag: Some(1),
            left_eyeball_sent: true,
            right_eyeball_sent: false,
        };

        let raw = raw_gaze_of(frame);

        assert_eq!([raw.left.status, raw.right.status], [4, 0]);
        assert_eq!(
            [
                raw.left.gaze_origin_mm,
                raw.left.gaze_origin_in_track_box,
                raw.left.gaze_point_mm,
            ],
            [
                f32s(frame.left.origin_tracker_mm.value),
                f32s(frame.left.track_box.value),
                f32s(frame.left.gaze_point_tracker_mm.value),
            ]
        );
        assert_eq!(
            raw.left.gaze_point_on_display,
            f32s(frame.left.gaze_point_norm.value)
        );
        assert_eq!(
            [raw.left.pupil_diameter_mm, raw.right.pupil_diameter_mm],
            [session1_pupils()[0], 0.0]
        );
        assert_eq!(raw.combined_gaze_point_on_display, f32s(frame.gaze.value));
        assert_eq!(raw.combined_gaze_validity, 2);
        assert_eq!(
            [raw.key_0e, raw.key_11, raw.frame_counter],
            [Some(9), None, None]
        );
        assert_eq!(
            [raw.left_origin_flag, raw.right_origin_flag],
            [Some(0), Some(1)]
        );
        assert_eq!(
            [raw.left_eyeball_center_mm, raw.right_eyeball_center_mm],
            [Some(f32s(frame.left.eyeball_center_mm.value)), None]
        );

        frame.record.combined_gaze_validity = None;
        assert_eq!(raw_gaze_of(frame).combined_gaze_validity, 0);
    }

    /// Every sample frame carries the sample's host time, as the Stream
    /// Engine's callbacks do. Gaze data keeps the device time beside it, and
    /// its system time is that device time on the host clock, not the read.
    /// Raw gaze, the exception, carries the device time alone, as the Stream
    /// Engine's raw gaze record does.
    #[test]
    fn sample_frames_carry_their_host_time_bar_raw_gaze() {
        let wanted = STREAM_GAZE
            | STREAM_GAZE_ORIGIN
            | STREAM_EYE_POSITION
            | STREAM_GAZE_DATA
            | STREAM_GAZE_RAW
            | STREAM_HEAD_POSE
            | STREAM_PRESENCE
            | STREAM_IMAGE;
        let [
            (_, ServerMsg::Gaze { ts_us, .. }),
            (_, ServerMsg::GazeOrigin(origin)),
            (_, ServerMsg::EyePosition(eyes)),
            (_, ServerMsg::GazeData(data)),
            (_, ServerMsg::GazeRaw(raw)),
        ] = &frames(&session1_sample(), wanted)[..]
        else {
            panic!("the gaze sample did not give its five frames");
        };
        assert_eq!([*ts_us, origin.ts_us, eyes.ts_us], [HOST_US; 3]);
        assert_eq!(
            (data.timestamp_tracker_us, data.timestamp_system_us),
            (DEVICE_US, HOST_US)
        );
        assert_eq!(raw.timestamp_tracker_us, DEVICE_US);

        let image = ImageFrame {
            device_ts_us: DEVICE_US.unsigned_abs(),
            width: 2,
            height: 2,
            pixels: vec![0; 4],
        };
        let others = [
            pose_sample(EngineHeadPose::default()),
            Sample::Presence(PresenceSample::new(DEVICE_US, HOST_US, true)),
            Sample::Image(ImageSample::new(Arc::new(image), HOST_US)),
        ];
        let stamps: Vec<i64> = others
            .iter()
            .flat_map(|s| frames(s, wanted))
            .map(|(_, msg)| match msg {
                ServerMsg::HeadPose(pose) => pose.ts_us,
                ServerMsg::Presence { ts_us, .. } => ts_us,
                ServerMsg::Image(image) => image.ts_us,
                other => panic!("unexpected frame {other:?}"),
            })
            .collect();
        assert_eq!(stamps, [HOST_US; 3], "head pose, presence and image");
    }

    /// The tracker's fault and warning lists reach notification subscribers
    /// as strings, as `TOBII_NOTIFICATION_TYPE_FAULTS_CHANGED` (10) and
    /// `_WARNINGS_CHANGED` (11).
    #[test]
    fn fault_and_warning_lists_reach_notification_subscribers() {
        let string = |kind, text: &str| {
            (
                STREAM_NOTIFICATIONS,
                ServerMsg::Notification(Notification {
                    kind,
                    value: NotificationValue::String(text.into()),
                }),
            )
        };
        let faults = Sample::Notification(DeviceNotification::FaultsChanged("FAULT_A".into()));
        let warnings = Sample::Notification(DeviceNotification::WarningsChanged("ok".into()));

        assert_eq!(
            frames(&faults, STREAM_NOTIFICATIONS),
            [string(10, "FAULT_A")]
        );
        assert_eq!(frames(&warnings, STREAM_NOTIFICATIONS), [string(11, "ok")]);
        assert!(frames(&faults, 0).is_empty(), "nobody subscribed");
    }

    /// A pose sample of the image of [`DEVICE_US`], [`HOST_US`] on the host
    /// clock.
    fn pose_sample(head: EngineHeadPose) -> Sample {
        Sample::Pose(PoseSample::new(DEVICE_US, HOST_US, head))
    }

    /// A valid head pose, its values exact in f32.
    fn valid_head() -> EngineHeadPose {
        EngineHeadPose {
            valid: true,
            position_mm: [-12.5, 30.25, 600.0],
            rotation_rad: [0.125, -0.25, 0.5],
        }
    }

    /// The pose's frames, `(bit, tag)`, of the streams in `wanted`.
    fn pose_tags(sample: &Sample, wanted: u32) -> Vec<(u32, u8)> {
        let mut out = Vec::new();
        push_sample_frames(sample, wanted, &mut out);
        out.iter().map(|(bit, body)| (*bit, body[0])).collect()
    }

    /// A valid pose goes out as `HEAD_POSE` with its four validities set,
    /// an invalid one with none, each with the values it carries (an
    /// invalid one those of the last valid pose) and the host time of its
    /// image.
    #[test]
    fn a_valid_and_an_invalid_pose_go_out_as_head_pose_with_their_validity() {
        let invalid = EngineHeadPose {
            valid: false,
            ..valid_head()
        };
        let sent = |head| match &frames(&pose_sample(head), STREAM_HEAD_POSE)[..] {
            [(STREAM_HEAD_POSE, ServerMsg::HeadPose(pose))] => *pose,
            other => panic!("expected a HEAD_POSE frame alone, got {other:?}"),
        };
        let on_the_wire = |valid| HeadPose {
            ts_us: HOST_US,
            position_valid: valid,
            position_mm: [-12.5, 30.25, 600.0],
            rotation_valid: [valid; 3],
            rotation_rad: [0.125, -0.25, 0.5],
        };

        assert_eq!(sent(valid_head()), on_the_wire(true));
        assert_eq!(sent(invalid), on_the_wire(false));
        assert_eq!(
            sent(EngineHeadPose::default()),
            HeadPose {
                ts_us: HOST_US,
                ..HeadPose::default()
            },
            "invalid before the first valid pose: zeros"
        );
    }

    /// HEAD, the retired legacy pose, is never sent: a pose sample, valid or
    /// not, gives a client of HEAD (bit 0) alone nothing, and one of HEAD
    /// and `HEAD_POSE`, or of every stream, the `HEAD_POSE` frame alone.
    /// That frame goes out under its own bit only while some client wants
    /// it, by which the pump writes it only to the clients that subscribe
    /// to it.
    #[test]
    fn a_pose_sample_goes_out_as_head_pose_alone_and_only_when_wanted() {
        let invalid = EngineHeadPose {
            valid: false,
            ..valid_head()
        };
        let head_pose = [(STREAM_HEAD_POSE, TAG_HEAD_POSE)];

        for head in [valid_head(), invalid] {
            let sample = pose_sample(head);

            assert_eq!(pose_tags(&sample, RETIRED_STREAM_HEAD), [], "{head:?}");
            assert_eq!(
                pose_tags(&sample, RETIRED_STREAM_HEAD | STREAM_HEAD_POSE),
                head_pose
            );
            assert_eq!(pose_tags(&sample, STREAM_HEAD_POSE), head_pose);
            assert_eq!(pose_tags(&sample, !0), head_pose, "every stream");
            assert_eq!(pose_tags(&sample, !STREAM_HEAD_POSE), [], "every other");
        }
    }

    /// The engine is asked for head-pose inference for `HEAD_POSE`, for the
    /// IR images for IMAGE, and for nothing for any other stream: not for
    /// HEAD (bit 0), the retired legacy pose, alone or beside the others.
    #[test]
    fn the_engine_is_asked_for_what_the_streams_take() {
        let ours = STREAM_HEAD_POSE | STREAM_IMAGE;
        let wanted = |head, image| Wanted { head, image };
        for (mask, want) in [
            (0, wanted(false, false)),
            (RETIRED_STREAM_HEAD, wanted(false, false)),
            (STREAM_HEAD_POSE, wanted(true, false)),
            (RETIRED_STREAM_HEAD | STREAM_HEAD_POSE, wanted(true, false)),
            (STREAM_IMAGE, wanted(false, true)),
            (STREAM_HEAD_POSE | STREAM_IMAGE, wanted(true, true)),
        ] {
            assert_eq!(engine_wanted(mask), want, "{mask:#x}");
            assert_eq!(engine_wanted(mask | !ours), want, "{mask:#x} and the rest");
        }
        let alone: Vec<u32> = (0..32)
            .map(|bit| 1 << bit)
            .filter(|&bit| engine_wanted(bit) != Wanted::default())
            .collect();
        assert_eq!(alone, [STREAM_IMAGE, STREAM_HEAD_POSE]);
    }
}
