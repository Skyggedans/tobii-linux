//! Engine samples -> IPC frames.
//!
//! Each sample is encoded once, and only into the streams some client wants;
//! the pump then writes the bodies to every client subscribed to each. What
//! goes out mirrors the Stream Engine: the gaze point is the device's
//! filtered combined point, unclamped; gaze origins are in the display frame
//! and gaze data in the tracker frame, both as the device sends them; values
//! are passed through with their validity flags (the GAZE frame's pupil
//! tail, which has none, sends `NaN` for an eye whose diameter is not
//! valid); every timestamp is the sample's host time (the engine's
//! `host_us`, on [`tobii_ipc::host_clock_us`]), bar gaze data's tracker
//! time, which stays the device clock. Gaze data's system time is that
//! tracker time on the host clock, as in the Stream Engine, not the time the
//! frame was read.

use tobii_ipc::{
    EyePair, EyePoint, GazeData, GazeDataEye, Notification, NotificationValue, PRESENCE_AWAY,
    PRESENCE_PRESENT, STREAM_EYE_POSITION, STREAM_GAZE, STREAM_GAZE_DATA, STREAM_GAZE_ORIGIN,
    STREAM_HEAD, STREAM_IMAGE, STREAM_NOTIFICATIONS, STREAM_PRESENCE, encode_eye_position,
    encode_gaze, encode_gaze_data, encode_gaze_origin, encode_head, encode_image,
    encode_notification, encode_presence, notification,
};
use tobii_proto::facts::DeviceNotification;
use tobii_proto::gaze83::{EyeFrame, GazeFrame, Valued};
use tobii_usb::engine::{GazeSample, PresenceSample, Sample};

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
/// has no name for.
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
        // The daemon reports the pause itself (see `pause`).
        DeviceNotification::DevicePausedChanged(_) => None,
        _ => None,
    }
}

/// Append a sample's `(stream bit, frame body)` pairs to `out`, encoding only
/// the streams in `wanted`.
pub(crate) fn push_sample_frames(s: &Sample, wanted: u32, out: &mut Vec<(u32, Vec<u8>)>) {
    match s {
        Sample::Pose(p) if wanted & STREAM_HEAD != 0 => {
            // cm -> mm; rotation about x=pitch, y=yaw, z=roll in radians.
            let pos = f32s(p.pos_cm.map(|c| c * 10.0));
            let rot = f32s([
                p.rot_deg[1].to_radians(),
                p.rot_deg[0].to_radians(),
                p.rot_deg[2].to_radians(),
            ]);
            out.push((STREAM_HEAD, encode_head(p.host_us, pos, rot)));
        }
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
    use tobii_ipc::{ServerMsg, decode_server};
    use tobii_proto::facts::parse_display_area;
    use tobii_proto::gaze83::decode_gaze_frame;
    use tobii_proto::image83::ImageFrame;
    use tobii_proto::protocol::{hex_to_bytes, parse_message};
    use tobii_proto::tlv::{UNITS_PER_MM, keyed_fields};
    use tobii_usb::engine::{ImageSample, PoseSample};

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

    fn session1_sample() -> Sample {
        let bytes = fixture("session1-gaze-frame");
        let frame = decode_gaze_frame(&parse_message(&bytes).expect("msg")).expect("frame");
        Sample::Gaze(Box::new(GazeSample::new(frame, READ_US, HOST_US)))
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
        assert!(frames(&sample, STREAM_HEAD | STREAM_PRESENCE).is_empty());
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
        let bytes = fixture("session1-gaze-frame");
        let mut frame = decode_gaze_frame(&parse_message(&bytes).expect("msg")).expect("frame");
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
        let gaze = fixture("session1-gaze-frame");
        let keys = keyed_fields(
            parse_message(&gaze)
                .expect("msg")
                .payload
                .get(2..)
                .expect("keys"),
        );
        let key = |k: u32| f32s(keys.point::<3>(k).expect("key").map(|c| c / UNITS_PER_MM));
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
            assert_eq!(eye.gaze_point_mm, key(k));
            let [x, y, z] = tracker_to_display(&area, eye.gaze_point_mm.map(f64::from));
            let [u, v] = eye.gaze_point_on_display.map(f64::from);
            assert!(z.abs() < 0.01, "{z} mm off the screen");
            assert!((x / width + 0.5 - u).abs() < 1e-4, "{x} vs {u}");
            assert!((0.5 - y / height - v).abs() < 1e-4, "{y} vs {v}");
        }
    }

    /// Every sample frame carries the sample's host time, as the Stream
    /// Engine's callbacks do. Gaze data keeps the device time beside it, and
    /// its system time is that device time on the host clock, not the read.
    #[test]
    fn every_sample_frame_carries_its_host_time() {
        let wanted = STREAM_GAZE
            | STREAM_GAZE_ORIGIN
            | STREAM_EYE_POSITION
            | STREAM_GAZE_DATA
            | STREAM_HEAD
            | STREAM_PRESENCE
            | STREAM_IMAGE;
        let [
            (_, ServerMsg::Gaze { ts_us, .. }),
            (_, ServerMsg::GazeOrigin(origin)),
            (_, ServerMsg::EyePosition(eyes)),
            (_, ServerMsg::GazeData(data)),
        ] = &frames(&session1_sample(), wanted)[..]
        else {
            panic!("the gaze sample did not give its four frames");
        };
        assert_eq!([*ts_us, origin.ts_us, eyes.ts_us], [HOST_US; 3]);
        assert_eq!(
            (data.timestamp_tracker_us, data.timestamp_system_us),
            (DEVICE_US, HOST_US)
        );

        let image = ImageFrame {
            device_ts_us: DEVICE_US.unsigned_abs(),
            width: 2,
            height: 2,
            pixels: vec![0; 4],
        };
        let others = [
            Sample::Pose(PoseSample::new(DEVICE_US, HOST_US, [0.0; 3], [0.0; 3])),
            Sample::Presence(PresenceSample::new(DEVICE_US, HOST_US, true)),
            Sample::Image(ImageSample::new(Arc::new(image), HOST_US)),
        ];
        let stamps: Vec<i64> = others
            .iter()
            .flat_map(|s| frames(s, wanted))
            .map(|(_, msg)| match msg {
                ServerMsg::Head { ts_us, .. } | ServerMsg::Presence { ts_us, .. } => ts_us,
                ServerMsg::Image(image) => image.ts_us,
                other => panic!("unexpected frame {other:?}"),
            })
            .collect();
        assert_eq!(stamps, [HOST_US; 3], "head, presence and image");
    }
}
