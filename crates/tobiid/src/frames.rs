//! Engine samples -> IPC frames.
//!
//! Each sample is encoded once, and only into the streams some client wants;
//! the pump then writes the bodies to every client subscribed to each. What
//! goes out mirrors the Stream Engine: the gaze point is the device's
//! filtered combined point, unclamped; gaze origins are in the display frame;
//! values are passed through with their validity flags; every timestamp is
//! the sample's host time (the engine's `host_us`, on
//! [`tobii_ipc::host_clock_us`]), bar gaze data's tracker time, which stays
//! the device clock. Gaze data's system time is that tracker time on the
//! host clock, as in the Stream Engine, not the time the frame was read.

use tobii_ipc::geometry::{DisplayArea, display_to_tracker};
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

/// The per-eye fields of `tobii_gaze_data_t`. The device reports the 3-D gaze
/// point in the display frame; the Stream Engine gives it in the tracker
/// frame, which needs the display area to convert — without one the point is
/// reported invalid.
fn gaze_data_eye(eye: &EyeFrame, display: Option<&DisplayArea>) -> GazeDataEye {
    let gaze_point = display.map(|area| display_to_tracker(area, eye.gaze_point_display_mm.value));
    GazeDataEye {
        gaze_origin_valid: eye.origin_tracker_mm.valid,
        gaze_origin_mm: f32s(eye.origin_tracker_mm.value),
        gaze_origin_in_track_box: f32s(eye.track_box.value),
        gaze_point_valid: eye.gaze_point_display_mm.valid && gaze_point.is_some(),
        gaze_point_mm: f32s(gaze_point.unwrap_or_default()),
        gaze_point_on_display: f32s(eye.gaze_point_norm.value),
        eyeball_center_valid: eye.eyeball_center_mm.valid,
        eyeball_center_mm: f32s(eye.eyeball_center_mm.value),
        // The ET5 wire carries no pupil diameter (see tobii_proto::gaze83).
        pupil_valid: false,
        pupil_diameter_mm: 0.0,
    }
}

fn gaze_frames(
    g: &GazeSample,
    wanted: u32,
    display: Option<&DisplayArea>,
    out: &mut Vec<(u32, Vec<u8>)>,
) {
    let frame: &GazeFrame = &g.frame;
    if wanted & STREAM_GAZE != 0 {
        out.push((
            STREAM_GAZE,
            encode_gaze(
                g.host_us,
                frame.gaze.valid,
                f32s(frame.gaze.value),
                [f32::NAN; 2],
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
                left: gaze_data_eye(&frame.left, display),
                right: gaze_data_eye(&frame.right, display),
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
pub(crate) fn push_sample_frames(
    s: &Sample,
    wanted: u32,
    display: Option<&DisplayArea>,
    out: &mut Vec<(u32, Vec<u8>)>,
) {
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
        Sample::Gaze(g) => gaze_frames(g, wanted, display, out),
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
    use tobii_ipc::{ServerMsg, decode_server};
    use tobii_proto::gaze83::decode_gaze_frame;
    use tobii_proto::image83::ImageFrame;
    use tobii_proto::protocol::{hex_to_bytes, parse_message};
    use tobii_usb::engine::{ImageSample, PoseSample};

    /// The session1 fixture frame's device timestamp.
    const DEVICE_US: i64 = 9_613_320_391;
    /// When the host read that frame.
    const READ_US: i64 = 12_000_008_000;
    /// The frame's device timestamp on the host clock: 6 ms before the read,
    /// as for a read that far above the latency floor.
    const HOST_US: i64 = 12_000_002_000;

    fn session1_sample() -> Sample {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tobii-proto/fixtures/session1-gaze-frame.hex"
        );
        let bytes = hex_to_bytes(&std::fs::read_to_string(path).expect("fixture")).expect("hex");
        let frame = decode_gaze_frame(&parse_message(&bytes).expect("msg")).expect("frame");
        Sample::Gaze(Box::new(GazeSample::new(frame, READ_US, HOST_US)))
    }

    fn frames(sample: &Sample, wanted: u32) -> Vec<(u32, ServerMsg)> {
        let mut out = Vec::new();
        push_sample_frames(sample, wanted, None, &mut out);
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
                assert!(
                    !data.left.gaze_point_valid,
                    "no display area, no tracker-frame gaze point"
                );
                assert!(!data.left.pupil_valid);
            }
            other => panic!("expected gaze data, got {other:?}"),
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
