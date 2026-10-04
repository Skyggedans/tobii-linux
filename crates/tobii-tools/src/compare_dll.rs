//! `compare-dll`: replay a captured session through the gaze decoder the
//! daemon uses and compare it with what the Windows Stream Engine delivered
//! for the same frames (its callbacks were logged to a JSONL file while the
//! USB traffic was captured). `--head` replays the session's IR images
//! through the head pose instead ([`crate::compare_head`]).
//!
//! The DLL's timestamps are the device clock minus a per-session constant,
//! so frames are paired by timestamp once that constant is known; it is taken
//! from the frames whose gaze point matches exactly ([`clock_offset`]).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, ensure};
use tobii_proto::gaze83::{GazeFrame, decode_gaze_frame};
use tobii_proto::log::read_log_payloads;
use tobii_proto::protocol::{BulkReassembler, parse_message};

/// One record of the DLL log that the comparisons use: a `gazePoint`, a
/// `gazeOrigin` or a `headPose`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DllRecord {
    /// A `gazePoint`.
    GazePoint {
        /// Its timestamp, µs.
        ts_us: i64,
        /// Its validity.
        valid: bool,
        /// The point, normalised display coordinates.
        xy: [f32; 2],
    },
    /// A `gazeOrigin`.
    GazeOrigin {
        /// Its timestamp, µs.
        ts_us: i64,
        /// The left eye's validity and origin, display frame, mm.
        left: (bool, [f32; 3]),
        /// The right eye's.
        right: (bool, [f32; 3]),
    },
    /// A `headPose`.
    HeadPose(DllHeadPose),
}

/// One `headPose` record of the DLL log: the Stream Engine's
/// `tobii_head_pose_t` as its callback received it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DllHeadPose {
    /// The pose's timestamp, µs: the device time of the IR image it was
    /// made of, less the session's clock offset.
    pub(crate) ts_us: i64,
    /// `position_validity`.
    pub(crate) position_valid: bool,
    /// `rotation_validity_xyz`.
    pub(crate) rotation_valid: [bool; 3],
    /// `position_xyz`: the display frame, mm.
    pub(crate) position_mm: [f64; 3],
    /// `rotation_xyz`: yxz Euler angles in the display frame, radians.
    pub(crate) rotation_rad: [f64; 3],
}

impl DllHeadPose {
    /// Whether the pose is valid: all four of its validity flags are set.
    /// The Stream Engine sets all four or none ([`Self::flags_agree`]).
    pub(crate) fn is_valid(&self) -> bool {
        self.position_valid && self.rotation_valid.iter().all(|&v| v)
    }

    /// Whether the four validity flags agree.
    pub(crate) fn flags_agree(&self) -> bool {
        self.rotation_valid
            .iter()
            .all(|&v| v == self.position_valid)
    }
}

/// The number following `"key":` in `line`.
fn number_after(line: &str, key: &str) -> Option<f64> {
    let start = line.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = &line[start..];
    let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// The numbers in the array following `"key":[`.
fn array_after(line: &str, key: &str) -> Option<Vec<f64>> {
    let start = line.find(&format!("\"{key}\":["))? + key.len() + 4;
    let rest = &line[start..];
    let end = rest.find(']')?;
    rest[..end]
        .split(',')
        .map(|v| v.trim().parse().ok())
        .collect()
}

/// The three numbers in the array following `"key":[`.
fn three_after(line: &str, key: &str) -> Option<[f64; 3]> {
    <[f64; 3]>::try_from(array_after(line, key)?).ok()
}

#[allow(clippy::cast_possible_truncation)] // reason: the DLL logged f32 values and i64 timestamps
fn parse_record(line: &str) -> Option<DllRecord> {
    let f32s = |v: Vec<f64>| v.into_iter().map(|c| c as f32).collect::<Vec<f32>>();
    if line.contains("\"gazePoint\"") {
        let xy = f32s(array_after(line, "position_xy")?);
        Some(DllRecord::GazePoint {
            ts_us: number_after(line, "timestamp_us")? as i64,
            valid: number_after(line, "validity")? != 0.0,
            xy: [*xy.first()?, *xy.get(1)?],
        })
    } else if line.contains("\"gazeOrigin\"") {
        let l = f32s(array_after(line, "left_xyz")?);
        let r = f32s(array_after(line, "right_xyz")?);
        Some(DllRecord::GazeOrigin {
            ts_us: number_after(line, "timestamp_us")? as i64,
            left: (
                number_after(line, "left_validity")? != 0.0,
                [l[0], l[1], l[2]],
            ),
            right: (
                number_after(line, "right_validity")? != 0.0,
                [r[0], r[1], r[2]],
            ),
        })
    } else if line.contains("\"headPose\"") {
        // The values stay f64: each is an f32 the DLL handed out, which the
        // log wrote out in full.
        Some(DllRecord::HeadPose(DllHeadPose {
            ts_us: number_after(line, "timestamp_us")? as i64,
            position_valid: number_after(line, "position_validity")? != 0.0,
            rotation_valid: three_after(line, "rotation_validity_xyz")?.map(|v| v != 0.0),
            position_mm: three_after(line, "position_xyz")?,
            rotation_rad: three_after(line, "rotation_xyz")?,
        }))
    } else {
        None
    }
}

/// Every record of the DLL log at `jsonl_path` that the comparisons use, in
/// log order. Reading stops at the first line that cannot be read.
///
/// # Errors
/// Fails when the file cannot be opened.
pub(crate) fn read_dll_records(jsonl_path: &str) -> Result<Vec<DllRecord>> {
    Ok(BufReader::new(
        File::open(jsonl_path).with_context(|| format!("failed to open {jsonl_path}"))?,
    )
    .lines()
    .map_while(Result::ok)
    .filter_map(|l| parse_record(&l))
    .collect())
}

#[allow(clippy::cast_possible_truncation)] // reason: compare as the f32 the DLL hands out
fn f32s<const N: usize>(v: [f64; N]) -> [f32; N] {
    v.map(|c| c as f32)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Decode every gaze frame in a TBI5LOG1 log, keyed by device timestamp.
fn frames(log_path: &str) -> Result<HashMap<i64, GazeFrame>> {
    let mut asm = BulkReassembler::new();
    let mut msgs = Vec::new();
    let mut out = HashMap::new();
    for record in read_log_payloads(log_path)? {
        asm.push_into(&record, &mut msgs);
        for msg in &msgs {
            if let Some(frame) = parse_message(msg).as_ref().and_then(decode_gaze_frame) {
                out.insert(i64::try_from(frame.device_ts_us)?, frame);
            }
        }
    }
    Ok(out)
}

/// A session's clock offset: device time less DLL time, µs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClockOffset {
    /// The offset: the median of the matches' offsets.
    pub(crate) k_us: i64,
    /// The matches whose offset is exactly `k_us`.
    pub(crate) agreeing: usize,
    /// The DLL's valid gaze points that match a decoded frame's.
    pub(crate) matches: usize,
}

/// The clock offset of a session, from the DLL's valid gaze points that
/// match a decoded gaze frame's exactly (xy is distinctive enough to pair
/// on its own).
///
/// # Errors
/// Fails when no DLL gaze point matches a frame, or a frame's time does not
/// fit an `i64`.
pub(crate) fn clock_offset<'a>(
    frames: impl IntoIterator<Item = &'a GazeFrame>,
    records: &[DllRecord],
) -> Result<ClockOffset> {
    let mut by_xy: HashMap<[u32; 2], i64> = HashMap::new();
    for f in frames {
        by_xy.insert(
            f32s(f.gaze.value).map(f32::to_bits),
            i64::try_from(f.device_ts_us).context("a gaze frame's time does not fit an i64")?,
        );
    }
    let mut offsets: Vec<i64> = records
        .iter()
        .filter_map(|r| match r {
            DllRecord::GazePoint {
                ts_us,
                xy,
                valid: true,
            } => by_xy.get(&xy.map(f32::to_bits)).map(|d| d - ts_us),
            _ => None,
        })
        .collect();
    ensure!(
        !offsets.is_empty(),
        "no DLL gaze point matches a decoded frame"
    );
    offsets.sort_unstable();
    let k_us = offsets[offsets.len() / 2];
    Ok(ClockOffset {
        k_us,
        agreeing: offsets.iter().filter(|&&o| o == k_us).count(),
        matches: offsets.len(),
    })
}

/// Compare a captured session with the DLL's log of it.
///
/// # Errors
/// Fails when either file cannot be read, or no frame can be paired.
pub(crate) fn compare_dll(log_path: &str, jsonl_path: &str) -> Result<()> {
    let frames = frames(log_path)?;
    // The head poses pair with the IR images, which `--head` replays.
    let records: Vec<DllRecord> = read_dll_records(jsonl_path)?
        .into_iter()
        .filter(|r| !matches!(r, DllRecord::HeadPose(_)))
        .collect();
    println!(
        "{} gaze frames decoded, {} DLL records",
        frames.len(),
        records.len()
    );

    // The rebase constant: device ts - DLL ts.
    let ClockOffset {
        k_us: k,
        agreeing,
        matches,
    } = clock_offset(frames.values(), &records)?;
    println!("clock offset {k} us (agreed by {agreeing}/{matches} xy matches)");

    let (mut points, mut origins, mut missing) = (0usize, 0usize, 0usize);
    let (mut validity_mismatch, mut gaze_err, mut origin_err) = (0usize, 0f32, 0f32);
    for r in &records {
        let ts = match r {
            DllRecord::GazePoint { ts_us, .. } | DllRecord::GazeOrigin { ts_us, .. } => ts_us + k,
            DllRecord::HeadPose(_) => continue,
        };
        let Some(f) = frames.get(&ts) else {
            missing += 1;
            continue;
        };
        match r {
            DllRecord::GazePoint { valid, xy, .. } => {
                points += 1;
                validity_mismatch += usize::from(*valid != f.gaze.valid);
                gaze_err = gaze_err.max(max_abs_diff(xy, &f32s(f.gaze.value)));
            }
            DllRecord::GazeOrigin { left, right, .. } => {
                origins += 1;
                validity_mismatch += usize::from(left.0 != f.left.origin_display_mm.valid);
                validity_mismatch += usize::from(right.0 != f.right.origin_display_mm.valid);
                origin_err = origin_err
                    .max(max_abs_diff(&left.1, &f32s(f.left.origin_display_mm.value)))
                    .max(max_abs_diff(
                        &right.1,
                        &f32s(f.right.origin_display_mm.value),
                    ));
            }
            DllRecord::HeadPose(_) => {}
        }
    }
    println!(
        "paired {points} gaze points and {origins} gaze origins; {missing} DLL records without a frame"
    );
    println!("max |gaze point error| {gaze_err:e}, max |gaze origin error| {origin_err:e} mm");
    println!("validity mismatches: {validity_mismatch}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_proto::gaze83::Valued;

    #[test]
    #[allow(clippy::cast_possible_truncation)] // reason: the DLL logged f32
    fn parses_the_dll_log_lines() {
        let point = r#"{"gazePoint":{"position_xy":[0.5872913599014282,0.09927384555339813],"timestamp_us":323209472,"validity":1},"hostUs":1}"#;
        assert_eq!(
            parse_record(point),
            Some(DllRecord::GazePoint {
                ts_us: 323_209_472,
                valid: true,
                xy: [
                    0.587_291_359_901_428_2_f64 as f32,
                    0.099_273_845_553_398_13_f64 as f32
                ],
            })
        );
        let origin = r#"{"gazeOrigin":{"left_validity":1,"left_xyz":[-57.9,112.4,618.2],"right_validity":0,"right_xyz":[16.0,108.9,621.4],"timestamp_us":7},"hostUs":1}"#;
        match parse_record(origin) {
            Some(DllRecord::GazeOrigin { ts_us, left, right }) => {
                assert_eq!(ts_us, 7);
                assert!(left.0 && !right.0);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(parse_record(r#"{"headPose":{}}"#), None);
        assert_eq!(parse_record(r#"{"presence":{"status":1}}"#), None);
    }

    /// A valid and an invalid `headPose` of session 2's log, as the DLL
    /// wrote them: the values keep every digit, and the invalid pose's
    /// leftovers are read too.
    #[test]
    fn parses_the_dll_head_poses() {
        let valid = r#"{"headPose":{"position_validity":1,"position_xyz":[-20.990829467773438,84.00411224365234,571.5678100585938],"rotation_validity_xyz":[1,1,1],"rotation_xyz":[0.18410883843898773,-0.03730017691850662,-0.049585357308387756],"timestamp_us":323201379},"hostUs":1780425659574342}"#;
        let pose = DllHeadPose {
            ts_us: 323_201_379,
            position_valid: true,
            rotation_valid: [true; 3],
            position_mm: [
                -20.990_829_467_773_438,
                84.004_112_243_652_34,
                571.567_810_058_593_8,
            ],
            rotation_rad: [
                0.184_108_838_438_987_73,
                -0.037_300_176_918_506_62,
                -0.049_585_357_308_387_756,
            ],
        };
        assert_eq!(parse_record(valid), Some(DllRecord::HeadPose(pose)));
        assert!(pose.is_valid() && pose.flags_agree());

        let invalid = r#"{"headPose":{"position_validity":0,"position_xyz":[0.0,0.0,0.0],"rotation_validity_xyz":[0,0,0],"rotation_xyz":[0.0,0.41282448172569275,0.14598959684371948],"timestamp_us":155536612},"hostUs":1780426902907389}"#;
        let Some(DllRecord::HeadPose(pose)) = parse_record(invalid) else {
            panic!("no head pose in {invalid}");
        };
        assert_eq!(pose.ts_us, 155_536_612);
        assert_eq!(pose.rotation_rad[2], 0.145_989_596_843_719_48);
        assert!(!pose.is_valid() && pose.flags_agree());

        let mixed = DllHeadPose {
            position_valid: true,
            rotation_valid: [true, false, true],
            ..pose
        };
        assert!(!mixed.is_valid() && !mixed.flags_agree());
        let short = r#"{"headPose":{"position_validity":1,"position_xyz":[1.0,2.0],"rotation_validity_xyz":[1,1,1],"rotation_xyz":[0.0,0.0,0.0],"timestamp_us":1}}"#;
        assert_eq!(parse_record(short), None);
    }

    /// A gaze frame of device time `ts_us` whose gaze point is `xy`.
    fn gaze_frame(ts_us: u64, xy: [f64; 2]) -> GazeFrame {
        GazeFrame {
            device_ts_us: ts_us,
            gaze: Valued {
                valid: true,
                value: xy,
            },
            ..GazeFrame::default()
        }
    }

    /// The clock offset is the median offset of the DLL's valid gaze points
    /// that match a frame's bit for bit; an invalid point, or one no frame
    /// has, counts for nothing.
    #[test]
    #[allow(clippy::cast_possible_truncation)] // reason: the DLL hands out f32
    fn the_clock_offset_comes_from_the_matching_gaze_points() {
        let frames = [
            gaze_frame(1_000, [0.25, 0.5]),
            gaze_frame(1_030, [0.26, 0.5]),
            gaze_frame(1_060, [0.27, 0.5]),
            gaze_frame(1_090, [0.28, 0.5]),
        ];
        let point = |ts_us: i64, valid: bool, xy: [f64; 2]| DllRecord::GazePoint {
            ts_us,
            valid,
            xy: xy.map(|c| c as f32),
        };
        let records = [
            point(900, true, [0.25, 0.5]),
            point(930, true, [0.26, 0.5]),
            // A stray match 7 µs off the others, an invalid point and a
            // point no frame has.
            point(953, true, [0.27, 0.5]),
            point(0, false, [0.28, 0.5]),
            point(5, true, [0.9, 0.9]),
        ];
        assert_eq!(
            clock_offset(&frames, &records).ok(),
            Some(ClockOffset {
                k_us: 100,
                agreeing: 2,
                matches: 3,
            })
        );
        assert!(clock_offset(&frames, &records[3..]).is_err());
    }
}
