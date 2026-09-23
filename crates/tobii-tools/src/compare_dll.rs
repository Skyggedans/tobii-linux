//! `compare-dll`: replay a captured session through the gaze decoder the
//! daemon uses and compare it with what the Windows Stream Engine delivered
//! for the same frames (its callbacks were logged to a JSONL file while the
//! USB traffic was captured).
//!
//! The DLL's timestamps are the device clock minus a per-session constant,
//! so frames are paired by timestamp once that constant is known; it is taken
//! from the frames whose gaze point matches exactly.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, ensure};
use tobii_proto::gaze83::{GazeFrame, decode_gaze_frame};
use tobii_proto::log::read_log_payloads;
use tobii_proto::protocol::{BulkReassembler, parse_message};

/// One `gazePoint` or `gazeOrigin` record of the DLL log.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DllRecord {
    GazePoint {
        ts_us: i64,
        valid: bool,
        xy: [f32; 2],
    },
    GazeOrigin {
        ts_us: i64,
        left: (bool, [f32; 3]),
        right: (bool, [f32; 3]),
    },
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
    } else {
        None
    }
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

/// Compare a captured session with the DLL's log of it.
///
/// # Errors
/// Fails when either file cannot be read, or no frame can be paired.
pub(crate) fn compare_dll(log_path: &str, jsonl_path: &str) -> Result<()> {
    let frames = frames(log_path)?;
    let records: Vec<DllRecord> = BufReader::new(
        File::open(jsonl_path).with_context(|| format!("failed to open {jsonl_path}"))?,
    )
    .lines()
    .map_while(Result::ok)
    .filter_map(|l| parse_record(&l))
    .collect();
    println!(
        "{} gaze frames decoded, {} DLL records",
        frames.len(),
        records.len()
    );

    // The rebase constant: device ts - DLL ts, from gaze points that match
    // exactly (xy is distinctive enough to pair on its own).
    let mut by_xy: HashMap<[u32; 2], i64> = HashMap::new();
    for (ts, f) in &frames {
        by_xy.insert(f32s(f.gaze.value).map(f32::to_bits), *ts);
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
    let k = offsets[offsets.len() / 2];
    let agreeing = offsets.iter().filter(|&&o| o == k).count();
    println!(
        "clock offset {k} us (agreed by {agreeing}/{} xy matches)",
        offsets.len()
    );

    let (mut points, mut origins, mut missing) = (0usize, 0usize, 0usize);
    let (mut validity_mismatch, mut gaze_err, mut origin_err) = (0usize, 0f32, 0f32);
    for r in &records {
        let ts = match r {
            DllRecord::GazePoint { ts_us, .. } | DllRecord::GazeOrigin { ts_us, .. } => ts_us + k,
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
    }
}
