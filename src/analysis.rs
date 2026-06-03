use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use crate::cli::LogInput;
use crate::decode::{
    decode_stream_payload, decode_stream_payload_with_status, field_value, head_point, LiveField,
    TrackingFrame,
};
use crate::track::{euler_deg, kabsch};
use crate::math::{
    dot3, normalize_angle_deg, solve_3x3, vector_pitch_deg, vector_roll_xy_deg, vector_yaw_deg,
};
use crate::opentrack::{
    DEFAULT_OPENTRACK_ANGLE_OCC, DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE, OPENTRACK_HEAD_SCALE,
};
use crate::protocol::{
    hex_to_bytes, main_stream_payloads, marker, read_init_packets, read_log_payloads, InitPacket,
};
use crate::sinks::PacketLog;

pub(crate) fn import_tsv(tsv_path: &str, log_path: &str) -> Result<()> {
    let input = BufReader::new(
        File::open(tsv_path).with_context(|| format!("failed to open TSV {tsv_path}"))?,
    );
    let mut log = PacketLog::create(log_path)?;
    let mut records = 0usize;
    let mut stream_records = 0usize;

    for (line_no, line) in input.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read TSV line {}", line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split('\t');
        let ts_s = parts
            .next()
            .with_context(|| format!("missing timestamp at TSV line {}", line_no + 1))?;
        let ep_s = parts
            .next()
            .with_context(|| format!("missing endpoint at TSV line {}", line_no + 1))?;
        let hex_s = parts
            .next()
            .with_context(|| format!("missing payload hex at TSV line {}", line_no + 1))?;

        let ts_us = parse_timestamp_us(ts_s)
            .with_context(|| format!("bad timestamp at TSV line {}", line_no + 1))?;
        let ep = u8::from_str_radix(ep_s.trim_start_matches("0x"), 16)
            .with_context(|| format!("bad endpoint at TSV line {}", line_no + 1))?;
        let data =
            hex_to_bytes(hex_s).with_context(|| format!("bad hex at TSV line {}", line_no + 1))?;

        if marker(&data) == Some(0x53) {
            stream_records += 1;
        }
        log.write_record_at(ts_us, ep, &data)?;
        records += 1;
    }

    println!("imported {records} records ({stream_records} stream packets) to {log_path}");
    Ok(())
}

pub(crate) fn parse_timestamp_us(s: &str) -> Result<u64> {
    let (seconds, fraction) = s.trim().split_once('.').unwrap_or((s.trim(), ""));
    let seconds: u64 = seconds.parse().context("bad seconds")?;
    let mut micros = String::from(fraction);
    micros.truncate(6);
    while micros.len() < 6 {
        micros.push('0');
    }
    let micros: u64 = if micros.is_empty() {
        0
    } else {
        micros.parse().context("bad fractional seconds")?
    };
    Ok(seconds * 1_000_000 + micros)
}

pub(crate) fn analyze_log(path: &str) -> Result<()> {
    let mut records = 0u64;
    let mut total_bytes = 0u64;
    let mut marker_52 = 0u64;
    let mut marker_53 = 0u64;
    let mut stream_payloads = Vec::new();

    for data in read_log_payloads(path)? {
        records += 1;
        total_bytes += data.len() as u64;

        let m = marker(&data);

        match m {
            Some(0x52) => marker_52 += 1,
            Some(0x53) => {
                marker_53 += 1;
                stream_payloads.push(data.clone());
            }
            _ => {}
        }
    }

    println!("records={records} total_payload_bytes={total_bytes}");
    println!("marker 0x52 responses={marker_52} marker 0x53 stream={marker_53}");

    print_numeric_candidates(&stream_payloads);
    print_stream_changes(&stream_payloads);

    Ok(())
}

#[derive(Clone)]
pub(crate) struct LogSummary {
    pub(crate) label: String,
    pub(crate) payloads: Vec<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct CompareCandidate {
    pub(crate) kind: &'static str,
    pub(crate) offset: usize,
    pub(crate) means: Vec<f64>,
    pub(crate) stddevs: Vec<f64>,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) score: f64,
}

pub(crate) fn compare_logs(inputs: &[LogInput]) -> Result<()> {
    let mut logs = Vec::new();

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        anyhow::ensure!(!payloads.is_empty(), "{} has no stream packets", input.path);
        println!(
            "{}: {} main stream packets, len={}",
            input.label,
            payloads.len(),
            payloads[0].len()
        );
        logs.push(LogSummary {
            label: input.label.clone(),
            payloads,
        });
    }

    let min_len = logs
        .iter()
        .flat_map(|log| log.payloads.iter().map(Vec::len))
        .min()
        .unwrap_or(0);

    let mut candidates = Vec::new();

    for offset in 0..min_len {
        push_compare_candidate(
            &logs,
            "u8",
            offset,
            |payload, offset| payload[offset] as f64,
            &mut candidates,
        );
    }

    for offset in 0..min_len.saturating_sub(1) {
        push_compare_candidate(
            &logs,
            "i16",
            offset,
            |payload, offset| {
                i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
        push_compare_candidate(
            &logs,
            "u16",
            offset,
            |payload, offset| {
                u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
    }

    for offset in 0..min_len.saturating_sub(3) {
        push_compare_candidate(
            &logs,
            "i32",
            offset,
            |payload, offset| {
                i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
        push_compare_candidate(
            &logs,
            "u32",
            offset,
            |payload, offset| {
                u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
        push_compare_candidate(
            &logs,
            "f32",
            offset,
            |payload, offset| {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.cmp(&b.offset))
    });

    print_compare_table("Top cross-log candidates", &logs, &candidates, 32);
    print_pair_table("head_left", "head_right", &logs, &candidates, 24);
    print_pair_table("eyes_left", "eyes_right", &logs, &candidates, 24);

    Ok(())
}

pub(crate) fn push_compare_candidate<F>(
    logs: &[LogSummary],
    kind: &'static str,
    offset: usize,
    read_value: F,
    out: &mut Vec<CompareCandidate>,
) where
    F: Fn(&[u8], usize) -> f64,
{
    let mut means = Vec::new();
    let mut stddevs = Vec::new();
    let mut all_min = f64::INFINITY;
    let mut all_max = f64::NEG_INFINITY;

    for log in logs {
        let values: Vec<f64> = log
            .payloads
            .iter()
            .map(|payload| read_value(payload, offset))
            .collect();

        if values.iter().any(|value| !value.is_finite()) {
            return;
        }

        if kind == "f32" && values.iter().any(|value| value.abs() > 1_000_000.0) {
            return;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        let stddev = variance.sqrt();

        all_min = all_min.min(values.iter().copied().fold(f64::INFINITY, f64::min));
        all_max = all_max.max(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        means.push(mean);
        stddevs.push(stddev);
    }

    let between = means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - means.iter().copied().fold(f64::INFINITY, f64::min);
    let within = stddevs.iter().sum::<f64>() / stddevs.len() as f64;

    if between <= 0.0 {
        return;
    }

    let score = between / (within + 1.0);
    if score < 0.5 {
        return;
    }

    out.push(CompareCandidate {
        kind,
        offset,
        means,
        stddevs,
        min: all_min,
        max: all_max,
        score,
    });
}

pub(crate) fn print_compare_table(
    title: &str,
    logs: &[LogSummary],
    candidates: &[CompareCandidate],
    limit: usize,
) {
    println!("{title}:");
    print!(
        "  {:>3} {:>5} {:>9} {:>12} {:>12}",
        "typ", "off", "score", "min", "max"
    );
    for log in logs {
        print!(" {:>12}", log.label);
    }
    println!();

    for candidate in candidates.iter().take(limit) {
        print!(
            "  {:>3} {:>5} {:>9.3} {:>12.5} {:>12.5}",
            candidate.kind, candidate.offset, candidate.score, candidate.min, candidate.max
        );
        for (mean, stddev) in candidate.means.iter().zip(&candidate.stddevs) {
            print!(" {:>7.2}/{:<4.1}", mean, stddev);
        }
        println!();
    }
}

pub(crate) fn print_pair_table(
    left_label: &str,
    right_label: &str,
    logs: &[LogSummary],
    candidates: &[CompareCandidate],
    limit: usize,
) {
    let Some(left) = logs.iter().position(|log| log.label == left_label) else {
        return;
    };
    let Some(right) = logs.iter().position(|log| log.label == right_label) else {
        return;
    };

    let mut pair_candidates: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let delta = candidate.means[right] - candidate.means[left];
            let noise = (candidate.stddevs[left] + candidate.stddevs[right]) / 2.0 + 1.0;
            (delta.abs() / noise, delta, candidate)
        })
        .filter(|(score, _, _)| *score >= 0.5)
        .collect();

    pair_candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.offset.cmp(&b.2.offset))
    });

    println!("Top {left_label} vs {right_label}:");
    println!(
        "  {:>3} {:>5} {:>9} {:>12} {:>12} {:>12}",
        "typ", "off", "score", left_label, right_label, "delta"
    );

    for (score, delta, candidate) in pair_candidates.into_iter().take(limit) {
        println!(
            "  {:>3} {:>5} {:>9.3} {:>12.5} {:>12.5} {:>12.5}",
            candidate.kind,
            candidate.offset,
            score,
            candidate.means[left],
            candidate.means[right],
            delta
        );
    }
}

#[derive(Clone, Default)]
pub(crate) struct StreamFieldStats {
    pub(crate) count: usize,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) sum: f64,
    pub(crate) first: f64,
    pub(crate) last: f64,
    pub(crate) changes: usize,
    pub(crate) prev: Option<f64>,
}

impl StreamFieldStats {
    pub(crate) fn push(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
            self.first = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }

        if let Some(prev) = self.prev {
            if (prev - value).abs() > f64::EPSILON {
                self.changes += 1;
            }
        }

        self.prev = Some(value);
        self.last = value;
        self.sum += value;
        self.count += 1;
    }

    pub(crate) fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    pub(crate) fn range(&self) -> f64 {
        self.max - self.min
    }
}

pub(crate) fn decode_stream(path: &str) -> Result<()> {
    let payloads = main_stream_payloads(path)?;
    anyhow::ensure!(!payloads.is_empty(), "{path} has no stream payloads");
    let (stats, malformed) = decode_stream_fields(&payloads)?;

    let mut fields: Vec<_> = stats
        .iter()
        .filter(|(_, stat)| stat.count > 1 && stat.changes > 0)
        .collect();
    fields.sort_by(|a, b| {
        b.1.range()
            .partial_cmp(&a.1.range())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });

    println!(
        "decoded {} stream payloads; malformed records={}",
        payloads.len(),
        malformed
    );
    print_decoded_table(fields.into_iter().take(120));

    Ok(())
}

pub(crate) fn decode_stream_fields(
    payloads: &[Vec<u8>],
) -> Result<(BTreeMap<(u32, usize, usize), StreamFieldStats>, usize)> {
    let mut stats = BTreeMap::<(u32, usize, usize), StreamFieldStats>::new();
    let mut malformed = 0usize;

    for payload in payloads {
        let (values, is_malformed) = decode_stream_payload_with_status(payload)?;
        if is_malformed {
            malformed += 1;
        }

        for (key, value) in values {
            stats.entry(key).or_default().push(value);
        }
    }

    Ok((stats, malformed))
}

pub(crate) fn print_decoded_table<'a, I>(fields: I)
where
    I: IntoIterator<Item = (&'a (u32, usize, usize), &'a StreamFieldStats)>,
{
    println!(
        "  {:>10} {:>3} {:>4} {:>6} {:>7} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "id", "occ", "comp", "count", "chg", "min", "max", "mean", "first", "last"
    );

    for ((id, occurrence, component), stat) in fields {
        println!(
            "  0x{id:08x} {:>3} {:>4} {:>6} {:>7} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            occurrence,
            component,
            stat.count,
            stat.changes,
            stat.min,
            stat.max,
            stat.mean(),
            stat.first,
            stat.last
        );
    }
}

pub(crate) fn pose_candidates(inputs: &[LogInput]) -> Result<()> {
    let occurrences = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut logs = Vec::<(String, Vec<BTreeMap<(u32, usize, usize), f64>>)>::new();

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        let mut frames = Vec::new();
        for payload in payloads {
            let (values, malformed) = decode_stream_payload_with_status(&payload)?;
            if !malformed && !values.is_empty() {
                frames.push(values);
            }
        }
        println!("{}: {} decoded frames", input.label, frames.len());
        logs.push((input.label.clone(), frames));
    }

    print_pose_candidate_axis("yaw", &logs, &occurrences, vector_yaw_deg);
    print_pose_candidate_axis("pitch", &logs, &occurrences, vector_pitch_deg);
    print_pose_candidate_axis("roll_xy", &logs, &occurrences, vector_roll_xy_deg);
    print_raw_field_candidates("yaw", &logs);
    print_raw_field_candidates("pitch", &logs);
    print_raw_field_candidates("roll", &logs);
    print_rotation_translation_fit(&logs);
    print_angle_translation_fit(&logs);

    Ok(())
}

pub(crate) fn print_raw_field_candidates(
    target: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
) {
    let mut keys = BTreeMap::<(u32, usize, usize), ()>::new();
    for (_, frames) in logs {
        for values in frames {
            for key in values.keys() {
                keys.insert(*key, ());
            }
        }
    }

    let mut rows = Vec::<((u32, usize, usize), Vec<f64>, f64)>::new();
    for key in keys.keys() {
        let mut ranges = Vec::new();
        let mut present = true;
        for (_, frames) in logs {
            let Some(range) = field_range(frames, *key) else {
                present = false;
                break;
            };
            ranges.push(range);
        }
        if !present {
            continue;
        }

        let score = score_raw_field_ranges(target, logs, &ranges);
        rows.push((*key, ranges, score));
    }

    rows.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("Top {target} raw field candidates:");
    print!("  {:>10} {:>3} {:>4} {:>8}", "id", "occ", "cmp", "score");
    for (label, _) in logs {
        print!(" {:>10}", label);
    }
    println!();

    for ((id, occurrence, component), ranges, score) in rows.into_iter().take(24) {
        print!(
            "  0x{id:08x} {:>3} {:>4} {:>8.3}",
            occurrence, component, score
        );
        for range in ranges {
            print!(" {:>10.2}", range);
        }
        println!();
    }
}

pub(crate) fn field_range(
    frames: &[BTreeMap<(u32, usize, usize), f64>],
    key: (u32, usize, usize),
) -> Option<f64> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0usize;

    for values in frames {
        let Some(value) = values.get(&key).copied() else {
            continue;
        };
        min = min.min(value);
        max = max.max(value);
        count += 1;
    }

    (count >= 8).then_some(max - min)
}

pub(crate) fn score_raw_field_ranges(
    target: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
    ranges: &[f64],
) -> f64 {
    let mut target_range = 0.0_f64;
    let mut shift_leak = 0.0_f64;
    let mut other_motion = 0.0_f64;

    for ((label, _), range) in logs.iter().zip(ranges) {
        if label.starts_with(target) {
            target_range = target_range.max(*range);
        } else if label.starts_with("shift") {
            shift_leak = shift_leak.max(*range);
        } else {
            other_motion = other_motion.max(*range);
        }
    }

    target_range / (shift_leak * 2.0 + other_motion * 0.5 + 1.0)
}

pub(crate) fn print_pose_candidate_axis<F>(
    axis: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
    occurrences: &[usize],
    angle_fn: F,
) where
    F: Fn([f64; 3]) -> f64 + Copy,
{
    let mut rows = Vec::<((usize, usize), Vec<f64>, f64)>::new();

    for (i, &a) in occurrences.iter().enumerate() {
        for &b in occurrences.iter().skip(i + 1) {
            let mut ranges = Vec::new();
            for (_, frames) in logs {
                let range = pair_angle_range(frames, a, b, angle_fn).unwrap_or(0.0);
                ranges.push(range);
            }
            let score = score_pose_ranges(axis, logs, &ranges);
            rows.push(((a, b), ranges, score));
        }
    }

    rows.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.1.iter()
                    .fold(0.0_f64, |m, v| m.max(*v))
                    .partial_cmp(&a.1.iter().fold(0.0_f64, |m, v| m.max(*v)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    println!("Top {axis} vector candidates:");
    print!("  {:>5} {:>8}", "pair", "score");
    for (label, _) in logs {
        print!(" {:>10}", label);
    }
    println!();

    for ((a, b), ranges, score) in rows.into_iter().take(24) {
        print!("  {:>2}-{:>2} {:>8.3}", a, b, score);
        for range in ranges {
            print!(" {:>10.3}", range);
        }
        println!();
    }
}

pub(crate) fn score_pose_ranges(
    axis: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
    ranges: &[f64],
) -> f64 {
    let target = match axis {
        "yaw" => "yaw",
        "pitch" => "pitch",
        "roll_xy" => "roll",
        _ => "",
    };

    let mut target_range = 0.0_f64;
    let mut shift_leak = 0.0_f64;
    let mut other_motion = 0.0_f64;
    for ((label, _), range) in logs.iter().zip(ranges) {
        if label.starts_with(target) {
            target_range = target_range.max(*range);
        } else if label.starts_with("shift") {
            shift_leak = shift_leak.max(*range);
        } else {
            other_motion = other_motion.max(*range);
        }
    }

    target_range / (shift_leak * 2.0 + other_motion * 0.5 + 1.0)
}

pub(crate) fn pair_angle_range<F>(
    frames: &[BTreeMap<(u32, usize, usize), f64>],
    a: usize,
    b: usize,
    angle_fn: F,
) -> Option<f64>
where
    F: Fn([f64; 3]) -> f64,
{
    let mut first = None::<f64>;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0usize;

    for values in frames {
        let (Some(pa), Some(pb)) = (head_point(values, a), head_point(values, b)) else {
            continue;
        };
        let vec = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
        if len < 1.0 {
            continue;
        }

        let angle = angle_fn(vec);
        let base = *first.get_or_insert(angle);
        let value = normalize_angle_deg(angle - base);
        min = min.min(value);
        max = max.max(value);
        count += 1;
    }

    (count >= 8).then_some(max - min)
}

pub(crate) fn print_angle_translation_fit(logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)]) {
    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [[0.0; 3]; 3];
    let mut samples = 0usize;

    for (label, frames) in logs {
        if !label.starts_with("shift") {
            continue;
        }

        let mut origin_head = None::<[f64; 3]>;
        let mut origin_angles = None::<[f64; 3]>;

        for (packet, values) in frames.iter().enumerate() {
            let frame = TrackingFrame::from_decoded(packet as u64, values);
            let Some(head) = frame.head_xyz() else {
                continue;
            };
            let Some(angles) = default_raw_angles(values) else {
                continue;
            };

            let origin_head = *origin_head.get_or_insert(head);
            let origin_angles = *origin_angles.get_or_insert(angles);
            let translation = [
                (head[0] - origin_head[0]) / OPENTRACK_HEAD_SCALE,
                (head[1] - origin_head[1]) / OPENTRACK_HEAD_SCALE,
                (head[2] - origin_head[2]) / OPENTRACK_HEAD_SCALE,
            ];
            if dot3(translation, translation).sqrt() < 0.01 {
                continue;
            }

            let angle_delta = [
                (angles[0] - origin_angles[0]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[0],
                (angles[1] - origin_angles[1]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[1],
                (angles[2] - origin_angles[2]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[2],
            ];

            for row in 0..3 {
                for col in 0..3 {
                    xtx[row][col] += translation[row] * translation[col];
                }
            }
            for axis in 0..3 {
                for col in 0..3 {
                    xty[axis][col] += angle_delta[axis] * translation[col];
                }
            }
            samples += 1;
        }
    }

    println!("Fitted --opentrack-angle-translation-comp from shift logs ({samples} samples):");
    if samples == 0 {
        println!("  not enough shift samples");
        return;
    }

    let mut matrix = [[0.0; 3]; 3];
    for axis in 0..3 {
        if let Some(row) = solve_3x3(xtx, xty[axis]) {
            matrix[axis] = row;
        }
    }

    println!(
        "  {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2]
    );
    println!("  rows are yaw,pitch,roll; columns are tx,ty,tz in cm");

    println!("  raw/compensated angle ranges by log:");
    println!(
        "  {:>10} {:>8} {:>8} {:>8} {:>8}",
        "log", "yaw_raw", "yaw_cmp", "pit_raw", "pit_cmp"
    );
    for (label, frames) in logs {
        if let Some((raw, comp)) = angle_ranges_with_comp(frames, matrix) {
            println!(
                "  {:>10} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                label, raw[0], comp[0], raw[1], comp[1]
            );
        }
    }
}

pub(crate) fn print_rotation_translation_fit(logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)]) {
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [[0.0; 3]; 3];
    let mut samples = 0usize;

    for (label, frames) in logs {
        if !(label.starts_with("yaw") || label.starts_with("pitch") || label.starts_with("roll")) {
            continue;
        }

        let mut origin_head = None::<[f64; 3]>;
        let mut origin_angles = None::<[f64; 3]>;

        for (packet, values) in frames.iter().enumerate() {
            let frame = TrackingFrame::from_decoded(packet as u64, values);
            let Some(head) = frame.head_xyz() else {
                continue;
            };
            let Some(angles) = default_raw_angles(values) else {
                continue;
            };

            let origin_head = *origin_head.get_or_insert(head);
            let origin_angles = *origin_angles.get_or_insert(angles);
            let angle_delta = [
                (angles[0] - origin_angles[0]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[0],
                (angles[1] - origin_angles[1]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[1],
                0.0,
            ];
            if dot3(angle_delta, angle_delta).sqrt() < 0.1 {
                continue;
            }

            let translation = [
                (head[0] - origin_head[0]) / OPENTRACK_HEAD_SCALE,
                (head[1] - origin_head[1]) / OPENTRACK_HEAD_SCALE,
                (head[2] - origin_head[2]) / OPENTRACK_HEAD_SCALE,
            ];

            for row in 0..3 {
                for col in 0..3 {
                    ata[row][col] += angle_delta[row] * angle_delta[col];
                }
            }
            for axis in 0..3 {
                for col in 0..3 {
                    atb[axis][col] += translation[axis] * angle_delta[col];
                }
            }
            samples += 1;
        }
    }

    println!("Fitted --opentrack-rotation-comp from rotation logs ({samples} samples):");
    if samples == 0 {
        println!("  not enough rotation samples");
        return;
    }

    let mut matrix = [[0.0; 3]; 3];
    for axis in 0..3 {
        if let Some(row) = solve_3x3(ata, atb[axis]) {
            matrix[axis] = row;
        }
    }

    println!(
        "  {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2]
    );
    println!("  rows are tx,ty,tz in cm; columns are yaw,pitch,roll in deg");
}

pub(crate) fn default_raw_angles(values: &BTreeMap<(u32, usize, usize), f64>) -> Option<[f64; 3]> {
    let yaw = field_value(
        values,
        LiveField::new("", 0x00031f41, DEFAULT_OPENTRACK_ANGLE_OCC, 0),
    )?;
    let pitch = field_value(
        values,
        LiveField::new("", 0x00031f41, DEFAULT_OPENTRACK_ANGLE_OCC, 1),
    )?;
    Some([yaw, -pitch, 0.0])
}

pub(crate) fn angle_ranges_with_comp(
    frames: &[BTreeMap<(u32, usize, usize), f64>],
    matrix: [[f64; 3]; 3],
) -> Option<([f64; 3], [f64; 3])> {
    let mut origin_head = None::<[f64; 3]>;
    let mut origin_angles = None::<[f64; 3]>;
    let mut raw_min = [f64::INFINITY; 3];
    let mut raw_max = [f64::NEG_INFINITY; 3];
    let mut comp_min = [f64::INFINITY; 3];
    let mut comp_max = [f64::NEG_INFINITY; 3];
    let mut count = 0usize;

    for (packet, values) in frames.iter().enumerate() {
        let frame = TrackingFrame::from_decoded(packet as u64, values);
        let Some(head) = frame.head_xyz() else {
            continue;
        };
        let Some(angles) = default_raw_angles(values) else {
            continue;
        };
        let origin_head = *origin_head.get_or_insert(head);
        let origin_angles = *origin_angles.get_or_insert(angles);
        let translation = [
            (head[0] - origin_head[0]) / OPENTRACK_HEAD_SCALE,
            (head[1] - origin_head[1]) / OPENTRACK_HEAD_SCALE,
            (head[2] - origin_head[2]) / OPENTRACK_HEAD_SCALE,
        ];
        let raw = [
            (angles[0] - origin_angles[0]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[0],
            (angles[1] - origin_angles[1]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[1],
            (angles[2] - origin_angles[2]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[2],
        ];
        let comp = [
            raw[0] - dot3(matrix[0], translation),
            raw[1] - dot3(matrix[1], translation),
            raw[2] - dot3(matrix[2], translation),
        ];

        for axis in 0..3 {
            raw_min[axis] = raw_min[axis].min(raw[axis]);
            raw_max[axis] = raw_max[axis].max(raw[axis]);
            comp_min[axis] = comp_min[axis].min(comp[axis]);
            comp_max[axis] = comp_max[axis].max(comp[axis]);
        }
        count += 1;
    }

    if count < 8 {
        return None;
    }

    Some((
        [
            raw_max[0] - raw_min[0],
            raw_max[1] - raw_min[1],
            raw_max[2] - raw_min[2],
        ],
        [
            comp_max[0] - comp_min[0],
            comp_max[1] - comp_min[1],
            comp_max[2] - comp_min[2],
        ],
    ))
}

pub(crate) fn compare_decoded(inputs: &[LogInput]) -> Result<()> {
    let mut logs = Vec::new();

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        let (fields, malformed) = decode_stream_fields(&payloads)?;
        println!(
            "{}: {} stream packets, {} decoded fields, malformed={}",
            input.label,
            payloads.len(),
            fields.len(),
            malformed
        );
        logs.push((input.label.clone(), fields));
    }

    let mut candidates = Vec::new();

    if let Some((_, first_fields)) = logs.first() {
        for key in first_fields.keys() {
            let mut means = Vec::new();
            let mut ranges = Vec::new();
            let mut present = true;

            for (_, fields) in &logs {
                let Some(stat) = fields.get(key) else {
                    present = false;
                    break;
                };
                means.push(stat.mean());
                ranges.push(stat.range());
            }

            if !present {
                continue;
            }

            let between = means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - means.iter().copied().fold(f64::INFINITY, f64::min);
            let within = ranges.iter().sum::<f64>() / ranges.len() as f64;
            let score = between / (within + 1.0);

            if score >= 0.02 && between.abs() > 1.0 {
                candidates.push((*key, means, ranges, score, between));
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.3.partial_cmp(&a.3)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("Top decoded cross-log candidates:");
    print!(
        "  {:>10} {:>3} {:>4} {:>9} {:>12}",
        "id", "occ", "cmp", "score", "between"
    );
    for (label, _) in &logs {
        print!(" {:>12}", label);
    }
    println!();

    for ((id, occurrence, component), means, _, score, between) in candidates.iter().take(80) {
        print!(
            "  0x{id:08x} {:>3} {:>4} {:>9.4} {:>12.4}",
            occurrence, component, score, between
        );
        for mean in means {
            print!(" {:>12.4}", mean);
        }
        println!();
    }

    print_decoded_pair("head_left", "head_right", &logs, &candidates, 32);
    print_decoded_pair("eyes_left", "eyes_right", &logs, &candidates, 32);

    Ok(())
}

/// Research: reconstruct gaze-independent head pose from the 0x83 stream by a
/// rigid (Kabsch) fit over the *rigid* head landmark points (occurrences that
/// move with the head but not with gaze — identified as 5,6,9 + 0,1,4). This is
/// how the Windows Stream Engine derives head pose without the camera.
pub(crate) fn head83(path: &str, occs: &[usize]) -> Result<()> {
    let payloads = main_stream_payloads(path)?;

    // Per-frame rigid point sets, skipping frames without a face (all-zero pts).
    let mut frames: Vec<Vec<[f64; 3]>> = Vec::new();
    for payload in &payloads {
        let values = decode_stream_payload(payload)?;
        let mut pts = Vec::with_capacity(occs.len());
        let mut ok = true;
        for &o in occs {
            match head_point(&values, o) {
                Some(p) if p.iter().any(|v| v.abs() > 1e-6) => pts.push(p),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            frames.push(pts);
        }
    }
    anyhow::ensure!(
        frames.len() > 30,
        "too few valid face frames in {path} ({})",
        frames.len()
    );

    // Reference shape = mean of the first 30 valid frames (neutral pose).
    let nref = 30.min(frames.len());
    let mut reference = vec![[0.0; 3]; occs.len()];
    for frame in &frames[..nref] {
        for (i, p) in frame.iter().enumerate() {
            for k in 0..3 {
                reference[i][k] += p[k] / nref as f64;
            }
        }
    }
    let ref_centroid = centroid3(&reference);

    // Per-frame pose: Kabsch rotation + centroid translation vs the reference.
    let mut eul = Vec::with_capacity(frames.len());
    let mut trans = Vec::with_capacity(frames.len());
    for frame in &frames {
        let r = kabsch(&reference, frame);
        eul.push(euler_deg(&r));
        let c = centroid3(frame);
        trans.push([
            c[0] - ref_centroid[0],
            c[1] - ref_centroid[1],
            c[2] - ref_centroid[2],
        ]);
    }

    let label = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    println!(
        "{label}: {} valid / {} packets, occ={:?}",
        frames.len(),
        payloads.len(),
        occs
    );
    println!("  axis        min        max      range        std");
    let names = ["rot0", "rot1", "rot2", "tx", "ty", "tz"];
    for (i, name) in names.iter().enumerate() {
        let series: Vec<f64> = if i < 3 {
            eul.iter().map(|e| e[i]).collect()
        } else {
            trans.iter().map(|t| t[i - 3]).collect()
        };
        let (mn, mx, sd) = min_max_std(&series);
        println!(
            "  {name:5} {mn:10.2} {mx:10.2} {:10.2} {sd:10.2}",
            mx - mn
        );
    }
    Ok(())
}

fn centroid3(pts: &[[f64; 3]]) -> [f64; 3] {
    let n = pts.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in pts {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    c
}

fn min_max_std(v: &[f64]) -> (f64, f64, f64) {
    let mn = v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    (mn, mx, var.sqrt())
}

pub(crate) type DecodedFields = BTreeMap<(u32, usize, usize), StreamFieldStats>;

pub(crate) fn print_decoded_pair(
    left_label: &str,
    right_label: &str,
    logs: &[(String, DecodedFields)],
    candidates: &[((u32, usize, usize), Vec<f64>, Vec<f64>, f64, f64)],
    limit: usize,
) {
    let Some(left) = logs.iter().position(|(label, _)| label == left_label) else {
        return;
    };
    let Some(right) = logs.iter().position(|(label, _)| label == right_label) else {
        return;
    };

    let mut pairs: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let delta = candidate.1[right] - candidate.1[left];
            let noise = (candidate.2[left] + candidate.2[right]) / 2.0 + 1.0;
            (delta.abs() / noise, delta, candidate)
        })
        .filter(|(score, delta, _)| *score >= 0.02 && delta.abs() > 1.0)
        .collect();

    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2 .0.cmp(&b.2 .0))
    });

    println!("Top decoded {left_label} vs {right_label}:");
    println!(
        "  {:>10} {:>3} {:>4} {:>9} {:>12} {:>12} {:>12}",
        "id", "occ", "cmp", "score", left_label, right_label, "delta"
    );
    for (score, delta, candidate) in pairs.into_iter().take(limit) {
        let (id, occurrence, component) = candidate.0;
        println!(
            "  0x{id:08x} {:>3} {:>4} {:>9.4} {:>12.4} {:>12.4} {:>12.4}",
            occurrence, component, score, candidate.1[left], candidate.1[right], delta
        );
    }
}

#[derive(Clone)]
pub(crate) struct NumericCandidate {
    pub(crate) kind: &'static str,
    pub(crate) offset: usize,
    pub(crate) count: usize,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) mean: f64,
    pub(crate) stddev: f64,
    pub(crate) first: f64,
    pub(crate) last: f64,
    pub(crate) changes: usize,
    pub(crate) score: f64,
}

pub(crate) fn print_numeric_candidates(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        println!("Need at least two stream packets for numeric analysis.");
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut f32_candidates = Vec::new();
    let mut i16_candidates = Vec::new();
    let mut u16_candidates = Vec::new();
    let mut u8_candidates = Vec::new();
    let mut u24_candidates = Vec::new();

    print_length_distribution(payloads);

    for offset in 0..min_len {
        let values: Vec<f64> = payloads
            .iter()
            .map(|payload| payload[offset] as f64)
            .collect();

        if let Some(candidate) = summarize_numeric("u8", offset, &values) {
            if candidate.stddev >= 1.0 {
                u8_candidates.push(candidate);
            }
        }
    }

    for offset in 0..min_len.saturating_sub(3) {
        let values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            })
            .collect();

        if let Some(candidate) = summarize_numeric("f32", offset, &values) {
            if candidate.min.is_finite()
                && candidate.max.is_finite()
                && candidate.min.abs() < 1000.0
                && candidate.max.abs() < 1000.0
                && candidate.stddev > 0.000001
            {
                f32_candidates.push(candidate);
            }
        }
    }

    for offset in 0..min_len.saturating_sub(1) {
        let i16_values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            })
            .collect();
        let u16_values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            })
            .collect();

        if let Some(candidate) = summarize_numeric("i16", offset, &i16_values) {
            if candidate.stddev >= 1.0 {
                i16_candidates.push(candidate);
            }
        }

        if let Some(candidate) = summarize_numeric("u16", offset, &u16_values) {
            if candidate.stddev >= 1.0 {
                u16_candidates.push(candidate);
            }
        }
    }

    for offset in 0..min_len.saturating_sub(2) {
        let values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                let bytes = [payload[offset], payload[offset + 1], payload[offset + 2], 0];
                u32::from_le_bytes(bytes) as f64
            })
            .collect();

        if let Some(candidate) = summarize_numeric("u24", offset, &values) {
            if candidate.stddev >= 1.0 {
                u24_candidates.push(candidate);
            }
        }
    }

    sort_candidates(&mut f32_candidates);
    sort_candidates(&mut i16_candidates);
    sort_candidates(&mut u16_candidates);
    sort_candidates(&mut u8_candidates);
    sort_candidates(&mut u24_candidates);

    print_candidate_table("Top varying u8 candidates", &u8_candidates, 16);
    print_candidate_table("Top varying f32 candidates", &f32_candidates, 24);
    print_candidate_table("Top varying i16 candidates", &i16_candidates, 16);
    print_candidate_table("Top varying u16 candidates", &u16_candidates, 16);
    print_candidate_table("Top varying u24 candidates", &u24_candidates, 16);

    print_f32_triplets(payloads);
}

pub(crate) fn print_length_distribution(payloads: &[Vec<u8>]) {
    let mut lengths: BTreeMap<usize, usize> = BTreeMap::new();

    for payload in payloads {
        *lengths.entry(payload.len()).or_default() += 1;
    }

    println!("Stream packet length distribution:");
    for (len, count) in lengths {
        println!("  len={len:<5} count={count}");
    }
}

pub(crate) fn summarize_numeric(
    kind: &'static str,
    offset: usize,
    values: &[f64],
) -> Option<NumericCandidate> {
    let first = *values.first()?;
    let last = *values.last()?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut changes = 0usize;
    let mut prev = first;

    for &value in values {
        if !value.is_finite() {
            return None;
        }

        min = min.min(value);
        max = max.max(value);
        sum += value;

        if (value - prev).abs() > f64::EPSILON {
            changes += 1;
        }

        prev = value;
    }

    let count = values.len();
    let mean = sum / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    let stddev = variance.sqrt();
    let range = max - min;

    if range <= 0.0 || changes == 0 {
        return None;
    }

    Some(NumericCandidate {
        kind,
        offset,
        count,
        min,
        max,
        mean,
        stddev,
        first,
        last,
        changes,
        score: stddev * (changes as f64).ln_1p(),
    })
}

pub(crate) fn sort_candidates(candidates: &mut [NumericCandidate]) {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.cmp(&b.offset))
    });
}

pub(crate) fn print_candidate_table(title: &str, candidates: &[NumericCandidate], limit: usize) {
    if candidates.is_empty() {
        println!("{title}: none");
        return;
    }

    println!("{title}:");
    for candidate in candidates.iter().take(limit) {
        println!(
            "  {:>3} off={:<4} count={:<4} changes={:<4} min={:>11.5} max={:>11.5} mean={:>11.5} std={:>11.5} first={:>11.5} last={:>11.5}",
            candidate.kind,
            candidate.offset,
            candidate.count,
            candidate.changes,
            candidate.min,
            candidate.max,
            candidate.mean,
            candidate.stddev,
            candidate.first,
            candidate.last
        );
    }
}

pub(crate) fn print_f32_triplets(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut candidates = Vec::new();

    for offset in 0..min_len.saturating_sub(11) {
        let xs: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            })
            .collect();
        let ys: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset + 4..offset + 8].try_into().unwrap()) as f64
            })
            .collect();
        let zs: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset + 8..offset + 12].try_into().unwrap()) as f64
            })
            .collect();

        let Some(x) = summarize_numeric("f32", offset, &xs) else {
            continue;
        };
        let Some(y) = summarize_numeric("f32", offset + 4, &ys) else {
            continue;
        };
        let Some(z) = summarize_numeric("f32", offset + 8, &zs) else {
            continue;
        };

        let plausible = [&x, &y, &z].iter().all(|c| {
            c.min.is_finite()
                && c.max.is_finite()
                && c.min.abs() < 1000.0
                && c.max.abs() < 1000.0
                && c.stddev > 0.000001
        });

        if plausible {
            candidates.push((x.score + y.score + z.score, offset, x, y, z));
        }
    }

    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    if candidates.is_empty() {
        println!("Top contiguous f32 triplets: none");
        return;
    }

    println!("Top contiguous f32 triplets:");
    for (_, offset, x, y, z) in candidates.iter().take(16) {
        println!(
            "  off={:<4} x=[{:>9.5}..{:>9.5}] std={:>9.5} y=[{:>9.5}..{:>9.5}] std={:>9.5} z=[{:>9.5}..{:>9.5}] std={:>9.5}",
            offset, x.min, x.max, x.stddev, y.min, y.max, y.stddev, z.min, z.max, z.stddev
        );
    }
}

pub(crate) fn print_stream_changes(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut ranges = Vec::new();
    let mut start = None;

    for offset in 0..min_len {
        let first = payloads[0][offset];
        let changed = payloads
            .iter()
            .skip(1)
            .any(|payload| payload[offset] != first);

        match (start, changed) {
            (None, true) => start = Some(offset),
            (Some(s), false) => {
                ranges.push((s, offset));
                start = None;
            }
            _ => {}
        }
    }

    if let Some(s) = start {
        ranges.push((s, min_len));
    }

    if ranges.is_empty() {
        println!("Stream packets have identical first {min_len} bytes.");
        return;
    }

    println!(
        "Changing byte ranges across first {} stream packets:",
        payloads.len()
    );
    for (start, end) in ranges.into_iter().take(64) {
        let width = end - start;
        let first_hex = short_hex(&payloads[0][start..end]);
        let last_hex = short_hex(&payloads[payloads.len() - 1][start..end]);

        print!(
            "  off={} len={} first={} last={}",
            start, width, first_hex, last_hex
        );

        if start + 4 <= min_len {
            let first_f32 = f32::from_le_bytes(payloads[0][start..start + 4].try_into().unwrap());
            let last_f32 = f32::from_le_bytes(
                payloads[payloads.len() - 1][start..start + 4]
                    .try_into()
                    .unwrap(),
            );

            if first_f32.is_finite()
                && last_f32.is_finite()
                && first_f32.abs() < 1000.0
                && last_f32.abs() < 1000.0
            {
                print!(" f32_first={first_f32:.6} f32_last={last_f32:.6}");
            }
        }

        println!();
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CalibrationRecord {
    pub(crate) offset: usize,
    pub(crate) values: Vec<f32>,
}

pub(crate) fn extract_calibration(path: &str, json_path: Option<&str>) -> Result<()> {
    let packets = read_init_packets(path)?;
    let blob = calibration_blob(&packets);
    anyhow::ensure!(
        !blob.is_empty(),
        "{path} does not contain large init blob packets"
    );

    let records = find_calibration_records(&blob);
    let point_records: Vec<_> = records
        .iter()
        .filter(|record| record.values.len() == 4)
        .collect();

    println!("input packets: {}", packets.len());
    println!("large blob bytes: {}", blob.len());
    println!(
        "calibration-like records: {} total, {} four-float point records",
        records.len(),
        point_records.len()
    );
    println!();

    println!("Records:");
    println!(
        "  {:>3} {:>8} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "#", "offset", "kind", "v0", "v1", "v2", "v3"
    );
    for (i, record) in records.iter().enumerate() {
        println!(
            "  {:>3} 0x{:06x} {:>7} {:>12} {:>12} {:>12} {:>12}",
            i + 1,
            record.offset,
            format!("{}f", record.values.len()),
            fmt_cal_value(record.values.first().copied()),
            fmt_cal_value(record.values.get(1).copied()),
            fmt_cal_value(record.values.get(2).copied()),
            fmt_cal_value(record.values.get(3).copied()),
        );
    }

    if !point_records.is_empty() {
        println!();
        println!("Likely calibration target/observed points:");
        println!(
            "  {:>3} {:>8} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
            "#", "offset", "target_x", "target_y", "observed_x", "observed_y", "err_x", "err_y"
        );
        for (i, record) in point_records.iter().enumerate() {
            let target_x = record.values[0];
            let target_y = record.values[1];
            let observed_x = record.values[2];
            let observed_y = record.values[3];
            println!(
                "  {:>3} 0x{:06x} {:>10.4} {:>10.4} {:>12.4} {:>12.4} {:>10.4} {:>10.4}",
                i + 1,
                record.offset,
                target_x,
                target_y,
                observed_x,
                observed_y,
                observed_x - target_x,
                observed_y - target_y,
            );
        }
    }

    if let Some(json_path) = json_path {
        write_calibration_json(json_path, &records)?;
        println!();
        println!("Wrote {json_path}");
    }

    Ok(())
}

pub(crate) fn calibration_blob(packets: &[InitPacket]) -> Vec<u8> {
    let mut blob = Vec::new();

    for packet in packets {
        if packet.data.len() >= 512 {
            blob.extend_from_slice(&packet.data[8..]);
        }
    }

    blob
}

pub(crate) fn find_calibration_records(blob: &[u8]) -> Vec<CalibrationRecord> {
    const MARKER: &[u8; 8] = b"\x01\0\0\0\0\0\0\0";
    let mut records = Vec::new();
    let search_start = blob.len().saturating_sub(8192);
    let mut offset = search_start;

    while offset + MARKER.len() <= blob.len() {
        let Some(relative) = find_bytes(&blob[offset..], MARKER) else {
            break;
        };
        let marker_offset = offset + relative;
        let value_offset = marker_offset + MARKER.len();
        let Some(next_relative) = find_bytes(&blob[value_offset..], MARKER) else {
            break;
        };
        let next_marker = value_offset + next_relative;

        if next_marker > value_offset && (next_marker - value_offset) % 4 == 0 {
            let value_count = (next_marker - value_offset) / 4;
            if matches!(value_count, 2 | 4) {
                let mut values = Vec::with_capacity(value_count);
                let mut plausible = true;

                for chunk in blob[value_offset..next_marker].chunks_exact(4) {
                    let value = f32::from_le_bytes(chunk.try_into().unwrap());
                    if !value.is_finite() || !(-0.50..=1.50).contains(&value) {
                        plausible = false;
                        break;
                    }
                    values.push(value);
                }

                if plausible {
                    records.push(CalibrationRecord {
                        offset: marker_offset,
                        values,
                    });
                }
            }
        }

        offset = marker_offset + 1;
    }

    records
}

pub(crate) fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub(crate) fn fmt_cal_value(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

pub(crate) fn write_calibration_json(path: &str, records: &[CalibrationRecord]) -> Result<()> {
    let mut out = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create calibration JSON {path}"))?,
    );

    writeln!(out, "{{")?;
    writeln!(out, "  \"records\": [")?;
    for (i, record) in records.iter().enumerate() {
        write!(
            out,
            "    {{\"offset\":{},\"kind\":\"{}f\",\"values\":[",
            record.offset,
            record.values.len()
        )?;
        for (j, value) in record.values.iter().enumerate() {
            if j > 0 {
                write!(out, ",")?;
            }
            write!(out, "{value:.8}")?;
        }
        write!(out, "]}}")?;
        if i + 1 < records.len() {
            writeln!(out, ",")?;
        } else {
            writeln!(out)?;
        }
    }
    writeln!(out, "  ],")?;
    writeln!(out, "  \"points\": [")?;

    let points: Vec<_> = records
        .iter()
        .filter(|record| record.values.len() == 4)
        .collect();
    for (i, record) in points.iter().enumerate() {
        let target_x = record.values[0];
        let target_y = record.values[1];
        let observed_x = record.values[2];
        let observed_y = record.values[3];
        write!(
            out,
            "    {{\"offset\":{},\"target\":{{\"x\":{target_x:.8},\"y\":{target_y:.8}}},\"observed\":{{\"x\":{observed_x:.8},\"y\":{observed_y:.8}}},\"error\":{{\"x\":{:.8},\"y\":{:.8}}}}}",
            record.offset,
            observed_x - target_x,
            observed_y - target_y
        )?;
        if i + 1 < points.len() {
            writeln!(out, ",")?;
        } else {
            writeln!(out)?;
        }
    }
    writeln!(out, "  ]")?;
    writeln!(out, "}}")?;
    out.flush()?;

    Ok(())
}

pub(crate) fn short_hex(bytes: &[u8]) -> String {
    const MAX: usize = 12;
    let mut out = String::new();

    for b in bytes.iter().take(MAX) {
        out.push_str(&format!("{b:02x}"));
    }

    if bytes.len() > MAX {
        out.push_str("...");
    }

    out
}
