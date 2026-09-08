//! Offline analysis and report subcommands over captured USB logs.
//!
//! Everything here is invoked from the `tobii5-init-replay` CLI and prints
//! human-readable tables with `println!` — that text *is* the program output,
//! not logging.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use crate::cli::LogInput;
use crate::decode::{
    LiveField, TrackingFrame, decode_stream_payload, decode_stream_payload_with_status,
    field_value, head_point,
};
use crate::log::{PacketLog, main_stream_payloads, read_log_payloads};
use crate::math::{
    dot3, normalize_angle_deg, solve_3x3, vector_pitch_deg, vector_roll_xy_deg, vector_yaw_deg,
};
use crate::opentrack::{
    DEFAULT_OPENTRACK_ANGLE_OCC, DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE, OPENTRACK_HEAD_SCALE,
};
use crate::protocol::{InitPacket, hex_to_bytes, marker, read_init_packets};
use crate::track::{euler_deg, kabsch};

/// Decoded stream field key: `(field id, occurrence, component)`.
pub(crate) type FieldKey = (u32, usize, usize);
/// One decoded stream packet: every field value keyed by [`FieldKey`].
pub(crate) type DecodedFrame = BTreeMap<FieldKey, f64>;
/// Per-field statistics over a whole log.
pub(crate) type DecodedFields = BTreeMap<FieldKey, StreamFieldStats>;
/// A labelled log reduced to its decoded frames.
pub(crate) type LabelledFrames = (String, Vec<DecodedFrame>);

/// Import a `timestamp<TAB>endpoint<TAB>hex` TSV capture into the binary packet log format.
///
/// # Errors
/// Returns an error when the TSV cannot be read or a line is malformed, or when the
/// packet log cannot be written.
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

/// Parse a `seconds[.fraction]` timestamp into microseconds.
///
/// # Errors
/// Returns an error when either part is not a decimal integer or the result
/// overflows `u64`.
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
    seconds
        .checked_mul(1_000_000)
        .and_then(|us| us.checked_add(micros))
        .context("timestamp overflows u64 microseconds")
}

/// Print record/marker counts for a packet log plus the raw numeric-candidate scan.
///
/// # Errors
/// Returns an error when the log cannot be read.
pub(crate) fn analyze_log(path: &str) -> Result<()> {
    let mut records = 0u64;
    let mut total_bytes = 0usize;
    let mut marker_52 = 0u64;
    let mut marker_53 = 0u64;
    let mut stream_payloads = Vec::new();

    for data in read_log_payloads(path)? {
        records += 1;
        total_bytes += data.len();

        match marker(&data) {
            Some(0x52) => marker_52 += 1,
            Some(0x53) => {
                marker_53 += 1;
                stream_payloads.push(data);
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

/// A labelled log reduced to its main-stream payloads.
#[derive(Clone, Debug)]
pub(crate) struct LogSummary {
    /// User-supplied label (`yaw_left`, `shift_ud`, ...).
    pub(crate) label: String,
    /// Raw main-stream payloads in capture order.
    pub(crate) payloads: Vec<Vec<u8>>,
}

/// A byte offset whose interpretation as `kind` separates the compared logs.
#[derive(Clone, Debug)]
pub(crate) struct CompareCandidate {
    /// Interpretation of the bytes at `offset` (`u8`, `i16`, `f32`, ...).
    pub(crate) kind: &'static str,
    /// Byte offset into the payload.
    pub(crate) offset: usize,
    /// Per-log mean of the value.
    pub(crate) means: Vec<f64>,
    /// Per-log standard deviation of the value.
    pub(crate) stddevs: Vec<f64>,
    /// Minimum across all logs.
    pub(crate) min: f64,
    /// Maximum across all logs.
    pub(crate) max: f64,
    /// Between-log spread divided by within-log noise.
    pub(crate) score: f64,
}

/// Little-endian `[u8; N]` at `offset`, or `None` when the payload is too short.
#[must_use]
fn le_bytes<const N: usize>(payload: &[u8], offset: usize) -> Option<[u8; N]> {
    payload.get(offset..)?.first_chunk().copied()
}

/// Little-endian `f32` at `offset`.
#[must_use]
fn f32_at(payload: &[u8], offset: usize) -> Option<f32> {
    le_bytes(payload, offset).map(f32::from_le_bytes)
}

/// Little-endian `f32` at `offset`, widened to `f64`.
#[must_use]
fn f32_at_f64(payload: &[u8], offset: usize) -> Option<f64> {
    f32_at(payload, offset).map(f64::from)
}

/// Little-endian `i16` at `offset`, widened to `f64`.
#[must_use]
fn i16_at(payload: &[u8], offset: usize) -> Option<f64> {
    le_bytes(payload, offset).map(|b| f64::from(i16::from_le_bytes(b)))
}

/// Little-endian `u16` at `offset`, widened to `f64`.
#[must_use]
fn u16_at(payload: &[u8], offset: usize) -> Option<f64> {
    le_bytes(payload, offset).map(|b| f64::from(u16::from_le_bytes(b)))
}

/// Little-endian `i32` at `offset`, widened to `f64`.
#[must_use]
fn i32_at(payload: &[u8], offset: usize) -> Option<f64> {
    le_bytes(payload, offset).map(|b| f64::from(i32::from_le_bytes(b)))
}

/// Little-endian `u32` at `offset`, widened to `f64`.
#[must_use]
fn u32_at(payload: &[u8], offset: usize) -> Option<f64> {
    le_bytes(payload, offset).map(|b| f64::from(u32::from_le_bytes(b)))
}

/// Little-endian unsigned 24-bit value at `offset`, widened to `f64`.
#[must_use]
fn u24_at(payload: &[u8], offset: usize) -> Option<f64> {
    le_bytes::<3>(payload, offset)
        .map(|[b0, b1, b2]| f64::from(u32::from_le_bytes([b0, b1, b2, 0])))
}

/// Unsigned byte at `offset`, widened to `f64`.
#[must_use]
fn u8_at(payload: &[u8], offset: usize) -> Option<f64> {
    payload.get(offset).copied().map(f64::from)
}

/// Scan every byte offset for fixed-width numeric interpretations whose value
/// differs between the given logs, and print the ranked candidate tables.
///
/// # Errors
/// Returns an error when a log cannot be read or contains no stream packets.
pub(crate) fn compare_logs(inputs: &[LogInput]) -> Result<()> {
    let mut logs = Vec::with_capacity(inputs.len());

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
        push_compare_candidate(&logs, "u8", offset, u8_at, &mut candidates);
    }

    for offset in 0..min_len.saturating_sub(1) {
        push_compare_candidate(&logs, "i16", offset, i16_at, &mut candidates);
        push_compare_candidate(&logs, "u16", offset, u16_at, &mut candidates);
    }

    for offset in 0..min_len.saturating_sub(3) {
        push_compare_candidate(&logs, "i32", offset, i32_at, &mut candidates);
        push_compare_candidate(&logs, "u32", offset, u32_at, &mut candidates);
        push_compare_candidate(&logs, "f32", offset, f32_at_f64, &mut candidates);
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

/// Score `kind` at `offset` across `logs` and append it to `out` when it
/// separates the logs by more than their noise.
///
/// `read_value` returns `None` when a payload is too short, in which case the
/// offset is skipped.
pub(crate) fn push_compare_candidate<F>(
    logs: &[LogSummary],
    kind: &'static str,
    offset: usize,
    read_value: F,
    out: &mut Vec<CompareCandidate>,
) where
    F: Fn(&[u8], usize) -> Option<f64>,
{
    let mut means = Vec::with_capacity(logs.len());
    let mut stddevs = Vec::with_capacity(logs.len());
    let mut all_min = f64::INFINITY;
    let mut all_max = f64::NEG_INFINITY;

    for log in logs {
        let values: Option<Vec<f64>> = log
            .payloads
            .iter()
            .map(|payload| read_value(payload, offset))
            .collect();
        let Some(values) = values else {
            return;
        };

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

/// Print the top `limit` candidates with one `mean/stddev` column per log.
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
            print!(" {mean:>7.2}/{stddev:<4.1}");
        }
        println!();
    }
}

/// Print the candidates that best separate the two named logs (e.g. a
/// left/right pair), ranked by delta over noise.
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

/// Running statistics of one decoded stream field.
#[derive(Clone, Debug, Default)]
pub(crate) struct StreamFieldStats {
    /// Number of samples seen.
    pub(crate) count: usize,
    /// Minimum sample.
    pub(crate) min: f64,
    /// Maximum sample.
    pub(crate) max: f64,
    /// Sum of all samples (for [`Self::mean`]).
    pub(crate) sum: f64,
    /// First sample.
    pub(crate) first: f64,
    /// Most recent sample.
    pub(crate) last: f64,
    /// Number of consecutive samples that differed.
    pub(crate) changes: usize,
    /// Previous sample, for change detection.
    pub(crate) prev: Option<f64>,
}

impl StreamFieldStats {
    /// Fold one sample into the statistics.
    pub(crate) fn push(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
            self.first = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }

        if let Some(prev) = self.prev
            && (prev - value).abs() > f64::EPSILON
        {
            self.changes += 1;
        }

        self.prev = Some(value);
        self.last = value;
        self.sum += value;
        self.count += 1;
    }

    /// Arithmetic mean of the samples (`NaN` when empty).
    #[must_use]
    pub(crate) fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    /// `max - min`.
    #[must_use]
    pub(crate) fn range(&self) -> f64 {
        self.max - self.min
    }
}

/// Decode every stream payload in a log and print the fields that vary,
/// widest range first.
///
/// # Errors
/// Returns an error when the log cannot be read, has no stream payloads, or a
/// payload fails to decode.
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

/// Accumulate per-field statistics over `payloads`; also returns the number of
/// payloads flagged as malformed by the decoder.
///
/// # Errors
/// Returns an error when a payload fails to decode.
pub(crate) fn decode_stream_fields(payloads: &[Vec<u8>]) -> Result<(DecodedFields, usize)> {
    let mut stats = DecodedFields::new();
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

/// Print one row per decoded field with its statistics.
pub(crate) fn print_decoded_table<'a, I>(fields: I)
where
    I: IntoIterator<Item = (&'a FieldKey, &'a StreamFieldStats)>,
{
    println!(
        "  {:>10} {:>3} {:>4} {:>6} {:>7} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "id", "occ", "comp", "count", "chg", "min", "max", "mean", "first", "last"
    );

    for ((id, occurrence, component), stat) in fields {
        println!(
            "  0x{id:08x} {occurrence:>3} {component:>4} {:>6} {:>7} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
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

/// Rank head-pose hypotheses (landmark-pair vectors, raw fields, and
/// translation/rotation compensation fits) across labelled motion logs.
///
/// # Errors
/// Returns an error when a log cannot be read or a payload fails to decode.
pub(crate) fn pose_candidates(inputs: &[LogInput]) -> Result<()> {
    let occurrences = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut logs = Vec::<LabelledFrames>::with_capacity(inputs.len());

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        let mut frames = Vec::with_capacity(payloads.len());
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

/// Print the decoded fields whose range responds most to logs labelled
/// `target*` and least to `shift*` logs.
pub(crate) fn print_raw_field_candidates(target: &str, logs: &[LabelledFrames]) {
    let keys: BTreeSet<FieldKey> = logs
        .iter()
        .flat_map(|(_, frames)| frames.iter().flat_map(BTreeMap::keys))
        .copied()
        .collect();

    let mut rows = Vec::<(FieldKey, Vec<f64>, f64)>::new();
    for key in &keys {
        let ranges: Option<Vec<f64>> = logs
            .iter()
            .map(|(_, frames)| field_range(frames, *key))
            .collect();
        let Some(ranges) = ranges else {
            continue;
        };

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
        print!(" {label:>10}");
    }
    println!();

    for ((id, occurrence, component), ranges, score) in rows.into_iter().take(24) {
        print!("  0x{id:08x} {occurrence:>3} {component:>4} {score:>8.3}");
        for range in ranges {
            print!(" {range:>10.2}");
        }
        println!();
    }
}

/// `max - min` of `key` over `frames`, or `None` with fewer than 8 samples.
#[must_use]
pub(crate) fn field_range(frames: &[DecodedFrame], key: FieldKey) -> Option<f64> {
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

/// Response to `target*` logs penalised by leak into `shift*` logs and other
/// motion logs.
#[must_use]
pub(crate) fn score_raw_field_ranges(target: &str, logs: &[LabelledFrames], ranges: &[f64]) -> f64 {
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

/// Print, for every landmark-occurrence pair, how much the angle of the vector
/// between them (via `angle_fn`) swings in each log.
pub(crate) fn print_pose_candidate_axis<F>(
    axis: &str,
    logs: &[LabelledFrames],
    occurrences: &[usize],
    angle_fn: F,
) where
    F: Fn([f64; 3]) -> f64 + Copy,
{
    let mut rows = Vec::<((usize, usize), Vec<f64>, f64)>::new();

    for (i, &a) in occurrences.iter().enumerate() {
        for &b in occurrences.iter().skip(i + 1) {
            let ranges: Vec<f64> = logs
                .iter()
                .map(|(_, frames)| pair_angle_range(frames, a, b, angle_fn).unwrap_or(0.0))
                .collect();
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
        print!(" {label:>10}");
    }
    println!();

    for ((a, b), ranges, score) in rows.into_iter().take(24) {
        print!("  {a:>2}-{b:>2} {score:>8.3}");
        for range in ranges {
            print!(" {range:>10.3}");
        }
        println!();
    }
}

/// [`score_raw_field_ranges`] with the vector-axis name (`roll_xy`) mapped to
/// its log-label prefix (`roll`).
#[must_use]
pub(crate) fn score_pose_ranges(axis: &str, logs: &[LabelledFrames], ranges: &[f64]) -> f64 {
    let target = match axis {
        "yaw" => "yaw",
        "pitch" => "pitch",
        "roll_xy" => "roll",
        _ => "",
    };
    score_raw_field_ranges(target, logs, ranges)
}

/// Range of `angle_fn(point_b - point_a)` over `frames`, relative to the first
/// valid frame; `None` with fewer than 8 valid frames.
#[must_use]
pub(crate) fn pair_angle_range<F>(
    frames: &[DecodedFrame],
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

/// `acc[i][j] += a[i] * b[j]` for every `i, j`.
fn add_outer_product(acc: &mut [[f64; 3]; 3], a: [f64; 3], b: [f64; 3]) {
    for (row, a_i) in acc.iter_mut().zip(a) {
        for (cell, b_j) in row.iter_mut().zip(b) {
            *cell += a_i * b_j;
        }
    }
}

/// Solve `lhs * x = rhs[axis]` for each axis; rows that cannot be solved stay zero.
fn solve_rows(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut matrix = [[0.0; 3]; 3];
    for (out, rhs_row) in matrix.iter_mut().zip(rhs) {
        if let Some(row) = solve_3x3(lhs, rhs_row) {
            *out = row;
        }
    }
    matrix
}

/// Print the 3x3 fit matrix as one comma-separated line of nine values.
fn print_fit_matrix(matrix: &[[f64; 3]; 3]) {
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
}

/// Least-squares fit of raw angle drift against head translation over the
/// `shift*` logs, printed as an `--opentrack-angle-translation-comp` matrix.
pub(crate) fn print_angle_translation_fit(logs: &[LabelledFrames]) {
    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [[0.0; 3]; 3];
    let mut samples = 0usize;

    for (label, frames) in logs {
        if !label.starts_with("shift") {
            continue;
        }

        let mut origin_head = None::<[f64; 3]>;
        let mut origin_angles = None::<[f64; 3]>;

        for (packet, values) in (0u64..).zip(frames) {
            let frame = TrackingFrame::from_decoded(packet, values);
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

            add_outer_product(&mut xtx, translation, translation);
            add_outer_product(&mut xty, angle_delta, translation);
            samples += 1;
        }
    }

    println!("Fitted --opentrack-angle-translation-comp from shift logs ({samples} samples):");
    if samples == 0 {
        println!("  not enough shift samples");
        return;
    }

    let matrix = solve_rows(xtx, xty);
    print_fit_matrix(&matrix);
    println!("  rows are yaw,pitch,roll; columns are tx,ty,tz in cm");

    println!("  raw/compensated angle ranges by log:");
    println!(
        "  {:>10} {:>8} {:>8} {:>8} {:>8}",
        "log", "yaw_raw", "yaw_cmp", "pit_raw", "pit_cmp"
    );
    for (label, frames) in logs {
        if let Some((raw, comp)) = angle_ranges_with_comp(frames, matrix) {
            println!(
                "  {label:>10} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                raw[0], comp[0], raw[1], comp[1]
            );
        }
    }
}

/// Least-squares fit of head translation against raw angle change over the
/// `yaw*`/`pitch*`/`roll*` logs, printed as an `--opentrack-rotation-comp` matrix.
pub(crate) fn print_rotation_translation_fit(logs: &[LabelledFrames]) {
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [[0.0; 3]; 3];
    let mut samples = 0usize;

    for (label, frames) in logs {
        if !(label.starts_with("yaw") || label.starts_with("pitch") || label.starts_with("roll")) {
            continue;
        }

        let mut origin_head = None::<[f64; 3]>;
        let mut origin_angles = None::<[f64; 3]>;

        for (packet, values) in (0u64..).zip(frames) {
            let frame = TrackingFrame::from_decoded(packet, values);
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

            add_outer_product(&mut ata, angle_delta, angle_delta);
            add_outer_product(&mut atb, translation, angle_delta);
            samples += 1;
        }
    }

    println!("Fitted --opentrack-rotation-comp from rotation logs ({samples} samples):");
    if samples == 0 {
        println!("  not enough rotation samples");
        return;
    }

    let matrix = solve_rows(ata, atb);
    print_fit_matrix(&matrix);
    println!("  rows are tx,ty,tz in cm; columns are yaw,pitch,roll in deg");
}

/// Raw `[yaw, -pitch, 0]` from the default `--opentrack` angle field, if present.
#[must_use]
pub(crate) fn default_raw_angles(values: &DecodedFrame) -> Option<[f64; 3]> {
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

/// Per-axis `(raw, compensated)` angle ranges over `frames`, where
/// `compensated = raw - matrix * translation`; `None` with fewer than 8 frames.
#[must_use]
pub(crate) fn angle_ranges_with_comp(
    frames: &[DecodedFrame],
    matrix: [[f64; 3]; 3],
) -> Option<([f64; 3], [f64; 3])> {
    let mut origin_head = None::<[f64; 3]>;
    let mut origin_angles = None::<[f64; 3]>;
    let mut raw_min = [f64::INFINITY; 3];
    let mut raw_max = [f64::NEG_INFINITY; 3];
    let mut comp_min = [f64::INFINITY; 3];
    let mut comp_max = [f64::NEG_INFINITY; 3];
    let mut count = 0usize;

    for (packet, values) in (0u64..).zip(frames) {
        let frame = TrackingFrame::from_decoded(packet, values);
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

/// A decoded field whose mean differs between the compared logs.
#[derive(Clone, Debug)]
pub(crate) struct DecodedCandidate {
    /// Field identity.
    pub(crate) key: FieldKey,
    /// Per-log mean.
    pub(crate) means: Vec<f64>,
    /// Per-log `max - min`.
    pub(crate) ranges: Vec<f64>,
    /// Between-log spread divided by mean within-log range.
    pub(crate) score: f64,
    /// Spread of the per-log means.
    pub(crate) between: f64,
}

/// Compare decoded field statistics across labelled logs and print the fields
/// whose means differ most relative to their in-log range.
///
/// # Errors
/// Returns an error when a log cannot be read or a payload fails to decode.
pub(crate) fn compare_decoded(inputs: &[LogInput]) -> Result<()> {
    let mut logs = Vec::<(String, DecodedFields)>::with_capacity(inputs.len());

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
            let mut means = Vec::with_capacity(logs.len());
            let mut ranges = Vec::with_capacity(logs.len());
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
                candidates.push(DecodedCandidate {
                    key: *key,
                    means,
                    ranges,
                    score,
                    between,
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.key.cmp(&b.key))
    });

    println!("Top decoded cross-log candidates:");
    print!(
        "  {:>10} {:>3} {:>4} {:>9} {:>12}",
        "id", "occ", "cmp", "score", "between"
    );
    for (label, _) in &logs {
        print!(" {label:>12}");
    }
    println!();

    for candidate in candidates.iter().take(80) {
        let (id, occurrence, component) = candidate.key;
        print!(
            "  0x{id:08x} {occurrence:>3} {component:>4} {:>9.4} {:>12.4}",
            candidate.score, candidate.between
        );
        for mean in &candidate.means {
            print!(" {mean:>12.4}");
        }
        println!();
    }

    print_decoded_pair("head_left", "head_right", &logs, &candidates, 32);
    print_decoded_pair("eyes_left", "eyes_right", &logs, &candidates, 32);

    Ok(())
}

/// Collect, per payload, the 3D points of the requested landmark occurrences.
/// Frames where any requested point is missing or all-zero (no face) are skipped.
fn collect_landmark_frames(payloads: &[Vec<u8>], occs: &[usize]) -> Result<Vec<Vec<[f64; 3]>>> {
    let mut frames = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let values = decode_stream_payload(payload)?;
        let pts: Option<Vec<[f64; 3]>> = occs
            .iter()
            .map(|&o| head_point(&values, o).filter(|p| p.iter().any(|v| v.abs() > 1e-6)))
            .collect();
        if let Some(pts) = pts {
            frames.push(pts);
        }
    }
    Ok(frames)
}

/// Point-wise mean of the first `nref` frames (the neutral reference shape).
fn reference_shape(frames: &[Vec<[f64; 3]>], nref: usize, n_points: usize) -> Vec<[f64; 3]> {
    let mut reference = vec![[0.0; 3]; n_points];
    for frame in &frames[..nref] {
        for (acc, p) in reference.iter_mut().zip(frame) {
            for (acc_k, p_k) in acc.iter_mut().zip(p) {
                *acc_k += p_k / nref as f64;
            }
        }
    }
    reference
}

/// Research: reconstruct gaze-independent head pose from the 0x83 stream by a
/// rigid (Kabsch) fit over the *rigid* head landmark points (occurrences that
/// move with the head but not with gaze — identified as 5,6,9 + 0,1,4). This is
/// how the Windows Stream Engine derives head pose without the camera.
///
/// # Errors
/// Returns an error when the log cannot be read, a payload fails to decode, or
/// fewer than 31 frames contain all requested landmarks.
pub(crate) fn head83(path: &str, occs: &[usize]) -> Result<()> {
    let payloads = main_stream_payloads(path)?;

    let frames = collect_landmark_frames(&payloads, occs)?;
    anyhow::ensure!(
        frames.len() > 30,
        "too few valid face frames in {path} ({})",
        frames.len()
    );

    // Reference shape = mean of the first 30 valid frames (neutral pose).
    let nref = 30.min(frames.len());
    let reference = reference_shape(&frames, nref, occs.len());
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
        "{label}: {} valid / {} packets, occ={occs:?}",
        frames.len(),
        payloads.len()
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
        println!("  {name:5} {mn:10.2} {mx:10.2} {:10.2} {sd:10.2}", mx - mn);
    }
    Ok(())
}

/// Mean of `pts` (zero for an empty slice).
#[must_use]
fn centroid3(pts: &[[f64; 3]]) -> [f64; 3] {
    let n = pts.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in pts {
        for (c_k, p_k) in c.iter_mut().zip(p) {
            *c_k += p_k / n;
        }
    }
    c
}

/// The rigid 0x00031f41 landmark subset that gives the strongest yaw response
/// with the least shift leak (see `head-axes` analysis).
const RIGID_OCCS: [usize; 5] = [3, 5, 6, 8, 9];

/// Rotation vector (axis * angle, degrees) of a rotation matrix. Translation-
/// invariant by construction — this is the head rotation with the centroid
/// (position) already factored out by Kabsch.
#[must_use]
fn rotvec_deg(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos(); // radians, 0..pi
    let axis = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n < 1e-9 {
        return [0.0; 3];
    }
    let s = angle.to_degrees() / n;
    [axis[0] * s, axis[1] * s, axis[2] * s]
}

/// Dominant eigenvector (unit) of a symmetric 3x3 via power iteration. Used to
/// recover a rotation axis from the spread of per-frame rotation vectors —
/// robust to symmetric left/right motion whose vectors would cancel in a mean.
#[must_use]
fn dominant_axis(m: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut v = [1.0, 0.3, 0.7];
    for _ in 0..200 {
        let w = [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)];
        let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if n < 1e-30 {
            break;
        }
        v = [w[0] / n, w[1] / n, w[2] / n];
    }
    v
}

/// Cross product `a × b`.
#[must_use]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `a` scaled to unit length (returned unchanged when (near) zero).
#[must_use]
fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot3(a, a).sqrt();
    if n < 1e-12 {
        a
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

/// Decode a dump into per-frame rotation vectors of the rigid head subset,
/// relative to the mean of the first 30 valid frames (neutral pose).
fn dump_rotvecs(path: &str) -> Result<Vec<[f64; 3]>> {
    let payloads = main_stream_payloads(path)?;
    let frames = collect_landmark_frames(&payloads, &RIGID_OCCS)?;
    anyhow::ensure!(frames.len() > 30, "too few valid frames in {path}");
    let nref = 30.min(frames.len());
    let reference = reference_shape(&frames, nref, RIGID_OCCS.len());
    Ok(frames
        .iter()
        .map(|f| rotvec_deg(&kabsch(&reference, f)))
        .collect())
}

/// Analysis: recover the head's rotation axes (in the tilted sensor frame) from
/// controlled yaw/pitch/roll dumps, then show that projecting each frame's
/// Kabsch rotation onto those axes yields yaw/pitch/roll that are decoupled
/// from each other AND from translation. Pass labelled dumps; labels starting
/// with `yaw`/`pitch`/`roll` define the basis, `shift*` are leak tests.
///
/// # Errors
/// Returns an error when a dump cannot be read or decoded, or has too few
/// valid face frames.
pub(crate) fn head_axes(inputs: &[LogInput]) -> Result<()> {
    let mut dumps: Vec<(String, Vec<[f64; 3]>, [f64; 3])> = Vec::with_capacity(inputs.len());
    println!("Per-dump rotation axis (Kabsch on occ {RIGID_OCCS:?}, sensor frame):");
    println!(
        "  {:<10} {:>5} {:>22} {:>8}",
        "dump", "n", "axis (x,y,z)", "rot°range"
    );
    for input in inputs {
        let rvs = dump_rotvecs(&input.path)?;
        // Dominant axis = top eigenvector of sum of outer products rv·rvᵀ.
        let mut m = [[0.0; 3]; 3];
        for &rv in &rvs {
            add_outer_product(&mut m, rv, rv);
        }
        let axis = dominant_axis(&m);
        // Orient the axis so the dump's net motion is positive along it.
        let mags: Vec<f64> = rvs.iter().map(|&rv| dot3(rv, axis)).collect();
        let (mn, mx, _) = min_max_std(&mags);
        println!(
            "  {:<10} {:>5} {:>7.3},{:>6.3},{:>6.3} {:>8.1}",
            input.label,
            rvs.len(),
            axis[0],
            axis[1],
            axis[2],
            mx - mn
        );
        dumps.push((input.label.clone(), rvs, axis));
    }

    // Build an orthonormal head basis from the yaw/pitch axes (roll = yaw×pitch).
    let find = |p: &str| {
        dumps
            .iter()
            .find(|(l, ..)| l.starts_with(p))
            .map(|(_, _, a)| *a)
    };
    let (Some(yaw_ax), Some(pitch_ax)) = (find("yaw"), find("pitch")) else {
        println!("\n(need both a `yaw…` and a `pitch…` dump to build the head basis)");
        return Ok(());
    };
    let u_yaw = unit(yaw_ax);
    // Orthogonalize pitch against yaw (Gram-Schmidt), roll = yaw × pitch.
    let p_proj = dot3(pitch_ax, u_yaw);
    let pitch_o = [
        pitch_ax[0] - p_proj * u_yaw[0],
        pitch_ax[1] - p_proj * u_yaw[1],
        pitch_ax[2] - p_proj * u_yaw[2],
    ];
    let u_pitch = unit(pitch_o);
    let u_roll = unit(cross(u_yaw, u_pitch));

    let yaw_pitch_angle = dot3(u_yaw, unit(pitch_ax))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    println!(
        "\nHead basis (sensor frame): yaw {u_yaw:.3?}, pitch⊥ {u_pitch:.3?}, roll {u_roll:.3?}"
    );
    println!("  raw yaw/pitch axes are {yaw_pitch_angle:.1}° apart (90° = orthogonal)");
    let tilt = u_yaw[2].atan2(u_yaw[1]).to_degrees();
    println!("  yaw axis tilt from sensor-Y toward Z: {tilt:.1}°  (the source of yaw→roll bleed)");

    // Project every dump's rotation onto the head basis → decoupled angles.
    println!("\nDecoupled angle ranges (projection onto head basis), degrees:");
    println!("  {:<10} {:>9} {:>9} {:>9}", "dump", "yaw", "pitch", "roll");
    for (label, rvs, _) in &dumps {
        let proj = |u: [f64; 3]| {
            let v: Vec<f64> = rvs.iter().map(|&rv| dot3(rv, u)).collect();
            let (mn, mx, _) = min_max_std(&v);
            mx - mn
        };
        println!(
            "  {label:<10} {:>9.1} {:>9.1} {:>9.1}",
            proj(u_yaw),
            proj(u_pitch),
            proj(u_roll)
        );
    }
    Ok(())
}

/// `(min, max, population std-dev)` of `v` (`NaN`s for an empty slice).
#[must_use]
fn min_max_std(v: &[f64]) -> (f64, f64, f64) {
    let mn = v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    (mn, mx, var.sqrt())
}

/// Print the decoded fields that best separate the two named logs.
pub(crate) fn print_decoded_pair(
    left_label: &str,
    right_label: &str,
    logs: &[(String, DecodedFields)],
    candidates: &[DecodedCandidate],
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
            let delta = candidate.means[right] - candidate.means[left];
            let noise = (candidate.ranges[left] + candidate.ranges[right]) / 2.0 + 1.0;
            (delta.abs() / noise, delta, candidate)
        })
        .filter(|(score, delta, _)| *score >= 0.02 && delta.abs() > 1.0)
        .collect();

    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.key.cmp(&b.2.key))
    });

    println!("Top decoded {left_label} vs {right_label}:");
    println!(
        "  {:>10} {:>3} {:>4} {:>9} {:>12} {:>12} {:>12}",
        "id", "occ", "cmp", "score", left_label, right_label, "delta"
    );
    for (score, delta, candidate) in pairs.into_iter().take(limit) {
        let (id, occurrence, component) = candidate.key;
        println!(
            "  0x{id:08x} {occurrence:>3} {component:>4} {score:>9.4} {:>12.4} {:>12.4} {delta:>12.4}",
            candidate.means[left], candidate.means[right]
        );
    }
}

/// Statistics of one fixed-width numeric interpretation at a byte offset.
#[derive(Clone, Debug)]
pub(crate) struct NumericCandidate {
    /// Interpretation of the bytes (`u8`, `f32`, `u24`, ...).
    pub(crate) kind: &'static str,
    /// Byte offset into the payload.
    pub(crate) offset: usize,
    /// Number of packets sampled.
    pub(crate) count: usize,
    /// Minimum value.
    pub(crate) min: f64,
    /// Maximum value.
    pub(crate) max: f64,
    /// Mean value.
    pub(crate) mean: f64,
    /// Population standard deviation.
    pub(crate) stddev: f64,
    /// Value in the first packet.
    pub(crate) first: f64,
    /// Value in the last packet.
    pub(crate) last: f64,
    /// Number of packet-to-packet changes.
    pub(crate) changes: usize,
    /// `stddev * ln(1 + changes)`.
    pub(crate) score: f64,
}

/// Read `kind` at `offset` from every payload; `None` when any payload is too short.
fn column<F>(payloads: &[Vec<u8>], offset: usize, read_value: F) -> Option<Vec<f64>>
where
    F: Fn(&[u8], usize) -> Option<f64>,
{
    payloads
        .iter()
        .map(|payload| read_value(payload, offset))
        .collect()
}

/// Scan every byte offset of the stream packets for varying `u8`/`f32`/`i16`/
/// `u16`/`u24` interpretations and print the top candidates per kind.
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
        let Some(values) = column(payloads, offset, u8_at) else {
            continue;
        };
        if let Some(candidate) = summarize_numeric("u8", offset, &values)
            && candidate.stddev >= 1.0
        {
            u8_candidates.push(candidate);
        }
    }

    for offset in 0..min_len.saturating_sub(3) {
        let Some(values) = column(payloads, offset, f32_at_f64) else {
            continue;
        };
        if let Some(candidate) = summarize_numeric("f32", offset, &values)
            && candidate.min.is_finite()
            && candidate.max.is_finite()
            && candidate.min.abs() < 1000.0
            && candidate.max.abs() < 1000.0
            && candidate.stddev > 0.000001
        {
            f32_candidates.push(candidate);
        }
    }

    for offset in 0..min_len.saturating_sub(1) {
        let (Some(i16_values), Some(u16_values)) = (
            column(payloads, offset, i16_at),
            column(payloads, offset, u16_at),
        ) else {
            continue;
        };

        if let Some(candidate) = summarize_numeric("i16", offset, &i16_values)
            && candidate.stddev >= 1.0
        {
            i16_candidates.push(candidate);
        }

        if let Some(candidate) = summarize_numeric("u16", offset, &u16_values)
            && candidate.stddev >= 1.0
        {
            u16_candidates.push(candidate);
        }
    }

    for offset in 0..min_len.saturating_sub(2) {
        let Some(values) = column(payloads, offset, u24_at) else {
            continue;
        };
        if let Some(candidate) = summarize_numeric("u24", offset, &values)
            && candidate.stddev >= 1.0
        {
            u24_candidates.push(candidate);
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

/// Print a histogram of stream packet lengths.
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

/// Summarise one column of values; `None` when empty, non-finite, or constant.
#[must_use]
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

/// Sort by descending score, then ascending offset.
pub(crate) fn sort_candidates(candidates: &mut [NumericCandidate]) {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.cmp(&b.offset))
    });
}

/// Print up to `limit` candidates under `title`.
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

/// Print the byte offsets where three consecutive `f32`s all vary plausibly
/// (candidate 3D vectors).
pub(crate) fn print_f32_triplets(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut candidates = Vec::new();

    for offset in 0..min_len.saturating_sub(11) {
        let (Some(xs), Some(ys), Some(zs)) = (
            column(payloads, offset, f32_at_f64),
            column(payloads, offset + 4, f32_at_f64),
            column(payloads, offset + 8, f32_at_f64),
        ) else {
            continue;
        };

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
            "  off={offset:<4} x=[{:>9.5}..{:>9.5}] std={:>9.5} y=[{:>9.5}..{:>9.5}] std={:>9.5} z=[{:>9.5}..{:>9.5}] std={:>9.5}",
            x.min, x.max, x.stddev, y.min, y.max, y.stddev, z.min, z.max, z.stddev
        );
    }
}

/// Print the contiguous byte ranges that differ between stream packets, with
/// the first/last packet's bytes (and `f32` interpretation when plausible).
pub(crate) fn print_stream_changes(payloads: &[Vec<u8>]) {
    let (Some(first_payload), Some(last_payload)) = (payloads.first(), payloads.last()) else {
        return;
    };
    if payloads.len() < 2 {
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut ranges = Vec::new();
    let mut start = None;

    for offset in 0..min_len {
        let first = first_payload[offset];
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
        let first_hex = short_hex(&first_payload[start..end]);
        let last_hex = short_hex(&last_payload[start..end]);

        print!("  off={start} len={width} first={first_hex} last={last_hex}");

        if start + 4 <= min_len
            && let (Some(first_f32), Some(last_f32)) =
                (f32_at(first_payload, start), f32_at(last_payload, start))
            && first_f32.is_finite()
            && last_f32.is_finite()
            && first_f32.abs() < 1000.0
            && last_f32.abs() < 1000.0
        {
            print!(" f32_first={first_f32:.6} f32_last={last_f32:.6}");
        }

        println!();
    }
}

/// A run of plausible calibration floats found in the init blob.
#[derive(Clone, Debug)]
pub(crate) struct CalibrationRecord {
    /// Offset of the record marker within the concatenated blob.
    pub(crate) offset: usize,
    /// Two or four floats following the marker.
    pub(crate) values: Vec<f32>,
}

/// Locate calibration-like float records in the large init blob packets of a
/// log, print them, and optionally write them as JSON.
///
/// # Errors
/// Returns an error when the log cannot be read, contains no large blob
/// packets, or the JSON file cannot be written.
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
                "  {:>3} 0x{:06x} {target_x:>10.4} {target_y:>10.4} {observed_x:>12.4} {observed_y:>12.4} {:>10.4} {:>10.4}",
                i + 1,
                record.offset,
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

/// Concatenate the bodies (after the 8-byte header) of every packet of at
/// least 512 bytes.
#[must_use]
pub(crate) fn calibration_blob(packets: &[InitPacket]) -> Vec<u8> {
    let mut blob = Vec::new();

    for packet in packets {
        if packet.data.len() >= 512 {
            blob.extend_from_slice(&packet.data[8..]);
        }
    }

    blob
}

/// Find `01 00.. 00` markers in the last 8192 bytes of `blob` followed by two or
/// four plausible (`-0.5..=1.5`) little-endian `f32`s.
#[must_use]
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

        if next_marker > value_offset && (next_marker - value_offset).is_multiple_of(4) {
            let value_count = (next_marker - value_offset) / 4;
            if matches!(value_count, 2 | 4) {
                let mut values = Vec::with_capacity(value_count);
                let mut plausible = true;

                let (chunks, _) = blob[value_offset..next_marker].as_chunks::<4>();
                for chunk in chunks {
                    let value = f32::from_le_bytes(*chunk);
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

/// Offset of the first occurrence of `needle` in `haystack`.
#[must_use]
pub(crate) fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// `value` with four decimals, or `-` when absent.
#[must_use]
pub(crate) fn fmt_cal_value(value: Option<f32>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.4}"))
}

/// Write `records` (and the four-float ones as target/observed points) as JSON.
///
/// # Errors
/// Returns an error when the file cannot be created or written.
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

/// Lowercase hex of the first 12 bytes, with `...` when truncated.
#[must_use]
pub(crate) fn short_hex(bytes: &[u8]) -> String {
    const MAX: usize = 12;
    let mut out = String::with_capacity(MAX * 2 + 3);

    for b in bytes.iter().take(MAX) {
        write!(out, "{b:02x}").expect("invariant: fmt::Write into a String cannot fail");
    }

    if bytes.len() > MAX {
        out.push_str("...");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_timestamp_us_handles_fraction_and_padding() {
        assert_eq!(parse_timestamp_us("12").ok(), Some(12_000_000));
        assert_eq!(parse_timestamp_us("12.5").ok(), Some(12_500_000));
        assert_eq!(parse_timestamp_us(" 1.2345678 ").ok(), Some(1_234_567));
        assert!(parse_timestamp_us("x.1").is_err());
        assert!(parse_timestamp_us(&u64::MAX.to_string()).is_err());
    }

    #[test]
    fn le_readers_return_none_when_short() {
        let payload = [0x01u8, 0x02, 0x03, 0x04, 0x05];
        assert_eq!(u8_at(&payload, 4), Some(5.0));
        assert_eq!(u8_at(&payload, 5), None);
        assert_eq!(u16_at(&payload, 0), Some(f64::from(0x0201_u16)));
        assert_eq!(i16_at(&payload, 3), Some(f64::from(0x0504_i16)));
        assert_eq!(i16_at(&payload, 4), None);
        assert_eq!(u24_at(&payload, 1), Some(f64::from(0x0004_0302_u32)));
        assert_eq!(u32_at(&payload, 1), Some(f64::from(0x0504_0302_u32)));
        assert_eq!(u32_at(&payload, 2), None);
        assert_eq!(f32_at(&1.5f32.to_le_bytes(), 0), Some(1.5));
        assert_eq!(f32_at_f64(&payload, 2), None);
    }

    #[test]
    fn add_outer_product_accumulates_a_i_times_b_j() {
        let mut acc = [[0.0; 3]; 3];
        add_outer_product(&mut acc, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);
        add_outer_product(&mut acc, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let want = [11.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0];
        for (got, want) in acc.iter().flatten().zip(want) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn find_calibration_records_accepts_two_and_four_float_runs() {
        let marker = b"\x01\0\0\0\0\0\0\0";
        let mut blob = Vec::new();
        blob.extend_from_slice(marker);
        for v in [0.25f32, 0.5, 0.75, 1.0] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(marker);
        blob.extend_from_slice(&5.0f32.to_le_bytes()); // implausible
        blob.extend_from_slice(&0.5f32.to_le_bytes());
        blob.extend_from_slice(marker);

        let records = find_calibration_records(&blob);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].offset, 0);
        assert_eq!(records[0].values, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn short_hex_truncates_after_twelve_bytes() {
        assert_eq!(short_hex(&[0xab, 0x01]), "ab01");
        let long: Vec<u8> = (0..13).collect();
        assert_eq!(short_hex(&long), "000102030405060708090a0b...");
        assert_eq!(fmt_cal_value(None), "-");
        assert_eq!(fmt_cal_value(Some(0.5)), "0.5000");
    }
}
