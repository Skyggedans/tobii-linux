//! File/stdout sinks for the replay/track CLI paths: the raw packet log,
//! the decoded-candidate CSV and the JSON Lines frame stream, plus the
//! per-packet fan-out that feeds them together with the `OpenTrack` sender and
//! the terminal dashboard.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::dashboard::{print_live_decoded, render_tracking_dashboard};
use crate::decode::{
    DERIVED_FIELDS, LIVE_FIELDS, TrackingFrame, decode_stream_payload, derive_live_values,
};
use crate::opentrack::OpentrackUdp;
use crate::protocol::{LOG_MAGIC, marker};

/// Raw USB packet log (`LOG_MAGIC` header, then one record per packet).
#[derive(Debug)]
pub(crate) struct PacketLog {
    pub(crate) out: BufWriter<File>,
}

impl PacketLog {
    /// Create (truncate) the log file at `path` and write the magic header.
    ///
    /// # Errors
    /// Fails when the file cannot be created or the header cannot be written.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let mut out = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create log {path}"))?,
        );
        out.write_all(LOG_MAGIC)?;
        info!(path, "logging IN packets");
        Ok(Self { out })
    }

    /// Append one packet stamped with the current wall-clock time.
    ///
    /// # Errors
    /// See [`PacketLog::write_record_at`].
    pub(crate) fn write_record(&mut self, ep: u8, data: &[u8]) -> Result<()> {
        self.write_record_at(now_us(), ep, data)
    }

    /// Append one packet record: `[0, ep, 0, 0]`, `ts_us` (LE u64), length
    /// (LE u32), payload; flushed immediately so a crash loses nothing.
    ///
    /// # Errors
    /// Fails when the payload exceeds `u32::MAX` bytes or the write fails.
    pub(crate) fn write_record_at(&mut self, ts_us: u64, ep: u8, data: &[u8]) -> Result<()> {
        let len = u32::try_from(data.len()).context("packet too large for the log record")?;
        self.out.write_all(&[0, ep, 0, 0])?;
        self.out.write_all(&ts_us.to_le_bytes())?;
        self.out.write_all(&len.to_le_bytes())?;
        self.out.write_all(data)?;
        self.out.flush()?;
        Ok(())
    }
}

/// CSV of the decoded stream candidates, one row per 0x53 packet.
#[derive(Debug)]
pub(crate) struct DecodedCsv {
    pub(crate) out: BufWriter<File>,
}

impl DecodedCsv {
    /// Create (truncate) the CSV at `path` and write the header row.
    ///
    /// # Errors
    /// Fails when the file cannot be created or the header cannot be written.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let mut out = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create decoded CSV {path}"))?,
        );

        write!(out, "ts_us,packet")?;
        for field in DERIVED_FIELDS {
            write!(out, ",{field}")?;
        }
        for field in LIVE_FIELDS {
            write!(out, ",{}", field.name)?;
        }
        writeln!(out)?;

        info!(path, "logging decoded stream candidates");
        Ok(Self { out })
    }

    /// Append one row: timestamp, packet number, derived fields, live fields.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn write_packet(
        &mut self,
        packet_no: u64,
        values: &BTreeMap<(u32, usize, usize), f64>,
    ) -> Result<()> {
        let derived = derive_live_values(values);
        write!(self.out, "{},{}", now_us(), packet_no)?;
        for value in derived {
            write_csv_value(&mut self.out, value)?;
        }
        for field in LIVE_FIELDS {
            write_csv_value(&mut self.out, values.get(&field.key()).copied())?;
        }
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

/// JSON Lines stream of [`TrackingFrame`]s, to a file or stdout (`-`).
pub(crate) struct JsonlOutput {
    pub(crate) out: BufWriter<Box<dyn Write>>,
}

impl JsonlOutput {
    /// Open the JSONL sink; `-` selects stdout.
    ///
    /// # Errors
    /// Fails when the file cannot be created.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let writer: Box<dyn Write> = if path == "-" {
            info!(path = "stdout", "logging tracking frames as JSON Lines");
            Box::new(io::stdout())
        } else {
            info!(path, "logging tracking frames as JSON Lines");
            Box::new(
                File::create(path)
                    .with_context(|| format!("failed to create JSONL output {path}"))?,
            )
        };

        Ok(Self {
            out: BufWriter::new(writer),
        })
    }

    /// Append one frame as a single JSON line and flush.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn write_frame(&mut self, frame: &TrackingFrame) -> Result<()> {
        frame.write_json(&mut self.out)?;
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

/// Write one `,value` CSV cell (6 decimals), or a bare `,` for `None`.
///
/// # Errors
/// Fails when the write fails.
pub(crate) fn write_csv_value<W: Write>(out: &mut W, value: Option<f64>) -> Result<()> {
    match value {
        Some(value) => write!(out, ",{value:.6}")?,
        None => write!(out, ",")?,
    }
    Ok(())
}

/// Wall-clock microseconds since the Unix epoch (0 before the epoch,
/// saturating at `u64::MAX`).
#[must_use]
pub(crate) fn now_us() -> u64 {
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    u64::try_from(micros).unwrap_or(u64::MAX)
}

/// Append `data` to the packet log when one is open.
///
/// # Errors
/// See [`PacketLog::write_record`].
pub(crate) fn log_packet(log: &mut Option<PacketLog>, ep: u8, data: &[u8]) -> Result<()> {
    if let Some(log) = log {
        log.write_record(ep, data)?;
    }
    Ok(())
}

/// Fan one IN packet out to every enabled sink. Non-stream packets (marker
/// other than 0x53) and empty decodes are ignored.
///
/// # Errors
/// Fails when decoding fails or any sink write/send fails.
pub(crate) fn handle_live_decoded(
    packet_no: u64,
    data: &[u8],
    live_csv: &mut Option<DecodedCsv>,
    jsonl: &mut Option<JsonlOutput>,
    opentrack: &mut Option<OpentrackUdp>,
    print_decoded: bool,
    dashboard: bool,
) -> Result<()> {
    if marker(data) != Some(0x53) {
        return Ok(());
    }

    let decoded = decode_stream_payload(data)?;
    if decoded.is_empty() {
        return Ok(());
    }

    let frame = TrackingFrame::from_decoded(packet_no, &decoded);

    if let Some(csv) = live_csv {
        csv.write_packet(packet_no, &decoded)?;
    }

    if let Some(jsonl) = jsonl {
        jsonl.write_frame(&frame)?;
    }

    let mut opentrack_pose = None;
    if let Some(opentrack) = opentrack {
        opentrack_pose = opentrack.send_frame(&frame, &decoded)?;
    }

    if print_decoded {
        print_live_decoded(&frame);
    }

    if dashboard {
        render_tracking_dashboard(&frame, opentrack_pose)?;
    }

    Ok(())
}
