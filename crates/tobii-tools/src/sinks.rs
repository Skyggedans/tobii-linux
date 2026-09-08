//! File/stdout sinks for the replay/track CLI paths: the decoded-candidate
//! CSV and the JSON Lines frame stream, plus the
//! per-packet fan-out that feeds them together with the `OpenTrack` sender and
//! the terminal dashboard.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use tracing::info;

use crate::dashboard::{print_live_decoded, render_tracking_dashboard};
use crate::opentrack::OpentrackUdp;
use tobii_proto::decode::{
    DERIVED_FIELDS, LIVE_FIELDS, TrackingFrame, decode_stream_payload, derive_live_values,
};
use tobii_proto::protocol::marker;
use tobii_proto::time::now_us;

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
