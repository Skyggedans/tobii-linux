use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::dashboard::{print_live_decoded, render_tracking_dashboard};
use crate::decode::{
    decode_stream_payload, derive_live_values, TrackingFrame, DERIVED_FIELDS, LIVE_FIELDS,
};
use crate::opentrack::OpentrackUdp;
use crate::protocol::{marker, LOG_MAGIC};

pub(crate) struct PacketLog {
    pub(crate) out: BufWriter<File>,
}

impl PacketLog {
    pub(crate) fn create(path: &str) -> Result<Self> {
        let mut out = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create log {path}"))?,
        );
        out.write_all(LOG_MAGIC)?;
        println!("Logging IN packets to {path}");
        Ok(Self { out })
    }

    pub(crate) fn write_record(&mut self, ep: u8, data: &[u8]) -> Result<()> {
        self.write_record_at(now_us(), ep, data)
    }

    pub(crate) fn write_record_at(&mut self, ts_us: u64, ep: u8, data: &[u8]) -> Result<()> {
        self.out.write_all(&[0, ep, 0, 0])?;
        self.out.write_all(&ts_us.to_le_bytes())?;
        self.out.write_all(&(data.len() as u32).to_le_bytes())?;
        self.out.write_all(data)?;
        self.out.flush()?;
        Ok(())
    }
}

pub(crate) struct DecodedCsv {
    pub(crate) out: BufWriter<File>,
}

impl DecodedCsv {
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

        println!("Logging decoded stream candidates to {path}");
        Ok(Self { out })
    }

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

pub(crate) struct JsonlOutput {
    pub(crate) out: BufWriter<Box<dyn Write>>,
}

impl JsonlOutput {
    pub(crate) fn create(path: &str) -> Result<Self> {
        let writer: Box<dyn Write> = if path == "-" {
            println!("Logging tracking frames as JSON Lines to stdout");
            Box::new(io::stdout())
        } else {
            println!("Logging tracking frames as JSON Lines to {path}");
            Box::new(
                File::create(path)
                    .with_context(|| format!("failed to create JSONL output {path}"))?,
            )
        };

        Ok(Self {
            out: BufWriter::new(writer),
        })
    }

    pub(crate) fn write_frame(&mut self, frame: &TrackingFrame) -> Result<()> {
        frame.write_json(&mut self.out)?;
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

pub(crate) fn write_csv_value(out: &mut BufWriter<File>, value: Option<f64>) -> Result<()> {
    match value {
        Some(value) => write!(out, ",{value:.6}")?,
        None => write!(out, ",")?,
    }
    Ok(())
}

pub(crate) fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

pub(crate) fn log_packet(log: &mut Option<PacketLog>, ep: u8, data: &[u8]) -> Result<()> {
    if let Some(log) = log {
        log.write_record(ep, data)?;
    }
    Ok(())
}

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
