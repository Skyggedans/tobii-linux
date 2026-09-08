//! The `TBI5LOG1` capture-log format: the writer used while recording a
//! session and the readers used to replay one offline.
//!
//! Writer and readers live together so the record layout is defined once.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use tracing::info;

use crate::decode::decode_stream_payload_with_status;
use crate::protocol::{MARKER_STREAM, marker};
use crate::time::now_us;

/// Magic at the start of a packet log written by [`PacketLog`].
pub const LOG_MAGIC: &[u8; 8] = b"TBI5LOG1";

/// Size of one packet-log record header (`ts_us` u64, `ep` u32, LE u32 payload length).
const LOG_RECORD_HEADER_LEN: usize = 16;

/// Raw USB packet log (`LOG_MAGIC` header, then one record per packet).
#[derive(Debug)]
pub struct PacketLog {
    out: BufWriter<File>,
}

impl PacketLog {
    /// Create (truncate) the log file at `path` and write the magic header.
    ///
    /// # Errors
    /// Fails when the file cannot be created or the header cannot be written.
    pub fn create(path: &str) -> Result<Self> {
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
    pub fn write_record(&mut self, ep: u8, data: &[u8]) -> Result<()> {
        self.write_record_at(now_us(), ep, data)
    }

    /// Append one packet record: `[0, ep, 0, 0]`, `ts_us` (LE u64), length
    /// (LE u32), payload; flushed immediately so a crash loses nothing.
    ///
    /// # Errors
    /// Fails when the payload exceeds `u32::MAX` bytes or the write fails.
    pub fn write_record_at(&mut self, ts_us: u64, ep: u8, data: &[u8]) -> Result<()> {
        let len = u32::try_from(data.len()).context("packet too large for the log record")?;
        self.out.write_all(&[0, ep, 0, 0])?;
        self.out.write_all(&ts_us.to_le_bytes())?;
        self.out.write_all(&len.to_le_bytes())?;
        self.out.write_all(data)?;
        self.out.flush()?;
        Ok(())
    }
}

/// Append `data` to the packet log when one is open.
///
/// # Errors
/// See [`PacketLog::write_record`].
pub fn log_packet(log: &mut Option<PacketLog>, ep: u8, data: &[u8]) -> Result<()> {
    if let Some(log) = log {
        log.write_record(ep, data)?;
    }
    Ok(())
}

/// Read every record payload of a packet log (see [`LOG_MAGIC`]).
///
/// # Errors
///
/// Fails if the file cannot be opened, the magic does not match, or a record
/// is truncated after its header.
pub fn read_log_payloads(path: &str) -> Result<Vec<Vec<u8>>> {
    let mut input =
        BufReader::new(File::open(path).with_context(|| format!("failed to open log {path}"))?);
    let mut magic = [0u8; LOG_MAGIC.len()];
    input
        .read_exact(&mut magic)
        .with_context(|| format!("failed to read log magic of {path}"))?;
    anyhow::ensure!(&magic == LOG_MAGIC, "bad log magic in {path}");

    let mut payloads = Vec::new();

    loop {
        let mut header = [0u8; LOG_RECORD_HEADER_LEN];
        match input.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("failed to read log record header"),
        }

        let len = usize::try_from(u32::from_le_bytes(header[12..16].try_into()?))
            .context("log record length does not fit in usize")?;
        let mut data = vec![0u8; len];
        input
            .read_exact(&mut data)
            .with_context(|| format!("truncated log record of {len} bytes"))?;
        payloads.push(data);
    }

    Ok(payloads)
}

/// Stream (`0x53`) payloads of the dominant gaze message length in a log.
///
/// The main stream is picked as the payload length with the most decodable
/// messages (ties broken by count), which skips presence/image messages that
/// share EP 0x83.
///
/// # Errors
///
/// Propagates [`read_log_payloads`] failures.
pub fn main_stream_payloads(path: &str) -> Result<Vec<Vec<u8>>> {
    let stream: Vec<Vec<u8>> = read_log_payloads(path)?
        .into_iter()
        .filter(|payload| marker(payload) == Some(MARKER_STREAM))
        .collect();

    let mut lengths = BTreeMap::<usize, (usize, usize)>::new();
    for payload in &stream {
        let entry = lengths.entry(payload.len()).or_default();
        entry.0 += 1;

        if let Ok((values, false)) = decode_stream_payload_with_status(payload)
            && !values.is_empty()
        {
            entry.1 += 1;
        }
    }

    let main_len = lengths
        .into_iter()
        .max_by_key(|(_, (count, decoded))| (*decoded, *count))
        .map(|(len, _)| len)
        .unwrap_or(0);

    Ok(stream
        .into_iter()
        .filter(|payload| payload.len() == main_len)
        .collect())
}
