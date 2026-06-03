use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};

use crate::decode::decode_stream_payload_with_status;

pub(crate) const LOG_MAGIC: &[u8; 8] = b"TBI5LOG1";

pub(crate) struct InitPacket {
    pub(crate) ep: u8,
    pub(crate) data: Vec<u8>,
}

pub(crate) fn read_log_payloads(path: &str) -> Result<Vec<Vec<u8>>> {
    let mut input =
        BufReader::new(File::open(path).with_context(|| format!("failed to open log {path}"))?);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == LOG_MAGIC, "bad log magic in {path}");

    let mut payloads = Vec::new();

    loop {
        let mut header = [0u8; 16];
        match input.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("failed to read log record header"),
        }

        let len = u32::from_le_bytes(header[12..16].try_into()?) as usize;
        let mut data = vec![0u8; len];
        input.read_exact(&mut data)?;
        payloads.push(data);
    }

    Ok(payloads)
}

pub(crate) fn main_stream_payloads(path: &str) -> Result<Vec<Vec<u8>>> {
    let stream: Vec<Vec<u8>> = read_log_payloads(path)?
        .into_iter()
        .filter(|payload| marker(payload) == Some(0x53))
        .collect();

    let mut lengths = BTreeMap::<usize, (usize, usize)>::new();
    for payload in &stream {
        let entry = lengths.entry(payload.len()).or_default();
        entry.0 += 1;

        if let Ok((values, false)) = decode_stream_payload_with_status(payload) {
            if !values.is_empty() {
                entry.1 += 1;
            }
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

pub(crate) fn read_init_packets(path: &str) -> Result<Vec<InitPacket>> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?;
    parse_init_packets(&text)
}

pub(crate) fn parse_init_packets(text: &str) -> Result<Vec<InitPacket>> {
    let mut packets = Vec::new();

    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();

        let ep_str = parts
            .next()
            .with_context(|| format!("missing endpoint at line {}", line_no + 1))?;

        let hex = parts
            .next()
            .with_context(|| format!("missing payload at line {}", line_no + 1))?;

        let ep = u8::from_str_radix(ep_str.trim_start_matches("0x"), 16)
            .with_context(|| format!("bad endpoint at line {}", line_no + 1))?;

        let data = hex_to_bytes(hex).with_context(|| format!("bad hex at line {}", line_no + 1))?;

        packets.push(InitPacket { ep, data });
    }

    Ok(packets)
}

pub(crate) fn hex_to_bytes(s: &str) -> Result<Vec<u8>> {
    let s = s.trim().replace(':', "");

    if s.len() % 2 != 0 {
        anyhow::bail!("odd hex length");
    }

    let mut bytes = Vec::with_capacity(s.len() / 2);

    for i in (0..s.len()).step_by(2) {
        let b = u8::from_str_radix(&s[i..i + 2], 16)
            .with_context(|| format!("bad hex byte at {}", i / 2))?;
        bytes.push(b);
    }

    Ok(bytes)
}

pub(crate) fn declared_len(buf: &[u8]) -> Option<u32> {
    if buf.len() < 8 {
        return None;
    }

    Some(u32::from_le_bytes(buf[4..8].try_into().ok()?))
}

pub(crate) fn marker(buf: &[u8]) -> Option<u32> {
    if buf.len() < 12 {
        return None;
    }

    Some(u32::from_be_bytes(buf[8..12].try_into().ok()?))
}

pub(crate) fn seq(buf: &[u8]) -> Option<u32> {
    if buf.len() < 16 {
        return None;
    }

    Some(u32::from_be_bytes(buf[12..16].try_into().ok()?))
}
