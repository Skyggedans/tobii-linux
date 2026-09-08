//! Wire-level helpers for the Tobii 5 USB protocol: the `init_packets` text
//! format, message prefix/marker/seq accessors, the bulk-read reassembler for
//! EP 0x83, and the stream start/stop commands.
//!
//! The capture-log format lives in [`crate::log`].
//!
//! Everything here is byte-exact: the constants and offsets are protocol facts
//! recovered from Windows Stream Engine captures, so keep them as they are.

use anyhow::{Context, Result};
use std::fs;

/// Marker (BE u32 at offset 8) of a host->device command message.
pub(crate) const MARKER_COMMAND: u32 = 0x51;
/// Marker of a device->host response to a command (same `seq` as the command).
pub(crate) const MARKER_RESPONSE: u32 = 0x52;
/// Marker of a device->host stream message (gaze, presence, image).
pub(crate) const MARKER_STREAM: u32 = 0x53;
/// Marker of an unsolicited device->host notification.
pub(crate) const MARKER_NOTIFICATION: u32 = 0x4e;

/// One line of an `init_packets` file: the OUT endpoint and the raw payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InitPacket {
    /// USB endpoint the payload was written to (e.g. `0x05`).
    pub(crate) ep: u8,
    /// Raw bytes as sent on the wire.
    pub(crate) data: Vec<u8>,
}

/// Read and parse an `init_packets` text file (see [`parse_init_packets`]).
///
/// # Errors
///
/// Fails if the file cannot be read or a line does not parse.
pub(crate) fn read_init_packets(path: &str) -> Result<Vec<InitPacket>> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?;
    parse_init_packets(&text)
}

/// Parse the `init_packets` format: one `<ep-hex> <payload-hex>` pair per
/// line; blank lines and `#` comments are skipped.
///
/// # Errors
///
/// Fails on a line missing either column, a non-hex endpoint, or a payload
/// that is not an even-length hex string.
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

/// Decode a hex string (optionally `:`-separated, surrounding whitespace
/// ignored) into bytes.
///
/// # Errors
///
/// Fails on an odd number of hex digits or any non-hex byte pair.
pub(crate) fn hex_to_bytes(s: &str) -> Result<Vec<u8>> {
    let s = s.trim().replace(':', "");

    if !s.len().is_multiple_of(2) {
        anyhow::bail!("odd hex length");
    }

    let mut bytes = Vec::with_capacity(s.len() / 2);

    // Walk the bytes, not char indices: slicing a `str` at an even byte
    // offset panics on a non-ASCII char boundary, whereas a non-UTF-8 pair
    // here is just a bad hex byte.
    for (i, pair) in s.as_bytes().chunks_exact(2).enumerate() {
        let b = std::str::from_utf8(pair)
            .ok()
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .with_context(|| format!("bad hex byte at {i}"))?;
        bytes.push(b);
    }

    Ok(bytes)
}

/// Total message length declared in the prefix (LE u32 at offset 4).
///
/// For device->host messages this includes the 8-byte prefix; for host->device
/// commands (see [`stream_start_packet`]) it is the body length after it.
#[must_use]
pub(crate) fn declared_len(buf: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(4..8)?.try_into().ok()?))
}

/// Message marker (BE u32 at offset 8): one of the `MARKER_*` constants.
#[must_use]
pub(crate) fn marker(buf: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(buf.get(8..12)?.try_into().ok()?))
}

/// Command / response sequence number (BE u32 at offset 12).
#[must_use]
pub(crate) fn seq(buf: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(buf.get(12..16)?.try_into().ok()?))
}

// ---------------------------------------------------------------------------
// Stream multiplexing on EP 0x83.
//
// Every device->host message starts with an 8-byte prefix (`01 00 00 00` +
// LE u32 total length including the prefix), then a BE u32 marker at +8
// (0x51 command, 0x52 response, 0x53 stream, 0x4e notification). Stream
// messages carry their stream id as a BE u32 at +20. The device multiplexes
// several streams on the one bulk endpoint; the ones we use:
//   0x500 gaze (1724 B, ~33 Hz), 0x504 presence (101 B), and
//   0x50e primary_camera_image (78609 B: 280x280 8-bit IR frame, ~33 Hz).
// Streams are started with command 1220 (0x4c4) and stopped with 1230 (0x4ce),
// each carrying the stream id (see `stream_start_packet`).
// ---------------------------------------------------------------------------

/// Gaze stream (1724-byte messages, ~33 Hz).
pub(crate) const STREAM_ID_GAZE: u32 = 0x500;
/// Presence stream (101-byte messages).
pub(crate) const STREAM_ID_PRESENCE: u32 = 0x504;
/// `primary_camera_image` stream (78609-byte messages, see [`crate::image83`]).
pub(crate) const STREAM_ID_IMAGE: u32 = 0x50e;

/// Stream id of a 0x53 stream message (BE u32 at payload offset 20).
#[must_use]
pub(crate) fn stream_id(buf: &[u8]) -> Option<u32> {
    if marker(buf) != Some(MARKER_STREAM) {
        return None;
    }
    Some(u32::from_be_bytes(buf.get(20..24)?.try_into().ok()?))
}

/// Length of the device->host message prefix plus marker: the minimum a
/// buffer needs before it can be classified.
const MIN_MESSAGE_LEN: u32 = 12;

/// Upper bound on a sane message: the image stream is 78609 bytes; anything
/// far beyond that means we lost sync and should drop the buffer.
const MAX_MESSAGE_LEN: u32 = 1 << 20;

/// Every device->host message starts with the tag `01 00 00 00`, a plausible
/// total length, and one of the known markers.
fn looks_like_prefix(b: &[u8]) -> bool {
    b.len() >= 12
        && b[..4] == [1, 0, 0, 0]
        && matches!(declared_len(b), Some(n) if (MIN_MESSAGE_LEN..=MAX_MESSAGE_LEN).contains(&n))
        && matches!(
            marker(b),
            Some(MARKER_COMMAND | MARKER_RESPONSE | MARKER_STREAM | MARKER_NOTIFICATION)
        )
}

/// Reassembles whole device->host messages from raw bulk reads.
///
/// A bulk read normally returns exactly one message (every message ends in a
/// short USB packet), but a read buffer smaller than the message, or a stall,
/// can split one message across reads. This stitches by the declared length
/// in the message prefix and resyncs by dropping bytes when the prefix is
/// implausible.
#[derive(Debug, Default)]
pub(crate) struct BulkReassembler {
    pending: Vec<u8>,
}

impl BulkReassembler {
    /// An empty reassembler.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed one bulk read; returns every complete message it completes.
    ///
    /// Allocating convenience wrapper around [`Self::push_into`], kept for the
    /// tests; the reader paths reuse one vector instead.
    #[cfg(test)]
    pub(crate) fn push(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        self.push_into(data, &mut out);
        out
    }

    /// Feed one bulk read, appending every completed message to `out`.
    ///
    /// `out` is cleared first, so a caller in a read loop can hand the same
    /// vector back on every iteration and keep its allocation.
    pub(crate) fn push_into(&mut self, data: &[u8], out: &mut Vec<Vec<u8>>) {
        out.clear();
        if data.is_empty() {
            return;
        }
        self.pending.extend_from_slice(data);
        loop {
            if self.pending.len() < 12 {
                return;
            }
            if !looks_like_prefix(&self.pending) {
                // Lost sync (e.g. the tail of a message whose head was read
                // elsewhere): skip forward to the next plausible message start
                // rather than dropping whole reads.
                match (1..self.pending.len().saturating_sub(11))
                    .find(|&i| looks_like_prefix(&self.pending[i..]))
                {
                    Some(i) => {
                        self.pending.drain(..i);
                    }
                    None => {
                        self.pending.clear();
                        return;
                    }
                }
                continue;
            }
            // `looks_like_prefix` bounds the declared length to
            // MIN_MESSAGE_LEN..=MAX_MESSAGE_LEN, so it always fits a usize;
            // the fallback only guards against a zero-length loop.
            let Some(total) = declared_len(&self.pending).and_then(|n| usize::try_from(n).ok())
            else {
                self.pending.clear();
                return;
            };
            if self.pending.len() < total {
                return;
            }
            let rest = self.pending.split_off(total);
            let msg = std::mem::replace(&mut self.pending, rest);
            out.push(msg);
        }
    }

    /// Bytes buffered but not yet forming a message.
    #[must_use]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

const CMD_STREAM_START: u32 = 1220;
const CMD_STREAM_STOP: u32 = 1230;

/// Bytes from the marker through the payload-length word of a command:
/// six BE u32 words (marker, seq, 0, command, 0, payload length).
const CMD_HEADER_LEN: u32 = 4 * 6;

/// Build a host->device command message on EP 0x05 for the stream start/stop
/// commands, mirroring the last lines of `init_packets_ep.txt`:
/// prefix `00 00 00 00` + LE u32 (body length), then BE marker 0x51, seq,
/// 0, command, 0, payload length, and the TLV payload
/// `00 00` + type 2/len 4/stream id [+ type 0x17/len 4/0 for start].
fn stream_command_packet(cmd: u32, seq: u32, id: u32, with_flags: bool) -> Vec<u8> {
    let mut payload = vec![0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04];
    payload.extend_from_slice(&id.to_be_bytes());
    if with_flags {
        payload.extend_from_slice(&[0x17, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00]);
    }
    let payload_len =
        u32::try_from(payload.len()).expect("invariant: stream command payload is 11 or 20 bytes");
    let body_len = CMD_HEADER_LEN + payload_len;
    let mut v = Vec::with_capacity(8 + payload.len() + 4 * 6);
    v.extend_from_slice(&[0, 0, 0, 0]);
    v.extend_from_slice(&body_len.to_le_bytes());
    for word in [MARKER_COMMAND, seq, 0, cmd, 0, payload_len] {
        v.extend_from_slice(&word.to_be_bytes());
    }
    v.extend_from_slice(&payload);
    v
}

/// Command 1220: start streaming `id` (e.g. `STREAM_ID_IMAGE`).
#[must_use]
pub(crate) fn stream_start_packet(seq: u32, id: u32) -> Vec<u8> {
    stream_command_packet(CMD_STREAM_START, seq, id, true)
}

/// Command 1230: stop streaming `id`.
#[must_use]
pub(crate) fn stream_stop_packet(seq: u32, id: u32) -> Vec<u8> {
    stream_command_packet(CMD_STREAM_STOP, seq, id, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(stream: u32, body_len: usize) -> Vec<u8> {
        // prefix + marker + seq + 2 words + stream id + body
        let total = 8 + 16 + body_len;
        let mut v = vec![1, 0, 0, 0];
        v.extend_from_slice(
            &u32::try_from(total)
                .expect("test message fits u32")
                .to_le_bytes(),
        );
        v.extend_from_slice(&MARKER_STREAM.to_be_bytes());
        v.extend_from_slice(&[0; 8]);
        v.extend_from_slice(&stream.to_be_bytes());
        v.extend(std::iter::repeat_n(0xabu8, body_len));
        assert_eq!(v.len(), total);
        v
    }

    #[test]
    fn stream_id_reads_offset_20() {
        let m = msg(STREAM_ID_IMAGE, 40);
        assert_eq!(stream_id(&m), Some(0x50e));
        assert_eq!(stream_id(&m[..20]), None);
        let mut cmd = m.clone();
        cmd[8..12].copy_from_slice(&MARKER_RESPONSE.to_be_bytes());
        assert_eq!(
            stream_id(&cmd),
            None,
            "only 0x53 messages carry a stream id"
        );
    }

    #[test]
    fn header_accessors_return_none_on_short_buffers() {
        let m = msg(STREAM_ID_GAZE, 4);
        assert_eq!(declared_len(&m[..7]), None);
        assert_eq!(marker(&m[..11]), None);
        assert_eq!(seq(&m[..15]), None);
        assert_eq!(declared_len(&m), Some(28));
        assert_eq!(marker(&m), Some(MARKER_STREAM));
        assert_eq!(seq(&m), Some(0));
    }

    #[test]
    fn hex_to_bytes_accepts_colons_and_rejects_bad_input() {
        assert_eq!(
            hex_to_bytes(" 01:ff:0A ").expect("valid hex"),
            vec![1, 0xff, 0x0a]
        );
        assert!(hex_to_bytes("abc").is_err(), "odd length");
        assert!(hex_to_bytes("zz").is_err(), "non-hex digit");
        // Multi-byte chars must produce an error, not a str-slicing panic.
        assert!(hex_to_bytes("aé").is_err());
        assert!(hex_to_bytes("aéa").is_err());
    }

    #[test]
    fn reassembler_passes_whole_messages_through() {
        let mut r = BulkReassembler::new();
        let a = msg(STREAM_ID_GAZE, 100);
        let got = r.push(&a);
        assert_eq!(got, vec![a]);
        assert_eq!(r.pending_len(), 0);
    }

    #[test]
    fn reassembler_stitches_split_message_and_splits_coalesced_reads() {
        let mut r = BulkReassembler::new();
        let big = msg(STREAM_ID_IMAGE, 70_000);
        let small = msg(STREAM_ID_GAZE, 50);
        // Split like a 16 KiB read buffer would.
        let mut got = Vec::new();
        for chunk in big.chunks(16384) {
            got.extend(r.push(chunk));
        }
        assert_eq!(got, vec![big.clone()]);
        // Two messages arriving in one read.
        let mut both = big.clone();
        both.extend_from_slice(&small);
        let got = r.push(&both);
        assert_eq!(got, vec![big, small]);
        assert_eq!(r.pending_len(), 0);
    }

    #[test]
    fn reassembler_resyncs_on_garbage() {
        let mut r = BulkReassembler::new();
        let garbage = vec![0xffu8; 32];
        assert!(r.push(&garbage).is_empty());
        assert_eq!(r.pending_len(), 0, "no prefix anywhere drops the buffer");
        let m = msg(STREAM_ID_PRESENCE, 10);
        assert_eq!(r.push(&m), vec![m]);
    }

    #[test]
    fn reassembler_skips_an_orphaned_tail_to_the_next_message() {
        // The tail of an image message (pixels containing a plausible-looking
        // length word) arrives first, then a real gaze message: the tail is
        // skipped and the gaze message comes out intact.
        let mut r = BulkReassembler::new();
        let mut tail = vec![0u8; 200];
        tail[4..8].copy_from_slice(&40u32.to_le_bytes()); // fake length, no tag
        let m = msg(STREAM_ID_GAZE, 60);
        let mut read = tail;
        read.extend_from_slice(&m);
        assert_eq!(r.push(&read), vec![m]);
        assert_eq!(r.pending_len(), 0);
        // A tail that itself contains a fake "01 00 00 00" tag with a bad marker
        // is still skipped.
        let mut tail2 = vec![0u8; 64];
        tail2[..4].copy_from_slice(&[1, 0, 0, 0]);
        tail2[4..8].copy_from_slice(&20u32.to_le_bytes());
        let m2 = msg(STREAM_ID_IMAGE, 30);
        let mut read2 = tail2;
        read2.extend_from_slice(&m2);
        assert_eq!(r.push(&read2), vec![m2]);
    }

    #[test]
    fn stream_start_packet_matches_captured_init_line() {
        // Last line of init_packets_ep.txt (start gaze 0x500, seq 0x28), then
        // the Windows session1 start of the image stream.
        let expected = hex_to_bytes(
            "000000002c000000000000510000002800000000000004c4000000000000001400000200000004000005\
             00170000000400000000",
        )
        .expect("valid hex");
        assert_eq!(stream_start_packet(0x28, STREAM_ID_GAZE), expected);
        let img = stream_start_packet(0x2a, STREAM_ID_IMAGE);
        assert_eq!(img.len(), expected.len());
        assert_eq!(&img[39..43], &[0, 0, 0x05, 0x0e]);
        assert_eq!(&img[12..16], &[0, 0, 0, 0x2a]);
    }

    #[test]
    fn stream_stop_packet_carries_only_the_id() {
        let p = stream_stop_packet(0x2b, STREAM_ID_IMAGE);
        assert_eq!(&p[8..12], &MARKER_COMMAND.to_be_bytes());
        assert_eq!(&p[20..24], &1230u32.to_be_bytes());
        assert_eq!(&p[28..32], &11u32.to_be_bytes(), "payload length");
        assert_eq!(p.len(), 8 + 24 + 11);
        let body_len = u32::try_from(p.len() - 8).expect("fits u32");
        assert_eq!(declared_len(&p), Some(body_len));
    }
}
