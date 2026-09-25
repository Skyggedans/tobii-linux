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
pub const MARKER_COMMAND: u32 = 0x51;
/// Marker of a device->host response to a command (same `seq` as the command).
pub const MARKER_RESPONSE: u32 = 0x52;
/// Marker of a device->host stream message (gaze, presence, image).
pub const MARKER_STREAM: u32 = 0x53;
/// Marker of an unsolicited device->host notification.
pub const MARKER_NOTIFICATION: u32 = 0x4e;

/// One line of an `init_packets` file: the OUT endpoint and the raw payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitPacket {
    /// USB endpoint the payload was written to (e.g. `0x05`).
    pub ep: u8,
    /// Raw bytes as sent on the wire.
    pub data: Vec<u8>,
}

/// Read and parse an `init_packets` text file (see [`parse_init_packets`]).
///
/// # Errors
///
/// Fails if the file cannot be read or a line does not parse.
pub fn read_init_packets(path: &str) -> Result<Vec<InitPacket>> {
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
pub fn parse_init_packets(text: &str) -> Result<Vec<InitPacket>> {
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
pub fn hex_to_bytes(s: &str) -> Result<Vec<u8>> {
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
pub fn declared_len(buf: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(4..8)?.try_into().ok()?))
}

/// Message marker (BE u32 at offset 8): one of the `MARKER_*` constants.
#[must_use]
pub fn marker(buf: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(buf.get(8..12)?.try_into().ok()?))
}

/// Command / response sequence number (BE u32 at offset 12).
#[must_use]
pub fn seq(buf: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(buf.get(12..16)?.try_into().ok()?))
}

fn be_word(buf: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        buf.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

/// Length of the message prefix plus the six-word header; the payload starts
/// here.
pub const HEADER_LEN: usize = 32;

/// The status word of every response ever captured.
pub const RESPONSE_STATUS_OK: u32 = 1;

/// TTP error codes a response carries in its error word, as the Windows
/// engine reads them (its parser at 0x18017b4c0). Every captured response
/// carries [`NONE`](ttp_error::NONE).
pub mod ttp_error {
    /// No error.
    pub const NONE: u32 = 0;
    /// The device is not in a state to run the command (for a command that
    /// needs a calibration session, taken to mean there is none; never
    /// captured).
    pub const BAD_STATE: u32 = 0x2000_0508;
    /// The device rejected a parameter.
    pub const INVALID_PARAMETER: u32 = 0x2000_0509;
}

/// A device->host (or host->device) message split into its header words and
/// payload.
///
/// Layout after the 8-byte prefix, as BE u32 words: marker, seq, status,
/// id, error, payload length. `id` is the command for a 0x51/0x52, the stream
/// id for a 0x53 and the notification id for a 0x4e; `status` is 1 on every
/// response ever captured and 0 elsewhere, and `error` is 0 on every message
/// ever captured. The payload is `00 00` + TLVs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Message<'a> {
    /// One of the `MARKER_*` values.
    pub marker: u32,
    /// Sequence number (commands and their responses).
    pub seq: u32,
    /// The response status word.
    pub status: u32,
    /// Command, stream or notification id.
    pub id: u32,
    /// The response's TTP error code, one of the [`ttp_error`] values; the
    /// Windows engine takes anything but 0 as a refusal.
    pub error: u32,
    /// Declared payload length; larger than `payload.len()` for the head of
    /// a chunked response.
    pub payload_len: u32,
    /// The payload bytes present in this message.
    pub payload: &'a [u8],
}

impl<'a> Message<'a> {
    /// The payload's TLV entries.
    #[must_use]
    pub fn tlvs(&self) -> crate::tlv::TlvIter<'a> {
        crate::tlv::payload_tlvs(self.payload)
    }
}

/// Split a whole message (prefix included) into header and payload.
#[must_use]
pub fn parse_message(buf: &[u8]) -> Option<Message<'_>> {
    Some(Message {
        marker: be_word(buf, 8)?,
        seq: be_word(buf, 12)?,
        status: be_word(buf, 16)?,
        id: be_word(buf, 20)?,
        error: be_word(buf, 24)?,
        payload_len: be_word(buf, 28)?,
        payload: buf.get(HEADER_LEN..)?,
    })
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
pub const STREAM_ID_GAZE: u32 = 0x500;
/// Presence stream (101-byte messages).
pub const STREAM_ID_PRESENCE: u32 = 0x504;
/// `primary_camera_image` stream (78609-byte messages, see [`crate::image83`]).
pub const STREAM_ID_IMAGE: u32 = 0x50e;

/// Stream id of a 0x53 stream message (BE u32 at payload offset 20).
#[must_use]
pub fn stream_id(buf: &[u8]) -> Option<u32> {
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

/// Upper bound on one continuation of a chunked response (a calibration blob
/// is ~660 KB and arrives as a single continuation).
const CONTINUATION_MAX_LEN: u32 = 4 << 20;

/// A continuation carries the 8-byte prefix but no header: whatever follows
/// the prefix is payload, so its "marker" word is arbitrary data.
fn looks_like_continuation(b: &[u8]) -> bool {
    b.len() >= 8
        && b[..4] == [1, 0, 0, 0]
        && matches!(declared_len(b), Some(n) if (9..=CONTINUATION_MAX_LEN).contains(&n))
}

fn is_known_marker(b: &[u8]) -> bool {
    matches!(
        marker(b),
        Some(MARKER_COMMAND | MARKER_RESPONSE | MARKER_STREAM | MARKER_NOTIFICATION)
    )
}

/// A response whose declared payload is larger than the message: the rest
/// follows in continuation messages.
fn missing_payload(msg: &[u8]) -> usize {
    if marker(msg) != Some(MARKER_RESPONSE) {
        return 0;
    }
    let declared = be_word(msg, 28)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(0);
    declared.saturating_sub(msg.len().saturating_sub(HEADER_LEN))
}

/// The head of a chunked response and how many payload bytes it still lacks.
#[derive(Debug)]
struct Continuation {
    msg: Vec<u8>,
    remaining: usize,
}

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
///
/// Large responses (a calibration read, ~660 KB) are chunked by the device:
/// a normal 0x52 message whose header declares more payload than it carries,
/// then continuation messages that have the 8-byte prefix but no header.
/// Those are appended to the head and the response is emitted whole, as if it
/// had been one message; stream messages that arrive in between still come
/// out on their own, in order.
#[derive(Debug, Default)]
pub struct BulkReassembler {
    pending: Vec<u8>,
    continuation: Option<Continuation>,
}

impl BulkReassembler {
    /// An empty reassembler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one bulk read; returns every complete message it completes.
    ///
    /// Allocating convenience wrapper around [`Self::push_into`], kept for the
    /// tests; the reader paths reuse one vector instead.
    #[cfg(test)]
    pub fn push(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        self.push_into(data, &mut out);
        out
    }

    /// Feed one bulk read, appending every completed message to `out`.
    ///
    /// `out` is cleared first, so a caller in a read loop can hand the same
    /// vector back on every iteration and keep its allocation.
    pub fn push_into(&mut self, data: &[u8], out: &mut Vec<Vec<u8>>) {
        out.clear();
        if data.is_empty() {
            return;
        }
        self.pending.extend_from_slice(data);
        loop {
            if self.continuation.is_some()
                && looks_like_continuation(&self.pending)
                && (self.pending.len() < 12 || !is_known_marker(&self.pending))
            {
                if !self.take_continuation(out) {
                    return;
                }
                continue;
            }
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
            let remaining = missing_payload(&msg);
            if remaining > 0 {
                if self.continuation.is_some() {
                    tracing::warn!(
                        "new chunked response before the last one completed; dropping it"
                    );
                }
                self.continuation = Some(Continuation { msg, remaining });
            } else {
                out.push(msg);
            }
        }
    }

    /// Consume one continuation at the head of `pending`. Returns `false` when
    /// it is not complete yet.
    fn take_continuation(&mut self, out: &mut Vec<Vec<u8>>) -> bool {
        let Some(total) = declared_len(&self.pending).and_then(|n| usize::try_from(n).ok()) else {
            return false;
        };
        if self.pending.len() < total {
            return false;
        }
        let rest = self.pending.split_off(total);
        let chunk = std::mem::replace(&mut self.pending, rest);
        let body = &chunk[8..];
        let Some(head) = self.continuation.as_mut() else {
            return true;
        };
        if body.len() > head.remaining {
            tracing::warn!(
                got = body.len(),
                expected = head.remaining,
                "continuation longer than the response declared; dropping the response"
            );
            self.continuation = None;
            return true;
        }
        head.msg.extend_from_slice(body);
        head.remaining -= body.len();
        if head.remaining == 0
            && let Some(done) = self.continuation.take()
        {
            let mut msg = done.msg;
            // The emitted message is whole: make its prefix length agree.
            if let Ok(len) = u32::try_from(msg.len()) {
                msg[4..8].copy_from_slice(&len.to_le_bytes());
            }
            out.push(msg);
        }
        true
    }

    /// Drop a chunked response in progress (its command timed out).
    pub fn abort_continuation(&mut self) {
        self.continuation = None;
    }

    /// Whether a chunked response is being collected.
    #[must_use]
    pub fn in_continuation(&self) -> bool {
        self.continuation.is_some()
    }

    /// Bytes buffered but not yet forming a message.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

// ---------------------------------------------------------------------------
// Host->device commands on EP 0x05.
//
// A command message is six BE u32 words (marker 0x51, seq, 0, command, 0,
// payload length) followed by the payload. On the wire it is cut into pieces
// of at most `OUT_CHUNK_BODY_MAX` bytes, each written as its own bulk
// transfer behind the 8-byte prefix `00 00 00 00` + LE u32 piece length; the
// device answers once, after the last piece.
// ---------------------------------------------------------------------------

/// Largest command piece after the prefix: 4095-byte bulk writes, as the
/// Windows engine sends them.
pub const OUT_CHUNK_BODY_MAX: usize = 4087;

/// Command ids seen in the Windows captures.
pub mod cmd {
    /// Read the stream catalogue.
    pub const STREAM_CATALOGUE: u32 = 1200;
    /// Start a stream.
    pub const STREAM_START: u32 = 1220;
    /// Stop a stream.
    pub const STREAM_STOP: u32 = 1230;
    /// Device property strings.
    pub const PROPERTIES: u32 = 1330;
    /// Track box.
    pub const TRACK_BOX: u32 = 1400;
    /// Device identity strings: serial, model, generation, firmware.
    pub const DEVICE_STRINGS: u32 = 1420;
    /// Read the display area.
    pub const DISPLAY_AREA_GET: u32 = 1430;
    /// Write the display area.
    pub const DISPLAY_AREA_SET: u32 = 1440;
    /// Status strings: 5 the fault list, 6 the warning list, 7 the
    /// calibration id.
    pub const STATUS: u32 = 1490;
    /// Output rate pair.
    pub const OUTPUT_RATE: u32 = 1650;
    /// Mounting geometry.
    pub const MOUNTING: u32 = 2110;
    /// The hardware configuration (the Windows service's, for the DLL's
    /// `tobii_hardware_configuration_get`). The init replay sends it inside
    /// realm 0x2712; on Linux the ET5 answers with an empty payload.
    pub const HARDWARE_CONFIGURATION: u32 = 2120;
    /// Pause (`u32 1`) or resume (`u32 0`) the device: the DLL's
    /// `tracker_pause_device` and `tracker_resume_device` (0x180199450,
    /// 0x18019e910). Every init replay resumes; a pause was never captured.
    pub const DEVICE_PAUSE: u32 = 3100;
}

/// Notification ids seen in the Windows captures, unless noted.
pub mod notify {
    /// The display area changed: three corners.
    pub const DISPLAY_AREA: u32 = 1450;
    /// The device paused (`u32 1`) or resumed (`u32 0`). From the DLL's
    /// decoder (0x1801882b4), which rejects anything above 1; never captured.
    pub const DEVICE_PAUSED: u32 = 3110;
    /// Unknown; `u32 3` once at init, the value command 3170 also returns.
    pub const STATE_3180: u32 = 3180;
    /// A new calibration is active: `u32` calibration id.
    pub const CALIBRATION_ID: u32 = 3220;
}

/// Bytes from the marker through the payload-length word of a command.
const CMD_HEADER_LEN: usize = 4 * 6;

/// A command message without the wire prefix: header words, then `payload`
/// (which begins with `00 00`, see [`crate::tlv::TlvWriter`]).
#[must_use]
pub fn command_message(cmd: u32, seq: u32, payload: &[u8]) -> Vec<u8> {
    // A payload is at most a calibration blob (< 4 MiB).
    let payload_len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    let mut v = Vec::with_capacity(CMD_HEADER_LEN + payload.len());
    for word in [MARKER_COMMAND, seq, 0, cmd, 0, payload_len] {
        v.extend_from_slice(&word.to_be_bytes());
    }
    v.extend_from_slice(payload);
    v
}

/// A command as the bulk writes that carry it, each already prefixed.
#[must_use]
pub fn chunk_command(cmd: u32, seq: u32, payload: &[u8]) -> Vec<Vec<u8>> {
    command_message(cmd, seq, payload)
        .chunks(OUT_CHUNK_BODY_MAX)
        .map(|piece| {
            let mut w = Vec::with_capacity(8 + piece.len());
            w.extend_from_slice(&[0, 0, 0, 0]);
            // `piece.len() <= OUT_CHUNK_BODY_MAX`.
            w.extend_from_slice(&u32::try_from(piece.len()).unwrap_or(0).to_le_bytes());
            w.extend_from_slice(piece);
            w
        })
        .collect()
}

/// A command that fits in one bulk write.
fn single_packet(cmd: u32, seq: u32, payload: &[u8]) -> Vec<u8> {
    chunk_command(cmd, seq, payload).swap_remove(0)
}

/// Build the stream start/stop commands, mirroring the last lines of
/// `init_packets_ep.txt`: payload `00 00` + type 2/len 4/stream id
/// [+ type 0x17/len 4/0 for start].
fn stream_command_packet(cmd: u32, seq: u32, id: u32, with_flags: bool) -> Vec<u8> {
    let mut payload = vec![0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04];
    payload.extend_from_slice(&id.to_be_bytes());
    if with_flags {
        payload.extend_from_slice(&[0x17, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00]);
    }
    single_packet(cmd, seq, &payload)
}

/// Command 1220: start streaming `id` (e.g. `STREAM_ID_IMAGE`).
#[must_use]
pub fn stream_start_packet(seq: u32, id: u32) -> Vec<u8> {
    stream_command_packet(cmd::STREAM_START, seq, id, true)
}

/// Command 1230: stop streaming `id`.
#[must_use]
pub fn stream_stop_packet(seq: u32, id: u32) -> Vec<u8> {
    stream_command_packet(cmd::STREAM_STOP, seq, id, false)
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

    /// The 1110 calibration upload in the init replay is 162 bulk writes;
    /// re-chunking the reassembled command must give back every one of them.
    #[test]
    fn chunk_command_reproduces_the_captured_calibration_upload() {
        let packets = parse_init_packets(crate::INIT_PACKETS).expect("init file parses");
        let run: Vec<&[u8]> = packets[38..200].iter().map(|p| p.data.as_slice()).collect();
        let message: Vec<u8> = run.iter().flat_map(|p| p[8..].iter().copied()).collect();
        let prefixed = [&[0u8; 8][..], &message].concat();
        let head = parse_message(&prefixed).expect("header");
        assert_eq!(
            (head.marker, head.id, head.seq),
            (MARKER_COMMAND, 1110, 0x27)
        );

        let chunks = chunk_command(1110, 0x27, &message[24..]);

        assert_eq!(chunks.len(), 162);
        for (i, (got, want)) in chunks.iter().zip(&run).enumerate() {
            assert_eq!(got.as_slice(), *want, "chunk {i}");
        }
    }

    #[test]
    fn parse_message_splits_header_and_payload() {
        let rsp = crate::fixture!("init-rsp-1420");
        let m = parse_message(&rsp).expect("response");
        assert_eq!(
            (m.marker, m.seq, m.status, m.id, m.error),
            (MARKER_RESPONSE, 3, 1, 1420, ttp_error::NONE)
        );
        assert_eq!(
            m.payload.len(),
            usize::try_from(m.payload_len).expect("fits")
        );
        assert_eq!(m.tlvs().count(), 4);

        let n = crate::fixture!("init-notify-3180");
        let m = parse_message(&n).expect("notification");
        assert_eq!((m.marker, m.id), (MARKER_NOTIFICATION, notify::STATE_3180));
        assert_eq!(m.tlvs().next().and_then(|t| t.u32()), Some(3));
    }

    /// Word 5 is the TTP error code: 0 in every capture, and whatever a
    /// refusing device puts there otherwise.
    #[test]
    fn parse_message_reads_the_error_word() {
        let mut rsp = crate::fixture!("change-display-rsp-1440");
        let m = parse_message(&rsp).expect("response");
        assert_eq!((m.id, m.status, m.error), (1440, 1, ttp_error::NONE));

        rsp[24..28].copy_from_slice(&ttp_error::BAD_STATE.to_be_bytes());
        let m = parse_message(&rsp).expect("response");
        assert_eq!(
            (m.marker, m.seq, m.status, m.id, m.error, m.payload_len),
            (MARKER_RESPONSE, 0x2a, 1, 1440, ttp_error::BAD_STATE, 0)
        );
    }

    /// The device splits a calibration read: a 0x52 head declaring 659067
    /// payload bytes but carrying 11, then header-less continuations. The
    /// reassembler must emit one whole response and keep interleaved stream
    /// messages separate.
    #[test]
    fn reassembler_joins_a_chunked_response() {
        let head = crate::fixture!("calib-rsp-1100-head");
        let declared =
            usize::try_from(parse_message(&head).expect("head").payload_len).expect("fits");
        let carried = head.len() - HEADER_LEN;
        let blob: Vec<u8> = (0..declared - carried)
            .map(|i| u8::try_from(i * 7 % 251).expect("below 251"))
            .collect();
        let split = blob.len() - 564;
        let continuation = |body: &[u8]| {
            let mut c = vec![1, 0, 0, 0];
            c.extend_from_slice(&u32::try_from(body.len() + 8).expect("fits").to_le_bytes());
            c.extend_from_slice(body);
            c
        };
        let gaze = msg(STREAM_ID_GAZE, 40);
        let mut wire = head.clone();
        wire.extend(continuation(&blob[..split]));
        wire.extend(gaze.clone());
        wire.extend(continuation(&blob[split..]));

        for read_size in [wire.len(), 16384, 1] {
            let mut r = BulkReassembler::new();
            let mut got = Vec::new();
            for piece in wire.chunks(read_size) {
                got.extend(r.push(piece));
            }
            assert_eq!(got.len(), 2, "read size {read_size}");
            assert_eq!(got[0], gaze);
            let whole = parse_message(&got[1]).expect("response");
            assert_eq!((whole.id, whole.seq), (1100, 43));
            assert_eq!(whole.payload.len(), declared);
            assert_eq!(&whole.payload[carried..], blob.as_slice());
            assert_eq!(
                declared_len(&got[1]),
                Some(u32::try_from(got[1].len()).expect("fits"))
            );
            assert!(!r.in_continuation());
        }
    }

    #[test]
    fn aborting_a_chunked_response_resumes_normal_parsing() {
        let head = crate::fixture!("calib-rsp-1100-head");
        let mut r = BulkReassembler::new();
        assert!(r.push(&head).is_empty());
        assert!(r.in_continuation());
        r.abort_continuation();
        let gaze = msg(STREAM_ID_GAZE, 40);
        assert_eq!(r.push(&gaze), vec![gaze]);
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
