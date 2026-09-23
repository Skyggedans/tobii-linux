//! The calibration commands, as captured from the Windows engine calibrating
//! seven points (calibration.pcapng):
//!
//! ```text
//! 1100 read      -> the active calibration blob (a chunked response)
//! 1010 start     -> notification 3220 with id 0
//! 1060 clear
//! 1110 write     -> the stored blob back (seeds the session's sample ring)
//! 1030 collect   x, y, eye mask 3   (blocks ~0.6-0.9 s)
//! 1070 compute   -> notification 3220 with the new id (~2 s)
//! 1100 read      -> the new blob
//! 1020 stop
//! ```
//!
//! Commands without parameters (start, stop, clear, compute, read) carry no
//! payload at all, not even the `00 00` a TLV payload starts with.
//!
//! The blob itself is opaque apart from its header and trailing point list;
//! see the `tobii-calib` crate.

use crate::tlv::{TYPE_BYTES, TlvWriter, payload_tlvs};

/// Calibration command ids.
pub mod cmd {
    /// Begin a session.
    pub const START: u32 = 1010;
    /// End the session.
    pub const STOP: u32 = 1020;
    /// Collect one 2-D point.
    pub const COLLECT_2D: u32 = 1030;
    /// Clear the collected points.
    pub const CLEAR: u32 = 1060;
    /// Compute and apply.
    pub const COMPUTE: u32 = 1070;
    /// Read the active calibration.
    pub const READ: u32 = 1100;
    /// Write a calibration.
    pub const WRITE: u32 = 1110;
}

/// The eye mask every captured collect carries: both eyes.
pub const EYES_BOTH: u32 = 3;

/// A normalised coordinate as the device wants it: `x * 1024` computed in
/// `f32` (the fractional bits in the capture show f32 rounding), then widened
/// to 32.32 fixed point.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // reason: |x * 1024 * 2^32| < 2^63 for x in 0..=1
pub fn fixed_32_32(v: f32) -> i64 {
    (f64::from(v * 1024.0) * 4_294_967_296.0).round() as i64
}

/// Payload of command 1030 (collect a 2-D point).
#[must_use]
pub fn collect_payload(x: f32, y: f32, eye_mask: u32) -> Vec<u8> {
    TlvWriter::new()
        .fixed32_raw(fixed_32_32(x))
        .fixed32_raw(fixed_32_32(y))
        .u32(eye_mask)
        .finish()
}

/// Payload of command 1110 (write a calibration): the blob as one byte
/// string.
#[must_use]
pub fn write_payload(blob: &[u8]) -> Vec<u8> {
    TlvWriter::new().bytes(blob).finish()
}

/// The blob inside a 1110 command payload or a 1100 response payload.
#[must_use]
pub fn blob_from_payload(payload: &[u8]) -> Option<&[u8]> {
    let entry = payload_tlvs(payload).next()?;
    (entry.typ == TYPE_BYTES).then(|| entry.bytes())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{chunk_command, parse_init_packets, parse_message};

    /// The captured collects, in target order.
    #[test]
    fn collect_reproduces_the_captured_commands() {
        let cases: [(Vec<u8>, u32, f32, f32); 4] = [
            (crate::fixture!("calib-cmd-1030-seq48"), 48, 0.5, 0.5),
            (crate::fixture!("calib-cmd-1030-seq51"), 51, 0.5, 0.1),
            (crate::fixture!("calib-cmd-1030-seq52"), 52, 0.9, 0.9),
            (crate::fixture!("calib-cmd-1030-seq53"), 53, 0.1, 0.9),
        ];
        for (captured, seq, x, y) in cases {
            let built = chunk_command(cmd::COLLECT_2D, seq, &collect_payload(x, y, EYES_BOTH));
            assert_eq!(built, vec![captured], "({x}, {y})");
        }
    }

    #[test]
    fn empty_commands_reproduce_the_capture() {
        let cases: [(Vec<u8>, u32, u32); 5] = [
            (crate::fixture!("calib-cmd-1100-seq43"), cmd::READ, 43),
            (crate::fixture!("calib-cmd-1010-seq44"), cmd::START, 44),
            (crate::fixture!("calib-cmd-1060-seq45"), cmd::CLEAR, 45),
            (crate::fixture!("calib-cmd-1070-seq49"), cmd::COMPUTE, 49),
            (crate::fixture!("calib-cmd-1020-seq61"), cmd::STOP, 61),
        ];
        for (captured, command, seq) in cases {
            assert_eq!(
                chunk_command(command, seq, &[]),
                vec![captured],
                "{command}"
            );
        }
    }

    /// The blob the init replay uploads round-trips through the payload
    /// framing, and carries the calibration id the device reports.
    #[test]
    fn write_payload_round_trips_the_embedded_blob() {
        let packets = parse_init_packets(crate::INIT_PACKETS).expect("init file");
        let message: Vec<u8> = packets[38..200]
            .iter()
            .flat_map(|p| p.data[8..].iter().copied())
            .collect();
        let prefixed = [&[0u8; 8][..], &message].concat();
        let payload = parse_message(&prefixed).expect("header").payload;

        let blob = blob_from_payload(payload).expect("blob");

        assert_eq!(blob.len(), 659_056);
        assert_eq!(&blob[20..24], &1_904_654_973u32.to_le_bytes());
        assert_eq!(write_payload(blob), payload);
    }
}
