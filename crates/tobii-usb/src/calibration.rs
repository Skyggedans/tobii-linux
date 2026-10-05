//! The init replay, adapted to this user: their saved calibration instead of
//! the one embedded in the capture, and their display area instead of the
//! capture author's monitor.
//!
//! Both are single commands inside the replay (1110 write-calibration, split
//! over 162 bulk writes, and 1440 set-display-area); they are rebuilt with the
//! same sequence numbers, so the rest of the replay and the command numbering
//! after it are unchanged.

use std::ops::Range;
use std::path::PathBuf;

use anyhow::{Context, Result};
use tobii_calib::store::{self, Location};
use tobii_ipc::geometry::DisplayArea;
use tobii_proto::calibration::{self as calib_cmd, blob_from_payload, write_payload};
use tobii_proto::facts::{DEFAULT_DISPLAY_ID, display_area_set_payload, parse_display_area};
use tobii_proto::protocol::{
    InitPacket, MARKER_COMMAND, chunk_command, cmd, parse_init_packets, parse_message,
};
use tracing::{info, warn};

use crate::device::EP_OUT;

/// The Windows engine's init sequence, captured once and replayed verbatim.
const EMBEDDED_INIT: &str = include_str!("../init_packets_ep.txt");

/// Where the calibration in use came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Built into the init replay (the capture author's).
    Embedded,
    /// The user's saved calibration.
    File(PathBuf),
}

/// The embedded init replay.
///
/// # Errors
///
/// Only if the embedded text is corrupt (a build problem).
pub fn embedded_packets() -> Result<Vec<InitPacket>> {
    parse_init_packets(EMBEDDED_INIT)
}

/// A command packet's id and seq, if `data` starts a command. A continuation
/// write carries payload right after its prefix, so a header is recognised by
/// the zero prefix tag, the 0x51 marker and the zero status word together.
fn command_of(data: &[u8]) -> Option<(u32, u32)> {
    let header = parse_message(data)?;
    (data.get(..4) == Some(&[0; 4]) && header.marker == MARKER_COMMAND && header.status == 0)
        .then_some((header.id, header.seq))
}

/// The packets carrying command `id`: its first write and the continuation
/// writes up to the next command.
fn command_run(packets: &[InitPacket], id: u32) -> Option<(Range<usize>, u32)> {
    let start = packets
        .iter()
        .position(|p| command_of(&p.data).is_some_and(|(c, _)| c == id))?;
    let (_, seq) = command_of(&packets[start].data)?;
    let end = packets[start + 1..]
        .iter()
        .position(|p| command_of(&p.data).is_some())
        .map_or(packets.len(), |n| start + 1 + n);
    Some((start..end, seq))
}

/// The whole message of a command written as `pieces`: each piece's body
/// after its 8-byte prefix, joined behind a zero prefix of that size, which
/// is where [`parse_message`] reads the header from.
fn command_message(pieces: &[InitPacket]) -> Vec<u8> {
    let mut message = vec![0u8; 8];
    for piece in pieces {
        message.extend_from_slice(piece.data.get(8..).unwrap_or_default());
    }
    message
}

/// The calibration blob the embedded replay uploads.
///
/// # Errors
///
/// Only if the embedded replay is corrupt.
pub fn embedded_blob() -> Result<Vec<u8>> {
    let packets = embedded_packets()?;
    let (run, _) = command_run(&packets, calib_cmd::cmd::WRITE)
        .context("no calibration upload in the init replay")?;
    let prefixed = command_message(&packets[run]);
    let message = parse_message(&prefixed).context("calibration upload header")?;
    let blob = blob_from_payload(message.payload).context("calibration upload payload")?;
    Ok(blob.to_vec())
}

/// The display area `packets` write (their 1440), as the device reads it:
/// the seq the device answers the write with, the corners as the wire
/// carries them (32.32 fixed point of 1/1024 mm, so an area set in `f64`
/// comes back rounded) and the display id. `None` when `packets` write no
/// display area, or one that does not parse.
#[must_use]
pub fn written_display_area(packets: &[InitPacket]) -> Option<(u32, DisplayArea, Option<u32>)> {
    let (run, seq) = command_run(packets, cmd::DISPLAY_AREA_SET)?;
    let message = command_message(&packets[run]);
    let (area, display_id) = parse_display_area(&parse_message(&message)?)?;
    Some((seq, area, display_id))
}

/// Replace the calibration upload in `packets` with `blob`, keeping its seq.
///
/// # Errors
///
/// Fails when `packets` has no calibration upload.
pub fn substitute_calibration(packets: &[InitPacket], blob: &[u8]) -> Result<Vec<InitPacket>> {
    let (run, seq) = command_run(packets, calib_cmd::cmd::WRITE)
        .context("no calibration upload in the init replay")?;
    Ok(splice(
        packets,
        run,
        chunk_command(calib_cmd::cmd::WRITE, seq, &write_payload(blob)),
    ))
}

/// Replace the display-area write in `packets` with `area`, keeping its seq.
///
/// # Errors
///
/// Fails when `packets` has no display-area write.
pub fn substitute_display_area(
    packets: &[InitPacket],
    area: &DisplayArea,
) -> Result<Vec<InitPacket>> {
    let (run, seq) = command_run(packets, cmd::DISPLAY_AREA_SET)
        .context("no display area in the init replay")?;
    let payload = display_area_set_payload(area, DEFAULT_DISPLAY_ID);
    Ok(splice(
        packets,
        run,
        chunk_command(cmd::DISPLAY_AREA_SET, seq, &payload),
    ))
}

fn splice(packets: &[InitPacket], run: Range<usize>, writes: Vec<Vec<u8>>) -> Vec<InitPacket> {
    let mut out = Vec::with_capacity(packets.len() - run.len() + writes.len());
    out.extend_from_slice(&packets[..run.start]);
    out.extend(
        writes
            .into_iter()
            .map(|data| InitPacket { ep: EP_OUT, data }),
    );
    out.extend_from_slice(&packets[run.end..]);
    out
}

/// The calibration the engine should upload: the user's saved file when there
/// is a valid one, else the embedded one. A file that fails validation is
/// reported and skipped, never uploaded.
///
/// # Errors
///
/// Only if the embedded replay is corrupt.
pub fn current_blob() -> Result<(Vec<u8>, Source)> {
    if let Location::File(path) = store::configured() {
        match store::load(&path) {
            Ok(Some((blob, _))) => return Ok((blob, Source::File(path))),
            Ok(None) => {}
            Err(e) => warn!(path = %path.display(), error = %e, "ignoring the saved calibration"),
        }
    }
    Ok((embedded_blob()?, Source::Embedded))
}

/// The init replay to send: the user's calibration and, when given, display
/// area substituted into the embedded one.
///
/// # Errors
///
/// Only if the embedded replay is corrupt.
pub fn init_packets(display_area: Option<&DisplayArea>) -> Result<Vec<InitPacket>> {
    let mut packets = embedded_packets()?;
    let (blob, source) = current_blob()?;
    if let Source::File(path) = &source {
        packets = substitute_calibration(&packets, &blob)?;
        info!(
            path = %path.display(),
            id = tobii_calib::blob::calibration_id(&blob),
            "calibration: using the saved calibration"
        );
    }
    if let Some(area) = display_area {
        packets = substitute_display_area(&packets, area)?;
    }
    Ok(packets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::next_command_seq;

    #[test]
    fn substituting_the_embedded_blob_changes_nothing() {
        let packets = embedded_packets().expect("embedded");
        let blob = embedded_blob().expect("blob");
        assert_eq!(
            tobii_calib::blob::calibration_id(&blob),
            Some(1_904_654_973)
        );

        let substituted = substitute_calibration(&packets, &blob).expect("substitute");

        assert_eq!(substituted, packets);
    }

    #[test]
    fn a_small_blob_becomes_a_single_write_with_the_same_seq() {
        let packets = embedded_packets().expect("embedded");
        let blob = vec![0x5a; 1000];

        let substituted = substitute_calibration(&packets, &blob).expect("substitute");

        assert_eq!(substituted.len(), packets.len() - 161);
        let (run, seq) = command_run(&substituted, calib_cmd::cmd::WRITE).expect("run");
        assert_eq!((run.len(), seq), (1, 0x27));
        assert_eq!(next_command_seq(&substituted), next_command_seq(&packets));
    }

    /// A display area a user might set: a 597 x 336 mm monitor.
    fn user_area() -> DisplayArea {
        tobii_ipc::geometry::display_area_basic(
            597.0,
            336.0,
            1.0,
            &tobii_ipc::geometry::GeometryMounting {
                guides: 2,
                width_mm: 184.0,
                angle_deg: 20.0,
                external_offset_mm: [0.0, -0.16, 13.85],
                internal_offset_mm: [0.0, 5.38, 9.86],
            },
        )
    }

    #[test]
    fn display_area_is_written_in_place() {
        let packets = embedded_packets().expect("embedded");
        let area = user_area();

        let substituted = substitute_display_area(&packets, &area).expect("substitute");

        assert_eq!(substituted.len(), packets.len());
        let (run, seq) = command_run(&substituted, cmd::DISPLAY_AREA_SET).expect("run");
        assert_eq!(seq, 0x0e);
        let msg = parse_message(&substituted[run.start].data).expect("msg");
        let (written, id) = tobii_proto::facts::parse_display_area(&msg).expect("area");
        assert_eq!(id, Some(DEFAULT_DISPLAY_ID));
        assert!((written.top_right_mm[0] - written.top_left_mm[0] - 597.0).abs() < 1e-6);
    }

    #[test]
    fn the_area_written_reads_back_as_the_wire_carries_it() {
        let packets = embedded_packets().expect("embedded");
        let (seq, embedded, id) = written_display_area(&packets).expect("the replay's 1440");
        assert_eq!((seq, id), (0x0e, Some(DEFAULT_DISPLAY_ID)));
        assert_eq!(written_display_area(&packets[..13]), None, "no 1440");

        let area = user_area();
        let substituted = substitute_display_area(&packets, &area).expect("substitute");
        let (seq, written, id) = written_display_area(&substituted).expect("the user's 1440");

        assert_eq!((seq, id), (0x0e, Some(DEFAULT_DISPLAY_ID)));
        assert_ne!(written, embedded);
        // 32.32 fixed point of 1/1024 mm: rounded to 2^-42 mm.
        let half_step = 0.5 / 1024.0 / 4_294_967_296.0;
        let corners = |a: DisplayArea| [a.top_left_mm, a.top_right_mm, a.bottom_left_mm];
        for (w, a) in corners(written)
            .iter()
            .flatten()
            .zip(corners(area).iter().flatten())
        {
            assert!((w - a).abs() <= half_step, "{w} for {a}");
        }
    }
}
