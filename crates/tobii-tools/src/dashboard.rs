//! Terminal output for the replay mode: the one-line decoded dump
//! and the full-screen tracking dashboard. Everything here is program output
//! that the user reads, so it goes to stdout via `print!`/`write!`.

use anyhow::Result;
use std::io::{self, BufWriter, Write};

use tobii_proto::decode::TrackingFrame;

/// Print one compact line with the decoded values of `frame`.
pub(crate) fn print_live_decoded(frame: &TrackingFrame) {
    println!(
        "  decoded #{} valid={} gaze=({},{}) norm=({},{}) left_eye=({},{}) right_eye=({},{}) head=({},{},{}) angles=({},{},{})",
        frame.packet,
        if frame.gaze_valid { "yes" } else { "no" },
        fmt_live(frame.gaze_x),
        fmt_live(frame.gaze_y),
        fmt_live(frame.gaze_norm_x),
        fmt_live(frame.gaze_norm_y),
        fmt_live(frame.left_eye_x),
        fmt_live(frame.left_eye_y),
        fmt_live(frame.right_eye_x),
        fmt_live(frame.right_eye_y),
        fmt_live(frame.head_x),
        fmt_live(frame.head_y),
        fmt_live(frame.head_z),
        fmt_live(frame.head_yaw),
        fmt_live(frame.head_pitch),
        fmt_live(frame.head_roll),
    );
}

/// Clear the terminal and draw the full dashboard for `frame`; the
/// `OpenTrack` block is shown when a pose was sent this frame.
///
/// # Errors
/// Fails when writing to stdout fails.
pub(crate) fn render_tracking_dashboard(
    frame: &TrackingFrame,
    opentrack_pose: Option<[f64; 6]>,
) -> Result<()> {
    // One buffered write per frame instead of ~30 line-buffered flushes.
    let mut out = BufWriter::new(io::stdout().lock());
    write_tracking_dashboard(&mut out, frame, opentrack_pose)?;
    out.flush()?;
    Ok(())
}

/// The dashboard of [`render_tracking_dashboard`], screen clear included.
fn write_tracking_dashboard<W: Write>(
    out: &mut W,
    frame: &TrackingFrame,
    opentrack_pose: Option<[f64; 6]>,
) -> io::Result<()> {
    write!(out, "\x1b[2J\x1b[H")?;
    writeln!(out, "Tobii Eye Tracker 5 - TrackingFrame")?;
    writeln!(
        out,
        "=============================================================="
    )?;
    writeln!(out, "packet        {:>18}", frame.packet)?;
    writeln!(out, "timestamp us  {:>18}", frame.ts_us)?;
    writeln!(
        out,
        "gaze valid    {:>18}",
        if frame.gaze_valid { "yes" } else { "no" }
    )?;
    writeln!(out)?;

    writeln!(out, "Gaze")?;
    dashboard_pair(out, "  px", frame.gaze_x, frame.gaze_y)?;
    dashboard_pair(out, "  norm", frame.gaze_norm_x, frame.gaze_norm_y)?;
    dashboard_bar(out, "  x", frame.gaze_norm_x)?;
    dashboard_bar(out, "  y", frame.gaze_norm_y)?;
    writeln!(out)?;

    writeln!(out, "Left Eye")?;
    dashboard_pair(out, "  px", frame.left_eye_x, frame.left_eye_y)?;
    dashboard_pair(out, "  norm", frame.left_eye_norm_x, frame.left_eye_norm_y)?;
    dashboard_bar(out, "  x", frame.left_eye_norm_x)?;
    dashboard_bar(out, "  y", frame.left_eye_norm_y)?;
    writeln!(out)?;

    writeln!(out, "Right Eye")?;
    dashboard_pair(out, "  px", frame.right_eye_x, frame.right_eye_y)?;
    dashboard_pair(
        out,
        "  norm",
        frame.right_eye_norm_x,
        frame.right_eye_norm_y,
    )?;
    dashboard_bar(out, "  x", frame.right_eye_norm_x)?;
    dashboard_bar(out, "  y", frame.right_eye_norm_y)?;
    writeln!(out)?;

    writeln!(out, "Head")?;
    writeln!(
        out,
        "  x/y/z       {:>14} {:>14} {:>14}",
        fmt_frame_value(frame.head_x),
        fmt_frame_value(frame.head_y),
        fmt_frame_value(frame.head_z)
    )?;
    writeln!(
        out,
        "  yaw/p/r raw {:>14} {:>14} {:>14}",
        fmt_frame_value(frame.head_yaw),
        fmt_frame_value(frame.head_pitch),
        fmt_frame_value(frame.head_roll)
    )?;
    writeln!(
        out,
        "  pupil L/R mm{:>14} {:>14}",
        fmt_frame_value(frame.pupil_diameter_left),
        fmt_frame_value(frame.pupil_diameter_right)
    )?;
    writeln!(
        out,
        "  sec range L/R{:>13} {:>14}",
        fmt_frame_value(frame.secondary_range_left),
        fmt_frame_value(frame.secondary_range_right)
    )?;
    if let Some(pose) = opentrack_pose {
        writeln!(out)?;
        writeln!(out, "OpenTrack UDP")?;
        writeln!(
            out,
            "  tx/ty/tz cm {:>14} {:>14} {:>14}",
            fmt_plain_value(pose[0]),
            fmt_plain_value(pose[1]),
            fmt_plain_value(pose[2])
        )?;
        writeln!(
            out,
            "  yaw/p/r deg{:>14} {:>14} {:>14}",
            fmt_plain_value(pose[3]),
            fmt_plain_value(pose[4]),
            fmt_plain_value(pose[5])
        )?;
    }
    writeln!(out)?;
    writeln!(out, "Press Ctrl+C to stop.")
}

/// Clear the terminal and show the dashboard header with a single status
/// line (used while waiting for, or after losing, the stream).
///
/// # Errors
/// Fails when writing to stdout fails.
pub(crate) fn render_dashboard_status(status: &str) -> Result<()> {
    let mut out = io::stdout().lock();
    write!(out, "\x1b[2J\x1b[H")?;
    writeln!(out, "Tobii Eye Tracker 5 - TrackingFrame")?;
    writeln!(
        out,
        "=============================================================="
    )?;
    writeln!(out, "{status}")?;
    out.flush()?;
    Ok(())
}

/// One `label x=... y=...` dashboard line.
fn dashboard_pair<W: Write>(
    out: &mut W,
    label: &str,
    x: Option<f64>,
    y: Option<f64>,
) -> io::Result<()> {
    writeln!(
        out,
        "{label:<8} x={:>12} y={:>12}",
        fmt_frame_value(x),
        fmt_frame_value(y)
    )
}

/// One `label [####....] value` bar for a 0..1 normalised value.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // reason: value is clamped to 0..=1 so the product is within 0..=WIDTH
fn dashboard_bar<W: Write>(out: &mut W, label: &str, value: Option<f64>) -> io::Result<()> {
    const WIDTH: usize = 32;

    let Some(value) = value else {
        return writeln!(out, "{label:<8} [{}] n/a", " ".repeat(WIDTH));
    };

    let value = value.clamp(0.0, 1.0);
    // cast: range-checked above, rounds to an integer in 0..=WIDTH.
    let filled = (value * WIDTH as f64).round() as usize;
    let empty = WIDTH.saturating_sub(filled);
    writeln!(
        out,
        "{label:<8} [{}{}] {:>7.3}",
        "#".repeat(filled),
        ".".repeat(empty),
        value
    )
}

/// Three-decimal rendering of an optional frame value, `n/a` when absent.
#[must_use]
pub(crate) fn fmt_frame_value(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
}

/// Three-decimal rendering of a value that is always present.
#[must_use]
pub(crate) fn fmt_plain_value(value: f64) -> String {
    format!("{value:.3}")
}

/// Two-decimal rendering for the compact decoded line, `-` when absent.
#[must_use]
pub(crate) fn fmt_live(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.2}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_proto::decode::decode_stream_payload;
    use tobii_proto::gaze83::decode_gaze_frame;
    use tobii_proto::protocol::{hex_to_bytes, parse_message};

    /// The pupil row shows keys `0x06`/`0x0c`, and keys `0x25`/`0x27`'s
    /// third component, which that row used to show, has a row of its own.
    #[test]
    fn the_pupil_row_shows_the_pupil_diameters() {
        let bytes = hex_to_bytes(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tobii-proto/fixtures/session1-gaze-frame.hex"
        )))
        .expect("fixture is valid hex");
        let values = decode_stream_payload(&bytes).expect("decodes");
        let gaze = parse_message(&bytes).as_ref().and_then(decode_gaze_frame);
        let frame = TrackingFrame::from_decoded(1, &values, gaze.as_ref());
        let mut out = Vec::new();

        write_tracking_dashboard(&mut out, &frame, None).expect("writes");

        let screen = String::from_utf8(out).expect("utf-8");
        let row = |label: &str| {
            screen
                .lines()
                .find(|line| line.starts_with(label))
                .map(str::split_whitespace)
                .map(|words| words.rev().take(2).collect::<Vec<_>>())
        };
        assert_eq!(row("  pupil L/R mm"), Some(vec!["5.997", "6.247"]));
        assert_eq!(row("  sec range L/R"), Some(vec!["430.605", "435.728"]));
    }
}
