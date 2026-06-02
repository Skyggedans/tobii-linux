use anyhow::Result;
use std::io::{self, Write};

use crate::decode::TrackingFrame;

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

pub(crate) fn render_tracking_dashboard(
    frame: &TrackingFrame,
    opentrack_pose: Option<[f64; 6]>,
) -> Result<()> {
    print!("\x1b[2J\x1b[H");
    println!("Tobii Eye Tracker 5 - TrackingFrame");
    println!("==============================================================");
    println!("packet        {:>18}", frame.packet);
    println!("timestamp us  {:>18}", frame.ts_us);
    println!(
        "gaze valid    {:>18}",
        if frame.gaze_valid { "yes" } else { "no" }
    );
    println!();

    println!("Gaze");
    dashboard_pair("  px", frame.gaze_x, frame.gaze_y);
    dashboard_pair("  norm", frame.gaze_norm_x, frame.gaze_norm_y);
    dashboard_bar("  x", frame.gaze_norm_x);
    dashboard_bar("  y", frame.gaze_norm_y);
    println!();

    println!("Left Eye");
    dashboard_pair("  px", frame.left_eye_x, frame.left_eye_y);
    dashboard_pair("  norm", frame.left_eye_norm_x, frame.left_eye_norm_y);
    dashboard_bar("  x", frame.left_eye_norm_x);
    dashboard_bar("  y", frame.left_eye_norm_y);
    println!();

    println!("Right Eye");
    dashboard_pair("  px", frame.right_eye_x, frame.right_eye_y);
    dashboard_pair("  norm", frame.right_eye_norm_x, frame.right_eye_norm_y);
    dashboard_bar("  x", frame.right_eye_norm_x);
    dashboard_bar("  y", frame.right_eye_norm_y);
    println!();

    println!("Head");
    println!(
        "  x/y/z       {:>14} {:>14} {:>14}",
        fmt_frame_value(frame.head_x),
        fmt_frame_value(frame.head_y),
        fmt_frame_value(frame.head_z)
    );
    println!(
        "  yaw/p/r raw {:>14} {:>14} {:>14}",
        fmt_frame_value(frame.head_yaw),
        fmt_frame_value(frame.head_pitch),
        fmt_frame_value(frame.head_roll)
    );
    if let Some(pose) = opentrack_pose {
        println!();
        println!("OpenTrack UDP");
        println!(
            "  tx/ty/tz cm {:>14} {:>14} {:>14}",
            fmt_plain_value(pose[0]),
            fmt_plain_value(pose[1]),
            fmt_plain_value(pose[2])
        );
        println!(
            "  yaw/p/r deg{:>14} {:>14} {:>14}",
            fmt_plain_value(pose[3]),
            fmt_plain_value(pose[4]),
            fmt_plain_value(pose[5])
        );
    }
    println!();
    println!("Press Ctrl+C to stop.");

    io::stdout().flush()?;
    Ok(())
}

pub(crate) fn render_dashboard_status(status: &str) -> Result<()> {
    print!("\x1b[2J\x1b[H");
    println!("Tobii Eye Tracker 5 - TrackingFrame");
    println!("==============================================================");
    println!("{status}");
    io::stdout().flush()?;
    Ok(())
}

pub(crate) fn dashboard_pair(label: &str, x: Option<f64>, y: Option<f64>) {
    println!(
        "{label:<8} x={:>12} y={:>12}",
        fmt_frame_value(x),
        fmt_frame_value(y)
    );
}

pub(crate) fn dashboard_bar(label: &str, value: Option<f64>) {
    const WIDTH: usize = 32;

    let Some(value) = value else {
        println!("{label:<8} [{}] {}", " ".repeat(WIDTH), "n/a");
        return;
    };

    let value = value.clamp(0.0, 1.0);
    let filled = (value * WIDTH as f64).round() as usize;
    let empty = WIDTH.saturating_sub(filled);
    println!(
        "{label:<8} [{}{}] {:>7.3}",
        "#".repeat(filled),
        ".".repeat(empty),
        value
    );
}

pub(crate) fn fmt_frame_value(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

pub(crate) fn fmt_plain_value(value: f64) -> String {
    format!("{value:.3}")
}

pub(crate) fn fmt_live(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_string())
}
