//! Tobii Eye Tracker 5 on Linux: USB transport and the live 0x83 engine, the
//! `tobiid` daemon, the IR-camera head-pose tracker, and the research and
//! diagnostic subcommands behind the `tobii5-init-replay` CLI.
//!
//! The Stream-Engine-like C ABI (`libtobii.so`) lives in the `tobii-ffi`
//! crate, and the client-facing IPC in `tobii-ipc`.
//!
//! The library only emits diagnostics through the `tracing` facade; the
//! executables install a subscriber via [`logging::init`].

/// Offline analysis of recorded stream logs (statistics, comparisons, head
/// axes, calibration extraction).
pub mod analysis;
/// `MediaPipe` canonical face mesh used by the head-pose fit.
pub mod canonical;
/// Command-line parsing for the `tobii5-init-replay` CLI.
pub mod cli;
pub mod daemon;
/// Terminal dashboard rendering live tracking frames.
pub mod dashboard;
/// Decoding of the 0x83 gaze / tracking stream into frames.
pub mod decode;
/// Research and diagnostic subcommands (replay, UVC camera, image83 tools).
pub mod devcmd;
/// USB transport and the live 0x83 gaze + image engine.
pub mod device;
pub mod engine;
pub mod image83;
pub use tobii_ipc as ipc;
pub mod log;
pub use tobii_log as logging;
/// Small numeric helpers (origin calibration, filtering, geometry).
pub mod math;
/// `OpenTrack` UDP sink and head/gaze-to-6DOF mapping.
pub mod opentrack;
/// Init-packet capture parsing and the device control protocol.
pub mod protocol;
/// Output sinks for decoded frames (CSV, JSONL) and the live fan-out.
pub mod sinks;
pub mod time;
pub mod track;

use anyhow::Result;
use cli::{Command, Options};

/// Parse argv and dispatch the CLI subcommand.
///
/// # Errors
/// Argument parsing failures (see [`cli`]) and any error the selected
/// subcommand returns (device access, I/O, malformed logs).
pub fn run() -> Result<()> {
    let opts = Options::parse()?;
    match &opts.command {
        Command::AnalyzeLog { path } => analysis::analyze_log(path),
        Command::CompareLogs { inputs } => analysis::compare_logs(inputs),
        Command::DecodeStream { path } => analysis::decode_stream(path),
        Command::ImportTsv { tsv_path, log_path } => analysis::import_tsv(tsv_path, log_path),
        Command::PoseCandidates { inputs } => analysis::pose_candidates(inputs),
        Command::CompareDecoded { inputs } => analysis::compare_decoded(inputs),
        Command::HeadAxes { inputs } => analysis::head_axes(inputs),
        Command::ExtractCalibration { path, json_path } => {
            analysis::extract_calibration(path, json_path.as_deref())
        }
        Command::Replay { .. } => devcmd::run(&opts),
        Command::Camera { .. } => devcmd::run_camera(&opts),
        Command::Track { .. } => devcmd::run_track(&opts),
        Command::Probe { .. } => devcmd::run_probe(&opts),
        Command::Image83 { .. } => devcmd::run_image83(&opts),
        Command::Image83Replay { path, csv } => devcmd::run_image83_replay(path, csv.as_deref()),
        Command::Head83 { path, occs } => analysis::head83(path, occs),
    }
}
