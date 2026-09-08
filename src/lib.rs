//! Tobii Eye Tracker 5 on Linux: USB decode, IR-camera head pose and a
//! Stream-Engine-like C ABI (see `ffi`). Built both as a binary CLI and as
//! `libtobii.so` (cdylib) for the FFI / OpenTrack integration.

pub mod analysis;
pub mod canonical;
pub mod cli;
pub mod daemon;
pub mod dashboard;
pub mod decode;
pub mod device;
pub mod engine;
pub mod ffi;
pub mod image83;
pub mod ipc;
pub mod math;
pub mod opentrack;
pub mod protocol;
pub mod sinks;
pub mod track;

use anyhow::Result;
use cli::{Command, Options};

/// Parse argv and dispatch the CLI subcommand.
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
        Command::Replay { .. } => device::run(&opts),
        Command::Camera { .. } => device::run_camera(&opts),
        Command::Track { .. } => device::run_track(&opts),
        Command::Probe { .. } => device::run_probe(&opts),
        Command::Image83 { .. } => device::run_image83(&opts),
        Command::Image83Replay { path, csv } => device::run_image83_replay(path, csv.as_deref()),
        Command::Head83 { path, occs } => analysis::head83(path, occs),
    }
}
