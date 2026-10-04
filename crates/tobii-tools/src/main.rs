//! Research and diagnostic CLI for the Tobii Eye Tracker 5: offline analysis
//! of recorded logs, the UVC IR camera path and the live 0x50e image tools.
//!
//! Nothing here is needed to run the driver — that is `tobiid`, `libtobii.so`
//! and the thin clients. This binary is built but not installed.

mod analysis;
mod cli;
mod compare_dll;
mod compare_head;
mod dashboard;
mod devcmd;
mod gates;
mod ipc_probe;
mod math;
mod opentrack;
mod sinks;

use anyhow::Result;
use cli::{Command, Options};

/// Parse argv and dispatch the selected subcommand.
fn run() -> Result<()> {
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
        Command::CompareDll {
            log_path,
            jsonl_path,
            head: None,
        } => compare_dll::compare_dll(log_path, jsonl_path),
        Command::CompareDll {
            log_path,
            jsonl_path,
            head: Some(options),
        } => compare_head::compare_head(log_path, jsonl_path, options),
        Command::IpcProbe {
            streams,
            secs,
            set_display,
        } => ipc_probe::run(*streams, *secs, *set_display),
    }
}

fn main() {
    tobii_log::init();
    if let Err(e) = run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
