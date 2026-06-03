mod analysis;
mod canonical;
mod cli;
mod dashboard;
mod decode;
mod device;
mod math;
mod opentrack;
mod protocol;
mod sinks;
mod track;

use anyhow::Result;

use cli::{Command, Options};

fn main() -> Result<()> {
    let opts = Options::parse()?;

    match &opts.command {
        Command::AnalyzeLog { path } => analysis::analyze_log(path),
        Command::CompareLogs { inputs } => analysis::compare_logs(inputs),
        Command::DecodeStream { path } => analysis::decode_stream(path),
        Command::ImportTsv { tsv_path, log_path } => analysis::import_tsv(tsv_path, log_path),
        Command::PoseCandidates { inputs } => analysis::pose_candidates(inputs),
        Command::CompareDecoded { inputs } => analysis::compare_decoded(inputs),
        Command::ExtractCalibration { path, json_path } => {
            analysis::extract_calibration(path, json_path.as_deref())
        }
        Command::Replay { .. } => device::run(&opts),
        Command::Camera { .. } => device::run_camera(&opts),
        Command::Track { .. } => device::run_track(&opts),
    }
}
