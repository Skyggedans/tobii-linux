use anyhow::{Context, Result};
use std::env;

use crate::opentrack::{
    AngleComponent, AngleSource, CouplingMode, DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
    DEFAULT_OPENTRACK_ANGLE_MAP, DEFAULT_OPENTRACK_ANGLE_OCC,
    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP, DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM, DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE,
    DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE, DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
    DEFAULT_OPENTRACK_ROTATION_COMP, DEFAULT_OPENTRACK_SMOOTHING,
    DEFAULT_OPENTRACK_TRANSLATION_SCALE,
};

pub(crate) const DEFAULT_LOG_PATH: &str = "tobii_stream.bin";

pub(crate) const DEFAULT_OPENTRACK_HOST: &str = "127.0.0.1";

pub(crate) const DEFAULT_OPENTRACK_PORT: u16 = 4242;

pub(crate) enum Command {
    Replay {
        init_path: String,
    },
    Camera {
        init_path: String,
        out_prefix: String,
        max_frames: usize,
        skip_replay: bool,
        fifo: Option<String>,
        frame_type: Option<u8>,
        interval: Option<u32>,
    },
    Track {
        init_path: String,
        skip_replay: bool,
        host: String,
        port: u16,
    },
    AnalyzeLog {
        path: String,
    },
    CompareLogs {
        inputs: Vec<LogInput>,
    },
    DecodeStream {
        path: String,
    },
    ImportTsv {
        tsv_path: String,
        log_path: String,
    },
    PoseCandidates {
        inputs: Vec<LogInput>,
    },
    CompareDecoded {
        inputs: Vec<LogInput>,
    },
    ExtractCalibration {
        path: String,
        json_path: Option<String>,
    },
}

pub(crate) struct LogInput {
    pub(crate) label: String,
    pub(crate) path: String,
}

pub(crate) struct Options {
    pub(crate) command: Command,
    pub(crate) log_path: Option<String>,
    pub(crate) max_init_packets: Option<usize>,
    pub(crate) max_stream_packets: Option<u64>,
    pub(crate) reconnect: bool,
    pub(crate) decoded_csv_path: Option<String>,
    pub(crate) jsonl_path: Option<String>,
    pub(crate) print_decoded: bool,
    pub(crate) dashboard: bool,
    pub(crate) opentrack_host: Option<String>,
    pub(crate) opentrack_port: Option<u16>,
    pub(crate) opentrack_translation: bool,
    pub(crate) opentrack_angle_source: AngleSource,
    pub(crate) opentrack_angle_occ: Option<usize>,
    pub(crate) opentrack_angle_scale: [f64; 3],
    pub(crate) opentrack_angle_map: [AngleComponent; 3],
    pub(crate) opentrack_origin_samples: usize,
    pub(crate) opentrack_angle_points: Option<(usize, usize)>,
    pub(crate) opentrack_roll_points: Option<(usize, usize)>,
    pub(crate) opentrack_smoothing: f64,
    pub(crate) opentrack_angle_deadzone: f64,
    pub(crate) opentrack_rotation_comp: [[f64; 3]; 3],
    pub(crate) opentrack_translation_scale: [f64; 3],
    pub(crate) opentrack_angle_translation_comp: [[f64; 3]; 3],
    pub(crate) opentrack_angle_translation_comp_scale: f64,
    pub(crate) opentrack_angle_translation_deadzone: f64,
    pub(crate) opentrack_coupling_mode: CouplingMode,
    pub(crate) opentrack_auto_decouple: bool,
}

impl Options {
    /// Build an `Options` carrying `command` with every other field at its
    /// default. Used by non-streaming subcommands that ignore the OpenTrack
    /// knobs.
    fn for_command(command: Command) -> Self {
        Self {
            command,
            log_path: None,
            max_init_packets: None,
            max_stream_packets: None,
            reconnect: false,
            decoded_csv_path: None,
            jsonl_path: None,
            print_decoded: false,
            dashboard: false,
            opentrack_host: None,
            opentrack_port: None,
            opentrack_translation: true,
            opentrack_angle_source: AngleSource::Head,
            opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
            opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
            opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
            opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
            opentrack_angle_points: None,
            opentrack_roll_points: None,
            opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
            opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
            opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
            opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
            opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
            opentrack_angle_translation_comp_scale: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
            opentrack_angle_translation_deadzone: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
            opentrack_coupling_mode: CouplingMode::Rotation,
            opentrack_auto_decouple: false,
        }
    }

    pub(crate) fn parse() -> Result<Self> {
        let mut args = env::args().skip(1).peekable();
        let mut init_path = None;
        let mut log_path = Some(DEFAULT_LOG_PATH.to_string());
        let mut max_init_packets = None;
        let mut max_stream_packets = None;
        let mut reconnect = true;
        let mut decoded_csv_path = None;
        let mut jsonl_path = None;
        let mut print_decoded = false;
        let mut dashboard = false;
        let mut opentrack_host = None;
        let mut opentrack_port = None;
        let mut opentrack_translation = true;
        let mut opentrack_angle_source = AngleSource::Head;
        let mut opentrack_angle_occ = Some(DEFAULT_OPENTRACK_ANGLE_OCC);
        let mut opentrack_angle_scale = DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE;
        let mut opentrack_angle_scale_set = false;
        let mut opentrack_angle_map = DEFAULT_OPENTRACK_ANGLE_MAP;
        let mut opentrack_origin_samples = DEFAULT_OPENTRACK_ORIGIN_SAMPLES;
        let mut opentrack_angle_points = None;
        let mut opentrack_roll_points = None;
        let mut opentrack_smoothing = DEFAULT_OPENTRACK_SMOOTHING;
        let mut opentrack_angle_deadzone = DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG;
        let mut opentrack_rotation_comp = DEFAULT_OPENTRACK_ROTATION_COMP;
        let mut opentrack_translation_scale = DEFAULT_OPENTRACK_TRANSLATION_SCALE;
        let mut opentrack_angle_translation_comp = DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP;
        let mut opentrack_angle_translation_comp_scale =
            DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE;
        let mut opentrack_angle_translation_deadzone =
            DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM;
        let mut opentrack_coupling_mode = CouplingMode::Rotation;
        let mut opentrack_auto_decouple = false;

        if args.peek().map(String::as_str) == Some("track") {
            args.next();
            let mut track_init_path = "init_packets_ep.txt".to_string();
            let mut skip_replay = false;
            let mut host = DEFAULT_OPENTRACK_HOST.to_string();
            let mut port = DEFAULT_OPENTRACK_PORT;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--no-replay" => skip_replay = true,
                    "--opentrack-host" => {
                        host = args.next().context("--opentrack-host requires a host")?;
                    }
                    "--opentrack-port" => {
                        port = args
                            .next()
                            .context("--opentrack-port requires a port")?
                            .parse()
                            .context("bad --opentrack-port value")?;
                    }
                    "-h" | "--help" => {
                        println!("usage: track [init_packets_ep.txt] [--no-replay] [--opentrack-host 127.0.0.1] [--opentrack-port 4242]");
                        std::process::exit(0);
                    }
                    s if s.starts_with('-') => anyhow::bail!("unknown option: {s}"),
                    other => track_init_path = other.to_string(),
                }
            }
            return Ok(Self::for_command(Command::Track {
                init_path: track_init_path,
                skip_replay,
                host,
                port,
            }));
        }

        if args.peek().map(String::as_str) == Some("camera") {
            args.next();
            let mut camera_init_path = "init_packets_ep.txt".to_string();
            let mut out_prefix = "frame".to_string();
            let mut max_frames = 10usize;
            let mut skip_replay = false;
            let mut fifo = None;
            let mut frame_type = None;
            let mut interval = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out_prefix = args.next().context("--out requires a prefix")?,
                    "--frames" => {
                        max_frames = args
                            .next()
                            .context("--frames requires a number")?
                            .parse()
                            .context("bad --frames value")?;
                    }
                    "--no-replay" => skip_replay = true,
                    "--fifo" => fifo = Some(args.next().context("--fifo requires a path")?),
                    "--type" => {
                        frame_type = Some(
                            args.next()
                                .context("--type requires a number")?
                                .parse()
                                .context("bad --type value")?,
                        );
                    }
                    "--interval" => {
                        interval = Some(
                            args.next()
                                .context("--interval requires a number (100ns units)")?
                                .parse()
                                .context("bad --interval value")?,
                        );
                    }
                    "-h" | "--help" => {
                        println!("usage: camera [init_packets_ep.txt] [--out frame] [--frames 10] [--no-replay] [--fifo path] [--type N] [--interval 100NS]");
                        std::process::exit(0);
                    }
                    s if s.starts_with('-') => anyhow::bail!("unknown option: {s}"),
                    other => camera_init_path = other.to_string(),
                }
            }
            return Ok(Self::for_command(Command::Camera {
                init_path: camera_init_path,
                out_prefix,
                max_frames,
                skip_replay,
                fifo,
                frame_type,
                interval,
            }));
        }

        if args.peek().map(String::as_str) == Some("analyze-log") {
            args.next();
            let path = args.next().context("usage: analyze-log <path>")?;
            return Ok(Self {
                command: Command::AnalyzeLog { path },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        if args.peek().map(String::as_str) == Some("compare-logs") {
            args.next();
            let inputs: Vec<LogInput> = args
                .map(|arg| parse_log_input(&arg))
                .collect::<Result<_>>()?;
            anyhow::ensure!(
                inputs.len() >= 2,
                "usage: compare-logs [label:]path.bin [label:]path.bin ..."
            );
            return Ok(Self {
                command: Command::CompareLogs { inputs },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        if args.peek().map(String::as_str) == Some("decode-stream") {
            args.next();
            let path = args.next().context("usage: decode-stream <path>")?;
            return Ok(Self {
                command: Command::DecodeStream { path },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        if args.peek().map(String::as_str) == Some("import-tsv") {
            args.next();
            let tsv_path = args
                .next()
                .context("usage: import-tsv <tshark.tsv> <out.bin>")?;
            let log_path = args
                .next()
                .context("usage: import-tsv <tshark.tsv> <out.bin>")?;
            anyhow::ensure!(
                args.next().is_none(),
                "usage: import-tsv <tshark.tsv> <out.bin>"
            );
            return Ok(Self {
                command: Command::ImportTsv { tsv_path, log_path },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        if args.peek().map(String::as_str) == Some("pose-candidates") {
            args.next();
            let inputs: Vec<LogInput> = args
                .map(|arg| parse_log_input(&arg))
                .collect::<Result<_>>()?;
            anyhow::ensure!(
                inputs.len() >= 2,
                "usage: pose-candidates [label:]path.bin [label:]path.bin ..."
            );
            return Ok(Self {
                command: Command::PoseCandidates { inputs },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        if args.peek().map(String::as_str) == Some("compare-decoded") {
            args.next();
            let inputs: Vec<LogInput> = args
                .map(|arg| parse_log_input(&arg))
                .collect::<Result<_>>()?;
            anyhow::ensure!(
                inputs.len() >= 2,
                "usage: compare-decoded [label:]path.bin [label:]path.bin ..."
            );
            return Ok(Self {
                command: Command::CompareDecoded { inputs },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        if args.peek().map(String::as_str) == Some("extract-calibration") {
            args.next();
            let path = args.next().context(
                "usage: extract-calibration <init_packets_ep.txt> [--json calibration.json]",
            )?;
            let mut json_path = None;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--json" => {
                        json_path = Some(args.next().context("--json requires a path")?);
                    }
                    "-h" | "--help" => {
                        println!(
                            "usage: extract-calibration <init_packets_ep.txt> [--json calibration.json]"
                        );
                        std::process::exit(0);
                    }
                    s if s.starts_with('-') => anyhow::bail!("unknown option: {s}"),
                    other => anyhow::bail!("unexpected argument for extract-calibration: {other}"),
                }
            }

            return Ok(Self {
                command: Command::ExtractCalibration { path, json_path },
                log_path: None,
                max_init_packets: None,
                max_stream_packets: None,
                reconnect: false,
                decoded_csv_path: None,
                jsonl_path: None,
                print_decoded: false,
                dashboard: false,
                opentrack_host: None,
                opentrack_port: None,
                opentrack_translation: true,
                opentrack_angle_source: AngleSource::Gaze,
                opentrack_angle_occ: Some(DEFAULT_OPENTRACK_ANGLE_OCC),
                opentrack_angle_scale: DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
                opentrack_angle_map: DEFAULT_OPENTRACK_ANGLE_MAP,
                opentrack_origin_samples: DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
                opentrack_angle_points: None,
                opentrack_roll_points: None,
                opentrack_smoothing: DEFAULT_OPENTRACK_SMOOTHING,
                opentrack_angle_deadzone: DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
                opentrack_rotation_comp: DEFAULT_OPENTRACK_ROTATION_COMP,
                opentrack_translation_scale: DEFAULT_OPENTRACK_TRANSLATION_SCALE,
                opentrack_angle_translation_comp: DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP,
                opentrack_angle_translation_comp_scale:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
                opentrack_angle_translation_deadzone:
                    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM,
                opentrack_coupling_mode: CouplingMode::Rotation,
                opentrack_auto_decouple: false,
            });
        }

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--log" => {
                    log_path = Some(args.next().context("--log requires a path")?);
                }
                "--no-log" => {
                    log_path = None;
                }
                "--max-init-packets" => {
                    max_init_packets = Some(
                        args.next()
                            .context("--max-init-packets requires a number")?
                            .parse()
                            .context("bad --max-init-packets value")?,
                    );
                }
                "--max-stream-packets" => {
                    max_stream_packets = Some(
                        args.next()
                            .context("--max-stream-packets requires a number")?
                            .parse()
                            .context("bad --max-stream-packets value")?,
                    );
                }
                "--no-reconnect" => {
                    reconnect = false;
                }
                "--decoded-csv" => {
                    decoded_csv_path = Some(args.next().context("--decoded-csv requires a path")?);
                }
                "--jsonl" => {
                    jsonl_path = Some(args.next().context("--jsonl requires a path")?);
                }
                "--print-decoded" => {
                    print_decoded = true;
                }
                "--dashboard" => {
                    dashboard = true;
                }
                "--opentrack-host" => {
                    opentrack_host = Some(args.next().context("--opentrack-host requires a host")?);
                }
                "--opentrack-port" => {
                    opentrack_port = Some(
                        args.next()
                            .context("--opentrack-port requires a port")?
                            .parse()
                            .context("bad --opentrack-port value")?,
                    );
                }
                "--opentrack-no-translation" => {
                    opentrack_translation = false;
                }
                "--opentrack-angle-source" => {
                    opentrack_angle_source = parse_angle_source(
                        &args
                            .next()
                            .context("--opentrack-angle-source requires gaze or head")?,
                    )?;
                    if !opentrack_angle_scale_set {
                        opentrack_angle_scale = match opentrack_angle_source {
                            AngleSource::Gaze => DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE,
                            AngleSource::Head | AngleSource::Model => {
                                DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE
                            }
                        };
                    }
                }
                "--opentrack-angle-occ" => {
                    opentrack_angle_source = AngleSource::Head;
                    if !opentrack_angle_scale_set {
                        opentrack_angle_scale = DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE;
                    }
                    opentrack_angle_occ = Some(
                        args.next()
                            .context("--opentrack-angle-occ requires an occurrence number")?
                            .parse()
                            .context("bad --opentrack-angle-occ value")?,
                    );
                }
                "--opentrack-no-angles" => {
                    opentrack_angle_source = AngleSource::Head;
                    opentrack_angle_occ = None;
                }
                "--opentrack-angle-scale" => {
                    opentrack_angle_scale = parse_angle_scale(
                        &args
                            .next()
                            .context("--opentrack-angle-scale requires a number or x,y,z")?,
                    )?;
                    opentrack_angle_scale_set = true;
                }
                "--opentrack-angle-map" => {
                    opentrack_angle_map = parse_angle_map(
                        &args
                            .next()
                            .context("--opentrack-angle-map requires values like x,-y,z")?,
                    )?;
                }
                "--opentrack-origin-samples" => {
                    opentrack_origin_samples = args
                        .next()
                        .context("--opentrack-origin-samples requires a number")?
                        .parse()
                        .context("bad --opentrack-origin-samples value")?;
                    anyhow::ensure!(
                        opentrack_origin_samples > 0,
                        "--opentrack-origin-samples must be greater than zero"
                    );
                }
                "--opentrack-angle-points" => {
                    opentrack_angle_points = Some(parse_point_pair(
                        &args
                            .next()
                            .context("--opentrack-angle-points requires A,B")?,
                        "--opentrack-angle-points",
                    )?);
                }
                "--opentrack-no-angle-points" => {
                    opentrack_angle_points = None;
                }
                "--opentrack-roll-points" => {
                    opentrack_roll_points = Some(parse_point_pair(
                        &args
                            .next()
                            .context("--opentrack-roll-points requires A,B")?,
                        "--opentrack-roll-points",
                    )?);
                }
                "--opentrack-no-roll-points" => {
                    opentrack_roll_points = None;
                }
                "--opentrack-smoothing" => {
                    opentrack_smoothing = args
                        .next()
                        .context("--opentrack-smoothing requires a number from 0 to 1")?
                        .parse()
                        .context("bad --opentrack-smoothing value")?;
                    anyhow::ensure!(
                        (0.0..=1.0).contains(&opentrack_smoothing),
                        "--opentrack-smoothing must be between 0 and 1"
                    );
                }
                "--opentrack-angle-deadzone" => {
                    opentrack_angle_deadzone = args
                        .next()
                        .context("--opentrack-angle-deadzone requires degrees")?
                        .parse()
                        .context("bad --opentrack-angle-deadzone value")?;
                    anyhow::ensure!(
                        opentrack_angle_deadzone >= 0.0,
                        "--opentrack-angle-deadzone must be non-negative"
                    );
                }
                "--opentrack-rotation-comp" => {
                    opentrack_rotation_comp = parse_matrix3(
                        &args
                            .next()
                            .context("--opentrack-rotation-comp requires values")?,
                        "--opentrack-rotation-comp",
                    )?;
                }
                "--opentrack-translation-scale" => {
                    opentrack_translation_scale = parse_three_f64(
                        &args
                            .next()
                            .context("--opentrack-translation-scale requires X,Y,Z")?,
                        "--opentrack-translation-scale",
                    )?;
                }
                "--opentrack-angle-translation-comp" => {
                    opentrack_angle_translation_comp = parse_angle_translation_comp(
                        &args
                            .next()
                            .context("--opentrack-angle-translation-comp requires values")?,
                    )?;
                }
                "--opentrack-angle-translation-comp-scale" => {
                    opentrack_angle_translation_comp_scale = args
                        .next()
                        .context("--opentrack-angle-translation-comp-scale requires a number")?
                        .parse()
                        .context("bad --opentrack-angle-translation-comp-scale value")?;
                    anyhow::ensure!(
                        opentrack_angle_translation_comp_scale >= 0.0,
                        "--opentrack-angle-translation-comp-scale must be non-negative"
                    );
                }
                "--opentrack-angle-translation-deadzone" => {
                    opentrack_angle_translation_deadzone = args
                        .next()
                        .context("--opentrack-angle-translation-deadzone requires centimeters")?
                        .parse()
                        .context("bad --opentrack-angle-translation-deadzone value")?;
                    anyhow::ensure!(
                        opentrack_angle_translation_deadzone >= 0.0,
                        "--opentrack-angle-translation-deadzone must be non-negative"
                    );
                }
                "--opentrack-coupling-mode" => {
                    opentrack_coupling_mode = parse_coupling_mode(&args.next().context(
                        "--opentrack-coupling-mode requires rotation|translation|hybrid|auto",
                    )?)?;
                }
                "--opentrack-auto-decouple" => {
                    opentrack_auto_decouple = true;
                }
                "--opentrack-no-auto-decouple" => {
                    opentrack_auto_decouple = false;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                s if s.starts_with('-') => anyhow::bail!("unknown option: {s}"),
                path => {
                    if init_path.replace(path.to_string()).is_some() {
                        anyhow::bail!("only one init packet file can be provided");
                    }
                }
            }
        }

        anyhow::ensure!(
            !(dashboard && jsonl_path.as_deref() == Some("-")),
            "--dashboard cannot be combined with --jsonl - because both write to stdout"
        );

        Ok(Self {
            command: Command::Replay {
                init_path: init_path.unwrap_or_else(|| "init_packets_ep.txt".to_string()),
            },
            log_path,
            max_init_packets,
            max_stream_packets,
            reconnect,
            decoded_csv_path,
            jsonl_path,
            print_decoded,
            dashboard,
            opentrack_host,
            opentrack_port,
            opentrack_translation,
            opentrack_angle_source,
            opentrack_angle_occ,
            opentrack_angle_scale,
            opentrack_angle_map,
            opentrack_origin_samples,
            opentrack_angle_points,
            opentrack_roll_points,
            opentrack_smoothing,
            opentrack_angle_deadzone,
            opentrack_rotation_comp,
            opentrack_translation_scale,
            opentrack_angle_translation_comp,
            opentrack_angle_translation_comp_scale,
            opentrack_angle_translation_deadzone,
            opentrack_coupling_mode,
            opentrack_auto_decouple,
        })
    }

    pub(crate) fn opentrack_target(&self) -> Option<(String, u16)> {
        if self.opentrack_host.is_none() && self.opentrack_port.is_none() {
            return None;
        }

        Some((
            self.opentrack_host
                .clone()
                .unwrap_or_else(|| DEFAULT_OPENTRACK_HOST.to_string()),
            self.opentrack_port.unwrap_or(DEFAULT_OPENTRACK_PORT),
        ))
    }
}

pub(crate) fn print_usage() {
    println!(
        "usage:\n  cargo run -- [init_packets_ep.txt] [--log tobii_stream.bin] [--decoded-csv decoded.csv] [--jsonl frames.jsonl] [--print-decoded] [--dashboard] [--opentrack-host 127.0.0.1] [--opentrack-port 4242] [--opentrack-no-translation] [--opentrack-translation-scale X,Y,Z] [--opentrack-angle-source model|gaze|head] [--opentrack-coupling-mode rotation|translation|hybrid|auto] [--opentrack-auto-decouple] [--opentrack-angle-points A,B] [--opentrack-angle-translation-comp X,Y,Z] [--opentrack-angle-translation-comp-scale N] [--opentrack-angle-translation-deadzone CM] [--opentrack-angle-occ N] [--opentrack-angle-map x,-y,off] [--opentrack-angle-scale N|YAW,PITCH,ROLL] [--opentrack-origin-samples N] [--opentrack-smoothing 0.35] [--opentrack-angle-deadzone 0.35] [--opentrack-rotation-comp X,Y,Z] [--opentrack-roll-points A,B] [--opentrack-no-angles] [--max-stream-packets N] [--max-init-packets N] [--no-reconnect]\n  cargo run -- analyze-log tobii_stream.bin\n  cargo run -- compare-logs [label:]path.bin [label:]path.bin ...\n  cargo run -- decode-stream tobii_stream.bin\n  cargo run -- import-tsv tshark.tsv out.bin\n  cargo run -- pose-candidates [label:]path.bin [label:]path.bin ...\n  cargo run -- compare-decoded [label:]path.bin [label:]path.bin ...\n  cargo run -- extract-calibration init_packets_ep.txt [--json calibration.json]\n  cargo run -- camera [init_packets_ep.txt] [--out frame] [--frames 10]"
    );
}

pub(crate) fn parse_three_f64(s: &str, option: &str) -> Result<[f64; 3]> {
    let parts: Vec<_> = s.split(',').collect();
    anyhow::ensure!(
        parts.len() == 3,
        "{option} must contain three comma-separated numbers"
    );

    Ok([
        parts[0]
            .trim()
            .parse()
            .with_context(|| format!("bad {option} x value"))?,
        parts[1]
            .trim()
            .parse()
            .with_context(|| format!("bad {option} y value"))?,
        parts[2]
            .trim()
            .parse()
            .with_context(|| format!("bad {option} z value"))?,
    ])
}

pub(crate) fn parse_angle_translation_comp(s: &str) -> Result<[[f64; 3]; 3]> {
    parse_matrix3(s, "--opentrack-angle-translation-comp")
}

pub(crate) fn parse_matrix3(s: &str, option: &str) -> Result<[[f64; 3]; 3]> {
    let parts: Vec<_> = s.split(',').map(str::trim).collect();
    if parts.len() == 3 {
        let diagonal = parse_three_f64(s, option)?;
        return Ok([
            [diagonal[0], 0.0, 0.0],
            [0.0, diagonal[1], 0.0],
            [0.0, 0.0, diagonal[2]],
        ]);
    }

    anyhow::ensure!(
        parts.len() == 9,
        "{option} must contain 3 diagonal values or 9 matrix values"
    );

    let mut matrix = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            matrix[row][col] = parts[row * 3 + col]
                .parse()
                .with_context(|| format!("bad {option} value"))?;
        }
    }
    Ok(matrix)
}

pub(crate) fn parse_point_pair(s: &str, option: &str) -> Result<(usize, usize)> {
    let (a, b) = s
        .split_once(',')
        .with_context(|| format!("{option} must look like A,B"))?;
    let a: usize = a
        .trim()
        .parse()
        .with_context(|| format!("bad first point for {option}"))?;
    let b: usize = b
        .trim()
        .parse()
        .with_context(|| format!("bad second point for {option}"))?;
    anyhow::ensure!(a != b, "{option} needs two different points");
    Ok((a, b))
}

pub(crate) fn parse_angle_scale(s: &str) -> Result<[f64; 3]> {
    let parts: Vec<_> = s.split(',').collect();
    if parts.len() == 1 {
        let scale = parse_nonzero_scale(parts[0])?;
        return Ok([scale, scale, scale]);
    }

    anyhow::ensure!(
        parts.len() == 3,
        "--opentrack-angle-scale must be one number or three comma-separated numbers"
    );

    Ok([
        parse_nonzero_scale(parts[0])?,
        parse_nonzero_scale(parts[1])?,
        parse_nonzero_scale(parts[2])?,
    ])
}

pub(crate) fn parse_angle_source(s: &str) -> Result<AngleSource> {
    match s.trim() {
        "gaze" => Ok(AngleSource::Gaze),
        "head" => Ok(AngleSource::Head),
        "model" => Ok(AngleSource::Model),
        other => anyhow::bail!("unknown --opentrack-angle-source: {other}"),
    }
}

pub(crate) fn parse_coupling_mode(s: &str) -> Result<CouplingMode> {
    match s.trim() {
        "rotation" => Ok(CouplingMode::Rotation),
        "translation" => Ok(CouplingMode::Translation),
        "hybrid" => Ok(CouplingMode::Hybrid),
        "auto" => Ok(CouplingMode::Auto),
        other => anyhow::bail!("unknown --opentrack-coupling-mode: {other}"),
    }
}

pub(crate) fn parse_nonzero_scale(s: &str) -> Result<f64> {
    let scale: f64 = s
        .trim()
        .parse()
        .context("bad --opentrack-angle-scale value")?;
    anyhow::ensure!(
        scale.abs() > f64::EPSILON,
        "--opentrack-angle-scale must not contain zero"
    );
    Ok(scale)
}

pub(crate) fn parse_angle_map(s: &str) -> Result<[AngleComponent; 3]> {
    let parts: Vec<_> = s.split(',').collect();
    anyhow::ensure!(
        parts.len() == 3,
        "--opentrack-angle-map must contain exactly three comma-separated axes"
    );

    Ok([
        parse_angle_component(parts[0])?,
        parse_angle_component(parts[1])?,
        parse_angle_component(parts[2])?,
    ])
}

pub(crate) fn parse_angle_component(s: &str) -> Result<AngleComponent> {
    let s = s.trim();
    let (sign, axis) = if let Some(axis) = s.strip_prefix('-') {
        (-1.0, axis)
    } else if let Some(axis) = s.strip_prefix('+') {
        (1.0, axis)
    } else {
        (1.0, s)
    };

    if matches!(axis, "off" | "none" | "_") {
        return Ok(AngleComponent::disabled());
    }

    let component = match axis {
        "x" | "0" => 0,
        "y" | "1" => 1,
        "z" | "2" => 2,
        _ => anyhow::bail!("bad angle axis {s:?}; use x, y, z, off or 0, 1, 2"),
    };

    Ok(AngleComponent::new(component, sign))
}

pub(crate) fn parse_log_input(arg: &str) -> Result<LogInput> {
    if let Some((label, path)) = arg.split_once(':') {
        anyhow::ensure!(!label.is_empty(), "empty compare-log label in {arg}");
        anyhow::ensure!(!path.is_empty(), "empty compare-log path in {arg}");
        return Ok(LogInput {
            label: label.to_string(),
            path: path.to_string(),
        });
    }

    let path = arg.to_string();
    let label = std::path::Path::new(arg)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(arg)
        .to_string();

    Ok(LogInput { label, path })
}
