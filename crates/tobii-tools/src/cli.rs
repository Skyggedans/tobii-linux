//! Argument parsing for the `tobii5-init-replay` CLI: the default replay /
//! stream mode (`Command::Replay`) plus the research and diagnostic
//! subcommands. Hand-rolled on `std::env::args` (no clap) so the binary stays
//! dependency-free; `print_usage` is the human-readable option surface.

use anyhow::{Context, Result, bail, ensure};
use std::env;
use std::iter::{Peekable, Skip};
use std::str::FromStr;
use tobii_ipc::{
    STREAM_EYE_POSITION, STREAM_GAZE, STREAM_GAZE_DATA, STREAM_GAZE_ORIGIN, STREAM_GAZE_RAW,
    STREAM_HEAD, STREAM_HEAD_POSE, STREAM_NOTIFICATIONS, STREAM_PRESENCE,
};

use crate::compare_head::{HeadOptions, parse_area_choice};
use crate::opentrack::{
    AngleComponent, AngleSource, CouplingMode, DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG,
    DEFAULT_OPENTRACK_ANGLE_MAP, DEFAULT_OPENTRACK_ANGLE_OCC,
    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP, DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE,
    DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM, DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE,
    DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE, DEFAULT_OPENTRACK_ORIGIN_SAMPLES,
    DEFAULT_OPENTRACK_ROLL_POINTS, DEFAULT_OPENTRACK_ROTATION_COMP, DEFAULT_OPENTRACK_SMOOTHING,
    DEFAULT_OPENTRACK_TRANSLATION_SCALE,
};

/// Default raw USB stream log written by the replay mode (`--log`).
pub(crate) const DEFAULT_LOG_PATH: &str = "tobii_stream.bin";

/// `OpenTrack` UDP host used when only `--opentrack-port` is given.
pub(crate) const DEFAULT_OPENTRACK_HOST: &str = "127.0.0.1";

/// `OpenTrack` UDP port used when only `--opentrack-host` is given.
pub(crate) const DEFAULT_OPENTRACK_PORT: u16 = 4242;

/// Streams `ipc-probe` watches when no `--streams` is given: every one but
/// the ~2.6 MB/s IR image stream.
const DEFAULT_IPC_PROBE_STREAMS: u32 = STREAM_HEAD
    | STREAM_GAZE
    | STREAM_PRESENCE
    | STREAM_GAZE_ORIGIN
    | STREAM_EYE_POSITION
    | STREAM_GAZE_DATA
    | STREAM_NOTIFICATIONS
    | STREAM_GAZE_RAW
    | STREAM_HEAD_POSE;

/// `image83-replay`'s synopsis.
const IMAGE83_REPLAY_USAGE: &str =
    "usage: image83-replay <log.bin> [--csv out.csv] [--fits fits.csv] [--landmarks landmarks.f32]";

/// What `image83-replay -h` says after the synopsis: what it does and the
/// layouts of its outputs, those of `--fits` and `--landmarks` as
/// `face_fits` documents them.
const IMAGE83_REPLAY_OUTPUTS: &str = "\
Runs the daemon's head tracker over every 0x50e image of a TBI5LOG1 log, in log order.
  --csv        per image: the legacy pose, raw and calibrated (cm, deg), and the head anchors of
               the last 0x83 gaze frame
  --fits       per image, one row: image_idx, device_ts_us, face (1 or 0; without a face every
               later field is empty), score (the landmark model's logit), found_by_detector (1 or
               0), r_cam_00 .. r_cam_22 (mesh -> camera rotation, row-major; camera x right, y
               down, z forward), t_cam_x_mm, t_cam_y_mm, t_cam_z_mm (mesh origin in the camera
               frame), then six points as <name>_u, <name>_v in the 280-px image's continuous
               pixels (x right, y down, pixel k spans [k, k+1)): centroid (of the 468 landmarks),
               nose_tip (landmark 1), eye_image_left and eye_image_right (the means of the eye
               contours 33 7 163 144 145 153 154 155 133 173 157 158 159 160 161 246 and 263 249
               390 373 374 380 381 382 362 398 384 385 386 387 388 466), corner_33, corner_263.
               Numbers in the shortest form that reads back as the same f64.
  --landmarks  per image, in log order like the rows above, no header: the 468 landmarks as (u, v)
               pairs of little-endian f32 in the same pixels, 3744 bytes, all NaN without a face.
               NumPy: np.fromfile(path, '<f4').reshape(-1, 468, 2)";

/// Init-packet capture replayed to bring the device up when none is given.
const DEFAULT_INIT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tobii-usb/init_packets_ep.txt"
);

/// The process arguments after the program name.
type Args = Peekable<Skip<env::Args>>;

/// Which subcommand was selected on the command line, with its own arguments.
#[derive(Debug)]
pub(crate) enum Command {
    /// Default mode: replay the init capture and stream gaze / head data.
    Replay {
        /// Init-packet capture to replay.
        init_path: String,
    },
    /// Grab frames from the UVC IR camera on interface 2.
    Camera {
        /// Init-packet capture to replay first (unless `skip_replay`).
        init_path: String,
        /// Prefix of the written `<prefix>NNN.pgm` frames.
        out_prefix: String,
        /// Number of frames to capture.
        max_frames: usize,
        /// Skip the init replay.
        skip_replay: bool,
        /// Optional FIFO to stream raw frames into.
        fifo: Option<String>,
        /// UVC frame type override.
        frame_type: Option<u8>,
        /// UVC frame interval override (100 ns units).
        interval: Option<u32>,
    },
    /// Head tracking from the UVC camera, forwarded to `OpenTrack`.
    Track {
        /// Init-packet capture to replay first (unless `skip_replay`).
        init_path: String,
        /// Skip the init replay.
        skip_replay: bool,
        /// `OpenTrack` UDP host.
        host: String,
        /// `OpenTrack` UDP port.
        port: u16,
    },
    /// Check whether the 0x83 stream and the UVC camera run concurrently.
    Probe {
        /// Init-packet capture to replay.
        init_path: String,
    },
    /// Live capture of the 0x50e IR image stream multiplexed on EP 0x83.
    Image83 {
        /// Init-packet capture to replay.
        init_path: String,
        /// Capture duration.
        secs: f64,
        /// Prefix of the written image frames.
        out_prefix: String,
        /// Maximum number of image frames to write.
        max_frames: usize,
        /// Optional raw stream log to write.
        log_path: Option<String>,
        /// Run the head tracker on each frame.
        pose: bool,
        /// Gaze-only baseline: do not enable the image stream.
        no_image: bool,
    },
    /// Replay a recorded 0x83 log through the image / pose decoder.
    Image83Replay {
        /// Recorded stream log.
        path: String,
        /// Optional CSV output of the decoded poses.
        csv: Option<String>,
        /// Optional CSV output of each image's face fit, at full precision.
        fits: Option<String>,
        /// Optional raw f32 output of each image's 468 landmarks.
        landmarks: Option<String>,
    },
    /// Head-axis analysis of the 0x83 3D points in a recorded log.
    Head83 {
        /// Recorded stream log.
        path: String,
        /// Point occurrences to fit the head frame to.
        occs: Vec<usize>,
    },
    /// Summarise a recorded stream log.
    AnalyzeLog {
        /// Recorded stream log.
        path: String,
    },
    /// Compare field statistics across several recorded logs.
    CompareLogs {
        /// Labelled logs to compare.
        inputs: Vec<LogInput>,
    },
    /// Decode and print every frame of a recorded log.
    DecodeStream {
        /// Recorded stream log.
        path: String,
    },
    /// Convert a tshark TSV export into the binary log format.
    ImportTsv {
        /// tshark TSV export.
        tsv_path: String,
        /// Binary log to write.
        log_path: String,
    },
    /// Rank candidate head-pose fields across labelled logs.
    PoseCandidates {
        /// Labelled logs to compare.
        inputs: Vec<LogInput>,
    },
    /// Compare decoded frames across labelled logs.
    CompareDecoded {
        /// Labelled logs to compare.
        inputs: Vec<LogInput>,
    },
    /// Derive the head axes from yaw / pitch / roll / shift labelled logs.
    HeadAxes {
        /// Labelled logs (labels: yaw / pitch / roll / shift*).
        inputs: Vec<LogInput>,
    },
    /// Replay a captured session through the daemon's gaze decoder and
    /// compare it with the Windows Stream Engine's log of the same session;
    /// with `--head`, replay its IR images through the daemon's head pose.
    CompareDll {
        /// TBI5LOG1 log of the captured session.
        log_path: String,
        /// The Windows Stream Engine's callback log of the same session.
        jsonl_path: String,
        /// Compare the head pose, with these options, instead of the gaze.
        head: Option<HeadOptions>,
    },
    /// Connect to a running tobiid, ask it everything and print what streams.
    IpcProbe {
        /// Stream mask to subscribe (`tobii_ipc::STREAM_*`).
        streams: u32,
        /// How long to watch the streams.
        secs: u64,
        /// Set a display area of this size (mm) and x offset first.
        set_display: Option<(f64, f64, f64)>,
    },
    /// Extract the calibration blob from an init-packet capture.
    ExtractCalibration {
        /// Init-packet capture.
        path: String,
        /// Optional JSON output path.
        json_path: Option<String>,
    },
}

/// One `[label:]path.bin` argument of the multi-log analysis subcommands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogInput {
    /// Display label; defaults to the file stem.
    pub(crate) label: String,
    /// Path of the recorded stream log.
    pub(crate) path: String,
}

/// Parsed command line: the selected [`Command`] plus the replay-mode and
/// `OpenTrack` options (which only `Command::Replay` reads).
#[derive(Debug)]
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
    /// default. Used by non-streaming subcommands that ignore the `OpenTrack`
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
            opentrack_roll_points: Some(DEFAULT_OPENTRACK_ROLL_POINTS),
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

    /// Parse the process arguments. The first argument may name a subcommand;
    /// otherwise the whole line is the replay / stream mode. `-h` / `--help`
    /// print usage and exit the process.
    ///
    /// # Errors
    /// Unknown options, options missing their value, values that do not parse
    /// and values that fail validation (ranges, point pairs, ...).
    pub(crate) fn parse() -> Result<Self> {
        let mut args: Args = env::args().skip(1).peekable();
        let parse_subcommand: fn(&mut Args) -> Result<Self> = match args.peek().map(String::as_str)
        {
            Some("track") => Self::parse_track,
            Some("head-axes") => Self::parse_head_axes,
            Some("head83") => Self::parse_head83,
            Some("image83-replay") => Self::parse_image83_replay,
            Some("image83") => Self::parse_image83,
            Some("probe") => Self::parse_probe,
            Some("camera") => Self::parse_camera,
            Some("analyze-log") => Self::parse_analyze_log,
            Some("compare-logs") => Self::parse_compare_logs,
            Some("decode-stream") => Self::parse_decode_stream,
            Some("import-tsv") => Self::parse_import_tsv,
            Some("pose-candidates") => Self::parse_pose_candidates,
            Some("compare-decoded") => Self::parse_compare_decoded,
            Some("extract-calibration") => Self::parse_extract_calibration,
            Some("compare-dll") => Self::parse_compare_dll,
            Some("ipc-probe") => Self::parse_ipc_probe,
            _ => return Self::parse_replay(&mut args),
        };
        args.next();
        parse_subcommand(&mut args)
    }

    fn parse_track(args: &mut Args) -> Result<Self> {
        let mut init_path = DEFAULT_INIT_PATH.to_string();
        let mut skip_replay = false;
        let mut host = DEFAULT_OPENTRACK_HOST.to_string();
        let mut port = DEFAULT_OPENTRACK_PORT;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--no-replay" => skip_replay = true,
                "--opentrack-host" => host = take_value(args, "--opentrack-host", "a host")?,
                "--opentrack-port" => port = parse_value(args, "--opentrack-port", "a port")?,
                "-h" | "--help" => print_help_and_exit(
                    "usage: track [init_packets_ep.txt] [--no-replay] [--opentrack-host 127.0.0.1] [--opentrack-port 4242]",
                ),
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                other => init_path = other.to_string(),
            }
        }
        Ok(Self::for_command(Command::Track {
            init_path,
            skip_replay,
            host,
            port,
        }))
    }

    fn parse_head_axes(args: &mut Args) -> Result<Self> {
        let inputs = parse_log_inputs(
            args,
            "usage: head-axes [label:]path.bin [label:]path.bin ... (labels: yaw/pitch/roll/shift*)",
        )?;
        Ok(Self::for_command(Command::HeadAxes { inputs }))
    }

    fn parse_head83(args: &mut Args) -> Result<Self> {
        let path = args
            .next()
            .context("usage: head83 <log.bin> [occ,occ,...]")?;
        let occs = match args.next() {
            Some(spec) => spec
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<usize>()
                        .context("bad occurrence in head83")
                })
                .collect::<Result<Vec<_>>>()?,
            None => vec![0, 1, 4, 5, 6, 9],
        };
        ensure!(occs.len() >= 3, "head83 needs at least 3 points");
        Ok(Self::for_command(Command::Head83 { path, occs }))
    }

    fn parse_image83_replay(args: &mut Args) -> Result<Self> {
        let mut path = None;
        let mut csv = None;
        let mut fits = None;
        let mut landmarks = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--csv" => csv = Some(take_value(args, "--csv", "a path")?),
                "--fits" => fits = Some(take_value(args, "--fits", "a path")?),
                "--landmarks" => landmarks = Some(take_value(args, "--landmarks", "a path")?),
                "-h" | "--help" => print_help_and_exit(&format!(
                    "{IMAGE83_REPLAY_USAGE}\n{IMAGE83_REPLAY_OUTPUTS}"
                )),
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                other => {
                    if path.replace(other.to_string()).is_some() {
                        bail!("image83-replay takes one log\n{IMAGE83_REPLAY_USAGE}");
                    }
                }
            }
        }
        let path = path.context(IMAGE83_REPLAY_USAGE)?;
        Ok(Self::for_command(Command::Image83Replay {
            path,
            csv,
            fits,
            landmarks,
        }))
    }

    fn parse_image83(args: &mut Args) -> Result<Self> {
        let mut init_path = DEFAULT_INIT_PATH.to_string();
        let mut secs = 10.0f64;
        let mut out_prefix = "image83_".to_string();
        let mut max_frames = 5usize;
        let mut log_path = None;
        let mut pose = false;
        let mut no_image = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--no-image" => no_image = true,
                "--secs" => secs = parse_value(args, "--secs", "a number")?,
                "--out" => out_prefix = take_value(args, "--out", "a prefix")?,
                "--frames" => max_frames = parse_value(args, "--frames", "a number")?,
                "--log" => log_path = Some(take_value(args, "--log", "a path")?),
                "--pose" => pose = true,
                "-h" | "--help" => print_help_and_exit(
                    "usage: image83 [init_packets_ep.txt] [--secs 10] [--out image83_] [--frames 5] [--log file.bin] [--pose] [--no-image]\n\
                     Starts gaze + the 0x50e IR image stream on EP 0x83 and reports per-stream rates and gaze validity; \
                     --pose runs the head tracker on each frame; --no-image is the gaze-only baseline for A/B.",
                ),
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                other => init_path = other.to_string(),
            }
        }
        Ok(Self::for_command(Command::Image83 {
            init_path,
            secs,
            out_prefix,
            max_frames,
            log_path,
            pose,
            no_image,
        }))
    }

    fn parse_probe(args: &mut Args) -> Result<Self> {
        let mut init_path = DEFAULT_INIT_PATH.to_string();
        for arg in args.by_ref() {
            match arg.as_str() {
                "-h" | "--help" => print_help_and_exit(
                    "usage: probe [init_packets_ep.txt]  (checks if 0x83 + camera stream concurrently)",
                ),
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                other => init_path = other.to_string(),
            }
        }
        Ok(Self::for_command(Command::Probe { init_path }))
    }

    fn parse_camera(args: &mut Args) -> Result<Self> {
        let mut init_path = DEFAULT_INIT_PATH.to_string();
        let mut out_prefix = "frame".to_string();
        let mut max_frames = 10usize;
        let mut skip_replay = false;
        let mut fifo = None;
        let mut frame_type = None;
        let mut interval = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out_prefix = take_value(args, "--out", "a prefix")?,
                "--frames" => max_frames = parse_value(args, "--frames", "a number")?,
                "--no-replay" => skip_replay = true,
                "--fifo" => fifo = Some(take_value(args, "--fifo", "a path")?),
                "--type" => frame_type = Some(parse_value(args, "--type", "a number")?),
                "--interval" => {
                    interval = Some(parse_value(args, "--interval", "a number (100ns units)")?);
                }
                "-h" | "--help" => print_help_and_exit(
                    "usage: camera [init_packets_ep.txt] [--out frame] [--frames 10] [--no-replay] [--fifo path] [--type N] [--interval 100NS]",
                ),
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                other => init_path = other.to_string(),
            }
        }
        Ok(Self::for_command(Command::Camera {
            init_path,
            out_prefix,
            max_frames,
            skip_replay,
            fifo,
            frame_type,
            interval,
        }))
    }

    fn parse_analyze_log(args: &mut Args) -> Result<Self> {
        let path = args.next().context("usage: analyze-log <path>")?;
        Ok(Self::for_command(Command::AnalyzeLog { path }))
    }

    fn parse_compare_logs(args: &mut Args) -> Result<Self> {
        let inputs = parse_log_inputs(
            args,
            "usage: compare-logs [label:]path.bin [label:]path.bin ...",
        )?;
        Ok(Self::for_command(Command::CompareLogs { inputs }))
    }

    fn parse_decode_stream(args: &mut Args) -> Result<Self> {
        let path = args.next().context("usage: decode-stream <path>")?;
        Ok(Self::for_command(Command::DecodeStream { path }))
    }

    fn parse_compare_dll(args: &mut Args) -> Result<Self> {
        const USAGE: &str = "usage: compare-dll <session.bin> <session.jsonl> [--head \
             [--display-area auto|TLx,TLy,TLz,TRx,TRy,TRz,BLx,BLy,BLz] [--gates gates.json] \
             [--csv per-image.csv]]\n\
             Replays a captured session's gaze frames through the daemon's decoder and compares \
             them with the Windows Stream Engine's log of it. --head replays its IR images through \
             the daemon's head pose instead and compares the poses with the Stream Engine's; the \
             display area is the log's (auto) or the given one (mm, tracker frame); --gates checks \
             the metrics against a gates file (tools/headpose/gates.json) and fails on a gate that \
             fails; --csv writes one row per image with a Stream Engine pose.";
        if matches!(args.peek().map(String::as_str), Some("-h" | "--help")) {
            print_help_and_exit(USAGE);
        }
        let log_path = args.next().context(USAGE)?;
        let jsonl_path = args.next().context(USAGE)?;
        let mut head = false;
        let mut options = HeadOptions::default();
        // The first option that only --head takes, to refuse it without one.
        let mut head_option = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--head" => head = true,
                "--display-area" => {
                    options.area = parse_area_choice(&take_value(
                        args,
                        "--display-area",
                        "auto or nine numbers",
                    )?)?;
                    head_option.get_or_insert("--display-area");
                }
                "--gates" => {
                    options.gates = Some(take_value(args, "--gates", "a path")?);
                    head_option.get_or_insert("--gates");
                }
                "--csv" => {
                    options.csv = Some(take_value(args, "--csv", "a path")?);
                    head_option.get_or_insert("--csv");
                }
                "-h" | "--help" => print_help_and_exit(USAGE),
                other => bail!("unknown compare-dll argument {other}\n{USAGE}"),
            }
        }
        if let Some(option) = head_option {
            ensure!(head, "{option} needs --head\n{USAGE}");
        }
        Ok(Self::for_command(Command::CompareDll {
            log_path,
            jsonl_path,
            head: head.then_some(options),
        }))
    }

    fn parse_ipc_probe(args: &mut Args) -> Result<Self> {
        const USAGE: &str =
            "usage: ipc-probe [--streams MASK] [--secs N] [--set-display W,H[,OFFSET_X]]";
        let mut streams = DEFAULT_IPC_PROBE_STREAMS;
        let mut secs = 5;
        let mut set_display = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--streams" => {
                    let v = args.next().context(USAGE)?;
                    streams = match v.strip_prefix("0x") {
                        Some(hex) => u32::from_str_radix(hex, 16),
                        None => v.parse(),
                    }
                    .with_context(|| format!("bad --streams {v}"))?;
                }
                "--secs" => secs = args.next().context(USAGE)?.parse().context("bad --secs")?,
                "--set-display" => {
                    let v = args.next().context(USAGE)?;
                    let parts: Vec<f64> = v
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .with_context(|| format!("bad --set-display {v}"))?;
                    ensure!(matches!(parts.len(), 2 | 3), USAGE);
                    set_display = Some((parts[0], parts[1], parts.get(2).copied().unwrap_or(0.0)));
                }
                _ => bail!("unknown ipc-probe option {arg}\n{USAGE}"),
            }
        }
        Ok(Self::for_command(Command::IpcProbe {
            streams,
            secs,
            set_display,
        }))
    }

    fn parse_import_tsv(args: &mut Args) -> Result<Self> {
        const USAGE: &str = "usage: import-tsv <tshark.tsv> <out.bin>";
        let tsv_path = args.next().context(USAGE)?;
        let log_path = args.next().context(USAGE)?;
        ensure!(args.next().is_none(), USAGE);
        Ok(Self::for_command(Command::ImportTsv { tsv_path, log_path }))
    }

    fn parse_pose_candidates(args: &mut Args) -> Result<Self> {
        let inputs = parse_log_inputs(
            args,
            "usage: pose-candidates [label:]path.bin [label:]path.bin ...",
        )?;
        Ok(Self::for_command(Command::PoseCandidates { inputs }))
    }

    fn parse_compare_decoded(args: &mut Args) -> Result<Self> {
        let inputs = parse_log_inputs(
            args,
            "usage: compare-decoded [label:]path.bin [label:]path.bin ...",
        )?;
        Ok(Self::for_command(Command::CompareDecoded { inputs }))
    }

    fn parse_extract_calibration(args: &mut Args) -> Result<Self> {
        const USAGE: &str =
            "usage: extract-calibration <init_packets_ep.txt> [--json calibration.json]";
        let path = args.next().context(USAGE)?;
        let mut json_path = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => json_path = Some(take_value(args, "--json", "a path")?),
                "-h" | "--help" => print_help_and_exit(USAGE),
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                other => bail!("unexpected argument for extract-calibration: {other}"),
            }
        }
        Ok(Self::for_command(Command::ExtractCalibration {
            path,
            json_path,
        }))
    }

    /// The default replay / stream mode: every option is optional and the
    /// single positional argument is the init-packet capture.
    fn parse_replay(args: &mut Args) -> Result<Self> {
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
        let mut opentrack_roll_points = Some(DEFAULT_OPENTRACK_ROLL_POINTS);
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

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--log" => log_path = Some(take_value(args, "--log", "a path")?),
                "--no-log" => log_path = None,
                "--max-init-packets" => {
                    max_init_packets = Some(parse_value(args, "--max-init-packets", "a number")?);
                }
                "--max-stream-packets" => {
                    max_stream_packets =
                        Some(parse_value(args, "--max-stream-packets", "a number")?);
                }
                "--no-reconnect" => reconnect = false,
                "--decoded-csv" => {
                    decoded_csv_path = Some(take_value(args, "--decoded-csv", "a path")?);
                }
                "--jsonl" => jsonl_path = Some(take_value(args, "--jsonl", "a path")?),
                "--print-decoded" => print_decoded = true,
                "--dashboard" => dashboard = true,
                "--opentrack-host" => {
                    opentrack_host = Some(take_value(args, "--opentrack-host", "a host")?);
                }
                "--opentrack-port" => {
                    opentrack_port = Some(parse_value(args, "--opentrack-port", "a port")?);
                }
                "--opentrack-no-translation" => opentrack_translation = false,
                "--opentrack-angle-source" => {
                    opentrack_angle_source = parse_angle_source(&take_value(
                        args,
                        "--opentrack-angle-source",
                        "gaze or head",
                    )?)?;
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
                    opentrack_angle_occ = Some(parse_value(
                        args,
                        "--opentrack-angle-occ",
                        "an occurrence number",
                    )?);
                }
                "--opentrack-no-angles" => {
                    opentrack_angle_source = AngleSource::Head;
                    opentrack_angle_occ = None;
                }
                "--opentrack-angle-scale" => {
                    opentrack_angle_scale = parse_angle_scale(&take_value(
                        args,
                        "--opentrack-angle-scale",
                        "a number or x,y,z",
                    )?)?;
                    opentrack_angle_scale_set = true;
                }
                "--opentrack-angle-map" => {
                    opentrack_angle_map = parse_angle_map(&take_value(
                        args,
                        "--opentrack-angle-map",
                        "values like x,-y,z",
                    )?)?;
                }
                "--opentrack-origin-samples" => {
                    opentrack_origin_samples =
                        parse_value(args, "--opentrack-origin-samples", "a number")?;
                    ensure!(
                        opentrack_origin_samples > 0,
                        "--opentrack-origin-samples must be greater than zero"
                    );
                }
                "--opentrack-angle-points" => {
                    opentrack_angle_points = Some(parse_point_pair(
                        &take_value(args, "--opentrack-angle-points", "A,B")?,
                        "--opentrack-angle-points",
                    )?);
                }
                "--opentrack-no-angle-points" => opentrack_angle_points = None,
                "--opentrack-roll-points" => {
                    opentrack_roll_points = Some(parse_point_pair(
                        &take_value(args, "--opentrack-roll-points", "A,B")?,
                        "--opentrack-roll-points",
                    )?);
                }
                "--opentrack-no-roll-points" => opentrack_roll_points = None,
                "--opentrack-smoothing" => {
                    opentrack_smoothing =
                        parse_value(args, "--opentrack-smoothing", "a number from 0 to 1")?;
                    ensure!(
                        (0.0..=1.0).contains(&opentrack_smoothing),
                        "--opentrack-smoothing must be between 0 and 1"
                    );
                }
                "--opentrack-angle-deadzone" => {
                    opentrack_angle_deadzone =
                        parse_value(args, "--opentrack-angle-deadzone", "degrees")?;
                    ensure!(
                        opentrack_angle_deadzone >= 0.0,
                        "--opentrack-angle-deadzone must be non-negative"
                    );
                }
                "--opentrack-rotation-comp" => {
                    opentrack_rotation_comp = parse_matrix3(
                        &take_value(args, "--opentrack-rotation-comp", "values")?,
                        "--opentrack-rotation-comp",
                    )?;
                }
                "--opentrack-translation-scale" => {
                    opentrack_translation_scale = parse_three_f64(
                        &take_value(args, "--opentrack-translation-scale", "X,Y,Z")?,
                        "--opentrack-translation-scale",
                    )?;
                }
                "--opentrack-angle-translation-comp" => {
                    opentrack_angle_translation_comp = parse_angle_translation_comp(&take_value(
                        args,
                        "--opentrack-angle-translation-comp",
                        "values",
                    )?)?;
                }
                "--opentrack-angle-translation-comp-scale" => {
                    opentrack_angle_translation_comp_scale =
                        parse_value(args, "--opentrack-angle-translation-comp-scale", "a number")?;
                    ensure!(
                        opentrack_angle_translation_comp_scale >= 0.0,
                        "--opentrack-angle-translation-comp-scale must be non-negative"
                    );
                }
                "--opentrack-angle-translation-deadzone" => {
                    opentrack_angle_translation_deadzone = parse_value(
                        args,
                        "--opentrack-angle-translation-deadzone",
                        "centimeters",
                    )?;
                    ensure!(
                        opentrack_angle_translation_deadzone >= 0.0,
                        "--opentrack-angle-translation-deadzone must be non-negative"
                    );
                }
                "--opentrack-coupling-mode" => {
                    opentrack_coupling_mode = parse_coupling_mode(&take_value(
                        args,
                        "--opentrack-coupling-mode",
                        "rotation|translation|hybrid|auto",
                    )?)?;
                }
                "--opentrack-auto-decouple" => opentrack_auto_decouple = true,
                "--opentrack-no-auto-decouple" => opentrack_auto_decouple = false,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                s if s.starts_with('-') => bail!("unknown option: {s}"),
                path => {
                    if init_path.replace(path.to_string()).is_some() {
                        bail!("only one init packet file can be provided");
                    }
                }
            }
        }

        ensure!(
            !(dashboard && jsonl_path.as_deref() == Some("-")),
            "--dashboard cannot be combined with --jsonl - because both write to stdout"
        );

        Ok(Self {
            command: Command::Replay {
                init_path: init_path.unwrap_or_else(|| DEFAULT_INIT_PATH.to_string()),
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

    /// `(host, port)` of the `OpenTrack` UDP sink, or `None` when neither
    /// `--opentrack-host` nor `--opentrack-port` was given; a missing half
    /// falls back to the default.
    #[must_use]
    pub(crate) fn opentrack_target(&self) -> Option<(String, u16)> {
        if self.opentrack_host.is_none() && self.opentrack_port.is_none() {
            return None;
        }

        Some((
            self.opentrack_host
                .as_deref()
                .unwrap_or(DEFAULT_OPENTRACK_HOST)
                .to_string(),
            self.opentrack_port.unwrap_or(DEFAULT_OPENTRACK_PORT),
        ))
    }
}

/// Print `usage` (a subcommand's `-h` text) and exit successfully.
fn print_help_and_exit(usage: &str) -> ! {
    println!("{usage}");
    std::process::exit(0)
}

/// Take the next argument as the value of `option`, erroring with
/// "`option` requires `what`" when the command line ends first.
fn take_value(args: &mut Args, option: &str, what: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{option} requires {what}"))
}

/// Take the next argument as the value of `option` and parse it as `T`.
fn parse_value<T>(args: &mut Args, option: &str, what: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    take_value(args, option, what)?
        .parse()
        .with_context(|| format!("bad {option} value"))
}

/// Parse the remaining arguments as `[label:]path.bin` inputs; the multi-log
/// subcommands need at least two.
fn parse_log_inputs(args: &mut Args, usage: &str) -> Result<Vec<LogInput>> {
    let inputs: Vec<LogInput> = args
        .map(|arg| parse_log_input(&arg))
        .collect::<Result<_>>()?;
    ensure!(inputs.len() >= 2, "{usage}");
    Ok(inputs)
}

/// Print the top-level usage text (replay mode plus subcommand one-liners).
pub(crate) fn print_usage() {
    println!(
        "usage:\n  cargo run -p tobii-tools -- [init_packets_ep.txt] [--log tobii_stream.bin] [--decoded-csv decoded.csv] [--jsonl frames.jsonl] [--print-decoded] [--dashboard] [--opentrack-host 127.0.0.1] [--opentrack-port 4242] [--opentrack-no-translation] [--opentrack-translation-scale X,Y,Z] [--opentrack-angle-source model|gaze|head] [--opentrack-coupling-mode rotation|translation|hybrid|auto] [--opentrack-auto-decouple] [--opentrack-angle-points A,B] [--opentrack-angle-translation-comp X,Y,Z] [--opentrack-angle-translation-comp-scale N] [--opentrack-angle-translation-deadzone CM] [--opentrack-angle-occ N] [--opentrack-angle-map x,-y,off] [--opentrack-angle-scale N|YAW,PITCH,ROLL] [--opentrack-origin-samples N] [--opentrack-smoothing 0.35] [--opentrack-angle-deadzone 0.35] [--opentrack-rotation-comp X,Y,Z] [--opentrack-roll-points A,B] [--opentrack-no-angles] [--max-stream-packets N] [--max-init-packets N] [--no-reconnect]\n  cargo run -p tobii-tools -- analyze-log tobii_stream.bin\n  cargo run -p tobii-tools -- compare-logs [label:]path.bin [label:]path.bin ...\n  cargo run -p tobii-tools -- decode-stream tobii_stream.bin\n  cargo run -p tobii-tools -- import-tsv tshark.tsv out.bin\n  cargo run -p tobii-tools -- pose-candidates [label:]path.bin [label:]path.bin ...\n  cargo run -p tobii-tools -- compare-decoded [label:]path.bin [label:]path.bin ...\n  cargo run -p tobii-tools -- extract-calibration init_packets_ep.txt [--json calibration.json]\n  cargo run -p tobii-tools -- compare-dll session.bin session.jsonl [--head [--display-area auto|TLx,TLy,TLz,TRx,TRy,TRz,BLx,BLy,BLz] [--gates gates.json] [--csv per-image.csv]]\n  cargo run -p tobii-tools -- ipc-probe [--streams MASK] [--secs N] [--set-display W,H[,OFFSET_X]]\n  cargo run -p tobii-tools -- camera [init_packets_ep.txt] [--out frame] [--frames 10]"
    );
}

/// Parse `x,y,z` for `option`.
///
/// # Errors
/// Not exactly three comma-separated values, or a value that is not a number.
pub(crate) fn parse_three_f64(s: &str, option: &str) -> Result<[f64; 3]> {
    let parts: Vec<&str> = s.split(',').collect();
    let [x, y, z] = parts.as_slice() else {
        bail!("{option} must contain three comma-separated numbers");
    };

    Ok([
        x.trim()
            .parse()
            .with_context(|| format!("bad {option} x value"))?,
        y.trim()
            .parse()
            .with_context(|| format!("bad {option} y value"))?,
        z.trim()
            .parse()
            .with_context(|| format!("bad {option} z value"))?,
    ])
}

/// Parse the `--opentrack-angle-translation-comp` matrix (see [`parse_matrix3`]).
///
/// # Errors
/// Same as [`parse_matrix3`].
pub(crate) fn parse_angle_translation_comp(s: &str) -> Result<[[f64; 3]; 3]> {
    parse_matrix3(s, "--opentrack-angle-translation-comp")
}

/// Parse a 3x3 matrix for `option`: either three diagonal values `x,y,z` or
/// nine row-major values.
///
/// # Errors
/// Neither three nor nine comma-separated values, or a value that is not a
/// number.
pub(crate) fn parse_matrix3(s: &str, option: &str) -> Result<[[f64; 3]; 3]> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() == 3 {
        let diagonal = parse_three_f64(s, option)?;
        return Ok([
            [diagonal[0], 0.0, 0.0],
            [0.0, diagonal[1], 0.0],
            [0.0, 0.0, diagonal[2]],
        ]);
    }

    ensure!(
        parts.len() == 9,
        "{option} must contain 3 diagonal values or 9 matrix values"
    );

    let mut matrix = [[0.0; 3]; 3];
    for (row, values) in matrix.iter_mut().zip(parts.as_chunks::<3>().0) {
        for (cell, value) in row.iter_mut().zip(values) {
            *cell = value
                .parse()
                .with_context(|| format!("bad {option} value"))?;
        }
    }
    Ok(matrix)
}

/// Parse `A,B` (two distinct point occurrences) for `option`.
///
/// # Errors
/// No comma, a side that is not an integer, or `A == B`.
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
    ensure!(a != b, "{option} needs two different points");
    Ok((a, b))
}

/// Parse `--opentrack-angle-scale`: one scale for all axes or `yaw,pitch,roll`.
///
/// # Errors
/// Neither one nor three values, a non-numeric value, or a zero scale.
pub(crate) fn parse_angle_scale(s: &str) -> Result<[f64; 3]> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.as_slice() {
        [scale] => {
            let scale = parse_nonzero_scale(scale)?;
            Ok([scale; 3])
        }
        [yaw, pitch, roll] => Ok([
            parse_nonzero_scale(yaw)?,
            parse_nonzero_scale(pitch)?,
            parse_nonzero_scale(roll)?,
        ]),
        _ => bail!("--opentrack-angle-scale must be one number or three comma-separated numbers"),
    }
}

/// Parse `--opentrack-angle-source` (`gaze`, `head` or `model`).
///
/// # Errors
/// Any other value.
pub(crate) fn parse_angle_source(s: &str) -> Result<AngleSource> {
    match s.trim() {
        "gaze" => Ok(AngleSource::Gaze),
        "head" => Ok(AngleSource::Head),
        "model" => Ok(AngleSource::Model),
        other => bail!("unknown --opentrack-angle-source: {other}"),
    }
}

/// Parse `--opentrack-coupling-mode` (`rotation`, `translation`, `hybrid` or
/// `auto`).
///
/// # Errors
/// Any other value.
pub(crate) fn parse_coupling_mode(s: &str) -> Result<CouplingMode> {
    match s.trim() {
        "rotation" => Ok(CouplingMode::Rotation),
        "translation" => Ok(CouplingMode::Translation),
        "hybrid" => Ok(CouplingMode::Hybrid),
        "auto" => Ok(CouplingMode::Auto),
        other => bail!("unknown --opentrack-coupling-mode: {other}"),
    }
}

/// Parse one `--opentrack-angle-scale` component, rejecting zero.
///
/// # Errors
/// Not a number, or (within `f64::EPSILON` of) zero.
pub(crate) fn parse_nonzero_scale(s: &str) -> Result<f64> {
    let scale: f64 = s
        .trim()
        .parse()
        .context("bad --opentrack-angle-scale value")?;
    ensure!(
        scale.abs() > f64::EPSILON,
        "--opentrack-angle-scale must not contain zero"
    );
    Ok(scale)
}

/// Parse `--opentrack-angle-map`: three axes like `x,-y,off`.
///
/// # Errors
/// Not exactly three values, or an axis [`parse_angle_component`] rejects.
pub(crate) fn parse_angle_map(s: &str) -> Result<[AngleComponent; 3]> {
    let parts: Vec<&str> = s.split(',').collect();
    let [yaw, pitch, roll] = parts.as_slice() else {
        bail!("--opentrack-angle-map must contain exactly three comma-separated axes");
    };

    Ok([
        parse_angle_component(yaw)?,
        parse_angle_component(pitch)?,
        parse_angle_component(roll)?,
    ])
}

/// Parse one angle-map axis: `[+|-](x|y|z|0|1|2)` or `off` / `none` / `_`.
///
/// # Errors
/// Any other axis name.
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
        _ => bail!("bad angle axis {s:?}; use x, y, z, off or 0, 1, 2"),
    };

    Ok(AngleComponent::new(component, sign))
}

/// Parse one `[label:]path.bin` argument; without a label the file stem is
/// used.
///
/// # Errors
/// An explicit label or path that is empty.
pub(crate) fn parse_log_input(arg: &str) -> Result<LogInput> {
    let Some((label, path)) = arg.split_once(':') else {
        let label = std::path::Path::new(arg)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(arg)
            .to_string();
        return Ok(LogInput {
            label,
            path: arg.to_string(),
        });
    };

    ensure!(!label.is_empty(), "empty compare-log label in {arg}");
    ensure!(!path.is_empty(), "empty compare-log path in {arg}");
    Ok(LogInput {
        label: label.to_string(),
        path: path.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_ipc::STREAM_IMAGE;

    /// Without `--streams`, `ipc-probe` watches every stream the daemon has
    /// but the IR images, the Stream Engine's head pose included.
    #[test]
    fn ipc_probe_watches_every_stream_but_the_images_by_default() {
        let every_stream = (STREAM_HEAD_POSE << 1) - 1;

        assert_eq!(DEFAULT_IPC_PROBE_STREAMS, every_stream & !STREAM_IMAGE);
        assert_eq!(DEFAULT_IPC_PROBE_STREAMS, 0x3bf);
    }
}
