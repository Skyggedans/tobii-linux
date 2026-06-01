use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::{
    env, error, fmt, fs, thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const VID: u16 = 0x2104;
const PID: u16 = 0x0313;

const IFACE: u8 = 0;
const EP_IN: u8 = 0x83;
const DEFAULT_LOG_PATH: &str = "tobii_stream.bin";
const LOG_MAGIC: &[u8; 8] = b"TBI5LOG1";
const STREAM_TLV_OFFSET: usize = 34;
const GAZE_COORD_MAX: f64 = 1024.0;
const DEFAULT_OPENTRACK_HOST: &str = "127.0.0.1";
const DEFAULT_OPENTRACK_PORT: u16 = 4242;
const OPENTRACK_HEAD_SCALE: f64 = 1000.0;
const DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE: [f64; 3] = [40.0, 60.0, 1.0];
const DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE: [f64; 3] = [2.0, 1.0, 2.0];
const DEFAULT_OPENTRACK_ANGLE_OCC: usize = 7;
const OPENTRACK_MAX_HEAD_ANGLE_DEG: f64 = 45.0;
const DEFAULT_OPENTRACK_ORIGIN_SAMPLES: usize = 30;
const DEFAULT_OPENTRACK_SMOOTHING: f64 = 0.35;
const DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG: f64 = 0.35;
const DEFAULT_OPENTRACK_ROTATION_COMP: [[f64; 3]; 3] = [[0.0; 3]; 3];
const DEFAULT_OPENTRACK_TRANSLATION_SCALE: [f64; 3] = [1.0, 1.0, 1.0];
const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP: [[f64; 3]; 3] = [[0.0; 3]; 3];
const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE: f64 = 1.0;
const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM: f64 = 0.0;
const OPENTRACK_COUPLING_SWITCH_FRAMES: usize = 4;
const OPENTRACK_COUPLING_SWITCH_MARGIN: f64 = 0.20;
const MAX_REPLAY_ATTEMPTS: usize = 5;
const STARTUP_STREAM_TIMEOUTS: u32 = 3;

const DEFAULT_OPENTRACK_ANGLE_MAP: [AngleComponent; 3] = [
    AngleComponent::new(0, 1.0),
    AngleComponent::new(1, -1.0),
    AngleComponent::disabled(),
];

const LIVE_FIELDS: &[LiveField] = &[
    LiveField::new("gaze0_x", 0x00021f40, 0, 0),
    LiveField::new("gaze0_y", 0x00021f40, 0, 1),
    LiveField::new("gaze1_x", 0x00021f40, 1, 0),
    LiveField::new("gaze1_y", 0x00021f40, 1, 1),
    LiveField::new("gaze3_x", 0x00021f40, 3, 0),
    LiveField::new("gaze3_y", 0x00021f40, 3, 1),
    LiveField::new("gaze5_x", 0x00021f40, 5, 0),
    LiveField::new("gaze5_y", 0x00021f40, 5, 1),
    LiveField::new("head3_x", 0x00031f41, 3, 0),
    LiveField::new("head3_y", 0x00031f41, 3, 1),
    LiveField::new("head3_z", 0x00031f41, 3, 2),
    LiveField::new("head5_x", 0x00031f41, 5, 0),
    LiveField::new("head5_y", 0x00031f41, 5, 1),
    LiveField::new("head5_z", 0x00031f41, 5, 2),
    LiveField::new("head6_x", 0x00031f41, 6, 0),
    LiveField::new("head6_y", 0x00031f41, 6, 1),
    LiveField::new("head6_z", 0x00031f41, 6, 2),
    LiveField::new("head8_x", 0x00031f41, 8, 0),
    LiveField::new("head8_y", 0x00031f41, 8, 1),
    LiveField::new("head8_z", 0x00031f41, 8, 2),
    LiveField::new("head9_x", 0x00031f41, 9, 0),
    LiveField::new("head9_y", 0x00031f41, 9, 1),
    LiveField::new("head9_z", 0x00031f41, 9, 2),
];

const DERIVED_FIELDS: &[&str] = &[
    "gaze_valid",
    "gaze_x",
    "gaze_y",
    "gaze_norm_x",
    "gaze_norm_y",
    "left_eye_x",
    "left_eye_y",
    "left_eye_norm_x",
    "left_eye_norm_y",
    "right_eye_x",
    "right_eye_y",
    "right_eye_norm_x",
    "right_eye_norm_y",
    "head_x",
    "head_y",
    "head_z",
    "head_yaw",
    "head_pitch",
    "head_roll",
];

struct InitPacket {
    ep: u8,
    data: Vec<u8>,
}

#[derive(Clone, Copy)]
struct LiveField {
    name: &'static str,
    id: u32,
    occurrence: usize,
    component: usize,
}

impl LiveField {
    const fn new(name: &'static str, id: u32, occurrence: usize, component: usize) -> Self {
        Self {
            name,
            id,
            occurrence,
            component,
        }
    }

    fn key(self) -> (u32, usize, usize) {
        (self.id, self.occurrence, self.component)
    }
}

fn main() -> Result<()> {
    let opts = Options::parse()?;

    if let Command::AnalyzeLog { path } = &opts.command {
        return analyze_log(path);
    }

    if let Command::CompareLogs { inputs } = &opts.command {
        return compare_logs(inputs);
    }

    if let Command::DecodeStream { path } = &opts.command {
        return decode_stream(path);
    }

    if let Command::ImportTsv { tsv_path, log_path } = &opts.command {
        return import_tsv(tsv_path, log_path);
    }

    if let Command::PoseCandidates { inputs } = &opts.command {
        return pose_candidates(inputs);
    }

    if let Command::CompareDecoded { inputs } = &opts.command {
        return compare_decoded(inputs);
    }

    if let Command::ExtractCalibration { path, json_path } = &opts.command {
        return extract_calibration(path, json_path.as_deref());
    }

    let Command::Replay { init_path } = &opts.command else {
        unreachable!();
    };

    let packets = read_init_packets(init_path)?;
    let init_limit = opts
        .max_init_packets
        .unwrap_or(packets.len())
        .min(packets.len());
    println!(
        "Loaded {} init packets from {}; replaying {}",
        packets.len(),
        init_path,
        init_limit
    );

    let ctx = UsbContext::new()?;
    let attempts = if opts.reconnect {
        MAX_REPLAY_ATTEMPTS
    } else {
        1
    };

    for attempt in 1..=attempts {
        if attempt > 1 {
            println!("Retrying initialization, attempt {attempt}/{attempts}");
        }

        match replay_and_read_stream(&ctx, &opts, &packets, init_limit) {
            Ok(()) => return Ok(()),
            Err(err)
                if opts.reconnect
                    && attempt < attempts
                    && err.downcast_ref::<StreamStartupTimeout>().is_some() =>
            {
                println!("Stream did not start after init; retrying full init replay");
                thread::sleep(Duration::from_millis(500));
            }
            Err(err) => return Err(err),
        }
    }

    anyhow::bail!("stream did not start after {attempts} initialization attempts")
}

#[derive(Debug)]
struct StreamStartupTimeout;

impl fmt::Display for StreamStartupTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream did not start after initialization")
    }
}

impl error::Error for StreamStartupTimeout {}

fn replay_and_read_stream(
    ctx: &UsbContext,
    opts: &Options,
    packets: &[InitPacket],
    init_limit: usize,
) -> Result<()> {
    let mut h = open_tobii(ctx)?;
    let mut log = opts
        .log_path
        .as_deref()
        .map(PacketLog::create)
        .transpose()?;
    let mut live_csv = opts
        .decoded_csv_path
        .as_deref()
        .map(DecodedCsv::create)
        .transpose()?;
    let mut jsonl = opts
        .jsonl_path
        .as_deref()
        .map(JsonlOutput::create)
        .transpose()?;
    let mut opentrack = opts
        .opentrack_target()
        .map(|(host, port)| {
            OpentrackUdp::connect(
                &host,
                port,
                opts.opentrack_translation,
                opts.opentrack_angle_source,
                opts.opentrack_angle_occ,
                opts.opentrack_angle_scale,
                opts.opentrack_angle_map,
                opts.opentrack_origin_samples,
                opts.opentrack_angle_points,
                opts.opentrack_roll_points,
                opts.opentrack_smoothing,
                opts.opentrack_angle_deadzone,
                opts.opentrack_rotation_comp,
                opts.opentrack_translation_scale,
                opts.opentrack_angle_translation_comp,
                opts.opentrack_angle_translation_comp_scale,
                opts.opentrack_angle_translation_deadzone,
                opts.opentrack_coupling_mode,
            )
        })
        .transpose()?;

    vendor_control_init(&mut h)?;

    for (i, pkt) in packets.iter().take(init_limit).enumerate() {
        let packet_no = i + 1;

        let expected_seq = if marker(&pkt.data) == Some(0x51) {
            seq(&pkt.data)
        } else {
            None
        };

        println!(
            "OUT #{:03}: ep=0x{:02x} {} bytes marker={:?} seq={:?}",
            packet_no,
            pkt.ep,
            pkt.data.len(),
            marker(&pkt.data),
            seq(&pkt.data)
        );

        match h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000)) {
            Ok(written) => {
                println!("  written={}", written);

                if let Some(expected_seq) = expected_seq {
                    if let Err(e) = wait_for_response_seq(&mut h, expected_seq, &mut log) {
                        println!("  no response for seq {}: {}", expected_seq, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("OUT #{:03} failed: {:?}", packet_no, e);
                break;
            }
        }

        if packet_no == 200 {
            println!("=== packet 200, waiting 500 ms ===");
            thread::sleep(Duration::from_millis(500));
            drain_in_limited(
                &mut h,
                20,
                &mut log,
                &mut live_csv,
                &mut jsonl,
                &mut opentrack,
                opts.print_decoded,
            );
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }

    println!("Init replay finished. Reading stream...");
    read_stream(
        ctx,
        h,
        &opts,
        &mut log,
        &mut live_csv,
        &mut jsonl,
        &mut opentrack,
    )
}

enum Command {
    Replay {
        init_path: String,
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

struct LogInput {
    label: String,
    path: String,
}

struct Options {
    command: Command,
    log_path: Option<String>,
    max_init_packets: Option<usize>,
    max_stream_packets: Option<u64>,
    reconnect: bool,
    decoded_csv_path: Option<String>,
    jsonl_path: Option<String>,
    print_decoded: bool,
    dashboard: bool,
    opentrack_host: Option<String>,
    opentrack_port: Option<u16>,
    opentrack_translation: bool,
    opentrack_angle_source: AngleSource,
    opentrack_angle_occ: Option<usize>,
    opentrack_angle_scale: [f64; 3],
    opentrack_angle_map: [AngleComponent; 3],
    opentrack_origin_samples: usize,
    opentrack_angle_points: Option<(usize, usize)>,
    opentrack_roll_points: Option<(usize, usize)>,
    opentrack_smoothing: f64,
    opentrack_angle_deadzone: f64,
    opentrack_rotation_comp: [[f64; 3]; 3],
    opentrack_translation_scale: [f64; 3],
    opentrack_angle_translation_comp: [[f64; 3]; 3],
    opentrack_angle_translation_comp_scale: f64,
    opentrack_angle_translation_deadzone: f64,
    opentrack_coupling_mode: CouplingMode,
}

#[derive(Clone, Copy)]
enum AngleSource {
    Gaze,
    Head,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CouplingMode {
    Rotation,
    Translation,
    Hybrid,
    Auto,
}

#[derive(Clone, Copy)]
struct AngleComponent {
    component: usize,
    sign: f64,
}

impl AngleComponent {
    const fn new(component: usize, sign: f64) -> Self {
        Self { component, sign }
    }

    const fn disabled() -> Self {
        Self {
            component: 0,
            sign: 0.0,
        }
    }

    fn is_disabled(self) -> bool {
        self.sign == 0.0
    }
}

impl Options {
    fn parse() -> Result<Self> {
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
        let mut opentrack_angle_source = AngleSource::Gaze;
        let mut opentrack_angle_occ = Some(DEFAULT_OPENTRACK_ANGLE_OCC);
        let mut opentrack_angle_scale = DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE;
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
                            AngleSource::Head => DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE,
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
        })
    }

    fn opentrack_target(&self) -> Option<(String, u16)> {
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

fn print_usage() {
    println!(
        "usage:\n  cargo run -- [init_packets_ep.txt] [--log tobii_stream.bin] [--decoded-csv decoded.csv] [--jsonl frames.jsonl] [--print-decoded] [--dashboard] [--opentrack-host 127.0.0.1] [--opentrack-port 4242] [--opentrack-no-translation] [--opentrack-translation-scale X,Y,Z] [--opentrack-angle-source gaze|head] [--opentrack-coupling-mode rotation|translation|hybrid|auto] [--opentrack-angle-points A,B] [--opentrack-angle-translation-comp X,Y,Z] [--opentrack-angle-translation-comp-scale N] [--opentrack-angle-translation-deadzone CM] [--opentrack-angle-occ N] [--opentrack-angle-map x,-y,off] [--opentrack-angle-scale N|YAW,PITCH,ROLL] [--opentrack-origin-samples N] [--opentrack-smoothing 0.35] [--opentrack-angle-deadzone 0.35] [--opentrack-rotation-comp X,Y,Z] [--opentrack-roll-points A,B] [--opentrack-no-angles] [--max-stream-packets N] [--max-init-packets N] [--no-reconnect]\n  cargo run -- analyze-log tobii_stream.bin\n  cargo run -- compare-logs [label:]path.bin [label:]path.bin ...\n  cargo run -- decode-stream tobii_stream.bin\n  cargo run -- import-tsv tshark.tsv out.bin\n  cargo run -- pose-candidates [label:]path.bin [label:]path.bin ...\n  cargo run -- compare-decoded [label:]path.bin [label:]path.bin ...\n  cargo run -- extract-calibration init_packets_ep.txt [--json calibration.json]"
    );
}

fn parse_three_f64(s: &str, option: &str) -> Result<[f64; 3]> {
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

fn parse_angle_translation_comp(s: &str) -> Result<[[f64; 3]; 3]> {
    parse_matrix3(s, "--opentrack-angle-translation-comp")
}

fn parse_matrix3(s: &str, option: &str) -> Result<[[f64; 3]; 3]> {
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

fn parse_point_pair(s: &str, option: &str) -> Result<(usize, usize)> {
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

fn parse_angle_scale(s: &str) -> Result<[f64; 3]> {
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

fn parse_angle_source(s: &str) -> Result<AngleSource> {
    match s.trim() {
        "gaze" => Ok(AngleSource::Gaze),
        "head" => Ok(AngleSource::Head),
        other => anyhow::bail!("unknown --opentrack-angle-source: {other}"),
    }
}

fn parse_coupling_mode(s: &str) -> Result<CouplingMode> {
    match s.trim() {
        "rotation" => Ok(CouplingMode::Rotation),
        "translation" => Ok(CouplingMode::Translation),
        "hybrid" => Ok(CouplingMode::Hybrid),
        "auto" => Ok(CouplingMode::Auto),
        other => anyhow::bail!("unknown --opentrack-coupling-mode: {other}"),
    }
}

fn parse_nonzero_scale(s: &str) -> Result<f64> {
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

fn parse_angle_map(s: &str) -> Result<[AngleComponent; 3]> {
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

fn parse_angle_component(s: &str) -> Result<AngleComponent> {
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

fn parse_log_input(arg: &str) -> Result<LogInput> {
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

fn open_tobii(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
    let devices = ctx.devices()?;
    let mut found = None;

    for dev in devices.iter() {
        let desc = dev.device_descriptor()?;

        if desc.vendor_id() == VID && desc.product_id() == PID {
            println!(
                "Found Tobii bus={} addr={}",
                dev.bus_number(),
                dev.address()
            );
            found = Some(dev);
            break;
        }
    }

    let dev = found.context("Tobii 2104:0313 not found by libusb")?;
    let h = dev
        .open()
        .context("failed to open Tobii device; try sudo")?;

    if let Err(e) = h.set_auto_detach_kernel_driver(true) {
        println!("auto detach kernel driver not available: {:?}", e);
    }

    match h.set_active_configuration(1) {
        Ok(()) => println!("Set active configuration 1"),
        Err(rusb::Error::Busy) => println!("Configuration 1 already active or busy"),
        Err(e) => println!("set_active_configuration(1) failed: {:?}", e),
    }

    if h.kernel_driver_active(IFACE).unwrap_or(false) {
        println!("Detaching kernel driver from interface {}", IFACE);
        let _ = h.detach_kernel_driver(IFACE);
    }

    h.claim_interface(IFACE)
        .context("failed to claim interface 0")?;

    println!("Claimed interface {}", IFACE);
    Ok(h)
}

fn read_stream(
    ctx: &UsbContext,
    mut h: rusb::DeviceHandle<UsbContext>,
    opts: &Options,
    log: &mut Option<PacketLog>,
    live_csv: &mut Option<DecodedCsv>,
    jsonl: &mut Option<JsonlOutput>,
    opentrack: &mut Option<OpentrackUdp>,
) -> Result<()> {
    let mut buf = [0u8; 8192];
    let mut read_count = 0u64;
    let mut reconnect_attempts = 0u32;
    let mut startup_timeouts = 0u32;

    loop {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(2000)) {
            Ok(n) => {
                let data = &buf[..n];
                read_count += 1;
                reconnect_attempts = 0;
                startup_timeouts = 0;

                log_packet(log, EP_IN, data)?;
                handle_live_decoded(
                    read_count,
                    data,
                    live_csv,
                    jsonl,
                    opentrack,
                    opts.print_decoded,
                    opts.dashboard,
                )?;

                if !opts.dashboard {
                    println!(
                        "STREAM/IN #{} len={} marker={:?} seq={:?} declared_len={:?}",
                        read_count,
                        data.len(),
                        marker(data),
                        seq(data),
                        declared_len(data)
                    );
                }

                if let Some(max) = opts.max_stream_packets {
                    if read_count >= max {
                        println!("Reached --max-stream-packets={max}");
                        return Ok(());
                    }
                }
            }
            Err(rusb::Error::Timeout) => {
                if read_count == 0 {
                    startup_timeouts += 1;
                    if startup_timeouts >= STARTUP_STREAM_TIMEOUTS {
                        anyhow::bail!(StreamStartupTimeout);
                    }
                }

                if opts.dashboard {
                    render_dashboard_status("timeout waiting for stream packet")?;
                } else {
                    println!("timeout");
                }
            }
            Err(rusb::Error::NoDevice) if opts.reconnect => {
                reconnect_attempts += 1;
                println!(
                    "read error: NoDevice; USB device probably re-enumerated, reconnect attempt {}",
                    reconnect_attempts
                );
                h = wait_for_reconnect(ctx)?;
            }
            Err(rusb::Error::NoDevice) => {
                anyhow::bail!("read error: NoDevice");
            }
            Err(e) => {
                if opts.dashboard {
                    render_dashboard_status(&format!("read error: {e:?}"))?;
                    continue;
                }
                println!("read error: {:?}", e);
            }
        }
    }
}

fn wait_for_reconnect(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
    for attempt in 1..=30 {
        thread::sleep(Duration::from_millis(500));
        match open_tobii(ctx) {
            Ok(h) => {
                println!("Reconnected on attempt {attempt}");
                return Ok(h);
            }
            Err(e) => {
                println!("  reconnect attempt {attempt} failed: {e}");
            }
        }
    }

    anyhow::bail!("Tobii did not reappear after 15 seconds")
}

fn vendor_control_init(h: &mut rusb::DeviceHandle<UsbContext>) -> Result<()> {
    let timeout = Duration::from_millis(2000);

    let init_data: [u8; 24] = [
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let n = h.write_control(0x41, 48, 0, 0, &init_data, timeout)?;
    println!("control OUT 48 written={}", n);

    let mut buf = [0u8; 512];
    let n = h.read_control(0xC1, 48, 0, 0, &mut buf, timeout)?;
    println!("control IN 48 read={}", n);

    let mut buf8 = [0u8; 8];
    let n = h.read_control(0xC1, 70, 1, 0, &mut buf8, timeout)?;
    println!("control IN 70 read={} data={:02x?}", n, &buf8[..n]);

    let n = h.write_control(0x41, 65, 0, 0, &[], timeout)?;
    println!("control OUT 65 written={}", n);

    Ok(())
}

fn wait_for_response_seq(
    h: &mut rusb::DeviceHandle<UsbContext>,
    expected_seq: u32,
    log: &mut Option<PacketLog>,
) -> Result<()> {
    let mut buf = [0u8; 8192];

    loop {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(2000)) {
            Ok(n) => {
                let data = &buf[..n];
                log_packet(log, EP_IN, data)?;

                println!(
                    "  IN len={} marker={:?} seq={:?} declared_len={:?}",
                    n,
                    marker(data),
                    seq(data),
                    declared_len(data)
                );

                if marker(data) == Some(0x52) && seq(data) == Some(expected_seq) {
                    return Ok(());
                }
            }
            Err(e) => {
                anyhow::bail!("waiting response seq {} failed: {:?}", expected_seq, e);
            }
        }
    }
}

fn drain_in_limited(
    h: &mut rusb::DeviceHandle<UsbContext>,
    max_reads: usize,
    log: &mut Option<PacketLog>,
    live_csv: &mut Option<DecodedCsv>,
    jsonl: &mut Option<JsonlOutput>,
    opentrack: &mut Option<OpentrackUdp>,
    print_decoded: bool,
) {
    let mut buf = [0u8; 8192];

    for _ in 0..max_reads {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(100)) {
            Ok(n) => {
                let data = &buf[..n];
                if let Err(e) = log_packet(log, EP_IN, data) {
                    println!("  log error: {e}");
                }
                if let Err(e) =
                    handle_live_decoded(0, data, live_csv, jsonl, opentrack, print_decoded, false)
                {
                    println!("  decoded csv error: {e}");
                }

                println!(
                    "  IN len={} marker={:?} seq={:?} declared_len={:?}",
                    n,
                    marker(data),
                    seq(data),
                    declared_len(data)
                );
            }
            Err(rusb::Error::Timeout) => {
                break;
            }
            Err(e) => {
                println!("  IN error: {:?}", e);
                break;
            }
        }
    }
}

struct PacketLog {
    out: BufWriter<File>,
}

impl PacketLog {
    fn create(path: &str) -> Result<Self> {
        let mut out = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create log {path}"))?,
        );
        out.write_all(LOG_MAGIC)?;
        println!("Logging IN packets to {path}");
        Ok(Self { out })
    }

    fn write_record(&mut self, ep: u8, data: &[u8]) -> Result<()> {
        self.write_record_at(now_us(), ep, data)
    }

    fn write_record_at(&mut self, ts_us: u64, ep: u8, data: &[u8]) -> Result<()> {
        self.out.write_all(&[0, ep, 0, 0])?;
        self.out.write_all(&ts_us.to_le_bytes())?;
        self.out.write_all(&(data.len() as u32).to_le_bytes())?;
        self.out.write_all(data)?;
        self.out.flush()?;
        Ok(())
    }
}

struct DecodedCsv {
    out: BufWriter<File>,
}

impl DecodedCsv {
    fn create(path: &str) -> Result<Self> {
        let mut out = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create decoded CSV {path}"))?,
        );

        write!(out, "ts_us,packet")?;
        for field in DERIVED_FIELDS {
            write!(out, ",{field}")?;
        }
        for field in LIVE_FIELDS {
            write!(out, ",{}", field.name)?;
        }
        writeln!(out)?;

        println!("Logging decoded stream candidates to {path}");
        Ok(Self { out })
    }

    fn write_packet(
        &mut self,
        packet_no: u64,
        values: &BTreeMap<(u32, usize, usize), f64>,
    ) -> Result<()> {
        let derived = derive_live_values(values);
        write!(self.out, "{},{}", now_us(), packet_no)?;
        for value in derived {
            write_csv_value(&mut self.out, value)?;
        }
        for field in LIVE_FIELDS {
            write_csv_value(&mut self.out, values.get(&field.key()).copied())?;
        }
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

struct JsonlOutput {
    out: BufWriter<Box<dyn Write>>,
}

impl JsonlOutput {
    fn create(path: &str) -> Result<Self> {
        let writer: Box<dyn Write> = if path == "-" {
            println!("Logging tracking frames as JSON Lines to stdout");
            Box::new(io::stdout())
        } else {
            println!("Logging tracking frames as JSON Lines to {path}");
            Box::new(
                File::create(path)
                    .with_context(|| format!("failed to create JSONL output {path}"))?,
            )
        };

        Ok(Self {
            out: BufWriter::new(writer),
        })
    }

    fn write_frame(&mut self, frame: &TrackingFrame) -> Result<()> {
        frame.write_json(&mut self.out)?;
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

struct OpentrackUdp {
    socket: UdpSocket,
    target: SocketAddr,
    send_translation: bool,
    angle_source: AngleSource,
    angle_occurrence: Option<usize>,
    angle_scale: [f64; 3],
    angle_map: [AngleComponent; 3],
    origin_samples: usize,
    angle_points: Option<(usize, usize)>,
    roll_points: Option<(usize, usize)>,
    origin: Option<[f64; 3]>,
    origin_sum: [f64; 3],
    origin_count: usize,
    angle_origin: Option<[f64; 3]>,
    angle_origin_sum: [f64; 3],
    angle_origin_count: usize,
    roll_origin: Option<f64>,
    roll_origin_sum: f64,
    roll_origin_count: usize,
    smoothing_alpha: f64,
    angle_deadzone: f64,
    rotation_comp: [[f64; 3]; 3],
    translation_scale: [f64; 3],
    angle_translation_comp: [[f64; 3]; 3],
    angle_translation_comp_scale: f64,
    angle_translation_deadzone: f64,
    coupling_mode: CouplingMode,
    coupling_state: CouplingMode,
    coupling_candidate: CouplingMode,
    coupling_candidate_count: usize,
    last_pose: Option<[f64; 6]>,
}

impl OpentrackUdp {
    fn connect(
        host: &str,
        port: u16,
        send_translation: bool,
        angle_source: AngleSource,
        angle_occurrence: Option<usize>,
        angle_scale: [f64; 3],
        angle_map: [AngleComponent; 3],
        origin_samples: usize,
        angle_points: Option<(usize, usize)>,
        roll_points: Option<(usize, usize)>,
        smoothing_alpha: f64,
        angle_deadzone: f64,
        rotation_comp: [[f64; 3]; 3],
        translation_scale: [f64; 3],
        angle_translation_comp: [[f64; 3]; 3],
        angle_translation_comp_scale: f64,
        angle_translation_deadzone: f64,
        coupling_mode: CouplingMode,
    ) -> Result<Self> {
        let target = (host, port)
            .to_socket_addrs()
            .with_context(|| format!("failed to resolve opentrack target {host}:{port}"))?
            .next()
            .with_context(|| format!("opentrack target {host}:{port} resolved to no addresses"))?;
        let bind_addr = if target.is_ipv4() {
            "0.0.0.0:0"
        } else {
            "[::]:0"
        };
        let socket = UdpSocket::bind(bind_addr).context("failed to create opentrack UDP socket")?;

        println!("Sending OpenTrack UDP frames to {target}");

        Ok(Self {
            socket,
            target,
            send_translation,
            angle_source,
            angle_occurrence,
            angle_scale,
            angle_map,
            origin_samples,
            angle_points,
            roll_points,
            origin: None,
            origin_sum: [0.0; 3],
            origin_count: 0,
            angle_origin: None,
            angle_origin_sum: [0.0; 3],
            angle_origin_count: 0,
            roll_origin: None,
            roll_origin_sum: 0.0,
            roll_origin_count: 0,
            smoothing_alpha,
            angle_deadzone,
            rotation_comp,
            translation_scale,
            angle_translation_comp,
            angle_translation_comp_scale,
            angle_translation_deadzone,
            coupling_mode,
            coupling_state: CouplingMode::Rotation,
            coupling_candidate: CouplingMode::Rotation,
            coupling_candidate_count: 0,
            last_pose: None,
        })
    }

    fn send_frame(
        &mut self,
        frame: &TrackingFrame,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
    ) -> Result<Option<[f64; 6]>> {
        let Some(head) = frame.head_xyz() else {
            return Ok(None);
        };
        let has_user = frame.gaze_valid;
        let origin = self.calibrate_translation_origin(head, has_user);
        let [yaw_deg, pitch_deg, mut roll_deg] = self.relative_angles(decoded, has_user);

        let raw_translation = if self.send_translation {
            origin
                .map(|origin| {
                    [
                        (head[0] - origin[0]) / OPENTRACK_HEAD_SCALE * self.translation_scale[0],
                        (head[1] - origin[1]) / OPENTRACK_HEAD_SCALE * self.translation_scale[1],
                        (head[2] - origin[2]) / OPENTRACK_HEAD_SCALE * self.translation_scale[2],
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0])
        } else {
            [0.0, 0.0, 0.0]
        };

        if let Some(roll) = self.relative_roll_from_points(decoded, has_user) {
            roll_deg = roll;
        }

        let raw_angles = [yaw_deg, pitch_deg, roll_deg];
        let (translation, angles) = if self.send_translation {
            let angle_comp = scale_matrix(
                self.angle_translation_comp,
                self.angle_translation_comp_scale,
            );
            match self.coupling_mode {
                CouplingMode::Rotation => (
                    [
                        raw_translation[0] - dot3(self.rotation_comp[0], raw_angles),
                        raw_translation[1] - dot3(self.rotation_comp[1], raw_angles),
                        raw_translation[2] - dot3(self.rotation_comp[2], raw_angles),
                    ],
                    raw_angles,
                ),
                CouplingMode::Translation => (
                    raw_translation,
                    angles_from_translation(raw_angles, raw_translation, angle_comp),
                ),
                CouplingMode::Hybrid => {
                    self.choose_pose_hypothesis_hybrid(raw_translation, raw_angles, angle_comp)
                }
                CouplingMode::Auto => choose_pose_hypothesis(
                    raw_translation,
                    raw_angles,
                    self.rotation_comp,
                    angle_comp,
                    self.angle_translation_deadzone,
                ),
            }
        } else {
            ([0.0, 0.0, 0.0], raw_angles)
        };
        let [x_cm, y_cm, z_cm] = translation;
        let [yaw_deg, pitch_deg, roll_deg] = [
            angles[0].clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            angles[1].clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            angles[2].clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        ];

        // OpenTrack "UDP over network" consumes pose axes in the same order
        // as its plugin API: TX, TY, TZ, Yaw, Pitch, Roll.
        let values = self.filter_pose([x_cm, y_cm, z_cm, yaw_deg, pitch_deg, roll_deg]);
        let mut packet = [0u8; 48];
        for (i, value) in values.iter().enumerate() {
            packet[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }

        self.socket
            .send_to(&packet, self.target)
            .context("failed to send OpenTrack UDP packet")?;

        Ok(Some(values))
    }

    fn choose_pose_hypothesis_hybrid(
        &mut self,
        raw_translation: [f64; 3],
        raw_angles: [f64; 3],
        angle_comp: [[f64; 3]; 3],
    ) -> ([f64; 3], [f64; 3]) {
        let rotation_pose = pose_for_coupling_mode(
            CouplingMode::Rotation,
            raw_translation,
            raw_angles,
            self.rotation_comp,
            angle_comp,
        );
        let translation_pose = pose_for_coupling_mode(
            CouplingMode::Translation,
            raw_translation,
            raw_angles,
            self.rotation_comp,
            angle_comp,
        );
        let rotation_score = norm3(rotation_pose.0) / (norm3(raw_translation) + 1.0);
        let translation_score = norm3(translation_pose.1) / (norm3(raw_angles) + 1.0);
        let desired =
            if dot3(raw_translation, raw_translation).sqrt() <= self.angle_translation_deadzone {
                CouplingMode::Rotation
            } else if translation_score + OPENTRACK_COUPLING_SWITCH_MARGIN < rotation_score {
                CouplingMode::Translation
            } else if rotation_score + OPENTRACK_COUPLING_SWITCH_MARGIN < translation_score {
                CouplingMode::Rotation
            } else {
                self.coupling_state
            };

        if desired == self.coupling_state {
            self.coupling_candidate = desired;
            self.coupling_candidate_count = 0;
        } else if desired == self.coupling_candidate {
            self.coupling_candidate_count += 1;
            if self.coupling_candidate_count >= OPENTRACK_COUPLING_SWITCH_FRAMES {
                self.coupling_state = desired;
                self.coupling_candidate_count = 0;
            }
        } else {
            self.coupling_candidate = desired;
            self.coupling_candidate_count = 1;
        }

        match self.coupling_state {
            CouplingMode::Translation => translation_pose,
            _ => rotation_pose,
        }
    }

    fn filter_pose(&mut self, mut values: [f64; 6]) -> [f64; 6] {
        for value in &mut values[3..] {
            if value.abs() < self.angle_deadzone {
                *value = 0.0;
            }
        }

        let Some(previous) = self.last_pose else {
            self.last_pose = Some(values);
            return values;
        };

        let mut filtered = values;
        for i in 0..6 {
            filtered[i] = previous[i] + self.smoothing_alpha * (values[i] - previous[i]);
        }
        self.last_pose = Some(filtered);
        filtered
    }

    fn calibrate_translation_origin(
        &mut self,
        head: [f64; 3],
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        calibrate_origin(
            &mut self.origin,
            &mut self.origin_sum,
            &mut self.origin_count,
            self.origin_samples,
            head,
            can_calibrate,
        )
    }

    fn relative_angles(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> [f64; 3] {
        if let Some(angles) = self.relative_angles_from_points(decoded, can_calibrate) {
            return angles;
        }

        if let AngleSource::Gaze = self.angle_source {
            if let Some(angles) = self.relative_angles_from_gaze(decoded, can_calibrate) {
                return angles;
            }
        };

        let Some(occurrence) = self.angle_occurrence else {
            return [0.0, 0.0, 0.0];
        };
        let mut angles = [0.0; 3];
        for (out_axis, source) in self.angle_map.iter().enumerate() {
            if source.is_disabled() {
                continue;
            }

            let Some(value) = field_value(
                decoded,
                LiveField::new("", 0x00031f41, occurrence, source.component),
            ) else {
                return [0.0, 0.0, 0.0];
            };
            angles[out_axis] = source.sign * value;
        }

        let Some(origin) = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            angles,
            can_calibrate,
        ) else {
            return [0.0, 0.0, 0.0];
        };

        [
            ((angles[0] - origin[0]) / self.angle_scale[0])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            ((angles[1] - origin[1]) / self.angle_scale[1])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            ((angles[2] - origin[2]) / self.angle_scale[2])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        ]
    }

    fn relative_angles_from_gaze(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        let yaw = mean_keys(
            decoded,
            &[
                LiveField::new("", 0x00021f40, 0, 0),
                LiveField::new("", 0x00021f40, 5, 0),
            ],
        )?;
        let pitch = mean_keys(
            decoded,
            &[
                LiveField::new("", 0x00021f40, 0, 1),
                LiveField::new("", 0x00021f40, 5, 1),
            ],
        )?;
        let angles = [yaw, -pitch, 0.0];

        let origin = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            angles,
            can_calibrate,
        )?;

        Some([
            ((angles[0] - origin[0]) / self.angle_scale[0])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            ((angles[1] - origin[1]) / self.angle_scale[1])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            0.0,
        ])
    }

    fn relative_angles_from_points(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        let (a, b) = self.angle_points?;
        let pa = head_point(decoded, a)?;
        let pb = head_point(decoded, b)?;
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let horizontal = (dx * dx + dz * dz).sqrt();
        if horizontal <= f64::EPSILON {
            return None;
        }

        let angles = [
            dx.atan2(dz).to_degrees(),
            (-dy).atan2(horizontal).to_degrees(),
            0.0,
        ];

        let origin = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            angles,
            can_calibrate,
        )?;

        Some([
            normalize_angle_deg(angles[0] - origin[0])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            normalize_angle_deg(angles[1] - origin[1])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            0.0,
        ])
    }

    fn relative_roll_from_points(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<f64> {
        let (a, b) = self.roll_points?;
        let pa = head_point(decoded, a)?;
        let pb = head_point(decoded, b)?;
        let roll = (pb[1] - pa[1]).atan2(pb[0] - pa[0]).to_degrees();
        let origin = calibrate_scalar_origin(
            &mut self.roll_origin,
            &mut self.roll_origin_sum,
            &mut self.roll_origin_count,
            self.origin_samples,
            roll,
            can_calibrate,
        )?;

        Some(
            normalize_angle_deg(roll - origin)
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        )
    }
}

fn calibrate_origin(
    origin: &mut Option<[f64; 3]>,
    sum: &mut [f64; 3],
    count: &mut usize,
    samples: usize,
    value: [f64; 3],
    can_calibrate: bool,
) -> Option<[f64; 3]> {
    if let Some(origin) = *origin {
        return Some(origin);
    }

    if !can_calibrate {
        return None;
    }

    for i in 0..3 {
        sum[i] += value[i];
    }
    *count += 1;

    if *count < samples {
        return None;
    }

    let calibrated = [
        sum[0] / *count as f64,
        sum[1] / *count as f64,
        sum[2] / *count as f64,
    ];
    *origin = Some(calibrated);
    Some(calibrated)
}

fn calibrate_scalar_origin(
    origin: &mut Option<f64>,
    sum: &mut f64,
    count: &mut usize,
    samples: usize,
    value: f64,
    can_calibrate: bool,
) -> Option<f64> {
    if let Some(origin) = *origin {
        return Some(origin);
    }

    if !can_calibrate {
        return None;
    }

    *sum += value;
    *count += 1;

    if *count < samples {
        return None;
    }

    let calibrated = *sum / *count as f64;
    *origin = Some(calibrated);
    Some(calibrated)
}

fn head_point(values: &BTreeMap<(u32, usize, usize), f64>, occurrence: usize) -> Option<[f64; 3]> {
    Some([
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 0))?,
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 1))?,
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 2))?,
    ])
}

fn normalize_angle_deg(mut angle: f64) -> f64 {
    while angle > 180.0 {
        angle -= 360.0;
    }
    while angle < -180.0 {
        angle += 360.0;
    }
    angle
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

fn scale_matrix(mut matrix: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    for row in &mut matrix {
        for value in row {
            *value *= scale;
        }
    }
    matrix
}

fn pose_for_coupling_mode(
    mode: CouplingMode,
    raw_translation: [f64; 3],
    raw_angles: [f64; 3],
    rotation_comp: [[f64; 3]; 3],
    angle_translation_comp: [[f64; 3]; 3],
) -> ([f64; 3], [f64; 3]) {
    match mode {
        CouplingMode::Translation => (
            raw_translation,
            angles_from_translation(raw_angles, raw_translation, angle_translation_comp),
        ),
        _ => (
            [
                raw_translation[0] - dot3(rotation_comp[0], raw_angles),
                raw_translation[1] - dot3(rotation_comp[1], raw_angles),
                raw_translation[2] - dot3(rotation_comp[2], raw_angles),
            ],
            raw_angles,
        ),
    }
}

fn choose_pose_hypothesis(
    raw_translation: [f64; 3],
    raw_angles: [f64; 3],
    rotation_comp: [[f64; 3]; 3],
    angle_translation_comp: [[f64; 3]; 3],
    translation_deadzone: f64,
) -> ([f64; 3], [f64; 3]) {
    let rotation_translation = [
        raw_translation[0] - dot3(rotation_comp[0], raw_angles),
        raw_translation[1] - dot3(rotation_comp[1], raw_angles),
        raw_translation[2] - dot3(rotation_comp[2], raw_angles),
    ];
    let translation_angles =
        angles_from_translation(raw_angles, raw_translation, angle_translation_comp);

    let raw_translation_len = dot3(raw_translation, raw_translation).sqrt();
    let rotation_translation_len = dot3(rotation_translation, rotation_translation).sqrt();

    if rotation_translation_len <= translation_deadzone
        || rotation_translation_len <= raw_translation_len * 0.45
    {
        (rotation_translation, raw_angles)
    } else if raw_translation_len > translation_deadzone {
        (raw_translation, translation_angles)
    } else {
        (rotation_translation, raw_angles)
    }
}

fn angles_from_translation(
    raw_angles: [f64; 3],
    translation: [f64; 3],
    angle_translation_comp: [[f64; 3]; 3],
) -> [f64; 3] {
    [
        raw_angles[0] - dot3(angle_translation_comp[0], translation),
        raw_angles[1] - dot3(angle_translation_comp[1], translation),
        raw_angles[2] - dot3(angle_translation_comp[2], translation),
    ]
}

fn write_csv_value(out: &mut BufWriter<File>, value: Option<f64>) -> Result<()> {
    match value {
        Some(value) => write!(out, ",{value:.6}")?,
        None => write!(out, ",")?,
    }
    Ok(())
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn log_packet(log: &mut Option<PacketLog>, ep: u8, data: &[u8]) -> Result<()> {
    if let Some(log) = log {
        log.write_record(ep, data)?;
    }
    Ok(())
}

fn handle_live_decoded(
    packet_no: u64,
    data: &[u8],
    live_csv: &mut Option<DecodedCsv>,
    jsonl: &mut Option<JsonlOutput>,
    opentrack: &mut Option<OpentrackUdp>,
    print_decoded: bool,
    dashboard: bool,
) -> Result<()> {
    if marker(data) != Some(0x53) {
        return Ok(());
    }

    let decoded = decode_stream_payload(data)?;
    if decoded.is_empty() {
        return Ok(());
    }

    let frame = TrackingFrame::from_decoded(packet_no, &decoded);

    if let Some(csv) = live_csv {
        csv.write_packet(packet_no, &decoded)?;
    }

    if let Some(jsonl) = jsonl {
        jsonl.write_frame(&frame)?;
    }

    let mut opentrack_pose = None;
    if let Some(opentrack) = opentrack {
        opentrack_pose = opentrack.send_frame(&frame, &decoded)?;
    }

    if print_decoded {
        print_live_decoded(&frame);
    }

    if dashboard {
        render_tracking_dashboard(&frame, opentrack_pose)?;
    }

    Ok(())
}

fn print_live_decoded(frame: &TrackingFrame) {
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

fn render_tracking_dashboard(
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

fn render_dashboard_status(status: &str) -> Result<()> {
    print!("\x1b[2J\x1b[H");
    println!("Tobii Eye Tracker 5 - TrackingFrame");
    println!("==============================================================");
    println!("{status}");
    io::stdout().flush()?;
    Ok(())
}

fn dashboard_pair(label: &str, x: Option<f64>, y: Option<f64>) {
    println!(
        "{label:<8} x={:>12} y={:>12}",
        fmt_frame_value(x),
        fmt_frame_value(y)
    );
}

fn dashboard_bar(label: &str, value: Option<f64>) {
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

fn fmt_frame_value(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_plain_value(value: f64) -> String {
    format!("{value:.3}")
}

#[derive(Clone, Debug)]
struct TrackingFrame {
    ts_us: u64,
    packet: u64,
    gaze_valid: bool,
    gaze_x: Option<f64>,
    gaze_y: Option<f64>,
    gaze_norm_x: Option<f64>,
    gaze_norm_y: Option<f64>,
    left_eye_x: Option<f64>,
    left_eye_y: Option<f64>,
    left_eye_norm_x: Option<f64>,
    left_eye_norm_y: Option<f64>,
    right_eye_x: Option<f64>,
    right_eye_y: Option<f64>,
    right_eye_norm_x: Option<f64>,
    right_eye_norm_y: Option<f64>,
    head_x: Option<f64>,
    head_y: Option<f64>,
    head_z: Option<f64>,
    head_yaw: Option<f64>,
    head_pitch: Option<f64>,
    head_roll: Option<f64>,
}

impl TrackingFrame {
    fn from_decoded(packet: u64, values: &BTreeMap<(u32, usize, usize), f64>) -> Self {
        let derived = derive_live_values(values);

        Self {
            ts_us: now_us(),
            packet,
            gaze_valid: derived[0].is_some_and(|value| value >= 0.5),
            gaze_x: derived[1],
            gaze_y: derived[2],
            gaze_norm_x: derived[3],
            gaze_norm_y: derived[4],
            left_eye_x: derived[5],
            left_eye_y: derived[6],
            left_eye_norm_x: derived[7],
            left_eye_norm_y: derived[8],
            right_eye_x: derived[9],
            right_eye_y: derived[10],
            right_eye_norm_x: derived[11],
            right_eye_norm_y: derived[12],
            head_x: derived[13],
            head_y: derived[14],
            head_z: derived[15],
            head_yaw: derived[16],
            head_pitch: derived[17],
            head_roll: derived[18],
        }
    }

    fn write_json<W: Write>(&self, out: &mut W) -> Result<()> {
        write!(
            out,
            "{{\"ts_us\":{},\"packet\":{},\"gaze_valid\":{},",
            self.ts_us, self.packet, self.gaze_valid
        )?;
        write!(out, "\"gaze\":{{")?;
        write_json_number(out, "x", self.gaze_x, true)?;
        write_json_number(out, "y", self.gaze_y, false)?;
        write_json_number(out, "norm_x", self.gaze_norm_x, false)?;
        write_json_number(out, "norm_y", self.gaze_norm_y, false)?;
        write!(out, "}},\"left_eye\":{{")?;
        write_json_number(out, "x", self.left_eye_x, true)?;
        write_json_number(out, "y", self.left_eye_y, false)?;
        write_json_number(out, "norm_x", self.left_eye_norm_x, false)?;
        write_json_number(out, "norm_y", self.left_eye_norm_y, false)?;
        write!(out, "}},\"right_eye\":{{")?;
        write_json_number(out, "x", self.right_eye_x, true)?;
        write_json_number(out, "y", self.right_eye_y, false)?;
        write_json_number(out, "norm_x", self.right_eye_norm_x, false)?;
        write_json_number(out, "norm_y", self.right_eye_norm_y, false)?;
        write!(out, "}},\"head\":{{")?;
        write_json_number(out, "x", self.head_x, true)?;
        write_json_number(out, "y", self.head_y, false)?;
        write_json_number(out, "z", self.head_z, false)?;
        write_json_number(out, "yaw", self.head_yaw, false)?;
        write_json_number(out, "pitch", self.head_pitch, false)?;
        write_json_number(out, "roll", self.head_roll, false)?;
        write!(out, "}}}}")?;
        Ok(())
    }

    fn head_xyz(&self) -> Option<[f64; 3]> {
        Some([self.head_x?, self.head_y?, self.head_z?])
    }
}

fn write_json_number<W: Write>(
    out: &mut W,
    name: &str,
    value: Option<f64>,
    first: bool,
) -> Result<()> {
    if !first {
        write!(out, ",")?;
    }

    write!(out, "\"{name}\":")?;
    match value {
        Some(value) if value.is_finite() => write!(out, "{value:.6}")?,
        _ => write!(out, "null")?,
    }
    Ok(())
}

fn derive_live_values(values: &BTreeMap<(u32, usize, usize), f64>) -> [Option<f64>; 19] {
    let left_eye_x = field_value(values, LiveField::new("", 0x00021f40, 1, 0));
    let left_eye_y = field_value(values, LiveField::new("", 0x00021f40, 1, 1));
    let right_eye_x = field_value(values, LiveField::new("", 0x00021f40, 3, 0));
    let right_eye_y = field_value(values, LiveField::new("", 0x00021f40, 3, 1));
    let gaze_x = mean_keys(
        values,
        &[
            LiveField::new("", 0x00021f40, 1, 0),
            LiveField::new("", 0x00021f40, 3, 0),
        ],
    );
    let gaze_y = mean_keys(
        values,
        &[
            LiveField::new("", 0x00021f40, 1, 1),
            LiveField::new("", 0x00021f40, 3, 1),
        ],
    );
    let head_x = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 5, 0),
            LiveField::new("", 0x00031f41, 6, 0),
            LiveField::new("", 0x00031f41, 9, 0),
        ],
    );
    let head_y = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 5, 1),
            LiveField::new("", 0x00031f41, 6, 1),
            LiveField::new("", 0x00031f41, 9, 1),
        ],
    );
    let head_z = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 5, 2),
            LiveField::new("", 0x00031f41, 6, 2),
            LiveField::new("", 0x00031f41, 9, 2),
        ],
    );
    let head_yaw = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 2, 0),
            LiveField::new("", 0x00031f41, 10, 0),
        ],
    );
    let head_pitch = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 2, 1),
            LiveField::new("", 0x00031f41, 10, 1),
        ],
    );
    let head_roll = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 2, 2),
            LiveField::new("", 0x00031f41, 10, 2),
        ],
    );
    let gaze_valid = gaze_valid(gaze_x, gaze_y);

    [
        Some(if gaze_valid { 1.0 } else { 0.0 }),
        gaze_x,
        gaze_y,
        norm_gaze(gaze_x),
        norm_gaze(gaze_y),
        left_eye_x,
        left_eye_y,
        norm_gaze(left_eye_x),
        norm_gaze(left_eye_y),
        right_eye_x,
        right_eye_y,
        norm_gaze(right_eye_x),
        norm_gaze(right_eye_y),
        head_x,
        head_y,
        head_z,
        head_yaw,
        head_pitch,
        head_roll,
    ]
}

fn gaze_valid(x: Option<f64>, y: Option<f64>) -> bool {
    let Some(x) = x else {
        return false;
    };
    let Some(y) = y else {
        return false;
    };

    (-0.25 * GAZE_COORD_MAX..=1.25 * GAZE_COORD_MAX).contains(&x)
        && (-0.25 * GAZE_COORD_MAX..=1.25 * GAZE_COORD_MAX).contains(&y)
}

fn norm_gaze(value: Option<f64>) -> Option<f64> {
    value.map(|value| (value / GAZE_COORD_MAX).clamp(0.0, 1.0))
}

fn field_value(values: &BTreeMap<(u32, usize, usize), f64>, field: LiveField) -> Option<f64> {
    let value = values.get(&field.key()).copied()?;
    if value.abs() == 1024.0 || value == 0.0 {
        None
    } else {
        Some(value)
    }
}

fn mean_keys(values: &BTreeMap<(u32, usize, usize), f64>, fields: &[LiveField]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;

    for field in fields {
        let Some(value) = field_value(values, *field) else {
            continue;
        };

        sum += value;
        count += 1;
    }

    (count > 0).then_some(sum / count as f64)
}

fn fmt_live(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_string())
}

fn import_tsv(tsv_path: &str, log_path: &str) -> Result<()> {
    let input = BufReader::new(
        File::open(tsv_path).with_context(|| format!("failed to open TSV {tsv_path}"))?,
    );
    let mut log = PacketLog::create(log_path)?;
    let mut records = 0usize;
    let mut stream_records = 0usize;

    for (line_no, line) in input.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read TSV line {}", line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split('\t');
        let ts_s = parts
            .next()
            .with_context(|| format!("missing timestamp at TSV line {}", line_no + 1))?;
        let ep_s = parts
            .next()
            .with_context(|| format!("missing endpoint at TSV line {}", line_no + 1))?;
        let hex_s = parts
            .next()
            .with_context(|| format!("missing payload hex at TSV line {}", line_no + 1))?;

        let ts_us = parse_timestamp_us(ts_s)
            .with_context(|| format!("bad timestamp at TSV line {}", line_no + 1))?;
        let ep = u8::from_str_radix(ep_s.trim_start_matches("0x"), 16)
            .with_context(|| format!("bad endpoint at TSV line {}", line_no + 1))?;
        let data =
            hex_to_bytes(hex_s).with_context(|| format!("bad hex at TSV line {}", line_no + 1))?;

        if marker(&data) == Some(0x53) {
            stream_records += 1;
        }
        log.write_record_at(ts_us, ep, &data)?;
        records += 1;
    }

    println!("imported {records} records ({stream_records} stream packets) to {log_path}");
    Ok(())
}

fn parse_timestamp_us(s: &str) -> Result<u64> {
    let (seconds, fraction) = s.trim().split_once('.').unwrap_or((s.trim(), ""));
    let seconds: u64 = seconds.parse().context("bad seconds")?;
    let mut micros = String::from(fraction);
    micros.truncate(6);
    while micros.len() < 6 {
        micros.push('0');
    }
    let micros: u64 = if micros.is_empty() {
        0
    } else {
        micros.parse().context("bad fractional seconds")?
    };
    Ok(seconds * 1_000_000 + micros)
}

fn analyze_log(path: &str) -> Result<()> {
    let mut records = 0u64;
    let mut total_bytes = 0u64;
    let mut marker_52 = 0u64;
    let mut marker_53 = 0u64;
    let mut stream_payloads = Vec::new();

    for data in read_log_payloads(path)? {
        records += 1;
        total_bytes += data.len() as u64;

        let m = marker(&data);

        match m {
            Some(0x52) => marker_52 += 1,
            Some(0x53) => {
                marker_53 += 1;
                stream_payloads.push(data.clone());
            }
            _ => {}
        }
    }

    println!("records={records} total_payload_bytes={total_bytes}");
    println!("marker 0x52 responses={marker_52} marker 0x53 stream={marker_53}");

    print_numeric_candidates(&stream_payloads);
    print_stream_changes(&stream_payloads);

    Ok(())
}

fn read_log_payloads(path: &str) -> Result<Vec<Vec<u8>>> {
    let mut input =
        BufReader::new(File::open(path).with_context(|| format!("failed to open log {path}"))?);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == LOG_MAGIC, "bad log magic in {path}");

    let mut payloads = Vec::new();

    loop {
        let mut header = [0u8; 16];
        match input.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("failed to read log record header"),
        }

        let len = u32::from_le_bytes(header[12..16].try_into()?) as usize;
        let mut data = vec![0u8; len];
        input.read_exact(&mut data)?;
        payloads.push(data);
    }

    Ok(payloads)
}

fn main_stream_payloads(path: &str) -> Result<Vec<Vec<u8>>> {
    let stream: Vec<Vec<u8>> = read_log_payloads(path)?
        .into_iter()
        .filter(|payload| marker(payload) == Some(0x53))
        .collect();

    let mut lengths = BTreeMap::<usize, (usize, usize)>::new();
    for payload in &stream {
        let entry = lengths.entry(payload.len()).or_default();
        entry.0 += 1;

        if let Ok((values, false)) = decode_stream_payload_with_status(payload) {
            if !values.is_empty() {
                entry.1 += 1;
            }
        }
    }

    let main_len = lengths
        .into_iter()
        .max_by_key(|(_, (count, decoded))| (*decoded, *count))
        .map(|(len, _)| len)
        .unwrap_or(0);

    Ok(stream
        .into_iter()
        .filter(|payload| payload.len() == main_len)
        .collect())
}

#[derive(Clone)]
struct LogSummary {
    label: String,
    payloads: Vec<Vec<u8>>,
}

#[derive(Clone)]
struct CompareCandidate {
    kind: &'static str,
    offset: usize,
    means: Vec<f64>,
    stddevs: Vec<f64>,
    min: f64,
    max: f64,
    score: f64,
}

fn compare_logs(inputs: &[LogInput]) -> Result<()> {
    let mut logs = Vec::new();

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        anyhow::ensure!(!payloads.is_empty(), "{} has no stream packets", input.path);
        println!(
            "{}: {} main stream packets, len={}",
            input.label,
            payloads.len(),
            payloads[0].len()
        );
        logs.push(LogSummary {
            label: input.label.clone(),
            payloads,
        });
    }

    let min_len = logs
        .iter()
        .flat_map(|log| log.payloads.iter().map(Vec::len))
        .min()
        .unwrap_or(0);

    let mut candidates = Vec::new();

    for offset in 0..min_len {
        push_compare_candidate(
            &logs,
            "u8",
            offset,
            |payload, offset| payload[offset] as f64,
            &mut candidates,
        );
    }

    for offset in 0..min_len.saturating_sub(1) {
        push_compare_candidate(
            &logs,
            "i16",
            offset,
            |payload, offset| {
                i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
        push_compare_candidate(
            &logs,
            "u16",
            offset,
            |payload, offset| {
                u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
    }

    for offset in 0..min_len.saturating_sub(3) {
        push_compare_candidate(
            &logs,
            "i32",
            offset,
            |payload, offset| {
                i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
        push_compare_candidate(
            &logs,
            "u32",
            offset,
            |payload, offset| {
                u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
        push_compare_candidate(
            &logs,
            "f32",
            offset,
            |payload, offset| {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            },
            &mut candidates,
        );
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.cmp(&b.offset))
    });

    print_compare_table("Top cross-log candidates", &logs, &candidates, 32);
    print_pair_table("head_left", "head_right", &logs, &candidates, 24);
    print_pair_table("eyes_left", "eyes_right", &logs, &candidates, 24);

    Ok(())
}

fn push_compare_candidate<F>(
    logs: &[LogSummary],
    kind: &'static str,
    offset: usize,
    read_value: F,
    out: &mut Vec<CompareCandidate>,
) where
    F: Fn(&[u8], usize) -> f64,
{
    let mut means = Vec::new();
    let mut stddevs = Vec::new();
    let mut all_min = f64::INFINITY;
    let mut all_max = f64::NEG_INFINITY;

    for log in logs {
        let values: Vec<f64> = log
            .payloads
            .iter()
            .map(|payload| read_value(payload, offset))
            .collect();

        if values.iter().any(|value| !value.is_finite()) {
            return;
        }

        if kind == "f32" && values.iter().any(|value| value.abs() > 1_000_000.0) {
            return;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        let stddev = variance.sqrt();

        all_min = all_min.min(values.iter().copied().fold(f64::INFINITY, f64::min));
        all_max = all_max.max(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        means.push(mean);
        stddevs.push(stddev);
    }

    let between = means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - means.iter().copied().fold(f64::INFINITY, f64::min);
    let within = stddevs.iter().sum::<f64>() / stddevs.len() as f64;

    if between <= 0.0 {
        return;
    }

    let score = between / (within + 1.0);
    if score < 0.5 {
        return;
    }

    out.push(CompareCandidate {
        kind,
        offset,
        means,
        stddevs,
        min: all_min,
        max: all_max,
        score,
    });
}

fn print_compare_table(
    title: &str,
    logs: &[LogSummary],
    candidates: &[CompareCandidate],
    limit: usize,
) {
    println!("{title}:");
    print!(
        "  {:>3} {:>5} {:>9} {:>12} {:>12}",
        "typ", "off", "score", "min", "max"
    );
    for log in logs {
        print!(" {:>12}", log.label);
    }
    println!();

    for candidate in candidates.iter().take(limit) {
        print!(
            "  {:>3} {:>5} {:>9.3} {:>12.5} {:>12.5}",
            candidate.kind, candidate.offset, candidate.score, candidate.min, candidate.max
        );
        for (mean, stddev) in candidate.means.iter().zip(&candidate.stddevs) {
            print!(" {:>7.2}/{:<4.1}", mean, stddev);
        }
        println!();
    }
}

fn print_pair_table(
    left_label: &str,
    right_label: &str,
    logs: &[LogSummary],
    candidates: &[CompareCandidate],
    limit: usize,
) {
    let Some(left) = logs.iter().position(|log| log.label == left_label) else {
        return;
    };
    let Some(right) = logs.iter().position(|log| log.label == right_label) else {
        return;
    };

    let mut pair_candidates: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let delta = candidate.means[right] - candidate.means[left];
            let noise = (candidate.stddevs[left] + candidate.stddevs[right]) / 2.0 + 1.0;
            (delta.abs() / noise, delta, candidate)
        })
        .filter(|(score, _, _)| *score >= 0.5)
        .collect();

    pair_candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.offset.cmp(&b.2.offset))
    });

    println!("Top {left_label} vs {right_label}:");
    println!(
        "  {:>3} {:>5} {:>9} {:>12} {:>12} {:>12}",
        "typ", "off", "score", left_label, right_label, "delta"
    );

    for (score, delta, candidate) in pair_candidates.into_iter().take(limit) {
        println!(
            "  {:>3} {:>5} {:>9.3} {:>12.5} {:>12.5} {:>12.5}",
            candidate.kind,
            candidate.offset,
            score,
            candidate.means[left],
            candidate.means[right],
            delta
        );
    }
}

#[derive(Clone, Default)]
struct StreamFieldStats {
    count: usize,
    min: f64,
    max: f64,
    sum: f64,
    first: f64,
    last: f64,
    changes: usize,
    prev: Option<f64>,
}

impl StreamFieldStats {
    fn push(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
            self.first = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }

        if let Some(prev) = self.prev {
            if (prev - value).abs() > f64::EPSILON {
                self.changes += 1;
            }
        }

        self.prev = Some(value);
        self.last = value;
        self.sum += value;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    fn range(&self) -> f64 {
        self.max - self.min
    }
}

fn decode_stream(path: &str) -> Result<()> {
    let payloads = main_stream_payloads(path)?;
    anyhow::ensure!(!payloads.is_empty(), "{path} has no stream payloads");
    let (stats, malformed) = decode_stream_fields(&payloads)?;

    let mut fields: Vec<_> = stats
        .iter()
        .filter(|(_, stat)| stat.count > 1 && stat.changes > 0)
        .collect();
    fields.sort_by(|a, b| {
        b.1.range()
            .partial_cmp(&a.1.range())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });

    println!(
        "decoded {} stream payloads; malformed records={}",
        payloads.len(),
        malformed
    );
    print_decoded_table(fields.into_iter().take(120));

    Ok(())
}

fn decode_stream_fields(
    payloads: &[Vec<u8>],
) -> Result<(BTreeMap<(u32, usize, usize), StreamFieldStats>, usize)> {
    let mut stats = BTreeMap::<(u32, usize, usize), StreamFieldStats>::new();
    let mut malformed = 0usize;

    for payload in payloads {
        let (values, is_malformed) = decode_stream_payload_with_status(payload)?;
        if is_malformed {
            malformed += 1;
        }

        for (key, value) in values {
            stats.entry(key).or_default().push(value);
        }
    }

    Ok((stats, malformed))
}

fn decode_stream_payload(payload: &[u8]) -> Result<BTreeMap<(u32, usize, usize), f64>> {
    let (values, _) = decode_stream_payload_with_status(payload)?;
    Ok(values)
}

fn decode_stream_payload_with_status(
    payload: &[u8],
) -> Result<(BTreeMap<(u32, usize, usize), f64>, bool)> {
    let mut values = BTreeMap::<(u32, usize, usize), f64>::new();
    let mut offset = STREAM_TLV_OFFSET;
    let mut current_id = None::<u32>;
    let mut current_occurrence = 0usize;
    let mut component = 0usize;
    let mut occurrences = BTreeMap::<u32, usize>::new();
    let mut malformed = false;

    while offset + 5 <= payload.len() {
        let typ = payload[offset];
        let len = u32::from_be_bytes(payload[offset + 1..offset + 5].try_into()?) as usize;
        offset += 5;

        if offset + len > payload.len() {
            malformed = true;
            break;
        }

        let value = &payload[offset..offset + len];

        match (typ, len) {
            (5, 4) => {
                let id = u32::from_be_bytes(value.try_into()?);
                let occurrence = occurrences.entry(id).or_default();
                current_id = Some(id);
                current_occurrence = *occurrence;
                *occurrence += 1;
                component = 0;
            }
            (4, 8) => {
                if let Some(id) = current_id {
                    let raw = i64::from_be_bytes(value.try_into()?);
                    let fixed = raw as f64 / 4_294_967_296.0;
                    values.insert((id, current_occurrence, component), fixed);
                    component += 1;
                }
            }
            (2, 4) | (3, 4) | (6, 8) => {}
            _ => {}
        }

        offset += len;
    }

    Ok((values, malformed))
}

fn print_decoded_table<'a, I>(fields: I)
where
    I: IntoIterator<Item = (&'a (u32, usize, usize), &'a StreamFieldStats)>,
{
    println!(
        "  {:>10} {:>3} {:>4} {:>6} {:>7} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "id", "occ", "comp", "count", "chg", "min", "max", "mean", "first", "last"
    );

    for ((id, occurrence, component), stat) in fields {
        println!(
            "  0x{id:08x} {:>3} {:>4} {:>6} {:>7} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            occurrence,
            component,
            stat.count,
            stat.changes,
            stat.min,
            stat.max,
            stat.mean(),
            stat.first,
            stat.last
        );
    }
}

fn pose_candidates(inputs: &[LogInput]) -> Result<()> {
    let occurrences = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut logs = Vec::<(String, Vec<BTreeMap<(u32, usize, usize), f64>>)>::new();

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        let mut frames = Vec::new();
        for payload in payloads {
            let (values, malformed) = decode_stream_payload_with_status(&payload)?;
            if !malformed && !values.is_empty() {
                frames.push(values);
            }
        }
        println!("{}: {} decoded frames", input.label, frames.len());
        logs.push((input.label.clone(), frames));
    }

    print_pose_candidate_axis("yaw", &logs, &occurrences, vector_yaw_deg);
    print_pose_candidate_axis("pitch", &logs, &occurrences, vector_pitch_deg);
    print_pose_candidate_axis("roll_xy", &logs, &occurrences, vector_roll_xy_deg);
    print_raw_field_candidates("yaw", &logs);
    print_raw_field_candidates("pitch", &logs);
    print_raw_field_candidates("roll", &logs);
    print_rotation_translation_fit(&logs);
    print_angle_translation_fit(&logs);

    Ok(())
}

fn print_raw_field_candidates(
    target: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
) {
    let mut keys = BTreeMap::<(u32, usize, usize), ()>::new();
    for (_, frames) in logs {
        for values in frames {
            for key in values.keys() {
                keys.insert(*key, ());
            }
        }
    }

    let mut rows = Vec::<((u32, usize, usize), Vec<f64>, f64)>::new();
    for key in keys.keys() {
        let mut ranges = Vec::new();
        let mut present = true;
        for (_, frames) in logs {
            let Some(range) = field_range(frames, *key) else {
                present = false;
                break;
            };
            ranges.push(range);
        }
        if !present {
            continue;
        }

        let score = score_raw_field_ranges(target, logs, &ranges);
        rows.push((*key, ranges, score));
    }

    rows.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("Top {target} raw field candidates:");
    print!("  {:>10} {:>3} {:>4} {:>8}", "id", "occ", "cmp", "score");
    for (label, _) in logs {
        print!(" {:>10}", label);
    }
    println!();

    for ((id, occurrence, component), ranges, score) in rows.into_iter().take(24) {
        print!(
            "  0x{id:08x} {:>3} {:>4} {:>8.3}",
            occurrence, component, score
        );
        for range in ranges {
            print!(" {:>10.2}", range);
        }
        println!();
    }
}

fn field_range(
    frames: &[BTreeMap<(u32, usize, usize), f64>],
    key: (u32, usize, usize),
) -> Option<f64> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0usize;

    for values in frames {
        let Some(value) = values.get(&key).copied() else {
            continue;
        };
        min = min.min(value);
        max = max.max(value);
        count += 1;
    }

    (count >= 8).then_some(max - min)
}

fn score_raw_field_ranges(
    target: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
    ranges: &[f64],
) -> f64 {
    let mut target_range = 0.0_f64;
    let mut shift_leak = 0.0_f64;
    let mut other_motion = 0.0_f64;

    for ((label, _), range) in logs.iter().zip(ranges) {
        if label.starts_with(target) {
            target_range = target_range.max(*range);
        } else if label.starts_with("shift") {
            shift_leak = shift_leak.max(*range);
        } else {
            other_motion = other_motion.max(*range);
        }
    }

    target_range / (shift_leak * 2.0 + other_motion * 0.5 + 1.0)
}

fn print_pose_candidate_axis<F>(
    axis: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
    occurrences: &[usize],
    angle_fn: F,
) where
    F: Fn([f64; 3]) -> f64 + Copy,
{
    let mut rows = Vec::<((usize, usize), Vec<f64>, f64)>::new();

    for (i, &a) in occurrences.iter().enumerate() {
        for &b in occurrences.iter().skip(i + 1) {
            let mut ranges = Vec::new();
            for (_, frames) in logs {
                let range = pair_angle_range(frames, a, b, angle_fn).unwrap_or(0.0);
                ranges.push(range);
            }
            let score = score_pose_ranges(axis, logs, &ranges);
            rows.push(((a, b), ranges, score));
        }
    }

    rows.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.1.iter()
                    .fold(0.0_f64, |m, v| m.max(*v))
                    .partial_cmp(&a.1.iter().fold(0.0_f64, |m, v| m.max(*v)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    println!("Top {axis} vector candidates:");
    print!("  {:>5} {:>8}", "pair", "score");
    for (label, _) in logs {
        print!(" {:>10}", label);
    }
    println!();

    for ((a, b), ranges, score) in rows.into_iter().take(24) {
        print!("  {:>2}-{:>2} {:>8.3}", a, b, score);
        for range in ranges {
            print!(" {:>10.3}", range);
        }
        println!();
    }
}

fn score_pose_ranges(
    axis: &str,
    logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)],
    ranges: &[f64],
) -> f64 {
    let target = match axis {
        "yaw" => "yaw",
        "pitch" => "pitch",
        "roll_xy" => "roll",
        _ => "",
    };

    let mut target_range = 0.0_f64;
    let mut shift_leak = 0.0_f64;
    let mut other_motion = 0.0_f64;
    for ((label, _), range) in logs.iter().zip(ranges) {
        if label.starts_with(target) {
            target_range = target_range.max(*range);
        } else if label.starts_with("shift") {
            shift_leak = shift_leak.max(*range);
        } else {
            other_motion = other_motion.max(*range);
        }
    }

    target_range / (shift_leak * 2.0 + other_motion * 0.5 + 1.0)
}

fn pair_angle_range<F>(
    frames: &[BTreeMap<(u32, usize, usize), f64>],
    a: usize,
    b: usize,
    angle_fn: F,
) -> Option<f64>
where
    F: Fn([f64; 3]) -> f64,
{
    let mut first = None::<f64>;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0usize;

    for values in frames {
        let pa = head_point(values, a)?;
        let pb = head_point(values, b)?;
        let vec = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
        if len < 1.0 {
            continue;
        }

        let angle = angle_fn(vec);
        let base = *first.get_or_insert(angle);
        let value = normalize_angle_deg(angle - base);
        min = min.min(value);
        max = max.max(value);
        count += 1;
    }

    (count >= 8).then_some(max - min)
}

fn vector_yaw_deg(vec: [f64; 3]) -> f64 {
    vec[0].atan2(vec[2]).to_degrees()
}

fn vector_pitch_deg(vec: [f64; 3]) -> f64 {
    (-vec[1])
        .atan2((vec[0] * vec[0] + vec[2] * vec[2]).sqrt())
        .to_degrees()
}

fn vector_roll_xy_deg(vec: [f64; 3]) -> f64 {
    vec[1].atan2(vec[0]).to_degrees()
}

fn print_angle_translation_fit(logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)]) {
    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [[0.0; 3]; 3];
    let mut samples = 0usize;

    for (label, frames) in logs {
        if !label.starts_with("shift") {
            continue;
        }

        let mut origin_head = None::<[f64; 3]>;
        let mut origin_angles = None::<[f64; 3]>;

        for (packet, values) in frames.iter().enumerate() {
            let frame = TrackingFrame::from_decoded(packet as u64, values);
            let Some(head) = frame.head_xyz() else {
                continue;
            };
            let Some(angles) = default_raw_angles(values) else {
                continue;
            };

            let origin_head = *origin_head.get_or_insert(head);
            let origin_angles = *origin_angles.get_or_insert(angles);
            let translation = [
                (head[0] - origin_head[0]) / OPENTRACK_HEAD_SCALE,
                (head[1] - origin_head[1]) / OPENTRACK_HEAD_SCALE,
                (head[2] - origin_head[2]) / OPENTRACK_HEAD_SCALE,
            ];
            if dot3(translation, translation).sqrt() < 0.01 {
                continue;
            }

            let angle_delta = [
                (angles[0] - origin_angles[0]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[0],
                (angles[1] - origin_angles[1]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[1],
                (angles[2] - origin_angles[2]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[2],
            ];

            for row in 0..3 {
                for col in 0..3 {
                    xtx[row][col] += translation[row] * translation[col];
                }
            }
            for axis in 0..3 {
                for col in 0..3 {
                    xty[axis][col] += angle_delta[axis] * translation[col];
                }
            }
            samples += 1;
        }
    }

    println!("Fitted --opentrack-angle-translation-comp from shift logs ({samples} samples):");
    if samples == 0 {
        println!("  not enough shift samples");
        return;
    }

    let mut matrix = [[0.0; 3]; 3];
    for axis in 0..3 {
        if let Some(row) = solve_3x3(xtx, xty[axis]) {
            matrix[axis] = row;
        }
    }

    println!(
        "  {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2]
    );
    println!("  rows are yaw,pitch,roll; columns are tx,ty,tz in cm");

    println!("  raw/compensated angle ranges by log:");
    println!(
        "  {:>10} {:>8} {:>8} {:>8} {:>8}",
        "log", "yaw_raw", "yaw_cmp", "pit_raw", "pit_cmp"
    );
    for (label, frames) in logs {
        if let Some((raw, comp)) = angle_ranges_with_comp(frames, matrix) {
            println!(
                "  {:>10} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                label, raw[0], comp[0], raw[1], comp[1]
            );
        }
    }
}

fn print_rotation_translation_fit(logs: &[(String, Vec<BTreeMap<(u32, usize, usize), f64>>)]) {
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [[0.0; 3]; 3];
    let mut samples = 0usize;

    for (label, frames) in logs {
        if !(label.starts_with("yaw") || label.starts_with("pitch") || label.starts_with("roll")) {
            continue;
        }

        let mut origin_head = None::<[f64; 3]>;
        let mut origin_angles = None::<[f64; 3]>;

        for (packet, values) in frames.iter().enumerate() {
            let frame = TrackingFrame::from_decoded(packet as u64, values);
            let Some(head) = frame.head_xyz() else {
                continue;
            };
            let Some(angles) = default_raw_angles(values) else {
                continue;
            };

            let origin_head = *origin_head.get_or_insert(head);
            let origin_angles = *origin_angles.get_or_insert(angles);
            let angle_delta = [
                (angles[0] - origin_angles[0]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[0],
                (angles[1] - origin_angles[1]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[1],
                0.0,
            ];
            if dot3(angle_delta, angle_delta).sqrt() < 0.1 {
                continue;
            }

            let translation = [
                (head[0] - origin_head[0]) / OPENTRACK_HEAD_SCALE,
                (head[1] - origin_head[1]) / OPENTRACK_HEAD_SCALE,
                (head[2] - origin_head[2]) / OPENTRACK_HEAD_SCALE,
            ];

            for row in 0..3 {
                for col in 0..3 {
                    ata[row][col] += angle_delta[row] * angle_delta[col];
                }
            }
            for axis in 0..3 {
                for col in 0..3 {
                    atb[axis][col] += translation[axis] * angle_delta[col];
                }
            }
            samples += 1;
        }
    }

    println!("Fitted --opentrack-rotation-comp from rotation logs ({samples} samples):");
    if samples == 0 {
        println!("  not enough rotation samples");
        return;
    }

    let mut matrix = [[0.0; 3]; 3];
    for axis in 0..3 {
        if let Some(row) = solve_3x3(ata, atb[axis]) {
            matrix[axis] = row;
        }
    }

    println!(
        "  {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2]
    );
    println!("  rows are tx,ty,tz in cm; columns are yaw,pitch,roll in deg");
}

fn default_raw_angles(values: &BTreeMap<(u32, usize, usize), f64>) -> Option<[f64; 3]> {
    let yaw = field_value(
        values,
        LiveField::new("", 0x00031f41, DEFAULT_OPENTRACK_ANGLE_OCC, 0),
    )?;
    let pitch = field_value(
        values,
        LiveField::new("", 0x00031f41, DEFAULT_OPENTRACK_ANGLE_OCC, 1),
    )?;
    Some([yaw, -pitch, 0.0])
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        a[i][i] += 1e-9;
        let mut pivot = i;
        for row in i + 1..3 {
            if a[row][i].abs() > a[pivot][i].abs() {
                pivot = row;
            }
        }
        if a[pivot][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);

        let div = a[i][i];
        for col in i..3 {
            a[i][col] /= div;
        }
        b[i] /= div;

        for row in 0..3 {
            if row == i {
                continue;
            }
            let factor = a[row][i];
            for col in i..3 {
                a[row][col] -= factor * a[i][col];
            }
            b[row] -= factor * b[i];
        }
    }

    Some(b)
}

fn angle_ranges_with_comp(
    frames: &[BTreeMap<(u32, usize, usize), f64>],
    matrix: [[f64; 3]; 3],
) -> Option<([f64; 3], [f64; 3])> {
    let mut origin_head = None::<[f64; 3]>;
    let mut origin_angles = None::<[f64; 3]>;
    let mut raw_min = [f64::INFINITY; 3];
    let mut raw_max = [f64::NEG_INFINITY; 3];
    let mut comp_min = [f64::INFINITY; 3];
    let mut comp_max = [f64::NEG_INFINITY; 3];
    let mut count = 0usize;

    for (packet, values) in frames.iter().enumerate() {
        let frame = TrackingFrame::from_decoded(packet as u64, values);
        let Some(head) = frame.head_xyz() else {
            continue;
        };
        let Some(angles) = default_raw_angles(values) else {
            continue;
        };
        let origin_head = *origin_head.get_or_insert(head);
        let origin_angles = *origin_angles.get_or_insert(angles);
        let translation = [
            (head[0] - origin_head[0]) / OPENTRACK_HEAD_SCALE,
            (head[1] - origin_head[1]) / OPENTRACK_HEAD_SCALE,
            (head[2] - origin_head[2]) / OPENTRACK_HEAD_SCALE,
        ];
        let raw = [
            (angles[0] - origin_angles[0]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[0],
            (angles[1] - origin_angles[1]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[1],
            (angles[2] - origin_angles[2]) / DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE[2],
        ];
        let comp = [
            raw[0] - dot3(matrix[0], translation),
            raw[1] - dot3(matrix[1], translation),
            raw[2] - dot3(matrix[2], translation),
        ];

        for axis in 0..3 {
            raw_min[axis] = raw_min[axis].min(raw[axis]);
            raw_max[axis] = raw_max[axis].max(raw[axis]);
            comp_min[axis] = comp_min[axis].min(comp[axis]);
            comp_max[axis] = comp_max[axis].max(comp[axis]);
        }
        count += 1;
    }

    if count < 8 {
        return None;
    }

    Some((
        [
            raw_max[0] - raw_min[0],
            raw_max[1] - raw_min[1],
            raw_max[2] - raw_min[2],
        ],
        [
            comp_max[0] - comp_min[0],
            comp_max[1] - comp_min[1],
            comp_max[2] - comp_min[2],
        ],
    ))
}

fn compare_decoded(inputs: &[LogInput]) -> Result<()> {
    let mut logs = Vec::new();

    for input in inputs {
        let payloads = main_stream_payloads(&input.path)?;
        let (fields, malformed) = decode_stream_fields(&payloads)?;
        println!(
            "{}: {} stream packets, {} decoded fields, malformed={}",
            input.label,
            payloads.len(),
            fields.len(),
            malformed
        );
        logs.push((input.label.clone(), fields));
    }

    let mut candidates = Vec::new();

    if let Some((_, first_fields)) = logs.first() {
        for key in first_fields.keys() {
            let mut means = Vec::new();
            let mut ranges = Vec::new();
            let mut present = true;

            for (_, fields) in &logs {
                let Some(stat) = fields.get(key) else {
                    present = false;
                    break;
                };
                means.push(stat.mean());
                ranges.push(stat.range());
            }

            if !present {
                continue;
            }

            let between = means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - means.iter().copied().fold(f64::INFINITY, f64::min);
            let within = ranges.iter().sum::<f64>() / ranges.len() as f64;
            let score = between / (within + 1.0);

            if score >= 0.02 && between.abs() > 1.0 {
                candidates.push((*key, means, ranges, score, between));
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.3.partial_cmp(&a.3)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    println!("Top decoded cross-log candidates:");
    print!(
        "  {:>10} {:>3} {:>4} {:>9} {:>12}",
        "id", "occ", "cmp", "score", "between"
    );
    for (label, _) in &logs {
        print!(" {:>12}", label);
    }
    println!();

    for ((id, occurrence, component), means, _, score, between) in candidates.iter().take(80) {
        print!(
            "  0x{id:08x} {:>3} {:>4} {:>9.4} {:>12.4}",
            occurrence, component, score, between
        );
        for mean in means {
            print!(" {:>12.4}", mean);
        }
        println!();
    }

    print_decoded_pair("head_left", "head_right", &logs, &candidates, 32);
    print_decoded_pair("eyes_left", "eyes_right", &logs, &candidates, 32);

    Ok(())
}

type DecodedFields = BTreeMap<(u32, usize, usize), StreamFieldStats>;

fn print_decoded_pair(
    left_label: &str,
    right_label: &str,
    logs: &[(String, DecodedFields)],
    candidates: &[((u32, usize, usize), Vec<f64>, Vec<f64>, f64, f64)],
    limit: usize,
) {
    let Some(left) = logs.iter().position(|(label, _)| label == left_label) else {
        return;
    };
    let Some(right) = logs.iter().position(|(label, _)| label == right_label) else {
        return;
    };

    let mut pairs: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let delta = candidate.1[right] - candidate.1[left];
            let noise = (candidate.2[left] + candidate.2[right]) / 2.0 + 1.0;
            (delta.abs() / noise, delta, candidate)
        })
        .filter(|(score, delta, _)| *score >= 0.02 && delta.abs() > 1.0)
        .collect();

    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2 .0.cmp(&b.2 .0))
    });

    println!("Top decoded {left_label} vs {right_label}:");
    println!(
        "  {:>10} {:>3} {:>4} {:>9} {:>12} {:>12} {:>12}",
        "id", "occ", "cmp", "score", left_label, right_label, "delta"
    );
    for (score, delta, candidate) in pairs.into_iter().take(limit) {
        let (id, occurrence, component) = candidate.0;
        println!(
            "  0x{id:08x} {:>3} {:>4} {:>9.4} {:>12.4} {:>12.4} {:>12.4}",
            occurrence, component, score, candidate.1[left], candidate.1[right], delta
        );
    }
}

#[derive(Clone)]
struct NumericCandidate {
    kind: &'static str,
    offset: usize,
    count: usize,
    min: f64,
    max: f64,
    mean: f64,
    stddev: f64,
    first: f64,
    last: f64,
    changes: usize,
    score: f64,
}

fn print_numeric_candidates(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        println!("Need at least two stream packets for numeric analysis.");
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut f32_candidates = Vec::new();
    let mut i16_candidates = Vec::new();
    let mut u16_candidates = Vec::new();
    let mut u8_candidates = Vec::new();
    let mut u24_candidates = Vec::new();

    print_length_distribution(payloads);

    for offset in 0..min_len {
        let values: Vec<f64> = payloads
            .iter()
            .map(|payload| payload[offset] as f64)
            .collect();

        if let Some(candidate) = summarize_numeric("u8", offset, &values) {
            if candidate.stddev >= 1.0 {
                u8_candidates.push(candidate);
            }
        }
    }

    for offset in 0..min_len.saturating_sub(3) {
        let values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            })
            .collect();

        if let Some(candidate) = summarize_numeric("f32", offset, &values) {
            if candidate.min.is_finite()
                && candidate.max.is_finite()
                && candidate.min.abs() < 1000.0
                && candidate.max.abs() < 1000.0
                && candidate.stddev > 0.000001
            {
                f32_candidates.push(candidate);
            }
        }
    }

    for offset in 0..min_len.saturating_sub(1) {
        let i16_values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            })
            .collect();
        let u16_values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                u16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap()) as f64
            })
            .collect();

        if let Some(candidate) = summarize_numeric("i16", offset, &i16_values) {
            if candidate.stddev >= 1.0 {
                i16_candidates.push(candidate);
            }
        }

        if let Some(candidate) = summarize_numeric("u16", offset, &u16_values) {
            if candidate.stddev >= 1.0 {
                u16_candidates.push(candidate);
            }
        }
    }

    for offset in 0..min_len.saturating_sub(2) {
        let values: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                let bytes = [payload[offset], payload[offset + 1], payload[offset + 2], 0];
                u32::from_le_bytes(bytes) as f64
            })
            .collect();

        if let Some(candidate) = summarize_numeric("u24", offset, &values) {
            if candidate.stddev >= 1.0 {
                u24_candidates.push(candidate);
            }
        }
    }

    sort_candidates(&mut f32_candidates);
    sort_candidates(&mut i16_candidates);
    sort_candidates(&mut u16_candidates);
    sort_candidates(&mut u8_candidates);
    sort_candidates(&mut u24_candidates);

    print_candidate_table("Top varying u8 candidates", &u8_candidates, 16);
    print_candidate_table("Top varying f32 candidates", &f32_candidates, 24);
    print_candidate_table("Top varying i16 candidates", &i16_candidates, 16);
    print_candidate_table("Top varying u16 candidates", &u16_candidates, 16);
    print_candidate_table("Top varying u24 candidates", &u24_candidates, 16);

    print_f32_triplets(payloads);
}

fn print_length_distribution(payloads: &[Vec<u8>]) {
    let mut lengths: BTreeMap<usize, usize> = BTreeMap::new();

    for payload in payloads {
        *lengths.entry(payload.len()).or_default() += 1;
    }

    println!("Stream packet length distribution:");
    for (len, count) in lengths {
        println!("  len={len:<5} count={count}");
    }
}

fn summarize_numeric(
    kind: &'static str,
    offset: usize,
    values: &[f64],
) -> Option<NumericCandidate> {
    let first = *values.first()?;
    let last = *values.last()?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut changes = 0usize;
    let mut prev = first;

    for &value in values {
        if !value.is_finite() {
            return None;
        }

        min = min.min(value);
        max = max.max(value);
        sum += value;

        if (value - prev).abs() > f64::EPSILON {
            changes += 1;
        }

        prev = value;
    }

    let count = values.len();
    let mean = sum / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    let stddev = variance.sqrt();
    let range = max - min;

    if range <= 0.0 || changes == 0 {
        return None;
    }

    Some(NumericCandidate {
        kind,
        offset,
        count,
        min,
        max,
        mean,
        stddev,
        first,
        last,
        changes,
        score: stddev * (changes as f64).ln_1p(),
    })
}

fn sort_candidates(candidates: &mut [NumericCandidate]) {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.cmp(&b.offset))
    });
}

fn print_candidate_table(title: &str, candidates: &[NumericCandidate], limit: usize) {
    if candidates.is_empty() {
        println!("{title}: none");
        return;
    }

    println!("{title}:");
    for candidate in candidates.iter().take(limit) {
        println!(
            "  {:>3} off={:<4} count={:<4} changes={:<4} min={:>11.5} max={:>11.5} mean={:>11.5} std={:>11.5} first={:>11.5} last={:>11.5}",
            candidate.kind,
            candidate.offset,
            candidate.count,
            candidate.changes,
            candidate.min,
            candidate.max,
            candidate.mean,
            candidate.stddev,
            candidate.first,
            candidate.last
        );
    }
}

fn print_f32_triplets(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut candidates = Vec::new();

    for offset in 0..min_len.saturating_sub(11) {
        let xs: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as f64
            })
            .collect();
        let ys: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset + 4..offset + 8].try_into().unwrap()) as f64
            })
            .collect();
        let zs: Vec<f64> = payloads
            .iter()
            .map(|payload| {
                f32::from_le_bytes(payload[offset + 8..offset + 12].try_into().unwrap()) as f64
            })
            .collect();

        let Some(x) = summarize_numeric("f32", offset, &xs) else {
            continue;
        };
        let Some(y) = summarize_numeric("f32", offset + 4, &ys) else {
            continue;
        };
        let Some(z) = summarize_numeric("f32", offset + 8, &zs) else {
            continue;
        };

        let plausible = [&x, &y, &z].iter().all(|c| {
            c.min.is_finite()
                && c.max.is_finite()
                && c.min.abs() < 1000.0
                && c.max.abs() < 1000.0
                && c.stddev > 0.000001
        });

        if plausible {
            candidates.push((x.score + y.score + z.score, offset, x, y, z));
        }
    }

    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    if candidates.is_empty() {
        println!("Top contiguous f32 triplets: none");
        return;
    }

    println!("Top contiguous f32 triplets:");
    for (_, offset, x, y, z) in candidates.iter().take(16) {
        println!(
            "  off={:<4} x=[{:>9.5}..{:>9.5}] std={:>9.5} y=[{:>9.5}..{:>9.5}] std={:>9.5} z=[{:>9.5}..{:>9.5}] std={:>9.5}",
            offset, x.min, x.max, x.stddev, y.min, y.max, y.stddev, z.min, z.max, z.stddev
        );
    }
}

fn print_stream_changes(payloads: &[Vec<u8>]) {
    if payloads.len() < 2 {
        return;
    }

    let min_len = payloads.iter().map(Vec::len).min().unwrap_or(0);
    let mut ranges = Vec::new();
    let mut start = None;

    for offset in 0..min_len {
        let first = payloads[0][offset];
        let changed = payloads
            .iter()
            .skip(1)
            .any(|payload| payload[offset] != first);

        match (start, changed) {
            (None, true) => start = Some(offset),
            (Some(s), false) => {
                ranges.push((s, offset));
                start = None;
            }
            _ => {}
        }
    }

    if let Some(s) = start {
        ranges.push((s, min_len));
    }

    if ranges.is_empty() {
        println!("Stream packets have identical first {min_len} bytes.");
        return;
    }

    println!(
        "Changing byte ranges across first {} stream packets:",
        payloads.len()
    );
    for (start, end) in ranges.into_iter().take(64) {
        let width = end - start;
        let first_hex = short_hex(&payloads[0][start..end]);
        let last_hex = short_hex(&payloads[payloads.len() - 1][start..end]);

        print!(
            "  off={} len={} first={} last={}",
            start, width, first_hex, last_hex
        );

        if start + 4 <= min_len {
            let first_f32 = f32::from_le_bytes(payloads[0][start..start + 4].try_into().unwrap());
            let last_f32 = f32::from_le_bytes(
                payloads[payloads.len() - 1][start..start + 4]
                    .try_into()
                    .unwrap(),
            );

            if first_f32.is_finite()
                && last_f32.is_finite()
                && first_f32.abs() < 1000.0
                && last_f32.abs() < 1000.0
            {
                print!(" f32_first={first_f32:.6} f32_last={last_f32:.6}");
            }
        }

        println!();
    }
}

#[derive(Clone, Debug)]
struct CalibrationRecord {
    offset: usize,
    values: Vec<f32>,
}

fn extract_calibration(path: &str, json_path: Option<&str>) -> Result<()> {
    let packets = read_init_packets(path)?;
    let blob = calibration_blob(&packets);
    anyhow::ensure!(
        !blob.is_empty(),
        "{path} does not contain large init blob packets"
    );

    let records = find_calibration_records(&blob);
    let point_records: Vec<_> = records
        .iter()
        .filter(|record| record.values.len() == 4)
        .collect();

    println!("input packets: {}", packets.len());
    println!("large blob bytes: {}", blob.len());
    println!(
        "calibration-like records: {} total, {} four-float point records",
        records.len(),
        point_records.len()
    );
    println!();

    println!("Records:");
    println!(
        "  {:>3} {:>8} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "#", "offset", "kind", "v0", "v1", "v2", "v3"
    );
    for (i, record) in records.iter().enumerate() {
        println!(
            "  {:>3} 0x{:06x} {:>7} {:>12} {:>12} {:>12} {:>12}",
            i + 1,
            record.offset,
            format!("{}f", record.values.len()),
            fmt_cal_value(record.values.first().copied()),
            fmt_cal_value(record.values.get(1).copied()),
            fmt_cal_value(record.values.get(2).copied()),
            fmt_cal_value(record.values.get(3).copied()),
        );
    }

    if !point_records.is_empty() {
        println!();
        println!("Likely calibration target/observed points:");
        println!(
            "  {:>3} {:>8} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
            "#", "offset", "target_x", "target_y", "observed_x", "observed_y", "err_x", "err_y"
        );
        for (i, record) in point_records.iter().enumerate() {
            let target_x = record.values[0];
            let target_y = record.values[1];
            let observed_x = record.values[2];
            let observed_y = record.values[3];
            println!(
                "  {:>3} 0x{:06x} {:>10.4} {:>10.4} {:>12.4} {:>12.4} {:>10.4} {:>10.4}",
                i + 1,
                record.offset,
                target_x,
                target_y,
                observed_x,
                observed_y,
                observed_x - target_x,
                observed_y - target_y,
            );
        }
    }

    if let Some(json_path) = json_path {
        write_calibration_json(json_path, &records)?;
        println!();
        println!("Wrote {json_path}");
    }

    Ok(())
}

fn calibration_blob(packets: &[InitPacket]) -> Vec<u8> {
    let mut blob = Vec::new();

    for packet in packets {
        if packet.data.len() >= 512 {
            blob.extend_from_slice(&packet.data[8..]);
        }
    }

    blob
}

fn find_calibration_records(blob: &[u8]) -> Vec<CalibrationRecord> {
    const MARKER: &[u8; 8] = b"\x01\0\0\0\0\0\0\0";
    let mut records = Vec::new();
    let search_start = blob.len().saturating_sub(8192);
    let mut offset = search_start;

    while offset + MARKER.len() <= blob.len() {
        let Some(relative) = find_bytes(&blob[offset..], MARKER) else {
            break;
        };
        let marker_offset = offset + relative;
        let value_offset = marker_offset + MARKER.len();
        let Some(next_relative) = find_bytes(&blob[value_offset..], MARKER) else {
            break;
        };
        let next_marker = value_offset + next_relative;

        if next_marker > value_offset && (next_marker - value_offset) % 4 == 0 {
            let value_count = (next_marker - value_offset) / 4;
            if matches!(value_count, 2 | 4) {
                let mut values = Vec::with_capacity(value_count);
                let mut plausible = true;

                for chunk in blob[value_offset..next_marker].chunks_exact(4) {
                    let value = f32::from_le_bytes(chunk.try_into().unwrap());
                    if !value.is_finite() || !(-0.50..=1.50).contains(&value) {
                        plausible = false;
                        break;
                    }
                    values.push(value);
                }

                if plausible {
                    records.push(CalibrationRecord {
                        offset: marker_offset,
                        values,
                    });
                }
            }
        }

        offset = marker_offset + 1;
    }

    records
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn fmt_cal_value(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

fn write_calibration_json(path: &str, records: &[CalibrationRecord]) -> Result<()> {
    let mut out = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create calibration JSON {path}"))?,
    );

    writeln!(out, "{{")?;
    writeln!(out, "  \"records\": [")?;
    for (i, record) in records.iter().enumerate() {
        write!(
            out,
            "    {{\"offset\":{},\"kind\":\"{}f\",\"values\":[",
            record.offset,
            record.values.len()
        )?;
        for (j, value) in record.values.iter().enumerate() {
            if j > 0 {
                write!(out, ",")?;
            }
            write!(out, "{value:.8}")?;
        }
        write!(out, "]}}")?;
        if i + 1 < records.len() {
            writeln!(out, ",")?;
        } else {
            writeln!(out)?;
        }
    }
    writeln!(out, "  ],")?;
    writeln!(out, "  \"points\": [")?;

    let points: Vec<_> = records
        .iter()
        .filter(|record| record.values.len() == 4)
        .collect();
    for (i, record) in points.iter().enumerate() {
        let target_x = record.values[0];
        let target_y = record.values[1];
        let observed_x = record.values[2];
        let observed_y = record.values[3];
        write!(
            out,
            "    {{\"offset\":{},\"target\":{{\"x\":{target_x:.8},\"y\":{target_y:.8}}},\"observed\":{{\"x\":{observed_x:.8},\"y\":{observed_y:.8}}},\"error\":{{\"x\":{:.8},\"y\":{:.8}}}}}",
            record.offset,
            observed_x - target_x,
            observed_y - target_y
        )?;
        if i + 1 < points.len() {
            writeln!(out, ",")?;
        } else {
            writeln!(out)?;
        }
    }
    writeln!(out, "  ]")?;
    writeln!(out, "}}")?;
    out.flush()?;

    Ok(())
}

fn short_hex(bytes: &[u8]) -> String {
    const MAX: usize = 12;
    let mut out = String::new();

    for b in bytes.iter().take(MAX) {
        out.push_str(&format!("{b:02x}"));
    }

    if bytes.len() > MAX {
        out.push_str("...");
    }

    out
}

fn read_init_packets(path: &str) -> Result<Vec<InitPacket>> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?;

    let mut packets = Vec::new();

    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();

        let ep_str = parts
            .next()
            .with_context(|| format!("missing endpoint at line {}", line_no + 1))?;

        let hex = parts
            .next()
            .with_context(|| format!("missing payload at line {}", line_no + 1))?;

        let ep = u8::from_str_radix(ep_str.trim_start_matches("0x"), 16)
            .with_context(|| format!("bad endpoint at line {}", line_no + 1))?;

        let data = hex_to_bytes(hex).with_context(|| format!("bad hex at line {}", line_no + 1))?;

        packets.push(InitPacket { ep, data });
    }

    Ok(packets)
}

fn hex_to_bytes(s: &str) -> Result<Vec<u8>> {
    let s = s.trim().replace(':', "");

    if s.len() % 2 != 0 {
        anyhow::bail!("odd hex length");
    }

    let mut bytes = Vec::with_capacity(s.len() / 2);

    for i in (0..s.len()).step_by(2) {
        let b = u8::from_str_radix(&s[i..i + 2], 16)
            .with_context(|| format!("bad hex byte at {}", i / 2))?;
        bytes.push(b);
    }

    Ok(bytes)
}

fn declared_len(buf: &[u8]) -> Option<u32> {
    if buf.len() < 8 {
        return None;
    }

    Some(u32::from_le_bytes(buf[4..8].try_into().ok()?))
}

fn marker(buf: &[u8]) -> Option<u32> {
    if buf.len() < 12 {
        return None;
    }

    Some(u32::from_be_bytes(buf[8..12].try_into().ok()?))
}

fn seq(buf: &[u8]) -> Option<u32> {
    if buf.len() < 16 {
        return None;
    }

    Some(u32::from_be_bytes(buf[12..16].try_into().ok()?))
}
