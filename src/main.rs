use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::{
    env, fs, thread,
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
    let mut h = open_tobii(&ctx)?;
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
                opts.print_decoded,
            );
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }

    println!("Init replay finished. Reading stream...");
    read_stream(&ctx, h, &opts, &mut log, &mut live_csv, &mut jsonl)
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
        })
    }
}

fn print_usage() {
    println!(
        "usage:\n  cargo run -- [init_packets_ep.txt] [--log tobii_stream.bin] [--decoded-csv decoded.csv] [--jsonl frames.jsonl] [--print-decoded] [--dashboard] [--max-stream-packets N] [--max-init-packets N] [--no-reconnect]\n  cargo run -- analyze-log tobii_stream.bin\n  cargo run -- compare-logs [label:]path.bin [label:]path.bin ...\n  cargo run -- decode-stream tobii_stream.bin\n  cargo run -- compare-decoded [label:]path.bin [label:]path.bin ...\n  cargo run -- extract-calibration init_packets_ep.txt [--json calibration.json]"
    );
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
) -> Result<()> {
    let mut buf = [0u8; 8192];
    let mut read_count = 0u64;
    let mut reconnect_attempts = 0u32;

    loop {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(2000)) {
            Ok(n) => {
                let data = &buf[..n];
                read_count += 1;
                reconnect_attempts = 0;

                log_packet(log, EP_IN, data)?;
                handle_live_decoded(
                    read_count,
                    data,
                    live_csv,
                    jsonl,
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
                if let Err(e) = handle_live_decoded(0, data, live_csv, jsonl, print_decoded, false)
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
        self.out.write_all(&[0, ep, 0, 0])?;
        self.out.write_all(&now_us().to_le_bytes())?;
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

    if print_decoded {
        print_live_decoded(&frame);
    }

    if dashboard {
        render_tracking_dashboard(&frame)?;
    }

    Ok(())
}

fn print_live_decoded(frame: &TrackingFrame) {
    println!(
        "  decoded #{} valid={} gaze=({},{}) norm=({},{}) left_eye=({},{}) right_eye=({},{}) head=({},{},{})",
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
    );
}

fn render_tracking_dashboard(frame: &TrackingFrame) -> Result<()> {
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
        write!(out, "}}}}")?;
        Ok(())
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

fn derive_live_values(values: &BTreeMap<(u32, usize, usize), f64>) -> [Option<f64>; 16] {
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

    let mut lengths = BTreeMap::<usize, usize>::new();
    for payload in &stream {
        *lengths.entry(payload.len()).or_default() += 1;
    }

    let main_len = lengths
        .into_iter()
        .max_by_key(|(_, count)| *count)
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
