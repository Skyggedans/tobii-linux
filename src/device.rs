use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::ToSocketAddrs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{error, fmt, thread};

use crate::cli::{Command, Options};
use crate::dashboard::render_dashboard_status;
use crate::decode::{decode_stream_payload, TrackingFrame};
use crate::engine::{GazeSample, PoseSample, Sample};
use crate::opentrack::OpentrackUdp;
use crate::protocol::{declared_len, marker, parse_init_packets, read_init_packets, seq, InitPacket};
use crate::sinks::{handle_live_decoded, log_packet, now_us, DecodedCsv, JsonlOutput, PacketLog};

pub(crate) const VID: u16 = 0x2104;

pub(crate) const PID: u16 = 0x0313;

pub(crate) const IFACE: u8 = 0;

pub(crate) const EP_IN: u8 = 0x83;

pub(crate) const MAX_REPLAY_ATTEMPTS: usize = 5;

pub(crate) const STARTUP_STREAM_TIMEOUTS: u32 = 3;

// UVC IR camera (interface 2 / EP 0x82): 560x560 8-bit grayscale. Each bulk
// payload is [2-byte UVC header][image]; the first payload of a frame also
// carries a 10-byte Tobii header (`XX 00 e8 03 00 00` + u32 counter).
const IFACE_VIDEO: u8 = 2;
const EP_VIDEO: u8 = 0x82;
const CAM_WIDTH: usize = 560;
const CAM_HEIGHT: usize = 560;
const TOBII_FRAME_MAGIC: [u8; 4] = [0xe8, 0x03, 0x00, 0x00];

pub(crate) fn run(opts: &Options) -> Result<()> {
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

        match replay_and_read_stream(&ctx, opts, &packets, init_limit) {
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
pub(crate) struct StreamStartupTimeout;

impl fmt::Display for StreamStartupTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream did not start after initialization")
    }
}

impl error::Error for StreamStartupTimeout {}

pub(crate) fn replay_and_read_stream(
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
                opts.opentrack_auto_decouple,
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

pub(crate) fn open_tobii(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
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

pub(crate) fn read_stream(
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

pub(crate) fn wait_for_reconnect(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
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

pub(crate) fn vendor_control_init(h: &mut rusb::DeviceHandle<UsbContext>) -> Result<()> {
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

pub(crate) fn wait_for_response_seq(
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

pub(crate) fn drain_in_limited(
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

pub(crate) fn run_camera(opts: &Options) -> Result<()> {
    let Command::Camera {
        init_path,
        out_prefix,
        max_frames,
        skip_replay,
        fifo,
        frame_type,
        interval,
    } = &opts.command
    else {
        unreachable!();
    };

    let ctx = UsbContext::new()?;
    let mut h = open_tobii(&ctx)?;

    // Power the sensor/illuminators the same way the streaming path does.
    vendor_control_init(&mut h)?;
    if *skip_replay {
        println!("--no-replay: skipping init packet replay (control init only)");
    } else {
        let packets = read_init_packets(init_path)?;
        println!("Replaying {} init packets to start the device...", packets.len());
        let mut no_log: Option<PacketLog> = None;
        for pkt in &packets {
            let expected_seq = if marker(&pkt.data) == Some(0x51) {
                seq(&pkt.data)
            } else {
                None
            };
            if let Err(e) = h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000)) {
                eprintln!("init OUT failed: {e:?}");
                break;
            }
            if let Some(s) = expected_seq {
                let _ = wait_for_response_seq(&mut h, s, &mut no_log);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    // Claim the UVC video-streaming interface and negotiate the stream.
    if h.kernel_driver_active(IFACE_VIDEO).unwrap_or(false) {
        println!("Detaching kernel driver from video interface {IFACE_VIDEO}");
        let _ = h.detach_kernel_driver(IFACE_VIDEO);
    }
    h.claim_interface(IFACE_VIDEO)
        .context("failed to claim video streaming interface 2")?;
    println!("Claimed video streaming interface {IFACE_VIDEO}");

    let frame_size = uvc_negotiate(&mut h, *interval)?;
    println!(
        "UVC probe/commit done; negotiated frame buffer = {frame_size} bytes (expected {}x{} = {})",
        CAM_WIDTH,
        CAM_HEIGHT,
        CAM_WIDTH * CAM_HEIGHT
    );

    // Give the sensor a moment and clear any stale halt on the stream endpoint.
    thread::sleep(Duration::from_millis(100));
    let _ = h.clear_halt(EP_VIDEO);

    let stream = fifo.is_some();
    let mut fifo_writer = match fifo {
        Some(path) => {
            println!("Streaming raw {CAM_WIDTH}x{CAM_HEIGHT} frames to {path} (waiting for reader)...");
            // Open the FIFO write-only WITHOUT O_CREAT (File::create uses it, and
            // fs.protected_fifos blocks a root process on a FIFO it doesn't own in
            // sticky /tmp). Blocks until a reader connects.
            Some(BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                    .with_context(|| format!("failed to open fifo {path} (mkfifo it first)"))?,
            ))
        }
        None => {
            println!("Writing up to {max_frames} PGM frames as {out_prefix}NNN.pgm");
            None
        }
    };
    let max = *max_frames;
    let mut saved = 0usize;
    read_camera_frames(&mut h, frame_size, *frame_type, |frame| {
        emit_frame(&mut fifo_writer, out_prefix, saved, frame)?;
        saved += 1;
        Ok(stream || saved < max)
    })?;
    println!("Saved {saved} frame(s)");
    Ok(())
}

pub(crate) fn run_track(opts: &Options) -> Result<()> {
    let Command::Track {
        init_path,
        skip_replay,
        host,
        port,
    } = &opts.command
    else {
        unreachable!();
    };

    let target = (host.as_str(), *port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve {host}:{port}"))?
        .next()
        .with_context(|| format!("{host}:{port} resolved to nothing"))?;
    let socket = std::net::UdpSocket::bind(if target.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    })
    .context("failed to create UDP socket")?;
    println!("Loading face model...");
    let mut tracker = crate::track::Tracker::new()?;

    let ctx = UsbContext::new()?;
    let mut h = open_tobii(&ctx)?;
    vendor_control_init(&mut h)?;
    if !*skip_replay {
        let packets = read_init_packets(init_path)?;
        println!("Replaying {} init packets...", packets.len());
        let mut no_log: Option<PacketLog> = None;
        for pkt in &packets {
            let expected_seq = if marker(&pkt.data) == Some(0x51) {
                seq(&pkt.data)
            } else {
                None
            };
            if h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000)).is_err() {
                break;
            }
            if let Some(s) = expected_seq {
                let _ = wait_for_response_seq(&mut h, s, &mut no_log);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }
    if h.kernel_driver_active(IFACE_VIDEO).unwrap_or(false) {
        let _ = h.detach_kernel_driver(IFACE_VIDEO);
    }
    h.claim_interface(IFACE_VIDEO)
        .context("failed to claim video streaming interface 2")?;
    let frame_size = uvc_negotiate(&mut h, None)?;
    thread::sleep(Duration::from_millis(100));
    let _ = h.clear_halt(EP_VIDEO);

    println!("Tracking head pose -> OpenTrack {target}. Sit still ~1s to calibrate, then move.");
    let mut frames = 0u64;
    read_camera_frames(&mut h, frame_size, Some(2), |frame| {
        if let Some(pose) = tracker.process(frame, CAM_WIDTH, CAM_HEIGHT)? {
            let mut packet = [0u8; 48];
            for (i, v) in pose.iter().enumerate() {
                packet[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            socket.send_to(&packet, target).ok();
            frames += 1;
            if frames % 8 == 0 {
                eprint!(
                    "\ryaw/pit/roll={:+5.1}/{:+5.1}/{:+5.1}  tx/ty/tz={:+5.1}/{:+5.1}/{:+5.1}   ",
                    pose[3], pose[4], pose[5], pose[0], pose[1], pose[2]
                );
            }
        }
        Ok(true)
    })
}

/// Drive the camera head-pose pipeline for the FFI/library engine: own the
/// device, run the tracker, and push pose samples to `tx` until `stop` is set.
/// Init packets are embedded so the library is self-contained.
///
/// The device init is flaky (~every other cold start it comes up without the
/// camera streaming), so the whole open+init+bring-up is retried: the startup
/// watchdog in `read_camera_frames` bails fast with `StreamStartupTimeout` and
/// we redo a full, fresh init instead of leaving the engine dead.
pub(crate) fn run_pose_engine(
    stop: &Arc<AtomicBool>,
    recenter: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
) -> Result<()> {
    let mut tracker = crate::track::Tracker::new()?;
    let ctx = UsbContext::new()?;
    // The init is a blind replay that assumes the device starts from a clean
    // baseline, but we never send a stop on exit, so a prior session leaves it
    // "hot" and the replay desyncs (~every other cold start). A USB reset forces
    // re-enumeration to a known state so the replay's preconditions hold.
    reset_device_baseline(&ctx);

    for attempt in 1..=MAX_REPLAY_ATTEMPTS {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        match pose_engine_attempt(&ctx, stop, recenter, tx, &mut tracker) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < MAX_REPLAY_ATTEMPTS && !stop.load(Ordering::Relaxed) => {
                eprintln!(
                    "tobii camera init attempt {attempt}/{MAX_REPLAY_ATTEMPTS} failed: {e}; \
                     redoing full init"
                );
                thread::sleep(Duration::from_millis(700));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Best-effort USB reset to drop any state a prior session left on the device,
/// so the replayed init starts from a known baseline. Re-enumeration is what we
/// want; we then wait for the device to reappear (by presence, not a blind
/// sleep) so the following init doesn't race a missing device.
///
/// Set `TOBII_NO_RESET=1` to skip it (faster start; try this now that uvcvideo
/// is kept off the device — the reset may no longer be needed).
fn reset_device_baseline(ctx: &UsbContext) {
    let skip = std::env::var("TOBII_NO_RESET")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    if skip {
        return;
    }
    if let Ok(h) = open_tobii(ctx) {
        match h.reset() {
            Ok(()) => println!("reset Tobii to baseline before init"),
            Err(e) => println!("reset best-effort failed: {e:?}"),
        }
        drop(h); // device may re-enumerate; reopen fresh below
    }
    // Poll for re-enumeration to finish (cap ~3s) instead of sleeping blindly.
    for _ in 0..30 {
        thread::sleep(Duration::from_millis(100));
        if tobii_present(ctx) {
            thread::sleep(Duration::from_millis(150)); // brief settle
            return;
        }
    }
}

/// Is the Tobii plugged in right now? (Cheap check for the daemon watchdog.)
pub(crate) fn device_present() -> bool {
    UsbContext::new().map(|ctx| tobii_present(&ctx)).unwrap_or(false)
}

/// Is the Tobii on the bus right now (without opening/claiming it)?
fn tobii_present(ctx: &UsbContext) -> bool {
    ctx.devices()
        .map(|devs| {
            devs.iter().any(|d| {
                d.device_descriptor()
                    .map(|x| x.vendor_id() == VID && x.product_id() == PID)
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// Replay the embedded init packet sequence on EP 0x05.
fn replay_init(
    h: &mut rusb::DeviceHandle<UsbContext>,
    packets: &[InitPacket],
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    let mut no_log: Option<PacketLog> = None;
    for pkt in packets {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let expected_seq = if marker(&pkt.data) == Some(0x51) {
            seq(&pkt.data)
        } else {
            None
        };
        h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000))
            .context("init packet write failed")?;
        if let Some(s) = expected_seq {
            let _ = wait_for_response_seq(h, s, &mut no_log);
        }
        thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}

/// Does the bulk camera endpoint deliver any payload within `dur`?
fn probe_camera(h: &mut rusb::DeviceHandle<UsbContext>, dur: Duration) -> bool {
    let mut buf = vec![0u8; 65536];
    let start = Instant::now();
    while start.elapsed() < dur {
        if let Ok(n) = h.read_bulk(EP_VIDEO, &mut buf, Duration::from_millis(300)) {
            if n > 0 {
                return true;
            }
        }
    }
    false
}

/// One open+init+camera bring-up, streaming poses until `stop` or a failure.
///
/// On a cold start the camera does not stream after a single init pass — the
/// device needs the init replayed again (re-COMMIT alone does nothing). We do
/// that on the *same* handle (no costly re-open/reset), replaying init + COMMIT
/// and probing until the camera flows, which avoids the slow second open cycle.
fn pose_engine_attempt(
    ctx: &UsbContext,
    stop: &Arc<AtomicBool>,
    recenter: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
    tracker: &mut crate::track::Tracker,
) -> Result<()> {
    const INIT_PACKETS: &str = include_str!("../init_packets_ep.txt");
    let packets = parse_init_packets(INIT_PACKETS)?;
    let mut h = open_tobii(ctx)?;
    vendor_control_init(&mut h)?;
    if h.kernel_driver_active(IFACE_VIDEO).unwrap_or(false) {
        let _ = h.detach_kernel_driver(IFACE_VIDEO);
    }
    h.claim_interface(IFACE_VIDEO)
        .context("failed to claim video streaming interface 2")?;

    // Bring up the camera. A cold start needs more than the first init+COMMIT,
    // and replaying init on the same handle alone doesn't help — so try cheap
    // escalating primes (re-claiming the camera interface) before the caller
    // falls back to a full device re-open. The log says which prime worked.
    let primes = ["init+commit", "reclaim iface2", "reinit + reclaim iface2"];
    let mut frame_size = CAM_WIDTH * CAM_HEIGHT;
    let mut started = false;
    for (pass, label) in primes.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        match pass {
            0 => replay_init(&mut h, &packets, stop)?,
            1 => {
                let _ = h.release_interface(IFACE_VIDEO);
                thread::sleep(Duration::from_millis(50));
                h.claim_interface(IFACE_VIDEO)
                    .context("re-claim video interface")?;
            }
            _ => {
                replay_init(&mut h, &packets, stop)?;
                let _ = h.release_interface(IFACE_VIDEO);
                thread::sleep(Duration::from_millis(50));
                h.claim_interface(IFACE_VIDEO)
                    .context("re-claim video interface")?;
            }
        }
        frame_size = uvc_negotiate(&mut h, None)?;
        thread::sleep(Duration::from_millis(80));
        let _ = h.clear_halt(EP_VIDEO);
        if probe_camera(&mut h, Duration::from_millis(1500)) {
            println!("camera streaming via prime: {label}");
            started = true;
            break;
        }
        println!("camera silent after prime: {label}");
    }
    if !started {
        anyhow::bail!(StreamStartupTimeout);
    }

    read_camera_frames(&mut h, frame_size, Some(2), |frame| {
        if stop.load(Ordering::Relaxed) {
            return Ok(false);
        }
        if recenter.swap(false, Ordering::Relaxed) {
            tracker.recenter();
        }
        if let Some(p) = tracker.process(frame, CAM_WIDTH, CAM_HEIGHT)? {
            let _ = tx.send(Sample::Pose(PoseSample {
                timestamp_us: now_us() as i64,
                pos_cm: [p[0], p[1], p[2]],
                rot_deg: [p[3], p[4], p[5]],
            }));
        }
        Ok(true)
    })
}

/// Drive the 0x83 processed stream for the FFI/library engine: own the device,
/// decode gaze + presence, and push gaze samples to `tx` until `stop` is set.
/// The camera is *not* claimed here — doing so would throttle this stream.
///
/// Same flaky-init story as the camera path (see `run_pose_engine`): reset to a
/// baseline, then retry the full init if the 0x83 stream doesn't start.
pub(crate) fn run_gaze_engine(stop: &Arc<AtomicBool>, tx: &Sender<Sample>) -> Result<()> {
    let ctx = UsbContext::new()?;
    reset_device_baseline(&ctx);

    for attempt in 1..=MAX_REPLAY_ATTEMPTS {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        match gaze_engine_attempt(&ctx, stop, tx) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < MAX_REPLAY_ATTEMPTS && !stop.load(Ordering::Relaxed) => {
                eprintln!(
                    "tobii gaze init attempt {attempt}/{MAX_REPLAY_ATTEMPTS} failed: {e}; \
                     redoing full init"
                );
                thread::sleep(Duration::from_millis(700));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// One open+init+read of the 0x83 stream, pushing gaze samples until `stop` or a
/// failure. Bails with `StreamStartupTimeout` if no stream packet arrives in 5s.
fn gaze_engine_attempt(
    ctx: &UsbContext,
    stop: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
) -> Result<()> {
    const INIT_PACKETS: &str = include_str!("../init_packets_ep.txt");
    let mut h = open_tobii(ctx)?;
    vendor_control_init(&mut h)?;
    let packets = parse_init_packets(INIT_PACKETS)?;
    let mut no_log: Option<PacketLog> = None;
    for pkt in &packets {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let expected_seq = if marker(&pkt.data) == Some(0x51) {
            seq(&pkt.data)
        } else {
            None
        };
        h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000))
            .context("init packet write failed")?;
        if let Some(s) = expected_seq {
            let _ = wait_for_response_seq(&mut h, s, &mut no_log);
        }
        thread::sleep(Duration::from_millis(2));
    }

    let mut buf = vec![0u8; 16384];
    let mut packet_no = 0u64;
    let mut delivered = 0u64;
    let t_start = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if delivered == 0 && t_start.elapsed().as_secs_f64() > 5.0 {
            anyhow::bail!(StreamStartupTimeout);
        }
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(500)) {
            Ok(n) if n > 0 => {
                let data = &buf[..n];
                if marker(data) != Some(0x53) {
                    continue;
                }
                let decoded = decode_stream_payload(data)?;
                if decoded.is_empty() {
                    continue;
                }
                let frame = TrackingFrame::from_decoded(packet_no, &decoded);
                packet_no += 1;
                delivered += 1;
                let _ = tx.send(Sample::Gaze(GazeSample {
                    timestamp_us: frame.ts_us as i64,
                    gaze_valid: frame.gaze_valid,
                    gaze_norm: [
                        frame.gaze_norm_x.unwrap_or(0.0),
                        frame.gaze_norm_y.unwrap_or(0.0),
                    ],
                    present: frame.gaze_valid,
                }));
            }
            Ok(_) => {}
            Err(rusb::Error::Timeout) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Diagnostic: after the normal init (which starts the 0x83 processed stream),
/// measure 0x83 alone, then bring up the camera (UVC) and check whether 0x83
/// (gaze/head) and 0x82 (camera) deliver data *concurrently*. This is the gate
/// for fusing the fast 0x83 translation with camera rotation.
pub(crate) fn run_probe(opts: &Options) -> Result<()> {
    let Command::Probe { init_path } = &opts.command else {
        unreachable!();
    };

    let ctx = UsbContext::new()?;
    let mut h = open_tobii(&ctx)?;
    vendor_control_init(&mut h)?;
    let packets = read_init_packets(init_path)?;
    println!("Replaying {} init packets...", packets.len());
    let mut no_log: Option<PacketLog> = None;
    for pkt in &packets {
        let expected_seq = if marker(&pkt.data) == Some(0x51) {
            seq(&pkt.data)
        } else {
            None
        };
        if h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000)).is_err() {
            break;
        }
        if let Some(s) = expected_seq {
            let _ = wait_for_response_seq(&mut h, s, &mut no_log);
        }
        thread::sleep(Duration::from_millis(2));
    }

    // Phase A: 0x83 only (camera not yet streaming).
    let (a_pkts, a_stream, a_bytes) = sample_0x83(&h, Duration::from_secs(2));
    let a_secs = 2.0;
    println!(
        "\n[A] 0x83 only:    {a_pkts} packets ({a_stream} stream/0x53), {a_bytes} bytes, {:.1} pkt/s",
        a_pkts as f64 / a_secs
    );

    // Bring up the camera.
    if h.kernel_driver_active(IFACE_VIDEO).unwrap_or(false) {
        let _ = h.detach_kernel_driver(IFACE_VIDEO);
    }
    h.claim_interface(IFACE_VIDEO)
        .context("failed to claim video streaming interface 2")?;
    let frame_size = uvc_negotiate(&mut h, None)?;
    thread::sleep(Duration::from_millis(100));
    let _ = h.clear_halt(EP_VIDEO);
    println!("Camera up (frame buffer {frame_size} bytes). Reading both endpoints for 6s...");

    // Phase B: one dedicated reader thread per endpoint (a shared DeviceHandle
    // is Send+Sync), so the low-rate 0x83 stream isn't starved by the camera.
    let h = Arc::new(h);
    let stop = Arc::new(AtomicBool::new(false));
    let cam = {
        let h = h.clone();
        let stop = stop.clone();
        let cap = frame_size.max(65536);
        thread::spawn(move || {
            let mut buf = vec![0u8; cap];
            let (mut n, mut b) = (0u64, 0u64);
            while !stop.load(Ordering::Relaxed) {
                if let Ok(k) = h.read_bulk(EP_VIDEO, &mut buf, Duration::from_millis(200)) {
                    if k > 0 {
                        n += 1;
                        b += k as u64;
                    }
                }
            }
            (n, b)
        })
    };

    let mut buf83 = vec![0u8; 16384];
    let (mut n83, mut s83, mut b83) = (0u64, 0u64, 0u64);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(6) {
        if let Ok(n) = h.read_bulk(EP_IN, &mut buf83, Duration::from_millis(200)) {
            if n > 0 {
                n83 += 1;
                b83 += n as u64;
                if marker(&buf83[..n]) == Some(0x53) {
                    s83 += 1;
                }
            }
        }
    }
    let dt = start.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let (n82, b82) = cam.join().unwrap_or((0, 0));
    println!(
        "[B] 0x83 (gaze):  {n83} packets ({s83} stream/0x53), {b83} bytes, {:.1} pkt/s",
        n83 as f64 / dt
    );
    println!(
        "[B] 0x82 (cam):   {n82} reads, {b82} bytes, {:.1} read/s",
        n82 as f64 / dt
    );

    let a_rate = a_pkts as f64 / a_secs;
    let b_rate = n83 as f64 / dt;
    println!("\n=== verdict ===");
    if n82 == 0 {
        println!("camera silent -> UVC/camera not delivering here.");
    } else if b_rate >= a_rate * 0.5 {
        println!(
            "0x83 keeps {b_rate:.1}/{a_rate:.1} pkt/s with the camera on -> concurrent fusion is viable."
        );
    } else if b_rate < a_rate * 0.2 {
        println!(
            "0x83 throttled to {b_rate:.1} pkt/s (was {a_rate:.1}) once the camera streams -> the device \
             will not run the processed pipeline and raw camera at the same time. Concurrent fusion is NOT viable; \
             camera-mode and 0x83-mode are mutually exclusive."
        );
    } else {
        println!(
            "0x83 partly throttled: {b_rate:.1} pkt/s (was {a_rate:.1}) -> degraded but not dead."
        );
    }
    Ok(())
}

/// Read EP 0x83 for `dur`, returning (packets, stream-marked, bytes).
fn sample_0x83(h: &rusb::DeviceHandle<UsbContext>, dur: Duration) -> (u64, u64, u64) {
    let mut buf = vec![0u8; 16384];
    let (mut n, mut s, mut b) = (0u64, 0u64, 0u64);
    let start = Instant::now();
    while start.elapsed() < dur {
        if let Ok(len) = h.read_bulk(EP_IN, &mut buf, Duration::from_millis(50)) {
            if len > 0 {
                n += 1;
                b += len as u64;
                if marker(&buf[..len]) == Some(0x53) {
                    s += 1;
                }
            }
        }
    }
    (n, s, b)
}

/// Run the UVC PROBE/COMMIT handshake on the video-streaming interface for
/// format 1 / frame 1 and return the negotiated `dwMaxVideoFrameSize`.
fn uvc_negotiate(h: &mut rusb::DeviceHandle<UsbContext>, interval: Option<u32>) -> Result<usize> {
    const SET_CUR: u8 = 0x01;
    const GET_CUR: u8 = 0x81;
    const GET_MIN: u8 = 0x82;
    const GET_MAX: u8 = 0x83;
    const GET_LEN: u8 = 0x85;
    const GET_DEF: u8 = 0x87;
    const PROBE: u16 = 0x0100; // VS_PROBE_CONTROL << 8
    const COMMIT: u16 = 0x0200; // VS_COMMIT_CONTROL << 8
    const RT_OUT: u8 = 0x21; // class, interface, host->device
    const RT_IN: u8 = 0xA1;
    let index = IFACE_VIDEO as u16;
    let timeout = Duration::from_millis(1000);

    let mut len_buf = [0u8; 2];
    let len = match h.read_control(RT_IN, GET_LEN, PROBE, index, &mut len_buf, timeout) {
        Ok(_) => u16::from_le_bytes(len_buf) as usize,
        Err(_) => 0,
    };
    let len = if (26..=64).contains(&len) { len } else { 34 };

    // Probe the supported frame-interval range. The descriptor only lists 24fps,
    // but the device may accept faster via the PROBE control. dwFrameInterval is
    // in 100ns units at offset 4; smaller = faster.
    let interval_of = |req: u8| -> Option<u32> {
        let mut b = vec![0u8; len];
        h.read_control(RT_IN, req, PROBE, index, &mut b, timeout)
            .ok()
            .filter(|_| b.len() >= 8)
            .map(|_| u32::from_le_bytes(b[4..8].try_into().unwrap()))
    };
    let fps = |iv: u32| if iv > 0 { 1e7 / iv as f64 } else { 0.0 };
    // Request the smallest (fastest) interval the device reports unless
    // overridden on the CLI. (The device caps actual streaming at ~24fps total
    // / 8fps per camera regardless, but we ask for its best.)
    let want_iv = interval.unwrap_or_else(|| {
        [interval_of(GET_MIN), interval_of(GET_MAX), interval_of(GET_DEF)]
            .into_iter()
            .flatten()
            .filter(|&v| v > 0)
            .min()
            .unwrap_or(416_667)
    });

    let mut ctrl = vec![0u8; len];
    // Seed from the device's current/default probe settings, then pin the
    // format, frame and the fastest interval.
    let _ = h.read_control(RT_IN, GET_CUR, PROBE, index, &mut ctrl, timeout);
    ctrl[2] = 1; // bFormatIndex
    ctrl[3] = 1; // bFrameIndex
    ctrl[4..8].copy_from_slice(&want_iv.to_le_bytes()); // dwFrameInterval

    h.write_control(RT_OUT, SET_CUR, PROBE, index, &ctrl, timeout)
        .context("UVC PROBE SET_CUR failed")?;
    let _ = h.read_control(RT_IN, GET_CUR, PROBE, index, &mut ctrl, timeout);
    let got_iv = u32::from_le_bytes(ctrl[4..8].try_into().unwrap());
    println!("UVC negotiated frame interval = {got_iv} ({:.1} fps total)", fps(got_iv));

    let frame_size = if ctrl.len() >= 22 {
        u32::from_le_bytes(ctrl[18..22].try_into().unwrap()) as usize
    } else {
        0
    };

    h.write_control(RT_OUT, SET_CUR, COMMIT, index, &ctrl, timeout)
        .context("UVC COMMIT SET_CUR failed")?;

    Ok(if frame_size == 0 {
        CAM_WIDTH * CAM_HEIGHT
    } else {
        frame_size
    })
}

/// Read the camera stream and hand each assembled frame (passing `type_filter`)
/// to `on_frame`. The callback returns `false` to stop. Frame assembly, the
/// Tobii/UVC header stripping and the type histogram live here; what to do with
/// a frame (PGM, FIFO, pose tracking) is the caller's.
fn read_camera_frames(
    h: &mut rusb::DeviceHandle<UsbContext>,
    frame_size: usize,
    type_filter: Option<u8>,
    mut on_frame: impl FnMut(&[u8]) -> Result<bool>,
) -> Result<()> {
    let mut buf = vec![0u8; 1024 * 1024];
    let mut frame: Vec<u8> = Vec::with_capacity(frame_size.max(CAM_WIDTH * CAM_HEIGHT) + 4096);
    let mut last_fid: Option<u8> = None;
    let mut timeouts = 0u32;
    let mut cur_type = 0u8;
    let mut type_hist = [0u64; 256];
    let mut typed = 0u64;
    let mut hist_printed = false;
    let mut delivered = 0u64;
    let t_start = Instant::now();

    loop {
        // Startup watchdog: a healthy init delivers matching frames within ~1s.
        // If none arrive in 5s the device came up half-initialised — bail with a
        // retryable error so the caller redoes the full init replay.
        if delivered == 0 && t_start.elapsed().as_secs_f64() > 5.0 {
            anyhow::bail!(StreamStartupTimeout);
        }

        match h.read_bulk(EP_VIDEO, &mut buf, Duration::from_millis(2000)) {
            Ok(0) => continue,
            Ok(n) => {
                timeouts = 0;

                // Each bulk read is one UVC payload: a 2-byte header
                // (buf[1]=bmHeaderInfo, bit0=FID, bit1=EOF) followed by image
                // bytes. The first payload of a frame inserts a 10-byte Tobii
                // header (magic `e8 03 00 00` at offset 4); buf[hle] is the
                // frame-type / camera id.
                let hle = (buf[0] as usize).min(n);
                let has_tobii = n >= 8 && buf[4..8] == TOBII_FRAME_MAGIC;
                let data_off = if has_tobii { hle + 10 } else { hle };
                let bfh = buf[1];
                let fid = bfh & 0x01;
                let eof = (bfh & 0x02) != 0;

                // Flush the previous frame on FID change (its type is `cur_type`).
                let boundary = last_fid.is_some() && last_fid != Some(fid);
                if boundary && !frame.is_empty() {
                    if type_filter.is_none_or(|t| t == cur_type) {
                        delivered += 1;
                        if !on_frame(&frame)? {
                            break;
                        }
                    }
                    frame.clear();
                }
                last_fid = Some(fid);

                if has_tobii && hle < n {
                    cur_type = buf[hle];
                    type_hist[cur_type as usize] += 1;
                    typed += 1;
                    if !hist_printed && typed >= 90 {
                        let dt = t_start.elapsed().as_secs_f64();
                        print_type_hist(&type_hist, typed as f64 / dt);
                        hist_printed = true;
                    }
                }
                if data_off < n {
                    frame.extend_from_slice(&buf[data_off..n]);
                }

                if eof && !frame.is_empty() {
                    let mut stop = false;
                    if type_filter.is_none_or(|t| t == cur_type) {
                        delivered += 1;
                        stop = !on_frame(&frame)?;
                    }
                    frame.clear();
                    last_fid = None;
                    if stop {
                        break;
                    }
                }
            }
            Err(rusb::Error::Timeout) => {
                timeouts += 1;
                println!("EP 0x82 read timeout ({timeouts}); camera may need different init/probe");
                if timeouts >= 10 {
                    anyhow::bail!("no camera data on EP 0x82 after {timeouts} timeouts");
                }
            }
            // A stalled bulk endpoint shows up as Pipe (and sometimes Io); clear
            // the halt and keep trying rather than giving up.
            Err(e @ (rusb::Error::Pipe | rusb::Error::Io)) => {
                timeouts += 1;
                println!("EP 0x82 {e:?}; clearing halt and retrying ({timeouts})");
                let _ = h.clear_halt(EP_VIDEO);
                thread::sleep(Duration::from_millis(50));
                if timeouts >= 10 {
                    anyhow::bail!("EP 0x82 kept failing ({e:?}) after {timeouts} retries");
                }
            }
            Err(e) => anyhow::bail!("EP 0x82 read error: {e:?}"),
        }
    }
    Ok(())
}

fn print_type_hist(hist: &[u64; 256], fps_total: f64) {
    let total: u64 = hist.iter().sum();
    let parts: Vec<String> = hist
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(t, &c)| format!("type {t}={}%", 100 * c / total.max(1)))
        .collect();
    println!(
        "Camera: {fps_total:.0} fps total ({}); face camera is usually type 2 (--type 2).",
        parts.join(", ")
    );
}

/// Emit one assembled frame: a fixed 560x560 blob to the FIFO (padded/truncated
/// so the CV consumer gets constant-size frames), otherwise a numbered PGM.
fn emit_frame(
    fifo: &mut Option<BufWriter<File>>,
    out_prefix: &str,
    index: usize,
    data: &[u8],
) -> Result<()> {
    if let Some(w) = fifo {
        let need = CAM_WIDTH * CAM_HEIGHT;
        if data.len() >= need {
            w.write_all(&data[..need])?;
        } else {
            w.write_all(data)?;
            w.write_all(&vec![0u8; need - data.len()])?;
        }
        w.flush()?;
    } else {
        save_pgm(out_prefix, index, data)?;
    }
    Ok(())
}

fn save_pgm(prefix: &str, index: usize, data: &[u8]) -> Result<()> {
    // Render at the sensor width and however many whole rows the payload fills.
    let height = (data.len() / CAM_WIDTH).max(1);
    let used = CAM_WIDTH * height;
    let path = format!("{prefix}{index:03}.pgm");
    let mut f =
        BufWriter::new(File::create(&path).with_context(|| format!("failed to create {path}"))?);
    write!(f, "P5\n{CAM_WIDTH} {height}\n255\n")?;
    f.write_all(&data[..used.min(data.len())])?;
    f.flush()?;
    println!("  wrote {path} ({} payload bytes, {CAM_WIDTH}x{height})", data.len());
    Ok(())
}
