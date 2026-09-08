use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::ToSocketAddrs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use std::{error, fmt, thread};

use crate::cli::{Command, Options};
use crate::dashboard::render_dashboard_status;
use crate::decode::{decode_stream_payload, TrackingFrame};
use crate::engine::{GazeSample, PoseSample, Sample};
use crate::image83::{decode_image_payload, upscale2x, write_pgm, ImageFrame};
use crate::opentrack::OpentrackUdp;
use crate::protocol::{
    declared_len, marker, parse_init_packets, read_init_packets, seq, stream_id,
    stream_start_packet, stream_stop_packet, BulkReassembler, InitPacket, STREAM_ID_GAZE,
    STREAM_ID_IMAGE, STREAM_ID_PRESENCE,
};
use crate::sinks::{handle_live_decoded, log_packet, now_us, DecodedCsv, JsonlOutput, PacketLog};

pub(crate) const VID: u16 = 0x2104;

pub(crate) const PID: u16 = 0x0313;

pub(crate) const IFACE: u8 = 0;

pub(crate) const EP_IN: u8 = 0x83;

/// Bulk OUT endpoint for command messages (what the init replay writes to).
pub(crate) const EP_OUT: u8 = 0x05;

/// One read must hold the largest multiplexed message on EP 0x83: the 0x50e
/// image stream is 78609 bytes. Every message ends in a short USB packet, so a
/// buffer this size normally returns exactly one whole message per read.
pub(crate) const READ_BUF: usize = 128 * 1024;

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

/// Mirror of the Windows Tobii Experience shutdown (captured in
/// `tobii-sys/teardown.pcapng`): a single vendor control OUT, request 66,
/// no data, which stops the 0x83 processed stream firmware-side. Without it the
/// device is left streaming and comes up in an undefined state on the next open
/// (the "works every other start" flakiness). Best-effort: errors are ignored
/// because we run it on the teardown path where the handle is about to drop.
pub(crate) fn vendor_control_deinit(h: &mut rusb::DeviceHandle<UsbContext>) {
    let timeout = Duration::from_millis(500);
    // Quiet on success: this also runs on each cold-start prime re-open, where a
    // "stop stream written" line would just be confusing noise.
    if let Err(e) = h.write_control(0x41, 66, 0, 0, &[], timeout) {
        eprintln!("tobii deinit: control OUT 66 failed: {e}");
    }
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

/// Drive the 0x83 streams for the daemon engine: own the device, decode gaze +
/// presence and the 0x50e IR image stream (head pose), and push samples to `tx`
/// until `stop` is set. The UVC camera is *not* claimed here — doing so would
/// throttle the 0x83 streams.
///
/// A cold gaze start normally arms on the *second* fresh open (the first accepts
/// the init but doesn't stream — a firmware quirk, not stale state). A USB reset
/// up front doesn't change that (verified: start behaves identically with and
/// without it), so we no longer pay its re-enumeration cost on every start. It's
/// kept only as a recovery escalation once several opens in a row fail to arm —
/// which is the signature of a device left hot by an unclean exit (`kill -9`,
/// crash) that skipped the teardown. `reset_device_baseline` still honors
/// `TOBII_NO_RESET=1` (skip even the escalation).
const RESET_ESCALATION_ATTEMPT: usize = 3;

pub(crate) fn run_gaze_engine(
    stop: &Arc<AtomicBool>,
    recenter: &Arc<AtomicBool>,
    head_wanted: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
) -> Result<()> {
    let ctx = UsbContext::new()?;

    // Head pose from the 0x50e image stream runs on its own thread so a slow
    // inference never blocks the USB reader (an unread IN buffer stalls the
    // firmware). The reader drops each new frame into a single-slot mailbox;
    // the worker always takes the newest one and skips whatever it missed.
    let mailbox: PoseMailbox = Arc::new((Mutex::new(None), Condvar::new()));
    // No worker (and no ONNX session) when the image stream is disabled.
    let worker = image_stream_enabled().then(|| {
        let mailbox = mailbox.clone();
        let stop = stop.clone();
        let recenter = recenter.clone();
        let head_wanted = head_wanted.clone();
        let tx = tx.clone();
        thread::spawn(move || pose_worker(mailbox, stop, recenter, head_wanted, tx))
    });

    let result = (|| {
        for attempt in 1..=MAX_REPLAY_ATTEMPTS {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            // Normal cold start arms by attempt 2 and never gets here; reaching the
            // escalation means the device is genuinely stuck — reset to recover.
            if attempt == RESET_ESCALATION_ATTEMPT {
                println!("gaze: stream still not arming after {} opens; USB-reset to recover", attempt - 1);
                reset_device_baseline(&ctx);
            }
            match gaze_engine_attempt(&ctx, stop, tx, &mailbox) {
                Ok(()) => return Ok(()),
                Err(e) if attempt < MAX_REPLAY_ATTEMPTS && !stop.load(Ordering::Relaxed) => {
                    // A cold device accepts the first init but doesn't start
                    // streaming; a fresh re-open is what arms it (expected, not an
                    // error). Report real failures loudly, the prime quietly.
                    if e.downcast_ref::<StreamStartupTimeout>().is_some() {
                        println!(
                            "gaze: stream not armed on open {attempt}; re-opening to prime \
                             (cold-start quirk)"
                        );
                    } else {
                        eprintln!("tobii gaze init attempt {attempt} failed: {e}; re-opening");
                    }
                    thread::sleep(Duration::from_millis(700));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    })();

    // Wake the worker so it notices `stop` (the caller sets it before joining
    // us; on an internal failure we set it ourselves so the worker exits too).
    stop.store(true, Ordering::Relaxed);
    mailbox.1.notify_all();
    if let Some(worker) = worker {
        let _ = worker.join();
    }
    result
}

/// Single-slot hand-off of the newest image frame to the pose worker.
pub(crate) type PoseMailbox = Arc<(Mutex<Option<ImageFrame>>, Condvar)>;

fn mailbox_put(mailbox: &PoseMailbox, frame: ImageFrame) {
    let (lock, cv) = &**mailbox;
    *lock.lock().unwrap() = Some(frame);
    cv.notify_one();
}

/// Head-pose inference loop over 0x50e frames (see `run_gaze_engine`).
fn pose_worker(
    mailbox: PoseMailbox,
    stop: Arc<AtomicBool>,
    recenter: Arc<AtomicBool>,
    head_wanted: Arc<AtomicBool>,
    tx: Sender<Sample>,
) {
    let mut tracker = match crate::track::Tracker::new_image83() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("image83 pose worker disabled: {e:?}");
            return;
        }
    };
    // `TOBII_IMAGE83_DEBUG=1`: report frames taken / poses / inference time.
    let debug = std::env::var("TOBII_IMAGE83_DEBUG").is_ok_and(|v| !v.is_empty() && v != "0");
    let (mut taken, mut posed, mut infer_us) = (0u64, 0u64, 0u64);
    let mut last_report = Instant::now();
    // The tracker outlives head clients (a gaze client keeps the engine
    // alive), so a new head subscriber must not inherit an old rest pose or
    // smoothing state: recalibrate on every off->on edge after the first.
    let mut was_wanted: Option<bool> = None;
    let (lock, cv) = &*mailbox;
    while !stop.load(Ordering::Relaxed) {
        let frame = {
            let mut slot = lock.lock().unwrap();
            while slot.is_none() && !stop.load(Ordering::Relaxed) {
                slot = cv.wait_timeout(slot, Duration::from_millis(200)).unwrap().0;
            }
            slot.take()
        };
        let Some(frame) = frame else { continue };
        if recenter.swap(false, Ordering::Relaxed) {
            tracker.recenter();
        }
        let wanted = head_wanted.load(Ordering::Relaxed);
        if was_wanted == Some(false) && wanted {
            tracker.recenter();
        }
        was_wanted = Some(wanted);
        if !wanted {
            continue;
        }
        let t0 = Instant::now();
        let big = upscale2x(&frame.pixels, frame.width, frame.height);
        match tracker.process(&big, frame.width * 2, frame.height * 2) {
            Ok(Some(p)) => {
                posed += 1;
                let _ = tx.send(Sample::Pose(PoseSample {
                    timestamp_us: now_us() as i64,
                    pos_cm: [p[0], p[1], p[2]],
                    rot_deg: [p[3], p[4], p[5]],
                }));
            }
            Ok(None) => {}
            Err(e) => eprintln!("image83 pose: {e}"),
        }
        taken += 1;
        infer_us += t0.elapsed().as_micros() as u64;
        if debug && last_report.elapsed() >= Duration::from_secs(5) {
            let dt = last_report.elapsed().as_secs_f64();
            println!(
                "image83 pose worker: {:.1} frames/s processed, {:.1} poses/s, mean {:.1} ms per frame",
                taken as f64 / dt,
                posed as f64 / dt,
                infer_us as f64 / 1000.0 / taken.max(1) as f64
            );
            taken = 0;
            posed = 0;
            infer_us = 0;
            last_report = Instant::now();
        }
    }
}

/// One open+init+read of the 0x83 stream, pushing gaze samples until `stop` or a
/// failure. Bails with `StreamStartupTimeout` if no stream packet arrives in 5s.
fn gaze_engine_attempt(
    ctx: &UsbContext,
    stop: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
) -> Result<()> {
    let mut h = open_tobii(ctx)?;
    vendor_control_init(&mut h)?;
    // From here the 0x83 stream is running firmware-side (started by request 65
    // inside vendor_control_init). Guarantee the Windows-style stop (request 66)
    // on every exit — clean stop, error, or timeout — so the device isn't left
    // mid-stream and the next open starts from a defined state.
    let result = gaze_stream_loop(&mut h, stop, tx, mailbox);
    vendor_control_deinit(&mut h);
    result
}

/// Replay the init-packet sequence once on `h`.
fn replay_gaze_init(
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

/// Read 0x83 for up to `dur`, returning true as soon as a decodable GAZE
/// (0x500) frame arrives. 0x52 handshake/config responses and any image /
/// presence messages are drained and ignored — an image frame must not count
/// as proof the gaze pipeline armed (the cold-start quirk leaves gaze silent
/// while a freshly requested 0x50e stream may run). Draining matters: an
/// unread IN buffer can stall the firmware. `asm` is shared with the pump so
/// a partially read message is not lost at the hand-off.
fn wait_for_gaze_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &Arc<AtomicBool>,
    dur: Duration,
    asm: &mut BulkReassembler,
) -> bool {
    let mut buf = vec![0u8; READ_BUF];
    let start = Instant::now();
    while start.elapsed() < dur && !stop.load(Ordering::Relaxed) {
        if let Ok(n) = h.read_bulk(EP_IN, &mut buf, Duration::from_millis(200)) {
            for msg in asm.push(&buf[..n]) {
                if stream_id(&msg) == Some(STREAM_ID_GAZE)
                    && decode_stream_payload(&msg).is_ok_and(|d| !d.is_empty())
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Longest the gaze stream may stay silent while running before the attempt
/// is torn down and the device re-opened (a stream that never armed, or died).
const GAZE_LIVENESS_TIMEOUT: Duration = Duration::from_secs(5);

/// `TOBII_NO_IMAGE=1` keeps the 0x50e image stream off (gaze only, as before).
fn image_stream_enabled() -> bool {
    !std::env::var("TOBII_NO_IMAGE")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// The command sequence number to use after the init replay: one past the
/// highest 0x51 seq in the replayed packets (the device echoes it in the 0x52
/// response, which is how we match replies).
fn next_command_seq(packets: &[InitPacket]) -> u32 {
    packets
        .iter()
        .filter(|p| marker(&p.data) == Some(0x51))
        .filter_map(|p| seq(&p.data))
        .max()
        .map_or(0x100, |s| s + 1)
}

/// Send one command message and wait (briefly) for its 0x52 response. Stream
/// messages that arrive meanwhile are discarded, so this is only for the few
/// commands sent around stream start/stop. Uses a full-size read buffer since
/// a 78 KB image message may already be in flight.
fn send_command(
    h: &mut rusb::DeviceHandle<UsbContext>,
    packet: &[u8],
    deadline: Duration,
) -> Result<()> {
    let expected = seq(packet).context("command packet without seq")?;
    h.write_bulk(EP_OUT, packet, Duration::from_millis(2000))
        .context("command write failed")?;
    let mut buf = vec![0u8; READ_BUF];
    let start = Instant::now();
    while start.elapsed() < deadline {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(200)) {
            Ok(n) if marker(&buf[..n]) == Some(0x52) && seq(&buf[..n]) == Some(expected) => {
                return Ok(());
            }
            Ok(_) | Err(rusb::Error::Timeout) => {}
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("no response to command seq {expected} within {deadline:?}")
}

pub(crate) fn start_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    cmd_seq: u32,
    id: u32,
) -> Result<()> {
    send_command(h, &stream_start_packet(cmd_seq, id), Duration::from_millis(1500))
        .with_context(|| format!("start stream {id:#x}"))
}

pub(crate) fn stop_stream(h: &mut rusb::DeviceHandle<UsbContext>, cmd_seq: u32, id: u32) -> Result<()> {
    send_command(h, &stream_stop_packet(cmd_seq, id), Duration::from_millis(700))
        .with_context(|| format!("stop stream {id:#x}"))
}

/// Replay the init packets, start the image stream, then pump 0x83 until `stop`
/// or failure. Split out from `gaze_engine_attempt` so the caller can always
/// run the device teardown after this returns, regardless of how it exits.
///
/// Cold-start note: the 0x53 gaze stream reliably arms only on a *fresh* open —
/// the first open after the device goes cold accepts the init but never starts
/// streaming, and re-running the init on the same handle does not help (unlike
/// the camera path; verified empirically). So if the stream doesn't arm here we
/// bail with `StreamStartupTimeout` and let the caller re-open, which is what
/// actually primes it.
fn gaze_stream_loop(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
) -> Result<()> {
    const INIT_PACKETS: &str = include_str!("../init_packets_ep.txt");
    let packets = parse_init_packets(INIT_PACKETS)?;
    replay_gaze_init(h, &packets, stop)?;
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    // The Windows Stream Engine subscribes the image stream right after its
    // init; we do the same. Failure here is not fatal — gaze still works.
    let mut cmd_seq = next_command_seq(&packets);
    let mut image = image_stream_enabled();
    if image {
        match start_stream(h, cmd_seq, STREAM_ID_IMAGE) {
            Ok(()) => println!("gaze: image stream 0x50e requested (head pose via IR frames)"),
            Err(e) => {
                eprintln!("gaze: image stream start failed ({e}); continuing gaze-only");
                image = false;
            }
        }
        cmd_seq += 1;
    }
    // The device's first 0x53 frame lands ~3s after a good init; give it margin.
    let mut asm = BulkReassembler::new();
    if !wait_for_gaze_stream(h, stop, Duration::from_secs_f64(4.5), &mut asm) {
        if image {
            let _ = stop_stream(h, cmd_seq, STREAM_ID_IMAGE);
        }
        anyhow::bail!(StreamStartupTimeout);
    }

    let result = pump_streams(h, stop, tx, mailbox, &mut asm);
    if image {
        // Mirror the Windows shutdown (1230 for 0x50e before the vendor stop).
        let _ = stop_stream(h, cmd_seq, STREAM_ID_IMAGE);
    }
    result
}

/// Demultiplex EP 0x83: gaze frames -> `GazeSample`s, image frames -> the pose
/// worker's mailbox, everything else (presence 0x504, responses) ignored.
fn pump_streams(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &Arc<AtomicBool>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
    asm: &mut BulkReassembler,
) -> Result<()> {
    let mut buf = vec![0u8; READ_BUF];
    let mut packet_no = 0u64;
    let mut image_live = false;
    let mut last_gaze = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if last_gaze.elapsed() > GAZE_LIVENESS_TIMEOUT {
            eprintln!("gaze: no gaze frame for {GAZE_LIVENESS_TIMEOUT:?}; re-opening the device");
            anyhow::bail!(StreamStartupTimeout);
        }
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(500)) {
            Ok(n) if n > 0 => {
                for msg in asm.push(&buf[..n]) {
                    match stream_id(&msg) {
                        Some(STREAM_ID_GAZE) => {
                            let decoded = decode_stream_payload(&msg)?;
                            if decoded.is_empty() {
                                continue;
                            }
                            last_gaze = Instant::now();
                            let frame = TrackingFrame::from_decoded(packet_no, &decoded);
                            packet_no += 1;
                            let _ = tx.send(Sample::Gaze(GazeSample {
                                timestamp_us: frame.ts_us as i64,
                                gaze_valid: frame.gaze_valid,
                                gaze_norm: [
                                    frame.gaze_norm_x.unwrap_or(0.0),
                                    frame.gaze_norm_y.unwrap_or(0.0),
                                ],
                                present: frame.gaze_valid,
                                pupil_mm: [
                                    frame.pupil_left.unwrap_or(f64::NAN),
                                    frame.pupil_right.unwrap_or(f64::NAN),
                                ],
                            }));
                        }
                        Some(STREAM_ID_IMAGE) => {
                            if let Some(frame) = decode_image_payload(&msg) {
                                if !image_live {
                                    println!(
                                        "gaze: image stream 0x50e live ({}x{})",
                                        frame.width, frame.height
                                    );
                                    image_live = true;
                                }
                                mailbox_put(mailbox, frame);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(_) => {}
            Err(rusb::Error::Timeout) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Offline: run the image head tracker over a TBI5LOG1 log that contains the
/// 0x50e stream (e.g. `image83 --log`, or `import-tsv` of a Windows capture)
/// and pair each pose with the latest 0x83 gaze-frame head anchors. Writes a
/// CSV for analysis and prints a summary (used to validate sign/scale against
/// the MediaPipe reference and the 0x83 eyeball-centre translation).
pub(crate) fn run_image83_replay(path: &str, csv: Option<&str>) -> Result<()> {
    use crate::protocol::read_log_payloads;
    let payloads = read_log_payloads(path)?;
    let mut tracker = crate::track::Tracker::new_image83()?;
    let mut asm = BulkReassembler::new();
    let mut out = csv
        .map(|p| File::create(p).map(BufWriter::new).with_context(|| format!("create {p}")))
        .transpose()?;
    if let Some(o) = out.as_mut() {
        writeln!(
            o,
            "image_idx,device_ts_us,face,raw_tx_cm,raw_ty_cm,raw_tz_cm,raw_pitch,raw_yaw,raw_roll,\
             rel_tx_cm,rel_ty_cm,rel_tz_cm,rel_yaw,rel_pitch,rel_roll,\
             g_head_x_mm,g_head_y_mm,g_head_z_mm,g_head_roll_deg,g_valid"
        )?;
    }
    let mut last_gaze: Option<TrackingFrame> = None;
    let (mut images, mut faces, mut poses, mut gaze_frames) = (0u64, 0u64, 0u64, 0u64);
    let mut yaw_vs_x: Vec<(f64, f64)> = Vec::new();
    for payload in &payloads {
        for msg in asm.push(payload) {
            match stream_id(&msg) {
                Some(STREAM_ID_GAZE) => {
                    let decoded = decode_stream_payload(&msg)?;
                    if !decoded.is_empty() {
                        last_gaze = Some(TrackingFrame::from_decoded(gaze_frames, &decoded));
                        gaze_frames += 1;
                    }
                }
                Some(STREAM_ID_IMAGE) => {
                    let Some(frame) = decode_image_payload(&msg) else { continue };
                    let big = upscale2x(&frame.pixels, frame.width, frame.height);
                    let rel = tracker.process(&big, frame.width * 2, frame.height * 2)?;
                    let raw = tracker.last_raw();
                    faces += raw.is_some() as u64;
                    poses += rel.is_some() as u64;
                    let g = last_gaze.as_ref();
                    if let (Some(r), Some(g)) = (raw, g) {
                        if let (Some(x), true) = (g.head_x, g.head_z.is_some()) {
                            yaw_vs_x.push((r[4], x / 1000.0));
                        }
                    }
                    if let Some(o) = out.as_mut() {
                        let f = |v: Option<f64>| v.map_or(String::from(""), |v| format!("{v:.3}"));
                        let r6 = |v: Option<[f64; 6]>, i: usize| f(v.map(|a| a[i]));
                        writeln!(
                            o,
                            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                            images,
                            frame.device_ts_us,
                            raw.is_some() as u8,
                            r6(raw, 0), r6(raw, 1), r6(raw, 2), r6(raw, 3), r6(raw, 4), r6(raw, 5),
                            r6(rel, 0), r6(rel, 1), r6(rel, 2), r6(rel, 3), r6(rel, 4), r6(rel, 5),
                            f(g.and_then(|g| g.head_x).map(|v| v / 1000.0)),
                            f(g.and_then(|g| g.head_y).map(|v| v / 1000.0)),
                            f(g.and_then(|g| g.head_z).map(|v| v / 1000.0)),
                            f(g.and_then(|g| g.head_roll)),
                            g.map_or(0, |g| g.gaze_valid as u8),
                        )?;
                    }
                    images += 1;
                }
                _ => {}
            }
        }
    }
    println!(
        "{path}: {gaze_frames} gaze frames, {images} images, face found in {faces}, {poses} calibrated poses"
    );
    if yaw_vs_x.len() > 10 {
        let n = yaw_vs_x.len() as f64;
        let (my, mx) = (
            yaw_vs_x.iter().map(|p| p.0).sum::<f64>() / n,
            yaw_vs_x.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (y, x) in &yaw_vs_x {
            sxy += (y - my) * (x - mx);
            sxx += (x - mx) * (x - mx);
            syy += (y - my) * (y - my);
        }
        println!(
            "raw image yaw vs 0x83 head_x: r = {:.3}, slope {:.2} mm/deg (neck lever; expect ~-1.5 for +yaw = left)",
            sxy / (sxx * syy).sqrt(),
            sxy / syy
        );
    }
    Ok(())
}

/// Diagnostic: bring the device up in gaze mode, additionally start the 0x50e
/// image stream, and report what arrives on EP 0x83 for `secs` seconds: per-
/// stream message rates, image timestamp cadence, the first frames as PGM, and
/// optionally the head pose from each frame.
pub(crate) fn run_image83(opts: &Options) -> Result<()> {
    let Command::Image83 {
        init_path,
        secs,
        out_prefix,
        max_frames,
        log_path,
        pose,
        no_image,
    } = &opts.command
    else {
        unreachable!();
    };

    let ctx = UsbContext::new()?;
    let packets = read_init_packets(init_path)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut tracker = if *pose {
        Some(crate::track::Tracker::new_image83()?)
    } else {
        None
    };
    let mut log = log_path.as_deref().map(PacketLog::create).transpose()?;

    for attempt in 1..=3 {
        let mut h = open_tobii(&ctx)?;
        vendor_control_init(&mut h)?;
        println!("Replaying {} init packets (attempt {attempt})...", packets.len());
        replay_gaze_init(&mut h, &packets, &stop)?;
        let mut cmd_seq = next_command_seq(&packets);
        let image = !*no_image;
        if image {
            match start_stream(&mut h, cmd_seq, STREAM_ID_IMAGE) {
                Ok(()) => println!("image stream 0x50e start acknowledged (cmd 1220, seq {cmd_seq:#x})"),
                Err(e) => println!("image stream start: {e}"),
            }
            cmd_seq += 1;
        } else {
            println!("image stream NOT requested (--no-image baseline)");
        }
        let mut asm = BulkReassembler::new();
        if !wait_for_gaze_stream(&mut h, &stop, Duration::from_secs_f64(4.5), &mut asm) {
            println!("no gaze stream within 4.5 s on open {attempt}; re-opening (cold-start quirk)");
            if image {
                let _ = stop_stream(&mut h, cmd_seq, STREAM_ID_IMAGE);
            }
            vendor_control_deinit(&mut h);
            drop(h);
            thread::sleep(Duration::from_millis(700));
            continue;
        }

        // Everything that can fail while the streams run lives in this
        // closure, so the teardown below always runs (a failed PGM write must
        // not leave the device hot for the next start).
        let result = (|| -> Result<()> {
        let mut buf = vec![0u8; READ_BUF];
        let mut counts: std::collections::BTreeMap<u32, (u64, u64)> = Default::default();
        let mut other = 0u64;
        let mut saved = 0usize;
        let mut image_ts = Vec::<u64>::new();
        let mut gaze_ts = Vec::<u64>::new();
        let (mut gaze_frames, mut gaze_valid, mut eyes_valid) = (0u64, 0u64, 0u64);
        let mut poses = 0u64;
        let start = Instant::now();
        let dur = Duration::from_secs_f64(*secs);
        while start.elapsed() < dur {
            let n = match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(300)) {
                Ok(n) => n,
                Err(rusb::Error::Timeout) => continue,
                Err(e) => return Err(e.into()),
            };
            for msg in asm.push(&buf[..n]) {
                log_packet(&mut log, EP_IN, &msg)?;
                let Some(id) = stream_id(&msg) else {
                    other += 1;
                    continue;
                };
                let e = counts.entry(id).or_default();
                e.0 += 1;
                e.1 += msg.len() as u64;
                match id {
                    STREAM_ID_GAZE => {
                        if let Ok(v) = decode_stream_payload(&msg) {
                            if !v.is_empty() {
                                let f = TrackingFrame::from_decoded(gaze_frames, &v);
                                gaze_frames += 1;
                                gaze_valid += f.gaze_valid as u64;
                                eyes_valid += f.head_xyz().is_some() as u64;
                            }
                        }
                        gaze_ts.push(now_us());
                    }
                    STREAM_ID_IMAGE => {
                        let Some(frame) = decode_image_payload(&msg) else {
                            println!("image message of {} bytes did not decode", msg.len());
                            continue;
                        };
                        image_ts.push(frame.device_ts_us);
                        if saved < *max_frames {
                            let path = format!("{out_prefix}{saved:03}.pgm");
                            write_pgm(&path, &frame)?;
                            println!(
                                "saved {path} ({}x{}, device ts {} us)",
                                frame.width, frame.height, frame.device_ts_us
                            );
                            saved += 1;
                        }
                        if let Some(t) = tracker.as_mut() {
                            let big = upscale2x(&frame.pixels, frame.width, frame.height);
                            if let Some(p) = t.process(&big, frame.width * 2, frame.height * 2)? {
                                poses += 1;
                                if poses % 10 == 1 {
                                    println!(
                                        "pose: yaw {:+6.1} pitch {:+6.1} roll {:+6.1}  t = [{:+5.1} {:+5.1} {:+5.1}] cm",
                                        p[3], p[4], p[5], p[0], p[1], p[2]
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let dt = start.elapsed().as_secs_f64();
        println!("\n=== EP 0x83 over {dt:.1} s ===");
        for (id, (n, bytes)) in &counts {
            let name = match *id {
                STREAM_ID_GAZE => "gaze",
                STREAM_ID_IMAGE => "image",
                STREAM_ID_PRESENCE => "presence",
                _ => "?",
            };
            println!(
                "stream {id:#05x} {name:8} {n:6} msgs  {:7.1} msg/s  {:9} bytes",
                *n as f64 / dt,
                bytes
            );
        }
        println!("non-stream messages: {other}; reassembler pending {} bytes", asm.pending_len());
        if gaze_frames > 0 {
            println!(
                "gaze frames {gaze_frames}: gaze valid {:.1}%, both eyeball centres valid {:.1}%",
                gaze_valid as f64 * 100.0 / gaze_frames as f64,
                eyes_valid as f64 * 100.0 / gaze_frames as f64
            );
        }
        if image_ts.len() > 2 {
            let mut d: Vec<f64> = image_ts.windows(2).map(|w| (w[1] as f64 - w[0] as f64) / 1000.0).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "image device-ts interval: median {:.1} ms, p95 {:.1} ms, max {:.1} ms",
                d[d.len() / 2],
                d[d.len() * 95 / 100],
                d[d.len() - 1]
            );
        }
        if *pose {
            println!("head poses emitted: {poses} (first ~30 frames calibrate the rest pose)");
        }
        Ok(())
        })();
        if image {
            let _ = stop_stream(&mut h, cmd_seq, STREAM_ID_IMAGE);
        }
        vendor_control_deinit(&mut h);
        return result;
    }
    anyhow::bail!("stream never armed after 3 opens")
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
