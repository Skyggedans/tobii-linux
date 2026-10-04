//! Research and diagnostic subcommands driven from the CLI: the legacy
//! `replay` stream mode, the UVC IR camera path (`camera`, `track`, `probe`)
//! and the 0x50e image tools (`image83`, `image83-replay`).
//!
//! None of this is needed to run the driver — the daemon uses
//! [`tobii_usb::device`] only. Everything here shares that module's USB transport
//! helpers rather than duplicating them.

use anyhow::{Context, Result};
use rusb::Context as UsbContext;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::cli::{Command, Options};
use crate::dashboard::render_dashboard_status;
use crate::opentrack::OpentrackUdp;
use crate::sinks::{DecodedCsv, JsonlOutput, handle_live_decoded};
use tobii_proto::decode::{TrackingFrame, decode_stream_payload};
use tobii_proto::image83::{decode_image_payload, write_pgm};
use tobii_proto::log::{PacketLog, log_packet};
use tobii_proto::protocol::{
    BulkReassembler, InitPacket, STREAM_ID_GAZE, STREAM_ID_IMAGE, STREAM_ID_PRESENCE, declared_len,
    marker, read_init_packets, seq, stream_id,
};
use tobii_proto::time::now_us;
use tobii_usb::device::{
    EP_IN, MAX_REPLAY_ATTEMPTS, READ_BUF, ResponseEcho, StreamStartupTimeout, command_seq,
    next_command_seq, open_tobii, replay_init_packets, start_stream, stop_stream,
    vendor_control_deinit, vendor_control_init, wait_for_gaze_stream, wait_for_response_seq,
};

/// Consecutive 2 s read timeouts before the first stream packet that count
/// as "the stream never started" in the `replay` subcommand.
const STARTUP_STREAM_TIMEOUTS: u32 = 3;

// UVC IR camera (interface 2 / EP 0x82): 560x560 8-bit grayscale. Each bulk
// payload is [2-byte UVC header][image]; the first payload of a frame also
// carries a 10-byte Tobii header (`XX 00 e8 03 00 00` + u32 counter).
const IFACE_VIDEO: u8 = 2;

const EP_VIDEO: u8 = 0x82;

const CAM_WIDTH: usize = 560;

const CAM_HEIGHT: usize = 560;

const TOBII_FRAME_MAGIC: [u8; 4] = [0xe8, 0x03, 0x00, 0x00];

/// The `replay` subcommand: replay a captured init sequence, then dump the
/// EP `0x83` stream (optionally logging / decoding / forwarding it).
///
/// # Errors
///
/// Fails if the init file cannot be read, the device cannot be opened, or
/// the stream never starts within the configured attempts.
///
/// # Panics
///
/// Panics if `opts.command` is not `Command::Replay`; `lib.rs` dispatches
/// on the variant before calling this.
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

/// One `replay` attempt: open, control-init, replay `init_limit` packets
/// (printing the OUT/IN handshake), then read the stream until it ends.
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
                opts.opentrack_auto_decouple,
            )
        })
        .transpose()?;

    vendor_control_init(&mut h)?;

    for (i, pkt) in packets.iter().take(init_limit).enumerate() {
        let packet_no = i + 1;

        let expected_seq = command_seq(&pkt.data);

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
                println!("  written={written}");

                if let Some(expected_seq) = expected_seq
                    && let Err(e) =
                        wait_for_response_seq(&mut h, expected_seq, &mut log, ResponseEcho::Stdout)
                {
                    println!("  no response for seq {expected_seq}: {e}");
                }
            }
            Err(e) => {
                error!(packet = packet_no, error = ?e, "init packet write failed");
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
        opts,
        &mut log,
        &mut live_csv,
        &mut jsonl,
        &mut opentrack,
    )
}

/// Dump the EP `0x83` stream for the `replay` subcommand until
/// `--max-stream-packets` is reached, reconnecting on `NoDevice` if asked.
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

                if let Some(max) = opts.max_stream_packets
                    && read_count >= max
                {
                    println!("Reached --max-stream-packets={max}");
                    return Ok(());
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
                    "read error: NoDevice; USB device probably re-enumerated, reconnect attempt {reconnect_attempts}"
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
                println!("read error: {e:?}");
            }
        }
    }
}

/// Poll for up to 15 s for the device to re-enumerate and re-open it.
fn wait_for_reconnect(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
    for attempt in 1..=30 {
        thread::sleep(Duration::from_millis(500));
        match open_tobii(ctx) {
            Ok(h) => {
                info!(attempt, "reconnected");
                return Ok(h);
            }
            Err(e) => {
                debug!(attempt, error = %e, "reconnect attempt failed");
            }
        }
    }

    anyhow::bail!("Tobii did not reappear after 15 seconds")
}

/// Read and print up to `max_reads` pending IN messages (stops on the first
/// timeout); used mid-replay to flush the device's early responses.
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
                println!("  IN error: {e:?}");
                break;
            }
        }
    }
}

/// The `camera` subcommand: bring the device up, negotiate the UVC IR
/// stream and write frames as PGM files or to a FIFO.
///
/// # Errors
///
/// Fails if the device cannot be opened, the UVC handshake fails, or a
/// frame cannot be written.
///
/// # Panics
///
/// Panics if `opts.command` is not `Command::Camera`; `lib.rs` dispatches
/// on the variant before calling this.
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
        println!(
            "Replaying {} init packets to start the device...",
            packets.len()
        );
        if let Err(e) = replay_init_packets(&mut h, &packets, None) {
            warn!(error = format_args!("{e:#}"), "init replay aborted");
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
            println!(
                "Streaming raw {CAM_WIDTH}x{CAM_HEIGHT} frames to {path} (waiting for reader)..."
            );
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

/// The `track` subcommand: camera head tracking straight to an `OpenTrack`
/// UDP target (no daemon).
///
/// # Errors
///
/// Fails if the target does not resolve, the face model cannot be loaded,
/// or the device / camera cannot be brought up.
///
/// # Panics
///
/// Panics if `opts.command` is not `Command::Track`; `lib.rs` dispatches
/// on the variant before calling this.
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
    let mut tracker = tobii_pose::track::Tracker::new()?;

    let ctx = UsbContext::new()?;
    let mut h = open_tobii(&ctx)?;
    vendor_control_init(&mut h)?;
    if !*skip_replay {
        let packets = read_init_packets(init_path)?;
        println!("Replaying {} init packets...", packets.len());
        if let Err(e) = replay_init_packets(&mut h, &packets, None) {
            warn!(error = format_args!("{e:#}"), "init replay aborted");
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
            if frames.is_multiple_of(8) {
                eprint!(
                    "\ryaw/pit/roll={:+5.1}/{:+5.1}/{:+5.1}  tx/ty/tz={:+5.1}/{:+5.1}/{:+5.1}   ",
                    pose[3], pose[4], pose[5], pose[0], pose[1], pose[2]
                );
            }
        }
        Ok(true)
    })
}

/// Offline: run the image head tracker over a TBI5LOG1 log that contains the
/// 0x50e stream (e.g. `image83 --log`, or `import-tsv` of a Windows capture)
/// and pair each pose with the latest 0x83 gaze-frame head anchors. Writes a
/// CSV for analysis and prints a summary (used to validate sign/scale against
/// the `MediaPipe` reference and the 0x83 eyeball-centre translation).
///
/// # Errors
///
/// Fails if the log cannot be read, the face model cannot be loaded, a gaze
/// payload does not decode, or the CSV cannot be written.
pub(crate) fn run_image83_replay(path: &str, csv: Option<&str>) -> Result<()> {
    use tobii_proto::log::read_log_payloads;
    let payloads = read_log_payloads(path)?;
    let mut tracker = tobii_pose::track::Tracker::new_image83()?;
    let mut asm = BulkReassembler::new();
    let mut out = csv
        .map(|p| {
            File::create(p)
                .map(BufWriter::new)
                .with_context(|| format!("create {p}"))
        })
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
    let mut msgs = Vec::new();
    for payload in &payloads {
        asm.push_into(payload, &mut msgs);
        for msg in &msgs {
            match stream_id(msg) {
                Some(STREAM_ID_GAZE) => {
                    let decoded = decode_stream_payload(msg)?;
                    if !decoded.is_empty() {
                        last_gaze = Some(TrackingFrame::from_decoded(gaze_frames, &decoded, None));
                        gaze_frames += 1;
                    }
                }
                Some(STREAM_ID_IMAGE) => {
                    let Some(frame) = decode_image_payload(msg) else {
                        continue;
                    };
                    let rel = tracker.process(&frame.pixels, frame.width, frame.height)?;
                    let raw = tracker.last_raw();
                    faces += u64::from(raw.is_some());
                    poses += u64::from(rel.is_some());
                    let g = last_gaze.as_ref();
                    if let Some(r) = raw
                        && let Some(g) = g
                        && let Some(x) = g.head_x
                        && g.head_z.is_some()
                    {
                        yaw_vs_x.push((r[4], x / 1000.0));
                    }
                    if let Some(o) = out.as_mut() {
                        let f = |v: Option<f64>| v.map_or(String::from(""), |v| format!("{v:.3}"));
                        let r6 = |v: Option<[f64; 6]>, i: usize| f(v.map(|a| a[i]));
                        writeln!(
                            o,
                            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                            images,
                            frame.device_ts_us,
                            u8::from(raw.is_some()),
                            r6(raw, 0),
                            r6(raw, 1),
                            r6(raw, 2),
                            r6(raw, 3),
                            r6(raw, 4),
                            r6(raw, 5),
                            r6(rel, 0),
                            r6(rel, 1),
                            r6(rel, 2),
                            r6(rel, 3),
                            r6(rel, 4),
                            r6(rel, 5),
                            f(g.and_then(|g| g.head_x).map(|v| v / 1000.0)),
                            f(g.and_then(|g| g.head_y).map(|v| v / 1000.0)),
                            f(g.and_then(|g| g.head_z).map(|v| v / 1000.0)),
                            f(g.and_then(|g| g.head_roll)),
                            g.map_or(0, |g| u8::from(g.gaze_valid)),
                        )?;
                    }
                    images += 1;
                }
                _ => {}
            }
        }
    }
    let runs = tracker.model_runs();
    println!(
        "{path}: {gaze_frames} gaze frames, {images} images, face found in {faces}, {poses} calibrated poses; \
         landmark model run {} times, face detector {}",
        runs.landmarks, runs.detector
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
///
/// # Errors
///
/// Fails if the device cannot be opened / initialised, a PGM or log write
/// fails, or the stream never arms in three opens.
///
/// # Panics
///
/// Panics if `opts.command` is not `Command::Image83`; `lib.rs` dispatches
/// on the variant before calling this.
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
        Some(tobii_pose::track::Tracker::new_image83()?)
    } else {
        None
    };
    let mut log = log_path.as_deref().map(PacketLog::create).transpose()?;

    for attempt in 1..=3 {
        let mut h = open_tobii(&ctx)?;
        vendor_control_init(&mut h)?;
        println!(
            "Replaying {} init packets (attempt {attempt})...",
            packets.len()
        );
        replay_init_packets(&mut h, &packets, Some(&stop))?;
        let mut cmd_seq = next_command_seq(&packets);
        let image = !*no_image;
        if image {
            match start_stream(&mut h, cmd_seq, STREAM_ID_IMAGE, Some(&stop)) {
                Ok(()) => {
                    println!("image stream 0x50e start acknowledged (cmd 1220, seq {cmd_seq:#x})");
                }
                Err(e) => println!("image stream start: {e}"),
            }
            cmd_seq += 1;
        } else {
            println!("image stream NOT requested (--no-image baseline)");
        }
        let mut asm = BulkReassembler::new();
        if !wait_for_gaze_stream(&mut h, &stop, Duration::from_secs_f64(4.5), &mut asm) {
            println!(
                "no gaze stream within 4.5 s on open {attempt}; re-opening (cold-start quirk)"
            );
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
            let mut counts = std::collections::BTreeMap::<u32, (u64, u64)>::new();
            let mut other = 0u64;
            let mut saved = 0usize;
            let mut image_ts = Vec::<u64>::new();
            let mut gaze_ts = Vec::<u64>::new();
            let (mut gaze_frames, mut gaze_valid, mut eyes_valid) = (0u64, 0u64, 0u64);
            let mut poses = 0u64;
            let start = Instant::now();
            let dur = Duration::from_secs_f64(*secs);
            let mut msgs = Vec::new();
            while start.elapsed() < dur {
                let n = match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(300)) {
                    Ok(n) => n,
                    Err(rusb::Error::Timeout) => continue,
                    Err(e) => return Err(e.into()),
                };
                asm.push_into(&buf[..n], &mut msgs);
                for msg in &msgs {
                    log_packet(&mut log, EP_IN, msg)?;
                    let Some(id) = stream_id(msg) else {
                        other += 1;
                        continue;
                    };
                    let e = counts.entry(id).or_default();
                    e.0 += 1;
                    e.1 += u64::try_from(msg.len()).unwrap_or(u64::MAX);
                    match id {
                        STREAM_ID_GAZE => {
                            if let Ok(v) = decode_stream_payload(msg)
                                && !v.is_empty()
                            {
                                let f = TrackingFrame::from_decoded(gaze_frames, &v, None);
                                gaze_frames += 1;
                                gaze_valid += u64::from(f.gaze_valid);
                                eyes_valid += u64::from(f.head_xyz().is_some());
                            }
                            gaze_ts.push(now_us());
                        }
                        STREAM_ID_IMAGE => {
                            let Some(frame) = decode_image_payload(msg) else {
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
                            if let Some(t) = tracker.as_mut()
                                && let Some(p) =
                                    t.process(&frame.pixels, frame.width, frame.height)?
                            {
                                poses += 1;
                                if poses % 10 == 1 {
                                    println!(
                                        "pose: yaw {:+6.1} pitch {:+6.1} roll {:+6.1}  t = [{:+5.1} {:+5.1} {:+5.1}] cm",
                                        p[3], p[4], p[5], p[0], p[1], p[2]
                                    );
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
            println!(
                "non-stream messages: {other}; reassembler pending {} bytes",
                asm.pending_len()
            );
            if gaze_frames > 0 {
                println!(
                    "gaze frames {gaze_frames}: gaze valid {:.1}%, both eyeball centres valid {:.1}%",
                    gaze_valid as f64 * 100.0 / gaze_frames as f64,
                    eyes_valid as f64 * 100.0 / gaze_frames as f64
                );
            }
            if image_ts.len() > 2 {
                let mut d: Vec<f64> = image_ts
                    .windows(2)
                    .map(|w| (w[1] as f64 - w[0] as f64) / 1000.0)
                    .collect();
                d.sort_by(f64::total_cmp);
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
///
/// # Errors
///
/// Fails if the device cannot be opened / initialised or the UVC handshake
/// fails.
///
/// # Panics
///
/// Panics if `opts.command` is not `Command::Probe`; `lib.rs` dispatches
/// on the variant before calling this.
pub(crate) fn run_probe(opts: &Options) -> Result<()> {
    let Command::Probe { init_path } = &opts.command else {
        unreachable!();
    };

    let ctx = UsbContext::new()?;
    let mut h = open_tobii(&ctx)?;
    vendor_control_init(&mut h)?;
    let packets = read_init_packets(init_path)?;
    println!("Replaying {} init packets...", packets.len());
    if let Err(e) = replay_init_packets(&mut h, &packets, None) {
        warn!(error = format_args!("{e:#}"), "init replay aborted");
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
                if let Ok(k) = h.read_bulk(EP_VIDEO, &mut buf, Duration::from_millis(200))
                    && k > 0
                {
                    n += 1;
                    b += u64::try_from(k).unwrap_or(u64::MAX);
                }
            }
            (n, b)
        })
    };

    let mut buf83 = vec![0u8; 16384];
    let (mut n83, mut s83, mut b83) = (0u64, 0u64, 0u64);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(6) {
        if let Ok(n) = h.read_bulk(EP_IN, &mut buf83, Duration::from_millis(200))
            && n > 0
        {
            n83 += 1;
            b83 += u64::try_from(n).unwrap_or(u64::MAX);
            if marker(&buf83[..n]) == Some(0x53) {
                s83 += 1;
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
        if let Ok(len) = h.read_bulk(EP_IN, &mut buf, Duration::from_millis(50))
            && len > 0
        {
            n += 1;
            b += u64::try_from(len).unwrap_or(u64::MAX);
            if marker(&buf[..len]) == Some(0x53) {
                s += 1;
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
    let index = u16::from(IFACE_VIDEO);
    let timeout = Duration::from_millis(1000);

    let mut len_buf = [0u8; 2];
    let len = h
        .read_control(RT_IN, GET_LEN, PROBE, index, &mut len_buf, timeout)
        .map_or(0, |_| usize::from(u16::from_le_bytes(len_buf)));
    let len = if (26..=64).contains(&len) { len } else { 34 };

    // Probe the supported frame-interval range. The descriptor only lists 24fps,
    // but the device may accept faster via the PROBE control. dwFrameInterval is
    // in 100ns units at offset 4; smaller = faster.
    let interval_of = |req: u8| -> Option<u32> {
        let mut b = vec![0u8; len];
        h.read_control(RT_IN, req, PROBE, index, &mut b, timeout)
            .ok()
            .and_then(|_| le_u32_at(&b, 4))
    };
    let fps = |iv: u32| if iv > 0 { 1e7 / f64::from(iv) } else { 0.0 };
    // Request the smallest (fastest) interval the device reports unless
    // overridden on the CLI. (The device caps actual streaming at ~24fps total
    // / 8fps per camera regardless, but we ask for its best.)
    let want_iv = interval.unwrap_or_else(|| {
        [
            interval_of(GET_MIN),
            interval_of(GET_MAX),
            interval_of(GET_DEF),
        ]
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
    let got_iv = le_u32_at(&ctrl, 4).unwrap_or(0);
    println!(
        "UVC negotiated frame interval = {got_iv} ({:.1} fps total)",
        fps(got_iv)
    );

    // dwMaxVideoFrameSize at offset 18.
    let frame_size = le_u32_at(&ctrl, 18).map_or(0, |v| usize::try_from(v).unwrap_or(0));

    h.write_control(RT_OUT, SET_CUR, COMMIT, index, &ctrl, timeout)
        .context("UVC COMMIT SET_CUR failed")?;

    Ok(if frame_size == 0 {
        CAM_WIDTH * CAM_HEIGHT
    } else {
        frame_size
    })
}

/// Little-endian `u32` at `at`, or `None` if the slice is too short.
#[must_use]
fn le_u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    let word: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(word))
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
                let hle = usize::from(buf[0]).min(n);
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
                    type_hist[usize::from(cur_type)] += 1;
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
                warn!(
                    timeouts,
                    "EP 0x82 read timeout; camera may need different init/probe"
                );
                if timeouts >= 10 {
                    anyhow::bail!("no camera data on EP 0x82 after {timeouts} timeouts");
                }
            }
            // A stalled bulk endpoint shows up as Pipe (and sometimes Io); clear
            // the halt and keep trying rather than giving up.
            Err(e @ (rusb::Error::Pipe | rusb::Error::Io)) => {
                timeouts += 1;
                warn!(error = ?e, retries = timeouts, "EP 0x82 failed; clearing halt and retrying");
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
        .filter(|&(_, &c)| c > 0)
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
            // Zero-pad without allocating a per-frame buffer.
            let pad = u64::try_from(need - data.len()).unwrap_or(u64::MAX);
            std::io::copy(&mut std::io::repeat(0).take(pad), w)?;
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
    println!(
        "  wrote {path} ({} payload bytes, {CAM_WIDTH}x{height})",
        data.len()
    );
    Ok(())
}
