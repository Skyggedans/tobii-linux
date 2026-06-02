use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};
use std::{error, fmt, thread};

use crate::cli::{Command, Options};
use crate::dashboard::render_dashboard_status;
use crate::opentrack::OpentrackUdp;
use crate::protocol::{declared_len, marker, read_init_packets, seq, InitPacket};
use crate::sinks::{handle_live_decoded, log_packet, DecodedCsv, JsonlOutput, PacketLog};

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

    read_camera_frames(
        &mut h,
        out_prefix,
        *max_frames,
        frame_size,
        fifo.as_deref(),
        *frame_type,
    )
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

fn read_camera_frames(
    h: &mut rusb::DeviceHandle<UsbContext>,
    out_prefix: &str,
    max_frames: usize,
    frame_size: usize,
    fifo: Option<&str>,
    type_filter: Option<u8>,
) -> Result<()> {
    // Two output modes: stream fixed-size raw frames to a FIFO (for a CV
    // consumer), or dump the first `max_frames` as PGM plus the raw bulk stream.
    let stream = fifo.is_some();
    let mut fifo_writer = match fifo {
        Some(path) => {
            println!("Streaming raw {CAM_WIDTH}x{CAM_HEIGHT} frames to {path} (waiting for reader)...");
            // Open the existing FIFO write-only WITHOUT O_CREAT/O_TRUNC: File::create
            // uses O_CREAT, which fs.protected_fifos blocks for a root process on a
            // FIFO it doesn't own in a sticky /tmp. This blocks until a reader opens.
            Some(BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                    .with_context(|| format!("failed to open fifo {path} (mkfifo it first)"))?,
            ))
        }
        None => None,
    };
    let mut raw = if stream {
        None
    } else {
        let raw_path = format!("{out_prefix}.raw");
        println!("Dumping raw EP 0x82 bulk to {raw_path}; writing up to {max_frames} PGM frames");
        Some(BufWriter::new(
            File::create(&raw_path).with_context(|| format!("failed to create {raw_path}"))?,
        ))
    };

    let mut buf = vec![0u8; 1024 * 1024];
    let mut frame: Vec<u8> = Vec::with_capacity(frame_size.max(CAM_WIDTH * CAM_HEIGHT) + 4096);
    let mut last_fid: Option<u8> = None;
    let mut saved = 0usize;
    let mut timeouts = 0u32;
    // Tobii frame-type byte (the wide face camera is type 2; eye cameras differ).
    let mut cur_type = 0u8;
    let mut type_hist = [0u64; 256];
    let mut typed = 0u64;
    let mut hist_printed = false;
    let t_start = Instant::now();

    loop {
        match h.read_bulk(EP_VIDEO, &mut buf, Duration::from_millis(2000)) {
            Ok(0) => continue,
            Ok(n) => {
                timeouts = 0;
                if let Some(raw) = raw.as_mut() {
                    raw.write_all(&buf[..n])?;
                }

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

                // Flush the previous frame on FID change (its type is `cur_type`,
                // not yet overwritten by this payload).
                let boundary = last_fid.is_some() && last_fid != Some(fid);
                if boundary && !frame.is_empty() {
                    if flush_camera_frame(
                        &mut fifo_writer, out_prefix, &mut saved, &frame, type_filter, cur_type,
                    )? && !stream
                        && saved >= max_frames
                    {
                        break;
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
                    let stop = flush_camera_frame(
                        &mut fifo_writer, out_prefix, &mut saved, &frame, type_filter, cur_type,
                    )? && !stream
                        && saved >= max_frames;
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

    println!("Saved {saved} PGM frame(s) as {out_prefix}NNN.pgm");
    Ok(())
}

/// Emit an assembled frame unless a `type_filter` excludes its camera id.
/// Returns whether it was actually written (so PGM mode can count toward its
/// `--frames` limit).
fn flush_camera_frame(
    fifo: &mut Option<BufWriter<File>>,
    out_prefix: &str,
    saved: &mut usize,
    frame: &[u8],
    type_filter: Option<u8>,
    cur_type: u8,
) -> Result<bool> {
    if type_filter.is_none_or(|t| t == cur_type) {
        emit_frame(fifo, out_prefix, *saved, frame)?;
        *saved += 1;
        Ok(true)
    } else {
        Ok(false)
    }
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
