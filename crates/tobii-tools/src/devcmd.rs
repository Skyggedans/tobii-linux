//! Research and diagnostic subcommands driven from the CLI: the legacy
//! `replay` stream mode, the UVC IR camera path (`camera`, `probe`) and the
//! 0x50e image tools (`image83`, `image83-replay`).
//!
//! None of this is needed to run the driver — the daemon uses
//! [`tobii_usb::device`] only. Everything here shares that module's USB transport
//! helpers rather than duplicating them.

use anyhow::{Context, Result};
use rusb::Context as UsbContext;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::cli::{Command, Options};
use crate::dashboard::render_dashboard_status;
use crate::face_fits::{FitFields, FitsCsv, LandmarksF32};
use crate::opentrack::OpentrackUdp;
use crate::sinks::{DecodedCsv, JsonlOutput, handle_live_decoded};
use tobii_ipc::geometry::{DisplayArea, DisplayFrame};
use tobii_pose::head::{FrameContext, HeadParams, HeadPose, HeadStep};
use tobii_pose::track::{ModelRuns, Tracker, euler_deg};
use tobii_proto::decode::{TrackingFrame, decode_stream_payload};
use tobii_proto::image83::{ImageFrame, decode_image_payload, write_pgm};
use tobii_proto::log::{PacketLog, log_packet};
use tobii_proto::protocol::{
    BulkReassembler, InitPacket, STREAM_ID_GAZE, STREAM_ID_IMAGE, STREAM_ID_PRESENCE, declared_len,
    marker, read_init_packets, seq, stream_id,
};
use tobii_proto::time::now_us;
use tobii_usb::calibration::written_display_area;
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

/// Offline: run the daemon's head tracker over every 0x50e image of a
/// TBI5LOG1 log (e.g. `image83 --log`, or `import-tsv` of a Windows
/// capture), in log order, and print how many images had a face and how
/// the fit's yaw follows the head anchors of the 0x83 gaze frames.
///
/// Each image gets its face fit ([`Tracker::fit`]) and nothing more. `csv`
/// gets one row per image ([`AnchorsCsv`]): whether it had a face, and the
/// head anchors of the last gaze frame before it. `fits` and `landmarks`
/// export each image's fit at full precision, as a CSV and as raw f32
/// landmarks ([`crate::face_fits`] gives their layouts).
///
/// # Errors
///
/// Fails if an output is the log or another output (before anything is
/// read or written), the log cannot be read, the face model cannot be
/// loaded, a gaze payload does not decode, the tracker fails on an image,
/// or an output cannot be written.
pub(crate) fn run_image83_replay(
    path: &str,
    csv: Option<&str>,
    fits: Option<&str>,
    landmarks: Option<&str>,
) -> Result<()> {
    use tobii_proto::log::read_log_payloads;
    let outputs: Vec<(&str, &str)> = [("--csv", csv), ("--fits", fits), ("--landmarks", landmarks)]
        .into_iter()
        .filter_map(|(option, output)| output.map(|o| (option, o)))
        .collect();
    ensure_distinct_paths(path, &outputs, resolved_path)?;
    let payloads = read_log_payloads(path)?;
    let mut tracker = Tracker::new_image83()?;
    let mut fits_csv = fits.map(FitsCsv::create).transpose()?;
    let mut landmarks_f32 = landmarks.map(LandmarksF32::create).transpose()?;
    let mut asm = BulkReassembler::new();
    let mut anchors_csv = csv.map(AnchorsCsv::create).transpose()?;
    let mut last_gaze: Option<TrackingFrame> = None;
    let (mut images, mut faces, mut gaze_frames) = (0u64, 0u64, 0u64);
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
                    let fit = tracker.fit(&frame.pixels, frame.width, frame.height)?;
                    faces += u64::from(fit.is_some());
                    if let Some(fits_csv) = fits_csv.as_mut() {
                        let fields = fit.as_ref().map(FitFields::from);
                        fits_csv.write_image(images, frame.device_ts_us, fields.as_ref())?;
                    }
                    if let Some(landmarks_f32) = landmarks_f32.as_mut() {
                        landmarks_f32.write_image(fit.as_ref().map(|f| f.landmarks))?;
                    }
                    let g = last_gaze.as_ref();
                    if let Some(fit) = &fit
                        && let Some(g) = g
                        && let Some(x) = g.head_x
                        && g.head_z.is_some()
                    {
                        yaw_vs_x.push((camera_yaw_deg(&fit.rotation), x / 1000.0));
                    }
                    if let Some(anchors_csv) = anchors_csv.as_mut() {
                        anchors_csv.write_image(images, frame.device_ts_us, fit.is_some(), g)?;
                    }
                    images += 1;
                }
                _ => {}
            }
        }
    }
    if let (Some(anchors_csv), Some(p)) = (anchors_csv, csv) {
        anchors_csv
            .finish()
            .with_context(|| format!("failed to write {p}"))?;
    }
    if let (Some(fits_csv), Some(p)) = (fits_csv, fits) {
        fits_csv.finish()?;
        println!("wrote the face fits of {images} images to {p}");
    }
    if let (Some(landmarks_f32), Some(p)) = (landmarks_f32, landmarks) {
        landmarks_f32.finish()?;
        println!("wrote the landmarks of {images} images to {p}");
    }
    let runs = tracker.model_runs();
    println!(
        "{path}: {gaze_frames} gaze frames, {images} images, face found in {faces}; \
         landmark model run {} times, face detector {}",
        runs.landmarks, runs.detector
    );
    if let Some((r, slope)) = yaw_vs_head_x(&yaw_vs_x) {
        println!(
            "raw image yaw vs 0x83 head_x: r = {r:.3}, slope {slope:.2} mm/deg (neck lever; expect ~-1.5 for +yaw = left)"
        );
    }
    Ok(())
}

/// The header of [`AnchorsCsv`]: the columns `image83-replay -h` describes.
const ANCHORS_CSV_HEADER: &str = "image_idx,device_ts_us,face,\
    g_head_x_mm,g_head_y_mm,g_head_z_mm,g_head_roll_deg,g_valid";

/// `image83-replay --csv`: one row per image, in log order: its number in
/// the log and device time, whether the tracker found a face in it, and the
/// head anchors of the last 0x83 gaze frame before it, which the decoder
/// makes of that frame's two eyeball centres ([`TrackingFrame::head_x`] and
/// the rest; µm there, mm here), with the frame's gaze validity.
struct AnchorsCsv<W: Write> {
    out: W,
}

impl AnchorsCsv<BufWriter<File>> {
    /// Create `path` and write the header.
    ///
    /// # Errors
    /// Fails when the file cannot be created or written.
    fn create(path: &str) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("create {path}"))?;
        Self::new(BufWriter::new(file)).with_context(|| format!("failed to write {path}"))
    }
}

impl<W: Write> AnchorsCsv<W> {
    /// Rows to `out`, after the header.
    fn new(mut out: W) -> io::Result<Self> {
        writeln!(out, "{ANCHORS_CSV_HEADER}")?;
        Ok(Self { out })
    }

    /// The row of image number `image` (of device time `device_ts_us`),
    /// which had a face or not, `gaze` being the last gaze frame before it.
    /// Numbers have three decimals. An anchor the frame lacks (it takes
    /// both eyes) is empty; before the first frame every anchor is, and the
    /// validity is 0.
    fn write_image(
        &mut self,
        image: u64,
        device_ts_us: u64,
        face: bool,
        gaze: Option<&TrackingFrame>,
    ) -> io::Result<()> {
        let three = |v: Option<f64>| v.map_or_else(String::new, |v| format!("{v:.3}"));
        let mm = |field: fn(&TrackingFrame) -> Option<f64>| {
            three(gaze.and_then(field).map(|um| um / 1000.0))
        };
        writeln!(
            self.out,
            "{image},{device_ts_us},{},{},{},{},{},{}",
            u8::from(face),
            mm(|g| g.head_x),
            mm(|g| g.head_y),
            mm(|g| g.head_z),
            three(gaze.and_then(|g| g.head_roll)),
            gaze.map_or(0, |g| u8::from(g.gaze_valid)),
        )
    }

    /// Flush the rows, and hand the writer back.
    fn finish(mut self) -> io::Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }
}

/// The yaw of a fit's `rotation` (mesh -> camera), degrees: its angle about
/// the camera's vertical axis, the yaw of its zyx Euler angles
/// ([`euler_deg`]), with the sign that makes a turn of the head to its own
/// left positive.
#[must_use]
fn camera_yaw_deg(rotation: &[[f64; 3]; 3]) -> f64 {
    -euler_deg(rotation)[1]
}

/// How the head anchors' x follows the fit's yaw over `pairs` of (yaw,
/// degrees; head x, mm): their correlation and the least-squares slope of
/// the x on the yaw, mm per degree. `None` for ten pairs or fewer.
#[must_use]
fn yaw_vs_head_x(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() <= 10 {
        return None;
    }
    let n = pairs.len() as f64;
    let (my, mx) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / n,
        pairs.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (y, x) in pairs {
        sxy += (y - my) * (x - mx);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    Some((sxy / (sxx * syy).sqrt(), sxy / syy))
}

/// Refuse outputs that would overwrite the log they are made from, or one
/// another. Each output is created, so truncated, once the log has been
/// read whole: `image83-replay session2.bin --fits session2.bin` would
/// leave the capture holding a CSV header and still succeed, and two
/// outputs on one path would interleave. `outputs` are (option, path);
/// paths are compared as `resolve` gives them ([`resolved_path`]).
///
/// # Errors
/// Names the output that is the log, or the two outputs that are one file.
fn ensure_distinct_paths(
    log: &str,
    outputs: &[(&str, &str)],
    resolve: impl Fn(&str) -> PathBuf,
) -> Result<()> {
    let log_path = resolve(log);
    let mut seen: Vec<(&str, PathBuf)> = Vec::with_capacity(outputs.len());
    for &(option, path) in outputs {
        let resolved = resolve(path);
        anyhow::ensure!(
            resolved != log_path,
            "{option} {path} is the log {log}: it would be overwritten"
        );
        if let Some((other, _)) = seen.iter().find(|(_, p)| *p == resolved) {
            anyhow::bail!("{option} {path} is also the {other} output");
        }
        seen.push((option, resolved));
    }
    Ok(())
}

/// `path` as the file system names it, to tell whether two paths are one
/// file: canonical (absolute, symbolic links and `..` resolved) when it
/// exists; for a file still to be created, its directory's canonical path
/// joined with its name; as given when neither resolves.
fn resolved_path(path: &str) -> PathBuf {
    let path = Path::new(path);
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return path.to_path_buf();
    };
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    std::fs::canonicalize(dir).map_or_else(|_| path.to_path_buf(), |dir| dir.join(name))
}

/// Diagnostic: bring the device up in gaze mode, additionally start the 0x50e
/// image stream, and report what arrives on EP 0x83 for `secs` seconds: per-
/// stream message rates, image timestamp cadence, the first frames as PGM, and
/// with `--pose` the Stream Engine's head pose of each image ([`LivePoses`]).
///
/// # Errors
///
/// Fails if the device cannot be opened / initialised, a PGM or log write
/// fails, the head pose models cannot be loaded or the tracker fails on an
/// image, or the stream never arms in three opens.
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
    let mut poses = if *pose {
        Some(LivePoses::new(&packets)?)
    } else {
        None
    };
    if let Some(poses) = &poses {
        println!("{}", poses.display_note());
    }
    let mut log = log_path.as_deref().map(PacketLog::create).transpose()?;

    for attempt in 1u64..=3 {
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
                            if let Some(poses) = poses.as_mut()
                                && let Some(line) = poses.image(&frame, attempt)?
                            {
                                println!("{line}");
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
            if let Some(poses) = &poses {
                println!("{}", poses.summary());
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

/// `image83 --pose`: tobiid's own step from an image to the Stream Engine's
/// head pose ([`HeadStep`] with [`HeadParams::FITTED`]), the poses in the
/// display frame of the area the init capture writes, and a tally of them.
struct LivePoses {
    step: HeadStep,
    /// The display area the init capture writes, if it writes one.
    area: Option<DisplayArea>,
    /// The frame `area` fixes; without one every pose is invalid.
    display: Option<DisplayFrame>,
    tally: PoseTally,
}

impl LivePoses {
    /// The step, its poses in the display frame of the area `packets`
    /// write (their 1440): the device holds that area once the init replay
    /// has written it, so it is the one in effect while the streams run.
    /// It is the capture's, not the user's, which tobiid writes instead.
    ///
    /// # Errors
    /// Fails when a model cannot be loaded.
    fn new(packets: &[InitPacket]) -> Result<Self> {
        let area = written_display_area(packets).map(|(_, area, _)| area);
        Ok(Self {
            step: HeadStep::new(HeadParams::FITTED).context("loading the head pose models")?,
            display: area.as_ref().and_then(DisplayFrame::new),
            area,
            tally: PoseTally::default(),
        })
    }

    /// The report's first line ([`display_note`]).
    fn display_note(&self) -> String {
        display_note(self.area.as_ref())
    }

    /// The pose of `frame`, an image read in the tracker's `open`th open (a
    /// new open restarts the filters), and the line of the report it makes:
    /// the first image's and every tenth after it ([`pose_line`]).
    ///
    /// # Errors
    /// Fails when the image's time does not fit an `i64`, or the tracker
    /// fails on the image.
    fn image(&mut self, frame: &ImageFrame, open: u64) -> Result<Option<String>> {
        let context = FrameContext {
            // The image's device time, as the replay tools step by; tobiid
            // maps it onto the host clock first.
            t_us: i64::try_from(frame.device_ts_us)
                .context("an image's time does not fit an i64")?,
            display: self.display.as_ref(),
            // The area stays the init capture's while the command runs.
            display_generation: 0,
            open,
        };
        let out = self
            .step
            .step(&frame.pixels, frame.width, frame.height, &context);
        if let Some(e) = out.error {
            return Err(e.context("the head tracker failed on an image"));
        }
        let face = out.face.is_some();
        self.tally.add(out.head.valid, face);
        Ok((self.tally.images % 10 == 1).then(|| pose_line(&out.head, face)))
    }

    /// The report's last line ([`PoseTally::summary`]).
    fn summary(&self) -> String {
        self.tally.summary(self.step.model_runs())
    }
}

/// What `image83 --pose` says of the display frame its poses are in: the
/// corners of `area`, the display area the init capture writes, or why
/// every pose is invalid.
fn display_note(area: Option<&DisplayArea>) -> String {
    let point = |p: [f64; 3]| format!("({:.1}, {:.1}, {:.1})", p[0], p[1], p[2]);
    match area {
        Some(area) if DisplayFrame::new(area).is_some() => format!(
            "head poses in the display frame of the area the init capture writes (mm): \
             top left {}, top right {}, bottom left {}",
            point(area.top_left_mm),
            point(area.top_right_mm),
            point(area.bottom_left_mm)
        ),
        Some(_) => "the display area the init capture writes fixes no display frame: \
                    every head pose is invalid"
            .to_string(),
        None => "the init capture writes no display area: every head pose is invalid".to_string(),
    }
}

/// One line of `image83 --pose`'s report. Of a valid pose, its position
/// (mm, in the display frame) and its rotation, the Stream Engine's x, y
/// and z angles in degrees; of an invalid one, which carries the last valid
/// pose's values, whether the tracker found a face (`face`).
#[must_use]
fn pose_line(pose: &HeadPose, face: bool) -> String {
    if pose.valid {
        let [x, y, z] = pose.position_mm;
        let [rx, ry, rz] = pose.rotation_rad.map(f64::to_degrees);
        format!(
            "head pose: position ({x:+6.1}, {y:+6.1}, {z:+6.1}) mm, \
             rotation x {rx:+5.1} y {ry:+5.1} z {rz:+5.1} deg"
        )
    } else if face {
        "head pose: invalid, a face but too near the image's edge, or no display area".to_string()
    } else {
        "head pose: invalid, no face".to_string()
    }
}

/// What `image83 --pose` made: the images it stepped, the valid poses among
/// them, and the images the tracker found a face in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PoseTally {
    images: u64,
    valid: u64,
    faces: u64,
}

impl PoseTally {
    /// Count one image, its pose valid or not and a face found or not.
    fn add(&mut self, valid: bool, face: bool) {
        self.images += 1;
        self.valid += u64::from(valid);
        self.faces += u64::from(face);
    }

    /// The tally as the report's last line, with the tracker's model `runs`.
    #[must_use]
    fn summary(&self, runs: ModelRuns) -> String {
        let valid_pct = if self.images == 0 {
            0.0
        } else {
            self.valid as f64 * 100.0 / self.images as f64
        };
        format!(
            "head poses of {} images: {} valid ({valid_pct:.1}%), a face in {}; \
             landmark model run {} times, face detector {}",
            self.images, self.valid, self.faces, runs.landmarks, runs.detector
        )
    }
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
/// a frame (PGM, FIFO) is the caller's.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Paths as they are written: the check itself, apart from the file
    /// system.
    fn as_written(path: &str) -> PathBuf {
        PathBuf::from(path)
    }

    /// A scratch directory, removed when dropped, after a failed assertion
    /// too.
    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            // Nothing to report a failure to here; a leftover is harmless.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn an_output_that_is_the_log_or_another_output_is_refused() {
        let outputs = [
            ("--csv", "poses.csv"),
            ("--fits", "fits.csv"),
            ("--landmarks", "landmarks.f32"),
        ];
        ensure_distinct_paths("session2.bin", &outputs, as_written).expect("distinct paths");
        ensure_distinct_paths("session2.bin", &[], as_written).expect("no outputs");

        for at in 0..outputs.len() {
            let mut onto_log = outputs;
            onto_log[at].1 = "session2.bin";
            let error = ensure_distinct_paths("session2.bin", &onto_log, as_written)
                .expect_err("an output onto the log")
                .to_string();
            assert_eq!(
                error,
                format!(
                    "{} session2.bin is the log session2.bin: it would be overwritten",
                    outputs[at].0
                )
            );
        }

        let twice = [("--csv", "out.csv"), ("--fits", "out.csv")];
        let error = ensure_distinct_paths("session2.bin", &twice, as_written)
            .expect_err("two outputs on one file")
            .to_string();
        assert_eq!(error, "--fits out.csv is also the --csv output");
    }

    /// One file under different names resolves to one path: through `..`,
    /// a relative path and, for a file still to be created, its directory.
    #[test]
    fn paths_to_one_file_resolve_alike() {
        let scratch = Scratch(
            std::env::temp_dir().join(format!("tobii-resolved-path-{}", std::process::id())),
        );
        let dir = &scratch.0;
        std::fs::create_dir_all(dir.join("sub")).expect("scratch directory");
        let log = dir.join("session.bin");
        std::fs::write(&log, b"TBI5LOG1").expect("scratch log");
        let name = |p: &Path| p.to_str().expect("a UTF-8 path").to_string();

        let roundabout = dir.join("sub").join("..").join("session.bin");
        assert_eq!(
            resolved_path(&name(&roundabout)),
            resolved_path(&name(&log))
        );
        let error = ensure_distinct_paths(
            &name(&log),
            &[("--fits", &name(&roundabout))],
            resolved_path,
        )
        .expect_err("the log under another name");
        assert!(error.to_string().starts_with("--fits "), "{error}");

        let new = dir.join("sub").join("fits.csv");
        let new_roundabout = dir.join("sub").join("..").join("sub").join("fits.csv");
        assert_eq!(
            resolved_path(&name(&new_roundabout)),
            resolved_path(&name(&new))
        );
        assert!(!new.exists());
        ensure_distinct_paths(
            &name(&log),
            &[
                ("--fits", &name(&new)),
                ("--landmarks", &name(&log.with_extension("f32"))),
            ],
            resolved_path,
        )
        .expect("other files");

        #[cfg(unix)]
        {
            let link = dir.join("link.bin");
            std::os::unix::fs::symlink(&log, &link).expect("scratch link");
            assert_eq!(resolved_path(&name(&link)), resolved_path(&name(&log)));
        }

        // Relative to the working directory, which cargo sets to the
        // package's own.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        for relative in ["Cargo.toml", "./Cargo.toml", "src/../Cargo.toml"] {
            assert_eq!(resolved_path(relative), resolved_path(&name(&manifest)));
        }
    }

    /// The 0x83 field that carries the eyeball centres (occurrences 4 and
    /// 9) and the one that carries the eyes' gaze points (1 and 3).
    const EYEBALL_CENTRES: u32 = 0x0003_1f41;
    const GAZE_POINTS: u32 = 0x0002_1f40;

    /// A gaze frame as `image83-replay` decodes one: the left eyeball
    /// centre at `left` and the right one at `right` (µm; `None`, the eye
    /// was not found), with both eyes' gaze points on screen when `gaze`.
    fn gaze_frame(left: [f64; 3], right: Option<[f64; 3]>, gaze: bool) -> TrackingFrame {
        let mut values = tobii_proto::decode::FieldValues::new();
        for (occurrence, centre) in [(4, Some(left)), (9, right)] {
            for (component, v) in centre.into_iter().flatten().enumerate() {
                values.insert((EYEBALL_CENTRES, occurrence, component), v);
            }
        }
        if gaze {
            for (occurrence, point) in [(1, [512.0, 400.0]), (3, [520.0, 410.0])] {
                for (component, v) in point.into_iter().enumerate() {
                    values.insert((GAZE_POINTS, occurrence, component), v);
                }
            }
        }
        TrackingFrame::from_decoded(0, &values, None)
    }

    /// `--csv`: a row per image of its number, device time and face, then
    /// the head anchors of the last gaze frame before it in mm (µm in the
    /// frame), to three decimals, and its gaze validity. Before the first
    /// gaze frame, and for a frame without both eyes, the anchors are empty;
    /// before the first, the validity is 0.
    #[test]
    fn the_csv_has_each_image_s_face_and_the_last_gaze_frame_s_head_anchors() {
        let level = gaze_frame(
            [-20_000.0, 3_000.0, 650_000.0],
            Some([40_000.0, 3_000.0, 650_000.0]),
            true,
        );
        let one_eye = gaze_frame([-20_000.0, 3_000.0, 650_000.0], None, false);
        let mut csv = AnchorsCsv::new(Vec::new()).expect("the header");

        for (image, face, gaze) in [
            (0, false, None),
            (1, true, Some(&level)),
            (2, true, Some(&one_eye)),
        ] {
            csv.write_image(image, 30_000 * image, face, gaze)
                .expect("a row");
        }

        let rows = String::from_utf8(csv.finish().expect("flushed")).expect("UTF-8");
        assert_eq!(
            rows,
            "image_idx,device_ts_us,face,g_head_x_mm,g_head_y_mm,g_head_z_mm,g_head_roll_deg,g_valid\n\
             0,0,0,,,,,0\n\
             1,30000,1,10.000,3.000,650.000,0.000,1\n\
             2,60000,1,,,,,0\n"
        );
    }

    /// A turn of the head to its own left takes its nose (the mesh's -z)
    /// towards the camera's +x, the image's right: a positive yaw of as
    /// many degrees.
    #[test]
    fn a_turn_of_the_head_to_its_left_is_a_positive_yaw() {
        let (s, c) = 20f64.to_radians().sin_cos();
        // 20° about the camera's y axis, which points down.
        let turned = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
        let nose = turned.map(|row| -row[2]);

        assert!(nose[0] > 0.0, "{nose:?}");
        assert!((camera_yaw_deg(&turned) - 20.0).abs() < 1e-9);
        let facing = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(camera_yaw_deg(&facing).abs() < 1e-12);
    }

    /// The head anchors' x against the yaw: the correlation and the slope
    /// in mm per degree, from eleven pairs on.
    #[test]
    fn the_head_x_is_fitted_to_the_yaw_from_eleven_pairs() {
        let pairs: Vec<(f64, f64)> = (0..11)
            .map(|k| {
                let yaw = f64::from(k) * 3.0 - 15.0;
                (yaw, -1.5 * yaw + 20.0)
            })
            .collect();

        let (r, slope) = yaw_vs_head_x(&pairs).expect("eleven pairs");
        assert!((r + 1.0).abs() < 1e-12, "r {r}");
        assert!((slope + 1.5).abs() < 1e-12, "slope {slope}");
        assert_eq!(yaw_vs_head_x(&pairs[..10]), None);
    }

    /// Of a valid pose, its position in mm and its rotation in degrees;
    /// of an invalid one, whose values are the last valid pose's, only
    /// whether there was a face.
    #[test]
    fn a_pose_line_gives_a_valid_pose_in_mm_and_degrees() {
        let pose = HeadPose {
            valid: true,
            position_mm: [12.34, -5.0, 600.0],
            rotation_rad: [0.1, -0.2, 0.3],
        };

        assert_eq!(
            pose_line(&pose, true),
            "head pose: position ( +12.3,   -5.0, +600.0) mm, rotation x  +5.7 y -11.5 z +17.2 deg"
        );
        let held = HeadPose {
            valid: false,
            ..pose
        };
        assert_eq!(
            pose_line(&held, true),
            "head pose: invalid, a face but too near the image's edge, or no display area"
        );
        assert_eq!(pose_line(&held, false), "head pose: invalid, no face");
    }

    /// The last line counts the images, the valid poses (with their share)
    /// and the faces, and gives the model runs.
    #[test]
    fn the_tally_counts_images_valid_poses_and_faces() {
        let mut tally = PoseTally::default();
        let mut runs = ModelRuns::default();
        assert_eq!(
            tally.summary(runs),
            "head poses of 0 images: 0 valid (0.0%), a face in 0; \
             landmark model run 0 times, face detector 0"
        );

        for (valid, face) in [(true, true), (false, true), (false, false)] {
            tally.add(valid, face);
        }
        (runs.landmarks, runs.detector) = (4, 1);

        assert_eq!(
            tally,
            PoseTally {
                images: 3,
                valid: 1,
                faces: 2
            }
        );
        assert_eq!(
            tally.summary(runs),
            "head poses of 3 images: 1 valid (33.3%), a face in 2; \
             landmark model run 4 times, face detector 1"
        );
    }

    /// The poses are in the display frame of the area the init capture
    /// writes, which the report names; without one, or with one that fixes
    /// no frame, it says every pose is invalid.
    #[test]
    fn the_report_names_the_display_area_the_init_capture_writes() {
        let packets = tobii_usb::calibration::embedded_packets().expect("the embedded init");
        let (_, area, _) = written_display_area(&packets).expect("a 1440");

        let note = display_note(Some(&area));
        assert!(
            note.starts_with(
                "head poses in the display frame of the area the init capture writes (mm): \
                 top left ("
            ),
            "{note}"
        );
        let corner = |p: [f64; 3]| format!("({:.1}, {:.1}, {:.1})", p[0], p[1], p[2]);
        assert!(note.ends_with(&format!(
            "top left {}, top right {}, bottom left {}",
            corner(area.top_left_mm),
            corner(area.top_right_mm),
            corner(area.bottom_left_mm)
        )));
        let point = DisplayArea {
            top_right_mm: area.top_left_mm,
            bottom_left_mm: area.top_left_mm,
            ..area
        };
        assert_eq!(
            display_note(Some(&point)),
            "the display area the init capture writes fixes no display frame: \
             every head pose is invalid"
        );
        assert_eq!(
            display_note(None),
            "the init capture writes no display area: every head pose is invalid"
        );
    }

    /// With the real models but no face: a black image makes an invalid
    /// pose (and the report's first line), and an image the tracker cannot
    /// take fails the command rather than counting as no face.
    #[test]
    fn a_live_pose_without_a_face_is_invalid_and_a_tracker_failure_fails() {
        let packets = tobii_usb::calibration::embedded_packets().expect("the embedded init");
        let mut poses = LivePoses::new(&packets).expect("the models");
        assert!(
            poses.display.is_some(),
            "the embedded init's area fixes a frame"
        );
        let black = ImageFrame {
            device_ts_us: 1_000_000,
            width: 280,
            height: 280,
            pixels: vec![0; 280 * 280],
        };

        let line = poses.image(&black, 1).expect("a black image");
        assert_eq!(line.as_deref(), Some("head pose: invalid, no face"));
        for k in 1..10 {
            let later = ImageFrame {
                device_ts_us: 1_000_000 + 30_000 * k,
                ..black.clone()
            };
            assert_eq!(poses.image(&later, 1).expect("a black image"), None);
        }
        assert_eq!(
            poses.tally,
            PoseTally {
                images: 10,
                valid: 0,
                faces: 0
            }
        );

        let small = ImageFrame {
            width: 10,
            height: 10,
            pixels: vec![0; 100],
            ..black
        };
        let error = poses.image(&small, 1).expect_err("a 10x10 image");
        assert!(
            format!("{error:#}").starts_with("the head tracker failed on an image: "),
            "{error:#}"
        );
        assert_eq!(poses.tally.images, 10, "a failure is not counted");
    }

    /// With a face (the 0x50e fixture, `TOBII_IMAGE83_FIXTURE`, the user's
    /// face and so not committed; without it the test returns at once): a
    /// valid pose, as tobiid's engine makes it.
    #[test]
    fn a_live_pose_of_a_face_is_valid() {
        let Ok(path) = std::env::var("TOBII_IMAGE83_FIXTURE") else {
            return;
        };
        let msg = std::fs::read(path).expect("the fixture");
        let face = decode_image_payload(&msg).expect("a 0x50e image");
        let packets = tobii_usb::calibration::embedded_packets().expect("the embedded init");
        let mut poses = LivePoses::new(&packets).expect("the models");

        let line = poses.image(&face, 1).expect("the fixture's image");

        let line = line.expect("the first image's line");
        assert!(line.starts_with("head pose: position ("), "{line}");
        assert_eq!(
            poses.tally,
            PoseTally {
                images: 1,
                valid: 1,
                faces: 1
            }
        );
    }
}
