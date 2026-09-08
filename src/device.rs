//! USB transport and the live 0x83 engine for the Tobii Eye Tracker 5
//! (`2104:0313`).
//!
//! This is the production path: open and claim the device, replay the captured
//! init sequence, then decode the multiplexed gaze (0x500) and IR image (0x50e)
//! streams on EP 0x83, running head-pose inference on the images. The UVC
//! camera and every offline/diagnostic subcommand live in [`crate::devcmd`].

use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};
use std::{error, fmt, thread};
use tracing::{debug, error, info, warn};

use crate::decode::{TrackingFrame, decode_stream_payload};
use crate::engine::{GazeSample, PoseSample, Sample};
use crate::image83::{ImageFrame, decode_image_payload, upscale2x_into};
use crate::log::{PacketLog, log_packet};
use crate::protocol::{
    BulkReassembler, InitPacket, STREAM_ID_GAZE, STREAM_ID_IMAGE, declared_len, marker,
    parse_init_packets, seq, stream_id, stream_start_packet, stream_stop_packet,
};
use crate::time::now_us;

/// USB vendor id of the Tobii Eye Tracker 5.
pub(crate) const VID: u16 = 0x2104;

/// USB product id of the Tobii Eye Tracker 5.
pub(crate) const PID: u16 = 0x0313;

/// Interface carrying the processed-stream endpoints.
pub(crate) const IFACE: u8 = 0;

/// Bulk IN endpoint of the multiplexed processed stream.
pub(crate) const EP_IN: u8 = 0x83;

/// Bulk OUT endpoint for command messages (what the init replay writes to).
pub(crate) const EP_OUT: u8 = 0x05;

/// One read must hold the largest multiplexed message on EP 0x83: the 0x50e
/// image stream is 78609 bytes. Every message ends in a short USB packet, so a
/// buffer this size normally returns exactly one whole message per read.
pub(crate) const READ_BUF: usize = 128 * 1024;

/// Full open+init attempts before giving up on a device that never streams.
pub(crate) const MAX_REPLAY_ATTEMPTS: usize = 5;

/// Bulk-flag parsing shared by the `TOBII_*` opt-out variables: set and
/// neither empty nor `"0"`.
fn is_env_flag_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Retryable failure: the device accepted the init but never streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamStartupTimeout;

impl fmt::Display for StreamStartupTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream did not start after initialization")
    }
}

impl error::Error for StreamStartupTimeout {}

/// Find the tracker on the bus, open it and claim interface 0.
pub(crate) fn open_tobii(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
    let devices = ctx.devices()?;
    let mut found = None;

    for dev in devices.iter() {
        let desc = dev.device_descriptor()?;

        if desc.vendor_id() == VID && desc.product_id() == PID {
            info!(bus = dev.bus_number(), addr = dev.address(), "found Tobii");
            found = Some(dev);
            break;
        }
    }

    let dev = found.context("Tobii 2104:0313 not found by libusb")?;
    let h = dev
        .open()
        .context("failed to open Tobii device; try sudo")?;

    if let Err(e) = h.set_auto_detach_kernel_driver(true) {
        debug!(error = ?e, "auto detach kernel driver not available");
    }

    match h.set_active_configuration(1) {
        Ok(()) => debug!("set active configuration 1"),
        Err(rusb::Error::Busy) => debug!("configuration 1 already active or busy"),
        Err(e) => warn!(error = ?e, "set_active_configuration(1) failed"),
    }

    if h.kernel_driver_active(IFACE).unwrap_or(false) {
        debug!(interface = IFACE, "detaching kernel driver");
        let _ = h.detach_kernel_driver(IFACE);
    }

    h.claim_interface(IFACE)
        .context("failed to claim interface 0")?;

    debug!(interface = IFACE, "claimed interface");
    Ok(h)
}

/// The vendor control handshake that powers the sensor and starts the
/// processed stream firmware-side (requests 48 / 70 / 65).
pub(crate) fn vendor_control_init(h: &mut rusb::DeviceHandle<UsbContext>) -> Result<()> {
    let timeout = Duration::from_millis(2000);

    let init_data: [u8; 24] = [
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let n = h.write_control(0x41, 48, 0, 0, &init_data, timeout)?;
    debug!(written = n, "control OUT 48");

    let mut buf = [0u8; 512];
    let n = h.read_control(0xC1, 48, 0, 0, &mut buf, timeout)?;
    debug!(read = n, "control IN 48");

    let mut buf8 = [0u8; 8];
    let n = h.read_control(0xC1, 70, 1, 0, &mut buf8, timeout)?;
    debug!(
        read = n,
        data = %format_args!("{:02x?}", &buf8[..n]),
        "control IN 70"
    );

    let n = h.write_control(0x41, 65, 0, 0, &[], timeout)?;
    debug!(written = n, "control OUT 65");

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
        warn!(error = %e, "tobii deinit: control OUT 66 failed");
    }
}

/// The seq an init/command packet expects echoed in its `0x52` response:
/// only `0x51` command packets get one.
#[must_use]
pub(crate) fn command_seq(data: &[u8]) -> Option<u32> {
    if marker(data) == Some(0x51) {
        seq(data)
    } else {
        None
    }
}

/// Where `wait_for_response_seq` reports each message it reads while waiting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResponseEcho {
    /// Echo to stdout as program output (the `replay` handshake trace).
    Stdout,
    /// `debug!` only (daemon / engine / diagnostics that own their stdout).
    Log,
}

/// Read EP `0x83` until the `0x52` response carrying `expected_seq` arrives,
/// logging every message read meanwhile to `log`.
pub(crate) fn wait_for_response_seq(
    h: &mut rusb::DeviceHandle<UsbContext>,
    expected_seq: u32,
    log: &mut Option<PacketLog>,
    echo: ResponseEcho,
) -> Result<()> {
    let mut buf = [0u8; 8192];

    loop {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(2000)) {
            Ok(n) => {
                let data = &buf[..n];
                log_packet(log, EP_IN, data)?;

                match echo {
                    // Part of the `replay` subcommand's handshake trace on
                    // stdout, paired with its `OUT #...` lines.
                    ResponseEcho::Stdout => println!(
                        "  IN len={} marker={:?} seq={:?} declared_len={:?}",
                        n,
                        marker(data),
                        seq(data),
                        declared_len(data)
                    ),
                    ResponseEcho::Log => debug!(
                        len = n,
                        marker = ?marker(data),
                        seq = ?seq(data),
                        declared_len = ?declared_len(data),
                        "IN"
                    ),
                }

                if marker(data) == Some(0x52) && seq(data) == Some(expected_seq) {
                    return Ok(());
                }
            }
            Err(e) => {
                anyhow::bail!("waiting response seq {expected_seq} failed: {e:?}");
            }
        }
    }
}

/// Best-effort USB reset to drop any state a prior session left on the device,
/// so the replayed init starts from a known baseline. Re-enumeration is what we
/// want; we then wait for the device to reappear (by presence, not a blind
/// sleep) so the following init doesn't race a missing device.
///
/// Set `TOBII_NO_RESET=1` to skip it (faster start; try this now that uvcvideo
/// is kept off the device — the reset may no longer be needed).
fn reset_device_baseline(ctx: &UsbContext) {
    if is_env_flag_set("TOBII_NO_RESET") {
        return;
    }
    if let Ok(h) = open_tobii(ctx) {
        match h.reset() {
            Ok(()) => info!("reset Tobii to baseline before init"),
            Err(e) => warn!(error = ?e, "best-effort reset failed"),
        }
        drop(h); // device may re-enumerate; reopen fresh below
    }
    // Poll for re-enumeration to finish (cap ~3s) instead of sleeping blindly.
    for _ in 0..30 {
        thread::sleep(Duration::from_millis(100));
        if is_tobii_present(ctx) {
            thread::sleep(Duration::from_millis(150)); // brief settle
            return;
        }
    }
}

/// Is the Tobii plugged in right now? (Cheap check for the daemon watchdog.)
#[must_use]
pub(crate) fn is_device_present() -> bool {
    UsbContext::new()
        .map(|ctx| is_tobii_present(&ctx))
        .unwrap_or(false)
}

/// Is the Tobii on the bus right now (without opening/claiming it)?
fn is_tobii_present(ctx: &UsbContext) -> bool {
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

/// Own the device for the daemon engine and stream samples into `tx` until
/// `stop` is set (see the note above [`RESET_ESCALATION_ATTEMPT`]).
///
/// `stop`, `recenter` and `head_wanted` are pure signals (no data is
/// published through them), so every access uses `Ordering::Relaxed`.
///
/// # Errors
///
/// Returns the last attempt's error once the device fails to stream after
/// [`MAX_REPLAY_ATTEMPTS`] opens, or immediately on a non-retryable USB error.
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
    let worker = is_image_stream_enabled().then(|| {
        let mailbox = mailbox.clone();
        let stop = stop.clone();
        let recenter = recenter.clone();
        let head_wanted = head_wanted.clone();
        let tx = tx.clone();
        thread::spawn(move || pose_worker(&mailbox, &stop, &recenter, &head_wanted, &tx))
    });

    let result = (|| {
        for attempt in 1..=MAX_REPLAY_ATTEMPTS {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            // Normal cold start arms by attempt 2 and never gets here; reaching the
            // escalation means the device is genuinely stuck — reset to recover.
            if attempt == RESET_ESCALATION_ATTEMPT {
                warn!(
                    opens = attempt - 1,
                    "gaze: stream still not arming; USB-reset to recover"
                );
                reset_device_baseline(&ctx);
            }
            match gaze_engine_attempt(&ctx, stop, tx, &mailbox) {
                Ok(()) => return Ok(()),
                Err(e) if attempt < MAX_REPLAY_ATTEMPTS && !stop.load(Ordering::Relaxed) => {
                    // A cold device accepts the first init but doesn't start
                    // streaming; a fresh re-open is what arms it (expected, not an
                    // error). Report real failures loudly, the prime quietly.
                    if e.downcast_ref::<StreamStartupTimeout>().is_some() {
                        info!(
                            attempt,
                            "gaze: stream not armed; re-opening to prime (cold-start quirk)"
                        );
                    } else {
                        error!(
                            attempt,
                            error = format_args!("{e:#}"),
                            "tobii gaze init failed; re-opening"
                        );
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
    // Relaxed: a pure signal, the mailbox mutex orders the data hand-off.
    stop.store(true, Ordering::Relaxed);
    mailbox.1.notify_all();
    if let Some(worker) = worker {
        let _ = worker.join();
    }
    result
}

/// Single-slot hand-off of the newest image frame to the pose worker.
type PoseMailbox = Arc<(Mutex<Option<ImageFrame>>, Condvar)>;

/// Replace the mailbox slot with `frame` (dropping any frame the worker
/// has not taken yet) and wake the worker.
///
/// The slot only ever holds an `Option<ImageFrame>` that is written whole,
/// so a poisoned lock (a panicking worker) leaves nothing half-updated and
/// the guard is reused rather than propagating the panic to the USB reader.
fn mailbox_put(mailbox: &PoseMailbox, frame: ImageFrame) {
    let (lock, cv) = &**mailbox;
    *lock.lock().unwrap_or_else(PoisonError::into_inner) = Some(frame);
    cv.notify_one();
}

/// Head-pose inference loop over 0x50e frames (see `run_gaze_engine`).
fn pose_worker(
    mailbox: &PoseMailbox,
    stop: &AtomicBool,
    recenter: &AtomicBool,
    head_wanted: &AtomicBool,
    tx: &Sender<Sample>,
) {
    let mut tracker = match crate::track::Tracker::new_image83() {
        Ok(t) => t,
        Err(e) => {
            error!(
                error = format_args!("{e:#}"),
                "image83 pose worker disabled"
            );
            return;
        }
    };
    // `TOBII_IMAGE83_DEBUG=1`: report frames taken / poses / inference time.
    let report_stats = is_env_flag_set("TOBII_IMAGE83_DEBUG");
    let (mut taken, mut posed, mut infer_us) = (0u64, 0u64, 0u64);
    let mut last_report = Instant::now();
    // The tracker outlives head clients (a gaze client keeps the engine
    // alive), so a new head subscriber must not inherit an old rest pose or
    // smoothing state: recalibrate on every off->on edge after the first.
    let mut was_wanted: Option<bool> = None;
    // Reused across frames so the 33 Hz loop does not allocate a fresh
    // 560x560 buffer every time (mem-reuse-collections).
    let mut upscaled = Vec::new();
    let (lock, cv) = &**mailbox;
    // Relaxed everywhere below: the flags are pure signals; the frame itself
    // is handed over under the mailbox mutex.
    while !stop.load(Ordering::Relaxed) {
        let frame = {
            // Poisoning cannot leave the slot half-written (see `mailbox_put`).
            let mut slot = lock.lock().unwrap_or_else(PoisonError::into_inner);
            while slot.is_none() && !stop.load(Ordering::Relaxed) {
                slot = cv
                    .wait_timeout(slot, Duration::from_millis(200))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
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
        upscale2x_into(&frame.pixels, frame.width, frame.height, &mut upscaled);
        match tracker.process(&upscaled, frame.width * 2, frame.height * 2) {
            Ok(Some(p)) => {
                posed += 1;
                let _ = tx.send(Sample::Pose(PoseSample {
                    timestamp_us: to_i64_us(now_us()),
                    pos_cm: [p[0], p[1], p[2]],
                    rot_deg: [p[3], p[4], p[5]],
                }));
            }
            Ok(None) => {}
            Err(e) => warn!(error = format_args!("{e:#}"), "image83 pose failed"),
        }
        taken += 1;
        infer_us += u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
        if report_stats && last_report.elapsed() >= Duration::from_secs(5) {
            let dt = last_report.elapsed().as_secs_f64();
            info!(
                frames_per_s = format_args!("{:.1}", taken as f64 / dt),
                poses_per_s = format_args!("{:.1}", posed as f64 / dt),
                mean_ms = format_args!("{:.1}", infer_us as f64 / 1000.0 / taken.max(1) as f64),
                "image83 pose worker"
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
    stop: &AtomicBool,
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

/// Replay the init-packet sequence once on `h`, waiting (best-effort) for
/// each command's `0x52` echo. Returns early without error once `stop` is
/// set; a failed write ends the replay with an error.
pub(crate) fn replay_init_packets(
    h: &mut rusb::DeviceHandle<UsbContext>,
    packets: &[InitPacket],
    stop: Option<&AtomicBool>,
) -> Result<()> {
    let mut no_log: Option<PacketLog> = None;
    for pkt in packets {
        if stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
            return Ok(());
        }
        let expected_seq = command_seq(&pkt.data);
        h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000))
            .context("init packet write failed")?;
        if let Some(s) = expected_seq {
            let _ = wait_for_response_seq(h, s, &mut no_log, ResponseEcho::Log);
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
pub(crate) fn wait_for_gaze_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &AtomicBool,
    dur: Duration,
    asm: &mut BulkReassembler,
) -> bool {
    let mut buf = vec![0u8; READ_BUF];
    let mut msgs = Vec::new();
    let start = Instant::now();
    while start.elapsed() < dur && !stop.load(Ordering::Relaxed) {
        if let Ok(n) = h.read_bulk(EP_IN, &mut buf, Duration::from_millis(200)) {
            asm.push_into(&buf[..n], &mut msgs);
            for msg in &msgs {
                if stream_id(msg) == Some(STREAM_ID_GAZE)
                    && decode_stream_payload(msg).is_ok_and(|d| !d.is_empty())
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
fn is_image_stream_enabled() -> bool {
    !is_env_flag_set("TOBII_NO_IMAGE")
}

/// Microsecond timestamps as the `i64` the sample structs carry. `u64`
/// values past `i64::MAX` (never in practice) saturate instead of wrapping.
#[must_use]
fn to_i64_us(us: u64) -> i64 {
    i64::try_from(us).unwrap_or(i64::MAX)
}

/// The command sequence number to use after the init replay: one past the
/// highest 0x51 seq in the replayed packets (the device echoes it in the 0x52
/// response, which is how we match replies).
pub(crate) fn next_command_seq(packets: &[InitPacket]) -> u32 {
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

/// Ask the device to start stream `id` (command 1220) and wait for the ack.
pub(crate) fn start_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    cmd_seq: u32,
    id: u32,
) -> Result<()> {
    send_command(
        h,
        &stream_start_packet(cmd_seq, id),
        Duration::from_millis(1500),
    )
    .with_context(|| format!("start stream {id:#x}"))
}

/// Ask the device to stop stream `id` (command 1230) and wait for the ack.
pub(crate) fn stop_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    cmd_seq: u32,
    id: u32,
) -> Result<()> {
    send_command(
        h,
        &stream_stop_packet(cmd_seq, id),
        Duration::from_millis(700),
    )
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
    stop: &AtomicBool,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
) -> Result<()> {
    const INIT_PACKETS: &str = include_str!("../init_packets_ep.txt");
    let packets = parse_init_packets(INIT_PACKETS)?;
    replay_init_packets(h, &packets, Some(stop))?;
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    // The Windows Stream Engine subscribes the image stream right after its
    // init; we do the same. Failure here is not fatal — gaze still works.
    let mut cmd_seq = next_command_seq(&packets);
    let mut image = is_image_stream_enabled();
    if image {
        match start_stream(h, cmd_seq, STREAM_ID_IMAGE) {
            Ok(()) => info!("gaze: image stream 0x50e requested (head pose via IR frames)"),
            Err(e) => {
                warn!(
                    error = format_args!("{e:#}"),
                    "gaze: image stream start failed; continuing gaze-only"
                );
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
    stop: &AtomicBool,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
    asm: &mut BulkReassembler,
) -> Result<()> {
    let mut buf = vec![0u8; READ_BUF];
    let mut packet_no = 0u64;
    let mut image_live = false;
    // Reused across reads (mem-reuse-collections).
    let mut msgs = Vec::new();
    let mut last_gaze = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if last_gaze.elapsed() > GAZE_LIVENESS_TIMEOUT {
            warn!(timeout = ?GAZE_LIVENESS_TIMEOUT, "gaze: no gaze frame; re-opening the device");
            anyhow::bail!(StreamStartupTimeout);
        }
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(500)) {
            Ok(n) if n > 0 => {
                asm.push_into(&buf[..n], &mut msgs);
                for msg in &msgs {
                    match stream_id(msg) {
                        Some(STREAM_ID_GAZE) => {
                            let decoded = decode_stream_payload(msg)?;
                            if decoded.is_empty() {
                                continue;
                            }
                            last_gaze = Instant::now();
                            let frame = TrackingFrame::from_decoded(packet_no, &decoded);
                            packet_no += 1;
                            let _ = tx.send(Sample::Gaze(GazeSample {
                                timestamp_us: to_i64_us(frame.ts_us),
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
                            if let Some(frame) = decode_image_payload(msg) {
                                if !image_live {
                                    info!(
                                        width = frame.width,
                                        height = frame.height,
                                        "gaze: image stream 0x50e live"
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
