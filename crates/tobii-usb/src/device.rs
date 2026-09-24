//! USB transport and the live 0x83 engine for the Tobii Eye Tracker 5
//! (`2104:0313`).
//!
//! This is the production path: open and claim the device, replay the captured
//! init sequence, then decode the multiplexed gaze (0x500) and IR image (0x50e)
//! streams on EP 0x83, running head-pose inference on the images. The UVC
//! camera and every offline/diagnostic subcommand live in the research CLI.

use anyhow::{Context, Result};
use rusb::{Context as UsbContext, UsbContext as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};
use std::{error, fmt, thread};
use tracing::{debug, error, info, warn};

use crate::engine::{
    CommandError, CommandResponse, GazeSample, PoseSample, PresenceSample, QueuedCommand, Sample,
    Shared,
};
use std::sync::mpsc::Receiver;
use tobii_proto::facts::{DeviceFacts, DeviceNotification, decode_notification};
use tobii_proto::gaze83::{GazeFrame, PresenceFrame, decode_gaze_frame, decode_presence_frame};
use tobii_proto::image83::{ImageFrame, decode_image_payload, upscale2x_into};
use tobii_proto::log::{PacketLog, log_packet};
use tobii_proto::protocol::{
    BulkReassembler, InitPacket, MARKER_COMMAND, MARKER_NOTIFICATION, MARKER_RESPONSE,
    MARKER_STREAM, STREAM_ID_GAZE, STREAM_ID_IMAGE, STREAM_ID_PRESENCE, chunk_command,
    declared_len, marker, parse_message, seq, stream_id, stream_start_packet, stream_stop_packet,
};
use tobii_proto::time::now_us;

/// USB vendor id of the Tobii Eye Tracker 5.
pub(crate) const VID: u16 = 0x2104;

/// USB product id of the Tobii Eye Tracker 5.
pub(crate) const PID: u16 = 0x0313;

/// Interface carrying the processed-stream endpoints.
pub(crate) const IFACE: u8 = 0;

/// Bulk IN endpoint of the multiplexed processed stream.
pub const EP_IN: u8 = 0x83;

/// Bulk OUT endpoint for command messages (what the init replay writes to).
pub(crate) const EP_OUT: u8 = 0x05;

/// One read must hold the largest multiplexed message on EP 0x83: the 0x50e
/// image stream is 78609 bytes. Every message ends in a short USB packet, so a
/// buffer this size normally returns exactly one whole message per read.
pub const READ_BUF: usize = 128 * 1024;

/// Full open+init attempts before giving up on a device that never streams.
pub const MAX_REPLAY_ATTEMPTS: usize = 5;

/// Bulk-flag parsing shared by the `TOBII_*` opt-out variables: set and
/// neither empty nor `"0"`.
fn is_env_flag_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Retryable failure: the device accepted the init but never streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamStartupTimeout;

impl fmt::Display for StreamStartupTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream did not start after initialization")
    }
}

impl error::Error for StreamStartupTimeout {}

/// Find the tracker on the bus, open it and claim interface 0.
///
/// # Errors
///
/// Fails when no matching device is present, or when it cannot be opened,
/// configured or claimed (usually a permissions or busy-device problem).
pub fn open_tobii(ctx: &UsbContext) -> Result<rusb::DeviceHandle<UsbContext>> {
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
///
/// # Errors
///
/// Propagates a failing control transfer.
pub fn vendor_control_init(h: &mut rusb::DeviceHandle<UsbContext>) -> Result<()> {
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
pub fn vendor_control_deinit(h: &mut rusb::DeviceHandle<UsbContext>) {
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
pub fn command_seq(data: &[u8]) -> Option<u32> {
    if marker(data) == Some(MARKER_COMMAND) {
        seq(data)
    } else {
        None
    }
}

/// Where `wait_for_response_seq` reports each message it reads while waiting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseEcho {
    /// Echo to stdout as program output (the `replay` handshake trace).
    Stdout,
    /// `debug!` only (daemon / engine / diagnostics that own their stdout).
    Log,
}

/// Read EP `0x83` until the `0x52` response carrying `expected_seq` arrives,
/// logging every message read meanwhile to `log`, and return that response.
///
/// # Errors
///
/// Fails when the bulk read fails, or when writing to `log` fails.
pub fn wait_for_response_seq(
    h: &mut rusb::DeviceHandle<UsbContext>,
    expected_seq: u32,
    log: &mut Option<PacketLog>,
    echo: ResponseEcho,
) -> Result<Vec<u8>> {
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

                if marker(data) == Some(MARKER_RESPONSE) && seq(data) == Some(expected_seq) {
                    return Ok(data.to_vec());
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
pub fn is_device_present() -> bool {
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
/// `shared.stop` is set, running the commands that arrive on `commands` in
/// between (see the note on the reset-escalation constant above).
///
/// # Errors
///
/// Returns the last attempt's error once the device fails to stream after
/// [`MAX_REPLAY_ATTEMPTS`] opens, or immediately on a non-retryable USB error.
pub(crate) fn run_gaze_engine(
    shared: &Arc<Shared>,
    commands: &Receiver<QueuedCommand>,
    tx: &Sender<Sample>,
) -> Result<()> {
    let ctx = UsbContext::new()?;
    let stop = &shared.stop;

    // Head pose from the 0x50e image stream runs on its own thread so a slow
    // inference never blocks the USB reader (an unread IN buffer stalls the
    // firmware). The reader drops each new frame into a single-slot mailbox;
    // the worker always takes the newest one and skips whatever it missed.
    let mailbox: PoseMailbox = Arc::new((Mutex::new(None), Condvar::new()));
    // No worker (and no ONNX session) when the image stream is disabled.
    let worker = is_image_stream_enabled().then(|| {
        let mailbox = mailbox.clone();
        let shared = Arc::clone(shared);
        let tx = tx.clone();
        thread::spawn(move || pose_worker(&mailbox, &shared, &tx))
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
            match gaze_engine_attempt(&ctx, shared, commands, tx, &mailbox) {
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
type PoseMailbox = Arc<(Mutex<Option<Arc<ImageFrame>>>, Condvar)>;

/// Replace the mailbox slot with `frame` (dropping any frame the worker
/// has not taken yet) and wake the worker.
///
/// The slot only ever holds an `Option` that is written whole, so a poisoned
/// lock (a panicking worker) leaves nothing half-updated and the guard is
/// reused rather than propagating the panic to the USB reader.
fn mailbox_put(mailbox: &PoseMailbox, frame: Arc<ImageFrame>) {
    let (lock, cv) = &**mailbox;
    *lock.lock().unwrap_or_else(PoisonError::into_inner) = Some(frame);
    cv.notify_one();
}

/// Head-pose inference loop over 0x50e frames (see `run_gaze_engine`).
fn pose_worker(mailbox: &PoseMailbox, shared: &Shared, tx: &Sender<Sample>) {
    let mut tracker = match tobii_pose::track::Tracker::new_image83() {
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
    while !shared.stop.load(Ordering::Relaxed) {
        let frame = {
            // Poisoning cannot leave the slot half-written (see `mailbox_put`).
            let mut slot = lock.lock().unwrap_or_else(PoisonError::into_inner);
            while slot.is_none() && !shared.stop.load(Ordering::Relaxed) {
                slot = cv
                    .wait_timeout(slot, Duration::from_millis(200))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            slot.take()
        };
        let Some(frame) = frame else { continue };
        if shared.recenter.swap(false, Ordering::Relaxed) {
            tracker.recenter();
        }
        let wanted = shared.head_wanted.load(Ordering::Relaxed);
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
                    timestamp_us: to_i64_us(frame.device_ts_us),
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

/// One open+init+read of the 0x83 stream, pushing samples until `stop` or a
/// failure. Bails with `StreamStartupTimeout` if no stream packet arrives in 5s.
fn gaze_engine_attempt(
    ctx: &UsbContext,
    shared: &Shared,
    commands: &Receiver<QueuedCommand>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
) -> Result<()> {
    let mut h = open_tobii(ctx)?;
    vendor_control_init(&mut h)?;
    // From here the 0x83 stream is running firmware-side (started by request 65
    // inside vendor_control_init). Guarantee the Windows-style stop (request 66)
    // on every exit — clean stop, error, or timeout — so the device isn't left
    // mid-stream and the next open starts from a defined state.
    let result = gaze_stream_loop(&mut h, shared, commands, tx, mailbox);
    vendor_control_deinit(&mut h);
    result
}

/// What the init replay collected: each command's response, and the other
/// messages (stream, notification) that arrived while it waited.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InitCapture {
    /// The 0x52 responses, in command order.
    pub responses: Vec<Vec<u8>>,
    /// Stream and notification messages read in between (bounded).
    pub side: Vec<Vec<u8>>,
}

/// How many non-response messages the init replay keeps for its caller.
const MAX_SIDE_MESSAGES: usize = 64;

/// How long the init replay waits for each command's response.
const INIT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(2);

/// How long one read waits while draining EP 0x83 between the pieces of a
/// chunked command: only what the device already sent is wanted.
const DRAIN_READ_TIMEOUT: Duration = Duration::from_millis(1);

/// Most reads per drain, so a device that keeps streaming cannot hold the
/// replay there.
const MAX_DRAIN_READS: usize = 16;

/// The init replay's read state: one reassembler and reusable buffers.
struct InitReader {
    asm: BulkReassembler,
    buf: Vec<u8>,
    msgs: Vec<Vec<u8>>,
}

impl InitReader {
    fn new() -> Self {
        Self {
            asm: BulkReassembler::new(),
            buf: vec![0u8; READ_BUF],
            msgs: Vec::new(),
        }
    }

    /// Read once, waiting up to `timeout`; the messages completed land in
    /// `self.msgs`. `Ok(false)` when nothing came; an error other than the
    /// timeout ends the waiting (retrying it would only spin: the next write
    /// reports it).
    fn read(
        &mut self,
        h: &mut rusb::DeviceHandle<UsbContext>,
        timeout: Duration,
    ) -> Result<bool, rusb::Error> {
        match h.read_bulk(EP_IN, &mut self.buf, timeout) {
            Ok(n) => {
                self.asm.push_into(&self.buf[..n], &mut self.msgs);
                Ok(true)
            }
            Err(rusb::Error::Timeout) => Ok(false),
            Err(e) => {
                debug!(error = ?e, "init: read failed");
                Err(e)
            }
        }
    }

    /// Read until the response to `expected_seq` is complete, keeping other
    /// messages in `side`. `None` on timeout.
    fn await_response(
        &mut self,
        h: &mut rusb::DeviceHandle<UsbContext>,
        expected_seq: u32,
        timeout: Duration,
        side: &mut Vec<Vec<u8>>,
    ) -> Option<Vec<u8>> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            match self.read(h, Duration::from_millis(200)) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(_) => return None,
            }
            let mut found = None;
            for msg in self.msgs.drain(..) {
                if marker(&msg) == Some(MARKER_RESPONSE) && seq(&msg) == Some(expected_seq) {
                    log_init_message(&msg);
                    found = Some(msg);
                } else {
                    keep_side(msg, side);
                }
            }
            if found.is_some() {
                return found;
            }
        }
        None
    }

    /// Take whatever EP 0x83 already holds, keeping it in `side`.
    fn drain(&mut self, h: &mut rusb::DeviceHandle<UsbContext>, side: &mut Vec<Vec<u8>>) {
        for _ in 0..MAX_DRAIN_READS {
            if !matches!(self.read(h, DRAIN_READ_TIMEOUT), Ok(true)) {
                return;
            }
            for msg in self.msgs.drain(..) {
                keep_side(msg, side);
            }
        }
    }
}

/// Trace one message read during the init replay, with a response's status
/// and error words (an empty 2120 answer shows up here).
fn log_init_message(msg: &[u8]) {
    let Some(m) = parse_message(msg) else {
        debug!(len = msg.len(), marker = ?marker(msg), seq = ?seq(msg), "IN");
        return;
    };
    debug!(
        len = msg.len(),
        marker = m.marker,
        seq = m.seq,
        id = m.id,
        status = m.status,
        error = format_args!("{:#x}", m.error),
        payload_len = m.payload_len,
        "IN"
    );
}

/// Keep a message read during the init replay for the caller, unless it is a
/// command response (those are matched by seq, not collected).
fn keep_side(msg: Vec<u8>, side: &mut Vec<Vec<u8>>) {
    log_init_message(&msg);
    if marker(&msg) != Some(MARKER_RESPONSE) && side.len() < MAX_SIDE_MESSAGES {
        side.push(msg);
    }
}

/// What the init replay does after writing one packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterWrite {
    /// More pieces of this command follow: take what the device sent.
    Drain,
    /// The command is complete: wait for its response.
    Await(u32),
    /// Nothing to wait for.
    Continue,
}

/// The step after writing `packets[i]`, given the seq of the command being
/// written (updated from `packets[i]` when it starts one).
fn after_write(packets: &[InitPacket], i: usize, awaiting: &mut Option<u32>) -> AfterWrite {
    if let Some(s) = packets.get(i).and_then(|p| command_seq(&p.data)) {
        *awaiting = Some(s);
    }
    let command_complete = packets
        .get(i + 1)
        .is_none_or(|next| command_seq(&next.data).is_some());
    if !command_complete {
        AfterWrite::Drain
    } else if let Some(s) = awaiting.take() {
        AfterWrite::Await(s)
    } else {
        AfterWrite::Continue
    }
}

/// Replay the init-packet sequence once on `h`, collecting each command's
/// `0x52` response (best-effort: a missing one is logged and skipped).
/// A command split over several writes is answered after its last piece, so
/// the wait happens there. Between the pieces EP 0x83 is drained: the device
/// sends a notification after the first piece of the calibration upload and
/// stops accepting further pieces until the host has read it. Returns early
/// without error once `stop` is set; a failed write ends the replay with an
/// error.
///
/// # Errors
///
/// Fails when an init packet cannot be written to the device.
pub fn replay_init_packets(
    h: &mut rusb::DeviceHandle<UsbContext>,
    packets: &[InitPacket],
    stop: Option<&AtomicBool>,
) -> Result<InitCapture> {
    let started = Instant::now();
    let mut capture = InitCapture::default();
    let mut reader = InitReader::new();
    let mut awaiting = None;
    for (i, pkt) in packets.iter().enumerate() {
        if stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
            return Ok(capture);
        }
        h.write_bulk(pkt.ep, &pkt.data, Duration::from_millis(2000))
            .with_context(|| {
                format!(
                    "init packet {i} write failed (command seq {:?}, {} ms in)",
                    command_seq(&pkt.data).or(awaiting),
                    started.elapsed().as_millis()
                )
            })?;
        match after_write(packets, i, &mut awaiting) {
            AfterWrite::Drain => reader.drain(h, &mut capture.side),
            AfterWrite::Await(s) => {
                match reader.await_response(h, s, INIT_RESPONSE_TIMEOUT, &mut capture.side) {
                    Some(rsp) => capture.responses.push(rsp),
                    None => debug!(seq = s, "init: no response"),
                }
            }
            AfterWrite::Continue => {}
        }
        thread::sleep(Duration::from_millis(2));
    }
    debug!(
        elapsed_ms = started.elapsed().as_millis(),
        responses = capture.responses.len(),
        "init replay done"
    );
    Ok(capture)
}

/// Read 0x83 for up to `dur`, returning true as soon as a decodable GAZE
/// (0x500) frame arrives. 0x52 handshake/config responses and any image /
/// presence messages are drained and ignored — an image frame must not count
/// as proof the gaze pipeline armed (the cold-start quirk leaves gaze silent
/// while a freshly requested 0x50e stream may run). Draining matters: an
/// unread IN buffer can stall the firmware. `asm` is shared with the pump so
/// a partially read message is not lost at the hand-off.
pub fn wait_for_gaze_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &AtomicBool,
    dur: Duration,
    asm: &mut BulkReassembler,
) -> bool {
    wait_for_gaze_stream_with(h, stop, dur, asm, |_| {})
}

/// [`wait_for_gaze_stream`], handing every other message to `side`.
fn wait_for_gaze_stream_with(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &AtomicBool,
    dur: Duration,
    asm: &mut BulkReassembler,
    mut side: impl FnMut(&[u8]),
) -> bool {
    let mut buf = vec![0u8; READ_BUF];
    let mut msgs = Vec::new();
    let mut seen = 0usize;
    let start = Instant::now();
    while start.elapsed() < dur && !stop.load(Ordering::Relaxed) {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(200)) {
            Ok(n) => {
                asm.push_into(&buf[..n], &mut msgs);
                for msg in &msgs {
                    seen += 1;
                    if stream_id(msg) == Some(STREAM_ID_GAZE)
                        && parse_message(msg)
                            .as_ref()
                            .and_then(decode_gaze_frame)
                            .is_some()
                    {
                        return true;
                    }
                    debug!(
                        len = msg.len(),
                        marker = ?marker(msg),
                        stream = ?stream_id(msg),
                        "arming: not a gaze frame"
                    );
                    side(msg);
                }
            }
            Err(rusb::Error::Timeout) => {}
            Err(e) => {
                // Not a stream that is merely slow to start: re-open.
                debug!(error = ?e, "arming: read failed");
                return false;
            }
        }
    }
    debug!(messages = seen, "arming: no gaze frame");
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
#[must_use]
pub fn next_command_seq(packets: &[InitPacket]) -> u32 {
    packets
        .iter()
        .filter(|p| marker(&p.data) == Some(MARKER_COMMAND))
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
            Ok(n)
                if marker(&buf[..n]) == Some(MARKER_RESPONSE)
                    && seq(&buf[..n]) == Some(expected) =>
            {
                return Ok(());
            }
            Ok(_) | Err(rusb::Error::Timeout) => {}
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("no response to command seq {expected} within {deadline:?}")
}

/// Ask the device to start stream `id` (command 1220) and wait for the ack.
///
/// # Errors
///
/// Fails when the command cannot be written or is not acknowledged in time.
pub fn start_stream(h: &mut rusb::DeviceHandle<UsbContext>, cmd_seq: u32, id: u32) -> Result<()> {
    send_command(
        h,
        &stream_start_packet(cmd_seq, id),
        Duration::from_millis(1500),
    )
    .with_context(|| format!("start stream {id:#x}"))
}

/// Ask the device to stop stream `id` (command 1230) and wait for the ack.
///
/// # Errors
///
/// Fails when the command cannot be written or is not acknowledged in time.
pub fn stop_stream(h: &mut rusb::DeviceHandle<UsbContext>, cmd_seq: u32, id: u32) -> Result<()> {
    send_command(
        h,
        &stream_stop_packet(cmd_seq, id),
        Duration::from_millis(700),
    )
    .with_context(|| format!("stop stream {id:#x}"))
}

/// A message from EP 0x83, classified.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Incoming {
    /// A 0x500 gaze frame (boxed: the largest variant by far).
    Gaze(Box<GazeFrame>),
    /// A 0x504 presence change.
    Presence(PresenceFrame),
    /// A 0x50e IR frame.
    Image(ImageFrame),
    /// A command response.
    Response {
        seq: u32,
        status: u32,
        error: u32,
        payload: Vec<u8>,
    },
    /// A notification.
    Notification(DeviceNotification),
    /// Anything else.
    Other,
}

/// Classify one whole message from EP 0x83.
pub(crate) fn classify(msg: &[u8]) -> Incoming {
    let Some(m) = parse_message(msg) else {
        return Incoming::Other;
    };
    match (m.marker, m.id) {
        (MARKER_STREAM, STREAM_ID_GAZE) => {
            decode_gaze_frame(&m).map_or(Incoming::Other, |f| Incoming::Gaze(Box::new(f)))
        }
        (MARKER_STREAM, STREAM_ID_PRESENCE) => {
            decode_presence_frame(&m).map_or(Incoming::Other, Incoming::Presence)
        }
        (MARKER_STREAM, STREAM_ID_IMAGE) => {
            decode_image_payload(msg).map_or(Incoming::Other, Incoming::Image)
        }
        (MARKER_RESPONSE, _) => Incoming::Response {
            seq: m.seq,
            status: m.status,
            error: m.error,
            payload: m.payload.to_vec(),
        },
        (MARKER_NOTIFICATION, _) => {
            decode_notification(&m).map_or(Incoming::Other, Incoming::Notification)
        }
        _ => Incoming::Other,
    }
}

/// A command written to the device and not yet answered.
#[derive(Debug)]
struct Outstanding {
    cmd: u32,
    seq: u32,
    deadline: Instant,
    reply: Sender<Result<CommandResponse, CommandError>>,
}

/// The live half of one attempt: reads EP 0x83, turns messages into samples,
/// and runs queued device commands one at a time between reads.
struct Pump<'a> {
    shared: &'a Shared,
    commands: &'a Receiver<QueuedCommand>,
    tx: &'a Sender<Sample>,
    mailbox: &'a PoseMailbox,
    cmd_seq: u32,
    outstanding: Option<Outstanding>,
    image_live: bool,
    last_gaze: Instant,
}

impl Pump<'_> {
    /// Route one message outside of command matching: samples to `tx`, image
    /// frames to the pose worker.
    fn deliver(&mut self, incoming: Incoming) {
        match incoming {
            Incoming::Gaze(frame) => {
                self.last_gaze = Instant::now();
                let _ = self.tx.send(Sample::Gaze(Box::new(GazeSample {
                    frame: *frame,
                    host_rx_us: to_i64_us(now_us()),
                })));
            }
            Incoming::Presence(p) => {
                let _ = self.tx.send(Sample::Presence(PresenceSample {
                    timestamp_us: to_i64_us(p.device_ts_us),
                    present: p.state == PRESENCE_STATE_PRESENT,
                }));
            }
            Incoming::Image(frame) => {
                if !self.image_live {
                    info!(
                        width = frame.width,
                        height = frame.height,
                        "gaze: image stream 0x50e live"
                    );
                    self.image_live = true;
                }
                let frame = Arc::new(frame);
                // Relaxed: a pure signal.
                if self.shared.image_wanted.load(Ordering::Relaxed) {
                    let _ = self.tx.send(Sample::Image(Arc::clone(&frame)));
                }
                mailbox_put(self.mailbox, frame);
            }
            Incoming::Notification(n) => {
                debug!(notification = ?n, "device notification");
                let _ = self.tx.send(Sample::Notification(n));
            }
            Incoming::Response {
                seq,
                status,
                error,
                payload,
            } => {
                if let Some(o) = self.outstanding.take_if(|o| o.seq == seq) {
                    debug!(
                        cmd = o.cmd,
                        seq,
                        status,
                        error = format_args!("{error:#x}"),
                        len = payload.len(),
                        "response"
                    );
                    let _ = o.reply.send(Ok(CommandResponse {
                        status,
                        error,
                        payload,
                    }));
                } else {
                    debug!(seq, "unsolicited response");
                }
            }
            Incoming::Other => {}
        }
    }

    /// Write the next queued command, if none is outstanding. Reads between
    /// the pieces of a large command so the stream keeps draining.
    fn send_next(&mut self, h: &mut rusb::DeviceHandle<UsbContext>, asm: &mut BulkReassembler) {
        if self.outstanding.is_some() {
            return;
        }
        let Ok(QueuedCommand { command, reply }) = self.commands.try_recv() else {
            return;
        };
        let seq = self.cmd_seq;
        self.cmd_seq += 1;
        let pieces = chunk_command(command.cmd, seq, &command.payload);
        debug!(cmd = command.cmd, seq, pieces = pieces.len(), "command");
        let mut buf = vec![0u8; READ_BUF];
        let mut msgs = Vec::new();
        for (i, piece) in pieces.iter().enumerate() {
            if let Err(e) = h.write_bulk(EP_OUT, piece, Duration::from_millis(2000)) {
                let _ = reply.send(Err(CommandError::Usb(e.to_string())));
                return;
            }
            if i + 1 < pieces.len() {
                // The device stops taking pieces while what it sent is unread
                // (see `replay_init_packets`); with the streams running that
                // includes image frames, so take everything it has.
                for _ in 0..PIECE_DRAIN_READS {
                    let Ok(n) = h.read_bulk(EP_IN, &mut buf, PIECE_DRAIN_TIMEOUT) else {
                        break;
                    };
                    asm.push_into(&buf[..n], &mut msgs);
                    for msg in msgs.drain(..) {
                        self.deliver(classify(&msg));
                    }
                }
            }
        }
        self.outstanding = Some(Outstanding {
            cmd: command.cmd,
            seq,
            deadline: Instant::now() + command.timeout,
            reply,
        });
    }

    /// Fail the outstanding command once its deadline has passed.
    fn expire(&mut self, asm: &mut BulkReassembler) {
        if let Some(o) = self.outstanding.take_if(|o| Instant::now() >= o.deadline) {
            warn!(cmd = o.cmd, seq = o.seq, "device command timed out");
            asm.abort_continuation();
            let _ = o.reply.send(Err(CommandError::Timeout));
        }
    }
}

/// How long one read waits between the pieces of a command while the
/// streams run: long enough for a whole 78 KB image message (about 2 ms on
/// the wire), so that a read is not cut off halfway through one and the
/// device's queue really empties. (A 1 ms read stalled a 164-piece
/// calibration upload that started as the image stream came up.)
const PIECE_DRAIN_TIMEOUT: Duration = Duration::from_millis(5);

/// Most reads between two pieces of a command.
const PIECE_DRAIN_READS: usize = 8;

/// Presence state the 0x504 stream uses for "a user is present".
const PRESENCE_STATE_PRESENT: u32 = 2;

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
    shared: &Shared,
    commands: &Receiver<QueuedCommand>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
) -> Result<()> {
    let stop = &shared.stop;
    let display_override = shared.display_override();
    let packets = crate::calibration::init_packets(display_override.as_ref())?;
    let capture = replay_init_packets(h, &packets, Some(stop))?;
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    let mut facts =
        DeviceFacts::from_messages(capture.responses.iter().filter_map(|r| parse_message(r)));
    if let Some(area) = display_override {
        facts.display_area = Some(area);
    }
    info!(
        model = %facts.info.model,
        firmware = %facts.info.firmware_version,
        calibration_id = ?facts.calibration_id,
        "device ready"
    );
    let _ = tx.send(Sample::DeviceReady(Arc::new(facts)));

    let mut pump = Pump {
        shared,
        commands,
        tx,
        mailbox,
        cmd_seq: next_command_seq(&packets),
        outstanding: None,
        image_live: false,
        last_gaze: Instant::now(),
    };
    for msg in &capture.side {
        pump.deliver(classify(msg));
    }

    // The Windows Stream Engine subscribes the image stream right after its
    // init; we do the same. Failure here is not fatal — gaze still works.
    let mut image = is_image_stream_enabled();
    if image {
        match start_stream(h, pump.cmd_seq, STREAM_ID_IMAGE) {
            Ok(()) => info!("gaze: image stream 0x50e requested (head pose via IR frames)"),
            Err(e) => {
                warn!(
                    error = format_args!("{e:#}"),
                    "gaze: image stream start failed; continuing gaze-only"
                );
                image = false;
            }
        }
        pump.cmd_seq += 1;
    }
    // The device's first 0x53 frame lands ~3s after a good init; give it margin.
    let mut asm = BulkReassembler::new();
    let mut early = Vec::new();
    let armed = wait_for_gaze_stream_with(h, stop, Duration::from_secs_f64(4.5), &mut asm, |msg| {
        if early.len() < MAX_SIDE_MESSAGES {
            early.push(msg.to_vec());
        }
    });
    for msg in &early {
        pump.deliver(classify(msg));
    }
    if !armed {
        if image {
            let _ = stop_stream(h, pump.cmd_seq, STREAM_ID_IMAGE);
        }
        anyhow::bail!(StreamStartupTimeout);
    }

    let result = pump_streams(h, &mut pump, &mut asm);
    if let Some(o) = pump.outstanding.take() {
        let _ = o.reply.send(Err(CommandError::EngineGone));
    }
    if image {
        // Mirror the Windows shutdown (1230 for 0x50e before the vendor stop).
        let _ = stop_stream(h, pump.cmd_seq, STREAM_ID_IMAGE);
    }
    result
}

/// Demultiplex EP 0x83 until `stop` or failure, running queued commands in
/// between reads.
fn pump_streams(
    h: &mut rusb::DeviceHandle<UsbContext>,
    pump: &mut Pump<'_>,
    asm: &mut BulkReassembler,
) -> Result<()> {
    let mut buf = vec![0u8; READ_BUF];
    // Reused across reads (mem-reuse-collections).
    let mut msgs = Vec::new();
    pump.last_gaze = Instant::now();
    while !pump.shared.stop.load(Ordering::Relaxed) {
        // A command the device is working on (a calibration point takes most
        // of a second) must not be mistaken for a dead stream.
        if pump.outstanding.is_none() && pump.last_gaze.elapsed() > GAZE_LIVENESS_TIMEOUT {
            warn!(timeout = ?GAZE_LIVENESS_TIMEOUT, "gaze: no gaze frame; re-opening the device");
            anyhow::bail!(StreamStartupTimeout);
        }
        pump.send_next(h, asm);
        pump.expire(asm);
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(100)) {
            Ok(n) if n > 0 => {
                asm.push_into(&buf[..n], &mut msgs);
                for msg in msgs.drain(..) {
                    pump.deliver(classify(&msg));
                }
            }
            Ok(_) | Err(rusb::Error::Timeout) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_proto::protocol::hex_to_bytes;

    fn fixture(name: &str) -> Vec<u8> {
        let path = format!(
            "{}/../tobii-proto/fixtures/{name}.hex",
            env!("CARGO_MANIFEST_DIR")
        );
        hex_to_bytes(&std::fs::read_to_string(path).expect("fixture")).expect("hex")
    }

    #[test]
    fn classifies_every_kind_of_message() {
        assert!(
            matches!(classify(&fixture("session1-gaze-frame")), Incoming::Gaze(f) if f.frame_counter == 43_780)
        );
        assert!(
            matches!(classify(&fixture("init-presence")), Incoming::Presence(p) if p.state == 2)
        );
        assert!(matches!(
            classify(&fixture("change-display-rsp-1440")),
            Incoming::Response {
                seq: 0x2a,
                status: 1,
                error: 0,
                ..
            }
        ));
        assert!(matches!(
            classify(&fixture("change-display-notify-1450")),
            Incoming::Notification(DeviceNotification::DisplayAreaChanged(_))
        ));
        assert_eq!(classify(&[1, 2, 3]), Incoming::Other);
    }

    #[test]
    fn init_responses_become_device_facts() {
        let responses = [
            fixture("init-rsp-1420"),
            fixture("init-rsp-1400"),
            fixture("init-rsp-1430"),
            fixture("init-rsp-2110"),
            fixture("init-rsp-1490"),
        ];
        let facts = DeviceFacts::from_messages(responses.iter().filter_map(|r| parse_message(r)));
        assert_eq!(facts.info.generation, "IS5");
        assert!(
            facts.track_box.is_some() && facts.display_area.is_some() && facts.mounting.is_some()
        );
        assert_eq!(facts.calibration_id, Some(1_904_654_973));
    }

    #[test]
    fn the_replay_drains_between_the_pieces_of_the_calibration_upload() {
        let packets = crate::calibration::embedded_packets().expect("embedded");
        let mut awaiting = None;
        let steps: Vec<AfterWrite> = (0..packets.len())
            .map(|i| after_write(&packets, i, &mut awaiting))
            .collect();

        // 38 single-write commands, each awaited by its own seq.
        for (i, step) in steps[..38].iter().enumerate() {
            assert_eq!(*step, AfterWrite::Await(u32::try_from(i + 1).expect("seq")));
        }
        // The calibration upload (seq 39) spans 162 writes. The device stops
        // taking them while its notification is unread, so every write but
        // the last drains; the response is awaited after the last.
        assert!(steps[38..199].iter().all(|s| *s == AfterWrite::Drain));
        assert_eq!(steps[199], AfterWrite::Await(39));
        assert_eq!(steps[200..], [AfterWrite::Await(40), AfterWrite::Await(41)]);
    }
}
