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
use tracing::{debug, error, info, info_span, warn};

use crate::engine::{
    CommandError, CommandResponse, DisplayGeneration, GazeSample, ImageSample, OpenNumber,
    PoseSample, PresenceSample, QueuedCommand, Sample, Shared,
};
use crate::time_map::{Stream, TimeMap};
use std::sync::mpsc::Receiver;
use tobii_ipc::geometry::{DisplayArea, DisplayFrame};
use tobii_ipc::host_clock_us;
use tobii_proto::facts::{
    DeviceFacts, DeviceNotification, STATUS_FAULTS, STATUS_WARNINGS, decode_notification,
    parse_display_area,
};
use tobii_proto::gaze83::{GazeFrame, PresenceFrame, decode_gaze_frame, decode_presence_frame};
use tobii_proto::image83::{ImageFrame, decode_image_payload, upscale2x_into};
use tobii_proto::log::{PacketLog, log_packet};
use tobii_proto::protocol::{
    BulkReassembler, InitPacket, MARKER_COMMAND, MARKER_NOTIFICATION, MARKER_RESPONSE,
    MARKER_STREAM, RESPONSE_STATUS_OK, STREAM_ID_GAZE, STREAM_ID_IMAGE, STREAM_ID_PRESENCE,
    chunk_command, cmd, declared_len, marker, notify, parse_message, seq, stream_id,
    stream_start_packet, stream_stop_packet, ttp_error,
};

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

/// Failed open+init attempts in a row after which the engine gives up on a
/// device that does not stream. It counts failures, not opens: the prime,
/// the first open since the engine started or a stream last armed that
/// accepts the init but does not stream (a cold-start quirk), is not one. An
/// open whose stream armed and then died within 30 s is one; a stream that
/// ran longer starts the count again. (That is the engine's count: `replay
/// --reconnect` spends it as plain opens, prime included.) A tracker that
/// may not be opened ([`OpenRefusal`]) ends the engine sooner.
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

/// A gaze stream that had armed sent no frame for [`GAZE_LIVENESS_TIMEOUT`]
/// while nothing excused the silence (see [`stream_stalled`]); after a resume,
/// not for the [`RESUME_GRACE`] either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GazeStalled;

impl fmt::Display for GazeStalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no gaze frame for over {GAZE_LIVENESS_TIMEOUT:?}")
    }
}

impl error::Error for GazeStalled {}

/// Context on a failure after the gaze stream armed (its first gaze frame
/// came): the stream ran for `ran`, then died, which is not the device
/// failing to start. [`after_arming`] adds it; [`OpenFailure::of`] reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamLost {
    /// Time from arming to the failure.
    ran: Duration,
}

impl fmt::Display for StreamLost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("gaze stream lost after arming")
    }
}

/// Why the tracker on the bus cannot be opened, when opening it again soon
/// would not help: the engine gives up on it sooner than on other failed
/// opens and without the USB reset, which can fix neither. As a context on
/// the engine's error, it says why the engine stopped (see
/// [`Engine::finish`](crate::engine::Engine::finish)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenRefusal {
    /// libusb may not open the device (`LIBUSB_ERROR_ACCESS`): the udev
    /// rule is missing or did not apply to this device, or, with the
    /// shipped `uaccess` rule, this user's session is not the active one
    /// on the seat (after a logout, say).
    NoPermission,
    /// Another process has claimed interface 0 (`LIBUSB_ERROR_BUSY`):
    /// another `tobiid`, or a research command.
    InUse,
}

impl OpenRefusal {
    /// The refusal `e` carries, if it is the error of the libusb call that
    /// opens the tracker or claims its interface in [`open_tobii`] (an
    /// [`OpenStep`] context) and says the tracker may not be opened: an
    /// `Access` or `Busy` there. Such an error elsewhere, or any other error
    /// of those calls (the tracker unplugged meanwhile, …), is none.
    fn of(e: &anyhow::Error) -> Option<Self> {
        e.downcast_ref::<OpenStep>()?;
        match e.downcast_ref::<rusb::Error>()? {
            rusb::Error::Access => Some(Self::NoPermission),
            rusb::Error::Busy => Some(Self::InUse),
            _ => None,
        }
    }

    /// How many more opens the engine tries, [`REOPEN_PAUSE`] apart, while
    /// the tracker keeps refusing, before it gives up (see [`FailedOpens`]).
    const fn retries(self) -> usize {
        match self {
            Self::NoPermission => NO_PERMISSION_RETRIES,
            Self::InUse => BUSY_RETRIES,
        }
    }
}

impl fmt::Display for OpenRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoPermission => {
                "no permission to open the tracker; check the udev rule (INSTALL §3) \
                 and that this user's session is the active one"
            }
            Self::InUse => "the tracker is in use by another process",
        })
    }
}

/// Context on the error of a libusb call in [`open_tobii`] whose failure
/// may say the tracker cannot be opened at all (see [`OpenRefusal::of`]).
/// It reads as the plain message it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenStep(&'static str);

impl fmt::Display for OpenStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// [`open_tobii`] opening the device.
const OPEN_DEVICE: OpenStep = OpenStep("failed to open Tobii device; try sudo");

/// [`open_tobii`] claiming interface 0.
const CLAIM_INTERFACE: OpenStep = OpenStep("failed to claim interface 0");

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
    let h = dev.open().context(OPEN_DEVICE)?;

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

    h.claim_interface(IFACE).context(CLAIM_INTERFACE)?;

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
/// The engine resets only as an escalation once opens keep failing (see
/// [`RESET_AFTER_FAILURES`]), and not at all with `TOBII_NO_RESET=1` (see
/// [`run_gaze_engine`]). It logs what came of the reset, a skip when the
/// tracker cannot be opened included: the escalation only says it tries one.
///
/// Whether the reset helps is unconfirmed on hardware: the logs show the
/// open after a reset finding the tracker starting its sensor (notification
/// 1271, about 0.3 s after control OUT 65), so that a piece of the
/// calibration upload is refused for about 3 s, and until the init waited
/// that out (see [`WRITE_DEADLINE`]) every such open failed on its 2 s write
/// timeout.
fn reset_device_baseline(ctx: &UsbContext) {
    match open_tobii(ctx) {
        Ok(h) => {
            match h.reset() {
                Ok(()) => info!("reset Tobii to baseline before init"),
                Err(e) => warn!(error = ?e, "best-effort reset failed"),
            }
            drop(h); // device may re-enumerate; reopen fresh below
        }
        // Unplugged, no access (udev rule), or interface 0 held elsewhere.
        Err(e) => warn!(
            error = format_args!("{e:#}"),
            "USB reset skipped: cannot open the tracker"
        ),
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

/// Where a device sits on the USB bus. The kernel gives a device the next
/// free address on its bus each time it enumerates (a USB reset normally
/// keeps the one it has), so the tracker at another address than before was
/// unplugged and plugged back in meanwhile, or re-enumerated, however
/// briefly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BusAddress {
    /// The bus number.
    pub bus: u8,
    /// The device's address on that bus.
    pub address: u8,
}

/// Is the Tobii plugged in right now? (Cheap check for the daemon watchdog.)
#[must_use]
pub fn is_device_present() -> bool {
    find_device().is_some()
}

/// Where the Tobii is on the bus right now, if it is plugged in (without
/// opening or claiming it; a cheap check for the daemon watchdog).
#[must_use]
pub fn find_device() -> Option<BusAddress> {
    UsbContext::new().ok().and_then(|ctx| tobii_address(&ctx))
}

/// Is the Tobii on the bus right now (without opening/claiming it)?
fn is_tobii_present(ctx: &UsbContext) -> bool {
    tobii_address(ctx).is_some()
}

/// Where the Tobii is on the bus right now, if it is on it (without
/// opening or claiming it).
fn tobii_address(ctx: &UsbContext) -> Option<BusAddress> {
    ctx.devices()
        .ok()?
        .iter()
        .find(|d| {
            d.device_descriptor()
                .is_ok_and(|x| x.vendor_id() == VID && x.product_id() == PID)
        })
        .map(|d| BusAddress {
            bus: d.bus_number(),
            address: d.address(),
        })
}

/// A cold gaze start normally arms on the *second* fresh open (the first accepts
/// the init but doesn't stream — a firmware quirk, not stale state), and so does
/// a re-open after the stream died; that first open is the prime, which is not
/// a failure (see [`FailedOpens`]). A USB reset up front doesn't change that
/// (verified: start behaves identically with and without it), so we no longer
/// pay its re-enumeration cost on every start. It's kept only as a recovery
/// escalation once opens keep failing — this many failures in a row: opens
/// whose init fails, that do not arm past the prime, or that lose the stream
/// within [`HEALTHY_STREAM`] — which is the signature of a device left hot by
/// an unclean exit (`kill -9`, crash) that skipped the teardown. A stream that
/// ran for [`HEALTHY_STREAM`] ends such a run, so the reset comes at most once
/// per run of failures, not once per engine. `TOBII_NO_RESET=1` skips even
/// the escalation (see [`run_gaze_engine`]).
const RESET_AFTER_FAILURES: usize = 2;

/// How long a gaze stream must run after arming for its loss to count as a
/// stream that died rather than an open that failed (see [`FailedOpens`]).
///
/// A device that arms and dies at once fails well inside it, so it is still
/// USB-reset and given up on: its silence is caught after the
/// [`GAZE_LIVENESS_TIMEOUT`] plus a command's write (a piece the device
/// refuses fails within [`WRITE_DEADLINE`] and a try) and deadline, which the
/// liveness check waits out (the daemon's longest deadline is 10 s), and
/// after a resume only once the [`RESUME_GRACE`] has passed too. A healthy
/// tracker streams for hours between stalls. Paused time and the silent tail
/// before the liveness check fires count toward the run on purpose: a resume
/// the tracker did not answer after a long pause is the pause's normal
/// recovery, not a failed open.
const HEALTHY_STREAM: Duration = Duration::from_secs(30);

// A stream that arms and dies at once must fail inside `HEALTHY_STREAM`, or
// the reset escalation and the give-up never come for it.
const _: () = assert!(
    GAZE_LIVENESS_TIMEOUT.as_millis()
        + RESUME_GRACE.as_millis()
        + WRITE_DEADLINE.as_millis()
        + WRITE_TRY.as_millis()
        < HEALTHY_STREAM.as_millis(),
    "HEALTHY_STREAM must outlast the liveness timeout, the resume grace and a refused write"
);

/// How long the engine waits after a failed open before the next.
const REOPEN_PAUSE: Duration = Duration::from_millis(700);

/// How many times the engine opens the tracker again, [`REOPEN_PAUSE`]
/// apart, while it may not open it ([`OpenRefusal::NoPermission`]), before
/// it gives up. A tracker just plugged in (or re-enumerated) refuses for a
/// moment even with the udev rule in place: its device node starts out
/// root's, until udev applies the rule, while libusb already lists it. One
/// more open covers that; a missing rule costs one [`REOPEN_PAUSE`] more.
/// No USB reset comes in between: it would open the tracker first, which
/// fails the same way, and it cannot fix a permission.
const NO_PERMISSION_RETRIES: usize = 1;

/// How many times the engine opens the tracker again, [`REOPEN_PAUSE`]
/// apart, while another process holds its interface
/// ([`OpenRefusal::InUse`]), before it gives up: about 1.4 s. That covers a
/// `tobiid` handing the tracker over, one stopping as another starts; a
/// holder that keeps it (a research command, a second daemon) is left to
/// the owner of the engine, whose restarts back off (tobiid's watchdog).
/// No USB reset comes in between, as for [`NO_PERMISSION_RETRIES`].
const BUSY_RETRIES: usize = 2;

/// What the engine does after an open failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterFailedOpen {
    /// Open the device again.
    Reopen,
    /// Open the device again: the open was the prime, which is not counted.
    ReopenPrimed,
    /// USB-reset the device, then open it again.
    ResetAndReopen,
    /// Give up: the engine stops.
    GiveUp,
}

/// How an open failed, as [`FailedOpens`] counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenFailure {
    /// The device accepted the init but the gaze stream never armed
    /// ([`StreamStartupTimeout`]).
    NotArmed,
    /// The gaze stream armed, ran this long, then died (a [`StreamLost`]
    /// context).
    Lost(Duration),
    /// The tracker is on the bus but may not be opened (see
    /// [`OpenRefusal`]), which a reset cannot fix.
    Refused(OpenRefusal),
    /// Anything else: no device, or an init that failed (a write timeout, …).
    Failed,
}

impl OpenFailure {
    /// Classify the error an open failed with. The [`StreamLost`] tag is
    /// checked first, defensively: a failure after arming is a lost stream
    /// whatever its cause. No tagged chain holds a [`StreamStartupTimeout`]
    /// today, which fails an open only before it arms, nor an
    /// [`OpenRefusal`], which fails it before the init.
    fn of(e: &anyhow::Error) -> Self {
        if let Some(lost) = e.downcast_ref::<StreamLost>() {
            Self::Lost(lost.ran)
        } else if e.downcast_ref::<StreamStartupTimeout>().is_some() {
            Self::NotArmed
        } else if let Some(refusal) = OpenRefusal::of(e) {
            Self::Refused(refusal)
        } else {
            Self::Failed
        }
    }
}

/// Failures counted in a row: opens whose init failed, that did not arm, or
/// that lost the stream within [`HEALTHY_STREAM`]. The first open that does
/// not arm since the engine started or a stream last armed is the prime and
/// is not counted: the ET5 needs one such open to arm, cold and after its
/// stream died alike (a firmware quirk). An init failure is counted and
/// leaves the prime due. The device is USB-reset once
/// [`RESET_AFTER_FAILURES`] failures are counted and given up on at
/// [`MAX_REPLAY_ATTEMPTS`]. An open whose gaze stream ran for
/// [`HEALTHY_STREAM`] ends the run, and its own loss starts none: the re-open
/// that follows gets the whole budget, prime included, as a new engine would.
///
/// An open the tracker refuses ([`OpenFailure::Refused`]) is none of those
/// failures and leaves their count and the prime alone: the engine gives up
/// once the tracker has refused [`OpenRefusal::retries`] more opens in a row
/// (either refusal counts toward the streak; the last one says how long it
/// may be), without permission after [`NO_PERMISSION_RETRIES`], on an
/// interface another process holds after [`BUSY_RETRIES`]. None gets a
/// reset.
#[derive(Debug, Default)]
struct FailedOpens {
    /// Failures counted since the engine started or a stream last ran.
    in_a_row: usize,
    /// An open accepted the init but did not arm (the prime) since the engine
    /// started or a stream last armed: the next such open is counted.
    prime_spent: bool,
    /// Opens in a row the tracker refused.
    refused_in_a_row: usize,
}

impl FailedOpens {
    /// Count an open that failed with `failure`, and say what comes next.
    #[must_use]
    fn record(&mut self, failure: OpenFailure) -> AfterFailedOpen {
        let refused_before = std::mem::take(&mut self.refused_in_a_row);
        match failure {
            OpenFailure::Refused(refusal) => {
                self.refused_in_a_row = refused_before + 1;
                return if self.refused_in_a_row > refusal.retries() {
                    AfterFailedOpen::GiveUp
                } else {
                    AfterFailedOpen::Reopen
                };
            }
            OpenFailure::NotArmed if !self.prime_spent => {
                self.prime_spent = true;
                return AfterFailedOpen::ReopenPrimed;
            }
            OpenFailure::Lost(ran) => {
                // It armed, so the next open that does not is a prime again.
                self.prime_spent = false;
                if ran >= HEALTHY_STREAM {
                    self.in_a_row = 0;
                    return AfterFailedOpen::Reopen;
                }
            }
            OpenFailure::NotArmed | OpenFailure::Failed => {}
        }
        self.in_a_row += 1;
        if self.in_a_row >= MAX_REPLAY_ATTEMPTS {
            AfterFailedOpen::GiveUp
        } else if self.in_a_row == RESET_AFTER_FAILURES {
            AfterFailedOpen::ResetAndReopen
        } else {
            AfterFailedOpen::Reopen
        }
    }

    /// Failures counted in the current run.
    fn in_a_row(&self) -> usize {
        self.in_a_row
    }

    /// Opens in a row the tracker refused.
    fn refused_in_a_row(&self) -> usize {
        self.refused_in_a_row
    }
}

/// Own the device for the daemon engine and stream samples into `tx` until
/// `shared.stop` is set, running the commands that arrive on `commands` in
/// between. A stream that dies (a stall, a USB error) is re-opened in place,
/// and only failures in a row count toward giving up, the prime aside (see
/// [`run_opens`] and the note on the reset-escalation constant above). Set
/// `TOBII_NO_RESET=1` to leave out the USB reset of that escalation.
///
/// # Errors
///
/// Returns the last open's error once [`MAX_REPLAY_ATTEMPTS`] failures in a
/// row are counted, or at once when libusb cannot be initialised. A tracker
/// that may not be opened ends it sooner, with an [`OpenRefusal`] context
/// saying why (see [`run_opens`]). A stop is not a failure, even one that
/// cuts an open short.
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

    let reset = (!is_env_flag_set("TOBII_NO_RESET")).then_some(|| reset_device_baseline(&ctx));
    let result = run_opens(
        stop,
        |opens| gaze_engine_attempt(&ctx, shared, commands, tx, &mailbox, opens),
        reset,
        REOPEN_PAUSE,
    );

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

/// The engine's open loop: call `open` until an open returns `Ok` (a clean
/// stop) or `stop` is set, waiting `pause` after each failed open. Every
/// error but a refusal to open (below) is retried. [`FailedOpens`] counts
/// the failures in a row by their [`OpenFailure`], which decides when
/// `reset` runs (once [`RESET_AFTER_FAILURES`] are counted) and when to give
/// up; the prime is not counted, and a failure whose [`StreamLost`] context
/// says the stream ran for [`HEALTHY_STREAM`] ends the run instead of adding
/// to it. With no `reset` (`TOBII_NO_RESET=1`) the escalation says it
/// skipped the reset and the opens go on as they would after one.
///
/// A tracker that may not be opened ([`OpenRefusal`]) is not retried that
/// way, since neither many more opens nor a reset fix it: the loop gives up
/// after a few more opens in a row that it refuses, without permission
/// after [`NO_PERMISSION_RETRIES`], on an interface another process holds
/// after [`BUSY_RETRIES`], with no reset in between.
///
/// A stop is not a failure: an open that fails once `stop` is set, as one
/// whose arming wait the stop cut short does, ends the loop `Ok` and its
/// error is logged at debug.
///
/// Every open gets the one [`Opens`] the loop keeps, as the opens before it
/// left it, whatever they failed with and across a reset: the count of
/// opens, the display area in effect and the check of the device's display
/// frame go on from one open to the next.
///
/// # Errors
///
/// Returns the last open's error once [`MAX_REPLAY_ATTEMPTS`] failures in a
/// row are counted, or once the tracker refused to be opened as above, then
/// with the [`OpenRefusal`] as its context.
fn run_opens(
    stop: &AtomicBool,
    mut open: impl FnMut(&mut Opens) -> Result<()>,
    mut reset: Option<impl FnMut()>,
    pause: Duration,
) -> Result<()> {
    let mut failed = FailedOpens::default();
    let mut opens = Opens::default();
    let mut reset_first = false;
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        // A normal cold start or re-open arms on the open after its prime and
        // never gets here; opens that keep failing (the init fails, they do
        // not arm past the prime, or lose the stream within `HEALTHY_STREAM`)
        // get a USB reset to recover, unless it is turned off.
        if reset_first {
            let in_a_row = failed.in_a_row();
            match reset.as_mut() {
                Some(reset) => {
                    warn!(in_a_row, "gaze: opens keep failing; trying a USB reset");
                    reset();
                }
                None => warn!(
                    in_a_row,
                    "gaze: opens keep failing; USB reset skipped (TOBII_NO_RESET=1)"
                ),
            }
        }
        let Err(e) = open(&mut opens) else {
            return Ok(());
        };
        if stop.load(Ordering::Relaxed) {
            debug!(
                error = format_args!("{e:#}"),
                "gaze: open ended by the stop"
            );
            return Ok(());
        }
        let failure = OpenFailure::of(&e);
        let after = failed.record(failure);
        reset_first = match after {
            AfterFailedOpen::GiveUp => {
                return Err(match failure {
                    OpenFailure::Refused(refusal) => e.context(refusal),
                    _ => e,
                });
            }
            AfterFailedOpen::Reopen | AfterFailedOpen::ReopenPrimed => false,
            AfterFailedOpen::ResetAndReopen => true,
        };
        let in_a_row = failed.in_a_row();
        match failure {
            OpenFailure::Refused(refusal) => warn!(
                reason = %refusal,
                refused_in_a_row = failed.refused_in_a_row(),
                error = format_args!("{e:#}"),
                "gaze: the tracker refused to open; re-opening"
            ),
            OpenFailure::Lost(ran) => warn!(
                ?ran,
                in_a_row,
                error = format_args!("{e:#}"),
                "gaze: stream lost; re-opening the device"
            ),
            // A cold device, or one whose stream just died, accepts the first
            // init but doesn't start streaming; a fresh re-open is what arms
            // it (expected, not an error). Report real failures loudly, the
            // prime quietly.
            OpenFailure::NotArmed if after == AfterFailedOpen::ReopenPrimed => info!(
                in_a_row,
                "gaze: stream not armed; re-opening to prime (cold-start quirk)"
            ),
            // An open past the prime that did not arm is a real failure too,
            // though its init did not fail.
            OpenFailure::NotArmed => error!(
                in_a_row,
                error = format_args!("{e:#}"),
                "gaze: stream not armed past the prime; re-opening"
            ),
            OpenFailure::Failed => error!(
                in_a_row,
                error = format_args!("{e:#}"),
                "tobii gaze init failed; re-opening"
            ),
        }
        thread::sleep(pause);
    }
}

/// Single-slot hand-off of the newest image frame to the pose worker, with
/// its host time, which the pose it makes carries (the pose may come after a
/// re-open, whose time map knows nothing of the image).
type PoseMailbox = Arc<(Mutex<Option<ImageSample>>, Condvar)>;

/// Replace the mailbox slot with `image` (dropping any frame the worker
/// has not taken yet) and wake the worker.
///
/// The slot only ever holds an `Option` that is written whole, so a poisoned
/// lock (a panicking worker) leaves nothing half-updated and the guard is
/// reused rather than propagating the panic to the USB reader.
fn mailbox_put(mailbox: &PoseMailbox, image: ImageSample) {
    let (lock, cv) = &**mailbox;
    *lock.lock().unwrap_or_else(PoisonError::into_inner) = Some(image);
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
        let image = {
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
        let Some(ImageSample { frame, host_us, .. }) = image else {
            continue;
        };
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
                let _ = tx.send(Sample::Pose(PoseSample::new(
                    to_i64_us(frame.device_ts_us),
                    host_us,
                    [p[0], p[1], p[2]],
                    [p[3], p[4], p[5]],
                )));
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
/// failure. Fails with [`StreamStartupTimeout`] when no gaze frame arrives
/// within 4.5 s of the init, and with a [`StreamLost`] context on any failure
/// after one did. `opens` is what the opens before this one left.
fn gaze_engine_attempt(
    ctx: &UsbContext,
    shared: &Shared,
    commands: &Receiver<QueuedCommand>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
    opens: &mut Opens,
) -> Result<()> {
    let mut h = open_tobii(ctx)?;
    vendor_control_init(&mut h)?;
    // From here the 0x83 stream is running firmware-side (started by request 65
    // inside vendor_control_init). Guarantee the Windows-style stop (request 66)
    // on every exit — clean stop, error, or timeout — so the device isn't left
    // mid-stream and the next open starts from a defined state.
    let result = gaze_stream_loop(&mut h, shared, commands, tx, mailbox, opens);
    vendor_control_deinit(&mut h);
    result
}

/// How long one try of a bulk OUT write waits for the device during the init
/// replay. A piece it refuses for longer is tried again after the caller's
/// step in between, which reads what the device sent meanwhile (see
/// [`write_piece`]). Nothing reads EP 0x83 while a try waits, so a refusal
/// because a message is unread costs a whole try; during the init only the
/// rare notification comes, and 500 ms still reads EP 0x83 twice a second
/// while a 3 s refusal costs only a handful of tries.
const WRITE_TRY: Duration = Duration::from_millis(500);

/// [`WRITE_TRY`] while the streams run (the pump's commands, the stream
/// start and stop). The gaze and image streams each send a message every
/// 30 ms, so a try spans under two frame periods and the step between
/// ([`PIECE_DRAIN_READS`] reads) takes all that came meanwhile: a refused
/// write holds the streams up by at most a try, and a refusal because a
/// message is unread costs only that.
const STREAMING_WRITE_TRY: Duration = Duration::from_millis(50);

/// How long the device may refuse a piece before its write fails; each piece
/// of a command gets its own. A try under way when it passes runs out first,
/// so a refused piece fails within the deadline plus a try. A tracker
/// starting its sensor (notification 1271) answers no command until
/// 3.60-3.66 s after it (22 starts in the logs), and takes no data on EP 0x05
/// from about 0.6 s after it until then. A cold start's init meets that window
/// inside the calibration upload, where one piece waits about 1.5 s; so does
/// the open after a USB reset, where one piece waits about 3 s. The single
/// 2 s write timeout used before cleared the first by only half a second and
/// failed every open after a reset (3 of 3 in the logs).
const WRITE_DEADLINE: Duration = Duration::from_secs(6);

/// A piece that waited this long for the device is logged (at debug).
const SLOW_WRITE: Duration = Duration::from_millis(100);

/// How [`write_piece`] paces a piece: each try waits up to `per_try`, and
/// the write fails once `deadline` has passed since it began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriteLimits {
    per_try: Duration,
    deadline: Duration,
}

impl WriteLimits {
    /// Whether the tries fit the deadline: libusb waits forever on a zero
    /// timeout, and a try as long as the deadline leaves nothing to retry.
    const fn is_paced(self) -> bool {
        self.per_try.as_millis() > 0 && self.per_try.as_millis() < self.deadline.as_millis()
    }
}

/// The limits of the init replay's writes.
const INIT_WRITE: WriteLimits = WriteLimits {
    per_try: WRITE_TRY,
    deadline: WRITE_DEADLINE,
};

/// The limits of a command written while the streams run.
const COMMAND_WRITE: WriteLimits = WriteLimits {
    per_try: STREAMING_WRITE_TRY,
    deadline: WRITE_DEADLINE,
};

/// The limits of the stream stop sent on the way out of an open. It is
/// best-effort (control OUT 66 then stops the streams anyway), so a tracker
/// that refuses it is not waited out, only read in case a message it sent
/// is what holds the write up.
const STREAM_STOP_WRITE: WriteLimits = WriteLimits {
    per_try: STREAMING_WRITE_TRY,
    deadline: Duration::from_millis(500),
};

const _: () = assert!(
    INIT_WRITE.is_paced() && COMMAND_WRITE.is_paced() && STREAM_STOP_WRITE.is_paced(),
    "every write's tries must be nonzero and shorter than its deadline"
);

/// Why [`write_piece`] did not write the whole piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteError {
    /// A try failed: [`rusb::Error::Timeout`] once the device still refused
    /// the rest of the piece at the deadline, any other error at once.
    Usb(rusb::Error),
    /// `stop` was set while the device refused all of the piece.
    Stopped,
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usb(e) => e.fmt(f),
            Self::Stopped => f.write_str("write stopped"),
        }
    }
}

impl error::Error for WriteError {}

/// Write one piece (`data`, at most 4 KB) in tries of `limits.per_try`,
/// until the device has taken all of it. After each try that leaves part of
/// it unwritten `between` runs; then, while the device has taken none of the
/// piece, the write ends if `stop` is set, and it fails once
/// `limits.deadline` has passed since the piece began; otherwise the next
/// try resumes after the bytes the device took. Returns how long the piece
/// took; one that waited [`SLOW_WRITE`] or more is logged at debug. The
/// callers enter a span saying which piece of what is written.
///
/// The ET5 refuses a piece while it starts its sensor (see
/// [`WRITE_DEADLINE`]) and while a message it sent is unread, so the callers
/// read EP 0x83 in `between`: the streams are read between the tries of a
/// long refusal, and a device waiting to be read is read.
///
/// A try that times out part-way is `Ok(n)`, `n` short of the rest (rusb
/// returns `Err(Timeout)` only when nothing moved). A high-speed bulk OUT
/// counts only the 512-byte packets the device acknowledged, so the next try
/// starts with the packet it refused. That has never been seen (the device
/// refuses whole pieces) and is logged at warn. A stop never cuts short a
/// piece the device has begun to take.
///
/// `write` is one try: it writes what it is given to the device `io`,
/// waiting up to the duration it is given. `between` gets the same `io`.
///
/// # Errors
///
/// [`WriteError::Usb`] with the error of a try that failed other than by
/// timing out, at once, or with [`rusb::Error::Timeout`] once the deadline
/// has passed. [`WriteError::Stopped`] when `stop` is set after a try while
/// the device has taken none of the piece.
fn write_piece<T>(
    io: &mut T,
    data: &[u8],
    limits: WriteLimits,
    stop: Option<&AtomicBool>,
    mut write: impl FnMut(&mut T, &[u8], Duration) -> Result<usize, rusb::Error>,
    mut between: impl FnMut(&mut T),
) -> Result<Duration, WriteError> {
    let start = Instant::now();
    let mut sent = 0;
    loop {
        let rest = &data[sent..];
        match write(io, rest, limits.per_try) {
            Ok(n) => {
                // Never more than the rest, even from a device that says so.
                sent += n.min(rest.len());
                if n > 0 && sent < data.len() {
                    warn!(
                        took = n,
                        sent,
                        len = data.len(),
                        "write: the device took part of a piece; writing the rest"
                    );
                }
            }
            Err(rusb::Error::Timeout) => {}
            Err(e) => return Err(WriteError::Usb(e)),
        }
        if sent == data.len() {
            let waited = start.elapsed();
            if waited >= SLOW_WRITE {
                debug!(
                    waited_ms = waited.as_millis(),
                    "write: the device made a piece wait"
                );
            }
            return Ok(waited);
        }
        between(io);
        // Relaxed: a pure signal.
        if sent == 0 && stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
            return Err(WriteError::Stopped);
        }
        if start.elapsed() >= limits.deadline {
            return Err(WriteError::Usb(rusb::Error::Timeout));
        }
    }
}

/// The error a packet's unfinished write fails the init replay with: none
/// for a stop, which ends the replay without error (a stop is not a
/// failure).
fn init_write_error(e: WriteError) -> Option<rusb::Error> {
    match e {
        WriteError::Usb(e) => Some(e),
        WriteError::Stopped => None,
    }
}

/// What a command's unfinished write answers its client: a piece the device
/// refused past the deadline is a command it did not take in time, a stop is
/// the engine going away.
impl From<WriteError> for CommandError {
    fn from(e: WriteError) -> Self {
        match e {
            WriteError::Usb(rusb::Error::Timeout) => Self::Timeout,
            WriteError::Usb(e) => Self::Usb(e.to_string()),
            WriteError::Stopped => Self::EngineGone,
        }
    }
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
/// chunked command, or the tries of a piece the device refuses: only what
/// the device already sent is wanted.
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
/// stops accepting further pieces until the host has read it. A packet the
/// device refuses is retried, draining between the tries, for up to
/// `WRITE_DEADLINE` per packet (see `write_piece`): a tracker starting its
/// sensor refuses one inside the calibration upload for up to about 3 s.
/// Returns early without error once `stop` is set, also while the device
/// refuses a packet; a failed write ends the replay with an error.
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
        let command = command_seq(&pkt.data).or(awaiting);
        let written = info_span!("init_packet", packet = i, seq = ?command).in_scope(|| {
            write_piece(
                h,
                &pkt.data,
                INIT_WRITE,
                stop,
                |h, rest, timeout| h.write_bulk(pkt.ep, rest, timeout),
                |h| reader.drain(h, &mut capture.side),
            )
        });
        if let Err(e) = written {
            let Some(e) = init_write_error(e) else {
                return Ok(capture);
            };
            return Err(e).with_context(|| {
                format!(
                    "init packet {i} write failed (command seq {command:?}, {} ms in)",
                    started.elapsed().as_millis()
                )
            });
        }
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

/// Where the display area an init leaves in effect came from (see
/// [`init_display_area`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AreaSource {
    /// The last display-area notification (1450) the replay read.
    Notified,
    /// The replay's display-area write (1440), which the device took.
    Written,
    /// The device's answer to the replay's display-area read (1430).
    Read,
}

impl AreaSource {
    /// The id of the message the area came in, for the log.
    const fn message_id(self) -> u32 {
        match self {
            Self::Notified => notify::DISPLAY_AREA,
            Self::Written => cmd::DISPLAY_AREA_SET,
            Self::Read => cmd::DISPLAY_AREA_GET,
        }
    }
}

/// The display area an init leaves in effect on the device, the display id
/// that goes with it, and where the two came from.
#[derive(Debug, Clone, Copy, PartialEq)]
struct InitArea {
    area: DisplayArea,
    display_id: Option<u32>,
    source: AreaSource,
}

/// The area in `msg`, if it is a `marker` message `id` that carries one.
fn display_area_in(msg: &[u8], marker: u32, id: u32) -> Option<(DisplayArea, Option<u32>)> {
    let m = parse_message(msg)?;
    if m.marker != marker || m.id != id {
        return None;
    }
    parse_display_area(&m)
}

/// The display area the init replay of `packets` left in effect, as the
/// device confirmed it in `capture`, the first of:
///
/// 1. the last display-area notification (1450) the replay read besides
///    the responses: the device sends one when an area it takes differs
///    from the one it holds;
/// 2. the area the replay wrote (its 1440, as the wire carries it), if the
///    device answered the write OK and with no error, as tobiid judges an
///    answer: an area the device holds already brings no 1450 (the Windows
///    inits), and the write stands in for a 1450 the side messages had no
///    room for;
/// 3. the device's answer to the replay's display-area read (1430).
///
/// The replay reads the area (seq 11) before it writes one (seq 14), so the
/// read answers what the device held before the init wrote: on Linux, the
/// 4 x 4 mm square around its origin it starts every open with (all 20
/// recorded logs, each followed by a 1450). The display id comes from the
/// same message; a 1450 carries none, and the next source down gives it.
fn init_display_area(packets: &[InitPacket], capture: &InitCapture) -> Option<InitArea> {
    let notified = capture
        .side
        .iter()
        .rev()
        .find_map(|m| display_area_in(m, MARKER_NOTIFICATION, notify::DISPLAY_AREA));
    let written = crate::calibration::written_display_area(packets).and_then(|(seq, area, id)| {
        capture
            .responses
            .iter()
            .filter_map(|r| parse_message(r))
            .any(|m| {
                m.marker == MARKER_RESPONSE
                    && m.id == cmd::DISPLAY_AREA_SET
                    && m.seq == seq
                    && m.status == RESPONSE_STATUS_OK
                    && m.error == ttp_error::NONE
            })
            .then_some((area, id))
    });
    let read = capture
        .responses
        .iter()
        .rev()
        .find_map(|m| display_area_in(m, MARKER_RESPONSE, cmd::DISPLAY_AREA_GET));
    let mut confirmed = [
        notified.map(|found| (found, AreaSource::Notified)),
        written.map(|found| (found, AreaSource::Written)),
        read.map(|found| (found, AreaSource::Read)),
    ]
    .into_iter()
    .flatten();
    let ((area, display_id), source) = confirmed.next()?;
    Some(InitArea {
        area,
        display_id: display_id.or_else(|| confirmed.find_map(|((_, id), _)| id)),
        source,
    })
}

/// What the init replay of `packets` says of the device in `capture`: the
/// facts its responses give, but for the display area and display id, which
/// are the ones the device confirmed ([`init_display_area`], also returned):
/// not the 1430 answer, which predates the init's write, nor the area the
/// init meant to write, which the device may have refused.
fn init_facts(packets: &[InitPacket], capture: &InitCapture) -> (DeviceFacts, Option<InitArea>) {
    let mut facts =
        DeviceFacts::from_messages(capture.responses.iter().filter_map(|r| parse_message(r)));
    let confirmed = init_display_area(packets, capture);
    facts.display_area = confirmed.map(|c| c.area);
    facts.display_id = confirmed.and_then(|c| c.display_id);
    (facts, confirmed)
}

/// The display area in effect on the device, as the device confirmed it:
/// the area an init left in effect (see [`init_display_area`]), or the one
/// a display-area notification (1450) gave since (see [`Pump::route`]).
/// Never the area handed to
/// [`Engine::set_display_area_override`](crate::engine::Engine::set_display_area_override),
/// which is what the next inits write and may run ahead of the device:
/// tobiid hands it the `TOBII_DISPLAY_MM` area before its own write of that
/// area, which the device may refuse, and `None` when a calibration session
/// puts back a configuration that had no area.
#[derive(Debug, Default, PartialEq)]
struct AreaInEffect {
    /// The area; `None` while the device has confirmed none.
    area: Option<DisplayArea>,
    /// The display frame the area fixes, which every image read while it
    /// is in effect shares; `None` without an area, and for one that fixes
    /// none.
    frame: Option<Arc<DisplayFrame>>,
    /// How many times the area changed since the engine started: 0 until
    /// the device first confirms one.
    generation: DisplayGeneration,
}

impl AreaInEffect {
    /// Take `area` as the area in effect. The generation moves on, and the
    /// frame is built anew, only if it differs from the one before, as the
    /// device sends a 1450 only for an area that changes the one it holds.
    ///
    /// An area that fixes no display frame (see [`DisplayFrame::new`]) is
    /// warned of as it is taken, so once for each generation: the images
    /// read while it is in effect carry no display frame, and the device's
    /// gaze origins are not checked. tobiid sets and loads such areas, a
    /// sheared one among them, and the device may take them.
    fn set(&mut self, area: Option<DisplayArea>) {
        if area == self.area {
            return;
        }
        let frame = area.as_ref().and_then(DisplayFrame::new).map(Arc::new);
        let generation = self.generation.next();
        if let (Some(area), None) = (&area, &frame) {
            warn!(
                generation = generation.get(),
                ?area,
                "gaze: the display area in effect fixes no display frame; \
                 images carry none and gaze origins go unchecked"
            );
        }
        *self = Self {
            area,
            frame,
            generation,
        };
    }
}

/// What the engine's USB thread keeps from one open of the tracker to the
/// next: how many it made, the display area in effect, and the check of the
/// device's display frame against that area's.
#[derive(Debug, Default)]
struct Opens {
    /// Opens whose init replay ran to its end, the one under way included:
    /// its number, from 1.
    count: OpenNumber,
    /// The display area in effect on the device.
    display: AreaInEffect,
    /// Whether the device converts its gaze origins as the display frame of
    /// that area does.
    check: DisplayFrameCheck,
}

impl Opens {
    /// Another open's init replay ran to its end and left `area` in effect.
    fn opened(&mut self, area: Option<DisplayArea>) {
        self.count = self.count.next();
        self.display.set(area);
    }

    /// Another open's init replay of `packets` ran to its end, reading
    /// `capture`: count the open, leave in effect the display area the
    /// device confirmed, and say what the init reports of the device and
    /// where that area came from (see [`init_facts`]).
    #[must_use]
    fn init_done(
        &mut self,
        packets: &[InitPacket],
        capture: &InitCapture,
    ) -> (DeviceFacts, Option<InitArea>) {
        let (facts, confirmed) = init_facts(packets, capture);
        self.opened(confirmed.map(|c| c.area));
        (facts, confirmed)
    }
}

/// How far, mm, the device's display-frame gaze origin may lie from where
/// the display frame in effect puts its tracker-frame one for the two
/// frames to be the same (see [`DisplayFrameCheck`]). On every gaze frame
/// recorded so far the two lie within 0.00016 mm with the area the device
/// held, and 0.847 mm apart with the other area recorded.
const ORIGIN_TOLERANCE_MM: f64 = 0.01;

/// How many gaze frames in a row must miss the display frame in effect by
/// more than [`ORIGIN_TOLERANCE_MM`] to say that the device's display frame
/// is not ours. A frame either side of a change of area may miss alone: one
/// the device converted with the new area before the pump routed its 1450,
/// or with the old area after. (In the one change recorded, change-display,
/// none did: the 1450 came between the last frame of the old area and the
/// first of the new.)
const MISSES_TO_WARN: u32 = 3;

/// The check that the device converts its gaze origins into the display
/// frame of the area in effect, as [`DisplayFrame`] builds it.
///
/// A 0x500 frame carries each eye's gaze origin (cornea centre) in the
/// tracker frame (keys 0x02/0x08) and in the display frame of the area the
/// device holds (0x22/0x24). An eye that has both, each valid, is judged by
/// how far the second lies from where the display frame in effect puts the
/// first, and a frame by its eye that lies further. A frame that misses by
/// more than [`ORIGIN_TOLERANCE_MM`] says the device's display frame is not
/// ours: the device holds an area the engine was not told of, or it builds
/// its frame from the area otherwise (every area seen so far was a
/// rectangle tilted 20 degrees about x; how it frames a rolled or sheared
/// one is not known). Anything the engine reports in its own display frame
/// is then off from the device's gaze origins by about as much.
///
/// The check counts the frames in a row, of one open and one display
/// generation, that miss; a frame with no eye to judge neither counts nor
/// breaks the run. The [`MISSES_TO_WARN`]th is to be warned of, once for
/// each display generation.
#[derive(Debug, Default, PartialEq)]
struct DisplayFrameCheck {
    /// The open and the display generation of the frames counted in
    /// `misses`.
    run_of: (OpenNumber, DisplayGeneration),
    /// Frames in a row, of that open and generation, whose origins missed.
    misses: u32,
    /// The last display generation that was to be warned of.
    warned: Option<DisplayGeneration>,
}

impl DisplayFrameCheck {
    /// Judge `frame`, read in open `open` while the display area of
    /// generation `generation`, whose display frame is `display`, was in
    /// effect. The error, mm, when this frame is the [`MISSES_TO_WARN`]th in
    /// a row to miss and the generation is not warned of yet; `None`
    /// otherwise.
    #[must_use]
    fn judge(
        &mut self,
        frame: &GazeFrame,
        display: DisplayFrame,
        open: OpenNumber,
        generation: DisplayGeneration,
    ) -> Option<f64> {
        let error_mm = origin_error_mm(frame, display)?;
        if self.run_of != (open, generation) {
            self.run_of = (open, generation);
            self.misses = 0;
        }
        if error_mm > ORIGIN_TOLERANCE_MM {
            self.misses = self.misses.saturating_add(1);
        } else {
            self.misses = 0;
        }
        if self.misses < MISSES_TO_WARN || self.warned == Some(generation) {
            return None;
        }
        self.warned = Some(generation);
        Some(error_mm)
    }
}

/// How far the device's display-frame gaze origins in `frame` lie from
/// where `display` puts its tracker-frame ones, mm: the larger distance of
/// the eyes that have both origins valid; `None` when neither eye has.
fn origin_error_mm(frame: &GazeFrame, display: DisplayFrame) -> Option<f64> {
    [&frame.left, &frame.right]
        .into_iter()
        .filter(|eye| eye.origin_tracker_mm.valid && eye.origin_display_mm.valid)
        .map(|eye| {
            let ours = display.to_display(eye.origin_tracker_mm.value);
            ours.iter()
                .zip(eye.origin_display_mm.value)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .reduce(f64::max)
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
    wait_for_gaze_stream_with(h, stop, dur, asm, |_, _| {}).is_some()
}

/// A message's device timestamp and the host time it was read at
/// ([`host_clock_us`]), microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Arrival {
    device_us: i64,
    host_rx_us: i64,
}

/// [`wait_for_gaze_stream`], handing every other message to `side` with the
/// host time it was read at. Returns the arrival of the gaze frame that armed
/// the stream, or `None` when the wait ended without one.
fn wait_for_gaze_stream_with(
    h: &mut rusb::DeviceHandle<UsbContext>,
    stop: &AtomicBool,
    dur: Duration,
    asm: &mut BulkReassembler,
    mut side: impl FnMut(&[u8], i64),
) -> Option<Arrival> {
    let mut buf = vec![0u8; READ_BUF];
    let mut msgs = Vec::new();
    let mut seen = 0usize;
    let start = Instant::now();
    while start.elapsed() < dur && !stop.load(Ordering::Relaxed) {
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(200)) {
            Ok(n) => {
                let host_rx_us = host_clock_us();
                asm.push_into(&buf[..n], &mut msgs);
                for msg in &msgs {
                    seen += 1;
                    let gaze = (stream_id(msg) == Some(STREAM_ID_GAZE))
                        .then(|| parse_message(msg).as_ref().and_then(decode_gaze_frame))
                        .flatten();
                    if let Some(frame) = gaze {
                        return Some(Arrival {
                            device_us: to_i64_us(frame.device_ts_us),
                            host_rx_us,
                        });
                    }
                    debug!(
                        len = msg.len(),
                        marker = ?marker(msg),
                        stream = ?stream_id(msg),
                        "arming: not a gaze frame"
                    );
                    side(msg, host_rx_us);
                }
            }
            Err(rusb::Error::Timeout) => {}
            Err(e) => {
                // Not a stream that is merely slow to start: re-open.
                debug!(error = ?e, "arming: read failed");
                return None;
            }
        }
    }
    debug!(messages = seen, "arming: no gaze frame");
    None
}

/// Longest the armed gaze stream may stay silent before the open is torn down
/// (a [`GazeStalled`], counted as a lost stream) and the device re-opened. An
/// open that never arms fails its own arming wait instead.
const GAZE_LIVENESS_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest the gaze stream may take to come back after a resume before the
/// device is re-opened (whose init replay resumes it).
const RESUME_GRACE: Duration = Duration::from_secs(10);

/// Whether the gaze stream has died and the device must be re-opened: no
/// frame for [`GAZE_LIVENESS_TIMEOUT`], and, since a resume that no frame has
/// followed yet (`since_resume`), not for [`RESUME_GRACE`] either. Never while
/// a command is outstanding (a calibration point takes most of a second) or
/// while the device is paused, when it sends nothing.
fn stream_stalled(
    outstanding: bool,
    paused: bool,
    since_gaze: Duration,
    since_resume: Option<Duration>,
) -> bool {
    !outstanding
        && !paused
        && since_gaze > GAZE_LIVENESS_TIMEOUT
        && since_resume.is_none_or(|t| t > RESUME_GRACE)
}

/// Catches a resume from the engine's pause hint, and keeps it until the
/// next gaze frame. A new attempt starts unpaused: its init replay resumed
/// the device.
#[derive(Debug, Default)]
struct ResumeWatch {
    /// The pause hint as last seen.
    paused: bool,
    /// When the device was resumed, until the next gaze frame.
    resumed_at: Option<Instant>,
}

impl ResumeWatch {
    /// Note the pause hint as it reads at `now`.
    fn note_hint(&mut self, paused: bool, now: Instant) {
        if self.paused && !paused {
            self.resumed_at = Some(now);
        }
        self.paused = paused;
    }

    /// A gaze frame came: the stream is back.
    fn on_gaze(&mut self) {
        self.resumed_at = None;
    }

    /// How long ago the resume that no frame has followed yet was.
    fn since_resume(&self, now: Instant) -> Option<Duration> {
        self.resumed_at.map(|t| now.saturating_duration_since(t))
    }
}

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
/// a 78 KB image message may already be in flight. A write the device
/// refuses is retried within `limits`, discarding what it sends between the
/// tries (see [`write_piece`]), and ends if `stop` is set while the device
/// has taken none of it; `deadline` counts from the end of the write.
fn send_command(
    h: &mut rusb::DeviceHandle<UsbContext>,
    packet: &[u8],
    deadline: Duration,
    limits: WriteLimits,
    stop: Option<&AtomicBool>,
) -> Result<()> {
    let (cmd, expected) = parse_message(packet)
        .map(|m| (m.id, m.seq))
        .context("command packet without a header")?;
    let mut buf = vec![0u8; READ_BUF];
    info_span!("command", cmd, seq = expected)
        .in_scope(|| {
            write_piece(
                h,
                packet,
                limits,
                stop,
                |h, rest, timeout| h.write_bulk(EP_OUT, rest, timeout),
                |h| discard_pending(h, &mut buf),
            )
        })
        .context("command write failed")?;
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

/// Read and drop what EP 0x83 holds, up to [`PIECE_DRAIN_READS`] reads and
/// until one finds nothing: [`send_command`]'s step between the tries of a
/// write the device refuses, which drops stream messages anyway.
fn discard_pending(h: &mut rusb::DeviceHandle<UsbContext>, buf: &mut [u8]) {
    for _ in 0..PIECE_DRAIN_READS {
        if h.read_bulk(EP_IN, buf, PIECE_DRAIN_TIMEOUT).is_err() {
            return;
        }
    }
}

/// Ask the device to start stream `id` (command 1220) and wait for the ack.
/// A tracker that refuses the command is waited out for up to
/// `WRITE_DEADLINE`, unless `stop` is set before it takes any of it.
///
/// # Errors
///
/// Fails when the command cannot be written (a stop included) or is not
/// acknowledged in time.
pub fn start_stream(
    h: &mut rusb::DeviceHandle<UsbContext>,
    cmd_seq: u32,
    id: u32,
    stop: Option<&AtomicBool>,
) -> Result<()> {
    send_command(
        h,
        &stream_start_packet(cmd_seq, id),
        Duration::from_millis(1500),
        COMMAND_WRITE,
        stop,
    )
    .with_context(|| format!("start stream {id:#x}"))
}

/// Ask the device to stop stream `id` (command 1230) and wait for the ack.
/// Best-effort, on the way out of an open: a tracker that refuses it is not
/// waited out (see `STREAM_STOP_WRITE`).
///
/// # Errors
///
/// Fails when the command cannot be written or is not acknowledged in time.
pub fn stop_stream(h: &mut rusb::DeviceHandle<UsbContext>, cmd_seq: u32, id: u32) -> Result<()> {
    send_command(
        h,
        &stream_stop_packet(cmd_seq, id),
        Duration::from_millis(700),
        STREAM_STOP_WRITE,
        None,
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

/// The device time of `frame`, or `None` when it carried none (which
/// [`decode_image_payload`] gives as 0).
fn image_device_us(frame: &ImageFrame) -> Option<i64> {
    (frame.device_ts_us != 0).then(|| to_i64_us(frame.device_ts_us))
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
    resume: ResumeWatch,
    /// This open's device-to-host time map: a new open may restart the
    /// tracker's clock, and a new pump starts a new map.
    time: TimeMap,
    /// This open's number and the display area in effect, which every image
    /// carries; a 1450 the pump routes changes the area. Each gaze frame is
    /// checked against its display frame.
    opens: &'a mut Opens,
    /// Set while the messages the init replay read are delivered (see
    /// [`Pump::deliver_init_side`]), whose gaze frames are not checked.
    in_init_side: bool,
}

impl Pump<'_> {
    /// Route one message outside of command matching, as read now: samples
    /// to `tx`, image frames to the pose worker.
    fn deliver(&mut self, incoming: Incoming) {
        self.deliver_at(incoming, host_clock_us());
    }

    /// [`Pump::deliver`] for a message read at host time `rx_us`. The time
    /// map learns from it before it is stamped: a gaze frame whose device
    /// clock went back must not be mapped with the offset from before.
    fn deliver_at(&mut self, incoming: Incoming, rx_us: i64) {
        self.observe(&incoming, rx_us);
        self.route(incoming, rx_us);
    }

    /// Deliver the messages the init replay read besides its responses,
    /// stamped as read now: no arrival of this open has come yet to map
    /// them. A display-area notification (1450) among them is passed on but
    /// changes nothing here: the init took the area of the last one already
    /// (see [`init_display_area`]), and routing any before it would make
    /// the area in effect go back and forth. A gaze frame among them is
    /// passed on but not checked against the area in effect (see
    /// [`Pump::check_display_frame`]): the device may have converted it with
    /// the area it held before the init wrote its own.
    fn deliver_init_side(&mut self, side: &[Vec<u8>]) {
        self.in_init_side = true;
        for msg in side {
            match classify(msg) {
                Incoming::Notification(n @ DeviceNotification::DisplayAreaChanged(_)) => {
                    debug!(notification = ?n, "device notification");
                    let _ = self.tx.send(Sample::Notification(n));
                }
                incoming => self.deliver(incoming),
            }
        }
        self.in_init_side = false;
    }

    /// Deliver the messages read while the stream armed, each with the host
    /// time it was read at, once the time map has learnt from them all and
    /// from the gaze frame that armed the stream (`armed`, itself not
    /// delivered): a message read early in the wait, presence at stream
    /// start among them, is stamped from the arrivals that followed it rather
    /// than when it was read or delivered.
    fn deliver_early(&mut self, early: Vec<(Incoming, i64)>, armed: Option<Arrival>) {
        for (incoming, rx_us) in &early {
            self.observe(incoming, *rx_us);
        }
        if let Some(armed) = armed {
            self.time
                .observe(Stream::Gaze, armed.device_us, armed.host_rx_us);
        }
        for (incoming, rx_us) in early {
            self.route(incoming, rx_us);
        }
    }

    /// Teach the time map a message read at host time `rx_us`: gaze frames and
    /// images, the two streams that arrive at 33 Hz. An image without a
    /// device time teaches it nothing.
    fn observe(&mut self, incoming: &Incoming, rx_us: i64) {
        let (stream, device_us) = match incoming {
            Incoming::Gaze(frame) => (Stream::Gaze, to_i64_us(frame.device_ts_us)),
            Incoming::Image(frame) => match image_device_us(frame) {
                Some(device_us) => (Stream::Image, device_us),
                None => return,
            },
            _ => return,
        };
        self.time.observe(stream, device_us, rx_us);
    }

    /// Stamp a message read at host time `rx_us` and route it (see
    /// [`Pump::deliver`]); the time map has already seen it. An image
    /// carries the display area in effect when it is routed, which a
    /// display-area notification (1450) changes as it is routed: messages
    /// are routed in the order they were read, so an image read after a
    /// 1450 carries the area it gave, and a gaze frame is checked against
    /// it (see [`Pump::check_display_frame`]).
    fn route(&mut self, incoming: Incoming, rx_us: i64) {
        match incoming {
            Incoming::Gaze(frame) => {
                self.last_gaze = Instant::now();
                self.resume.on_gaze();
                if !self.in_init_side {
                    self.check_display_frame(&frame);
                }
                let host_us = self
                    .time
                    .stamp(Stream::Gaze, to_i64_us(frame.device_ts_us), rx_us);
                let _ = self.tx.send(Sample::Gaze(Box::new(GazeSample {
                    frame: *frame,
                    host_rx_us: rx_us,
                    host_us,
                })));
            }
            Incoming::Presence(p) => {
                let timestamp_us = to_i64_us(p.device_ts_us);
                let _ = self.tx.send(Sample::Presence(PresenceSample {
                    timestamp_us,
                    host_us: self.time.stamp(Stream::Presence, timestamp_us, rx_us),
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
                let host_us = match image_device_us(&frame) {
                    Some(device_us) => self.time.stamp(Stream::Image, device_us, rx_us),
                    None => {
                        debug!("image without a device time: stamped as read");
                        self.time.stamp_read(Stream::Image, rx_us)
                    }
                };
                let in_effect = &self.opens.display;
                let image = ImageSample {
                    frame: Arc::new(frame),
                    host_us,
                    display_frame: in_effect.frame.clone(),
                    display_generation: in_effect.generation,
                    open: self.opens.count,
                };
                // Relaxed: a pure signal.
                if self.shared.image_wanted.load(Ordering::Relaxed) {
                    let _ = self.tx.send(Sample::Image(image.clone()));
                }
                mailbox_put(self.mailbox, image);
            }
            Incoming::Notification(n) => {
                debug!(notification = ?n, "device notification");
                if let DeviceNotification::DisplayAreaChanged(area) = &n {
                    // Not named `display`: tracing's macros take that name.
                    let in_effect = &mut self.opens.display;
                    in_effect.set(Some(*area));
                    debug!(
                        generation = in_effect.generation.get(),
                        has_frame = in_effect.frame.is_some(),
                        "display area in effect"
                    );
                }
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

    /// Check the gaze origins of `frame` against the display frame of the
    /// area in effect (see [`DisplayFrameCheck`]), and warn once for each
    /// display generation when the device's display frame is not ours. A
    /// few dozen flops; nothing to check without a display frame.
    fn check_display_frame(&mut self, frame: &GazeFrame) {
        let Opens {
            count,
            display: in_effect,
            check,
        } = &mut *self.opens;
        let Some(display) = in_effect.frame.as_deref().copied() else {
            return;
        };
        let generation = in_effect.generation;
        if let Some(error_mm) = check.judge(frame, display, *count, generation) {
            warn!(
                error_mm,
                generation = generation.get(),
                "gaze: the device's gaze origins are not in the display frame in effect"
            );
        }
    }

    /// Write the next queued command, if none is outstanding. Reads between
    /// the pieces of a large command so the stream keeps draining, and
    /// between the tries of a piece the device refuses, which is retried for
    /// up to [`WRITE_DEADLINE`] per piece (see [`write_piece`]). The liveness
    /// check waits for the write as it waits for an outstanding command; the
    /// command's deadline starts once it is written. A stop while the device
    /// refuses the first piece, before it has taken any of the command, fails
    /// it with [`CommandError::EngineGone`]; once it has taken some, the
    /// command is written to its end, so the teardown's stream stop never
    /// lands inside it.
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
        let shared = self.shared;
        let mut buf = vec![0u8; READ_BUF];
        let mut msgs = Vec::new();
        for (i, piece) in pieces.iter().enumerate() {
            let span = info_span!("command", cmd = command.cmd, seq, piece = i);
            let written = span.in_scope(|| {
                write_piece(
                    h,
                    piece,
                    COMMAND_WRITE,
                    (i == 0).then_some(&shared.stop),
                    |h, rest, timeout| h.write_bulk(EP_OUT, rest, timeout),
                    |h| self.drain_between_writes(h, asm, &mut buf, &mut msgs),
                )
            });
            if let Err(e) = written {
                // The daemon logs the failure it answers its client with.
                debug!(cmd = command.cmd, seq, piece = i, error = %e, "command write failed");
                let _ = reply.send(Err(e.into()));
                return;
            }
            if i + 1 < pieces.len() {
                // The device stops taking pieces while what it sent is unread
                // (see `replay_init_packets`); with the streams running that
                // includes image frames, so take everything it has.
                self.drain_between_writes(h, asm, &mut buf, &mut msgs);
            }
        }
        self.outstanding = Some(Outstanding {
            cmd: command.cmd,
            seq,
            deadline: Instant::now() + command.timeout,
            reply,
        });
    }

    /// Take what the device sent and deliver it, up to
    /// [`PIECE_DRAIN_READS`] reads and until one finds nothing: the step
    /// between the pieces of a command and between the tries of a piece the
    /// device refuses.
    fn drain_between_writes(
        &mut self,
        h: &mut rusb::DeviceHandle<UsbContext>,
        asm: &mut BulkReassembler,
        buf: &mut [u8],
        msgs: &mut Vec<Vec<u8>>,
    ) {
        for _ in 0..PIECE_DRAIN_READS {
            let Ok(n) = h.read_bulk(EP_IN, buf, PIECE_DRAIN_TIMEOUT) else {
                break;
            };
            // Before the decoding, as in `pump_streams`.
            let rx_us = host_clock_us();
            asm.push_into(&buf[..n], msgs);
            for msg in msgs.drain(..) {
                self.deliver_at(classify(&msg), rx_us);
            }
        }
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
/// actually arms it; the first such bail is the prime, not a failure (see
/// [`FailedOpens`]). A failure after arming carries a [`StreamLost`] context
/// saying how long the stream ran (see [`after_arming`]).
///
/// An init that runs to its end counts an open in `opens` and leaves the
/// display area the device confirmed in effect (see [`Opens::init_done`]);
/// the facts it reports carry that area and its display id.
fn gaze_stream_loop(
    h: &mut rusb::DeviceHandle<UsbContext>,
    shared: &Shared,
    commands: &Receiver<QueuedCommand>,
    tx: &Sender<Sample>,
    mailbox: &PoseMailbox,
    opens: &mut Opens,
) -> Result<()> {
    let stop = &shared.stop;
    let display_override = shared.display_override();
    let packets = crate::calibration::init_packets(display_override.as_ref())?;
    let capture = replay_init_packets(h, &packets, Some(stop))?;
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    let (facts, confirmed) = opens.init_done(&packets, &capture);
    // This init's own 1330 answer, so empty after one that lost it; tobiid
    // then keeps the previous init's properties, if there was one, and tells
    // its clients those (`keep_unreported`).
    info!(
        model = %facts.info.model,
        firmware = %facts.info.firmware_version,
        init_integration_type = %facts.device_info().integration_type,
        calibration_id = ?facts.calibration_id,
        faults = ?facts.status_string(STATUS_FAULTS),
        warnings = ?facts.status_string(STATUS_WARNINGS),
        display_area_from = ?confirmed.map(|c| c.source.message_id()),
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
        resume: ResumeWatch::default(),
        time: TimeMap::default(),
        opens,
        in_init_side: false,
    };
    pump.deliver_init_side(&capture.side);

    // The Windows Stream Engine subscribes the image stream right after its
    // init; we do the same. Failure here is not fatal — gaze still works.
    let mut image = is_image_stream_enabled();
    if image {
        match start_stream(h, pump.cmd_seq, STREAM_ID_IMAGE, Some(stop)) {
            Ok(()) => info!("gaze: image stream 0x50e requested (head pose via IR frames)"),
            // The arming wait below ends on the stop too.
            Err(e) if stop.load(Ordering::Relaxed) => {
                debug!(
                    error = format_args!("{e:#}"),
                    "gaze: image stream start ended by the stop"
                );
                image = false;
            }
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
    let armed = wait_for_gaze_stream_with(
        h,
        stop,
        Duration::from_secs_f64(4.5),
        &mut asm,
        |msg, rx_us| {
            if early.len() < MAX_SIDE_MESSAGES {
                early.push((classify(msg), rx_us));
            }
        },
    );
    pump.deliver_early(early, armed);
    if armed.is_none() {
        if image {
            let _ = stop_stream(h, pump.cmd_seq, STREAM_ID_IMAGE);
        }
        // Also when a stop cut the wait short: `run_opens` ends cleanly on
        // any failure after a stop.
        anyhow::bail!(StreamStartupTimeout);
    }

    // Armed: this open's first gaze frame has come.
    let armed_at = Instant::now();
    let result = after_arming(pump_streams(h, &mut pump, &mut asm), armed_at);
    if let Some(o) = pump.outstanding.take() {
        let _ = o.reply.send(Err(CommandError::EngineGone));
    }
    if image {
        // Mirror the Windows shutdown (1230 for 0x50e before the vendor stop).
        let _ = stop_stream(h, pump.cmd_seq, STREAM_ID_IMAGE);
    }
    result
}

/// Tag the pump's `result` as a failure after arming: a [`StreamLost`]
/// context with the time from `armed_at` to now, taken as it fails. A clean
/// stop passes through.
fn after_arming(result: Result<()>, armed_at: Instant) -> Result<()> {
    result.with_context(|| StreamLost {
        ran: armed_at.elapsed(),
    })
}

/// Demultiplex EP 0x83 until `stop` or failure, running queued commands in
/// between reads. Fails with [`GazeStalled`] when the gaze stream goes silent.
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
        // Relaxed: a pure signal.
        let paused = pump.shared.paused.load(Ordering::Relaxed);
        let now = Instant::now();
        pump.resume.note_hint(paused, now);
        if stream_stalled(
            pump.outstanding.is_some(),
            paused,
            now.saturating_duration_since(pump.last_gaze),
            pump.resume.since_resume(now),
        ) {
            // Logged once, by the engine's open loop, as a lost stream.
            anyhow::bail!(GazeStalled);
        }
        pump.send_next(h, asm);
        pump.expire(asm);
        match h.read_bulk(EP_IN, &mut buf, Duration::from_millis(100)) {
            Ok(n) if n > 0 => {
                // The read time of every message in the transfer, taken
                // before they are decoded (an image's copies 78 KB), as the
                // arming wait takes it.
                let rx_us = host_clock_us();
                asm.push_into(&buf[..n], &mut msgs);
                for msg in msgs.drain(..) {
                    pump.deliver_at(classify(&msg), rx_us);
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
    use AfterFailedOpen::{GiveUp, Reopen, ReopenPrimed, ResetAndReopen};
    use OpenFailure::{Failed, Lost, NotArmed, Refused};
    use OpenRefusal::{InUse, NoPermission};
    use Seen::{Between, Try};
    use Step::{Open, Reset};
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use tobii_proto::facts::DEFAULT_DISPLAY_ID;
    use tobii_proto::protocol::hex_to_bytes;

    const MS: i64 = 1_000;
    const S: i64 = 1_000_000;

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
    fn a_gaze_frame_is_stamped_with_the_host_clock_when_read() {
        let rig = Rig::new();
        let mut opens = Opens::default();
        let mut pump = rig.pump(&mut opens);

        let before = host_clock_us();
        pump.deliver(classify(&fixture("session1-gaze-frame")));
        let after = host_clock_us();

        let [Sample::Gaze(gaze)] = &rig.samples()[..] else {
            panic!("the frame was not delivered as gaze");
        };
        // The daemon's TIMESYNC pair takes its host time from this.
        assert!(
            (before..=after).contains(&gaze.host_rx_us),
            "{} is not between {before} and {after}",
            gaze.host_rx_us
        );
        // The first arrival of an open maps to when it was read.
        assert_eq!(gaze.host_us, gaze.host_rx_us);
    }

    /// What a [`Pump`] borrows, to drive one without a device.
    struct Rig {
        shared: Shared,
        _queue: Sender<QueuedCommand>,
        commands: Receiver<QueuedCommand>,
        tx: Sender<Sample>,
        samples: Receiver<Sample>,
        mailbox: PoseMailbox,
    }

    impl Rig {
        fn new() -> Self {
            let (queue, commands) = std::sync::mpsc::channel();
            let (tx, samples) = std::sync::mpsc::channel();
            Self {
                shared: Shared::default(),
                _queue: queue,
                commands,
                tx,
                samples,
                mailbox: PoseMailbox::default(),
            }
        }

        /// A pump as a fresh open starts one, after the opens before it left
        /// `opens` (an open's init counts it there first, see
        /// [`Opens::init_done`]).
        fn pump<'a>(&'a self, opens: &'a mut Opens) -> Pump<'a> {
            Pump {
                shared: &self.shared,
                commands: &self.commands,
                tx: &self.tx,
                mailbox: &self.mailbox,
                cmd_seq: 0,
                outstanding: None,
                image_live: false,
                last_gaze: Instant::now(),
                resume: ResumeWatch::default(),
                time: TimeMap::default(),
                opens,
                in_init_side: false,
            }
        }

        /// The samples sent so far.
        fn samples(&self) -> Vec<Sample> {
            self.samples.try_iter().collect()
        }

        /// The kind and host time of each sample sent so far.
        fn host_times(&self) -> Vec<(&'static str, i64)> {
            self.samples()
                .iter()
                .map(|sample| match sample {
                    Sample::Gaze(g) => ("gaze", g.host_us),
                    Sample::Presence(p) => ("presence", p.host_us),
                    Sample::Image(i) => ("image", i.host_us),
                    other => panic!("unexpected sample {other:?}"),
                })
                .collect()
        }

        /// The host time of the image waiting for the pose worker, which the
        /// pose made from it carries.
        fn pose_host_us(&self) -> Option<i64> {
            self.pose_image().map(|image| image.host_us)
        }

        /// The image waiting for the pose worker.
        fn pose_image(&self) -> Option<ImageSample> {
            let slot = self
                .mailbox
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            slot.clone()
        }

        /// What the image waiting for the pose worker says of the display
        /// area in effect and the open: its frame, generation and open.
        fn pose_display(&self) -> (Option<DisplayFrame>, DisplayGeneration, OpenNumber) {
            let image = self.pose_image().expect("an image for the pose worker");
            (
                image.display_frame.as_deref().copied(),
                image.display_generation,
                image.open,
            )
        }
    }

    fn device_us(t: i64) -> u64 {
        u64::try_from(t).expect("a device time")
    }

    fn gaze_at(t: i64) -> Incoming {
        Incoming::Gaze(Box::new(GazeFrame {
            device_ts_us: device_us(t),
            ..GazeFrame::default()
        }))
    }

    fn image_at(t: i64) -> Incoming {
        Incoming::Image(ImageFrame {
            device_ts_us: device_us(t),
            width: 2,
            height: 1,
            pixels: vec![0, 0],
        })
    }

    fn presence_at(t: i64) -> Incoming {
        Incoming::Presence(PresenceFrame {
            device_ts_us: device_us(t),
            state: PRESENCE_STATE_PRESENT,
        })
    }

    #[test]
    fn the_pump_maps_every_stream_from_gaze_and_image_arrivals() {
        let rig = Rig::new();
        rig.shared.image_wanted.store(true, Ordering::Relaxed);
        let mut opens = Opens::default();
        let mut pump = rig.pump(&mut opens);

        // Device time plus 40 s, and the latency: 4 ms for the first gaze
        // frame, 1 ms for the image that follows.
        pump.deliver_at(gaze_at(10 * S), 50 * S + 4 * MS);
        pump.deliver_at(image_at(10 * S + 15 * MS), 50 * S + 16 * MS);
        pump.deliver_at(gaze_at(10 * S + 30 * MS), 50 * S + 36 * MS);
        pump.deliver_at(presence_at(10 * S + 31 * MS), 50 * S + 40 * MS);

        // From the image on, everything maps at 40 s + 1 ms: gaze fed alone
        // would have put the second frame at 50.034 s.
        assert_eq!(
            rig.host_times(),
            [
                ("gaze", 50 * S + 4 * MS),
                ("image", 50 * S + 16 * MS),
                ("gaze", 50 * S + 31 * MS),
                ("presence", 50 * S + 32 * MS),
            ]
        );
        assert_eq!(rig.pose_host_us(), Some(50 * S + 16 * MS));
    }

    #[test]
    fn a_gaze_frame_whose_device_clock_went_back_maps_from_itself() {
        let rig = Rig::new();
        let mut opens = Opens::default();
        let mut pump = rig.pump(&mut opens);

        pump.deliver_at(gaze_at(600 * S), 640 * S + 3 * MS);
        // The device clock restarts: mapped with the old offset, this frame
        // would come out 60 s before it was read.
        pump.deliver_at(gaze_at(12 * S), 700 * S);

        assert_eq!(
            rig.host_times(),
            [("gaze", 640 * S + 3 * MS), ("gaze", 700 * S)]
        );
    }

    #[test]
    fn an_image_without_a_device_time_is_stamped_as_read_and_teaches_nothing() {
        let rig = Rig::new();
        rig.shared.image_wanted.store(true, Ordering::Relaxed);
        let mut opens = Opens::default();
        let mut pump = rig.pump(&mut opens);

        pump.deliver_at(image_at(10 * S), 50 * S + MS);
        pump.deliver_at(image_at(0), 50 * S + 31 * MS);
        assert_eq!(rig.pose_host_us(), Some(50 * S + 31 * MS));
        // No restart, so the first image's offset stays: taken for a device
        // time, the 0 would have started the map over, and the gaze frame
        // would map to 50.045 s, the image after it to 50.062 s.
        pump.deliver_at(gaze_at(10 * S + 40 * MS), 50 * S + 45 * MS);
        pump.deliver_at(image_at(10 * S + 60 * MS), 50 * S + 62 * MS);

        assert_eq!(
            rig.host_times(),
            [
                ("image", 50 * S + MS),
                ("image", 50 * S + 31 * MS),
                ("gaze", 50 * S + 41 * MS),
                ("image", 50 * S + 61 * MS),
            ]
        );
    }

    #[test]
    fn the_gaze_frame_that_arms_the_stream_stamps_what_was_read_before_it() {
        let rig = Rig::new();
        let mut opens = Opens::default();
        let mut pump = rig.pump(&mut opens);

        // Presence at stream start, read 5 ms after device time plus 40 s,
        // then the gaze frame that arms the stream, 1 ms after.
        let early = vec![(presence_at(9 * S + 900 * MS), 49 * S + 905 * MS)];
        let armed = Arrival {
            device_us: 10 * S,
            host_rx_us: 50 * S + MS,
        };
        pump.deliver_early(early, Some(armed));

        // Stamped as read, or as delivered, presence would be 4 ms late or
        // more; the arming frame itself is not delivered.
        assert_eq!(rig.host_times(), [("presence", 49 * S + 901 * MS)]);
    }

    #[test]
    fn messages_read_while_arming_are_stamped_from_the_arrivals_that_followed() {
        let rig = Rig::new();
        let mut opens = Opens::default();
        let mut pump = rig.pump(&mut opens);

        // Presence at stream start, then an image (which nobody wants) read
        // 1 ms after device time plus 40 s, then the gaze frame that arms the
        // stream, 3 ms after.
        let early = vec![
            (presence_at(9 * S + 900 * MS), 49 * S + 905 * MS),
            (image_at(9 * S + 950 * MS), 49 * S + 951 * MS),
        ];
        let armed = Arrival {
            device_us: 10 * S,
            host_rx_us: 50 * S + 3 * MS,
        };
        pump.deliver_early(early, Some(armed));

        // Presence maps from the image read after it, not from the arming
        // frame (which would make it 49.903 s).
        assert_eq!(rig.host_times(), [("presence", 49 * S + 901 * MS)]);
        assert_eq!(rig.pose_host_us(), Some(49 * S + 951 * MS));
    }

    #[test]
    fn init_responses_become_device_facts() {
        let responses = [
            fixture("init-rsp-1420"),
            fixture("init-rsp-1400"),
            fixture("init-rsp-1430"),
            fixture("init-rsp-2110"),
            fixture("init-rsp-1490"),
            fixture("init-rsp-2120"),
        ];
        let facts = DeviceFacts::from_messages(responses.iter().filter_map(|r| parse_message(r)));
        assert_eq!(facts.info.generation, "IS5");
        assert!(
            facts.track_box.is_some() && facts.display_area.is_some() && facts.mounting.is_some()
        );
        assert!(facts.hardware.is_some());
        assert_eq!(facts.calibration_id, Some(1_904_654_973));
    }

    /// The display area and display id the fixture `name` carries.
    fn area_of(name: &str) -> (DisplayArea, Option<u32>) {
        let msg = fixture(name);
        parse_display_area(&parse_message(&msg).expect("a message")).expect("an area")
    }

    /// The device's answer to the captured 1440, as it would answer one of
    /// seq `seq` with `status` and TTP `error`.
    fn answer_1440(seq: u32, status: u32, error: u32) -> Vec<u8> {
        let mut msg = fixture("change-display-rsp-1440");
        msg[12..16].copy_from_slice(&seq.to_be_bytes());
        msg[16..20].copy_from_slice(&status.to_be_bytes());
        msg[24..28].copy_from_slice(&error.to_be_bytes());
        msg
    }

    /// A replay that writes `area` with `display_id` as its 1440, of seq 42
    /// as the captured one.
    fn replay_writing(area: &DisplayArea, display_id: u32) -> Vec<InitPacket> {
        let payload = tobii_proto::facts::display_area_set_payload(area, display_id);
        chunk_command(cmd::DISPLAY_AREA_SET, 0x2a, &payload)
            .into_iter()
            .map(|data| InitPacket { ep: EP_OUT, data })
            .collect()
    }

    /// A display-area notification (1450) of `area`: the captured one's
    /// corners, without its trailing zero.
    fn notification_of(area: &DisplayArea) -> Vec<u8> {
        let payload = tobii_proto::tlv::TlvWriter::new()
            .point_mm(area.top_left_mm)
            .point_mm(area.top_right_mm)
            .point_mm(area.bottom_left_mm)
            .finish();
        let mut msg = chunk_command(notify::DISPLAY_AREA, 0, &payload).swap_remove(0);
        msg[..4].copy_from_slice(&[1, 0, 0, 0]);
        msg[8..12].copy_from_slice(&MARKER_NOTIFICATION.to_be_bytes());
        msg
    }

    /// What `init_display_area` makes of an init that replayed `packets`
    /// and read `responses`, and `side` besides.
    fn init_area(
        packets: &[InitPacket],
        responses: &[Vec<u8>],
        side: &[Vec<u8>],
    ) -> Option<(DisplayArea, Option<u32>, AreaSource)> {
        let capture = InitCapture {
            responses: responses.to_vec(),
            side: side.to_vec(),
        };
        init_display_area(packets, &capture).map(|d| (d.area, d.display_id, d.source))
    }

    #[test]
    fn the_replay_writes_the_area_its_capture_read_back() {
        let packets = crate::calibration::embedded_packets().expect("embedded");
        let (a, id) = area_of("init-rsp-1430");
        assert_eq!(
            crate::calibration::written_display_area(&packets),
            Some((0x0e, a, id)),
            "the capture author's monitor, bit for bit as the Windows device read it back"
        );
        let (b, _) = area_of("change-display-cmd-1440");
        assert_ne!(DisplayFrame::new(&a), DisplayFrame::new(&b), "A and B");
    }

    #[test]
    fn an_init_takes_the_last_display_area_notification_first() {
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-notify-1450");
        // The embedded replay writes A at seq 14, which the device takes,
        // and reads A before it: only the 1450 says B.
        let packets = crate::calibration::embedded_packets().expect("embedded");
        let responses = [fixture("init-rsp-1430"), answer_1440(0x0e, 1, 0)];
        let b_1450 = fixture("change-display-notify-1450");

        assert_eq!(
            init_area(&packets, &responses, std::slice::from_ref(&b_1450)),
            Some((b, Some(DEFAULT_DISPLAY_ID), AreaSource::Notified)),
            "a 1450 carries no display id: the write's"
        );
        let side = [
            fixture("init-notify-3180"),
            notification_of(&a),
            b_1450.clone(),
            fixture("init-presence"),
        ];
        assert_eq!(
            init_area(&packets, &responses, &side).map(|(area, ..)| area),
            Some(b),
            "the last 1450, among other messages"
        );
        let side = [b_1450, notification_of(&a)];
        assert_eq!(
            init_area(&packets, &responses, &side),
            Some((a, Some(DEFAULT_DISPLAY_ID), AreaSource::Notified))
        );
        assert_eq!(
            init_area(&packets, &[], &side[..1]),
            Some((b, None, AreaSource::Notified)),
            "nothing else gives a display id"
        );
    }

    #[test]
    fn without_a_notification_an_init_takes_the_area_the_device_took() {
        let (b, _) = area_of("change-display-cmd-1440");
        let packets = vec![InitPacket {
            ep: EP_OUT,
            data: fixture("change-display-cmd-1440"),
        }];
        let read = fixture("init-rsp-1430");
        let took = fixture("change-display-rsp-1440");

        for responses in [[read.clone(), took.clone()], [took, read]] {
            assert_eq!(
                init_area(&packets, &responses, &[fixture("init-presence")]),
                Some((b, Some(DEFAULT_DISPLAY_ID), AreaSource::Written))
            );
        }
        // A write of the area the device holds brings no 1450.
        let replay = replay_writing(&b, 77);
        assert_eq!(
            init_area(&replay, &[fixture("change-display-rsp-1440")], &[]),
            Some((b, Some(77), AreaSource::Written))
        );
    }

    #[test]
    fn an_init_whose_write_was_refused_takes_the_area_it_read() {
        let (a, _) = area_of("init-rsp-1430");
        let packets = vec![InitPacket {
            ep: EP_OUT,
            data: fixture("change-display-cmd-1440"),
        }];
        let read = fixture("init-rsp-1430");

        for (answer, why) in [
            (None, "no answer"),
            (
                Some(answer_1440(0x2a, 1, ttp_error::INVALID_PARAMETER)),
                "an error",
            ),
            (Some(answer_1440(0x2a, 0, 0)), "another status"),
            (Some(answer_1440(0x2b, 1, 0)), "the answer to another seq"),
        ] {
            let responses: Vec<Vec<u8>> =
                [Some(read.clone()), answer].into_iter().flatten().collect();
            assert_eq!(
                init_area(&packets, &responses, &[]),
                Some((a, Some(DEFAULT_DISPLAY_ID), AreaSource::Read)),
                "{why}"
            );
        }
        assert_eq!(
            init_area(&packets, &[answer_1440(0x2a, 0, 0)], &[]),
            None,
            "nothing confirmed"
        );
        assert_eq!(init_area(&[], &[], &[]), None);
    }

    #[test]
    fn an_inits_facts_carry_the_area_the_device_confirmed() {
        let (b, _) = area_of("change-display-notify-1450");
        let packets = replay_writing(&b, 77);
        let responses: Vec<Vec<u8>> = ["1420", "1400", "1430", "2110", "1490"]
            .iter()
            .map(|cmd| fixture(&format!("init-rsp-{cmd}")))
            .chain([answer_1440(0x2a, 1, 0)])
            .collect();
        let capture = InitCapture {
            responses,
            side: vec![fixture("change-display-notify-1450")],
        };

        let (facts, confirmed) = init_facts(&packets, &capture);

        assert_eq!(
            (facts.display_area, facts.display_id),
            (Some(b), Some(77)),
            "not the 1430's"
        );
        assert_eq!(confirmed.map(|c| c.source), Some(AreaSource::Notified));
        let from_responses =
            DeviceFacts::from_messages(capture.responses.iter().filter_map(|r| parse_message(r)));
        assert_eq!(from_responses.display_id, Some(DEFAULT_DISPLAY_ID));
        assert_eq!(
            DeviceFacts {
                display_area: from_responses.display_area,
                display_id: from_responses.display_id,
                ..facts
            },
            from_responses,
            "the rest as the responses give it"
        );

        let none = init_facts(&packets, &InitCapture::default());
        assert_eq!(none, (DeviceFacts::default(), None));
    }

    #[test]
    fn a_notified_area_takes_its_display_id_from_the_next_source_down() {
        let (b, _) = area_of("change-display-notify-1450");
        let replay = replay_writing(&b, 77);
        let side = [fixture("change-display-notify-1450")];
        let read = fixture("init-rsp-1430");

        assert_eq!(
            init_area(&replay, &[read.clone(), answer_1440(0x2a, 1, 0)], &side),
            Some((b, Some(77), AreaSource::Notified)),
            "the write's, which the device took"
        );
        assert_eq!(
            init_area(&replay, &[read, answer_1440(0x2a, 0, 0)], &side),
            Some((b, Some(DEFAULT_DISPLAY_ID), AreaSource::Notified)),
            "the read's, the write refused"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: the 4 x 4 mm square maps exactly
    fn the_area_in_effect_moves_its_generation_on_only_when_it_changes() {
        let mut display = AreaInEffect::default();
        display.set(None);
        assert_eq!(display, AreaInEffect::default(), "nothing confirmed yet");

        // The square the tracker starts each open with fixes the tracker
        // frame itself.
        let start = DisplayArea {
            top_left_mm: [-2.0, 2.0, 0.0],
            top_right_mm: [2.0, 2.0, 0.0],
            bottom_left_mm: [-2.0, -2.0, 0.0],
        };
        display.set(Some(start));
        let frame = display.frame.as_deref().copied().expect("a frame");
        assert_eq!(frame.to_display([30.0, -40.0, 600.0]), [30.0, -40.0, 600.0]);
        assert_eq!(display.generation, DisplayGeneration(1));
        display.set(Some(start));
        assert_eq!(display.generation, DisplayGeneration(1), "the same area");

        let (b, _) = area_of("change-display-notify-1450");
        display.set(Some(b));
        assert_eq!(
            (display.frame.as_deref().copied(), display.generation),
            (DisplayFrame::new(&b), DisplayGeneration(2))
        );
        let narrow = DisplayArea {
            top_right_mm: [-1.5, 2.0, 0.0],
            ..start
        };
        display.set(Some(narrow));
        assert_eq!(
            (display.area, display.frame.as_deref(), display.generation),
            (Some(narrow), None, DisplayGeneration(3)),
            "an area that fixes no frame"
        );
        display.set(None);
        assert_eq!(
            (display.area, display.frame.as_deref(), display.generation),
            (None, None, DisplayGeneration(4))
        );
    }

    #[test]
    fn an_image_routed_after_a_display_area_notification_carries_its_area() {
        let rig = Rig::new();
        rig.shared.image_wanted.store(true, Ordering::Relaxed);
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-notify-1450");
        let mut opens = Opens::default();
        opens.opened(Some(a));
        let mut pump = rig.pump(&mut opens);

        pump.deliver_at(image_at(10 * S), 50 * S);
        let (a_frame, b_frame) = (DisplayFrame::new(&a), DisplayFrame::new(&b));
        assert_eq!(
            rig.pose_display(),
            (a_frame, DisplayGeneration(1), OpenNumber(1))
        );
        let notified = || classify(&fixture("change-display-notify-1450"));
        pump.deliver_at(notified(), 50 * S + 5 * MS);
        pump.deliver_at(image_at(10 * S + 30 * MS), 50 * S + 31 * MS);
        assert_eq!(
            rig.pose_display(),
            (b_frame, DisplayGeneration(2), OpenNumber(1))
        );
        pump.deliver_at(notified(), 50 * S + 40 * MS);
        pump.deliver_at(image_at(10 * S + 60 * MS), 50 * S + 61 * MS);
        assert_eq!(
            rig.pose_display(),
            (b_frame, DisplayGeneration(2), OpenNumber(1)),
            "the same area again"
        );

        // The daemon gets the same images and both notifications.
        let samples = rig.samples();
        let images: Vec<(Option<DisplayFrame>, DisplayGeneration, OpenNumber)> = samples
            .iter()
            .filter_map(|sample| match sample {
                Sample::Image(i) => Some((
                    i.display_frame.as_deref().copied(),
                    i.display_generation,
                    i.open,
                )),
                _ => None,
            })
            .collect();
        assert_eq!(
            images,
            [
                (a_frame, DisplayGeneration(1), OpenNumber(1)),
                (b_frame, DisplayGeneration(2), OpenNumber(1)),
                (b_frame, DisplayGeneration(2), OpenNumber(1)),
            ]
        );
        let notifications = samples
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    Sample::Notification(DeviceNotification::DisplayAreaChanged(area)) if *area == b
                )
            })
            .count();
        assert_eq!(notifications, 2);
    }

    /// The images read while one display area is in effect share the frame
    /// built for its generation, the one waiting for the pose worker
    /// included: a 1450 of the same area builds none, one of another area
    /// builds its own.
    #[test]
    fn the_images_of_a_display_generation_share_its_frame() {
        let rig = Rig::new();
        rig.shared.image_wanted.store(true, Ordering::Relaxed);
        let (a, _) = area_of("init-rsp-1430");
        let mut opens = Opens::default();
        opens.opened(Some(a));
        let mut pump = rig.pump(&mut opens);
        let notified = || classify(&fixture("change-display-notify-1450"));

        pump.deliver_at(image_at(10 * S), 50 * S);
        pump.deliver_at(image_at(10 * S + 30 * MS), 50 * S + 31 * MS);
        pump.deliver_at(notified(), 50 * S + 40 * MS);
        pump.deliver_at(image_at(10 * S + 60 * MS), 50 * S + 61 * MS);
        pump.deliver_at(notified(), 50 * S + 70 * MS);
        pump.deliver_at(image_at(10 * S + 90 * MS), 50 * S + 91 * MS);

        let frames: Vec<Arc<DisplayFrame>> = rig
            .samples()
            .into_iter()
            .filter_map(|sample| match sample {
                Sample::Image(image) => image.display_frame,
                _ => None,
            })
            .collect();
        let [a1, a2, b1, b2] = &frames[..] else {
            panic!("four images with a display frame: {frames:?}");
        };
        assert!(Arc::ptr_eq(a1, a2), "A's generation");
        assert!(Arc::ptr_eq(b1, b2), "B's, notified twice");
        assert!(!Arc::ptr_eq(a1, b1));
        let waiting = rig.pose_image().and_then(|image| image.display_frame);
        assert!(waiting.is_some_and(|frame| Arc::ptr_eq(&frame, b2)));
    }

    #[test]
    fn the_display_area_override_leaves_the_area_in_effect_alone() {
        let rig = Rig::new();
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-cmd-1440");
        let mut opens = Opens::default();
        opens.opened(Some(a));
        let mut pump = rig.pump(&mut opens);

        // What `Engine::set_display_area_override` does. tobiid sets it to
        // the TOBII_DISPLAY_MM area before it writes that area, and to None
        // when a calibration session puts back a configuration without one.
        let in_effect = (DisplayFrame::new(&a), DisplayGeneration(1), OpenNumber(1));
        rig.shared.set_display_override(Some(b));
        pump.deliver_at(image_at(10 * S), 50 * S);
        assert_eq!(rig.pose_display(), in_effect);
        rig.shared.set_display_override(None);
        pump.deliver_at(image_at(10 * S + 30 * MS), 50 * S + 31 * MS);
        assert_eq!(rig.pose_display(), in_effect);
    }

    /// Open again on `opens`, the init leaving `area` in effect, and say
    /// what an image the open reads carries.
    fn image_after_open(
        rig: &Rig,
        opens: &mut Opens,
        area: Option<DisplayArea>,
    ) -> (Option<DisplayFrame>, DisplayGeneration, OpenNumber) {
        opens.opened(area);
        rig.pump(opens).deliver_at(image_at(10 * S), 50 * S);
        rig.pose_display()
    }

    #[test]
    fn a_re_open_is_counted_and_keeps_the_generation_of_an_area_that_stayed() {
        let rig = Rig::new();
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-notify-1450");
        let mut opens = Opens::default();

        let a_frame = DisplayFrame::new(&a);
        let b_frame = DisplayFrame::new(&b);
        assert_eq!(
            image_after_open(&rig, &mut opens, Some(a)),
            (a_frame, DisplayGeneration(1), OpenNumber(1))
        );
        assert_eq!(
            image_after_open(&rig, &mut opens, Some(a)),
            (a_frame, DisplayGeneration(1), OpenNumber(2)),
            "the same area"
        );
        assert_eq!(
            image_after_open(&rig, &mut opens, Some(b)),
            (b_frame, DisplayGeneration(2), OpenNumber(3))
        );
        assert_eq!(
            image_after_open(&rig, &mut opens, None),
            (None, DisplayGeneration(3), OpenNumber(4)),
            "an init that confirmed no area"
        );

        // A 1450 during an open is what the next one compares with.
        assert_eq!(
            image_after_open(&rig, &mut opens, Some(a)),
            (a_frame, DisplayGeneration(4), OpenNumber(5))
        );
        rig.pump(&mut opens)
            .deliver(classify(&fixture("change-display-notify-1450")));
        assert_eq!(
            image_after_open(&rig, &mut opens, Some(b)),
            (b_frame, DisplayGeneration(5), OpenNumber(6))
        );
    }

    /// What an open does once its init replay has run, as `gaze_stream_loop`
    /// does it: take the init's facts from [`Opens::init_done`], then pump.
    /// Each init counts its open, and leaves in effect the area the device
    /// confirmed, which the images the open reads carry; a re-open does it
    /// again on the same `Opens`.
    #[test]
    fn each_init_counts_its_open_and_leaves_the_area_it_confirmed_in_effect() {
        let rig = Rig::new();
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-notify-1450");
        // A replay that reads the area, then writes B.
        let packets = replay_writing(&b, 77);
        // The device answered the read with A, took the write and sent B's
        // 1450.
        let took_b = InitCapture {
            responses: vec![fixture("init-rsp-1430"), answer_1440(0x2a, 1, 0)],
            side: vec![fixture("change-display-notify-1450")],
        };
        // Only the read was answered: A.
        let read_a = InitCapture {
            responses: vec![fixture("init-rsp-1430")],
            side: Vec::new(),
        };
        let mut opens = Opens::default();
        let mut open = |capture: &InitCapture| {
            let init = opens.init_done(&packets, capture);
            assert_eq!(init, init_facts(&packets, capture), "what the init says");
            rig.pump(&mut opens).deliver_at(image_at(10 * S), 50 * S);
            (init.0.display_area, rig.pose_display())
        };

        let (a_frame, b_frame) = (DisplayFrame::new(&a), DisplayFrame::new(&b));
        assert_eq!(
            open(&took_b),
            (Some(b), (b_frame, DisplayGeneration(1), OpenNumber(1)))
        );
        assert_eq!(
            open(&took_b),
            (Some(b), (b_frame, DisplayGeneration(1), OpenNumber(2))),
            "the same area"
        );
        assert_eq!(
            open(&read_a),
            (Some(a), (a_frame, DisplayGeneration(2), OpenNumber(3)))
        );
        assert_eq!(
            open(&InitCapture::default()),
            (None, (None, DisplayGeneration(3), OpenNumber(4))),
            "an init that confirmed no area"
        );
    }

    #[test]
    fn the_inits_own_notifications_leave_the_area_it_took_alone() {
        let rig = Rig::new();
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-notify-1450");
        let mut opens = Opens::default();
        // An init whose replay read a 1450 of A, then one of B, took B.
        opens.opened(Some(b));
        let mut pump = rig.pump(&mut opens);

        pump.deliver_init_side(&[
            notification_of(&a),
            fixture("init-presence"),
            fixture("change-display-notify-1450"),
        ]);
        pump.deliver_at(image_at(10 * S), 50 * S);

        assert_eq!(
            rig.pose_display(),
            (DisplayFrame::new(&b), DisplayGeneration(1), OpenNumber(1))
        );
        let delivered: Vec<&str> = rig
            .samples()
            .iter()
            .map(|sample| match sample {
                Sample::Notification(DeviceNotification::DisplayAreaChanged(area))
                    if *area == a =>
                {
                    "A"
                }
                Sample::Notification(DeviceNotification::DisplayAreaChanged(area))
                    if *area == b =>
                {
                    "B"
                }
                Sample::Presence(_) => "presence",
                other => panic!("unexpected sample {other:?}"),
            })
            .collect();
        assert_eq!(delivered, ["A", "presence", "B"], "all passed on, in order");
    }

    /// The session-1 gaze frame, which the device converted with the
    /// captured area A (the init's 1430): both eyes have both origins valid.
    fn session1_frame() -> GazeFrame {
        let msg = fixture("session1-gaze-frame");
        decode_gaze_frame(&parse_message(&msg).expect("a message")).expect("a gaze frame")
    }

    /// `frame` as the device holding `area` would send it: its display-frame
    /// origins converted from its tracker-frame ones with that area's frame.
    fn converted_with(mut frame: GazeFrame, area: &DisplayArea) -> GazeFrame {
        let display = DisplayFrame::new(area).expect("a display frame");
        for eye in [&mut frame.left, &mut frame.right] {
            eye.origin_display_mm.value = display.to_display(eye.origin_tracker_mm.value);
        }
        frame
    }

    fn gaze(frame: GazeFrame) -> Incoming {
        Incoming::Gaze(Box::new(frame))
    }

    /// The square the tracker starts each open with, whose frame is the
    /// tracker frame itself: the session-1 origins miss it by some 75 mm.
    fn start_square() -> DisplayArea {
        DisplayArea {
            top_left_mm: [-2.0, 2.0, 0.0],
            top_right_mm: [2.0, 2.0, 0.0],
            bottom_left_mm: [-2.0, -2.0, 0.0],
        }
    }

    /// A warning logged: its message, and its other fields as `Debug`
    /// formats them.
    #[derive(Debug, Default)]
    struct Logged {
        message: String,
        fields: Vec<(&'static str, String)>,
    }

    impl Logged {
        fn field(&self, name: &str) -> &str {
            self.fields
                .iter()
                .find_map(|(n, v)| (*n == name).then_some(v.as_str()))
                .unwrap_or_else(|| panic!("no field {name} in {self:?}"))
        }

        /// The error and generation of a warning that the device's display
        /// frame is not ours.
        fn mismatch(&self) -> (f64, u64) {
            assert!(
                self.message.contains("not in the display frame in effect"),
                "{self:?}"
            );
            (
                self.field("error_mm").parse().expect("error_mm"),
                self.field("generation").parse().expect("generation"),
            )
        }

        /// The generation of a warning that the display area in effect
        /// fixes no display frame.
        fn frameless(&self) -> u64 {
            assert!(self.message.contains("fixes no display frame"), "{self:?}");
            self.field("generation").parse().expect("generation")
        }
    }

    impl tracing::field::Visit for Logged {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
            if field.name() == "message" {
                self.message = format!("{value:?}");
            } else {
                self.fields.push((field.name(), format!("{value:?}")));
            }
        }
    }

    /// Collects what is logged at warn level on a thread that has it as its
    /// default subscriber.
    #[derive(Debug, Default)]
    struct Warnings(Mutex<Vec<Logged>>);

    impl Warnings {
        /// Collect the warnings logged on this thread while the guard lives.
        fn collect() -> (Arc<Self>, tracing::subscriber::DefaultGuard) {
            let warnings = Arc::new(Self::default());
            let guard = tracing::subscriber::set_default(Arc::clone(&warnings));
            (warnings, guard)
        }

        /// The warnings logged so far, taking them.
        fn take(&self) -> Vec<Logged> {
            let mut logged = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            logged.drain(..).collect()
        }

        /// The error and generation of each warning so far that the
        /// device's display frame is not ours, taking them: there must be
        /// no other.
        fn mismatches(&self) -> Vec<(f64, u64)> {
            self.take().iter().map(Logged::mismatch).collect()
        }

        /// The generations of [`Warnings::mismatches`].
        fn generations(&self) -> Vec<u64> {
            self.mismatches().into_iter().map(|(_, g)| g).collect()
        }

        /// The generation of each warning so far that the display area in
        /// effect fixes no display frame, taking them: there must be no
        /// other.
        fn frameless(&self) -> Vec<u64> {
            self.take().iter().map(Logged::frameless).collect()
        }
    }

    impl tracing::Subscriber for Warnings {
        fn register_callsite(
            &self,
            _: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            // Asked at every event, as other tests log on their own threads.
            tracing::subscriber::Interest::sometimes()
        }

        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            *metadata.level() == tracing::Level::WARN
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            let mut logged = Logged::default();
            event.record(&mut logged);
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(logged);
        }

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// The device converts its gaze origins as the display frame of the
    /// area it holds does, to 0.00016 mm (here 0.00013), and that of the
    /// other area recorded misses them by 0.847 mm.
    #[test]
    fn the_device_converts_its_gaze_origins_with_the_frame_of_its_area() {
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-cmd-1440");
        let frame_of = |area: &DisplayArea| DisplayFrame::new(area).expect("a display frame");
        let frame = session1_frame();

        let on_a = origin_error_mm(&frame, frame_of(&a)).expect("eyes to judge");
        let on_b = origin_error_mm(&frame, frame_of(&b)).expect("eyes to judge");

        assert!(on_a < 0.000_16, "{on_a}");
        assert!((on_b - 0.847).abs() < 0.001, "{on_b}");
        let as_b = converted_with(frame, &b);
        assert!(origin_error_mm(&as_b, frame_of(&b)).expect("eyes") < 1e-12);
    }

    #[test]
    fn gaze_frames_that_miss_the_area_in_effect_three_times_in_a_row_warn_once() {
        let (warnings, _guard) = Warnings::collect();
        let rig = Rig::new();
        let (a, _) = area_of("init-rsp-1430");
        let (b, _) = area_of("change-display-cmd-1440");

        let mut opens = Opens::default();
        opens.opened(Some(a));
        let mut pump = rig.pump(&mut opens);
        for _ in 0..5 {
            pump.deliver(gaze(session1_frame()));
        }
        assert_eq!(warnings.mismatches(), [], "the area the device held");
        assert_eq!(pump.opens.check.misses, 0);

        let mut opens = Opens::default();
        opens.opened(Some(b));
        let mut pump = rig.pump(&mut opens);
        pump.deliver(gaze(session1_frame()));
        pump.deliver(gaze(session1_frame()));
        assert_eq!(warnings.mismatches(), [], "two frames");
        pump.deliver(gaze(session1_frame()));
        let [(error_mm, generation)] = warnings.mismatches()[..] else {
            panic!("one warning on the third frame");
        };
        assert!((error_mm - 0.847).abs() < 0.001, "{error_mm}");
        assert_eq!(generation, 1);
        for _ in 0..5 {
            pump.deliver(gaze(session1_frame()));
        }
        assert_eq!(warnings.mismatches(), [], "once for the area");
        let gazes = rig
            .samples()
            .iter()
            .filter(|s| matches!(s, Sample::Gaze(_)))
            .count();
        assert_eq!(gazes, 13, "every frame passed on");
    }

    #[test]
    fn gaze_frames_the_init_replay_read_are_not_checked() {
        let (warnings, _guard) = Warnings::collect();
        let rig = Rig::new();
        let (b, _) = area_of("change-display-cmd-1440");
        let mut opens = Opens::default();
        opens.opened(Some(b));
        let mut pump = rig.pump(&mut opens);

        // The device may have converted these with the area it held before
        // the init wrote B.
        pump.deliver_init_side(&vec![fixture("session1-gaze-frame"); 4]);
        assert_eq!(pump.opens.check, DisplayFrameCheck::default());
        pump.deliver(gaze(session1_frame()));
        pump.deliver(gaze(session1_frame()));
        assert_eq!(warnings.mismatches(), [], "two frames after the init");
        pump.deliver(gaze(session1_frame()));
        assert_eq!(warnings.mismatches().len(), 1, "the third");
        assert_eq!(rig.samples().len(), 7, "the init's frames passed on");
    }

    /// A frame that fits breaks a run of misses, and a frame misses when
    /// either eye does; a frame with no eye to judge neither breaks a run
    /// nor counts, and one eye is enough to judge.
    #[test]
    fn a_frame_that_fits_breaks_a_run_and_one_without_origins_does_not() {
        let (b, _) = area_of("change-display-cmd-1440");
        let display = DisplayFrame::new(&b).expect("a display frame");
        let miss = session1_frame();
        let fit = converted_with(miss, &b);
        // The left eye fits, the right one misses.
        let mut right_misses = fit;
        right_misses.right.origin_display_mm.value = miss.right.origin_display_mm.value;
        // Each eye has one of its origins valid, not both.
        let mut no_origins = miss;
        no_origins.left.origin_tracker_mm.valid = false;
        no_origins.right.origin_display_mm.valid = false;
        // Only the right eye has both, and it misses.
        let mut right_alone = right_misses;
        right_alone.left.origin_display_mm.valid = false;
        let mut check = DisplayFrameCheck::default();
        let mut judge = |frame: &GazeFrame| {
            check
                .judge(frame, display, OpenNumber(1), DisplayGeneration(1))
                .is_some()
        };

        assert!(!judge(&miss) && !judge(&miss));
        assert!(!judge(&fit), "fits");
        assert!(!judge(&miss) && !judge(&right_misses), "a new run");
        assert!(!judge(&no_origins), "no eye to judge");
        assert!(judge(&right_alone), "the third miss");
        assert!(!judge(&miss) && !judge(&miss) && !judge(&miss), "warned of");
    }

    /// A frame misses when an eye's origin lies more than 0.01 mm from
    /// where the frame in effect puts it.
    #[test]
    fn a_frame_misses_by_more_than_a_hundredth_of_a_millimetre() {
        let (b, _) = area_of("change-display-cmd-1440");
        let display = DisplayFrame::new(&b).expect("a display frame");
        let off_by = |mm: f64| {
            let mut frame = converted_with(session1_frame(), &b);
            frame.left.origin_display_mm.value[2] += mm;
            frame
        };

        for (mm, warns) in [(0.0099, false), (0.0101, true), (-0.0101, true)] {
            let mut check = DisplayFrameCheck::default();
            let frame = off_by(mm);
            let warned = (0..3).any(|_| {
                check
                    .judge(&frame, display, OpenNumber(1), DisplayGeneration(1))
                    .is_some()
            });
            assert_eq!(warned, warns, "{mm} mm");
        }
    }

    /// A new display generation, or a new open, starts a run over, and each
    /// generation is warned of once whatever the opens.
    #[test]
    fn each_display_generation_is_warned_of_once_and_each_open_starts_a_run() {
        let (warnings, _guard) = Warnings::collect();
        let rig = Rig::new();
        let (b, _) = area_of("change-display-cmd-1440");
        let mut opens = Opens::default();
        let miss = || gaze(session1_frame());

        // Generation 1, B: two misses, then a 1450 of the start square.
        opens.opened(Some(b));
        let mut pump = rig.pump(&mut opens);
        pump.deliver(miss());
        pump.deliver(miss());
        pump.deliver(classify(&notification_of(&start_square())));
        pump.deliver(miss());
        assert_eq!(warnings.generations(), [], "a new run at the 1450");
        pump.deliver(miss());
        pump.deliver(miss());
        assert_eq!(warnings.generations(), [2]);
        // B again, twice: generation 3, a generation of its own.
        pump.deliver(classify(&notification_of(&b)));
        pump.deliver(classify(&notification_of(&b)));
        for _ in 0..3 {
            pump.deliver(miss());
        }
        assert_eq!(warnings.generations(), [3]);

        // A re-open that leaves B in effect keeps generation 3.
        opens.opened(Some(b));
        for _ in 0..3 {
            rig.pump(&mut opens).deliver(miss());
        }
        assert_eq!(warnings.generations(), [], "generation 3 again");
        // Generation 4, over two opens.
        opens.opened(Some(start_square()));
        let mut pump = rig.pump(&mut opens);
        pump.deliver(miss());
        pump.deliver(miss());
        opens.opened(Some(start_square()));
        rig.pump(&mut opens).deliver(miss());
        assert_eq!(warnings.generations(), [], "a new run at the open");
        let mut pump = rig.pump(&mut opens);
        pump.deliver(miss());
        pump.deliver(miss());
        assert_eq!(warnings.generations(), [4]);
    }

    #[test]
    fn nothing_is_checked_without_a_display_frame() {
        let (warnings, _guard) = Warnings::collect();
        let rig = Rig::new();
        let narrow = DisplayArea {
            top_right_mm: [-1.5, 2.0, 0.0],
            ..start_square()
        };

        for area in [None, Some(narrow)] {
            let mut opens = Opens::default();
            opens.opened(area);
            let mut pump = rig.pump(&mut opens);
            for _ in 0..5 {
                pump.deliver(gaze(session1_frame()));
            }
            assert_eq!(pump.opens.check, DisplayFrameCheck::default(), "{area:?}");
        }
        assert_eq!(
            warnings.frameless(),
            [1],
            "the narrow area, as it was taken"
        );
    }

    /// An area that fixes no display frame is warned of as an init or a
    /// 1450 makes it the area in effect, once for each generation: a re-open
    /// that keeps it, or a 1450 of it again, says nothing more. An init that
    /// confirms no area at all is no such area (its "device ready" line
    /// says where the area came from).
    #[test]
    fn an_area_in_effect_that_fixes_no_frame_is_warned_of_once_for_each_generation() {
        let (warnings, _guard) = Warnings::collect();
        let rig = Rig::new();
        let (a, _) = area_of("init-rsp-1430");
        // Its left edge is 30 mm long, as tobiid asks of an area it loads,
        // but it is 1 mm high across its top edge.
        let sheared = DisplayArea {
            top_left_mm: [-50.0, 60.0, 0.0],
            top_right_mm: [50.0, 60.0, 0.0],
            bottom_left_mm: [-80.0, 59.0, 0.0],
        };
        let mut opens = Opens::default();

        opens.opened(Some(sheared));
        assert_eq!(warnings.frameless(), [1], "taken at an init");
        opens.opened(Some(sheared));
        assert_eq!(warnings.frameless(), [], "a re-open that keeps it");
        let mut pump = rig.pump(&mut opens);
        pump.deliver(classify(&notification_of(&a)));
        assert_eq!(warnings.frameless(), [], "A fixes one");
        pump.deliver(classify(&notification_of(&sheared)));
        assert_eq!(warnings.frameless(), [3], "taken from a 1450");
        pump.deliver(classify(&notification_of(&sheared)));
        assert_eq!(warnings.frameless(), [], "the same area again");
        pump.deliver_at(image_at(10 * S), 50 * S);
        assert_eq!(
            rig.pose_display(),
            (None, DisplayGeneration(3), OpenNumber(2))
        );

        opens.opened(None);
        assert_eq!(warnings.frameless(), [], "no area");
        assert_eq!(opens.display.generation, DisplayGeneration(4));
    }

    #[test]
    fn a_silent_stream_is_dead_unless_busy_paused_or_just_resumed() {
        let secs = Duration::from_secs;
        assert!(!stream_stalled(false, false, secs(4), None));
        assert!(stream_stalled(false, false, secs(6), None));
        assert!(
            !stream_stalled(true, false, secs(60), None),
            "a command outstanding"
        );
        assert!(!stream_stalled(false, true, secs(600), None), "paused");
        assert!(
            !stream_stalled(false, false, secs(60), Some(secs(9))),
            "inside the grace after a resume"
        );
        assert!(stream_stalled(false, false, secs(60), Some(secs(11))));
        assert!(
            !stream_stalled(false, false, secs(4), Some(secs(11))),
            "a frame came since"
        );
    }

    #[test]
    fn a_resume_is_watched_until_the_next_gaze_frame() {
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        let mut w = ResumeWatch::default();
        w.note_hint(false, at(0));
        assert_eq!(w.since_resume(at(1)), None, "a new attempt is not resuming");

        w.note_hint(true, at(1));
        w.note_hint(true, at(2));
        assert_eq!(w.since_resume(at(3)), None, "still paused");

        w.note_hint(false, at(3));
        w.note_hint(false, at(4));
        assert_eq!(w.since_resume(at(7)), Some(Duration::from_secs(4)));

        w.on_gaze();
        assert_eq!(w.since_resume(at(8)), None);
    }

    /// What `failed` says after each open in turn, given how each one failed.
    fn after_each(failed: &mut FailedOpens, failures: &[OpenFailure]) -> Vec<AfterFailedOpen> {
        failures
            .iter()
            .map(|&failure| failed.record(failure))
            .collect()
    }

    #[test]
    fn the_prime_of_a_cold_start_is_not_counted() {
        let mut failed = FailedOpens::default();
        assert_eq!(after_each(&mut failed, &[NotArmed]), [ReopenPrimed]);
        assert_eq!(failed.in_a_row(), 0, "the re-open arms and runs");
    }

    #[test]
    fn a_stall_soon_after_a_primed_start_is_not_reset() {
        // The hardware log: a cold start's prime, a stall 8 s after arming,
        // then the re-open's own prime before it arms.
        let mut failed = FailedOpens::default();
        let stall = Lost(Duration::from_secs(8));
        assert_eq!(
            after_each(&mut failed, &[NotArmed, stall, NotArmed]),
            [ReopenPrimed, Reopen, ReopenPrimed]
        );
        assert_eq!(failed.in_a_row(), 1);
    }

    #[test]
    fn opens_that_never_arm_are_primed_once_reset_after_two_and_given_up_after_five() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[NotArmed; 6]),
            [ReopenPrimed, Reopen, ResetAndReopen, Reopen, Reopen, GiveUp]
        );
    }

    #[test]
    fn a_device_that_arms_and_dies_is_reset_after_the_second_loss_and_given_up_after_the_fifth() {
        let mut failed = FailedOpens::default();
        let quick = Lost(Duration::from_secs(1));
        assert_eq!(
            after_each(&mut failed, &[NotArmed, quick].repeat(5)),
            [
                ReopenPrimed,
                Reopen,
                ReopenPrimed,
                ResetAndReopen,
                ReopenPrimed,
                Reopen,
                ReopenPrimed,
                Reopen,
                ReopenPrimed,
                GiveUp
            ],
            "each re-open after a loss is primed again"
        );
    }

    #[test]
    fn an_init_failure_counts_and_leaves_the_prime_due() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[Failed, NotArmed, NotArmed]),
            [Reopen, ReopenPrimed, ResetAndReopen]
        );
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[Failed; 5]),
            [Reopen, ResetAndReopen, Reopen, Reopen, GiveUp]
        );
    }

    #[test]
    fn an_init_failure_after_the_prime_leaves_it_spent() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[NotArmed, Failed, NotArmed]),
            [ReopenPrimed, Reopen, ResetAndReopen],
            "only a stream that armed makes the prime due again"
        );
    }

    #[test]
    fn a_stream_that_ran_gives_the_next_open_the_whole_budget() {
        let mut failed = FailedOpens::default();
        for _ in 0..100 {
            assert_eq!(
                after_each(&mut failed, &[NotArmed, Lost(HEALTHY_STREAM)]),
                [ReopenPrimed, Reopen],
                "a prime, then a stream that ran exactly long enough"
            );
        }
        assert_eq!(failed.in_a_row(), 0);
        assert_eq!(
            after_each(&mut failed, &[NotArmed; 6]),
            [ReopenPrimed, Reopen, ResetAndReopen, Reopen, Reopen, GiveUp],
            "as a new engine would"
        );
    }

    #[test]
    fn a_stream_that_dies_soon_after_arming_is_a_failed_open() {
        let mut failed = FailedOpens::default();
        let short = Lost(HEALTHY_STREAM - Duration::from_secs(1));
        assert_eq!(
            after_each(&mut failed, &[short; 5]),
            [Reopen, ResetAndReopen, Reopen, Reopen, GiveUp]
        );
    }

    #[test]
    fn a_stream_that_ran_forgives_the_failures_before_it() {
        let mut failed = FailedOpens::default();
        let hour = Lost(Duration::from_secs(3600));
        assert_eq!(
            after_each(&mut failed, &[NotArmed, NotArmed, NotArmed, hour]),
            [ReopenPrimed, Reopen, ResetAndReopen, Reopen]
        );
        assert_eq!(
            after_each(&mut failed, &[NotArmed; 3]),
            [ReopenPrimed, Reopen, ResetAndReopen],
            "a new run is primed and reset again on its own"
        );
    }

    #[test]
    fn failures_of_every_kind_share_one_count() {
        let mut failed = FailedOpens::default();
        let short = Lost(Duration::from_secs(10));
        assert_eq!(
            after_each(
                &mut failed,
                &[
                    NotArmed, NotArmed, short, Failed, NotArmed, NotArmed, Failed
                ]
            ),
            [
                ReopenPrimed,
                Reopen,
                ResetAndReopen,
                Reopen,
                ReopenPrimed,
                Reopen,
                GiveUp
            ]
        );
    }

    #[test]
    fn an_open_without_permission_is_retried_once_without_a_reset_then_given_up() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[Refused(NoPermission); 2]),
            [Reopen, GiveUp]
        );
        assert_eq!(NO_PERMISSION_RETRIES, 1);
        assert_eq!(failed.in_a_row(), 0, "not counted as failed opens");

        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(
                &mut failed,
                &[
                    NotArmed,
                    Failed,
                    Refused(NoPermission),
                    Failed,
                    Refused(NoPermission),
                    Refused(NoPermission),
                ]
            ),
            [ReopenPrimed, Reopen, Reopen, ResetAndReopen, Reopen, GiveUp],
            "another failure ends the streak; the refusal is not the reset's second failure"
        );
    }

    #[test]
    fn a_busy_interface_is_retried_twice_without_a_reset_then_given_up() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[Refused(InUse); 3]),
            [Reopen, Reopen, GiveUp]
        );
        assert_eq!(BUSY_RETRIES, 2);
        assert_eq!(failed.in_a_row(), 0, "not counted as failed opens");
    }

    #[test]
    fn refusals_of_either_kind_in_a_row_make_one_streak() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(&mut failed, &[Refused(InUse), Refused(NoPermission)]),
            [Reopen, GiveUp],
            "the second refusal in a row, one past a missing permission's retry"
        );
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(
                &mut failed,
                &[Refused(NoPermission), Refused(InUse), Refused(InUse)]
            ),
            [Reopen, Reopen, GiveUp],
            "never more than the longest streak"
        );
    }

    #[test]
    fn a_busy_interface_leaves_the_failure_count_and_the_prime_alone() {
        let mut failed = FailedOpens::default();
        assert_eq!(
            after_each(
                &mut failed,
                &[
                    Refused(InUse),
                    Refused(InUse),
                    NotArmed,
                    Refused(InUse),
                    NotArmed,
                    Refused(InUse),
                    Refused(InUse),
                    NotArmed,
                ]
            ),
            [
                Reopen,
                Reopen,
                ReopenPrimed,
                Reopen,
                Reopen,
                Reopen,
                Reopen,
                ResetAndReopen
            ],
            "another failure ends a busy streak; the prime and the reset come as without it"
        );
    }

    #[test]
    fn an_open_refused_by_libusb_is_classified_by_its_error_in_the_open_step() {
        let refused = |step: OpenStep, error: rusb::Error| {
            Err::<(), _>(error).context(step).expect_err("refused")
        };

        let access = refused(OPEN_DEVICE, rusb::Error::Access);
        assert_eq!(OpenRefusal::of(&access), Some(NoPermission));
        assert_eq!(OpenFailure::of(&access), Refused(NoPermission));
        assert_eq!(
            format!("{access:#}"),
            "failed to open Tobii device; try sudo: Access denied (insufficient permissions)",
            "the message open_tobii always gave"
        );
        let busy = refused(CLAIM_INTERFACE, rusb::Error::Busy);
        assert_eq!(OpenFailure::of(&busy), Refused(InUse));
        assert_eq!(
            OpenFailure::of(&busy.context("open 3")),
            Refused(InUse),
            "wrapped further"
        );

        // Other errors of those calls, and those errors anywhere else.
        assert_eq!(
            OpenFailure::of(&refused(OPEN_DEVICE, rusb::Error::NoDevice)),
            Failed
        );
        assert_eq!(
            OpenFailure::of(&refused(CLAIM_INTERFACE, rusb::Error::NotFound)),
            Failed
        );
        assert_eq!(OpenFailure::of(&rusb::Error::Access.into()), Failed);
        let control = Err::<(), _>(rusb::Error::Busy)
            .context("control OUT 48")
            .expect_err("control");
        assert_eq!(OpenFailure::of(&control), Failed);
        let armed_at = Instant::now();
        let lost = after_arming(Err(access), armed_at).expect_err("lost");
        assert!(
            matches!(OpenFailure::of(&lost), Lost(_)),
            "the lost tag wins"
        );
    }

    #[test]
    fn a_failed_open_is_classified_by_its_lost_tag_then_its_cause() {
        let armed_at = Instant::now()
            .checked_sub(HEALTHY_STREAM)
            .expect("a monotonic clock older than HEALTHY_STREAM");
        let lost_long =
            |e: &anyhow::Error| matches!(OpenFailure::of(e), Lost(ran) if ran >= HEALTHY_STREAM);

        // The two ways the pump fails, tagged as `gaze_stream_loop` tags them.
        let stalled = after_arming(Err(GazeStalled.into()), armed_at).expect_err("stalled");
        assert!(lost_long(&stalled));
        assert_eq!(
            format!("{stalled:#}"),
            "gaze stream lost after arming: no gaze frame for over 5s"
        );
        let usb = after_arming(Err(rusb::Error::NoDevice.into()), armed_at).expect_err("usb");
        assert!(lost_long(&usb));
        let tagged = after_arming(Err(StreamStartupTimeout.into()), armed_at).expect_err("tagged");
        assert!(lost_long(&tagged), "the tag wins over any cause");

        // An open that accepted the init but never armed, bare or wrapped.
        assert_eq!(OpenFailure::of(&StreamStartupTimeout.into()), NotArmed);
        let wrapped = anyhow::Error::new(StreamStartupTimeout).context("open 2");
        assert_eq!(OpenFailure::of(&wrapped), NotArmed);

        // Every other way an open fails before it arms.
        let not_found = None::<()>
            .context("Tobii 2104:0313 not found by libusb")
            .expect_err("not found");
        assert_eq!(OpenFailure::of(&not_found), Failed);
        let write_timeout = Err::<(), _>(rusb::Error::Timeout)
            .context("init packet 57 write failed (command seq Some(39), 2961 ms in)")
            .expect_err("write timeout");
        assert_eq!(
            OpenFailure::of(&write_timeout),
            Failed,
            "an init that failed"
        );
        assert_eq!(
            OpenFailure::of(&rusb::Error::NoDevice.into()),
            Failed,
            "a USB error before arming"
        );

        assert!(after_arming(Ok(()), armed_at).is_ok(), "a clean stop");
    }

    /// What `run_opens` did, in order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Step {
        Open,
        Reset,
    }

    /// An open that never armed.
    fn never_armed() -> Result<()> {
        Err(StreamStartupTimeout.into())
    }

    /// An open whose stream armed and stalled `ran` later, as tagged by
    /// `after_arming`.
    fn lost_after(ran: Duration) -> Result<()> {
        Err(anyhow::Error::new(GazeStalled).context(StreamLost { ran }))
    }

    /// Run `run_opens` with no pause, each open returning the next of
    /// `outcomes`, and say what it did and what it returned.
    fn drive(outcomes: Vec<Result<()>>) -> (Vec<Step>, Result<()>) {
        drive_with(outcomes, true)
    }

    /// [`drive`], with no reset to run unless `can_reset` (as with
    /// `TOBII_NO_RESET=1`).
    fn drive_with(outcomes: Vec<Result<()>>, can_reset: bool) -> (Vec<Step>, Result<()>) {
        let stop = AtomicBool::new(false);
        let steps = RefCell::new(Vec::new());
        let mut outcomes = outcomes.into_iter();
        let result = run_opens(
            &stop,
            |_| {
                steps.borrow_mut().push(Step::Open);
                outcomes.next().expect("an open past the script")
            },
            can_reset.then_some(|| steps.borrow_mut().push(Step::Reset)),
            Duration::ZERO,
        );
        (steps.into_inner(), result)
    }

    #[test]
    fn the_open_loop_resets_before_the_third_open_and_gives_up_after_the_fifth() {
        let (steps, result) = drive((1..=5).map(|n| Err(anyhow::anyhow!("open {n}"))).collect());
        assert_eq!(steps, [Open, Open, Reset, Open, Open, Open]);
        assert_eq!(
            result.expect_err("gave up").to_string(),
            "open 5",
            "the last open's error"
        );
    }

    #[test]
    fn the_open_loop_restores_the_whole_budget_after_each_stream_that_ran() {
        let mut outcomes = Vec::new();
        for _ in 0..10 {
            outcomes.push(never_armed());
            outcomes.push(lost_after(HEALTHY_STREAM));
        }
        outcomes.extend((0..6).map(|_| never_armed()));
        let (steps, result) = drive(outcomes);
        assert_eq!(steps[..20], [Open; 20], "no reset, no give-up");
        assert_eq!(steps[20..], [Open, Open, Open, Reset, Open, Open, Open]);
        assert!(
            result
                .expect_err("gave up")
                .downcast_ref::<StreamStartupTimeout>()
                .is_some()
        );
    }

    #[test]
    fn the_open_loop_counts_a_stream_lost_soon_after_arming() {
        let short = Duration::from_secs(1);
        let (steps, result) = drive((0..5).map(|_| lost_after(short)).collect());
        assert_eq!(steps, [Open, Open, Reset, Open, Open, Open]);
        assert_eq!(OpenFailure::of(&result.expect_err("gave up")), Lost(short));
    }

    #[test]
    fn the_open_loop_does_not_reset_a_tracker_that_stalls_soon_after_its_prime() {
        // The hardware log: a cold start's prime, a stall 8 s after arming,
        // the re-open's own prime, then a stream until the stop.
        let (steps, result) = drive(vec![
            never_armed(),
            lost_after(Duration::from_secs(8)),
            never_armed(),
            Ok(()),
        ]);
        assert_eq!(steps, [Open; 4], "no reset");
        assert!(result.is_ok(), "a clean stop");
    }

    #[test]
    fn the_open_loop_counts_an_init_failure_after_the_prime_and_primes_no_more() {
        // A cold start's prime, then an init write timeout: the next open that
        // does not arm is the second failure, not a second prime.
        let (steps, result) = drive(vec![
            never_armed(),
            Err(anyhow::anyhow!("init packet 57 write failed")),
            never_armed(),
            Ok(()),
        ]);
        assert_eq!(steps, [Open, Open, Open, Reset, Open]);
        assert!(result.is_ok(), "a clean stop");
    }

    #[test]
    fn the_open_loop_resets_a_tracker_that_arms_and_dies_after_its_second_loss() {
        let short = Duration::from_secs(1);
        let mut outcomes = Vec::new();
        for _ in 0..5 {
            outcomes.push(never_armed());
            outcomes.push(lost_after(short));
        }
        let (steps, result) = drive(outcomes);
        assert_eq!(
            steps,
            [
                Open, Open, Open, Open, Reset, Open, Open, Open, Open, Open, Open
            ]
        );
        assert_eq!(
            OpenFailure::of(&result.expect_err("gave up")),
            Lost(short),
            "the fifth loss"
        );
    }

    #[test]
    fn the_open_loop_resets_once_per_run_of_failures() {
        let (steps, result) = drive(vec![
            never_armed(),
            never_armed(),
            never_armed(),
            lost_after(HEALTHY_STREAM),
            never_armed(),
            never_armed(),
            never_armed(),
            Ok(()),
        ]);
        assert_eq!(
            steps,
            [Open, Open, Open, Reset, Open, Open, Open, Open, Reset, Open]
        );
        assert!(result.is_ok(), "a clean stop");
    }

    #[test]
    fn the_open_loop_without_a_reset_skips_it_and_still_gives_up_after_the_fifth() {
        let (steps, result) = drive_with(
            (1..=5).map(|n| Err(anyhow::anyhow!("open {n}"))).collect(),
            false,
        );
        assert_eq!(steps, [Open; 5], "no reset, nor an open in its place");
        assert_eq!(result.expect_err("gave up").to_string(), "open 5");
    }

    /// An open the tracker refused with `error` at `step`.
    fn refused_at(step: OpenStep, error: rusb::Error) -> Result<()> {
        Err(error).context(step)
    }

    #[test]
    fn the_open_loop_gives_up_on_a_second_open_without_permission_and_resets_nothing() {
        let (steps, result) = drive(vec![
            refused_at(OPEN_DEVICE, rusb::Error::Access),
            refused_at(OPEN_DEVICE, rusb::Error::Access),
        ]);
        assert_eq!(steps, [Open, Open], "no reset, no third open");
        let e = result.expect_err("gave up");
        assert_eq!(e.downcast_ref::<OpenRefusal>(), Some(&NoPermission));
        assert_eq!(
            format!("{e:#}"),
            "no permission to open the tracker; check the udev rule (INSTALL §3) and that \
             this user's session is the active one: \
             failed to open Tobii device; try sudo: Access denied (insufficient permissions)"
        );

        // A tracker just plugged in, or back from the reset, refuses for a
        // moment until udev applies the rule: the next open goes on as usual.
        let (steps, result) = drive(vec![
            refused_at(OPEN_DEVICE, rusb::Error::Access),
            never_armed(),
            Ok(()),
        ]);
        assert_eq!(steps, [Open; 3]);
        assert!(result.is_ok(), "a clean stop");
        let (steps, result) = drive(vec![
            Err(anyhow::anyhow!("open 1")),
            Err(anyhow::anyhow!("open 2")),
            refused_at(OPEN_DEVICE, rusb::Error::Access),
            Ok(()),
        ]);
        assert_eq!(steps, [Open, Open, Reset, Open, Open]);
        assert!(result.is_ok(), "a clean stop");

        // Past a run of failures that already reset the tracker.
        let (steps, result) = drive(vec![
            Err(anyhow::anyhow!("open 1")),
            Err(anyhow::anyhow!("open 2")),
            refused_at(OPEN_DEVICE, rusb::Error::Access),
            refused_at(OPEN_DEVICE, rusb::Error::Access),
        ]);
        assert_eq!(steps, [Open, Open, Reset, Open, Open]);
        assert_eq!(
            result.expect_err("gave up").downcast_ref::<OpenRefusal>(),
            Some(&NoPermission)
        );
    }

    #[test]
    fn the_open_loop_tries_a_busy_interface_three_times_without_a_reset() {
        let (steps, result) = drive(
            (0..3)
                .map(|_| refused_at(CLAIM_INTERFACE, rusb::Error::Busy))
                .collect(),
        );
        assert_eq!(steps, [Open; 3]);
        let e = result.expect_err("gave up");
        assert_eq!(e.downcast_ref::<OpenRefusal>(), Some(&InUse));
        assert_eq!(
            format!("{e:#}"),
            "the tracker is in use by another process: failed to claim interface 0: \
             Resource busy"
        );

        // Released while the loop retries: the next open goes on as usual.
        let (steps, result) = drive(vec![
            refused_at(CLAIM_INTERFACE, rusb::Error::Busy),
            refused_at(CLAIM_INTERFACE, rusb::Error::Busy),
            never_armed(),
            Ok(()),
        ]);
        assert_eq!(steps, [Open; 4]);
        assert!(result.is_ok(), "a clean stop");
    }

    /// Every open gets the `Opens` the ones before it left, across a failed
    /// open, a reset and a refused one: each init counts its open there, and
    /// the area it leaves in effect keeps its generation while it stays.
    #[test]
    fn the_open_loop_hands_every_open_what_the_opens_before_it_left() {
        let stop = AtomicBool::new(false);
        let (b, _) = area_of("change-display-notify-1450");
        let packets = replay_writing(&b, 77);
        // The device took the write: B.
        let took_b = InitCapture {
            responses: vec![answer_1440(0x2a, 1, 0)],
            side: Vec::new(),
        };
        let steps = RefCell::new(Vec::new());
        // What each open was handed: the opens counted, and the generation
        // of the area in effect.
        let mut handed = Vec::new();
        let result = run_opens(
            &stop,
            |opens| {
                steps.borrow_mut().push(Step::Open);
                handed.push((opens.count, opens.display.generation));
                let n = handed.len();
                if n == 3 {
                    // Refused before its init ran: not an open counted.
                    return refused_at(OPEN_DEVICE, rusb::Error::Access);
                }
                let _ = opens.init_done(&packets, &took_b);
                if n == 4 {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("open {n}"))
                }
            },
            Some(|| steps.borrow_mut().push(Step::Reset)),
            Duration::ZERO,
        );

        assert!(result.is_ok(), "a clean stop");
        assert_eq!(steps.into_inner(), [Open, Open, Reset, Open, Open]);
        assert_eq!(
            handed,
            [
                (OpenNumber(0), DisplayGeneration(0)),
                (OpenNumber(1), DisplayGeneration(1)),
                (OpenNumber(2), DisplayGeneration(1)),
                (OpenNumber(2), DisplayGeneration(1)),
            ]
        );
    }

    #[test]
    fn a_stop_during_an_open_ends_the_loop_ok_without_another_open() {
        let stop = AtomicBool::new(false);
        let (mut opens, mut resets) = (0, 0);
        let result = run_opens(
            &stop,
            |_| {
                opens += 1;
                // Stopped during the open whose failure calls for the reset.
                if opens == RESET_AFTER_FAILURES {
                    stop.store(true, Ordering::Relaxed);
                }
                Err(anyhow::anyhow!("open {opens}"))
            },
            Some(|| resets += 1),
            Duration::ZERO,
        );
        assert!(result.is_ok(), "a stop is not a failure: {result:?}");
        assert_eq!((opens, resets), (RESET_AFTER_FAILURES, 0));
    }

    #[test]
    fn a_stop_that_cuts_the_arming_wait_short_ends_the_loop_ok() {
        // The wait gives up on the stop, and the open fails as one that never
        // armed.
        let stop = AtomicBool::new(false);
        let mut opens = 0;
        let result = run_opens(
            &stop,
            |_| {
                opens += 1;
                stop.store(true, Ordering::Relaxed);
                never_armed()
            },
            None::<fn()>,
            Duration::ZERO,
        );
        assert!(result.is_ok(), "a stop is not a failure: {result:?}");
        assert_eq!(opens, 1);
    }

    #[test]
    fn a_stop_before_the_first_open_returns_ok_without_opening() {
        let stop = AtomicBool::new(true);
        let mut opens = 0;
        let result = run_opens(
            &stop,
            |_| {
                opens += 1;
                Ok(())
            },
            Some(|| {}),
            Duration::ZERO,
        );
        assert!(result.is_ok());
        assert_eq!(opens, 0);
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

    /// What a fake EP 0x05 saw, in order.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Seen {
        /// A try, offered this many bytes.
        Try(usize),
        /// The caller's step between two tries.
        Between,
    }

    /// More tries than any test here makes: past it the write has missed
    /// its deadline and the fake fails the test rather than spin.
    const MAX_FAKE_TRIES: usize = 1000;

    /// A fake EP 0x05 for [`write_piece`]: each try returns the next of
    /// `tries` (past the end, a refusal), and one that takes `n` bytes keeps
    /// the first `n` it was offered. A refused try waits its whole timeout,
    /// as libusb does.
    struct FakeOut {
        tries: VecDeque<Result<usize, rusb::Error>>,
        taken: Vec<u8>,
        seen: Vec<Seen>,
    }

    impl FakeOut {
        fn new(tries: impl IntoIterator<Item = Result<usize, rusb::Error>>) -> Self {
            Self {
                tries: tries.into_iter().collect(),
                taken: Vec::new(),
                seen: Vec::new(),
            }
        }

        fn write(&mut self, data: &[u8], timeout: Duration) -> Result<usize, rusb::Error> {
            self.seen.push(Try(data.len()));
            assert!(
                self.seen.len() <= 2 * MAX_FAKE_TRIES,
                "the write went on past its deadline"
            );
            let result = self.tries.pop_front().unwrap_or(Err(rusb::Error::Timeout));
            match result {
                Ok(n) => self.taken.extend_from_slice(&data[..n]),
                Err(rusb::Error::Timeout) => thread::sleep(timeout),
                Err(_) => {}
            }
            result
        }

        fn between(&mut self) {
            self.seen.push(Between);
        }
    }

    /// A 4095-byte piece, as the replay and `chunk_command` write. Its bytes
    /// repeat every 251 (a prime), so no whole number of 512-byte packets
    /// lines it up with itself: a rest resumed at the wrong offset differs.
    fn a_piece() -> Vec<u8> {
        (0..4095u32)
            .map(|i| u8::try_from(i % 251).expect("below 251"))
            .collect()
    }

    /// Short tries and a deadline no test here reaches unless it means to.
    const QUICK_TRIES: WriteLimits = WriteLimits {
        per_try: Duration::from_millis(1),
        deadline: Duration::from_secs(5),
    };

    /// Write `data` to `out` as the callers do, with `limits` and `stop`.
    fn write_to(
        out: &mut FakeOut,
        data: &[u8],
        limits: WriteLimits,
        stop: Option<&AtomicBool>,
    ) -> Result<Duration, WriteError> {
        write_piece(out, data, limits, stop, FakeOut::write, FakeOut::between)
    }

    #[test]
    fn every_write_is_paced_to_fit_its_deadline() {
        for limits in [INIT_WRITE, COMMAND_WRITE, STREAM_STOP_WRITE] {
            assert!(limits.is_paced(), "{limits:?}");
        }
        let zero_try = WriteLimits {
            per_try: Duration::ZERO,
            deadline: WRITE_DEADLINE,
        };
        assert!(!zero_try.is_paced(), "libusb would wait forever");
        let one_try = WriteLimits {
            per_try: WRITE_TRY,
            deadline: WRITE_TRY,
        };
        assert!(!one_try.is_paced(), "nothing left to retry");
    }

    #[test]
    fn a_refused_piece_is_tried_again_after_the_step_between_until_it_is_taken() {
        let piece = a_piece();
        let mut out = FakeOut::new([
            Err(rusb::Error::Timeout),
            Err(rusb::Error::Timeout),
            Ok(piece.len()),
        ]);
        let waited = write_to(&mut out, &piece, QUICK_TRIES, None).expect("written");
        assert!(
            waited >= 2 * QUICK_TRIES.per_try,
            "counted from the first try: {waited:?}"
        );
        assert_eq!(out.taken, piece);
        assert_eq!(
            out.seen,
            [Try(4095), Between, Try(4095), Between, Try(4095)],
            "no step after the try that took it"
        );
    }

    #[test]
    fn a_piece_taken_in_part_is_written_on_from_where_the_device_stopped() {
        // The device takes the first two 512-byte packets, refuses a try,
        // then takes the rest.
        let piece = a_piece();
        let mut out = FakeOut::new([Ok(1024), Err(rusb::Error::Timeout), Ok(3071)]);
        let result = write_to(&mut out, &piece, QUICK_TRIES, None);
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(out.taken, piece, "every byte once, in order");
        assert_eq!(
            out.seen,
            [Try(4095), Between, Try(3071), Between, Try(3071)]
        );

        // Split three ways.
        let mut out = FakeOut::new([Ok(512), Ok(1536), Ok(2047)]);
        let result = write_to(&mut out, &piece, QUICK_TRIES, None);
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(out.taken, piece, "every byte once, in order");
        assert_eq!(
            out.seen,
            [Try(4095), Between, Try(3583), Between, Try(2047)]
        );
    }

    #[test]
    fn a_piece_refused_past_the_deadline_times_out_after_the_step_between() {
        let piece = a_piece();
        let mut out = FakeOut::new([]);
        let limits = WriteLimits {
            per_try: Duration::from_millis(2),
            deadline: Duration::from_millis(100),
        };
        let start = Instant::now();
        let result = write_to(&mut out, &piece, limits, None);
        assert_eq!(result, Err(WriteError::Usb(rusb::Error::Timeout)));
        assert!(start.elapsed() >= limits.deadline);
        assert!(out.taken.is_empty());
        let tries = out.seen.iter().filter(|s| **s == Try(4095)).count();
        assert!(tries >= 2, "tried again before giving up: {:?}", out.seen);
        let expected: Vec<Seen> = (0..tries).flat_map(|_| [Try(4095), Between]).collect();
        assert_eq!(out.seen, expected, "a step after every refused try");
    }

    #[test]
    fn a_stop_ends_a_refused_write_after_the_try_under_way() {
        let piece = a_piece();
        let stop = AtomicBool::new(true);
        let mut out = FakeOut::new([]);
        let start = Instant::now();
        let result = write_to(&mut out, &piece, QUICK_TRIES, Some(&stop));
        assert_eq!(result, Err(WriteError::Stopped));
        assert!(start.elapsed() < Duration::from_secs(1), "not the deadline");
        assert_eq!(out.seen, [Try(4095), Between]);

        // A stop does not cut a piece the device takes.
        let mut out = FakeOut::new([Ok(piece.len())]);
        let result = write_to(&mut out, &piece, QUICK_TRIES, Some(&stop));
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(out.taken, piece);
    }

    #[test]
    fn a_stop_that_comes_while_the_device_refuses_a_piece_ends_the_write() {
        let piece = a_piece();
        let stop = AtomicBool::new(false);
        let mut out = FakeOut::new([]);
        let start = Instant::now();
        let result = write_piece(
            &mut out,
            &piece,
            QUICK_TRIES,
            Some(&stop),
            FakeOut::write,
            |out| {
                out.between();
                if out.seen.iter().filter(|s| **s == Between).count() == 2 {
                    stop.store(true, Ordering::Relaxed);
                }
            },
        );
        assert_eq!(result, Err(WriteError::Stopped));
        assert!(start.elapsed() < Duration::from_secs(1), "not the deadline");
        assert_eq!(out.seen, [Try(4095), Between, Try(4095), Between]);
    }

    #[test]
    fn a_stop_does_not_cut_short_a_piece_the_device_has_begun_to_take() {
        let piece = a_piece();
        let stop = AtomicBool::new(true);
        let mut out = FakeOut::new([Ok(512), Err(rusb::Error::Timeout), Ok(3583)]);
        let result = write_to(&mut out, &piece, QUICK_TRIES, Some(&stop));
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(out.taken, piece);
    }

    #[test]
    fn a_write_error_other_than_a_timeout_ends_the_write_at_once() {
        let piece = a_piece();
        let mut out = FakeOut::new([Err(rusb::Error::NoDevice), Ok(piece.len())]);
        let result = write_to(&mut out, &piece, QUICK_TRIES, None);
        assert_eq!(result, Err(WriteError::Usb(rusb::Error::NoDevice)));
        assert_eq!(out.seen, [Try(4095)], "no step between, no retry");

        // Also after the device took part of the piece.
        let mut out = FakeOut::new([Ok(512), Err(rusb::Error::Pipe)]);
        let result = write_to(&mut out, &piece, QUICK_TRIES, None);
        assert_eq!(result, Err(WriteError::Usb(rusb::Error::Pipe)));
        assert_eq!(out.taken, piece[..512]);
    }

    #[test]
    fn a_stop_ends_the_init_replay_without_error_and_a_failed_write_fails_it() {
        assert_eq!(init_write_error(WriteError::Stopped), None, "not a failure");
        assert_eq!(
            init_write_error(WriteError::Usb(rusb::Error::Timeout)),
            Some(rusb::Error::Timeout)
        );
        assert_eq!(
            init_write_error(WriteError::Usb(rusb::Error::NoDevice)),
            Some(rusb::Error::NoDevice)
        );
    }

    #[test]
    fn a_command_write_that_did_not_finish_answers_its_client() {
        assert_eq!(
            CommandError::from(WriteError::Usb(rusb::Error::Timeout)),
            CommandError::Timeout,
            "a tracker that refused it past the deadline is there, only busy"
        );
        assert_eq!(
            CommandError::from(WriteError::Stopped),
            CommandError::EngineGone
        );
        assert_eq!(
            CommandError::from(WriteError::Usb(rusb::Error::NoDevice)),
            CommandError::Usb(rusb::Error::NoDevice.to_string())
        );
    }
}
