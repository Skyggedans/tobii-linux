//! Background engine that owns the USB device and produces tracking samples:
//! gaze from the 0x83 gaze stream (~33 Hz), presence from the 0x504 stream,
//! gaze-independent 6DOF head pose from the device's own 280x280 IR image
//! stream (0x50e), which the firmware multiplexes on the same endpoint
//! concurrently, and the device's notifications. Head-pose inference only runs
//! while a client wants it (`set_head_wanted`).
//!
//! Besides streaming, the engine runs device commands for its owner
//! ([`Engine::commands`]): they are queued to the USB thread, which writes
//! them between reads and matches the device's answer by sequence number, so
//! the streams keep flowing while a slow command (a calibration point takes
//! most of a second) is outstanding.
//!
//! The UVC camera (interface 2) is never used by the engine: streaming it
//! throttles the 0x83 streams from ~33 Hz to <1 Hz (firmware mode, not
//! bandwidth — verified with `probe`). The UVC path survives only in the
//! standalone research subcommands (`camera`, `track`, `probe`).
//!
//! The `stop` / `recenter` / `head_wanted` / `image_wanted` / `paused` flags
//! are pure signals (no data is published alongside them), so every access
//! uses `Ordering::Relaxed`.
//!
//! # Timestamps
//!
//! Every sample keeps the device's timestamp (the tracker's clock, in
//! microseconds) and also gives it on the host clock
//! ([`tobii_ipc::host_clock_us`]) as `host_us`. The engine maps the one to the
//! other with the smallest `host_rx - device` over the last 120 s of gaze and
//! image arrivals, which follows the drift between the clocks and starts
//! afresh with each open of the device (a new open may restart the tracker's
//! clock). A host time strictly increases within each stream and is no later
//! than the host read its message, except for a message read during the
//! device's init: delivered before any arrival of its open, it gets the time
//! it was delivered, a few ms after it was read. (The order may also pass the
//! read time by a microsecond when two messages of a stream share it, as the
//! messages of one transfer do.) The messages read while the stream arms,
//! delivered once it has, map from all the arrivals of that wait, the gaze
//! frame that ends it included.
//! A message read with no arrival in the 120 s before it (presence as a long
//! pause ends), and an image without a device timestamp, get the time they
//! were read. A head pose has the host time of the image it was made from.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tobii_ipc::geometry::DisplayArea;
use tobii_proto::facts::{DeviceFacts, DeviceNotification};
use tobii_proto::gaze83::GazeFrame;
use tobii_proto::image83::ImageFrame;
use tobii_proto::protocol::{RESPONSE_STATUS_OK, ttp_error};
use tracing::error;

/// One 6DOF head-pose estimate from the IR image stream.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PoseSample {
    /// Device timestamp of the source image frame, microseconds.
    pub timestamp_us: i64,
    /// The source image's host time ([`ImageSample::host_us`]).
    pub host_us: i64,
    /// Head translation `[tx, ty, tz]` in centimetres.
    pub pos_cm: [f64; 3],
    /// Head rotation `[yaw, pitch, roll]` in degrees.
    pub rot_deg: [f64; 3],
}

impl PoseSample {
    /// A pose made from the image of device time `timestamp_us`, which is
    /// `host_us` on the host clock.
    #[must_use]
    pub fn new(timestamp_us: i64, host_us: i64, pos_cm: [f64; 3], rot_deg: [f64; 3]) -> Self {
        Self {
            timestamp_us,
            host_us,
            pos_cm,
            rot_deg,
        }
    }
}

/// One decoded 0x500 gaze frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GazeSample {
    /// Everything the frame carries, device timestamp included.
    pub frame: GazeFrame,
    /// Host time when the frame was read, microseconds
    /// ([`tobii_ipc::host_clock_us`]).
    pub host_rx_us: i64,
    /// The frame's device timestamp on the host clock, microseconds (see
    /// [Timestamps](crate::engine#timestamps)).
    pub host_us: i64,
}

impl GazeSample {
    /// A sample of `frame` read at host time `host_rx_us`, whose device
    /// timestamp is `host_us` on the host clock.
    #[must_use]
    pub fn new(frame: GazeFrame, host_rx_us: i64, host_us: i64) -> Self {
        Self {
            frame,
            host_rx_us,
            host_us,
        }
    }
}

/// A presence change from the 0x504 stream (sent at stream start and on
/// change only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PresenceSample {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// The device timestamp on the host clock, microseconds (see
    /// [Timestamps](crate::engine#timestamps)).
    pub host_us: i64,
    /// Whether a user is in front of the tracker.
    pub present: bool,
}

impl PresenceSample {
    /// Presence `present` at device time `timestamp_us`, which is `host_us`
    /// on the host clock.
    #[must_use]
    pub fn new(timestamp_us: i64, host_us: i64, present: bool) -> Self {
        Self {
            timestamp_us,
            host_us,
            present,
        }
    }
}

/// One 0x50e IR frame.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImageSample {
    /// The frame, device timestamp included (shared with the pose worker).
    pub frame: Arc<ImageFrame>,
    /// The frame's device timestamp on the host clock, microseconds (see
    /// [Timestamps](crate::engine#timestamps)).
    pub host_us: i64,
}

impl ImageSample {
    /// A sample of `frame`, whose device timestamp is `host_us` on the host
    /// clock.
    #[must_use]
    pub fn new(frame: Arc<ImageFrame>, host_us: i64) -> Self {
        Self { frame, host_us }
    }
}

/// One item out of the engine.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Sample {
    /// Head pose (only while head pose is wanted, see [`Engine::set_head_wanted`]).
    Pose(PoseSample),
    /// A gaze frame (boxed: it is ten times the size of the other samples).
    Gaze(Box<GazeSample>),
    /// A presence change.
    Presence(PresenceSample),
    /// An IR frame (only while images are wanted, see [`Engine::set_image_wanted`]).
    Image(ImageSample),
    /// A device notification.
    Notification(DeviceNotification),
    /// The device finished its init; what it reported about itself.
    DeviceReady(Arc<DeviceFacts>),
}

/// A command for the device: id and TLV payload (empty for a parameterless
/// command), plus how long the device may take to answer once it is sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCommand {
    /// Command id.
    pub cmd: u32,
    /// Payload (`00 00` + TLVs), or empty.
    pub payload: Vec<u8>,
    /// Answer deadline, counted from the end of the write (which waits out
    /// a tracker that refuses it for a while).
    pub timeout: Duration,
}

/// The device's answer to a [`DeviceCommand`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResponse {
    /// The response's status word (1 on every answer ever captured).
    pub status: u32,
    /// The response's TTP error code (0 on every answer ever captured; see
    /// [`tobii_proto::protocol::ttp_error`]).
    pub error: u32,
    /// The whole payload, chunked responses joined.
    pub payload: Vec<u8>,
}

impl CommandResponse {
    /// A successful answer carrying `payload`.
    #[must_use]
    pub fn ok(payload: Vec<u8>) -> Self {
        Self {
            status: RESPONSE_STATUS_OK,
            error: ttp_error::NONE,
            payload,
        }
    }
}

/// Why a command got no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommandError {
    /// The device did not answer in time, or refused the command's write
    /// past the write deadline (it is there, but busy).
    Timeout,
    /// The engine stopped (device lost or shutting down).
    EngineGone,
    /// The command could not be written (a USB error).
    Usb(String),
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("device did not answer in time"),
            Self::EngineGone => f.write_str("engine stopped"),
            Self::Usb(e) => write!(f, "USB write failed: {e}"),
        }
    }
}

impl std::error::Error for CommandError {}

/// A command waiting for the USB thread, with where to send its answer.
#[derive(Debug)]
pub(crate) struct QueuedCommand {
    pub(crate) command: DeviceCommand,
    pub(crate) reply: Sender<Result<CommandResponse, CommandError>>,
}

/// How long a command may wait for the USB thread to reach it (queued behind
/// others, or the device still initialising) and to write it (a tracker
/// starting its sensor refuses writes for up to about 3 s; its sensor start
/// takes 3.6 s) before its own deadline starts.
const QUEUE_ALLOWANCE: Duration = Duration::from_secs(30);

/// Runs commands on the engine's device; cheap to clone and usable from any
/// thread.
#[derive(Debug, Clone)]
pub struct Commands(Sender<QueuedCommand>);

impl Commands {
    /// Send `command` and block until the device answers.
    ///
    /// # Errors
    ///
    /// [`CommandError::Timeout`] when no answer arrives within the command's
    /// timeout (plus the time spent queued and writing it) or the device
    /// refuses the write past its deadline, [`CommandError::EngineGone`] when
    /// the engine stops first, [`CommandError::Usb`] when the write fails
    /// otherwise.
    pub fn run(&self, command: DeviceCommand) -> Result<CommandResponse, CommandError> {
        let deadline = command.timeout + QUEUE_ALLOWANCE;
        let (reply, answer) = mpsc::channel();
        self.0
            .send(QueuedCommand { command, reply })
            .map_err(|_| CommandError::EngineGone)?;
        match answer.recv_timeout(deadline) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(CommandError::Timeout),
            Err(RecvTimeoutError::Disconnected) => Err(CommandError::EngineGone),
        }
    }
}

/// State shared between the [`Engine`] handle and its USB thread.
#[derive(Debug, Default)]
pub(crate) struct Shared {
    pub(crate) stop: AtomicBool,
    pub(crate) recenter: AtomicBool,
    pub(crate) head_wanted: AtomicBool,
    pub(crate) image_wanted: AtomicBool,
    /// The device was told to pause: its streams are expected to stop.
    pub(crate) paused: AtomicBool,
    /// Display area to write in place of the one in the init replay.
    pub(crate) display_override: Mutex<Option<DisplayArea>>,
}

impl Shared {
    pub(crate) fn display_override(&self) -> Option<DisplayArea> {
        // Written whole; a poisoned lock cannot hold a torn value.
        *self
            .display_override
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Owns the device thread. The daemon drives `wait`/`drain` from a single
/// consumer thread (the fan-out pump), so `&mut self` access is sufficient
/// and lock-free.
#[derive(Debug)]
pub struct Engine {
    shared: Arc<Shared>,
    commands: Commands,
    rx: Receiver<Sample>,
    pending: VecDeque<Sample>,
    handle: Option<JoinHandle<()>>,
}

impl Engine {
    /// Spawn the device thread and start streaming. Failures inside the thread
    /// are logged once; [`Engine::is_alive`] turns false when it has exited.
    #[must_use]
    pub fn start() -> Self {
        Self::start_with(None)
    }

    /// Like [`Engine::start`], writing `display_area` instead of the display
    /// area embedded in the init replay.
    #[must_use]
    pub fn start_with(display_area: Option<DisplayArea>) -> Self {
        let shared = Arc::new(Shared {
            display_override: Mutex::new(display_area),
            ..Shared::default()
        });
        let (tx, rx) = mpsc::channel();
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let thread_shared = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            if let Err(e) = crate::device::run_gaze_engine(&thread_shared, &cmd_rx, &tx) {
                error!(error = %format_args!("{e:#}"), "tobii engine stopped");
            }
        });
        Engine {
            shared,
            commands: Commands(cmd_tx),
            rx,
            pending: VecDeque::new(),
            handle: Some(handle),
        }
    }

    /// A handle for running device commands.
    #[must_use]
    pub fn commands(&self) -> Commands {
        self.commands.clone()
    }

    /// Ask the head tracker to recalibrate its rest pose on the next frames.
    pub fn request_recenter(&self) {
        // Relaxed: a pure signal, no data is published with it.
        self.shared.recenter.store(true, Ordering::Relaxed);
    }

    /// Run head-pose inference on the image stream (costs CPU) while some
    /// client consumes head pose; frames are dropped otherwise.
    pub fn set_head_wanted(&self, wanted: bool) {
        // Relaxed: a pure signal, no data is published with it.
        self.shared.head_wanted.store(wanted, Ordering::Relaxed);
    }

    /// Emit [`Sample::Image`] for every IR frame while some client wants them.
    pub fn set_image_wanted(&self, wanted: bool) {
        // Relaxed: a pure signal, no data is published with it.
        self.shared.image_wanted.store(wanted, Ordering::Relaxed);
    }

    /// Tell the engine whether the device is paused. A paused device sends no
    /// gaze, so the engine does not take the silence for a dead stream; after
    /// a resume it gives the stream a longer grace to come back. The owner
    /// sets this before it sends the pause and clears it once the device is
    /// resumed (or given up on: a device still paused is then re-opened,
    /// which resumes it).
    pub fn set_paused(&self, paused: bool) {
        // Relaxed: a pure signal, no data is published with it.
        self.shared.paused.store(paused, Ordering::Relaxed);
    }

    /// Keep `area` as the display area across device re-inits (the init
    /// replay would otherwise restore the one it embeds).
    pub fn set_display_area_override(&self, area: Option<DisplayArea>) {
        *self
            .shared
            .display_override
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = area;
    }

    /// Is the device thread still running (false once it has exited/failed)?
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Block up to `timeout` until at least one sample is available.
    pub fn wait(&mut self, timeout: Duration) -> bool {
        if !self.pending.is_empty() {
            return true;
        }
        match self.rx.recv_timeout(timeout) {
            Ok(sample) => {
                self.pending.push_back(sample);
                true
            }
            Err(_) => false,
        }
    }

    /// Take all queued samples (for the callback pump).
    pub fn drain(&mut self) -> Vec<Sample> {
        let mut out = Vec::new();
        self.drain_into(&mut out);
        out
    }

    /// Append all queued samples to `out`, reusing its allocation (for the
    /// per-tick fan-out loop).
    pub fn drain_into(&mut self, out: &mut Vec<Sample>) {
        while let Ok(sample) = self.rx.try_recv() {
            self.pending.push_back(sample);
        }
        out.extend(self.pending.drain(..));
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Relaxed: a pure signal; the join below is the synchronisation point.
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // A panicked device thread has already logged; nothing to recover.
            let _ = h.join();
        }
    }
}
