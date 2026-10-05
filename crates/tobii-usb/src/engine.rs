//! Background engine that owns the USB device and produces tracking samples:
//! gaze from the 0x83 gaze stream (~33 Hz), presence from the 0x504 stream,
//! gaze-independent 6DOF head pose from the device's own 280x280 IR image
//! stream (0x50e), which the firmware multiplexes on the same endpoint
//! concurrently, and the device's notifications. Head-pose inference only runs
//! while a client wants a head pose ([`Wanted::head`]); it then makes one
//! [`PoseSample`] of every image it takes: the Stream Engine's head pose,
//! valid or not, and the legacy relative pose while that one is wanted too
//! ([`Wanted::legacy_head`]). The engine's owner says what is wanted with
//! [`Engine::set_wanted`].
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
//! standalone research subcommands (`camera`, `probe`).
//!
//! The `stop`, `recenter`, `paused` and `image_wanted` flags are pure
//! signals (no data is published alongside them), so every access uses
//! `Ordering::Relaxed`. The head flags of [`Wanted`] are kept with how many
//! times each was set, under a lock (see [`Engine::set_wanted`]).
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
//!
//! # Display area
//!
//! The engine keeps the display area in effect on the device as the device
//! confirms it: at each init, the display-area notification (1450) the init
//! brought, else the area the init wrote if the device took it, else the
//! device's answer to the init's read, which comes before the write (on
//! Linux, the 4 x 4 mm square the tracker starts each open with); after
//! that, each 1450. [`Sample::DeviceReady`] reports that area, and every
//! [`ImageSample`] carries its display frame. The area handed to
//! [`Engine::set_display_area_override`] is only what the next inits write.
//!
//! Each gaze frame read after an init is checked against that display
//! frame: the device sends its gaze origins in both the tracker frame and
//! the display frame of the area it holds, and the display frame in effect
//! must turn the one into the other (it does to 0.00016 mm on every frame
//! recorded). When three frames in a row miss by more than 0.01 mm, the
//! device's display frame is not the engine's, and the engine logs a
//! warning, once for each [display
//! generation](ImageSample::display_generation).

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tobii_ipc::deadline;
use tobii_ipc::geometry::{DisplayArea, DisplayFrame};
use tobii_proto::facts::{DeviceFacts, DeviceNotification};
use tobii_proto::gaze83::GazeFrame;
use tobii_proto::image83::ImageFrame;
use tobii_proto::protocol::{RESPONSE_STATUS_OK, ttp_error};
use tracing::error;

use crate::device::OpenRefusal;

/// The Stream Engine's head pose, as [`PoseSample::head`] carries it:
/// re-exported so that the engine's users need not depend on `tobii_pose`.
pub use tobii_pose::head::HeadPose;

/// The head poses made of one 0x50e image: one for every image the engine
/// takes while a head pose is wanted ([`Wanted::head`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PoseSample {
    /// Device timestamp of the source image frame, microseconds (0 for an
    /// image that carries none).
    pub timestamp_us: i64,
    /// The source image's host time ([`ImageSample::host_us`]).
    pub host_us: i64,
    /// The Stream Engine's head pose, valid or not: absolute, in the display
    /// frame of the display area in effect when the image was read
    /// ([`ImageSample::display_frame`]). An invalid pose carries the values
    /// of the last valid one, zeros before the first.
    pub head: HeadPose,
    /// The legacy pose, relative to a rest pose: only while it is wanted
    /// ([`Wanted::legacy_head`]), for an image with a face, once its rest
    /// pose has calibrated.
    pub legacy: Option<LegacyPose>,
}

impl PoseSample {
    /// The poses `head` and `legacy` made from the image of device time
    /// `timestamp_us`, which is `host_us` on the host clock.
    #[must_use]
    pub fn new(
        timestamp_us: i64,
        host_us: i64,
        head: HeadPose,
        legacy: Option<LegacyPose>,
    ) -> Self {
        Self {
            timestamp_us,
            host_us,
            head,
            legacy,
        }
    }
}

/// The legacy head pose, the engine's own from before it made the Stream
/// Engine's, which the daemon keeps publishing for the `OpenTrack` UDP
/// bridge: that of a pivot at the neck, relative to a rest pose that the
/// first fits after a recenter ([`Engine::request_recenter`]) calibrate, its
/// translation along the camera's axes and its angles in the camera's
/// upright frame, clamped to ±45°, the whole of it smoothed (see
/// `tobii_pose::track::RestPose`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct LegacyPose {
    /// Translation `[tx, ty, tz]` from the rest pose, centimetres.
    pub pos_cm: [f64; 3],
    /// Rotation `[yaw, pitch, roll]` from the rest pose, degrees.
    pub rot_deg: [f64; 3],
}

impl LegacyPose {
    /// A pose of translation `pos_cm` and rotation `rot_deg`
    /// (`[yaw, pitch, roll]`).
    #[must_use]
    pub fn new(pos_cm: [f64; 3], rot_deg: [f64; 3]) -> Self {
        Self { pos_cm, rot_deg }
    }

    /// The pose `RestPose::update` returns: `[TX, TY, TZ (cm), Yaw, Pitch,
    /// Roll (deg)]`, the `OpenTrack` order.
    #[must_use]
    pub(crate) fn of_open_track(pose: [f64; 6]) -> Self {
        let [tx, ty, tz, yaw, pitch, roll] = pose;
        Self::new([tx, ty, tz], [yaw, pitch, roll])
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

/// Which of the engine's opens of the tracker read something
/// ([`ImageSample::open`]): the engine counts from 1 each open whose init
/// ran to its end, across re-opens and USB resets. What one open reads does
/// not follow on from what the open before it read: an open starts the
/// tracker over, its clock included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct OpenNumber(pub(crate) u64);

impl OpenNumber {
    /// The number, from 1; 0 for none of the engine's opens.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The number of the open after this one.
    #[must_use]
    pub(crate) const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

/// Which display area the device confirmed was in effect when something
/// was read ([`ImageSample::display_generation`]): how many times the area
/// in effect had changed since the engine started, so that all that one
/// generation covers shares one area. 0 before the device first confirmed
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DisplayGeneration(pub(crate) u64);

impl DisplayGeneration {
    /// How many times the area in effect had changed.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The generation after this one.
    #[must_use]
    pub(crate) const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

/// One 0x50e IR frame, with the display area in effect on the device when
/// it was read and the open that read it.
///
/// The display frame, its generation and the open are for the head pose the
/// Stream Engine reports ([`PoseSample::head`]), which is in the display
/// frame: a new generation or open restarts its filters.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImageSample {
    /// The frame, device timestamp included (shared with the pose worker).
    pub frame: Arc<ImageFrame>,
    /// The frame's device timestamp on the host clock, microseconds (see
    /// [Timestamps](crate::engine#timestamps)).
    pub host_us: i64,
    /// The display frame of the display area in effect on the device when
    /// the frame was read, as the device confirmed that area: the one the
    /// open's init left in effect, or the one a display-area notification
    /// gave since. Never the area handed to
    /// [`Engine::set_display_area_override`], which only the next init
    /// writes. `None` while the device has confirmed no area, or one that
    /// fixes no frame (see [`DisplayFrame::new`]). Built once for each
    /// [display generation](Self::display_generation) and shared by all
    /// its images.
    pub display_frame: Option<Arc<DisplayFrame>>,
    /// How many times that display area had changed since the engine
    /// started: frames of one generation share one area. 0 before the
    /// device first confirmed one.
    pub display_generation: DisplayGeneration,
    /// Which of the engine's opens of the tracker read the frame, from 1. A
    /// frame of another open does not follow on from the ones before it: an
    /// open starts the tracker over, its clock included.
    pub open: OpenNumber,
}

impl ImageSample {
    /// A sample of `frame`, whose device timestamp is `host_us` on the host
    /// clock, with no display frame, display generation 0 and open 0 (no
    /// open the engine makes is 0).
    #[must_use]
    pub fn new(frame: Arc<ImageFrame>, host_us: i64) -> Self {
        Self {
            frame,
            host_us,
            display_frame: None,
            display_generation: DisplayGeneration::default(),
            open: OpenNumber::default(),
        }
    }
}

// An image sample carries its frame and its display frame by pointer: five
// words. The display frame inline (104 bytes) would make it the largest
// sample by far, and so every sample larger.
const _: () = assert!(size_of::<ImageSample>() <= 5 * size_of::<u64>());

/// One item out of the engine.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Sample {
    /// The head poses of an image, one for every image while a head pose is
    /// wanted (see [`Wanted::head`]); boxed: with its two poses it is 1.6
    /// times the size of the largest other sample.
    Pose(Box<PoseSample>),
    /// A gaze frame (boxed: it is ten times the size of the other samples).
    Gaze(Box<GazeSample>),
    /// A presence change.
    Presence(PresenceSample),
    /// An IR frame (only while images are wanted, see [`Wanted::image`]).
    Image(ImageSample),
    /// A device notification.
    Notification(DeviceNotification),
    /// The device finished its init; what it reported about itself, the
    /// display area it confirmed included (see [Display
    /// area](crate::engine#display-area)).
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
    /// timeout, plus up to [`deadline::QUEUE_ALLOWANCE`] spent queued and
    /// writing it (see [`deadline::command`]), or the device refuses the write
    /// past its deadline, [`CommandError::EngineGone`] when the engine stops
    /// first, [`CommandError::Usb`] when the write fails otherwise.
    pub fn run(&self, command: DeviceCommand) -> Result<CommandResponse, CommandError> {
        let deadline = deadline::command(command.timeout);
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

/// The optional work the engine does, as its owner asks for it with
/// [`Engine::set_wanted`]: what some client consumes. Gaze, presence and the
/// device's notifications are made whatever is wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Wanted {
    /// A head pose, of either kind: the engine runs head-pose inference on
    /// the image stream (it costs CPU) and makes a [`PoseSample`] of every
    /// image it takes, the Stream Engine's pose valid or not; it drops the
    /// images otherwise.
    pub head: bool,
    /// The legacy pose too ([`PoseSample::legacy`]), of the images a head
    /// pose is made of: nothing without [`Wanted::head`].
    pub legacy_head: bool,
    /// [`Sample::Image`] for every IR frame.
    pub image: bool,
}

/// One head flag of [`Wanted`] as the pose worker reads it: whether it is
/// set, and how many times it has been set when it was not. The worker
/// reads the flags once an image, and sees from the count that a client
/// subscribed since the image before even when the flag was cleared and set
/// again between the two, or while no image came (a pause, a re-open).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct HeadFlag {
    /// Whether the flag is set.
    pub(crate) on: bool,
    /// How many times it has been set when it was not.
    pub(crate) raised: u64,
}

impl HeadFlag {
    /// Set the flag `on` or clear it, counting a raise.
    pub(crate) fn set(&mut self, on: bool) {
        if on && !self.on {
            self.raised = self.raised.wrapping_add(1);
        }
        self.on = on;
    }

    /// Whether the flag is set, and was raised since it was `before`.
    #[must_use]
    pub(crate) fn is_raised_since(self, before: Self) -> bool {
        self.on && self.raised != before.raised
    }
}

/// The head flags of [`Wanted`], as the pose worker reads them (see
/// [`HeadFlag`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct HeadFlags {
    /// [`Wanted::head`].
    pub(crate) head: HeadFlag,
    /// [`Wanted::legacy_head`].
    pub(crate) legacy: HeadFlag,
}

/// State shared between the [`Engine`] handle and its USB thread.
#[derive(Debug, Default)]
pub(crate) struct Shared {
    pub(crate) stop: AtomicBool,
    pub(crate) recenter: AtomicBool,
    /// The head flags of what is wanted, which the pose worker reads (see
    /// [`Shared::heads`]): both under one lock, so that it never reads one
    /// changed and the other not yet.
    heads: Mutex<HeadFlags>,
    pub(crate) image_wanted: AtomicBool,
    /// The device was told to pause: its streams are expected to stop.
    pub(crate) paused: AtomicBool,
    /// Display area to write in place of the one in the init replay.
    pub(crate) display_override: Mutex<Option<DisplayArea>>,
}

impl Shared {
    /// Keep `wanted` in the flags the USB thread and the pose worker read
    /// (see [`Engine::set_wanted`]).
    pub(crate) fn set_wanted(&self, wanted: Wanted) {
        {
            // Each flag is written whole and nothing under the lock can
            // panic: a poisoned lock holds no half-updated flags.
            let mut heads = self.heads.lock().unwrap_or_else(PoisonError::into_inner);
            heads.head.set(wanted.head);
            heads.legacy.set(wanted.legacy_head);
        }
        // Relaxed: a pure signal, no data is published with it.
        self.image_wanted.store(wanted.image, Ordering::Relaxed);
    }

    /// The head flags as they are now.
    pub(crate) fn heads(&self) -> HeadFlags {
        *self.heads.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The display area the next init writes, if not the replay's own.
    pub(crate) fn display_override(&self) -> Option<DisplayArea> {
        // Written whole; a poisoned lock cannot hold a torn value.
        *self
            .display_override
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Have the next inits write `area` (see
    /// [`Engine::set_display_area_override`]).
    pub(crate) fn set_display_override(&self, area: Option<DisplayArea>) {
        *self
            .display_override
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = area;
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
    /// The device thread; it hands back why it gave up when the tracker
    /// refused to be opened (see [`Engine::finish`]).
    handle: Option<JoinHandle<Option<OpenRefusal>>>,
}

impl Engine {
    /// Spawn the device thread and start streaming. Failures inside the thread
    /// are logged once; [`Engine::is_alive`] turns false when it has exited,
    /// and [`Engine::finish`] then says whether the tracker refused to be
    /// opened.
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
            ended(crate::device::run_gaze_engine(&thread_shared, &cmd_rx, &tx))
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

    /// Have the legacy pose's rest pose ([`PoseSample::legacy`]) calibrate
    /// afresh from the next images with a face. The Stream Engine's head
    /// pose is absolute: a recenter leaves it alone.
    pub fn request_recenter(&self) {
        // Relaxed: a pure signal, no data is published with it.
        self.shared.recenter.store(true, Ordering::Relaxed);
    }

    /// Do the optional work `wanted` asks for, and none that it does not:
    /// head-pose inference, the legacy pose, the IR images (see [`Wanted`]).
    ///
    /// The pose worker reads the head flags at each image it takes, with how
    /// many times each was set when it was not. The first image after a
    /// head pose was wanted anew starts the poses over: the Stream Engine's
    /// restarts its filters, its invalid poses carrying zeros until the
    /// next valid one, and the legacy pose's rest pose calibrates afresh.
    /// The first image after the legacy pose alone was wanted anew has the
    /// rest pose calibrate afresh, the Stream Engine's pose going on as it
    /// was. Wanted anew is since the image before, even if the flag was
    /// cleared and set again between the two or while no image came (a
    /// pause, a re-open); not at the worker's first image, whose poses are
    /// new. So a new subscriber does not inherit an earlier one's poses,
    /// unless a head pose stays wanted throughout: [`Wanted::head`] covers
    /// both kinds, and while it stays set the Stream Engine's pose goes on
    /// without a restart, so a client that comes to want it while another
    /// keeps the legacy pose going gets its filters as they are.
    pub fn set_wanted(&self, wanted: Wanted) {
        self.shared.set_wanted(wanted);
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
    /// replay would otherwise restore the one it embeds); `None` goes back
    /// to the replay's own. This is only what the next inits write: the
    /// area in effect, which the device's facts and the image samples carry
    /// ([`ImageSample::display_frame`]), is the one the device confirms.
    pub fn set_display_area_override(&self, area: Option<DisplayArea>) {
        self.shared.set_display_override(area);
    }

    /// Is the device thread still running (false once it has exited/failed)?
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Stop the engine as dropping it does, and say why it gave up if the
    /// tracker refused to be opened: `None` for any other end, a stop
    /// included. It waits for the device thread, which is at once for an
    /// engine no longer [alive](Engine::is_alive).
    #[must_use]
    pub fn finish(mut self) -> Option<OpenRefusal> {
        self.stop_and_join()
    }

    /// Tell the device thread to stop and wait for it; what it handed back,
    /// the first time (see [`Engine::finish`]).
    fn stop_and_join(&mut self) -> Option<OpenRefusal> {
        // Relaxed: a pure signal; the join below is the synchronisation point.
        self.shared.stop.store(true, Ordering::Relaxed);
        // A panicked device thread has already logged; nothing to recover.
        self.handle.take().and_then(|h| h.join().ok().flatten())
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
        let _ = self.stop_and_join();
    }
}

/// Log how the device thread's `result` ended, once, and hand back why it
/// gave up if the tracker refused to be opened (an [`OpenRefusal`] context,
/// see [`crate::device::run_gaze_engine`]).
fn ended(result: anyhow::Result<()>) -> Option<OpenRefusal> {
    let e = result.err()?;
    error!(error = %format_args!("{e:#}"), "tobii engine stopped");
    e.downcast_ref::<OpenRefusal>().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_engine_given_up_on_a_refusal_says_why() {
        let e = || anyhow::anyhow!("open 5");
        assert_eq!(
            ended(Err(e().context(OpenRefusal::NoPermission))),
            Some(OpenRefusal::NoPermission)
        );
        assert_eq!(
            ended(Err(e().context(OpenRefusal::InUse).context("engine"))),
            Some(OpenRefusal::InUse),
            "wrapped further"
        );
        assert_eq!(ended(Err(e())), None);
        assert_eq!(ended(Ok(())), None, "a stop");
    }

    /// The rest pose's `OpenTrack` order, [TX, TY, TZ, Yaw, Pitch, Roll],
    /// splits into the translation and the rotation as the legacy pose has
    /// them, the yaw first.
    #[test]
    fn a_legacy_pose_takes_the_rest_poses_open_track_order() {
        let pose = LegacyPose::of_open_track([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            (pose.pos_cm, pose.rot_deg),
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        );
    }
}
