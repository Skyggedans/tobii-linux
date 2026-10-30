//! Background engine that owns the USB device and produces tracking samples:
//! gaze from the 0x83 gaze stream (~33 Hz), presence from the 0x504 stream,
//! gaze-independent 6DOF head pose from the device's own 280x280 IR image
//! stream (0x50e), which the firmware multiplexes on the same endpoint
//! concurrently, and the device's notifications. Head-pose inference only runs
//! while a client wants a head pose ([`Wanted::head`]); it then makes one
//! [`PoseSample`] of every image it takes, the Stream Engine's head pose,
//! valid or not. The engine's owner says what is wanted with
//! [`Engine::set_wanted`].
//!
//! Besides streaming, the engine runs device commands for its owner
//! ([`Engine::commands`]): they are queued to the USB thread, which writes
//! them between reads and matches the device's answer by sequence number, so
//! the streams keep flowing while a slow command (a calibration point takes
//! most of a second) is outstanding. It takes them once an open's gaze stream
//! has armed. It also starts and stops the tracker's catalogue streams for
//! its owner, and keeps them running across re-opens (see [Catalogue
//! streams](crate::engine#catalogue-streams)).
//!
//! The UVC camera (interface 2) is never used by the engine: streaming it
//! throttles the 0x83 streams from ~33 Hz to <1 Hz (firmware mode, not
//! bandwidth — verified with `probe`). The UVC path survives only in the
//! standalone research subcommands (`camera`, `probe`).
//!
//! The `stop`, `paused` and `image_wanted` flags are pure signals (no data is
//! published alongside them), so every access uses `Ordering::Relaxed`. The
//! head flag of [`Wanted`] is kept with how many times it was set, under a
//! lock (see [`Engine::set_wanted`]).
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
//! A frame of the custom stream carries no device timestamp: its sample has
//! the time it was read, and the map learns nothing from it.
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
//!
//! # Stream columns
//!
//! Each init, the prime's included, also asks the tracker the columns of its
//! streams (command 1210, which the Stream Engine's
//! `tobii_enumerate_stream_type_columns` sends): of each stream the init's
//! catalogue (command 1200) lists that the Stream Engine can ask about (see
//! [`tobii_proto::facts::column_ids`]), or the last catalogue an earlier init
//! got when this one lost its own. It asks right after the init's own
//! commands, one stream at a time, and gives up at the first stream not
//! answered within 500 ms; nothing of it fails the open.
//! [`Sample::DeviceReady`] carries the answers in its facts' `stream_columns`:
//! the columns of each stream answered, `None` for an answer the tracker
//! refused or that does not parse, and nothing for a stream not answered.
//!
//! # Custom stream
//!
//! The 0x1770 `algodbg` stream is the Stream Engine's custom stream 0: the
//! DLL starts it when a client subscribes to that stream
//! (`tobii_custom_stream_subscribe`; command 1220 from 0x180160810) and hands
//! the subscriber each frame's key 1 (`process_custom_stream`, 0x180189920).
//! Once the gaze stream has armed, each 0x1770 frame the engine reads becomes
//! a [`Sample::Custom`], with the time it was read and the data the DLL
//! reads in it ([`tobii_proto::custom83`]); a frame the DLL drops becomes
//! none. The first of each open is logged. The engine starts the stream when
//! its owner asks (see [Catalogue streams](crate::engine#catalogue-streams)).
//!
//! Neither the messages an init reads besides its answers nor those read
//! while the gaze stream arms, which the engine keeps to deliver later,
//! include a frame of 0x1770 or of 0x508 (image collection, which the ET5
//! refuses to start: none was ever seen). A 0x1770 left running, by a
//! process that started it and ended without stopping it, sends some 133
//! frames a second: kept, they would fill those two buffers of 64 messages
//! within half a second and keep out what the buffers are for, the
//! display-area notification (1450) that decides the init's area, presence
//! at stream start and the device's notifications.
//!
//! # Catalogue streams
//!
//! Two streams of the tracker's catalogue (command 1200) are the Stream
//! Engine's to start for a client who subscribes to one: 0x1770, its custom
//! stream 0 (`tobii_custom_stream_subscribe`, whose tracker module sends the
//! 1220 from 0x180160810), and 0x508, image collection
//! (`tobii_image_collection_subscribe`, from 0x180160210). The engine starts
//! and stops them when its owner asks, with [`Commands::change_stream`]
//! (commands 1220 and 1230), and starts or stops no other stream for anyone:
//! a change of any other stream, the engine's own 0x500, 0x504 and 0x50e
//! among them, is [`CommandError::NotSupported`] at once, with nothing
//! queued, and so is a 1220 or 1230 handed to [`Commands::run`]. A change of
//! a stream that the last catalogue an init got does not list is answered
//! [`CommandError::NotSupported`] too, with nothing written, once the USB
//! thread takes it (after an open has armed): the DLL sends nothing for a
//! stream its own catalogue lacks (0x1801608b0, 0x1801602a8).
//!
//! The engine keeps the streams it is to run: each whose start the tracker
//! took (an answer without an error, whatever its status word: the DLL reads
//! the error word alone, which 0x18017b4c0 maps), and those
//! [`Engine::start_seeded`] was given, until a stop. A stop takes its stream
//! out of them before its 1230 is written, and its 1230 is always written:
//! the tracker takes a 1230 for a stream that does not run, and a start it
//! did not answer in time may have started one. A start of a stream that
//! runs is answered OK, and the stream kept, with nothing written: what the
//! tracker does with a second 1220 is not known. A start the tracker refused
//! leaves the streams kept as they were.
//!
//! What is kept outlasts the open, across the re-opens and USB resets of the
//! engine: each open starts the kept streams again (a 1220 each, its answer
//! due within 1.5 s, as the 0x50e start's) once its gaze stream has armed,
//! before any command queued meanwhile. The prime of a cold start, which
//! does not arm, starts none. A restart the tracker refuses is warned of,
//! once in the open, and the stream stays kept: a tracker that kept the
//! stream running across the re-open may refuse to start it again. A kept
//! stream the catalogue does not list is warned of, and not started.
//!
//! Each open ends by stopping each catalogue stream it started, unless the
//! tracker has answered a stop of it since, and each it tried to start or to
//! stop without an answer, even if the tracker refused a later start of it:
//! a 1230 each, best-effort, and none after the first that fails, as the
//! engine's owner may be waiting on its end (tobiid drops an engine under its
//! state lock). Then come the 1230 of 0x50e and the vendor control 66. A stop
//! outstanding as the open ends is so sent twice, which the tracker takes as
//! it takes any 1230. A stream whose only start in the open the tracker
//! refused, a refused restart's among them, is not stopped, though that
//! refusal may mean the stream runs still; its owner's stop writes its 1230
//! all the same.
//!
//! Not known on the ET5: whether the vendor control 66 that ends each open
//! stops 0x1770, whether the vendor init of the next open does, and whether
//! a pause (command 3100) does; a stream a pause stopped would not come back
//! with the resume, which starts nothing. An open that ends without its
//! stops, as a killed process's does, may leave 0x1770 running into the next
//! (see [Custom stream](crate::engine#custom-stream)).

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tobii_ipc::deadline;
use tobii_ipc::geometry::{DisplayArea, DisplayFrame};
use tobii_proto::facts::{DeviceFacts, DeviceNotification};
use tobii_proto::gaze83::GazeFrame;
use tobii_proto::image83::ImageFrame;
use tobii_proto::protocol::{
    RESPONSE_STATUS_OK, cmd, stream_id_payload, stream_start_payload, ttp_error,
};
use tracing::{error, warn};

use crate::device::{CATALOGUE_STREAMS, OpenRefusal};

/// The Stream Engine's head pose, as [`PoseSample::head`] carries it:
/// re-exported so that the engine's users need not depend on `tobii_pose`.
#[cfg(feature = "head-pose")]
pub use tobii_pose::head::HeadPose;

/// Stand-in for [`tobii_pose::head::HeadPose`] without the `head-pose`
/// feature, so [`PoseSample`] and [`Sample::Pose`] still compile: a gaze-only
/// build never produces one of these, as nothing starts the pose worker that
/// would.
#[cfg(not(feature = "head-pose"))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadPose {
    /// Always `false` in this build.
    pub valid: bool,
    /// Always zero in this build.
    pub position_mm: [f64; 3],
    /// Always zero in this build.
    pub rotation_rad: [f64; 3],
}

/// The head pose made of one 0x50e image: one for every image the engine
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
}

impl PoseSample {
    /// The pose `head` made from the image of device time `timestamp_us`,
    /// which is `host_us` on the host clock.
    #[must_use]
    pub fn new(timestamp_us: i64, host_us: i64, head: HeadPose) -> Self {
        Self {
            timestamp_us,
            host_us,
            head,
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

// A pose sample goes inline: it is no larger than a notification (one of a
// display area, 80 bytes), so it makes no sample larger.
const _: () = assert!(size_of::<PoseSample>() <= size_of::<DeviceNotification>());

/// One frame of the 0x1770 `algodbg` stream, the Stream Engine's custom
/// stream 0: the data the DLL hands the stream's subscriber, with the device
/// stream and the time the frame was read (see [Custom
/// stream](crate::engine#custom-stream)).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CustomSample {
    /// The device stream the frame came on: 0x1770.
    pub stream_id: u32,
    /// Host time when the frame was read, microseconds
    /// ([`tobii_ipc::host_clock_us`]): the frame carries no device time.
    pub host_rx_us: i64,
    /// The frame's data, key 1's bytes
    /// ([`CustomFrame::data`](tobii_proto::custom83::CustomFrame::data)):
    /// empty in every frame of the ET5.
    pub data: Vec<u8>,
}

impl CustomSample {
    /// A frame of device stream `stream_id` carrying `data`, read at host
    /// time `host_rx_us`.
    #[must_use]
    pub fn new(stream_id: u32, host_rx_us: i64, data: Vec<u8>) -> Self {
        Self {
            stream_id,
            host_rx_us,
            data,
        }
    }
}

// A custom sample goes inline: its data is on the heap, so it is no larger
// than a notification and makes no sample larger.
const _: () = assert!(size_of::<CustomSample>() <= size_of::<DeviceNotification>());

/// One item out of the engine.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Sample {
    /// The head pose of an image, one for every image while a head pose is
    /// wanted (see [`Wanted::head`]).
    Pose(PoseSample),
    /// A gaze frame (boxed: it is ten times the size of the other samples).
    Gaze(Box<GazeSample>),
    /// A presence change.
    Presence(PresenceSample),
    /// An IR frame (only while images are wanted, see [`Wanted::image`]).
    Image(ImageSample),
    /// A frame of the custom stream (see [Custom
    /// stream](crate::engine#custom-stream)).
    Custom(CustomSample),
    /// A device notification.
    Notification(DeviceNotification),
    /// The device finished its init; what it reported about itself, the
    /// display area it confirmed (see [Display
    /// area](crate::engine#display-area)) and the columns of its streams
    /// (see [Stream columns](crate::engine#stream-columns)) included.
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

/// A start or a stop of one of the tracker's catalogue streams, which the
/// engine makes for its owner (see [`Commands::change_stream`] and
/// [Catalogue streams](crate::engine#catalogue-streams)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamChange {
    /// Start the device stream of this id (command 1220).
    Start(u32),
    /// Stop the device stream of this id (command 1230).
    Stop(u32),
}

impl StreamChange {
    /// The device stream it changes.
    #[must_use]
    pub const fn stream_id(self) -> u32 {
        match self {
            Self::Start(id) | Self::Stop(id) => id,
        }
    }

    /// The device command that makes the change, its answer due within
    /// `timeout`: for a start, command 1220 with the stream id and an empty
    /// list ([`stream_start_payload`]), as every Stream Engine start carries
    /// one; for a stop, command 1230 with the stream id alone
    /// ([`stream_id_payload`]).
    #[must_use]
    pub fn device_command(self, timeout: Duration) -> DeviceCommand {
        let (command, payload) = match self {
            Self::Start(id) => (cmd::STREAM_START, stream_start_payload(id)),
            Self::Stop(id) => (cmd::STREAM_STOP, stream_id_payload(id)),
        };
        DeviceCommand {
            cmd: command,
            payload,
            timeout,
        }
    }
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
    /// The engine does not run the command, and wrote nothing: a change of
    /// a stream it does not start or stop, or of one the tracker's catalogue
    /// does not list, or a stream start or stop (command 1220 or 1230)
    /// handed to [`Commands::run`] (see [Catalogue
    /// streams](crate::engine#catalogue-streams)).
    NotSupported,
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("device did not answer in time"),
            Self::EngineGone => f.write_str("engine stopped"),
            Self::Usb(e) => write!(f, "USB write failed: {e}"),
            Self::NotSupported => f.write_str("the engine does not run that stream start or stop"),
        }
    }
}

impl std::error::Error for CommandError {}

/// A command waiting for the USB thread, with where to send its answer.
#[derive(Debug)]
pub(crate) struct QueuedCommand {
    pub(crate) command: DeviceCommand,
    pub(crate) reply: Sender<Result<CommandResponse, CommandError>>,
    /// The catalogue-stream change the command makes (see
    /// [`Commands::change_stream`]); `None` for any other command.
    pub(crate) stream: Option<StreamChange>,
}

/// Runs commands on the engine's device, and starts and stops its catalogue
/// streams; cheap to clone and usable from any thread.
#[derive(Debug, Clone)]
pub struct Commands(pub(crate) Sender<QueuedCommand>);

impl Commands {
    /// A handle that no engine serves, as an engine's is once the engine has
    /// stopped: each command it queues, a change of a catalogue stream among
    /// them, is answered [`CommandError::EngineGone`] at once. What a handle
    /// refuses before it queues anything is refused all the same: a stream
    /// start or stop handed to [`Commands::run`], and a change of any stream
    /// but the catalogue streams, are [`CommandError::NotSupported`]. For the
    /// tests of an engine's owner, which start no engine: the answer tells
    /// what reached the engine's queue from what never did.
    #[must_use]
    pub fn unserved() -> Self {
        let (queue, taken) = mpsc::channel();
        // Nothing takes from the queue: its receiving end goes now, as an
        // engine's goes with its USB thread, and so each send fails.
        drop(taken);
        Self(queue)
    }

    /// Send `command` and block until the device answers.
    ///
    /// # Errors
    ///
    /// [`CommandError::Timeout`] when no answer arrives within the command's
    /// timeout, plus up to [`deadline::QUEUE_ALLOWANCE`] spent queued and
    /// writing it (see [`deadline::command`]), or the device refuses the write
    /// past its deadline, [`CommandError::EngineGone`] when the engine stops
    /// first, [`CommandError::Usb`] when the write fails otherwise.
    /// [`CommandError::NotSupported`] at once, with nothing queued, for a
    /// stream start or stop (command 1220 or 1230): streams are started and
    /// stopped with [`Commands::change_stream`] alone, so that the streams
    /// the engine keeps running are the ones it was asked for (see [Catalogue
    /// streams](crate::engine#catalogue-streams)).
    pub fn run(&self, command: DeviceCommand) -> Result<CommandResponse, CommandError> {
        if matches!(command.cmd, cmd::STREAM_START | cmd::STREAM_STOP) {
            return Err(CommandError::NotSupported);
        }
        let deadline = deadline::command(command.timeout);
        let (reply, answer) = mpsc::channel();
        self.0
            .send(QueuedCommand {
                command,
                reply,
                stream: None,
            })
            .map_err(|_| CommandError::EngineGone)?;
        match answer.recv_timeout(deadline) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(CommandError::Timeout),
            Err(RecvTimeoutError::Disconnected) => Err(CommandError::EngineGone),
        }
    }

    /// Queue `change`, its answer due within `timeout` once it is written,
    /// and return without waiting: [`StreamAnswer::wait`] waits for the
    /// answer. The engine runs the commands in the order they were queued,
    /// changes among them, once an open's gaze stream has armed and the open
    /// has started again the streams it keeps; dropping the [`StreamAnswer`]
    /// does not take the change back.
    ///
    /// A change of a stream other than the catalogue streams, 0x1770 and
    /// 0x508, is answered [`CommandError::NotSupported`] at once and not
    /// queued; the engine answers so a change of one the tracker's catalogue
    /// does not list, writing nothing (see [Catalogue
    /// streams](crate::engine#catalogue-streams)).
    #[must_use = "the answer comes through it; dropping it leaves the change queued"]
    pub fn change_stream(&self, change: StreamChange, timeout: Duration) -> StreamAnswer {
        let queued = Instant::now();
        if !CATALOGUE_STREAMS.contains(&change.stream_id()) {
            return StreamAnswer::answered(Err(CommandError::NotSupported));
        }
        let (reply, answer) = mpsc::channel();
        // The send fails only once the USB thread has gone; the reply's
        // sender goes with the command, and the wait answers EngineGone.
        let _ = self.0.send(QueuedCommand {
            command: change.device_command(timeout),
            reply,
            stream: Some(change),
        });
        StreamAnswer {
            answer,
            queued,
            deadline: deadline::command(timeout),
        }
    }
}

/// The answer to a [`StreamChange`] queued with [`Commands::change_stream`],
/// to wait for with [`StreamAnswer::wait`]. Dropping it does not take the
/// change back: the engine runs it all the same, and keeps the stream
/// running or not as the tracker answers.
#[derive(Debug)]
pub struct StreamAnswer {
    answer: Receiver<Result<CommandResponse, CommandError>>,
    /// When the change was queued: the wait counts from then.
    queued: Instant,
    /// How long the wait lasts from then at most: [`deadline::command`] of
    /// the change's timeout.
    deadline: Duration,
}

impl StreamAnswer {
    /// A handle whose answer is there already, `result`: what a stand-in for
    /// the engine (a test's) hands out.
    #[must_use]
    pub fn answered(result: Result<CommandResponse, CommandError>) -> Self {
        let (reply, answer) = mpsc::channel();
        // The receiver is right here, so the send cannot fail, and the
        // answer stays in the channel once its sender is gone.
        let _ = reply.send(result);
        Self {
            answer,
            queued: Instant::now(),
            deadline: Duration::ZERO,
        }
    }

    /// Wait for the tracker's answer to the change, until
    /// [`deadline::command`] of its timeout has passed since it was queued:
    /// up to [`deadline::QUEUE_ALLOWANCE`] for the engine to take it (once an
    /// open's gaze stream has armed) and write it, then the timeout.
    ///
    /// # Errors
    ///
    /// [`CommandError::Timeout`] when the device did not answer in time, or
    /// the wait gave up first (the engine may still write the change then,
    /// and the tracker take it), [`CommandError::EngineGone`] when the engine
    /// stopped, or the open the change was written in ended, before the
    /// answer, [`CommandError::Usb`] when its write failed, and
    /// [`CommandError::NotSupported`] for a change the engine does not run.
    pub fn wait(self) -> Result<CommandResponse, CommandError> {
        let left = self.deadline.saturating_sub(self.queued.elapsed());
        match self.answer.recv_timeout(left) {
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
    /// A head pose: the engine runs head-pose inference on the image stream
    /// (it costs CPU) and makes a [`PoseSample`] of every image it takes,
    /// the Stream Engine's pose valid or not; it drops the images otherwise.
    pub head: bool,
    /// [`Sample::Image`] for every IR frame.
    pub image: bool,
}

/// The head flag of [`Wanted`] as the pose worker reads it: whether it is
/// set, and how many times it has been set when it was not. The worker
/// reads the flag once an image, and sees from the count that a client
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

/// State shared between the [`Engine`] handle and its USB thread.
#[derive(Debug, Default)]
pub(crate) struct Shared {
    pub(crate) stop: AtomicBool,
    /// [`Wanted::head`], which the pose worker reads (see [`Shared::head`]):
    /// under a lock, so that it never reads whether the flag is set from
    /// one change and how many times it was raised from another.
    head: Mutex<HeadFlag>,
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
        // Nothing under the lock can panic: a poisoned lock holds no
        // half-updated flag.
        self.head
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .set(wanted.head);
        // Relaxed: a pure signal, no data is published with it.
        self.image_wanted.store(wanted.image, Ordering::Relaxed);
    }

    /// The head flag as it is now.
    pub(crate) fn head(&self) -> HeadFlag {
        *self.head.lock().unwrap_or_else(PoisonError::into_inner)
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
        Self::start_seeded(display_area, &[])
    }

    /// Like [`Engine::start_with`], keeping `streams` running as if the
    /// tracker had taken a start of each: the first open whose gaze stream
    /// arms starts them, and so does each open after it (see [Catalogue
    /// streams](crate::engine#catalogue-streams)). For the streams an engine
    /// before this one ran. Of `streams`, the catalogue streams alone, 0x1770
    /// and 0x508, are kept; any other is warned of and left out.
    #[must_use]
    pub fn start_seeded(display_area: Option<DisplayArea>, streams: &[u32]) -> Self {
        let keep = seed(streams);
        let shared = Arc::new(Shared {
            display_override: Mutex::new(display_area),
            ..Shared::default()
        });
        let (tx, rx) = mpsc::channel();
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let thread_shared = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            ended(crate::device::run_gaze_engine(
                &thread_shared,
                &cmd_rx,
                &tx,
                keep,
            ))
        });
        Engine {
            shared,
            commands: Commands(cmd_tx),
            rx,
            pending: VecDeque::new(),
            handle: Some(handle),
        }
    }

    /// A handle for running device commands and starting and stopping the
    /// catalogue streams.
    #[must_use]
    pub fn commands(&self) -> Commands {
        self.commands.clone()
    }

    /// Do the optional work `wanted` asks for, and none that it does not:
    /// head-pose inference, the IR images (see [`Wanted`]).
    ///
    /// The pose worker reads the head flag at each image it takes, with how
    /// many times it was set when it was not. The first image after a head
    /// pose was wanted anew starts the pose over: the Stream Engine's
    /// restarts its filters, its invalid poses carrying zeros until the
    /// next valid one. Wanted anew is since the image before, even if the
    /// flag was cleared and set again between the two or while no image
    /// came (a pause, a re-open); not at the worker's first image, whose
    /// pose is new. So a new subscriber does not inherit an earlier one's
    /// filters, unless a head pose stays wanted throughout: a client that
    /// comes to want it while another keeps it wanted gets the filters as
    /// they are.
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

/// The streams of `streams` a new engine keeps (see
/// [`Engine::start_seeded`]): the [`CATALOGUE_STREAMS`] among them, each
/// once. Any other is warned of: the engine starts and stops no other
/// stream.
fn seed(streams: &[u32]) -> BTreeSet<u32> {
    let mut keep = BTreeSet::new();
    for &id in streams {
        if CATALOGUE_STREAMS.contains(&id) {
            keep.insert(id);
        } else {
            warn!(
                stream = format_args!("{id:#x}"),
                "engine: a stream to keep that is not a catalogue stream is left out"
            );
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_proto::protocol::{
        STREAM_ID_ALGODBG, STREAM_ID_GAZE, STREAM_ID_IMAGE, STREAM_ID_IMAGE_COLLECTION,
        STREAM_ID_PRESENCE, chunk_command, stream_start_packet, stream_stop_packet,
    };

    /// How long the tracker may take to answer a change in these tests: the
    /// DLL's wait for a 1220.
    const ANSWER_DUE: Duration = Duration::from_secs(3);

    /// A handle on a queue no engine takes from, and the queue.
    fn queue() -> (Commands, Receiver<QueuedCommand>) {
        let (tx, rx) = mpsc::channel();
        (Commands(tx), rx)
    }

    #[test]
    fn a_stream_change_is_the_1220_or_1230_proto_builds() {
        let start = StreamChange::Start(STREAM_ID_ALGODBG).device_command(ANSWER_DUE);
        assert_eq!(
            (start.cmd, start.timeout),
            (cmd::STREAM_START, ANSWER_DUE),
            "its answer due within the timeout given"
        );
        // 00 00, the id as a type-2 u32, then an empty list of u32 (type
        // 0x17, length 4, count 0).
        assert_eq!(
            start.payload,
            [
                0, 0, 2, 0, 0, 0, 4, 0, 0, 0x17, 0x70, 0x17, 0, 0, 0, 4, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            chunk_command(start.cmd, 0x2f, &start.payload),
            [stream_start_packet(0x2f, STREAM_ID_ALGODBG)],
            "the packet the 0x50e start writes, for another stream"
        );

        let stop = StreamChange::Stop(STREAM_ID_IMAGE_COLLECTION).device_command(ANSWER_DUE);
        assert_eq!(stop.cmd, cmd::STREAM_STOP);
        assert_eq!(
            stop.payload,
            [0, 0, 2, 0, 0, 0, 4, 0, 0, 5, 8],
            "the id alone"
        );
        assert_eq!(
            chunk_command(stop.cmd, 0x30, &stop.payload),
            [stream_stop_packet(0x30, STREAM_ID_IMAGE_COLLECTION)]
        );
        assert_eq!(StreamChange::Stop(0x508).stream_id(), 0x508);
    }

    #[test]
    fn a_change_of_a_stream_the_engine_does_not_start_is_not_supported_and_not_queued() {
        let (commands, queued) = queue();
        // The engine's own streams, and one of the catalogue the DLL does
        // not start.
        for id in [STREAM_ID_GAZE, STREAM_ID_PRESENCE, STREAM_ID_IMAGE, 0x501] {
            for change in [StreamChange::Start(id), StreamChange::Stop(id)] {
                assert_eq!(
                    commands.change_stream(change, ANSWER_DUE).wait(),
                    Err(CommandError::NotSupported),
                    "{change:?}"
                );
            }
        }
        assert!(queued.try_recv().is_err(), "nothing queued");
    }

    #[test]
    fn a_raw_stream_start_or_stop_is_not_supported_and_not_queued() {
        let (commands, queued) = queue();
        for change in [
            StreamChange::Start(STREAM_ID_ALGODBG),
            StreamChange::Stop(STREAM_ID_ALGODBG),
            StreamChange::Stop(STREAM_ID_GAZE),
        ] {
            assert_eq!(
                commands.run(change.device_command(ANSWER_DUE)),
                Err(CommandError::NotSupported),
                "{change:?}"
            );
        }
        assert!(queued.try_recv().is_err(), "nothing queued");
    }

    #[test]
    fn a_change_is_queued_in_turn_and_dropping_its_answer_takes_nothing_back() {
        let (commands, queued) = queue();

        let start = commands.change_stream(StreamChange::Start(STREAM_ID_ALGODBG), ANSWER_DUE);
        drop(commands.change_stream(StreamChange::Stop(STREAM_ID_IMAGE_COLLECTION), ANSWER_DUE));

        let first = queued.try_recv().expect("the start, queued");
        assert_eq!(
            (first.stream, first.command),
            (
                Some(StreamChange::Start(STREAM_ID_ALGODBG)),
                StreamChange::Start(STREAM_ID_ALGODBG).device_command(ANSWER_DUE)
            )
        );
        let second = queued.try_recv().expect("the stop, queued all the same");
        assert_eq!(
            second.stream,
            Some(StreamChange::Stop(STREAM_ID_IMAGE_COLLECTION))
        );
        // Its answer goes nowhere, and the engine does not mind.
        assert!(
            second
                .reply
                .send(Ok(CommandResponse::ok(Vec::new())))
                .is_err()
        );
        let refused = CommandResponse {
            status: RESPONSE_STATUS_OK,
            error: ttp_error::UNAUTHORIZED,
            payload: Vec::new(),
        };
        first
            .reply
            .send(Ok(refused.clone()))
            .expect("the answer is waited for");
        assert_eq!(start.wait(), Ok(refused), "the tracker's words");
    }

    #[test]
    fn an_answer_ends_its_wait_at_once_and_a_wait_ends_at_its_deadline_or_the_engines_end() {
        assert_eq!(
            StreamAnswer::answered(Ok(CommandResponse::ok(vec![7]))).wait(),
            Ok(CommandResponse::ok(vec![7]))
        );
        assert_eq!(
            StreamAnswer::answered(Err(CommandError::NotSupported)).wait(),
            Err(CommandError::NotSupported)
        );

        // No answer: the wait gives up once its deadline has passed since
        // the change was queued, not since the wait began.
        let (_reply, answer) = mpsc::channel();
        let queued = Instant::now()
            .checked_sub(Duration::from_secs(10))
            .expect("a monotonic clock older than 10 s");
        let late = StreamAnswer {
            answer,
            queued,
            deadline: Duration::from_secs(10),
        };
        let began = Instant::now();
        assert_eq!(late.wait(), Err(CommandError::Timeout));
        assert!(began.elapsed() < Duration::from_secs(5), "not another 10 s");

        // The engine went away with the change: its reply's sender too.
        let (commands, queued) = queue();
        let gone = commands.change_stream(StreamChange::Start(STREAM_ID_ALGODBG), ANSWER_DUE);
        drop(queued);
        assert_eq!(gone.wait(), Err(CommandError::EngineGone));
        assert_eq!(
            commands
                .change_stream(StreamChange::Stop(STREAM_ID_ALGODBG), ANSWER_DUE)
                .wait(),
            Err(CommandError::EngineGone),
            "queued after it went"
        );
    }

    #[test]
    fn a_handle_no_engine_serves_answers_engine_gone_to_what_it_queues() {
        let commands = Commands::unserved();
        assert_eq!(
            commands.run(DeviceCommand {
                cmd: cmd::DISPLAY_AREA_GET,
                payload: Vec::new(),
                timeout: ANSWER_DUE,
            }),
            Err(CommandError::EngineGone)
        );
        for change in [
            StreamChange::Start(STREAM_ID_ALGODBG),
            StreamChange::Stop(STREAM_ID_IMAGE_COLLECTION),
        ] {
            assert_eq!(
                commands.change_stream(change, ANSWER_DUE).wait(),
                Err(CommandError::EngineGone),
                "{change:?}"
            );
        }

        // Refused before anything is queued, as by an engine's handle.
        assert_eq!(
            commands.run(StreamChange::Start(STREAM_ID_ALGODBG).device_command(ANSWER_DUE)),
            Err(CommandError::NotSupported)
        );
        assert_eq!(
            commands
                .change_stream(StreamChange::Stop(STREAM_ID_GAZE), ANSWER_DUE)
                .wait(),
            Err(CommandError::NotSupported)
        );
    }

    #[test]
    fn a_new_engine_keeps_the_catalogue_streams_it_is_given_and_no_other() {
        assert_eq!(
            seed(&[
                STREAM_ID_GAZE,
                STREAM_ID_ALGODBG,
                STREAM_ID_IMAGE_COLLECTION,
                STREAM_ID_ALGODBG,
                STREAM_ID_IMAGE,
                0x1774,
            ]),
            BTreeSet::from([STREAM_ID_IMAGE_COLLECTION, STREAM_ID_ALGODBG])
        );
        assert!(seed(&[]).is_empty());
    }

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
}
