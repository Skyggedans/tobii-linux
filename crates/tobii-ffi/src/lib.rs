//! C ABI in the shape of the Tobii Stream Engine, backed by the `tobiid` daemon.
//! Built into `libtobii.so`.
//!
//! `tobii_device_create` connects to the daemon (auto-spawning it if needed), so
//! several processes can consume the same device, and head pose, gaze and
//! presence can all be subscribed at once (`TOBII_ERROR_CONFLICTING_API` is
//! kept for ABI compatibility; the daemon no longer refuses a subscription).
//! Pump with `tobii_wait_for_callbacks` + `tobii_device_process_callbacks`
//! exactly like the Stream Engine.
//!
//! Every entry point takes raw handles from C, so each is an `unsafe fn` whose
//! `# Safety` section states what the caller must uphold; the `unsafe` blocks
//! inside are kept to the single pointer operation that needs them.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt;
use std::io;
use std::net::Shutdown;
use std::os::raw::c_char;
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tobii_ipc::{
    self, STREAM_GAZE, STREAM_HEAD, STREAM_PRESENCE, ServerMsg, decode_server, encode_subscribe,
    read_frame, write_frame,
};

/// Status code returned by every entry point; one of the `TOBII_ERROR_*` values.
pub type Status = i32;
/// The call succeeded.
pub const TOBII_ERROR_NO_ERROR: Status = 0;
/// A null or otherwise unusable argument was passed.
pub const TOBII_ERROR_INVALID_PARAMETER: Status = 1;
/// The daemon could not be reached (or spawned), or the connection dropped.
pub const TOBII_ERROR_CONNECTION_FAILED: Status = 2;
/// The daemon refused the subscription. Kept for ABI compatibility; the
/// current daemon never refuses.
pub const TOBII_ERROR_CONFLICTING_API: Status = 9;
/// No sample (or no daemon acknowledgement) arrived before the timeout.
pub const TOBII_ERROR_TIMED_OUT: Status = 11;

/// Validity flag carried by sample fields; one of the `TOBII_VALIDITY_*` values.
pub type Validity = u32;
/// The field holds no usable measurement.
pub const TOBII_VALIDITY_INVALID: Validity = 0;
/// The field holds a valid measurement.
pub const TOBII_VALIDITY_VALID: Validity = 1;

/// User presence as delivered to [`PresenceFn`]; one of the
/// `TOBII_USER_PRESENCE_STATUS_*` values.
pub type PresenceStatus = u32;
/// Presence could not be determined.
pub const TOBII_USER_PRESENCE_STATUS_UNKNOWN: PresenceStatus = 0;
/// Nobody is in front of the tracker.
pub const TOBII_USER_PRESENCE_STATUS_AWAY: PresenceStatus = 1;
/// A user is in front of the tracker.
pub const TOBII_USER_PRESENCE_STATUS_PRESENT: PresenceStatus = 2;

/// How long a subscription change waits for the daemon's acknowledgement.
const SUBSCRIBE_ACK_TIMEOUT: Duration = Duration::from_secs(2);
/// How long `tobii_wait_for_callbacks` blocks per idle device before giving up.
const WAIT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Opaque API handle. Carries no state; it exists so the entry points keep the
/// Stream Engine signatures.
#[derive(Debug)]
pub struct Api {
    _private: u8,
}

/// Opaque device handle: a connection to the daemon plus a reader thread that
/// funnels decoded messages into a channel.
///
/// Invariant: every stored callback was registered through the matching
/// `tobii_*_subscribe` entry point, whose safety contract makes it sound to
/// invoke with the stored `user_data` until it is unsubscribed or the device
/// is destroyed.
pub struct Device {
    stream: UnixStream,
    rx: Receiver<ServerMsg>,
    pending: VecDeque<ServerMsg>,
    reader: Option<JoinHandle<()>>,
    streams: u32,
    head: Option<(HeadPoseFn, *mut c_void)>,
    gaze: Option<(GazePointFn, *mut c_void)>,
    presence: Option<(PresenceFn, *mut c_void)>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("streams", &self.streams)
            .field("pending", &self.pending.len())
            .field("head", &self.head.is_some())
            .field("gaze", &self.gaze.is_some())
            .field("presence", &self.presence.is_some())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Connect to the daemon (spawning it if needed) and start the reader thread.
    fn connect() -> io::Result<Self> {
        let stream = tobii_ipc::connect_or_spawn()?;
        let reader_stream = stream.try_clone()?;
        let (tx, rx) = mpsc::channel();
        let reader = thread::Builder::new()
            .name("tobii-ffi-reader".into())
            .spawn(move || reader_loop(reader_stream, &tx))?;
        Ok(Self {
            stream,
            rx,
            pending: VecDeque::new(),
            reader: Some(reader),
            streams: 0,
            head: None,
            gaze: None,
            presence: None,
        })
    }

    /// Resend the current subscription mask and wait for the daemon's ack,
    /// queueing any samples that arrive meanwhile. Returns the ack's `ok` flag.
    fn resend_subscription(&mut self) -> Result<bool, Status> {
        if write_frame(&mut self.stream, &encode_subscribe(self.streams)).is_err() {
            return Err(TOBII_ERROR_CONNECTION_FAILED);
        }
        loop {
            match self.rx.recv_timeout(SUBSCRIBE_ACK_TIMEOUT) {
                Ok(ServerMsg::Subscribed { ok }) => return Ok(ok),
                Ok(other) => self.pending.push_back(other),
                Err(_) => return Err(TOBII_ERROR_TIMED_OUT),
            }
        }
    }

    /// Drop `bit` from the subscription and resend the mask, so the daemon
    /// stops sending (and computing) that stream for us.
    fn unsubscribe(&mut self, bit: u32) -> Status {
        self.streams &= !bit;
        match self.resend_subscription() {
            Ok(_) => TOBII_ERROR_NO_ERROR,
            Err(status) => status,
        }
    }

    /// Add `bit` to the subscription, resend, and wait for the daemon's ack.
    fn subscribe(&mut self, bit: u32) -> Status {
        self.streams |= bit;
        match self.resend_subscription() {
            Ok(true) => TOBII_ERROR_NO_ERROR,
            Ok(false) => TOBII_ERROR_CONFLICTING_API,
            Err(status) => status,
        }
    }

    /// Deliver one daemon message to the matching subscribed callback, if any.
    fn dispatch(&self, msg: &ServerMsg) {
        match *msg {
            ServerMsg::Head {
                ts_us,
                pos_mm,
                rot_rad,
            } => {
                if let Some((cb, ud)) = self.head {
                    let hp = HeadPose {
                        timestamp_us: ts_us,
                        position_validity: TOBII_VALIDITY_VALID,
                        position_xyz: pos_mm,
                        rotation_validity_xyz: [TOBII_VALIDITY_VALID; 3],
                        rotation_xyz: rot_rad,
                    };
                    // SAFETY: `cb`/`ud` were registered through
                    // `tobii_head_pose_subscribe`, whose contract makes `cb` sound
                    // to call with a valid `HeadPose` pointer and `ud`; `hp`
                    // outlives the call.
                    unsafe { cb(&raw const hp, ud) };
                }
            }
            ServerMsg::Gaze {
                ts_us, valid, xy, ..
            } => {
                if let Some((cb, ud)) = self.gaze {
                    let gp = GazePoint {
                        timestamp_us: ts_us,
                        validity: if valid {
                            TOBII_VALIDITY_VALID
                        } else {
                            TOBII_VALIDITY_INVALID
                        },
                        position_xy: xy,
                    };
                    // SAFETY: `cb`/`ud` were registered through
                    // `tobii_gaze_point_subscribe`, whose contract makes `cb` sound
                    // to call with a valid `GazePoint` pointer and `ud`; `gp`
                    // outlives the call.
                    unsafe { cb(&raw const gp, ud) };
                }
            }
            ServerMsg::Presence { ts_us, status } => {
                if let Some((cb, ud)) = self.presence {
                    // SAFETY: `cb`/`ud` were registered through
                    // `tobii_user_presence_subscribe`, whose contract makes `cb`
                    // sound to call with any status/timestamp and `ud`.
                    unsafe { cb(PresenceStatus::from(status), ts_us, ud) };
                }
            }
            ServerMsg::Subscribed { .. } => {}
            // `ServerMsg` is #[non_exhaustive]: a message kind added by a newer
            // daemon is ignored rather than breaking the C ABI.
            _ => {}
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Shutting down the socket makes the reader's blocking read return, so
        // the join below cannot hang; a shutdown error only means it is already
        // closed.
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(h) = self.reader.take()
            && h.join().is_err()
        {
            tracing::warn!("ffi reader thread panicked");
        }
    }
}

/// Pump frames from the daemon into `tx` until EOF, a read error, or the
/// receiving `Device` going away.
fn reader_loop(mut stream: UnixStream, tx: &Sender<ServerMsg>) {
    while let Ok(Some(body)) = read_frame(&mut stream) {
        if let Some(msg) = decode_server(&body)
            && tx.send(msg).is_err()
        {
            break;
        }
    }
}

/// Head pose sample handed to [`HeadPoseFn`]. Layout matches the Stream
/// Engine's `tobii_head_pose_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPose {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Validity of `position_xyz`.
    pub position_validity: Validity,
    /// Head position in millimetres, tracker coordinates.
    pub position_xyz: [f32; 3],
    /// Per-axis validity of `rotation_xyz`.
    pub rotation_validity_xyz: [Validity; 3],
    /// Head rotation in radians about x, y, z.
    pub rotation_xyz: [f32; 3],
}

/// Gaze point sample handed to [`GazePointFn`]. Layout matches the Stream
/// Engine's `tobii_gaze_point_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazePoint {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Validity of `position_xy`.
    pub validity: Validity,
    /// Gaze point in normalised screen coordinates, `[0, 1]` each axis.
    pub position_xy: [f32; 2],
}

/// Head pose callback: `(sample, user_data)`. The sample pointer is valid only
/// for the duration of the call.
pub type HeadPoseFn = unsafe extern "C" fn(*const HeadPose, *mut c_void);
/// Gaze point callback: `(sample, user_data)`. The sample pointer is valid only
/// for the duration of the call.
pub type GazePointFn = unsafe extern "C" fn(*const GazePoint, *mut c_void);
/// Presence callback: `(status, timestamp_us, user_data)`.
pub type PresenceFn = unsafe extern "C" fn(PresenceStatus, i64, *mut c_void);

/// Create the API handle. `custom_alloc` and `custom_log` are accepted for
/// signature compatibility and ignored.
///
/// # Safety
/// `api` must be null or valid for writing one `*mut Api`. The handle written
/// there must be released with `tobii_api_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_api_create(
    api: *mut *mut Api,
    _custom_alloc: *const c_void,
    _custom_log: *const c_void,
) -> Status {
    if api.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let handle = Box::into_raw(Box::new(Api { _private: 0 }));
    // SAFETY: `api` is non-null (checked above) and the caller guarantees it is
    // valid for a write of one pointer.
    unsafe { api.write(handle) };
    TOBII_ERROR_NO_ERROR
}

/// Release an API handle. Null is accepted and ignored.
///
/// # Safety
/// `api` must be null or a handle from `tobii_api_create` that has not been
/// destroyed yet; it must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_api_destroy(api: *mut Api) -> Status {
    if !api.is_null() {
        // SAFETY: non-null, and the caller guarantees it came from
        // `Box::into_raw` in `tobii_api_create` and is destroyed only once.
        drop(unsafe { Box::from_raw(api) });
    }
    TOBII_ERROR_NO_ERROR
}

/// Connect to the daemon (spawning it if needed) and create a device handle.
/// `api`, `url` and `field_of_use` are accepted for signature compatibility
/// and ignored: there is exactly one device, owned by the daemon.
///
/// # Safety
/// `device` must be null or valid for writing one `*mut Device`. The handle
/// written there must be released with `tobii_device_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_create(
    _api: *mut Api,
    _url: *const c_char,
    _field_of_use: i32,
    device: *mut *mut Device,
) -> Status {
    if device.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let Ok(d) = Device::connect()
        .inspect_err(|e| tracing::warn!(error = %e, "could not connect to tobiid"))
    else {
        return TOBII_ERROR_CONNECTION_FAILED;
    };
    let handle = Box::into_raw(Box::new(d));
    // SAFETY: `device` is non-null (checked above) and the caller guarantees
    // it is valid for a write of one pointer.
    unsafe { device.write(handle) };
    TOBII_ERROR_NO_ERROR
}

/// Release a device handle: unsubscribes everything, closes the daemon
/// connection and joins the reader thread. Null is accepted and ignored.
///
/// # Safety
/// `device` must be null or a handle from `tobii_device_create` that has not
/// been destroyed yet and is not in use by another thread; it must not be used
/// afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_destroy(device: *mut Device) -> Status {
    if !device.is_null() {
        // SAFETY: non-null, and the caller guarantees it came from
        // `Box::into_raw` in `tobii_device_create` and is destroyed only once.
        drop(unsafe { Box::from_raw(device) });
    }
    TOBII_ERROR_NO_ERROR
}

/// Subscribe to head pose. `callback` runs on the thread that calls
/// `tobii_device_process_callbacks`, with `user_data` as its second argument.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call. `callback` must be sound to invoke with
/// a valid `*const HeadPose` and `user_data`, must not re-enter this library
/// with the same device, and `user_data` must stay valid until the stream is
/// unsubscribed or the device destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_head_pose_subscribe(
    device: *mut Device,
    callback: HeadPoseFn,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let status = d.subscribe(STREAM_HEAD);
    if status == TOBII_ERROR_NO_ERROR {
        d.head = Some((callback, user_data));
    }
    status
}

/// Stop head pose delivery and drop the registered callback.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_head_pose_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    d.head = None;
    d.unsubscribe(STREAM_HEAD)
}

/// Subscribe to gaze points. `callback` runs on the thread that calls
/// `tobii_device_process_callbacks`, with `user_data` as its second argument.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call. `callback` must be sound to invoke with
/// a valid `*const GazePoint` and `user_data`, must not re-enter this library
/// with the same device, and `user_data` must stay valid until the stream is
/// unsubscribed or the device destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_gaze_point_subscribe(
    device: *mut Device,
    callback: GazePointFn,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let status = d.subscribe(STREAM_GAZE);
    if status == TOBII_ERROR_NO_ERROR {
        d.gaze = Some((callback, user_data));
    }
    status
}

/// Stop gaze point delivery and drop the registered callback.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_gaze_point_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    d.gaze = None;
    d.unsubscribe(STREAM_GAZE)
}

/// Subscribe to user presence. `callback` runs on the thread that calls
/// `tobii_device_process_callbacks`, with `user_data` as its third argument.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call. `callback` must be sound to invoke with
/// any status, timestamp and `user_data`, must not re-enter this library with
/// the same device, and `user_data` must stay valid until the stream is
/// unsubscribed or the device destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_user_presence_subscribe(
    device: *mut Device,
    callback: PresenceFn,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let status = d.subscribe(STREAM_PRESENCE);
    if status == TOBII_ERROR_NO_ERROR {
        d.presence = Some((callback, user_data));
    }
    status
}

/// Stop presence delivery and drop the registered callback.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_user_presence_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    d.presence = None;
    d.unsubscribe(STREAM_PRESENCE)
}

/// Recalibrate the head rest pose now (extension; not in the original Stream
/// Engine). Affects whichever client currently holds the device's mode.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_recenter(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if write_frame(&mut d.stream, &tobii_ipc::encode_recenter()).is_err() {
        return TOBII_ERROR_CONNECTION_FAILED;
    }
    TOBII_ERROR_NO_ERROR
}

/// Block until at least one of `devices` has a sample queued, waiting up to
/// ~100 ms per idle device. Returns `TOBII_ERROR_TIMED_OUT` when none has.
///
/// # Safety
/// `devices` must be null or point to `num_devices` initialised `*mut Device`
/// values, each null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_wait_for_callbacks(
    num_devices: i32,
    devices: *const *mut Device,
) -> Status {
    let Ok(n) = usize::try_from(num_devices) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if devices.is_null() || n == 0 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: `devices` is non-null (checked above) and the caller guarantees
    // it points to `n` initialised, readable handle pointers that stay valid
    // and unmodified for the duration of this call.
    let handles = unsafe { std::slice::from_raw_parts(devices, n) };
    let mut any = false;
    for &handle in handles {
        // SAFETY: caller guarantees each handle is null or a live, unaliased
        // device; the reference is dropped before the next iteration.
        if let Some(d) = unsafe { handle.as_mut() } {
            if d.pending.is_empty() {
                if let Ok(msg) = d.rx.recv_timeout(WAIT_POLL_TIMEOUT) {
                    d.pending.push_back(msg);
                    any = true;
                }
            } else {
                any = true;
            }
        }
    }
    if any {
        TOBII_ERROR_NO_ERROR
    } else {
        TOBII_ERROR_TIMED_OUT
    }
}

/// Dispatch every queued sample to the subscribed callbacks, on this thread.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call. The callbacks registered on it are
/// invoked under the contracts stated on the `tobii_*_subscribe` functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_process_callbacks(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let Some(d) = (unsafe { device.as_mut() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    while let Ok(msg) = d.rx.try_recv() {
        d.pending.push_back(msg);
    }
    while let Some(msg) = d.pending.pop_front() {
        d.dispatch(&msg);
    }
    TOBII_ERROR_NO_ERROR
}
