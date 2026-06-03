//! C ABI in the shape of the Tobii Stream Engine, backed by the `tobiid` daemon.
//! Built into `libtobii.so`.
//!
//! `tobii_device_create` connects to the daemon (auto-spawning it if needed), so
//! several processes can consume the same device. Subscriptions select the mode;
//! a head+gaze conflict (or a daemon already in the other mode) returns
//! `TOBII_ERROR_CONFLICTING_API`. Pump with `tobii_wait_for_callbacks` +
//! `tobii_device_process_callbacks` exactly like the Stream Engine.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::net::Shutdown;
use std::os::raw::c_char;
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::ipc::{
    self, decode_server, encode_subscribe, read_frame, write_frame, ServerMsg, STREAM_GAZE,
    STREAM_HEAD, STREAM_PRESENCE,
};

pub type Status = i32;
pub const TOBII_ERROR_NO_ERROR: Status = 0;
pub const TOBII_ERROR_INVALID_PARAMETER: Status = 1;
pub const TOBII_ERROR_CONNECTION_FAILED: Status = 2;
pub const TOBII_ERROR_CONFLICTING_API: Status = 9;
pub const TOBII_ERROR_TIMED_OUT: Status = 11;

pub type Validity = u32;
pub const TOBII_VALIDITY_INVALID: Validity = 0;
pub const TOBII_VALIDITY_VALID: Validity = 1;

pub type PresenceStatus = u32;
pub const TOBII_USER_PRESENCE_STATUS_UNKNOWN: PresenceStatus = 0;
pub const TOBII_USER_PRESENCE_STATUS_AWAY: PresenceStatus = 1;
pub const TOBII_USER_PRESENCE_STATUS_PRESENT: PresenceStatus = 2;

pub struct Api {
    _private: u8,
}

/// Opaque device handle: a connection to the daemon plus a reader thread that
/// funnels decoded messages into a channel.
pub struct Device {
    stream: UnixStream,
    rx: Receiver<ServerMsg>,
    pending: VecDeque<ServerMsg>,
    reader: Option<JoinHandle<()>>,
    streams: u8,
    head: Option<(HeadPoseFn, *mut c_void)>,
    gaze: Option<(GazePointFn, *mut c_void)>,
    presence: Option<(PresenceFn, *mut c_void)>,
}

impl Device {
    /// Add `bit` to the subscription, resend, and wait for the daemon's ack.
    fn subscribe(&mut self, bit: u8) -> Status {
        self.streams |= bit;
        if write_frame(&mut self.stream, &encode_subscribe(self.streams)).is_err() {
            return TOBII_ERROR_CONNECTION_FAILED;
        }
        loop {
            match self.rx.recv_timeout(Duration::from_secs(2)) {
                Ok(ServerMsg::Subscribed { ok: true }) => return TOBII_ERROR_NO_ERROR,
                Ok(ServerMsg::Subscribed { ok: false }) => return TOBII_ERROR_CONFLICTING_API,
                Ok(other) => self.pending.push_back(other),
                Err(_) => return TOBII_ERROR_TIMED_OUT,
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

#[repr(C)]
pub struct HeadPose {
    pub timestamp_us: i64,
    pub position_validity: Validity,
    pub position_xyz: [f32; 3],
    pub rotation_validity_xyz: [Validity; 3],
    pub rotation_xyz: [f32; 3],
}

#[repr(C)]
pub struct GazePoint {
    pub timestamp_us: i64,
    pub validity: Validity,
    pub position_xy: [f32; 2],
}

pub type HeadPoseFn = unsafe extern "C" fn(*const HeadPose, *mut c_void);
pub type GazePointFn = unsafe extern "C" fn(*const GazePoint, *mut c_void);
pub type PresenceFn = unsafe extern "C" fn(PresenceStatus, i64, *mut c_void);

/// # Safety
/// `api` must be a valid pointer to write the handle into.
#[no_mangle]
pub unsafe extern "C" fn tobii_api_create(
    api: *mut *mut Api,
    _custom_alloc: *const c_void,
    _custom_log: *const c_void,
) -> Status {
    if api.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    *api = Box::into_raw(Box::new(Api { _private: 0 }));
    TOBII_ERROR_NO_ERROR
}

/// # Safety
/// `api` must come from `tobii_api_create`.
#[no_mangle]
pub unsafe extern "C" fn tobii_api_destroy(api: *mut Api) -> Status {
    if !api.is_null() {
        drop(Box::from_raw(api));
    }
    TOBII_ERROR_NO_ERROR
}

/// # Safety
/// `device` must be a valid pointer to write the handle into.
#[no_mangle]
pub unsafe extern "C" fn tobii_device_create(
    _api: *mut Api,
    _url: *const c_char,
    _field_of_use: i32,
    device: *mut *mut Device,
) -> Status {
    if device.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let stream = match ipc::connect_or_spawn() {
        Ok(s) => s,
        Err(_) => return TOBII_ERROR_CONNECTION_FAILED,
    };
    let reader_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return TOBII_ERROR_CONNECTION_FAILED,
    };
    let (tx, rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut s = reader_stream;
        while let Ok(Some(body)) = read_frame(&mut s) {
            if let Some(msg) = decode_server(&body) {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        }
    });
    let d = Box::new(Device {
        stream,
        rx,
        pending: VecDeque::new(),
        reader: Some(reader),
        streams: 0,
        head: None,
        gaze: None,
        presence: None,
    });
    *device = Box::into_raw(d);
    TOBII_ERROR_NO_ERROR
}

/// # Safety
/// `device` must come from `tobii_device_create`.
#[no_mangle]
pub unsafe extern "C" fn tobii_device_destroy(device: *mut Device) -> Status {
    if !device.is_null() {
        drop(Box::from_raw(device));
    }
    TOBII_ERROR_NO_ERROR
}

/// # Safety
/// `device` must be valid; `callback` is invoked from `process_callbacks`.
#[no_mangle]
pub unsafe extern "C" fn tobii_head_pose_subscribe(
    device: *mut Device,
    callback: HeadPoseFn,
    user_data: *mut c_void,
) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let status = d.subscribe(STREAM_HEAD);
    if status == TOBII_ERROR_NO_ERROR {
        d.head = Some((callback, user_data));
    }
    status
}

/// # Safety
/// `device` must be valid.
#[no_mangle]
pub unsafe extern "C" fn tobii_head_pose_unsubscribe(device: *mut Device) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    d.head = None;
    TOBII_ERROR_NO_ERROR
}

/// # Safety
/// `device` must be valid; `callback` is invoked from `process_callbacks`.
#[no_mangle]
pub unsafe extern "C" fn tobii_gaze_point_subscribe(
    device: *mut Device,
    callback: GazePointFn,
    user_data: *mut c_void,
) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let status = d.subscribe(STREAM_GAZE);
    if status == TOBII_ERROR_NO_ERROR {
        d.gaze = Some((callback, user_data));
    }
    status
}

/// # Safety
/// `device` must be valid.
#[no_mangle]
pub unsafe extern "C" fn tobii_gaze_point_unsubscribe(device: *mut Device) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    d.gaze = None;
    TOBII_ERROR_NO_ERROR
}

/// # Safety
/// `device` must be valid; `callback` is invoked from `process_callbacks`.
#[no_mangle]
pub unsafe extern "C" fn tobii_user_presence_subscribe(
    device: *mut Device,
    callback: PresenceFn,
    user_data: *mut c_void,
) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let status = d.subscribe(STREAM_PRESENCE);
    if status == TOBII_ERROR_NO_ERROR {
        d.presence = Some((callback, user_data));
    }
    status
}

/// # Safety
/// `device` must be valid.
#[no_mangle]
pub unsafe extern "C" fn tobii_user_presence_unsubscribe(device: *mut Device) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    d.presence = None;
    TOBII_ERROR_NO_ERROR
}

/// Block until a device has data or a short timeout elapses.
///
/// # Safety
/// `devices` must point to `num_devices` valid device handles.
#[no_mangle]
pub unsafe extern "C" fn tobii_wait_for_callbacks(
    num_devices: i32,
    devices: *const *mut Device,
) -> Status {
    if devices.is_null() || num_devices < 1 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let mut any = false;
    for i in 0..num_devices as isize {
        if let Some(d) = (*devices.offset(i)).as_mut() {
            if !d.pending.is_empty() {
                any = true;
            } else if let Ok(msg) = d.rx.recv_timeout(Duration::from_millis(100)) {
                d.pending.push_back(msg);
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

/// Dispatch queued samples to the subscribed callbacks.
///
/// # Safety
/// `device` must be valid.
#[no_mangle]
pub unsafe extern "C" fn tobii_device_process_callbacks(device: *mut Device) -> Status {
    let Some(d) = device.as_mut() else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    while let Ok(msg) = d.rx.try_recv() {
        d.pending.push_back(msg);
    }
    while let Some(msg) = d.pending.pop_front() {
        match msg {
            ServerMsg::Head { ts_us, pos_mm, rot_rad } => {
                if let Some((cb, ud)) = d.head {
                    let hp = HeadPose {
                        timestamp_us: ts_us,
                        position_validity: TOBII_VALIDITY_VALID,
                        position_xyz: pos_mm,
                        rotation_validity_xyz: [TOBII_VALIDITY_VALID; 3],
                        rotation_xyz: rot_rad,
                    };
                    cb(&hp, ud);
                }
            }
            ServerMsg::Gaze { ts_us, valid, xy } => {
                if let Some((cb, ud)) = d.gaze {
                    let gp = GazePoint {
                        timestamp_us: ts_us,
                        validity: if valid {
                            TOBII_VALIDITY_VALID
                        } else {
                            TOBII_VALIDITY_INVALID
                        },
                        position_xy: xy,
                    };
                    cb(&gp, ud);
                }
            }
            ServerMsg::Presence { ts_us, status } => {
                if let Some((cb, ud)) = d.presence {
                    cb(status as PresenceStatus, ts_us, ud);
                }
            }
            ServerMsg::Subscribed { .. } => {}
        }
    }
    TOBII_ERROR_NO_ERROR
}
