//! `tobii.h`: API and device lifetime, callbacks, device information, states
//! and capabilities.

use std::ffi::{c_char, c_void};
use std::time::Duration;

use tobii_ipc::request::{self, decode_device_info, decode_track_box, kind};

use crate::device::{Api, Device, device_mut, in_callback};
use crate::status::{
    Status, TOBII_ERROR_CALLBACK_IN_PROGRESS, TOBII_ERROR_CONFLICTING_API_INSTANCES,
    TOBII_ERROR_CONNECTION_FAILED, TOBII_ERROR_INTERNAL, TOBII_ERROR_INVALID_PARAMETER,
    TOBII_ERROR_NO_ERROR, TOBII_ERROR_NOT_SUPPORTED, TOBII_ERROR_TIMED_OUT,
};
use crate::types::{
    DeviceInfo, DeviceUrlReceiver, FieldOfUse, StateString, TOBII_CAPABILITY_CALIBRATION_2D,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_XY,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_Z, TOBII_CAPABILITY_DISPLAY_AREA_WRITABLE,
    TOBII_FIELD_OF_USE_ANALYTICAL, TOBII_FIELD_OF_USE_INTERACTIVE, TOBII_NOT_SUPPORTED,
    TOBII_STATE_BOOL_FALSE, TOBII_STATE_CALIBRATION_ACTIVE, TOBII_STATE_CALIBRATION_ID,
    TOBII_STATE_DEVICE_PAUSED, TOBII_STATE_FAULT, TOBII_STATE_WARNING,
    TOBII_STREAM_EYE_POSITION_NORMALIZED, TOBII_STREAM_GAZE_DATA, TOBII_STREAM_GAZE_ORIGIN,
    TOBII_STREAM_GAZE_POINT, TOBII_STREAM_HEAD_POSE, TOBII_STREAM_USER_POSITION_GUIDE,
    TOBII_STREAM_USER_PRESENCE, TOBII_SUPPORTED, TrackBox, Version, copy_c_string,
};

/// The URL `tobii_enumerate_local_device_urls` reports. There is exactly one
/// device, owned by the daemon, so `tobii_device_create` ignores the URL again.
const DEVICE_URL: &std::ffi::CStr = c"tobii-ffi://tobiid";

/// The Stream Engine version this library imitates: clients gate features
/// on it (4.x has the four-argument `tobii_device_create`).
const API_VERSION: Version = Version {
    major: 4,
    minor: 1,
    revision: 0,
    build: 3,
};

/// Device facts are ready once the daemon's engine has initialised the
/// tracker, which a cold start can take most of 10 s to do.
pub(crate) const FACTS_TIMEOUT: Duration = Duration::from_secs(12);
/// How long a state query may take.
const STATE_TIMEOUT: Duration = Duration::from_secs(3);

/// `tobii_get_api_version`: 4.1.0.3.
///
/// # Safety
/// `version` must be null or valid for writing one `tobii_version_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_api_version(version: *mut Version) -> Status {
    if version.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { version.write(API_VERSION) };
    TOBII_ERROR_NO_ERROR
}

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
    let handle = Box::into_raw(Box::new(Api::new()));
    // SAFETY: `api` is non-null (checked above) and the caller guarantees it
    // is valid for a write of one pointer.
    unsafe { api.write(handle) };
    TOBII_ERROR_NO_ERROR
}

/// Release an API handle. A null handle is `TOBII_ERROR_INVALID_PARAMETER`,
/// then a call from inside a callback `TOBII_ERROR_CALLBACK_IN_PROGRESS`, in
/// the DLL's order; neither releases anything. As in the DLL, devices created
/// from the handle are not checked for; here they keep working, since the
/// handle carries no state.
///
/// # Safety
/// `api` must be null or a handle from `tobii_api_create` that has not been
/// destroyed yet; once this returns `TOBII_ERROR_NO_ERROR` it must not be
/// used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_api_destroy(api: *mut Api) -> Status {
    if api.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    // SAFETY: non-null (checked above), and the caller guarantees it came
    // from `Box::into_raw` in `tobii_api_create` and is destroyed only once.
    drop(unsafe { Box::from_raw(api) });
    TOBII_ERROR_NO_ERROR
}

/// Report the local devices by handing each URL to `receiver`. The daemon owns
/// the one device this library speaks to, so `receiver` is called exactly once.
///
/// Whether a tracker is actually attached is not decided here: it surfaces at
/// `tobii_device_create`, which returns `TOBII_ERROR_CONNECTION_FAILED` when
/// the daemon cannot be reached.
///
/// # Safety
/// `api` must be null or a live handle from `tobii_api_create`. `receiver`
/// must be sound to invoke with a NUL-terminated string and `user_data`, and
/// must not re-enter this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_enumerate_local_device_urls(
    api: *mut Api,
    // `Option` rather than a bare `extern "C" fn`: a C caller may legally pass
    // null here, and that is a defined `None` instead of an invalid fn pointer.
    receiver: Option<DeviceUrlReceiver>,
    user_data: *mut c_void,
) -> Status {
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if api.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: `receiver` is non-null (checked above) and the caller guarantees
    // it is sound to call with a NUL-terminated string and `user_data`;
    // `DEVICE_URL` is a `'static` C string, so it outlives the call.
    unsafe { receiver(DEVICE_URL.as_ptr(), user_data) };
    TOBII_ERROR_NO_ERROR
}

/// Like `tobii_enumerate_local_device_urls`, filtered by device generation.
/// The ET5 is a PRP device, which the Stream Engine always enumerates, so any
/// non-empty filter reports it; an empty one is invalid, as in the DLL.
///
/// # Safety
/// As `tobii_enumerate_local_device_urls`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_enumerate_local_device_urls_ex(
    api: *mut Api,
    receiver: Option<DeviceUrlReceiver>,
    user_data: *mut c_void,
    device_generations: u32,
) -> Status {
    if device_generations == 0 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: forwarded under the same contract.
    unsafe { tobii_enumerate_local_device_urls(api, receiver, user_data) }
}

/// Validate the arguments of the device constructors, refuse a call from
/// inside a callback (after the arguments, as in the DLL) and connect.
///
/// # Safety
/// `device` must be null or valid for writing one `*mut Device`.
pub(crate) unsafe fn create_device(
    api: *mut Api,
    field_of_use: FieldOfUse,
    device: *mut *mut Device,
) -> Status {
    if api.is_null() || device.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    if !matches!(
        field_of_use,
        TOBII_FIELD_OF_USE_INTERACTIVE | TOBII_FIELD_OF_USE_ANALYTICAL
    ) {
        tracing::warn!(
            field_of_use,
            "rejecting tobii_device_create: field_of_use must be 1 or 2 — a \
             caller built against the 3-argument Stream Engine header lands here"
        );
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // A callback could not destroy the device it made.
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    let Ok(d) = Device::connect_daemon(api as usize, field_of_use)
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

/// Connect to the daemon (spawning it if needed) and create a device handle.
/// `url` is accepted for signature compatibility and ignored: there is one
/// device, owned by the daemon.
///
/// `field_of_use` is validated exactly as the Stream Engine validates it, and
/// that check is load-bearing: a caller compiled against the three-argument
/// Stream Engine 3.x header lands here with a stack address in `field_of_use`
/// and an uninitialised `device`, and is rejected instead of corrupting memory.
/// A call from inside a callback is `TOBII_ERROR_CALLBACK_IN_PROGRESS` once
/// the arguments have been checked, as in the DLL.
///
/// # Safety
/// `api` must be null or a live handle from `tobii_api_create`. `device` must
/// be null or valid for writing one `*mut Device`; the handle written there
/// must be released with `tobii_device_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_create(
    api: *mut Api,
    _url: *const c_char,
    field_of_use: FieldOfUse,
    device: *mut *mut Device,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { create_device(api, field_of_use, device) }
}

/// Release a device handle: unsubscribes everything, closes the daemon
/// connection and joins the reader thread. A null handle is
/// `TOBII_ERROR_INVALID_PARAMETER`, and a call from inside a callback
/// `TOBII_ERROR_CALLBACK_IN_PROGRESS` for any device, since the one a callback
/// runs on is still being dispatched; as in the DLL, neither releases
/// anything.
///
/// # Safety
/// `device` must be null or a handle from `tobii_device_create` that has not
/// been destroyed yet and is not in use by another thread; once this returns
/// `TOBII_ERROR_NO_ERROR` it must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_destroy(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
        return status;
    }
    // SAFETY: non-null (checked above), and the caller guarantees it came
    // from `Box::into_raw` in `tobii_device_create` and is destroyed only
    // once; no callback runs on this thread, so no dispatch loop borrows it.
    drop(unsafe { Box::from_raw(device) });
    TOBII_ERROR_NO_ERROR
}

/// How long `tobii_wait_for_callbacks` blocks per idle device before giving up.
const WAIT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Block until at least one of `devices` has something to process, waiting
/// up to ~100 ms per idle device. Returns `TOBII_ERROR_TIMED_OUT` when none
/// has. Devices from different API handles are refused, as in the DLL. A call
/// from inside a callback is refused after the count and null checks and
/// before any device is read, so before that API check, which the DLL makes
/// first.
///
/// Something to process is a queued sample, or a lost daemon connection that
/// `tobii_device_process_callbacks` has not reported yet. A loss wakes every
/// wait until a process call reports it as `TOBII_ERROR_CONNECTION_FAILED`,
/// so a wait-and-process loop wakes once for it. After that, until
/// `tobii_device_reconnect`, a lost device waits out its ~100 ms like a quiet
/// one rather than answering at once and spinning the caller's loop; a live
/// device waited on with it is still reported, once that ~100 ms is up. The
/// wake is libtobii's own choice, within what the 4.1 documentation promises
/// for `TOBII_ERROR_NO_ERROR` ("there is something to process"); the DLL
/// shows none. As in the DLL, whose internal wait gives back only 0 or 1,
/// this never returns `TOBII_ERROR_CONNECTION_FAILED` itself.
///
/// # Safety
/// `devices` must be null or point to `device_count` initialised
/// `*mut Device` values, each null or a live handle from
/// `tobii_device_create` that no other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_wait_for_callbacks(
    device_count: i32,
    devices: *const *mut Device,
) -> Status {
    let Ok(n) = usize::try_from(device_count) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if devices.is_null() || n == 0 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: `devices` is non-null (checked above) and the caller guarantees
    // it points to `n` initialised, readable handle pointers that stay valid
    // and unmodified for the duration of this call.
    let handles = unsafe { std::slice::from_raw_parts(devices, n) };
    if handles.iter().any(|h| h.is_null()) {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // Before any device is read: the one a callback runs on is still
    // borrowed by the dispatch loop.
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    let mut api = None;
    for &handle in handles {
        // SAFETY: non-null (checked above) and live per the contract; the
        // shared borrow ends before the next iteration.
        let this = unsafe { &*handle }.api;
        if *api.get_or_insert(this) != this {
            return TOBII_ERROR_CONFLICTING_API_INSTANCES;
        }
    }
    let mut any = false;
    for &handle in handles {
        // SAFETY: the caller guarantees each handle is live and unaliased;
        // the reference is dropped before the next iteration.
        match unsafe { device_mut(handle) } {
            Ok(d) => any |= d.wait(WAIT_POLL_TIMEOUT),
            Err(status) => return status,
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
/// Once the daemon connection is lost (tobiid stopped, crashed or was
/// restarted, or dropped this client), the samples that had arrived are still
/// delivered, and then this returns `TOBII_ERROR_CONNECTION_FAILED`, on that
/// call and on every later one until `tobii_device_reconnect` connects again,
/// which is what the 4.1 documentation says to call. The DLL returns the error
/// without emptying its queue on that call. A tracker unplugged while the
/// daemon runs does not lose the connection: the daemon starts the tracker
/// again once it is back, and its samples resume. Requests that need the
/// tracker fail with `TOBII_ERROR_CONNECTION_FAILED` all the same once the
/// daemon has given it up, a few seconds after the unplug (see
/// `tobii_device_reconnect`), so this is the call that says whether the
/// connection is gone.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call. The callbacks registered on it are
/// invoked under the contracts stated on the `tobii_*_subscribe` functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_process_callbacks(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(d) => d.process(),
        Err(status) => status,
    }
}

/// Drop every sample queued for `device` without delivering it. A lost
/// daemon connection is still reported by the next process call.
///
/// # Safety
/// As `tobii_device_process_callbacks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_clear_callback_buffers(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(d) => {
            d.clear_buffers();
            TOBII_ERROR_NO_ERROR
        }
        Err(status) => status,
    }
}

/// Connect to the daemon again and subscribe every registered callback's
/// stream on the new connection; only then is the old one closed. Once this
/// has succeeded, a later loss of the new connection is reported again, and
/// the device info is asked for again, since a restarted daemon may serve
/// another tracker.
///
/// It only connects: unlike `tobii_device_create` it never spawns a daemon,
/// and it waits ~500 ms at most for the subscriptions' ack, leaving longer
/// waits (a daemon systemd is still restarting) to the caller's next try. As
/// in the DLL, any failure is `TOBII_ERROR_CONNECTION_FAILED`, never
/// `TOBII_ERROR_TIMED_OUT`. A failure leaves the device as it was, so a lost
/// one stays lost. A calibration session or pause the old connection held is
/// not restored: the daemon ends both when that connection closes. Nor is a
/// tracker: when a request fails with `TOBII_ERROR_CONNECTION_FAILED` because
/// the daemon has no tracker, a reconnect succeeds without bringing it back.
///
/// # Safety
/// As `tobii_device_process_callbacks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_reconnect(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(d) => d.reconnect(),
        Err(status) => status,
    }
}

/// Accepted and a no-op: the daemon maps the device clock to the host clock
/// for every sample and keeps the mapping current itself, so there is no
/// offset here to refresh (`tobii_timesync` takes a fresh clock pair from the
/// daemon on every call).
///
/// # Safety
/// As `tobii_device_process_callbacks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_update_timesync(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(_) => TOBII_ERROR_NO_ERROR,
        Err(status) => status,
    }
}

/// The host clock, `CLOCK_MONOTONIC` in microseconds
/// ([`tobii_ipc::host_clock_us`], which the daemon reads too): the clock of
/// every callback timestamp (bar gaze data's `timestamp_tracker_us`) and of
/// `tobii_timesync`'s host times. Its epoch is undefined, as that of the
/// DLL's `QueryPerformanceCounter` is.
///
/// # Safety
/// `api` must be null or a live handle; `timestamp_us` must be null or valid
/// for writing one `int64_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_system_clock(api: *mut Api, timestamp_us: *mut i64) -> Status {
    if api.is_null() || timestamp_us.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let now = tobii_ipc::host_clock_us();
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { timestamp_us.write(now) };
    TOBII_ERROR_NO_ERROR
}

/// Fetch the device's identity from the daemon, once per connection.
pub(crate) fn fetch_device_info(d: &mut Device) -> Result<request::DeviceInfo, Status> {
    if let Some(info) = &d.device_info {
        return Ok(info.clone());
    }
    let payload = d.request(kind::DEVICE_INFO, &[], FACTS_TIMEOUT)?;
    let info = decode_device_info(&payload).ok_or(TOBII_ERROR_INTERNAL)?;
    d.device_info = Some(info.clone());
    Ok(info)
}

fn device_info_c(info: &request::DeviceInfo) -> DeviceInfo {
    let mut c = DeviceInfo {
        serial_number: [0; 256],
        model: [0; 256],
        generation: [0; 256],
        firmware_version: [0; 256],
        integration_id: [0; 128],
        hw_calibration_version: [0; 128],
        hw_calibration_date: [0; 128],
        lot_id: [0; 128],
        integration_type: [0; 256],
        runtime_build_version: [0; 256],
    };
    copy_c_string(&mut c.serial_number, &info.serial_number);
    copy_c_string(&mut c.model, &info.model);
    copy_c_string(&mut c.generation, &info.generation);
    copy_c_string(&mut c.firmware_version, &info.firmware_version);
    copy_c_string(
        &mut c.runtime_build_version,
        concat!("libtobii.so ", env!("CARGO_PKG_VERSION")),
    );
    c
}

/// The device's serial, model, generation and firmware.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `device_info` must be null or
/// valid for writing one `tobii_device_info_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_device_info(
    device: *mut Device,
    device_info: *mut DeviceInfo,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if device_info.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match fetch_device_info(d) {
        Ok(info) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { device_info.write(device_info_c(&info)) };
            TOBII_ERROR_NO_ERROR
        }
        Err(status) => status,
    }
}

/// The volume the tracker sees the user in, mm.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `track_box` must be null or
/// valid for writing one `tobii_track_box_t`.
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)] // reason: the C ABI carries float
pub unsafe extern "C" fn tobii_get_track_box(
    device: *mut Device,
    track_box: *mut TrackBox,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if track_box.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let b = match d.request(kind::TRACK_BOX, &[], FACTS_TIMEOUT) {
        Ok(p) => match decode_track_box(&p) {
            Some(b) => b,
            None => return TOBII_ERROR_INTERNAL,
        },
        Err(status) => return status,
    };
    let c = b.corners_mm.map(|p| p.map(|v| v as f32));
    let out = TrackBox {
        front_upper_right_xyz: c[0],
        front_upper_left_xyz: c[1],
        front_lower_left_xyz: c[2],
        front_lower_right_xyz: c[3],
        back_upper_right_xyz: c[4],
        back_upper_left_xyz: c[5],
        back_lower_left_xyz: c[6],
        back_lower_right_xyz: c[7],
    };
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { track_box.write(out) };
    TOBII_ERROR_NO_ERROR
}

/// Ask the daemon for a state value.
fn query_state(d: &mut Device, state: u32) -> Result<Vec<u8>, Status> {
    d.request(kind::STATE, &request::encode_u32(state), STATE_TIMEOUT)
}

/// A boolean state. Power save, remote wake, exclusive mode and the
/// fault/warning flags are always false here; paused and calibration-active
/// come from the daemon.
///
/// Unlike the DLL, which learns it from the tracker, the paused state
/// changes as soon as the tracker accepts a pause or resume; a tracker
/// re-init ends a pause. A daemon too old to know it answers false.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `value` must be null or valid
/// for writing one `tobii_state_bool_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_state_bool(
    device: *mut Device,
    state: u32,
    value: *mut u32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if value.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let answer = match state {
        TOBII_STATE_DEVICE_PAUSED => match query_state(d, state) {
            Ok(p) => u32::from(p.first().copied().unwrap_or(0)),
            Err(TOBII_ERROR_NOT_SUPPORTED) => TOBII_STATE_BOOL_FALSE,
            Err(status) => return status,
        },
        s if s <= TOBII_STATE_WARNING => TOBII_STATE_BOOL_FALSE,
        TOBII_STATE_CALIBRATION_ACTIVE => match query_state(d, state) {
            Ok(p) => u32::from(p.first().copied().unwrap_or(0)),
            Err(status) => return status,
        },
        _ => return TOBII_ERROR_INVALID_PARAMETER,
    };
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { value.write(answer) };
    TOBII_ERROR_NO_ERROR
}

/// A `u32` state: only the calibration id. `TOBII_STATE_DEVICE_PAUSED`, which
/// an example in the 4.1 documentation reads this way, is
/// `TOBII_ERROR_INVALID_PARAMETER`, as in the DLL (whose converter at
/// 0x1800013e0 takes the calibration id only); it is a bool state.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `value` must be null or valid
/// for writing one `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_state_uint32(
    device: *mut Device,
    state: u32,
    value: *mut u32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if value.is_null() || state != TOBII_STATE_CALIBRATION_ID {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match query_state(d, state).map(|p| request::decode_u32(&p)) {
        Ok(Some(id)) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { value.write(id) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => TOBII_ERROR_INTERNAL,
        Err(status) => status,
    }
}

/// A string state: the fault and warning lists, always empty here.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `value` must be null or valid
/// for writing a `tobii_state_string_t` (512 bytes).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_state_string(
    device: *mut Device,
    state: u32,
    value: *mut StateString,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
        return status;
    }
    if value.is_null() || !matches!(state, TOBII_STATE_FAULT | TOBII_STATE_WARNING) {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees 512 writable bytes.
    unsafe { value.write([0; 512]) };
    TOBII_ERROR_NO_ERROR
}

/// Capabilities this library provides.
const fn capability_supported(capability: u32) -> bool {
    matches!(
        capability,
        TOBII_CAPABILITY_DISPLAY_AREA_WRITABLE
            | TOBII_CAPABILITY_CALIBRATION_2D
            | TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_XY
            | TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_Z
    )
}

/// Streams this library delivers: the `tobii_stream_t` values whose subscribe
/// is implemented. Digital syncport (6), diagnostics image (7) and the
/// wearable streams (9..=11) have stub subscribes, so they are not listed.
const fn stream_supported(stream: u32) -> bool {
    matches!(
        stream,
        TOBII_STREAM_GAZE_POINT
            | TOBII_STREAM_GAZE_ORIGIN
            | TOBII_STREAM_EYE_POSITION_NORMALIZED
            | TOBII_STREAM_USER_PRESENCE
            | TOBII_STREAM_HEAD_POSE
            | TOBII_STREAM_GAZE_DATA
            | TOBII_STREAM_USER_POSITION_GUIDE
    )
}

/// Shared shape of the `*_supported` queries: an unknown value is "not
/// supported", not an error (as in the DLL).
///
/// # Safety
/// As `tobii_capability_supported`.
pub(crate) unsafe fn write_supported(
    device: *mut Device,
    value: u32,
    supported: *mut u32,
    known: fn(u32) -> bool,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
        return status;
    }
    if supported.is_null() || i32::try_from(value).is_err() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let answer = if known(value) {
        TOBII_SUPPORTED
    } else {
        TOBII_NOT_SUPPORTED
    };
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { supported.write(answer) };
    TOBII_ERROR_NO_ERROR
}

/// Whether a capability is available: display-area writes, 2-D calibration
/// and the user position guide.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `supported` must be null or
/// valid for writing one `tobii_supported_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_capability_supported(
    device: *mut Device,
    capability: u32,
    supported: *mut u32,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { write_supported(device, capability, supported, capability_supported) }
}

/// Whether a stream is available: gaze point, gaze origin, eye position,
/// presence, head pose, gaze data and the user position guide.
///
/// This follows what libtobii delivers, not the DLL, which answers from what
/// the device reports (its stream lists and capability flags): its TTP path
/// never reports the user position guide, its PRP path does when the tracker
/// lists the compound stream `USER_POSITION_GUIDE_XYZ`, and its answer for 6,
/// 7 and 9..=11 depends on the device too. Here those are unsupported because
/// their subscribes are stubs. As in the DLL, a value of 12 or more is
/// reported unsupported, not an error, and a value above `i32::MAX` (a
/// negative `tobii_stream_t`) is an invalid parameter.
///
/// # Safety
/// As `tobii_capability_supported`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_stream_supported(
    device: *mut Device,
    stream: u32,
    supported: *mut u32,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { write_supported(device, stream, supported, stream_supported) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CalibrationPointData, EyePair, EyePairFn, LicenseKey, TOBII_STATE_EXCLUSIVE_MODE,
    };
    use std::ptr;

    fn api() -> *mut Api {
        let mut api: *mut Api = ptr::null_mut();
        // SAFETY: `api` is a live local valid for one pointer write.
        let status = unsafe { tobii_api_create(&raw mut api, ptr::null(), ptr::null()) };
        assert_eq!(status, 0);
        api
    }

    #[test]
    fn reports_the_imitated_version() {
        let mut v = Version::default();
        // SAFETY: `v` is a live local.
        let status = unsafe { tobii_get_api_version(&raw mut v) };
        assert_eq!(status, 0);
        assert_eq!(v, API_VERSION);
        // SAFETY: null is rejected before any write.
        let status = unsafe { tobii_get_api_version(ptr::null_mut()) };
        assert_eq!(status, TOBII_ERROR_INVALID_PARAMETER);
    }

    #[test]
    fn the_system_clock_is_the_daemons_host_clock() {
        let api = api();
        let mut t = 0i64;
        let before = tobii_ipc::host_clock_us();
        // SAFETY: `api` is live; `t` is a live local.
        let status = unsafe { tobii_system_clock(api, &raw mut t) };
        let after = tobii_ipc::host_clock_us();
        // SAFETY: `api` is live and not used again.
        assert_eq!(unsafe { tobii_api_destroy(api) }, 0);

        assert_eq!(status, 0);
        // That this is `CLOCK_MONOTONIC` and not the wall clock is
        // `tobii_ipc`'s to show.
        assert!(
            (before..=after).contains(&t),
            "{t} is not between {before} and {after}"
        );
    }

    /// Through the C entry points, a callback gets the daemon's stamp
    /// unchanged, on the clock `tobii_system_clock` reads. The fake daemon
    /// stands in for tobiid, stamping a frame on [`tobii_ipc::host_clock_us`]
    /// a latency before it is sent; the stamp lands within the
    /// `tobii_system_clock` bracket around the callbacks, less the most a
    /// capture precedes its read, only while libtobii neither converts it nor
    /// reads another clock. The daemon's own mapping (a stamp no later than
    /// its read, on the latency floor) is checked by `tobii-usb`'s `time_map`
    /// tests. Gaze data gives the tracker time beside the same stamp.
    #[test]
    fn a_frame_the_daemon_stamps_reads_on_the_system_clock() {
        use crate::device::tests::{Stamps, fake_daemon, stamp_gaze, stamp_gaze_data};
        use crate::types::{GazeDataFn, GazePointFn};
        /// The frame's tracker time.
        const TRACKER_US: i64 = 9_613_320_391;
        /// The frame was taken this long before the daemon read it.
        const LATENCY_US: i64 = 8_000;
        /// The most a gaze frame is taken before the daemon reads it: the
        /// bracket `tobii_timesync` gives one.
        const TAKEN_BEFORE_READ_US: i64 = 30_000;
        let both = tobii_ipc::STREAM_GAZE | tobii_ipc::STREAM_GAZE_DATA;
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                let mut out = Vec::new();
                if tobii_ipc::decode_subscribe(body) == Some(both) {
                    // tobiid's stamp, stood in for, of a frame it reads now.
                    let host_us = tobii_ipc::host_clock_us() - LATENCY_US;
                    out.push(tobii_ipc::encode_gaze(
                        host_us,
                        true,
                        [0.5; 2],
                        [f32::NAN; 2],
                    ));
                    out.push(tobii_ipc::encode_gaze_data(&tobii_ipc::GazeData {
                        timestamp_tracker_us: TRACKER_US,
                        timestamp_system_us: host_us,
                        ..tobii_ipc::GazeData::default()
                    }));
                }
                out.push(tobii_ipc::encode_subscribed(true));
                out
            }
            _ => vec![],
        });
        let api = api();
        let device = Box::into_raw(Box::new(
            Device::new(connect, api as usize, 1).expect("device"),
        ));
        let mut stamps = Stamps::default();
        let ud = (&raw mut stamps).cast::<c_void>();
        let (mut before, mut after) = (0i64, 0i64);
        // SAFETY: `api` and `device` are live and each destroyed once, last;
        // the callbacks match their slots and `ud` is the live `stamps`,
        // which outlives the device; `before` and `after` are live locals.
        unsafe {
            assert_eq!(tobii_system_clock(api, &raw mut before), 0);
            assert_eq!(
                crate::streams::tobii_gaze_point_subscribe(
                    device,
                    Some(stamp_gaze as GazePointFn),
                    ud
                ),
                0
            );
            assert_eq!(
                crate::advanced::tobii_gaze_data_subscribe(
                    device,
                    Some(stamp_gaze_data as GazeDataFn),
                    ud
                ),
                0
            );
            assert_eq!(tobii_device_process_callbacks(device), 0);
            assert_eq!(tobii_system_clock(api, &raw mut after), 0);
            assert_eq!(tobii_device_destroy(device), 0);
            assert_eq!(tobii_api_destroy(api), 0);
        }

        let earliest = before - TAKEN_BEFORE_READ_US;
        assert!(
            (earliest..=after).contains(&stamps.gaze),
            "{} is not between {earliest} and {after}",
            stamps.gaze
        );
        assert_eq!(stamps.gaze_data, (TRACKER_US, stamps.gaze));
    }

    unsafe extern "C" fn count_urls(_url: *const c_char, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut u32`.
        unsafe { *ud.cast::<u32>() += 1 };
    }

    #[test]
    fn enumerate_ex_rejects_an_empty_filter() {
        let api = api();
        let mut n = 0u32;
        let ud = (&raw mut n).cast::<c_void>();
        // SAFETY: `api` is live and `count_urls` matches the receiver contract.
        unsafe {
            assert_eq!(
                tobii_enumerate_local_device_urls_ex(api, Some(count_urls), ud, 0),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_enumerate_local_device_urls_ex(api, Some(count_urls), ud, 8),
                0
            );
            assert_eq!(tobii_api_destroy(api), 0);
        }
        assert_eq!(n, 1);
    }

    #[test]
    fn device_create_rejects_bad_arguments_before_connecting() {
        let api = api();
        let mut device: *mut Device = ptr::null_mut();
        for field_of_use in [0, 3, -1, 0x7fff_0000] {
            // SAFETY: `api` is live, `device` a local; rejected before any work.
            let status =
                unsafe { tobii_device_create(api, ptr::null(), field_of_use, &raw mut device) };
            assert_eq!(status, TOBII_ERROR_INVALID_PARAMETER, "{field_of_use}");
            assert!(device.is_null());
        }
        // SAFETY: null out-pointer and null api are rejected first.
        unsafe {
            assert_eq!(
                tobii_device_create(api, ptr::null(), 1, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_device_create(ptr::null_mut(), ptr::null(), 1, &raw mut device),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_api_destroy(api), 0);
        }
    }

    /// As in the DLL and the 4.1 documentation.
    #[test]
    fn destroying_a_null_handle_is_an_invalid_parameter() {
        // SAFETY: null handles are rejected before anything is freed.
        unsafe {
            assert_eq!(
                tobii_device_destroy(ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_api_destroy(ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
        }
    }

    /// A device over a fake daemon that sends two gaze-origin samples ahead of
    /// every subscription ack, so each subscription change leaves both queued
    /// for `tobii_device_process_callbacks`.
    fn device_with_two_samples_per_ack() -> *mut Device {
        let connect = crate::device::tests::fake_daemon(|body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                let sample = tobii_ipc::encode_gaze_origin(&tobii_ipc::EyePair::default());
                vec![sample.clone(), sample, tobii_ipc::encode_subscribed(true)]
            }
            _ => vec![],
        });
        Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")))
    }

    unsafe extern "C" fn ignore_pair(_p: *const EyePair, _ud: *mut c_void) {}

    #[test]
    fn wait_for_callbacks_rejects_bad_arguments_then_waits() {
        let idle = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let connect = crate::device::tests::fake_daemon(|_| vec![]);
        let elsewhere = Box::into_raw(Box::new(Device::new(connect, 2, 1).expect("device")));
        let busy = device_with_two_samples_per_ack();
        let (one, none) = ([idle], [ptr::null_mut()]);
        let (mixed, either) = ([idle, elsewhere], [idle, busy]);
        // SAFETY: the devices are live and each destroyed once below; the
        // arrays are live locals of the length passed.
        unsafe {
            assert_eq!(
                tobii_wait_for_callbacks(0, ptr::null()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_wait_for_callbacks(-1, one.as_ptr()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_wait_for_callbacks(1, none.as_ptr()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_wait_for_callbacks(2, mixed.as_ptr()),
                TOBII_ERROR_CONFLICTING_API_INSTANCES
            );
            assert_eq!(
                tobii_wait_for_callbacks(1, one.as_ptr()),
                TOBII_ERROR_TIMED_OUT
            );
            assert_eq!(
                crate::streams::tobii_gaze_origin_subscribe(
                    busy,
                    Some(ignore_pair as EyePairFn),
                    ptr::null_mut()
                ),
                0
            );
            assert_eq!(tobii_wait_for_callbacks(2, either.as_ptr()), 0);
            for d in [idle, elsewhere, busy] {
                assert_eq!(tobii_device_destroy(d), 0);
            }
        }
    }

    /// A lost daemon connection wakes the wait once, is reported by the
    /// process call, and then waits out the timeout like a quiet device; a
    /// live device waited on with it is still reported, and its samples
    /// delivered.
    #[test]
    fn wait_for_callbacks_wakes_once_for_a_lost_connection_then_times_out() {
        let (mut lost_hits, mut busy_hits) = (0u32, 0u32);
        // Nothing queued, so only the loss can wake the first wait.
        let (lost, _daemons) =
            crate::device::tests::lost_device(0, vec![], (&raw mut lost_hits).cast());
        let lost = Box::into_raw(Box::new(lost));
        let busy = device_with_two_samples_per_ack();
        let (one, both) = ([lost], [lost, busy]);
        // SAFETY: the devices are live and each destroyed once below; the
        // counters outlive them; the arrays are live locals of the length
        // passed.
        unsafe {
            assert_eq!(
                tobii_wait_for_callbacks(1, one.as_ptr()),
                TOBII_ERROR_NO_ERROR,
                "woken by the loss"
            );
            assert_eq!(
                tobii_device_process_callbacks(lost),
                TOBII_ERROR_CONNECTION_FAILED
            );
            let t = std::time::Instant::now();
            assert_eq!(
                tobii_wait_for_callbacks(1, one.as_ptr()),
                TOBII_ERROR_TIMED_OUT
            );
            assert!(t.elapsed() >= WAIT_POLL_TIMEOUT, "slept out the timeout");
            assert_eq!(
                tobii_device_process_callbacks(lost),
                TOBII_ERROR_CONNECTION_FAILED
            );

            assert_eq!(
                crate::streams::tobii_gaze_origin_subscribe(
                    busy,
                    Some(crate::device::tests::count_pair as EyePairFn),
                    (&raw mut busy_hits).cast()
                ),
                TOBII_ERROR_NO_ERROR
            );
            assert_eq!(
                tobii_wait_for_callbacks(2, both.as_ptr()),
                TOBII_ERROR_NO_ERROR
            );
            assert_eq!(tobii_device_process_callbacks(busy), TOBII_ERROR_NO_ERROR);
            for d in [lost, busy] {
                assert_eq!(tobii_device_destroy(d), TOBII_ERROR_NO_ERROR);
            }
        }
        assert_eq!(lost_hits, 0);
        assert_eq!(busy_hits, 2, "the live device's samples");
    }

    /// As in the DLL, whatever stops a reconnect is
    /// `TOBII_ERROR_CONNECTION_FAILED`, never `TOBII_ERROR_TIMED_OUT`, and
    /// the device stays lost.
    #[test]
    fn a_failed_reconnect_is_connection_failed_and_the_device_stays_lost() {
        let mut hits = 0u32;
        let (d, daemons) = crate::device::tests::lost_device(0, vec![], (&raw mut hits).cast());
        let d = Box::into_raw(Box::new(d));
        // SAFETY: `d` is live and destroyed once below; `hits` outlives it.
        unsafe {
            assert_eq!(
                tobii_device_process_callbacks(d),
                TOBII_ERROR_CONNECTION_FAILED
            );
            // A daemon that takes the connection and never acks.
            assert_eq!(tobii_device_reconnect(d), TOBII_ERROR_CONNECTION_FAILED);
            drop(daemons);
            // Nothing listens.
            assert_eq!(tobii_device_reconnect(d), TOBII_ERROR_CONNECTION_FAILED);
            assert_eq!(
                tobii_device_process_callbacks(d),
                TOBII_ERROR_CONNECTION_FAILED
            );
            assert_eq!(tobii_device_destroy(d), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(hits, 0);
    }

    /// `tobii_recenter` to a daemon that stopped reading fails, and loses
    /// the connection: the next process call reports it, after the sample
    /// that came first, although the reader has seen nothing.
    #[test]
    fn a_failed_recenter_loses_the_connection() {
        let mut hits = 0u32;
        let (d, daemon) = crate::device::tests::deaf_daemon_device((&raw mut hits).cast());
        let d = Box::into_raw(Box::new(d));
        // SAFETY: `d` is live and destroyed once below; `hits` outlives it.
        unsafe {
            assert_eq!(
                crate::streams::tobii_recenter(d),
                TOBII_ERROR_CONNECTION_FAILED
            );
            assert_eq!(
                tobii_device_process_callbacks(d),
                TOBII_ERROR_CONNECTION_FAILED
            );
            assert_eq!(tobii_device_destroy(d), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(hits, 1);
        drop(daemon);
    }

    /// The handles a callback tries to release, and what it got back: one
    /// row per sample. `other` belongs to another API instance.
    struct Teardown {
        own: *mut Device,
        other: *mut Device,
        api: *mut Api,
        got: Vec<[Status; 8]>,
    }

    unsafe extern "C" fn tear_down(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Teardown` as `ud`.
        let t = unsafe { &mut *ud.cast::<Teardown>() };
        let (own, pair) = ([t.own], [t.own, t.other]);
        // SAFETY: each call is refused, or rejected, before it reads a device
        // or frees a handle; `own` and `pair` are live locals.
        t.got.push(unsafe {
            [
                tobii_device_destroy(t.own),
                tobii_device_destroy(t.other),
                tobii_device_destroy(ptr::null_mut()),
                tobii_api_destroy(t.api),
                tobii_api_destroy(ptr::null_mut()),
                tobii_wait_for_callbacks(1, own.as_ptr()),
                tobii_wait_for_callbacks(0, ptr::null()),
                tobii_wait_for_callbacks(2, pair.as_ptr()),
            ]
        });
    }

    /// A callback that destroyed its own device used to free it under the
    /// dispatch loop. No release goes through now and the device keeps
    /// working: the second sample is still delivered, and the daemon link
    /// still round-trips.
    #[test]
    fn a_callback_cannot_destroy_a_device_or_the_api() {
        let api = api();
        let own = device_with_two_samples_per_ack();
        let connect = crate::device::tests::fake_daemon(|_| vec![]);
        let other = Box::into_raw(Box::new(Device::new(connect, 2, 1).expect("device")));
        let mut t = Teardown {
            own,
            other,
            api,
            got: Vec::new(),
        };
        // SAFETY: `own`, `other` and `api` are live and each destroyed once,
        // after the callbacks; `t` outlives the subscription.
        unsafe {
            assert_eq!(
                crate::streams::tobii_gaze_origin_subscribe(
                    own,
                    Some(tear_down as EyePairFn),
                    (&raw mut t).cast()
                ),
                0
            );
            assert_eq!(tobii_device_process_callbacks(own), 0);
            assert_eq!(crate::streams::tobii_gaze_origin_unsubscribe(own), 0);
            assert_eq!(tobii_device_destroy(own), 0);
            assert_eq!(tobii_device_destroy(other), 0);
            assert_eq!(tobii_api_destroy(api), 0);
        }
        let (refused, invalid) = (
            TOBII_ERROR_CALLBACK_IN_PROGRESS,
            TOBII_ERROR_INVALID_PARAMETER,
        );
        // A null API handle and a zero count are checked before the callback,
        // as in the DLL; a null device after it, as wherever a device handle
        // is taken. The mixed-API wait is refused before either device is
        // read, where the DLL answers `TOBII_ERROR_CONFLICTING_API_INSTANCES`.
        let row = [
            refused, refused, refused, refused, invalid, refused, invalid, refused,
        ];
        assert_eq!(t.got, [row; 2], "both samples are delivered");
        assert!(!in_callback());
    }

    /// The device and calibration a callback tries to make, and what it got
    /// back: one row per sample.
    struct Creation {
        api: *mut Api,
        device: *mut Device,
        license_results: [u32; 1],
        got: Vec<[Status; 5]>,
    }

    unsafe extern "C" fn ignore_point(_p: *const CalibrationPointData, _ud: *mut c_void) {}

    unsafe extern "C" fn create(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Creation` as `ud`.
        let c = unsafe { &mut *ud.cast::<Creation>() };
        let blob = [0u8; 8];
        let data = blob.as_ptr().cast();
        let key = LicenseKey {
            license_key: ptr::null(),
            size_in: 0,
        };
        // SAFETY: `c.api` is live and `c.device` and `c.license_results` live
        // fields; `key` is a live local, `data` points to `blob.len()`
        // readable bytes and `ignore_point` is sound to call. Each call is
        // refused, or rejected, before it connects or parses.
        c.got.push(unsafe {
            [
                tobii_device_create(c.api, ptr::null(), 1, &raw mut c.device),
                tobii_device_create(c.api, ptr::null(), 0, &raw mut c.device),
                crate::licensing::tobii_device_create_ex(
                    c.api,
                    ptr::null(),
                    1,
                    &raw const key,
                    1,
                    c.license_results.as_mut_ptr(),
                    &raw mut c.device,
                ),
                crate::calibration::tobii_calibration_parse(
                    c.api,
                    data,
                    blob.len(),
                    Some(ignore_point),
                    ptr::null_mut(),
                ),
                crate::calibration::tobii_calibration_parse(
                    c.api,
                    data,
                    blob.len(),
                    None,
                    ptr::null_mut(),
                ),
            ]
        });
    }

    /// The DLL refuses these once their arguments check out; a callback
    /// could not destroy a device it made.
    #[test]
    fn a_callback_cannot_create_a_device_or_parse_a_calibration() {
        let api = api();
        let d = device_with_two_samples_per_ack();
        let mut c = Creation {
            api,
            device: ptr::null_mut(),
            license_results: [99],
            got: Vec::new(),
        };
        // SAFETY: `d` and `api` are live and each destroyed once, after the
        // callbacks; `c` outlives the subscription.
        unsafe {
            assert_eq!(
                crate::streams::tobii_gaze_origin_subscribe(
                    d,
                    Some(create as EyePairFn),
                    (&raw mut c).cast()
                ),
                0
            );
            assert_eq!(tobii_device_process_callbacks(d), 0);
            assert_eq!(tobii_device_destroy(d), 0);
            assert_eq!(tobii_api_destroy(api), 0);
        }
        let (refused, invalid) = (
            TOBII_ERROR_CALLBACK_IN_PROGRESS,
            TOBII_ERROR_INVALID_PARAMETER,
        );
        assert_eq!(c.got, [[refused, invalid, refused, refused, invalid]; 2]);
        assert!(c.device.is_null(), "no device was written");
        assert_eq!(c.license_results, [99], "no licence result was written");
        assert!(!in_callback());
    }

    #[test]
    fn capability_and_stream_tables() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut s = 9u32;
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `s` is a live local.
        unsafe {
            assert_eq!(
                tobii_capability_supported(d, TOBII_CAPABILITY_CALIBRATION_2D, &raw mut s),
                0
            );
            assert_eq!(s, TOBII_SUPPORTED);
            assert_eq!(tobii_capability_supported(d, 2, &raw mut s), 0);
            assert_eq!(s, TOBII_NOT_SUPPORTED);
            assert_eq!(
                tobii_capability_supported(d, 99, &raw mut s),
                0,
                "unknown is not an error"
            );
            assert_eq!(s, TOBII_NOT_SUPPORTED);
            // Literal values: the DLL's numbering, not the constants', is
            // what a client compiled against Tobii's header sends.
            for stream in (0..=12).chain([1000, 0x7fff_ffff]) {
                s = 9;
                assert_eq!(
                    tobii_stream_supported(d, stream, &raw mut s),
                    0,
                    "stream {stream}: not an error"
                );
                let expected = if matches!(stream, 0..=5 | 8) {
                    TOBII_SUPPORTED
                } else {
                    TOBII_NOT_SUPPORTED
                };
                assert_eq!(s, expected, "stream {stream}");
            }
            s = 9;
            assert_eq!(
                tobii_stream_supported(d, 0x8000_0000, &raw mut s),
                TOBII_ERROR_INVALID_PARAMETER,
                "a negative tobii_stream_t"
            );
            assert_eq!(s, 9, "nothing written");
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    #[test]
    fn device_info_is_fetched_once_and_filled_in() {
        let payload = request::encode_device_info(&request::DeviceInfo {
            serial_number: "IS50F-000000000000".into(),
            model: "IS5_Large_Eyetracker_5".into(),
            generation: "IS5".into(),
            firmware_version: "02a1a6a977".into(),
        });
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, payload)));
        let mut info = device_info_c(&request::DeviceInfo::default());
        // SAFETY: `d` is live and destroyed once; `info` a live local.
        unsafe {
            assert_eq!(tobii_get_device_info(d, &raw mut info), 0);
            assert_eq!(tobii_device_destroy(d), 0);
        }
        let model: Vec<u8> = info
            .model
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| c.to_ne_bytes()[0])
            .collect();
        assert_eq!(model, b"IS5_Large_Eyetracker_5");
        assert_eq!(info.runtime_build_version[0].to_ne_bytes()[0], b'l');
    }

    #[test]
    fn states() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![1])));
        let mut v = 7u32;
        // SAFETY: `d` is live and destroyed once; `v` a live local.
        unsafe {
            assert_eq!(
                tobii_get_state_bool(d, TOBII_STATE_EXCLUSIVE_MODE, &raw mut v),
                0
            );
            assert_eq!(v, TOBII_STATE_BOOL_FALSE);
            assert_eq!(
                tobii_get_state_bool(d, TOBII_STATE_CALIBRATION_ACTIVE, &raw mut v),
                0
            );
            assert_eq!(v, 1);
            assert_eq!(
                tobii_get_state_bool(d, 8, &raw mut v),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_get_state_uint32(d, 0, &raw mut v),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_get_state_uint32(d, TOBII_STATE_DEVICE_PAUSED, &raw mut v),
                TOBII_ERROR_INVALID_PARAMETER,
                "a bool state, as in the DLL"
            );
            let mut s: StateString = [1; 512];
            assert_eq!(tobii_get_state_string(d, TOBII_STATE_FAULT, &raw mut s), 0);
            assert_eq!(s[0], 0);
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    /// The paused state a daemon answering `status`/`payload` gives. It must
    /// be asked for with STATE 2.
    fn paused_state(status: u8, payload: Vec<u8>) -> (Status, u32) {
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let connect = crate::device::tests::fake_daemon(move |body| {
            let req = request::decode_request(body).expect("request");
            log.lock()
                .expect("log")
                .push((req.kind, req.payload.to_vec()));
            vec![tobii_ipc::encode_reply(req.id, status, &payload)]
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        let mut v = 7u32;
        // SAFETY: `d` is live and destroyed once; `v` a live local.
        let got = unsafe {
            let got = tobii_get_state_bool(d, TOBII_STATE_DEVICE_PAUSED, &raw mut v);
            assert_eq!(tobii_device_destroy(d), 0);
            got
        };
        assert_eq!(
            *seen.lock().expect("log"),
            [(
                kind::STATE,
                request::encode_u32(request::state::DEVICE_PAUSED)
            )]
        );
        (got, v)
    }

    #[test]
    fn the_paused_state_comes_from_the_daemon() {
        assert_eq!(paused_state(0, vec![1]), (TOBII_ERROR_NO_ERROR, 1));
        assert_eq!(paused_state(0, vec![0]), (TOBII_ERROR_NO_ERROR, 0));
        assert_eq!(
            paused_state(tobii_ipc::request::status::NOT_SUPPORTED, vec![]),
            (TOBII_ERROR_NO_ERROR, TOBII_STATE_BOOL_FALSE),
            "an older daemon"
        );
        assert_eq!(
            paused_state(tobii_ipc::request::status::TIMED_OUT, vec![]),
            (TOBII_ERROR_TIMED_OUT, 7)
        );
    }
}
