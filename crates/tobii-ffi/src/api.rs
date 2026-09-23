//! `tobii.h`: API and device lifetime, callbacks, device information, states
//! and capabilities.

use std::ffi::{c_char, c_void};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tobii_ipc::request::{self, decode_device_info, decode_track_box, kind};

use crate::device::{Api, Device, device_mut};
use crate::status::{
    Status, TOBII_ERROR_CONFLICTING_API_INSTANCES, TOBII_ERROR_CONNECTION_FAILED,
    TOBII_ERROR_INTERNAL, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
    TOBII_ERROR_TIMED_OUT,
};
use crate::types::{
    DeviceInfo, DeviceUrlReceiver, FieldOfUse, StateString, TOBII_CAPABILITY_CALIBRATION_2D,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_XY,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_Z, TOBII_CAPABILITY_DISPLAY_AREA_WRITABLE,
    TOBII_FIELD_OF_USE_ANALYTICAL, TOBII_FIELD_OF_USE_INTERACTIVE, TOBII_NOT_SUPPORTED,
    TOBII_STATE_BOOL_FALSE, TOBII_STATE_CALIBRATION_ACTIVE, TOBII_STATE_CALIBRATION_ID,
    TOBII_STATE_FAULT, TOBII_STATE_WARNING, TOBII_STREAM_EYE_POSITION_NORMALIZED,
    TOBII_STREAM_GAZE_DATA, TOBII_STREAM_GAZE_ORIGIN, TOBII_STREAM_GAZE_POINT,
    TOBII_STREAM_HEAD_POSE, TOBII_STREAM_USER_PRESENCE, TOBII_SUPPORTED, TrackBox, Version,
    copy_c_string,
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

/// Validate the arguments of the device constructors and connect.
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

/// How long `tobii_wait_for_callbacks` blocks per idle device before giving up.
const WAIT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Block until at least one of `devices` has a sample queued, waiting up to
/// ~100 ms per idle device. Returns `TOBII_ERROR_TIMED_OUT` when none has.
/// Devices from different API handles are refused, as in the DLL.
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
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call. The callbacks registered on it are
/// invoked under the contracts stated on the `tobii_*_subscribe` functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_process_callbacks(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(d) => {
            d.process();
            TOBII_ERROR_NO_ERROR
        }
        Err(status) => status,
    }
}

/// Drop every sample queued for `device` without delivering it.
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

/// Reconnect to the daemon, keeping the subscriptions.
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

/// Accepted and a no-op: samples already carry the device clock, and the
/// daemon keeps the device/host clock pair current by itself.
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

/// The host clock `timestamp_system_us` in gaze data is taken from:
/// microseconds since the Unix epoch.
///
/// # Safety
/// `api` must be null or a live handle; `timestamp_us` must be null or valid
/// for writing one `int64_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_system_clock(api: *mut Api, timestamp_us: *mut i64) -> Status {
    if api.is_null() || timestamp_us.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX));
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { timestamp_us.write(now) };
    TOBII_ERROR_NO_ERROR
}

/// Fetch (once) the device's identity from the daemon.
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

/// A boolean state. Power save, remote wake, paused, exclusive mode and the
/// fault/warning flags are always false here; calibration-active comes from
/// the daemon.
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

/// A `u32` state: only the calibration id.
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

/// Streams this library delivers.
const fn stream_supported(stream: u32) -> bool {
    matches!(
        stream,
        TOBII_STREAM_GAZE_POINT
            | TOBII_STREAM_GAZE_ORIGIN
            | TOBII_STREAM_EYE_POSITION_NORMALIZED
            | TOBII_STREAM_USER_PRESENCE
            | TOBII_STREAM_HEAD_POSE
            | TOBII_STREAM_GAZE_DATA
    )
}

/// Shared shape of the two `*_supported` queries: an unknown value is "not
/// supported", not an error (as in the DLL).
///
/// # Safety
/// As `tobii_capability_supported`.
unsafe fn write_supported(
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
/// presence, head pose and gaze data.
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
    use crate::types::TOBII_STATE_EXCLUSIVE_MODE;
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
    fn the_system_clock_moves_forward() {
        let api = api();
        let (mut a, mut b) = (0i64, 0i64);
        // SAFETY: `api` is live; `a`/`b` are live locals.
        unsafe {
            assert_eq!(tobii_system_clock(api, &raw mut a), 0);
            assert_eq!(tobii_system_clock(api, &raw mut b), 0);
            assert_eq!(tobii_api_destroy(api), 0);
        }
        assert!(a > 1_600_000_000_000_000 && b >= a);
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
            assert_eq!(
                tobii_stream_supported(d, TOBII_STREAM_GAZE_DATA, &raw mut s),
                0
            );
            assert_eq!(s, TOBII_SUPPORTED);
            assert_eq!(tobii_stream_supported(d, 9, &raw mut s), 0);
            assert_eq!(s, TOBII_NOT_SUPPORTED);
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
            let mut s: StateString = [1; 512];
            assert_eq!(tobii_get_state_string(d, TOBII_STATE_FAULT, &raw mut s), 0);
            assert_eq!(s[0], 0);
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }
}
