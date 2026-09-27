//! `tobii.h`: API and device lifetime, callbacks, device information, states
//! and capabilities.

use std::ffi::{c_char, c_void};
use std::time::Duration;

use tobii_ipc::request::{self, decode_track_box, kind};

use crate::device::{Api, Device, device_ref, in_callback};
use crate::logger::{self, Level, Logger};
use crate::status::{
    Status, TOBII_ERROR_CALLBACK_IN_PROGRESS, TOBII_ERROR_CONFLICTING_API_INSTANCES,
    TOBII_ERROR_CONNECTION_FAILED, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
    TOBII_ERROR_NOT_SUPPORTED, TOBII_ERROR_TIMED_OUT,
};
use crate::timeouts;
use crate::types::{
    CustomAlloc, CustomLog, DeviceInfo, DeviceUrlReceiver, FieldOfUse, StateString,
    TOBII_CAPABILITY_CALIBRATION_2D, TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_XY,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_Z, TOBII_CAPABILITY_DISPLAY_AREA_WRITABLE,
    TOBII_FIELD_OF_USE_ANALYTICAL, TOBII_FIELD_OF_USE_INTERACTIVE, TOBII_NOT_SUPPORTED,
    TOBII_STATE_BOOL_FALSE, TOBII_STATE_CALIBRATION_ACTIVE, TOBII_STATE_CALIBRATION_ID,
    TOBII_STATE_DEVICE_PAUSED, TOBII_STATE_FAULT, TOBII_STATE_WARNING,
    TOBII_STREAM_EYE_POSITION_NORMALIZED, TOBII_STREAM_GAZE_DATA, TOBII_STREAM_GAZE_ORIGIN,
    TOBII_STREAM_GAZE_POINT, TOBII_STREAM_HEAD_POSE, TOBII_STREAM_USER_POSITION_GUIDE,
    TOBII_STREAM_USER_PRESENCE, TOBII_SUPPORTED, TrackBox, Version, copy_c_bytes, copy_c_string,
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

/// Create the API handle. Its arguments are checked in the DLL's order
/// (0x180144abc..0x180144ae7): a null `api`, a `custom_alloc` without
/// `malloc_func` or `free_func`, then a `custom_log` without `log_func` is
/// `TOBII_ERROR_INVALID_PARAMETER`, and `*api` is left as it was.
///
/// The logger is copied into the handle, as the DLL copies the struct, and
/// hears libtobii's own diagnostics (see `logger`); with `custom_log` null
/// nothing is logged. The allocator is never called: libtobii allocates with
/// Rust's global allocator, which cannot be switched per handle, so
/// `TOBII_ERROR_ALLOCATION_FAILED` never comes from it. (The DLL allocates
/// through it: the handle (0x180144b09), each device (0x180157a3c), more
/// through the sub-libraries it hands it to (0x180144c37), and a log line of
/// 255 characters or more (0x18015e3dd).)
///
/// # Safety
/// `api` must be null or valid for writing one `*mut Api`. The handle written
/// there must be released with `tobii_api_destroy`. `custom_alloc` and
/// `custom_log` must each be null or valid for reading one struct during the
/// call. A `log_func` must be sound to call with `log_context`, any
/// `tobii_log_level_t` and a NUL-terminated string valid for the call, on any
/// thread that calls into the handle or a device created from it, from
/// several at once, until the handle and every such device are destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_api_create(
    api: *mut *mut Api,
    custom_alloc: *const CustomAlloc,
    custom_log: *const CustomLog,
) -> Status {
    if api.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: null or readable during the call, per the contract.
    if let Some(alloc) = unsafe { custom_alloc.as_ref() }
        && (alloc.malloc_func.is_none() || alloc.free_func.is_none())
    {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: null or readable during the call, per the contract.
    let logger = match unsafe { Logger::from_c(custom_log) } {
        Ok(logger) => logger,
        Err(status) => return status,
    };
    let handle = Box::into_raw(Box::new(Api::new(logger)));
    // SAFETY: `api` is non-null (checked above) and the caller guarantees it
    // is valid for a write of one pointer.
    unsafe { api.write(handle) };
    TOBII_ERROR_NO_ERROR
}

/// Release an API handle. A null handle is `TOBII_ERROR_INVALID_PARAMETER`,
/// then a call from inside a callback `TOBII_ERROR_CALLBACK_IN_PROGRESS`, in
/// the DLL's order; neither releases anything. As in the DLL, devices created
/// from the handle are not checked for; here they keep working, and keep
/// logging to the handle's logger, which each copied when it was created.
///
/// # Safety
/// `api` must be null or a handle from `tobii_api_create` that has not been
/// destroyed yet and that no other thread is inside a call on; once this
/// returns `TOBII_ERROR_NO_ERROR` no thread may use it again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_api_destroy(api: *mut Api) -> Status {
    if api.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    // SAFETY: non-null (checked above), and the caller guarantees it came
    // from `Box::into_raw` in `tobii_api_create`, is destroyed only once, and
    // that no other thread is inside a call on it or uses it later.
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
/// inside a callback (after the arguments, as in the DLL) and connect. A
/// refused `field_of_use` is logged at ERROR, as in the DLL (0x1801442b3), and
/// so is a failed connect; a connected device takes the API's logger, and
/// logs so at INFO.
///
/// # Safety
/// `api` must be null or a live handle from `tobii_api_create`; `device` must
/// be null or valid for writing one `*mut Device`.
pub(crate) unsafe fn create_device(
    api: *mut Api,
    field_of_use: FieldOfUse,
    device: *mut *mut Device,
) -> Status {
    if api.is_null() || device.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null (checked above), and live per the contract.
    let logger = unsafe { &*api }.logger;
    if !matches!(
        field_of_use,
        TOBII_FIELD_OF_USE_INTERACTIVE | TOBII_FIELD_OF_USE_ANALYTICAL
    ) {
        logger::emit(
            logger,
            Level::Error,
            format_args!(
                "refused field_of_use {field_of_use}: it must be 1 or 2 (a caller built \
                 against the 3-argument Stream Engine header lands here)"
            ),
        );
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // A callback could not destroy the device it made.
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    let mut d = match Device::connect_daemon(api as usize, field_of_use) {
        Ok(d) => d,
        Err(e) => {
            logger::emit(
                logger,
                Level::Error,
                format_args!(
                    "could not open a connection to tobiid ({}): {e}",
                    tobii_ipc::socket_path().display()
                ),
            );
            return TOBII_ERROR_CONNECTION_FAILED;
        }
    };
    d.adopt(logger);
    let handle = Box::into_raw(Box::new(d));
    // SAFETY: `device` is non-null (checked above) and the caller guarantees
    // it is valid for a write of one pointer.
    unsafe { device.write(handle) };
    TOBII_ERROR_NO_ERROR
}

/// Connect to the daemon (spawning it if needed) and create a device handle.
/// `url` is accepted for signature compatibility and ignored: there is one
/// device, owned by the daemon. Threads that create devices at once while no
/// daemon runs spawn one between them: the others wait for that spawn, then
/// connect to its daemon or fail as it did.
///
/// `field_of_use` is validated exactly as the Stream Engine validates it, and
/// that check is load-bearing: a caller compiled against the three-argument
/// Stream Engine 3.x header lands here with a stack address in `field_of_use`
/// and an uninitialised `device`, and is rejected instead of corrupting memory.
/// A call from inside a callback is `TOBII_ERROR_CALLBACK_IN_PROGRESS` once
/// the arguments have been checked, as in the DLL. The device logs to
/// `api`'s logger (see `tobii_api_create`).
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
/// anything. It takes none of the device's locks, as the DLL's destroy takes
/// none (0x18015a320): the caller keeps every other thread out.
///
/// # Safety
/// `device` must be null or a handle from `tobii_device_create` that has not
/// been destroyed yet and that no other thread is inside a call on; once
/// this returns `TOBII_ERROR_NO_ERROR` no thread may use it again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_destroy(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    // SAFETY: non-null (checked above), and the caller guarantees it came
    // from `Box::into_raw` in `tobii_device_create`, is destroyed only once,
    // and that no other thread is inside a call on it or uses it later; no
    // callback runs on this thread, so no dispatch loop of its own holds it.
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
/// The devices are waited on one after another, each holding none of its
/// locks while it sleeps. A device another thread is dispatching is waited
/// on as usual, on its doorbell, until that thread leaves it something to
/// process; the DLL instead skips a device whose process mutex another
/// thread holds, and with nothing else to wait on returns
/// `TOBII_ERROR_NO_ERROR` at once (its wait at 0x1801596e0: the try-enter at
/// 0x18000e940, read from the code, not observed), which would spin a
/// wait-and-process loop.
///
/// # Safety
/// `devices` must be null or point to `device_count` initialised
/// `*mut Device` values, each null or a live handle from
/// `tobii_device_create` that is not destroyed before the call returns.
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
    // Before any device is read: a callback runs under its device's locks,
    // and may call into no device.
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
        // SAFETY: the caller guarantees each handle is live until this
        // returns; the reference is dropped before the next iteration.
        match unsafe { device_ref(handle) } {
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
/// The callbacks run one at a time per device, on whichever thread calls
/// this. A call made while another thread holds the device's dispatch (a
/// process running the callbacks, a look of `tobii_wait_for_callbacks`, a
/// clear, or a reconnect waiting for its new connection's ack) returns at
/// once and delivers nothing, what is queued staying for the next call:
/// `TOBII_ERROR_NO_ERROR`, or `TOBII_ERROR_CONNECTION_FAILED` once the loss
/// has been reported, until a reconnect. The DLL's returns
/// `TOBII_ERROR_NO_ERROR` when another thread holds its process mutex
/// (0x18000e9d9..0x18000e9ea), even after a loss, once it has delivered the
/// device's queued notifications on this thread (0x180159515..0x180159566).
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that is
/// not destroyed before the call returns. The callbacks registered on it are
/// invoked under the contracts stated on the `tobii_*_subscribe` functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_process_callbacks(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    match unsafe { device_ref(device) } {
        Ok(d) => d.process(),
        Err(status) => status,
    }
}

/// Drop every sample queued for `device` without delivering it. A lost
/// daemon connection is still reported by the next process call. While
/// another thread is dispatching the device, or reconnecting it, this waits
/// for it to finish; it never waits for a request. The DLL's waits for
/// requests and reconnects, under its API mutex (0x180143a85), and while
/// another thread processes it clears only the device's queued
/// notifications (its try-enter at 0x18000e9d9, through 0x180158a20,
/// fails), leaving the rest.
///
/// # Safety
/// As `tobii_device_process_callbacks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_clear_callback_buffers(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    match unsafe { device_ref(device) } {
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
/// On a device shared between threads it first waits for the requests and
/// subscription changes other threads have under way or queued ahead of it,
/// then for a dispatch another thread runs (the ~500 ms count from then),
/// before it asks for the subscriptions back. Until it has swapped connections
/// or failed, a `tobii_device_process_callbacks` on another thread then
/// returns at once, delivering nothing: tobiid sends the new connection what
/// it sends the old one from its ack on, so delivering from the old one
/// meanwhile would deliver those samples twice, their stamps stepping back.
///
/// Of what the old connection brought and was not delivered, the samples are
/// dropped, so a reconnect of a live connection may lose those tobiid sent
/// the old one alone, just before it took the new one's subscriptions. The
/// notifications are kept, and delivered ahead of the new connection's:
/// tobiid sends each once and does not repeat it to a new connection, so
/// dropping one would leave the application with a stale state (a
/// calibration, a pause, the faults) until the next change. One tobiid sent
/// both connections may be delivered again, after later ones, so a state may
/// be seen to step back before it settles: the last delivered is current.
///
/// # Safety
/// As `tobii_device_process_callbacks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_reconnect(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    match unsafe { device_ref(device) } {
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
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    match unsafe { device_ref(device) } {
        Ok(_) => TOBII_ERROR_NO_ERROR,
        Err(status) => status,
    }
}

/// The host clock, `CLOCK_MONOTONIC` in microseconds
/// ([`tobii_ipc::host_clock_us`], which the daemon reads too): the clock of
/// every callback timestamp (bar gaze data's and raw gaze's
/// `timestamp_tracker_us`) and of `tobii_timesync`'s host times. Its epoch is
/// undefined, as that of the DLL's `QueryPerformanceCounter` is.
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
    copy_c_string(&mut c.integration_type, &info.integration_type);
    // integration_id, hw_calibration_version, hw_calibration_date and
    // lot_id stay empty: for a tracker it drives over USB the DLL's
    // built-in tracker module never passes them on (`platmod_start`,
    // 0x18016ac30), so the DLL returns them empty too.
    copy_c_string(
        &mut c.runtime_build_version,
        concat!("libtobii.so ", env!("CARGO_PKG_VERSION")),
    );
    c
}

/// The device's serial, model, generation, firmware and integration type,
/// as the tracker reports them; `runtime_build_version` names libtobii.so.
/// `integration_id`, `hw_calibration_version`, `hw_calibration_date` and
/// `lot_id` are empty, as the DLL leaves them for a tracker it drives over
/// USB. From a daemon that predates it, the integration type is empty too.
/// Fetched once per connection; a read of the one kept waits, too, for the
/// requests other threads have under way on the device or queued ahead of it.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `device_info` must be null or
/// valid for writing one `tobii_device_info_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_device_info(
    device: *mut Device,
    device_info: *mut DeviceInfo,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if device_info.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d.device_info() {
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
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if track_box.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let b = match d.request(kind::TRACK_BOX, &[], timeouts::FACTS) {
        Ok(p) => match decode_track_box(&p) {
            Some(b) => b,
            None => return d.malformed("track box"),
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
fn query_state(d: &Device, state: u32) -> Result<Vec<u8>, Status> {
    d.request(kind::STATE, &request::encode_u32(state), timeouts::STATE)
}

/// A boolean state. Power save, remote wake, exclusive mode and the
/// fault/warning flags are always false here; paused and calibration-active
/// come from the daemon. The DLL refuses FAULT and WARNING as bool states
/// (`TOBII_ERROR_INVALID_PARAMETER`, its state map at 0x180142ba0); they are
/// string states (`tobii_get_state_string`).
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
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
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
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
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
        Ok(None) => d.malformed("calibration id"),
        Err(status) => status,
    }
}

/// A string state: the tracker's fault or warning list ("ok" when there are
/// none), as its last init reported them (status strings 5 and 6 of command
/// 1490) or a notification 3200 or 3210 has replaced them since, from the
/// daemon.
///
/// The DLL (0x1801422c0) copies them from a cache that its create and
/// reconnect fill from 1490 (0x18016db40, through `tracker_get_status` at
/// 0x1801a0ac0), and that the tracker's notifications 3200 and 3210 update
/// (0x18016efa0); the daemon follows those the same way, and passes them on
/// as `TOBII_NOTIFICATION_TYPE_FAULTS_CHANGED` and `_WARNINGS_CHANGED` (a
/// daemon older than that answers the init's list and sends neither). Up
/// to the first NUL and at most 511 bytes are copied, and all 512 written,
/// as the DLL's `strncpy(value, .., 0x200)` and forced NUL at `[0x1ff]` do;
/// the DLL's own copy of a 1490 string holds at most 119 bytes (its TTP
/// record, 0x18017c3a4, inferred to be the 1490 path), where a longer one
/// is passed on here. `TOBII_ERROR_NOT_SUPPORTED` when the init reported no
/// such list, whatever notifications came since (they never set the DLL's
/// "present" flag, which only a 1490 sets: 0x18016dce1, 0x18016dd8b), as
/// the DLL, and from a daemon too old to know these states. Unlike the DLL,
/// which answers from its cache, `TOBII_ERROR_TIMED_OUT` if the daemon has
/// not seen a tracker yet (the call waits for its first init) and
/// `TOBII_ERROR_CONNECTION_FAILED` when the daemon is gone. On any error
/// `value` is left untouched.
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
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if value.is_null() || !matches!(state, TOBII_STATE_FAULT | TOBII_STATE_WARNING) {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // The daemon may wait for the tracker's first init: the facts timeout.
    let bytes = match d.request(kind::STATE, &request::encode_u32(state), timeouts::FACTS) {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    let text = bytes
        .iter()
        .position(|&b| b == 0)
        .map_or(&bytes[..], |end| &bytes[..end]);
    let mut out: StateString = [0; 512];
    copy_c_bytes(&mut out, text);
    // SAFETY: non-null, and the caller guarantees 512 writable bytes.
    unsafe { value.write(out) };
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
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
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
    use crate::logger::tests::Recorder;
    use crate::status::TOBII_ERROR_INTERNAL;
    use crate::types::{
        CalibrationPointData, EyePair, EyePairFn, LicenseKey, TOBII_LOG_LEVEL_ERROR,
        TOBII_LOG_LEVEL_INFO, TOBII_STATE_EXCLUSIVE_MODE,
    };
    use std::cell::Cell;
    use std::ptr;

    fn api() -> *mut Api {
        let mut api: *mut Api = ptr::null_mut();
        // SAFETY: `api` is a live local valid for one pointer write.
        let status = unsafe { tobii_api_create(&raw mut api, ptr::null(), ptr::null()) };
        assert_eq!(status, 0);
        api
    }

    /// An API handle whose logger records into `recorder`, which must
    /// outlive it and every device created from it.
    fn api_logging_to(recorder: &Recorder) -> *mut Api {
        let log = recorder.custom_log();
        let mut api: *mut Api = ptr::null_mut();
        // SAFETY: `api` and `log` are live locals.
        let status = unsafe { tobii_api_create(&raw mut api, ptr::null(), &raw const log) };
        assert_eq!(status, 0);
        api
    }

    // The `counting_*` allocator functions count their calls in the
    // `Cell<u32>` the tests pass as `mem_context`, and allocate nothing.

    unsafe extern "C" fn counting_malloc(context: *mut c_void, _size: usize) -> *mut c_void {
        // SAFETY: `context` is a live `Cell<u32>` (see above).
        let calls = unsafe { &*context.cast::<Cell<u32>>() };
        calls.set(calls.get() + 1);
        ptr::null_mut()
    }

    unsafe extern "C" fn counting_free(context: *mut c_void, _ptr: *mut c_void) {
        // SAFETY: `context` is a live `Cell<u32>` (see above).
        let calls = unsafe { &*context.cast::<Cell<u32>>() };
        calls.set(calls.get() + 1);
    }

    /// As in the DLL (0x180144abc..0x180144ae7), an allocator without either
    /// function or a logger without `log_func` is refused, and `*api` left as
    /// it was; the allocator is never called, from the create to the destroy.
    #[test]
    fn api_create_checks_custom_alloc_and_custom_log_as_the_dll() {
        let (calls, recorder) = (Cell::new(0u32), Recorder::default());
        let alloc = CustomAlloc {
            mem_context: ptr::from_ref(&calls).cast_mut().cast(),
            malloc_func: Some(counting_malloc),
            free_func: Some(counting_free),
        };
        let no_malloc = CustomAlloc {
            malloc_func: None,
            ..alloc
        };
        let no_free = CustomAlloc {
            free_func: None,
            ..alloc
        };
        let log = recorder.custom_log();
        let no_log_func = CustomLog {
            log_func: None,
            ..log
        };
        let untouched = ptr::NonNull::<Api>::dangling().as_ptr();
        let mut api = untouched;
        let mut device: *mut Device = ptr::null_mut();
        // SAFETY: `api`, `device` and the structs are live locals; a refused
        // create writes nothing, and the handle made is destroyed once.
        unsafe {
            for (a, l) in [
                (&raw const no_malloc, ptr::null()),
                (&raw const no_free, ptr::null()),
                (ptr::null(), &raw const no_log_func),
                (&raw const alloc, &raw const no_log_func),
            ] {
                assert_eq!(
                    tobii_api_create(&raw mut api, a, l),
                    TOBII_ERROR_INVALID_PARAMETER
                );
                assert_eq!(api, untouched);
            }
            assert_eq!(
                tobii_api_create(ptr::null_mut(), &raw const alloc, &raw const log),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_api_create(&raw mut api, &raw const alloc, &raw const log),
                TOBII_ERROR_NO_ERROR
            );
            assert_ne!(api, untouched);
            assert_eq!(
                tobii_device_create(api, ptr::null(), 0, &raw mut device),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_api_destroy(api), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(calls.get(), 0, "the allocator is never called");
        assert_eq!(recorder.lines().len(), 1, "the refused field_of_use");
    }

    /// A refused `field_of_use` and a failed connect are logged at ERROR, from
    /// either constructor; a null argument is not, its status says it all.
    #[test]
    fn device_create_failures_reach_the_application_logger() {
        let recorder = Recorder::default();
        let api = api_logging_to(&recorder);
        let mut device: *mut Device = ptr::null_mut();
        let mut results = [99u32];
        let key = LicenseKey {
            license_key: ptr::null(),
            size_in: 0,
        };
        // SAFETY: `api` is live and destroyed once; the rest are live
        // locals. Nothing reaches a daemon: under test, connecting fails.
        unsafe {
            assert_eq!(
                tobii_device_create(api, ptr::null(), 0, &raw mut device),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                crate::licensing::tobii_device_create_ex(
                    api,
                    ptr::null(),
                    3,
                    &raw const key,
                    1,
                    results.as_mut_ptr(),
                    &raw mut device,
                ),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_device_create(api, ptr::null(), 1, &raw mut device),
                TOBII_ERROR_CONNECTION_FAILED
            );
            assert_eq!(
                tobii_device_create(ptr::null_mut(), ptr::null(), 0, &raw mut device),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_device_create(api, ptr::null(), 0, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_api_destroy(api), TOBII_ERROR_NO_ERROR);
        }
        assert!(device.is_null());
        assert_eq!(results, [99], "no licence result was written");
        let lines = recorder.lines();
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines
                .iter()
                .all(|(level, _)| *level == TOBII_LOG_LEVEL_ERROR)
        );
        assert!(lines[0].1.contains("field_of_use 0"), "{lines:?}");
        assert!(lines[1].1.contains("field_of_use 3"), "{lines:?}");
        assert!(
            lines[2]
                .1
                .starts_with("could not open a connection to tobiid ("),
            "{lines:?}"
        );
    }

    /// Every undecodable reply goes through `Device::malformed`: one is
    /// enough to show it is logged and `TOBII_ERROR_INTERNAL`.
    #[test]
    fn a_reply_that_does_not_decode_is_logged_and_internal() {
        let recorder = Recorder::default();
        let mut d = crate::device::tests::device_with(0, vec![0xff]);
        d.set_logger(recorder.logger());
        let d = Box::into_raw(Box::new(d));
        let mut track_box = TrackBox::default();
        // SAFETY: `d` is live and destroyed once; `track_box` a live local.
        unsafe {
            assert_eq!(
                tobii_get_track_box(d, &raw mut track_box),
                TOBII_ERROR_INTERNAL
            );
            assert_eq!(tobii_device_destroy(d), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(track_box, TrackBox::default(), "nothing written");
        assert_eq!(
            recorder.lines(),
            [(
                TOBII_LOG_LEVEL_ERROR,
                "tobiid sent a track box reply that does not decode".to_owned()
            )]
        );
    }

    /// `tobii_device_create` hands the device it connects its API's logger,
    /// which hears the connect at INFO, once. The device keeps the logger
    /// once the API is destroyed, as `tobii_api_create`'s contract says: its
    /// daemon hanging up then still reaches it.
    #[test]
    fn a_created_device_logs_to_its_apis_logger_even_once_the_api_is_gone() {
        let recorder = Recorder::default();
        let api = api_logging_to(&recorder);
        let (connect, daemons) = crate::device::tests::scripted_daemon(vec![]);
        crate::device::tests::DAEMON.set(Some(connect));
        let mut device: *mut Device = ptr::null_mut();
        // SAFETY: `api` is live until it is destroyed, once, here; `device`
        // is a live local.
        unsafe {
            assert_eq!(
                tobii_device_create(api, ptr::null(), 1, &raw mut device),
                TOBII_ERROR_NO_ERROR
            );
            assert_eq!(tobii_api_destroy(api), TOBII_ERROR_NO_ERROR);
        }
        assert_eq!(
            recorder.lines(),
            [(TOBII_LOG_LEVEL_INFO, "connected to tobiid".to_owned())]
        );
        drop(daemons.recv().expect("the daemon's end"));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        // SAFETY: `device` is live and destroyed once; `recorder` outlives it.
        unsafe {
            while tobii_wait_for_callbacks(1, &raw const device) != TOBII_ERROR_NO_ERROR {
                assert!(std::time::Instant::now() < deadline, "the hang-up is seen");
            }
            assert_eq!(
                tobii_device_process_callbacks(device),
                TOBII_ERROR_CONNECTION_FAILED
            );
            assert_eq!(tobii_device_destroy(device), TOBII_ERROR_NO_ERROR);
        }
        let lines = recorder.lines();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[1].0, TOBII_LOG_LEVEL_ERROR);
        assert!(lines[1].1.starts_with("lost the connection"), "{lines:?}");
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

    /// Run [`create`] on two samples, calling into `api`, then check that
    /// every call was refused or rejected before it did anything.
    fn assert_a_callback_creates_nothing(api: *mut Api) {
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

    /// The DLL refuses these once their arguments check out; a callback
    /// could not destroy a device it made.
    #[test]
    fn a_callback_cannot_create_a_device_or_parse_a_calibration() {
        assert_a_callback_creates_nothing(api());
    }

    /// The refused `field_of_use` is logged from inside the callback, and
    /// the guard must outlive the line: were it cleared, the
    /// `tobii_device_create_ex` after it would try to connect.
    #[test]
    fn a_callback_cannot_create_a_device_through_a_logging_api_either() {
        let recorder = Recorder::default();

        assert_a_callback_creates_nothing(api_logging_to(&recorder));

        let lines = recorder.lines();
        assert_eq!(lines.len(), 2, "one per sample: {lines:?}");
        assert!(
            lines
                .iter()
                .all(|(level, text)| *level == TOBII_LOG_LEVEL_ERROR
                    && text.contains("field_of_use 0")),
            "{lines:?}"
        );
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

    /// A C string field's bytes up to its NUL.
    fn c_bytes(s: &[c_char]) -> Vec<u8> {
        s.iter()
            .take_while(|c| **c != 0)
            .map(|c| c.to_ne_bytes()[0])
            .collect()
    }

    /// `tobii_get_device_info` against a daemon that answers `payload`.
    fn device_info_from(payload: Vec<u8>) -> (Status, DeviceInfo) {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, payload)));
        let mut info = device_info_c(&request::DeviceInfo::default());
        // SAFETY: `d` is live and destroyed once; `info` a live local.
        let status = unsafe {
            let status = tobii_get_device_info(d, &raw mut info);
            assert_eq!(tobii_device_destroy(d), 0);
            status
        };
        (status, info)
    }

    #[test]
    fn device_info_is_fetched_once_and_filled_in() {
        let payload = request::encode_device_info(&request::DeviceInfo {
            serial_number: "IS50F-000000000000".into(),
            model: "IS5_Large_Eyetracker_5".into(),
            generation: "IS5".into(),
            firmware_version: "02a1a6a977".into(),
            integration_type: "Peripheral".into(),
        });

        let (status, info) = device_info_from(payload);

        assert_eq!(status, 0);
        assert_eq!(c_bytes(&info.model), b"IS5_Large_Eyetracker_5");
        assert_eq!(c_bytes(&info.integration_type), b"Peripheral");
        for (name, field) in [
            ("integration_id", &info.integration_id),
            ("hw_calibration_version", &info.hw_calibration_version),
            ("hw_calibration_date", &info.hw_calibration_date),
            ("lot_id", &info.lot_id),
        ] {
            assert_eq!(c_bytes(field), b"", "{name} stays empty, as in the DLL");
        }
        assert_eq!(
            c_bytes(&info.runtime_build_version),
            concat!("libtobii.so ", env!("CARGO_PKG_VERSION")).as_bytes()
        );
    }

    #[test]
    fn device_info_from_an_older_daemon_leaves_the_integration_type_empty() {
        let mut payload = request::encode_device_info(&request::DeviceInfo {
            model: "IS5_Large_Eyetracker_5".into(),
            ..request::DeviceInfo::default()
        });
        // Drop the empty tail's length: the four strings an older daemon
        // sends.
        payload.truncate(payload.len() - 2);

        let (status, info) = device_info_from(payload);

        assert_eq!(status, 0);
        assert_eq!(c_bytes(&info.model), b"IS5_Large_Eyetracker_5");
        assert_eq!(c_bytes(&info.integration_type), b"");
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
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    /// Requests a fake daemon was sent, as `(kind, payload)`.
    type Requests = std::sync::Arc<std::sync::Mutex<Vec<(u8, Vec<u8>)>>>;

    /// A device on a fake daemon that answers every request with
    /// `status`/`payload`, and the requests it is sent.
    fn recording_device(status: u8, payload: Vec<u8>) -> (*mut Device, Requests) {
        let seen = Requests::default();
        let log = std::sync::Arc::clone(&seen);
        let connect = crate::device::tests::fake_daemon(move |body| {
            let req = request::decode_request(body).expect("request");
            log.lock()
                .expect("log")
                .push((req.kind, req.payload.to_vec()));
            vec![tobii_ipc::encode_reply(req.id, status, &payload)]
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        (d, seen)
    }

    /// The paused state a daemon answering `status`/`payload` gives. It must
    /// be asked for with STATE 2.
    fn paused_state(status: u8, payload: Vec<u8>) -> (Status, u32) {
        let (d, seen) = recording_device(status, payload);
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

    /// What `tobii_get_state_string(state)` gives on a daemon answering
    /// `status`/`payload`: its status, the buffer (filled with 1s before the
    /// call) and the requests sent.
    fn state_string(
        state: u32,
        status: u8,
        payload: &[u8],
    ) -> (Status, StateString, Vec<(u8, Vec<u8>)>) {
        let (d, seen) = recording_device(status, payload.to_vec());
        let mut s: StateString = [1; 512];
        // SAFETY: `d` is live and destroyed once; `s` a live local.
        let got = unsafe {
            let got = tobii_get_state_string(d, state, &raw mut s);
            assert_eq!(tobii_device_destroy(d), 0);
            got
        };
        let requests = seen.lock().expect("log").clone();
        (got, s, requests)
    }

    fn bytes(s: &StateString) -> Vec<u8> {
        s.iter().map(|c| c.to_ne_bytes()[0]).collect()
    }

    /// `text` then zeros, as a 512-byte state string.
    fn padded(text: &[u8]) -> Vec<u8> {
        let mut v = text.to_vec();
        v.resize(512, 0);
        v
    }

    #[test]
    fn the_fault_and_warning_strings_come_from_the_daemon() {
        use tobii_ipc::request::state;
        let (got, s, requests) = state_string(TOBII_STATE_FAULT, 0, b"ok");
        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        assert_eq!(bytes(&s), padded(b"ok"), "all 512 bytes written");
        assert_eq!(requests, [(kind::STATE, request::encode_u32(state::FAULT))]);

        let (got, s, requests) = state_string(TOBII_STATE_WARNING, 0, b"W_A,W_B");
        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        assert_eq!(bytes(&s), padded(b"W_A,W_B"));
        assert_eq!(
            requests,
            [(kind::STATE, request::encode_u32(state::WARNING))]
        );

        // Up to the first NUL, as strncpy.
        let (got, s, _) = state_string(TOBII_STATE_FAULT, 0, b"a\0b");
        assert_eq!((got, bytes(&s)), (TOBII_ERROR_NO_ERROR, padded(b"a")));
    }

    #[test]
    fn a_long_state_string_is_cut_to_511_bytes() {
        let (got, s, _) = state_string(TOBII_STATE_FAULT, 0, &[b'x'; 600]);
        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        let s = bytes(&s);
        assert!(s[..511].iter().all(|&b| b == b'x'));
        assert_eq!(s[511], 0);
    }

    #[test]
    fn a_state_string_the_daemon_lacks_leaves_the_value_untouched() {
        for (status, want, what) in [
            (
                tobii_ipc::request::status::NOT_SUPPORTED,
                TOBII_ERROR_NOT_SUPPORTED,
                "an older daemon, or an init without the list",
            ),
            (
                tobii_ipc::request::status::TIMED_OUT,
                TOBII_ERROR_TIMED_OUT,
                "no init yet",
            ),
        ] {
            let (got, s, requests) = state_string(TOBII_STATE_WARNING, status, &[]);
            assert_eq!(got, want, "{what}");
            assert_eq!(s, [1; 512], "{what}");
            assert_eq!(requests.len(), 1, "{what}");
        }
    }

    #[test]
    fn other_string_states_are_refused_unasked() {
        for state in [
            0,
            TOBII_STATE_DEVICE_PAUSED,
            TOBII_STATE_CALIBRATION_ID,
            TOBII_STATE_CALIBRATION_ACTIVE,
            8,
        ] {
            let (got, s, requests) = state_string(state, 0, b"ok");
            assert_eq!(got, TOBII_ERROR_INVALID_PARAMETER, "state {state}");
            assert_eq!(s, [1; 512], "state {state}");
            assert_eq!(requests, [], "state {state}");
        }
        let (d, seen) = recording_device(0, b"ok".to_vec());
        // SAFETY: `d` is live and destroyed once; a null value is refused.
        unsafe {
            assert_eq!(
                tobii_get_state_string(d, TOBII_STATE_FAULT, std::ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_device_destroy(d), 0);
        }
        assert_eq!(*seen.lock().expect("log"), []);
    }

    /// STATE 4 and 5 now carry the lists: a bool read of them must not ask,
    /// or "ok" would read as TRUE.
    #[test]
    fn the_fault_and_warning_bools_are_not_asked() {
        let (d, seen) = recording_device(0, b"ok".to_vec());
        let mut v = 7u32;
        // SAFETY: `d` is live and destroyed once; `v` a live local.
        unsafe {
            for state in [TOBII_STATE_FAULT, TOBII_STATE_WARNING] {
                assert_eq!(tobii_get_state_bool(d, state, &raw mut v), 0);
                assert_eq!(v, TOBII_STATE_BOOL_FALSE);
            }
            assert_eq!(tobii_device_destroy(d), 0);
        }
        assert_eq!(*seen.lock().expect("log"), []);
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
