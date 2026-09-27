//! The 73 exports of `tobii_stream_engine.dll` 4.1.0.3 that its documentation
//! does not cover (all but `tobii_calibration_stimulus_points_get`, which
//! lives with calibration). Their argument counts come from the DLL (see
//! `tools/abi`); their types are best-effort, which is safe because only the
//! field-of-use, image, raw gaze, internal-stream, internal-capability,
//! timesync, stream-type, pause and hardware-configuration functions below
//! read their arguments, and the subscribes and unsubscribes of the internal
//! streams the DLL refuses only compare theirs with null.

use std::ffi::c_void;

use tobii_ipc::request::{
    self, decode_hardware_configuration, decode_stream_types, decode_timesync, kind,
};

use crate::api::write_supported;
use crate::calibration::request;
use crate::device::{Device, device_ref, in_callback};
use crate::status::{
    Status, TOBII_ERROR_CALLBACK_IN_PROGRESS, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
    TOBII_ERROR_NOT_SUPPORTED,
};
use crate::streams::{subscribe, unsubscribe};
use crate::stub::not_supported;
use crate::timeouts;
use crate::types::{
    FieldOfUse, FieldOfUseFn, GazeRawFn, HardwareConfiguration, HardwareConfigurationEntry,
    ImageFn, StreamType, StreamTypeReceiver, TimesyncData, copy_c_string,
};

/// The field of use the device was created with.
///
/// # Safety
/// `device` must be null or a live handle that is not destroyed before the
/// call returns; `field_of_use` must be null or valid for writing one
/// `tobii_field_of_use_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_field_of_use(
    device: *mut Device,
    field_of_use: *mut FieldOfUse,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if field_of_use.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { field_of_use.write(d.field_of_use) };
    TOBII_ERROR_NO_ERROR
}

/// Register a field-of-use callback. It is never called: the field of use is
/// fixed when the device is created.
///
/// # Safety
/// As the subscribe functions in `tobii_streams.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_field_of_use_subscribe(
    device: *mut Device,
    callback: Option<FieldOfUseFn>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { subscribe(device, |c| &mut c.field_of_use, callback, user_data) }
}

/// Undo `tobii_field_of_use_subscribe`.
///
/// # Safety
/// `device` must be null or a live handle that is not destroyed before the
/// call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_field_of_use_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { unsubscribe(device, |c| &mut c.field_of_use) }
}

/// The tracker's IR camera: 280x280, 8 bits per pixel, ~33 Hz (~2.6 MB/s).
/// `tobii_image_t.data` is valid only during the callback.
///
/// # Safety
/// As the subscribe functions in `tobii_streams.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_image_subscribe(
    device: *mut Device,
    callback: Option<ImageFn>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { subscribe(device, |c| &mut c.image, callback, user_data) }
}

/// Undo `tobii_image_subscribe`.
///
/// # Safety
/// `device` must be null or a live handle that is not destroyed before the
/// call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_image_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { unsubscribe(device, |c| &mut c.image) }
}

/// The Stream Engine's raw gaze: its own record of each gaze frame,
/// `tobii_gaze_raw_t` (see `tobii_internal.h`), ~33 Hz, stamped with the
/// tracker's clock. Every value is passed as the tracker sent it; a key's
/// `tobii_validity_t` flag says only that the frame had the key.
///
/// The DLL serves it only from its in-process tracker module, which it runs
/// for any URL, libtobii's `tobii-ffi://` included, but `tobii-prp://` and
/// `tprp-tcp://`, and only with the internal feature group:
/// `TOBII_ERROR_INSUFFICIENT_LICENSE` below it (0x180175d15), and
/// `TOBII_ERROR_NOT_SUPPORTED` behind the Tobii service, where that module
/// does not exist (0x18014e3e8). No licence is checked here (see
/// `licensing`). A daemon older than this library acks the subscription and
/// never sends the stream.
///
/// # Safety
/// As the subscribe functions in `tobii_streams.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_gaze_raw_subscribe(
    device: *mut Device,
    callback: Option<GazeRawFn>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { subscribe(device, |c| &mut c.gaze_raw, callback, user_data) }
}

/// Undo `tobii_gaze_raw_subscribe`.
///
/// # Safety
/// `device` must be null or a live handle that is not destroyed before the
/// call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_gaze_raw_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { unsubscribe(device, |c| &mut c.gaze_raw) }
}

/// Internal streams this library delivers: the IR image (id 0) only.
///
/// The DLL's internal stream ids are 0 image, 1 clean IR, 2 custom, 3 and 4
/// low-frequency head rotation and position, 5 multiple faces, 6 image
/// collection, 7 wearable limited image and 8 secondary camera image. The
/// names of 3..8 come from the exports that subscribe to them; those of 0..2
/// are inferred.
const fn internal_stream_supported(stream: u32) -> bool {
    stream == 0
}

/// Whether an internal stream is available: the IR image only.
///
/// This matches what libtobii delivers (`tobii_image_subscribe` works; the
/// clean IR, custom and image-collection streams are stubs). For 3, 4, 5, 7
/// and 8 it is also the DLL's answer on its TTP path, for any tracker it
/// drives itself: its check (0x180157860, called at 0x18014cc59) has no case
/// for them, and their subscribes refuse them the same way (see
/// `refused_internal_subscribe`). For an ET5 the DLL would support 0, 2 and 6
/// on that path. On its PRP path, behind the Tobii service, it never
/// supports 0 or 1 and answers 2..8 from the streams the service lists,
/// never captured for an ET5. As in the DLL, an unknown id is reported
/// unsupported, not an error, and an id above `i32::MAX` is an invalid
/// parameter.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `supported` must be null or
/// valid for writing one `tobii_supported_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_internal_stream_supported(
    device: *mut Device,
    stream: u32,
    supported: *mut u32,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { write_supported(device, stream, supported, internal_stream_supported) }
}

/// A subscribe to an internal stream the DLL refuses for a tracker it drives
/// itself: 3 and 4 low-frequency head rotation and position, 5 multiple
/// faces position, 7 wearable limited image and 8 secondary camera image.
///
/// Each such export passes its stream's id to one helper of the DLL
/// (0x18015bbf0, `tobii_internal_stream_subscribe` in its log). Its support
/// check (0x180157860) has no case for these ids when the DLL runs its own
/// tracker module, as it does for any URL but `tobii-prp://` and
/// `tprp-tcp://`, so the helper answers `TOBII_ERROR_NOT_SUPPORTED` before it
/// registers anything (0x18015bd29); the module's own subscribes for them
/// answer `PLATMOD_ERROR_NOT_SUPPORTED` whatever they are given. Behind the
/// Tobii service the answer depends on the streams the service lists, never
/// captured for an ET5. So nothing is sent to the daemon, and libtobii's head
/// pose is not re-published as the low-frequency streams, whose units, frame
/// and rate would all be invented.
///
/// A call from inside a callback is `TOBII_ERROR_CALLBACK_IN_PROGRESS`, then a
/// null device or callback is `TOBII_ERROR_INVALID_PARAMETER`, in that order:
/// the DLL checks both nulls before the callback. Unlike the DLL, which logs
/// every refusal but a null device's at ERROR, nothing reaches the
/// application's logger. The device is only compared with null, never
/// borrowed.
fn refused_internal_subscribe(
    export: &'static str,
    stream: u32,
    device: *mut Device,
    callback: *const c_void,
) -> Status {
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    if device.is_null() || callback.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    tracing::trace!(export, stream, "internal stream not supported");
    TOBII_ERROR_NOT_SUPPORTED
}

/// The unsubscribe of an internal stream [`refused_internal_subscribe`]
/// refuses. The DLL's helper (0x18015bee0) runs the same support check before
/// it looks for a subscription (0x18015bf2d), so the answer is
/// `TOBII_ERROR_NOT_SUPPORTED`, never `TOBII_ERROR_NOT_SUBSCRIBED`; the
/// checks before it are the subscribe's, less the callback.
fn refused_internal_unsubscribe(export: &'static str, stream: u32, device: *mut Device) -> Status {
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    if device.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    tracing::trace!(export, stream, "internal stream not supported");
    TOBII_ERROR_NOT_SUPPORTED
}

/// Define the subscribe and unsubscribe of internal streams the DLL refuses
/// for a tracker it drives itself (see `refused_internal_subscribe`).
macro_rules! refused_internal_streams {
    ($(
        $(#[$doc:meta])*
        $sub:ident / $unsub:ident: $stream:literal;
    )+) => { $(
        $(#[$doc])*
        ///
        /// `TOBII_ERROR_NOT_SUPPORTED` once the device and callback check out,
        /// as the DLL answers for a tracker it drives itself, with nothing sent
        /// to the daemon (see `refused_internal_subscribe`). The pointers are
        /// only compared with null, so the function is safe.
        #[unsafe(no_mangle)]
        pub extern "C" fn $sub(
            device: *mut Device,
            callback: *const c_void,
            _user_data: *mut c_void,
        ) -> Status {
            refused_internal_subscribe(stringify!($sub), $stream, device, callback)
        }

        #[doc = concat!(
            "The unsubscribe of `", stringify!($sub), "`: `TOBII_ERROR_NOT_SUPPORTED` ",
            "once the device checks out, as in the DLL, whose support check comes ",
            "before its subscription lookup (see `refused_internal_unsubscribe`)."
        )]
        #[unsafe(no_mangle)]
        pub extern "C" fn $unsub(device: *mut Device) -> Status {
            refused_internal_unsubscribe(stringify!($unsub), $stream, device)
        }
    )+ };
}

refused_internal_streams! {
    /// Internal stream 3, low-frequency head rotation (the DLL's PRP stream 9,
    /// `LOW_FREQUENCY_HEAD_ROTATION`). What the DLL would deliver behind the
    /// Tobii service is described in `tobii_internal.h`.
    tobii_low_frequency_head_rotation_subscribe / tobii_low_frequency_head_rotation_unsubscribe: 3;
    /// Internal stream 4, low-frequency head position (the DLL's PRP stream 8,
    /// `LOW_FREQUENCY_HEAD_POSITION`). What the DLL would deliver behind the
    /// Tobii service is described in `tobii_internal.h`.
    tobii_low_frequency_head_position_subscribe / tobii_low_frequency_head_position_unsubscribe: 4;
    /// Internal stream 5, multiple faces position (the DLL's PRP stream 10,
    /// `MULTIPLE_FACES_POSITION`).
    tobii_multiple_faces_position_subscribe / tobii_multiple_faces_position_unsubscribe: 5;
    /// Internal stream 7, wearable limited image (the DLL's PRP stream 11,
    /// `WEARABLE_LIMITED_IMAGE`).
    tobii_wearable_limited_image_subscribe / tobii_wearable_limited_image_unsubscribe: 7;
    /// Internal stream 8, secondary camera image (the DLL's PRP stream 0x17,
    /// `SECONDARY_CAMERA_IMAGE`).
    tobii_secondary_camera_image_subscribe / tobii_secondary_camera_image_unsubscribe: 8;
}

/// Internal capabilities this library provides: eyeball centres (id 0) only.
///
/// The DLL's ids, from what its jump table at 0x18014d268 leads to: 0
/// eyeball center, 1 diagnostic images (command 0x15), 2 remote wake
/// (settable property 3), 3 power save (settable property 2), 4 face id
/// (properties 13 and 14, commands 0x1a and 0x1b) and 5 logs (command
/// 0x19). The name of 0 is the DLL's; those of 1..5 are inferred from what
/// they look for.
const fn internal_capability_supported(capability: u32) -> bool {
    capability == 0
}

/// Whether an internal capability is available: eyeball centres only.
///
/// This matches what libtobii delivers (`tobii_gaze_data_t` carries each
/// eye's eyeball centre; diagnostic images, remote wake, power save, face id
/// and logs are stubs), not the DLL. For an ET5 the DLL would support 0 on
/// its TTP path (inferred: it asks for gaze columns 0x17 and 0x18, the keys
/// of the eyeball centres in every ET5 gaze frame) and answer
/// `TOBII_ERROR_NOT_SUPPORTED` for 0 on its PRP path, the one behind the
/// Tobii service; it answers 1..5 from device lists never captured for an
/// ET5. As in the DLL, an id above 5 is reported unsupported, not an error,
/// and a negative id is an invalid parameter.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `supported` must be null or
/// valid for writing one `tobii_supported_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_internal_capability_supported(
    device: *mut Device,
    capability: i32,
    supported: *mut u32,
) -> Status {
    // A negative id is one above `i32::MAX` here, which `write_supported`
    // refuses once the device checks out.
    let capability = capability.cast_unsigned();
    // SAFETY: forwarded under the same contract.
    unsafe { write_supported(device, capability, supported, internal_capability_supported) }
}

/// A fresh tracker/host clock pair: the tracker clock read `tracker_us` at
/// some host time between `system_start_us` and `system_end_us`.
///
/// The daemon answers from the first gaze frame it receives after the
/// request (starting the tracker if needed): `tracker_us` is the frame's
/// device timestamp and the bracket is the 30 ms before the daemon read it.
/// The DLL instead times a round trip to its service, and its own offset
/// estimator skips pairs wider than 6 ms, though it still returns them. The
/// host clock is `tobii_system_clock`'s, `CLOCK_MONOTONIC` rather than the
/// DLL's QPC, though like it monotonic with an undefined epoch. The
/// callbacks' `timestamp_us` (and gaze data's `timestamp_system_us`) need no
/// pair: the daemon sends them on the host clock already; gaze data's
/// `timestamp_tracker_us` is the tracker time a pair maps. The tracker's
/// clock may restart when the tracker re-initialises, so a pair holds until
/// then only. Nothing is written unless the call succeeds;
/// `TOBII_ERROR_NOT_AVAILABLE` while the tracker is paused,
/// `TOBII_ERROR_CONNECTION_FAILED` when no tracker is plugged in.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `timesync` must be null or
/// valid for writing one `tobii_timesync_data_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_timesync(
    device: *mut Device,
    timesync: *mut TimesyncData,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if timesync.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::TIMESYNC, &[], timeouts::TIMESYNC)
        .map(|p| decode_timesync(&p))
    {
        Ok(Some(t)) => {
            // The reply's order is start, device, end; the struct's is
            // start, end, tracker.
            let out = TimesyncData {
                system_start_us: t.host_start_us,
                system_end_us: t.host_end_us,
                tracker_us: t.device_us,
            };
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { timesync.write(out) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => d.malformed("timesync"),
        Err(status) => status,
    }
}

/// The Stream Engine's stream type for a device stream id: the DLL's three
/// tables composed (id to TTP code at 0x18017c784, TTP to tracker at
/// 0x180190ae8, tracker to Stream Engine at 0x180001e74). 0x509 is 0 in the
/// DLL's table, and so are the ids the ET5 lists that the DLL has no type
/// for (0x50e, 0x1771, 0x1772, 0x1774).
const fn se_stream_type(id: u32) -> i32 {
    match id {
        0x500 => 1,
        0x501 => 2,
        0x502 => 3,
        0x503 => 14,
        0x504 => 4,
        0x505 => 5,
        0x506 => 8,
        0x507 => 9,
        0x508 => 11,
        0x50a => 6,
        0x1770 => 7,
        _ => 0,
    }
}

/// A catalogue entry as the DLL hands it over: the device's stream id
/// becomes a Stream Engine type, and the strings are cut to 63 bytes.
fn stream_type_c(t: &request::StreamType) -> StreamType {
    let mut c = StreamType {
        type_: se_stream_type(t.id),
        value: t.value,
        name: [0; 64],
        text: [0; 64],
    };
    copy_c_string(&mut c.name, &t.name);
    copy_c_string(&mut c.text, &t.text);
    c
}

/// The tracker's stream catalogue: `receiver` is called once per stream,
/// in the tracker's order.
///
/// The daemon answers from the catalogue the tracker reported at its last
/// init (starting the tracker if needed), where the DLL asks the tracker on
/// every call. The list is the tracker's own, so it names streams libtobii
/// does not deliver, such as `image_collection` (internal stream 6 is
/// unsupported). The DLL answers `TOBII_ERROR_NOT_SUPPORTED` on its PRP path
/// and needs the internal feature group on its TTP path; libtobii has no
/// licence gate here. Every entry is built before the first call, so the
/// receiver may call back into this library, as the DLL allows.
///
/// `TOBII_ERROR_NOT_SUPPORTED` if the tracker reported no catalogue (the DLL
/// answers `TOBII_ERROR_NO_ERROR` with no calls) or the daemon is older than
/// this library; `TOBII_ERROR_TIMED_OUT` if no tracker has been seen.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `receiver` must be null or
/// sound to call with a `tobii_stream_type_t` (valid during the call only)
/// and `user_data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_enumerate_stream_types(
    device: *mut Device,
    receiver: Option<StreamTypeReceiver>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let entries: Vec<StreamType> = match d
        .request(kind::STREAM_TYPES, &[], timeouts::FACTS)
        .map(|p| decode_stream_types(&p))
    {
        Ok(Some(types)) => types.iter().map(stream_type_c).collect(),
        Ok(None) => return d.malformed("stream types"),
        Err(status) => return status,
    };
    for entry in &entries {
        // SAFETY: the caller guarantees `receiver` is sound to call like
        // this; `entry` outlives the call.
        unsafe { receiver(entry, user_data) };
    }
    TOBII_ERROR_NO_ERROR
}

/// Pause the tracker: it stops sending data until resumed.
///
/// The pause is one state for the tracker, shared by every client, as in the
/// DLL: the last call wins and any client may resume. Unlike the DLL, a
/// pause the tracker accepts shows at once in `TOBII_STATE_DEVICE_PAUSED`
/// and a `TOBII_NOTIFICATION_TYPE_DEVICE_PAUSED_STATE_CHANGED`
/// notification. A pause ends when the client that paused last disconnects
/// and whenever the tracker re-initialises. `TOBII_ERROR_CALIBRATION_BUSY`
/// while a calibration session runs.
///
/// # Safety
/// `device` must be null or a live handle that is not destroyed before the
/// call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_pause_device(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { request(device, kind::DEVICE_PAUSE, &[1], timeouts::DEVICE_PAUSE) }
}

/// Resume the tracker, whichever client paused it. A resume the tracker does
/// not answer still succeeds: the daemon re-opens a tracker that stays
/// silent, which resumes it.
///
/// # Safety
/// As `tobii_pause_device`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_resume_device(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { request(device, kind::DEVICE_PAUSE, &[0], timeouts::DEVICE_PAUSE) }
}

/// A hardware configuration entry with nothing set.
const NO_HARDWARE_ENTRY: HardwareConfigurationEntry = HardwareConfigurationEntry {
    id: 0,
    param_a: 0.0,
    param_b: 0.0,
    position_xyz: [0.0; 3],
    values: [0.0; 15],
    width: 0,
    height: 0,
    param_c: 0,
    coefficient_count: 0,
    coefficients: [0.0; 64],
    point_a_xyz: [0.0; 3],
    point_b_xyz: [0.0; 3],
    param_d: 0.0,
};

/// A count of slots, which is at most an array's length.
fn c_count(n: usize) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// The daemon's hardware configuration in the DLL's layout: every slot past
/// a count is zero, and the tracker's 32-bit words keep their bits, as the
/// DLL copies them. A mode outside 0..=2 is 0, as the DLL's service maps it.
#[allow(clippy::cast_possible_truncation)] // reason: the C fields are float (inferred); 16.16 values fit
fn hardware_configuration_c(h: &request::HardwareConfiguration) -> HardwareConfiguration {
    let mut c = HardwareConfiguration {
        entry_count: c_count(h.entries.len().min(2)),
        entries: [NO_HARDWARE_ENTRY; 2],
        point_count: c_count(h.points_mm.len().min(40)),
        points_xyz: [[0.0; 3]; 40],
        mode: match h.mode {
            mode @ 0..=2 => mode.cast_signed(),
            _ => 0,
        },
    };
    for (dst, e) in c.entries.iter_mut().zip(&h.entries) {
        let n = e.coefficients.len().min(dst.coefficients.len());
        dst.coefficients[..n].copy_from_slice(&e.coefficients[..n]);
        dst.coefficient_count = c_count(n);
        dst.id = e.id.cast_signed();
        dst.param_a = e.param_a as f32;
        dst.param_b = e.param_b as f32;
        dst.position_xyz = e.position_mm;
        dst.values = e.values;
        dst.width = e.width.cast_signed();
        dst.height = e.height.cast_signed();
        dst.param_c = e.param_c.cast_signed();
        dst.point_a_xyz = e.point_a_mm;
        dst.point_b_xyz = e.point_b_mm;
        dst.param_d = e.param_d;
    }
    for (dst, p) in c.points_xyz.iter_mut().zip(&h.points_mm) {
        *dst = *p;
    }
    c
}

/// The tracker's hardware configuration: provisional.
///
/// The DLL reads it from its service (PRP property 12), which has it from
/// the tracker's command 2120. That answer's layout is inferred from one
/// Windows capture, and on Linux the ET5 has so far answered 2120 with no
/// data, so this is `TOBII_ERROR_NOT_SUPPORTED` there, as the DLL answers
/// when a device does not list the property. When the daemon has one, the
/// whole struct is written, with every slot past a count zero (the DLL
/// leaves those as the caller had them); nothing is written unless the call
/// succeeds. The field names and units are ours: 16.16 values unscaled,
/// 32.32 values as lengths in mm.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `configuration` must be
/// null or valid for writing one `tobii_hardware_configuration_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_hardware_configuration_get(
    device: *mut Device,
    configuration: *mut HardwareConfiguration,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if configuration.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::HARDWARE_CONFIGURATION, &[], timeouts::FACTS)
        .map(|p| decode_hardware_configuration(&p))
    {
        Ok(Some(h)) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { configuration.write(hardware_configuration_c(&h)) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => d.malformed("hardware configuration"),
        Err(status) => status,
    }
}

type P = *mut c_void;
type C = *const c_void;

not_supported! {
    fn tobii_clean_ir_subscribe(device: P, callback: C, user_data: P);
    fn tobii_clean_ir_unsubscribe(device: P);
    fn tobii_custom_stream_subscribe(device: P, callback: C, stream: u32, user_data: P);
    fn tobii_custom_stream_unsubscribe(device: P, stream: u32);
    fn tobii_diagnostic_images_retrieve(device: P, receiver: C, user_data: P);
    fn tobii_diagnostics_dump_images(device: P, a: u32, b: u32);
    fn tobii_diagnostics_get_data(device: P, kind: u32, receiver: C, user_data: P);
    fn tobii_diagnostics_image_subscribe(device: P, callback: C, user_data: P);
    fn tobii_diagnostics_image_unsubscribe(device: P);
    fn tobii_display_id_subscribe(device: P, callback: C, user_data: P);
    fn tobii_display_id_unsubscribe(device: P);
    fn tobii_enable_extension(device: P, extension: u32);
    fn tobii_enumerate_enabled_extensions(device: P, receiver: C, user_data: P);
    fn tobii_enumerate_extensions(device: P, receiver: C, user_data: P);
    fn tobii_enumerate_illumination_modes(device: P, receiver: C, user_data: P);
    fn tobii_enumerate_stream_type_columns(device: P, stream_type: u32, receiver: C, user_data: P);
    fn tobii_face_id_enroll(device: P, a: P, b: P);
    fn tobii_face_id_enroll_clear(device: P, a: P);
    fn tobii_face_id_parameters_subscribe(device: P, callback: C, user_data: P);
    fn tobii_face_id_parameters_unsubscribe(device: P);
    fn tobii_face_id_state_subscribe(device: P, callback: C, user_data: P);
    fn tobii_face_id_state_unsubscribe(device: P);
    fn tobii_foveated_rendering_gaze_point_subscribe(device: P, callback: C, user_data: P);
    fn tobii_foveated_rendering_gaze_point_unsubscribe(device: P);
    fn tobii_get_combined_gaze_hid_track_box(device: P, track_box: P);
    fn tobii_get_configuration_key(device: P, key: P, value: P);
    fn tobii_get_device_info_internal(device: P, info: P);
    fn tobii_get_display_id(device: P, display_id: P);
    fn tobii_get_display_info(device: P, display_info: P);
    fn tobii_get_face_id_parameters(device: P, parameters: P);
    fn tobii_get_face_id_state(device: P, state: P);
    fn tobii_get_gaze_hid_enabled(device: P, enabled: P);
    fn tobii_get_illumination_mode(device: P, mode: P);
    fn tobii_image_collection_subscribe(device: P, callback: C, user_data: P);
    fn tobii_image_collection_unsubscribe(device: P);
    fn tobii_logs_retrieve(device: P, receiver: C, user_data: P);
    fn tobii_open_realm(device: P, realm: u32, key: C, key_size: u32);
    fn tobii_power_save_activate(device: P);
    fn tobii_power_save_deactivate(device: P);
    fn tobii_remote_wake_activate(device: P);
    fn tobii_remote_wake_deactivate(device: P);
    fn tobii_send_custom_command(device: P, command: u32, data: C, size: usize, receiver: C, user_data: P);
    fn tobii_send_statistics(device: P, data: C, size: usize);
    fn tobii_set_display_id(device: P, display_id: u32);
    fn tobii_set_display_info(device: P, display_info: C);
    fn tobii_set_face_id_parameters(device: P, parameters: C);
    fn tobii_set_fw_upgrade_allowed(device: P, allowed: u32);
    fn tobii_set_illumination_mode(device: P, mode: C);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tobii_device_destroy;
    use crate::status::{
        TOBII_ERROR_ALREADY_SUBSCRIBED, TOBII_ERROR_CONNECTION_FAILED, TOBII_ERROR_INTERNAL,
        TOBII_ERROR_NOT_AVAILABLE, TOBII_ERROR_NOT_SUBSCRIBED,
    };
    use crate::types::{GazeRaw, TOBII_NOT_SUPPORTED, TOBII_SUPPORTED};
    use std::ffi::CStr;
    use std::ptr;
    use tobii_ipc::request::{Timesync, encode_stream_types, encode_timesync, status};

    #[test]
    fn only_the_ir_image_is_a_supported_internal_stream() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut s = 9u32;
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `s` is a live local.
        unsafe {
            assert_eq!(tobii_internal_stream_supported(d, 0, &raw mut s), 0);
            assert_eq!(s, TOBII_SUPPORTED);
            // For 3, 4, 5, 7 and 8 this is the DLL's own answer for a tracker
            // it drives itself.
            for stream in (1..=8).chain([9, 1000, 0x7fff_ffff]) {
                s = 9;
                assert_eq!(
                    tobii_internal_stream_supported(d, stream, &raw mut s),
                    0,
                    "{stream}: unknown is not an error"
                );
                assert_eq!(s, TOBII_NOT_SUPPORTED, "{stream}");
            }
            s = 9;
            assert_eq!(
                tobii_internal_stream_supported(d, 0x8000_0000, &raw mut s),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(s, 9, "nothing written");
            assert_eq!(
                tobii_internal_stream_supported(d, 0, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_internal_stream_supported(ptr::null_mut(), 0, &raw mut s),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(s, 9, "nothing written");
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    type RefusedSubscribe = extern "C" fn(*mut Device, *const c_void, *mut c_void) -> Status;
    type RefusedUnsubscribe = extern "C" fn(*mut Device) -> Status;

    /// The subscribe and unsubscribe of each internal stream the DLL refuses
    /// for a tracker it drives itself.
    const REFUSED: [(u32, RefusedSubscribe, RefusedUnsubscribe); 5] = [
        (
            3,
            tobii_low_frequency_head_rotation_subscribe,
            tobii_low_frequency_head_rotation_unsubscribe,
        ),
        (
            4,
            tobii_low_frequency_head_position_subscribe,
            tobii_low_frequency_head_position_unsubscribe,
        ),
        (
            5,
            tobii_multiple_faces_position_subscribe,
            tobii_multiple_faces_position_unsubscribe,
        ),
        (
            7,
            tobii_wearable_limited_image_subscribe,
            tobii_wearable_limited_image_unsubscribe,
        ),
        (
            8,
            tobii_secondary_camera_image_subscribe,
            tobii_secondary_camera_image_unsubscribe,
        ),
    ];

    /// A callback the refused subscribes are given and never call.
    fn never_called() -> *const c_void {
        ptr::NonNull::<c_void>::dangling().as_ptr().cast_const()
    }

    unsafe extern "C" fn ignore_head_pose(_: *const crate::types::HeadPose, _: *mut c_void) {}

    /// The daemon fails every request, as with no tracker plugged in, and
    /// counts every frame: only the head-pose subscribe at the end reaches
    /// it. An unsubscribe is refused, not `TOBII_ERROR_NOT_SUBSCRIBED`, as in
    /// the DLL, whose support check comes first.
    #[test]
    fn refused_internal_streams_are_not_supported_without_asking_the_daemon() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tobii_ipc::request::decode_request;
        use tobii_ipc::{encode_reply, encode_subscribed};
        let frames = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&frames);
        let connect = crate::device::tests::fake_daemon(move |body| {
            seen.fetch_add(1, Ordering::Relaxed);
            match body.first() {
                Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![encode_subscribed(true)],
                Some(&tobii_ipc::TAG_REQUEST) => {
                    let req = decode_request(body).expect("request");
                    vec![encode_reply(req.id, status::CONNECTION_FAILED, &[])]
                }
                _ => vec![],
            }
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        let (refused, invalid) = (TOBII_ERROR_NOT_SUPPORTED, TOBII_ERROR_INVALID_PARAMETER);
        let n = ptr::null_mut();
        for (stream, subscribe, unsubscribe) in REFUSED {
            let got = [
                unsubscribe(d),
                subscribe(d, never_called(), n),
                subscribe(d, ptr::null(), n),
                subscribe(n.cast(), never_called(), n),
                subscribe(n.cast(), ptr::null(), n),
                unsubscribe(d),
                unsubscribe(n.cast()),
            ];
            let want = [
                refused, refused, invalid, invalid, invalid, refused, invalid,
            ];
            assert_eq!(got, want, "internal stream {stream}");
        }
        assert_eq!(frames.load(Ordering::Relaxed), 0, "nothing sent");
        // The daemon handles frames in order, so once the head-pose subscribe
        // is acked, a frame the calls sent without waiting would have been
        // counted too. The refused subscribes left head pose's slot free.
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once.
        unsafe {
            assert_eq!(
                crate::streams::tobii_head_pose_subscribe(d, Some(ignore_head_pose), n),
                TOBII_ERROR_NO_ERROR
            );
            assert_eq!(frames.load(Ordering::Relaxed), 1, "only the head pose");
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    /// Inside a callback every call is refused before its arguments are
    /// read, as wherever a device handle is taken. The DLL checks the nulls
    /// first, so for those it would answer `TOBII_ERROR_INVALID_PARAMETER`.
    #[test]
    fn refused_internal_streams_are_refused_inside_a_callback() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let n = ptr::null_mut();
        let mut got = Vec::new();
        crate::device::call(|| {
            for (_, subscribe, unsubscribe) in REFUSED {
                got.extend([
                    subscribe(d, never_called(), n),
                    subscribe(n.cast(), never_called(), n),
                    subscribe(d, ptr::null(), n),
                    subscribe(n.cast(), ptr::null(), n),
                    unsubscribe(d),
                    unsubscribe(n.cast()),
                ]);
            }
        });
        assert_eq!(got, [TOBII_ERROR_CALLBACK_IN_PROGRESS; 30]);
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once;
        // the callback guard is down again.
        unsafe { assert_eq!(tobii_device_destroy(d), 0) };
    }

    unsafe extern "C" fn ignore_raw(_: *const GazeRaw, _: *mut c_void) {}

    /// Raw gaze is subscribed with a bit of its own, and follows the Stream
    /// Engine's subscription rules. The daemon stand-in acks every change
    /// and never sends the stream, as a daemon older than this library
    /// would: the subscribe succeeds all the same.
    #[test]
    fn raw_gaze_takes_its_own_stream_bit_by_the_stream_engines_rules() {
        use std::sync::{Arc, Mutex};
        use tobii_ipc::{STREAM_GAZE_RAW, decode_subscribe, encode_subscribed};
        let masks = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&masks);
        let connect = crate::device::tests::fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => {
                log.lock().expect("log").push(decode_subscribe(body));
                vec![encode_subscribed(true)]
            }
            _ => vec![],
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        let n = ptr::null_mut();
        let callback = Some(ignore_raw as GazeRawFn);
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; a null device is allowed; the callback ignores its
        // arguments.
        let got = unsafe {
            let got = [
                tobii_gaze_raw_subscribe(d, None, n),
                tobii_gaze_raw_subscribe(n.cast(), callback, n),
                tobii_gaze_raw_unsubscribe(d),
                tobii_gaze_raw_subscribe(d, callback, n),
                tobii_gaze_raw_subscribe(d, callback, n),
                tobii_gaze_raw_unsubscribe(d),
                tobii_gaze_raw_unsubscribe(d),
                tobii_gaze_raw_unsubscribe(n.cast()),
            ];
            assert_eq!(tobii_device_destroy(d), 0);
            got
        };
        let want = [
            TOBII_ERROR_INVALID_PARAMETER,
            TOBII_ERROR_INVALID_PARAMETER,
            TOBII_ERROR_NOT_SUBSCRIBED,
            TOBII_ERROR_NO_ERROR,
            TOBII_ERROR_ALREADY_SUBSCRIBED,
            TOBII_ERROR_NO_ERROR,
            TOBII_ERROR_NOT_SUBSCRIBED,
            TOBII_ERROR_INVALID_PARAMETER,
        ];
        assert_eq!(got, want);
        assert_eq!(
            *masks.lock().expect("log"),
            [Some(STREAM_GAZE_RAW), Some(0)]
        );
    }

    /// What the raw gaze callback saw.
    #[derive(Default)]
    struct SeenRaw {
        records: Vec<GazeRaw>,
        /// The record's bytes as the callback read them.
        bytes: Vec<u8>,
        /// Whether the callback guard was up while it ran.
        guarded: bool,
    }

    unsafe extern "C" fn keep_raw(p: *const GazeRaw, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut SeenRaw` as `ud`, which outlives
        // the device; `p` is a live record for the call.
        let (seen, record) = unsafe { (&mut *ud.cast::<SeenRaw>(), &*p) };
        seen.records.push(*record);
        // SAFETY: `record` is live for the call, and all its bytes are
        // initialised: `GazeRaw` has no padding (see the layout test).
        let bytes = unsafe {
            std::slice::from_raw_parts(ptr::from_ref(record).cast::<u8>(), size_of::<GazeRaw>())
        };
        seen.bytes = bytes.to_vec();
        seen.guarded = in_callback();
    }

    /// A raw gaze frame reaches the callback in the DLL's layout: every value
    /// as the daemon sent it, a lost eye's included, the tracker time
    /// unchanged, a flagged key's validity 1 when the frame had it (a 0 sent
    /// too) and 0, with a value of 0, when it did not, and the slots the
    /// DLL's record never fills zero. The callback runs under the callback
    /// guard.
    #[test]
    fn raw_gaze_reaches_its_callback_in_the_dlls_layout() {
        use crate::types::{TOBII_VALIDITY_INVALID as NO, TOBII_VALIDITY_VALID as YES};
        use tobii_ipc::{STREAM_GAZE_RAW, decode_subscribe, encode_gaze_raw, encode_subscribed};
        let sample = tobii_ipc::GazeRaw {
            timestamp_tracker_us: 9_613_320_391,
            left: tobii_ipc::GazeRawEye {
                gaze_origin_mm: [-31.5, 12.25, 612.0],
                gaze_origin_in_track_box: [0.25, 0.5, 0.75],
                gaze_point_mm: [40.5, 292.5, 99.625],
                gaze_point_on_display: [0.5625, 0.125],
                pupil_diameter_mm: 6.25,
                status: 0,
            },
            // Lost, its values passed on all the same.
            right: tobii_ipc::GazeRawEye {
                gaze_origin_mm: [31.5, 11.0, 610.5],
                gaze_origin_in_track_box: [0.375, 0.625, 0.875],
                gaze_point_mm: [-2.0, 1.5, -0.5],
                gaze_point_on_display: [-1.0, 2.0],
                pupil_diameter_mm: 5.75,
                status: 4,
            },
            combined_gaze_point_on_display: [0.5, 0.25],
            combined_gaze_validity: 1,
            key_0e: None,
            key_11: Some(4),
            frame_counter: Some(43_780),
            left_origin_flag: Some(1),
            right_origin_flag: Some(0),
            left_eyeball_center_mm: Some([-30.0, 10.5, 620.25]),
            right_eyeball_center_mm: None,
        };
        let frame = encode_gaze_raw(&sample);
        // Sent ahead of the ack of raw gaze alone, so that it waits for
        // `process`.
        let connect = crate::device::tests::fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) if decode_subscribe(body) == Some(STREAM_GAZE_RAW) => {
                vec![frame.clone(), encode_subscribed(true)]
            }
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![encode_subscribed(true)],
            _ => vec![],
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        let mut seen = SeenRaw::default();
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `keep_raw` gets the live `seen`, which outlives the device.
        unsafe {
            assert_eq!(
                tobii_gaze_raw_subscribe(
                    d,
                    Some(keep_raw as GazeRawFn),
                    (&raw mut seen).cast::<c_void>()
                ),
                TOBII_ERROR_NO_ERROR
            );
            assert_eq!(
                crate::api::tobii_device_process_callbacks(d),
                TOBII_ERROR_NO_ERROR
            );
            assert_eq!(tobii_device_destroy(d), 0);
        }

        let eye = |e: &tobii_ipc::GazeRawEye| crate::types::GazeRawEye {
            gaze_origin_from_eye_tracker_mm_xyz: e.gaze_origin_mm,
            gaze_origin_in_track_box_normalized_xyz: e.gaze_origin_in_track_box,
            gaze_point_from_eye_tracker_mm_xyz: e.gaze_point_mm,
            gaze_point_on_display_normalized_xy: e.gaze_point_on_display,
            pupil_diameter_mm: e.pupil_diameter_mm,
            status: e.status,
        };
        let want = GazeRaw {
            timestamp_tracker_us: 9_613_320_391,
            left: eye(&sample.left),
            right: eye(&sample.right),
            combined_gaze_point_on_display_normalized_xy: [0.5, 0.25],
            combined_gaze_validity: 1,
            key_0e_validity: NO,
            key_0e: 0,
            reserved_84: NO,
            reserved_88: 0.0,
            reserved_8c: NO,
            reserved_90: 0.0,
            key_11_validity: YES,
            key_11: 4,
            reserved_9c: NO,
            reserved_a0: 0.0,
            reserved_a4: NO,
            reserved_a8: 0.0,
            frame_counter_validity: YES,
            frame_counter: 43_780,
            left_origin_flag_validity: YES,
            left_origin_flag: 1,
            right_origin_flag_validity: YES,
            right_origin_flag: 0,
            left_eyeball_center_validity: YES,
            left_eyeball_center_from_eye_tracker_mm_xyz: [-30.0, 10.5, 620.25],
            right_eyeball_center_validity: NO,
            right_eyeball_center_from_eye_tracker_mm_xyz: [0.0; 3],
            reserved_e4: 0,
        };
        assert_eq!(seen.records, [want]);
        assert!(seen.guarded, "run under the callback guard");
        // The bytes the DLL's dispatch leaves zero, and those of the keys
        // the frame lacked: 0x0e's pair and the right eyeball.
        let zero = |range: std::ops::Range<usize>| seen.bytes[range].iter().all(|&b| b == 0);
        assert_eq!(seen.bytes.len(), 232);
        for range in [0x7c..0x84, 0x84..0x94, 0x9c..0xac, 0xd4..0xe4, 0xe4..0xe8] {
            assert!(zero(range.clone()), "{range:#x?}");
        }
        assert_eq!(seen.bytes[0x78..0x7c], 1u32.to_ne_bytes(), "combined");
        assert_eq!(
            seen.bytes[0xac..0xb4],
            [1u32.to_ne_bytes(), 43_780u32.to_ne_bytes()].concat(),
            "frame counter"
        );
    }

    /// Inside a callback the raw gaze calls are refused before their
    /// arguments are read, as wherever a device handle is taken, and
    /// subscribe nothing.
    #[test]
    fn raw_gaze_is_refused_inside_a_callback() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let n = ptr::null_mut();
        let callback = Some(ignore_raw as GazeRawFn);
        let mut got = [0; 5];
        crate::device::call(|| {
            // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed
            // once below; a null device is allowed.
            got = unsafe {
                [
                    tobii_gaze_raw_subscribe(d, callback, n),
                    tobii_gaze_raw_subscribe(n.cast(), callback, n),
                    tobii_gaze_raw_subscribe(d, None, n),
                    tobii_gaze_raw_unsubscribe(d),
                    tobii_gaze_raw_unsubscribe(n.cast()),
                ]
            };
        });
        assert_eq!(got, [TOBII_ERROR_CALLBACK_IN_PROGRESS; 5]);
        // SAFETY: as above; the callback guard is down again.
        unsafe {
            assert_eq!(tobii_gaze_raw_unsubscribe(d), TOBII_ERROR_NOT_SUBSCRIBED);
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    /// The daemon fails every request, as with no tracker plugged in: the
    /// answer is local.
    #[test]
    fn only_eyeball_centres_are_a_supported_internal_capability() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(
            status::CONNECTION_FAILED,
            vec![],
        )));
        let mut s = 9u32;
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `s` is a live local.
        unsafe {
            assert_eq!(tobii_internal_capability_supported(d, 0, &raw mut s), 0);
            assert_eq!(s, TOBII_SUPPORTED);
            for capability in (1..=5).chain([6, 1000, i32::MAX]) {
                s = 9;
                assert_eq!(
                    tobii_internal_capability_supported(d, capability, &raw mut s),
                    0,
                    "{capability}: unknown is not an error"
                );
                assert_eq!(s, TOBII_NOT_SUPPORTED, "{capability}");
            }
            for capability in [-1, i32::MIN] {
                s = 9;
                assert_eq!(
                    tobii_internal_capability_supported(d, capability, &raw mut s),
                    TOBII_ERROR_INVALID_PARAMETER,
                    "{capability}"
                );
                assert_eq!(s, 9, "{capability}: nothing written");
            }
            assert_eq!(
                tobii_internal_capability_supported(d, 0, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_internal_capability_supported(d, -1, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_internal_capability_supported(ptr::null_mut(), 0, &raw mut s),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(s, 9, "nothing written");
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    /// From inside a callback the query is refused before its arguments are
    /// read, as wherever a device handle is taken, and writes nothing.
    #[test]
    fn internal_capability_support_is_refused_inside_a_callback() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut s = 9u32;
        let mut got = [0; 4];
        crate::device::call(|| {
            // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed
            // once below; `s` is a live local.
            got = unsafe {
                [
                    tobii_internal_capability_supported(d, 0, &raw mut s),
                    tobii_internal_capability_supported(ptr::null_mut(), 0, &raw mut s),
                    tobii_internal_capability_supported(d, 0, ptr::null_mut()),
                    tobii_internal_capability_supported(d, -1, &raw mut s),
                ]
            };
        });
        assert_eq!(got, [TOBII_ERROR_CALLBACK_IN_PROGRESS; 4]);
        assert_eq!(s, 9, "nothing written");
        // SAFETY: as above; the callback guard is down again.
        unsafe { assert_eq!(tobii_device_destroy(d), 0) };
    }

    /// Ask a daemon answering `status`/`payload` for a clock pair, into a
    /// sentinel.
    fn timesync_with(status: u8, payload: Vec<u8>) -> (Status, TimesyncData) {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(status, payload)));
        let mut out = TimesyncData {
            system_start_us: -1,
            system_end_us: -2,
            tracker_us: -3,
        };
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `out` is a live local.
        unsafe {
            let got = tobii_timesync(d, &raw mut out);
            assert_eq!(tobii_device_destroy(d), 0);
            (got, out)
        }
    }

    #[test]
    fn timesync_puts_the_daemon_pair_in_stream_engine_order() {
        let payload = encode_timesync(&Timesync {
            host_start_us: 12_000_000_000,
            device_us: 5_000_000,
            host_end_us: 12_000_030_000,
        });
        assert_eq!(
            timesync_with(0, payload),
            (
                TOBII_ERROR_NO_ERROR,
                TimesyncData {
                    system_start_us: 12_000_000_000,
                    system_end_us: 12_000_030_000,
                    tracker_us: 5_000_000,
                }
            )
        );
    }

    #[test]
    fn timesync_writes_nothing_on_an_error() {
        let untouched = TimesyncData {
            system_start_us: -1,
            system_end_us: -2,
            tracker_us: -3,
        };
        assert_eq!(
            timesync_with(status::NOT_AVAILABLE, vec![]),
            (TOBII_ERROR_NOT_AVAILABLE, untouched),
            "the daemon's status passes through"
        );
        assert_eq!(
            timesync_with(status::CONNECTION_FAILED, vec![]),
            (TOBII_ERROR_CONNECTION_FAILED, untouched),
            "no tracker"
        );
        assert_eq!(
            timesync_with(0, vec![1, 2, 3]),
            (TOBII_ERROR_INTERNAL, untouched),
            "a malformed reply"
        );

        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut out = untouched;
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `out` is a live local.
        unsafe {
            assert_eq!(
                tobii_timesync(d, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_timesync(ptr::null_mut(), &raw mut out),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_device_destroy(d), 0);
        }
        assert_eq!(out, untouched);
    }

    /// What a stream-type receiver saw, and the device it may call back into.
    #[derive(Default)]
    struct Seen {
        entries: Vec<(i32, u32, String, String)>,
        reentry: Vec<Status>,
        device: Option<*mut Device>,
    }

    unsafe extern "C" fn collect_type(t: *const StreamType, ud: *mut c_void) {
        // SAFETY: the tests pass `&raw mut Seen` as `ud`, and the library a
        // valid entry as `t`.
        let (seen, t) = unsafe { (&mut *ud.cast::<Seen>(), &*t) };
        let text = |s: &[std::ffi::c_char; 64]| {
            // SAFETY: the library NUL-terminates both strings.
            unsafe { CStr::from_ptr(s.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        };
        seen.entries
            .push((t.type_, t.value, text(&t.name), text(&t.text)));
        if let Some(d) = seen.device {
            let mut fou = 9;
            // SAFETY: `d` is the live handle the enumeration runs on, and
            // the library no longer borrows it while the receiver runs.
            seen.reentry
                .push(unsafe { tobii_get_field_of_use(d, &raw mut fou) });
        }
    }

    /// The catalogue in the init fixture, as the daemon would send it.
    fn fixture_catalogue() -> Vec<u8> {
        let hex = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tobii-proto/fixtures/init-rsp-1200.hex"
        ));
        let bytes = tobii_proto::protocol::hex_to_bytes(hex).expect("hex");
        let msg = tobii_proto::protocol::parse_message(&bytes).expect("msg");
        encode_stream_types(&tobii_proto::facts::parse_stream_catalogue(&msg))
    }

    /// Enumerate from a daemon answering `status`/`payload`.
    fn enumerate_with(status: u8, payload: Vec<u8>, reenter: bool) -> (Status, Seen) {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(status, payload)));
        let mut seen = Seen {
            device: reenter.then_some(d),
            ..Seen::default()
        };
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `seen` is a live local.
        unsafe {
            let got = tobii_enumerate_stream_types(
                d,
                Some(collect_type),
                (&raw mut seen).cast::<c_void>(),
            );
            assert_eq!(tobii_device_destroy(d), 0);
            (got, seen)
        }
    }

    #[test]
    fn stream_types_come_in_the_trackers_order_with_the_dlls_types() {
        let (got, seen) = enumerate_with(0, fixture_catalogue(), false);

        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        let types: Vec<i32> = seen.entries.iter().map(|e| e.0).collect();
        assert_eq!(types, [1, 2, 4, 11, 0, 7, 0, 0, 0]);
        let values: Vec<u32> = seen.entries.iter().map(|e| e.1).collect();
        assert_eq!(values, [0, 0, 0, 1000, 0, 0, 0, 0, 0]);
        assert_eq!(
            seen.entries[4],
            (0, 0, "primary_camera_image".into(), String::new())
        );
    }

    #[test]
    fn stream_type_strings_are_cut_to_63_bytes() {
        let payload = encode_stream_types(&[request::StreamType {
            id: 0x500,
            name: "n".repeat(80),
            text: "t".repeat(64),
            value: 0,
        }]);
        let (got, seen) = enumerate_with(0, payload, false);
        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        assert_eq!(seen.entries[0].2, "n".repeat(63));
        assert_eq!(seen.entries[0].3, "t".repeat(63));
    }

    #[test]
    fn a_stream_type_receiver_may_call_back_into_the_library() {
        let (got, seen) = enumerate_with(0, fixture_catalogue(), true);
        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        assert_eq!(seen.reentry, [TOBII_ERROR_NO_ERROR; 9]);
    }

    #[test]
    fn stream_type_failures_call_nothing() {
        let (got, seen) = enumerate_with(status::TIMED_OUT, vec![], false);
        assert_eq!(got, crate::status::TOBII_ERROR_TIMED_OUT, "passed through");
        assert!(seen.entries.is_empty());
        let (got, seen) = enumerate_with(0, vec![9, 0, 0, 0], false);
        assert_eq!(got, TOBII_ERROR_INTERNAL, "a malformed reply");
        assert!(seen.entries.is_empty());

        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below.
        unsafe {
            assert_eq!(
                tobii_enumerate_stream_types(d, None, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_enumerate_stream_types(ptr::null_mut(), Some(collect_type), ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    #[test]
    fn pause_and_resume_ask_the_daemon() {
        use std::sync::{Arc, Mutex};
        use tobii_ipc::encode_reply;
        use tobii_ipc::request::decode_request;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let connect = crate::device::tests::fake_daemon(move |body| {
            let req = decode_request(body).expect("request");
            log.lock()
                .expect("log")
                .push((req.kind, req.payload.to_vec()));
            vec![encode_reply(req.id, 0, &[])]
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        // SAFETY: `d` is live and destroyed once; a null device is allowed.
        unsafe {
            assert_eq!(tobii_pause_device(d), TOBII_ERROR_NO_ERROR);
            assert_eq!(tobii_resume_device(d), TOBII_ERROR_NO_ERROR);
            assert_eq!(tobii_device_destroy(d), 0);
            assert_eq!(
                tobii_pause_device(ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_resume_device(ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
        }
        assert_eq!(
            *seen.lock().expect("log"),
            vec![(kind::DEVICE_PAUSE, vec![1]), (kind::DEVICE_PAUSE, vec![0])]
        );
    }

    #[test]
    fn pause_and_resume_carry_the_daemon_status() {
        // The daemon says a calibration session runs.
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(
            status::CALIBRATION_BUSY,
            vec![],
        )));
        // SAFETY: `d` is live and destroyed once.
        unsafe {
            assert_eq!(
                tobii_pause_device(d),
                crate::status::TOBII_ERROR_CALIBRATION_BUSY
            );
            assert_eq!(
                tobii_resume_device(d),
                crate::status::TOBII_ERROR_CALIBRATION_BUSY
            );
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }

    /// A configuration with every field set, to see what a call writes.
    fn sentinel_configuration() -> HardwareConfiguration {
        let entry = HardwareConfigurationEntry {
            id: -7,
            coefficient_count: -7,
            coefficients: [-7.0; 64],
            param_d: -7.0,
            ..NO_HARDWARE_ENTRY
        };
        HardwareConfiguration {
            entry_count: -7,
            entries: [entry; 2],
            point_count: -7,
            points_xyz: [[-7.0; 3]; 40],
            mode: -7,
        }
    }

    /// Ask a daemon answering `status`/`payload` for the hardware
    /// configuration, into a sentinel.
    fn hardware_with(status: u8, payload: Vec<u8>) -> (Status, HardwareConfiguration) {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(status, payload)));
        let mut out = sentinel_configuration();
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `out` is a live local.
        unsafe {
            let got = tobii_hardware_configuration_get(d, &raw mut out);
            assert_eq!(tobii_device_destroy(d), 0);
            (got, out)
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: values copied, not computed
    fn the_hardware_configuration_fills_the_dll_layout() {
        let h = request::HardwareConfiguration {
            entries: vec![request::HardwareEntry {
                id: u32::MAX,
                param_a: 16.0,
                param_b: 100.0,
                position_mm: [0.0, 0.0, 4.14],
                values: [0.5; 15],
                width: 2240,
                height: 2241,
                param_c: 3,
                coefficients: vec![1.0, -0.5],
                point_a_mm: [1.0, 2.0, 3.0],
                point_b_mm: [4.0, 5.0, 6.0],
                param_d: 0.25,
            }],
            points_mm: vec![[130.0, 0.76, 1.62], [-130.0, 0.76, 1.62]],
            mode: 2,
        };

        let (got, c) = hardware_with(0, request::encode_hardware_configuration(&h));

        assert_eq!(got, TOBII_ERROR_NO_ERROR);
        assert_eq!((c.entry_count, c.point_count, c.mode), (1, 2, 2));
        let e = &c.entries[0];
        assert_eq!((e.id, e.param_a, e.param_b), (-1, 16.0, 100.0));
        assert_eq!((e.position_xyz, e.values), ([0.0, 0.0, 4.14], [0.5; 15]));
        assert_eq!((e.width, e.height, e.param_c), (2240, 2241, 3));
        assert_eq!(e.coefficient_count, 2);
        assert_eq!(e.coefficients[..3], [1.0, -0.5, 0.0], "zero past the count");
        assert_eq!(
            (e.point_a_xyz, e.point_b_xyz),
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        );
        assert_eq!(e.param_d, 0.25);
        assert_eq!(c.entries[1], NO_HARDWARE_ENTRY, "zero past the count");
        assert_eq!(c.points_xyz[1], [-130.0, 0.76, 1.62]);
        assert_eq!(c.points_xyz[2], [0.0; 3], "zero past the count");
    }

    #[test]
    fn a_hardware_mode_outside_the_dlls_range_is_zero() {
        for (mode, want) in [(0, 0), (1, 1), (2, 2), (3, 0), (u32::MAX, 0)] {
            let h = request::HardwareConfiguration {
                mode,
                ..request::HardwareConfiguration::default()
            };
            let (got, c) = hardware_with(0, request::encode_hardware_configuration(&h));
            assert_eq!((got, c.mode), (TOBII_ERROR_NO_ERROR, want), "{mode}");
        }
    }

    #[test]
    fn a_hardware_configuration_failure_writes_nothing() {
        let (got, c) = hardware_with(status::NOT_SUPPORTED, vec![]);
        assert_eq!(
            got,
            crate::status::TOBII_ERROR_NOT_SUPPORTED,
            "what the ET5 answers on Linux, passed through"
        );
        assert_eq!(c, sentinel_configuration());
        let (got, c) = hardware_with(0, vec![3]);
        assert_eq!(got, TOBII_ERROR_INTERNAL, "a malformed reply");
        assert_eq!(c, sentinel_configuration());

        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut out = sentinel_configuration();
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `out` is a live local.
        unsafe {
            assert_eq!(
                tobii_hardware_configuration_get(d, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_hardware_configuration_get(ptr::null_mut(), &raw mut out),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_device_destroy(d), 0);
        }
        assert_eq!(out, sentinel_configuration());
    }

    #[test]
    fn stream_ids_map_as_in_the_dll() {
        let ids = [
            (0x500, 1),
            (0x501, 2),
            (0x502, 3),
            (0x503, 14),
            (0x504, 4),
            (0x505, 5),
            (0x506, 8),
            (0x507, 9),
            (0x508, 11),
            (0x509, 0),
            (0x50a, 6),
            (0x1770, 7),
            (0x50e, 0),
            (0x1771, 0),
            (0x1774, 0),
            (0, 0),
        ];
        for (id, want) in ids {
            assert_eq!(se_stream_type(id), want, "{id:#x}");
        }
    }
}
