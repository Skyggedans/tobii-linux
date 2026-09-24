//! The 73 exports of `tobii_stream_engine.dll` 4.1.0.3 that its documentation
//! does not cover (all but `tobii_calibration_stimulus_points_get`, which
//! lives with calibration). Their argument counts come from the DLL (see
//! `tools/abi`); their types are best-effort, which is safe because only the
//! field-of-use, image, internal-stream, timesync and stream-type functions
//! below read their arguments.

use std::ffi::c_void;

use tobii_ipc::request::{self, decode_stream_types, decode_timesync, kind};

use crate::api::{FACTS_TIMEOUT, write_supported};
use crate::device::{Device, device_mut};
use crate::status::{
    Status, TOBII_ERROR_INTERNAL, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
};
use crate::streams::{subscribe, unsubscribe};
use crate::stub::not_supported;
use crate::types::{
    FieldOfUse, FieldOfUseFn, ImageFn, StreamType, StreamTypeReceiver, TimesyncData, copy_c_string,
};

/// The field of use the device was created with.
///
/// # Safety
/// `device` must be null or a live handle that no other thread uses during
/// the call; `field_of_use` must be null or valid for writing one
/// `tobii_field_of_use_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_field_of_use(
    device: *mut Device,
    field_of_use: *mut FieldOfUse,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
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
/// `device` must be null or a live handle that no other thread uses during
/// the call.
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
/// `device` must be null or a live handle that no other thread uses during
/// the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_image_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { unsubscribe(device, |c| &mut c.image) }
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
/// clean IR, custom and image-collection streams are stubs), not the DLL.
/// For an ET5 the DLL would support 0, 2 and 6 on its TTP path and none on
/// its PRP path. As in the DLL, an unknown id is reported unsupported, not
/// an error, and an id above `i32::MAX` is an invalid parameter.
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

/// A fresh tracker/host clock pair: the tracker clock read `tracker_us` at
/// some host time between `system_start_us` and `system_end_us`.
///
/// The daemon answers from the first gaze frame it receives after the
/// request (starting the tracker if needed): `tracker_us` is the frame's
/// device timestamp and the bracket is the 30 ms before the daemon read it.
/// The DLL instead times a round trip to its service, and its own offset
/// estimator skips pairs wider than 6 ms, though it still returns them. The
/// host clock is `tobii_system_clock`'s, `CLOCK_REALTIME` rather than the
/// DLL's monotonic QPC. Nothing is written unless the call succeeds.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `timesync` must be null or
/// valid for writing one `tobii_timesync_data_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_timesync(
    device: *mut Device,
    timesync: *mut TimesyncData,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if timesync.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::TIMESYNC, &[], FACTS_TIMEOUT)
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
        Ok(None) => TOBII_ERROR_INTERNAL,
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let entries: Vec<StreamType> = match d
        .request(kind::STREAM_TYPES, &[], FACTS_TIMEOUT)
        .map(|p| decode_stream_types(&p))
    {
        Ok(Some(types)) => types.iter().map(stream_type_c).collect(),
        Ok(None) => return TOBII_ERROR_INTERNAL,
        Err(status) => return status,
    };
    for entry in &entries {
        // SAFETY: the caller guarantees `receiver` is sound to call like
        // this; `entry` outlives the call.
        unsafe { receiver(entry, user_data) };
    }
    TOBII_ERROR_NO_ERROR
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
    fn tobii_gaze_raw_subscribe(device: P, callback: C, user_data: P);
    fn tobii_gaze_raw_unsubscribe(device: P);
    fn tobii_get_combined_gaze_hid_track_box(device: P, track_box: P);
    fn tobii_get_configuration_key(device: P, key: P, value: P);
    fn tobii_get_device_info_internal(device: P, info: P);
    fn tobii_get_display_id(device: P, display_id: P);
    fn tobii_get_display_info(device: P, display_info: P);
    fn tobii_get_face_id_parameters(device: P, parameters: P);
    fn tobii_get_face_id_state(device: P, state: P);
    fn tobii_get_gaze_hid_enabled(device: P, enabled: P);
    fn tobii_get_illumination_mode(device: P, mode: P);
    fn tobii_hardware_configuration_get(device: P, configuration: P);
    fn tobii_image_collection_subscribe(device: P, callback: C, user_data: P);
    fn tobii_image_collection_unsubscribe(device: P);
    fn tobii_internal_capability_supported(device: P, capability: u32, supported: P);
    fn tobii_logs_retrieve(device: P, receiver: C, user_data: P);
    fn tobii_low_frequency_head_position_subscribe(device: P, callback: C, user_data: P);
    fn tobii_low_frequency_head_position_unsubscribe(device: P);
    fn tobii_low_frequency_head_rotation_subscribe(device: P, callback: C, user_data: P);
    fn tobii_low_frequency_head_rotation_unsubscribe(device: P);
    fn tobii_multiple_faces_position_subscribe(device: P, callback: C, user_data: P);
    fn tobii_multiple_faces_position_unsubscribe(device: P);
    fn tobii_open_realm(device: P, realm: u32, key: C, key_size: u32);
    fn tobii_pause_device(device: P);
    fn tobii_power_save_activate(device: P);
    fn tobii_power_save_deactivate(device: P);
    fn tobii_remote_wake_activate(device: P);
    fn tobii_remote_wake_deactivate(device: P);
    fn tobii_resume_device(device: P);
    fn tobii_secondary_camera_image_subscribe(device: P, callback: C, user_data: P);
    fn tobii_secondary_camera_image_unsubscribe(device: P);
    fn tobii_send_custom_command(device: P, command: u32, data: C, size: usize, receiver: C, user_data: P);
    fn tobii_send_statistics(device: P, data: C, size: usize);
    fn tobii_set_display_id(device: P, display_id: u32);
    fn tobii_set_display_info(device: P, display_info: C);
    fn tobii_set_face_id_parameters(device: P, parameters: C);
    fn tobii_set_fw_upgrade_allowed(device: P, allowed: u32);
    fn tobii_set_illumination_mode(device: P, mode: C);
    fn tobii_wearable_limited_image_subscribe(device: P, callback: C, user_data: P);
    fn tobii_wearable_limited_image_unsubscribe(device: P);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tobii_device_destroy;
    use crate::status::TOBII_ERROR_NOT_AVAILABLE;
    use crate::types::{TOBII_NOT_SUPPORTED, TOBII_SUPPORTED};
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
            host_start_us: 1_700_000_000_000_000,
            device_us: 5_000_000,
            host_end_us: 1_700_000_000_030_000,
        });
        assert_eq!(
            timesync_with(0, payload),
            (
                TOBII_ERROR_NO_ERROR,
                TimesyncData {
                    system_start_us: 1_700_000_000_000_000,
                    system_end_us: 1_700_000_000_030_000,
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
