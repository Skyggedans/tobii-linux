//! The 73 exports of `tobii_stream_engine.dll` 4.1.0.3 that its documentation
//! does not cover (all but `tobii_calibration_stimulus_points_get`, which
//! lives with calibration). Their argument counts come from the DLL (see
//! `tools/abi`); their types are best-effort, which is safe because only the
//! field-of-use, image and internal-stream functions below read their
//! arguments.

use std::ffi::c_void;

use crate::api::write_supported;
use crate::device::{Device, device_mut};
use crate::status::{Status, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR};
use crate::streams::{subscribe, unsubscribe};
use crate::stub::not_supported;
use crate::types::{FieldOfUse, FieldOfUseFn, ImageFn};

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
    fn tobii_enumerate_stream_types(device: P, receiver: C, user_data: P);
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
    fn tobii_timesync(device: P, timesync: P);
    fn tobii_wearable_limited_image_subscribe(device: P, callback: C, user_data: P);
    fn tobii_wearable_limited_image_unsubscribe(device: P);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tobii_device_destroy;
    use crate::types::{TOBII_NOT_SUPPORTED, TOBII_SUPPORTED};
    use std::ptr;

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
}
