//! `tobii_licensing.h`. There is nothing to license on Linux: every key
//! validates, and the feature group is consumer. No entry point checks it.
//! What the Stream Engine reserves for a higher group is served anyway: gaze
//! data and `tobii_timesync` (professional), calibration and the
//! display-area and device-name writes (config or professional), the IR
//! image (additional features), the stream catalogue (internal) and, in the
//! DLL's own service, pause (internal).

use std::ffi::{c_char, c_void};

use crate::api::create_device;
use crate::device::{Api, Device, device_ref};
use crate::status::{Status, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR};
use crate::stub::not_supported;
use crate::types::{
    FieldOfUse, LicenseKey, TOBII_FEATURE_GROUP_CONSUMER, TOBII_LICENSE_VALIDATION_RESULT_OK,
};

/// `tobii_device_create` with license keys: each key is reported valid
/// without being read.
///
/// # Safety
/// As `tobii_device_create`; `license_results` must be null or valid for
/// writing `license_count` `tobii_license_validation_result_t` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_device_create_ex(
    api: *mut Api,
    _url: *const c_char,
    field_of_use: FieldOfUse,
    license_keys: *const LicenseKey,
    license_count: i32,
    license_results: *mut u32,
    device: *mut *mut Device,
) -> Status {
    let Ok(count) = usize::try_from(license_count) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if count > 0 && license_keys.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: forwarded under the same contract.
    let status = unsafe { create_device(api, field_of_use, device) };
    if status == TOBII_ERROR_NO_ERROR && !license_results.is_null() {
        for i in 0..count {
            // SAFETY: the caller guarantees `license_count` writable results.
            unsafe {
                license_results
                    .add(i)
                    .write(TOBII_LICENSE_VALIDATION_RESULT_OK);
            }
        }
    }
    status
}

/// Consumer.
///
/// # Safety
/// `device` must be null or a live handle that is not destroyed before the
/// call returns; `feature_group` must be null or valid for writing one
/// `tobii_feature_group_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_feature_group(
    device: *mut Device,
    feature_group: *mut u32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    if feature_group.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { feature_group.write(TOBII_FEATURE_GROUP_CONSUMER) };
    TOBII_ERROR_NO_ERROR
}

not_supported! {
    /// There is no license storage on the device we drive.
    fn tobii_license_key_store(device: *mut c_void, data: *mut c_void, size: usize);
    /// There is no license storage on the device we drive.
    fn tobii_license_key_retrieve(device: *mut c_void, receiver: *const c_void, user_data: *mut c_void);
}
