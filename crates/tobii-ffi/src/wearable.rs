//! `tobii_wearable.h`: head-mounted devices only. Nothing here applies to the
//! screen-based ET5; only the lens-configuration writability query answers,
//! and its answer is "not writable".

use std::ffi::c_void;

use crate::device::{Device, device_mut};
use crate::status::{Status, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR};
use crate::stub::not_supported;
use crate::types::TOBII_LENS_CONFIGURATION_NOT_WRITABLE;

/// Whether the lens configuration can be written: never on the ET5.
///
/// The DLL answers from the device's list of writable properties (lens
/// configuration is property 0xa) and has no `TOBII_ERROR_NOT_SUPPORTED`
/// path. The ET5 has no lens configuration, so the answer is
/// `TOBII_LENS_CONFIGURATION_NOT_WRITABLE`.
///
/// # Safety
/// `device` as `tobii_device_process_callbacks`; `writable` must be null or
/// valid for writing one `tobii_lens_configuration_writable_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_lens_configuration_writable(
    device: *mut Device,
    writable: *mut u32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
        return status;
    }
    if writable.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { writable.write(TOBII_LENS_CONFIGURATION_NOT_WRITABLE) };
    TOBII_ERROR_NO_ERROR
}

not_supported! {
    /// Wearable devices only.
    fn tobii_wearable_consumer_data_subscribe(device: *mut c_void, callback: *const c_void, user_data: *mut c_void);
    /// Wearable devices only.
    fn tobii_wearable_consumer_data_unsubscribe(device: *mut c_void);
    /// Wearable devices only.
    fn tobii_wearable_advanced_data_subscribe(device: *mut c_void, callback: *const c_void, user_data: *mut c_void);
    /// Wearable devices only.
    fn tobii_wearable_advanced_data_unsubscribe(device: *mut c_void);
    /// Wearable devices only.
    fn tobii_get_lens_configuration(device: *mut c_void, lens_config: *mut c_void);
    /// Wearable devices only.
    fn tobii_set_lens_configuration(device: *mut c_void, lens_config: *const c_void);
    /// Wearable devices only.
    fn tobii_wearable_foveated_gaze_subscribe(device: *mut c_void, callback: *const c_void, user_data: *mut c_void);
    /// Wearable devices only.
    fn tobii_wearable_foveated_gaze_unsubscribe(device: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tobii_device_destroy;
    use std::ptr;

    #[test]
    fn the_lens_configuration_is_never_writable() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut w = 0xdead_beef_u32;
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // below; `w` is a live local.
        unsafe {
            assert_eq!(tobii_lens_configuration_writable(d, &raw mut w), 0);
            assert_eq!(w, TOBII_LENS_CONFIGURATION_NOT_WRITABLE);
            assert_eq!(
                tobii_lens_configuration_writable(d, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            w = 0xdead_beef;
            assert_eq!(
                tobii_lens_configuration_writable(ptr::null_mut(), &raw mut w),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(w, 0xdead_beef, "nothing written");
            assert_eq!(tobii_device_destroy(d), 0);
        }
    }
}
