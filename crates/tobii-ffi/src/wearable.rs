//! `tobii_wearable.h`: head-mounted devices only; nothing here applies to the
//! screen-based ET5.

use std::ffi::c_void;

use crate::stub::not_supported;

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
    fn tobii_lens_configuration_writable(device: *mut c_void, writable: *mut c_void);
    /// Wearable devices only.
    fn tobii_wearable_foveated_gaze_subscribe(device: *mut c_void, callback: *const c_void, user_data: *mut c_void);
    /// Wearable devices only.
    fn tobii_wearable_foveated_gaze_unsubscribe(device: *mut c_void);
}
