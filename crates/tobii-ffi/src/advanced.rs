//! `tobii_advanced.h`: per-eye gaze data; the sync port and face types are not
//! available.

use std::ffi::c_void;

use crate::device::Device;
use crate::status::Status;
use crate::streams::{subscribe, unsubscribe};
use crate::stub::not_supported;
use crate::types::GazeDataFn;

/// Per-eye gaze data: origins, gaze points and eyeball centres in the tracker
/// frame, normalised positions and display gaze points. The ET5 reports no
/// pupil diameter, so `pupil_validity` is always invalid.
///
/// # Safety
/// As the subscribe functions in `tobii_streams.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_gaze_data_subscribe(
    device: *mut Device,
    callback: Option<GazeDataFn>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { subscribe(device, |c| &mut c.gaze_data, callback, user_data) }
}

/// Undo `tobii_gaze_data_subscribe`.
///
/// # Safety
/// `device` must be null or a live handle that no other thread uses during
/// the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_gaze_data_unsubscribe(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe { unsubscribe(device, |c| &mut c.gaze_data) }
}

not_supported! {
    /// The ET5 has no digital sync port.
    fn tobii_digital_syncport_subscribe(device: *mut c_void, callback: *const c_void, user_data: *mut c_void);
    /// The ET5 has no digital sync port.
    fn tobii_digital_syncport_unsubscribe(device: *mut c_void);
    /// Face types were never captured.
    fn tobii_enumerate_face_types(device: *mut c_void, receiver: *const c_void, user_data: *mut c_void);
    /// Face types were never captured.
    fn tobii_set_face_type(device: *mut c_void, face_type: *const c_void);
    /// Face types were never captured.
    fn tobii_get_face_type(device: *mut c_void, face_type: *mut c_void);
}
