//! `tobii_streams.h`: the sample streams, plus the `tobii_recenter` extension.
//!
//! Every subscribe entry point has the same contract: `device` must be null
//! or a live handle from `tobii_device_create` that no other thread uses
//! during the call; `callback` must be null (rejected) or sound to invoke with
//! a valid pointer to its sample type and `user_data`, must not re-enter this
//! library (such calls are refused with `TOBII_ERROR_CALLBACK_IN_PROGRESS`),
//! and `user_data` must stay valid until the stream is unsubscribed or the
//! device destroyed. Callbacks run on the thread that calls
//! `tobii_device_process_callbacks`.

use std::ffi::c_void;

use tobii_ipc::write_frame;

use crate::device::{Callbacks, Device, Slot, device_mut};
use crate::status::{Status, TOBII_ERROR_CONNECTION_FAILED, TOBII_ERROR_NO_ERROR};
use crate::types::{EyePairFn, GazePointFn, HeadPoseFn, NotificationsFn, PresenceFn};

/// Subscribe `callback` into `slot` of the device behind `device`.
///
/// # Safety
/// See the module documentation.
pub(crate) unsafe fn subscribe<F: Copy>(
    device: *mut Device,
    slot: fn(&mut Callbacks) -> &mut Slot<F>,
    callback: Option<F>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(d) => d.subscribe(slot, callback, user_data),
        Err(status) => status,
    }
}

/// Unsubscribe `slot` of the device behind `device`.
///
/// # Safety
/// `device` must be null or a live handle that no other thread uses during
/// the call.
pub(crate) unsafe fn unsubscribe<F: Copy>(
    device: *mut Device,
    slot: fn(&mut Callbacks) -> &mut Slot<F>,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
        Ok(d) => d.unsubscribe(slot),
        Err(status) => status,
    }
}

/// Define a subscribe/unsubscribe pair over one callback slot.
macro_rules! stream_pair {
    ($(
        $(#[$doc:meta])*
        $sub:ident / $unsub:ident: $slot:ident, $fn_ty:ty;
    )+) => { $(
        $(#[$doc])*
        ///
        /// # Safety
        /// See the module documentation.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $sub(device: *mut Device, callback: Option<$fn_ty>, user_data: *mut c_void) -> Status {
            // SAFETY: forwarded under the same contract.
            unsafe { subscribe(device, |c| &mut c.$slot, callback, user_data) }
        }

        #[doc = concat!("Undo `", stringify!($sub), "`.")]
        ///
        /// # Safety
        /// `device` must be null or a live handle that no other thread uses
        /// during the call.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $unsub(device: *mut Device) -> Status {
            // SAFETY: forwarded under the same contract.
            unsafe { unsubscribe(device, |c| &mut c.$slot) }
        }
    )+ };
}

stream_pair! {
    /// Gaze point, normalised display coordinates (the device's filtered
    /// combined gaze, unclamped, as the Stream Engine reports it).
    tobii_gaze_point_subscribe / tobii_gaze_point_unsubscribe: gaze, GazePointFn;
    /// Per-eye gaze origin in the display frame, mm.
    tobii_gaze_origin_subscribe / tobii_gaze_origin_unsubscribe: gaze_origin, EyePairFn;
    /// Per-eye position normalised to the track box.
    tobii_eye_position_normalized_subscribe / tobii_eye_position_normalized_unsubscribe: eye_position, EyePairFn;
    /// User presence, reported once on subscribe and then on change.
    tobii_user_presence_subscribe / tobii_user_presence_unsubscribe: presence, PresenceFn;
    /// Head pose from the tracker's IR camera: position in mm, rotation in
    /// radians about x (pitch), y (yaw) and z (roll).
    tobii_head_pose_subscribe / tobii_head_pose_unsubscribe: head, HeadPoseFn;
    /// Device notifications: display-area and calibration changes.
    tobii_notifications_subscribe / tobii_notifications_unsubscribe: notifications, NotificationsFn;
    /// The user position guide: the track-box-normalised eye positions.
    tobii_user_position_guide_subscribe / tobii_user_position_guide_unsubscribe: user_position_guide, EyePairFn;
}

/// Recalibrate the head rest pose now (extension; not in the original Stream
/// Engine). Affects whichever client currently holds the device's mode.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_recenter(device: *mut Device) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    match write_frame(&mut d.stream(), &tobii_ipc::encode_recenter()) {
        Ok(()) => TOBII_ERROR_NO_ERROR,
        Err(_) => TOBII_ERROR_CONNECTION_FAILED,
    }
}
