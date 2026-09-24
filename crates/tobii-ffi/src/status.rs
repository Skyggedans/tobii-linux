//! `tobii_error_t`: the status every entry point returns, in the Stream
//! Engine's own numbering, and `tobii_error_message`.

use std::ffi::c_char;

/// Status code returned by every entry point; one of the `TOBII_ERROR_*` values.
pub type Status = i32;

// The numbering below is the Stream Engine's own, recovered from the jump table
// of `tobii_error_message` in `tobii_stream_engine.dll`. A client that switches
// on these values has to see the same numbers this library returns, so keep
// them as they are. Only the codes marked "returned here" ever leave this
// library; the rest exist so the shared header can name them.

/// The call succeeded.
pub const TOBII_ERROR_NO_ERROR: Status = 0;
/// Unrecoverable internal failure.
pub const TOBII_ERROR_INTERNAL: Status = 1;
/// A restricted feature was used without the permission to do so.
pub const TOBII_ERROR_INSUFFICIENT_LICENSE: Status = 2;
/// The device does not support the feature.
pub const TOBII_ERROR_NOT_SUPPORTED: Status = 3;
/// No device is available.
pub const TOBII_ERROR_NOT_AVAILABLE: Status = 4;
/// Returned here: the daemon could not be reached (or spawned), the
/// connection dropped, or the daemon has no tracker for a call that needs
/// one live (a clock pair, a pause, a calibration, a display-area write).
pub const TOBII_ERROR_CONNECTION_FAILED: Status = 5;
/// Returned here: no sample (or no daemon acknowledgement) arrived before the
/// timeout.
pub const TOBII_ERROR_TIMED_OUT: Status = 6;
/// Memory could not be allocated.
pub const TOBII_ERROR_ALLOCATION_FAILED: Status = 7;
/// Returned here: a null or otherwise unusable argument was passed.
pub const TOBII_ERROR_INVALID_PARAMETER: Status = 8;
/// Calibration has already been started.
pub const TOBII_ERROR_CALIBRATION_ALREADY_STARTED: Status = 9;
/// Calibration has not been started.
pub const TOBII_ERROR_CALIBRATION_NOT_STARTED: Status = 10;
/// The stream is already subscribed on this device.
pub const TOBII_ERROR_ALREADY_SUBSCRIBED: Status = 11;
/// The stream is not subscribed on this device.
pub const TOBII_ERROR_NOT_SUBSCRIBED: Status = 12;
/// The operation failed.
pub const TOBII_ERROR_OPERATION_FAILED: Status = 13;
/// Returned here: the daemon refused the subscription. Kept for ABI
/// compatibility; the current daemon never refuses.
pub const TOBII_ERROR_CONFLICTING_API_INSTANCES: Status = 14;
/// Another client is calibrating the device.
pub const TOBII_ERROR_CALIBRATION_BUSY: Status = 15;
/// An API function was called from inside an API callback.
pub const TOBII_ERROR_CALLBACK_IN_PROGRESS: Status = 16;
/// The stream already has as many subscribers as it accepts.
pub const TOBII_ERROR_TOO_MANY_SUBSCRIBERS: Status = 17;
/// The platform runtime driver failed to connect.
pub const TOBII_ERROR_CONNECTION_FAILED_DRIVER: Status = 18;
/// The caller is not authorised to use the feature.
pub const TOBII_ERROR_UNAUTHORIZED: Status = 19;
/// A firmware upgrade is in progress.
pub const TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS: Status = 20;

/// Return a static, NUL-terminated description of a `TOBII_ERROR_*` code.
/// Unknown codes get a generic message rather than a null pointer, so callers
/// can print the result unconditionally.
#[unsafe(no_mangle)]
pub extern "C" fn tobii_error_message(error: Status) -> *const c_char {
    let message = match error {
        TOBII_ERROR_NO_ERROR => c"no error",
        TOBII_ERROR_INTERNAL => c"unrecoverable internal error",
        TOBII_ERROR_INSUFFICIENT_LICENSE => c"insufficient license for the feature",
        TOBII_ERROR_NOT_SUPPORTED => c"feature not supported by the device",
        TOBII_ERROR_NOT_AVAILABLE => c"no device is available",
        TOBII_ERROR_CONNECTION_FAILED => c"connection to the eye tracker failed or was lost",
        TOBII_ERROR_TIMED_OUT => c"the wait timed out",
        TOBII_ERROR_ALLOCATION_FAILED => c"memory could not be allocated",
        TOBII_ERROR_INVALID_PARAMETER => c"invalid parameter",
        TOBII_ERROR_CALIBRATION_ALREADY_STARTED => c"calibration already started",
        TOBII_ERROR_CALIBRATION_NOT_STARTED => c"calibration not started",
        TOBII_ERROR_ALREADY_SUBSCRIBED => c"already subscribed",
        TOBII_ERROR_NOT_SUBSCRIBED => c"not subscribed",
        TOBII_ERROR_OPERATION_FAILED => c"operation failed",
        TOBII_ERROR_CONFLICTING_API_INSTANCES => c"conflicting API instances",
        TOBII_ERROR_CALIBRATION_BUSY => c"another client is calibrating the device",
        TOBII_ERROR_CALLBACK_IN_PROGRESS => c"an API function was called from an API callback",
        TOBII_ERROR_TOO_MANY_SUBSCRIBERS => c"too many subscribers for the stream",
        TOBII_ERROR_CONNECTION_FAILED_DRIVER => c"the platform runtime driver failed to connect",
        TOBII_ERROR_UNAUTHORIZED => c"unauthorized",
        TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS => c"firmware upgrade in progress",
        _ => c"unknown error code",
    };
    message.as_ptr()
}
