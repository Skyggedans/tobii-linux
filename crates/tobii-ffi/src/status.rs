//! `tobii_error_t`: the status every entry point returns, in the Stream
//! Engine's own numbering, and `tobii_error_message`.

use std::ffi::c_char;
use std::io::Write as _;
use std::sync::atomic::{AtomicU8, Ordering};

/// Status code returned by every entry point; one of the `TOBII_ERROR_*` values.
pub type Status = i32;

// The numbering below is the Stream Engine's own, recovered from the jump table
// of `tobii_error_message` in `tobii_stream_engine.dll`, and so are the texts
// that function returns. A client that switches on these values has to see
// the same numbers this library returns, so keep them as they are. Codes 2,
// 7 and 17 to 20 never leave this library; they exist so the shared header
// can name them.

/// The call succeeded.
pub const TOBII_ERROR_NO_ERROR: Status = 0;
/// Unrecoverable internal failure; here, a reply from tobiid that does not
/// decode, which is logged too.
pub const TOBII_ERROR_INTERNAL: Status = 1;
/// A restricted feature was used without the permission to do so.
pub const TOBII_ERROR_INSUFFICIENT_LICENSE: Status = 2;
/// The device does not support the feature.
pub const TOBII_ERROR_NOT_SUPPORTED: Status = 3;
/// No device is available; here, the tracker is paused (a clock pair, a
/// calibration start) or tobiid knows no calibration id yet.
pub const TOBII_ERROR_NOT_AVAILABLE: Status = 4;
/// The daemon could not be reached (or spawned), the connection dropped, the
/// daemon has no tracker for a call that needs one live (a clock pair, a
/// pause, starting, retrieving or applying a calibration, a display-area
/// write), or the tracker re-initialised or was lost while the call ran.
pub const TOBII_ERROR_CONNECTION_FAILED: Status = 5;
/// No sample (or no daemon acknowledgement) arrived before the timeout.
pub const TOBII_ERROR_TIMED_OUT: Status = 6;
/// Memory could not be allocated.
pub const TOBII_ERROR_ALLOCATION_FAILED: Status = 7;
/// A null or otherwise unusable argument was passed.
pub const TOBII_ERROR_INVALID_PARAMETER: Status = 8;
/// Calibration has already been started.
pub const TOBII_ERROR_CALIBRATION_ALREADY_STARTED: Status = 9;
/// Calibration has not been started; the daemon ended the session without
/// its owner (the tracker re-initialised or went away); or the tracker
/// refused a collect, discard, clear or compute for having no session, and
/// the daemon keeps its session for the owner to stop.
pub const TOBII_ERROR_CALIBRATION_NOT_STARTED: Status = 10;
/// The stream is already subscribed on this device.
pub const TOBII_ERROR_ALREADY_SUBSCRIBED: Status = 11;
/// The stream is not subscribed on this device.
pub const TOBII_ERROR_NOT_SUBSCRIBED: Status = 12;
/// The operation failed: the tracker refused a request or tobiid could not
/// complete it, or `tobii_calibration_parse` refused the blob.
pub const TOBII_ERROR_OPERATION_FAILED: Status = 13;
/// `tobii_wait_for_callbacks` was given devices from different API handles,
/// as in the DLL, or the daemon refused a subscription, which the current
/// daemon never does.
pub const TOBII_ERROR_CONFLICTING_API_INSTANCES: Status = 14;
/// Another client is calibrating the device.
pub const TOBII_ERROR_CALIBRATION_BUSY: Status = 15;
/// An API function was called from inside an API callback, or from inside
/// the logger, on the thread that runs it.
pub const TOBII_ERROR_CALLBACK_IN_PROGRESS: Status = 16;
/// The stream already has as many subscribers as it accepts.
pub const TOBII_ERROR_TOO_MANY_SUBSCRIBERS: Status = 17;
/// The platform runtime driver failed to connect.
pub const TOBII_ERROR_CONNECTION_FAILED_DRIVER: Status = 18;
/// The caller is not authorised to use the feature.
pub const TOBII_ERROR_UNAUTHORIZED: Status = 19;
/// A firmware upgrade is in progress.
pub const TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS: Status = 20;

/// Return a NUL-terminated description of a `TOBII_ERROR_*` code, never null
/// and never to be freed: the DLL's own text, verbatim. Its function
/// (0x180144cb0) picks one from a jump table (0x180144df4) of `.rdata` strings
/// (0x180220b90..0x180220f30), and codes 2 and 19 share one case (0x180144dac)
/// and so one text. A code outside 0..=20, a negative one included (the DLL
/// compares it unsigned), gets the DLL's generic text instead, in one buffer
/// the whole process shares (see `undefined_message`).
#[unsafe(no_mangle)]
pub extern "C" fn tobii_error_message(error: Status) -> *const c_char {
    let message = match error {
        TOBII_ERROR_NO_ERROR => c"No error.",
        TOBII_ERROR_INTERNAL => c"Internal error. Not recoverable. Please contact support.",
        TOBII_ERROR_INSUFFICIENT_LICENSE | TOBII_ERROR_UNAUTHORIZED => {
            c"Insufficient permissions when using a restricted feature."
        }
        TOBII_ERROR_NOT_SUPPORTED => c"Attempt to use a feature which is not supported.",
        TOBII_ERROR_NOT_AVAILABLE => c"No device is available.",
        TOBII_ERROR_CONNECTION_FAILED => {
            c"Connection to the eye tracker was lost or could not be established."
        }
        TOBII_ERROR_TIMED_OUT => c"The wait timed out after the specified time period.",
        TOBII_ERROR_ALLOCATION_FAILED => c"Memory could not be allocated.",
        TOBII_ERROR_INVALID_PARAMETER => c"API usage error: Invalid parameter.",
        TOBII_ERROR_CALIBRATION_ALREADY_STARTED => c"API usage error: Calibration already started.",
        TOBII_ERROR_CALIBRATION_NOT_STARTED => c"API usage error: Calibration not started.",
        TOBII_ERROR_ALREADY_SUBSCRIBED => c"API usage error: Already subscribed.",
        TOBII_ERROR_NOT_SUBSCRIBED => c"API usage error: Not subscribed.",
        TOBII_ERROR_OPERATION_FAILED => c"Operation failed.",
        TOBII_ERROR_CONFLICTING_API_INSTANCES => c"API usage error: Conflicting API instances.",
        TOBII_ERROR_CALIBRATION_BUSY => c"Another client is currently calibrating the device.",
        TOBII_ERROR_CALLBACK_IN_PROGRESS => {
            c"API usage error: An API function was called from within an API callback."
        }
        TOBII_ERROR_TOO_MANY_SUBSCRIBERS => c"Too many subscribers for requested stream.",
        TOBII_ERROR_CONNECTION_FAILED_DRIVER => {
            c"A connection failure occurred in the platform runtime driver."
        }
        TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS => c"Tracker firmware upgrade is in progress.",
        _ => return undefined_message(error),
    };
    message.as_ptr()
}

/// Size of the DLL's buffer for an out-of-range code's text: it formats the
/// text with `snprintf(buffer, 0x40, ...)` (0x180144dd5).
const UNDEFINED_SIZE: usize = 64;

/// The DLL's one buffer for an out-of-range code's text (0x18024ee60, zeroed
/// `.bss`), shared by the whole process: every such call returns it, holding
/// the text of the last one made on any thread. Its bytes 53.. only ever hold
/// zero, so a reader, even one racing a writer, always finds a NUL in it.
static UNDEFINED: [AtomicU8; UNDEFINED_SIZE] = [const { AtomicU8::new(0) }; UNDEFINED_SIZE];

/// Write `error`'s generic text into [`UNDEFINED`] and return the buffer, as
/// the DLL's default case (0x180144dc4) does. The pointer never dangles; the
/// next out-of-range call, on any thread, rewrites what it points at.
fn undefined_message(error: Status) -> *const c_char {
    // Relaxed: the caller reads it on its own thread, and the DLL orders
    // nothing between threads either.
    for (slot, byte) in UNDEFINED.iter().zip(undefined_text(error)) {
        slot.store(byte, Ordering::Relaxed);
    }
    // AtomicU8 has u8's size, alignment and bit validity.
    UNDEFINED.as_ptr().cast()
}

/// The DLL's generic text for an out-of-range code (format at 0x180220f60),
/// NUL-padded to the buffer's size.
fn undefined_text(error: Status) -> [u8; UNDEFINED_SIZE] {
    let mut text = [0; UNDEFINED_SIZE];
    let mut rest = &mut text[..UNDEFINED_SIZE - 1];
    // `{:x}` of an i32 is its two's complement, as the DLL's `%x`: -1 gives
    // 0xffffffff. The text is at most 53 bytes, so this cannot fail; a longer
    // one would be cut short at 63, as snprintf cuts it.
    let _ = write!(
        rest,
        "Undefined error (0x{error:x}). Please contact support."
    );
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// What `tobii_error_message` returns for `error`, as text.
    fn text(error: Status) -> String {
        // SAFETY: the pointer is never null and stays NUL-terminated for the
        // life of the process: a literal, or the shared buffer, whose last
        // bytes stay zero. Nothing writes the shared buffer while the CStr
        // lives: `out_of_range_codes_share_one_buffer` is the only test that
        // passes an out-of-range code, and it reads on its own thread.
        unsafe { CStr::from_ptr(tobii_error_message(error)) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn messages_are_the_dlls() {
        // The DLL's jump table (0x180144df4), in code order.
        let dll = [
            (TOBII_ERROR_NO_ERROR, "No error."),
            (
                TOBII_ERROR_INTERNAL,
                "Internal error. Not recoverable. Please contact support.",
            ),
            (
                TOBII_ERROR_INSUFFICIENT_LICENSE,
                "Insufficient permissions when using a restricted feature.",
            ),
            (
                TOBII_ERROR_NOT_SUPPORTED,
                "Attempt to use a feature which is not supported.",
            ),
            (TOBII_ERROR_NOT_AVAILABLE, "No device is available."),
            (
                TOBII_ERROR_CONNECTION_FAILED,
                "Connection to the eye tracker was lost or could not be established.",
            ),
            (
                TOBII_ERROR_TIMED_OUT,
                "The wait timed out after the specified time period.",
            ),
            (
                TOBII_ERROR_ALLOCATION_FAILED,
                "Memory could not be allocated.",
            ),
            (
                TOBII_ERROR_INVALID_PARAMETER,
                "API usage error: Invalid parameter.",
            ),
            (
                TOBII_ERROR_CALIBRATION_ALREADY_STARTED,
                "API usage error: Calibration already started.",
            ),
            (
                TOBII_ERROR_CALIBRATION_NOT_STARTED,
                "API usage error: Calibration not started.",
            ),
            (
                TOBII_ERROR_ALREADY_SUBSCRIBED,
                "API usage error: Already subscribed.",
            ),
            (
                TOBII_ERROR_NOT_SUBSCRIBED,
                "API usage error: Not subscribed.",
            ),
            (TOBII_ERROR_OPERATION_FAILED, "Operation failed."),
            (
                TOBII_ERROR_CONFLICTING_API_INSTANCES,
                "API usage error: Conflicting API instances.",
            ),
            (
                TOBII_ERROR_CALIBRATION_BUSY,
                "Another client is currently calibrating the device.",
            ),
            (
                TOBII_ERROR_CALLBACK_IN_PROGRESS,
                "API usage error: An API function was called from within an API callback.",
            ),
            (
                TOBII_ERROR_TOO_MANY_SUBSCRIBERS,
                "Too many subscribers for requested stream.",
            ),
            (
                TOBII_ERROR_CONNECTION_FAILED_DRIVER,
                "A connection failure occurred in the platform runtime driver.",
            ),
            (
                TOBII_ERROR_UNAUTHORIZED,
                "Insufficient permissions when using a restricted feature.",
            ),
            (
                TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS,
                "Tracker firmware upgrade is in progress.",
            ),
        ];
        for (index, (code, expected)) in dll.into_iter().enumerate() {
            assert_eq!(usize::try_from(code), Ok(index), "row {index}");
            assert_eq!(text(code), expected, "code {code}");
        }
        // One case in the DLL (0x180144dac).
        assert_eq!(
            text(TOBII_ERROR_INSUFFICIENT_LICENSE),
            text(TOBII_ERROR_UNAUTHORIZED)
        );
    }

    #[test]
    fn an_out_of_range_code_gets_the_dlls_generic_text() {
        for (code, expected) in [
            (21, c"Undefined error (0x15). Please contact support."),
            (9999, c"Undefined error (0x270f). Please contact support."),
            (-1, c"Undefined error (0xffffffff). Please contact support."),
            (
                i32::MIN,
                c"Undefined error (0x80000000). Please contact support.",
            ),
            (
                i32::MAX,
                c"Undefined error (0x7fffffff). Please contact support.",
            ),
        ] {
            let text = undefined_text(code);
            assert_eq!(
                CStr::from_bytes_until_nul(&text),
                Ok(expected),
                "code {code}"
            );
            // At most 53 bytes, so the shared buffer's bytes 53.. stay zero.
            assert!(text[53..].iter().all(|&b| b == 0), "code {code}");
        }
    }

    /// The only test that writes the shared buffer: tests run on parallel
    /// threads, and one passing an out-of-range code elsewhere would race it.
    #[test]
    fn out_of_range_codes_share_one_buffer() {
        let first = tobii_error_message(21);
        assert_eq!(
            text(-1),
            "Undefined error (0xffffffff). Please contact support."
        );
        assert_eq!(tobii_error_message(-1), first);
        // SAFETY: as in `text`.
        let now = unsafe { CStr::from_ptr(first) };
        assert_eq!(
            now,
            c"Undefined error (0xffffffff). Please contact support."
        );
    }
}
