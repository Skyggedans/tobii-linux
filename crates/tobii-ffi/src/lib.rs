//! `libtobii.so`: the C ABI of Tobii's Stream Engine 4.1, backed by the
//! `tobiid` daemon.
//!
//! Every one of the 153 entry points `tobii_stream_engine.dll` 4.1.0.3
//! exports is exported here with the same signature (plus the
//! `tobii_recenter` extension), so any Stream Engine client links and runs.
//! What stands behind each:
//!
//! - **Implemented** — device lifetime and callbacks, device info, track box,
//!   display area and mounting (read and write), states, capabilities, the
//!   gaze point, gaze origin, eye position, user position guide, presence,
//!   head pose, gaze data, IR image and notification streams, 2-D
//!   calibration (the result is saved as the user's calibration) and
//!   discarding a 2-D point, the device/host clock pair (`tobii_timesync`),
//!   the tracker's stream catalogue, device pause and resume, and the device
//!   name (kept by the host, not the tracker). The hardware configuration is
//!   implemented provisionally; the ET5 has reported none on Linux, so it
//!   answers `TOBII_ERROR_NOT_SUPPORTED` there.
//! - **Answered locally** — API version (4.1.0.3), system clock, output
//!   frequency (33 Hz), enabled eye (both), feature group (consumer),
//!   license validation (every key valid), display-area calculation,
//!   calibration parsing, internal-stream support (the IR image only), lens
//!   configuration writability (never).
//! - **`TOBII_ERROR_NOT_SUPPORTED`** — everything the ET5 was never observed
//!   doing: wearable, face id, illumination, power control, firmware,
//!   diagnostics, extensions, custom streams, 3-D and per-eye calibration.
//!
//! Nothing is gated by a licence. What the Stream Engine reserves for a
//! higher feature group or an extra licence (gaze data, timesync,
//! calibration, display-area and name writes, the IR image, the stream
//! catalogue, pause) works here under the consumer group this library
//! reports (see `licensing`).
//!
//! An entry point that takes a device checks its arguments in one order: a
//! call from inside a callback is `TOBII_ERROR_CALLBACK_IN_PROGRESS`, then a
//! null device is `TOBII_ERROR_INVALID_PARAMETER`, then its other arguments
//! are checked (`TOBII_ERROR_INVALID_PARAMETER` again). Two check an
//! argument first: `tobii_calibration_start` its eye and
//! `tobii_calibration_apply` its blob. The DLL checks null pointers before
//! the callback in some entry points, so the two differ only for an invalid
//! argument passed from inside a callback.
//!
//! `tobii_device_create` connects to the daemon (auto-spawning it if needed),
//! so several processes can use the device at once. Pump with
//! `tobii_wait_for_callbacks` + `tobii_device_process_callbacks` exactly like
//! the Stream Engine. Every sample carries the device clock.
//!
//! Every entry point takes raw handles from C, so each is an `unsafe fn` whose
//! `# Safety` section states what the caller must uphold; the `unsafe` blocks
//! inside are kept to the single pointer operation that needs them. Entry
//! points without an implementation read none of their arguments and are safe
//! functions (see `stub`).

mod advanced;
mod api;
mod calibration;
mod config;
mod device;
mod internal;
mod licensing;
mod status;
mod streams;
mod stub;
mod types;
mod wearable;

pub use device::{Api, Device};
pub use status::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use crate::status::{Status, TOBII_ERROR_NOT_SUPPORTED};
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};

    /// Every stub answers `TOBII_ERROR_NOT_SUPPORTED` whatever it is given,
    /// and naming each one here catches a typo before `make verify-abi` does.
    #[test]
    fn stubs_are_not_supported() {
        let n: *mut c_void = null_mut();
        let c: *const c_void = null();
        let results: Vec<(&str, Status)> = vec![
            (
                "license_key_store",
                crate::licensing::tobii_license_key_store(n, n, 0),
            ),
            (
                "license_key_retrieve",
                crate::licensing::tobii_license_key_retrieve(n, c, n),
            ),
            (
                "digital_syncport_subscribe",
                crate::advanced::tobii_digital_syncport_subscribe(n, c, n),
            ),
            ("get_face_type", crate::advanced::tobii_get_face_type(n, n)),
            (
                "wearable_consumer_data_subscribe",
                crate::wearable::tobii_wearable_consumer_data_subscribe(n, c, n),
            ),
            (
                "collect_data_3d",
                crate::calibration::tobii_calibration_collect_data_3d(n, 0.5, 0.5, 0.5),
            ),
            (
                "stimulus_points_get",
                crate::calibration::tobii_calibration_stimulus_points_get(n, n),
            ),
            ("open_realm", crate::internal::tobii_open_realm(n, 0, c, 0)),
            (
                "send_custom_command",
                crate::internal::tobii_send_custom_command(n, 0, c, 0, c, n),
            ),
        ];
        for (name, status) in results {
            assert_eq!(status, TOBII_ERROR_NOT_SUPPORTED, "{name}");
        }
    }
}
