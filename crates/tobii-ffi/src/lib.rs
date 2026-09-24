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
//!   head pose, gaze data, IR image and notification streams, and 2-D
//!   calibration (the result is saved as the user's calibration).
//! - **Answered locally** — API version (4.1.0.3), system clock, output
//!   frequency (33 Hz), enabled eye (both), feature group (consumer),
//!   license validation (every key valid), display-area calculation,
//!   calibration parsing.
//! - **`TOBII_ERROR_NOT_SUPPORTED`** — everything the ET5 was never observed
//!   doing: wearable, face id, illumination, power and pause control,
//!   firmware, diagnostics, extensions, custom streams, 3-D and per-eye
//!   calibration.
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
                "set_device_name",
                crate::config::tobii_set_device_name(n, c.cast()),
            ),
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
                "discard_data_2d",
                crate::calibration::tobii_calibration_discard_data_2d(n, 0.5, 0.5),
            ),
            (
                "stimulus_points_get",
                crate::calibration::tobii_calibration_stimulus_points_get(n, n),
            ),
            ("open_realm", crate::internal::tobii_open_realm(n, 0, c, 0)),
            ("pause_device", crate::internal::tobii_pause_device(n)),
            ("timesync", crate::internal::tobii_timesync(n, n)),
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
