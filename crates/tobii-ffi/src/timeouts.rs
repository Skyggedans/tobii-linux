//! How long a request waits for tobiid's answer: longer than the daemon may
//! take over it, so that the call reports what the daemon did. One that gave
//! up first would return `TOBII_ERROR_TIMED_OUT` while the daemon carried
//! the request out (a calibration started, a display area set), and the
//! device's next request would wait behind it, since the daemon runs a
//! connection's requests one at a time. `TOBII_ERROR_TIMED_OUT` from a
//! request is then the daemon's own answer, or a daemon held up past its
//! deadlines by what they leave out (its state lock held while it stops an
//! engine no client wants any more, whose thread it joins, say), which may
//! still carry the request out.
//!
//! The daemon's waits are its deadlines (see [`tobii_ipc::deadline`]). A
//! request answered from the facts the tracker reported at its last init
//! waits for its first init. One that needs the tracker live runs device
//! commands, which the daemon's engine takes only once the tracker streams
//! (a cold start takes ~12 s: an init, a re-open to prime the stream, a
//! second init): each may wait 30 s for the engine to take and write it,
//! then its own deadline for the answer
//! ([`deadline::command`](tobii_ipc::deadline::command)). A request that
//! runs several commands waits for each in turn, the whole 30 s each at
//! worst: the engine may re-open the tracker between two of them (after a
//! stall), and the next waits for the stream to arm again. The worst cases
//! are [`worst`]'s, which tobiid's tests hold its requests' device commands
//! to, and each timeout is at least [`MARGIN`] past its request's:
//!
//! | Timeout | Calls | The daemon waits for | Worst case |
//! |---|---|---|---|
//! | [`FACTS`] 12 s | the facts reads (below) | the facts, at the tracker's first init | 10 s |
//! | [`FACTS`] 12 s | `tobii_set_device_name` | nothing: it saves the name in a file | — |
//! | [`STATE`] 3 s | `tobii_get_state_bool`, `tobii_get_state_uint32` | nothing: its own state | — |
//! | [`TIMESYNC`] 27 s | `tobii_timesync` | a gaze frame newer than the request | 25 s |
//! | [`DEVICE_PAUSE`] 60 s | `tobii_pause_device`, `tobii_resume_device` | another pause or resume to finish (20 s), then 3100 (6 s) | 20 + 36 = 56 s |
//! | [`DISPLAY_AREA_SET`] 75 s | `tobii_set_display_area` | 1440 (5 s); when the caller's calibration session ended meanwhile, 1440 again with the area that end put back | 35 + 35 = 70 s |
//! | [`CALIBRATION_START`] 190 s | `tobii_calibration_start` | 1010, 1060 (5 s each), 1110 (10 s); when one fails, 1020 (5 s) and 1110 (10 s) to put the calibration back | 35 + 35 + 40 + 35 + 40 = 185 s |
//! | [`CALIBRATION_STOP`] 120 s | `tobii_calibration_stop` | 1020 (5 s), 1110 (10 s), then 1110 again for an engine started in place of a lost one, or 1440 (5 s) to put the display area back | 35 + 40 + 40 = 115 s |
//! | [`CALIBRATION_COLLECT_2D`] 40 s | `tobii_calibration_collect_data_2d` | 1030 (5 s) | 35 s |
//! | [`CALIBRATION_DISCARD_2D`] 40 s | `tobii_calibration_discard_data_2d` | 1080 (5 s) | 35 s |
//! | [`CALIBRATION_CLEAR`] 40 s | `tobii_calibration_clear` | 1060 (5 s) | 35 s |
//! | [`CALIBRATION_COMPUTE`] 80 s | `tobii_calibration_compute_and_apply` | 1070 (10 s), then 1100 (5 s) to read the result back | 40 + 35 = 75 s |
//! | [`CALIBRATION_RETRIEVE`] 40 s | `tobii_calibration_retrieve` | 1100 (5 s) | 35 s |
//! | [`CALIBRATION_APPLY`] 45 s | `tobii_calibration_apply` | 1110 (10 s) | 40 s |
//!
//! The facts reads are `tobii_get_device_info`, `tobii_get_track_box`,
//! `tobii_get_display_area`, `tobii_get_geometry_mounting`,
//! `tobii_get_device_name` (before a name is set), `tobii_get_state_string`,
//! `tobii_enumerate_stream_types` and `tobii_hardware_configuration_get`.
//! A subscription change and a reconnect make no request: they wait for an
//! acknowledgement the daemon gives without the tracker, 2 s and ~500 ms
//! (see `device`). Each timeout is checked against its worst case at
//! compile time, and that each call passes its own in the tests.

use std::time::Duration;

use tobii_ipc::deadline::worst;

/// How much longer than its request's worst case in the daemon a timeout is
/// at least: for what the table leaves out, which takes milliseconds (a bus
/// scan when no engine runs, the daemon's state lock but while it stops an
/// engine, a file saved, the socket both ways, the daemon's poll steps of 5
/// and 50 ms).
const MARGIN: Duration = Duration::from_secs(2);

/// Device facts are ready once the daemon's engine has initialised the
/// tracker, which a cold start can take most of 10 s to do.
pub(crate) const FACTS: Duration = Duration::from_secs(12);
/// A state the daemon keeps itself.
pub(crate) const STATE: Duration = Duration::from_secs(3);
/// A clock pair: the daemon waits 25 s for a gaze frame, since a cold
/// tracker streams gaze only after its second init.
pub(crate) const TIMESYNC: Duration = Duration::from_secs(27);
/// A pause or resume.
pub(crate) const DEVICE_PAUSE: Duration = Duration::from_secs(60);
/// A display-area write.
pub(crate) const DISPLAY_AREA_SET: Duration = Duration::from_secs(75);
/// A calibration start.
pub(crate) const CALIBRATION_START: Duration = Duration::from_secs(190);
/// A calibration stop.
pub(crate) const CALIBRATION_STOP: Duration = Duration::from_secs(120);
/// Collecting a 2-D point.
pub(crate) const CALIBRATION_COLLECT_2D: Duration = Duration::from_secs(40);
/// Discarding a 2-D point.
pub(crate) const CALIBRATION_DISCARD_2D: Duration = Duration::from_secs(40);
/// Clearing the points collected.
pub(crate) const CALIBRATION_CLEAR: Duration = Duration::from_secs(40);
/// Computing a calibration.
pub(crate) const CALIBRATION_COMPUTE: Duration = Duration::from_secs(80);
/// Reading the active calibration.
pub(crate) const CALIBRATION_RETRIEVE: Duration = Duration::from_secs(40);
/// Writing a calibration.
pub(crate) const CALIBRATION_APPLY: Duration = Duration::from_secs(45);

/// Whether `timeout` outlasts a daemon that takes `worst` by [`MARGIN`].
const fn outlasts(timeout: Duration, worst: Duration) -> bool {
    timeout.as_nanos() >= worst.as_nanos() + MARGIN.as_nanos()
}

// The table, row by row.
const _: () = {
    assert!(outlasts(FACTS, worst::FACTS));
    assert!(outlasts(STATE, Duration::ZERO));
    assert!(outlasts(TIMESYNC, worst::TIMESYNC));
    assert!(outlasts(DEVICE_PAUSE, worst::DEVICE_PAUSE));
    assert!(outlasts(DISPLAY_AREA_SET, worst::DISPLAY_AREA_SET));
    assert!(outlasts(CALIBRATION_START, worst::CALIBRATION_START));
    assert!(outlasts(CALIBRATION_STOP, worst::CALIBRATION_STOP));
    assert!(outlasts(
        CALIBRATION_COLLECT_2D,
        worst::CALIBRATION_COLLECT_2D
    ));
    assert!(outlasts(
        CALIBRATION_DISCARD_2D,
        worst::CALIBRATION_DISCARD_2D
    ));
    assert!(outlasts(CALIBRATION_CLEAR, worst::CALIBRATION_CLEAR));
    assert!(outlasts(CALIBRATION_COMPUTE, worst::CALIBRATION_COMPUTE));
    assert!(outlasts(CALIBRATION_RETRIEVE, worst::CALIBRATION_RETRIEVE));
    assert!(outlasts(CALIBRATION_APPLY, worst::CALIBRATION_APPLY));
};

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::ptr;

    use tobii_ipc::request::status;

    use super::*;
    use crate::api::{
        tobii_device_destroy, tobii_get_device_info, tobii_get_state_bool, tobii_get_state_string,
        tobii_get_state_uint32, tobii_get_track_box,
    };
    use crate::calibration::{
        tobii_calibration_apply, tobii_calibration_clear, tobii_calibration_collect_data_2d,
        tobii_calibration_compute_and_apply, tobii_calibration_discard_data_2d,
        tobii_calibration_retrieve, tobii_calibration_start, tobii_calibration_stop,
    };
    use crate::config::{
        tobii_get_device_name, tobii_get_display_area, tobii_get_geometry_mounting,
        tobii_set_device_name, tobii_set_display_area,
    };
    use crate::device::tests::{device_with, take_request_timeouts};
    use crate::internal::{
        tobii_enumerate_stream_types, tobii_hardware_configuration_get, tobii_pause_device,
        tobii_resume_device, tobii_timesync,
    };
    use crate::types::{
        DeviceInfo, DeviceName, DisplayArea, GeometryMounting, HardwareConfiguration, StateString,
        StreamType, TOBII_ENABLED_EYE_BOTH, TOBII_STATE_CALIBRATION_ACTIVE,
        TOBII_STATE_CALIBRATION_ID, TOBII_STATE_DEVICE_PAUSED, TOBII_STATE_FAULT, TimesyncData,
        TrackBox,
    };

    unsafe extern "C" fn ignore_blob(_: *const c_void, _: usize, _: *mut c_void) {}

    unsafe extern "C" fn ignore_stream_type(_: *const StreamType, _: *mut c_void) {}

    /// Each call that asks the daemon waits for its answer as long as its
    /// row of the table says. The daemon refuses every request, so no answer
    /// needs decoding: the timeout is noted as the request goes out.
    #[test]
    fn each_call_waits_for_the_daemon_as_long_as_its_row_says() {
        let d = Box::into_raw(Box::new(device_with(status::OPERATION_FAILED, vec![])));
        let _ = take_request_timeouts();
        let area = DisplayArea::default();
        let blob = [0u8; 4];
        let mut info = MaybeUninit::<DeviceInfo>::uninit();
        let mut track_box = MaybeUninit::<TrackBox>::uninit();
        let mut shown = MaybeUninit::<DisplayArea>::uninit();
        let mut mounting = MaybeUninit::<GeometryMounting>::uninit();
        let mut name = MaybeUninit::<DeviceName>::uninit();
        let mut text = MaybeUninit::<StateString>::uninit();
        let mut hardware = MaybeUninit::<HardwareConfiguration>::uninit();
        let mut timesync = MaybeUninit::<TimesyncData>::uninit();
        let mut value = 0u32;
        let mut got = Vec::new();
        let mut took = |call: &'static str| got.push((call, take_request_timeouts()));

        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once
        // at the end; every other pointer is to a live local of the type the
        // call reads or writes, and every receiver has the type it takes.
        unsafe {
            tobii_get_device_info(d, info.as_mut_ptr());
            took("tobii_get_device_info");
            tobii_get_track_box(d, track_box.as_mut_ptr());
            took("tobii_get_track_box");
            tobii_get_display_area(d, shown.as_mut_ptr());
            took("tobii_get_display_area");
            tobii_get_geometry_mounting(d, mounting.as_mut_ptr());
            took("tobii_get_geometry_mounting");
            tobii_get_device_name(d, name.as_mut_ptr());
            took("tobii_get_device_name");
            tobii_get_state_string(d, TOBII_STATE_FAULT, text.as_mut_ptr());
            took("tobii_get_state_string");
            tobii_enumerate_stream_types(d, Some(ignore_stream_type), ptr::null_mut());
            took("tobii_enumerate_stream_types");
            tobii_hardware_configuration_get(d, hardware.as_mut_ptr());
            took("tobii_hardware_configuration_get");
            tobii_set_device_name(d, c"Desk".as_ptr());
            took("tobii_set_device_name");
            tobii_get_state_bool(d, TOBII_STATE_DEVICE_PAUSED, &raw mut value);
            tobii_get_state_bool(d, TOBII_STATE_CALIBRATION_ACTIVE, &raw mut value);
            took("tobii_get_state_bool");
            tobii_get_state_uint32(d, TOBII_STATE_CALIBRATION_ID, &raw mut value);
            took("tobii_get_state_uint32");
            tobii_timesync(d, timesync.as_mut_ptr());
            took("tobii_timesync");
            tobii_pause_device(d);
            tobii_resume_device(d);
            took("tobii_pause_device, tobii_resume_device");
            tobii_set_display_area(d, &raw const area);
            took("tobii_set_display_area");
            tobii_calibration_start(d, TOBII_ENABLED_EYE_BOTH);
            took("tobii_calibration_start");
            tobii_calibration_stop(d);
            took("tobii_calibration_stop");
            tobii_calibration_collect_data_2d(d, 0.5, 0.5);
            took("tobii_calibration_collect_data_2d");
            tobii_calibration_discard_data_2d(d, 0.5, 0.5);
            took("tobii_calibration_discard_data_2d");
            tobii_calibration_clear(d);
            took("tobii_calibration_clear");
            tobii_calibration_compute_and_apply(d);
            took("tobii_calibration_compute_and_apply");
            tobii_calibration_retrieve(d, Some(ignore_blob), ptr::null_mut());
            took("tobii_calibration_retrieve");
            tobii_calibration_apply(d, blob.as_ptr().cast(), blob.len());
            took("tobii_calibration_apply");
            assert_eq!(tobii_device_destroy(d), 0);
        }

        assert_eq!(
            got,
            [
                ("tobii_get_device_info", vec![FACTS]),
                ("tobii_get_track_box", vec![FACTS]),
                ("tobii_get_display_area", vec![FACTS]),
                ("tobii_get_geometry_mounting", vec![FACTS]),
                ("tobii_get_device_name", vec![FACTS]),
                ("tobii_get_state_string", vec![FACTS]),
                ("tobii_enumerate_stream_types", vec![FACTS]),
                ("tobii_hardware_configuration_get", vec![FACTS]),
                ("tobii_set_device_name", vec![FACTS]),
                ("tobii_get_state_bool", vec![STATE, STATE]),
                ("tobii_get_state_uint32", vec![STATE]),
                ("tobii_timesync", vec![TIMESYNC]),
                (
                    "tobii_pause_device, tobii_resume_device",
                    vec![DEVICE_PAUSE, DEVICE_PAUSE]
                ),
                ("tobii_set_display_area", vec![DISPLAY_AREA_SET]),
                ("tobii_calibration_start", vec![CALIBRATION_START]),
                ("tobii_calibration_stop", vec![CALIBRATION_STOP]),
                (
                    "tobii_calibration_collect_data_2d",
                    vec![CALIBRATION_COLLECT_2D]
                ),
                (
                    "tobii_calibration_discard_data_2d",
                    vec![CALIBRATION_DISCARD_2D]
                ),
                ("tobii_calibration_clear", vec![CALIBRATION_CLEAR]),
                (
                    "tobii_calibration_compute_and_apply",
                    vec![CALIBRATION_COMPUTE]
                ),
                ("tobii_calibration_retrieve", vec![CALIBRATION_RETRIEVE]),
                ("tobii_calibration_apply", vec![CALIBRATION_APPLY]),
            ]
        );
    }
}
