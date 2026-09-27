//! How long each call that asks tobiid waits for its answer: the shared
//! table in [`tobii_ipc::timeout`], which outlasts what the daemon may take
//! over each request, so that the call reports what the daemon did. One
//! that gave up first would return `TOBII_ERROR_TIMED_OUT` while the daemon
//! carried the request out (a calibration started, a display area set), and
//! the device's next request would wait behind it, since the daemon runs a
//! connection's requests one at a time. `TOBII_ERROR_TIMED_OUT` from a
//! request is then the daemon's own answer, or a daemon held up past its
//! deadlines by what they leave out, which may still carry the request out.
//! What the daemon waits for over each request, and its worst case, are in
//! [`tobii_ipc::timeout`]'s table; here, which calls wait which timeout:
//!
//! | Timeout | Calls |
//! |---|---|
//! | [`FACTS`] 12 s | the facts reads (below), `tobii_set_device_name` |
//! | [`STATE`] 3 s | `tobii_get_state_bool`, `tobii_get_state_uint32` |
//! | [`TIMESYNC`] 27 s | `tobii_timesync` |
//! | [`DEVICE_PAUSE`] 60 s | `tobii_pause_device`, `tobii_resume_device` |
//! | [`DISPLAY_AREA_SET`] 75 s | `tobii_set_display_area` |
//! | [`CALIBRATION_START`] 190 s | `tobii_calibration_start` |
//! | [`CALIBRATION_STOP`] 120 s | `tobii_calibration_stop` |
//! | [`CALIBRATION_COLLECT_2D`] 40 s | `tobii_calibration_collect_data_2d` |
//! | [`CALIBRATION_DISCARD_2D`] 40 s | `tobii_calibration_discard_data_2d` |
//! | [`CALIBRATION_CLEAR`] 40 s | `tobii_calibration_clear` |
//! | [`CALIBRATION_COMPUTE`] 80 s | `tobii_calibration_compute_and_apply` |
//! | [`CALIBRATION_RETRIEVE`] 40 s | `tobii_calibration_retrieve` |
//! | [`CALIBRATION_APPLY`] 45 s | `tobii_calibration_apply` |
//!
//! The facts reads are `tobii_get_device_info`, `tobii_get_track_box`,
//! `tobii_get_display_area`, `tobii_get_geometry_mounting`,
//! `tobii_get_device_name` (before a name is set), `tobii_get_state_string`,
//! `tobii_enumerate_stream_types` and `tobii_hardware_configuration_get`.
//! A subscription change and a reconnect make no request: they wait for an
//! acknowledgement the daemon gives without the tracker, 2 s and ~500 ms
//! (see `device`). Each timeout is checked against its worst case at
//! compile time (in `tobii-ipc`), and that each call passes its own in the
//! tests.

pub(crate) use tobii_ipc::timeout::{
    CALIBRATION_APPLY, CALIBRATION_CLEAR, CALIBRATION_COLLECT_2D, CALIBRATION_COMPUTE,
    CALIBRATION_DISCARD_2D, CALIBRATION_RETRIEVE, CALIBRATION_START, CALIBRATION_STOP,
    DEVICE_PAUSE, DISPLAY_AREA_SET, FACTS, STATE, TIMESYNC,
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
