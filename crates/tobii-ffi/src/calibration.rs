//! `tobii_config.h` calibration: the session runs in the daemon (one owner at
//! a time; the result is saved as the user's calibration only when the owner
//! stops the session); parsing a blob is done here.
//!
//! 2-D calibration of both eyes is what the Windows engine was captured doing;
//! discarding a 2-D point uses the command the DLL sends for it. 3-D and
//! per-eye variants were never observed and return
//! `TOBII_ERROR_NOT_SUPPORTED`. So do the undocumented stimulus points, once
//! their arguments check out, as the DLL's in-process tracker module answers
//! for an ET5.

use std::ffi::c_void;
use std::time::Duration;

use tobii_ipc::request::{STOP_KEEP, encode_point_2d, kind};

use crate::device::{Api, Device, call, device_ref, in_callback};
use crate::status::{
    Status, TOBII_ERROR_CALLBACK_IN_PROGRESS, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
    TOBII_ERROR_NOT_SUPPORTED, TOBII_ERROR_OPERATION_FAILED,
};
use crate::stub::not_supported;
use crate::timeouts;
use crate::types::{
    CalibrationPointData, CalibrationPointReceiver, CalibrationStimulusPoints, DataReceiver,
    TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
    TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
    TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
};

/// Run a request on `device`, discarding the reply payload.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that is
/// not destroyed before the call returns.
pub(crate) unsafe fn request(
    device: *mut Device,
    request: u8,
    payload: &[u8],
    timeout: Duration,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    match unsafe { device_ref(device) } {
        Ok(d) => match d.request(request, payload, timeout) {
            Ok(_) => TOBII_ERROR_NO_ERROR,
            Err(status) => status,
        },
        Err(status) => status,
    }
}

/// Start a calibration session. Another client's session makes this
/// `TOBII_ERROR_CALIBRATION_BUSY`; only `TOBII_ENABLED_EYE_BOTH` is supported.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that is
/// not destroyed before the call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_start(device: *mut Device, enabled_eye: u32) -> Status {
    let Ok(eye) = u8::try_from(enabled_eye) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_START,
            &[eye],
            timeouts::CALIBRATION_START,
        )
    }
}

/// End the session, keeping the calibration it computed last (and the
/// display area set during it). Only if nothing was computed, or the daemon
/// could not save it (`TOBII_ERROR_OPERATION_FAILED`), are the previous
/// calibration and display area restored. Once saved, both are kept even if
/// the tracker then refuses the calibration or goes away before taking it
/// (`TOBII_ERROR_OPERATION_FAILED` or `TOBII_ERROR_CONNECTION_FAILED` all
/// the same): it loads them at its next init. A session the daemon already
/// ended, because the tracker re-initialised or went away, saved nothing:
/// its stop is `TOBII_ERROR_CALIBRATION_NOT_STARTED`.
///
/// # Safety
/// As `tobii_calibration_start`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_stop(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_STOP,
            STOP_KEEP,
            timeouts::CALIBRATION_STOP,
        )
    }
}

/// Collect the user's gaze at `(x, y)` (normalised display coordinates)
/// while they look at a stimulus there. Blocks for most of a second.
///
/// # Safety
/// As `tobii_calibration_start`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_collect_data_2d(
    device: *mut Device,
    x: f32,
    y: f32,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_COLLECT_2D,
            &encode_point_2d(x, y),
            timeouts::CALIBRATION_COLLECT_2D,
        )
    }
}

/// Discard the data collected at `(x, y)` in this session: the point as it
/// was given to `tobii_calibration_collect_data_2d`.
///
/// # Safety
/// As `tobii_calibration_start`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_discard_data_2d(
    device: *mut Device,
    x: f32,
    y: f32,
) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_DISCARD_2D,
            &encode_point_2d(x, y),
            timeouts::CALIBRATION_DISCARD_2D,
        )
    }
}

/// Clear the points collected so far in this session.
///
/// # Safety
/// As `tobii_calibration_start`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_clear(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_CLEAR,
            &[],
            timeouts::CALIBRATION_CLEAR,
        )
    }
}

/// Compute a calibration from the collected points and make it active. The
/// daemon saves the last one computed when the session is stopped with
/// `tobii_calibration_stop`, as the user's calibration from then on.
///
/// # Safety
/// As `tobii_calibration_start`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_compute_and_apply(device: *mut Device) -> Status {
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_COMPUTE,
            &[],
            timeouts::CALIBRATION_COMPUTE,
        )
    }
}

/// Read the active calibration and hand it to `receiver`, on this thread.
///
/// The receiver runs as a callback does, under the callback guard (see
/// [`crate::device::call`]): a call from inside it that a stream callback
/// could not make either is `TOBII_ERROR_CALLBACK_IN_PROGRESS`, a destroy
/// included, while `tobii_system_clock` goes through. So it is in the DLL,
/// which sets its callback flag around the command that calls the receiver
/// (0x180147bd9..0x180147c66), and the 4.1 documentation names retrieve
/// among the calls whose callbacks may not call in. Unlike a stream
/// callback, and unlike the DLL's receiver, which runs under the device's
/// API mutex (dev+0x4e0, 0x180147bbc..0x180147c79), it runs with none of the
/// device's locks held: other threads' calls into the device go on
/// meanwhile, and it may wait for them.
///
/// # Safety
/// As `tobii_calibration_start`; `receiver` must be null or sound to call,
/// on the calling thread, with a data pointer, its size and `user_data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_retrieve(
    device: *mut Device,
    receiver: Option<DataReceiver>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    match d.request(
        kind::CALIBRATION_RETRIEVE,
        &[],
        timeouts::CALIBRATION_RETRIEVE,
    ) {
        Ok(blob) => {
            // The request has let the command lock go, so no lock is held.
            // `call` leaves the guard as it found it, down (a call from
            // inside a callback was refused above), where the DLL clears its
            // flag outright (0x180147c66).
            // SAFETY: the caller guarantees `receiver` is sound to call like
            // this; `blob` outlives the call. The guard refuses a destroy of
            // the device from inside it, and `d` is not used after it.
            call(|| unsafe { receiver(blob.as_ptr().cast(), blob.len(), user_data) });
            TOBII_ERROR_NO_ERROR
        }
        Err(status) => status,
    }
}

/// Make `data` the active calibration (and the user's saved one).
///
/// # Safety
/// As `tobii_calibration_start`; `data` must be null or point to `size`
/// readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_apply(
    device: *mut Device,
    data: *const c_void,
    size: usize,
) -> Status {
    if data.is_null() || size == 0 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees `size` readable bytes.
    let blob = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size) };
    // SAFETY: forwarded under the same contract.
    unsafe {
        request(
            device,
            kind::CALIBRATION_APPLY,
            blob,
            timeouts::CALIBRATION_APPLY,
        )
    }
}

/// A record's status word as a `tobii_calibration_point_status_t`, mapped as
/// the DLL maps it (0x180147910): only the low 32 bits count; 1 was used in
/// the calibration, 0 is valid but was not used, and anything else (the DLL
/// tests for -1 explicitly) failed or is invalid. The mask only mirrors the
/// DLL: `tobii_calib::blob::points` passes an upper half only as the sign of
/// a -1.
fn point_status(word: u64) -> u32 {
    match word & 0xffff_ffff {
        1 => TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
        0 => TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
        _ => TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
    }
}

/// Hand each calibration point stored in `data` to `receiver`. The checks run
/// in the DLL's order: a null `api` or `data`, a `data_size` under 8 or a
/// null `receiver` is `TOBII_ERROR_INVALID_PARAMETER`; then a call from inside
/// a callback is `TOBII_ERROR_CALLBACK_IN_PROGRESS`; then data that
/// `tobii_calib::blob::points` refuses is not a valid calibration,
/// `TOBII_ERROR_OPERATION_FAILED` with no point handed out. The DLL returns
/// that only for a negative point count (0x18014783f, 13 at 0x1801478a7) and
/// reads the rest as given. Here a status word whose low 32 bits are outside
/// -1..=2 is refused where the DLL reports that eye as `FAILED_OR_INVALID`,
/// and one with a non-zero upper half, unless it is a -1's sign, where the
/// DLL ignores that half and maps the low word as usual; so are a target, or
/// the measurement of an eye not marked failed (-1), that is not finite or
/// lies outside the display by more than half its size, and a blob shorter
/// than a header or with bytes past its point list, where the DLL may
/// succeed. The first measurement of each record is reported as the left
/// eye, and each eye's status word is mapped as the DLL maps it; a failed
/// eye's mapping is passed on as the blob holds it.
///
/// # Safety
/// `api` must be null or a live handle; `data` must be null or point to
/// `data_size` readable bytes; `receiver` must be null or sound to call with a
/// point and `user_data`, and must not re-enter this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_parse(
    api: *mut Api,
    data: *const c_void,
    data_size: usize,
    receiver: Option<CalibrationPointReceiver>,
    user_data: *mut c_void,
) -> Status {
    if api.is_null() || data.is_null() || data_size < 8 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    // SAFETY: non-null, and the caller guarantees `data_size` readable bytes.
    let blob = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), data_size) };
    let points = match tobii_calib::blob::points(blob) {
        Ok(points) => points,
        Err(e) => {
            tracing::debug!(error = %e, "tobii_calibration_parse: not a valid calibration");
            return TOBII_ERROR_OPERATION_FAILED;
        }
    };
    for p in points {
        let c = CalibrationPointData {
            point_xy: p.target,
            left_status: point_status(p.a_status),
            left_mapping_xy: p.a,
            right_status: point_status(p.b_status),
            right_mapping_xy: p.b,
        };
        // SAFETY: the caller guarantees `receiver` is sound to call like this;
        // `c` outlives the call.
        unsafe { receiver(&raw const c, user_data) };
    }
    TOBII_ERROR_NO_ERROR
}

/// The calibration's stimulus points (undocumented): always
/// `TOBII_ERROR_NOT_SUPPORTED` once the arguments check out, with nothing
/// written. That is the answer of the DLL's in-process tracker module
/// (legacy TTP) for an ET5; behind Tobii's service (`tobii-prp://`) the
/// answer was never captured.
///
/// The DLL (0x180149d10) reads PRP property 0x13, which its id-to-name
/// switch calls `CALIBRATION_STIMULUS_POINTS` (0x18002e8b0, table
/// 0x18002ea00, case 19 at 0x18002e9b8), through its `tobii_property_get`
/// (0x18015d300). That answers `TOBII_ERROR_NOT_SUPPORTED` without a request
/// when the device's property list lacks the id (0x18015d450), and on the
/// DLL's own path the list is what the legacy TTP module reported at
/// connect, which never holds 0x13 (0x18016b536..0x18016b8d2). No capture
/// has a tracker command for the points either, so the answer is given here,
/// without asking the daemon. A call from inside a callback is
/// `TOBII_ERROR_CALLBACK_IN_PROGRESS`, then a null device or `points` is
/// `TOBII_ERROR_INVALID_PARAMETER`, in that order: the DLL checks the two
/// nulls before the callback. The device is only compared with null, never
/// borrowed, and `points` is never written, so the function is safe.
#[unsafe(no_mangle)]
pub extern "C" fn tobii_calibration_stimulus_points_get(
    device: *mut Device,
    points: *mut CalibrationStimulusPoints,
) -> Status {
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    if device.is_null() || points.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    tracing::trace!("tobii_calibration_stimulus_points_get: not supported");
    TOBII_ERROR_NOT_SUPPORTED
}

not_supported! {
    /// 3-D calibration was never captured.
    fn tobii_calibration_collect_data_3d(device: *mut c_void, x: f32, y: f32, z: f32);
    /// Per-eye calibration was never captured.
    fn tobii_calibration_collect_data_per_eye_2d(device: *mut c_void, x: f32, y: f32, requested_eyes: u32, collected_eyes: *mut c_void);
    /// 3-D calibration was never captured.
    fn tobii_calibration_discard_data_3d(device: *mut c_void, x: f32, y: f32, z: f32);
    /// Per-eye calibration was never captured.
    fn tobii_calibration_discard_data_per_eye_2d(device: *mut c_void, x: f32, y: f32, eyes: u32);
    /// Per-eye calibration was never captured.
    fn tobii_calibration_compute_and_apply_per_eye(device: *mut c_void, calibrated_eyes: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{
        tobii_api_create, tobii_device_create, tobii_device_destroy,
        tobii_device_process_callbacks, tobii_get_state_bool, tobii_system_clock,
    };
    use crate::device::tests::{Gate, PROMPT, device_with};
    use crate::logger::tests::Recorder;
    use crate::types::{
        CalibrationStimulusPoint, TOBII_LOG_LEVEL_ERROR, TOBII_STATE_CALIBRATION_ACTIVE,
    };
    use std::ptr;
    use std::thread;
    use std::time::Instant;

    unsafe extern "C" fn collect(p: *const CalibrationPointData, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Vec<CalibrationPointData>` and the
        // library a valid point.
        unsafe { (*ud.cast::<Vec<CalibrationPointData>>()).push(*p) };
    }

    fn embedded_blob() -> Vec<u8> {
        use tobii_proto::protocol::{parse_init_packets, parse_message};
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tobii-usb/init_packets_ep.txt"
        ));
        let packets = parse_init_packets(text).expect("init file");
        let body: Vec<u8> = packets[38..200]
            .iter()
            .flat_map(|p| p.data[8..].iter().copied())
            .collect();
        let prefixed = [&[0u8; 8][..], &body].concat();
        let payload = parse_message(&prefixed).expect("header").payload;
        tobii_proto::calibration::blob_from_payload(payload)
            .expect("blob")
            .to_vec()
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: compares rounded values
    fn parses_the_embedded_calibration() {
        let blob = embedded_blob();
        let mut api: *mut Api = ptr::null_mut();
        let mut points: Vec<CalibrationPointData> = Vec::new();
        // SAFETY: live locals; `collect` matches the receiver contract.
        unsafe {
            assert_eq!(
                crate::api::tobii_api_create(&raw mut api, ptr::null(), ptr::null()),
                0
            );
            let status = tobii_calibration_parse(
                api,
                blob.as_ptr().cast(),
                blob.len(),
                Some(collect),
                (&raw mut points).cast(),
            );
            assert_eq!(status, 0);
            assert_eq!(
                tobii_calibration_parse(
                    api,
                    blob.as_ptr().cast(),
                    4,
                    Some(collect),
                    ptr::null_mut()
                ),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
        }
        assert_eq!(points.len(), 14);
        assert_eq!(points[1].point_xy.map(|v| (v * 10.0).round()), [5.0, 1.0]);
        assert_eq!(
            points[1].left_status,
            TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION
        );
    }

    #[test]
    fn point_status_maps_status_words_as_the_dll_does() {
        // The numbers the DLL writes (0x180147932, 0x180147928, 0x18014791e).
        assert_eq!(TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID, 0);
        assert_eq!(
            TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
            1
        );
        assert_eq!(
            TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
            2
        );
        for (word, status) in [
            (
                1,
                TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
            ),
            (
                0,
                TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
            ),
            // -1 as the DLL reads it, a 32-bit int, and a 64-bit -1.
            (
                0xffff_ffff,
                TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
            ),
            (u64::MAX, TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID),
            (2, TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID),
            // The upper half is never read.
            (
                0x1_0000_0001,
                TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
            ),
            (
                0x1_0000_0000,
                TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
            ),
        ] {
            assert_eq!(point_status(word), status, "{word:#x}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: the points are copied, not computed
    fn parse_reports_each_eyes_own_status() {
        let mut blob = embedded_blob();
        let records = tobii_calib::blob::points(&blob).expect("points");
        let total =
            usize::try_from(tobii_calib::blob::header(&blob).expect("header").total).expect("fits");
        let first = total + 8;
        // The first measurement's word is at +16, the second's at +32. The
        // blob check lets 2 through, which the DLL reports as failed.
        for (i, offset, word) in [(2, 16, 0u64), (2, 32, 2), (4, 16, 2), (4, 32, 0)] {
            let o = first + tobii_calib::blob::RECORD_LEN * i + offset;
            blob[o..o + 8].copy_from_slice(&word.to_le_bytes());
        }
        let (status, points) = parse(&blob);
        assert_eq!(status, TOBII_ERROR_NO_ERROR);
        assert_eq!(points.len(), 14);
        for (i, (p, r)) in points.iter().zip(&records).enumerate() {
            let statuses = match i {
                2 => (
                    TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
                    TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
                ),
                4 => (
                    TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
                    TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
                ),
                _ => (
                    TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
                    TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
                ),
            };
            assert_eq!((p.left_status, p.right_status), statuses, "point {i}");
            // The first measurement is the left eye, as in the DLL.
            assert_eq!(p.point_xy, r.target, "point {i}");
            assert_eq!(p.left_mapping_xy, r.a, "point {i}");
            assert_eq!(p.right_mapping_xy, r.b, "point {i}");
        }
    }

    #[test]
    fn parse_reports_a_failed_eye_and_passes_its_mapping_on() {
        let mut blob = embedded_blob();
        let total =
            usize::try_from(tobii_calib::blob::header(&blob).expect("header").total).expect("fits");
        let first = total + 8;
        // -1 in either encoding the tracker may write: a 32-bit int (record
        // 3's first eye) and a 64-bit one (record 6's second); the failed
        // eye's mapping is not a number, or off the display.
        let failed_32 = 0xffff_ffffu64.to_le_bytes();
        let failed_64 = u64::MAX.to_le_bytes();
        let nan = f32::NAN.to_le_bytes();
        let off = (-7.5f32).to_le_bytes();
        for (i, offset, bytes) in [
            (3, 16, &failed_32[..]),
            (3, 8, &nan[..]),
            (6, 32, &failed_64[..]),
            (6, 28, &off[..]),
        ] {
            let o = first + tobii_calib::blob::RECORD_LEN * i + offset;
            blob[o..o + bytes.len()].copy_from_slice(bytes);
        }
        let (status, points) = parse(&blob);
        assert_eq!(status, TOBII_ERROR_NO_ERROR);
        assert_eq!(points.len(), 14);
        let used = TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION;
        let failed = TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID;
        for (i, p) in points.iter().enumerate() {
            let statuses = match i {
                3 => (failed, used),
                6 => (used, failed),
                _ => (used, used),
            };
            assert_eq!((p.left_status, p.right_status), statuses, "point {i}");
        }
        assert!(points[3].left_mapping_xy[0].is_nan());
        assert_eq!(points[6].right_mapping_xy[1].to_bits(), (-7.5f32).to_bits());
    }

    /// The DLL's own invalid calibration (0x18014783f, 13 at 0x1801478a7) in
    /// the fewest bytes the argument checks pass: a point list at offset 0
    /// whose count, the next word, is negative. Here it is shorter than any
    /// header, with the same answer.
    static NEGATIVE_COUNT: [u8; 8] = [0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff];

    /// Parse `data` with a fresh API, as a client would: the status and the
    /// points the receiver was handed.
    fn parse(data: &[u8]) -> (Status, Vec<CalibrationPointData>) {
        let mut api: *mut Api = ptr::null_mut();
        let mut points: Vec<CalibrationPointData> = Vec::new();
        // SAFETY: live locals; `data` points to `data.len()` readable bytes
        // and `collect` matches the receiver contract.
        let status = unsafe {
            assert_eq!(
                crate::api::tobii_api_create(&raw mut api, ptr::null(), ptr::null()),
                0
            );
            let status = tobii_calibration_parse(
                api,
                data.as_ptr().cast(),
                data.len(),
                Some(collect),
                (&raw mut points).cast(),
            );
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
            status
        };
        (status, points)
    }

    #[test]
    fn parse_checks_its_arguments_before_the_data() {
        let data = NEGATIVE_COUNT.as_ptr().cast();
        let mut api: *mut Api = ptr::null_mut();
        let mut points: Vec<CalibrationPointData> = Vec::new();
        // SAFETY: live locals; `data` points to 8 readable bytes, null
        // handles are allowed and `collect` matches the receiver contract.
        let got = unsafe {
            assert_eq!(
                crate::api::tobii_api_create(&raw mut api, ptr::null(), ptr::null()),
                0
            );
            let ud = (&raw mut points).cast();
            let got = [
                tobii_calibration_parse(ptr::null_mut(), data, 8, Some(collect), ud),
                tobii_calibration_parse(api, ptr::null(), 8, Some(collect), ud),
                tobii_calibration_parse(api, data, 7, Some(collect), ud),
                tobii_calibration_parse(api, data, 0, Some(collect), ud),
                tobii_calibration_parse(api, data, 8, None, ud),
                tobii_calibration_parse(api, data, 8, Some(collect), ud),
            ];
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
            got
        };
        let invalid = TOBII_ERROR_INVALID_PARAMETER;
        assert_eq!(
            got,
            [
                invalid,
                invalid,
                invalid,
                invalid,
                invalid,
                TOBII_ERROR_OPERATION_FAILED
            ]
        );
        assert!(points.is_empty());
    }

    #[test]
    fn parse_refuses_data_that_is_not_a_calibration() {
        let blob = embedded_blob();
        let total =
            usize::try_from(tobii_calib::blob::header(&blob).expect("header").total).expect("fits");
        let mut negative = blob.clone();
        negative[total + 4..total + 8].copy_from_slice(&(-1i32).to_le_bytes());
        // Record 5's target x: the five records before it are sound, and
        // still none is handed out.
        let mut not_finite = blob.clone();
        let o = total + 8 + tobii_calib::blob::RECORD_LEN * 5;
        not_finite[o..o + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        // Record 5's first status word: 3 means nothing to the DLL.
        let mut status_3 = blob.clone();
        status_3[o + 16..o + 24].copy_from_slice(&3u64.to_le_bytes());
        let long = [&blob[..], &[0]].concat();
        for (name, data) in [
            ("a negative count in 8 bytes", &NEGATIVE_COUNT[..]),
            ("a negative count", &negative[..]),
            ("a list past the end", &blob[..blob.len() - 1]),
            ("bytes after the list", &long[..]),
            ("a NaN target", &not_finite[..]),
            ("a status word of 3", &status_3[..]),
        ] {
            let (status, points) = parse(data);
            assert_eq!(status, TOBII_ERROR_OPERATION_FAILED, "{name}");
            assert!(points.is_empty(), "{name}");
        }
        assert_eq!(parse(&blob).0, TOBII_ERROR_NO_ERROR);
    }

    #[test]
    fn a_discard_sends_its_point_to_the_daemon() {
        use std::sync::{Arc, Mutex};
        use tobii_ipc::encode_reply;
        use tobii_ipc::request::{decode_point_2d, decode_request};
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let connect = crate::device::tests::fake_daemon(move |body| {
            let req = decode_request(body).expect("request");
            log.lock()
                .expect("log")
                .push((req.kind, decode_point_2d(req.payload)));
            vec![encode_reply(req.id, 0, &[])]
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        // SAFETY: `d` is live and destroyed once; a null device is allowed.
        unsafe {
            assert_eq!(tobii_calibration_discard_data_2d(d, 0.25, 0.75), 0);
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
            assert_eq!(
                tobii_calibration_discard_data_2d(ptr::null_mut(), 0.25, 0.75),
                TOBII_ERROR_INVALID_PARAMETER
            );
        }
        assert_eq!(
            *seen.lock().expect("log"),
            vec![(kind::CALIBRATION_DISCARD_2D, Some((0.25, 0.75)))]
        );
    }

    #[test]
    fn requests_carry_the_daemon_status() {
        // The daemon says another client is calibrating.
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(15, vec![])));
        // SAFETY: `d` is live and destroyed once.
        unsafe {
            assert_eq!(tobii_calibration_start(d, 2), 15);
            assert_eq!(tobii_calibration_discard_data_2d(d, 0.5, 0.5), 15);
            assert_eq!(
                tobii_calibration_apply(d, ptr::null(), 0),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
    }

    unsafe extern "C" fn keep_blob(data: *const c_void, size: usize, ud: *mut c_void) {
        // SAFETY: the tests pass a live `Vec<u8>` as `ud`, and the library
        // `size` readable bytes at `data`.
        unsafe { *ud.cast::<Vec<u8>>() = std::slice::from_raw_parts(data.cast(), size).to_vec() };
    }

    /// A retrieve receiver's user data: the handles it calls into, and what
    /// it saw from inside.
    struct Inside {
        api: *mut Api,
        device: *mut Device,
        /// Another device: the guard refuses a call into any.
        other: *mut Device,
        blob: Vec<u8>,
        guarded: bool,
        got: Vec<Status>,
    }

    impl Inside {
        fn new(api: *mut Api, device: *mut Device, other: *mut Device) -> Self {
            Self {
                api,
                device,
                other,
                blob: Vec::new(),
                guarded: false,
                got: Vec::new(),
            }
        }
    }

    unsafe extern "C" fn call_back_in(data: *const c_void, size: usize, ud: *mut c_void) {
        // SAFETY: the test passes a live `Inside` as `ud`, whose handles are
        // live.
        let inside = unsafe { &mut *ud.cast::<Inside>() };
        // SAFETY: the library passes `size` readable bytes at `data`.
        unsafe { keep_blob(data, size, (&raw mut inside.blob).cast()) };
        inside.guarded = in_callback();
        let (mut blob, mut active, mut now) = (Vec::<u8>::new(), 7u32, 0i64);
        let mut points: Vec<CalibrationPointData> = Vec::new();
        // SAFETY: live handles and locals; `NEGATIVE_COUNT` is 8 readable
        // bytes, and the receivers match their contracts.
        inside.got = unsafe {
            vec![
                tobii_calibration_retrieve(inside.device, Some(keep_blob), (&raw mut blob).cast()),
                tobii_calibration_clear(inside.device),
                tobii_device_process_callbacks(inside.device),
                tobii_get_state_bool(
                    inside.other,
                    TOBII_STATE_CALIBRATION_ACTIVE,
                    &raw mut active,
                ),
                tobii_calibration_parse(
                    inside.api,
                    NEGATIVE_COUNT.as_ptr().cast(),
                    NEGATIVE_COUNT.len(),
                    Some(collect),
                    (&raw mut points).cast(),
                ),
                tobii_device_destroy(inside.other),
                tobii_system_clock(inside.api, &raw mut now),
            ]
        };
    }

    /// The receiver runs as a callback does, as in the DLL
    /// (0x180147bd9..0x180147c66): each call from inside it that a callback
    /// could not make either is refused, into its own device or another, a
    /// nested retrieve and a destroy included, while `tobii_system_clock`
    /// goes through; once retrieve has returned, the guard is down again.
    /// Without the guard every call goes through, the destroy too, and the
    /// test fails before it touches that device again.
    #[test]
    fn a_retrieve_receiver_is_refused_what_a_callback_is() {
        let blob = embedded_blob();
        let d = Box::into_raw(Box::new(device_with(0, blob.clone())));
        let other = Box::into_raw(Box::new(device_with(0, vec![1])));
        let mut api: *mut Api = ptr::null_mut();
        // SAFETY: a live local; a null allocator and logger are allowed.
        let created = unsafe { tobii_api_create(&raw mut api, ptr::null(), ptr::null()) };
        assert_eq!(created, TOBII_ERROR_NO_ERROR);
        let mut inside = Inside::new(api, d, other);

        // SAFETY: `d` is live, and `call_back_in` matches the receiver
        // contract for the live `inside`.
        let status =
            unsafe { tobii_calibration_retrieve(d, Some(call_back_in), (&raw mut inside).cast()) };

        assert_eq!(status, TOBII_ERROR_NO_ERROR);
        assert!(inside.guarded, "the receiver runs under the guard");
        let refused = TOBII_ERROR_CALLBACK_IN_PROGRESS;
        assert_eq!(
            inside.got,
            [
                refused,
                refused,
                refused,
                refused,
                refused,
                refused,
                TOBII_ERROR_NO_ERROR
            ]
        );
        assert_eq!(inside.blob, blob);
        assert!(!in_callback(), "the guard is down again");
        let mut again = Vec::<u8>::new();
        // SAFETY: live handles, each destroyed once; `keep_blob` matches the
        // receiver contract for the live `again`.
        unsafe {
            let retrieved = tobii_calibration_retrieve(d, Some(keep_blob), (&raw mut again).cast());
            assert_eq!(retrieved, TOBII_ERROR_NO_ERROR);
            assert_eq!(tobii_device_destroy(other), 0);
            assert_eq!(tobii_device_destroy(d), 0);
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
        }
        assert_eq!(again, blob);
    }

    unsafe extern "C" fn log_then_call_back_in(
        _data: *const c_void,
        _size: usize,
        ud: *mut c_void,
    ) {
        // SAFETY: the test passes a live `Inside` as `ud`, whose handles are
        // live.
        let inside = unsafe { &mut *ud.cast::<Inside>() };
        let mut device: *mut Device = ptr::null_mut();
        // SAFETY: a live handle and local.
        let created = unsafe { tobii_device_create(inside.api, ptr::null(), 99, &raw mut device) };
        inside.guarded = in_callback();
        // SAFETY: a live handle.
        inside.got = vec![created, unsafe { tobii_calibration_clear(inside.device) }];
    }

    /// A line logged from inside the receiver (a refused `field_of_use`,
    /// which the create checks ahead of the guard, as in the DLL) runs under
    /// the guard too, and leaves it up: the receiver's next call is still
    /// refused, and the guard is down once retrieve returns.
    #[test]
    fn a_line_logged_inside_a_retrieve_receiver_leaves_the_guard_up() {
        let recorder = Recorder::default();
        let log = recorder.custom_log();
        let d = Box::into_raw(Box::new(device_with(0, vec![1])));
        let mut api: *mut Api = ptr::null_mut();
        // SAFETY: `api` and `log` are live locals.
        let created = unsafe { tobii_api_create(&raw mut api, ptr::null(), &raw const log) };
        assert_eq!(created, TOBII_ERROR_NO_ERROR);
        let mut inside = Inside::new(api, d, ptr::null_mut());

        // SAFETY: `d` is live, and `log_then_call_back_in` matches the
        // receiver contract for the live `inside`.
        let status = unsafe {
            tobii_calibration_retrieve(d, Some(log_then_call_back_in), (&raw mut inside).cast())
        };

        assert_eq!(status, TOBII_ERROR_NO_ERROR);
        assert_eq!(
            inside.got,
            [
                TOBII_ERROR_INVALID_PARAMETER,
                TOBII_ERROR_CALLBACK_IN_PROGRESS
            ]
        );
        assert!(inside.guarded, "the guard outlives the line");
        assert!(!in_callback(), "and is down once retrieve returns");
        let lines = recorder.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0].0, TOBII_LOG_LEVEL_ERROR);
        assert!(
            lines[0].1.starts_with("refused field_of_use 99"),
            "{lines:?}"
        );
        // SAFETY: live handles, each destroyed once.
        unsafe {
            assert_eq!(tobii_device_destroy(d), 0);
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
        }
    }

    unsafe extern "C" fn held_blob(_data: *const c_void, _size: usize, ud: *mut c_void) {
        // SAFETY: the test passes a live `Gate` as `ud`.
        unsafe { &*ud.cast::<Gate>() }.hold();
    }

    /// A guard against locking too much, which passes with the receiver
    /// unguarded too: it runs with none of the device's locks held, and the
    /// guard is its own thread's, so another thread's request on the device
    /// goes through while it runs, and the receiver may wait for it, as this
    /// one does. It fails should the receiver run under the command lock, as
    /// the DLL's runs under its API mutex (dev+0x4e0,
    /// 0x180147bbc..0x180147c79): the request then waits out the receiver's
    /// hold. It fails too should the guard be one flag for the whole
    /// process: the request is refused.
    #[test]
    fn another_threads_request_goes_through_while_a_retrieve_receiver_runs() {
        let gate = Gate::default();
        let device = device_with(0, vec![1]);
        let handle = || ptr::from_ref(&device).cast_mut();

        let ((ours, took), retrieved) = thread::scope(|s| {
            // SAFETY: `device` outlives the scope, and the call borrows it
            // shared only; `held_blob` matches the receiver contract for the
            // live `gate`.
            let retrieving = s.spawn(|| unsafe {
                tobii_calibration_retrieve(handle(), Some(held_blob), gate.ud())
            });
            assert!(gate.entered(1), "the receiver runs on the other thread");
            let started = Instant::now();
            let mut active = 7u32;
            // SAFETY: as above; `active` is a live local.
            let status = unsafe {
                tobii_get_state_bool(handle(), TOBII_STATE_CALIBRATION_ACTIVE, &raw mut active)
            };
            let took = started.elapsed();
            gate.open();
            (
                ((status, active), took),
                retrieving.join().expect("retrieve"),
            )
        });

        assert_eq!(ours, (TOBII_ERROR_NO_ERROR, 1));
        assert!(took < PROMPT, "{took:?}");
        assert_eq!(retrieved, TOBII_ERROR_NO_ERROR);
    }

    /// Stimulus points no call may write.
    fn stimulus_sentinel() -> CalibrationStimulusPoints {
        CalibrationStimulusPoints {
            point_count: -7,
            points: [CalibrationStimulusPoint {
                words: [0xdead_beef; 9],
            }; 32],
        }
    }

    #[test]
    fn stimulus_points_are_not_supported_without_asking_the_daemon() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tobii_ipc::encode_reply;
        use tobii_ipc::request::decode_request;
        let requests = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&requests);
        let connect = crate::device::tests::fake_daemon(move |body| {
            if body.first() != Some(&tobii_ipc::TAG_REQUEST) {
                return vec![];
            }
            seen.fetch_add(1, Ordering::Relaxed);
            let req = decode_request(body).expect("request");
            vec![encode_reply(req.id, 0, &[])]
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        let mut out = stimulus_sentinel();
        let got = [
            tobii_calibration_stimulus_points_get(d, &raw mut out),
            tobii_calibration_stimulus_points_get(d, ptr::null_mut()),
            tobii_calibration_stimulus_points_get(ptr::null_mut(), &raw mut out),
            tobii_calibration_stimulus_points_get(ptr::null_mut(), ptr::null_mut()),
        ];
        let invalid = TOBII_ERROR_INVALID_PARAMETER;
        assert_eq!(got, [TOBII_ERROR_NOT_SUPPORTED, invalid, invalid, invalid]);
        assert_eq!(out, stimulus_sentinel(), "nothing written");
        assert_eq!(requests.load(Ordering::Relaxed), 0, "no request");
        // The daemon handles frames in order, so once a request of the
        // test's own is answered, one the calls sent without waiting would
        // have been counted too.
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once.
        unsafe {
            assert_eq!(tobii_calibration_clear(d), 0);
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
        assert_eq!(requests.load(Ordering::Relaxed), 1, "only the clear");
    }

    /// Inside a callback every call is refused before its arguments are
    /// read, as wherever a device handle is taken, and writes nothing.
    #[test]
    fn stimulus_points_are_refused_inside_a_callback() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut out = stimulus_sentinel();
        let mut got = [0; 4];
        crate::device::call(|| {
            got = [
                tobii_calibration_stimulus_points_get(d, &raw mut out),
                tobii_calibration_stimulus_points_get(ptr::null_mut(), &raw mut out),
                tobii_calibration_stimulus_points_get(d, ptr::null_mut()),
                tobii_calibration_stimulus_points_get(ptr::null_mut(), ptr::null_mut()),
            ];
        });
        assert_eq!(got, [TOBII_ERROR_CALLBACK_IN_PROGRESS; 4]);
        assert_eq!(out, stimulus_sentinel(), "nothing written");
        // SAFETY: `d` is a live handle from `Box::into_raw`, destroyed once;
        // the callback guard is down again.
        unsafe { assert_eq!(crate::api::tobii_device_destroy(d), 0) };
    }
}
