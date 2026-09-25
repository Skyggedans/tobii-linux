//! `tobii_config.h` calibration: the session runs in the daemon (one owner at
//! a time; the result is saved as the user's calibration only when the owner
//! stops the session); parsing a blob is done here.
//!
//! 2-D calibration of both eyes is what the Windows engine was captured doing;
//! discarding a 2-D point uses the command the DLL sends for it. 3-D and
//! per-eye variants were never observed and return
//! `TOBII_ERROR_NOT_SUPPORTED`.

use std::ffi::c_void;
use std::time::Duration;

use tobii_ipc::request::{STOP_KEEP, encode_point_2d, kind};

use crate::device::{Api, Device, device_mut, in_callback};
use crate::status::{
    Status, TOBII_ERROR_CALLBACK_IN_PROGRESS, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
};
use crate::stub::not_supported;
use crate::types::{
    CalibrationPointData, CalibrationPointReceiver, DataReceiver,
    TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
    TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
    TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
};

// Client-side timeouts cover the daemon's own device timeouts plus queueing.
const START_TIMEOUT: Duration = Duration::from_secs(25);
const STOP_TIMEOUT: Duration = Duration::from_secs(20);
const COLLECT_TIMEOUT: Duration = Duration::from_secs(8);
/// The daemon's 5 s command timeout plus the 30 s a command may wait queued.
const DISCARD_TIMEOUT: Duration = Duration::from_secs(40);
const COMPUTE_TIMEOUT: Duration = Duration::from_secs(20);
const RETRIEVE_TIMEOUT: Duration = Duration::from_secs(8);
const APPLY_TIMEOUT: Duration = Duration::from_secs(15);
const CLEAR_TIMEOUT: Duration = Duration::from_secs(8);

/// Run a request on `device`, discarding the reply payload.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
pub(crate) unsafe fn request(
    device: *mut Device,
    request: u8,
    payload: &[u8],
    timeout: Duration,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    match unsafe { device_mut(device) } {
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
/// `device` must be null or a live handle that no other thread uses during
/// the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_start(device: *mut Device, enabled_eye: u32) -> Status {
    let Ok(eye) = u8::try_from(enabled_eye) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    // SAFETY: forwarded under the same contract.
    unsafe { request(device, kind::CALIBRATION_START, &[eye], START_TIMEOUT) }
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
    unsafe { request(device, kind::CALIBRATION_STOP, STOP_KEEP, STOP_TIMEOUT) }
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
            COLLECT_TIMEOUT,
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
            DISCARD_TIMEOUT,
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
    unsafe { request(device, kind::CALIBRATION_CLEAR, &[], CLEAR_TIMEOUT) }
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
    unsafe { request(device, kind::CALIBRATION_COMPUTE, &[], COMPUTE_TIMEOUT) }
}

/// Read the active calibration and hand it to `receiver`.
///
/// # Safety
/// As `tobii_calibration_start`; `receiver` must be null or sound to call
/// with a data pointer, its size and `user_data`, and must not re-enter this
/// library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calibration_retrieve(
    device: *mut Device,
    receiver: Option<DataReceiver>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    match d.request(kind::CALIBRATION_RETRIEVE, &[], RETRIEVE_TIMEOUT) {
        Ok(blob) => {
            // SAFETY: the caller guarantees `receiver` is sound to call like
            // this; `blob` outlives the call.
            unsafe { receiver(blob.as_ptr().cast(), blob.len(), user_data) };
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
    unsafe { request(device, kind::CALIBRATION_APPLY, blob, APPLY_TIMEOUT) }
}

/// A record's status word as a `tobii_calibration_point_status_t`, mapped as
/// the DLL maps it (0x180147910): only the low 32 bits count; 1 was used in
/// the calibration, 0 is valid but was not used, and anything else (the DLL
/// tests for -1 explicitly) failed or is invalid. The mask only mirrors the
/// DLL: `tobii_calib::blob::points` passes no word above 2.
fn point_status(word: u64) -> u32 {
    match word & 0xffff_ffff {
        1 => TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
        0 => TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
        _ => TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
    }
}

/// Hand each calibration point stored in `data` to `receiver`. The first
/// measurement of each record is reported as the left eye, and each eye's
/// status word is mapped as the DLL maps it; a word above 2, the DLL's -1
/// included, makes the blob invalid here (`TOBII_ERROR_INVALID_PARAMETER`, no
/// points), where the DLL reports that eye as `FAILED_OR_INVALID`. A call from
/// inside a callback is `TOBII_ERROR_CALLBACK_IN_PROGRESS` once the arguments
/// have been checked, before `data` is parsed, as in the DLL.
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
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if api.is_null() || data.is_null() || data_size < 8 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    if in_callback() {
        return TOBII_ERROR_CALLBACK_IN_PROGRESS;
    }
    // SAFETY: non-null, and the caller guarantees `data_size` readable bytes.
    let blob = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), data_size) };
    let Ok(points) = tobii_calib::blob::points(blob) else {
        return TOBII_ERROR_INVALID_PARAMETER;
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
    /// Undocumented; its point type is unknown.
    fn tobii_calibration_stimulus_points_get(device: *mut c_void, points: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

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
            // -1 as the DLL reads it, a 32-bit int.
            (
                0xffff_ffff,
                TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
            ),
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
        // The first measurement's word is at +16, the second's at +32. 2 is
        // the one word besides 0 and 1 that the blob check lets through.
        for (i, offset, word) in [(2, 16, 0u64), (2, 32, 2), (4, 16, 2), (4, 32, 0)] {
            let o = first + tobii_calib::blob::RECORD_LEN * i + offset;
            blob[o..o + 8].copy_from_slice(&word.to_le_bytes());
        }
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
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
        }
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
}
