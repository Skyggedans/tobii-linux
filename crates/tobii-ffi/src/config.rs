//! `tobii_config.h`, apart from calibration: enabled eye, display area and
//! mounting geometry, device name and output frequency.

use std::ffi::{c_char, c_void};

use tobii_ipc::geometry::{self, GeometryMounting as WireMounting};
use tobii_ipc::request::{
    DEVICE_NAME_MAX, decode_display_area, decode_geometry_mounting, encode_display_area, kind,
};

use crate::device::{Api, Device, device_ref};
use crate::status::{
    Status, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR, TOBII_ERROR_NOT_SUPPORTED,
};
use crate::timeouts;
use crate::types::{
    DeviceName, DisplayArea, GeometryMounting, OutputFrequencyReceiver, TOBII_ENABLED_EYE_BOTH,
    copy_c_bytes,
};

/// The ET5's output frequency, Hz (what command 1650 reports).
const OUTPUT_FREQUENCY_HZ: f32 = 33.0;

/// Only both eyes: per-eye tracking was never captured.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that is
/// not destroyed before the call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_set_enabled_eye(device: *mut Device, enabled_eye: u32) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    match enabled_eye {
        TOBII_ENABLED_EYE_BOTH => TOBII_ERROR_NO_ERROR,
        0 | 1 => TOBII_ERROR_NOT_SUPPORTED,
        _ => TOBII_ERROR_INVALID_PARAMETER,
    }
}

/// Always both eyes.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `enabled_eye` must be null or valid
/// for writing one `tobii_enabled_eye_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_enabled_eye(
    device: *mut Device,
    enabled_eye: *mut u32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    if enabled_eye.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { enabled_eye.write(TOBII_ENABLED_EYE_BOTH) };
    TOBII_ERROR_NO_ERROR
}

#[allow(clippy::cast_possible_truncation)] // reason: the C ABI carries float
fn mounting_c(m: &WireMounting) -> GeometryMounting {
    GeometryMounting {
        guides: m.guides,
        width_mm: m.width_mm as f32,
        angle_deg: m.angle_deg as f32,
        external_offset_mm_xyz: m.external_offset_mm.map(|v| v as f32),
        internal_offset_mm_xyz: m.internal_offset_mm.map(|v| v as f32),
    }
}

fn mounting_wire(m: &GeometryMounting) -> WireMounting {
    WireMounting {
        guides: m.guides,
        width_mm: f64::from(m.width_mm),
        angle_deg: f64::from(m.angle_deg),
        external_offset_mm: m.external_offset_mm_xyz.map(f64::from),
        internal_offset_mm: m.internal_offset_mm_xyz.map(f64::from),
    }
}

fn display_area_wire(a: &DisplayArea) -> geometry::DisplayArea {
    geometry::DisplayArea {
        top_left_mm: a.top_left_mm_xyz.map(f64::from),
        top_right_mm: a.top_right_mm_xyz.map(f64::from),
        bottom_left_mm: a.bottom_left_mm_xyz.map(f64::from),
    }
}

/// How the tracker is mounted.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `geometry_mounting` must be null or
/// valid for writing one `tobii_geometry_mounting_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_geometry_mounting(
    device: *mut Device,
    geometry_mounting: *mut GeometryMounting,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if geometry_mounting.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::GEOMETRY_MOUNTING, &[], timeouts::FACTS)
        .map(|p| decode_geometry_mounting(&p))
    {
        Ok(Some(m)) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { geometry_mounting.write(mounting_c(&m)) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => d.malformed("geometry mounting"),
        Err(status) => status,
    }
}

/// The display area in effect, tracker frame, mm.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `display_area` must be null or valid
/// for writing one `tobii_display_area_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_display_area(
    device: *mut Device,
    display_area: *mut DisplayArea,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if display_area.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::DISPLAY_AREA_GET, &[], timeouts::FACTS)
        .map(|p| decode_display_area(&p))
    {
        Ok(Some(a)) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { display_area.write(crate::device::display_area(&a)) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => d.malformed("display area"),
        Err(status) => status,
    }
}

/// Write the display area to the device. The daemon keeps it and re-applies
/// it whenever the device is re-initialised.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `display_area` must be null or point
/// to a readable `tobii_display_area_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_set_display_area(
    device: *mut Device,
    display_area: *const DisplayArea,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    // SAFETY: the caller guarantees `display_area` is null or readable.
    let Some(area) = (unsafe { display_area.as_ref() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let payload = encode_display_area(&display_area_wire(area));
    match d.request(kind::DISPLAY_AREA_SET, &payload, timeouts::DISPLAY_AREA_SET) {
        Ok(_) => TOBII_ERROR_NO_ERROR,
        Err(status) => status,
    }
}

/// The display area of a `width_mm` x `height_mm` screen centred
/// `offset_x_mm` right of the tracker, for the given mounting. Pure
/// computation; reproduces the Windows engine's areas to under a micrometre.
///
/// # Safety
/// `api` must be null or a live handle; `geometry_mounting` must be null or
/// point to a readable `tobii_geometry_mounting_t`; `display_area` must be
/// null or valid for writing one `tobii_display_area_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_calculate_display_area_basic(
    api: *mut Api,
    width_mm: f32,
    height_mm: f32,
    offset_x_mm: f32,
    geometry_mounting: *const GeometryMounting,
    display_area: *mut DisplayArea,
) -> Status {
    // SAFETY: the caller guarantees `geometry_mounting` is null or readable.
    let mounting = unsafe { geometry_mounting.as_ref() };
    let (Some(mounting), false, false) = (mounting, api.is_null(), display_area.is_null()) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    if !(width_mm.is_finite() && height_mm.is_finite() && offset_x_mm.is_finite()) {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let area = geometry::display_area_basic(
        f64::from(width_mm),
        f64::from(height_mm),
        f64::from(offset_x_mm),
        &mounting_wire(mounting),
    );
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { display_area.write(crate::device::display_area(&area)) };
    TOBII_ERROR_NO_ERROR
}

/// The device's name: the one a client set, kept by the daemon, else its
/// model string. Asked of the daemon on every call, since another process
/// may rename the device; a daemon that predates names gives the model.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `device_name` must be null or valid
/// for writing a `tobii_device_name_t` (64 bytes).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_device_name(
    device: *mut Device,
    device_name: *mut DeviceName,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if device_name.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    let bytes = match d.request(kind::DEVICE_NAME_GET, &[], timeouts::FACTS) {
        Ok(bytes) => bytes,
        Err(TOBII_ERROR_NOT_SUPPORTED) => match d.device_info() {
            Ok(info) => info.model.into_bytes(),
            Err(status) => return status,
        },
        Err(status) => return status,
    };
    let mut name: DeviceName = [0; 64];
    copy_c_bytes(&mut name, &bytes);
    // SAFETY: non-null, and the caller guarantees 64 writable bytes.
    unsafe { device_name.write(name) };
    TOBII_ERROR_NO_ERROR
}

/// Name the device. The daemon keeps the name, for every client and later
/// sessions; nothing is written to the tracker. At most 63 bytes are read,
/// up to the NUL, and kept as they are (empty and non-UTF-8 names too). A
/// null name is `TOBII_ERROR_INVALID_PARAMETER`, where the DLL crashes.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `device_name` must be null, or
/// readable up to its NUL or for 63 bytes, whichever comes first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_set_device_name(
    device: *mut Device,
    device_name: *const c_char,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    let d = match unsafe { device_ref(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if device_name.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is readable that far.
    let name = unsafe { name_bytes(device_name) };
    match d.request(kind::DEVICE_NAME_SET, &name, timeouts::FACTS) {
        Ok(_) => TOBII_ERROR_NO_ERROR,
        Err(status) => status,
    }
}

/// The bytes of a C name before its NUL, at most [`DEVICE_NAME_MAX`]: never
/// past the 64 of a `tobii_device_name_t`, however it ends.
///
/// # Safety
/// `name` must be readable up to its NUL or for [`DEVICE_NAME_MAX`] bytes,
/// whichever comes first.
unsafe fn name_bytes(name: *const c_char) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(DEVICE_NAME_MAX);
    for i in 0..DEVICE_NAME_MAX {
        // SAFETY: `i` is below the limit and no NUL came before it, so the
        // caller guarantees this byte is readable.
        let c = unsafe { name.add(i).read() };
        if c == 0 {
            break;
        }
        bytes.push(c.to_ne_bytes()[0]);
    }
    bytes
}

/// The one output frequency the ET5 runs at.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `receiver` must be null or sound to
/// call with a frequency and `user_data`, and must not re-enter this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_enumerate_output_frequencies(
    device: *mut Device,
    receiver: Option<OutputFrequencyReceiver>,
    user_data: *mut c_void,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    let Some(receiver) = receiver else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    // SAFETY: the caller guarantees `receiver` is sound to call like this.
    unsafe { receiver(OUTPUT_FREQUENCY_HZ, user_data) };
    TOBII_ERROR_NO_ERROR
}

/// Accept the frequency the device already runs at; others are unsupported
/// and a negative one invalid, as in the DLL.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_set_output_frequency(
    device: *mut Device,
    output_frequency: f32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    if output_frequency.is_nan() || output_frequency < 0.0 {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    if (output_frequency - OUTPUT_FREQUENCY_HZ).abs() < 0.5 {
        TOBII_ERROR_NO_ERROR
    } else {
        TOBII_ERROR_NOT_SUPPORTED
    }
}

/// 33 Hz.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `output_frequency` must be null or
/// valid for writing one `float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_output_frequency(
    device: *mut Device,
    output_frequency: *mut f32,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live handle, not
    // destroyed before this returns.
    if let Err(status) = unsafe { device_ref(device) } {
        return status;
    }
    if output_frequency.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    // SAFETY: non-null, and the caller guarantees it is writable.
    unsafe { output_frequency.write(OUTPUT_FREQUENCY_HZ) };
    TOBII_ERROR_NO_ERROR
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::sync::{Arc, Mutex};
    use tobii_ipc::encode_reply;
    use tobii_ipc::request::{DeviceInfo, decode_request, encode_device_info};

    /// The mounting and monitor from the Windows capture: the result must be
    /// the display area the Windows engine wrote.
    #[test]
    fn calculates_the_captured_display_area() {
        let mut api: *mut Api = ptr::null_mut();
        let mounting = GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 20.0,
            external_offset_mm_xyz: [0.0, -0.16, 13.85],
            internal_offset_mm_xyz: [0.0, 5.38, 9.86],
        };
        let mut area = DisplayArea::default();
        // SAFETY: live locals; the api handle is destroyed once.
        unsafe {
            assert_eq!(
                crate::api::tobii_api_create(&raw mut api, ptr::null(), ptr::null()),
                0
            );
            assert_eq!(
                tobii_calculate_display_area_basic(
                    api,
                    597.0,
                    336.0,
                    1.002_258_3,
                    &raw const mounting,
                    &raw mut area
                ),
                0
            );
            assert_eq!(
                tobii_calculate_display_area_basic(
                    api,
                    f32::NAN,
                    336.0,
                    0.0,
                    &raw const mounting,
                    &raw mut area
                ),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(crate::api::tobii_api_destroy(api), 0);
        }
        // change-display.pcapng: TL (-304637.69, 333828.16, 114502.40) / 1024.
        let expected_tl = [-297.497_73, 326.004_06, 111.818_75];
        for (got, want) in area.top_left_mm_xyz.iter().zip(expected_tl) {
            assert!((got - want).abs() < 1e-3, "{:?}", area.top_left_mm_xyz);
        }
        assert!((area.top_right_mm_xyz[0] - area.top_left_mm_xyz[0] - 597.0).abs() < 1e-3);
    }

    /// Every request a fake daemon got: (kind, payload).
    type RequestLog = Arc<Mutex<Vec<(u8, Vec<u8>)>>>;

    /// A daemon that knows the device as `name`, or predates names when it
    /// is `None`; every request it gets is logged as (kind, payload).
    fn named_device(name: Option<&'static [u8]>) -> (*mut Device, RequestLog) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let connect = crate::device::tests::fake_daemon(move |body| {
            let req = decode_request(body).expect("request");
            log.lock()
                .expect("log")
                .push((req.kind, req.payload.to_vec()));
            let (status, payload) = match (req.kind, name) {
                (kind::DEVICE_NAME_GET, Some(name)) => (0, name.to_vec()),
                (kind::DEVICE_NAME_SET, Some(_)) => (0, Vec::new()),
                (kind::DEVICE_INFO, _) => (
                    0,
                    encode_device_info(&DeviceInfo {
                        model: "IS5_Large_Eyetracker_5".into(),
                        ..DeviceInfo::default()
                    }),
                ),
                _ => (tobii_ipc::request::status::NOT_SUPPORTED, Vec::new()),
            };
            vec![encode_reply(req.id, status, &payload)]
        });
        let d = Box::into_raw(Box::new(Device::new(connect, 1, 1).expect("device")));
        (d, seen)
    }

    fn c_name(bytes: &[u8]) -> DeviceName {
        let mut name: DeviceName = [0; 64];
        for (c, b) in name.iter_mut().zip(bytes) {
            *c = c_char::from_ne_bytes([*b]);
        }
        name
    }

    #[test]
    fn a_set_name_is_read_up_to_its_nul_and_never_past_63_bytes() {
        let (d, seen) = named_device(Some(b""));
        // No NUL anywhere in the 64 bytes.
        let full = c_name(&[b'x'; 64]);
        let desk = c_name(b"Desk\0junk");
        let raw = c_name(&[0xff, 0xfe]);
        // SAFETY: `d` is live and destroyed once; the names are live locals.
        unsafe {
            assert_eq!(tobii_set_device_name(d, full.as_ptr()), 0);
            assert_eq!(tobii_set_device_name(d, desk.as_ptr()), 0);
            assert_eq!(tobii_set_device_name(d, raw.as_ptr()), 0);
            assert_eq!(
                tobii_set_device_name(d, ptr::null()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_set_device_name(ptr::null_mut(), desk.as_ptr()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
        assert_eq!(
            *seen.lock().expect("log"),
            vec![
                (kind::DEVICE_NAME_SET, vec![b'x'; 63]),
                (kind::DEVICE_NAME_SET, b"Desk".to_vec()),
                (kind::DEVICE_NAME_SET, vec![0xff, 0xfe]),
            ]
        );
    }

    #[test]
    fn every_name_get_asks_the_daemon() {
        let (d, seen) = named_device(Some(&[b'D', b'e', b's', b'k', 0xff]));
        let mut name: DeviceName = [1; 64];
        // SAFETY: `d` is live and destroyed once; `name` a live local.
        unsafe {
            assert_eq!(tobii_get_device_name(d, &raw mut name), 0);
            assert_eq!(tobii_get_device_name(d, &raw mut name), 0);
            assert_eq!(
                tobii_get_device_name(d, ptr::null_mut()),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(
                tobii_get_device_name(ptr::null_mut(), &raw mut name),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
        assert_eq!(name, c_name(&[b'D', b'e', b's', b'k', 0xff]));
        assert_eq!(
            *seen.lock().expect("log"),
            vec![(kind::DEVICE_NAME_GET, vec![]); 2],
            "no cached answer"
        );
    }

    #[test]
    fn a_daemon_without_names_gives_the_model_and_refuses_a_set() {
        let (d, seen) = named_device(None);
        let mut name: DeviceName = [1; 64];
        let desk = c_name(b"Desk");
        // SAFETY: `d` is live and destroyed once; the names are live locals.
        unsafe {
            assert_eq!(tobii_get_device_name(d, &raw mut name), 0);
            assert_eq!(
                tobii_set_device_name(d, desk.as_ptr()),
                TOBII_ERROR_NOT_SUPPORTED
            );
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
        assert_eq!(name, c_name(b"IS5_Large_Eyetracker_5"));
        let kinds: Vec<u8> = seen.lock().expect("log").iter().map(|r| r.0).collect();
        assert_eq!(
            kinds,
            vec![
                kind::DEVICE_NAME_GET,
                kind::DEVICE_INFO,
                kind::DEVICE_NAME_SET
            ]
        );

        // Any other failure is the daemon's status.
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(6, vec![])));
        // SAFETY: `d` is live and destroyed once; the names are live locals.
        unsafe {
            assert_eq!(tobii_get_device_name(d, &raw mut name), 6);
            assert_eq!(tobii_set_device_name(d, desk.as_ptr()), 6);
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
    }

    #[test]
    fn output_frequency_is_fixed_at_33_hz() {
        let d = Box::into_raw(Box::new(crate::device::tests::device_with(0, vec![])));
        let mut hz = 0.0f32;
        // SAFETY: `d` is live and destroyed once; `hz` a live local.
        unsafe {
            assert_eq!(tobii_get_output_frequency(d, &raw mut hz), 0);
            assert_eq!(tobii_set_output_frequency(d, 33.0), 0);
            assert_eq!(
                tobii_set_output_frequency(d, 60.0),
                TOBII_ERROR_NOT_SUPPORTED
            );
            assert_eq!(
                tobii_set_output_frequency(d, -1.0),
                TOBII_ERROR_INVALID_PARAMETER
            );
            assert_eq!(tobii_set_enabled_eye(d, TOBII_ENABLED_EYE_BOTH), 0);
            assert_eq!(tobii_set_enabled_eye(d, 0), TOBII_ERROR_NOT_SUPPORTED);
            assert_eq!(crate::api::tobii_device_destroy(d), 0);
        }
        assert!((hz - 33.0).abs() < f32::EPSILON);
    }
}
