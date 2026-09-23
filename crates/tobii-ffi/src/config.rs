//! `tobii_config.h`, apart from calibration: enabled eye, display area and
//! mounting geometry, device name and output frequency.

use std::ffi::{c_char, c_void};
use std::time::Duration;

use tobii_ipc::geometry::{self, GeometryMounting as WireMounting};
use tobii_ipc::request::{
    decode_display_area, decode_geometry_mounting, encode_display_area, kind,
};

use crate::api::{FACTS_TIMEOUT, fetch_device_info};
use crate::device::{Api, Device, device_mut};
use crate::status::{
    Status, TOBII_ERROR_INTERNAL, TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR,
    TOBII_ERROR_NOT_SUPPORTED,
};
use crate::stub::not_supported;
use crate::types::{
    DeviceName, DisplayArea, GeometryMounting, OutputFrequencyReceiver, TOBII_ENABLED_EYE_BOTH,
    copy_c_string,
};

/// The ET5's output frequency, Hz (what command 1650 reports).
const OUTPUT_FREQUENCY_HZ: f32 = 33.0;
/// How long the device may take to acknowledge a display area.
const DISPLAY_AREA_TIMEOUT: Duration = Duration::from_secs(8);

/// Only both eyes: per-eye tracking was never captured.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_set_enabled_eye(device: *mut Device, enabled_eye: u32) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if geometry_mounting.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::GEOMETRY_MOUNTING, &[], FACTS_TIMEOUT)
        .map(|p| decode_geometry_mounting(&p))
    {
        Ok(Some(m)) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { geometry_mounting.write(mounting_c(&m)) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => TOBII_ERROR_INTERNAL,
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if display_area.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match d
        .request(kind::DISPLAY_AREA_GET, &[], FACTS_TIMEOUT)
        .map(|p| decode_display_area(&p))
    {
        Ok(Some(a)) => {
            // SAFETY: non-null, and the caller guarantees it is writable.
            unsafe { display_area.write(crate::device::display_area(&a)) };
            TOBII_ERROR_NO_ERROR
        }
        Ok(None) => TOBII_ERROR_INTERNAL,
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    // SAFETY: the caller guarantees `display_area` is null or readable.
    let Some(area) = (unsafe { display_area.as_ref() }) else {
        return TOBII_ERROR_INVALID_PARAMETER;
    };
    let payload = encode_display_area(&display_area_wire(area));
    match d.request(kind::DISPLAY_AREA_SET, &payload, DISPLAY_AREA_TIMEOUT) {
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

/// The device's name: its model string.
///
/// # Safety
/// `device` as `tobii_set_enabled_eye`; `device_name` must be null or valid
/// for writing a `tobii_device_name_t` (64 bytes).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tobii_get_device_name(
    device: *mut Device,
    device_name: *mut DeviceName,
) -> Status {
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    let d = match unsafe { device_mut(device) } {
        Ok(d) => d,
        Err(status) => return status,
    };
    if device_name.is_null() {
        return TOBII_ERROR_INVALID_PARAMETER;
    }
    match fetch_device_info(d) {
        Ok(info) => {
            let mut name: DeviceName = [0; 64];
            copy_c_string(&mut name, &info.model);
            // SAFETY: non-null, and the caller guarantees 64 writable bytes.
            unsafe { device_name.write(name) };
            TOBII_ERROR_NO_ERROR
        }
        Err(status) => status,
    }
}

not_supported! {
    /// Renaming the device was never captured.
    fn tobii_set_device_name(device: *mut c_void, device_name: *const c_char);
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
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
    // SAFETY: caller guarantees `device` is null or a live, unaliased handle.
    if let Err(status) = unsafe { device_mut(device) } {
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
