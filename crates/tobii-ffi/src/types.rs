//! The C types of the Stream Engine 4.1 ABI that this library reads or
//! writes. Layouts follow `tobii_stream_engine.dll` 4.1.0.3 (see
//! `tools/abi/README.md` for how each was established); the tests pin every
//! size and offset, and `abi-smoke.c` pins the same numbers from the C side.

use std::ffi::{c_char, c_void};

/// Validity flag carried by sample fields; one of the `TOBII_VALIDITY_*` values.
pub type Validity = u32;
/// The field holds no usable measurement.
pub const TOBII_VALIDITY_INVALID: Validity = 0;
/// The field holds a valid measurement.
pub const TOBII_VALIDITY_VALID: Validity = 1;

/// Intended use of the data, declared to `tobii_device_create`; one of the
/// `TOBII_FIELD_OF_USE_*` values.
pub type FieldOfUse = i32;
/// The data drives an interactive experience.
pub const TOBII_FIELD_OF_USE_INTERACTIVE: FieldOfUse = 1;
/// The data is recorded or analysed.
pub const TOBII_FIELD_OF_USE_ANALYTICAL: FieldOfUse = 2;

/// User presence as delivered to [`PresenceFn`]; one of the
/// `TOBII_USER_PRESENCE_STATUS_*` values.
pub type PresenceStatus = u32;
/// Presence could not be determined.
pub const TOBII_USER_PRESENCE_STATUS_UNKNOWN: PresenceStatus = 0;
/// Nobody is in front of the tracker.
pub const TOBII_USER_PRESENCE_STATUS_AWAY: PresenceStatus = 1;
/// A user is in front of the tracker.
pub const TOBII_USER_PRESENCE_STATUS_PRESENT: PresenceStatus = 2;

/// `tobii_state_t`.
pub type State = u32;
/// Power save is active (bool).
pub const TOBII_STATE_POWER_SAVE_ACTIVE: State = 0;
/// Remote wake is active (bool).
pub const TOBII_STATE_REMOTE_WAKE_ACTIVE: State = 1;
/// The device is paused (bool).
pub const TOBII_STATE_DEVICE_PAUSED: State = 2;
/// Exclusive mode (bool).
pub const TOBII_STATE_EXCLUSIVE_MODE: State = 3;
/// Faults (string).
pub const TOBII_STATE_FAULT: State = 4;
/// Warnings (string).
pub const TOBII_STATE_WARNING: State = 5;
/// The active calibration id (`u32`).
pub const TOBII_STATE_CALIBRATION_ID: State = 6;
/// A calibration session is running (bool).
pub const TOBII_STATE_CALIBRATION_ACTIVE: State = 7;

/// `tobii_state_bool_t` false.
pub const TOBII_STATE_BOOL_FALSE: u32 = 0;
/// `tobii_state_bool_t` true.
pub const TOBII_STATE_BOOL_TRUE: u32 = 1;

/// `tobii_supported_t` no.
pub const TOBII_NOT_SUPPORTED: u32 = 0;
/// `tobii_supported_t` yes.
pub const TOBII_SUPPORTED: u32 = 1;

/// `tobii_lens_configuration_writable_t` no.
pub const TOBII_LENS_CONFIGURATION_NOT_WRITABLE: u32 = 0;
/// `tobii_lens_configuration_writable_t` yes.
pub const TOBII_LENS_CONFIGURATION_WRITABLE: u32 = 1;

/// `tobii_capability_t`: the display area can be written.
pub const TOBII_CAPABILITY_DISPLAY_AREA_WRITABLE: u32 = 0;
/// `tobii_capability_t`: 2-D calibration.
pub const TOBII_CAPABILITY_CALIBRATION_2D: u32 = 1;
/// `tobii_capability_t`: user position guide x/y.
pub const TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_XY: u32 = 7;
/// `tobii_capability_t`: user position guide z.
pub const TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_Z: u32 = 8;
/// Highest `tobii_capability_t` in 4.1.
pub const TOBII_CAPABILITY_MAX: u32 = 18;

/// `tobii_stream_t`: gaze point.
///
/// The `tobii_stream_t` values are the 4.1 DLL's, not the pre-4.0 list its
/// documentation still shows. `tobii_stream_supported` passes the value
/// unchanged to the DLL's helper at 0x1801591b0, which the ten stream
/// subscribes other than user presence and digital syncport call with their
/// own number. Its map at 0x180153a50 sends 0, 1, 2, 4, 5, 7 and 11 to
/// tracker streams the DLL names (4 `HEADPOSE`, 5 `ADVANCED_GAZE`,
/// 7 `DIAGNOSTICS_IMAGE`, 11 `WEARABLE_FOVEATED`) and the rest to `INVALID`.
/// 8 is identified by the compound stream `USER_POSITION_GUIDE_XYZ` the
/// helper checks for it (0x1801592f4), and is the number the user position
/// guide subscribe passes (0x18015111d); 3 and 6 by the checks the helper
/// shares with the user presence and digital syncport subscribes (property
/// 0xb, `[+0xa84c] != 2`).
pub const TOBII_STREAM_GAZE_POINT: u32 = 0;
/// `tobii_stream_t`: gaze origin.
pub const TOBII_STREAM_GAZE_ORIGIN: u32 = 1;
/// `tobii_stream_t`: eye position normalised.
pub const TOBII_STREAM_EYE_POSITION_NORMALIZED: u32 = 2;
/// `tobii_stream_t`: user presence.
pub const TOBII_STREAM_USER_PRESENCE: u32 = 3;
/// `tobii_stream_t`: head pose.
pub const TOBII_STREAM_HEAD_POSE: u32 = 4;
/// `tobii_stream_t`: gaze data.
pub const TOBII_STREAM_GAZE_DATA: u32 = 5;
/// `tobii_stream_t`: user position guide.
pub const TOBII_STREAM_USER_POSITION_GUIDE: u32 = 8;

/// `tobii_enabled_eye_t`: left.
pub const TOBII_ENABLED_EYE_LEFT: u32 = 0;
/// `tobii_enabled_eye_t`: right.
pub const TOBII_ENABLED_EYE_RIGHT: u32 = 1;
/// `tobii_enabled_eye_t`: both.
pub const TOBII_ENABLED_EYE_BOTH: u32 = 2;

/// `tobii_feature_group_t`: consumer.
pub const TOBII_FEATURE_GROUP_CONSUMER: u32 = 1;
/// `tobii_license_validation_result_t`: ok.
pub const TOBII_LICENSE_VALIDATION_RESULT_OK: u32 = 0;

/// `tobii_calibration_point_status_t`: failed or invalid.
pub const TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID: u32 = 0;
/// `tobii_calibration_point_status_t`: valid and used.
pub const TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION: u32 = 2;

/// `tobii_notification_value_type_t` numbering.
pub const TOBII_NOTIFICATION_VALUE_TYPE_NONE: u32 = 0;
/// A float.
pub const TOBII_NOTIFICATION_VALUE_TYPE_FLOAT: u32 = 1;
/// A `tobii_state_bool_t`.
pub const TOBII_NOTIFICATION_VALUE_TYPE_STATE: u32 = 2;
/// A display area.
pub const TOBII_NOTIFICATION_VALUE_TYPE_DISPLAY_AREA: u32 = 3;
/// A `u32`.
pub const TOBII_NOTIFICATION_VALUE_TYPE_UINT: u32 = 4;
/// A `tobii_enabled_eye_t`.
pub const TOBII_NOTIFICATION_VALUE_TYPE_ENABLED_EYE: u32 = 5;
/// A string.
pub const TOBII_NOTIFICATION_VALUE_TYPE_STRING: u32 = 6;

/// `tobii_version_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Version {
    /// Major.
    pub major: i32,
    /// Minor.
    pub minor: i32,
    /// Revision.
    pub revision: i32,
    /// Build.
    pub build: i32,
}

/// `tobii_device_info_t`: 2048 bytes of NUL-terminated strings.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Serial number.
    pub serial_number: [c_char; 256],
    /// Model.
    pub model: [c_char; 256],
    /// Generation.
    pub generation: [c_char; 256],
    /// Firmware version.
    pub firmware_version: [c_char; 256],
    /// Integration id.
    pub integration_id: [c_char; 128],
    /// Hardware calibration version.
    pub hw_calibration_version: [c_char; 128],
    /// Hardware calibration date.
    pub hw_calibration_date: [c_char; 128],
    /// Lot id.
    pub lot_id: [c_char; 128],
    /// Integration type.
    pub integration_type: [c_char; 256],
    /// Runtime build version.
    pub runtime_build_version: [c_char; 256],
}

/// `tobii_track_box_t`: eight corners, mm.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TrackBox {
    /// Front face, upper right.
    pub front_upper_right_xyz: [f32; 3],
    /// Front face, upper left.
    pub front_upper_left_xyz: [f32; 3],
    /// Front face, lower left.
    pub front_lower_left_xyz: [f32; 3],
    /// Front face, lower right.
    pub front_lower_right_xyz: [f32; 3],
    /// Back face, upper right.
    pub back_upper_right_xyz: [f32; 3],
    /// Back face, upper left.
    pub back_upper_left_xyz: [f32; 3],
    /// Back face, lower left.
    pub back_lower_left_xyz: [f32; 3],
    /// Back face, lower right.
    pub back_lower_right_xyz: [f32; 3],
}

/// `tobii_display_area_t`: three corners, mm, tracker frame.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DisplayArea {
    /// Top-left corner.
    pub top_left_mm_xyz: [f32; 3],
    /// Top-right corner.
    pub top_right_mm_xyz: [f32; 3],
    /// Bottom-left corner.
    pub bottom_left_mm_xyz: [f32; 3],
}

/// `tobii_geometry_mounting_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeometryMounting {
    /// Number of mounting guides.
    pub guides: i32,
    /// Tracker width, mm.
    pub width_mm: f32,
    /// Camera tilt, degrees.
    pub angle_deg: f32,
    /// External offset, mm.
    pub external_offset_mm_xyz: [f32; 3],
    /// Internal offset, mm.
    pub internal_offset_mm_xyz: [f32; 3],
}

/// `tobii_head_pose_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPose {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Validity of `position_xyz`.
    pub position_validity: Validity,
    /// Head position in millimetres, tracker coordinates.
    pub position_xyz: [f32; 3],
    /// Per-axis validity of `rotation_xyz`.
    pub rotation_validity_xyz: [Validity; 3],
    /// Head rotation in radians about x, y, z.
    pub rotation_xyz: [f32; 3],
}

/// `tobii_gaze_point_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazePoint {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Validity of `position_xy`.
    pub validity: Validity,
    /// Gaze point in normalised display coordinates.
    pub position_xy: [f32; 2],
}

/// `tobii_gaze_origin_t`, `tobii_eye_position_normalized_t` and
/// `tobii_user_position_guide_t` share this shape.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyePair {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Validity of `left_xyz`.
    pub left_validity: Validity,
    /// Left eye.
    pub left_xyz: [f32; 3],
    /// Validity of `right_xyz`.
    pub right_validity: Validity,
    /// Right eye.
    pub right_xyz: [f32; 3],
}

/// `tobii_gaze_data_eye_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeDataEye {
    /// Validity of the gaze origin fields.
    pub gaze_origin_validity: Validity,
    /// Gaze origin, tracker frame, mm.
    pub gaze_origin_from_eye_tracker_mm_xyz: [f32; 3],
    /// Gaze origin, track-box-normalised.
    pub gaze_origin_in_track_box_normalized_xyz: [f32; 3],
    /// Validity of the gaze point fields.
    pub gaze_point_validity: Validity,
    /// Gaze point, tracker frame, mm.
    pub gaze_point_from_eye_tracker_mm_xyz: [f32; 3],
    /// Gaze point, display-normalised.
    pub gaze_point_on_display_normalized_xy: [f32; 2],
    /// Validity of the eyeball centre.
    pub eyeball_center_validity: Validity,
    /// Eyeball centre, tracker frame, mm.
    pub eyeball_center_from_eye_tracker_mm_xyz: [f32; 3],
    /// Validity of the pupil diameter.
    pub pupil_validity: Validity,
    /// Pupil diameter, mm.
    pub pupil_diameter_mm: f32,
}

/// `tobii_gaze_data_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeData {
    /// Device timestamp, microseconds.
    pub timestamp_tracker_us: i64,
    /// Host timestamp, microseconds (the `tobii_system_clock` clock).
    pub timestamp_system_us: i64,
    /// Left eye.
    pub left: GazeDataEye,
    /// Right eye.
    pub right: GazeDataEye,
}

/// The value union of [`Notification`].
#[repr(C)]
#[derive(Clone, Copy)]
pub union NotificationValue {
    /// `TOBII_NOTIFICATION_VALUE_TYPE_FLOAT`.
    pub float_: f32,
    /// `TOBII_NOTIFICATION_VALUE_TYPE_STATE`.
    pub state: u32,
    /// `TOBII_NOTIFICATION_VALUE_TYPE_DISPLAY_AREA`.
    pub display_area: DisplayArea,
    /// `TOBII_NOTIFICATION_VALUE_TYPE_UINT`.
    pub uint_: u32,
    /// `TOBII_NOTIFICATION_VALUE_TYPE_ENABLED_EYE`.
    pub enabled_eye: u32,
    /// `TOBII_NOTIFICATION_VALUE_TYPE_STRING`, NUL-terminated.
    pub string_: [c_char; 512],
}

/// `tobii_notification_t`: 520 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Notification {
    /// `tobii_notification_type_t`.
    pub type_: u32,
    /// `tobii_notification_value_type_t`.
    pub value_type: u32,
    /// The value, per `value_type`.
    pub value: NotificationValue,
}

/// `tobii_image_t` (undocumented; layout from the DLL's image dispatcher).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Image {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Width, pixels.
    pub width: i32,
    /// Padding at the end of each row, pixels.
    pub padding_per_row: i32,
    /// Height, pixels.
    pub height: i32,
    /// Bits per pixel.
    pub bits_per_pixel: i32,
    /// Pixel data, valid for the duration of the callback.
    pub data: *const c_void,
}

/// `tobii_timesync_data_t` (undocumented; layout from the DLL's
/// `device_timesync`, field names from the older public headers): the
/// tracker clock read `tracker_us` at some host time between
/// `system_start_us` and `system_end_us`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimesyncData {
    /// Host time before the tracker time was taken, microseconds (the
    /// `tobii_system_clock` clock).
    pub system_start_us: i64,
    /// Host time after the tracker time was taken, microseconds.
    pub system_end_us: i64,
    /// Tracker time, microseconds (the clock of the samples' timestamps).
    pub tracker_us: i64,
}

/// `tobii_stream_type_t` (undocumented; offsets from the DLL's receiver
/// trampoline at 0x180001d90, size inferred from its 0x88-byte records, field
/// names ours): one entry of the tracker's stream catalogue.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamType {
    /// The Stream Engine's stream type, mapped from the device's stream id;
    /// 0 for a stream it has no type for.
    pub type_: i32,
    /// The device's number for the stream (1000 for `image_collection`, 0
    /// for the rest on the ET5); its meaning is unknown.
    pub value: u32,
    /// Stream name, NUL-terminated.
    pub name: [c_char; 64],
    /// A second string, NUL-terminated; empty on the ET5.
    pub text: [c_char; 64],
}

/// One entry of [`HardwareConfiguration`] (provisional: offsets from the
/// DLL's copy at 0x18014abe0 and its PRP (de)serialisers, which load the
/// 64-bit fields as `double`; the names are ours, and what the fields mean
/// is not known).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareConfigurationEntry {
    /// An id.
    pub id: i32,
    /// A 16.16 value from the tracker, unscaled.
    pub param_a: f32,
    /// A 16.16 value from the tracker, unscaled.
    pub param_b: f32,
    /// A 3-D point, mm.
    pub position_xyz: [f64; 3],
    /// Fifteen values, scaled as lengths (mm).
    pub values: [f64; 15],
    /// A count (2240 on the ET5).
    pub width: i32,
    /// A count (2240 on the ET5).
    pub height: i32,
    /// A number.
    pub param_c: i32,
    /// How many of `coefficients` are set.
    pub coefficient_count: i32,
    /// Values scaled as lengths (mm); the first `coefficient_count` are set.
    pub coefficients: [f64; 64],
    /// A 3-D point, mm.
    pub point_a_xyz: [f64; 3],
    /// A 3-D point, mm.
    pub point_b_xyz: [f64; 3],
    /// A value scaled as a length (mm).
    pub param_d: f64,
}

/// `tobii_hardware_configuration_t` (undocumented, provisional: 2472 bytes,
/// offsets from the DLL's copy at 0x18014abe0 and its PRP deserialiser at
/// 0x180045056; the names and units are ours).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardwareConfiguration {
    /// How many of `entries` are set.
    pub entry_count: i32,
    /// The tracker's entries; the first `entry_count` are set.
    pub entries: [HardwareConfigurationEntry; 2],
    /// How many of `points_xyz` are set.
    pub point_count: i32,
    /// 3-D points, mm; the first `point_count` are set.
    pub points_xyz: [[f64; 3]; 40],
    /// A mode, 0..=2.
    pub mode: i32,
}

/// `tobii_calibration_point_data_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationPointData {
    /// Stimulus point, display-normalised.
    pub point_xy: [f32; 2],
    /// `tobii_calibration_point_status_t` of the left eye.
    pub left_status: u32,
    /// Where the left eye was measured looking.
    pub left_mapping_xy: [f32; 2],
    /// `tobii_calibration_point_status_t` of the right eye.
    pub right_status: u32,
    /// Where the right eye was measured looking.
    pub right_mapping_xy: [f32; 2],
}

/// `tobii_license_key_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LicenseKey {
    /// UTF-16 key text.
    pub license_key: *const u16,
    /// Length in `u16` units.
    pub size_in: usize,
}

/// `tobii_device_name_t`.
pub type DeviceName = [c_char; 64];
/// `tobii_state_string_t`.
pub type StateString = [c_char; 512];

/// `tobii_device_url_receiver_t`.
pub type DeviceUrlReceiver = unsafe extern "C" fn(*const c_char, *mut c_void);
/// `tobii_head_pose_callback_t`.
pub type HeadPoseFn = unsafe extern "C" fn(*const HeadPose, *mut c_void);
/// `tobii_gaze_point_callback_t`.
pub type GazePointFn = unsafe extern "C" fn(*const GazePoint, *mut c_void);
/// `tobii_user_presence_callback_t`.
pub type PresenceFn = unsafe extern "C" fn(PresenceStatus, i64, *mut c_void);
/// `tobii_gaze_origin_callback_t`, `tobii_eye_position_normalized_callback_t`
/// and `tobii_user_position_guide_callback_t`.
pub type EyePairFn = unsafe extern "C" fn(*const EyePair, *mut c_void);
/// `tobii_gaze_data_callback_t`.
pub type GazeDataFn = unsafe extern "C" fn(*const GazeData, *mut c_void);
/// `tobii_image_callback_t`.
pub type ImageFn = unsafe extern "C" fn(*const Image, *mut c_void);
/// `tobii_notifications_callback_t`.
pub type NotificationsFn = unsafe extern "C" fn(*const Notification, *mut c_void);
/// `tobii_field_of_use_callback_t`.
pub type FieldOfUseFn = unsafe extern "C" fn(FieldOfUse, *mut c_void);
/// `tobii_data_receiver_t`.
pub type DataReceiver = unsafe extern "C" fn(*const c_void, usize, *mut c_void);
/// `tobii_output_frequency_receiver_t`.
pub type OutputFrequencyReceiver = unsafe extern "C" fn(f32, *mut c_void);
/// `tobii_stream_type_receiver_t`.
pub type StreamTypeReceiver = unsafe extern "C" fn(*const StreamType, *mut c_void);
/// `tobii_calibration_point_data_receiver_t`.
pub type CalibrationPointReceiver = unsafe extern "C" fn(*const CalibrationPointData, *mut c_void);

/// Copy `s` into a fixed C string buffer, truncated to leave room for the NUL.
pub(crate) fn copy_c_string(dst: &mut [c_char], s: &str) {
    copy_c_bytes(dst, s.as_bytes());
}

/// Copy raw bytes (not necessarily UTF-8) into a fixed C string buffer,
/// truncated to leave room for the NUL.
pub(crate) fn copy_c_bytes(dst: &mut [c_char], bytes: &[u8]) {
    let n = bytes.len().min(dst.len().saturating_sub(1));
    for (d, b) in dst.iter_mut().zip(&bytes[..n]) {
        *d = c_char::from_ne_bytes([*b]);
    }
    if let Some(end) = dst.get_mut(n) {
        *end = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn layouts_match_the_stream_engine() {
        assert_eq!(size_of::<Version>(), 16);
        assert_eq!(size_of::<DeviceInfo>(), 2048);
        assert_eq!(offset_of!(DeviceInfo, model), 0x100);
        assert_eq!(offset_of!(DeviceInfo, integration_id), 0x400);
        assert_eq!(offset_of!(DeviceInfo, runtime_build_version), 0x700);
        assert_eq!(size_of::<TrackBox>(), 96);
        assert_eq!(size_of::<DisplayArea>(), 36);
        assert_eq!(size_of::<GeometryMounting>(), 36);

        assert_eq!(size_of::<HeadPose>(), 48);
        assert_eq!(align_of::<HeadPose>(), 8);
        assert_eq!(offset_of!(HeadPose, position_validity), 8);
        assert_eq!(offset_of!(HeadPose, position_xyz), 12);
        assert_eq!(offset_of!(HeadPose, rotation_validity_xyz), 24);
        assert_eq!(offset_of!(HeadPose, rotation_xyz), 36);
        assert_eq!(size_of::<GazePoint>(), 24);
        assert_eq!(offset_of!(GazePoint, position_xy), 12);

        assert_eq!(size_of::<EyePair>(), 40);
        assert_eq!(offset_of!(EyePair, right_validity), 24);
        assert_eq!(size_of::<GazeDataEye>(), 76);
        assert_eq!(size_of::<GazeData>(), 168);
        assert_eq!(offset_of!(GazeData, left), 16);
        assert_eq!(offset_of!(GazeData, right), 92);

        assert_eq!(size_of::<Notification>(), 520);
        assert_eq!(align_of::<Notification>(), 4);
        assert_eq!(offset_of!(Notification, value), 8);

        assert_eq!(size_of::<Image>(), 32);
        assert_eq!(offset_of!(Image, padding_per_row), 12);
        assert_eq!(offset_of!(Image, data), 24);

        assert_eq!(size_of::<TimesyncData>(), 24);
        assert_eq!(align_of::<TimesyncData>(), 8);
        assert_eq!(offset_of!(TimesyncData, system_end_us), 8);
        assert_eq!(offset_of!(TimesyncData, tracker_us), 16);

        assert_eq!(size_of::<StreamType>(), 136);
        assert_eq!(offset_of!(StreamType, type_), 0);
        assert_eq!(offset_of!(StreamType, value), 4);
        assert_eq!(offset_of!(StreamType, name), 8);
        assert_eq!(offset_of!(StreamType, text), 72);

        assert_eq!(size_of::<HardwareConfigurationEntry>(), 0x2e8);
        assert_eq!(offset_of!(HardwareConfigurationEntry, param_a), 4);
        assert_eq!(offset_of!(HardwareConfigurationEntry, param_b), 8);
        assert_eq!(offset_of!(HardwareConfigurationEntry, position_xyz), 0x10);
        assert_eq!(offset_of!(HardwareConfigurationEntry, values), 0x28);
        assert_eq!(offset_of!(HardwareConfigurationEntry, width), 0xa0);
        assert_eq!(offset_of!(HardwareConfigurationEntry, height), 0xa4);
        assert_eq!(offset_of!(HardwareConfigurationEntry, param_c), 0xa8);
        assert_eq!(
            offset_of!(HardwareConfigurationEntry, coefficient_count),
            0xac
        );
        assert_eq!(offset_of!(HardwareConfigurationEntry, coefficients), 0xb0);
        assert_eq!(offset_of!(HardwareConfigurationEntry, point_a_xyz), 0x2b0);
        assert_eq!(offset_of!(HardwareConfigurationEntry, point_b_xyz), 0x2c8);
        assert_eq!(offset_of!(HardwareConfigurationEntry, param_d), 0x2e0);
        assert_eq!(size_of::<HardwareConfiguration>(), 0x9a8);
        assert_eq!(align_of::<HardwareConfiguration>(), 8);
        assert_eq!(offset_of!(HardwareConfiguration, entries), 8);
        assert_eq!(offset_of!(HardwareConfiguration, point_count), 0x5d8);
        assert_eq!(offset_of!(HardwareConfiguration, points_xyz), 0x5e0);
        assert_eq!(offset_of!(HardwareConfiguration, mode), 0x9a0);

        assert_eq!(size_of::<CalibrationPointData>(), 32);
        assert_eq!(offset_of!(CalibrationPointData, right_status), 20);
        assert_eq!(size_of::<LicenseKey>(), 16);
        assert_eq!(size_of::<DeviceName>(), 64);
        assert_eq!(size_of::<StateString>(), 512);
    }

    #[test]
    fn c_strings_are_truncated_and_terminated() {
        let mut buf: [c_char; 4] = [1; 4];
        copy_c_string(&mut buf, "IS5_Large");
        assert_eq!(buf.map(|c| c.to_ne_bytes()[0]), *b"IS5\0");
        copy_c_string(&mut buf, "");
        assert_eq!(buf[0], 0);
        copy_c_bytes(&mut buf, &[0xff, 0xfe]);
        assert_eq!(buf.map(|c| c.to_ne_bytes()[0]), [0xff, 0xfe, 0, 0]);
        let mut name: DeviceName = [1; 64];
        copy_c_bytes(&mut name, &[b'x'; 100]);
        assert_eq!(name[62].to_ne_bytes(), *b"x");
        assert_eq!(name[63], 0);
    }
}
