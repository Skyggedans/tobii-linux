//! Daemon -> client frames: the sample streams, subscription acks and request
//! replies, and their codecs.
//!
//! Every sample timestamp is the **host** clock ([`crate::host_clock_us`],
//! microseconds): the device time the sample was taken at, which the daemon
//! maps onto the host clock, as the Stream Engine's callbacks carry it. Only
//! [`GazeData::timestamp_tracker_us`] and [`GazeRaw::timestamp_tracker_us`]
//! keep the device clock, which [`crate::request::Timesync`] pairs with the
//! host clock.

use crate::geometry::DisplayArea;
use crate::wire::{Reader, Writer};
use crate::{
    TAG_EYE_POSITION, TAG_GAZE, TAG_GAZE_DATA, TAG_GAZE_ORIGIN, TAG_GAZE_RAW, TAG_HEAD, TAG_IMAGE,
    TAG_NOTIFICATION, TAG_PRESENCE, TAG_REPLY, TAG_SUBSCRIBED,
};

/// One eye's 3-D point and whether it is usable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EyePoint {
    /// Whether `xyz` holds a measurement for this frame.
    pub valid: bool,
    /// The point; its frame and unit depend on the stream carrying it.
    pub xyz: [f32; 3],
}

/// A per-eye 3-D sample: `tobii_gaze_origin_t` (display frame, mm) or
/// `tobii_eye_position_normalized_t` / `tobii_user_position_guide_t`
/// (track-box-normalised, `0..1` per axis).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EyePair {
    /// When the sample was taken, on the host clock, microseconds.
    pub ts_us: i64,
    /// The user's left eye.
    pub left: EyePoint,
    /// The user's right eye.
    pub right: EyePoint,
}

/// One eye's half of a [`GazeData`] sample: `tobii_gaze_data_eye_t`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GazeDataEye {
    /// Whether the gaze origin fields are valid.
    pub gaze_origin_valid: bool,
    /// Gaze origin (the cornea centre) in the tracker frame, mm.
    pub gaze_origin_mm: [f32; 3],
    /// Gaze origin normalised to the track box, `0..1` per axis.
    pub gaze_origin_in_track_box: [f32; 3],
    /// Whether the gaze point fields are valid.
    pub gaze_point_valid: bool,
    /// Gaze point on the screen plane, in the tracker frame, mm.
    pub gaze_point_mm: [f32; 3],
    /// Gaze point in normalised display coordinates, `0..1` per axis.
    pub gaze_point_on_display: [f32; 2],
    /// Whether the eyeball centre is valid.
    pub eyeball_center_valid: bool,
    /// Eyeball rotation centre in the tracker frame, mm.
    pub eyeball_center_mm: [f32; 3],
    /// Whether the pupil diameter is valid.
    pub pupil_valid: bool,
    /// Pupil diameter, mm.
    pub pupil_diameter_mm: f32,
}

/// The per-eye "gaze data" sample: `tobii_gaze_data_t`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GazeData {
    /// When the sample was taken, on the device clock, microseconds.
    pub timestamp_tracker_us: i64,
    /// `timestamp_tracker_us` on the host clock ([`crate::host_clock_us`],
    /// the clock of `tobii_system_clock`), microseconds: as in the Stream
    /// Engine, not the time the sample was received.
    pub timestamp_system_us: i64,
    /// The user's left eye.
    pub left: GazeDataEye,
    /// The user's right eye.
    pub right: GazeDataEye,
}

/// One eye's half of a [`GazeRaw`] sample: its block of `tobii_gaze_raw_t`,
/// the device's values as sent (keys `0x02`..`0x07` for the left eye,
/// `0x08`..`0x0d` for the right). No field has a validity of its own: a key
/// the frame lacks leaves its field 0, as in the Stream Engine's zeroed
/// record.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GazeRawEye {
    /// Gaze origin (the cornea centre) in the tracker frame, mm.
    pub gaze_origin_mm: [f32; 3],
    /// Gaze origin normalised to the track box, `0..1` per axis.
    pub gaze_origin_in_track_box: [f32; 3],
    /// Gaze point on the screen plane, in the tracker frame, mm.
    pub gaze_point_mm: [f32; 3],
    /// Gaze point in normalised display coordinates, unclamped.
    pub gaze_point_on_display: [f32; 2],
    /// Pupil diameter, mm.
    pub pupil_diameter_mm: f32,
    /// Tracking status as sent: the ET5 sends 0 for a tracked eye and 4 for
    /// a lost one. The Stream Engine's gaze data takes this eye's values as
    /// usable while it is below 2.
    pub status: u32,
}

/// The raw gaze sample: `tobii_gaze_raw_t`, the Stream Engine's own record of
/// one gaze frame, which its raw gaze callback receives. Values pass as the
/// device sent them, filtered by nothing. A key the frame lacks is `None`
/// where the record flags it, and 0 elsewhere; `Some` says only that the
/// frame had the key, not that its value is usable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GazeRaw {
    /// When the frame was taken, on the **device** clock, microseconds, as
    /// the Stream Engine hands its raw gaze out: the clock of
    /// [`GazeData::timestamp_tracker_us`], not the host clock of the other
    /// stamps, and with no host time beside it. Nothing keeps it from going
    /// back: a re-open of the device (a cold start's prime re-open, a
    /// reconnect) may restart the device clock.
    pub timestamp_tracker_us: i64,
    /// The user's left eye.
    pub left: GazeRawEye,
    /// The user's right eye.
    pub right: GazeRawEye,
    /// Combined gaze point in normalised display coordinates, unclamped
    /// (key `0x1c`).
    pub combined_gaze_point_on_display: [f32; 2],
    /// Combined gaze validity as sent (key `0x1b`): 1 when the point is
    /// valid.
    pub combined_gaze_validity: u32,
    /// Key `0x0e`, of unknown meaning; the ET5 never sends it.
    pub key_0e: Option<u32>,
    /// Key `0x11`, of unknown meaning; the ET5 sends 4.
    pub key_11: Option<u32>,
    /// The device's frame counter (key `0x14`).
    pub frame_counter: Option<u32>,
    /// The left eye's gaze-origin validity as sent (key `0x16`): 1 when
    /// valid.
    pub left_origin_flag: Option<u32>,
    /// The right eye's gaze-origin validity as sent (key `0x15`).
    pub right_origin_flag: Option<u32>,
    /// The left eyeball's centre in the tracker frame, mm (key `0x17`).
    pub left_eyeball_center_mm: Option<[f32; 3]>,
    /// The right eyeball's centre in the tracker frame, mm (key `0x18`).
    pub right_eyeball_center_mm: Option<[f32; 3]>,
}

/// One IR camera frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Image {
    /// When the frame was taken, on the host clock, microseconds.
    pub ts_us: i64,
    /// Width, pixels.
    pub width: u32,
    /// Height, pixels.
    pub height: u32,
    /// Bits per pixel (8 on the ET5).
    pub bits_per_pixel: u8,
    /// Row-major pixels, `width * height * bits_per_pixel / 8` bytes.
    pub pixels: Vec<u8>,
}

/// `tobii_notification_type_t`: what a [`Notification`] reports.
pub mod notification {
    /// Calibration started or stopped ([`super::NotificationValue::State`]).
    pub const CALIBRATION_STATE_CHANGED: u8 = 0;
    /// Exclusive mode changed.
    pub const EXCLUSIVE_MODE_STATE_CHANGED: u8 = 1;
    /// The track box changed.
    pub const TRACK_BOX_CHANGED: u8 = 2;
    /// The display area changed ([`super::NotificationValue::DisplayArea`]).
    pub const DISPLAY_AREA_CHANGED: u8 = 3;
    /// The output frequency changed ([`super::NotificationValue::Float`]).
    pub const FRAMERATE_CHANGED: u8 = 4;
    /// Power save was entered or left.
    pub const POWER_SAVE_STATE_CHANGED: u8 = 5;
    /// The device was paused or resumed.
    pub const DEVICE_PAUSED_STATE_CHANGED: u8 = 6;
    /// The calibrated eye selection changed.
    pub const CALIBRATION_ENABLED_EYE_CHANGED: u8 = 7;
    /// A new calibration is active ([`super::NotificationValue::Uint`] id).
    pub const CALIBRATION_ID_CHANGED: u8 = 8;
    /// The combined-gaze eye selection changed.
    pub const COMBINED_GAZE_EYE_SELECTION_CHANGED: u8 = 9;
    /// The fault list changed ([`super::NotificationValue::String`], the new
    /// list).
    pub const FAULTS_CHANGED: u8 = 10;
    /// The warning list changed ([`super::NotificationValue::String`], the
    /// new list).
    pub const WARNINGS_CHANGED: u8 = 11;
    /// The face type changed.
    pub const FACE_TYPE_CHANGED: u8 = 12;
}

/// The value a [`Notification`] carries (`tobii_notification_value_type_t`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NotificationValue {
    /// No value.
    None,
    /// A float, e.g. a frequency.
    Float(f32),
    /// An on/off state.
    State(bool),
    /// A display area.
    DisplayArea(DisplayArea),
    /// An unsigned integer, e.g. a calibration id.
    Uint(u32),
    /// A `tobii_enabled_eye_t`.
    EnabledEye(u8),
    /// A string (at most 511 bytes reach C).
    String(String),
}

/// A device notification: `tobii_notification_t`.
#[derive(Debug, Clone, PartialEq)]
pub struct Notification {
    /// One of the [`notification`] constants.
    pub kind: u8,
    /// The value.
    pub value: NotificationValue,
}

/// A decoded daemon -> client frame.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ServerMsg {
    /// Reply to a SUBSCRIBE frame.
    Subscribed {
        /// `true` if the subscription was accepted; `false` if the daemon is busy.
        ok: bool,
    },
    /// Reply to a REQUEST frame.
    Reply {
        /// The id the request carried.
        request_id: u32,
        /// A Stream Engine `tobii_error_t` value; `0` is success.
        status: u8,
        /// Kind-specific payload (see [`crate::request`]).
        payload: Vec<u8>,
    },
    /// Head pose.
    Head {
        /// When the sample was taken, on the host clock, microseconds.
        ts_us: i64,
        /// Translation `[x, y, z]` in millimetres.
        pos_mm: [f32; 3],
        /// Rotation `[pitch, yaw, roll]` in radians (Stream-Engine axis order).
        rot_rad: [f32; 3],
    },
    /// Gaze point.
    Gaze {
        /// When the sample was taken, on the host clock, microseconds.
        ts_us: i64,
        /// Whether `xy` holds a usable gaze point.
        valid: bool,
        /// Gaze point in normalised display coordinates: nominally `0..1`
        /// per axis, unclamped (the Stream Engine passes looks off-screen
        /// through).
        xy: [f32; 2],
        /// Pupil diameter `[left, right]` in millimetres; `NaN` for an eye
        /// whose diameter is not valid, and for both when the daemon did not
        /// send the optional tail.
        pupil_mm: [f32; 2],
    },
    /// User presence.
    Presence {
        /// When the sample was taken, on the host clock, microseconds.
        ts_us: i64,
        /// One of [`crate::PRESENCE_AWAY`] / [`crate::PRESENCE_PRESENT`].
        status: u8,
    },
    /// Gaze origins in the display frame, mm (`tobii_gaze_origin_t`).
    GazeOrigin(EyePair),
    /// Eye positions normalised to the track box (serves both
    /// `tobii_eye_position_normalized_t` and `tobii_user_position_guide_t`).
    EyePosition(EyePair),
    /// Per-eye gaze data (`tobii_gaze_data_t`).
    GazeData(Box<GazeData>),
    /// The Stream Engine's raw record of a gaze frame (`tobii_gaze_raw_t`),
    /// on the device clock.
    GazeRaw(Box<GazeRaw>),
    /// An IR camera frame.
    Image(Box<Image>),
    /// A device notification.
    Notification(Notification),
}

/// Daemon -> client SUBSCRIBED reply body.
#[must_use]
pub fn encode_subscribed(ok: bool) -> Vec<u8> {
    vec![TAG_SUBSCRIBED, u8::from(ok)]
}

/// Daemon -> client REPLY body.
#[must_use]
pub fn encode_reply(request_id: u32, status: u8, payload: &[u8]) -> Vec<u8> {
    Writer::with_tag(TAG_REPLY, 5 + payload.len())
        .u32(request_id)
        .u8(status)
        .bytes(payload)
        .finish()
}

/// Daemon -> client HEAD body: position in mm, rotation in radians.
#[must_use]
pub fn encode_head(ts_us: i64, pos_mm: [f32; 3], rot_rad: [f32; 3]) -> Vec<u8> {
    Writer::with_tag(TAG_HEAD, 32)
        .i64(ts_us)
        .f32s(&pos_mm)
        .f32s(&rot_rad)
        .finish()
}

/// Daemon -> client GAZE body, including the pupil tail.
#[must_use]
pub fn encode_gaze(ts_us: i64, valid: bool, xy: [f32; 2], pupil_mm: [f32; 2]) -> Vec<u8> {
    Writer::with_tag(TAG_GAZE, 25)
        .i64(ts_us)
        .bool(valid)
        .f32s(&xy)
        .f32s(&pupil_mm)
        .finish()
}

/// Daemon -> client PRESENCE body (`status` is a `PRESENCE_*` value).
#[must_use]
pub fn encode_presence(ts_us: i64, status: u8) -> Vec<u8> {
    Writer::with_tag(TAG_PRESENCE, 9)
        .i64(ts_us)
        .u8(status)
        .finish()
}

fn encode_pair(tag: u8, pair: &EyePair) -> Vec<u8> {
    Writer::with_tag(tag, 34)
        .i64(pair.ts_us)
        .bool(pair.left.valid)
        .f32s(&pair.left.xyz)
        .bool(pair.right.valid)
        .f32s(&pair.right.xyz)
        .finish()
}

/// Daemon -> client `GAZE_ORIGIN` body.
#[must_use]
pub fn encode_gaze_origin(pair: &EyePair) -> Vec<u8> {
    encode_pair(TAG_GAZE_ORIGIN, pair)
}

/// Daemon -> client `EYE_POSITION` body.
#[must_use]
pub fn encode_eye_position(pair: &EyePair) -> Vec<u8> {
    encode_pair(TAG_EYE_POSITION, pair)
}

fn write_gaze_data_eye(w: &mut Writer, eye: &GazeDataEye) {
    w.bool(eye.gaze_origin_valid)
        .f32s(&eye.gaze_origin_mm)
        .f32s(&eye.gaze_origin_in_track_box)
        .bool(eye.gaze_point_valid)
        .f32s(&eye.gaze_point_mm)
        .f32s(&eye.gaze_point_on_display)
        .bool(eye.eyeball_center_valid)
        .f32s(&eye.eyeball_center_mm)
        .bool(eye.pupil_valid)
        .f32(eye.pupil_diameter_mm);
}

/// Daemon -> client `GAZE_DATA` body.
#[must_use]
pub fn encode_gaze_data(data: &GazeData) -> Vec<u8> {
    let mut w = Writer::with_tag(TAG_GAZE_DATA, 16 + 2 * 60);
    w.i64(data.timestamp_tracker_us)
        .i64(data.timestamp_system_us);
    write_gaze_data_eye(&mut w, &data.left);
    write_gaze_data_eye(&mut w, &data.right);
    w.finish()
}

fn write_gaze_raw_eye(w: &mut Writer, eye: &GazeRawEye) {
    w.f32s(&eye.gaze_origin_mm)
        .f32s(&eye.gaze_origin_in_track_box)
        .f32s(&eye.gaze_point_mm)
        .f32s(&eye.gaze_point_on_display)
        .f32(eye.pupil_diameter_mm)
        .u32(eye.status);
}

/// Daemon -> client `GAZE_RAW` body: `i64` device time; per eye `12 x f32`
/// and a `u32` status; the combined gaze `2 x f32` and its `u32` validity;
/// then each field the frame may lack as `u8 present` and its value, zero
/// when absent: five `u32`s, then the two eyeball centres, `3 x f32` each.
/// 175 bytes after the tag; a later daemon may append a tail, which this
/// decoder ignores.
#[must_use]
pub fn encode_gaze_raw(raw: &GazeRaw) -> Vec<u8> {
    let mut w = Writer::with_tag(TAG_GAZE_RAW, 8 + 2 * 52 + 12 + 5 * 5 + 2 * 13);
    w.i64(raw.timestamp_tracker_us);
    write_gaze_raw_eye(&mut w, &raw.left);
    write_gaze_raw_eye(&mut w, &raw.right);
    w.f32s(&raw.combined_gaze_point_on_display)
        .u32(raw.combined_gaze_validity);
    for word in [
        raw.key_0e,
        raw.key_11,
        raw.frame_counter,
        raw.left_origin_flag,
        raw.right_origin_flag,
    ] {
        w.bool(word.is_some()).u32(word.unwrap_or(0));
    }
    for point in [raw.left_eyeball_center_mm, raw.right_eyeball_center_mm] {
        w.bool(point.is_some()).f32s(&point.unwrap_or_default());
    }
    w.finish()
}

/// Daemon -> client IMAGE body. Takes the pixels by reference so one encode
/// can be shared by every subscriber.
#[must_use]
pub fn encode_image(
    ts_us: i64,
    width: u32,
    height: u32,
    bits_per_pixel: u8,
    pixels: &[u8],
) -> Vec<u8> {
    Writer::with_tag(TAG_IMAGE, 17 + pixels.len())
        .i64(ts_us)
        .u32(width)
        .u32(height)
        .u8(bits_per_pixel)
        .bytes(pixels)
        .finish()
}

/// `tobii_notification_value_type_t` numbering.
const VALUE_NONE: u8 = 0;
const VALUE_FLOAT: u8 = 1;
const VALUE_STATE: u8 = 2;
const VALUE_DISPLAY_AREA: u8 = 3;
const VALUE_UINT: u8 = 4;
const VALUE_ENABLED_EYE: u8 = 5;
const VALUE_STRING: u8 = 6;

/// Daemon -> client NOTIFICATION body.
#[must_use]
pub fn encode_notification(n: &Notification) -> Vec<u8> {
    let mut w = Writer::with_tag(TAG_NOTIFICATION, 40);
    w.u8(n.kind);
    match &n.value {
        NotificationValue::None => {
            w.u8(VALUE_NONE);
        }
        NotificationValue::Float(v) => {
            w.u8(VALUE_FLOAT).f32(*v);
        }
        NotificationValue::State(v) => {
            w.u8(VALUE_STATE).bool(*v);
        }
        NotificationValue::DisplayArea(a) => {
            w.u8(VALUE_DISPLAY_AREA)
                .f64_as_f32(&a.top_left_mm)
                .f64_as_f32(&a.top_right_mm)
                .f64_as_f32(&a.bottom_left_mm);
        }
        NotificationValue::Uint(v) => {
            w.u8(VALUE_UINT).u32(*v);
        }
        NotificationValue::EnabledEye(v) => {
            w.u8(VALUE_ENABLED_EYE).u8(*v);
        }
        NotificationValue::String(s) => {
            w.u8(VALUE_STRING).str(s);
        }
    }
    w.finish()
}

fn read_pair(r: &mut Reader<'_>) -> Option<EyePair> {
    Some(EyePair {
        ts_us: r.i64()?,
        left: EyePoint {
            valid: r.bool()?,
            xyz: r.f32s()?,
        },
        right: EyePoint {
            valid: r.bool()?,
            xyz: r.f32s()?,
        },
    })
}

fn read_gaze_data_eye(r: &mut Reader<'_>) -> Option<GazeDataEye> {
    Some(GazeDataEye {
        gaze_origin_valid: r.bool()?,
        gaze_origin_mm: r.f32s()?,
        gaze_origin_in_track_box: r.f32s()?,
        gaze_point_valid: r.bool()?,
        gaze_point_mm: r.f32s()?,
        gaze_point_on_display: r.f32s()?,
        eyeball_center_valid: r.bool()?,
        eyeball_center_mm: r.f32s()?,
        pupil_valid: r.bool()?,
        pupil_diameter_mm: r.f32()?,
    })
}

fn read_gaze_raw_eye(r: &mut Reader<'_>) -> Option<GazeRawEye> {
    Some(GazeRawEye {
        gaze_origin_mm: r.f32s()?,
        gaze_origin_in_track_box: r.f32s()?,
        gaze_point_mm: r.f32s()?,
        gaze_point_on_display: r.f32s()?,
        pupil_diameter_mm: r.f32()?,
        status: r.u32()?,
    })
}

/// A `u32` the frame may lack: `u8 present`, then the value.
fn read_present_u32(r: &mut Reader<'_>) -> Option<Option<u32>> {
    let present = r.bool()?;
    let value = r.u32()?;
    Some(present.then_some(value))
}

/// A point the frame may lack: `u8 present`, then `3 x f32`.
fn read_present_point(r: &mut Reader<'_>) -> Option<Option<[f32; 3]>> {
    let present = r.bool()?;
    let value = r.f32s()?;
    Some(present.then_some(value))
}

fn read_gaze_raw(r: &mut Reader<'_>) -> Option<GazeRaw> {
    Some(GazeRaw {
        timestamp_tracker_us: r.i64()?,
        left: read_gaze_raw_eye(r)?,
        right: read_gaze_raw_eye(r)?,
        combined_gaze_point_on_display: r.f32s()?,
        combined_gaze_validity: r.u32()?,
        key_0e: read_present_u32(r)?,
        key_11: read_present_u32(r)?,
        frame_counter: read_present_u32(r)?,
        left_origin_flag: read_present_u32(r)?,
        right_origin_flag: read_present_u32(r)?,
        left_eyeball_center_mm: read_present_point(r)?,
        right_eyeball_center_mm: read_present_point(r)?,
    })
}

fn read_notification(r: &mut Reader<'_>) -> Option<Notification> {
    let kind = r.u8()?;
    let value = match r.u8()? {
        VALUE_NONE => NotificationValue::None,
        VALUE_FLOAT => NotificationValue::Float(r.f32()?),
        VALUE_STATE => NotificationValue::State(r.bool()?),
        VALUE_DISPLAY_AREA => NotificationValue::DisplayArea(DisplayArea {
            top_left_mm: r.f64s()?,
            top_right_mm: r.f64s()?,
            bottom_left_mm: r.f64s()?,
        }),
        VALUE_UINT => NotificationValue::Uint(r.u32()?),
        VALUE_ENABLED_EYE => NotificationValue::EnabledEye(r.u8()?),
        VALUE_STRING => NotificationValue::String(r.str()?),
        _ => return None,
    };
    Some(Notification { kind, value })
}

/// Decode a daemon -> client frame body. `None` for an unknown tag or a body
/// too short for its tag, so an older client ignores frames it does not know.
#[must_use]
pub fn decode_server(body: &[u8]) -> Option<ServerMsg> {
    let mut r = Reader::new(body);
    match r.u8()? {
        TAG_SUBSCRIBED => Some(ServerMsg::Subscribed { ok: r.bool()? }),
        TAG_REPLY => Some(ServerMsg::Reply {
            request_id: r.u32()?,
            status: r.u8()?,
            payload: r.rest().to_vec(),
        }),
        TAG_HEAD => Some(ServerMsg::Head {
            ts_us: r.i64()?,
            pos_mm: r.f32s()?,
            rot_rad: r.f32s()?,
        }),
        TAG_GAZE => {
            let ts_us = r.i64()?;
            let valid = r.bool()?;
            let xy = r.f32s()?;
            // Pupil is an optional tail: older daemons omit it (18-byte frame).
            let pupil_mm = r.f32s().unwrap_or([f32::NAN; 2]);
            Some(ServerMsg::Gaze {
                ts_us,
                valid,
                xy,
                pupil_mm,
            })
        }
        TAG_PRESENCE => Some(ServerMsg::Presence {
            ts_us: r.i64()?,
            status: r.u8()?,
        }),
        TAG_GAZE_ORIGIN => read_pair(&mut r).map(ServerMsg::GazeOrigin),
        TAG_EYE_POSITION => read_pair(&mut r).map(ServerMsg::EyePosition),
        TAG_GAZE_DATA => Some(ServerMsg::GazeData(Box::new(GazeData {
            timestamp_tracker_us: r.i64()?,
            timestamp_system_us: r.i64()?,
            left: read_gaze_data_eye(&mut r)?,
            right: read_gaze_data_eye(&mut r)?,
        }))),
        TAG_IMAGE => {
            let ts_us = r.i64()?;
            let width = r.u32()?;
            let height = r.u32()?;
            let bits_per_pixel = r.u8()?;
            let pixels = r.rest().to_vec();
            Some(ServerMsg::Image(Box::new(Image {
                ts_us,
                width,
                height,
                bits_per_pixel,
                pixels,
            })))
        }
        TAG_NOTIFICATION => read_notification(&mut r).map(ServerMsg::Notification),
        TAG_GAZE_RAW => read_gaze_raw(&mut r).map(|raw| ServerMsg::GazeRaw(Box::new(raw))),
        _ => None,
    }
}
