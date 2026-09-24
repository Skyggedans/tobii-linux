//! Client -> daemon requests and the payloads of their replies.
//!
//! A request is `TAG_REQUEST`, `u32 request_id`, `u8 kind`, payload; the
//! daemon answers with exactly one `TAG_REPLY` carrying the same id, a
//! Stream Engine status code and a kind-specific payload. Samples may arrive
//! between the two, so a client must match replies by id.

use crate::TAG_REQUEST;
use crate::geometry::{DisplayArea, GeometryMounting, TrackBox};
use crate::wire::{Reader, Writer};

/// Request kinds.
pub mod kind {
    /// Device identity: reply is a [`super::DeviceInfo`].
    pub const DEVICE_INFO: u8 = 1;
    /// Track box: reply is a [`super::TrackBox`](crate::geometry::TrackBox).
    pub const TRACK_BOX: u8 = 2;
    /// Current display area: reply is a display area.
    pub const DISPLAY_AREA_GET: u8 = 3;
    /// Set the display area: payload is a display area, reply is empty.
    pub const DISPLAY_AREA_SET: u8 = 4;
    /// Mounting geometry: reply is a geometry mounting.
    pub const GEOMETRY_MOUNTING: u8 = 5;
    /// A `tobii_state_t` value: payload `u32` state id (see [`super::state`]).
    pub const STATE: u8 = 6;
    /// Device/host clock pair: reply is a [`super::Timesync`].
    pub const TIMESYNC: u8 = 7;
    /// The tracker's stream catalogue: reply is a list of
    /// [`super::StreamType`] (see [`super::encode_stream_types`]).
    pub const STREAM_TYPES: u8 = 8;
    /// Pause (payload `u8 1`) or resume (`u8 0`) the device; reply is empty.
    /// One state for every client: the last request wins, any client may
    /// resume, and the device resumes when the client that paused it goes
    /// away or the device is re-initialised.
    pub const DEVICE_PAUSE: u8 = 0x0a;
    /// The device's name: reply is its bytes, at most
    /// [`super::DEVICE_NAME_MAX`] and no NUL. The name a client set, else
    /// the model.
    pub const DEVICE_NAME_GET: u8 = 0x0b;
    /// Name the device: payload is the name's bytes, reply is empty. The
    /// daemon keeps what comes before the first NUL, at most
    /// [`super::DEVICE_NAME_MAX`] bytes, and saves it for later sessions.
    pub const DEVICE_NAME_SET: u8 = 0x0c;
    /// Start a calibration session: payload `u8 tobii_enabled_eye_t`.
    pub const CALIBRATION_START: u8 = 0x10;
    /// End the calibration session: payload [`super::STOP_KEEP`] (keep
    /// what it computed) or [`super::STOP_DISCARD`] (put back what was
    /// there before it).
    pub const CALIBRATION_STOP: u8 = 0x11;
    /// Collect a 2-D point: payload `f32 x, f32 y` (normalised display).
    pub const CALIBRATION_COLLECT_2D: u8 = 0x12;
    /// Discard the data collected at a 2-D point in this session: payload
    /// `f32 x, f32 y`, as collected.
    pub const CALIBRATION_DISCARD_2D: u8 = 0x13;
    /// Compute and apply: reply `u32` new calibration id.
    pub const CALIBRATION_COMPUTE: u8 = 0x14;
    /// Read the active calibration: reply is the blob.
    pub const CALIBRATION_RETRIEVE: u8 = 0x15;
    /// Apply a calibration blob (empty payload: revert to the built-in one).
    pub const CALIBRATION_APPLY: u8 = 0x16;
    /// Clear the points collected in this session.
    pub const CALIBRATION_CLEAR: u8 = 0x17;
}

/// [`kind::CALIBRATION_STOP`] payload: keep the session's calibration (and
/// the display area it was made on). What `tobii_calibration_stop` sends.
pub const STOP_KEEP: &[u8] = &[];
/// [`kind::CALIBRATION_STOP`] payload: discard the session, putting back the
/// calibration and display area it started from.
pub const STOP_DISCARD: &[u8] = &[1];

/// The longest device name, in bytes: a `tobii_device_name_t` less its NUL.
/// Names are raw bytes, as the Stream Engine passes them, and need not be
/// UTF-8.
pub const DEVICE_NAME_MAX: usize = 63;

/// `tobii_state_t` ids understood by [`kind::STATE`].
pub mod state {
    /// Reply `u8`: whether the device is paused.
    pub const DEVICE_PAUSED: u32 = 2;
    /// Reply `u32`: the active calibration id.
    pub const CALIBRATION_ID: u32 = 6;
    /// Reply `u8`: whether a calibration session is running.
    pub const CALIBRATION_ACTIVE: u32 = 7;
}

/// Reply status codes: the Stream Engine's own `tobii_error_t` numbering.
pub mod status {
    /// Success.
    pub const OK: u8 = 0;
    /// Not supported by this device or daemon.
    pub const NOT_SUPPORTED: u8 = 3;
    /// Not available yet (e.g. no sample seen).
    pub const NOT_AVAILABLE: u8 = 4;
    /// The device connection failed.
    pub const CONNECTION_FAILED: u8 = 5;
    /// The device did not answer in time.
    pub const TIMED_OUT: u8 = 6;
    /// The request payload was malformed.
    pub const INVALID_PARAMETER: u8 = 8;
    /// A calibration session is already running for this client.
    pub const CALIBRATION_ALREADY_STARTED: u8 = 9;
    /// No calibration session is running for this client.
    pub const CALIBRATION_NOT_STARTED: u8 = 10;
    /// The device reported a failure.
    pub const OPERATION_FAILED: u8 = 13;
    /// Another client is calibrating.
    pub const CALIBRATION_BUSY: u8 = 15;
}

/// A decoded request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request<'a> {
    /// Echoed in the reply.
    pub id: u32,
    /// One of the [`kind`] constants.
    pub kind: u8,
    /// Kind-specific payload.
    pub payload: &'a [u8],
}

/// Client -> daemon REQUEST body.
#[must_use]
pub fn encode_request(id: u32, kind: u8, payload: &[u8]) -> Vec<u8> {
    Writer::with_tag(TAG_REQUEST, 5 + payload.len())
        .u32(id)
        .u8(kind)
        .bytes(payload)
        .finish()
}

/// Decode a client REQUEST body.
#[must_use]
pub fn decode_request(body: &[u8]) -> Option<Request<'_>> {
    let mut r = Reader::new(body);
    if r.u8()? != TAG_REQUEST {
        return None;
    }
    Some(Request {
        id: r.u32()?,
        kind: r.u8()?,
        payload: r.rest(),
    })
}

/// Device identity, the four strings the tracker reports about itself.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceInfo {
    /// Serial number.
    pub serial_number: String,
    /// Model name.
    pub model: String,
    /// Hardware generation.
    pub generation: String,
    /// Firmware version.
    pub firmware_version: String,
}

/// Reply payload of [`kind::DEVICE_INFO`].
#[must_use]
pub fn encode_device_info(info: &DeviceInfo) -> Vec<u8> {
    Writer::default()
        .str(&info.serial_number)
        .str(&info.model)
        .str(&info.generation)
        .str(&info.firmware_version)
        .finish()
}

/// Decode a [`kind::DEVICE_INFO`] reply payload.
#[must_use]
pub fn decode_device_info(payload: &[u8]) -> Option<DeviceInfo> {
    let mut r = Reader::new(payload);
    Some(DeviceInfo {
        serial_number: r.str()?,
        model: r.str()?,
        generation: r.str()?,
        firmware_version: r.str()?,
    })
}

/// Reply payload of [`kind::TRACK_BOX`]: 24 `f32` mm, corners in order.
#[must_use]
pub fn encode_track_box(track_box: &TrackBox) -> Vec<u8> {
    let mut w = Writer::default();
    for corner in &track_box.corners_mm {
        w.f64_as_f32(corner);
    }
    w.finish()
}

/// Decode a [`kind::TRACK_BOX`] reply payload.
#[must_use]
pub fn decode_track_box(payload: &[u8]) -> Option<TrackBox> {
    let mut r = Reader::new(payload);
    let mut corners_mm = [[0.0; 3]; 8];
    for corner in &mut corners_mm {
        *corner = r.f64s()?;
    }
    Some(TrackBox { corners_mm })
}

/// Payload of [`kind::DISPLAY_AREA_SET`] and reply of
/// [`kind::DISPLAY_AREA_GET`]: 9 `f32` mm, top-left, top-right, bottom-left.
#[must_use]
pub fn encode_display_area(area: &DisplayArea) -> Vec<u8> {
    Writer::default()
        .f64_as_f32(&area.top_left_mm)
        .f64_as_f32(&area.top_right_mm)
        .f64_as_f32(&area.bottom_left_mm)
        .finish()
}

/// Decode a display area payload.
#[must_use]
pub fn decode_display_area(payload: &[u8]) -> Option<DisplayArea> {
    let mut r = Reader::new(payload);
    Some(DisplayArea {
        top_left_mm: r.f64s()?,
        top_right_mm: r.f64s()?,
        bottom_left_mm: r.f64s()?,
    })
}

/// Reply payload of [`kind::GEOMETRY_MOUNTING`].
#[must_use]
pub fn encode_geometry_mounting(m: &GeometryMounting) -> Vec<u8> {
    Writer::default()
        .i32(m.guides)
        .f64_as_f32(&[m.width_mm, m.angle_deg])
        .f64_as_f32(&m.external_offset_mm)
        .f64_as_f32(&m.internal_offset_mm)
        .finish()
}

/// Decode a [`kind::GEOMETRY_MOUNTING`] reply payload.
#[must_use]
pub fn decode_geometry_mounting(payload: &[u8]) -> Option<GeometryMounting> {
    let mut r = Reader::new(payload);
    let guides = r.i32()?;
    let [width_mm, angle_deg] = r.f64s()?;
    Some(GeometryMounting {
        guides,
        width_mm,
        angle_deg,
        external_offset_mm: r.f64s()?,
        internal_offset_mm: r.f64s()?,
    })
}

/// A device timestamp bracketed by two host timestamps: the device clock read
/// `device_us` at some host time between `host_start_us` and `host_end_us`
/// (microseconds, the host clock of `tobii_system_clock`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Timesync {
    /// Host time before the device timestamp was taken.
    pub host_start_us: i64,
    /// The device timestamp.
    pub device_us: i64,
    /// Host time after the device timestamp was taken.
    pub host_end_us: i64,
}

/// Reply payload of [`kind::TIMESYNC`].
#[must_use]
pub fn encode_timesync(t: &Timesync) -> Vec<u8> {
    Writer::default()
        .i64(t.host_start_us)
        .i64(t.device_us)
        .i64(t.host_end_us)
        .finish()
}

/// Decode a [`kind::TIMESYNC`] reply payload.
#[must_use]
pub fn decode_timesync(payload: &[u8]) -> Option<Timesync> {
    let mut r = Reader::new(payload);
    Some(Timesync {
        host_start_us: r.i64()?,
        device_us: r.i64()?,
        host_end_us: r.i64()?,
    })
}

/// One entry of the tracker's stream catalogue (command 1200), as the device
/// reports it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamType {
    /// The device's stream id (0x500 gaze, 0x501 image, ...).
    pub id: u32,
    /// Stream name.
    pub name: String,
    /// A second string, empty on the ET5.
    pub text: String,
    /// A number whose meaning is unknown (1000 for `image_collection`, 0
    /// for the rest on the ET5).
    pub value: u32,
}

/// The smallest encoded [`StreamType`]: id, two empty strings, value.
const STREAM_TYPE_MIN_LEN: usize = 4 + 2 + 2 + 4;

/// Reply payload of [`kind::STREAM_TYPES`]: `u32` count, then per entry
/// `u32 id`, name, text, `u32 value`, in the device's order.
#[must_use]
pub fn encode_stream_types(types: &[StreamType]) -> Vec<u8> {
    let count = u32::try_from(types.len()).unwrap_or(u32::MAX);
    let mut w = Writer::default();
    w.u32(count);
    // Never more entries than the count says (only past `u32::MAX` would
    // that drop any).
    for t in types.iter().take(count as usize) {
        w.u32(t.id).str(&t.name).str(&t.text).u32(t.value);
    }
    w.finish()
}

/// Decode a [`kind::STREAM_TYPES`] reply payload. A count the payload is too
/// short to hold is `None` before anything is allocated.
#[must_use]
pub fn decode_stream_types(payload: &[u8]) -> Option<Vec<StreamType>> {
    let mut r = Reader::new(payload);
    let count = usize::try_from(r.u32()?).ok()?;
    if count > payload.len().saturating_sub(4) / STREAM_TYPE_MIN_LEN {
        return None;
    }
    let mut types = Vec::new();
    for _ in 0..count {
        types.push(StreamType {
            id: r.u32()?,
            name: r.str()?,
            text: r.str()?,
            value: r.u32()?,
        });
    }
    Some(types)
}

/// A single `u32` payload (state id, state value, calibration id).
#[must_use]
pub fn encode_u32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Decode a single-`u32` payload.
#[must_use]
pub fn decode_u32(payload: &[u8]) -> Option<u32> {
    Reader::new(payload).u32()
}

/// A normalised display point: the payload of
/// [`kind::CALIBRATION_COLLECT_2D`] and [`kind::CALIBRATION_DISCARD_2D`].
#[must_use]
pub fn encode_point_2d(x: f32, y: f32) -> Vec<u8> {
    Writer::default().f32(x).f32(y).finish()
}

/// Decode a normalised display point payload.
#[must_use]
pub fn decode_point_2d(payload: &[u8]) -> Option<(f32, f32)> {
    let mut r = Reader::new(payload);
    Some((r.f32()?, r.f32()?))
}
