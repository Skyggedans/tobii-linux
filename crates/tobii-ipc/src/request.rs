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
    /// Device identity: reply is a [`super::DeviceInfo`] (see
    /// [`super::encode_device_info`]).
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
    /// The tracker's hardware configuration (command 2120): reply is a
    /// [`super::HardwareConfiguration`]. Provisional: the layout is inferred
    /// from one Windows capture, and the ET5 has never sent one to Linux.
    pub const HARDWARE_CONFIGURATION: u8 = 9;
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
    /// Reply: the tracker's fault list, its bytes with no NUL, as status
    /// string 5 of the last init's command 1490 gave it ("ok" when there are
    /// none). `NOT_SUPPORTED` when that init reported none.
    pub const FAULT: u32 = 4;
    /// Reply: the tracker's warning list, as [`FAULT`] (status string 6).
    pub const WARNING: u32 = 5;
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

/// Device identity, the strings the tracker reports about itself.
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
    /// Integration type (`Peripheral` on the ET5); empty from a daemon that
    /// predates it.
    pub integration_type: String,
}

/// Reply payload of [`kind::DEVICE_INFO`]: 4 x (`u16` length + UTF-8), the
/// serial, model, generation and firmware, then the integration type the
/// same way. That fifth string is an optional tail: a daemon from before it
/// omits it, and an older client reads four strings and ignores the rest.
#[must_use]
pub fn encode_device_info(info: &DeviceInfo) -> Vec<u8> {
    Writer::default()
        .str(&info.serial_number)
        .str(&info.model)
        .str(&info.generation)
        .str(&info.firmware_version)
        .str(&info.integration_type)
        .finish()
}

/// Decode a [`kind::DEVICE_INFO`] reply payload. The integration type is
/// empty when the tail is missing (an older daemon) or cut short.
#[must_use]
pub fn decode_device_info(payload: &[u8]) -> Option<DeviceInfo> {
    let mut r = Reader::new(payload);
    Some(DeviceInfo {
        serial_number: r.str()?,
        model: r.str()?,
        generation: r.str()?,
        firmware_version: r.str()?,
        integration_type: r.str().unwrap_or_default(),
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
/// (microseconds, [`crate::host_clock_us`], the clock of
/// `tobii_system_clock`).
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

/// Most entries a [`HardwareConfiguration`] carries: the slots of
/// `tobii_hardware_configuration_t`.
pub const HARDWARE_ENTRIES_MAX: usize = 2;
/// Most points a [`HardwareConfiguration`] carries.
pub const HARDWARE_POINTS_MAX: usize = 40;
/// Most coefficients a [`HardwareEntry`] carries.
pub const HARDWARE_COEFFICIENTS_MAX: usize = 64;

/// One entry of the tracker's hardware configuration (command 2120). What
/// each field means is not known; the names are neutral. The 32.32 values
/// are scaled as the tracker's lengths are (1/1024 mm to mm).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HardwareEntry {
    /// An id (0 in both entries the ET5 sent to Windows).
    pub id: u32,
    /// A 16.16 value, unscaled (16 and 62 on the ET5).
    pub param_a: f64,
    /// A 16.16 value, unscaled (100 on the ET5).
    pub param_b: f64,
    /// A 3-D point, mm.
    pub position_mm: [f64; 3],
    /// Fifteen 32.32 values, scaled to mm.
    pub values: [f64; 15],
    /// A count (2240 on the ET5).
    pub width: u32,
    /// A count (2240 on the ET5).
    pub height: u32,
    /// A number (0 on the ET5).
    pub param_c: u32,
    /// A list of 32.32 values, scaled to mm; at most
    /// [`HARDWARE_COEFFICIENTS_MAX`] (empty on the ET5).
    pub coefficients: Vec<f64>,
    /// A 3-D point, mm.
    pub point_a_mm: [f64; 3],
    /// A 3-D point, mm.
    pub point_b_mm: [f64; 3],
    /// A 32.32 value, scaled to mm.
    pub param_d: f64,
}

/// The tracker's hardware configuration (command 2120), the data behind
/// `tobii_hardware_configuration_get`. Provisional: see
/// [`kind::HARDWARE_CONFIGURATION`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HardwareConfiguration {
    /// At most [`HARDWARE_ENTRIES_MAX`] entries.
    pub entries: Vec<HardwareEntry>,
    /// At most [`HARDWARE_POINTS_MAX`] 3-D points, mm.
    pub points_mm: Vec<[f64; 3]>,
    /// The tracker's mode word (1 on the ET5).
    pub mode: u32,
}

/// Reply payload of [`kind::HARDWARE_CONFIGURATION`]: `u8` entry count, then
/// per entry `u32 id`, `f64 param_a, param_b`, 3 `f64` position, 15 `f64`
/// values, `u32 width, height, param_c`, `u8` coefficient count and the
/// `f64` coefficients, 3 + 3 `f64` points, `f64 param_d`; then `u8` point
/// count and 3 `f64` per point; then `u32 mode`. Lists past their limit are
/// cut to it.
#[must_use]
pub fn encode_hardware_configuration(h: &HardwareConfiguration) -> Vec<u8> {
    // Each list is cut to its limit, which fits a `u8`.
    let count = |n: usize, max: usize| u8::try_from(n.min(max)).unwrap_or(u8::MAX);
    let mut w = Writer::default();
    w.u8(count(h.entries.len(), HARDWARE_ENTRIES_MAX));
    for e in h.entries.iter().take(HARDWARE_ENTRIES_MAX) {
        let coefficients = &e.coefficients[..e.coefficients.len().min(HARDWARE_COEFFICIENTS_MAX)];
        w.u32(e.id)
            .f64(e.param_a)
            .f64(e.param_b)
            .f64_array(&e.position_mm)
            .f64_array(&e.values)
            .u32(e.width)
            .u32(e.height)
            .u32(e.param_c)
            .u8(count(coefficients.len(), HARDWARE_COEFFICIENTS_MAX))
            .f64_array(coefficients)
            .f64_array(&e.point_a_mm)
            .f64_array(&e.point_b_mm)
            .f64(e.param_d);
    }
    w.u8(count(h.points_mm.len(), HARDWARE_POINTS_MAX));
    for p in h.points_mm.iter().take(HARDWARE_POINTS_MAX) {
        w.f64_array(p);
    }
    w.u32(h.mode).finish()
}

/// Decode a [`kind::HARDWARE_CONFIGURATION`] reply payload; a count past its
/// limit is `None`.
#[must_use]
pub fn decode_hardware_configuration(payload: &[u8]) -> Option<HardwareConfiguration> {
    let mut r = Reader::new(payload);
    let entry_count = usize::from(r.u8()?);
    if entry_count > HARDWARE_ENTRIES_MAX {
        return None;
    }
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let id = r.u32()?;
        let param_a = r.f64()?;
        let param_b = r.f64()?;
        let position_mm = r.f64_array()?;
        let values = r.f64_array()?;
        let width = r.u32()?;
        let height = r.u32()?;
        let param_c = r.u32()?;
        let coefficient_count = usize::from(r.u8()?);
        if coefficient_count > HARDWARE_COEFFICIENTS_MAX {
            return None;
        }
        let coefficients = (0..coefficient_count)
            .map(|_| r.f64())
            .collect::<Option<Vec<_>>>()?;
        entries.push(HardwareEntry {
            id,
            param_a,
            param_b,
            position_mm,
            values,
            width,
            height,
            param_c,
            coefficients,
            point_a_mm: r.f64_array()?,
            point_b_mm: r.f64_array()?,
            param_d: r.f64()?,
        });
    }
    let point_count = usize::from(r.u8()?);
    if point_count > HARDWARE_POINTS_MAX {
        return None;
    }
    let points_mm = (0..point_count)
        .map(|_| r.f64_array())
        .collect::<Option<Vec<_>>>()?;
    Some(HardwareConfiguration {
        entries,
        points_mm,
        mode: r.u32()?,
    })
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
