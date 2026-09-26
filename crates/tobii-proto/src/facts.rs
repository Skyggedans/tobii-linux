//! What the device says about itself during init, and its notifications.
//!
//! The init replay asks for identity, geometry and status; the answers used
//! to be discarded. [`DeviceFacts`] collects them so the daemon can answer
//! `tobii_get_device_info`, `tobii_get_track_box` and friends without touching
//! USB again. Lengths on the wire are 32.32 fixed point in 1/1024 mm and come
//! out here in millimetres.

use tobii_ipc::geometry::{DisplayArea, GeometryMounting, TrackBox};
use tobii_ipc::request::{
    DeviceInfo, HARDWARE_COEFFICIENTS_MAX, HARDWARE_ENTRIES_MAX, HARDWARE_POINTS_MAX,
    HardwareConfiguration, HardwareEntry, StreamType,
};
use tracing::warn;

use crate::protocol::{MARKER_NOTIFICATION, MARKER_RESPONSE, Message, cmd, notify};
use crate::tlv::{
    FIELD_POINT_3D, FIXED32_ONE, TYPE_U32, TYPE_U32_ALT, Tlv, TlvIter, TlvWriter, UNITS_PER_MM,
};

/// Field id of an indexed string (`05 ID, 02 index, 14 string`).
const FIELD_INDEXED_STRING: u32 = 0x0002_2710;
/// Field id of a stream catalogue entry.
const FIELD_STREAM_ENTRY: u32 = 0x0004_1389;
/// Low half of a list header's field id; the high half's low 12 bits are
/// the entry count plus one.
const FIELD_LIST: u32 = 0x0100;
/// Field id of a hardware configuration entry: 26 components.
const FIELD_HARDWARE_ENTRY: u32 = 0x001a_332c;
/// Field id announcing the display id after the three corners.
const FIELD_DISPLAY_ID: u32 = 0x0001_0100;
/// Field id announcing the output rate pair.
const FIELD_OUTPUT_RATE: u32 = 0x0002_2af8;
/// Index of the fault list in the status strings (the DLL's
/// `tracker_get_status`, 0x1801a0ac0, index table at RVA 0x1a1610); per the
/// 4.1 docs comma-separated, "ok" when there are none.
pub const STATUS_FAULTS: u32 = 5;
/// Index of the warning list in the status strings, as [`STATUS_FAULTS`].
pub const STATUS_WARNINGS: u32 = 6;
/// Index of the calibration id in the status strings.
const STATUS_CALIBRATION_ID: u32 = 7;
/// Index of the integration type in the property strings (command 1330;
/// the DLL's `setup_device_info`, 0x18016e250, stores it at 0x18016e33a).
const PROPERTY_INTEGRATION_TYPE: u32 = 0;
/// The display id the Windows engine writes with every display area.
pub const DEFAULT_DISPLAY_ID: u32 = 12345;

/// Everything the init replay learns about the device.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceFacts {
    /// Serial, model, generation, firmware (command 1420). Its integration
    /// type is always empty: [`DeviceFacts::device_info`] adds it from the
    /// properties.
    pub info: DeviceInfo,
    /// Property strings by index (command 1330).
    pub properties: Vec<(u32, String)>,
    /// Stream catalogue, in the device's order (command 1200).
    pub streams: Vec<StreamType>,
    /// The display area in effect (command 1430, or the last 1440/1450).
    pub display_area: Option<DisplayArea>,
    /// The display id that accompanies the display area.
    pub display_id: Option<u32>,
    /// Track box (command 1400).
    pub track_box: Option<TrackBox>,
    /// Mounting geometry (command 2110).
    pub mounting: Option<GeometryMounting>,
    /// Hardware configuration (command 2120); provisional, and never sent
    /// to Linux so far.
    pub hardware: Option<HardwareConfiguration>,
    /// Status strings by index (the last command 1490).
    pub status: Vec<(u32, String)>,
    /// The active calibration id (status index 7).
    pub calibration_id: Option<u32>,
    /// Output frequency, Hz (second word of command 1650).
    pub output_hz: Option<u32>,
}

impl DeviceFacts {
    /// Fold one message into the facts. Responses are recognised by their
    /// command; a display-area notification updates the area. Returns whether
    /// the message contributed anything.
    pub fn apply(&mut self, msg: &Message<'_>) -> bool {
        let is_response = msg.marker == MARKER_RESPONSE;
        match msg.id {
            cmd::DEVICE_STRINGS if is_response => parse_device_strings(msg)
                .map(|info| self.info = info)
                .is_some(),
            cmd::PROPERTIES if is_response => {
                self.properties = parse_indexed_strings(msg);
                true
            }
            cmd::STREAM_CATALOGUE if is_response => {
                self.streams = parse_stream_catalogue(msg);
                true
            }
            cmd::DISPLAY_AREA_GET if is_response => parse_display_area(msg)
                .map(|(area, id)| {
                    self.display_area = Some(area);
                    self.display_id = id.or(self.display_id);
                })
                .is_some(),
            cmd::TRACK_BOX if is_response => parse_track_box(msg)
                .map(|b| self.track_box = Some(b))
                .is_some(),
            cmd::MOUNTING if is_response => parse_mounting(msg)
                .map(|m| self.mounting = Some(m))
                .is_some(),
            cmd::HARDWARE_CONFIGURATION if is_response => {
                let parsed = parse_hardware_configuration(msg);
                if parsed.is_none() && !msg.payload.is_empty() {
                    warn!(
                        len = msg.payload.len(),
                        "hardware configuration (2120) in an unknown layout; ignored"
                    );
                }
                parsed.map(|h| self.hardware = Some(h)).is_some()
            }
            cmd::STATUS if is_response => {
                self.status = parse_indexed_strings(msg);
                self.calibration_id = self
                    .status_string(STATUS_CALIBRATION_ID)
                    .and_then(|s| s.parse().ok());
                true
            }
            cmd::OUTPUT_RATE if is_response => parse_output_rate(msg)
                .map(|(_, hz)| self.output_hz = Some(hz))
                .is_some(),
            notify::DISPLAY_AREA if msg.marker == MARKER_NOTIFICATION => parse_display_area(msg)
                .map(|(area, _)| self.display_area = Some(area))
                .is_some(),
            _ => false,
        }
    }

    /// Status string `index` of the last command 1490 (the first entry for
    /// it); `None` when the tracker left it out or the 1490 was lost.
    #[must_use]
    pub fn status_string(&self, index: u32) -> Option<&str> {
        self.status
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, s)| s.as_str())
    }

    /// What `tobii_get_device_info` reports: [`Self::info`] and the
    /// integration type, property 0 of command 1330 (`Peripheral` on the
    /// ET5). Of the device's strings, the DLL's built-in tracker module
    /// passes on only these and properties 3 and 4 (`platmod_start`,
    /// 0x18016ac30), so its other device-info fields stay empty. The
    /// integration type is the last record for index 0, as the DLL's
    /// `tracker_get_properties` (0x1801a22a0) keeps the last, and empty
    /// when the tracker left it out or the 1330 was lost. The DLL then skips
    /// its store (0x18016e328, 0x18016e332) and keeps what it held before,
    /// which tobiid mirrors by keeping the previous init's properties across
    /// a lost 1330. Not modelled: the DLL voids the whole 1330, index 0
    /// included, when one of the boolean records (index 1, 5 or 7) holds
    /// anything but `true` or `false` (0x1801a2824); the ET5 sends only
    /// those.
    #[must_use]
    pub fn device_info(&self) -> DeviceInfo {
        let integration_type = self
            .properties
            .iter()
            .rfind(|(i, _)| *i == PROPERTY_INTEGRATION_TYPE)
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        DeviceInfo {
            integration_type,
            ..self.info.clone()
        }
    }

    /// Facts from the init replay's responses, in order.
    #[must_use]
    pub fn from_messages<'a>(msgs: impl IntoIterator<Item = Message<'a>>) -> Self {
        let mut facts = Self::default();
        for msg in msgs {
            facts.apply(&msg);
        }
        facts
    }
}

/// Command 1420: serial, model, generation, firmware. The integration type
/// comes from command 1330 (see [`DeviceFacts::device_info`]), so a 1420
/// answered after it cannot blank it.
#[must_use]
pub fn parse_device_strings(msg: &Message<'_>) -> Option<DeviceInfo> {
    let mut strings = msg.tlvs().filter_map(|t| t.string());
    Some(DeviceInfo {
        serial_number: strings.next()?,
        model: strings.next()?,
        generation: strings.next()?,
        firmware_version: strings.next()?,
        ..DeviceInfo::default()
    })
}

/// Commands 1330 and 1490: `(index, string)` pairs.
#[must_use]
pub fn parse_indexed_strings(msg: &Message<'_>) -> Vec<(u32, String)> {
    let entries: Vec<Tlv<'_>> = msg.tlvs().collect();
    entries
        .windows(3)
        .filter(|w| w[0].field_id() == Some(FIELD_INDEXED_STRING))
        .filter_map(|w| Some((w[1].u32()?, w[2].string()?)))
        .collect()
}

/// Command 1200: the stream catalogue, in wire order.
///
/// The payload is a list header (field `((n + 1) << 16) | 0x0100`, then a
/// `u32` entry type, which is not checked, as the DLL does not check it),
/// then `n` entries of field `0x0004_1389`, `u32` id, string name, string
/// text, `u32` value (the DLL's reader at 0x180004b20). One malformed entry
/// empties the whole catalogue, as it fails the DLL's whole reply.
#[must_use]
pub fn parse_stream_catalogue(msg: &Message<'_>) -> Vec<StreamType> {
    stream_entries(msg).unwrap_or_default()
}

fn stream_entries(msg: &Message<'_>) -> Option<Vec<StreamType>> {
    let mut entries = msg.tlvs();
    let count = list_header(&mut entries)?;
    entries.next()?.u32()?;
    (0..count)
        .map(|_| {
            if entries.next()?.field_id()? != FIELD_STREAM_ENTRY {
                return None;
            }
            Some(StreamType {
                id: typed_u32(entries.next()?)?,
                name: entries.next()?.string()?,
                text: entries.next()?.string()?,
                value: typed_u32(entries.next()?)?,
            })
        })
        .collect()
}

/// A list header's entry count: field `((n + 1) << 16) | 0x0100`. The
/// entry type follows it.
fn list_header(entries: &mut TlvIter<'_>) -> Option<u32> {
    let header = entries.next()?.field_id()?;
    if header & 0xffff != FIELD_LIST {
        return None;
    }
    ((header >> 16) & 0xfff).checked_sub(1)
}

/// A `u32` of type 0x02 (`Tlv::u32` reads any 4 bytes).
fn typed_u32(t: Tlv<'_>) -> Option<u32> {
    (t.typ == TYPE_U32).then(|| t.u32()).flatten()
}

/// A `u32` of type 0x01.
fn alt_u32(t: Tlv<'_>) -> Option<u32> {
    (t.typ == TYPE_U32_ALT).then(|| t.u32()).flatten()
}

/// A 32.32 value (type 0x04) in the device's length unit, as mm.
fn fixed32_mm(t: Tlv<'_>) -> Option<f64> {
    t.fixed32().map(|v| v / UNITS_PER_MM)
}

/// A 3-D point: `05 FIELD_POINT_3D` and three 32.32 components, mm.
fn point_mm(entries: &mut TlvIter<'_>) -> Option<[f64; 3]> {
    if entries.next()?.field_id()? != FIELD_POINT_3D {
        return None;
    }
    Some([
        fixed32_mm(entries.next()?)?,
        fixed32_mm(entries.next()?)?,
        fixed32_mm(entries.next()?)?,
    ])
}

/// Every 3-D point in a payload, mm.
#[must_use]
pub fn parse_points_mm(msg: &Message<'_>) -> Vec<[f64; 3]> {
    let entries: Vec<Tlv<'_>> = msg.tlvs().collect();
    entries
        .windows(4)
        .filter(|w| w[0].field_id() == Some(FIELD_POINT_3D))
        .filter_map(|w| {
            Some([
                w[1].fixed32()? / UNITS_PER_MM,
                w[2].fixed32()? / UNITS_PER_MM,
                w[3].fixed32()? / UNITS_PER_MM,
            ])
        })
        .collect()
}

/// Commands 1430/1440 and notification 1450: the three corners and, when
/// present, the display id.
#[must_use]
pub fn parse_display_area(msg: &Message<'_>) -> Option<(DisplayArea, Option<u32>)> {
    let points = parse_points_mm(msg);
    let [top_left_mm, top_right_mm, bottom_left_mm] =
        <[[f64; 3]; 3]>::try_from(points.get(..3)?).ok()?;
    let entries: Vec<Tlv<'_>> = msg.tlvs().collect();
    let display_id = entries
        .windows(2)
        .find(|w| w[0].field_id() == Some(FIELD_DISPLAY_ID))
        .and_then(|w| w[1].u32());
    Some((
        DisplayArea {
            top_left_mm,
            top_right_mm,
            bottom_left_mm,
        },
        display_id,
    ))
}

/// Command 1400: eight corners in `tobii_track_box_t` order.
#[must_use]
pub fn parse_track_box(msg: &Message<'_>) -> Option<TrackBox> {
    let points = parse_points_mm(msg);
    Some(TrackBox {
        corners_mm: <[[f64; 3]; 8]>::try_from(points.get(..8)?).ok()?,
    })
}

/// Command 2110: guides, width, angle, external and internal offsets.
#[must_use]
pub fn parse_mounting(msg: &Message<'_>) -> Option<GeometryMounting> {
    let mut entries = msg.tlvs();
    let guides = i32::try_from(entries.next()?.u32()?).ok()?;
    let width_mm = entries.next()?.fixed16()?;
    let angle_deg = entries.next()?.fixed16()?;
    let points = parse_points_mm(msg);
    Some(GeometryMounting {
        guides,
        width_mm,
        angle_deg,
        external_offset_mm: *points.first()?,
        internal_offset_mm: *points.get(1)?,
    })
}

/// Command 2120: the hardware configuration, provisional.
///
/// The layout comes from the one response ever captured (Windows,
/// `init.pcapng` frame 773): a list of entries (field `0x001a_332c`, 26
/// components each: `u32` id, two 16.16 values, a point, fifteen 32.32
/// values, three `u32`, a 0x19 list, two points, a 32.32 value), a list of
/// points, then the `u32` mode. Every TLV's type is checked, and anything
/// else (the empty payload the ET5 sends to Linux among it) is `None`.
/// Beyond [`HARDWARE_ENTRIES_MAX`] entries and [`HARDWARE_POINTS_MAX`]
/// points the rest is dropped, as the DLL clamps. Beyond
/// [`HARDWARE_COEFFICIENTS_MAX`] list items too; the DLL does not check
/// that count, but its struct holds only 64. 32.32 values are scaled as lengths (to mm), 16.16 values are
/// not; both are inferred.
#[must_use]
pub fn parse_hardware_configuration(msg: &Message<'_>) -> Option<HardwareConfiguration> {
    let mut tlvs = msg.tlvs();
    let entry_count = list_header(&mut tlvs)?;
    if typed_u32(tlvs.next()?)? != FIELD_HARDWARE_ENTRY & 0xffff {
        return None;
    }
    let mut entries = Vec::new();
    for _ in 0..entry_count {
        let entry = hardware_entry(&mut tlvs)?;
        if entries.len() < HARDWARE_ENTRIES_MAX {
            entries.push(entry);
        }
    }
    let point_count = list_header(&mut tlvs)?;
    if typed_u32(tlvs.next()?)? != FIELD_POINT_3D & 0xffff {
        return None;
    }
    let mut points_mm = Vec::new();
    for _ in 0..point_count {
        let point = point_mm(&mut tlvs)?;
        if points_mm.len() < HARDWARE_POINTS_MAX {
            points_mm.push(point);
        }
    }
    let mode = alt_u32(tlvs.next()?)?;
    if tlvs.next().is_some() || tlvs.truncated() {
        return None;
    }
    Some(HardwareConfiguration {
        entries,
        points_mm,
        mode,
    })
}

/// One entry of a 2120 list.
#[allow(clippy::cast_precision_loss)] // reason: exact for the device's value range
fn hardware_entry(tlvs: &mut TlvIter<'_>) -> Option<HardwareEntry> {
    if tlvs.next()?.field_id()? != FIELD_HARDWARE_ENTRY {
        return None;
    }
    let id = alt_u32(tlvs.next()?)?;
    let param_a = tlvs.next()?.fixed16()?;
    let param_b = tlvs.next()?.fixed16()?;
    let position_mm = point_mm(tlvs)?;
    let mut values = [0.0; 15];
    for v in &mut values {
        *v = fixed32_mm(tlvs.next()?)?;
    }
    let width = alt_u32(tlvs.next()?)?;
    let height = alt_u32(tlvs.next()?)?;
    let param_c = alt_u32(tlvs.next()?)?;
    let coefficients = tlvs
        .next()?
        .fixed32_list_raw()?
        .into_iter()
        .take(HARDWARE_COEFFICIENTS_MAX)
        .map(|raw| raw as f64 / FIXED32_ONE / UNITS_PER_MM)
        .collect();
    Some(HardwareEntry {
        id,
        param_a,
        param_b,
        position_mm,
        values,
        width,
        height,
        param_c,
        coefficients,
        point_a_mm: point_mm(tlvs)?,
        point_b_mm: point_mm(tlvs)?,
        param_d: fixed32_mm(tlvs.next()?)?,
    })
}

/// Command 1650: the two words after the output-rate field id (132, 33 on the
/// ET5: sensor and output rate, Hz).
#[must_use]
pub fn parse_output_rate(msg: &Message<'_>) -> Option<(u32, u32)> {
    let entries: Vec<Tlv<'_>> = msg.tlvs().collect();
    entries
        .windows(3)
        .find(|w| w[0].field_id() == Some(FIELD_OUTPUT_RATE))
        .and_then(|w| Some((w[1].u32()?, w[2].u32()?)))
}

/// Payload of command 1440 (set display area), exactly as the Windows engine
/// writes it: three corners, then the display id.
#[must_use]
pub fn display_area_set_payload(area: &DisplayArea, display_id: u32) -> Vec<u8> {
    TlvWriter::new()
        .point_mm(area.top_left_mm)
        .point_mm(area.top_right_mm)
        .point_mm(area.bottom_left_mm)
        .field_id(FIELD_DISPLAY_ID)
        .u32(display_id)
        .finish()
}

/// Payload of command 3100: `u32 1` pauses the device, `u32 0` resumes it
/// (the DLL's encoder at 0x18017efa0; the init replay sends the resume).
#[must_use]
pub fn device_pause_payload(paused: bool) -> Vec<u8> {
    TlvWriter::new().u32(u32::from(paused)).finish()
}

/// A decoded 0x4e notification.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DeviceNotification {
    /// The display area changed.
    DisplayAreaChanged(DisplayArea),
    /// A new calibration is active.
    CalibrationIdChanged(u32),
    /// The device paused (`true`) or resumed.
    DevicePausedChanged(bool),
    /// Anything else: its id and first `u32`, if any.
    Other {
        /// Notification id.
        id: u32,
        /// The first `u32` in the payload.
        value: Option<u32>,
    },
}

/// Decode a notification message.
#[must_use]
pub fn decode_notification(msg: &Message<'_>) -> Option<DeviceNotification> {
    if msg.marker != MARKER_NOTIFICATION {
        return None;
    }
    let first_u32 = || {
        msg.tlvs()
            .find(|t| t.typ == crate::tlv::TYPE_U32)
            .and_then(|t| t.u32())
    };
    Some(match msg.id {
        notify::DISPLAY_AREA => match parse_display_area(msg) {
            Some((area, _)) => DeviceNotification::DisplayAreaChanged(area),
            None => DeviceNotification::Other {
                id: msg.id,
                value: None,
            },
        },
        notify::CALIBRATION_ID => match first_u32() {
            Some(id) => DeviceNotification::CalibrationIdChanged(id),
            None => DeviceNotification::Other {
                id: msg.id,
                value: None,
            },
        },
        // The DLL rejects a value above 1.
        notify::DEVICE_PAUSED => match first_u32() {
            Some(v @ (0 | 1)) => DeviceNotification::DevicePausedChanged(v == 1),
            value => DeviceNotification::Other { id: msg.id, value },
        },
        id => DeviceNotification::Other {
            id,
            value: first_u32(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{chunk_command, parse_message};

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn facts_from(fixtures: &[Vec<u8>]) -> DeviceFacts {
        DeviceFacts::from_messages(fixtures.iter().filter_map(|b| parse_message(b)))
    }

    #[test]
    fn collects_identity_catalogue_and_status() {
        let facts = facts_from(&[
            crate::fixture!("init-rsp-1420"),
            crate::fixture!("init-rsp-1330"),
            crate::fixture!("init-rsp-1200"),
            crate::fixture!("init-rsp-1490"),
            crate::fixture!("init-rsp-1650"),
        ]);
        assert_eq!(facts.info.model, "IS5_Large_Eyetracker_5");
        assert_eq!(facts.info.generation, "IS5");
        assert_eq!(facts.info.firmware_version, "02a1a6a977");
        assert!(facts.info.serial_number.starts_with("IS50F-"));
        assert!(facts.properties.iter().any(|(_, s)| s == "IS5LEYETRACKER5"));
        assert_eq!(facts.device_info().integration_type, "Peripheral");
        assert_eq!(facts.calibration_id, Some(1_904_654_973));
        assert_eq!(facts.status_string(STATUS_FAULTS), Some("ok"));
        assert_eq!(facts.status_string(STATUS_WARNINGS), Some("ok"));
        assert_eq!(facts.status_string(9), None);
        assert_eq!(facts.output_hz, Some(33));
        assert_eq!(facts.streams.len(), 9);
    }

    #[test]
    fn device_info_adds_the_integration_type_whichever_answer_comes_first() {
        let identity_first = facts_from(&[
            crate::fixture!("init-rsp-1420"),
            crate::fixture!("init-rsp-1330"),
        ]);
        let properties_first = facts_from(&[
            crate::fixture!("init-rsp-1330"),
            crate::fixture!("init-rsp-1420"),
        ]);

        for facts in [&identity_first, &properties_first] {
            let info = facts.device_info();
            assert_eq!(info.model, "IS5_Large_Eyetracker_5");
            assert_eq!(info.firmware_version, "02a1a6a977");
            assert_eq!(info.integration_type, "Peripheral");
            assert_eq!(facts.info.integration_type, "", "only the property");
        }
        assert_eq!(identity_first.device_info(), properties_first.device_info());
    }

    #[test]
    fn the_integration_type_is_the_last_property_0() {
        let with = |properties: &[(u32, &str)]| DeviceFacts {
            properties: properties
                .iter()
                .map(|(i, s)| (*i, (*s).to_owned()))
                .collect(),
            ..DeviceFacts::default()
        };

        // A 1330 that was lost, and one without index 0.
        assert_eq!(DeviceFacts::default().device_info().integration_type, "");
        assert_eq!(
            with(&[(3, "IS5LEYETRACKER5")])
                .device_info()
                .integration_type,
            ""
        );
        // The DLL's reader keeps the last record for an index.
        assert_eq!(
            with(&[(0, "Peripheral"), (1, "true"), (0, "HMD")])
                .device_info()
                .integration_type,
            "HMD"
        );
    }

    #[test]
    fn the_stream_catalogue_keeps_every_field_in_wire_order() {
        let bytes = crate::fixture!("init-rsp-1200");
        let streams = parse_stream_catalogue(&parse_message(&bytes).expect("msg"));

        let got: Vec<(u32, &str, &str, u32)> = streams
            .iter()
            .map(|t| (t.id, t.name.as_str(), t.text.as_str(), t.value))
            .collect();
        assert_eq!(
            got,
            [
                (0x500, "gaze", "", 0),
                (0x501, "image", "", 0),
                (0x504, "presence", "", 0),
                (0x508, "image_collection", "", 1000),
                (0x50e, "primary_camera_image", "", 0),
                (0x1770, "algodbg", "", 0),
                (0x1771, "is5_sync_stream", "", 0),
                (0x1772, "log", "", 0),
                (0x1774, "custom", "", 0),
            ]
        );
    }

    #[test]
    fn a_malformed_catalogue_entry_empties_the_catalogue() {
        let good = crate::fixture!("init-rsp-1200");
        let catalogue = |bytes: &[u8]| parse_stream_catalogue(&parse_message(bytes).expect("msg"));
        // The first entry's id type byte: after the 32-byte header, `00 00`,
        // and three 9-byte TLVs (list header, entry type, entry field).
        let id_type = 32 + 2 + 3 * 9;
        assert_eq!(good[id_type], TYPE_U32);

        let mut wrong_type = good.clone();
        wrong_type[id_type] = crate::tlv::TYPE_U32_ALT;
        assert_eq!(catalogue(&wrong_type), vec![]);

        // The header announces one entry more than the payload holds.
        let mut too_many = good.clone();
        too_many[32 + 2 + 5 + 1] += 1;
        assert_eq!(catalogue(&too_many), vec![]);

        // One entry fewer: the rest is ignored.
        let mut fewer = good;
        fewer[32 + 2 + 5 + 1] -= 1;
        assert_eq!(catalogue(&fewer).len(), 8);
    }

    /// `init.pcapng` frame 773 with `edit` applied (offsets are into the
    /// message, header included), parsed as a 2120.
    fn hardware_edited(edit: impl FnOnce(&mut Vec<u8>)) -> Option<HardwareConfiguration> {
        let mut bytes = crate::fixture!("init-rsp-2120");
        edit(&mut bytes);
        parse_hardware_configuration(&parse_message(&bytes).expect("msg"))
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: fixed-point values scale exactly
    fn parses_the_captured_hardware_configuration() {
        let facts = facts_from(&[crate::fixture!("init-rsp-2120")]);
        let h = facts.hardware.expect("hardware configuration");

        assert_eq!(h.entries.len(), 2);
        let e = &h.entries[0];
        assert_eq!((e.id, e.param_a, e.param_b), (0, 16.0, 100.0));
        #[allow(clippy::cast_precision_loss)] // reason: exact for these values
        let mm = |raw: i64| raw as f64 / FIXED32_ONE / UNITS_PER_MM;
        assert_eq!(e.position_mm, [0.0, 0.0, mm(0x0000_108f_5c28_f5c2)]);
        assert!(close(e.position_mm, [0.0, 0.0, 4.14], 1e-9));
        let mut values = [0.0; 15];
        values[3] = mm(0x0000_14e1_47ae_147a);
        values[13] = mm(0x0000_0001_cac0_8312);
        values[14] = values[13];
        assert_eq!(e.values, values);
        assert!((values[3] - 5.22).abs() < 1e-9 && (values[13] - 0.001_75).abs() < 1e-9);
        assert_eq!((e.width, e.height, e.param_c), (2240, 2240, 0));
        assert!(e.coefficients.is_empty());
        assert_eq!(
            (e.point_a_mm, e.point_b_mm, e.param_d),
            ([0.0; 3], [0.0; 3], 0.0)
        );
        assert_eq!(
            h.entries[1],
            HardwareEntry {
                param_a: 62.0,
                param_b: 100.0,
                ..HardwareEntry::default()
            }
        );

        assert_eq!(h.points_mm.len(), 3);
        assert_eq!(h.points_mm[0], [0.0; 3]);
        assert!(close(h.points_mm[1], [130.0, 0.76, 1.62], 1e-9));
        assert!(close(h.points_mm[2], [-130.0, 0.76, 1.62], 1e-9));
        assert_eq!(h.mode, 1);
    }

    /// What the ET5 sends to Linux: the header alone.
    #[test]
    fn an_empty_hardware_configuration_is_not_a_fact() {
        let bytes = crate::fixture!("init-rsp-2120");
        let msg = parse_message(&bytes[..32]).expect("msg");
        assert_eq!(msg.id, cmd::HARDWARE_CONFIGURATION);

        assert_eq!(parse_hardware_configuration(&msg), None);
        let mut facts = DeviceFacts::default();
        assert!(!facts.apply(&msg));
        assert_eq!(facts.hardware, None);
    }

    #[test]
    fn a_cut_hardware_configuration_is_none() {
        let bytes = crate::fixture!("init-rsp-2120");
        for len in 32..bytes.len() {
            let msg = parse_message(&bytes[..len]).expect("msg");
            assert_eq!(parse_hardware_configuration(&msg), None, "cut at {len}");
        }
        assert_eq!(
            hardware_edited(|b| b.extend_from_slice(&[1, 0, 0, 0, 4, 0, 0, 0, 0])),
            None,
            "a TLV after the mode"
        );
    }

    #[test]
    fn every_hardware_configuration_type_is_checked() {
        // The first entry's id (0x01), its param_a (0x03), a value (0x04),
        // its width (0x01), its list (0x19), the mode (0x01) and the entry
        // list's element type (0x02).
        for (at, other) in [
            (0x3d, TYPE_U32),
            (0x46, crate::tlv::TYPE_FIXED32),
            (0x88, crate::tlv::TYPE_FIXED16),
            (0x14b, TYPE_U32),
            (0x166, crate::tlv::TYPE_LIST),
            (0x426, TYPE_U32),
            (0x2b, TYPE_U32_ALT),
        ] {
            assert_eq!(hardware_edited(|b| b[at] = other), None, "type at {at:#x}");
        }
        // Each entry's field id, with 27 components in place of 26.
        for at in [0x3a, 0x1e2] {
            assert_eq!(
                hardware_edited(|b| b[at] = 0x1b),
                None,
                "an entry of 27 components at {at:#x}"
            );
        }
    }

    /// Replace the first entry's empty 0x19 list (at 0x166) with `value`.
    fn with_list(value: &[u8]) -> Option<HardwareConfiguration> {
        hardware_edited(|b| {
            let mut tlv = vec![crate::tlv::TYPE_FIXED32_LIST];
            tlv.extend_from_slice(&u32::try_from(value.len()).expect("len").to_be_bytes());
            tlv.extend_from_slice(value);
            b.splice(0x166..0x166 + 9, tlv);
        })
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: fixed-point values scale exactly
    fn a_fixed32_list_becomes_the_coefficients() {
        let mut two = 2u32.to_be_bytes().to_vec();
        two.extend_from_slice(&(1024i64 << 32).to_be_bytes());
        two.extend_from_slice(&(-512i64 << 32).to_be_bytes());
        let h = with_list(&two).expect("parses");
        assert_eq!(h.entries[0].coefficients, [1.0, -0.5]);
        assert_eq!(h.entries[0].width, 2240, "the rest still lines up");

        let mut three = two;
        three[3] = 3;
        assert_eq!(with_list(&three), None, "a count the length does not hold");

        let mut many = 65u32.to_be_bytes().to_vec();
        many.extend((0..65i64).flat_map(|i| (i << 42).to_be_bytes()));
        let h = with_list(&many).expect("parses");
        assert_eq!(h.entries[0].coefficients.len(), HARDWARE_COEFFICIENTS_MAX);
        assert_eq!(h.entries[0].coefficients[63], 63.0);
    }

    #[test]
    fn extra_hardware_entries_and_points_are_dropped() {
        let h = hardware_edited(|b| {
            // A third entry: the second one again.
            let second = b[0x1dc..0x384].to_vec();
            b.splice(0x384..0x384, second);
            b[0x22 + 5 + 1] = 4;
        })
        .expect("parses");
        assert_eq!(h.entries.len(), HARDWARE_ENTRIES_MAX);
        assert_eq!((h.points_mm.len(), h.mode), (3, 1));

        let h = hardware_edited(|b| {
            // 41 points: the second one 38 more times.
            let point = b[0x3c6..0x3f6].to_vec();
            for _ in 0..38 {
                b.splice(0x426..0x426, point.iter().copied());
            }
            b[0x384 + 5 + 1] = 42;
        })
        .expect("parses");
        assert_eq!(h.points_mm.len(), HARDWARE_POINTS_MAX);
        assert!(close(h.points_mm[39], [130.0, 0.76, 1.62], 1e-9));
        assert_eq!(h.mode, 1);
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: fixed-point values scale exactly
    fn collects_geometry() {
        let facts = facts_from(&[
            crate::fixture!("init-rsp-1400"),
            crate::fixture!("init-rsp-1430"),
            crate::fixture!("init-rsp-2110"),
        ]);
        let track_box = facts.track_box.expect("track box");
        assert_eq!(track_box.corners_mm[0], [125.0, 100.0, 450.0]);
        assert_eq!(track_box.corners_mm[2], [-125.0, -100.0, 450.0]);
        assert_eq!(track_box.corners_mm[4], [250.0, 200.0, 900.0]);

        let area = facts.display_area.expect("display area");
        assert_eq!(facts.display_id, Some(DEFAULT_DISPLAY_ID));
        assert!(close(
            area.top_left_mm,
            [
                -315.791_717_529_296_9,
                324.411_468_505_859_4,
                111.239_097_595_214_8
            ],
            1e-12
        ));

        let m = facts.mounting.expect("mounting");
        assert_eq!((m.guides, m.width_mm, m.angle_deg), (2, 184.0, 20.0));
        assert!(close(m.external_offset_mm, [0.0, -0.16, 13.85], 1e-9));
        assert!(close(m.internal_offset_mm, [0.0, 5.38, 9.86], 1e-9));
    }

    /// `calculate_display_area_basic` against both monitors the Windows
    /// engine configured: the init one (read back with 1430) and the one set
    /// at runtime in change-display.pcapng (1440).
    #[test]
    fn display_area_basic_reproduces_both_captured_monitors() {
        let mounting =
            parse_mounting(&parse_message(&crate::fixture!("init-rsp-2110")).expect("msg"))
                .expect("mounting");
        let offset_x = 1026.3125 / UNITS_PER_MM;
        for bytes in [
            crate::fixture!("init-rsp-1430"),
            crate::fixture!("change-display-cmd-1440"),
        ] {
            let (area, _) = parse_display_area(&parse_message(&bytes).expect("msg")).expect("area");
            let width = area.top_right_mm[0] - area.top_left_mm[0];
            let d: Vec<f64> = (0..3)
                .map(|i| area.top_left_mm[i] - area.bottom_left_mm[i])
                .collect();
            let height = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();

            let got = tobii_ipc::geometry::display_area_basic(width, height, offset_x, &mounting);

            assert!(
                close(got.top_left_mm, area.top_left_mm, 1e-4),
                "{got:?} vs {area:?}"
            );
            assert!(close(got.top_right_mm, area.top_right_mm, 1e-4));
            assert!(close(got.bottom_left_mm, area.bottom_left_mm, 1e-4));
        }
    }

    /// The runtime display change: parsing the captured 1440 and writing it
    /// back must give the captured bytes, header and all.
    #[test]
    fn display_area_set_round_trips_the_captured_command() {
        let captured = crate::fixture!("change-display-cmd-1440");
        let msg = parse_message(&captured).expect("msg");
        let (area, id) = parse_display_area(&msg).expect("area");

        let rebuilt = chunk_command(
            cmd::DISPLAY_AREA_SET,
            msg.seq,
            &display_area_set_payload(&area, id.expect("id")),
        );

        assert_eq!(rebuilt, vec![captured]);
    }

    #[test]
    fn decodes_notifications() {
        let area = crate::fixture!("change-display-notify-1450");
        match decode_notification(&parse_message(&area).expect("msg")) {
            Some(DeviceNotification::DisplayAreaChanged(a)) => {
                assert!((a.top_right_mm[0] - a.top_left_mm[0] - 597.0).abs() < 1e-9);
            }
            other => panic!("expected a display area, got {other:?}"),
        }
        let calib = crate::fixture!("calib-notify-3220-frame1699");
        assert!(matches!(
            decode_notification(&parse_message(&calib).expect("msg")),
            Some(DeviceNotification::CalibrationIdChanged(0x7186_ba7d))
        ));
        let other = crate::fixture!("init-notify-3180");
        assert_eq!(
            decode_notification(&parse_message(&other).expect("msg")),
            Some(DeviceNotification::Other {
                id: 3180,
                value: Some(3)
            })
        );
    }
    /// The resume every init replay sends: seq 37 of `init_packets_ep.txt`,
    /// which is `init.pcapng` frame 913.
    #[test]
    fn a_resume_is_the_captured_init_command() {
        let frame_913 = crate::protocol::hex_to_bytes(
            "000000002300000000000051000000250000000000000c1c000000000000000b0000020000000400000000",
        )
        .expect("hex");
        let packets =
            crate::protocol::parse_init_packets(crate::INIT_PACKETS).expect("init file parses");
        assert_eq!(packets[36].data, frame_913);

        let resume = device_pause_payload(false);

        assert_eq!(resume, frame_913[32..]);
        assert_eq!(
            chunk_command(cmd::DEVICE_PAUSE, 0x25, &resume),
            vec![frame_913]
        );
        assert_eq!(
            device_pause_payload(true),
            [0, 0, 0x02, 0, 0, 0, 4, 0, 0, 0, 1]
        );
    }

    /// A 3110 carrying `payload`, laid out as the DLL decodes it (none was
    /// ever captured).
    fn paused_notification(payload: &[u8]) -> Vec<u8> {
        let mut msg = chunk_command(notify::DEVICE_PAUSED, 0, payload).swap_remove(0);
        msg[..4].copy_from_slice(&[1, 0, 0, 0]);
        msg[8..12].copy_from_slice(&MARKER_NOTIFICATION.to_be_bytes());
        msg
    }

    #[test]
    fn decodes_a_synthetic_pause_notification() {
        let decode = |payload: &[u8]| {
            let msg = paused_notification(payload);
            decode_notification(&parse_message(&msg).expect("msg"))
        };
        assert_eq!(
            decode(&device_pause_payload(true)),
            Some(DeviceNotification::DevicePausedChanged(true))
        );
        assert_eq!(
            decode(&device_pause_payload(false)),
            Some(DeviceNotification::DevicePausedChanged(false))
        );
        assert_eq!(
            decode(&TlvWriter::new().u32(2).finish()),
            Some(DeviceNotification::Other {
                id: 3110,
                value: Some(2)
            }),
            "the DLL rejects values above 1"
        );
        assert_eq!(
            decode(&TlvWriter::new().finish()),
            Some(DeviceNotification::Other {
                id: 3110,
                value: None
            })
        );
    }
}
