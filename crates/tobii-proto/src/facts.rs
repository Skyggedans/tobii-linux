//! What the device says about itself during init, and its notifications.
//!
//! The init replay asks for identity, geometry and status; the answers used
//! to be discarded. [`DeviceFacts`] collects them so the daemon can answer
//! `tobii_get_device_info`, `tobii_get_track_box` and friends without touching
//! USB again. Lengths on the wire are 32.32 fixed point in 1/1024 mm and come
//! out here in millimetres.

use tobii_ipc::geometry::{DisplayArea, GeometryMounting, TrackBox};
use tobii_ipc::request::{DeviceInfo, StreamType};

use crate::protocol::{MARKER_NOTIFICATION, MARKER_RESPONSE, Message, cmd, notify};
use crate::tlv::{FIELD_POINT_3D, TYPE_U32, Tlv, TlvWriter, UNITS_PER_MM};

/// Field id of an indexed string (`05 ID, 02 index, 14 string`).
const FIELD_INDEXED_STRING: u32 = 0x0002_2710;
/// Field id of a stream catalogue entry.
const FIELD_STREAM_ENTRY: u32 = 0x0004_1389;
/// Low half of a list header's field id; the high half's low 12 bits are
/// the entry count plus one.
const FIELD_LIST: u32 = 0x0100;
/// Field id announcing the display id after the three corners.
const FIELD_DISPLAY_ID: u32 = 0x0001_0100;
/// Field id announcing the output rate pair.
const FIELD_OUTPUT_RATE: u32 = 0x0002_2af8;
/// Index of the calibration id in the status strings.
const STATUS_CALIBRATION_ID: u32 = 7;
/// The display id the Windows engine writes with every display area.
pub const DEFAULT_DISPLAY_ID: u32 = 12345;

/// Everything the init replay learns about the device.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceFacts {
    /// Serial, model, generation, firmware (command 1420).
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
            cmd::STATUS if is_response => {
                self.status = parse_indexed_strings(msg);
                self.calibration_id = self
                    .status
                    .iter()
                    .find(|(i, _)| *i == STATUS_CALIBRATION_ID)
                    .and_then(|(_, s)| s.parse().ok());
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

/// Command 1420: serial, model, generation, firmware.
#[must_use]
pub fn parse_device_strings(msg: &Message<'_>) -> Option<DeviceInfo> {
    let mut strings = msg.tlvs().filter_map(|t| t.string());
    Some(DeviceInfo {
        serial_number: strings.next()?,
        model: strings.next()?,
        generation: strings.next()?,
        firmware_version: strings.next()?,
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
    let header = entries.next()?.field_id()?;
    if header & 0xffff != FIELD_LIST {
        return None;
    }
    let count = ((header >> 16) & 0xfff).checked_sub(1)?;
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

/// A `u32` of type 0x02 (`Tlv::u32` reads any 4 bytes).
fn typed_u32(t: Tlv<'_>) -> Option<u32> {
    (t.typ == TYPE_U32).then(|| t.u32()).flatten()
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

/// A decoded 0x4e notification.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DeviceNotification {
    /// The display area changed.
    DisplayAreaChanged(DisplayArea),
    /// A new calibration is active.
    CalibrationIdChanged(u32),
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
        assert_eq!(facts.calibration_id, Some(1_904_654_973));
        assert_eq!(facts.output_hz, Some(33));
        assert_eq!(facts.streams.len(), 9);
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
}
