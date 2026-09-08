use anyhow::Result;
use std::collections::BTreeMap;
use std::io::Write;

use crate::sinks::now_us;

pub(crate) const STREAM_TLV_OFFSET: usize = 34;

pub(crate) const GAZE_COORD_MAX: f64 = 1024.0;

/// Lever arm (µm) from the inter-eye line down to the head-roll pivot. A pure
/// head roll swings the eye midpoint sideways about a point below the eyes;
/// adding `lever * sin(roll)` back to the midpoint x removes that arc from the
/// reported lateral translation, so a roll nets ~zero apparent tx. Fit on the
/// roll_only captures (native ≈137 mm; the Windows replication set gives
/// ≈120–125 mm) — user/mounting dependent, hence a single tunable constant.
pub(crate) const HEAD_ROLL_LEVER_UM: f64 = 137_000.0;

/// Pupil diameter is reported in hundredths of a millimeter (raw 342 ≈ 3.42 mm;
/// binocular readings track to r ≈ 1.0). occ10 comp2 = left, occ11 comp2 = right.
pub(crate) const PUPIL_RAW_PER_MM: f64 = 100.0;

pub(crate) const LIVE_FIELDS: &[LiveField] = &[
    LiveField::new("gaze0_x", 0x00021f40, 0, 0),
    LiveField::new("gaze0_y", 0x00021f40, 0, 1),
    LiveField::new("gaze1_x", 0x00021f40, 1, 0),
    LiveField::new("gaze1_y", 0x00021f40, 1, 1),
    LiveField::new("gaze3_x", 0x00021f40, 3, 0),
    LiveField::new("gaze3_y", 0x00021f40, 3, 1),
    LiveField::new("gaze5_x", 0x00021f40, 5, 0),
    LiveField::new("gaze5_y", 0x00021f40, 5, 1),
    LiveField::new("head3_x", 0x00031f41, 3, 0),
    LiveField::new("head3_y", 0x00031f41, 3, 1),
    LiveField::new("head3_z", 0x00031f41, 3, 2),
    LiveField::new("head4_x", 0x00031f41, 4, 0),
    LiveField::new("head4_y", 0x00031f41, 4, 1),
    LiveField::new("head4_z", 0x00031f41, 4, 2),
    LiveField::new("head5_x", 0x00031f41, 5, 0),
    LiveField::new("head5_y", 0x00031f41, 5, 1),
    LiveField::new("head5_z", 0x00031f41, 5, 2),
    LiveField::new("head6_x", 0x00031f41, 6, 0),
    LiveField::new("head6_y", 0x00031f41, 6, 1),
    LiveField::new("head6_z", 0x00031f41, 6, 2),
    LiveField::new("head8_x", 0x00031f41, 8, 0),
    LiveField::new("head8_y", 0x00031f41, 8, 1),
    LiveField::new("head8_z", 0x00031f41, 8, 2),
    LiveField::new("head9_x", 0x00031f41, 9, 0),
    LiveField::new("head9_y", 0x00031f41, 9, 1),
    LiveField::new("head9_z", 0x00031f41, 9, 2),
];

pub(crate) const DERIVED_FIELDS: &[&str] = &[
    "gaze_valid",
    "gaze_x",
    "gaze_y",
    "gaze_norm_x",
    "gaze_norm_y",
    "left_eye_x",
    "left_eye_y",
    "left_eye_norm_x",
    "left_eye_norm_y",
    "right_eye_x",
    "right_eye_y",
    "right_eye_norm_x",
    "right_eye_norm_y",
    "head_x",
    "head_y",
    "head_z",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "pupil_left",
    "pupil_right",
];

#[derive(Clone, Copy)]
pub(crate) struct LiveField {
    pub(crate) name: &'static str,
    pub(crate) id: u32,
    pub(crate) occurrence: usize,
    pub(crate) component: usize,
}

impl LiveField {
    pub(crate) const fn new(name: &'static str, id: u32, occurrence: usize, component: usize) -> Self {
        Self {
            name,
            id,
            occurrence,
            component,
        }
    }

    pub(crate) fn key(self) -> (u32, usize, usize) {
        (self.id, self.occurrence, self.component)
    }
}

pub(crate) fn head_point(values: &BTreeMap<(u32, usize, usize), f64>, occurrence: usize) -> Option<[f64; 3]> {
    Some([
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 0))?,
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 1))?,
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 2))?,
    ])
}

#[derive(Clone, Debug)]
pub(crate) struct TrackingFrame {
    pub(crate) ts_us: u64,
    pub(crate) packet: u64,
    pub(crate) gaze_valid: bool,
    pub(crate) gaze_x: Option<f64>,
    pub(crate) gaze_y: Option<f64>,
    pub(crate) gaze_norm_x: Option<f64>,
    pub(crate) gaze_norm_y: Option<f64>,
    pub(crate) left_eye_x: Option<f64>,
    pub(crate) left_eye_y: Option<f64>,
    pub(crate) left_eye_norm_x: Option<f64>,
    pub(crate) left_eye_norm_y: Option<f64>,
    pub(crate) right_eye_x: Option<f64>,
    pub(crate) right_eye_y: Option<f64>,
    pub(crate) right_eye_norm_x: Option<f64>,
    pub(crate) right_eye_norm_y: Option<f64>,
    pub(crate) head_x: Option<f64>,
    pub(crate) head_y: Option<f64>,
    pub(crate) head_z: Option<f64>,
    pub(crate) head_yaw: Option<f64>,
    pub(crate) head_pitch: Option<f64>,
    pub(crate) head_roll: Option<f64>,
    pub(crate) pupil_left: Option<f64>,
    pub(crate) pupil_right: Option<f64>,
}

impl TrackingFrame {
    pub(crate) fn from_decoded(packet: u64, values: &BTreeMap<(u32, usize, usize), f64>) -> Self {
        let derived = derive_live_values(values);

        Self {
            ts_us: now_us(),
            packet,
            gaze_valid: derived[0].is_some_and(|value| value >= 0.5),
            gaze_x: derived[1],
            gaze_y: derived[2],
            gaze_norm_x: derived[3],
            gaze_norm_y: derived[4],
            left_eye_x: derived[5],
            left_eye_y: derived[6],
            left_eye_norm_x: derived[7],
            left_eye_norm_y: derived[8],
            right_eye_x: derived[9],
            right_eye_y: derived[10],
            right_eye_norm_x: derived[11],
            right_eye_norm_y: derived[12],
            head_x: derived[13],
            head_y: derived[14],
            head_z: derived[15],
            head_yaw: derived[16],
            head_pitch: derived[17],
            head_roll: derived[18],
            pupil_left: derived[19],
            pupil_right: derived[20],
        }
    }

    pub(crate) fn write_json<W: Write>(&self, out: &mut W) -> Result<()> {
        write!(
            out,
            "{{\"ts_us\":{},\"packet\":{},\"gaze_valid\":{},",
            self.ts_us, self.packet, self.gaze_valid
        )?;
        write!(out, "\"gaze\":{{")?;
        write_json_number(out, "x", self.gaze_x, true)?;
        write_json_number(out, "y", self.gaze_y, false)?;
        write_json_number(out, "norm_x", self.gaze_norm_x, false)?;
        write_json_number(out, "norm_y", self.gaze_norm_y, false)?;
        write!(out, "}},\"left_eye\":{{")?;
        write_json_number(out, "x", self.left_eye_x, true)?;
        write_json_number(out, "y", self.left_eye_y, false)?;
        write_json_number(out, "norm_x", self.left_eye_norm_x, false)?;
        write_json_number(out, "norm_y", self.left_eye_norm_y, false)?;
        write!(out, "}},\"right_eye\":{{")?;
        write_json_number(out, "x", self.right_eye_x, true)?;
        write_json_number(out, "y", self.right_eye_y, false)?;
        write_json_number(out, "norm_x", self.right_eye_norm_x, false)?;
        write_json_number(out, "norm_y", self.right_eye_norm_y, false)?;
        write!(out, "}},\"head\":{{")?;
        write_json_number(out, "x", self.head_x, true)?;
        write_json_number(out, "y", self.head_y, false)?;
        write_json_number(out, "z", self.head_z, false)?;
        write_json_number(out, "yaw", self.head_yaw, false)?;
        write_json_number(out, "pitch", self.head_pitch, false)?;
        write_json_number(out, "roll", self.head_roll, false)?;
        write!(out, "}},\"pupil\":{{")?;
        write_json_number(out, "left", self.pupil_left, true)?;
        write_json_number(out, "right", self.pupil_right, false)?;
        write!(out, "}}}}")?;
        Ok(())
    }

    pub(crate) fn head_xyz(&self) -> Option<[f64; 3]> {
        Some([self.head_x?, self.head_y?, self.head_z?])
    }
}

pub(crate) fn write_json_number<W: Write>(
    out: &mut W,
    name: &str,
    value: Option<f64>,
    first: bool,
) -> Result<()> {
    if !first {
        write!(out, ",")?;
    }

    write!(out, "\"{name}\":")?;
    match value {
        Some(value) if value.is_finite() => write!(out, "{value:.6}")?,
        _ => write!(out, "null")?,
    }
    Ok(())
}

pub(crate) fn derive_live_values(values: &BTreeMap<(u32, usize, usize), f64>) -> [Option<f64>; 21] {
    let left_eye_x = field_value(values, LiveField::new("", 0x00021f40, 1, 0));
    let left_eye_y = field_value(values, LiveField::new("", 0x00021f40, 1, 1));
    let right_eye_x = field_value(values, LiveField::new("", 0x00021f40, 3, 0));
    let right_eye_y = field_value(values, LiveField::new("", 0x00021f40, 3, 1));
    let gaze_x = mean_keys(
        values,
        &[
            LiveField::new("", 0x00021f40, 1, 0),
            LiveField::new("", 0x00021f40, 3, 0),
        ],
    );
    let gaze_y = mean_keys(
        values,
        &[
            LiveField::new("", 0x00021f40, 1, 1),
            LiveField::new("", 0x00021f40, 3, 1),
        ],
    );
    // Head pose from the two eyeball rotation centers: occ4 = left, occ9 =
    // right of 0x00031f41. Unlike the cornea / gaze-origin points (occ0/5/6),
    // these do not follow the eyes, so translation and roll come out
    // gaze-invariant. `head_point` applies the per-field sentinel filter, so a
    // frame with either eye dropped yields None here rather than a one-eyed
    // (IPD/2-biased) estimate. See stream-0x83 field analysis (2026-07-06).
    let (head_x, head_y, head_z, head_roll) = match (head_point(values, 4), head_point(values, 9)) {
        (Some(left), Some(right)) => {
            let mid_x = (left[0] + right[0]) / 2.0;
            let mid_y = (left[1] + right[1]) / 2.0;
            let mid_z = (left[2] + right[2]) / 2.0;
            // Inter-eye line (left -> right); its tilt off horizontal is head roll.
            let roll = (right[1] - left[1]).atan2(right[0] - left[0]);
            // Undo the sideways arc a roll about the sub-eye pivot induces, so a
            // pure roll leaves the reported lateral translation unchanged.
            let tx = mid_x + HEAD_ROLL_LEVER_UM * roll.sin();
            (Some(tx), Some(mid_y), Some(mid_z), Some(roll.to_degrees()))
        }
        _ => (None, None, None, None),
    };
    // Yaw and pitch are not recoverable from this stream: the two eyes' depths
    // are a synthetic equal-camera-range reconstruction, so any positional
    // "yaw" is just lateral head position in disguise (proven on the labelled
    // captures — genuine yaw content of a deliberate turn is ~0.07°). Report
    // None rather than a translation artifact dressed up as an angle.
    let head_yaw = None;
    let head_pitch = None;
    // Pupil diameters (occ10 = left, occ11 = right; comp2), converted to mm.
    // Gaze-invariant and binocular-correlated — a genuine attention/arousal
    // and blink signal, independent of the head-pose channels above.
    let pupil_left =
        field_value(values, LiveField::new("", 0x00031f41, 10, 2)).map(|v| v / PUPIL_RAW_PER_MM);
    let pupil_right =
        field_value(values, LiveField::new("", 0x00031f41, 11, 2)).map(|v| v / PUPIL_RAW_PER_MM);
    let gaze_valid = gaze_valid(gaze_x, gaze_y);

    [
        Some(if gaze_valid { 1.0 } else { 0.0 }),
        gaze_x,
        gaze_y,
        norm_gaze(gaze_x),
        norm_gaze(gaze_y),
        left_eye_x,
        left_eye_y,
        norm_gaze(left_eye_x),
        norm_gaze(left_eye_y),
        right_eye_x,
        right_eye_y,
        norm_gaze(right_eye_x),
        norm_gaze(right_eye_y),
        head_x,
        head_y,
        head_z,
        head_yaw,
        head_pitch,
        head_roll,
        pupil_left,
        pupil_right,
    ]
}

pub(crate) fn gaze_valid(x: Option<f64>, y: Option<f64>) -> bool {
    let Some(x) = x else {
        return false;
    };
    let Some(y) = y else {
        return false;
    };

    (-0.25 * GAZE_COORD_MAX..=1.25 * GAZE_COORD_MAX).contains(&x)
        && (-0.25 * GAZE_COORD_MAX..=1.25 * GAZE_COORD_MAX).contains(&y)
}

pub(crate) fn norm_gaze(value: Option<f64>) -> Option<f64> {
    value.map(|value| (value / GAZE_COORD_MAX).clamp(0.0, 1.0))
}

pub(crate) fn field_value(values: &BTreeMap<(u32, usize, usize), f64>, field: LiveField) -> Option<f64> {
    let value = values.get(&field.key()).copied()?;
    if value.abs() == 1024.0 || value == 0.0 {
        None
    } else {
        Some(value)
    }
}

pub(crate) fn mean_keys(values: &BTreeMap<(u32, usize, usize), f64>, fields: &[LiveField]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;

    for field in fields {
        let Some(value) = field_value(values, *field) else {
            continue;
        };

        sum += value;
        count += 1;
    }

    (count > 0).then_some(sum / count as f64)
}

pub(crate) fn decode_stream_payload(payload: &[u8]) -> Result<BTreeMap<(u32, usize, usize), f64>> {
    let (values, _) = decode_stream_payload_with_status(payload)?;
    Ok(values)
}

pub(crate) fn decode_stream_payload_with_status(
    payload: &[u8],
) -> Result<(BTreeMap<(u32, usize, usize), f64>, bool)> {
    let mut values = BTreeMap::<(u32, usize, usize), f64>::new();
    let mut offset = STREAM_TLV_OFFSET;
    let mut current_id = None::<u32>;
    let mut current_occurrence = 0usize;
    let mut component = 0usize;
    let mut occurrences = BTreeMap::<u32, usize>::new();
    let mut malformed = false;

    while offset + 5 <= payload.len() {
        let typ = payload[offset];
        let len = u32::from_be_bytes(payload[offset + 1..offset + 5].try_into()?) as usize;
        offset += 5;

        if offset + len > payload.len() {
            malformed = true;
            break;
        }

        let value = &payload[offset..offset + len];

        match (typ, len) {
            (5, 4) => {
                let id = u32::from_be_bytes(value.try_into()?);
                let occurrence = occurrences.entry(id).or_default();
                current_id = Some(id);
                current_occurrence = *occurrence;
                *occurrence += 1;
                component = 0;
            }
            (4, 8) => {
                if let Some(id) = current_id {
                    let raw = i64::from_be_bytes(value.try_into()?);
                    let fixed = raw as f64 / 4_294_967_296.0;
                    values.insert((id, current_occurrence, component), fixed);
                    component += 1;
                }
            }
            (2, 4) | (3, 4) | (6, 8) => {}
            _ => {}
        }

        offset += len;
    }

    Ok((values, malformed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(values: &mut BTreeMap<(u32, usize, usize), f64>, occ: usize, p: [f64; 3]) {
        for (component, v) in p.into_iter().enumerate() {
            values.insert((0x00031f41, occ, component), v);
        }
    }

    #[test]
    fn head_from_eyeball_centers_with_roll_compensation() {
        // Left (occ4) / right (occ9) eyeball centers tilted so the inter-eye
        // line rolls ~10° about the viewing axis; midpoint sits at x = 0.
        let mut values = BTreeMap::new();
        eye(&mut values, 4, [-35_000.0, -6_170.0, 600_000.0]);
        eye(&mut values, 9, [35_000.0, 6_170.0, 600_000.0]);

        let derived = derive_live_values(&values);
        let head_x = derived[13].expect("head_x");
        let head_z = derived[15].expect("head_z");
        let head_yaw = derived[16];
        let head_pitch = derived[17];
        let head_roll = derived[18].expect("head_roll");

        let roll = (6_170.0f64 - -6_170.0).atan2(35_000.0 - -35_000.0);
        assert!((head_roll - roll.to_degrees()).abs() < 1e-6);
        // midpoint x is 0, so head_x is pure roll compensation.
        assert!((head_x - HEAD_ROLL_LEVER_UM * roll.sin()).abs() < 1e-6);
        assert!((head_z - 600_000.0).abs() < 1e-6);
        // Yaw and pitch are deliberately absent (not recoverable from 0x83).
        assert!(head_yaw.is_none());
        assert!(head_pitch.is_none());
    }

    #[test]
    fn pupil_diameters_scaled_to_mm() {
        let mut values = BTreeMap::new();
        values.insert((0x00031f41, 10, 2), 342.0); // left pupil, hundredths of mm
        values.insert((0x00031f41, 11, 2), 335.0); // right pupil

        let derived = derive_live_values(&values);
        assert!((derived[19].expect("pupil_left") - 3.42).abs() < 1e-9);
        assert!((derived[20].expect("pupil_right") - 3.35).abs() < 1e-9);
    }

    #[test]
    fn head_none_when_an_eye_center_drops() {
        // Only the left eyeball center present -> no biased one-eyed estimate.
        let mut values = BTreeMap::new();
        eye(&mut values, 4, [-35_000.0, 1_000.0, 600_000.0]);

        let derived = derive_live_values(&values);
        assert!(derived[13].is_none()); // head_x
        assert!(derived[14].is_none()); // head_y
        assert!(derived[15].is_none()); // head_z
        assert!(derived[18].is_none()); // head_roll
    }
}
