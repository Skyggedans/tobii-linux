//! Decoder for the gaze stream (id 0x500) on EP 0x83: TLV parsing into a
//! `(field id, occurrence, component)` map of 32.32 fixed-point values, and the
//! derived per-frame quantities (`TrackingFrame`) built from them.
//!
//! Field ids: `0x00021f40` carries 2-D gaze/eye points (occ1 = left eye,
//! occ3 = right eye) in a 0..1024 screen space; `0x00031f41` carries 3-D head
//! points in µm (occ4/occ9 = eyeball rotation centres; occ10/occ11 are keys
//! `0x25`/`0x27`, whose third component, reported as `pupil_*`, tracks the
//! eye's range, not its pupil, which is keys `0x06`/`0x0c`: see
//! [`crate::gaze83`]). See the stream-0x83 field map for the full table.

use anyhow::Result;
use std::collections::BTreeMap;
use std::io::Write;

use crate::time::now_us;

/// Key of a decoded stream value: `(field id, occurrence, component)`.
pub type FieldKey = (u32, usize, usize);

/// All values of one decoded stream message, keyed by [`FieldKey`].
pub type FieldValues = BTreeMap<FieldKey, f64>;

/// Offset of the first TLV entry in a 0x53 stream message (after the common
/// 34-byte stream header).
pub const STREAM_TLV_OFFSET: usize = 34;

/// Full-scale gaze coordinate; gaze points are reported in `0..=1024`.
pub const GAZE_COORD_MAX: f64 = 1024.0;

/// The device reports an absent field as exactly `±1024.0` or `0.0` (32.32
/// fixed point, so these are bit-exact).
const FIELD_SENTINEL: f64 = 1024.0;

/// Denominator of the 32.32 fixed-point wire format (`2^32`).
const FIXED_POINT_ONE: f64 = 4_294_967_296.0;

/// Lever arm (µm) from the inter-eye line down to the head-roll pivot. A pure
/// head roll swings the eye midpoint sideways about a point below the eyes;
/// adding `lever * sin(roll)` back to the midpoint x removes that arc from the
/// reported lateral translation, so a roll nets ~zero apparent tx. Fit on the
/// `roll_only` captures (native ≈137 mm; the Windows replication set gives
/// ≈120–125 mm) — user/mounting dependent, hence a single tunable constant.
pub const HEAD_ROLL_LEVER_UM: f64 = 137_000.0;

/// Scale of the `pupil_*` columns: occ10 comp2 = left, occ11 comp2 = right
/// (keys `0x25`/`0x27`), over 100 (raw 342 → 3.42). Despite the name these
/// track the eye's range, not its pupil; the pupil diameter is
/// [`crate::gaze83::EyeFrame::pupil_diameter_mm`] (keys `0x06`/`0x0c`).
pub const PUPIL_RAW_PER_MM: f64 = 100.0;

/// Raw stream fields written to the decoded CSV, in column order.
pub const LIVE_FIELDS: &[LiveField] = &[
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

/// Column names of [`derive_live_values`], in the same order as its output.
pub const DERIVED_FIELDS: &[&str] = &[
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

/// Address of one scalar in a decoded stream message, plus its CSV name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveField {
    /// CSV column name (empty for ad-hoc lookups).
    pub name: &'static str,
    /// TLV field id (type-5 entry).
    pub id: u32,
    /// Zero-based index of this id's repetition within the message.
    pub occurrence: usize,
    /// Zero-based index of the type-4 value following that id.
    pub component: usize,
}

impl LiveField {
    /// A field address; `name` is only used for CSV headers.
    #[must_use]
    pub const fn new(name: &'static str, id: u32, occurrence: usize, component: usize) -> Self {
        Self {
            name,
            id,
            occurrence,
            component,
        }
    }

    /// The map key this field is stored under.
    #[must_use]
    pub fn key(self) -> FieldKey {
        (self.id, self.occurrence, self.component)
    }
}

/// The 3-D head point of occurrence `occurrence` of field `0x00031f41`, or
/// `None` if any component is absent (sentinel-filtered).
#[must_use]
pub fn head_point(values: &FieldValues, occurrence: usize) -> Option<[f64; 3]> {
    Some([
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 0))?,
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 1))?,
        field_value(values, LiveField::new("", 0x00031f41, occurrence, 2))?,
    ])
}

/// One decoded gaze-stream message as the derived quantities the sinks emit.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackingFrame {
    /// Host wall-clock time (µs since the Unix epoch) when decoded.
    pub ts_us: u64,
    /// Running stream message counter.
    pub packet: u64,
    /// Whether the binocular gaze point lies within the plausible screen range.
    pub gaze_valid: bool,
    /// Binocular gaze x (mean of both eyes), 0..1024.
    pub gaze_x: Option<f64>,
    /// Binocular gaze y (mean of both eyes), 0..1024.
    pub gaze_y: Option<f64>,
    /// `gaze_x / 1024` clamped to 0..1.
    pub gaze_norm_x: Option<f64>,
    /// `gaze_y / 1024` clamped to 0..1.
    pub gaze_norm_y: Option<f64>,
    /// Left-eye gaze x, 0..1024.
    pub left_eye_x: Option<f64>,
    /// Left-eye gaze y, 0..1024.
    pub left_eye_y: Option<f64>,
    /// Normalised left-eye gaze x.
    pub left_eye_norm_x: Option<f64>,
    /// Normalised left-eye gaze y.
    pub left_eye_norm_y: Option<f64>,
    /// Right-eye gaze x, 0..1024.
    pub right_eye_x: Option<f64>,
    /// Right-eye gaze y, 0..1024.
    pub right_eye_y: Option<f64>,
    /// Normalised right-eye gaze x.
    pub right_eye_norm_x: Option<f64>,
    /// Normalised right-eye gaze y.
    pub right_eye_norm_y: Option<f64>,
    /// Head lateral position (µm), roll-compensated.
    pub head_x: Option<f64>,
    /// Head vertical position (µm).
    pub head_y: Option<f64>,
    /// Head distance from the tracker (µm).
    pub head_z: Option<f64>,
    /// Always `None`: yaw is not recoverable from this stream.
    pub head_yaw: Option<f64>,
    /// Always `None`: pitch is not recoverable from this stream.
    pub head_pitch: Option<f64>,
    /// Head roll (degrees) from the inter-eye line.
    pub head_roll: Option<f64>,
    /// Key `0x25`'s third component / 100: range-like, not the left pupil
    /// (that is [`crate::gaze83::EyeFrame::pupil_diameter_mm`]).
    pub pupil_left: Option<f64>,
    /// Key `0x27`'s third component / 100: range-like, not the right pupil.
    pub pupil_right: Option<f64>,
}

impl TrackingFrame {
    /// Build a frame from a decoded value map, stamping it with the host clock.
    #[must_use]
    pub fn from_decoded(packet: u64, values: &FieldValues) -> Self {
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

    /// Write the frame as one JSON object (no trailing newline); absent or
    /// non-finite numbers become `null`.
    ///
    /// # Errors
    ///
    /// Propagates write failures from `out`.
    pub fn write_json<W: Write>(&self, out: &mut W) -> Result<()> {
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

    /// Head position as `[x, y, z]` (µm), or `None` if any axis is absent.
    #[must_use]
    pub fn head_xyz(&self) -> Option<[f64; 3]> {
        Some([self.head_x?, self.head_y?, self.head_z?])
    }
}

/// Write `"name":value` (six decimals, `null` when absent or non-finite),
/// preceded by a comma unless `first`.
///
/// # Errors
///
/// Propagates write failures from `out`.
pub fn write_json_number<W: Write>(
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

/// Compute the [`DERIVED_FIELDS`] columns (same order) from a decoded message.
#[must_use]
pub fn derive_live_values(values: &FieldValues) -> [Option<f64>; 21] {
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
    // The `pupil_*` columns: occ10 = left, occ11 = right (keys 0x25/0x27),
    // comp2, / 100. Despite the name they track the eye's range, not its
    // pupil; the pupil diameter is keys 0x06/0x0c (see crate::gaze83).
    let pupil_left =
        field_value(values, LiveField::new("", 0x00031f41, 10, 2)).map(|v| v / PUPIL_RAW_PER_MM);
    let pupil_right =
        field_value(values, LiveField::new("", 0x00031f41, 11, 2)).map(|v| v / PUPIL_RAW_PER_MM);
    let gaze_valid = is_gaze_valid(gaze_x, gaze_y);

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

/// Whether both gaze coordinates are present and within 25% beyond the
/// screen edges (`-256..=1280`).
#[must_use]
pub fn is_gaze_valid(x: Option<f64>, y: Option<f64>) -> bool {
    let Some(x) = x else {
        return false;
    };
    let Some(y) = y else {
        return false;
    };

    (-0.25 * GAZE_COORD_MAX..=1.25 * GAZE_COORD_MAX).contains(&x)
        && (-0.25 * GAZE_COORD_MAX..=1.25 * GAZE_COORD_MAX).contains(&y)
}

/// Gaze coordinate scaled to `0..=1` (clamped).
#[must_use]
pub fn norm_gaze(value: Option<f64>) -> Option<f64> {
    value.map(|value| (value / GAZE_COORD_MAX).clamp(0.0, 1.0))
}

/// Look up a field, treating the device's "absent" sentinels (`±1024.0`,
/// `0.0`) as `None`.
#[must_use]
// reason: the sentinels are exact 32.32 fixed-point constants from the wire,
// so a bit-exact comparison is the correct check (num-float-compare).
#[allow(clippy::float_cmp)]
pub fn field_value(values: &FieldValues, field: LiveField) -> Option<f64> {
    let value = values.get(&field.key()).copied()?;
    if value.abs() == FIELD_SENTINEL || value == 0.0 {
        None
    } else {
        Some(value)
    }
}

/// Mean of the present (non-sentinel) fields, or `None` if none are present.
#[must_use]
pub fn mean_keys(values: &FieldValues, fields: &[LiveField]) -> Option<f64> {
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

/// Decode a stream message into its value map, ignoring truncation.
///
/// # Errors
///
/// See [`decode_stream_payload_with_status`].
pub fn decode_stream_payload(payload: &[u8]) -> Result<FieldValues> {
    let (values, _) = decode_stream_payload_with_status(payload)?;
    Ok(values)
}

/// Decode the TLV list of a stream message starting at [`STREAM_TLV_OFFSET`].
///
/// Entries are `type: u8, len: BE u32, value`. A type-5/len-4 entry announces
/// a field id (its n-th repetition is occurrence n); each following type-4/
/// len-8 entry is one BE i64 32.32 fixed-point component of that field. Other
/// entries (2/3: u32, 6: u64 timestamp) are skipped.
///
/// Returns the values and whether the list was truncated (an entry's declared
/// length ran past the end of the payload); parsing stops at that entry.
///
/// # Errors
///
/// Fails only if a wire length does not fit in `usize` (32-bit hosts).
pub fn decode_stream_payload_with_status(payload: &[u8]) -> Result<(FieldValues, bool)> {
    let mut values = FieldValues::new();
    let mut offset = STREAM_TLV_OFFSET;
    let mut current_id = None::<u32>;
    let mut current_occurrence = 0usize;
    let mut component = 0usize;
    let mut occurrences = BTreeMap::<u32, usize>::new();
    let mut malformed = false;

    while offset + 5 <= payload.len() {
        let typ = payload[offset];
        let len = usize::try_from(u32::from_be_bytes(
            payload[offset + 1..offset + 5].try_into()?,
        ))?;
        offset += 5;

        // `offset <= payload.len()` here, so this cannot underflow, and the
        // subtraction form cannot overflow the way `offset + len` could.
        if len > payload.len() - offset {
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
                    // cast: 32.32 fixed point -> f64; the integer part is
                    // small (µm / screen units), well within f64's 53 bits.
                    let fixed = raw as f64 / FIXED_POINT_ONE;
                    values.insert((id, current_occurrence, component), fixed);
                    component += 1;
                }
            }
            // u32 fields and the u64 device timestamp: not used by the live
            // decoder, listed so the known wire types are documented here.
            (2 | 3, 4) | (6, 8) => {}
            _ => {}
        }

        offset += len;
    }

    Ok((values, malformed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(values: &mut FieldValues, occ: usize, p: [f64; 3]) {
        for (component, v) in p.into_iter().enumerate() {
            values.insert((0x00031f41, occ, component), v);
        }
    }

    /// One TLV entry: `type`, BE u32 length, value.
    fn tlv(typ: u8, value: &[u8]) -> Vec<u8> {
        let mut v = vec![typ];
        v.extend_from_slice(&u32::try_from(value.len()).expect("fits u32").to_be_bytes());
        v.extend_from_slice(value);
        v
    }

    /// A stream message with header padding and one id followed by `raw`
    /// 32.32 fixed-point components (already in wire units).
    fn message(id: u32, raw: &[i64]) -> Vec<u8> {
        let mut msg = vec![0u8; STREAM_TLV_OFFSET];
        msg.extend(tlv(5, &id.to_be_bytes()));
        for r in raw {
            msg.extend(tlv(4, &r.to_be_bytes()));
        }
        msg
    }

    /// `v` in 32.32 fixed point.
    const fn fixed(v: i64) -> i64 {
        v << 32
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
        values.insert((0x00031f41, 10, 2), 342.0); // pupil_left: key 0x25 comp2
        values.insert((0x00031f41, 11, 2), 335.0); // pupil_right: key 0x27 comp2

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

    #[test]
    fn field_value_filters_exact_sentinels_only() {
        let mut values = BTreeMap::new();
        values.insert((0x00021f40, 1, 0), 1024.0);
        values.insert((0x00021f40, 1, 1), -1024.0);
        values.insert((0x00021f40, 3, 0), 0.0);
        values.insert((0x00021f40, 3, 1), 1023.5);
        assert_eq!(
            field_value(&values, LiveField::new("", 0x00021f40, 1, 0)),
            None
        );
        assert_eq!(
            field_value(&values, LiveField::new("", 0x00021f40, 1, 1)),
            None
        );
        assert_eq!(
            field_value(&values, LiveField::new("", 0x00021f40, 3, 0)),
            None
        );
        assert_eq!(
            field_value(&values, LiveField::new("", 0x00021f40, 3, 1)),
            Some(1023.5)
        );
        assert_eq!(
            field_value(&values, LiveField::new("", 0x00021f40, 9, 9)),
            None
        );
    }

    #[test]
    fn decodes_fixed_point_components_by_occurrence() {
        let mut msg = message(0x00031f41, &[fixed(3) >> 1, fixed(-2), fixed(3)]);
        msg.extend(message(0x00031f41, &[fixed(4)])[STREAM_TLV_OFFSET..].to_vec());
        let (values, malformed) = decode_stream_payload_with_status(&msg).expect("decodes");
        assert!(!malformed);
        assert_eq!(values.get(&(0x00031f41, 0, 0)), Some(&1.5));
        assert_eq!(values.get(&(0x00031f41, 0, 1)), Some(&-2.0));
        assert_eq!(values.get(&(0x00031f41, 0, 2)), Some(&3.0));
        assert_eq!(values.get(&(0x00031f41, 1, 0)), Some(&4.0));
        assert_eq!(values.len(), 4);
    }

    #[test]
    fn truncated_entry_sets_malformed_without_panicking() {
        let mut msg = message(0x00031f41, &[fixed(1)]);
        // Append an entry whose declared length overruns the payload.
        msg.push(4);
        msg.extend_from_slice(&u32::MAX.to_be_bytes());
        msg.push(0xaa);
        let (values, malformed) = decode_stream_payload_with_status(&msg).expect("decodes");
        assert!(malformed);
        assert_eq!(values.len(), 1, "entries before the truncation are kept");
        // A payload shorter than the header decodes to nothing.
        let (values, malformed) = decode_stream_payload_with_status(&[0u8; 10]).expect("decodes");
        assert!(values.is_empty());
        assert!(!malformed);
    }
}
