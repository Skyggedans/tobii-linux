use anyhow::Result;
use std::collections::BTreeMap;
use std::io::Write;

use crate::sinks::now_us;

pub(crate) const STREAM_TLV_OFFSET: usize = 34;

pub(crate) const GAZE_COORD_MAX: f64 = 1024.0;

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

pub(crate) fn derive_live_values(values: &BTreeMap<(u32, usize, usize), f64>) -> [Option<f64>; 19] {
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
    let head_x = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 5, 0),
            LiveField::new("", 0x00031f41, 6, 0),
            LiveField::new("", 0x00031f41, 9, 0),
        ],
    );
    let head_y = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 5, 1),
            LiveField::new("", 0x00031f41, 6, 1),
            LiveField::new("", 0x00031f41, 9, 1),
        ],
    );
    let head_z = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 5, 2),
            LiveField::new("", 0x00031f41, 6, 2),
            LiveField::new("", 0x00031f41, 9, 2),
        ],
    );
    let head_yaw = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 2, 0),
            LiveField::new("", 0x00031f41, 10, 0),
        ],
    );
    let head_pitch = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 2, 1),
            LiveField::new("", 0x00031f41, 10, 1),
        ],
    );
    let head_roll = mean_keys(
        values,
        &[
            LiveField::new("", 0x00031f41, 2, 2),
            LiveField::new("", 0x00031f41, 10, 2),
        ],
    );
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
