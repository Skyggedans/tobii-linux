//! Layout of the calibration blob, as read back from the device:
//!
//! ```text
//! +0   u32 LE  total        offset of the point list's 4-byte lead-in
//! +4   u32 LE  110          header size, always 110 so far
//! +8   u32 LE  84           format version, always 84 so far
//! +12  u32 LE  1            profile count
//! +16  u32 LE  0
//! +20  u32 LE  id           the calibration id the device reports
//! +24  u32 LE  total - 88   body length
//! +28  [u8; 8] "human\0\0\0" profile name
//! ...          opaque
//! +total      4 opaque bytes
//! +total+4    u32 LE count
//! +total+8    count x 40-byte point records
//! ```
//!
//! A record is `f32 target_x, target_y, a_x, a_y; u64 a_status; f32 b_x, b_y;
//! u64 b_status`, all little-endian: the stimulus point, then where each eye
//! was measured looking. `a` is the measurement the Stream Engine reports as
//! the left eye and `b` the one it reports as the right
//! (`tobii_calibration_parse`, 0x1801478f4 and 0x180147937). The device keeps
//! a ring of 14 records (two per target of the 7-point pattern); a new session
//! pushes the oldest out.

use std::fmt;

/// Smallest plausible blob: a header and an empty point list.
pub const MIN_BLOB_LEN: usize = 44;
/// Largest plausible blob; the device's are about 660 KB.
pub const MAX_BLOB_LEN: usize = 4 << 20;
/// Size of one point record.
pub const RECORD_LEN: usize = 40;
/// More points than this means the list offset is garbage.
pub const MAX_POINTS: usize = 256;

/// The 7-point pattern the Windows engine calibrates with, in its order,
/// normalised display coordinates.
pub const STIMULUS_POINTS: [[f32; 2]; 7] = [
    [0.5, 0.5],
    [0.5, 0.1],
    [0.9, 0.9],
    [0.1, 0.9],
    [0.1, 0.1],
    [0.9, 0.1],
    [0.5, 0.9],
];

/// The blob's header words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobHeader {
    /// Offset of the point list's lead-in.
    pub total: u32,
    /// Header size (110).
    pub header_len: u32,
    /// Format version (84).
    pub version: u32,
    /// Profile count (1).
    pub profiles: u32,
    /// Calibration id.
    pub id: u32,
    /// Body length (`total - 88`).
    pub body_len: u32,
    /// Profile name, NUL-padded.
    pub name: [u8; 8],
}

/// One calibration point: the target and the two eyes' measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointRecord {
    /// Where the stimulus was, normalised display coordinates.
    pub target: [f32; 2],
    /// The measurement the Stream Engine reports as the left eye
    /// (`tobii_calibration_parse`, 0x1801478f4).
    pub a: [f32; 2],
    /// Status word of `a` (1 on every captured record). The Stream Engine
    /// reads its low 32 bits: 1 used in the calibration, 0 valid but not
    /// used, anything else failed. [`points`] accepts only `0..=2`.
    pub a_status: u64,
    /// The measurement the Stream Engine reports as the right eye
    /// (0x180147937).
    pub b: [f32; 2],
    /// Status word of `b`, read as `a_status` is.
    pub b_status: u64,
}

/// What [`validate`] learned about a blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobInfo {
    /// Calibration id.
    pub id: u32,
    /// Number of point records.
    pub points: usize,
    /// Blob length.
    pub len: usize,
    /// Header size, version and name have the values every captured blob
    /// has; `false` means a format this code has not seen, not an error.
    pub conventional: bool,
}

/// Why a blob was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlobError {
    /// Shorter than a header.
    TooShort(usize),
    /// Longer than any blob the device produces.
    TooLong(usize),
    /// The point list offset or count does not fit the blob.
    PointList,
    /// The point list does not end exactly at the end of the blob.
    Trailer {
        /// Where the list ends.
        expected: usize,
        /// Where the blob ends.
        actual: usize,
    },
    /// A record holds non-finite or out-of-range values.
    Record(usize),
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort(n) => write!(f, "calibration blob too short ({n} bytes)"),
            Self::TooLong(n) => write!(f, "calibration blob too long ({n} bytes)"),
            Self::PointList => f.write_str("calibration point list offset or count out of range"),
            Self::Trailer { expected, actual } => write!(
                f,
                "calibration point list ends at {expected}, blob at {actual}"
            ),
            Self::Record(i) => write!(f, "calibration point {i} is out of range"),
        }
    }
}

impl std::error::Error for BlobError {}

fn u32_at(blob: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        blob.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn u64_at(blob: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        blob.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn f32_at(blob: &[u8], offset: usize) -> Option<f32> {
    u32_at(blob, offset).map(f32::from_bits)
}

/// The header words.
///
/// # Errors
///
/// [`BlobError::TooShort`] / [`BlobError::TooLong`] when the length is
/// implausible.
pub fn header(blob: &[u8]) -> Result<BlobHeader, BlobError> {
    if blob.len() < MIN_BLOB_LEN {
        return Err(BlobError::TooShort(blob.len()));
    }
    if blob.len() > MAX_BLOB_LEN {
        return Err(BlobError::TooLong(blob.len()));
    }
    let word = |o| u32_at(blob, o).ok_or(BlobError::TooShort(blob.len()));
    let mut name = [0u8; 8];
    name.copy_from_slice(&blob[28..36]);
    Ok(BlobHeader {
        total: word(0)?,
        header_len: word(4)?,
        version: word(8)?,
        profiles: word(12)?,
        id: word(20)?,
        body_len: word(24)?,
        name,
    })
}

/// The calibration id (header word at +20).
#[must_use]
pub fn calibration_id(blob: &[u8]) -> Option<u32> {
    header(blob).ok().map(|h| h.id)
}

/// The point records.
///
/// # Errors
///
/// Any [`BlobError`]: the list must sit where the header says and end
/// exactly at the end of the blob, every value must be finite and within a
/// margin of the display, and each status word at most 2.
pub fn points(blob: &[u8]) -> Result<Vec<PointRecord>, BlobError> {
    let h = header(blob)?;
    let list = usize::try_from(h.total).map_err(|_| BlobError::PointList)?;
    let count = u32_at(blob, list.checked_add(4).ok_or(BlobError::PointList)?)
        .and_then(|c| usize::try_from(c).ok())
        .ok_or(BlobError::PointList)?;
    if count > MAX_POINTS {
        return Err(BlobError::PointList);
    }
    let first = list + 8;
    let expected = first + RECORD_LEN * count;
    if expected != blob.len() {
        return Err(BlobError::Trailer {
            expected,
            actual: blob.len(),
        });
    }
    (0..count)
        .map(|i| {
            let o = first + RECORD_LEN * i;
            let f = |k: usize| f32_at(blob, o + 4 * k).ok_or(BlobError::Record(i));
            let record = PointRecord {
                target: [f(0)?, f(1)?],
                a: [f(2)?, f(3)?],
                a_status: u64_at(blob, o + 16).ok_or(BlobError::Record(i))?,
                b: [
                    f32_at(blob, o + 24).ok_or(BlobError::Record(i))?,
                    f32_at(blob, o + 28).ok_or(BlobError::Record(i))?,
                ],
                b_status: u64_at(blob, o + 32).ok_or(BlobError::Record(i))?,
            };
            let values = [record.target, record.a, record.b];
            let plausible = values
                .iter()
                .flatten()
                .all(|v| v.is_finite() && (-0.5..=1.5).contains(v))
                && record.a_status <= 2
                && record.b_status <= 2;
            if plausible {
                Ok(record)
            } else {
                Err(BlobError::Record(i))
            }
        })
        .collect()
}

/// Check a blob before storing or uploading it.
///
/// # Errors
///
/// Whatever [`points`] rejects.
pub fn validate(blob: &[u8]) -> Result<BlobInfo, BlobError> {
    let h = header(blob)?;
    let points = points(blob)?.len();
    Ok(BlobInfo {
        id: h.id,
        points,
        len: blob.len(),
        conventional: h.header_len == 110
            && h.version == 84
            && h.profiles == 1
            && h.body_len == h.total.wrapping_sub(88)
            && h.name == *b"human\0\0\0",
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tobii_proto::calibration::blob_from_payload;
    use tobii_proto::protocol::{parse_init_packets, parse_message};

    /// The calibration the init replay uploads (the author's).
    pub(crate) fn embedded_blob() -> Vec<u8> {
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tobii-usb/init_packets_ep.txt"
        ));
        let packets = parse_init_packets(text).expect("init file");
        let message: Vec<u8> = packets[38..200]
            .iter()
            .flat_map(|p| p.data[8..].iter().copied())
            .collect();
        let prefixed = [&[0u8; 8][..], &message].concat();
        let payload = parse_message(&prefixed).expect("header").payload;
        blob_from_payload(payload).expect("blob").to_vec()
    }

    #[test]
    fn reads_the_embedded_blob() {
        let blob = embedded_blob();

        let info = validate(&blob).expect("valid");
        let records = points(&blob).expect("points");

        assert_eq!(info.id, 1_904_654_973);
        assert_eq!(info.points, 14);
        assert!(info.conventional);
        for (record, target) in records.iter().zip(STIMULUS_POINTS.iter().cycle()) {
            assert!(
                (record.target[0] - target[0]).abs() < 1e-6
                    && (record.target[1] - target[1]).abs() < 1e-6
            );
            assert_eq!((record.a_status, record.b_status), (1, 1));
        }
        // Values from calibration.json (offsets 658563 / 658587).
        assert!((records[1].a[0] - 0.501_088_9).abs() < 1e-6);
        assert!((records[1].b[1] - 0.128_229_7).abs() < 1e-6);
    }

    #[test]
    fn rejects_damaged_blobs() {
        let blob = embedded_blob();
        assert!(matches!(
            validate(&blob[..blob.len() - 1]),
            Err(BlobError::Trailer { .. })
        ));
        assert_eq!(validate(&blob[..10]), Err(BlobError::TooShort(10)));

        let mut bad = blob.clone();
        let total = usize::try_from(header(&blob).expect("header").total).expect("fits");
        let record3 = total + 8 + 3 * RECORD_LEN;
        bad[record3..record3 + 4].copy_from_slice(&5.0f32.to_le_bytes());
        assert_eq!(validate(&bad), Err(BlobError::Record(3)));
    }
}
