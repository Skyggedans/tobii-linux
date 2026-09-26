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
//! (`tobii_calibration_parse`, 0x1801478f4 and 0x180147937). The Stream
//! Engine reads only the low 32 bits of a status word, as an `i32`: -1 the
//! eye failed at that point, 0 valid but not used, 1 used (0x180147910 and
//! 0x180147950). The device keeps a ring of 14 records (two per target of the
//! 7-point pattern); a new session pushes the oldest out.

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
    /// (`tobii_calibration_parse`, 0x1801478f4). Anything, not a number
    /// included, when the eye failed ([`Self::a_failed`]).
    pub a: [f32; 2],
    /// Status word of `a` (1 on every captured record). The Stream Engine
    /// reads its low 32 bits as an `i32`: 1 used in the calibration, 0 valid
    /// but not used, -1 failed, and it reports any other value as failed
    /// too. [`points`] accepts `-1..=2` there, with the upper 32 bits 0, or
    /// all ones with -1.
    pub a_status: u64,
    /// The measurement the Stream Engine reports as the right eye
    /// (0x180147937). Anything when the eye failed ([`Self::b_failed`]).
    pub b: [f32; 2],
    /// Status word of `b`, read as `a_status` is.
    pub b_status: u64,
}

impl PointRecord {
    /// Whether the Stream Engine reads `a_status` as -1 (its low 32 bits, the
    /// upper half unchecked), the eye failed at this point: `a` may then hold
    /// anything.
    #[must_use]
    pub const fn a_failed(&self) -> bool {
        status(self.a_status) == STATUS_FAILED
    }

    /// Whether the Stream Engine reads `b_status` as -1, as
    /// [`Self::a_failed`] for `a`.
    #[must_use]
    pub const fn b_failed(&self) -> bool {
        status(self.b_status) == STATUS_FAILED
    }
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
    /// A record holds a status word other than `-1..=2` (see
    /// [`PointRecord::a_status`]), or a non-finite or out-of-range value that
    /// is not a failed eye's.
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

/// The status the Stream Engine gives an eye that failed at a point.
const STATUS_FAILED: i32 = -1;

/// A status word as the Stream Engine reads it: its low 32 bits, as an `i32`
/// (`tobii_calibration_parse`, 0x1801478eb and 0x180147941).
#[allow(clippy::cast_possible_truncation)] // reason: the upper half is never read
const fn status(word: u64) -> i32 {
    (word as u32).cast_signed()
}

/// Whether [`points`] lets `word` through: -1, 0 and 1, the statuses the
/// Stream Engine names, and 2, which it reports as failed and this check has
/// always let through, with the upper 32 bits 0. A -1 may also be a 64-bit
/// one, the upper half all ones: no captured record holds a failed eye, so
/// how the tracker writes one is inferred, and both are let through.
fn status_plausible(word: u64) -> bool {
    let upper = word >> 32;
    match status(word) {
        STATUS_FAILED => upper == 0 || upper == 0xffff_ffff,
        0..=2 => upper == 0,
        _ => false,
    }
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
/// exactly at the end of the blob, each status word must read as `-1..=2`
/// with its upper half 0, or all ones with -1 (see [`PointRecord::a_status`]),
/// and every value must be finite and in `-0.5..=1.5`, no more than half a
/// display past its edges, but for the measurement of an eye whose status is
/// -1, failed, which may hold anything.
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
    let on_display = |xy: &[f32; 2]| xy.iter().all(|v| v.is_finite() && (-0.5..=1.5).contains(v));
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
            let plausible = status_plausible(record.a_status)
                && status_plausible(record.b_status)
                && on_display(&record.target)
                && (record.a_failed() || on_display(&record.a))
                && (record.b_failed() || on_display(&record.b));
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

    /// Offsets in a record: the first eye's values and status word, then the
    /// second's.
    const A: usize = 8;
    const A_STATUS: usize = 16;
    const B: usize = 24;
    const B_STATUS: usize = 32;

    /// The embedded blob with `bytes` written at each offset into record `i`.
    fn edited(i: usize, edits: &[(usize, &[u8])]) -> Vec<u8> {
        let mut blob = embedded_blob();
        let total = usize::try_from(header(&blob).expect("header").total).expect("fits");
        let record = total + 8 + i * RECORD_LEN;
        for (offset, bytes) in edits {
            let o = record + offset;
            blob[o..o + bytes.len()].copy_from_slice(bytes);
        }
        blob
    }

    #[test]
    fn a_failed_eye_passes_in_either_encoding_whatever_its_values() {
        let nan = f32::NAN.to_le_bytes();
        let off = 5.0f32.to_le_bytes();
        // -1 as a 32-bit int with a zero upper half, and as a 64-bit one.
        for failed in [0xffff_ffff, u64::MAX] {
            let word = failed.to_le_bytes();
            let first = edited(3, &[(A_STATUS, &word), (A, &nan), (A + 4, &off)]);
            let second = edited(5, &[(B_STATUS, &word), (B, &off)]);

            assert_eq!(validate(&first).map(|info| info.points), Ok(14));
            assert_eq!(validate(&second).map(|info| info.points), Ok(14));
            let record = points(&first).expect("first")[3];
            assert_eq!((record.a_status, record.b_status), (failed, 1));
            assert!(record.a[0].is_nan());
            let record = points(&second).expect("second")[5];
            assert_eq!((record.a_status, record.b_status), (1, failed));
        }
    }

    #[test]
    fn a_value_off_the_display_passes_only_for_a_failed_eye() {
        let nan = f32::NAN.to_le_bytes();
        let failed = u64::MAX.to_le_bytes();
        for (name, blob) in [
            ("a used eye", edited(3, &[(A, &nan)])),
            (
                "an eye valid but not used",
                edited(3, &[(A_STATUS, &0u64.to_le_bytes()), (A, &nan)]),
            ),
            (
                "an eye with status 2",
                edited(3, &[(B_STATUS, &2u64.to_le_bytes()), (B, &nan)]),
            ),
            (
                "the eye that did not fail",
                edited(3, &[(A_STATUS, &failed), (B, &nan)]),
            ),
            (
                "the target of two failed eyes",
                edited(3, &[(A_STATUS, &failed), (B_STATUS, &failed), (0, &nan)]),
            ),
        ] {
            assert_eq!(validate(&blob), Err(BlobError::Record(3)), "{name}");
        }
    }

    #[test]
    fn a_status_word_passes_only_as_minus_one_to_two() {
        for (word, passes) in [
            (0u64, true),
            (1, true),
            (2, true),
            (0xffff_ffff, true),
            (u64::MAX, true),
            (3, false),
            (0xffff_fffe, false),
            (0x8000_0000, false),
            // An upper half only as the sign of a -1.
            (0x1_0000_0001, false),
            (0xffff_ffff_0000_0000, false),
            (0xffff_ffff_0000_0001, false),
            (0x1_ffff_ffff, false),
        ] {
            for at in [A_STATUS, B_STATUS] {
                let blob = edited(3, &[(at, &word.to_le_bytes())]);
                let expected = if passes {
                    Ok(14)
                } else {
                    Err(BlobError::Record(3))
                };
                assert_eq!(
                    validate(&blob).map(|info| info.points),
                    expected,
                    "{word:#x} at +{at}"
                );
            }
        }
    }
}
