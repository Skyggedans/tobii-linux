//! The TLV encoding every command, response, notification and stream message
//! carries after its 32-byte header.
//!
//! A payload is `00 00` followed by entries of `u8 type`, `u32 BE length`,
//! value. The types seen on the wire:
//!
//! | type | value |
//! |---|---|
//! | `0x01`, `0x02`, `0x1a` | `u32` BE (0x01 carries the presence state, 0x1a a trailer word) |
//! | `0x03` | 16.16 fixed point, signed |
//! | `0x04` | 32.32 fixed point, signed; lengths are in 1/1024 mm |
//! | `0x05` | a field id; the entries that follow are its components |
//! | `0x06` | `u64` BE (the device timestamp, µs) |
//! | `0x14` | string: `u32` BE length + bytes |
//! | `0x15` | bytes: `u32` BE length + bytes |
//! | `0x17` | a list header (`u32` BE count, then the items) |
//!
//! Stream messages are **keyed**: each field is announced by
//! `05 KEY_FIELD_ID` plus a `02` key, then either one scalar or a `05 <id>`
//! point with its fixed-point components. [`keyed_fields`] resolves that into
//! a lookup by key, so a decoder names fields by what they are rather than by
//! the order they happen to arrive in.

/// `u32` BE, used by the presence state.
pub const TYPE_U32_ALT: u8 = 0x01;
/// `u32` BE.
pub const TYPE_U32: u8 = 0x02;
/// 16.16 signed fixed point.
pub const TYPE_FIXED16: u8 = 0x03;
/// 32.32 signed fixed point.
pub const TYPE_FIXED32: u8 = 0x04;
/// A field id announcing the entries that follow.
pub const TYPE_FIELD_ID: u8 = 0x05;
/// `u64` BE.
pub const TYPE_U64: u8 = 0x06;
/// Length-prefixed string.
pub const TYPE_STRING: u8 = 0x14;
/// Length-prefixed bytes.
pub const TYPE_BYTES: u8 = 0x15;
/// List header.
pub const TYPE_LIST: u8 = 0x17;
/// `u32` BE trailer word.
pub const TYPE_U32_TRAILER: u8 = 0x1a;

/// One unit of 32.32 fixed point.
pub const FIXED32_ONE: f64 = 4_294_967_296.0;
/// One unit of 16.16 fixed point.
pub const FIXED16_ONE: f64 = 65_536.0;
/// The device's length unit: 1/1024 mm.
pub const UNITS_PER_MM: f64 = 1024.0;
/// Field id that announces a keyed field in a stream message.
pub const KEY_FIELD_ID: u32 = 0x0002_0bb9;
/// Field id of a 3-D point (tracker or display frame, 1/1024 mm).
pub const FIELD_POINT_3D: u32 = 0x0003_1f41;
/// Field id of a 2-D display point (x1024, normalised).
pub const FIELD_POINT_2D: u32 = 0x0002_1f40;

/// One TLV entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tlv<'a> {
    /// The type byte.
    pub typ: u8,
    /// The raw value, exactly the declared length.
    pub value: &'a [u8],
}

impl<'a> Tlv<'a> {
    /// A 4-byte BE integer (types 0x01, 0x02, 0x1a).
    #[must_use]
    pub fn u32(&self) -> Option<u32> {
        Some(u32::from_be_bytes(self.value.try_into().ok()?))
    }

    /// An 8-byte BE integer (type 0x06).
    #[must_use]
    pub fn u64(&self) -> Option<u64> {
        Some(u64::from_be_bytes(self.value.try_into().ok()?))
    }

    /// A field id (type 0x05).
    #[must_use]
    pub fn field_id(&self) -> Option<u32> {
        (self.typ == TYPE_FIELD_ID).then(|| self.u32()).flatten()
    }

    /// The raw 32.32 value (type 0x04).
    #[must_use]
    pub fn fixed32_raw(&self) -> Option<i64> {
        (self.typ == TYPE_FIXED32).then(|| Some(i64::from_be_bytes(self.value.try_into().ok()?)))?
    }

    /// A 32.32 value as `f64`: exact, the integer parts on this wire stay far
    /// below 2^21.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // reason: exact for the device's value range
    pub fn fixed32(&self) -> Option<f64> {
        self.fixed32_raw().map(|raw| raw as f64 / FIXED32_ONE)
    }

    /// A 16.16 value as `f64` (type 0x03).
    #[must_use]
    pub fn fixed16(&self) -> Option<f64> {
        (self.typ == TYPE_FIXED16).then(|| {
            Some(f64::from(i32::from_be_bytes(self.value.try_into().ok()?)) / FIXED16_ONE)
        })?
    }

    fn length_prefixed(&self) -> Option<&'a [u8]> {
        let (len, rest) = self.value.split_first_chunk::<4>()?;
        rest.get(..usize::try_from(u32::from_be_bytes(*len)).ok()?)
    }

    /// A string (type 0x14), lossily decoded.
    #[must_use]
    pub fn string(&self) -> Option<String> {
        (self.typ == TYPE_STRING).then(|| {
            self.length_prefixed()
                .map(|b| String::from_utf8_lossy(b).into_owned())
        })?
    }

    /// A byte string (type 0x15).
    #[must_use]
    pub fn bytes(&self) -> Option<&'a [u8]> {
        (self.typ == TYPE_BYTES).then(|| self.length_prefixed())?
    }
}

/// Iterates the entries of a TLV list; stops at the first entry whose declared
/// length runs past the end, and records that in [`TlvIter::truncated`].
#[derive(Debug, Clone)]
pub struct TlvIter<'a> {
    buf: &'a [u8],
    pos: usize,
    truncated: bool,
}

impl<'a> TlvIter<'a> {
    /// Iterate `list`, which starts directly with an entry (no `00 00`).
    #[must_use]
    pub fn new(list: &'a [u8]) -> Self {
        Self {
            buf: list,
            pos: 0,
            truncated: false,
        }
    }

    /// Whether iteration stopped at an entry that ran past the end.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

impl<'a> Iterator for TlvIter<'a> {
    type Item = Tlv<'a>;

    fn next(&mut self) -> Option<Tlv<'a>> {
        let rest = self.buf.get(self.pos..)?;
        if rest.len() < 5 {
            if !rest.is_empty() {
                self.truncated = true;
                self.pos = self.buf.len();
            }
            return None;
        }
        let typ = rest[0];
        let declared = u32::from_be_bytes([rest[1], rest[2], rest[3], rest[4]]);
        let len = usize::try_from(declared).unwrap_or(usize::MAX);
        let Some(value) = rest.get(5..).and_then(|v| v.get(..len)) else {
            self.truncated = true;
            self.pos = self.buf.len();
            return None;
        };
        self.pos += 5 + len;
        Some(Tlv { typ, value })
    }
}

/// Iterate the TLV list of a message payload that begins with `00 00`.
#[must_use]
pub fn payload_tlvs(payload: &[u8]) -> TlvIter<'_> {
    TlvIter::new(payload.get(2..).unwrap_or_default())
}

/// Builds a TLV payload, including its leading `00 00`.
#[derive(Debug, Clone)]
pub struct TlvWriter(Vec<u8>);

impl Default for TlvWriter {
    fn default() -> Self {
        Self(vec![0, 0])
    }
}

impl TlvWriter {
    /// An empty payload (`00 00`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn entry(&mut self, typ: u8, value: &[u8]) -> &mut Self {
        self.0.push(typ);
        // Values here are at most a calibration blob (< 4 MiB).
        let len = u32::try_from(value.len()).unwrap_or(u32::MAX);
        self.0.extend_from_slice(&len.to_be_bytes());
        self.0.extend_from_slice(value);
        self
    }

    /// A `u32` (type 0x02).
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.entry(TYPE_U32, &v.to_be_bytes())
    }

    /// A field id (type 0x05).
    pub fn field_id(&mut self, id: u32) -> &mut Self {
        self.entry(TYPE_FIELD_ID, &id.to_be_bytes())
    }

    /// A raw 32.32 value (type 0x04).
    pub fn fixed32_raw(&mut self, raw: i64) -> &mut Self {
        self.entry(TYPE_FIXED32, &raw.to_be_bytes())
    }

    /// A length in millimetres as 32.32 fixed point in 1/1024 mm.
    #[allow(clippy::cast_possible_truncation)] // reason: rounded and range-checked by the device
    pub fn mm(&mut self, mm: f64) -> &mut Self {
        self.fixed32_raw((mm * UNITS_PER_MM * FIXED32_ONE).round() as i64)
    }

    /// A 3-D point in millimetres: `05 FIELD_POINT_3D` + three components.
    pub fn point_mm(&mut self, p: [f64; 3]) -> &mut Self {
        self.field_id(FIELD_POINT_3D).mm(p[0]).mm(p[1]).mm(p[2])
    }

    /// A byte string (type 0x15).
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        let mut value = Vec::with_capacity(4 + b.len());
        value.extend_from_slice(&u32::try_from(b.len()).unwrap_or(u32::MAX).to_be_bytes());
        value.extend_from_slice(b);
        self.entry(TYPE_BYTES, &value)
    }

    /// The payload.
    #[must_use]
    pub fn finish(&self) -> Vec<u8> {
        self.0.clone()
    }
}

/// A keyed field of a stream message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyedValue {
    /// A scalar: `u32` (types 0x01/0x02), 16.16 (0x03, raw bits) or `u64` (0x06).
    Scalar {
        /// Its wire type.
        typ: u8,
        /// Its bits, zero-extended.
        bits: u64,
    },
    /// A point: a field id and its 32.32 components (raw, unscaled).
    Point {
        /// The point's field id ([`FIELD_POINT_3D`] or [`FIELD_POINT_2D`]).
        id: u32,
        /// Components; only the first `len` are meaningful.
        raw: [i64; 3],
        /// Number of components.
        len: usize,
    },
}

/// Highest key a stream message is expected to use, plus one.
pub const MAX_KEY: usize = 0x40;

/// The fields of a keyed stream message, looked up by key.
#[derive(Debug, Clone)]
pub struct KeyedFields([Option<KeyedValue>; MAX_KEY]);

impl KeyedFields {
    /// The field with `key`, if present.
    #[must_use]
    pub fn get(&self, key: u32) -> Option<KeyedValue> {
        *self.0.get(usize::try_from(key).ok()?)?
    }

    /// A scalar field's bits.
    #[must_use]
    pub fn scalar(&self, key: u32) -> Option<u64> {
        match self.get(key)? {
            KeyedValue::Scalar { bits, .. } => Some(bits),
            KeyedValue::Point { .. } => None,
        }
    }

    /// A 16.16 scalar field as `f64`.
    #[must_use]
    pub fn fixed16(&self, key: u32) -> Option<f64> {
        match self.get(key)? {
            KeyedValue::Scalar {
                typ: TYPE_FIXED16,
                bits,
            } => Some(
                f64::from(i32::from_be_bytes(u32::try_from(bits).ok()?.to_be_bytes()))
                    / FIXED16_ONE,
            ),
            _ => None,
        }
    }

    /// A point field's components in wire units (32.32 decoded, not scaled).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // reason: exact for the device's value range
    pub fn point<const N: usize>(&self, key: u32) -> Option<[f64; N]> {
        match self.get(key)? {
            KeyedValue::Point { raw, len, .. } if len >= N => {
                let mut out = [0.0; N];
                for (o, r) in out.iter_mut().zip(raw) {
                    *o = r as f64 / FIXED32_ONE;
                }
                Some(out)
            }
            _ => None,
        }
    }
}

/// Resolve a keyed stream message's TLV list (after `00 00`) into fields by
/// key. Unkeyed entries and keys past [`MAX_KEY`] are ignored.
#[must_use]
pub fn keyed_fields(list: &[u8]) -> KeyedFields {
    let mut fields = KeyedFields([None; MAX_KEY]);
    let mut entries = TlvIter::new(list).peekable();
    while let Some(entry) = entries.next() {
        if entry.field_id() != Some(KEY_FIELD_ID) {
            continue;
        }
        let Some(key) = entries.next().and_then(|k| k.u32()) else {
            continue;
        };
        let Some(value) = entries.next() else { break };
        let parsed = if let Some(id) = value.field_id() {
            let mut raw = [0i64; 3];
            let mut len = 0;
            while let Some(component) = entries.peek().and_then(Tlv::fixed32_raw) {
                entries.next();
                if let Some(slot) = raw.get_mut(len) {
                    *slot = component;
                }
                len += 1;
            }
            KeyedValue::Point {
                id,
                raw,
                len: len.min(3),
            }
        } else {
            let bits = match value.value.len() {
                4 => value.u32().map(u64::from),
                8 => value.u64(),
                _ => None,
            };
            let Some(bits) = bits else { continue };
            KeyedValue::Scalar {
                typ: value.typ,
                bits,
            }
        };
        if let Some(slot) = usize::try_from(key).ok().and_then(|k| fields.0.get_mut(k)) {
            *slot = Some(parsed);
        }
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(typ: u8, value: &[u8]) -> Vec<u8> {
        let mut v = vec![typ];
        v.extend_from_slice(&u32::try_from(value.len()).unwrap_or(0).to_be_bytes());
        v.extend_from_slice(value);
        v
    }

    #[test]
    fn iterates_entries_and_flags_truncation() {
        let mut list = entry(TYPE_U32, &7u32.to_be_bytes());
        list.extend(entry(TYPE_U64, &9u64.to_be_bytes()));
        let got: Vec<_> = TlvIter::new(&list).collect();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].u32(), Some(7));
        assert_eq!(got[1].u64(), Some(9));

        list.extend([TYPE_U32, 0, 0, 0, 9, 1, 2]);
        let mut it = TlvIter::new(&list);
        assert_eq!(it.by_ref().count(), 2);
        assert!(it.truncated());
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: exact fixed-point values
    fn decodes_strings_bytes_and_fixed_point() {
        let mut s = 3u32.to_be_bytes().to_vec();
        s.extend_from_slice(b"IS5");
        let t = Tlv {
            typ: TYPE_STRING,
            value: &s,
        };
        assert_eq!(t.string().as_deref(), Some("IS5"));
        assert_eq!(t.bytes(), None, "type mismatch");

        let fx16 = (184u32 << 16).to_be_bytes();
        assert_eq!(
            Tlv {
                typ: TYPE_FIXED16,
                value: &fx16
            }
            .fixed16(),
            Some(184.0)
        );

        let fx32 = (-(3i64 << 31)).to_be_bytes();
        assert_eq!(
            Tlv {
                typ: TYPE_FIXED32,
                value: &fx32
            }
            .fixed32(),
            Some(-1.5)
        );
    }

    #[test]
    fn writer_round_trips_a_point() {
        let payload = TlvWriter::new()
            .point_mm([-1.5, 2.0, 0.25])
            .u32(12345)
            .finish();
        assert_eq!(&payload[..2], &[0, 0]);
        let entries: Vec<_> = payload_tlvs(&payload).collect();
        assert_eq!(entries[0].field_id(), Some(FIELD_POINT_3D));
        let mm: Vec<f64> = entries[1..4]
            .iter()
            .filter_map(Tlv::fixed32)
            .map(|v| v / UNITS_PER_MM)
            .collect();
        assert_eq!(mm, vec![-1.5, 2.0, 0.25]);
        assert_eq!(entries[4].u32(), Some(12345));
    }

    #[test]
    fn keyed_fields_resolve_scalars_and_points() {
        let mut list = Vec::new();
        list.extend(entry(TYPE_FIELD_ID, &KEY_FIELD_ID.to_be_bytes()));
        list.extend(entry(TYPE_U32, &0x14u32.to_be_bytes()));
        list.extend(entry(TYPE_U32, &43780u32.to_be_bytes()));
        list.extend(entry(TYPE_FIELD_ID, &KEY_FIELD_ID.to_be_bytes()));
        list.extend(entry(TYPE_U32, &0x1cu32.to_be_bytes()));
        list.extend(entry(TYPE_FIELD_ID, &FIELD_POINT_2D.to_be_bytes()));
        list.extend(entry(TYPE_FIXED32, &(512i64 << 32).to_be_bytes()));
        list.extend(entry(TYPE_FIXED32, &(256i64 << 32).to_be_bytes()));

        let fields = keyed_fields(&list);
        assert_eq!(fields.scalar(0x14), Some(43780));
        assert_eq!(fields.point::<2>(0x1c), Some([512.0, 256.0]));
        assert_eq!(fields.point::<3>(0x1c), None, "only two components");
        assert_eq!(fields.get(0x02), None);
    }
}
