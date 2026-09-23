//! Little-endian field readers and writers shared by the frame codecs.
//!
//! Every frame body is a flat sequence of LE scalars; these two types keep the
//! offset bookkeeping out of the codecs, and make a short body a `None`
//! rather than a panic.

/// Appends LE scalars to a frame body.
#[derive(Debug, Default)]
pub(crate) struct Writer(pub(crate) Vec<u8>);

impl Writer {
    pub(crate) fn with_tag(tag: u8, capacity: usize) -> Self {
        let mut body = Vec::with_capacity(capacity + 1);
        body.push(tag);
        Self(body)
    }

    pub(crate) fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }

    pub(crate) fn bool(&mut self, v: bool) -> &mut Self {
        self.u8(u8::from(v))
    }

    pub(crate) fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn i32(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn i64(&mut self, v: i64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn f32(&mut self, v: f32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn f32s(&mut self, vs: &[f32]) -> &mut Self {
        for v in vs {
            self.f32(*v);
        }
        self
    }

    /// Narrowing to `f32` is the wire format: the C ABI this protocol feeds
    /// carries `float`, and millimetre geometry needs no more than 24 bits.
    #[allow(clippy::cast_possible_truncation)] // reason: f32 is the wire type
    pub(crate) fn f64_as_f32(&mut self, vs: &[f64]) -> &mut Self {
        for v in vs {
            self.f32(*v as f32);
        }
        self
    }

    /// `u16 LE length` + UTF-8, truncated at a char boundary to fit `u16`.
    pub(crate) fn str(&mut self, s: &str) -> &mut Self {
        let mut end = s.len().min(usize::from(u16::MAX));
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        // `end <= u16::MAX` by construction.
        self.u16(u16::try_from(end).unwrap_or(u16::MAX));
        self.0.extend_from_slice(&s.as_bytes()[..end]);
        self
    }

    pub(crate) fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }

    pub(crate) fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

/// Reads LE scalars from a frame body; every read is `None` past the end.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.pos.checked_add(N)?;
        let bytes: [u8; N] = self.buf.get(self.pos..end)?.try_into().ok()?;
        self.pos = end;
        Some(bytes)
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|[b]| b)
    }

    pub(crate) fn bool(&mut self) -> Option<bool> {
        self.u8().map(|b| b != 0)
    }

    pub(crate) fn u16(&mut self) -> Option<u16> {
        self.take().map(u16::from_le_bytes)
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }

    pub(crate) fn i32(&mut self) -> Option<i32> {
        self.take().map(i32::from_le_bytes)
    }

    pub(crate) fn i64(&mut self) -> Option<i64> {
        self.take().map(i64::from_le_bytes)
    }

    pub(crate) fn f32(&mut self) -> Option<f32> {
        self.take().map(f32::from_le_bytes)
    }

    pub(crate) fn f32s<const N: usize>(&mut self) -> Option<[f32; N]> {
        let mut out = [0.0; N];
        for v in &mut out {
            *v = self.f32()?;
        }
        Some(out)
    }

    pub(crate) fn f64s<const N: usize>(&mut self) -> Option<[f64; N]> {
        let mut out = [0.0; N];
        for v in &mut out {
            *v = f64::from(self.f32()?);
        }
        Some(out)
    }

    pub(crate) fn str(&mut self) -> Option<String> {
        let len = usize::from(self.u16()?);
        let end = self.pos.checked_add(len)?;
        let bytes = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Everything not read yet.
    pub(crate) fn rest(&mut self) -> &'a [u8] {
        let rest = self.buf.get(self.pos..).unwrap_or_default();
        self.pos = self.buf.len();
        rest
    }
}
