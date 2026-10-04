//! Decoder for the device's `primary_camera_image` stream (id 0x50e) that the
//! firmware multiplexes on EP 0x83 next to the gaze stream: a 280x280 8-bit IR
//! frame from the tracker camera at ~33 Hz. Unlike the UVC camera on interface
//! 2, this stream runs *concurrently* with the gaze pipeline, so it is what
//! gives gaze-independent head yaw/pitch while gaze is being tracked.
//!
//! Message layout (after the common 34-byte stream header): a TLV list where
//! each field is announced by id `0x20bb9` + a type-2 key, then the value:
//!   key 1 -> type 6 u64 device timestamp (µs)
//!   key 2 -> type 2 u32 bits per pixel (8)
//!   key 3 -> type 2 u32 width (280)
//!   key 4 -> type 2 u32 height (280)
//!   key 5 -> type 2 u32 stride (280)
//!   key 6 -> type 0x15 blob: BE u32 pixel-count + row-major pixels
//! Verified against the Windows Stream Engine captures (session1-3.pcapng and
//! the `pcap_analysis` replication set); the iris positions in these frames match
//! the projected 0x83 eyeball centres to <1 px with f≈376 px (280-px frame).

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::decode::STREAM_TLV_OFFSET;

const KEY_ID: u32 = 0x20bb9;
const KEY_TIMESTAMP: u32 = 1;
const KEY_BPP: u32 = 2;
const KEY_WIDTH: u32 = 3;
const KEY_HEIGHT: u32 = 4;
const KEY_STRIDE: u32 = 5;
const KEY_PIXELS: u32 = 6;

/// Largest frame we accept (guards allocation on a corrupt header).
const MAX_DIM: u32 = 4096;

/// One decoded camera frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageFrame {
    /// Device timestamp, microseconds (same clock as the gaze stream's key 1).
    pub device_ts_us: u64,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Row-major 8-bit grayscale, `width * height` bytes (stride removed).
    pub pixels: Vec<u8>,
}

/// Which keyed field the next value entry belongs to, while walking the TLVs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyState {
    /// Not inside a `0x20bb9` field; values are ignored.
    None,
    /// Saw the `0x20bb9` field id; the next type-2 entry is the key.
    Expecting,
    /// Inside the keyed field `key`; the next entry is its value.
    Key(u32),
}

fn be_u32(b: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(..4)?.try_into().ok()?))
}

/// Decode one 0x50e stream message into a frame; `None` if it is not a
/// well-formed image message.
#[must_use]
pub fn decode_image_payload(payload: &[u8]) -> Option<ImageFrame> {
    let mut off = STREAM_TLV_OFFSET;
    let mut key = KeyState::None;
    let mut ts = None;
    let mut bpp = None;
    let mut width = None;
    let mut height = None;
    let mut stride = None;
    let mut blob: Option<&[u8]> = None;

    while off + 5 <= payload.len() {
        let typ = payload[off];
        let len = usize::try_from(be_u32(&payload[off + 1..])?).ok()?;
        off += 5;
        let end = off.checked_add(len)?;
        let value = payload.get(off..end)?;
        off = end;
        match typ {
            5 if len == 4 => {
                // A field id. Only 0x20bb9 announces a keyed field; the key
                // itself is the next type-2 entry.
                key = if be_u32(value)? == KEY_ID {
                    KeyState::Expecting
                } else {
                    KeyState::None
                };
            }
            2 if len == 4 => {
                let v = be_u32(value)?;
                match key {
                    KeyState::Expecting => key = KeyState::Key(v),
                    KeyState::Key(KEY_BPP) => bpp = Some(v),
                    KeyState::Key(KEY_WIDTH) => width = Some(v),
                    KeyState::Key(KEY_HEIGHT) => height = Some(v),
                    KeyState::Key(KEY_STRIDE) => stride = Some(v),
                    KeyState::None | KeyState::Key(_) => {}
                }
            }
            6 if len == 8 && key == KeyState::Key(KEY_TIMESTAMP) => {
                ts = Some(u64::from_be_bytes(value.try_into().ok()?));
            }
            0x15 if key == KeyState::Key(KEY_PIXELS) && len >= 4 => {
                let n = usize::try_from(be_u32(value)?).ok()?;
                blob = value.get(4..).and_then(|px| px.get(..n));
            }
            _ => {}
        }
    }

    let (width, height) = (width?, height?);
    if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM || bpp? != 8 {
        return None;
    }
    let stride = usize::try_from(stride.unwrap_or(width)).ok()?;
    let (w, h) = (usize::try_from(width).ok()?, usize::try_from(height).ok()?);
    let blob = blob?;
    // `w`, `h` <= MAX_DIM, so the products below stay far below usize::MAX;
    // `stride` is unbounded on the wire, hence checked.
    let needed = stride.checked_mul(h - 1)?.checked_add(w)?;
    if stride < w || blob.len() < needed {
        return None;
    }
    let pixels = if stride == w {
        blob[..w * h].to_vec()
    } else {
        let mut px = Vec::with_capacity(w * h);
        // The last chunk may be shorter than `stride` but holds >= `w` bytes
        // (checked above).
        for row in blob.chunks(stride).take(h) {
            px.extend_from_slice(&row[..w]);
        }
        px
    };
    Some(ImageFrame {
        device_ts_us: ts.unwrap_or(0),
        width: w,
        height: h,
        pixels,
    })
}

/// 2x nearest-neighbour upscale (280x280 -> 560x560): an allocating
/// wrapper around [`upscale2x_into`], for the tests.
///
/// # Panics
///
/// Panics if `src` holds fewer than `w * h` bytes.
#[cfg(test)]
#[must_use]
pub fn upscale2x(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::new();
    upscale2x_into(src, w, h, &mut out);
    out
}

/// 2x nearest-neighbour upscale writing into `out`, which is resized to
/// `w * h * 4` bytes: each pixel becomes a 2x2 block, and a vector handed
/// back keeps its allocation.
///
/// This is how the 0x50e stream's frames used to be enlarged to 560x560 for
/// the head tracker. No per-frame path calls it any more: `tobii-pose`'s
/// tracker takes the 280x280 frames as they come and enlarges them itself,
/// and its tests check that it does so exactly as this function does.
///
/// # Panics
///
/// Panics if `src` holds fewer than `w * h` bytes.
pub fn upscale2x_into(src: &[u8], w: usize, h: usize, out: &mut Vec<u8>) {
    assert!(
        src.len() >= w * h,
        "upscale2x: source has {} bytes, need {}",
        src.len(),
        w * h
    );
    out.clear();
    if w == 0 || h == 0 {
        return;
    }
    let ow = w * 2;
    out.resize(ow * h * 2, 0);
    for (src_row, out_rows) in src[..w * h]
        .chunks_exact(w)
        .zip(out.chunks_exact_mut(ow * 2))
    {
        let (r0, r1) = out_rows.split_at_mut(ow);
        for (pair, &v) in r0.as_chunks_mut::<2>().0.iter_mut().zip(src_row) {
            *pair = [v; 2];
        }
        r1.copy_from_slice(r0);
    }
}

/// Write a frame as a binary PGM.
///
/// # Errors
///
/// Propagates file creation and write failures.
pub fn write_pgm(path: impl AsRef<Path>, frame: &ImageFrame) -> std::io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P5\n{} {}\n255\n", frame.width, frame.height)?;
    f.write_all(&frame.pixels)?;
    f.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tlv(typ: u8, value: &[u8]) -> Vec<u8> {
        let mut v = vec![typ];
        v.extend_from_slice(&u32::try_from(value.len()).expect("fits u32").to_be_bytes());
        v.extend_from_slice(value);
        v
    }

    fn keyed(key: u32, value: Vec<u8>) -> Vec<u8> {
        let mut v = tlv(5, &KEY_ID.to_be_bytes());
        v.extend(tlv(2, &key.to_be_bytes()));
        v.extend(value);
        v
    }

    /// Synthesize an image message the way the firmware lays it out.
    fn synth_message(w: u32, h: u32, stride: u32, ts: u64, pixels: &[u8]) -> Vec<u8> {
        let mut body = tlv(5, &0x60bb8u32.to_be_bytes());
        body.extend(keyed(KEY_TIMESTAMP, tlv(6, &ts.to_be_bytes())));
        body.extend(keyed(KEY_BPP, tlv(2, &8u32.to_be_bytes())));
        body.extend(keyed(KEY_WIDTH, tlv(2, &w.to_be_bytes())));
        body.extend(keyed(KEY_HEIGHT, tlv(2, &h.to_be_bytes())));
        body.extend(keyed(KEY_STRIDE, tlv(2, &stride.to_be_bytes())));
        let mut blob = u32::try_from(pixels.len())
            .expect("fits u32")
            .to_be_bytes()
            .to_vec();
        blob.extend_from_slice(pixels);
        body.extend(keyed(KEY_PIXELS, tlv(0x15, &blob)));

        let total = STREAM_TLV_OFFSET + body.len();
        let mut msg = vec![1, 0, 0, 0];
        msg.extend_from_slice(&u32::try_from(total).expect("fits u32").to_le_bytes());
        msg.extend_from_slice(&0x53u32.to_be_bytes());
        msg.extend_from_slice(&[0; 8]);
        msg.extend_from_slice(&0x50eu32.to_be_bytes());
        msg.resize(STREAM_TLV_OFFSET, 0);
        msg.extend(body);
        msg
    }

    #[test]
    fn decodes_synthetic_frame() {
        let px: Vec<u8> = (0..12).collect();
        let msg = synth_message(4, 3, 4, 1_443_372_769, &px);
        assert_eq!(crate::protocol::stream_id(&msg), Some(0x50e));
        let f = decode_image_payload(&msg).expect("decodes");
        assert_eq!(f.width, 4);
        assert_eq!(f.height, 3);
        assert_eq!(f.device_ts_us, 1_443_372_769);
        assert_eq!(f.pixels, px);
    }

    #[test]
    fn strips_stride_padding() {
        // 3 wide, stride 4, 2 rows: pad byte 0xff after each row.
        let px = vec![1, 2, 3, 0xff, 4, 5, 6, 0xff];
        let msg = synth_message(3, 2, 4, 7, &px);
        let f = decode_image_payload(&msg).expect("decodes");
        assert_eq!(f.pixels, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rejects_truncated_blob_and_bad_dims() {
        let px = vec![0u8; 12];
        let mut msg = synth_message(4, 3, 4, 0, &px);
        msg.truncate(msg.len() - 5);
        assert!(decode_image_payload(&msg).is_none(), "short blob");
        let msg = synth_message(0, 3, 0, 0, &[]);
        assert!(decode_image_payload(&msg).is_none(), "zero width");
    }

    #[test]
    fn rejects_pixel_count_smaller_than_frame() {
        // BE count 10 < w*h = 12: the blob is intact at TLV level, so this is
        // the `blob.len() < stride*(h-1)+w` guard, not TLV truncation.
        let msg = synth_message(4, 3, 4, 0, &[0u8; 10]);
        assert!(decode_image_payload(&msg).is_none());
        // With stride padding: 3 wide, stride 4, 2 rows needs 7 bytes; 6 fails.
        let msg = synth_message(3, 2, 4, 0, &[0u8; 6]);
        assert!(decode_image_payload(&msg).is_none());
        let msg = synth_message(3, 2, 4, 0, &[0u8; 7]);
        assert!(
            decode_image_payload(&msg).is_some(),
            "last row needs only `width` bytes"
        );
    }

    #[test]
    fn rejects_overlong_tlv_and_huge_stride_without_panicking() {
        // A TLV whose declared length overruns the message: None, no panic.
        let mut msg = synth_message(4, 3, 4, 0, &[0u8; 12]);
        msg.extend(tlv(2, &[0, 0, 0, 0]));
        let n = msg.len();
        msg[n - 8..n - 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(decode_image_payload(&msg).is_none());
        // A stride far larger than the blob must fail the size check cleanly.
        let msg = synth_message(4, 3, u32::MAX, 0, &[0u8; 12]);
        assert!(decode_image_payload(&msg).is_none());
    }

    #[test]
    fn upscale_into_reuses_the_buffer() {
        let mut out = vec![0xffu8; 999];
        upscale2x_into(&[1, 2, 3, 4], 2, 2, &mut out);
        assert_eq!(out, upscale2x(&[1, 2, 3, 4], 2, 2));
        upscale2x_into(&[], 0, 0, &mut out);
        assert!(out.is_empty(), "a zero-sized frame empties the buffer");
    }

    #[test]
    fn upscale_doubles_each_pixel() {
        let src = vec![1, 2, 3, 4];
        let out = upscale2x(&src, 2, 2);
        assert_eq!(out, vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    }

    /// Optional check against a real captured message (contains the user's
    /// face, so it is not committed): set `TOBII_IMAGE83_FIXTURE` to its path.
    #[test]
    fn decodes_real_capture_if_available() {
        let Ok(path) = std::env::var("TOBII_IMAGE83_FIXTURE") else {
            return;
        };
        let msg = std::fs::read(path).expect("fixture readable");
        assert_eq!(msg.len(), 78609);
        let f = decode_image_payload(&msg).expect("real message decodes");
        assert_eq!((f.width, f.height), (280, 280));
        assert_eq!(f.pixels.len(), 280 * 280);
        assert!(f.device_ts_us > 0);
        let mean = f.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / f.pixels.len() as f64;
        assert!(
            mean > 5.0 && mean < 200.0,
            "plausible IR exposure, mean {mean}"
        );
    }
}
