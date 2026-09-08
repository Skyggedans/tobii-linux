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
//! the pcap_analysis replication set); the iris positions in these frames match
//! the projected 0x83 eyeball centres to <1 px with f≈376 px (280-px frame).

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImageFrame {
    /// Device timestamp, microseconds (same clock as the gaze stream's key 1).
    pub(crate) device_ts_us: u64,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Row-major 8-bit grayscale, `width * height` bytes (stride removed).
    pub(crate) pixels: Vec<u8>,
}

fn be_u32(b: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(..4)?.try_into().ok()?))
}

/// Decode one 0x50e stream message into a frame; `None` if it is not a
/// well-formed image message.
pub(crate) fn decode_image_payload(payload: &[u8]) -> Option<ImageFrame> {
    let mut off = STREAM_TLV_OFFSET;
    let mut key = None::<u32>;
    let mut ts = None;
    let mut bpp = None;
    let mut width = None;
    let mut height = None;
    let mut stride = None;
    let mut blob: Option<&[u8]> = None;

    while off + 5 <= payload.len() {
        let typ = payload[off];
        let len = be_u32(&payload[off + 1..])? as usize;
        off += 5;
        let value = payload.get(off..off + len)?;
        off += len;
        match typ {
            5 if len == 4 => {
                // A field id. Only 0x20bb9 announces a keyed field; the key
                // itself is the next type-2 entry.
                if be_u32(value)? == KEY_ID {
                    key = Some(u32::MAX); // expect the key next
                } else {
                    key = None;
                }
            }
            2 if len == 4 => {
                let v = be_u32(value)?;
                match key {
                    Some(u32::MAX) => key = Some(v),
                    Some(KEY_BPP) => bpp = Some(v),
                    Some(KEY_WIDTH) => width = Some(v),
                    Some(KEY_HEIGHT) => height = Some(v),
                    Some(KEY_STRIDE) => stride = Some(v),
                    _ => {}
                }
            }
            6 if len == 8 => {
                if key == Some(KEY_TIMESTAMP) {
                    ts = Some(u64::from_be_bytes(value.try_into().ok()?));
                }
            }
            0x15 => {
                if key == Some(KEY_PIXELS) && len >= 4 {
                    let n = be_u32(value)? as usize;
                    blob = value.get(4..4 + n);
                }
            }
            _ => {}
        }
    }

    let (width, height) = (width?, height?);
    if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM || bpp? != 8 {
        return None;
    }
    let stride = stride.unwrap_or(width) as usize;
    let (w, h) = (width as usize, height as usize);
    let blob = blob?;
    if stride < w || blob.len() < stride * (h - 1) + w {
        return None;
    }
    let pixels = if stride == w {
        blob[..w * h].to_vec()
    } else {
        let mut px = Vec::with_capacity(w * h);
        for row in 0..h {
            px.extend_from_slice(&blob[row * stride..row * stride + w]);
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

/// 2x nearest-neighbour upscale (280x280 -> 560x560), so the frame can go
/// through the same face crop / landmark pipeline as the UVC camera path.
pub(crate) fn upscale2x(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    let ow = w * 2;
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        let o0 = (2 * y) * ow;
        let o1 = o0 + ow;
        for (x, &v) in row.iter().enumerate() {
            out[o0 + 2 * x] = v;
            out[o0 + 2 * x + 1] = v;
            out[o1 + 2 * x] = v;
            out[o1 + 2 * x + 1] = v;
        }
    }
    out
}

/// Write a frame as a binary PGM.
pub(crate) fn write_pgm(path: &str, frame: &ImageFrame) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "P5\n{} {}\n255\n", frame.width, frame.height)?;
    f.write_all(&frame.pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tlv(typ: u8, value: &[u8]) -> Vec<u8> {
        let mut v = vec![typ];
        v.extend_from_slice(&(value.len() as u32).to_be_bytes());
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
    pub(crate) fn synth_message(w: u32, h: u32, stride: u32, ts: u64, pixels: &[u8]) -> Vec<u8> {
        let mut body = tlv(5, &0x60bb8u32.to_be_bytes());
        body.extend(keyed(KEY_TIMESTAMP, tlv(6, &ts.to_be_bytes())));
        body.extend(keyed(KEY_BPP, tlv(2, &8u32.to_be_bytes())));
        body.extend(keyed(KEY_WIDTH, tlv(2, &w.to_be_bytes())));
        body.extend(keyed(KEY_HEIGHT, tlv(2, &h.to_be_bytes())));
        body.extend(keyed(KEY_STRIDE, tlv(2, &stride.to_be_bytes())));
        let mut blob = (pixels.len() as u32).to_be_bytes().to_vec();
        blob.extend_from_slice(pixels);
        body.extend(keyed(KEY_PIXELS, tlv(0x15, &blob)));

        let total = STREAM_TLV_OFFSET + body.len();
        let mut msg = vec![1, 0, 0, 0];
        msg.extend_from_slice(&(total as u32).to_le_bytes());
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
        let f = decode_image_payload(&msg).unwrap();
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
        assert!(decode_image_payload(&msg).is_some(), "last row needs only `width` bytes");
    }

    #[test]
    fn upscale_doubles_each_pixel() {
        let src = vec![1, 2, 3, 4];
        let out = upscale2x(&src, 2, 2);
        assert_eq!(out, vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    }

    /// Optional check against a real captured message (contains the user's
    /// face, so it is not committed): set TOBII_IMAGE83_FIXTURE to its path.
    #[test]
    fn decodes_real_capture_if_available() {
        let Ok(path) = std::env::var("TOBII_IMAGE83_FIXTURE") else {
            return;
        };
        let msg = std::fs::read(path).unwrap();
        assert_eq!(msg.len(), 78609);
        let f = decode_image_payload(&msg).expect("real message decodes");
        assert_eq!((f.width, f.height), (280, 280));
        assert_eq!(f.pixels.len(), 280 * 280);
        assert!(f.device_ts_us > 0);
        let mean = f.pixels.iter().map(|&p| p as f64).sum::<f64>() / f.pixels.len() as f64;
        assert!(mean > 5.0 && mean < 200.0, "plausible IR exposure, mean {mean}");
    }
}
