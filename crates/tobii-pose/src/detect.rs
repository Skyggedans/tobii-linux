//! Face detection on the camera's own frame, for the tracker to find a face it
//! has lost: `MediaPipe`'s `BlazeFace` short-range model via ONNX Runtime,
//! decoded through its SSD anchors and merged by weighted non-maximum
//! suppression as `MediaPipe`'s face-detection graph does. The frame is
//! resized to the model's input as `cv2.resize` does it, bit for bit, so that
//! the detector sees what it saw in the Python prototype the tracker was
//! validated with.

use anyhow::{Context, Result, ensure};
use ort::session::Session;
use ort::value::TensorRef;

// MediaPipe's BlazeFace short-range model converted to ONNX; Apache-2.0, see
// models/README.md.
const MODEL: &[u8] = include_bytes!("../models/blaze_face_short_range.onnx");
/// Side (px) of the model's square input.
const SIDE: usize = 128;
/// The model's regressors are in input pixels.
const SCALE: f32 = 128.0;
/// SSD anchors: one per row of the model's outputs.
const ANCHORS: usize = 896;
/// The anchor layers' strides (input pixels per grid cell). Layers of the
/// same stride share a grid, and each puts two anchors in every cell.
const STRIDES: [usize; 4] = [8, 16, 16, 16];
/// Values per anchor in `regressors`: the box's centre (relative to the
/// anchor) and size, then x and y of the six keypoints (relative to the
/// anchor).
const REG: usize = 16;
/// Keypoints per detection.
const KEYPOINTS: usize = 6;
/// The logits are clamped to ±100 before the sigmoid.
const SCORE_CLIP: f32 = 100.0;
/// The lowest score (after the sigmoid) a candidate needs.
const MIN_SCORE: f32 = 0.3;
/// Candidates that overlap the best remaining one by more than this
/// intersection over union are merged into it.
const MERGE_IOU: f64 = 0.3;

/// A face the detector found, in the pixels of the frame it was given: x
/// right, y down, pixel k spanning `[k, k + 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Detection {
    /// Left edge of the box.
    pub x: f64,
    /// Top edge of the box.
    pub y: f64,
    /// Width of the box.
    pub w: f64,
    /// Height of the box.
    pub h: f64,
    /// The keypoints: the eye on the image's left (the subject's right eye),
    /// the other eye, the nose tip, the mouth, and the tragion on the image's
    /// left and on its right.
    pub keypoints: [[f64; 2]; KEYPOINTS],
    /// The best score among the candidates merged into it.
    pub score: f32,
}

/// The face-detection ONNX session, its anchors and the buffers it reuses on
/// every call.
pub(crate) struct FaceDetector {
    session: Session,
    /// Anchor centres (x, y, as fractions of the input's side), one per
    /// output row, in the order of the rows.
    anchors: Vec<[f64; 2]>,
    /// Model input, 1x3x`SIDE`x`SIDE` (NCHW): the frame resized, scaled to
    /// [-1, 1] and put in all three channels (kept across calls).
    input: Vec<f32>,
    /// Decoding scratch and the detections of the last call.
    decoder: Decoder,
}

impl FaceDetector {
    /// Load the embedded face-detection model into a new ONNX Runtime
    /// session that runs on one thread.
    ///
    /// # Errors
    /// Fails when ONNX Runtime cannot be initialised or rejects the model.
    pub(crate) fn new() -> Result<Self> {
        // One intra-op thread, as the prototype ran it (0.8 ms a call). In ort
        // 2.0.0-rc.12 the builder's setters fail with an error that hands the
        // builder back, which is not Send; convert it to a plain ort::Error.
        let session = Session::builder()?
            .with_intra_threads(1)
            .map_err(ort::Error::<()>::from)?
            .commit_from_memory(MODEL)
            .context("failed to load face detection model")?;
        Ok(Self {
            session,
            anchors: anchors(),
            input: vec![0f32; 3 * SIDE * SIDE],
            decoder: Decoder::default(),
        })
    }

    /// Look for faces in the `n`x`n` grayscale `frame`: resize it to the
    /// model's input (`prepare_input`), run the model and decode its outputs
    /// (`Decoder::decode`). Returns the faces found, best first; they borrow
    /// an internal buffer that the next call overwrites.
    ///
    /// # Errors
    /// Fails when `frame` holds fewer than `n * n` pixels or `n` is 0, when
    /// the ONNX session rejects the input or the run fails, or when an output
    /// is missing or not of the model's size.
    pub(crate) fn detect(&mut self, frame: &[u8], n: usize) -> Result<&[Detection]> {
        ensure!(
            n > 0 && n.checked_mul(n).is_some_and(|len| frame.len() >= len),
            "a {n}x{n} frame of {} bytes",
            frame.len()
        );
        prepare_input(frame, n, &mut self.input);

        let value = TensorRef::from_array_view(([1usize, 3, SIDE, SIDE], self.input.as_slice()))?;
        let outputs = self.session.run(ort::inputs!["input" => value])?;
        let (_, regressors) = outputs
            .get("regressors")
            .context("the face detection model has no regressors output")?
            .try_extract_tensor::<f32>()?;
        let (_, classificators) = outputs
            .get("classificators")
            .context("the face detection model has no classificators output")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            regressors.len() == ANCHORS * REG && classificators.len() == ANCHORS,
            "the face detection model returned {} regressors and {} scores, not {} and {ANCHORS}",
            regressors.len(),
            classificators.len(),
            ANCHORS * REG
        );
        Ok(self
            .decoder
            .decode(regressors, classificators, &self.anchors, n as f64))
    }
}

#[cfg(test)]
impl FaceDetector {
    /// The ONNX session, for the tests of the model's inputs and outputs.
    pub(crate) fn session(&self) -> &Session {
        &self.session
    }
}

/// The model's 896 anchor centres, as fractions of the input's side (x, y):
/// for each run of layers with the same stride, the cells of its grid row by
/// row, each cell two anchors per layer of the run (512 at stride 8, 384 at
/// 16). All anchors are of unit size, so only the centres matter.
fn anchors() -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(ANCHORS);
    let mut layer = 0;
    while let Some(&stride) = STRIDES.get(layer) {
        let layers = STRIDES[layer..]
            .iter()
            .take_while(|&&s| s == stride)
            .count();
        let cells = SIDE / stride;
        for y in 0..cells {
            for x in 0..cells {
                let centre = [
                    (x as f64 + 0.5) / cells as f64,
                    (y as f64 + 0.5) / cells as f64,
                ];
                out.extend(std::iter::repeat_n(centre, 2 * layers));
            }
        }
        layer += layers;
    }
    out
}

/// Where `cv2.resize` with `INTER_LINEAR` takes one output pixel of an 8-bit
/// image from: two neighbouring source pixels and their weights, in 2048ths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tap {
    first: usize,
    second: usize,
    weights: [i32; 2],
}

/// `OpenCV`'s tap for output pixel `d` of [`SIDE`] from `n` source pixels
/// (`resizeGeneric_`): the source position (d + 0.5)·scale − 0.5, worked out
/// in f64 with scale = 1 / (`SIDE` / n) and rounded to f32; its floor and the
/// fraction f; the weights (1 − f)·2048 and f·2048, in f32 and rounded half
/// to even. Across the frame (`clamp_weights`), a position before the first
/// pixel or past the last takes that edge pixel alone; down it, only the rows
/// are clamped and the weights stay.
// reason: the position, within (-1, n), is rounded to the f32 OpenCV works
// in, and its floor fits an i64; the weights are within [0, 2048]
// (num-cast-try-from).
#[allow(clippy::cast_possible_truncation)]
#[must_use]
fn tap(d: usize, n: usize, clamp_weights: bool) -> Tap {
    let scale = 1.0 / (SIDE as f64 / n as f64);
    let pos = ((d as f64 + 0.5) * scale - 0.5) as f32; // cast: see the reason above
    let floor = pos.floor();
    let mut f = pos - floor;
    let last = i64::try_from(n.saturating_sub(1)).unwrap_or(i64::MAX);
    let mut s = floor as i64; // cast: see the reason above
    if clamp_weights {
        if s < 0 {
            (s, f) = (0, 0.0);
        }
        if s >= last {
            (s, f) = (last, 0.0);
        }
    }
    let index = |i: i64| usize::try_from(i.clamp(0, last)).unwrap_or(0);
    Tap {
        first: index(s),
        second: index(s + 1),
        weights: [
            ((1.0 - f) * 2048.0).round_ties_even() as i32, // cast: within [0, 2048]
            (f * 2048.0).round_ties_even() as i32,         // cast: within [0, 2048]
        ],
    }
}

/// One row of the frame resized across ([`SIDE`] values, in 2048ths of a
/// grey level): `OpenCV`'s `HResizeLinear` for 8-bit images.
fn resize_row(row: &[u8], taps: &[Tap; SIDE], out: &mut [i32; SIDE]) {
    for (o, t) in out.iter_mut().zip(taps) {
        *o = i32::from(row[t.first]) * t.weights[0] + i32::from(row[t.second]) * t.weights[1];
    }
}

/// Resize the `n`x`n` grey `frame` to the model's [`SIDE`]x[`SIDE`] input as
/// `cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)` does it
/// with `OpenCV`'s 8-bit fixed-point arithmetic (`VResizeLinear`'s vectorised
/// rounding), then scale it to [-1, 1] (x / 127.5 − 1, in f32) into all three
/// channel planes of `input`. `frame` holds at least `n * n` pixels, `n > 0`.
fn prepare_input(frame: &[u8], n: usize, input: &mut [f32]) {
    let across: [Tap; SIDE] = std::array::from_fn(|d| tap(d, n, true));
    let mut rows = [[0i32; SIDE]; 2];
    let (plane, others) = input.split_at_mut(SIDE * SIDE);
    for (d, out) in plane.as_chunks_mut::<SIDE>().0.iter_mut().enumerate() {
        let down = tap(d, n, false);
        let [top, bottom] = &mut rows;
        resize_row(&frame[down.first * n..][..n], &across, top);
        resize_row(&frame[down.second * n..][..n], &across, bottom);
        let [b0, b1] = down.weights;
        for ((o, &t), &b) in out.iter_mut().zip(top.iter()).zip(bottom.iter()) {
            let v = (((b0 * (t >> 4)) >> 16) + ((b1 * (b >> 4)) >> 16) + 2) >> 2;
            let v = u8::try_from(v.clamp(0, 255)).unwrap_or(u8::MAX);
            *o = f32::from(v) / 127.5 - 1.0;
        }
    }
    for copy in others.as_chunks_mut::<{ SIDE * SIDE }>().0 {
        copy.copy_from_slice(plane);
    }
}

/// The score of a logit: its sigmoid, the logit clamped to ±[`SCORE_CLIP`],
/// in f32.
#[must_use]
fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit.clamp(-SCORE_CLIP, SCORE_CLIP)).exp())
}

/// `NumPy`'s sum of a float32 vector (`np.add.reduce`): pairwise in blocks of
/// up to 128, each summed eight ways, as the prototype's decoder summed the
/// weights of a merged group.
#[must_use]
fn pairwise_sum(v: &[f32]) -> f32 {
    if v.len() > 128 {
        let mut half = v.len() / 2;
        half -= half % 8;
        return pairwise_sum(&v[..half]) + pairwise_sum(&v[half..]);
    }
    let (blocks, rest) = v.as_chunks::<8>();
    let Some((first, others)) = blocks.split_first() else {
        // Fewer than eight: in order.
        return v.iter().fold(0.0, |s, x| s + x);
    };
    let mut r = *first;
    for block in others {
        for (acc, x) in r.iter_mut().zip(block) {
            *acc += x;
        }
    }
    let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for x in rest {
        sum += x;
    }
    sum
}

/// An anchor whose score passed [`MIN_SCORE`], decoded into fractions of the
/// input: the box (left, top, right, bottom) and the keypoints.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    score: f32,
    bbox: [f64; 4],
    keypoints: [[f64; 2]; KEYPOINTS],
}

impl Candidate {
    #[must_use]
    fn area(&self) -> f64 {
        (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    }

    /// Intersection over union with `other` (plus 1e-12 under the line, as
    /// the prototype computed it).
    #[must_use]
    fn iou(&self, other: &Self) -> f64 {
        let (a, b) = (&self.bbox, &other.bbox);
        let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let inter = w * h;
        inter / (self.area() + other.area() - inter + 1e-12)
    }
}

/// The decoder's buffers, reused from call to call.
#[derive(Debug, Default)]
struct Decoder {
    candidates: Vec<Candidate>,
    /// Indices into `candidates`, best score first.
    order: Vec<usize>,
    used: Vec<bool>,
    /// Scores of the group being merged.
    weights: Vec<f32>,
    detections: Vec<Detection>,
}

impl Decoder {
    /// Decode the model's outputs for a frame of `side` pixels: every anchor
    /// that scores at least [`MIN_SCORE`] is a candidate, its box centre and
    /// keypoints the regressors / 128 plus the anchor's centre, its size the
    /// regressors / 128. Taking the best remaining candidate each time (a
    /// stable sort, ties in anchor order), it and every other remaining
    /// candidate that overlaps it by more than [`MERGE_IOU`] make one
    /// detection: their boxes and keypoints averaged with their scores as
    /// weights, scaled to the frame's pixels, and the best score. Returns the
    /// detections, best first.
    fn decode(
        &mut self,
        regressors: &[f32],
        classificators: &[f32],
        anchors: &[[f64; 2]],
        side: f64,
    ) -> &[Detection] {
        self.candidates.clear();
        for ((reg, &logit), anchor) in regressors
            .as_chunks::<REG>()
            .0
            .iter()
            .zip(classificators)
            .zip(anchors)
        {
            let score = sigmoid(logit);
            if score.is_nan() || score < MIN_SCORE {
                continue;
            }
            let r = reg.map(|v| v / SCALE);
            let (cx, cy) = (f64::from(r[0]) + anchor[0], f64::from(r[1]) + anchor[1]);
            let (hw, hh) = (f64::from(r[2] / 2.0), f64::from(r[3] / 2.0));
            self.candidates.push(Candidate {
                score,
                bbox: [cx - hw, cy - hh, cx + hw, cy + hh],
                keypoints: std::array::from_fn(|k| {
                    [
                        f64::from(r[4 + 2 * k]) + anchor[0],
                        f64::from(r[5 + 2 * k]) + anchor[1],
                    ]
                }),
            });
        }

        let candidates = &self.candidates;
        self.order.clear();
        self.order.extend(0..candidates.len());
        self.order
            .sort_by(|&a, &b| candidates[b].score.total_cmp(&candidates[a].score));
        self.used.clear();
        self.used.resize(candidates.len(), false);
        self.detections.clear();
        for &i in &self.order {
            if self.used[i] {
                continue;
            }
            let best = &candidates[i];
            self.weights.clear();
            let mut bbox = [0.0; 4];
            let mut keypoints = [[0.0; 2]; KEYPOINTS];
            for (j, (c, used)) in candidates.iter().zip(&mut self.used).enumerate() {
                if *used {
                    continue;
                }
                // A NaN overlap (a box that is not finite) merges nothing.
                let member = j == i || best.iou(c) > MERGE_IOU;
                if !member {
                    continue;
                }
                *used = true;
                self.weights.push(c.score);
                let w = f64::from(c.score);
                for (s, v) in bbox.iter_mut().zip(&c.bbox) {
                    *s += v * w;
                }
                for (s, v) in keypoints.iter_mut().zip(&c.keypoints) {
                    s[0] += v[0] * w;
                    s[1] += v[1] * w;
                }
            }
            let total = f64::from(pairwise_sum(&self.weights));
            let bbox = bbox.map(|v| v / total);
            self.detections.push(Detection {
                x: bbox[0] * side,
                y: bbox[1] * side,
                w: (bbox[2] - bbox[0]) * side,
                h: (bbox[3] - bbox[1]) * side,
                keypoints: keypoints.map(|p| [p[0] / total * side, p[1] / total * side]),
                score: best.score,
            });
        }
        &self.detections
    }
}

#[cfg(test)]
// reason: unwrap on fixtures is the idiomatic test failure (test-* rules).
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn there_are_896_anchors_in_two_grids() {
        // Exactly: the centres are fractions with power-of-two denominators.
        let a: Vec<[u64; 2]> = anchors().iter().map(|c| c.map(f64::to_bits)).collect();
        let at = |x: f64, y: f64| [x.to_bits(), y.to_bits()];
        assert_eq!(a.len(), ANCHORS);
        // Stride 8: 16x16 cells, two anchors each; the first cell's centre.
        assert_eq!(a[0], at(0.5 / 16.0, 0.5 / 16.0));
        assert_eq!(a[1], a[0]);
        assert_eq!(a[2], at(1.5 / 16.0, 0.5 / 16.0));
        assert_eq!(a[511], at(15.5 / 16.0, 15.5 / 16.0));
        // Stride 16: 8x8 cells, six anchors each (three layers).
        assert_eq!(a[512], at(0.5 / 8.0, 0.5 / 8.0));
        assert_eq!(a[517], a[512]);
        assert_eq!(a[518], at(1.5 / 8.0, 0.5 / 8.0));
        assert_eq!(a[ANCHORS - 1], at(7.5 / 8.0, 7.5 / 8.0));
    }

    /// The logit whose score is `p`.
    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    /// Model outputs with every anchor below the threshold but the given
    /// ones: (anchor, score, regressors in input pixels).
    fn outputs(set: &[(usize, f32, [f32; REG])]) -> (Vec<f32>, Vec<f32>) {
        let mut regressors = vec![0f32; ANCHORS * REG];
        let mut classificators = vec![-10f32; ANCHORS];
        for (anchor, p, reg) in set {
            classificators[*anchor] = logit(*p);
            regressors[anchor * REG..][..REG].copy_from_slice(reg);
        }
        (regressors, classificators)
    }

    /// Regressors of a box `w`x`h` (input px) moved by `(dx, dy)` from its
    /// anchor, its keypoints at `kp` from the anchor.
    fn reg(dx: f32, dy: f32, w: f32, h: f32, kp: [[f32; 2]; KEYPOINTS]) -> [f32; REG] {
        let mut r = [0f32; REG];
        r[..4].copy_from_slice(&[dx, dy, w, h]);
        for (k, p) in kp.iter().enumerate() {
            r[4 + 2 * k..][..2].copy_from_slice(p);
        }
        r
    }

    fn assert_close(got: f64, want: f64, what: &str) {
        assert!((got - want).abs() < 1e-9, "{what}: {got} != {want}");
    }

    #[test]
    fn one_candidate_decodes_to_one_detection_in_frame_pixels() {
        let anchors = anchors();
        // Anchor 600 is in cell (600 - 512) / 6 = 14 of the 8x8 grid: x 6,
        // y 1, centre (6.5 / 8, 1.5 / 8) = (0.8125, 0.1875).
        let kp = [
            [-8.0, -4.0],
            [8.0, -4.0],
            [0.0, 2.0],
            [0.0, 10.0],
            [-20.0, 0.0],
            [20.0, 0.0],
        ];
        let (r, c) = outputs(&[(600, 0.9, reg(3.0, -6.0, 32.0, 48.0, kp))]);
        let mut decoder = Decoder::default();
        let d = decoder.decode(&r, &c, &anchors, 280.0);
        assert_eq!(d.len(), 1);
        let d = d[0];
        // Centre (0.8125 + 3/128, 0.1875 - 6/128), size 0.25 x 0.375 of the
        // input, in a 280-px frame.
        assert_close(d.x, (0.835_937_5 - 0.125) * 280.0, "x");
        assert_close(d.y, (0.140_625 - 0.1875) * 280.0, "y");
        assert_close(d.w, 0.25 * 280.0, "w");
        assert_close(d.h, 0.375 * 280.0, "h");
        assert_close(d.keypoints[0][0], (0.8125 - 0.0625) * 280.0, "eye x");
        assert_close(d.keypoints[1][1], (0.1875 - 0.031_25) * 280.0, "eye y");
        assert_close(d.keypoints[5][0], (0.8125 + 0.156_25) * 280.0, "tragion x");
        assert!((d.score - 0.9).abs() < 1e-6, "{}", d.score);
    }

    #[test]
    fn overlapping_candidates_merge_weighted_by_score() {
        let anchors = anchors();
        let kp = [[0.0; 2]; KEYPOINTS];
        // Anchors 100 and 101 share a centre; the boxes are 32 px, one 4 px
        // to the right of the other (IoU 28/36), the keypoints 8 px apart.
        // Anchor 300 holds a box far from both.
        let mut a = reg(0.0, 0.0, 32.0, 32.0, kp);
        let mut b = reg(4.0, 0.0, 32.0, 32.0, kp);
        a[4] = 0.0;
        b[4] = 8.0;
        let far = reg(0.0, 0.0, 16.0, 16.0, kp);
        let (r, c) = outputs(&[(100, 0.6, a), (101, 0.9, b), (300, 0.5, far)]);
        let mut decoder = Decoder::default();
        let d = decoder.decode(&r, &c, &anchors, 128.0);
        assert_eq!(d.len(), 2, "{d:?}");
        // The pair, best score first: weights 0.6 and 0.9 put the box 0.9 /
        // 1.5 of the way from the first to the second, 2.4 px right.
        let ax = anchors[100][0] * 128.0;
        assert!((d[0].score - 0.9).abs() < 1e-6, "{d:?}");
        assert!((d[0].x - (ax - 16.0 + 2.4)).abs() < 1e-4, "{d:?}");
        assert!((d[0].w - 32.0).abs() < 1e-4, "{d:?}");
        assert!((d[0].keypoints[0][0] - (ax + 4.8)).abs() < 1e-4, "{d:?}");
        // Then the far box alone.
        assert!((d[1].score - 0.5).abs() < 1e-6, "{d:?}");
        assert!((d[1].w - 16.0).abs() < 1e-4, "{d:?}");
    }

    #[test]
    fn candidates_below_the_threshold_are_dropped() {
        let anchors = anchors();
        let kp = [[0.0; 2]; KEYPOINTS];
        let (r, c) = outputs(&[(10, 0.29, reg(0.0, 0.0, 20.0, 20.0, kp))]);
        let mut decoder = Decoder::default();
        assert!(decoder.decode(&r, &c, &anchors, 280.0).is_empty());
        // A huge logit is clamped, not overflowed; a NaN one is no face.
        assert!((sigmoid(1e30) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-1e30).abs() < 1e-30);
        let (r, mut c) = outputs(&[]);
        c[10] = f32::NAN;
        assert!(decoder.decode(&r, &c, &anchors, 280.0).is_empty());
    }

    #[test]
    fn the_pairwise_sum_is_numpys() {
        // Under eight values: in order. From eight on, eight running sums:
        // 1e8 and fifteen 1s, summed in order in f32, lose every 1; NumPy
        // 2.4.6's np.float32 sum of them is 100000008.
        let mut v = vec![1e8f32];
        v.extend([1.0; 15]);
        let bits = f32::to_bits;
        assert_eq!(bits(v.iter().fold(0.0f32, |s, x| s + x)), bits(1e8));
        assert_eq!(bits(pairwise_sum(&v)), bits(100_000_008.0));
        assert_eq!(bits(pairwise_sum(&[0.25, 0.5, 2.0])), bits(2.75));
        assert_eq!(bits(pairwise_sum(&[])), bits(0.0));
        let long: Vec<f32> = (0..300).map(|i| (i % 7) as f32 * 0.1).collect();
        let exact: f64 = (0..300).map(|i| f64::from(i % 7) * 0.1).sum();
        assert!((f64::from(pairwise_sum(&long)) - exact).abs() < 1e-3);
    }

    /// A grey frame with detail at every scale: a hash of the pixel position
    /// (the same as track.rs's tests).
    fn textured_frame(n: usize) -> Vec<u8> {
        (0..n * n)
            .map(|i| {
                let (x, y) = (i % n, i / n);
                u8::try_from(((x * 31 + y * 17) ^ (x * y / 7)) % 251).unwrap()
            })
            .collect()
    }

    /// The resized frame as grey levels, back from the model input.
    // reason: `g` is a whole number in [0, 255] (checked against the input)
    // (num-cast-try-from).
    #[allow(clippy::cast_possible_truncation)]
    fn resized(frame: &[u8], n: usize) -> Vec<u8> {
        let mut input = vec![0f32; 3 * SIDE * SIDE];
        prepare_input(frame, n, &mut input);
        let (plane, others) = input.split_at(SIDE * SIDE);
        assert_eq!(&others[..SIDE * SIDE], plane);
        assert_eq!(&others[SIDE * SIDE..], plane);
        plane
            .iter()
            .map(|v| {
                let g = ((v + 1.0) * 127.5).round();
                assert!((g / 127.5 - 1.0 - v).abs() < 1e-6, "{v}");
                u8::try_from(g as i64).unwrap()
            })
            .collect()
    }

    #[test]
    fn the_input_is_cv2_resize_bit_for_bit() {
        // cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) of
        // `textured_frame`, OpenCV 4.13.0: the sum of the 16384 grey levels,
        // and a few of them (x, y, value).
        for (n, sum, samples) in [
            (
                280,
                2_045_993u64,
                [(0, 0, 28u8), (17, 3, 102), (64, 64, 151), (127, 127, 97)],
            ),
            (
                560,
                2_045_701,
                [(0, 0, 81), (17, 3, 192), (64, 64, 170), (127, 127, 135)],
            ),
        ] {
            let got = resized(&textured_frame(n), n);
            assert_eq!(got.iter().map(|&v| u64::from(v)).sum::<u64>(), sum, "{n}");
            for (x, y, v) in samples {
                assert_eq!(got[y * SIDE + x], v, "{n}: ({x}, {y})");
            }
        }
        // A flat frame stays flat.
        let flat = resized(&[77u8; 280 * 280], 280);
        assert!(flat.iter().all(|&v| v == 77));
    }

    #[test]
    fn opencv_taps_split_the_position_into_pixel_and_weights() {
        // 280 -> 128: output 0 is at 0.59375, output 127 at 278.40625.
        assert_eq!(
            tap(0, 280, true),
            Tap {
                first: 0,
                second: 1,
                weights: [832, 1216]
            }
        );
        assert_eq!(
            tap(127, 280, false),
            Tap {
                first: 278,
                second: 279,
                weights: [1216, 832]
            }
        );
        // Enlarging (64 -> 128): the first position, -0.25, takes the edge
        // pixel alone across, and keeps its weights down.
        assert_eq!(
            tap(0, 64, true),
            Tap {
                first: 0,
                second: 1,
                weights: [2048, 0]
            }
        );
        assert_eq!(
            tap(0, 64, false),
            Tap {
                first: 0,
                second: 0,
                weights: [512, 1536]
            }
        );
    }
}
