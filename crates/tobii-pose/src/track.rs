//! Head pose from the IR face frame, fully in Rust: crop the 560x560 frame,
//! run the `MediaPipe` face-landmark model via ONNX Runtime (`ort`) and fit
//! the canonical face mesh to the landmarks (Kabsch start, perspective `PnP`):
//! [`Tracker::fit`] gives that [`FaceFit`] per frame. [`RestPose`] reads the
//! legacy yaw/pitch/roll and translation relative to a calibrated rest pose
//! out of the fits. Validated against `MediaPipe` to within ~3 degrees.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use tracing::info;

use crate::canonical::CANONICAL_FACE;

// MediaPipe Face Mesh V2 converted to ONNX; Apache-2.0, see models/README.md.
const MODEL: &[u8] = include_bytes!("../models/face_landmarks.onnx");
const IN: usize = 256; // model input is 256x256x3
const NLM: usize = 468; // canonical landmarks (model emits 478 incl. iris)

// Tunables (flip a sign if an axis is reversed; raise a gain if too weak).
const ANGLE_SIGN: [f64; 3] = [1.0, 1.0, 1.0]; // pitch, yaw, roll
const SEND_TRANSLATION: bool = true;
const TRANS_SIGN: [f64; 3] = [-1.0, -1.0, -1.0]; // tx, ty, tz (camera y down, z fwd)
const TRANS_GAIN: [f64; 3] = [1.0, 1.0, 1.0]; // solvePnP translation is metric cm
// Report translation at a pivot (the neck) instead of the face origin, so pure
// head rotation rotates in place instead of sliding. The pivot is offset from
// the face origin in the canonical model frame: +y is down, +z is toward the
// back of the head (nose points to -z). Raise these if rotations still slide.
const PIVOT_NECK_DOWN_CM: f64 = 11.0;
const PIVOT_NECK_BACK_CM: f64 = 6.0;
const SMOOTH: f64 = 0.5;
const CALIB_FRAMES: usize = 30;
const CLAMP_DEG: f64 = 45.0;
const CROP_HALF: f32 = 160.0;
const FOCAL: f64 = 457.0; // ~63deg vertical FOV on a 560px frame (matches MediaPipe)
/// The device's own 280x280 IR stream (EP 0x83, stream 0x50e), upscaled 2x to
/// 560x560 so the same crop/landmark pipeline applies. Focal length fitted on
/// the Windows captures: iris pixels vs projected 0x83 eyeball centres give
/// f = 376 px in the 280 frame (r² 0.999), i.e. 752 px at 560.
const IMAGE83_FOCAL: f64 = 752.0;
const IMAGE83_CY_FRAC: f32 = 0.5;
/// Face crop (half-size, px) for the 2x-upscaled 0x50e stream. The face is only
/// ~150 px wide at 560 (65-75 cm), so the 320-px UVC crop leaves it at <50%
/// fill and the landmark score goes negative beyond ~38° yaw / ~13° right
/// roll. Replaying the Windows sweeps: 105-115 lose 0 frames on all six
/// captures; 100 loses 13 (yaw), 120 loses 18 (roll), 160 loses 18+27.
const IMAGE83_CROP_HALF: f32 = 110.0;
/// The tracker camera looks up ~20° at the user (the device's S frame is
/// tilted 20.0° relative to the display frame). Relative head rotations are
/// expressed in the upright (display) frame, so yaw is a turn about the true
/// vertical: with the camera-frame decomposition a 30° turn read as 28° yaw +
/// 11° roll and a 20° tilt as 19° roll + 7° yaw. Override: `TOBII_CAMERA_TILT_DEG`.
const CAMERA_TILT_DEG: f64 = 20.0;

/// The face-landmark ONNX session plus the buffers it reuses every frame: the
/// 256x256x3 model input and the 468 landmarks it emits.
pub(crate) struct FaceModel {
    session: Session,
    /// Model input, `IN * IN * 3` normalised grey values (kept across frames).
    input: Vec<f32>,
    /// Landmarks of the last `landmarks` call (kept across frames).
    pts: Vec<[f32; 3]>,
}

impl FaceModel {
    /// Load the embedded face-landmark model into a new ONNX Runtime session.
    ///
    /// # Errors
    /// Fails when ONNX Runtime cannot be initialised or rejects the model.
    pub fn new() -> Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(MODEL)
            .context("failed to load face landmark model")?;
        Ok(Self {
            session,
            input: vec![0f32; IN * IN * 3],
            pts: Vec::with_capacity(NLM),
        })
    }

    /// Crop a centred square around `(cx, cy)` of the wWxhH grayscale frame,
    /// bilinearly resize to 256x256, normalise to [0,1] and run the model.
    /// Returns the 468 face landmarks (x,y in input px, z relative) and the
    /// face-presence score. The landmarks borrow an internal buffer that the
    /// next call overwrites.
    ///
    /// # Errors
    /// Fails when the ONNX session rejects the input or the run fails.
    pub(crate) fn landmarks(
        &mut self,
        gray: &[u8],
        w: usize,
        h: usize,
        cx: f32,
        cy: f32,
        half: f32,
    ) -> Result<(&[[f32; 3]], f32)> {
        let x0 = cx - half;
        let y0 = cy - half;
        let span = half * 2.0;
        // Rows of IN pixels, each pixel three (identical) channels.
        let (rows, _) = self.input.as_chunks_mut::<{ IN * 3 }>();
        for (oy, row) in rows.iter_mut().enumerate() {
            let sy = y0 + (oy as f32 + 0.5) / IN as f32 * span - 0.5;
            for (ox, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let sx = x0 + (ox as f32 + 0.5) / IN as f32 * span - 0.5;
                let v = bilinear(gray, w, h, sx, sy) / 255.0;
                px.fill(v);
            }
        }

        let value = TensorRef::from_array_view(([1usize, IN, IN, 3], self.input.as_slice()))?;
        let outputs = self.session.run(ort::inputs!["input_12" => value])?;
        let (_, lm) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, score) = outputs["Identity_1"].try_extract_tensor::<f32>()?;

        self.pts.clear();
        self.pts
            .extend(lm.as_chunks::<3>().0.iter().take(NLM).copied());
        Ok((self.pts.as_slice(), score[0]))
    }
}

// reason: `x`/`y` are clamped to `[0, w-1]`/`[0, h-1]` first, so the
// floor->usize cast is in range and non-negative (num-cast-try-from).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[must_use]
fn bilinear(g: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize; // cast: clamped to [0, w-1] above
    let y0 = y.floor() as usize; // cast: clamped to [0, h-1] above
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p = |xx: usize, yy: usize| f32::from(g[yy * w + xx]);
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// Rotation (canonical -> observed) via Horn's quaternion method on the
/// cross-covariance, no external SVD needed.
#[must_use]
pub fn kabsch(reference: &[[f64; 3]], current: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let n = reference.len();
    let mut rc = [0.0; 3];
    let mut cc = [0.0; 3];
    for (r, c) in reference.iter().zip(current) {
        for k in 0..3 {
            rc[k] += r[k];
            cc[k] += c[k];
        }
    }
    for k in 0..3 {
        rc[k] /= n as f64;
        cc[k] /= n as f64;
    }
    let mut hm = [[0.0; 3]; 3];
    for (r, c) in reference.iter().zip(current) {
        for a in 0..3 {
            for b in 0..3 {
                hm[a][b] += (r[a] - rc[a]) * (c[b] - cc[b]);
            }
        }
    }
    let (sxx, sxy, sxz) = (hm[0][0], hm[0][1], hm[0][2]);
    let (syx, syy, syz) = (hm[1][0], hm[1][1], hm[1][2]);
    let (szx, szy, szz) = (hm[2][0], hm[2][1], hm[2][2]);
    let mut nn = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let mut shift = 0.0f64;
    for row in &nn {
        shift = shift.max(row.iter().map(|v| v.abs()).sum());
    }
    for (i, row) in nn.iter_mut().enumerate() {
        row[i] += shift;
    }
    let mut v = [1.0, 0.2, 0.1, 0.05];
    for _ in 0..200 {
        let mut wv = [0.0; 4];
        for (wr, row) in wv.iter_mut().zip(&nn) {
            for (a, b) in row.iter().zip(&v) {
                *wr += a * b;
            }
        }
        let m = (wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2] + wv[3] * wv[3]).sqrt();
        for (vi, wi) in v.iter_mut().zip(&wv) {
            *vi = wi / m;
        }
    }
    let (qw, qx, qy, qz) = (v[0], v[1], v[2], v[3]);
    [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
}

/// [pitch, yaw, roll] in degrees from a rotation matrix.
#[must_use]
pub fn euler_deg(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let sy = (r[0][0] * r[0][0] + r[1][0] * r[1][0]).sqrt();
    if sy > 1e-6 {
        [
            r[2][1].atan2(r[2][2]).to_degrees(),
            (-r[2][0]).atan2(sy).to_degrees(),
            r[1][0].atan2(r[0][0]).to_degrees(),
        ]
    } else {
        [
            (-r[1][2]).atan2(r[1][1]).to_degrees(),
            (-r[2][0]).atan2(sy).to_degrees(),
            0.0,
        ]
    }
}

const IDENTITY3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Inverse of `euler_deg`: R = Rz(roll) * Ry(yaw) * Rx(pitch), degrees in.
#[must_use]
fn from_euler_deg(e: &[f64; 3]) -> [[f64; 3]; 3] {
    let (sa, ca) = e[0].to_radians().sin_cos();
    let (sb, cb) = e[1].to_radians().sin_cos();
    let (sg, cg) = e[2].to_radians().sin_cos();
    [
        [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
        [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
        [-sb, cb * sa, cb * ca],
    ]
}

/// `T · R · R0^T · T^T`: the rotation that takes the rest pose `R0` to `R`,
/// expressed in the upright frame (`T` = camera -> upright).
#[must_use]
fn relative_upright(r: &[[f64; 3]; 3], r0: &[[f64; 3]; 3], t: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    matmul3(&matmul3(t, &matmul3(r, &transpose3(r0))), &transpose3(t))
}

#[must_use]
fn transpose3(r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = r[j][i];
        }
    }
    o
}

#[must_use]
fn matvec3(r: &[[f64; 3]; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    ]
}

#[must_use]
fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    o
}

/// Rodrigues: axis-angle vector -> rotation matrix.
#[must_use]
fn expmap(w: [f64; 3]) -> [[f64; 3]; 3] {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if th < 1e-12 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let (k0, k1, k2) = (w[0] / th, w[1] / th, w[2] / th);
    let (c, s) = (th.cos(), th.sin());
    let c1 = 1.0 - c;
    [
        [
            c + k0 * k0 * c1,
            k0 * k1 * c1 - k2 * s,
            k0 * k2 * c1 + k1 * s,
        ],
        [
            k1 * k0 * c1 + k2 * s,
            c + k1 * k1 * c1,
            k1 * k2 * c1 - k0 * s,
        ],
        [
            k2 * k0 * c1 - k1 * s,
            k2 * k1 * c1 + k0 * s,
            c + k2 * k2 * c1,
        ],
    ]
}

/// Gauss-Jordan solve of the 6x6 normal equations with partial pivoting;
/// singular pivots are skipped and their unknowns left at zero.
#[must_use]
fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> [f64; 6] {
    let mut m = *a;
    let mut y = *b;
    for i in 0..6 {
        let mut piv = i;
        for r in i + 1..6 {
            if m[r][i].abs() > m[piv][i].abs() {
                piv = r;
            }
        }
        m.swap(i, piv);
        y.swap(i, piv);
        let d = m[i][i];
        if d.abs() < 1e-12 {
            continue;
        }
        let row_i = m[i];
        let y_i = y[i];
        for (r, (row, yr)) in m.iter_mut().zip(&mut y).enumerate() {
            if r == i {
                continue;
            }
            let f = row[i] / d;
            for (mc, ic) in row.iter_mut().zip(&row_i).skip(i) {
                *mc -= f * ic;
            }
            *yr -= f * y_i;
        }
    }
    let mut x = [0.0; 6];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = if m[i][i].abs() > 1e-12 {
            y[i] / m[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Perspective `PnP` via Gauss-Newton minimising reprojection error. Returns the
/// rotation (object->camera) and translation (cm), initialised from `init_r`.
#[must_use]
pub(crate) fn solve_pnp(
    model: &[[f64; 3]],
    image: &[[f64; 2]],
    focal: f64,
    cx: f64,
    cy: f64,
    init_r: [[f64; 3]; 3],
    init_depth: f64,
) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut r = init_r;
    let mut t = [0.0, 0.0, init_depth];
    for _ in 0..15 {
        let mut jtj = [[0.0; 6]; 6];
        let mut jtr = [0.0; 6];
        for (p, im) in model.iter().zip(image) {
            let rp = matvec3(&r, p);
            let q = [rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]];
            if q[2] < 1e-3 {
                continue;
            }
            let iz = 1.0 / q[2];
            let res = [
                focal * q[0] * iz + cx - im[0],
                focal * q[1] * iz + cy - im[1],
            ];
            let duq = [focal * iz, 0.0, -focal * q[0] * iz * iz];
            let dvq = [0.0, focal * iz, -focal * q[1] * iz * iz];
            // dQ/ddelta = -skew(rp)
            let sk = [
                [0.0, rp[2], -rp[1]],
                [-rp[2], 0.0, rp[0]],
                [rp[1], -rp[0], 0.0],
            ];
            let mut ju = [0.0; 6];
            let mut jv = [0.0; 6];
            for j in 0..3 {
                ju[j] = duq[0] * sk[0][j] + duq[1] * sk[1][j] + duq[2] * sk[2][j];
                jv[j] = dvq[0] * sk[0][j] + dvq[1] * sk[1][j] + dvq[2] * sk[2][j];
            }
            ju[3..6].copy_from_slice(&duq);
            jv[3..6].copy_from_slice(&dvq);
            for ((jtr_a, jtj_row), (ua, va)) in jtr.iter_mut().zip(&mut jtj).zip(ju.iter().zip(&jv))
            {
                *jtr_a += ua * res[0] + va * res[1];
                for (cell, (ub, vb)) in jtj_row.iter_mut().zip(ju.iter().zip(&jv)) {
                    *cell += ua * ub + va * vb;
                }
            }
        }
        for (i, row) in jtj.iter_mut().enumerate() {
            row[i] += row[i] * 1e-3 + 1e-9;
        }
        let d = solve6(&jtj, &jtr);
        r = matmul3(&expmap([-d[0], -d[1], -d[2]]), &r);
        t[0] -= d[3];
        t[1] -= d[4];
        t[2] -= d[5];
    }
    (r, t)
}

/// RMS distance of the points from their centroid.
#[must_use]
fn spread3(p: &[[f64; 3]]) -> f64 {
    let n = p.len() as f64;
    let mut c = [0.0; 3];
    for q in p {
        for (ck, qk) in c.iter_mut().zip(q) {
            *ck += qk / n;
        }
    }
    let mut s = 0.0;
    for q in p {
        s += (0..3).map(|k| (q[k] - c[k]).powi(2)).sum::<f64>();
    }
    (s / n).sqrt()
}

/// RMS distance of the points from their centroid.
#[must_use]
fn spread2(p: &[[f64; 2]]) -> f64 {
    let n = p.len() as f64;
    let mut c = [0.0; 2];
    for q in p {
        for (ck, qk) in c.iter_mut().zip(q) {
            *ck += qk / n;
        }
    }
    let mut s = 0.0;
    for q in p {
        s += (0..2).map(|k| (q[k] - c[k]).powi(2)).sum::<f64>();
    }
    (s / n).sqrt()
}

/// Canonical mesh as f64 for the fit.
#[must_use]
pub(crate) fn canonical_f64() -> Vec<[f64; 3]> {
    CANONICAL_FACE
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        .collect()
}

/// One frame's fit of the canonical face mesh to the landmarks the model
/// found, as [`Tracker::fit`] returns it. The mesh is `MediaPipe`'s, in cm:
/// x towards the subject's left, y down, z towards the back of the head (the
/// nose points to -z).
///
/// It borrows the tracker's landmark buffer, so it lasts until the next
/// frame. Only this crate builds one, so fields can be added (how the face
/// was found, say) without breaking a reader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceFit<'a> {
    /// Rotation of the mesh into the camera frame (object -> camera; camera x
    /// right, y down, z forward), as the perspective fit gives it: no
    /// `MediaPipe` sign flips and no rest pose.
    pub rotation: [[f64; 3]; 3],
    /// The mesh origin in the camera frame, mm.
    pub translation_mm: [f64; 3],
    /// The 468 landmarks in the pixels of the frame given to
    /// [`Tracker::fit`]: x right, y down, pixel k spanning `[k, k + 1)`, so
    /// the frame centre, the fit's principal point, is (w/2, h/2). The 0x50e
    /// stream is fit upscaled 2x; halve them for its own 280x280 image.
    pub landmarks: &'a [[f64; 2]; NLM],
    /// The landmark model's face-presence score, a logit. Never negative:
    /// below zero the tracker reports no face.
    pub score: f32,
    /// `translation_mm` in cm, exactly as the solver returned it. The legacy
    /// pose is computed in cm, and a cm -> mm -> cm round trip changes the
    /// last bit of about one value in eleven.
    translation_cm: [f64; 3],
}

impl<'a> FaceFit<'a> {
    /// A fit from the solver's rotation and translation (cm).
    fn new(
        rotation: [[f64; 3]; 3],
        translation_cm: [f64; 3],
        landmarks: &'a [[f64; 2]; NLM],
        score: f32,
    ) -> Self {
        Self {
            rotation,
            translation_mm: translation_cm.map(|v| v * 10.0),
            landmarks,
            score,
            translation_cm,
        }
    }
}

/// Live head-pose tracker: face-following crop, landmark inference and the
/// perspective fit ([`Tracker::fit`]), plus the legacy pose relative to a
/// calibrated rest pose: `process` returns the `OpenTrack` pose
/// [TX, TY, TZ, Yaw, Pitch, Roll] of its own [`RestPose`] once calibrated,
/// else `None`.
pub struct Tracker {
    face: FaceFitter,
    rest: RestPose,
}

/// The half of [`Tracker`] that makes a [`FaceFit`] of each frame: the
/// face-following crop, the landmark model and the perspective fit.
struct FaceFitter {
    model: FaceModel,
    canonical: Vec<[f64; 3]>,
    /// Landmarks in full-frame pixels (reused every frame).
    image2d: Vec<[f64; 2]>,
    /// Landmarks in model crop coordinates as f64 (reused every frame).
    observed: Vec<[f64; 3]>,
    cx: f32,
    cy: f32,
    /// Pinhole focal length (px) of the frames fed to `fit`.
    focal: f64,
    /// Half-size (px) of the face crop fed to the landmark model.
    crop_half: f32,
    /// Initial crop centre as a fraction of the frame height.
    cy_frac: f32,
    /// While the face is lost: index into `SEARCH_GRID` of the crop centre to
    /// try on the next frame (one candidate per frame); `None` once tracking.
    searching: Option<usize>,
}

/// The legacy head pose, the one tobiid publishes on the `HEAD` stream: each
/// fit moved to a neck pivot, taken relative to the mean of the first 30 fits
/// (`CALIB_FRAMES`), read in the upright frame with `MediaPipe`'s signs,
/// clamped to ±45° and smoothed.
///
/// [`RestPose::update`] takes one frame at a time and returns the `OpenTrack`
/// pose [TX, TY, TZ (cm), Yaw, Pitch, Roll (deg)] once the rest pose is
/// calibrated. [`Default`] gives the built-in settings,
/// [`RestPose::from_env`] the daemon's (with the environment overrides).
#[derive(Debug, Clone)]
pub struct RestPose {
    pivot: [f64; 3], // neck offset in model cm: [0, down, back]
    debug: bool,
    roll_eyeline: bool,
    /// Fits taken, for the `TOBII_POSE_DEBUG` cadence.
    frame: u64,
    /// Pose of the last frame before rest-pose subtraction and smoothing:
    /// [tx, ty, tz (cm at the pivot), pitch, yaw, roll (deg)]; `None` when no
    /// face was found in it.
    last_raw: Option<[f64; 6]>,
    origin: Option<[f64; 6]>,
    accum: Vec<[f64; 6]>,
    out: [f64; 6],
    have: bool,
    /// Rest-pose rotation (head -> camera) captured at calibration. Angles are
    /// reported from the relative rotation `T · R · rot0^T · T^T`, i.e. in the
    /// upright frame (`T` undoes the camera tilt): subtracting euler angles per
    /// axis instead would leak yaw into roll (and back) through the camera's
    /// ~20° upward tilt (0.4 deg/deg).
    rot0: Option<[[f64; 3]; 3]>,
    /// `T` = Rx(camera tilt): camera frame -> upright frame.
    untilt: [[f64; 3]; 3],
}

/// Crop centres (fractions of frame width/height) cycled while the face is
/// lost. Centre first, then the cardinal offsets, then the corners; a 3x3 grid
/// with a face-sized crop covers the whole frame within 9 frames (~0.3 s).
const SEARCH_GRID: [(f32, f32); 9] = [
    (0.5, 0.5),
    (0.5, 0.25),
    (0.5, 0.75),
    (0.25, 0.5),
    (0.75, 0.5),
    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
];

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Tracker {
    /// Tracker for the UVC camera path (560x560 frames, `MediaPipe`-like FOV).
    ///
    /// # Errors
    /// Fails when the landmark model cannot be loaded.
    pub fn new() -> Result<Self> {
        Self::with_geometry(FOCAL, CROP_HALF, 0.42)
    }

    /// Tracker for the 0x50e image stream: feed it 280x280 frames upscaled 2x
    /// (`image83::upscale2x_into`), i.e. 560x560 at the fitted focal length.
    ///
    /// # Errors
    /// Fails when the landmark model cannot be loaded.
    pub fn new_image83() -> Result<Self> {
        Self::with_geometry(IMAGE83_FOCAL, IMAGE83_CROP_HALF, IMAGE83_CY_FRAC)
    }

    /// Tracker for frames with pinhole focal length `focal` (px), a face crop of
    /// half-size `crop_half` (px) first centred at `cy_frac` of the frame height.
    /// Its [`RestPose`] reads the `TOBII_PIVOT_DOWN`, `TOBII_PIVOT_BACK`,
    /// `TOBII_POSE_DEBUG`, `TOBII_ROLL_EYELINE` and `TOBII_CAMERA_TILT_DEG`
    /// overrides ([`RestPose::from_env`]).
    ///
    /// # Errors
    /// Fails when the landmark model cannot be loaded.
    pub(crate) fn with_geometry(focal: f64, crop_half: f32, cy_frac: f32) -> Result<Self> {
        let rest = RestPose::from_env();
        Ok(Self {
            face: FaceFitter {
                model: FaceModel::new()?,
                canonical: canonical_f64(),
                image2d: Vec::with_capacity(NLM),
                observed: Vec::with_capacity(NLM),
                cx: 0.0,
                cy: 0.0,
                focal,
                crop_half,
                cy_frac,
                searching: None,
            },
            rest,
        })
    }

    /// Unfiltered pose of the last frame `process` took (see
    /// [`RestPose::last_raw`]), for offline analysis.
    #[must_use]
    pub fn last_raw(&self) -> Option<[f64; 6]> {
        self.rest.last_raw()
    }

    /// Drop the calibrated rest pose so it recalibrates from the next frames.
    pub fn recenter(&mut self) {
        self.rest.recenter();
    }

    /// Fit the face mesh to one `w`x`h` grayscale frame; `None` when the
    /// landmark model finds no face in the crop (it then tries the next
    /// `SEARCH_GRID` position on the next frame). The crop follows the face
    /// from frame to frame.
    ///
    /// The fit borrows the tracker until the next frame. It leaves the legacy
    /// pose alone: `process` is `fit` followed by the tracker's [`RestPose`].
    ///
    /// # Errors
    /// Fails when landmark inference fails or the model returns fewer than 468
    /// landmarks.
    pub fn fit(&mut self, gray: &[u8], w: usize, h: usize) -> Result<Option<FaceFit<'_>>> {
        self.face.fit(gray, w, h)
    }

    /// Run one `w`x`h` grayscale frame through the pipeline: [`Tracker::fit`],
    /// then [`RestPose::update`]. Returns the smoothed `OpenTrack` pose once the
    /// rest pose is calibrated; `None` while calibrating or when no face is
    /// found.
    ///
    /// # Errors
    /// Fails when landmark inference fails (see [`Tracker::fit`]).
    pub fn process(&mut self, gray: &[u8], w: usize, h: usize) -> Result<Option<[f64; 6]>> {
        let fit = self.face.fit(gray, w, h)?;
        Ok(self.rest.update(fit.as_ref()))
    }
}

impl FaceFitter {
    /// See [`Tracker::fit`].
    fn fit(&mut self, gray: &[u8], w: usize, h: usize) -> Result<Option<FaceFit<'_>>> {
        if self.cx == 0.0 {
            self.cx = w as f32 / 2.0;
            self.cy = h as f32 * self.cy_frac;
        }
        // While the face is lost, sweep the crop over a grid of candidate
        // centres (one per frame); otherwise a face outside the initial crop
        // would never be found. Once found, the crop follows the face.
        if let Some(i) = self.searching {
            let (fx, fy) = SEARCH_GRID[i % SEARCH_GRID.len()];
            self.cx = w as f32 * fx;
            self.cy = h as f32 * fy;
        }
        let crop_half = self.crop_half;
        let focal = self.focal;
        let (pts, score) = self
            .model
            .landmarks(gray, w, h, self.cx, self.cy, crop_half)?;
        if score < 0.0 {
            // Face lost - try the next search position on the next frame.
            self.searching = Some(self.searching.map_or(1, |i| i + 1));
            return Ok(None);
        }
        self.searching = None;

        // Landmark centroid in the 256 crop, mapped back to the frame.
        let span = crop_half * 2.0;
        let (mut mx, mut my) = (0.0f32, 0.0f32);
        for p in pts {
            mx += p[0];
            my += p[1];
        }
        mx /= pts.len() as f32;
        my /= pts.len() as f32;
        let crop_x0 = self.cx - crop_half;
        let crop_y0 = self.cy - crop_half;
        self.cx = crop_x0 + mx / IN as f32 * span; // follow the face next frame
        self.cy = crop_y0 + my / IN as f32 * span;

        // Map landmarks to full-frame 2D, and keep the model's 3D for the init.
        self.image2d.clear();
        self.image2d.extend(pts.iter().map(|p| {
            [
                f64::from(crop_x0 + p[0] / IN as f32 * span),
                f64::from(crop_y0 + p[1] / IN as f32 * span),
            ]
        }));
        self.observed.clear();
        self.observed.extend(
            pts.iter()
                .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]),
        );
        let init_r = kabsch(&self.canonical, &self.observed);
        let init_depth = focal * spread3(&self.canonical) / spread2(&self.image2d).max(1e-6);

        // Perspective solve: metric, rotation-decoupled rotation + translation.
        let (r, t) = solve_pnp(
            &self.canonical,
            &self.image2d,
            focal,
            w as f64 / 2.0,
            h as f64 / 2.0,
            init_r,
            init_depth,
        );
        let landmarks =
            <&[[f64; 2]; NLM]>::try_from(self.image2d.as_slice()).with_context(|| {
                format!(
                    "the landmark model returned {} points, not {NLM}",
                    self.image2d.len()
                )
            })?;
        Ok(Some(FaceFit::new(r, t, landmarks, score)))
    }
}

impl Default for RestPose {
    /// The built-in settings, whatever the environment says: the neck pivot
    /// 11 cm down and 6 cm back, a 20° camera tilt, roll from the fit and no
    /// debug log.
    fn default() -> Self {
        Self::new(
            [0.0, PIVOT_NECK_DOWN_CM, PIVOT_NECK_BACK_CM],
            CAMERA_TILT_DEG,
            false,
            false,
        )
    }
}

impl RestPose {
    /// `pivot` in model cm ([0, down, back]), the camera tilt in degrees.
    fn new(pivot: [f64; 3], camera_tilt_deg: f64, roll_eyeline: bool, debug: bool) -> Self {
        Self {
            pivot,
            debug,
            roll_eyeline,
            frame: 0,
            last_raw: None,
            origin: None,
            accum: Vec::with_capacity(CALIB_FRAMES),
            out: [0.0; 6],
            have: false,
            rot0: None,
            untilt: from_euler_deg(&[camera_tilt_deg, 0.0, 0.0]),
        }
    }

    /// The legacy pose as the daemon runs it: the built-in settings with the
    /// `TOBII_PIVOT_DOWN`, `TOBII_PIVOT_BACK` (cm), `TOBII_CAMERA_TILT_DEG`,
    /// `TOBII_ROLL_EYELINE` and `TOBII_POSE_DEBUG` overrides applied.
    #[must_use]
    pub fn from_env() -> Self {
        let pivot = [
            0.0,
            env_f64("TOBII_PIVOT_DOWN", PIVOT_NECK_DOWN_CM),
            env_f64("TOBII_PIVOT_BACK", PIVOT_NECK_BACK_CM),
        ];
        let debug = std::env::var("TOBII_POSE_DEBUG").is_ok();
        let roll_eyeline =
            std::env::var("TOBII_ROLL_EYELINE").is_ok_and(|v| !v.is_empty() && v != "0");
        info!(
            pivot_down_cm = pivot[1],
            pivot_back_cm = pivot[2],
            roll_eyeline,
            "tracker pivot"
        );
        Self::new(
            pivot,
            env_f64("TOBII_CAMERA_TILT_DEG", CAMERA_TILT_DEG),
            roll_eyeline,
            debug,
        )
    }

    /// Unfiltered pose of the last frame [`RestPose::update`] took, before
    /// rest-pose subtraction and smoothing: [tx, ty, tz (cm at the neck
    /// pivot), pitch, yaw, roll (deg, `MediaPipe` signs)]; `None` when that
    /// frame had no face. For offline analysis.
    #[must_use]
    pub fn last_raw(&self) -> Option<[f64; 6]> {
        self.last_raw
    }

    /// Drop the calibrated rest pose so it recalibrates from the next frames.
    pub fn recenter(&mut self) {
        self.origin = None;
        self.rot0 = None;
        self.accum.clear();
        self.have = false;
        info!("recenter: recalibrating rest pose");
    }

    /// Take one frame: its fit, or `None` when no face was found in it.
    /// Returns the smoothed `OpenTrack` pose [TX, TY, TZ (cm), Yaw, Pitch,
    /// Roll (deg)] once the rest pose is calibrated; `None` while calibrating
    /// (the first 30 fits after construction or [`RestPose::recenter`]) and
    /// for a frame without a face, which changes nothing but
    /// [`RestPose::last_raw`].
    pub fn update(&mut self, fit: Option<&FaceFit<'_>>) -> Option<[f64; 6]> {
        let Some(fit) = fit else {
            self.last_raw = None;
            return None;
        };
        let (r, t) = (&fit.rotation, &fit.translation_cm);
        let e = euler_deg(r);
        // Match MediaPipe's convention (yaw/roll negate vs our euler).
        let mut mp = [e[0], -e[1], -e[2]]; // pitch, yaw, roll

        // Optional: measure roll directly from the eye line (outer corners 33 &
        // 263). This is decoupled from yaw/pitch and symmetric by construction,
        // avoiding euler cross-axis coupling. Flip the sign here if reversed.
        if self.roll_eyeline {
            let r_eye = fit.landmarks[33]; // subject's right eye outer (image left)
            let l_eye = fit.landmarks[263]; // subject's left eye outer (image right)
            let dx = l_eye[0] - r_eye[0];
            let dy = l_eye[1] - r_eye[1];
            mp[2] = -dy.atan2(dx).to_degrees();
        }

        // Translate the reported position to the neck pivot: t' = t + R*pivot.
        // A pure head rotation about the neck then leaves t' ~constant (rotates
        // in place) instead of swinging the face origin sideways.
        let rp = matvec3(r, &self.pivot);
        let tp = [t[0] + rp[0], t[1] + rp[1], t[2] + rp[2]];
        let raw = [tp[0], tp[1], tp[2], mp[0], mp[1], mp[2]]; // translation in cm
        self.last_raw = Some(raw);

        self.frame += 1;
        if self.debug && self.frame.is_multiple_of(8) {
            // Opted in via TOBII_POSE_DEBUG, so it stays visible at the
            // default (info) filter level.
            info!(
                yaw = mp[1],
                pitch = mp[0],
                tx = t[0],
                ty = t[1],
                tz = t[2],
                pivot_tx = tp[0],
                pivot_ty = tp[1],
                pivot_tz = tp[2],
                "pose (deg, cm)"
            );
        }

        let Some(o) = self.origin else {
            self.accum.push(raw);
            if self.accum.len() >= CALIB_FRAMES {
                let n = self.accum.len() as f64;
                let mut o = [0.0; 6];
                for a in &self.accum {
                    for (oi, ai) in o.iter_mut().zip(a) {
                        *oi += ai / n;
                    }
                }
                self.origin = Some(o);
                // Rest rotation from the averaged eulers (undo the MediaPipe
                // sign flip applied to `mp`).
                self.rot0 = Some(from_euler_deg(&[o[3], -o[4], -o[5]]));
                info!("calibrated rest pose");
            }
            return None;
        };

        let tx = (raw[0] - o[0]) * TRANS_SIGN[0] * TRANS_GAIN[0];
        let ty = (raw[1] - o[1]) * TRANS_SIGN[1] * TRANS_GAIN[1];
        let tz = (raw[2] - o[2]) * TRANS_SIGN[2] * TRANS_GAIN[2];
        let clamp = |v: f64| v.clamp(-CLAMP_DEG, CLAMP_DEG);
        // Relative rotation R · R0^T (R is head -> camera), expressed in the
        // upright frame so yaw is about the true vertical.
        let r_rel = relative_upright(r, &self.rot0.unwrap_or(IDENTITY3), &self.untilt);
        let er = euler_deg(&r_rel);
        let mut rel = [er[0], -er[1], -er[2]]; // pitch, yaw, roll (MediaPipe sign)
        if self.roll_eyeline {
            rel[2] = raw[5] - o[5]; // eye-line roll is an image-plane measure
        }
        let pitch = clamp(rel[0] * ANGLE_SIGN[0]);
        let yaw = clamp(rel[1] * ANGLE_SIGN[1]);
        let roll = clamp(rel[2] * ANGLE_SIGN[2]);
        let (tx, ty, tz) = if SEND_TRANSLATION {
            (tx, ty, tz)
        } else {
            (0.0, 0.0, 0.0)
        };
        let target = [tx, ty, tz, yaw, pitch, roll]; // OpenTrack order

        if self.have {
            for (out, tgt) in self.out.iter_mut().zip(&target) {
                *out += SMOOTH * (tgt - *out);
            }
        } else {
            self.out = target;
            self.have = true;
        }
        Some(self.out)
    }
}

#[cfg(test)]
// reason: unwrap on fixtures is the idiomatic test failure (test-* rules).
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn from_euler_round_trips_euler_deg() {
        for e in [[-20.0, 30.0, 5.0], [10.0, -40.0, -12.0], [0.0, 0.0, 0.0]] {
            let back = euler_deg(&from_euler_deg(&e));
            for k in 0..3 {
                assert!((back[k] - e[k]).abs() < 1e-9, "{e:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn relative_rotation_does_not_leak_yaw_into_roll() {
        // Camera looks up 20°: camera = Rx(-20) · upright. The head at rest is
        // additionally pitched 8° down in the upright frame; it then turns 30°
        // about the WORLD vertical and, separately, rolls 20°.
        let tilt = from_euler_deg(&[20.0, 0.0, 0.0]); // upright -> camera is its transpose
        let cam_from_up = transpose3(&tilt);
        let rest_up = from_euler_deg(&[-8.0, 0.0, 0.0]);
        let r0 = matmul3(&cam_from_up, &rest_up);
        let turned = matmul3(
            &cam_from_up,
            &matmul3(&from_euler_deg(&[0.0, 30.0, 0.0]), &rest_up),
        );
        let naive = euler_deg(&turned);
        let e0 = euler_deg(&r0);
        assert!(
            (naive[2] - e0[2]).abs() > 8.0,
            "per-axis subtraction leaks roll: {naive:?} - {e0:?}"
        );
        let e = euler_deg(&relative_upright(&turned, &r0, &tilt));
        assert!(
            e[0].abs() < 1e-9 && (e[1] - 30.0).abs() < 1e-9 && e[2].abs() < 1e-9,
            "{e:?}"
        );
        let rolled = matmul3(
            &cam_from_up,
            &matmul3(&from_euler_deg(&[0.0, 0.0, 20.0]), &rest_up),
        );
        let e = euler_deg(&relative_upright(&rolled, &r0, &tilt));
        assert!(e[1].abs() < 1e-9 && (e[2] - 20.0).abs() < 1e-9, "{e:?}");
    }

    /// Where the synthetic face rests: 60 cm in front of the camera.
    const REST: [f64; 3] = [0.0, 0.0, 60.0];

    /// Landmarks for a synthetic fit: all at the origin but the outer eye
    /// corners, 60 px apart on a line rolled `deg` (eye-line roll sign).
    fn eye_line(deg: f64) -> [[f64; 2]; NLM] {
        let mut lm = [[0.0; 2]; NLM];
        let (s, c) = deg.to_radians().sin_cos();
        lm[33] = [100.0, 200.0]; // subject's right eye outer (image left)
        lm[263] = [100.0 + 60.0 * c, 200.0 - 60.0 * s]; // image y is down
        lm
    }

    /// Head -> camera rotation whose rotation from the identity, read in the
    /// default `RestPose`'s upright frame, is `from_euler_deg(e)`: the legacy
    /// pose of a head that calibrated at the identity is then
    /// pitch `e[0]`, yaw `-e[1]`, roll `-e[2]`.
    fn upright(e: [f64; 3]) -> [[f64; 3]; 3] {
        let t = from_euler_deg(&[CAMERA_TILT_DEG, 0.0, 0.0]);
        matmul3(&transpose3(&t), &matmul3(&from_euler_deg(&e), &t))
    }

    /// A synthetic fit: rotation `r`, the mesh origin at `t_cm`.
    fn synthetic(r: [[f64; 3]; 3], t_cm: [f64; 3], lm: &[[f64; 2]; NLM]) -> FaceFit<'_> {
        FaceFit::new(r, t_cm, lm, 1.0)
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-9, "{got:?} != {want:?} at {k}");
        }
    }

    /// Feed `fit` until `rest` is calibrated on it: no pose comes out yet.
    fn calibrate(rest: &mut RestPose, fit: &FaceFit<'_>) {
        for i in 0..CALIB_FRAMES {
            assert!(rest.update(Some(fit)).is_none(), "calibrating, fit {i}");
        }
    }

    /// A default rest pose calibrated on the face at `REST`, facing the camera.
    fn calibrated(lm: &[[f64; 2]; NLM]) -> RestPose {
        let mut rest = RestPose::default();
        calibrate(&mut rest, &synthetic(IDENTITY3, REST, lm));
        rest
    }

    #[test]
    fn face_fit_is_in_millimetres_and_keeps_the_solver_centimetres() {
        let lm = eye_line(0.0);
        let fit = synthetic(IDENTITY3, [1.25, -2.5, 61.0], &lm);
        let bits = |v: [f64; 3]| v.map(f64::to_bits);
        assert_eq!(bits(fit.translation_mm), bits([12.5, -25.0, 610.0]));
        assert_eq!(bits(fit.translation_cm), bits([1.25, -2.5, 61.0]));
    }

    #[test]
    fn rest_pose_is_the_mean_of_the_first_fits() {
        let lm = eye_line(0.0);
        let near = synthetic(IDENTITY3, [1.0, -2.0, 60.0], &lm);
        let far = synthetic(IDENTITY3, [3.0, -2.0, 62.0], &lm);
        let mut rest = RestPose::default();
        for i in 0..CALIB_FRAMES {
            let fit = if i % 2 == 0 { &near } else { &far };
            assert!(rest.update(Some(fit)).is_none(), "calibrating, fit {i}");
        }
        // Raw: the translation at the neck pivot, 11 cm down and 6 cm back.
        assert_close(&rest.last_raw().unwrap(), &[3.0, 9.0, 68.0, 0.0, 0.0, 0.0]);
        // Rest = the mean, (2, -2, 61); the pose is its negated offset from it.
        let moved = synthetic(IDENTITY3, [3.5, -2.5, 63.0], &lm);
        let out = rest.update(Some(&moved)).unwrap();
        assert_close(&out, &[-1.5, 0.5, -2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rest_pose_reads_the_rotation_in_the_upright_frame_with_mediapipe_signs() {
        let lm = eye_line(0.0);
        let mut rest = calibrated(&lm);
        let turned = synthetic(upright([-10.0, 20.0, 5.0]), REST, &lm);
        let out = rest.update(Some(&turned)).unwrap();
        // [.., yaw, pitch, roll]; the translation moves too, as the neck pivot
        // swings with the head.
        assert_close(&out[3..], &[-20.0, -10.0, -5.0]);
        assert!(out[..3].iter().any(|v| v.abs() > 0.1), "{out:?}");
    }

    #[test]
    fn rest_pose_moves_the_translation_to_the_neck_pivot() {
        let lm = eye_line(0.0);
        let mut rest = RestPose::default();
        // A 30° turn about the camera's y axis swings the pivot, (0, 11, 6) cm
        // in the face frame, to (6 sin 30°, 11, 6 cos 30°).
        let turned = synthetic(from_euler_deg(&[0.0, 30.0, 0.0]), [1.0, 2.0, 50.0], &lm);
        assert!(rest.update(Some(&turned)).is_none());
        let (s, c) = 30f64.to_radians().sin_cos();
        let raw = rest.last_raw().unwrap();
        assert_close(
            &raw,
            &[1.0 + 6.0 * s, 13.0, 50.0 + 6.0 * c, 0.0, -30.0, 0.0],
        );
    }

    #[test]
    fn rest_pose_clamps_the_angles_to_45_degrees() {
        let lm = eye_line(0.0);
        let turned = synthetic(upright([-70.0, -60.0, 0.0]), REST, &lm);
        let out = calibrated(&lm).update(Some(&turned)).unwrap();
        assert_close(&out[3..5], &[45.0, -45.0]);
        let rolled = synthetic(upright([0.0, 0.0, 50.0]), REST, &lm);
        let out = calibrated(&lm).update(Some(&rolled)).unwrap();
        assert_close(&out[5..], &[-45.0]);
    }

    #[test]
    fn rest_pose_smooths_all_six_outputs_after_the_first() {
        let lm = eye_line(0.0);
        let a = synthetic(upright([2.0, -4.0, 6.0]), [2.0, 0.0, 60.0], &lm);
        let b = synthetic(upright([10.0, -12.0, 14.0]), [6.0, -4.0, 64.0], &lm);
        // What `b` reads unsmoothed: as the first pose after calibration.
        let target = calibrated(&lm).update(Some(&b)).unwrap();
        let mut rest = calibrated(&lm);
        let mut want = rest.update(Some(&a)).unwrap();
        assert_close(&want[3..4], &[4.0]); // `a`'s yaw, as it is
        for step in 0..3 {
            for (w, t) in want.iter_mut().zip(&target) {
                *w += 0.5 * (t - *w);
            }
            assert_close(&rest.update(Some(&b)).unwrap(), &want);
            assert!(
                want.iter().zip(&target).all(|(w, t)| (w - t).abs() > 0.01),
                "every output still on its way at step {step}: {want:?} vs {target:?}"
            );
        }
    }

    #[test]
    fn recenter_recalibrates_and_restarts_the_smoothing() {
        let lm = eye_line(0.0);
        let mut rest = calibrated(&lm);
        let moved = synthetic(IDENTITY3, [2.0, 0.0, 60.0], &lm);
        assert_close(
            &rest.update(Some(&moved)).unwrap(),
            &[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        rest.recenter();
        // A new rest pose, 5 cm right, 1 cm down and 10 cm further away.
        calibrate(&mut rest, &synthetic(IDENTITY3, [5.0, 1.0, 70.0], &lm));
        let away = synthetic(IDENTITY3, [5.0, 1.0, 73.0], &lm);
        // Relative to the new rest pose, and not blended with the -2 cm before.
        assert_close(
            &rest.update(Some(&away)).unwrap(),
            &[0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn a_frame_without_a_face_only_clears_the_raw_pose() {
        let lm = eye_line(0.0);
        let at_rest = synthetic(IDENTITY3, REST, &lm);
        let mut rest = RestPose::default();
        for _ in 1..CALIB_FRAMES {
            assert!(rest.update(Some(&at_rest)).is_none());
        }
        assert!(rest.update(None).is_none());
        assert!(rest.last_raw().is_none());
        // The calibration goes on where it was: one more fit completes it.
        assert!(rest.update(Some(&at_rest)).is_none());
        assert!(rest.last_raw().is_some());
        let moved = synthetic(IDENTITY3, [2.0, 0.0, 60.0], &lm);
        assert_close(
            &rest.update(Some(&moved)).unwrap(),
            &[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        // So does the smoothing, across the frame without a face.
        assert!(rest.update(None).is_none());
        let further = synthetic(IDENTITY3, [4.0, 0.0, 60.0], &lm);
        assert_close(
            &rest.update(Some(&further)).unwrap(),
            &[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn eye_line_roll_replaces_the_fitted_roll() {
        let pivot = [0.0, PIVOT_NECK_DOWN_CM, PIVOT_NECK_BACK_CM];
        let mut rest = RestPose::new(pivot, CAMERA_TILT_DEG, true, false);
        calibrate(&mut rest, &synthetic(IDENTITY3, REST, &eye_line(0.0)));
        // Rolled 25° by the fit and 12° by the eye line: the eye line wins.
        let rolled = eye_line(12.0);
        let fit = synthetic(upright([0.0, 0.0, 25.0]), REST, &rolled);
        assert_close(&rest.update(Some(&fit)).unwrap()[5..], &[12.0]);
        assert_close(&rest.last_raw().unwrap()[5..], &[12.0]);
    }

    /// A face far from the initial crop (frame shifted so the face sits in a
    /// corner) must be found by the grid search within one sweep.
    #[test]
    fn image83_tracker_finds_face_outside_initial_crop() {
        let Ok(path) = std::env::var("TOBII_IMAGE83_FIXTURE") else {
            return;
        };
        let msg = std::fs::read(path).unwrap();
        let frame = tobii_proto::image83::decode_image_payload(&msg).unwrap();
        let (w, h) = (frame.width, frame.height);
        // Shift the 280 image by (-70, -60): the face (centre ~ (140, 130))
        // moves to ~ (70, 70), i.e. outside the centred 110-px crop at 560.
        let mut shifted = vec![0u8; w * h];
        for y in 0..h - 60 {
            for x in 0..w - 70 {
                shifted[y * w + x] = frame.pixels[(y + 60) * w + (x + 70)];
            }
        }
        let mut big = Vec::new();
        tobii_proto::image83::upscale2x_into(&shifted, w, h, &mut big);
        let mut t = Tracker::new_image83().unwrap();
        let mut found_at = None;
        for i in 0..12 {
            t.process(&big, w * 2, h * 2).unwrap();
            if t.last_raw().is_some() {
                found_at = Some(i);
                break;
            }
        }
        assert!(
            matches!(found_at, Some(i) if i < SEARCH_GRID.len()),
            "face found within one sweep, got {found_at:?}"
        );
        assert!(
            t.face.cx < 300.0 && t.face.cy < 300.0,
            "crop re-seated onto the shifted face: ({}, {})",
            t.face.cx,
            t.face.cy
        );
    }

    /// Head pose from a real 0x50e frame (user's face, so the fixture is not
    /// committed): set `TOBII_IMAGE83_FIXTURE` to a captured 78609-byte message.
    #[test]
    fn image83_frame_yields_a_face_pose() {
        let Ok(path) = std::env::var("TOBII_IMAGE83_FIXTURE") else {
            return;
        };
        let msg = std::fs::read(path).unwrap();
        let frame = tobii_proto::image83::decode_image_payload(&msg).unwrap();
        let mut big = Vec::new();
        tobii_proto::image83::upscale2x_into(&frame.pixels, frame.width, frame.height, &mut big);
        let (w, h) = (frame.width * 2, frame.height * 2);
        let mut fm = FaceModel::new().unwrap();
        let (cx, cy) = (w as f32 / 2.0, h as f32 * IMAGE83_CY_FRAC);
        let (pts, score) = fm.landmarks(&big, w, h, cx, cy, IMAGE83_CROP_HALF).unwrap();
        assert!(score > 0.0, "face detected in the IR frame, score {score}");
        let span = IMAGE83_CROP_HALF * 2.0;
        let (x0, y0) = (cx - IMAGE83_CROP_HALF, cy - IMAGE83_CROP_HALF);
        let canon = canonical_f64();
        let image2d: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| {
                [
                    f64::from(x0 + p[0] / IN as f32 * span),
                    f64::from(y0 + p[1] / IN as f32 * span),
                ]
            })
            .collect();
        let observed: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
            .collect();
        let init_r = kabsch(&canon, &observed);
        let init_depth = IMAGE83_FOCAL * spread3(&canon) / spread2(&image2d);
        let (r, t) = solve_pnp(
            &canon,
            &image2d,
            IMAGE83_FOCAL,
            w as f64 / 2.0,
            h as f64 / 2.0,
            init_r,
            init_depth,
        );
        let e = euler_deg(&r);
        println!(
            ">>> image83 score={score:.2} pitch/yaw/roll = {:.1}/{:.1}/{:.1}  t = {:.1}/{:.1}/{:.1} cm",
            e[0], e[1], e[2], t[0], t[1], t[2]
        );
        assert!(
            t[2] > 30.0 && t[2] < 150.0,
            "depth in a plausible range, got {} cm",
            t[2]
        );
        assert!(e[1].abs() < 45.0 && e[2].abs() < 45.0, "yaw/roll plausible");
    }

    /// The fit of a real 0x50e frame read back through its documented
    /// conventions (same fixture as above): a proper object -> camera
    /// rotation, the mesh origin in mm, the landmarks in the pixels of the
    /// upscaled frame. Projected with them, the canonical mesh must land on
    /// the landmarks.
    #[test]
    fn image83_fit_projects_the_mesh_onto_its_landmarks() {
        let Ok(path) = std::env::var("TOBII_IMAGE83_FIXTURE") else {
            return;
        };
        let msg = std::fs::read(path).unwrap();
        let frame = tobii_proto::image83::decode_image_payload(&msg).unwrap();
        let mut big = Vec::new();
        tobii_proto::image83::upscale2x_into(&frame.pixels, frame.width, frame.height, &mut big);
        let (w, h) = (frame.width * 2, frame.height * 2);
        let mut tracker = Tracker::new_image83().unwrap();
        let fit = tracker
            .fit(&big, w, h)
            .unwrap()
            .expect("a face in the start crop");
        let r = fit.rotation;
        let rrt = matmul3(&r, &transpose3(&r));
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        for (i, row) in rrt.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-9, "R R^T = {rrt:?}");
            }
        }
        assert!((det - 1.0).abs() < 1e-9, "det R = {det}");
        let t = fit.translation_mm;
        assert!(t[2] > 300.0 && t[2] < 1500.0, "depth {} mm", t[2]);
        let mut err: Vec<f64> = CANONICAL_FACE
            .iter()
            .zip(fit.landmarks)
            .map(|(p, lm)| {
                let q = matvec3(&r, &[f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                // The mesh is in cm, the translation in mm.
                let c = [q[0] * 10.0 + t[0], q[1] * 10.0 + t[1], q[2] * 10.0 + t[2]];
                let u = IMAGE83_FOCAL * c[0] / c[2] + w as f64 / 2.0;
                let v = IMAGE83_FOCAL * c[1] / c[2] + h as f64 / 2.0;
                (u - lm[0]).hypot(v - lm[1])
            })
            .collect();
        err.sort_by(f64::total_cmp);
        let (median, max) = (err[err.len() / 2], err[err.len() - 1]);
        println!(
            ">>> image83 fit score={:.2} t = {:.1}/{:.1}/{:.1} mm, reprojection median {median:.2} px, max {max:.2} px (560 frame)",
            fit.score, t[0], t[1], t[2]
        );
        assert!(fit.score >= 0.0);
        // The rigid canonical mesh is not the user's face: ~4.5 px median is
        // the fit's own residual on session-1 frames. Reading a convention
        // wrong (the translation as cm, the landmarks as 280-px ones, the
        // rotation as camera -> object) misses by 45-200 px.
        assert!(
            median < 8.0 && max < 30.0,
            "reprojection {median} / {max} px"
        );
    }

    fn read_pgm(path: &str) -> (Vec<u8>, usize, usize) {
        let bytes = std::fs::read(path).unwrap();
        // header: "P5\n<w> <h>\n255\n"
        let mut idx = 0;
        let mut fields = Vec::new();
        while fields.len() < 4 {
            while bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            let start = idx;
            while !bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            fields.push(std::str::from_utf8(&bytes[start..idx]).unwrap().to_string());
        }
        idx += 1; // single whitespace after maxval
        let w: usize = fields[1].parse().unwrap();
        let h: usize = fields[2].parse().unwrap();
        (bytes[idx..idx + w * h].to_vec(), w, h)
    }

    /// Regression against the reference Python `solvePnP` result for one
    /// recorded IR frame. The frame is a photograph of the user's face, so it
    /// is not committed: point `TOBII_POSE_PGM_FIXTURE` at a 560x560 binary
    /// PGM to run this (same convention as the two tests above).
    #[test]
    fn pose_matches_python() {
        let Ok(path) = std::env::var("TOBII_POSE_PGM_FIXTURE") else {
            return;
        };
        let (gray, w, h) = read_pgm(&path);
        let mut fm = FaceModel::new().unwrap();
        let cx = w as f32 / 2.0;
        let cy = h as f32 * 0.42;
        let half = 160.0f32;
        let (pts, score) = fm.landmarks(&gray, w, h, cx, cy, half).unwrap();
        let span = half * 2.0;
        let (x0, y0) = (cx - half, cy - half);
        let canon = canonical_f64();
        let image2d: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| {
                [
                    f64::from(x0 + p[0] / IN as f32 * span),
                    f64::from(y0 + p[1] / IN as f32 * span),
                ]
            })
            .collect();
        let observed: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
            .collect();
        let init_r = kabsch(&canon, &observed);
        let init_depth = FOCAL * spread3(&canon) / spread2(&image2d);
        let (r, t) = solve_pnp(
            &canon,
            &image2d,
            FOCAL,
            w as f64 / 2.0,
            h as f64 / 2.0,
            init_r,
            init_depth,
        );
        let e = euler_deg(&r);
        // Python cv2.solvePnP (flip Y/Z, fov63): pitch -19.2, yaw +1.4, roll -4.0, t.z +39.3
        println!(
            ">>> score={score:.2} pnp pitch/yaw/roll = {:.1}/{:.1}/{:.1}  t = {:.1}/{:.1}/{:.1}",
            e[0], e[1], e[2], t[0], t[1], t[2]
        );
        assert!(score > 0.0, "face detected");
        assert!(
            (e[0] - (-19.2)).abs() < 4.0,
            "pitch close to python solvePnP"
        );
        assert!((t[2] - 39.3).abs() < 5.0, "depth ~40cm metric");
    }
}
