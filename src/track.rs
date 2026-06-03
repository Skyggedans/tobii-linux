//! Head pose from the IR face frame, fully in Rust: crop the 560x560 frame,
//! run the MediaPipe face-landmark model via ONNX Runtime (`ort`), fit the
//! canonical face mesh to the landmarks (Kabsch) and read out yaw/pitch/roll.
//! Validated against MediaPipe to within ~3 degrees.

use anyhow::{Context, Result};
use ort::session::Session;

use crate::canonical::CANONICAL_FACE;

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

pub(crate) struct FaceModel {
    session: Session,
}

impl FaceModel {
    pub(crate) fn new() -> Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(MODEL)
            .context("failed to load face landmark model")?;
        Ok(Self { session })
    }

    /// Crop a centred square around `(cx, cy)` of the wWxhH grayscale frame,
    /// bilinearly resize to 256x256, normalise to [0,1] and run the model.
    /// Returns the 468 face landmarks (x,y in input px, z relative) and the
    /// face-presence score.
    pub(crate) fn landmarks(
        &mut self,
        gray: &[u8],
        w: usize,
        h: usize,
        cx: f32,
        cy: f32,
        half: f32,
    ) -> Result<(Vec<[f32; 3]>, f32)> {
        let x0 = cx - half;
        let y0 = cy - half;
        let span = half * 2.0;
        let mut input = vec![0f32; IN * IN * 3];
        for oy in 0..IN {
            let sy = y0 + (oy as f32 + 0.5) / IN as f32 * span - 0.5;
            for ox in 0..IN {
                let sx = x0 + (ox as f32 + 0.5) / IN as f32 * span - 0.5;
                let v = bilinear(gray, w, h, sx, sy) / 255.0;
                let o = (oy * IN + ox) * 3;
                input[o] = v;
                input[o + 1] = v;
                input[o + 2] = v;
            }
        }

        let value = ort::value::Tensor::from_array(([1usize, IN, IN, 3], input))?;
        let outputs = self.session.run(ort::inputs!["input_12" => value])?;
        let (_, lm) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, score) = outputs["Identity_1"].try_extract_tensor::<f32>()?;

        let pts: Vec<[f32; 3]> = (0..NLM)
            .map(|i| [lm[i * 3], lm[i * 3 + 1], lm[i * 3 + 2]])
            .collect();
        Ok((pts, score[0]))
    }
}

fn bilinear(g: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p = |xx: usize, yy: usize| g[yy * w + xx] as f32;
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// Rotation (canonical -> observed) via Horn's quaternion method on the
/// cross-covariance, no external SVD needed.
pub(crate) fn kabsch(reference: &[[f64; 3]], current: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let n = reference.len();
    let mut rc = [0.0; 3];
    let mut cc = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            rc[k] += reference[i][k];
            cc[k] += current[i][k];
        }
    }
    for k in 0..3 {
        rc[k] /= n as f64;
        cc[k] /= n as f64;
    }
    let mut hm = [[0.0; 3]; 3];
    for i in 0..n {
        for a in 0..3 {
            for b in 0..3 {
                hm[a][b] += (reference[i][a] - rc[a]) * (current[i][b] - cc[b]);
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
    for i in 0..4 {
        nn[i][i] += shift;
    }
    let mut v = [1.0, 0.2, 0.1, 0.05];
    for _ in 0..200 {
        let mut wv = [0.0; 4];
        for r in 0..4 {
            for c in 0..4 {
                wv[r] += nn[r][c] * v[c];
            }
        }
        let m = (wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2] + wv[3] * wv[3]).sqrt();
        for i in 0..4 {
            v[i] = wv[i] / m;
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
pub(crate) fn euler_deg(r: &[[f64; 3]; 3]) -> [f64; 3] {
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

fn matvec3(r: &[[f64; 3]; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    ]
}

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
        for r in 0..6 {
            if r == i {
                continue;
            }
            let f = m[r][i] / d;
            for c in i..6 {
                m[r][c] -= f * m[i][c];
            }
            y[r] -= f * y[i];
        }
    }
    let mut x = [0.0; 6];
    for i in 0..6 {
        x[i] = if m[i][i].abs() > 1e-12 { y[i] / m[i][i] } else { 0.0 };
    }
    x
}

/// Perspective PnP via Gauss-Newton minimising reprojection error. Returns the
/// rotation (object->camera) and translation (cm), initialised from `init_r`.
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
            let res = [focal * q[0] * iz + cx - im[0], focal * q[1] * iz + cy - im[1]];
            let duq = [focal * iz, 0.0, -focal * q[0] * iz * iz];
            let dvq = [0.0, focal * iz, -focal * q[1] * iz * iz];
            // dQ/ddelta = -skew(rp)
            let sk = [[0.0, rp[2], -rp[1]], [-rp[2], 0.0, rp[0]], [rp[1], -rp[0], 0.0]];
            let mut ju = [0.0; 6];
            let mut jv = [0.0; 6];
            for j in 0..3 {
                ju[j] = duq[0] * sk[0][j] + duq[1] * sk[1][j] + duq[2] * sk[2][j];
                jv[j] = dvq[0] * sk[0][j] + dvq[1] * sk[1][j] + dvq[2] * sk[2][j];
            }
            ju[3..6].copy_from_slice(&duq);
            jv[3..6].copy_from_slice(&dvq);
            for a in 0..6 {
                jtr[a] += ju[a] * res[0] + jv[a] * res[1];
                for b in 0..6 {
                    jtj[a][b] += ju[a] * ju[b] + jv[a] * jv[b];
                }
            }
        }
        for i in 0..6 {
            jtj[i][i] += jtj[i][i] * 1e-3 + 1e-9;
        }
        let d = solve6(&jtj, &jtr);
        r = matmul3(&expmap([-d[0], -d[1], -d[2]]), &r);
        t[0] -= d[3];
        t[1] -= d[4];
        t[2] -= d[5];
    }
    (r, t)
}

fn spread3(p: &[[f64; 3]]) -> f64 {
    let n = p.len() as f64;
    let mut c = [0.0; 3];
    for q in p {
        for k in 0..3 {
            c[k] += q[k] / n;
        }
    }
    let mut s = 0.0;
    for q in p {
        s += (0..3).map(|k| (q[k] - c[k]).powi(2)).sum::<f64>();
    }
    (s / n).sqrt()
}

fn spread2(p: &[[f64; 2]]) -> f64 {
    let n = p.len() as f64;
    let mut c = [0.0; 2];
    for q in p {
        for k in 0..2 {
            c[k] += q[k] / n;
        }
    }
    let mut s = 0.0;
    for q in p {
        s += (0..2).map(|k| (q[k] - c[k]).powi(2)).sum::<f64>();
    }
    (s / n).sqrt()
}

/// Canonical mesh as f64 for the fit.
pub(crate) fn canonical_f64() -> Vec<[f64; 3]> {
    CANONICAL_FACE
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect()
}

/// Live head-pose tracker: face-following crop, landmark inference, Kabsch fit,
/// rest-pose calibration and smoothing. `process` returns the OpenTrack pose
/// [TX, TY, TZ, Yaw, Pitch, Roll] once calibrated, else `None`.
pub(crate) struct Tracker {
    model: FaceModel,
    canonical: Vec<[f64; 3]>,
    origin: Option<[f64; 6]>,
    accum: Vec<[f64; 6]>,
    out: [f64; 6],
    have: bool,
    cx: f32,
    cy: f32,
    pivot: [f64; 3], // neck offset in model cm: [0, down, back]
    debug: bool,
    frame: u64,
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Tracker {
    pub(crate) fn new() -> Result<Self> {
        let pivot = [
            0.0,
            env_f64("TOBII_PIVOT_DOWN", PIVOT_NECK_DOWN_CM),
            env_f64("TOBII_PIVOT_BACK", PIVOT_NECK_BACK_CM),
        ];
        let debug = std::env::var("TOBII_POSE_DEBUG").is_ok();
        eprintln!("tracker pivot = [down {:.1}, back {:.1}] cm", pivot[1], pivot[2]);
        Ok(Self {
            model: FaceModel::new()?,
            canonical: canonical_f64(),
            origin: None,
            accum: Vec::new(),
            out: [0.0; 6],
            have: false,
            cx: 0.0,
            cy: 0.0,
            pivot,
            debug,
            frame: 0,
        })
    }

    pub(crate) fn process(&mut self, gray: &[u8], w: usize, h: usize) -> Result<Option<[f64; 6]>> {
        if self.cx == 0.0 {
            self.cx = w as f32 / 2.0;
            self.cy = h as f32 * 0.42;
        }
        let (pts, score) = self
            .model
            .landmarks(gray, w, h, self.cx, self.cy, CROP_HALF)?;
        if score < 0.0 {
            // Face lost - re-centre the crop and wait.
            self.cx = w as f32 / 2.0;
            self.cy = h as f32 * 0.42;
            return Ok(None);
        }

        // Landmark centroid + spread in the 256 crop, mapped back to the frame.
        let span = CROP_HALF * 2.0;
        let (mut mx, mut my) = (0.0f32, 0.0f32);
        for p in &pts {
            mx += p[0];
            my += p[1];
        }
        mx /= pts.len() as f32;
        my /= pts.len() as f32;
        let mut size = 0.0f32;
        for p in &pts {
            size += ((p[0] - mx).powi(2) + (p[1] - my).powi(2)).sqrt();
        }
        size /= pts.len() as f32;
        let _ = size;
        let crop_x0 = self.cx - CROP_HALF;
        let crop_y0 = self.cy - CROP_HALF;
        self.cx = crop_x0 + mx / IN as f32 * span; // follow the face next frame
        self.cy = crop_y0 + my / IN as f32 * span;

        // Map landmarks to full-frame 2D, and keep the model's 3D for the init.
        let image2d: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| {
                [
                    (crop_x0 + p[0] / IN as f32 * span) as f64,
                    (crop_y0 + p[1] / IN as f32 * span) as f64,
                ]
            })
            .collect();
        let observed: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let init_r = kabsch(&self.canonical, &observed);
        let init_depth = FOCAL * spread3(&self.canonical) / spread2(&image2d).max(1e-6);

        // Perspective solve: metric, rotation-decoupled rotation + translation.
        let (r, t) = solve_pnp(
            &self.canonical,
            &image2d,
            FOCAL,
            w as f64 / 2.0,
            h as f64 / 2.0,
            init_r,
            init_depth,
        );
        let e = euler_deg(&r);
        // Match MediaPipe's convention (yaw/roll negate vs our euler).
        let mp = [e[0], -e[1], -e[2]]; // pitch, yaw, roll

        // Translate the reported position to the neck pivot: t' = t + R*pivot.
        // A pure head rotation about the neck then leaves t' ~constant (rotates
        // in place) instead of swinging the face origin sideways.
        let rp = matvec3(&r, &self.pivot);
        let tp = [t[0] + rp[0], t[1] + rp[1], t[2] + rp[2]];
        let raw = [tp[0], tp[1], tp[2], mp[0], mp[1], mp[2]]; // translation in cm

        self.frame += 1;
        if self.debug && self.frame % 8 == 0 {
            eprintln!(
                "yaw/pit={:+5.1}/{:+5.1}  t=[{:+5.1} {:+5.1} {:+5.1}]  t'=[{:+5.1} {:+5.1} {:+5.1}] cm",
                mp[1], mp[0], t[0], t[1], t[2], tp[0], tp[1], tp[2]
            );
        }

        if self.origin.is_none() {
            self.accum.push(raw);
            if self.accum.len() >= CALIB_FRAMES {
                let mut o = [0.0; 6];
                for a in &self.accum {
                    for i in 0..6 {
                        o[i] += a[i] / self.accum.len() as f64;
                    }
                }
                self.origin = Some(o);
                eprintln!("calibrated rest pose");
            }
            return Ok(None);
        }

        let o = self.origin.unwrap();
        let tx = (raw[0] - o[0]) * TRANS_SIGN[0] * TRANS_GAIN[0];
        let ty = (raw[1] - o[1]) * TRANS_SIGN[1] * TRANS_GAIN[1];
        let tz = (raw[2] - o[2]) * TRANS_SIGN[2] * TRANS_GAIN[2];
        let clamp = |v: f64| v.clamp(-CLAMP_DEG, CLAMP_DEG);
        let pitch = clamp((raw[3] - o[3]) * ANGLE_SIGN[0]);
        let yaw = clamp((raw[4] - o[4]) * ANGLE_SIGN[1]);
        let roll = clamp((raw[5] - o[5]) * ANGLE_SIGN[2]);
        let (tx, ty, tz) = if SEND_TRANSLATION {
            (tx, ty, tz)
        } else {
            (0.0, 0.0, 0.0)
        };
        let target = [tx, ty, tz, yaw, pitch, roll]; // OpenTrack order

        if !self.have {
            self.out = target;
            self.have = true;
        } else {
            for i in 0..6 {
                self.out[i] += SMOOTH * (target[i] - self.out[i]);
            }
        }
        Ok(Some(self.out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn pose_matches_python() {
        let (gray, w, h) = read_pgm("frame000.pgm");
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
            .map(|p| [
                (x0 + p[0] / IN as f32 * span) as f64,
                (y0 + p[1] / IN as f32 * span) as f64,
            ])
            .collect();
        let observed: Vec<[f64; 3]> = pts.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
        let init_r = kabsch(&canon, &observed);
        let init_depth = FOCAL * spread3(&canon) / spread2(&image2d);
        let (r, t) = solve_pnp(&canon, &image2d, FOCAL, w as f64 / 2.0, h as f64 / 2.0, init_r, init_depth);
        let e = euler_deg(&r);
        // Python cv2.solvePnP (flip Y/Z, fov63): pitch -19.2, yaw +1.4, roll -4.0, t.z +39.3
        println!(">>> score={score:.2} pnp pitch/yaw/roll = {:.1}/{:.1}/{:.1}  t = {:.1}/{:.1}/{:.1}",
            e[0], e[1], e[2], t[0], t[1], t[2]);
        assert!(score > 0.0, "face detected");
        assert!((e[0] - (-19.2)).abs() < 4.0, "pitch close to python solvePnP");
        assert!((t[2] - 39.3).abs() < 5.0, "depth ~40cm metric");
    }
}
