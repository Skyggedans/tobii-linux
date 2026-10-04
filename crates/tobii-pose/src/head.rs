//! The head pose the Stream Engine reports (`tobii_head_pose_t`), made from
//! the tracker's fits ([`FaceFit`]): absolute, in the display frame, one for
//! every 0x50e image, and invalid where the Stream Engine's would be.
//!
//! Three frames: C, the camera's (x right in the image, y down, z forward),
//! which a fit is in; S, the tracker's ([`tobii_ipc::geometry`]), with
//! `p_S = D p_C` and `D = diag(-1, -1, 1)`; and T, the display's, which the
//! display area fixes ([`DisplayFrame`]). For each image:
//!
//! - **Validity**: a face whose 468 landmarks have their centroid at least
//!   6 px inside the 280 px frame and the tip of the nose at most 4 px
//!   outside it ([`FaceEdges`]), a display frame, and every output finite.
//!   The Stream Engine's four validity flags all take this one value.
//! - **Rotation**: the head's frame in S is `H = D R D Q`, with `R` the
//!   fit's rotation and `Q` a fixed turn from the canonical mesh's zero to
//!   the Stream Engine's ([`HeadParams::rotation_offset`]); the pose's
//!   angles are the yxz angles ([`euler_yxz`]) of `R_TS H`.
//! - **Position**: the `PnP` branch moves the fit's translation to a point
//!   between the eyes, keeps its distance from the camera and corrects its
//!   direction. An eye branch, which ranges the head by how far apart its
//!   eyes are in the image, can be blended in; it is off in
//!   [`HeadParams::FITTED`].
//! - **Filters**: an EMA on the position, in S, and a one-euro filter on
//!   each yxz angle, in degrees. They keep their state across invalid
//!   frames, stepping by the time since the last valid one, and start again
//!   from the input (rates 0) at the first valid frame, when the time does
//!   not move forward or jumps by more than a second, after a result that is
//!   not finite and on [`HeadPoseEstimator::reset`].
//! - **Output**: a valid pose carries the filtered values; an invalid one
//!   those of the last valid pose, zeros before the first.
//!
//! [`HeadPoseEstimator`] does this for a stream of fits. [`HeadStep`] is the
//! whole step from an image to the pose, the legacy relative pose
//! ([`RestPose`]) included, that the engine's pose worker and the replay
//! tools share.
//!
//! The model is the one the head-pose study fitted to the Windows Stream
//! Engine's output on three captured sessions; each function here follows
//! the study's Python reference implementation, whose test vectors the tests
//! reproduce.

use std::f64::consts::PI;

use anyhow::Result;
use tobii_ipc::geometry::DisplayFrame;
use tracing::debug;

use crate::track::{
    FaceFit, FaceFitter, Geometry, ModelRuns, NLM, OnnxModels, RestPose, matmul3, matvec3,
};

/// Side of the 0x50e stream's frames, px: a [`FaceFit`]'s landmarks are in
/// their pixels.
const FRAME_PX: f64 = 280.0;
const _: () = assert!(Geometry::IMAGE83.native_size == 280);

/// The landmark at the tip of the nose.
const NOSE_TIP: usize = 1;

/// Floor under `1 - (x_head · m)²` in the eye branch, the square of how much
/// of the eyes' baseline lies across the line of sight `m`: a head turned
/// side-on keeps a finite range. The value the study fitted with.
const ACROSS_SIGHT_FLOOR: f64 = 1e-6;

/// One one-euro filter's settings, for one yxz angle in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneEuro {
    /// Cutoff frequency at rest, Hz.
    pub min_cutoff_hz: f64,
    /// How fast the cutoff rises with the angle's rate, Hz per °/s.
    pub beta: f64,
}

/// A pinhole camera's intrinsics, as the eye branch maps a pixel `(u, v)`
/// of the 0x50e frame to the ray `((u - cu) / fu, (v - cv) / fv, 1)` of the
/// tracker frame S (so the focal lengths are negative: S flips x and y).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayIntrinsics {
    /// Focal length across, px.
    pub fu: f64,
    /// Principal point across, px.
    pub cu: f64,
    /// Focal length down, px.
    pub fv: f64,
    /// Principal point down, px.
    pub cv: f64,
}

/// How a branch of the position model corrects the direction of the point
/// it found: the point keeps its distance from the camera, and the tangents
/// `x / z` and `y / z` of its direction become `gain * tangent + offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionMap {
    /// Gain on both tangents.
    pub gain: f64,
    /// Offsets added to the x and the y tangent.
    pub offset: [f64; 2],
}

/// Every constant of the head pose model, so that tools can try others; the
/// study's names in parentheses. [`HeadParams::FITTED`] is what the engine
/// runs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HeadParams {
    /// (`Q`) The head's frame in the canonical mesh's, x and y flipped as S
    /// is from C: the head's rotation in S is `D R D Q`.
    pub rotation_offset: [[f64; 3]; 3],
    /// (`s`) Scale on the fit's translation.
    pub pnp_scale: f64,
    /// (`d`) The point the position is of, in the head's frame, mm from the
    /// canonical mesh's origin: about the middle of the eyes' corners, a few
    /// mm in front of them.
    pub pnp_point_mm: [f64; 3],
    /// (`g_p`, `ox_p`, `oy_p`) The `PnP` branch's direction correction.
    pub pnp_direction: DirectionMap,
    /// (`fu`, `cu`, `fv`, `cv`) The intrinsics the eye branch casts rays
    /// with: the 0x50e camera's own.
    pub eye_rays: RayIntrinsics,
    /// The landmarks whose centroid is an eye in the eye branch: the 16
    /// around the eye on the image's left (the subject's right eye), then
    /// the 16 around the other.
    pub eye_contours: [[usize; 16]; 2],
    /// (`K`) How far apart the eye branch takes the eyes to be, mm: it
    /// ranges the head at `K fs / sin(angle)`, for eyes seen `angle` apart
    /// whose baseline lies `fs` across the line of sight.
    pub eye_range_mm: f64,
    /// (`g_e`, `ox_e`, `oy_e`) The eye branch's direction correction.
    pub eye_direction: DirectionMap,
    /// (`w`) How much of the eye branch's difference from the `PnP` branch
    /// the position takes; 0 turns the eye branch off.
    pub eye_weight: f64,
    /// (`tau_pos`) Time constant of the position's EMA, s: the EMA moves
    /// `dt / (dt + tau)` of the way to its input.
    pub position_tau_s: f64,
    /// (`tau_beta`) Time constant of the eye correction's low-pass, s.
    pub correction_tau_s: f64,
    /// The one-euro filters of the yxz angles x, y and z.
    pub rotation_filters: [OneEuro; 3],
    /// Cutoff of the one-euro filters' rate low-pass, Hz.
    pub rotation_derivative_cutoff_hz: f64,
    /// Longest time between two valid frames that the filters carry their
    /// state across, s.
    pub reset_gap_s: f64,
    /// Validity: least distance of the landmarks' centroid inside the
    /// frame's nearest edge, px.
    pub min_centroid_edge_px: f64,
    /// Validity: least signed distance of the tip of the nose inside the
    /// frame's nearest edge, px (negative: outside).
    pub min_nose_tip_edge_px: f64,
}

impl HeadParams {
    /// The constants of the Python port of the head-pose study, fitted on all
    /// three Windows sessions (in sample, 2026-09-29) against the Stream
    /// Engine's own head pose, with the study's Python port of the tracker
    /// (`f_mp150_bf` crops, `BlazeFace` re-acquisition) — but for two
    /// decisions: the position is the `PnP` branch's alone (the eye weight is
    /// 0, where the fit gave 0.383; the eye branch keeps its constants, for
    /// comparison), and the three angles share one one-euro filter of 1.0 Hz
    /// and 0.2 Hz per °/s (the fit had 1.0 and 0.4, 0.7 and 0.2, 1.5 and 0.1
    /// for x, y and z). They are to be fitted again on this tracker's fits.
    pub const FITTED: Self = Self {
        rotation_offset: [
            [
                0.999_860_075_374_677_1,
                -0.007_035_297_618_850_274,
                -0.015_176_767_085_183_904,
            ],
            [
                0.006_109_694_286_072_543,
                0.998_167_924_520_407_6,
                -0.060_195_233_153_067_08,
            ],
            [
                0.015_572_453_482_815_5,
                0.060_094_084_950_480_346,
                0.998_071_239_765_223,
            ],
        ],
        pnp_scale: 0.971_429_553_944_983_1,
        pnp_point_mm: [
            -2.824_343_736_189_661_5,
            27.404_881_620_713_358,
            -36.483_212_590_351_55,
        ],
        pnp_direction: DirectionMap {
            gain: 1.013_556_746_417_381_4,
            offset: [0.001_815_721_453_967_718_2, -0.007_409_409_876_383_765_5],
        },
        eye_rays: RayIntrinsics {
            fu: -375.9,
            cu: 140.9,
            fv: -383.3,
            cv: 139.8,
        },
        eye_contours: [
            [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            ],
            [
                263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
            ],
        ],
        eye_range_mm: 64.291_648_924_177_8,
        eye_direction: DirectionMap {
            gain: 1.020_828_932_951_275_4,
            offset: [-0.004_033_985_018_082_697, -0.000_262_390_956_082_422_75],
        },
        eye_weight: 0.0,
        // Each EMA moves 0.3 of the way (dt / (dt + tau) = 0.30) at the
        // stream's 30.208 ms frame interval.
        position_tau_s: 0.070_485_333_333_333_33,
        correction_tau_s: 0.070_485_333_333_333_33,
        rotation_filters: [OneEuro {
            min_cutoff_hz: 1.0,
            beta: 0.2,
        }; 3],
        rotation_derivative_cutoff_hz: 1.0,
        reset_gap_s: 1.0,
        min_centroid_edge_px: 6.0,
        min_nose_tip_edge_px: -4.0,
    };

    /// A fingerprint of the constants, to tell a model by in reports:
    /// FNV-1a (64 bits) of every number in the order of the fields, each
    /// `f64` as the little-endian bytes of its bits and each landmark index
    /// as those of a `u64`. Unlike their `Debug` form, which a Rust release
    /// may print otherwise, the bytes stay as long as the constants do.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        // Every field, so that one added must be added here too.
        let Self {
            rotation_offset,
            pnp_scale,
            pnp_point_mm,
            pnp_direction,
            eye_rays,
            eye_contours,
            eye_range_mm,
            eye_direction,
            eye_weight,
            position_tau_s,
            correction_tau_s,
            rotation_filters,
            rotation_derivative_cutoff_hz,
            reset_gap_s,
            min_centroid_edge_px,
            min_nose_tip_edge_px,
        } = *self;
        let direction = |map: DirectionMap| [map.gain, map.offset[0], map.offset[1]];
        let RayIntrinsics { fu, cu, fv, cv } = eye_rays;
        let numbers = rotation_offset
            .into_iter()
            .flatten()
            .chain([pnp_scale])
            .chain(pnp_point_mm)
            .chain(direction(pnp_direction))
            .chain([fu, cu, fv, cv])
            .map(f64::to_bits)
            // cast: a landmark index, below 468
            .chain(eye_contours.into_iter().flatten().map(|i| i as u64))
            .chain(
                [eye_range_mm]
                    .into_iter()
                    .chain(direction(eye_direction))
                    .chain([eye_weight, position_tau_s, correction_tau_s])
                    .chain(
                        rotation_filters
                            .into_iter()
                            .flat_map(|f| [f.min_cutoff_hz, f.beta]),
                    )
                    .chain([
                        rotation_derivative_cutoff_hz,
                        reset_gap_s,
                        min_centroid_edge_px,
                        min_nose_tip_edge_px,
                    ])
                    .map(f64::to_bits),
            );
        numbers
            .flat_map(u64::to_le_bytes)
            .fold(0xcbf2_9ce4_8422_2325, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            })
    }
}

/// One head pose, as the Stream Engine reports it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadPose {
    /// Whether the pose is valid: the value of all four of the Stream
    /// Engine's validity flags. An invalid pose carries the values of the
    /// last valid one, zeros before the first.
    pub valid: bool,
    /// The head's position in the display frame T, mm: a point between the
    /// eyes, from the centre of the display area, x to the user's right, y
    /// up and z towards the user.
    pub position_mm: [f64; 3],
    /// The head's rotation in T, radians: the yxz angles (x, y, z) of
    /// [`euler_yxz`]. All zero when the face looks along −z; x grows as the
    /// chin rises, y as the head turns to its left, z as it tips its top
    /// towards its left shoulder.
    pub rotation_rad: [f64; 3],
}

/// The yxz Euler angles `[x, y, z]` of the rotation `r`, radians:
/// `r = Ry(y) Rx(x) Rz(z)` ([`compose_yxz`]), with x in [−π/2, π/2]. This is
/// how the Stream Engine reads its head rotation; the zyx order of
/// [`euler_deg`](crate::track::euler_deg) reads combined turns quite
/// differently.
#[must_use]
pub fn euler_yxz(r: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        (-r[1][2]).clamp(-1.0, 1.0).asin(),
        r[0][2].atan2(r[2][2]),
        r[1][0].atan2(r[1][1]),
    ]
}

/// The rotation `Ry(y) Rx(x) Rz(z)` of the yxz Euler angles `[x, y, z]`,
/// radians: the inverse of [`euler_yxz`].
#[must_use]
pub fn compose_yxz(angles: [f64; 3]) -> [[f64; 3]; 3] {
    let [x, y, z] = angles;
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    matmul3(&matmul3(&ry, &rx), &rz)
}

/// How far a face's landmarks are inside the 280 px frame, the two measures
/// of the pose's validity: each the signed distance of a point from the
/// frame's nearest edge, `min(u, 280 − u, v, 280 − v)`, px, negative
/// outside; NaN for a point that is NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceEdges {
    /// The distance of the centroid of the 468 landmarks.
    pub centroid_px: f64,
    /// The distance of the tip of the nose (landmark 1).
    pub nose_tip_px: f64,
}

impl FaceEdges {
    /// The measures of `fit`'s landmarks.
    #[must_use]
    pub fn of(fit: &FaceFit<'_>) -> Self {
        let mut sum = [0.0; 2];
        for p in fit.landmarks {
            sum[0] += p[0];
            sum[1] += p[1];
        }
        let n = NLM as f64;
        Self {
            centroid_px: edge_px([sum[0] / n, sum[1] / n]),
            nose_tip_px: edge_px(fit.landmarks[NOSE_TIP]),
        }
    }
}

/// The signed distance of `p` from the frame's nearest edge, px.
fn edge_px(p: [f64; 2]) -> f64 {
    let [u, v] = p;
    // f64::min would pass over a NaN: an unusable point must fail the check.
    if u.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    u.min(FRAME_PX - u).min(v).min(FRAME_PX - v)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `v` scaled to unit length (divided by its length, as the reference does).
fn unit(v: [f64; 3]) -> [f64; 3] {
    let length = dot(v, v).sqrt();
    v.map(|x| x / length)
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// The head's rotation in S, `D R D Q`, of a fit's rotation `r` (object →
/// camera) and the offset `Q`.
fn head_rotation(r: &[[f64; 3]; 3], offset: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // D R D: the entries that pair z with x or y change sign.
    let [[a, b, c], [d, e, f], [g, h, i]] = *r;
    matmul3(&[[a, b, -c], [d, e, -f], [-g, -h, i]], offset)
}

/// The point `distance` from the camera in the direction `direction` has
/// after `map`.
fn mapped(distance: f64, direction: [f64; 3], map: &DirectionMap) -> [f64; 3] {
    let [x, y, z] = direction;
    unit([
        map.gain * x / z + map.offset[0],
        map.gain * y / z + map.offset[1],
        1.0,
    ])
    .map(|v| distance * v)
}

/// The `PnP` branch's position in S, mm: the fit's translation (the mesh's
/// origin in C, mm) scaled and moved to the head's point, then its direction
/// corrected.
fn pnp_position(translation_mm: [f64; 3], head: &[[f64; 3]; 3], params: &HeadParams) -> [f64; 3] {
    let [x, y, z] = translation_mm;
    let s = params.pnp_scale;
    let point = add([s * -x, s * -y, s * z], matvec3(head, &params.pnp_point_mm));
    mapped(dot(point, point).sqrt(), point, &params.pnp_direction)
}

/// The ray in S through the pixel `p` of the 0x50e frame, unit length.
fn ray(p: [f64; 2], k: &RayIntrinsics) -> [f64; 3] {
    unit([(p[0] - k.cu) / k.fu, (p[1] - k.cv) / k.fv, 1.0])
}

/// The eye branch's position in S, mm, of the eyes seen at `left` and
/// `right` (px of the 0x50e frame) on a head turned `head` in S: the
/// distance at which eyes `K` apart look as far apart as they do (less
/// where the head turns its eyes' baseline along the line of sight), in the
/// direction between them, corrected.
fn eye_position(
    left: [f64; 2],
    right: [f64; 2],
    head: &[[f64; 3]; 3],
    params: &HeadParams,
) -> [f64; 3] {
    let (a, b) = (ray(left, &params.eye_rays), ray(right, &params.eye_rays));
    let sight = unit(add(a, b));
    let apart = dot(a, b).clamp(-1.0, 1.0).acos();
    let x_head = [head[0][0], head[1][0], head[2][0]];
    let across = (1.0 - dot(x_head, sight).powi(2))
        .max(ACROSS_SIGHT_FLOOR)
        .sqrt();
    let range = params.eye_range_mm * across / apart.sin();
    mapped(range, sight, &params.eye_direction)
}

/// The centroid of the landmarks `contour` picks out; `None` when an index
/// is out of range.
fn contour_centroid(landmarks: &[[f64; 2]; NLM], contour: &[usize; 16]) -> Option<[f64; 2]> {
    let mut sum = [0.0; 2];
    for &i in contour {
        let p = landmarks.get(i)?;
        sum[0] += p[0];
        sum[1] += p[1];
    }
    Some([sum[0] / 16.0, sum[1] / 16.0])
}

/// The eye correction `w (p_eye − p_pnp)` (S, mm) of `fit`, or `None` when
/// the eye branch gives nothing finite.
fn eye_correction(
    fit: &FaceFit<'_>,
    head: &[[f64; 3]; 3],
    pnp_mm: [f64; 3],
    params: &HeadParams,
) -> Option<[f64; 3]> {
    let [left, right] = params
        .eye_contours
        .map(|contour| contour_centroid(fit.landmarks, &contour));
    let eye = eye_position(left?, right?, head, params);
    let correction = sub(eye, pnp_mm).map(|d| params.eye_weight * d);
    all_finite(&correction).then_some(correction)
}

/// `a` wrapped to [−180, 180) degrees.
fn wrap_deg(a: f64) -> f64 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

/// The share of the way to its input a first-order low-pass of cutoff `f`
/// Hz moves in `dt` s: `1 / (1 + 1 / (2π f dt))`.
fn smoothing(f: f64, dt: f64) -> f64 {
    1.0 / (1.0 + 1.0 / (2.0 * PI * f * dt))
}

/// An EMA of time constant `tau` s stepped `dt` s from `y` towards `x`.
fn ema(y: [f64; 3], x: [f64; 3], dt: f64, tau: f64) -> [f64; 3] {
    let a = dt / (dt + tau);
    [
        y[0] + a * (x[0] - y[0]),
        y[1] + a * (x[1] - y[1]),
        y[2] + a * (x[2] - y[2]),
    ]
}

/// One valid frame's unfiltered values, as the filters take them.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Raw {
    /// The `PnP` branch's position, S, mm.
    pnp_mm: [f64; 3],
    /// The eye correction, S, mm; `None` when the eye branch is off or gave
    /// nothing finite, which leaves the low-passed one as it was.
    correction_mm: Option<[f64; 3]>,
    /// The yxz angles, degrees.
    angles_deg: [f64; 3],
}

/// One yxz angle's one-euro filter, degrees. Its rate is that of its raw
/// input, not of its output.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AngleFilter {
    /// The filtered angle.
    out: f64,
    /// The last input.
    input: f64,
    /// The input's rate, low-passed, °/s.
    rate: f64,
}

impl AngleFilter {
    /// A filter started on `x`.
    const fn start(x: f64) -> Self {
        Self {
            out: x,
            input: x,
            rate: 0.0,
        }
    }

    /// The filter stepped `dt` s (positive) to `x`.
    fn step(self, x: f64, dt: f64, settings: &OneEuro, rate_cutoff_hz: f64) -> Self {
        let raw_rate = wrap_deg(x - self.input) / dt;
        let rate = self.rate + smoothing(rate_cutoff_hz, dt) * (raw_rate - self.rate);
        let cutoff = settings.min_cutoff_hz + settings.beta * rate.abs();
        Self {
            out: wrap_deg(self.out + smoothing(cutoff, dt) * wrap_deg(x - self.out)),
            input: x,
            rate,
        }
    }
}

/// The estimator's filter state after a valid frame.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Filters {
    /// Time of the last valid frame, µs.
    t_us: i64,
    /// The eye correction, low-passed, S, mm.
    correction_mm: [f64; 3],
    /// The filtered position, S, mm.
    position_mm: [f64; 3],
    /// The filters of the yxz angles.
    angles: [AngleFilter; 3],
}

impl Filters {
    /// The state started on `raw` at `t_us`: every output its input, the
    /// correction `raw`'s or none.
    fn start(t_us: i64, raw: &Raw) -> Self {
        let correction_mm = raw.correction_mm.unwrap_or([0.0; 3]);
        Self {
            t_us,
            correction_mm,
            position_mm: add(raw.pnp_mm, correction_mm),
            angles: raw.angles_deg.map(AngleFilter::start),
        }
    }

    /// The state stepped `dt` s (positive) to `raw` at `t_us`.
    fn step(&self, t_us: i64, raw: &Raw, dt: f64, params: &HeadParams) -> Self {
        let correction_mm = raw.correction_mm.map_or(self.correction_mm, |c| {
            ema(self.correction_mm, c, dt, params.correction_tau_s)
        });
        let position_mm = ema(
            self.position_mm,
            add(raw.pnp_mm, correction_mm),
            dt,
            params.position_tau_s,
        );
        let mut angles = self.angles;
        for ((filter, x), settings) in angles
            .iter_mut()
            .zip(raw.angles_deg)
            .zip(&params.rotation_filters)
        {
            *filter = filter.step(x, dt, settings, params.rotation_derivative_cutoff_hz);
        }
        Self {
            t_us,
            correction_mm,
            position_mm,
            angles,
        }
    }

    /// The state stepped from `last`, the state of the last valid frame (if
    /// any), to `raw` at `t_us`, or started on `raw` (see the [module
    /// docs](self)); and the step, s, when it was one.
    fn next(last: Option<&Self>, t_us: i64, raw: &Raw, params: &HeadParams) -> (Self, Option<f64>) {
        // cast: µs to s; an i64 of µs is exact in f64 below 2^53 (285 years).
        let dt = last
            .and_then(|last| t_us.checked_sub(last.t_us))
            .map(|us| us as f64 * 1e-6);
        match (last, dt) {
            (Some(last), Some(dt)) if !(dt <= 0.0 || dt > params.reset_gap_s) => {
                (last.step(t_us, raw, dt, params), Some(dt))
            }
            _ => (Self::start(t_us, raw), None),
        }
    }
}

/// The Stream Engine's head pose from a stream of fits, one step per 0x50e
/// image (see the [module docs](self)).
#[derive(Debug, Clone)]
pub struct HeadPoseEstimator {
    params: HeadParams,
    /// The filter state; `None` until the first valid frame and after a
    /// reset.
    filters: Option<Filters>,
    /// The last valid pose, whose values invalid ones carry.
    last: HeadPose,
}

impl HeadPoseEstimator {
    /// An estimator of the model `params`, before its first frame.
    #[must_use]
    pub const fn new(params: HeadParams) -> Self {
        Self {
            params,
            filters: None,
            last: HeadPose {
                valid: false,
                position_mm: [0.0; 3],
                rotation_rad: [0.0; 3],
            },
        }
    }

    /// The model's constants.
    #[must_use]
    pub const fn params(&self) -> &HeadParams {
        &self.params
    }

    /// The pose of one image: of time `t_us` (µs, what the filters step
    /// by), read while `display` was the display frame in effect, with `fit`
    /// the tracker's fit of it (`None`: no face). Invalid without a display
    /// frame, for a face too near the frame's edge and when a value is not
    /// finite; an invalid pose carries the last valid pose's values.
    #[must_use]
    pub fn step(
        &mut self,
        t_us: i64,
        display: Option<&DisplayFrame>,
        fit: Option<&FaceFit<'_>>,
    ) -> HeadPose {
        let pose = match (display, fit) {
            (Some(display), Some(fit)) if self.is_valid_face(fit) => self.track(t_us, display, fit),
            _ => None,
        };
        pose.unwrap_or(HeadPose {
            valid: false,
            ..self.last
        })
    }

    /// Restart the filters: the next valid pose is its frame's unfiltered
    /// one. For a new display area or a new open of the tracker, whose
    /// frames do not follow on from the ones before; invalid poses carry the
    /// last valid one's values as before.
    pub fn reset(&mut self) {
        self.filters = None;
    }

    /// Whether `fit` is of a face far enough inside the frame for a valid
    /// pose.
    fn is_valid_face(&self, fit: &FaceFit<'_>) -> bool {
        let edges = FaceEdges::of(fit);
        fit.score >= 0.0
            && edges.centroid_px >= self.params.min_centroid_edge_px
            && edges.nose_tip_px >= self.params.min_nose_tip_edge_px
    }

    /// The valid pose of a face that passed the edge checks, or `None` when
    /// a value is not finite (which restarts the filters).
    fn track(&mut self, t_us: i64, display: &DisplayFrame, fit: &FaceFit<'_>) -> Option<HeadPose> {
        let params = &self.params;
        let head = head_rotation(&fit.rotation, &params.rotation_offset);
        let angles_deg = euler_yxz(&matmul3(&display.rotation(), &head)).map(f64::to_degrees);
        let pnp_mm = pnp_position(fit.translation_mm, &head, params);
        if !(all_finite(&angles_deg) && all_finite(&pnp_mm)) {
            self.filters = None;
            return None;
        }
        let correction_mm = if params.eye_weight == 0.0 {
            None
        } else {
            eye_correction(fit, &head, pnp_mm, params)
        };
        let raw = Raw {
            pnp_mm,
            correction_mm,
            angles_deg,
        };
        let (filters, _) = Filters::next(self.filters.as_ref(), t_us, &raw, params);
        let pose = HeadPose {
            valid: true,
            position_mm: display.to_display(filters.position_mm),
            rotation_rad: filters.angles.map(|a| a.out.to_radians()),
        };
        if all_finite(&pose.position_mm) && all_finite(&pose.rotation_rad) {
            self.filters = Some(filters);
            self.last = pose;
            Some(pose)
        } else {
            self.filters = None;
            None
        }
    }
}

/// What [`HeadStep::step`] needs to know of a frame besides its pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameContext<'a> {
    /// The frame's time, µs, which the filters step by: the image's host
    /// time live (it never goes back, not even across a re-open), its device
    /// time in a replay.
    pub t_us: i64,
    /// The display frame of the display area in effect when the frame was
    /// read; without one the pose is invalid.
    pub display: Option<&'a DisplayFrame>,
    /// A number that changes whenever the display area in effect does (the
    /// engine's display generation): a change restarts the filters.
    pub display_generation: u64,
    /// A number that changes with each open of the tracker (the engine's
    /// open number): a change restarts the filters.
    pub open: u64,
    /// Whether to make the legacy relative pose too.
    pub legacy_wanted: bool,
}

/// What the tracker found on a frame, for analysis: copied out of its
/// [`FaceFit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FaceSeen {
    /// The landmark model's face-presence score ([`FaceFit::score`]).
    pub score: f32,
    /// Whether the face detector found the face
    /// ([`FaceFit::found_by_detector`]).
    pub found_by_detector: bool,
    /// How far inside the frame the landmarks are, which decides the pose's
    /// validity.
    pub edges: FaceEdges,
}

impl FaceSeen {
    fn of(fit: &FaceFit<'_>) -> Self {
        Self {
            score: fit.score,
            found_by_detector: fit.found_by_detector,
            edges: FaceEdges::of(fit),
        }
    }
}

/// What [`HeadStep::step`] made of a frame.
#[derive(Debug)]
#[non_exhaustive]
pub struct StepOutput {
    /// The head pose: one for every frame, valid or not.
    pub head: HeadPose,
    /// The legacy pose [TX, TY, TZ (cm), Yaw, Pitch, Roll (deg)] of
    /// [`RestPose::update`]: only when it was wanted, the frame had a face
    /// and the rest pose is calibrated.
    pub legacy: Option<[f64; 6]>,
    /// What the tracker found; `None` without a face.
    pub face: Option<FaceSeen>,
    /// How many times the tracker has run each model so far.
    pub model_runs: ModelRuns,
    /// Why the tracker failed on the frame, if it did. The frame then
    /// counts as one without a face, for both poses.
    pub error: Option<anyhow::Error>,
}

/// The step from one 0x50e image to the head poses made of it, which the
/// engine's pose worker and the replay tools share: the tracker
/// ([`Geometry::IMAGE83`]: feed it the stream's 280x280 frames as they
/// come), the absolute pose of a [`HeadPoseEstimator`] for every frame and,
/// when wanted, the legacy relative one of a [`RestPose`] — with the rules
/// for restarting them.
pub struct HeadStep {
    fitter: FaceFitter<OnnxModels>,
    poses: Poses,
}

impl HeadStep {
    /// A step for the head pose model `params`, its legacy pose the daemon's
    /// ([`RestPose::from_env`]).
    ///
    /// # Errors
    /// Fails when a model cannot be loaded.
    pub fn new(params: HeadParams) -> Result<Self> {
        Ok(Self {
            fitter: FaceFitter::with_geometry(Geometry::IMAGE83)?,
            poses: Poses::new(params, RestPose::from_env()),
        })
    }

    /// Make the poses of one `width`x`height` grey `frame` of the 0x50e
    /// stream, of which `context` tells the rest: always a head pose, and
    /// the legacy pose when wanted. A tracker failure (a frame of another
    /// size, a model that fails) makes an invalid pose too, and comes with
    /// it ([`StepOutput::error`]).
    ///
    /// The filters of the head pose restart when the display generation or
    /// the open changes; the rest pose of the legacy one recalibrates when
    /// it becomes wanted again after the last step, or after
    /// [`HeadStep::reset`], did not want it.
    #[must_use]
    pub fn step(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        context: &FrameContext<'_>,
    ) -> StepOutput {
        let fit = self.fitter.fit(frame, width, height);
        let mut out = self.poses.step_with_fit(fit, context);
        out.model_runs = self.fitter.model_runs();
        out
    }

    /// Start over as a new step would, but for the tracker, which goes on
    /// following the face: for the head pose wanted again after a time it was
    /// not. The head pose restarts its filters and invalid poses carry zeros
    /// until the next valid one; the legacy pose recalibrates once it is
    /// wanted.
    pub fn reset(&mut self) {
        self.poses.reset();
    }

    /// Recalibrate the legacy pose's rest pose from the next frames. The
    /// head pose is absolute and does not change.
    pub fn recenter(&mut self) {
        self.poses.recenter();
    }

    /// The head pose model's constants.
    #[must_use]
    pub const fn params(&self) -> &HeadParams {
        self.poses.estimator.params()
    }

    /// How many times the tracker has run each model.
    #[must_use]
    pub const fn model_runs(&self) -> ModelRuns {
        self.fitter.model_runs()
    }
}

/// Everything of a [`HeadStep`] but the tracker: the two poses it makes of
/// each fit and the rules that restart them. The tests drive it with
/// synthetic fits.
#[derive(Debug, Clone)]
struct Poses {
    estimator: HeadPoseEstimator,
    rest: RestPose,
    /// The display generation and the open of the last frame.
    last_frame: Option<(u64, u64)>,
    /// Whether the last frame wanted the legacy pose; `Some(false)` after a
    /// reset, `None` before the first frame.
    legacy_was_wanted: Option<bool>,
}

impl Poses {
    const fn new(params: HeadParams, rest: RestPose) -> Self {
        Self {
            estimator: HeadPoseEstimator::new(params),
            rest,
            last_frame: None,
            legacy_was_wanted: None,
        }
    }

    /// See [`HeadStep::step`]: the poses of a frame the tracker made `fit`
    /// of.
    fn step_with_fit(
        &mut self,
        fit: Result<Option<FaceFit<'_>>>,
        context: &FrameContext<'_>,
    ) -> StepOutput {
        let frame = (context.display_generation, context.open);
        if self.last_frame.is_some_and(|last| last != frame) {
            debug!(
                display_generation = context.display_generation,
                open = context.open,
                "new display area or open: head pose filters restart"
            );
            self.estimator.reset();
        }
        self.last_frame = Some(frame);
        let (fit, error) = match fit {
            Ok(fit) => (fit, None),
            Err(e) => (None, Some(e)),
        };
        let head = self
            .estimator
            .step(context.t_us, context.display, fit.as_ref());
        let legacy = if context.legacy_wanted {
            if self.legacy_was_wanted == Some(false) {
                self.rest.recenter();
            }
            self.rest.update(fit.as_ref())
        } else {
            None
        };
        self.legacy_was_wanted = Some(context.legacy_wanted);
        StepOutput {
            head,
            legacy,
            face: fit.as_ref().map(FaceSeen::of),
            model_runs: ModelRuns::default(),
            error,
        }
    }

    /// See [`HeadStep::reset`].
    fn reset(&mut self) {
        self.estimator = HeadPoseEstimator::new(self.estimator.params);
        self.legacy_was_wanted = Some(false);
    }

    /// See [`HeadStep::recenter`].
    fn recenter(&mut self) {
        self.rest.recenter();
    }
}

#[cfg(test)]
// reason: unwrap on fixtures is the idiomatic test failure (test-* rules).
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::track::canonical_f64;
    use anyhow::anyhow;
    use std::f64::consts::FRAC_PI_2;
    use tobii_ipc::geometry::DisplayArea;

    type Mat3 = [[f64; 3]; 3];

    const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// The vectors' tolerance: 1e-9 of the value, absolute below 1.
    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "{what}: {got:?} != {want:?} at {k}"
            );
        }
    }

    fn assert_close_mat(got: &Mat3, want: &Mat3, what: &str) {
        assert_close(got.as_flattened(), want.as_flattened(), what);
    }

    fn assert_pose(got: &HeadPose, want: &HeadPose, what: &str) {
        assert_eq!(got.valid, want.valid, "{what}: {got:?}");
        assert_close(&got.position_mm, &want.position_mm, what);
        assert_close(&got.rotation_rad, &want.rotation_rad, what);
    }

    /// The constants the study's test vectors (`test_vectors.json`) were
    /// made with: [`HeadParams::FITTED`] but for the eye weight and the
    /// per-axis one-euro filters the fit gave.
    fn vector_params() -> HeadParams {
        HeadParams {
            eye_weight: 0.383_418_503_706_659_53,
            rotation_filters: [
                OneEuro {
                    min_cutoff_hz: 1.0,
                    beta: 0.4,
                },
                OneEuro {
                    min_cutoff_hz: 0.7,
                    beta: 0.2,
                },
                OneEuro {
                    min_cutoff_hz: 1.5,
                    beta: 0.1,
                },
            ],
            ..HeadParams::FITTED
        }
    }

    /// The display frame of area A, the display area of the three Windows
    /// sessions.
    fn frame_a() -> DisplayFrame {
        DisplayFrame::new(&DisplayArea {
            top_left_mm: [-315.7917175292969, 324.4114685058594, 111.23909759521484],
            top_right_mm: [317.7962341308594, 324.4114685058594, 111.23909759521484],
            bottom_left_mm: [-315.7917175292969, 10.267330169677734, -3.100020170211792],
        })
        .expect("a display frame")
    }

    /// `x` rounded to 1e-4 px, half to even, as the vectors' generator
    /// (`NumPy`) rounded its landmarks.
    fn round4(x: f64) -> f64 {
        (x * 10_000.0).round_ties_even() / 10_000.0
    }

    /// The vectors' landmark arrays: the canonical mesh (in mm) posed by `r`
    /// and `t_mm` in front of a pinhole camera, `u = 376 X / Z + 140`,
    /// `v = 376 Y / Z + 140`, rounded. Worked out in the generator's order,
    /// they come out bit for bit.
    fn projected(r: &Mat3, t_mm: [f64; 3]) -> Box<[[f64; 2]; NLM]> {
        let mut landmarks = Box::new([[0.0; 2]; NLM]);
        for (out, p) in landmarks.iter_mut().zip(canonical_f64()) {
            let c = p.map(|v| 10.0 * v);
            let x = [0, 1, 2].map(|j| c[0] * r[j][0] + c[1] * r[j][1] + c[2] * r[j][2] + t_mm[j]);
            *out = [
                round4(376.0 * x[0] / x[2] + 140.0),
                round4(376.0 * x[1] / x[2] + 140.0),
            ];
        }
        landmarks
    }

    /// The centroid of `landmarks`, summed in order, as the generator's
    /// `NumPy` sums down a column.
    fn centroid(landmarks: &[[f64; 2]; NLM]) -> [f64; 2] {
        let mut sum = [0.0; 2];
        for p in landmarks {
            sum[0] += p[0];
            sum[1] += p[1];
        }
        [sum[0] / 468.0, sum[1] / 468.0]
    }

    /// `landmarks` moved by `(du, dv)` and rounded again.
    fn shifted(landmarks: &[[f64; 2]; NLM], du: f64, dv: f64) -> Box<[[f64; 2]; NLM]> {
        let mut out = Box::new(*landmarks);
        for p in out.iter_mut() {
            *p = [round4(p[0] + du), round4(p[1] + dv)];
        }
        out
    }

    /// A fit of rotation `r` and translation `t_mm` (camera frame) to
    /// `landmarks`.
    fn fit_of(r: Mat3, t_mm: [f64; 3], landmarks: &[[f64; 2]; NLM], score: f32) -> FaceFit<'_> {
        let mut fit = FaceFit::new(r, t_mm.map(|v| v / 10.0), landmarks, score, false);
        // Exactly the vectors' millimetres, which the trip through cm could
        // change in the last bit.
        fit.translation_mm = t_mm;
        fit
    }

    /// `test_vectors.json` "`euler_yxz`": angles (x, y, z) in degrees,
    /// `R = Ry(y) Rx(x) Rz(z)`, and the yxz angles of `R` in radians.
    struct Euler {
        degrees: [f64; 3],
        r: Mat3,
        yxz: [f64; 3],
    }

    /// `test_vectors.json` "rotation": a fit's rotation, the head's in S,
    /// the head's in T (area A) and its yxz angles.
    struct Rotation {
        r_cam: Mat3,
        head: Mat3,
        r_t: Mat3,
        yxz: [f64; 3],
    }

    /// `test_vectors.json` "`position_model`": a fit's rotation and
    /// translation and the centroids of its eye contours (image left, image
    /// right; px); the head's rotation in S; the `PnP` branch's position and
    /// the eye branch's, the correction (weight 0.383), the blend (all S,
    /// mm); the blend and the `PnP` branch's position in T (area A).
    struct Position {
        r_cam: Mat3,
        t_mm: [f64; 3],
        eyes: [[f64; 2]; 2],
        head: Mat3,
        pnp: [f64; 3],
        eye: [f64; 3],
        correction: [f64; 3],
        raw: [f64; 3],
        raw_t: [f64; 3],
        pnp_t: [f64; 3],
    }

    /// `test_vectors.json` "filters" (the rule that keeps the state across
    /// invalid frames): a frame's time, its input (`None`: invalid), the step
    /// the filters took (`None`: they restarted), and their position (S,
    /// mm) and angles (degrees).
    struct FilterRow {
        t_us: i64,
        input: Option<Raw>,
        dt_s: Option<f64>,
        out_mm: [f64; 3],
        out_deg: [f64; 3],
    }

    /// A face of `test_vectors.json` "`estimator_sequences`": score, fit and
    /// landmarks [`projected`] from it, moved, for the frame that is too
    /// near the edge, so that their centroid is `left_edge_px` from the left
    /// edge.
    struct Face {
        score: f32,
        r_cam: Mat3,
        t_mm: [f64; 3],
        left_edge_px: Option<f64>,
    }

    impl Face {
        fn landmarks(&self) -> Box<[[f64; 2]; NLM]> {
            let landmarks = projected(&self.r_cam, self.t_mm);
            match self.left_edge_px {
                Some(edge) => shifted(&landmarks, edge - centroid(&landmarks)[0], 0.0),
                None => landmarks,
            }
        }
    }

    /// A frame of `test_vectors.json` "`estimator_sequences`": its time and
    /// its face, if any.
    struct Frame {
        t_us: i64,
        face: Option<Face>,
    }

    /// The landmarks of each frame of [`SEQUENCE`].
    fn sequence_landmarks() -> Vec<Option<Box<[[f64; 2]; NLM]>>> {
        SEQUENCE
            .iter()
            .map(|frame| frame.face.as_ref().map(Face::landmarks))
            .collect()
    }

    /// The fit of [`SEQUENCE`]'s frame `i`, whose landmarks are `landmarks`.
    fn sequence_fit(i: usize, landmarks: &[Option<Box<[[f64; 2]; NLM]>>]) -> Option<FaceFit<'_>> {
        let face = SEQUENCE[i].face.as_ref()?;
        let landmarks = landmarks[i].as_deref()?;
        Some(fit_of(face.r_cam, face.t_mm, landmarks, face.score))
    }

    #[test]
    fn fitted_turns_the_eye_branch_off_and_shares_one_filter_between_the_angles() {
        let fitted = HeadParams::FITTED;
        assert_eq!(fitted.eye_weight.to_bits(), 0.0f64.to_bits());
        let shared = OneEuro {
            min_cutoff_hz: 1.0,
            beta: 0.2,
        };
        assert_eq!(fitted.rotation_filters, [shared; 3]);
        // The rest is the fit's: Q turns the mesh by the yxz angles the study
        // gave, and the EMA moves 0.3 of the way at the stream's frame
        // interval.
        let q = euler_yxz(&fitted.rotation_offset).map(f64::to_degrees);
        assert_close(
            &q,
            &[
                3.451_019_058_543_119_7,
                -0.871_177_981_576_780_2,
                0.350_697_829_977_867_97,
            ],
            "Q",
        );
        let dt = 0.030_208;
        assert_close(&[dt / (dt + fitted.position_tau_s)], &[0.3], "position EMA");
        assert_close(
            &[dt / (dt + fitted.correction_tau_s)],
            &[0.3],
            "correction EMA",
        );
    }

    /// The fingerprint changes with each constant, and FITTED's is pinned:
    /// it changes with them, or with how the fingerprint reads them.
    #[test]
    fn the_fingerprint_follows_every_constant() {
        let fitted = HeadParams::FITTED.fingerprint();
        assert_eq!(fitted, HeadParams::FITTED.fingerprint());
        assert_eq!(fitted, FITTED_FINGERPRINT, "{fitted:016x}");
        let changed = |change: fn(&mut HeadParams)| {
            let mut params = HeadParams::FITTED;
            change(&mut params);
            params.fingerprint()
        };
        let prints = [
            changed(|p| p.rotation_offset[2][1] += 1e-9),
            changed(|p| p.pnp_scale = 0.97),
            changed(|p| p.pnp_point_mm[2] = -36.0),
            changed(|p| p.pnp_direction.gain = 1.0),
            changed(|p| p.pnp_direction.offset[1] = 0.0),
            changed(|p| p.eye_rays.fu = -376.0),
            changed(|p| p.eye_rays.cu = 140.0),
            changed(|p| p.eye_rays.fv = -383.0),
            changed(|p| p.eye_rays.cv = 140.0),
            changed(|p| p.eye_contours[1][15] = 467),
            changed(|p| p.eye_range_mm = 64.0),
            changed(|p| p.eye_direction.gain = 1.0),
            changed(|p| p.eye_direction.offset[0] = 0.0),
            changed(|p| p.eye_weight = 0.383),
            changed(|p| p.position_tau_s = 0.07),
            changed(|p| p.correction_tau_s = 0.07),
            changed(|p| p.rotation_filters[2].min_cutoff_hz = 1.5),
            changed(|p| p.rotation_filters[0].beta = 0.4),
            changed(|p| p.rotation_derivative_cutoff_hz = 1.2),
            changed(|p| p.reset_gap_s = 1.000_000_000_000_001),
            changed(|p| p.min_centroid_edge_px = 6.5),
            changed(|p| p.min_nose_tip_edge_px = -3.5),
        ];
        for (i, print) in prints.iter().enumerate() {
            assert_ne!(*print, fitted, "change {i}");
            assert!(!prints[..i].contains(print), "change {i}");
        }
    }

    #[test]
    fn euler_yxz_and_compose_yxz_match_the_vectors() {
        for v in &EULER {
            let what = format!("{:?}", v.degrees);
            assert_close_mat(&compose_yxz(v.degrees.map(f64::to_radians)), &v.r, &what);
            assert_close(&euler_yxz(&v.r), &v.yxz, &what);
        }
    }

    #[test]
    fn euler_yxz_reads_a_turn_about_one_axis_as_its_angle() {
        for deg in [-89.0, -45.0, -10.0, 0.0, 15.0, 60.0, 89.0] {
            let a = f64::to_radians(deg);
            let (s, c) = a.sin_cos();
            let rx = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
            assert_close(&euler_yxz(&rx), &[a, 0.0, 0.0], "x");
        }
        // y and z go all the way round.
        for deg in [-179.0, -90.0, -30.0, 0.0, 45.0, 120.0, 179.0] {
            let a = f64::to_radians(deg);
            let (s, c) = a.sin_cos();
            let ry = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
            let rz = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
            assert_close(&euler_yxz(&ry), &[0.0, a, 0.0], "y");
            assert_close(&euler_yxz(&rz), &[0.0, 0.0, a], "z");
        }
    }

    #[test]
    fn euler_yxz_clamps_a_sine_rounded_past_one() {
        // The chin straight up or down, the sine rounded just past 1.
        let mut r = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        r[1][2] = -1.0 - 2.0 * f64::EPSILON;
        assert_close(&euler_yxz(&r), &[FRAC_PI_2, 0.0, 0.0], "up");
        r[1][2] = 1.0 + 2.0 * f64::EPSILON;
        r[2][1] = -1.0;
        assert_close(&euler_yxz(&r), &[-FRAC_PI_2, 0.0, 0.0], "down");
    }

    #[test]
    fn area_a_gives_the_vectors_display_frame() {
        let frame = frame_a();
        assert_close_mat(&frame.rotation(), &A_ROTATION, "R_TS");
        assert_close(&frame.centre(), &A_CENTRE, "centre");
        assert_close(&frame.to_display([0.0; 3]), &A_TRACKER_IN_T, "tracker");
        for (s, t) in A_POINTS {
            assert_close(&frame.to_display(s), &t, "point");
        }
    }

    #[test]
    fn the_head_rotation_matches_the_vectors() {
        let r_ts = frame_a().rotation();
        for (i, v) in ROTATION.iter().enumerate() {
            let what = format!("vector {i}");
            let head = head_rotation(&v.r_cam, &vector_params().rotation_offset);
            assert_close_mat(&head, &v.head, &what);
            let r_t = matmul3(&r_ts, &head);
            assert_close_mat(&r_t, &v.r_t, &what);
            assert_close(&euler_yxz(&r_t), &v.yxz, &what);
        }
    }

    #[test]
    fn the_position_branches_match_the_vectors() {
        let params = vector_params();
        let frame = frame_a();
        for (i, v) in POSITION.iter().enumerate() {
            let what = format!("vector {i}");
            let head = head_rotation(&v.r_cam, &params.rotation_offset);
            assert_close_mat(&head, &v.head, &what);
            let pnp = pnp_position(v.t_mm, &head, &params);
            assert_close(&pnp, &v.pnp, &what);
            assert_close(
                &eye_position(v.eyes[0], v.eyes[1], &head, &params),
                &v.eye,
                &what,
            );
            // The eye correction of a fit whose eye contours are all at the
            // eyes' centroids.
            let mut landmarks = [[140.0, 140.0]; NLM];
            for (contour, eye) in params.eye_contours.iter().zip(v.eyes) {
                for &k in contour {
                    landmarks[k] = eye;
                }
            }
            let fit = fit_of(v.r_cam, v.t_mm, &landmarks, 1.0);
            let correction = eye_correction(&fit, &head, pnp, &params).unwrap();
            assert_close(&correction, &v.correction, &what);
            assert_close(&add(pnp, correction), &v.raw, &what);
            assert_close(&frame.to_display(add(pnp, correction)), &v.raw_t, &what);
            // With the eye branch off, the position is the PnP branch's.
            assert_close(&frame.to_display(pnp), &v.pnp_t, &what);
        }
    }

    #[test]
    fn the_validity_rule_matches_the_vectors() {
        // test_vectors.json "g3": the canonical face 62 cm away, facing a
        // camera tilted 24°, moved about the frame.
        let rest = [
            [1.0, 0.0, 0.0],
            [0.0, 0.913_545_457_642_600_9, -0.406_736_643_075_800_2],
            [0.0, 0.406_736_643_075_800_2, 0.913_545_457_642_600_9],
        ];
        let t_mm = [0.0, -10.0, 620.0];
        let base = projected(&rest, t_mm);
        let [cu, cv] = centroid(&base);
        let right = shifted(&base, (280.0 - 40.0) - cu, 0.0);
        let with_nose = |landmarks: &[[f64; 2]; NLM], nose: [f64; 2]| {
            let mut out = Box::new(*landmarks);
            out[NOSE_TIP] = nose;
            out
        };
        let cases = [
            (
                "centroid 6.1 px from the left edge",
                shifted(&base, 6.1 - cu, 0.0),
                true,
                [6.100_000_000_000_000_5, 6.1],
            ),
            (
                "centroid 5.9 px from the left edge",
                shifted(&base, 5.9 - cu, 0.0),
                false,
                [5.899_999_999_999_999_5, 5.9],
            ),
            (
                "centroid 6.1 px from the top edge",
                shifted(&base, 0.0, 6.1 - cv),
                true,
                [6.100_030_341_880_342, 18.8315],
            ),
            (
                "centroid 5.9 px from the top edge",
                shifted(&base, 0.0, 5.9 - cv),
                false,
                [5.900_030_341_880_343, 18.6315],
            ),
            (
                "nose tip 3.9 px beyond the right edge",
                with_nose(&right, [280.0 + 3.9, right[NOSE_TIP][1]]),
                true,
                [39.906_196_581_196_58, -3.899_999_999_999_977_3],
            ),
            (
                "nose tip 4.1 px beyond the right edge",
                with_nose(&right, [280.0 + 4.1, right[NOSE_TIP][1]]),
                false,
                [39.905_769_230_769_21, -4.100_000_000_000_023],
            ),
            (
                "nose tip 3.9 px beyond the top edge",
                with_nose(&base, [base[NOSE_TIP][0], -3.9]),
                true,
                [132.332_487_606_837_62, -3.9],
            ),
            (
                "nose tip 4.1 px beyond the top edge",
                with_nose(&base, [base[NOSE_TIP][0], -4.1]),
                false,
                [132.332_914_957_264_93, -4.1],
            ),
        ];
        let frame = frame_a();
        for (what, landmarks, valid, edges) in cases {
            let fit = fit_of(rest, t_mm, &landmarks, 12.5);
            let got = FaceEdges::of(&fit);
            assert_close(&[got.centroid_px, got.nose_tip_px], &edges, what);
            let pose = HeadPoseEstimator::new(vector_params()).step(0, Some(&frame), Some(&fit));
            assert_eq!(pose.valid, valid, "{what}");
        }
        // A face scored below 0 is none, and so is no fit at all.
        let fit = fit_of(rest, t_mm, &base, -0.5);
        let mut estimator = HeadPoseEstimator::new(vector_params());
        assert!(!estimator.step(0, Some(&frame), Some(&fit)).valid);
        assert!(!estimator.step(0, Some(&frame), None).valid);
    }

    #[test]
    fn the_validity_measures_of_a_point_that_is_not_a_number_fail() {
        let mut landmarks = [[140.0, 140.0]; NLM];
        landmarks[NOSE_TIP] = [f64::NAN, 140.0];
        let fit = fit_of(IDENTITY, [0.0, 0.0, 600.0], &landmarks, 1.0);
        assert!(FaceEdges::of(&fit).nose_tip_px.is_nan());
        assert!(FaceEdges::of(&fit).centroid_px.is_nan());
        let pose = HeadPoseEstimator::new(HeadParams::FITTED).step(0, Some(&frame_a()), Some(&fit));
        assert!(!pose.valid);
    }

    #[test]
    fn the_filters_keep_their_state_across_invalid_frames_as_the_vectors_do() {
        let params = vector_params();
        let mut last: Option<Filters> = None;
        for (i, row) in FILTER_ROWS.iter().enumerate() {
            // An invalid frame leaves the filters as they were.
            let Some(raw) = row.input else {
                continue;
            };
            let (filters, step) = Filters::next(last.as_ref(), row.t_us, &raw, &params);
            assert_eq!(
                step.map(f64::to_bits),
                row.dt_s.map(f64::to_bits),
                "row {i}"
            );
            assert_close(&filters.position_mm, &row.out_mm, &format!("row {i}"));
            assert_close(
                &filters.angles.map(|a| a.out),
                &row.out_deg,
                &format!("row {i}"),
            );
            last = Some(filters);
        }
    }

    #[test]
    fn the_filters_restart_after_more_than_the_gap_and_when_time_does_not_move_on() {
        let params = HeadParams::FITTED;
        let raw = |k: f64| Raw {
            pnp_mm: [10.0 * k, 0.0, 600.0],
            correction_mm: None,
            angles_deg: [k, -k, 2.0 * k],
        };
        let start = Filters::start(5_000_000, &raw(0.0));
        let restarted = |t_us| Filters::start(t_us, &raw(1.0));
        for (t_us, restarts) in [
            (5_000_000 + 1_000_000, false),
            (5_000_000 + 1_000_001, true),
            (5_000_000 + 1, false),
            (5_000_000, true),
            (4_999_999, true),
        ] {
            let (next, step) = Filters::next(Some(&start), t_us, &raw(1.0), &params);
            assert_eq!(next == restarted(t_us), restarts, "{t_us} µs");
            assert_eq!(step.is_none(), restarts, "{t_us} µs");
        }
        // A time so far back the step overflows restarts them too.
        let (next, step) = Filters::next(Some(&start), i64::MIN, &raw(1.0), &params);
        assert!(step.is_none() && next == restarted(i64::MIN));
    }

    #[test]
    fn the_correction_holds_while_the_eye_branch_gives_nothing() {
        let params = vector_params();
        let first = Raw {
            pnp_mm: [0.0, 0.0, 600.0],
            correction_mm: Some([1.0, 2.0, 3.0]),
            angles_deg: [0.0; 3],
        };
        let filters = Filters::start(0, &first);
        let next = filters.step(
            30_208,
            &Raw {
                correction_mm: None,
                ..first
            },
            0.030_208,
            &params,
        );
        assert_eq!(next.correction_mm, [1.0, 2.0, 3.0]);
        assert_close(&next.position_mm, &[1.0, 2.0, 603.0], "position");
    }

    /// The position's EMA and the eye correction's low-pass each move by
    /// their own time constant, `dt / (dt + tau)` of the way.
    #[test]
    fn the_position_and_the_correction_each_take_their_own_time_constant() {
        let params = HeadParams {
            position_tau_s: 0.05,
            correction_tau_s: 0.1,
            ..vector_params()
        };
        let first = Raw {
            pnp_mm: [0.0, 0.0, 600.0],
            correction_mm: Some([0.0; 3]),
            angles_deg: [0.0; 3],
        };
        let dt = 0.030_208;
        let next = Filters::start(0, &first).step(
            30_208,
            &Raw {
                pnp_mm: [10.0, 0.0, 600.0],
                correction_mm: Some([0.0, 4.0, 0.0]),
                ..first
            },
            dt,
            &params,
        );
        let (a_position, a_correction) = (dt / (dt + 0.05), dt / (dt + 0.1));
        assert_close(
            &next.correction_mm,
            &[0.0, 4.0 * a_correction, 0.0],
            "correction",
        );
        assert_close(
            &next.position_mm,
            &[10.0 * a_position, 4.0 * a_correction * a_position, 600.0],
            "position",
        );
    }

    #[test]
    fn the_one_euro_filter_takes_the_short_way_round_180_degrees() {
        let settings = OneEuro {
            min_cutoff_hz: 0.7,
            beta: 0.2,
        };
        let mut filter = AngleFilter::start(WRAP[0].0);
        assert_close(&[filter.out], &[WRAP[0].1], "start");
        for (x, want) in &WRAP[1..] {
            filter = filter.step(*x, 0.030_208, &settings, 1.0);
            assert_close(&[filter.out], &[*want], &format!("{x}°"));
        }
    }

    /// Run the estimator of `params` over [`SEQUENCE`] in area A.
    fn run_sequence(params: HeadParams, want: &[HeadPose]) {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let mut estimator = HeadPoseEstimator::new(params);
        for (i, want) in want.iter().enumerate() {
            let fit = sequence_fit(i, &landmarks);
            let pose = estimator.step(SEQUENCE[i].t_us, Some(&frame), fit.as_ref());
            assert_pose(&pose, want, &format!("frame {i}"));
        }
    }

    #[test]
    fn the_estimator_makes_the_vectors_blended_sequence() {
        run_sequence(vector_params(), &BLLP);
    }

    #[test]
    fn the_estimator_makes_the_vectors_pnp_only_sequence() {
        let params = HeadParams {
            eye_weight: 0.0,
            ..vector_params()
        };
        run_sequence(params, &PNP_ONLY);
    }

    #[test]
    fn the_sequence_frame_near_the_edge_is_the_one_the_vectors_made() {
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(9, &landmarks).unwrap();
        let edges = FaceEdges::of(&fit);
        assert_close(
            &[edges.centroid_px, edges.nose_tip_px],
            &[5.900_044_230_769_232_5, -2.2514],
            "frame 9",
        );
    }

    /// Two of [`SEQUENCE`]'s fits with an estimator that has taken the
    /// first: the pose stays valid throughout.
    fn after_one_frame(params: HeadParams) -> (HeadPoseEstimator, HeadPose) {
        let landmarks = sequence_landmarks();
        let mut estimator = HeadPoseEstimator::new(params);
        let fit = sequence_fit(0, &landmarks).unwrap();
        let first = estimator.step(0, Some(&frame_a()), Some(&fit));
        assert!(first.valid);
        (estimator, first)
    }

    /// The pose a new estimator makes of [`SEQUENCE`]'s frame `i` at
    /// `t_us`: unfiltered.
    fn unfiltered(params: HeadParams, i: usize, t_us: i64) -> HeadPose {
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(i, &landmarks).unwrap();
        HeadPoseEstimator::new(params).step(t_us, Some(&frame_a()), Some(&fit))
    }

    fn bits(pose: &HeadPose) -> Vec<u64> {
        pose.position_mm
            .iter()
            .chain(&pose.rotation_rad)
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn a_fit_that_is_not_finite_makes_an_invalid_pose_and_restarts_the_filters() {
        let params = vector_params();
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(1, &landmarks).unwrap();
        let frame = frame_a();
        let mut broken_translation = fit;
        broken_translation.translation_mm[2] = f64::NAN;
        let mut broken_rotation = fit;
        broken_rotation.rotation[0][0] = f64::INFINITY;
        for broken in [broken_translation, broken_rotation] {
            let (mut estimator, first) = after_one_frame(params);
            let pose = estimator.step(30_208, Some(&frame), Some(&broken));
            assert!(!pose.valid);
            assert_eq!(bits(&pose), bits(&first), "the last valid values");
            // Nothing that is not finite went into the filters, which start
            // again: the next pose is its frame's unfiltered one.
            let next = estimator.step(60_416, Some(&frame), Some(&fit));
            assert_eq!(next, unfiltered(params, 1, 60_416));
            assert!(
                next.position_mm
                    .iter()
                    .chain(&next.rotation_rad)
                    .all(|v| v.is_finite())
            );
        }
    }

    #[test]
    fn an_eye_branch_that_is_not_finite_leaves_the_eye_correction_out() {
        // An eye branch that ranges every head infinitely far.
        let params = HeadParams {
            eye_range_mm: f64::INFINITY,
            ..vector_params()
        };
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(1, &landmarks).unwrap();
        let head = head_rotation(&fit.rotation, &params.rotation_offset);
        let pnp = pnp_position(fit.translation_mm, &head, &params);
        assert!(eye_correction(&fit, &head, pnp, &params).is_none());
        // The pose is valid all the same: the PnP branch's position, with no
        // correction to hold yet.
        let frame = frame_a();
        let pose = HeadPoseEstimator::new(params).step(0, Some(&frame), Some(&fit));
        assert!(pose.valid);
        assert_close(&pose.position_mm, &frame.to_display(pnp), "position");
    }

    #[test]
    fn an_eye_contour_out_of_range_leaves_the_eye_correction_out() {
        let mut params = vector_params();
        params.eye_contours[1][15] = NLM;
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(0, &landmarks).unwrap();
        let head = head_rotation(&fit.rotation, &params.rotation_offset);
        let pnp = pnp_position(fit.translation_mm, &head, &params);
        assert!(eye_correction(&fit, &head, pnp, &params).is_none());
    }

    #[test]
    fn a_head_turned_side_on_keeps_a_finite_eye_range() {
        // The head's x axis along the line of sight to the eyes: none of
        // their baseline lies across it, and the range takes the floor the
        // study's reference has, sqrt(1e-6) of what it would be square on.
        let params = vector_params();
        let (left, right) = ([120.0, 130.0], [160.0, 131.0]);
        let (a, b) = (ray(left, &params.eye_rays), ray(right, &params.eye_rays));
        let sight = unit(add(a, b));
        let head = [
            [sight[0], 0.0, 0.0],
            [sight[1], 0.0, 0.0],
            [sight[2], 0.0, 0.0],
        ];
        let eye = eye_position(left, right, &head, &params);
        let want = params.eye_range_mm * 1e-3 / dot(a, b).acos().sin();
        assert_close(&[dot(eye, eye).sqrt()], &[want], "range");
    }

    #[test]
    fn without_a_display_frame_the_pose_is_invalid_and_the_filters_wait() {
        let params = vector_params();
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let (mut estimator, first) = after_one_frame(params);
        let fit = sequence_fit(1, &landmarks).unwrap();
        let pose = estimator.step(30_208, None, Some(&fit));
        assert!(!pose.valid);
        assert_eq!(bits(&pose), bits(&first));
        // The next valid frame steps from the first, as if the one without a
        // display frame had had no face.
        let fit = sequence_fit(2, &landmarks).unwrap();
        let next = estimator.step(60_416, Some(&frame), Some(&fit));
        let (mut skipped, _) = after_one_frame(params);
        assert_eq!(next, skipped.step(60_416, Some(&frame), Some(&fit)));
        assert_ne!(next, unfiltered(params, 2, 60_416));
    }

    #[test]
    fn reset_restarts_the_filters_and_keeps_the_values_invalid_poses_carry() {
        let params = vector_params();
        let (mut estimator, first) = after_one_frame(params);
        estimator.reset();
        let pose = estimator.step(30_208, Some(&frame_a()), None);
        assert!(!pose.valid);
        assert_eq!(bits(&pose), bits(&first));
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(2, &landmarks).unwrap();
        let next = estimator.step(60_416, Some(&frame_a()), Some(&fit));
        assert_eq!(next, unfiltered(params, 2, 60_416));
    }

    /// The context of a frame at `t_us` in `display`, of display generation
    /// and open 1, the legacy pose not wanted.
    fn context(t_us: i64, display: &DisplayFrame) -> FrameContext<'_> {
        FrameContext {
            t_us,
            display: Some(display),
            display_generation: 1,
            open: 1,
            legacy_wanted: false,
        }
    }

    #[test]
    fn the_step_makes_one_pose_per_frame_and_the_vectors_sequence() {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let mut poses = Poses::new(vector_params(), RestPose::default());
        for (i, want) in BLLP.iter().enumerate() {
            let fit = sequence_fit(i, &landmarks);
            let face = fit.as_ref().map(FaceSeen::of);
            let out = poses.step_with_fit(Ok(fit), &context(SEQUENCE[i].t_us, &frame));
            assert_pose(&out.head, want, &format!("frame {i}"));
            assert_eq!(out.face, face, "frame {i}");
            assert!(out.legacy.is_none() && out.error.is_none(), "frame {i}");
        }
    }

    #[test]
    fn invalid_poses_carry_the_last_valid_values_and_zeros_before_the_first() {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let mut poses = Poses::new(HeadParams::FITTED, RestPose::default());
        let before = poses.step_with_fit(Ok(None), &context(0, &frame)).head;
        assert_eq!(before, HeadPose::default());
        let valid = poses
            .step_with_fit(Ok(sequence_fit(0, &landmarks)), &context(30_208, &frame))
            .head;
        assert!(valid.valid);
        for (t_us, fit) in [
            (60_416, Ok(None)),
            (90_624, Err(anyhow!("the landmark model failed"))),
            (120_832, Ok(sequence_fit(9, &landmarks))),
        ] {
            let pose = poses.step_with_fit(fit, &context(t_us, &frame)).head;
            assert!(!pose.valid, "{t_us} µs");
            assert_eq!(bits(&pose), bits(&valid), "{t_us} µs");
        }
        let without_display = FrameContext {
            display: None,
            ..context(151_040, &frame)
        };
        let pose = poses
            .step_with_fit(Ok(sequence_fit(1, &landmarks)), &without_display)
            .head;
        assert!(!pose.valid);
        assert_eq!(bits(&pose), bits(&valid));
    }

    /// A change of the display generation or the open restarts the filters
    /// once, at the frame that brings it: the frames after it step on, and
    /// going back is a change again.
    #[test]
    fn a_new_display_generation_or_open_restarts_the_filters() {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let params = HeadParams::FITTED;
        let at = |t_us, display_generation, open| FrameContext {
            display_generation,
            open,
            ..context(t_us, &frame)
        };
        let fit = |i| Ok(sequence_fit(i, &landmarks));
        for (generation, open, restarts) in [(1, 1, false), (2, 1, true), (1, 2, true)] {
            let what = format!("generation {generation}, open {open}");
            let mut poses = Poses::new(params, RestPose::default());
            let _ = poses.step_with_fit(fit(0), &at(0, 1, 1));
            let _ = poses.step_with_fit(fit(1), &at(30_208, 1, 1));
            let out = poses.step_with_fit(fit(2), &at(60_416, generation, open));
            assert_eq!(
                out.head == unfiltered(params, 2, 60_416),
                restarts,
                "{what}"
            );
            let out = poses.step_with_fit(fit(3), &at(90_624, generation, open));
            assert_ne!(out.head, unfiltered(params, 3, 90_624), "{what}");
            let out = poses.step_with_fit(fit(4), &at(120_832, 1, 1));
            assert_eq!(
                out.head == unfiltered(params, 4, 120_832),
                restarts,
                "{what}"
            );
            let out = poses.step_with_fit(fit(1), &at(151_040, 1, 1));
            assert_ne!(out.head, unfiltered(params, 1, 151_040), "{what}");
        }
    }

    #[test]
    fn the_legacy_pose_comes_only_when_wanted_and_recalibrates_when_wanted_again() {
        let landmarks = sequence_landmarks();
        let fit = sequence_fit(0, &landmarks);
        let frame = frame_a();
        let mut poses = Poses::new(HeadParams::FITTED, RestPose::default());
        let mut t_us = 0;
        let mut step = |poses: &mut Poses, legacy_wanted| {
            t_us += 30_208;
            let wanted = FrameContext {
                legacy_wanted,
                ..context(t_us, &frame)
            };
            poses.step_with_fit(Ok(fit), &wanted)
        };
        for _ in 0..40 {
            let out = step(&mut poses, false);
            assert!(out.head.valid && out.legacy.is_none());
        }
        // Wanted: the rest pose calibrates on 30 fits, then the pose comes.
        for _ in 0..2 {
            for i in 0..30 {
                assert!(
                    step(&mut poses, true).legacy.is_none(),
                    "calibrating, fit {i}"
                );
            }
            assert!(step(&mut poses, true).legacy.is_some());
            assert!(step(&mut poses, true).legacy.is_some());
            // Not wanted for a frame: wanted again, it calibrates again.
            assert!(step(&mut poses, false).legacy.is_none());
        }
    }

    #[test]
    fn recenter_leaves_the_head_pose_alone() {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let mut poses = Poses::new(HeadParams::FITTED, RestPose::default());
        let mut plain = poses.clone();
        for i in 0..40 {
            let fit = sequence_fit(i % 5, &landmarks);
            let wanted = FrameContext {
                legacy_wanted: true,
                ..context(30_208 * i64::try_from(i).unwrap(), &frame)
            };
            if i == 35 {
                poses.recenter();
            }
            let out = poses.step_with_fit(Ok(fit), &wanted);
            let want = plain.step_with_fit(Ok(fit), &wanted);
            assert_eq!(out.head, want.head, "frame {i}");
            // The legacy pose recalibrates.
            assert_eq!(out.legacy.is_some(), (30..35).contains(&i), "frame {i}");
        }
    }

    #[test]
    fn reset_starts_the_head_pose_over_and_the_legacy_pose_recalibrates() {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let params = HeadParams::FITTED;
        let mut poses = Poses::new(params, RestPose::default());
        let wanted = |t_us| FrameContext {
            legacy_wanted: true,
            ..context(t_us, &frame)
        };
        for i in 0..31 {
            let out = poses.step_with_fit(Ok(sequence_fit(0, &landmarks)), &wanted(30_208 * i));
            assert_eq!(out.legacy.is_some(), i == 30, "frame {i}");
        }
        poses.reset();
        // Zeros until the first valid pose, which is unfiltered.
        let out = poses.step_with_fit(Ok(None), &wanted(2_000_000));
        assert_eq!(out.head, HeadPose::default());
        let out = poses.step_with_fit(Ok(sequence_fit(2, &landmarks)), &wanted(2_030_208));
        assert_eq!(out.head, unfiltered(params, 2, 2_030_208));
        // The legacy pose calibrates anew, on the 30 fits from the reset.
        assert!(out.legacy.is_none());
        for i in 1..=30 {
            let out = poses.step_with_fit(
                Ok(sequence_fit(2, &landmarks)),
                &wanted(2_030_208 + 30_208 * i),
            );
            assert_eq!(
                out.legacy.is_some(),
                i == 30,
                "fit {} after the reset",
                i + 1
            );
        }
    }

    #[test]
    fn a_tracker_error_counts_as_a_frame_without_a_face() {
        let landmarks = sequence_landmarks();
        let frame = frame_a();
        let wanted = |t_us| FrameContext {
            legacy_wanted: true,
            ..context(t_us, &frame)
        };
        let mut poses = Poses::new(HeadParams::FITTED, RestPose::default());
        for i in 0..31 {
            let _ = poses.step_with_fit(
                Ok(sequence_fit(i % 3, &landmarks)),
                &wanted(30_208 * i64::try_from(i).unwrap()),
            );
        }
        let mut no_face = poses.clone();
        let out = poses.step_with_fit(
            Err(anyhow!("the landmark model failed")),
            &wanted(31 * 30_208),
        );
        let want = no_face.step_with_fit(Ok(None), &wanted(31 * 30_208));
        assert!(out.error.is_some() && want.error.is_none());
        assert!(!out.head.valid && out.legacy.is_none() && out.face.is_none());
        assert_eq!((out.head, out.legacy), (want.head, want.legacy));
        assert_eq!(poses.rest.last_raw(), no_face.rest.last_raw());
        // And so are the frames after it.
        let next = poses.step_with_fit(Ok(sequence_fit(1, &landmarks)), &wanted(32 * 30_208));
        let want = no_face.step_with_fit(Ok(sequence_fit(1, &landmarks)), &wanted(32 * 30_208));
        assert!(next.head.valid && next.legacy.is_some());
        assert_eq!((next.head, next.legacy), (want.head, want.legacy));
    }

    #[test]
    fn a_step_on_a_frame_without_a_face_makes_one_invalid_pose() {
        let mut step = HeadStep::new(HeadParams::FITTED).unwrap();
        let frame = frame_a();
        let n = 280;
        let black = vec![0u8; n * n];
        let wanted = FrameContext {
            legacy_wanted: true,
            ..context(0, &frame)
        };
        let out = step.step(&black, n, n, &wanted);
        assert_eq!(out.head, HeadPose::default());
        assert!(out.legacy.is_none() && out.face.is_none() && out.error.is_none());
        let runs = ModelRuns {
            landmarks: 1,
            detector: 1,
        };
        assert_eq!((out.model_runs, step.model_runs()), (runs, runs));
        // A frame of another size: the tracker refuses it, and the pose is
        // invalid all the same.
        let out = step.step(&black[..100 * 100], 100, 100, &context(30_208, &frame));
        assert!(!out.head.valid && out.error.is_some());
        assert_eq!(out.model_runs, runs);
    }

    /// [`HeadParams::fingerprint`] of [`HeadParams::FITTED`], as Python
    /// packing the same numbers with `struct` gets it too. It changes with
    /// the constants.
    const FITTED_FINGERPRINT: u64 = 0x8443_11fe_3b47_e7ed;

    // The study's test vectors (`test_vectors.json`, generated by its
    // Python reference implementation), as its generator wrote them.

    /// `test_vectors.json` "`euler_yxz`".
    // reason: the reference's values as it computed them, some within a bit
    // of π/4 and 1/√2 (a 45° roll), not those constants.
    #[allow(clippy::approx_constant)]
    const EULER: [Euler; 18] = [
        Euler {
            degrees: [0.0, 0.0, 0.0],
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            yxz: [-0.0, 0.0, 0.0],
        },
        Euler {
            degrees: [20.0, 0.0, 0.0],
            r: [
                [1.0, 0.0, 0.0],
                [0.0, 0.9396926207859084, -0.3420201433256687],
                [0.0, 0.3420201433256687, 0.9396926207859084],
            ],
            yxz: [0.3490658503988659, 0.0, 0.0],
        },
        Euler {
            degrees: [0.0, 30.0, 0.0],
            r: [
                [0.8660254037844387, 0.0, 0.49999999999999994],
                [0.0, 1.0, 0.0],
                [-0.49999999999999994, 0.0, 0.8660254037844387],
            ],
            yxz: [-0.0, 0.5235987755982988, 0.0],
        },
        Euler {
            degrees: [0.0, 0.0, 25.0],
            r: [
                [0.9063077870366499, -0.42261826174069944, 0.0],
                [0.42261826174069944, 0.9063077870366499, 0.0],
                [0.0, 0.0, 1.0],
            ],
            yxz: [-0.0, 0.0, 0.4363323129985824],
        },
        Euler {
            degrees: [15.0, 25.0, 0.0],
            r: [
                [0.9063077870366499, 0.109381654946615, 0.40821789367673483],
                [0.0, 0.9659258262890683, -0.25881904510252074],
                [-0.42261826174069944, 0.23456971600980447, 0.875426098065593],
            ],
            yxz: [0.2617993877991494, 0.4363323129985824, 0.0],
        },
        Euler {
            degrees: [15.0, 0.0, 20.0],
            r: [
                [0.9396926207859084, -0.3420201433256687, 0.0],
                [
                    0.33036608954935215,
                    0.9076733711903687,
                    -0.25881904510252074,
                ],
                [0.08852132690137686, 0.24321034680169396, 0.9659258262890683],
            ],
            yxz: [0.2617993877991494, 0.0, 0.3490658503988659],
        },
        Euler {
            degrees: [0.0, 25.0, 20.0],
            r: [
                [
                    0.8516507396391465,
                    -0.30997551921944466,
                    0.42261826174069944,
                ],
                [0.3420201433256687, 0.9396926207859084, 0.0],
                [-0.39713126196710286, 0.144543958452599, 0.9063077870366499],
            ],
            yxz: [-0.0, 0.4363323129985824, 0.3490658503988659],
        },
        Euler {
            degrees: [-20.0, 30.0, -15.0],
            r: [
                [
                    0.8807769671884964,
                    0.058960823267337356,
                    0.46984631039295416,
                ],
                [-0.24321034680169396, 0.9076733711903687, 0.3420201433256687],
                [
                    -0.4063011952712349,
                    -0.41551494864992405,
                    0.8137976813493738,
                ],
            ],
            yxz: [-0.3490658503988659, 0.5235987755982988, -0.2617993877991494],
        },
        Euler {
            degrees: [25.0, -35.0, 10.0],
            r: [
                [0.7646142926969142, -0.3809654766663491, -0.5198367907256845],
                [
                    0.15737869562426265,
                    0.8925389352890299,
                    -0.42261826174069944,
                ],
                [0.6249775432503181, 0.2413287272197532, 0.7424038765061041],
            ],
            yxz: [0.4363323129985824, -0.6108652381980153, 0.17453292519943298],
        },
        Euler {
            degrees: [-10.0, -40.0, 30.0],
            r: [
                [0.7192233966934132, -0.28635742117269863, -0.633022221559489],
                [0.49240387650610395, 0.8528685319524433, 0.17364817766693033],
                [0.4901592884466749, -0.4365944279816291, 0.7544065067354889],
            ],
            yxz: [
                -0.17453292519943295,
                -0.6981317007977318,
                0.5235987755982988,
            ],
        },
        Euler {
            degrees: [-35.0, 0.0, 0.0],
            r: [
                [1.0, 0.0, 0.0],
                [0.0, 0.8191520442889918, 0.573576436351046],
                [0.0, -0.573576436351046, 0.8191520442889918],
            ],
            yxz: [-0.6108652381980153, 0.0, 0.0],
        },
        Euler {
            degrees: [0.0, -60.0, 0.0],
            r: [
                [0.5000000000000001, 0.0, -0.8660254037844386],
                [0.0, 1.0, 0.0],
                [0.8660254037844386, 0.0, 0.5000000000000001],
            ],
            yxz: [-0.0, -1.0471975511965976, 0.0],
        },
        Euler {
            degrees: [0.0, 0.0, -45.0],
            r: [
                [0.7071067811865476, 0.7071067811865475, 0.0],
                [-0.7071067811865475, 0.7071067811865476, 0.0],
                [0.0, 0.0, 1.0],
            ],
            yxz: [-0.0, 0.0, -0.7853981633974483],
        },
        Euler {
            degrees: [41.0, 59.0, -30.0],
            r: [
                [0.16485988329781867, 0.7445304558184107, 0.6469123736935778],
                [
                    -0.37735479011138595,
                    0.6535976689524103,
                    -0.6560590289905073,
                ],
                [
                    -0.9112763473606794,
                    -0.13595766803974324,
                    0.38870416931411156,
                ],
            ],
            yxz: [0.7155849933176751, 1.0297442586766545, -0.5235987755982988],
        },
        Euler {
            degrees: [-17.97, 34.167, 1.72],
            r: [
                [0.8218307641467915, -0.19802299210140686, 0.5342107165726899],
                [0.02855096082524736, 0.9507896095398874, 0.3085189800108633],
                [
                    -0.5690158501639967,
                    -0.23829815985482228,
                    0.7870418980409807,
                ],
            ],
            yxz: [-0.313635666583381, 0.5963266455289026, 0.030019663134302467],
        },
        Euler {
            degrees: [-49.203, -59.895, 4.728],
            r: [
                [0.553860913760473, 0.6113397748325274, -0.5652448742944323],
                [0.05385528407793232, 0.6511576585728234, 0.7570292676368884],
                [0.8308656308846208, -0.4497303451955105, 0.3277268985375291],
            ],
            yxz: [-0.85875435185877, -1.0453649554820037, 0.08251916703429191],
        },
        Euler {
            degrees: [-16.431, 18.737, 22.29],
            r: [
                [0.8417757447087016, -0.4432653876682877, 0.3081061370946599],
                [0.3638046865690525, 0.8874886553769219, 0.28286045430150986],
                [
                    -0.39882295025550407,
                    -0.12601461293257085,
                    0.9083284492280036,
                ],
            ],
            yxz: [
                -0.28677504939518833,
                0.3270223419461775,
                0.38903389026953605,
            ],
        },
        Euler {
            degrees: [-25.466, -21.65, 1.816],
            r: [
                [0.9340151335638206, 0.1290993416326238, -0.3330902134013871],
                [0.028610903582998382, 0.902387144025474, 0.42997541615040363],
                [0.35608586951734394, -0.411133557724946, 0.8391496000371481],
            ],
            yxz: [
                -0.44446554731287596,
                -0.37786378305677226,
                0.03169517921621703,
            ],
        },
    ];
    /// `test_vectors.json` "`display_frame`", area A: `R_TS`, the centre, the
    /// tracker in T, and points in S and in T.
    const A_ROTATION: [[f64; 3]; 3] = [
        [1.0, 0.0, 0.0],
        [0.0, 0.9396926187267778, 0.34202014898308336],
        [0.0, -0.34202014898308336, 0.9396926187267778],
    ];
    const A_CENTRE: [f64; 3] = [1.00225830078125, 167.33939933776855, 54.069538712501526];
    const A_TRACKER_IN_T: [f64; 3] = [-1.00225830078125, -175.74047006577013, 6.424699866143832];
    const A_POINTS: [([f64; 3], [f64; 3]); 3] = [
        (
            [0.0, 150.0, 600.0],
            [-1.00225830078125, 170.42551213309656, 518.937248754748],
        ),
        (
            [-40.0, 60.0, 550.0],
            [-41.00225830078125, 68.75216899853238, 502.7344312268866],
        ),
        (
            [55.5, 120.25, 700.0],
            [54.49774169921875, 176.67167162428325, 623.0816100596725],
        ),
    ];
    /// `test_vectors.json` "rotation".
    const ROTATION: [Rotation; 5] = [
        Rotation {
            r_cam: [
                [1.0, 0.0, 0.0],
                [0.0, 0.9135454576426009, -0.4067366430758002],
                [0.0, 0.4067366430758002, 0.9135454576426009],
            ],
            head: [
                [
                    0.9998600753746771,
                    -0.007035297618850274,
                    -0.015176767085183904,
                ],
                [0.011915370916680955, 0.9363142397916311, 0.3509610637938872],
                [
                    0.011741107599440255,
                    -0.3510927925076694,
                    0.9362670545530823,
                ],
            ],
            r_t: [
                [
                    0.9998600753746771,
                    -0.007035297618850274,
                    -0.015176767085183904,
                ],
                [0.015212481470183782, 0.75976677074061, 0.6500177185938114],
                [
                    0.006957735210758966,
                    -0.6501576413961452,
                    0.7597674849945462,
                ],
            ],
            yxz: [
                -0.7076077529155185,
                -0.01997288544899144,
                0.020019892464277225,
            ],
        },
        Rotation {
            r_cam: [
                [0.9361168066628592, -0.08189960831908934, 0.3420201433256687],
                [0.16699414721038502, 0.959392003563287, -0.2273322201015467],
                [-0.3095129707795902, 0.2699248740964818, 0.9117797339616576],
            ],
            head: [
                [
                    0.930159346528325,
                    -0.10888880994356884,
                    -0.35063772919506536,
                ],
                [0.17637249288185633, 0.9701207932187741, 0.16660849413885012],
                [
                    0.32201915132587705,
                    -0.21681529843082825,
                    0.9215719139306017,
                ],
            ],
            r_t: [
                [
                    0.930159346528325,
                    -0.10888880994356884,
                    -0.35063772919506536,
                ],
                [0.27587296781940407, 0.8374601479899252, 0.47175693546063074],
                [0.24227607329761763, -0.5355405937907649, 0.8090108631592791],
            ],
            yxz: [
                -0.49128232166015834,
                -0.40897688029741813,
                0.31822103171258437,
            ],
        },
        Rotation {
            r_cam: [
                [0.847100670886274, 0.18005680599195542, -0.49999999999999994],
                [
                    -0.4354887420128307,
                    0.7744280054504655,
                    -0.45892354478071395,
                ],
                [0.3045816950575112, 0.6064988136756652, 0.7344311949024933],
            ],
            head: [
                [0.8558684594224347, 0.20381436547519133, 0.47534080888538516],
                [-0.4235497225006305, 0.8036515782865168, 0.41803082815413106],
                [-0.2968077032954264, -0.5591098685799915, 0.7741455561591493],
            ],
            r_t: [
                [0.8558684594224347, 0.20381436547519133, 0.47534080888538516],
                [-0.4995207627980461, 0.5639586155943243, 0.6575938620688232],
                [
                    -0.13404546877655044,
                    -0.8002564490979613,
                    0.5844838988180912,
                ],
            ],
            yxz: [-0.7176204692920763, 0.6827777085217357, -0.7248806115609879],
        },
        Rotation {
            r_cam: [
                [
                    0.8619499266302462,
                    -0.28006450832867547,
                    0.42261826174069944,
                ],
                [0.4930964779493586, 0.6569228633625117, -0.5703579709494173],
                [-0.11789057390669819, 0.7000115875942454, 0.704333436532537],
            ],
            head: [
                [
                    0.8535370068615025,
                    -0.31101234099932223,
                    -0.4180261973417783,
                ],
                [0.5059229524452727, 0.6865253909737988, 0.5222306518554133],
                [0.12456542099220058, -0.6572322354346241, 0.7433231091514929],
            ],
            r_t: [
                [
                    0.8535370068615025,
                    -0.31101234099932223,
                    -0.4180261973417783,
                ],
                [0.5180159479031744, 0.420336175386759, 0.7449677693559679],
                [
                    -0.05598263691432907,
                    -0.8524017969287443,
                    0.5198818336475757,
                ],
            ],
            yxz: [-0.8404865252661324, -0.6772233478630346, 0.889121723187907],
        },
        Rotation {
            r_cam: [
                [0.696364240320019, -0.12278780396897285, -0.7071067811865475],
                [0.1246492655874294, 0.9909740550457953, -0.04932527561613236],
                [0.7067810165758821, -0.05379198288379073, 0.7053843046066397],
            ],
            head: [
                [0.7065279933277063, -0.08496904214821911, 0.7025656243517312],
                [0.131454488167464, 0.9912457284003366, -0.01231354817732707],
                [-0.6953689036654958, 0.10105527103768247, 0.7115124173270247],
            ],
            r_t: [
                [0.7065279933277063, -0.08496904214821911, 0.7025656243517312],
                [
                    -0.11430336380040373,
                    0.9660292331780788,
                    0.23178063264493198,
                ],
                [
                    -0.6983931096941292,
                    -0.24406511942878398,
                    0.6728144482767716,
                ],
            ],
            yxz: [
                -0.2339077650114157,
                0.8070260302489793,
                -0.11777528996905463,
            ],
        },
    ];
    /// `test_vectors.json` "`position_model`".
    const POSITION: [Position; 6] = [
        Position {
            r_cam: [
                [
                    0.9841256823661367,
                    -0.04987287035978048,
                    0.17032127908591518,
                ],
                [0.13406472633902528, 0.8377611228028367, -0.5293231057414484],
                [-0.11628968337315196, 0.5437545383003316, 0.8311484293546882],
            ],
            t_mm: [-27.524850807863743, -16.81878251724813, 573.497310874498],
            eyes: [
                [95.21220624999998, 123.058925],
                [139.07861250000002, 128.88034375],
            ],
            head: [
                [
                    0.9810309507619948,
                    -0.06694041798096077,
                    -0.1819265073801768,
                ],
                [0.1474072911690436, 0.8670922836246877, 0.4758382731440673],
                [0.12589425784053856, -0.4936293671562275, 0.8605118730870143],
            ],
            pnp: [30.09360742351783, 18.837000066966095, 511.89871353492236],
            eye: [32.48510465568619, 19.704444267529652, 539.2409338021613],
            correction: [0.9169442903766112, 0.33259415742909876, 10.48351318288265],
            raw: [31.010551713894444, 19.169594224395194, 522.382226717805],
            raw_t: [30.008293413113194, 20.93830313901999, 490.7470349943505],
            pnp_t: [29.09134912273658, 17.040194123576573, 481.0095089413456],
        },
        Position {
            r_cam: [
                [
                    0.9241995839870967,
                    -0.23835491907548037,
                    -0.2983991647283861,
                ],
                [0.08515605803093886, 0.8902761091566922, -0.4473889753284638],
                [0.37229501035967255, 0.38806620829098554, 0.8430901750370174],
            ],
            t_mm: [19.59008322953075, -13.900598287519477, 577.9997776123156],
            eyes: [
                [145.3493125, 122.88939374999998],
                [185.11116875000002, 127.313875],
            ],
            head: [
                [0.9272607971314948, -0.22648822925102877, 0.2981450923859003],
                [0.09755040146559343, 0.9149313890040406, 0.3916432976434522],
                [
                    -0.36148490048266896,
                    -0.3340713029068341,
                    0.8704740267791824,
                ],
            ],
            pnp: [-38.324773457800354, 20.481048921385735, 521.7749581101051],
            eye: [-38.58060174173732, 21.4604726920124, 551.6556273837913],
            correction: [
                -0.09808929783295313,
                0.37552919662840983,
                11.456801502670313,
            ],
            raw: [-38.42286275563331, 20.856578118014145, 533.2317596127754],
            raw_t: [-39.42512105641456, 26.234308308899305, 500.3652784897599],
            pnp_t: [-39.327031758581604, 21.962969337898326, 489.7278452352611],
        },
        Position {
            r_cam: [
                [0.989160177940258, 0.04254517294971721, -0.1405419888712111],
                [
                    -0.09935156803232392,
                    0.8986800982480979,
                    -0.4272041045475893,
                ],
                [0.10812681585395684, 0.4365363550398383, 0.8931654955392179],
            ],
            t_mm: [55.1954939229997, 2.384865646955536, 602.2696806150905],
            eyes: [
                [158.91479375, 138.14265625],
                [200.502525, 133.96707500000002],
            ],
            head: [
                [0.9914702916571256, 0.043953932955392426, 0.1226976468359757],
                [-0.08719438959438307, 0.9234050560759834, 0.3737917078221133],
                [
                    -0.09687001179207062,
                    -0.3813019199939363,
                    0.9193612166196382,
                ],
            ],
            pnp: [-59.52495738541075, 5.718065078991532, 541.4185056962222],
            eye: [-62.31858534198135, 5.529162636710397, 571.1783093209344],
            correction: [
                -1.0711286510213907,
                -0.07242869176596638,
                11.410459376391204,
            ],
            raw: [-60.59608603643214, 5.645636387225565, 552.8289650726134],
            raw_t: [-61.59834433721339, 18.643337771620086, 523.9830763649791],
            pnp_t: [-60.527215686192, 14.808791462777851, 513.235979840754],
        },
        Position {
            r_cam: [
                [
                    0.9963216154262884,
                    0.07410339906304889,
                    -0.04303399681244427,
                ],
                [
                    -0.08460514794658054,
                    0.7708878413916384,
                    -0.6313270982109649,
                ],
                [
                    -0.013609098988843974,
                    0.6326457320189891,
                    0.7743217484888754,
                ],
            ],
            t_mm: [-25.093009705258464, -22.040093297034492, 631.5743791297155],
            eyes: [
                [103.52828750000002, 129.20931875],
                [143.8337125, 125.766025],
            ],
            head: [
                [0.9973050995249243, 0.06954430561425912, 0.02336938206735496],
                [-0.0700521086938693, 0.8080097633472297, 0.584989678885957],
                [0.021800012135817753, -0.5850502644151042, 0.81070398270805],
            ],
            pnp: [23.953855820380184, 18.50845042438859, 567.9436708460427],
            eye: [25.500859362398664, 19.477484939002817, 598.586720657412],
            correction: [0.593149783309628, 0.3715457636334955, 11.749112307683866],
            raw: [24.54700560368981, 18.879996188022087, 579.6927831537265],
            raw_t: [23.54474730290856, 40.267535052359726, 544.7003902158559],
            pnp_t: [22.951597519598934, 35.89996309886112, 533.7869122411654],
        },
        Position {
            r_cam: [
                [
                    0.9867520392461167,
                    -0.15867075462189476,
                    -0.033823138106163525,
                ],
                [0.12349506274796629, 0.8698203530211479, -0.4776625618017822],
                [0.10521113306350242, 0.46715751636667907, 0.8778920619196928],
            ],
            t_mm: [-35.22009439555708, -3.1235538232141344, 591.7268269139147],
            eyes: [[97.8724125, 131.227075], [140.92034375, 136.65900625]],
            head: [
                [0.9861712477786594, -0.1632895815604838, 0.02833341858093848],
                [0.1362304972124857, 0.8960626465264172, 0.42250797049205],
                [
                    -0.09437966774639266,
                    -0.4128053367555638,
                    0.9059162390984088,
                ],
            ],
            pnp: [27.235197920069204, 8.019166963558101, 530.7284142048542],
            eye: [30.31607395521518, 8.56977923199374, 559.6800693559916],
            correction: [1.1812648795013763, 0.2111149320861223, 11.100600297880304],
            raw: [28.41646279957058, 8.230281895644223, 541.8290145027345],
            raw_t: [27.41420449878933, 17.309905345189875, 512.7625031662467],
            pnp_t: [26.232939619287954, 13.314893234122845, 502.4035565634169],
        },
        Position {
            r_cam: [
                [
                    0.9104030211376942,
                    -0.10852861482434063,
                    -0.3992341153604849,
                ],
                [
                    -0.08358334233946622,
                    0.8968342915574667,
                    -0.4343985248248206,
                ],
                [0.40519151519586716, 0.42884705111604365, 0.8074094641273105],
            ],
            t_mm: [-11.839186504535812, -26.169900675208407, 598.9327045298605],
            eyes: [
                [123.94082499999999, 118.29808125],
                [162.527675, 115.80136250000001],
            ],
            head: [
                [0.9158296113684052, -0.0907431295734881, 0.39118001914208733],
                [
                    -0.07132761280408538,
                    0.9218840389851691,
                    0.38084404986301257,
                ],
                [
                    -0.39518159698101457,
                    -0.37669019512007423,
                    0.8378162103396751,
                ],
            ],
            pnp: [-6.969199337593287, 33.493423613503715, 542.2871453886302],
            eye: [-5.930088433487557, 34.575983658029784, 573.1531741154055],
            correction: [0.3984143480374931, 0.4150735524448001, 11.83460654978696],
            raw: [-6.570784989555794, 33.90849716594852, 554.1217519384171],
            raw_t: [-7.573043290337044, 45.64389858593414, 515.5314307861348],
            pnp_t: [-7.971457638374537, 41.20618313715876, 504.55250188401055],
        },
    ];
    /// `test_vectors.json` "filters": frames 4-6 and 13 invalid, frame 11
    /// 1.51 s after frame 10.
    const FILTER_ROWS: [FilterRow; 14] = [
        FilterRow {
            t_us: 0,
            input: Some(Raw {
                pnp_mm: [10.0, 100.0, 600.0],
                correction_mm: Some([0.0, -0.3, 1.0]),
                angles_deg: [5.3, -10.0, 2.0],
            }),
            dt_s: None,
            out_mm: [10.0, 99.7, 601.0],
            out_deg: [5.3, -10.0, 2.0],
        },
        FilterRow {
            t_us: 30211,
            input: Some(Raw {
                pnp_mm: [12.0, 98.5, 603.0],
                correction_mm: Some([0.42073549240394825, -0.1620906917604419, 1.1]),
                angles_deg: [5.9, -6.0, 1.3],
            }),
            dt_s: Some(0.030211),
            out_mm: [10.63791316866974, 99.2623822812708, 601.909063815674],
            out_deg: [5.480536455059905, -8.067398559949027, 1.8166431818578133],
        },
        FilterRow {
            t_us: 60416,
            input: Some(Raw {
                pnp_mm: [14.0, 97.0, 606.0],
                correction_mm: Some([0.45464871341284085, 0.12484405096414272, 1.2]),
                angles_deg: [7.7, -2.0, 0.6000000000000001],
            }),
            dt_s: Some(0.030205),
            out_mm: [11.713888918933511, 98.54064023163535, 603.4605361792968],
            out_deg: [6.649891689715048, -4.325999293439395, 1.460550714272614],
        },
        FilterRow {
            t_us: 90624,
            input: Some(Raw {
                pnp_mm: [16.0, 95.5, 609.0],
                correction_mm: Some([0.0705600040299336, 0.2969977489801336, 1.3]),
                angles_deg: [8.299999999999999, 2.0, -0.09999999999999964],
            }),
            dt_s: Some(0.030208),
            out_mm: [13.053269799205808, 97.62502367853772, 605.4663848875222],
            out_deg: [7.552277370023859, 0.007622968152418252, 0.9663872237850057],
        },
        FilterRow {
            t_us: 120835,
            input: None,
            dt_s: None,
            out_mm: [0.0; 3],
            out_deg: [0.0; 3],
        },
        FilterRow {
            t_us: 151040,
            input: None,
            dt_s: None,
            out_mm: [0.0; 3],
            out_deg: [0.0; 3],
        },
        FilterRow {
            t_us: 181248,
            input: None,
            dt_s: None,
            out_mm: [0.0; 3],
            out_deg: [0.0; 3],
        },
        FilterRow {
            t_us: 211459,
            input: Some(Raw {
                pnp_mm: [24.0, 89.5, 606.0],
                correction_mm: Some([
                    0.32849329935939453,
                    -0.22617067630299137,
                    1.7000000000000002,
                ]),
                angles_deg: [13.1, 18.0, -2.8999999999999995],
            }),
            dt_s: Some(0.120835),
            out_mm: [20.13962557052025, 92.40050742965283, 606.7483568678504],
            out_deg: [12.502061838495933, 16.788577741776294, -1.7299326987025836],
        },
        FilterRow {
            t_us: 241664,
            input: Some(Raw {
                pnp_mm: [26.0, 88.0, 609.0],
                correction_mm: Some([0.4946791233116909, 0.04365001014258406, 1.8]),
                angles_deg: [14.9, 22.0, -3.5999999999999996],
            }),
            dt_s: Some(0.030205),
            out_mm: [21.999506726621703, 91.0534953686396, 607.8999604521879],
            out_deg: [14.213123843358431, 20.898985346550546, -2.430999047523045],
        },
        FilterRow {
            t_us: 271872,
            input: Some(Raw {
                pnp_mm: [28.0, 86.5, 612.0],
                correction_mm: Some([0.2060592426208783, 0.27333907856540307, 1.9]),
                angles_deg: [15.499999999999998, 26.0, -4.3],
            }),
            dt_s: Some(0.030208),
            out_mm: [23.88952873520436, 89.6931798370089, 609.6343058134138],
            out_deg: [15.117539937067647, 24.971798270074657, -3.1460640674666536],
        },
        FilterRow {
            t_us: 302083,
            input: Some(Raw {
                pnp_mm: [30.0, 85.0, 600.0],
                correction_mm: Some([-0.2720105554446849, 0.2517214587229357, 2.0]),
                angles_deg: [17.3, 30.0, -5.0],
            }),
            dt_s: Some(0.030211),
            out_mm: [25.76122751098266, 88.31179940650033, 607.2768856454298],
            out_deg: [16.71640487461974, 29.02414485966119, -3.8671050753614793],
        },
        // restarted after 1.510397 s
        FilterRow {
            t_us: 1812480,
            input: Some(Raw {
                pnp_mm: [32.0, 83.5, 603.0],
                correction_mm: Some([-0.49999510327535174, -0.0013277093964152355, 2.1]),
                angles_deg: [17.9, 34.0, -5.699999999999999],
            }),
            dt_s: None,
            out_mm: [31.500004896724647, 83.49867229060358, 605.1],
            out_deg: [17.9, 34.0, -5.699999999999999],
        },
        FilterRow {
            t_us: 1842691,
            input: Some(Raw {
                pnp_mm: [34.0, 82.0, 606.0],
                correction_mm: Some([-0.26828645900021747, -0.2531561876197476, 2.2]),
                angles_deg: [19.7, 38.0, -6.399999999999999],
            }),
            dt_s: Some(0.030211),
            out_mm: [32.120903283712444, 83.02597329418546, 606.009063815674],
            out_deg: [18.758357112191362, 35.93260144005097, -5.883356818142204],
        },
        FilterRow {
            t_us: 1872896,
            input: None,
            dt_s: None,
            out_mm: [0.0; 3],
            out_deg: [0.0; 3],
        },
    ];
    /// `test_vectors.json` "filters", "`one_euro_wrap_yaw`": an angle crossing
    /// ±180° and the one-euro filter's output (0.7 Hz, 0.2 Hz per °/s,
    /// 30.208 ms frames).
    const WRAP: [(f64, f64); 5] = [
        (178.0, 178.0),
        (179.5, 178.45365240636954),
        (-179.0, 179.48996691601826),
        (-177.5, -179.0853146021321),
        (-178.0, -178.6474480437757),
    ];
    /// `test_vectors.json` "`estimator_sequences`": 30.208 ms frames; 5-7 without
    /// a face, 9 too near the left edge, 11 1.51 s after 10, 13 scored below 0.
    const SEQUENCE: [Frame; 15] = [
        Frame {
            t_us: 0,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.9135454576426009, -0.4067366430758002],
                    [0.0, 0.4067366430758002, 0.9135454576426009],
                ],
                t_mm: [5.0, -20.0, 620.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 30208,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [
                        0.9993527732787075,
                        -0.008721219528731424,
                        0.03489949670250097,
                    ],
                    [0.02265753010506906, 0.9061445685044691, -0.4223608141144103],
                    [
                        -0.02794048800028539,
                        0.42287818730667726,
                        0.9057556888204041,
                    ],
                ],
                t_mm: [6.0, -19.0, 621.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 60416,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [0.9974121164231596, -0.01740989325235717, 0.0697564737441253],
                    [0.04626068702645717, 0.8981234745424449, -0.437303296707956],
                    [
                        -0.055036522856313275,
                        0.43939858908825363,
                        0.896604629175613,
                    ],
                ],
                t_mm: [8.0, -18.0, 622.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 90624,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [
                        0.9941810975534692,
                        -0.026033548246103322,
                        0.10452846326765347,
                    ],
                    [0.07076249938249594, 0.8894589732753043, -0.4515034922801636],
                    [
                        -0.08121954166653264,
                        0.45627293282174797,
                        0.8861254972213128,
                    ],
                ],
                t_mm: [10.0, -18.0, 624.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 120832,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [
                        0.9896648241902408,
                        -0.03455985719963844,
                        0.13917310096006544,
                    ],
                    [0.09424045536969887, 0.8882586829166373, -0.4495722954041014],
                    [-0.10808456102613251, 0.4580416231015926, 0.8823353099441544],
                ],
                t_mm: [12.0, -17.0, 625.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 151040,
            face: None,
        },
        Frame {
            t_us: 181248,
            face: None,
        },
        Frame {
            t_us: 211456,
            face: None,
        },
        Frame {
            t_us: 241664,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [0.9374035767904598, -0.06554964362940052, 0.3420201433256687],
                    [
                        0.23100437806403454,
                        0.8519867498531749,
                        -0.46984631039295416,
                    ],
                    [-0.2605983720895067, 0.5194437623984739, 0.8137976813493738],
                ],
                t_mm: [30.0, -10.0, 630.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 271872,
            face: Some(Face {
                score: 9.0,
                r_cam: [
                    [0.9249252812971602, -0.064677076207065, 0.374606593415912],
                    [
                        0.24725791363184088,
                        0.8508501919281077,
                        -0.46359192728339366,
                    ],
                    [-0.28875032149557517, 0.5214123384704352, 0.8029647720336144],
                ],
                t_mm: [32.0, -9.0, 631.0],
                left_edge_px: Some(5.9),
            }),
        },
        Frame {
            t_us: 302080,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [0.9100691413693562, -0.07962073289459017, 0.4067366430758002],
                    [
                        0.28339475726828833,
                        0.8356477119746079,
                        -0.47051069384706956,
                    ],
                    [-0.3024261388836793, 0.5434642953910875, 0.7830612939961841],
                ],
                t_mm: [35.0, -8.0, 632.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 1812480,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [0.994829447880333, 0.05213680212878224, -0.08715574274765817],
                    [
                        -0.08519344800947809,
                        0.8955627032919189,
                        -0.4367030122276772,
                    ],
                    [0.05528513404495024, 0.4418701147806385, 0.8953738635996571],
                ],
                t_mm: [-20.0, -25.0, 600.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 1842688,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [
                        0.9931589376748557,
                        0.05204925439864352,
                        -0.10452846326765347,
                    ],
                    [-0.09154758095777378, 0.902753749484343, -0.4203031146836054],
                    [0.07248699840200136, 0.42699712283398983, 0.9013429381507145],
                ],
                t_mm: [-22.0, -25.0, 601.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 1872896,
            face: Some(Face {
                score: -1.25,
                r_cam: [
                    [
                        0.9939160595006973,
                        0.03470831360797007,
                        -0.10452846326765347,
                    ],
                    [
                        -0.07437193361380127,
                        0.9115051789176384,
                        -0.40450849718747367,
                    ],
                    [0.08123842783529675, 0.40982147549001735, 0.908540960039796],
                ],
                t_mm: [-23.0, -24.0, 602.0],
                left_edge_px: None,
            }),
        },
        Frame {
            t_us: 1903104,
            face: Some(Face {
                score: 15.0,
                r_cam: [
                    [
                        0.9919415193434417,
                        0.034639361146286345,
                        -0.12186934340514748,
                    ],
                    [
                        -0.08142080838747401,
                        0.9112590267864907,
                        -0.40370488981639546,
                    ],
                    [0.09707045979161145, 0.4103743422285498, 0.9067360283325738],
                ],
                t_mm: [-24.0, -24.0, 603.0],
                left_edge_px: None,
            }),
        },
    ];
    /// The poses of [`SEQUENCE`] with [`vector_params`] (the eye branch at
    /// weight 0.383), area A.
    const BLLP: [HeadPose; 15] = [
        HeadPose {
            valid: true,
            position_mm: [-7.305263801573662, 46.39083599160854, 532.0782260946238],
            rotation_rad: [
                -0.7076077529155185,
                -0.01997288544899144,
                0.020019892464277225,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-7.3088952504325055, 45.947597546419935, 532.456511774309],
            rotation_rad: [
                -0.7134719150210815,
                -0.03834548769400342,
                0.034332599565995416,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-7.611783230751511, 45.19919969762165, 533.1099740932896],
            rotation_rad: [
                -0.7251706021913319,
                -0.07808737559128301,
                0.0644320820465975,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-8.124550439968292, 44.620693414473294, 534.1424778277404],
            rotation_rad: [
                -0.7393257057339666,
                -0.12885836423833694,
                0.10602658736037733,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-8.785292524115594, 44.03048258759454, 535.2428019487741],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-8.785292524115594, 44.03048258759454, 535.2428019487741],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-8.785292524115594, 44.03048258759454, 535.2428019487741],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-8.785292524115594, 44.03048258759454, 535.2428019487741],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-17.068197213284343, 38.88793402163968, 541.8191760221097],
            rotation_rad: [
                -0.7428975947219212,
                -0.47561470604682005,
                0.4133086082306029,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-17.068197213284343, 38.88793402163968, 541.8191760221097],
            rotation_rad: [
                -0.7428975947219212,
                -0.47561470604682005,
                0.4133086082306029,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-20.722146846901538, 36.73086424730629, 544.9832072322762],
            rotation_rad: [-0.7386866048373887, -0.575832488263852, 0.5138415001127281],
        },
        HeadPose {
            valid: true,
            position_mm: [16.24131974721774, 42.99708606023846, 512.1244420697997],
            rotation_rad: [
                -0.7411842882790189,
                0.09333567696131931,
                -0.11353690695767861,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [16.65291008070488, 43.35198608986978, 512.3964733636012],
            rotation_rad: [
                -0.7340575179744467,
                0.09934521483776439,
                -0.11734430740825331,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [16.65291008070488, 43.35198608986978, 512.3964733636012],
            rotation_rad: [
                -0.7340575179744467,
                0.09934521483776439,
                -0.11734430740825331,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [17.50370524210054, 44.024048510659185, 513.7336584822908],
            rotation_rad: [
                -0.7144129484575245,
                0.1190394762258413,
                -0.12010518856689958,
            ],
        },
    ];
    /// The poses of [`SEQUENCE`] with the eye branch off.
    const PNP_ONLY: [HeadPose; 15] = [
        HeadPose {
            valid: true,
            position_mm: [-7.410097005963494, 42.17926122969773, 521.6491393809121],
            rotation_rad: [
                -0.7076077529155185,
                -0.01997288544899144,
                0.020019892464277225,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-7.413498536066379, 41.73653549001725, 522.0294410727525],
            rotation_rad: [
                -0.7134719150210815,
                -0.03834548769400342,
                0.034332599565995416,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-7.71393437515307, 40.98879417137359, 522.6867913109792],
            rotation_rad: [
                -0.7251706021913319,
                -0.07808737559128301,
                0.0644320820465975,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-8.221107599842918, 40.408663482585574, 523.7234748429979],
            rotation_rad: [
                -0.7393257057339666,
                -0.12885836423833694,
                0.10602658736037733,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-8.872912829762626, 39.81592034453105, 524.8300106098162],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-8.872912829762626, 39.81592034453105, 524.8300106098162],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-8.872912829762626, 39.81592034453105, 524.8300106098162],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-8.872912829762626, 39.81592034453105, 524.8300106098162],
            rotation_rad: [
                -0.7431804625374093,
                -0.18050150735957685,
                0.15117139989023604,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-17.005654339184, 34.63769423121939, 531.3939169311982],
            rotation_rad: [
                -0.7428975947219212,
                -0.47561470604682005,
                0.4133086082306029,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [-17.005654339184, 34.63769423121939, 531.3939169311982],
            rotation_rad: [
                -0.7428975947219212,
                -0.47561470604682005,
                0.4133086082306029,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [-20.574649637506635, 32.44796635489947, 534.5240043162531],
            rotation_rad: [-0.7386866048373887, -0.575832488263852, 0.5138415001127281],
        },
        HeadPose {
            valid: true,
            position_mm: [15.675816861389677, 38.89702134868061, 501.85328901980864],
            rotation_rad: [
                -0.7411842882790189,
                0.09333567696131931,
                -0.11353690695767861,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [16.083982359538822, 39.2506498784798, 502.1224439053334],
            rotation_rad: [
                -0.7340575179744467,
                0.09934521483776439,
                -0.11734430740825331,
            ],
        },
        HeadPose {
            valid: false,
            position_mm: [16.083982359538822, 39.2506498784798, 502.1224439053334],
            rotation_rad: [
                -0.7340575179744467,
                0.09934521483776439,
                -0.11734430740825331,
            ],
        },
        HeadPose {
            valid: true,
            position_mm: [16.912906561018406, 39.91397852712639, 503.4408826881866],
            rotation_rad: [
                -0.7144129484575245,
                0.1190394762258413,
                -0.12010518856689958,
            ],
        },
    ];
}
