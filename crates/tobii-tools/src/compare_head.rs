//! `compare-dll --head`: replay a captured session's IR images (stream
//! 0x50e) through the head pose the daemon publishes, and compare the poses
//! with the ones the Windows Stream Engine reported for the same images (the
//! `headPose` records of the session's JSONL).
//!
//! The replay is the daemon's own step from an image to its pose,
//! [`HeadStep`] with [`HeadParams::FITTED`], over every image of the log in
//! log order. Its clock is the images' device time: a replay has no host
//! time. The Stream Engine stamps a pose with the device time of its image
//! less the session's clock offset K, which the gaze points give
//! ([`clock_offset`]), so the DLL's pose of time `ts` pairs with the image
//! of device time `ts + K`.
//!
//! The display area the session ran on fixes the frame of both poses. By
//! default it comes from the log: the area of its display-area
//! notifications (1450), else of its display-area writes (1440), else a
//! rigid fit of the gaze origins the device sent in both frames (keys
//! 0x02/0x08 in the tracker frame, 0x22/0x24 in the display frame). The
//! fit takes origins that stray from the line between the eyes
//! ([`MIN_SPREAD_MM`]): those of a head held quite still may not, and the
//! tool then asks for `--display-area`. Any area is checked against those
//! origins, one given on the command line too: the device's own area maps
//! the one onto the other to 0.0002 mm, any other area misses by 0.85 mm or
//! more.
//!
//! The report follows the head-pose study's acceptance tables: the DLL's
//! validity against ours and its losses of the face, the errors of the
//! rotation and the position on the images both call valid, on all of them
//! and on subsets ([`Subset`]), and what the replay cost. `--gates` checks
//! the metrics against a gates file ([`crate::gates`]). `--csv` writes one
//! row per paired image, for the lag and the jitter of the filters, which
//! are measured in Python against a reference this tool does not have.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use tobii_ipc::geometry::{DisplayArea, DisplayFrame};
use tobii_pose::head::{FaceSeen, FrameContext, HeadParams, HeadPose, HeadStep, compose_yxz};
use tobii_proto::facts::parse_display_area;
use tobii_proto::gaze83::{GazeFrame, decode_gaze_frame};
use tobii_proto::image83::decode_image_payload;
use tobii_proto::log::read_log_payloads;
use tobii_proto::protocol::{
    BulkReassembler, MARKER_COMMAND, MARKER_NOTIFICATION, MARKER_STREAM, STREAM_ID_IMAGE, cmd,
    notify, parse_message,
};

use crate::compare_dll::{ClockOffset, DllHeadPose, DllRecord, clock_offset, read_dll_records};
use crate::gates::{Checked, Gates};
use crate::math::{norm3, symmetric_eigen};

/// How far the display frame in use may put a gaze origin from where the
/// device put it, mm, for the frame to be the device's. The device's own
/// area misses by 0.00016 mm at most on every frame recorded, another area
/// by 0.85 mm or more.
const AREA_TOLERANCE_MM: f64 = 0.001;

/// The least RMS distance of a log's gaze origins (tracker frame) from the
/// line through them for a rigid fit of them to fix the display frame, mm.
/// The origins of a head held still lie near the line between its eyes,
/// and only how far they stray from it fixes the frame's turn about that
/// line. At 0.1 mm the device's float rounding of the origins turns the
/// fitted frame by 0.005° at most over 50 gaze frames, less over more (200
/// simulated sessions each).
const MIN_SPREAD_MM: f64 = 0.1;

/// Half the side of the square display area a rigid fit of the gaze
/// origins stands for, mm: the origins fix the display frame, not the size
/// of the screen.
const FITTED_AREA_HALF_MM: f64 = 100.0;

/// How much older than an image a gaze frame may be for its eyes to count
/// for the image, µs (the stream sends one every 30 ms).
const EYES_MAX_AGE_US: i64 = 100_000;

/// The DLL-valid images after a loss of the face that are a re-acquisition.
const REACQUISITION_IMAGES: usize = 10;

/// The DLL's |yaw| from which an image is in [`Subset::LargeYaw`], degrees.
const LARGE_YAW_DEG: f64 = 20.0;

/// The DLL's |yaw|, |pitch| and |roll| above which an image is in
/// [`Subset::Combined`] (the yaw, and the pitch or the roll), degrees.
const COMBINED_DEG: [f64; 3] = [15.0, 8.0, 10.0];

/// The columns of the CSV, one row per image with a DLL pose: the image's
/// number in the log ([`Frame::image`]) and device time; the DLL's pose
/// (its timestamp, all four flags as one validity, position in the display
/// frame, mm, rotation as yxz angles, radians) and ours alike; what the
/// tracker found (whether a face, its score, whether the detector found it,
/// how far inside the frame the landmarks' centroid and the tip of the nose
/// are, px; empty without a face); the image's subsets ([`Subset`], 0 or
/// 1); and what the step cost (µs, model runs). An invalid pose carries the
/// values its side gave: ours the last valid pose's, the DLL's whatever its
/// record held.
const CSV_HEADER: &str = "image,device_ts_us,dll_ts_us,dll_valid,\
dll_pos_x_mm,dll_pos_y_mm,dll_pos_z_mm,dll_rot_x_rad,dll_rot_y_rad,dll_rot_z_rad,\
ours_valid,ours_pos_x_mm,ours_pos_y_mm,ours_pos_z_mm,ours_rot_x_rad,ours_rot_y_rad,ours_rot_z_rad,\
face,face_score,by_detector,centroid_edge_px,nose_tip_edge_px,\
both_eyes,no_eyes,yaw20,reacq,comb,step_us,landmark_runs,detector_runs";

/// Where `compare-dll --head` takes the display area from.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) enum AreaChoice {
    /// From the log (see the [module docs](self)).
    #[default]
    Auto,
    /// This area, whatever the log says.
    Given(DisplayArea),
}

/// The options of `compare-dll --head`.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct HeadOptions {
    /// Where to take the display area from.
    pub(crate) area: AreaChoice,
    /// A gates file to check the metrics against.
    pub(crate) gates: Option<String>,
    /// A CSV file to write the paired images to.
    pub(crate) csv: Option<String>,
}

/// Parse the value of `--display-area`: `auto`, or the top-left, top-right
/// and bottom-left corners of the area in the tracker frame, mm, as nine
/// comma-separated numbers.
///
/// # Errors
/// Anything else, and an area that fixes no display frame.
pub(crate) fn parse_area_choice(s: &str) -> Result<AreaChoice> {
    if s.trim() == "auto" {
        return Ok(AreaChoice::Auto);
    }
    let values: Vec<f64> = s
        .split(',')
        .map(|v| v.trim().parse())
        .collect::<Result<_, _>>()
        .with_context(|| format!("bad --display-area {s}"))?;
    let Ok([a, b, c, d, e, f, g, h, i]) = <[f64; 9]>::try_from(values) else {
        bail!(
            "--display-area takes auto or nine numbers, TLx,TLy,TLz,TRx,TRy,TRz,BLx,BLy,BLz (mm)"
        );
    };
    let area = DisplayArea {
        top_left_mm: [a, b, c],
        top_right_mm: [d, e, f],
        bottom_left_mm: [g, h, i],
    };
    ensure!(
        DisplayFrame::new(&area).is_some(),
        "--display-area {s} fixes no display frame"
    );
    Ok(AreaChoice::Given(area))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `m v`.
fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

/// `mᵀ v`.
fn mat_t_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = *m;
    [0, 1, 2].map(|k| x[k] * v[0] + y[k] * v[1] + z[k] * v[2])
}

/// `a` wrapped to [−180, 180) degrees.
fn wrap_deg(a: f64) -> f64 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

/// `100 a / b`; NaN when `b` is 0.
// cast: image counts, far below 2^53
#[allow(clippy::cast_precision_loss)]
fn percent(a: usize, b: usize) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        100.0 * a as f64 / b as f64
    }
}

/// `values` sorted ascending.
fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

/// The `q`th percentile of `sorted` (ascending), as `NumPy`'s default
/// method gives it: interpolated linearly between the two nearest ranks.
/// NaN for no values.
// cast: the rank is within [0, len - 1], so its floor is an index; the
// length is far below 2^53
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f64::NAN;
    };
    let rank = q / 100.0 * last as f64;
    let below = rank.floor();
    let i = below as usize;
    let (Some(&a), Some(&b)) = (sorted.get(i), sorted.get((i + 1).min(last))) else {
        return f64::NAN;
    };
    let t = rank - below;
    // NumPy's lerp, from the nearer end: t = 1 gives b exactly.
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// The angle between the rotations of the yxz angles `a` and `b`
/// (radians), degrees: the angle of `Ra^T Rb`.
fn geodesic_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (ra, rb) = (compose_yxz(a), compose_yxz(b));
    let trace: f64 = ra
        .iter()
        .flatten()
        .zip(rb.iter().flatten())
        .map(|(x, y)| x * y)
        .sum();
    ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

/// The maximal runs of consecutive items of `items` that `pred` holds for,
/// as (first index, length).
fn runs<T>(items: &[T], pred: impl Fn(&T) -> bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, item) in items.iter().enumerate() {
        match (pred(item), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i - s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, items.len() - s));
    }
    out
}

/// `a − b` as a signed number of images.
fn signed_diff(a: usize, b: usize) -> i64 {
    let magnitude = i64::try_from(a.abs_diff(b)).unwrap_or(i64::MAX);
    if a >= b { magnitude } else { -magnitude }
}

/// A gaze origin the device sent in both frames: (tracker frame, display
/// frame), mm.
type OriginPair = ([f64; 3], [f64; 3]);

/// The gaze origins of every eye of `frames` that has both valid.
fn origin_pairs(frames: &[GazeFrame]) -> Vec<OriginPair> {
    frames
        .iter()
        .flat_map(|f| [&f.left, &f.right])
        .filter(|eye| eye.origin_tracker_mm.valid && eye.origin_display_mm.valid)
        .map(|eye| (eye.origin_tracker_mm.value, eye.origin_display_mm.value))
        .collect()
}

/// The rotation `R` that takes points `p` closest to points `q`, `q ≈ R p`
/// in the least-squares sense, of their cross-covariance `cross`,
/// `Σ (p − p̄)(q − q̄)ᵀ`: Horn's unit quaternion, the eigenvector of the
/// largest eigenvalue of his symmetric 4x4 matrix, solved for to convergence
/// ([`symmetric_eigen`]). `None` should the eigen-solve not converge.
fn horn_rotation(cross: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = *cross;
    let horn = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (values, vectors) = symmetric_eigen(horn)?;
    let top = (0..4).max_by(|&i, &j| values[i].total_cmp(&values[j]))?;
    let q = vectors.map(|row| row[top]);
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let [w, x, y, z] = q.map(|v| v / norm);
    Some([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
}

/// A rigid map from the tracker frame to the display frame:
/// `p_T = R p_S + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RigidMap {
    rotation: [[f64; 3]; 3],
    translation_mm: [f64; 3],
    /// How far the tracker-frame points it was fitted to lie from the line
    /// through them, mm (RMS): how well they fixed its turn about that line.
    spread_mm: f64,
}

impl RigidMap {
    /// The rigid map that takes the tracker-frame point of each pair
    /// closest to its display-frame point in the least-squares sense: the
    /// rotation of the centred points ([`horn_rotation`]), then the
    /// translation of the centroids. Not the tracker's
    /// [`kabsch`](tobii_pose::track::kabsch): its fixed
    /// 200 power steps stop short when the points lie near one line, and
    /// the gaze origins of a head held still do, along the line between
    /// the eyes.
    ///
    /// # Errors
    /// Fails for fewer than three pairs, a point that is not finite, points
    /// that lie within [`MIN_SPREAD_MM`] (RMS) of one line, which leave the
    /// turn about it open, and should an eigen-solve not converge.
    fn fit(pairs: &[OriginPair]) -> Result<Self> {
        ensure!(
            pairs.len() >= 3,
            "{} gaze origins: too few to fit a frame to",
            pairs.len()
        );
        ensure!(
            pairs
                .iter()
                .all(|(s, t)| s.iter().chain(t).all(|v| v.is_finite())),
            "a gaze origin that is not finite"
        );
        // cast: a count of points, far below 2^53
        #[allow(clippy::cast_precision_loss)]
        let n = pairs.len() as f64;
        let centroid = |pick: fn(&OriginPair) -> [f64; 3]| {
            pairs
                .iter()
                .fold([0.0; 3], |s, p| add(s, pick(p)))
                .map(|c| c / n)
        };
        let (from, to) = (centroid(|p| p.0), centroid(|p| p.1));
        let zero = [[0.0; 3]; 3];
        let (cross, scatter) = pairs.iter().fold((zero, zero), |(cross, scatter), (s, t)| {
            let (ds, dt) = (sub(*s, from), sub(*t, to));
            (
                std::array::from_fn(|a| std::array::from_fn(|b| cross[a][b] + ds[a] * dt[b])),
                std::array::from_fn(|a| std::array::from_fn(|b| scatter[a][b] + ds[a] * ds[b])),
            )
        });
        // The tracker-frame points' squared distances from the line along
        // their principal axis sum to the scatter's trace less its largest
        // eigenvalue.
        let (spread, _) =
            symmetric_eigen(scatter).context("the eigen-solve of the origins' spread")?;
        let largest = spread.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let trace = scatter[0][0] + scatter[1][1] + scatter[2][2];
        let spread_mm = ((trace - largest).max(0.0) / n).sqrt();
        ensure!(
            spread_mm >= MIN_SPREAD_MM,
            "the {} gaze origins lie {spread_mm:.4} mm (rms) off one line, as those of a head \
             held still lie along the line between its eyes: they leave the turn about it open \
             (it takes {MIN_SPREAD_MM} mm)",
            pairs.len()
        );
        let rotation = horn_rotation(&cross).context("the eigen-solve of the rotation")?;
        Ok(Self {
            rotation,
            translation_mm: sub(to, mat_vec(&rotation, from)),
            spread_mm,
        })
    }

    /// The display frame of the map, as the display area that has it gives
    /// it: a square around the map's centre `−Rᵀ t`, in its plane.
    fn frame(&self) -> Option<DisplayFrame> {
        let back = |v: [f64; 3]| mat_t_vec(&self.rotation, v);
        let centre = back(self.translation_mm).map(|c| -c);
        let corner = |x: f64, y: f64| add(centre, back([x, y, 0.0]));
        let h = FITTED_AREA_HALF_MM;
        DisplayFrame::new(&DisplayArea {
            top_left_mm: corner(-h, h),
            top_right_mm: corner(h, h),
            bottom_left_mm: corner(-h, -h),
        })
    }
}

/// How far a display frame puts the gaze origins of a log from where the
/// device put them.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OriginCheck {
    /// The origins checked.
    pairs: usize,
    /// The largest distance, mm.
    max_mm: f64,
    /// The RMS distance, mm.
    rms_mm: f64,
}

impl OriginCheck {
    /// How far `frame` puts the tracker-frame origin of each pair from its
    /// display-frame one; `None` without pairs.
    fn of(frame: &DisplayFrame, pairs: &[OriginPair]) -> Option<Self> {
        if pairs.is_empty() {
            return None;
        }
        let (mut max_mm, mut squares) = (0.0f64, 0.0);
        for (tracker, display) in pairs {
            let d = norm3(sub(frame.to_display(*tracker), *display));
            if d.is_nan() || d > max_mm {
                max_mm = d;
            }
            squares += d * d;
        }
        // cast: a count of pairs, far below 2^53
        #[allow(clippy::cast_precision_loss)]
        let n = pairs.len() as f64;
        Some(Self {
            pairs: pairs.len(),
            max_mm,
            rms_mm: (squares / n).sqrt(),
        })
    }

    /// Whether the frame is the device's: no origin off by more than
    /// [`AREA_TOLERANCE_MM`].
    fn fits(&self) -> bool {
        self.max_mm <= AREA_TOLERANCE_MM
    }
}

/// Where the display area of a replay came from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum AreaSource {
    /// The log's display-area notifications (1450), this many, all alike.
    Notified(usize),
    /// The log's display-area writes (1440), this many, all alike.
    Written(usize),
    /// A rigid fit of the log's gaze origins ([`RigidMap::fit`]), which lie
    /// this far off the line through them, mm (RMS).
    Fitted {
        /// [`RigidMap::spread_mm`].
        spread_mm: f64,
    },
    /// The command line.
    Given,
}

/// What a log holds besides its images' pixels, from a first pass over it.
#[derive(Debug, Default)]
struct LogScan {
    /// Its gaze frames, in log order.
    gaze: Vec<GazeFrame>,
    /// The areas of its display-area notifications (1450), in log order.
    notified: Vec<DisplayArea>,
    /// The areas of its display-area writes (1440), in log order.
    written: Vec<DisplayArea>,
    /// How many image messages it has.
    images: usize,
}

/// The messages of a log's records in log order, reassembled as the
/// daemon's reader reassembles what the device sends.
fn messages(payloads: &[Vec<u8>]) -> impl Iterator<Item = Vec<u8>> + '_ {
    let mut asm = BulkReassembler::new();
    payloads.iter().flat_map(move |record| {
        let mut msgs = Vec::new();
        asm.push_into(record, &mut msgs);
        msgs
    })
}

/// The first pass over a log's records.
fn scan_log(payloads: &[Vec<u8>]) -> LogScan {
    let mut scan = LogScan {
        // A write is a host-to-device record, which the reassembler, taking
        // in what the device sent, passes over: its command is the record's
        // first message.
        written: payloads
            .iter()
            .filter_map(|record| parse_message(record))
            .filter(|m| m.marker == MARKER_COMMAND && m.id == cmd::DISPLAY_AREA_SET)
            .filter_map(|m| parse_display_area(&m).map(|(area, _)| area))
            .collect(),
        ..LogScan::default()
    };
    for msg in messages(payloads) {
        let Some(m) = parse_message(&msg) else {
            continue;
        };
        match (m.marker, m.id) {
            (MARKER_STREAM, STREAM_ID_IMAGE) => scan.images += 1,
            (MARKER_STREAM, _) => scan.gaze.extend(decode_gaze_frame(&m)),
            (MARKER_NOTIFICATION, notify::DISPLAY_AREA) => scan
                .notified
                .extend(parse_display_area(&m).map(|(area, _)| area)),
            _ => {}
        }
    }
    scan
}

/// The one area `areas` (a log's messages of one kind) all carry; `None`
/// without any.
///
/// # Errors
/// Fails when they carry different areas: the area changed during the
/// session, which a replay of one frame cannot follow.
fn one_area(areas: &[DisplayArea], what: &str) -> Result<Option<DisplayArea>> {
    let Some(first) = areas.first() else {
        return Ok(None);
    };
    ensure!(
        areas.iter().all(|a| a == first),
        "the log's {what} do not all carry the same area: the display area changed during \
         the session; give --display-area"
    );
    Ok(Some(*first))
}

/// The display frame of the replay and where its area came from (see the
/// [module docs](self)).
///
/// # Errors
/// Fails when the log's display-area messages disagree, when an area fixes
/// no display frame, and when the area must be fitted and the log's gaze
/// origins fix none ([`RigidMap::fit`]).
fn display_frame(
    choice: AreaChoice,
    scan: &LogScan,
    pairs: &[OriginPair],
) -> Result<(DisplayFrame, AreaSource)> {
    let of_area = |area: DisplayArea, source: AreaSource| {
        DisplayFrame::new(&area)
            .map(|frame| (frame, source))
            .with_context(|| format!("the display area {area:?} fixes no display frame"))
    };
    match choice {
        AreaChoice::Given(area) => of_area(area, AreaSource::Given),
        AreaChoice::Auto => {
            if let Some(area) = one_area(&scan.notified, "display-area notifications (1450)")? {
                of_area(area, AreaSource::Notified(scan.notified.len()))
            } else if let Some(area) = one_area(&scan.written, "display-area writes (1440)")? {
                of_area(area, AreaSource::Written(scan.written.len()))
            } else {
                let fitted = RigidMap::fit(pairs).and_then(|map| {
                    let frame = map
                        .frame()
                        .context("the fitted frame fixes no display area")?;
                    Ok((frame, map.spread_mm))
                });
                let (frame, spread_mm) = fitted.context(
                    "the log has no display-area message (1450 or 1440), and its gaze origins \
                     fix no display frame; give --display-area",
                )?;
                Ok((frame, AreaSource::Fitted { spread_mm }))
            }
        }
    }
}

/// Whether a replay may run in the display frame of an area from `source`,
/// which puts the log's gaze origins as `check` says (`None`: the log has
/// none): an area of the log's, or the fit, must put each within
/// [`AREA_TOLERANCE_MM`] of where the device put it. One given on the
/// command line is used whatever it misses by: the note to print, if any.
///
/// # Errors
/// Fails for an area of the log's messages that misses, which is not the
/// one the device held, and for a fit that misses: no one display frame
/// maps the origins.
fn judge_area(source: AreaSource, check: Option<&OriginCheck>) -> Result<Option<String>> {
    let Some(check) = check.filter(|c| !c.fits()) else {
        return Ok(None);
    };
    match source {
        AreaSource::Given => Ok(Some(format!(
            "the given area misses the gaze origins by more than {AREA_TOLERANCE_MM} mm: it is \
             not the area the device held"
        ))),
        AreaSource::Fitted { .. } => bail!(
            "no one display frame maps the log's gaze origins to within {AREA_TOLERANCE_MM} mm \
             (the best rigid fit misses one by {:.6} mm): did the display area change during the \
             session? give --display-area",
            check.max_mm
        ),
        AreaSource::Notified(_) | AreaSource::Written(_) => bail!(
            "the display area does not map the log's gaze origins to within {AREA_TOLERANCE_MM} \
             mm (it misses one by {:.6} mm): it is not the session's; give --display-area",
            check.max_mm
        ),
    }
}

/// What the tracker found on an image.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Face {
    /// The landmark model's face-presence score.
    score: f32,
    /// Whether the face detector found the face.
    by_detector: bool,
    /// How far inside the frame the landmarks' centroid is, px.
    centroid_edge_px: f64,
    /// How far inside the frame the tip of the nose is, px.
    nose_tip_edge_px: f64,
}

impl From<&FaceSeen> for Face {
    fn from(seen: &FaceSeen) -> Self {
        Self {
            score: seen.score,
            by_detector: seen.found_by_detector,
            centroid_edge_px: seen.edges.centroid_px,
            nose_tip_edge_px: seen.edges.nose_tip_px,
        }
    }
}

/// What the step of one image cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct StepCost {
    /// The time `HeadStep::step` took, µs.
    micros: u64,
    /// Runs of the landmark model.
    landmark_runs: u64,
    /// Runs of the face detector.
    detector_runs: u64,
}

/// The device's eyes at an image: whether the latest gaze frame had each
/// eyeball centre (keys 0x17 and 0x18) valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Eyes {
    left: bool,
    right: bool,
}

/// The device's eyes at device time `t_us`: those of the latest gaze frame
/// of `gaze` (device time, eyes; ascending) at or before it, when it is less
/// than [`EYES_MAX_AGE_US`] older; none otherwise.
fn eyes_at(gaze: &[(i64, Eyes)], t_us: i64) -> Eyes {
    let after = gaze.partition_point(|(t, _)| *t <= t_us);
    after
        .checked_sub(1)
        .and_then(|i| gaze.get(i))
        .filter(|(t, _)| t_us.saturating_sub(*t) < EYES_MAX_AGE_US)
        .map_or_else(Eyes::default, |(_, eyes)| *eyes)
}

/// The DLL's state of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DllState {
    /// No pose: the image came before the DLL's subscription, or after it.
    Missing,
    /// An invalid pose.
    Invalid,
    /// A valid pose.
    Valid,
}

/// One image of the session: what the replay and the DLL made of it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Frame {
    /// Its number among the log's image messages, from 0: those that did
    /// not decode count too, as the study numbers its images.
    image: usize,
    /// Its device time, µs.
    device_ts_us: i64,
    /// Our pose.
    ours: HeadPose,
    /// What the tracker found; `None` without a face.
    face: Option<Face>,
    /// What the step cost.
    cost: StepCost,
    /// The DLL's pose of it, if the DLL reported one.
    dll: Option<DllHeadPose>,
    /// The device's eyes at it.
    eyes: Eyes,
}

impl Frame {
    fn dll_state(&self) -> DllState {
        match &self.dll {
            None => DllState::Missing,
            Some(pose) if pose.is_valid() => DllState::Valid,
            Some(_) => DllState::Invalid,
        }
    }

    /// The DLL's pose, if valid.
    fn dll_valid(&self) -> Option<HeadPose> {
        self.dll.filter(DllHeadPose::is_valid).map(|p| HeadPose {
            valid: true,
            position_mm: p.position_mm,
            rotation_rad: p.rotation_rad,
        })
    }
}

/// The replay of a log's images through [`HeadStep`].
#[derive(Debug)]
struct Replay {
    /// One frame per image that decoded, in log order, not paired yet; the
    /// analysis indexes them by their position here, the report and the
    /// CSV number them as the log does ([`Frame::image`]).
    frames: Vec<Frame>,
    /// Image messages that did not decode.
    undecoded: usize,
    /// Images the tracker failed on, and the first error.
    tracker_errors: (usize, Option<String>),
    /// The fingerprint of the model's constants ([`fingerprint`]).
    fingerprint: u64,
    /// How long the whole replay took, s.
    seconds: f64,
}

/// A fingerprint of the head pose model's constants: FNV-1a (64 bits) of
/// their `Debug` form, which has every field and each number in the
/// shortest form that reads back to it.
fn fingerprint(params: &HeadParams) -> u64 {
    format!("{params:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// Replay every image of a log's records, in log order, through the
/// daemon's head pose step, in the display frame `display`, its time the
/// images' device time.
///
/// # Errors
/// Fails when the models cannot be loaded, or an image's time does not fit
/// an `i64`.
fn replay(payloads: &[Vec<u8>], display: &DisplayFrame) -> Result<Replay> {
    let mut step = HeadStep::new(HeadParams::FITTED).context("loading the head pose models")?;
    let mut out = Replay {
        frames: Vec::new(),
        undecoded: 0,
        tracker_errors: (0, None),
        fingerprint: fingerprint(step.params()),
        seconds: 0.0,
    };
    let mut runs_before = step.model_runs();
    let mut images = 0;
    let start = Instant::now();
    for msg in messages(payloads) {
        let is_image = parse_message(&msg)
            .is_some_and(|m| (m.marker, m.id) == (MARKER_STREAM, STREAM_ID_IMAGE));
        if !is_image {
            continue;
        }
        let number = images;
        images += 1;
        let Some(image) = decode_image_payload(&msg) else {
            out.undecoded += 1;
            continue;
        };
        let device_ts_us =
            i64::try_from(image.device_ts_us).context("an image's time does not fit an i64")?;
        let context = FrameContext {
            t_us: device_ts_us,
            display: Some(display),
            display_generation: 1,
            open: 1,
            legacy_wanted: false,
        };
        let begun = Instant::now();
        let stepped = step.step(&image.pixels, image.width, image.height, &context);
        let micros = u64::try_from(begun.elapsed().as_micros()).unwrap_or(u64::MAX);
        if let Some(e) = &stepped.error {
            out.tracker_errors.0 += 1;
            out.tracker_errors.1.get_or_insert_with(|| format!("{e:#}"));
        }
        let runs = stepped.model_runs;
        out.frames.push(Frame {
            image: number,
            device_ts_us,
            ours: stepped.head,
            face: stepped.face.as_ref().map(Face::from),
            cost: StepCost {
                micros,
                landmark_runs: runs.landmarks.saturating_sub(runs_before.landmarks),
                detector_runs: runs.detector.saturating_sub(runs_before.detector),
            },
            dll: None,
            eyes: Eyes::default(),
        });
        runs_before = runs;
    }
    out.seconds = start.elapsed().as_secs_f64();
    Ok(out)
}

/// How the DLL's poses paired with the images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pairing {
    /// Images with a DLL pose.
    paired: usize,
    /// DLL poses of no image.
    unpaired_poses: usize,
    /// DLL poses that share their time with another.
    duplicate_poses: usize,
}

/// Give each frame the DLL's pose of its image (the one of time `ts` with
/// `ts + k_us` the image's device time; of poses that share a time, the
/// later) and the device's eyes at it, of `gaze` (device time, eyes;
/// ascending).
fn pair(frames: &mut [Frame], poses: &[DllHeadPose], k_us: i64, gaze: &[(i64, Eyes)]) -> Pairing {
    let mut by_ts = HashMap::with_capacity(poses.len());
    let mut duplicate_poses = 0;
    for pose in poses {
        duplicate_poses += usize::from(by_ts.insert(pose.ts_us, *pose).is_some());
    }
    for frame in frames.iter_mut() {
        frame.dll = frame
            .device_ts_us
            .checked_sub(k_us)
            .and_then(|ts| by_ts.get(&ts))
            .copied();
        frame.eyes = eyes_at(gaze, frame.device_ts_us);
    }
    let images: HashSet<i64> = frames.iter().map(|f| f.device_ts_us).collect();
    Pairing {
        paired: frames.iter().filter(|f| f.dll.is_some()).count(),
        unpaired_poses: poses
            .iter()
            .filter(|p| {
                p.ts_us
                    .checked_add(k_us)
                    .is_none_or(|t| !images.contains(&t))
            })
            .count(),
        duplicate_poses,
    }
}

/// Our validity against the DLL's, on the images with a DLL pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Confusion {
    /// Images both call valid.
    both_valid: usize,
    /// Images only the DLL calls valid.
    dll_only: usize,
    /// Of those, the ones the tracker found no face on.
    dll_only_no_face: usize,
    /// Images only we call valid.
    ours_only: usize,
    /// Images neither calls valid.
    neither: usize,
    /// Of those, the ones the tracker found no face on.
    neither_no_face: usize,
}

impl Confusion {
    /// The table of images with DLL state `dll`, our validity `ours` and
    /// whether the tracker found a face, `face`.
    fn of(dll: &[DllState], ours: &[bool], face: &[bool]) -> Self {
        let mut c = Self::default();
        for ((state, &valid), &face) in dll.iter().zip(ours).zip(face) {
            match (state, valid) {
                (DllState::Missing, _) => {}
                (DllState::Valid, true) => c.both_valid += 1,
                (DllState::Valid, false) => {
                    c.dll_only += 1;
                    c.dll_only_no_face += usize::from(!face);
                }
                (DllState::Invalid, true) => c.ours_only += 1,
                (DllState::Invalid, false) => {
                    c.neither += 1;
                    c.neither_no_face += usize::from(!face);
                }
            }
        }
        c
    }

    /// The images with a DLL pose.
    fn paired(&self) -> usize {
        self.both_valid + self.dll_only + self.ours_only + self.neither
    }

    /// Our valid share of the DLL-valid images, %.
    fn coverage_pct(&self) -> f64 {
        percent(self.both_valid, self.both_valid + self.dll_only)
    }

    /// The share of the paired images whose validity agrees with the DLL's, %.
    fn agreement_pct(&self) -> f64 {
        percent(self.both_valid + self.neither, self.paired())
    }
}

/// A run of images the DLL calls invalid, a loss of the face, and whether
/// we lost the face too. It counts the images that decoded, by their
/// position in the replay's frames ([`LossEvent::images`] numbers them as
/// the log does).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LossEvent {
    /// The run's first image.
    start: usize,
    /// Its length, images.
    len: usize,
    /// When runs of our invalid poses overlap it: how many images after the
    /// run's start the first of them starts (onset), and after the run's
    /// end the last of them ends (offset); negative when before.
    seen: Option<(i64, i64)>,
}

impl LossEvent {
    /// The log's numbers ([`Frame::image`]) of the run's first image and of
    /// the one after its last, of `frames`, the frames it was found in.
    fn images(&self, frames: &[Frame]) -> (usize, usize) {
        let number = |i: usize| frames.get(i).map_or(i, |f| f.image);
        let last = (self.start + self.len).saturating_sub(1);
        (number(self.start), number(last) + 1)
    }
}

/// The DLL's loss events, and the runs of our invalid poses that overlap
/// none of them; our runs count on the images with a DLL pose.
fn loss_events(dll: &[DllState], ours_valid: &[bool]) -> (Vec<LossEvent>, Vec<(usize, usize)>) {
    let ours_invalid: Vec<bool> = dll
        .iter()
        .zip(ours_valid)
        .map(|(state, &valid)| *state != DllState::Missing && !valid)
        .collect();
    let ours = runs(&ours_invalid, |&invalid| invalid);
    let events = runs(dll, |state| *state == DllState::Invalid)
        .into_iter()
        .map(|(start, len)| {
            let end = start + len;
            let mut overlapping = ours.iter().filter(|(s, l)| *s < end && s + l > start);
            let seen = overlapping.next().map(|&(first, first_len)| {
                let (last, last_len) = overlapping
                    .next_back()
                    .copied()
                    .unwrap_or((first, first_len));
                (signed_diff(first, start), signed_diff(last + last_len, end))
            });
            LossEvent { start, len, seen }
        })
        .collect();
    let unmatched = ours
        .into_iter()
        .filter(|&(s, l)| {
            dll.get(s..s + l)
                .is_some_and(|states| !states.contains(&DllState::Invalid))
        })
        .collect();
    (events, unmatched)
}

/// Which images are among the first [`REACQUISITION_IMAGES`] DLL-valid ones
/// after a loss of the face, a run of DLL-invalid ones; the start of the
/// session is no re-acquisition.
fn reacquisition(dll: &[DllState]) -> Vec<bool> {
    let mut out = vec![false; dll.len()];
    for (start, len) in runs(dll, |state| *state == DllState::Valid) {
        let after_loss = start.checked_sub(1).and_then(|i| dll.get(i)) == Some(&DllState::Invalid);
        if after_loss {
            for flag in out
                .iter_mut()
                .skip(start)
                .take(len.min(REACQUISITION_IMAGES))
            {
                *flag = true;
            }
        }
    }
    out
}

/// The subsets of the DLL-valid images that the errors are also given on,
/// as the head-pose study defined them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Subset {
    /// Every DLL-valid image.
    All,
    /// Both of the device's eyes at the image ([`eyes_at`]).
    BothEyes,
    /// Neither of them.
    NoEyes,
    /// The DLL's |yaw| at least 20°.
    LargeYaw,
    /// The first 10 DLL-valid images after a loss of the face
    /// ([`reacquisition`]).
    Reacquired,
    /// The DLL's |yaw| above 15° and its |pitch| above 8° or |roll| above
    /// 10°: combined turns, where the order of the Euler angles shows.
    Combined,
}

impl Subset {
    const ALL: [Self; 6] = [
        Self::All,
        Self::BothEyes,
        Self::NoEyes,
        Self::LargeYaw,
        Self::Reacquired,
        Self::Combined,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::All => "ALL",
            Self::BothEyes => "BOTH",
            Self::NoEyes => "NONE",
            Self::LargeYaw => "YAW20",
            Self::Reacquired => "REACQ",
            Self::Combined => "COMB",
        }
    }

    /// Whether `frame`, which is a re-acquisition or not, is in the subset:
    /// never when the DLL's pose of it is not valid.
    fn contains(self, frame: &Frame, reacquired: bool) -> bool {
        let Some(dll) = frame.dll_valid() else {
            return false;
        };
        let [pitch, yaw, roll] = dll.rotation_rad.map(|a| a.to_degrees().abs());
        let [yaw_above, pitch_above, roll_above] = COMBINED_DEG;
        match self {
            Self::All => true,
            Self::BothEyes => frame.eyes.left && frame.eyes.right,
            Self::NoEyes => !frame.eyes.left && !frame.eyes.right,
            Self::LargeYaw => yaw >= LARGE_YAW_DEG,
            Self::Reacquired => reacquired,
            Self::Combined => yaw > yaw_above && (pitch > pitch_above || roll > roll_above),
        }
    }
}

/// Our errors against the DLL on the images of a subset that both call
/// valid. Each is the median or the 95th percentile of the images' errors
/// ([`percentile`]); NaN without images.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Errors {
    /// The subset's images (all DLL-valid).
    subset: usize,
    /// Those our pose is valid on too, which the errors are of.
    both_valid: usize,
    /// |wrap(ours − DLL)| of each yxz angle, degrees: median.
    rotation_median_deg: [f64; 3],
    /// The same, 95th percentile.
    rotation_p95_deg: [f64; 3],
    /// The median of the signed errors, degrees.
    rotation_signed_median_deg: [f64; 3],
    /// The angle between the two rotations ([`geodesic_deg`]): median.
    geodesic_median_deg: f64,
    /// The same, 95th percentile.
    geodesic_p95_deg: f64,
    /// |ours − DLL| along each axis of the display frame, mm: median.
    position_median_mm: [f64; 3],
    /// The same, 95th percentile.
    position_p95_mm: [f64; 3],
    /// The distance between the two positions, mm: median.
    position_median_3d_mm: f64,
}

impl Errors {
    /// The errors of `pairs`, our pose and the DLL's of each image both
    /// call valid, among the `subset` images of a subset.
    fn of(pairs: &[(HeadPose, HeadPose)], subset: usize) -> Self {
        let mut signed: [Vec<f64>; 3] = Default::default();
        let mut position: [Vec<f64>; 3] = Default::default();
        let (mut geodesic, mut distance) = (Vec::new(), Vec::new());
        for (ours, dll) in pairs {
            let angle_errors = ours
                .rotation_rad
                .into_iter()
                .zip(dll.rotation_rad)
                .map(|(o, d)| wrap_deg(o.to_degrees() - d.to_degrees()));
            for (errors, e) in signed.iter_mut().zip(angle_errors) {
                errors.push(e);
            }
            let offsets = sub(ours.position_mm, dll.position_mm);
            for (errors, e) in position.iter_mut().zip(offsets) {
                errors.push(e.abs());
            }
            geodesic.push(geodesic_deg(ours.rotation_rad, dll.rotation_rad));
            distance.push(norm3(offsets));
        }
        let rotation = signed
            .clone()
            .map(|errors| sorted(errors.into_iter().map(f64::abs).collect()));
        let signed = signed.map(sorted);
        let position = position.map(sorted);
        let (geodesic, distance) = (sorted(geodesic), sorted(distance));
        Self {
            subset,
            both_valid: pairs.len(),
            rotation_median_deg: rotation.each_ref().map(|e| percentile(e, 50.0)),
            rotation_p95_deg: rotation.each_ref().map(|e| percentile(e, 95.0)),
            rotation_signed_median_deg: signed.each_ref().map(|e| percentile(e, 50.0)),
            geodesic_median_deg: percentile(&geodesic, 50.0),
            geodesic_p95_deg: percentile(&geodesic, 95.0),
            position_median_mm: position.each_ref().map(|e| percentile(e, 50.0)),
            position_p95_mm: position.each_ref().map(|e| percentile(e, 95.0)),
            position_median_3d_mm: percentile(&distance, 50.0),
        }
    }

    /// Our valid share of the subset, %.
    fn coverage_pct(&self) -> f64 {
        percent(self.both_valid, self.subset)
    }
}

/// Everything the report and the gates read of a paired session.
#[derive(Debug, Clone, PartialEq)]
struct Analysis {
    confusion: Confusion,
    events: Vec<LossEvent>,
    /// Our runs of invalid poses that overlap no loss of the DLL's.
    unmatched_runs: Vec<(usize, usize)>,
    /// The errors on each subset, in the order of [`Subset::ALL`].
    errors: [(Subset, Errors); 6],
    /// Which images are re-acquisitions ([`reacquisition`]).
    reacquired: Vec<bool>,
}

impl Analysis {
    fn of(frames: &[Frame]) -> Self {
        let dll: Vec<DllState> = frames.iter().map(Frame::dll_state).collect();
        let ours: Vec<bool> = frames.iter().map(|f| f.ours.valid).collect();
        let face: Vec<bool> = frames.iter().map(|f| f.face.is_some()).collect();
        let (events, unmatched_runs) = loss_events(&dll, &ours);
        let reacquired = reacquisition(&dll);
        let errors = Subset::ALL.map(|subset| {
            let members: Vec<&Frame> = frames
                .iter()
                .zip(&reacquired)
                .filter(|(f, r)| subset.contains(f, **r))
                .map(|(f, _)| f)
                .collect();
            let pairs: Vec<(HeadPose, HeadPose)> = members
                .iter()
                .filter(|f| f.ours.valid)
                .filter_map(|f| Some((f.ours, f.dll_valid()?)))
                .collect();
            (subset, Errors::of(&pairs, members.len()))
        });
        Self {
            confusion: Confusion::of(&dll, &ours, &face),
            events,
            unmatched_runs,
            errors,
            reacquired,
        }
    }

    fn errors(&self, subset: Subset) -> &Errors {
        self.errors
            .iter()
            .find(|(s, _)| *s == subset)
            .map_or(&self.errors[0].1, |(_, e)| e)
    }

    /// The value of the metric `name` that a gate may bound; `None` for a
    /// name this tool does not compute. See `tools/headpose/gates.json`.
    fn metric(&self, name: &str) -> Option<f64> {
        let all = self.errors(Subset::All);
        let combined = self.errors(Subset::Combined);
        // cast: a count of events, far below 2^53
        #[allow(clippy::cast_precision_loss)]
        let found = self.events.iter().filter(|e| e.seen.is_some()).count() as f64;
        Some(match name {
            "coverage_pct" => self.confusion.coverage_pct(),
            "agreement_pct" => self.confusion.agreement_pct(),
            "loss_events_found" => found,
            "rotation_median_abs_x_deg" => all.rotation_median_deg[0],
            "rotation_median_abs_y_deg" => all.rotation_median_deg[1],
            "rotation_median_abs_z_deg" => all.rotation_median_deg[2],
            "rotation_median_signed_x_deg" => all.rotation_signed_median_deg[0],
            "rotation_geodesic_p95_deg" => all.geodesic_p95_deg,
            "combined_rotation_p95_abs_x_deg" => combined.rotation_p95_deg[0],
            "combined_rotation_geodesic_p95_deg" => combined.geodesic_p95_deg,
            "position_median_abs_x_mm" => all.position_median_mm[0],
            "position_median_abs_y_mm" => all.position_median_mm[1],
            "position_median_abs_z_mm" => all.position_median_mm[2],
            "position_p95_abs_z_mm" => all.position_p95_mm[2],
            _ => return None,
        })
    }
}

/// The CSV row of `frame`, a re-acquisition or not ([`CSV_HEADER`]); `None`
/// for an image without a DLL pose.
fn csv_row(frame: &Frame, reacquired: bool) -> Option<String> {
    let dll = frame.dll?;
    let bit = |b: bool| u8::from(b).to_string();
    let face = |value: fn(&Face) -> String| frame.face.as_ref().map_or_else(String::new, value);
    let in_subset = |subset: Subset| bit(subset.contains(frame, reacquired));
    let fields = [
        frame.image.to_string(),
        frame.device_ts_us.to_string(),
        dll.ts_us.to_string(),
        bit(dll.is_valid()),
    ]
    .into_iter()
    .chain(
        dll.position_mm
            .iter()
            .chain(&dll.rotation_rad)
            .map(f64::to_string),
    )
    .chain([bit(frame.ours.valid)])
    .chain(
        frame
            .ours
            .position_mm
            .iter()
            .chain(&frame.ours.rotation_rad)
            .map(f64::to_string),
    )
    .chain([
        bit(frame.face.is_some()),
        face(|f| f.score.to_string()),
        face(|f| u8::from(f.by_detector).to_string()),
        face(|f| f.centroid_edge_px.to_string()),
        face(|f| f.nose_tip_edge_px.to_string()),
        in_subset(Subset::BothEyes),
        in_subset(Subset::NoEyes),
        in_subset(Subset::LargeYaw),
        in_subset(Subset::Reacquired),
        in_subset(Subset::Combined),
        frame.cost.micros.to_string(),
        frame.cost.landmark_runs.to_string(),
        frame.cost.detector_runs.to_string(),
    ]);
    Some(fields.collect::<Vec<_>>().join(","))
}

/// Write the CSV of `frames` to `path`: [`CSV_HEADER`], then one row per
/// image with a DLL pose. Returns the rows written.
///
/// # Errors
/// Fails when the file cannot be written.
fn write_csv(path: &str, frames: &[Frame], reacquired: &[bool]) -> Result<usize> {
    let mut out =
        BufWriter::new(File::create(path).with_context(|| format!("failed to create {path}"))?);
    writeln!(out, "{CSV_HEADER}")?;
    let mut rows = 0;
    for (frame, &r) in frames.iter().zip(reacquired) {
        if let Some(row) = csv_row(frame, r) {
            writeln!(out, "{row}")?;
            rows += 1;
        }
    }
    out.flush()
        .with_context(|| format!("failed to write {path}"))?;
    Ok(rows)
}

/// `v` to two decimals; `-` for a NaN.
fn f2(v: f64) -> String {
    if v.is_nan() {
        "-".to_string()
    } else {
        format!("{v:.2}")
    }
}

/// Three values to two decimals, `a / b / c`.
fn f2x3(v: [f64; 3]) -> String {
    v.map(f2).join(" / ")
}

/// The median and the largest of `values`.
// cast: image counts, far below 2^53
#[allow(clippy::cast_precision_loss)]
fn median_max(values: &[u64]) -> (f64, u64) {
    let sorted = sorted(values.iter().map(|&v| v as f64).collect());
    (
        percentile(&sorted, 50.0),
        values.iter().copied().max().unwrap_or(0),
    )
}

/// Print what the replay cost.
// cast: microseconds and run counts, far below 2^53
#[allow(clippy::cast_precision_loss)]
fn print_cost(replay: &Replay) {
    let frames = &replay.frames;
    let ms = sorted(
        frames
            .iter()
            .map(|f| f.cost.micros as f64 / 1000.0)
            .collect(),
    );
    let n = frames.len().max(1) as f64;
    let mean = |v: fn(&StepCost) -> u64| frames.iter().map(|f| v(&f.cost) as f64).sum::<f64>() / n;
    let mut landmark_runs: BTreeMap<u64, usize> = BTreeMap::new();
    for f in frames {
        *landmark_runs.entry(f.cost.landmark_runs).or_default() += 1;
    }
    let by_runs: Vec<String> = landmark_runs
        .iter()
        .map(|(runs, images)| format!("{runs}: {images}"))
        .collect();
    println!(
        "replay: {} images in {:.1} s; HeadStep::step per image: mean {:.2} ms, p99 {:.2} ms, \
         max {:.2} ms",
        frames.len(),
        replay.seconds,
        ms.iter().sum::<f64>() / n,
        percentile(&ms, 99.0),
        ms.last().copied().unwrap_or(f64::NAN),
    );
    println!(
        "model runs per image: landmarks {:.4} (images by runs {}), detector {:.4} \
         ({} images); faces the detector found {}",
        mean(|c| c.landmark_runs),
        by_runs.join(", "),
        mean(|c| c.detector_runs),
        frames.iter().filter(|f| f.cost.detector_runs > 0).count(),
        frames
            .iter()
            .filter(|f| f.face.is_some_and(|face| face.by_detector))
            .count(),
    );
    if replay.undecoded > 0 {
        println!("image messages that did not decode: {}", replay.undecoded);
    }
    let (errors, first) = &replay.tracker_errors;
    if let Some(first) = first {
        println!(
            "images the tracker failed on (each an invalid pose): {errors}; the first: {first}"
        );
    }
}

/// Print the validity of our poses against the DLL's, on `frames`, the
/// frames `analysis` is of.
fn print_validity(analysis: &Analysis, frames: &[Frame]) {
    let c = &analysis.confusion;
    println!(
        "validity on the {} images with a DLL pose (ours invalid: without a face / with one):",
        c.paired()
    );
    println!("                  ours valid   ours invalid");
    println!(
        "  DLL valid     {:>12}   {:>12} ({} / {})",
        c.both_valid,
        c.dll_only,
        c.dll_only_no_face,
        c.dll_only - c.dll_only_no_face
    );
    println!(
        "  DLL invalid   {:>12}   {:>12} ({} / {})",
        c.ours_only,
        c.neither,
        c.neither_no_face,
        c.neither - c.neither_no_face
    );
    println!(
        "coverage of DLL-valid images {} %, agreement {} %",
        f2(c.coverage_pct()),
        f2(c.agreement_pct())
    );
    let seen: Vec<(i64, i64)> = analysis.events.iter().filter_map(|e| e.seen).collect();
    let (onset, onset_max) =
        median_max(&seen.iter().map(|s| s.0.unsigned_abs()).collect::<Vec<_>>());
    let (offset, offset_max) =
        median_max(&seen.iter().map(|s| s.1.unsigned_abs()).collect::<Vec<_>>());
    print!(
        "DLL losses of the face (runs of DLL-invalid images): {} of {} seen",
        seen.len(),
        analysis.events.len()
    );
    if seen.is_empty() {
        println!();
    } else {
        println!(
            "; |onset| median {}, max {onset_max}; |offset| median {}, max {offset_max} images",
            f2(onset),
            f2(offset)
        );
    }
    for e in &analysis.events {
        let seen = e.seen.map_or_else(
            || "not seen".to_string(),
            |(on, off)| format!("onset {on:+}, offset {off:+}"),
        );
        let (first, end) = e.images(frames);
        println!("  images {first}..{end} ({}): {seen}", e.len);
    }
    println!(
        "runs of our invalid poses with no DLL-invalid image: {} ({} of 3 images or more)",
        analysis.unmatched_runs.len(),
        analysis
            .unmatched_runs
            .iter()
            .filter(|(_, l)| *l >= 3)
            .count()
    );
}

/// Print the errors on every subset.
fn print_errors(analysis: &Analysis) {
    let head = |what: &str| {
        println!("{what}, on the images both call valid:");
        println!(
            "  subset  DLL-valid  ours valid           |e| median x / y / z    |e| p95 x / y / z"
        );
    };
    let lead = |subset: Subset, e: &Errors| {
        format!(
            "  {:<6} {:>10} {:>11} {:>8} %",
            subset.name(),
            e.subset,
            e.both_valid,
            f2(e.coverage_pct())
        )
    };
    head("rotation error, degrees (then the geodesic angle: median / p95)");
    for (subset, e) in &analysis.errors {
        println!(
            "{}   {:<22}  {:<22}  {} / {}",
            lead(*subset, e),
            f2x3(e.rotation_median_deg),
            f2x3(e.rotation_p95_deg),
            f2(e.geodesic_median_deg),
            f2(e.geodesic_p95_deg)
        );
    }
    println!(
        "  median signed error x / y / z, ALL: {}",
        f2x3(analysis.errors(Subset::All).rotation_signed_median_deg)
    );
    head("position error, mm (then the 3-D distance: median)");
    for (subset, e) in &analysis.errors {
        println!(
            "{}   {:<22}  {:<22}  {}",
            lead(*subset, e),
            f2x3(e.position_median_mm),
            f2x3(e.position_p95_mm),
            f2(e.position_median_3d_mm)
        );
    }
}

/// Print the gates checked; returns those that fail.
fn print_gates(path: &str, status: &str, session: &str, checked: &[Checked<'_>]) -> Vec<String> {
    println!("gates of {path}, session {session}");
    if !status.is_empty() {
        println!("  {status}");
    }
    let mut failed = Vec::new();
    for c in checked {
        let (value, verdict) = match c.value {
            Some(v) if c.fails() => {
                failed.push(c.gate.id.clone());
                (f2(v), "FAIL")
            }
            Some(v) => (f2(v), "PASS"),
            None => ("-".to_string(), "not checked here: Python, from the CSV"),
        };
        println!(
            "  {:<4} {:<44} {:>14} {:>9}  {verdict}",
            c.gate.id,
            c.gate.name,
            c.bound.to_string(),
            value
        );
    }
    let here = checked.iter().filter(|c| c.value.is_some()).count();
    let python: Vec<&str> = checked
        .iter()
        .filter(|c| c.value.is_none())
        .map(|c| c.gate.id.as_str())
        .collect();
    println!(
        "{here} gates checked: {} PASS, {} FAIL{}",
        here - failed.len(),
        failed.len(),
        if python.is_empty() {
            String::new()
        } else {
            format!("; left to Python: {}", python.join(", "))
        }
    );
    failed
}

/// Replay a captured session's IR images through the daemon's head pose
/// and compare the poses with the Stream Engine's (see the [module
/// docs](self)): print the report, write the CSV and check the gates that
/// `options` ask for.
///
/// # Errors
/// Fails when a file cannot be read or written, no gaze point pairs the two
/// logs, the gates file has no bounds for the session, the display area
/// does not fit the log's gaze origins (unless given), and when a gate
/// fails.
pub(crate) fn compare_head(log_path: &str, jsonl_path: &str, options: &HeadOptions) -> Result<()> {
    // A gates file that does not read fails before the replay.
    let gates = options.gates.as_deref().map(Gates::load).transpose()?;
    let payloads = read_log_payloads(log_path)?;
    let scan = scan_log(&payloads);
    let records = read_dll_records(jsonl_path)?;
    let poses: Vec<DllHeadPose> = records
        .iter()
        .filter_map(|r| match r {
            DllRecord::HeadPose(pose) => Some(*pose),
            _ => None,
        })
        .collect();
    let ClockOffset {
        k_us,
        agreeing,
        matches,
    } = clock_offset(&scan.gaze, &records)?;
    let session = gates.as_ref().map(|g| g.session(k_us)).transpose()?;

    println!("head pose replay of {log_path} against {jsonl_path}");
    println!(
        "log: {} images, {} gaze frames; DLL: {} head poses ({} valid, {} invalid, {} with \
         flags that disagree)",
        scan.images,
        scan.gaze.len(),
        poses.len(),
        poses.iter().filter(|p| p.is_valid()).count(),
        poses.iter().filter(|p| !p.is_valid()).count(),
        poses.iter().filter(|p| !p.flags_agree()).count(),
    );
    println!("clock offset K = {k_us} us: agreed by {agreeing} of {matches} matching gaze points");

    let pairs = origin_pairs(&scan.gaze);
    let (display, source) = display_frame(options.area, &scan, &pairs)?;
    let check = OriginCheck::of(&display, &pairs);
    let from = match source {
        AreaSource::Notified(n) => format!("the log's display-area notification (1450; {n} alike)"),
        AreaSource::Written(n) => format!("the log's display-area write (1440; {n} alike)"),
        AreaSource::Fitted { spread_mm } => format!(
            "a rigid fit of the log's gaze origins (0x02/0x08 -> 0x22/0x24), which lie \
             {spread_mm:.4} mm (rms) off the line through them"
        ),
        AreaSource::Given => "the command line".to_string(),
    };
    let vector = |v: [f64; 3]| format!("({:.6}, {:.6}, {:.6})", v[0], v[1], v[2]);
    let [x, y, z] = display.rotation().map(vector);
    println!("display area: {from}");
    println!(
        "  display frame p_T = R p_S + t: t {} mm; R's rows x {x}, y {y}, z {z}",
        vector(display.to_display([0.0; 3]))
    );
    match check {
        Some(c) => println!(
            "  gaze origins it maps: {}, off by {:.6} mm at most (rms {:.6} mm)",
            c.pairs, c.max_mm, c.rms_mm
        ),
        None => println!("  gaze origins it maps: none in the log, unchecked"),
    }
    if let Some(note) = judge_area(source, check.as_ref())? {
        println!("  note: {note}");
    }

    let mut replay = replay(&payloads, &display)?;
    drop(payloads);
    println!(
        "model: HeadStep with HeadParams::FITTED, fingerprint {:016x}; time: the images' device time",
        replay.fingerprint
    );
    let mut gaze: Vec<(i64, Eyes)> = scan
        .gaze
        .iter()
        .filter_map(|f| {
            let t = i64::try_from(f.device_ts_us).ok()?;
            Some((
                t,
                Eyes {
                    left: f.left.eyeball_center_mm.valid,
                    right: f.right.eyeball_center_mm.valid,
                },
            ))
        })
        .collect();
    gaze.sort_by_key(|(t, _)| *t);
    let pairing = pair(&mut replay.frames, &poses, k_us, &gaze);
    println!(
        "paired: {} of {} images have a DLL pose; {} of the {} DLL poses fall on an image at K{}",
        pairing.paired,
        replay.frames.len(),
        poses.len() - pairing.unpaired_poses,
        poses.len(),
        if pairing.duplicate_poses > 0 {
            format!("; DLL poses sharing a time: {}", pairing.duplicate_poses)
        } else {
            String::new()
        }
    );
    ensure!(pairing.paired > 0, "no image has a DLL pose");
    print_cost(&replay);

    let analysis = Analysis::of(&replay.frames);
    print_validity(&analysis, &replay.frames);
    print_errors(&analysis);

    if let Some(path) = &options.csv {
        let rows = write_csv(path, &replay.frames, &analysis.reacquired)?;
        println!("CSV: {rows} rows to {path}");
    }

    if let (Some(gates), Some(session), Some(path)) = (&gates, session, &options.gates) {
        let checked = gates.check(session, |name| analysis.metric(name))?;
        let failed = print_gates(path, &gates.status, &session.name, &checked);
        if !failed.is_empty() {
            bail!("gates failed: {}", failed.join(", "));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_proto::decode::STREAM_TLV_OFFSET;
    use tobii_proto::facts::display_area_set_payload;
    use tobii_proto::protocol::chunk_command;
    use tobii_proto::tlv::{KEY_FIELD_ID, TlvWriter};

    use crate::gates::Checker;
    use DllState::{Invalid as I, Missing as M, Valid as V};

    /// Area A, the display area of the three Windows sessions.
    const AREA_A: DisplayArea = DisplayArea {
        top_left_mm: [
            -315.791_717_529_296_9,
            324.411_468_505_859_4,
            111.239_097_595_214_84,
        ],
        top_right_mm: [
            317.796_234_130_859_4,
            324.411_468_505_859_4,
            111.239_097_595_214_84,
        ],
        bottom_left_mm: [
            -315.791_717_529_296_9,
            10.267_330_169_677_734,
            -3.100_020_170_211_792,
        ],
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn percentiles_interpolate_as_numpy_does() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!(close(percentile(&v, 50.0), 2.5, 1e-12));
        // np.percentile([1, 2, 3, 4], 95) = 3.85, (.., 99) = 3.97
        assert!(close(percentile(&v, 95.0), 3.85, 1e-12));
        assert!(close(percentile(&v, 99.0), 3.97, 1e-12));
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 100.0), 4.0);
        assert_eq!(percentile(&[7.0], 95.0), 7.0);
        assert!(percentile(&[], 50.0).is_nan());
    }

    /// The geodesic angle is that of the rotation from one pose to the
    /// other, whatever the Euler angles say: a pure turn about one axis,
    /// a turn about an oblique axis, and angles that wrap.
    #[test]
    fn the_geodesic_angle_is_the_angle_between_the_rotations() {
        let deg = |a: [f64; 3]| a.map(f64::to_radians);
        assert!(close(geodesic_deg(deg([0.0; 3]), deg([0.0; 3])), 0.0, 1e-6));
        assert!(close(
            geodesic_deg(deg([0.0, 10.0, 0.0]), deg([0.0; 3])),
            10.0,
            1e-9
        ));
        assert!(close(
            geodesic_deg(deg([0.0; 3]), deg([0.0, 0.0, -25.0])),
            25.0,
            1e-9
        ));
        // Turns about one axis compose: from yaw 30 to yaw 50 is 20 degrees,
        // and so from pitch 5 to pitch -5 at the same yaw.
        assert!(close(
            geodesic_deg(deg([0.0, 30.0, 0.0]), deg([0.0, 50.0, 0.0])),
            20.0,
            1e-9
        ));
        assert!(close(
            geodesic_deg(deg([5.0, 30.0, 0.0]), deg([-5.0, 30.0, 0.0])),
            10.0,
            1e-9
        ));
        // Yaw 179 and yaw -179 are 2 degrees apart.
        assert!(close(
            geodesic_deg(deg([0.0, 179.0, 0.0]), deg([0.0, -179.0, 0.0])),
            2.0,
            1e-9
        ));
        // Two rotations 12 degrees apart about an oblique axis, whose yxz
        // angles all differ.
        let a = deg([10.0, -20.0, 5.0]);
        let ra = compose_yxz(a);
        let axis = [1.0, 2.0, -2.0].map(|c: f64| c / 3.0);
        let (s, c) = 12f64.to_radians().sin_cos();
        let k = [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ];
        let turn: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let kk: f64 = (0..3).map(|m| k[i][m] * k[m][j]).sum();
                f64::from(u8::from(i == j)) + s * k[i][j] + (1.0 - c) * kk
            })
        });
        let rb: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|m| ra[i][m] * turn[m][j]).sum())
        });
        let b = tobii_pose::head::euler_yxz(&rb);
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() > 1e-3));
        assert!(close(geodesic_deg(a, b), 12.0, 1e-9));
        assert!(close(geodesic_deg(b, a), 12.0, 1e-9));
    }

    #[test]
    fn runs_are_maximal() {
        let v = [true, true, false, true, false, false, true];
        assert_eq!(runs(&v, |&b| b), [(0, 2), (3, 1), (6, 1)]);
        assert_eq!(runs(&v, |&b| !b), [(2, 1), (4, 2)]);
        assert_eq!(runs::<bool>(&[], |&b| b), []);
    }

    /// `n` copies of each item of `parts`, in order.
    fn seq<T: Copy>(parts: &[(T, usize)]) -> Vec<T> {
        parts
            .iter()
            .flat_map(|&(item, n)| std::iter::repeat_n(item, n))
            .collect()
    }

    /// Loss events are found by overlap: one we see a frame early and end
    /// two late, one two of our runs overlap (onset from the first, offset
    /// from the last), one we do not see; our run that overlaps no loss is
    /// unmatched, and images without a DLL pose count for neither side.
    #[test]
    fn loss_events_are_matched_with_onset_and_offset() {
        // Losses at 7..10, 16..21 and 28..30; no DLL pose at 0..2, 42..44.
        let dll = seq(&[
            (M, 2),
            (V, 5),
            (I, 3),
            (V, 6),
            (I, 5),
            (V, 7),
            (I, 2),
            (V, 12),
            (M, 2),
        ]);
        // Ours invalid at 6..12, 15..17, 18..21 and 35..38 (and where the
        // DLL has no pose).
        let ours = seq(&[
            (false, 2),
            (true, 4),
            (false, 6),
            (true, 3),
            (false, 2),
            (true, 1),
            (false, 3),
            (true, 14),
            (false, 3),
            (true, 4),
            (false, 2),
        ]);
        let (events, unmatched) = loss_events(&dll, &ours);
        assert_eq!(
            events,
            [
                LossEvent {
                    start: 7,
                    len: 3,
                    seen: Some((-1, 2)),
                },
                LossEvent {
                    start: 16,
                    len: 5,
                    seen: Some((-1, 0)),
                },
                LossEvent {
                    start: 28,
                    len: 2,
                    seen: None,
                },
            ]
        );
        assert_eq!(unmatched, [(35, 3)]);

        // No face at 7, 8 (a DLL loss) and 10, 11 (DLL valid).
        let face = seq(&[(true, 7), (false, 2), (true, 1), (false, 2), (true, 32)]);
        let c = Confusion::of(&dll, &ours, &face);
        assert_eq!(
            c,
            Confusion {
                both_valid: 23,
                dll_only: 7,
                dll_only_no_face: 2,
                ours_only: 3,
                neither: 7,
                neither_no_face: 2,
            }
        );
        assert_eq!(c.paired(), 40);
        assert!(close(c.coverage_pct(), 100.0 * 23.0 / 30.0, 1e-12));
        assert!(close(c.agreement_pct(), 100.0 * 30.0 / 40.0, 1e-12));
    }

    /// A loss is printed in the log's image numbers: those of the images
    /// that decoded, past one that did not.
    #[test]
    fn a_loss_spans_the_log_s_image_numbers() {
        let numbered = |image: usize| Frame {
            image,
            ..frame([0.0; 3], Eyes::default())
        };
        let frames = [numbered(0), numbered(2), numbered(3), numbered(4)];
        let event = |start, len| LossEvent {
            start,
            len,
            seen: None,
        };
        assert_eq!(event(1, 2).images(&frames), (2, 4));
        assert_eq!(event(0, 4).images(&frames), (0, 5));
        assert_eq!(event(3, 1).images(&frames), (4, 5));
    }

    /// A run of our invalid poses that only touches a loss, ending where it
    /// starts or starting where it ends, does not see it.
    #[test]
    fn a_run_that_only_touches_a_loss_does_not_see_it() {
        // A loss at 5..8; ours invalid at 2..5 and 8..10.
        let dll = seq(&[(V, 5), (I, 3), (V, 4)]);
        let ours = seq(&[(true, 2), (false, 3), (true, 3), (false, 2), (true, 2)]);
        let (events, unmatched) = loss_events(&dll, &ours);
        assert_eq!(
            events,
            [LossEvent {
                start: 5,
                len: 3,
                seen: None,
            }]
        );
        assert_eq!(unmatched, [(2, 3), (8, 2)]);
    }

    /// Re-acquisitions are the first 10 DLL-valid images after a loss, and
    /// no more than the valid run has: not the start of the session, nor
    /// what follows images with no DLL pose.
    #[test]
    fn reacquisitions_follow_a_loss() {
        let dll = seq(&[
            (M, 2),
            (V, 15),
            (I, 2),
            (V, 14),
            (M, 2),
            (V, 13),
            (M, 1),
            (I, 1),
            (V, 3),
            (I, 2),
            (V, 3),
            (M, 1),
            (V, 4),
        ]);
        let flags = reacquisition(&dll);
        let marked: Vec<usize> = (0..dll.len()).filter(|&i| flags[i]).collect();
        let expected: Vec<usize> = (19..29).chain(50..53).chain(55..58).collect();
        assert_eq!(marked, expected);
    }

    /// A frame of `rotation_deg` (pitch, yaw, roll) on the DLL's side, valid,
    /// with the device's `eyes`; ours valid too.
    fn frame(rotation_deg: [f64; 3], eyes: Eyes) -> Frame {
        let pose = DllHeadPose {
            ts_us: 0,
            position_valid: true,
            rotation_valid: [true; 3],
            position_mm: [0.0, 0.0, 600.0],
            rotation_rad: rotation_deg.map(f64::to_radians),
        };
        Frame {
            image: 0,
            device_ts_us: 0,
            ours: HeadPose::default(),
            face: None,
            cost: StepCost::default(),
            dll: Some(pose),
            eyes,
        }
    }

    #[test]
    fn subsets_follow_the_dll_pose_and_the_device_eyes() {
        let both = Eyes {
            left: true,
            right: true,
        };
        let one = Eyes {
            left: true,
            right: false,
        };
        let member = |f: &Frame| -> Vec<&str> {
            Subset::ALL
                .into_iter()
                .filter(|s| s.contains(f, false))
                .map(Subset::name)
                .collect()
        };
        assert_eq!(member(&frame([0.0; 3], both)), ["ALL", "BOTH"]);
        assert_eq!(member(&frame([0.0; 3], one)), ["ALL"]);
        assert_eq!(member(&frame([0.0; 3], Eyes::default())), ["ALL", "NONE"]);
        assert_eq!(member(&frame([0.0, -20.0, 0.0], one)), ["ALL", "YAW20"]);
        assert_eq!(member(&frame([0.0, 19.9, 0.0], one)), ["ALL"]);
        // Combined: yaw above 15 and pitch above 8 or roll above 10.
        assert_eq!(member(&frame([8.1, 15.1, 0.0], one)), ["ALL", "COMB"]);
        assert_eq!(member(&frame([0.0, -15.1, -10.1], one)), ["ALL", "COMB"]);
        assert_eq!(member(&frame([8.0, 15.1, 10.0], one)), ["ALL"]);
        assert_eq!(member(&frame([9.0, 15.0, 0.0], one)), ["ALL"]);
        assert!(Subset::Reacquired.contains(&frame([0.0; 3], one), true));
        // An image whose DLL pose is invalid, or missing, is in none.
        let mut invalid = frame([0.0, 30.0, 0.0], both);
        invalid.dll = invalid.dll.map(|p| DllHeadPose {
            position_valid: false,
            rotation_valid: [false; 3],
            ..p
        });
        assert!(member(&invalid).is_empty() && !Subset::Reacquired.contains(&invalid, true));
        let missing = Frame {
            dll: None,
            ..invalid
        };
        assert!(member(&missing).is_empty());
    }

    #[test]
    fn the_eyes_are_the_latest_gaze_frame_s() {
        let both = Eyes {
            left: true,
            right: true,
        };
        let left = Eyes {
            left: true,
            right: false,
        };
        let gaze = [(1_000, both), (31_000, left), (161_000, both)];
        assert_eq!(eyes_at(&gaze, 999), Eyes::default());
        assert_eq!(eyes_at(&gaze, 1_000), both);
        assert_eq!(eyes_at(&gaze, 30_999), both);
        assert_eq!(eyes_at(&gaze, 31_000), left);
        // 100 ms after the last frame its eyes no longer count.
        assert_eq!(eyes_at(&gaze, 130_999), left);
        assert_eq!(eyes_at(&gaze, 131_000), Eyes::default());
        assert_eq!(eyes_at(&gaze, 161_001), both);
        assert_eq!(eyes_at(&[], 5), Eyes::default());
    }

    /// The errors of a subset are those of its pairs, the coverage its
    /// valid share; yaw 179 against -179 is 2 degrees off, not 358.
    #[test]
    fn errors_are_per_axis_geodesic_and_positional() {
        let pose = |position_mm: [f64; 3], rotation_deg: [f64; 3]| HeadPose {
            valid: true,
            position_mm,
            rotation_rad: rotation_deg.map(f64::to_radians),
        };
        let pairs = [
            (
                pose([1.0, 2.0, 600.0], [1.0, 179.0, 0.0]),
                pose([0.0, 0.0, 598.0], [0.0, -179.0, 0.0]),
            ),
            (
                pose([0.0, 0.0, 600.0], [-2.0, 0.0, 0.5]),
                pose([3.0, 4.0, 600.0], [0.0, 0.0, 0.0]),
            ),
            (
                pose([0.0, 0.0, 600.0], [0.0, 0.0, 0.0]),
                pose([0.0, 0.0, 600.0], [0.0, 0.0, 0.0]),
            ),
        ];
        let e = Errors::of(&pairs, 4);
        assert_eq!((e.subset, e.both_valid), (4, 3));
        assert!(close(e.coverage_pct(), 75.0, 1e-12));
        let all_close =
            |a: [f64; 3], b: [f64; 3]| a.iter().zip(&b).all(|(x, y)| close(*x, *y, 1e-9));
        assert!(all_close(e.rotation_median_deg, [1.0, 0.0, 0.0]));
        assert!(all_close(e.rotation_signed_median_deg, [0.0, 0.0, 0.0]));
        // p95 of [0, 1, 2] is 1.9; of [0, 0, 2] (yaw) is 1.8.
        assert!(all_close(e.rotation_p95_deg, [1.9, 1.8, 0.45]));
        assert!(all_close(e.position_median_mm, [1.0, 2.0, 0.0]));
        assert!(close(e.position_median_3d_mm, 3.0, 1e-9));
        assert!(close(
            e.geodesic_median_deg,
            geodesic_deg(pairs[1].0.rotation_rad, [0.0; 3]),
            1e-12
        ));
        let none = Errors::of(&[], 0);
        assert!(none.rotation_median_deg[0].is_nan() && none.coverage_pct().is_nan());
    }

    /// The analysis of a session: which images each subset takes, which of
    /// them both call valid, and the errors of those alone, never of the
    /// values our invalid poses hold; the validity table, the losses and
    /// the re-acquisitions.
    #[test]
    fn the_analysis_takes_each_subset_s_images_and_only_our_valid_poses() {
        let both = Eyes {
            left: true,
            right: true,
        };
        let one = Eyes {
            left: true,
            right: false,
        };
        let none = Eyes::default();
        // Ours valid: the DLL's pose moved `dx` mm along x.
        let ours = |dx: f64, f: Frame| Frame {
            ours: HeadPose {
                valid: true,
                position_mm: [dx, 0.0, 600.0],
                rotation_rad: f.dll.map_or([0.0; 3], |p| p.rotation_rad),
            },
            ..f
        };
        // Ours invalid, holding values far from the DLL's.
        let held = |f: Frame| Frame {
            ours: HeadPose {
                valid: false,
                position_mm: [300.0, -200.0, 100.0],
                rotation_rad: [1.0, -1.0, 0.5],
            },
            ..f
        };
        let invalid = |f: Frame| Frame {
            dll: f.dll.map(|p| DllHeadPose {
                position_valid: false,
                rotation_valid: [false; 3],
                ..p
            }),
            ..f
        };
        let missing = |f: Frame| Frame { dll: None, ..f };
        let level = [0.0; 3];
        let mut gated = held(frame(level, one));
        gated.face = Some(Face {
            score: 2.0,
            by_detector: false,
            centroid_edge_px: 3.0,
            nose_tip_edge_px: 10.0,
        });
        let frames = [
            missing(ours(0.0, frame(level, both))),
            ours(1.0, frame(level, both)),
            held(frame(level, none)),
            ours(2.0, frame([0.0, 25.0, 0.0], one)),
            // A loss of two images, then a valid run of three, re-acquired.
            invalid(held(frame(level, none))),
            invalid(held(frame(level, none))),
            ours(4.0, frame(level, both)),
            gated,
            ours(8.0, frame(level, none)),
            missing(held(frame(level, one))),
            ours(16.0, frame(level, one)),
            ours(32.0, frame([9.0, 16.0, 0.0], both)),
        ];
        let analysis = Analysis::of(&frames);
        let reacquired: Vec<usize> = (0..frames.len())
            .filter(|&i| analysis.reacquired[i])
            .collect();
        assert_eq!(reacquired, [6, 7, 8]);
        assert_eq!(
            analysis.confusion,
            Confusion {
                both_valid: 6,
                dll_only: 2,
                dll_only_no_face: 1,
                ours_only: 0,
                neither: 2,
                neither_no_face: 2,
            }
        );
        assert_eq!(
            analysis.events,
            [LossEvent {
                start: 4,
                len: 2,
                seen: Some((0, 0)),
            }]
        );
        assert_eq!(analysis.unmatched_runs, [(2, 1), (7, 1)]);
        // Each subset: its DLL-valid images, those ours is valid on, and
        // the median of our x errors there (1, 2, 4, 8, 16 and 32 mm on
        // the images both call valid; 300 mm on those ours is not).
        for (subset, images, both_valid, median_x) in [
            (Subset::All, 8, 6, 6.0),
            (Subset::BothEyes, 3, 3, 4.0),
            (Subset::NoEyes, 2, 1, 8.0),
            (Subset::LargeYaw, 1, 1, 2.0),
            (Subset::Reacquired, 3, 2, 6.0),
            (Subset::Combined, 1, 1, 32.0),
        ] {
            let e = analysis.errors(subset);
            assert_eq!((e.subset, e.both_valid), (images, both_valid), "{subset:?}");
            assert!(
                close(e.position_median_mm[0], median_x, 1e-12),
                "{subset:?}: {e:?}"
            );
            assert!(e.rotation_p95_deg.iter().all(|&a| a < 1e-9), "{subset:?}");
        }
        assert!(close(
            analysis.metric("coverage_pct").expect("a metric"),
            75.0,
            1e-12
        ));
        assert!(close(
            analysis.metric("agreement_pct").expect("a metric"),
            80.0,
            1e-12
        ));
        assert_eq!(analysis.metric("loss_events_found"), Some(1.0));
    }

    /// Each image takes the DLL's pose of its time less the clock offset
    /// (of two that share a time, the later), and the device's eyes at it;
    /// the report counts the images with a pose, the poses of no image and
    /// the poses that share a time.
    #[test]
    fn the_dll_poses_pair_with_the_images_at_the_clock_offset() {
        let k = 9_290_110_919;
        let image = |t_us: i64| Frame {
            device_ts_us: k + t_us,
            dll: None,
            ..frame([0.0; 3], Eyes::default())
        };
        let mut frames = [image(100), image(130), image(160), image(190)];
        let pose = |ts_us: i64, x: f64| DllHeadPose {
            ts_us,
            position_valid: true,
            rotation_valid: [true; 3],
            position_mm: [x, 0.0, 600.0],
            rotation_rad: [0.0; 3],
        };
        let poses = [
            pose(100, 1.0),
            // None for the image at 130; two at 160, the later of which it
            // takes; one of no image, and one whose time overflows.
            pose(160, 2.0),
            pose(160, 3.0),
            pose(190, 4.0),
            pose(175, 5.0),
            pose(i64::MAX, 6.0),
        ];
        let both = Eyes {
            left: true,
            right: true,
        };
        let left = Eyes {
            left: true,
            right: false,
        };
        let gaze = [(k + 90, both), (k + 150, left)];
        assert_eq!(
            pair(&mut frames, &poses, k, &gaze),
            Pairing {
                paired: 3,
                unpaired_poses: 2,
                duplicate_poses: 1,
            }
        );
        let taken: Vec<Option<f64>> = frames
            .iter()
            .map(|f| f.dll.map(|p| p.position_mm[0]))
            .collect();
        assert_eq!(taken, [Some(1.0), None, Some(3.0), Some(4.0)]);
        let eyes: Vec<Eyes> = frames.iter().map(|f| f.eyes).collect();
        assert_eq!(eyes, [both, both, left, left]);
    }

    /// A TLV entry: its type, its length (BE u32) and its value.
    fn tlv(typ: u8, value: &[u8]) -> Vec<u8> {
        let len = u32::try_from(value.len()).expect("a short value");
        [&[typ][..], &len.to_be_bytes(), value].concat()
    }

    /// A message of stream `id` whose TLVs are `body`, as the device sends
    /// one.
    fn stream_message(id: u32, body: &[u8]) -> Vec<u8> {
        let len = u32::try_from(STREAM_TLV_OFFSET + body.len()).expect("a short message");
        let mut msg = [
            &[1, 0, 0, 0][..],
            &len.to_le_bytes(),
            &MARKER_STREAM.to_be_bytes(),
            &[0; 8],
            &id.to_be_bytes(),
        ]
        .concat();
        msg.resize(STREAM_TLV_OFFSET, 0);
        msg.extend_from_slice(body);
        msg
    }

    /// A 0x50e message of a `side` x `side` image of `pixels` at device
    /// time `t_us`, as the device lays one out.
    fn image_message(t_us: u64, side: u32, pixels: &[u8]) -> Vec<u8> {
        let keyed = |key: u32, entry: Vec<u8>| {
            [
                tlv(5, &KEY_FIELD_ID.to_be_bytes()),
                tlv(2, &key.to_be_bytes()),
                entry,
            ]
            .concat()
        };
        let count = u32::try_from(pixels.len()).expect("a short image");
        let body = [
            keyed(1, tlv(6, &t_us.to_be_bytes())),
            keyed(2, tlv(2, &8u32.to_be_bytes())),
            keyed(3, tlv(2, &side.to_be_bytes())),
            keyed(4, tlv(2, &side.to_be_bytes())),
            keyed(5, tlv(2, &side.to_be_bytes())),
            keyed(6, tlv(0x15, &[&count.to_be_bytes()[..], pixels].concat())),
        ]
        .concat();
        stream_message(STREAM_ID_IMAGE, &body)
    }

    /// A replay steps every image of the log once, in log order, at its
    /// device time, each with the model runs of its own step, and passes
    /// over the other messages; an image that does not decode is counted
    /// and keeps its number. Black images make invalid poses: the first
    /// runs the landmark model in the tracker's starting crop and then the
    /// detector; with the face lost, the others run the detector alone.
    #[test]
    fn a_replay_steps_every_image_once_with_its_own_model_runs() {
        let display = DisplayFrame::new(&AREA_A).expect("area A fixes a frame");
        let black = vec![0u8; 280 * 280];
        let log = [
            image_message(1_000_000, 280, &black),
            stream_message(0x500, &[]),
            // Short of its pixels: it does not decode, and keeps its number.
            image_message(1_015_104, 280, &black[..1000]),
            image_message(1_030_208, 280, &black),
            image_message(1_060_416, 280, &black),
        ];
        let replay = replay(&log, &display).expect("the models load");
        let images: Vec<(usize, i64)> = replay
            .frames
            .iter()
            .map(|f| (f.image, f.device_ts_us))
            .collect();
        assert_eq!(images, [(0, 1_000_000), (2, 1_030_208), (3, 1_060_416)]);
        for (f, runs) in replay.frames.iter().zip([(1, 1), (0, 1), (0, 1)]) {
            assert!(
                !f.ours.valid && f.face.is_none() && f.dll.is_none(),
                "{f:?}"
            );
            assert_eq!((f.cost.landmark_runs, f.cost.detector_runs), runs, "{f:?}");
        }
        assert_eq!((replay.undecoded, &replay.tracker_errors), (1, &(0, None)));
        assert_eq!(replay.fingerprint, fingerprint(&HeadParams::FITTED));
    }

    /// The tracker-frame and display-frame points of `points` through area
    /// A's frame.
    fn pairs_of(frame: &DisplayFrame, points: &[[f64; 3]]) -> Vec<OriginPair> {
        points.iter().map(|&p| (p, frame.to_display(p))).collect()
    }

    /// Gaze origins of a head moving about 600 mm out: both eyes, a few
    /// centimetres of movement.
    fn origins() -> Vec<[f64; 3]> {
        (0..200)
            .flat_map(|i| {
                let t = f64::from(i) * 0.1;
                let head = [
                    40.0 * t.sin(),
                    20.0 * (0.7 * t).cos(),
                    600.0 + 30.0 * (0.3 * t).sin(),
                ];
                [add(head, [-31.0, 0.0, 0.0]), add(head, [31.0, 0.0, 0.0])]
            })
            .collect()
    }

    /// The display frame a rigid fit of `pairs` gives, which must be area
    /// A's: its axes to 1e-9 and its centre to 1e-6 mm.
    fn assert_fits_area_a(pairs: &[OriginPair]) -> DisplayFrame {
        let frame_a = DisplayFrame::new(&AREA_A).expect("area A fixes a frame");
        let fitted = RigidMap::fit(pairs)
            .ok()
            .and_then(|map| map.frame())
            .expect("a frame");
        for (row, want) in fitted.rotation().iter().zip(frame_a.rotation()) {
            assert!(
                row.iter().zip(&want).all(|(a, b)| close(*a, *b, 1e-9)),
                "{row:?} {want:?}"
            );
        }
        assert!(
            fitted
                .centre()
                .iter()
                .zip(frame_a.centre())
                .all(|(a, b)| close(*a, b, 1e-6))
        );
        fitted
    }

    /// A rigid fit of the gaze origins area A maps gives area A's frame,
    /// and maps them to a micrometre's thousandth; area B's do not fit it.
    #[test]
    fn a_rigid_fit_of_the_origins_gives_the_display_frame() {
        let frame_a = DisplayFrame::new(&AREA_A).expect("area A fixes a frame");
        let pairs = pairs_of(&frame_a, &origins());
        let fitted = assert_fits_area_a(&pairs);
        let check = OriginCheck::of(&fitted, &pairs).expect("pairs");
        assert_eq!(check.pairs, 400);
        assert!(check.fits() && check.max_mm < 1e-6, "{check:?}");

        // The same origins under area A moved 0.85 mm sideways, as the
        // Linux area is: not the device's frame.
        let moved = DisplayArea {
            top_left_mm: add(AREA_A.top_left_mm, [0.85, 0.0, 0.0]),
            top_right_mm: add(AREA_A.top_right_mm, [0.85, 0.0, 0.0]),
            bottom_left_mm: add(AREA_A.bottom_left_mm, [0.85, 0.0, 0.0]),
        };
        let frame_b = DisplayFrame::new(&moved).expect("a frame");
        let check = OriginCheck::of(&frame_b, &pairs).expect("pairs");
        assert!(
            !check.fits() && close(check.max_mm, 0.85, 1e-9),
            "{check:?}"
        );
        assert!(RigidMap::fit(&pairs[..2]).is_err());
        assert!(OriginCheck::of(&frame_a, &[]).is_none());
        let mut broken = pairs;
        broken[7].1[2] = f64::NAN;
        assert!(RigidMap::fit(&broken).is_err());
    }

    /// The gaze origins of a head held still about 600 mm out: both eyes,
    /// the head swaying by a few tenths of a millimetre, 0.229 mm (rms)
    /// across the line between the eyes.
    fn still_origins() -> Vec<[f64; 3]> {
        (0..300)
            .flat_map(|i| {
                let t = f64::from(i) * 0.1;
                let head = [
                    0.3 * (1.3 * t).sin(),
                    150.0 + 0.2 * (0.7 * t).cos(),
                    600.0 + 0.25 * (0.45 * t + 1.0).sin(),
                ];
                [add(head, [-31.0, 0.0, 0.0]), add(head, [31.0, 0.0, 0.0])]
            })
            .collect()
    }

    /// The origins of a head held still lie near one line, which the fit
    /// still turns the frame about as the device did; the tracker's kabsch,
    /// 200 power steps, stops some 40 degrees short on these.
    #[test]
    fn a_rigid_fit_fixes_the_frame_of_a_head_held_still() {
        let frame_a = DisplayFrame::new(&AREA_A).expect("area A fixes a frame");
        let pairs = pairs_of(&frame_a, &still_origins());
        let map = RigidMap::fit(&pairs).expect("a fit");
        // NumPy's eigvalsh of the same points: 0.228 730 326 233 mm.
        assert!(
            close(map.spread_mm, 0.228_730_326, 1e-9),
            "{}",
            map.spread_mm
        );
        let fitted = assert_fits_area_a(&pairs);
        let check = OriginCheck::of(&fitted, &pairs).expect("pairs");
        assert!(check.fits() && check.max_mm < 1e-6, "{check:?}");
    }

    /// Origins that lie along one line leave the turn about it open, and
    /// fix no frame: a head that never moves, and origins `d` mm off the
    /// line on either side, just within and just past the 0.1 mm a fit
    /// takes.
    #[test]
    fn origins_along_one_line_fix_no_frame() {
        let frame_a = DisplayFrame::new(&AREA_A).expect("area A fixes a frame");
        let frozen: Vec<[f64; 3]> = (0..100)
            .flat_map(|_| [[-31.0, 150.0, 600.0], [31.0, 150.0, 600.0]])
            .collect();
        let pairs = pairs_of(&frame_a, &frozen);
        let error = RigidMap::fit(&pairs).expect_err("a line");
        assert!(error.to_string().contains("off one line"), "{error}");
        let error = display_frame(AreaChoice::Auto, &LogScan::default(), &pairs)
            .expect_err("nothing to take the area from");
        assert!(
            format!("{error:#}").contains("give --display-area"),
            "{error:#}"
        );

        let off_line = |d: f64| -> Vec<[f64; 3]> {
            [-31.0, -10.0, 10.0, 31.0]
                .into_iter()
                .flat_map(|x| [[x, 150.0 + d, 600.0], [x, 150.0 - d, 600.0]])
                .collect()
        };
        assert!(RigidMap::fit(&pairs_of(&frame_a, &off_line(0.0999))).is_err());
        let pairs = pairs_of(&frame_a, &off_line(0.1001));
        let map = RigidMap::fit(&pairs).expect("a fit");
        assert!(close(map.spread_mm, 0.1001, 1e-9), "{}", map.spread_mm);
        assert_fits_area_a(&pairs);
    }

    /// The fit, or an area of the log's, that misses the gaze origins stops
    /// the replay, each with its own reason; an area given on the command
    /// line comes with a note; one that maps them passes.
    #[test]
    fn an_area_that_misses_the_gaze_origins_is_judged_by_its_source() {
        let check = |max_mm| OriginCheck {
            pairs: 10,
            max_mm,
            rms_mm: max_mm / 2.0,
        };
        let fitted = AreaSource::Fitted { spread_mm: 5.0 };
        let sources = [
            fitted,
            AreaSource::Notified(1),
            AreaSource::Written(2),
            AreaSource::Given,
        ];
        for source in sources {
            assert_eq!(judge_area(source, Some(&check(0.000_2))).ok(), Some(None));
            assert_eq!(judge_area(source, None).ok(), Some(None));
        }
        let missed = check(0.85);
        let error = judge_area(fitted, Some(&missed)).expect_err("a fit that misses");
        assert!(
            error.to_string().contains("no one display frame"),
            "{error}"
        );
        for source in [AreaSource::Notified(1), AreaSource::Written(2)] {
            let error = judge_area(source, Some(&missed)).expect_err("an area that misses");
            assert!(error.to_string().contains("not the session's"), "{error}");
        }
        let note = judge_area(AreaSource::Given, Some(&missed)).expect("a note");
        assert!(note.is_some_and(|n| n.contains("not the area the device held")));
    }

    /// The log's display-area messages give the area, a notification
    /// before a write; areas that disagree stop the replay.
    #[test]
    fn the_log_s_area_messages_give_the_area() {
        // As the device sends a 1450: the prefix 01 00 00 00 and the whole
        // length, the notification marker.
        let notification = |area: &DisplayArea| {
            let payload = TlvWriter::new()
                .point_mm(area.top_left_mm)
                .point_mm(area.top_right_mm)
                .point_mm(area.bottom_left_mm)
                .finish();
            let mut msg = chunk_command(notify::DISPLAY_AREA, 0, &payload).swap_remove(0);
            let len = u32::try_from(msg.len()).expect("a short message");
            msg[..4].copy_from_slice(&[1, 0, 0, 0]);
            msg[4..8].copy_from_slice(&len.to_le_bytes());
            msg[8..12].copy_from_slice(&MARKER_NOTIFICATION.to_be_bytes());
            msg
        };
        // As the host sends a 1440: one write, prefix 00 00 00 00.
        let write = |area: &DisplayArea| {
            chunk_command(
                cmd::DISPLAY_AREA_SET,
                14,
                &display_area_set_payload(area, 1),
            )
            .swap_remove(0)
        };
        let other = DisplayArea {
            top_left_mm: [-300.0, 330.0, 110.0],
            top_right_mm: [300.0, 330.0, 110.0],
            bottom_left_mm: [-300.0, 10.0, -3.0],
        };
        let scan = scan_log(&[write(&other), notification(&AREA_A), notification(&AREA_A)]);
        assert_eq!(scan.notified, [AREA_A, AREA_A]);
        assert_eq!(scan.written, [other]);
        let (frame, source) = display_frame(AreaChoice::Auto, &scan, &[]).expect("an area");
        assert_eq!(source, AreaSource::Notified(2));
        assert_eq!(Some(frame), DisplayFrame::new(&AREA_A));

        let scan = scan_log(&[write(&other)]);
        let (frame, source) = display_frame(AreaChoice::Auto, &scan, &[]).expect("an area");
        assert_eq!(source, AreaSource::Written(1));
        assert_eq!(Some(frame), DisplayFrame::new(&other));
        // The command line wins over the log.
        let (frame, source) =
            display_frame(AreaChoice::Given(AREA_A), &scan, &[]).expect("an area");
        assert_eq!(
            (Some(frame), source),
            (DisplayFrame::new(&AREA_A), AreaSource::Given)
        );

        let changed = scan_log(&[notification(&AREA_A), notification(&other)]);
        assert!(display_frame(AreaChoice::Auto, &changed, &[]).is_err());
        // No message and no origins: nothing to fit.
        assert!(display_frame(AreaChoice::Auto, &LogScan::default(), &[]).is_err());
    }

    #[test]
    fn display_area_arguments_parse() {
        assert_eq!(parse_area_choice("auto").ok(), Some(AreaChoice::Auto));
        let given = parse_area_choice(
            "-315.7917175292969,324.4114685058594,111.23909759521484,\
             317.7962341308594,324.4114685058594,111.23909759521484,\
             -315.7917175292969,10.267330169677734,-3.100020170211792",
        );
        assert_eq!(given.ok(), Some(AreaChoice::Given(AREA_A)));
        assert!(parse_area_choice("1,2,3,4,5,6,7,8").is_err());
        assert!(parse_area_choice("1,2,3,4,5,6,7,8,x").is_err());
        // An area of no width fixes no frame.
        assert!(parse_area_choice("0,0,0,0,0,0,0,-300,0").is_err());
    }

    #[test]
    fn the_csv_has_a_column_for_every_field() {
        assert_eq!(
            CSV_HEADER,
            "image,device_ts_us,dll_ts_us,dll_valid,\
             dll_pos_x_mm,dll_pos_y_mm,dll_pos_z_mm,dll_rot_x_rad,dll_rot_y_rad,dll_rot_z_rad,\
             ours_valid,ours_pos_x_mm,ours_pos_y_mm,ours_pos_z_mm,\
             ours_rot_x_rad,ours_rot_y_rad,ours_rot_z_rad,\
             face,face_score,by_detector,centroid_edge_px,nose_tip_edge_px,\
             both_eyes,no_eyes,yaw20,reacq,comb,step_us,landmark_runs,detector_runs"
        );
        let mut f = frame([1.0, 25.0, 0.0], Eyes::default());
        f.image = 7;
        f.device_ts_us = 1_000_123;
        f.ours = HeadPose {
            valid: true,
            position_mm: [1.5, -2.25, 601.0],
            rotation_rad: [0.1, -0.2, 0.3],
        };
        f.face = Some(Face {
            score: 3.5,
            by_detector: true,
            centroid_edge_px: 40.25,
            nose_tip_edge_px: -1.5,
        });
        f.cost = StepCost {
            micros: 6012,
            landmark_runs: 2,
            detector_runs: 1,
        };
        let row = csv_row(&f, true).expect("a DLL pose");
        assert_eq!(row.split(',').count(), CSV_HEADER.split(',').count());
        let rotation = [1.0f64, 25.0, 0.0].map(f64::to_radians);
        assert_eq!(
            row,
            format!(
                "7,1000123,0,1,0,0,600,{},{},0,1,1.5,-2.25,601,0.1,-0.2,0.3,1,3.5,1,40.25,-1.5,\
                 0,1,1,1,0,6012,2,1",
                rotation[0], rotation[1]
            )
        );
        // Without a face its columns are empty; without a DLL pose, no row.
        f.face = None;
        let row = csv_row(&f, false).expect("a DLL pose");
        assert!(row.contains(",0,,,,,"), "{row}");
        assert_eq!(row.split(',').count(), CSV_HEADER.split(',').count());
        f.dll = None;
        assert_eq!(csv_row(&f, false), None);
    }

    /// The committed gates file reads, and every gate this tool checks
    /// bounds a metric it computes; the others are the lag and jitter ones.
    #[test]
    fn the_committed_gates_file_matches_the_metrics() {
        let gates = Gates::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/headpose/gates.json"
        )))
        .expect("the committed gates file");
        assert_eq!(
            gates
                .sessions
                .iter()
                .map(|s| (s.name.as_str(), s.clock_offset_us))
                .collect::<Vec<_>>(),
            [
                ("s1", 9_290_110_919),
                ("s2", 10_701_480_021),
                ("s3", 12_670_044_474)
            ]
        );
        assert_eq!(gates.gates.len(), 18);
        let analysis = Analysis::of(&[]);
        for gate in &gates.gates {
            match gate.checker {
                Checker::CompareDll => assert!(
                    analysis.metric(&gate.metric).is_some(),
                    "{}: no metric {}",
                    gate.id,
                    gate.metric
                ),
                Checker::Python => assert!(
                    [
                        "rotation_lag_max_ms",
                        "position_lag_max_ms",
                        "rotation_rest_jitter_ratio",
                        "position_rest_jitter_ratio"
                    ]
                    .contains(&gate.metric.as_str()),
                    "{}: {}",
                    gate.id,
                    gate.metric
                ),
            }
        }
    }

    /// Each gate metric reads its own number: of ALL, or of COMB for the
    /// gates on combined turns.
    #[test]
    fn the_gate_metrics_read_their_numbers() {
        let errors = |base: f64| Errors {
            subset: 10,
            both_valid: 9,
            rotation_median_deg: [base + 1.0, base + 2.0, base + 3.0],
            rotation_p95_deg: [base + 4.0, base + 5.0, base + 6.0],
            rotation_signed_median_deg: [base + 7.0, base + 8.0, base + 9.0],
            geodesic_median_deg: base + 10.0,
            geodesic_p95_deg: base + 11.0,
            position_median_mm: [base + 12.0, base + 13.0, base + 14.0],
            position_p95_mm: [base + 15.0, base + 16.0, base + 17.0],
            position_median_3d_mm: base + 18.0,
        };
        let analysis = Analysis {
            confusion: Confusion {
                both_valid: 90,
                dll_only: 10,
                dll_only_no_face: 4,
                ours_only: 5,
                neither: 15,
                neither_no_face: 1,
            },
            events: vec![
                LossEvent {
                    start: 3,
                    len: 2,
                    seen: Some((0, 1)),
                },
                LossEvent {
                    start: 9,
                    len: 1,
                    seen: None,
                },
                LossEvent {
                    start: 20,
                    len: 4,
                    seen: Some((-1, 0)),
                },
            ],
            unmatched_runs: Vec::new(),
            errors: Subset::ALL.map(|s| {
                let base = match s {
                    Subset::All => 0.0,
                    Subset::Combined => 100.0,
                    _ => 50.0,
                };
                (s, errors(base))
            }),
            reacquired: Vec::new(),
        };
        for (name, value) in [
            ("coverage_pct", 90.0),
            ("agreement_pct", 100.0 * 105.0 / 120.0),
            ("loss_events_found", 2.0),
            ("rotation_median_abs_x_deg", 1.0),
            ("rotation_median_abs_y_deg", 2.0),
            ("rotation_median_abs_z_deg", 3.0),
            ("rotation_median_signed_x_deg", 7.0),
            ("rotation_geodesic_p95_deg", 11.0),
            ("combined_rotation_p95_abs_x_deg", 104.0),
            ("combined_rotation_geodesic_p95_deg", 111.0),
            ("position_median_abs_x_mm", 12.0),
            ("position_median_abs_y_mm", 13.0),
            ("position_median_abs_z_mm", 14.0),
            ("position_p95_abs_z_mm", 17.0),
        ] {
            let got = analysis.metric(name).expect(name);
            assert!(close(got, value, 1e-12), "{name}: {got} != {value}");
        }
        assert_eq!(analysis.metric("rotation_lag_max_ms"), None);
    }

    /// The fingerprint follows every constant.
    #[test]
    fn the_fingerprint_changes_with_the_constants() {
        let fitted = HeadParams::FITTED;
        let mut other = fitted;
        other.reset_gap_s = 1.000_000_000_000_001;
        assert_eq!(fingerprint(&fitted), fingerprint(&HeadParams::FITTED));
        assert_ne!(fingerprint(&fitted), fingerprint(&other));
    }
}
