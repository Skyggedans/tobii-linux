use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

use crate::decode::{field_value, head_point, mean_keys, LiveField, TrackingFrame};
use crate::math::{
    angles_from_translation, calibrate_origin, calibrate_scalar_origin, choose_pose_hypothesis,
    dot3, norm3, normalize_angle_deg, scale_matrix, solve_3x3,
};

pub(crate) const OPENTRACK_HEAD_SCALE: f64 = 1000.0;

pub(crate) const DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE: [f64; 3] = [40.0, 60.0, 1.0];

pub(crate) const DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE: [f64; 3] = [2.0, 1.0, 2.0];

pub(crate) const DEFAULT_OPENTRACK_ANGLE_OCC: usize = 7;

pub(crate) const OPENTRACK_MAX_HEAD_ANGLE_DEG: f64 = 45.0;

pub(crate) const DEFAULT_OPENTRACK_ORIGIN_SAMPLES: usize = 30;

pub(crate) const DEFAULT_OPENTRACK_SMOOTHING: f64 = 0.35;

pub(crate) const DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG: f64 = 0.35;

pub(crate) const DEFAULT_OPENTRACK_ROTATION_COMP: [[f64; 3]; 3] = [[0.0; 3]; 3];

pub(crate) const DEFAULT_OPENTRACK_TRANSLATION_SCALE: [f64; 3] = [1.0, 1.0, 1.0];

pub(crate) const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP: [[f64; 3]; 3] = [[0.0; 3]; 3];

pub(crate) const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE: f64 = 1.0;

pub(crate) const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM: f64 = 0.0;

pub(crate) const OPENTRACK_COUPLING_SWITCH_FRAMES: usize = 4;

pub(crate) const OPENTRACK_COUPLING_SWITCH_MARGIN: f64 = 0.20;

// Opt-in online decoupling of rotation-induced translation
// (--opentrack-auto-decouple, off by default). The rotation->translation lever
// arm is learned live by recursive least squares and translation -= comp*angles
// is applied so a pure head turn nets ~zero translation.
//
// NOTE: this only works if the angle signal is a clean rotation measurement.
// On this device the angle telemetry is dominated by translation (a lateral
// shift fakes more "yaw" than a real turn), so the subtraction also corrupts
// genuine translation -- hence it is opt-in, not the default.
// Learned head-rotation model: maps the translation-invariant (centroid-
// centred) positions of head landmarks occ 0..9 of field 0x00031f41 to head
// rotation. Trained by least squares against Tobii Stream Engine ground truth
// (sessions 2+3, 11934 frames); validated cross-session: roll ~3.5deg, yaw
// ~5.3deg (both generalise), pitch ~9deg (weak, does not transfer well).
// Input: 30 centred coords (occ0 x,y,z .. occ9 x,y,z) + bias. Columns are
// [pitch, yaw, roll] in radians per device unit.
const HEAD_ROT_MODEL: [[f64; 3]; 31] = [
    [-2.12985456e+00, -2.38572003e+00, 9.55928008e-01],
    [1.18586057e-01, 5.53841562e-03, 8.98765202e-04],
    [9.13773401e-02, -1.78883689e-03, -4.32098753e-03],
    [2.12968468e+00, 2.38566516e+00, -9.55911135e-01],
    [-1.42725874e-01, -4.77259199e-03, 7.42218880e-04],
    [-4.53771958e-02, 3.34149717e-03, 4.62164279e-03],
    [-3.41912661e-03, -5.18894976e-03, 2.81220605e-03],
    [-1.18569896e-03, -6.92863288e-03, 2.86735564e-03],
    [8.10672947e-04, -3.50993387e-03, 3.17978155e-03],
    [7.43591838e-07, 1.12094381e-07, -1.01573344e-07],
    [4.28470166e-03, 5.38870189e-03, 1.57597919e-04],
    [-1.17689661e-02, -1.48055616e-02, -4.32420573e-04],
    [1.63783559e-04, 4.43278133e-05, -1.29065086e-05],
    [1.03758545e-05, 6.75623169e-05, -2.14991982e-05],
    [7.93585111e-05, 2.85798415e-04, -2.76731413e-04],
    [-1.84981362e+01, 2.47395747e+00, -2.06032864e+00],
    [-1.54553070e-02, -5.57211532e-03, 4.33653916e-03],
    [-3.31372445e-02, -2.33707915e-04, 8.70321342e-03],
    [1.84983783e+01, -2.47388931e+00, 2.06029072e+00],
    [2.59134491e-02, 5.39192632e-03, -7.06917280e-03],
    [2.58485095e-02, -1.47389614e-03, -6.93496271e-03],
    [3.41964410e-03, 5.18891774e-03, -2.81226661e-03],
    [4.18599572e-04, 5.69092774e-03, -2.15887371e-03],
    [2.29563109e-04, 5.16340762e-03, -3.93403577e-03],
    [-1.20750315e-06, -8.32250555e-07, 5.70208289e-07],
    [1.02067617e-02, -4.82377132e-03, 3.04714129e-04],
    [-2.80436044e-02, 1.32553022e-02, -8.39676744e-04],
    [-2.36071579e-04, -5.68495266e-05, 3.35462533e-05],
    [-5.31147603e-05, 1.95805790e-05, -5.76490850e-05],
    [-1.81345494e-05, -2.34129999e-04, 2.34219671e-04],
    [-2.71108232e-03, 2.77695447e-04, -2.64032690e-04], // bias
];

const OPENTRACK_DECOUPLE_FORGET: f64 = 0.997;
const OPENTRACK_DECOUPLE_RIDGE: f64 = 0.6;
const OPENTRACK_DECOUPLE_MIN_WEIGHT: f64 = 90.0;
const OPENTRACK_DECOUPLE_ANGLE_GATE_DEG: f64 = 1.0;
const OPENTRACK_DECOUPLE_BLEND: f64 = 0.1;

pub(crate) const DEFAULT_OPENTRACK_ANGLE_MAP: [AngleComponent; 3] = [
    AngleComponent::new(0, 1.0),
    AngleComponent::new(1, -1.0),
    AngleComponent::disabled(),
];

#[derive(Clone, Copy)]
pub(crate) enum AngleSource {
    Gaze,
    Head,
    /// Learned translation-invariant rigid-rotation model (occ 0-9 landmarks).
    Model,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CouplingMode {
    Rotation,
    Translation,
    Hybrid,
    Auto,
}

#[derive(Clone, Copy)]
pub(crate) struct AngleComponent {
    pub(crate) component: usize,
    pub(crate) sign: f64,
}

impl AngleComponent {
    pub(crate) const fn new(component: usize, sign: f64) -> Self {
        Self { component, sign }
    }

    pub(crate) const fn disabled() -> Self {
        Self {
            component: 0,
            sign: 0.0,
        }
    }

    pub(crate) fn is_disabled(self) -> bool {
        self.sign == 0.0
    }
}

pub(crate) struct OpentrackUdp {
    pub(crate) socket: UdpSocket,
    pub(crate) target: SocketAddr,
    pub(crate) send_translation: bool,
    pub(crate) angle_source: AngleSource,
    pub(crate) angle_occurrence: Option<usize>,
    pub(crate) angle_scale: [f64; 3],
    pub(crate) angle_map: [AngleComponent; 3],
    pub(crate) origin_samples: usize,
    pub(crate) angle_points: Option<(usize, usize)>,
    pub(crate) roll_points: Option<(usize, usize)>,
    pub(crate) origin: Option<[f64; 3]>,
    pub(crate) origin_sum: [f64; 3],
    pub(crate) origin_count: usize,
    pub(crate) angle_origin: Option<[f64; 3]>,
    pub(crate) angle_origin_sum: [f64; 3],
    pub(crate) angle_origin_count: usize,
    pub(crate) roll_origin: Option<f64>,
    pub(crate) roll_origin_sum: f64,
    pub(crate) roll_origin_count: usize,
    pub(crate) smoothing_alpha: f64,
    pub(crate) angle_deadzone: f64,
    pub(crate) rotation_comp: [[f64; 3]; 3],
    pub(crate) translation_scale: [f64; 3],
    pub(crate) angle_translation_comp: [[f64; 3]; 3],
    pub(crate) angle_translation_comp_scale: f64,
    pub(crate) angle_translation_deadzone: f64,
    pub(crate) coupling_mode: CouplingMode,
    pub(crate) coupling_state: CouplingMode,
    pub(crate) coupling_candidate: CouplingMode,
    pub(crate) coupling_candidate_count: usize,
    pub(crate) last_pose: Option<[f64; 6]>,
    // Online rotation->translation decoupling (active when rotation_comp is zero).
    pub(crate) auto_decouple: bool,
    pub(crate) decouple_ata: [[f64; 3]; 3],
    pub(crate) decouple_atb: [[f64; 3]; 3],
    pub(crate) decouple_weight: f64,
    pub(crate) learned_rotation_comp: [[f64; 3]; 3],
    // Last valid relative angle, held across head-angle dropouts.
    pub(crate) last_angles: Option<[f64; 3]>,
}

impl OpentrackUdp {
    pub(crate) fn connect(
        host: &str,
        port: u16,
        send_translation: bool,
        angle_source: AngleSource,
        angle_occurrence: Option<usize>,
        angle_scale: [f64; 3],
        angle_map: [AngleComponent; 3],
        origin_samples: usize,
        angle_points: Option<(usize, usize)>,
        roll_points: Option<(usize, usize)>,
        smoothing_alpha: f64,
        angle_deadzone: f64,
        rotation_comp: [[f64; 3]; 3],
        translation_scale: [f64; 3],
        angle_translation_comp: [[f64; 3]; 3],
        angle_translation_comp_scale: f64,
        angle_translation_deadzone: f64,
        coupling_mode: CouplingMode,
        auto_decouple: bool,
    ) -> Result<Self> {
        let target = (host, port)
            .to_socket_addrs()
            .with_context(|| format!("failed to resolve opentrack target {host}:{port}"))?
            .next()
            .with_context(|| format!("opentrack target {host}:{port} resolved to no addresses"))?;
        let bind_addr = if target.is_ipv4() {
            "0.0.0.0:0"
        } else {
            "[::]:0"
        };
        let socket = UdpSocket::bind(bind_addr).context("failed to create opentrack UDP socket")?;

        let source = match angle_source {
            AngleSource::Model => "model (learned, translation-invariant)",
            AngleSource::Head => "head (occ-7)",
            AngleSource::Gaze => "gaze",
        };
        let decouple = auto_decouple && is_zero_matrix(rotation_comp);
        println!(
            "Sending OpenTrack UDP frames to {target}; angle source: {source}; rotation->translation auto-decouple: {}",
            if decouple { "on" } else { "off" }
        );

        Ok(Self {
            socket,
            target,
            send_translation,
            angle_source,
            angle_occurrence,
            angle_scale,
            angle_map,
            origin_samples,
            angle_points,
            roll_points,
            origin: None,
            origin_sum: [0.0; 3],
            origin_count: 0,
            angle_origin: None,
            angle_origin_sum: [0.0; 3],
            angle_origin_count: 0,
            roll_origin: None,
            roll_origin_sum: 0.0,
            roll_origin_count: 0,
            smoothing_alpha,
            angle_deadzone,
            rotation_comp,
            translation_scale,
            angle_translation_comp,
            angle_translation_comp_scale,
            angle_translation_deadzone,
            coupling_mode,
            coupling_state: CouplingMode::Rotation,
            coupling_candidate: CouplingMode::Rotation,
            coupling_candidate_count: 0,
            last_pose: None,
            auto_decouple: auto_decouple && is_zero_matrix(rotation_comp),
            decouple_ata: [[0.0; 3]; 3],
            decouple_atb: [[0.0; 3]; 3],
            decouple_weight: 0.0,
            learned_rotation_comp: [[0.0; 3]; 3],
            last_angles: None,
        })
    }

    /// Returns the rotation->translation compensation matrix to apply this
    /// frame. With an explicit `--opentrack-rotation-comp` it is constant;
    /// otherwise the lever arm is learned online from rotation-excited frames
    /// so that pure head turns produce ~zero translation ("rotate in place").
    fn rotation_compensation(
        &mut self,
        translation: [f64; 3],
        angles: [f64; 3],
        learn: bool,
    ) -> [[f64; 3]; 3] {
        if !self.auto_decouple {
            return self.rotation_comp;
        }

        if learn && norm3(angles) >= OPENTRACK_DECOUPLE_ANGLE_GATE_DEG {
            for r in 0..3 {
                for c in 0..3 {
                    self.decouple_ata[r][c] =
                        self.decouple_ata[r][c] * OPENTRACK_DECOUPLE_FORGET + angles[r] * angles[c];
                }
            }
            for axis in 0..3 {
                for c in 0..3 {
                    self.decouple_atb[axis][c] = self.decouple_atb[axis][c]
                        * OPENTRACK_DECOUPLE_FORGET
                        + translation[axis] * angles[c];
                }
            }
            self.decouple_weight = self.decouple_weight * OPENTRACK_DECOUPLE_FORGET + 1.0;

            if self.decouple_weight >= OPENTRACK_DECOUPLE_MIN_WEIGHT {
                // Ridge keeps unexcited axes (e.g. roll with no roll motion)
                // from inventing a compensation; it shrinks toward zero there.
                let ridge = OPENTRACK_DECOUPLE_RIDGE * self.decouple_weight;
                let mut ata = self.decouple_ata;
                for i in 0..3 {
                    ata[i][i] += ridge;
                }
                for axis in 0..3 {
                    if let Some(row) = solve_3x3(ata, self.decouple_atb[axis]) {
                        for c in 0..3 {
                            self.learned_rotation_comp[axis][c] = self.learned_rotation_comp[axis]
                                [c]
                                * (1.0 - OPENTRACK_DECOUPLE_BLEND)
                                + row[c] * OPENTRACK_DECOUPLE_BLEND;
                        }
                    }
                }
            }
        }

        self.learned_rotation_comp
    }

    pub(crate) fn send_frame(
        &mut self,
        frame: &TrackingFrame,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
    ) -> Result<Option<[f64; 6]>> {
        let Some(head) = frame.head_xyz() else {
            return Ok(None);
        };
        let has_user = frame.gaze_valid;
        let origin = self.calibrate_translation_origin(head, has_user);
        let [yaw_deg, pitch_deg, mut roll_deg] = self.relative_angles(decoded, has_user);

        let raw_translation = if self.send_translation {
            origin
                .map(|origin| {
                    [
                        (head[0] - origin[0]) / OPENTRACK_HEAD_SCALE * self.translation_scale[0],
                        (head[1] - origin[1]) / OPENTRACK_HEAD_SCALE * self.translation_scale[1],
                        (head[2] - origin[2]) / OPENTRACK_HEAD_SCALE * self.translation_scale[2],
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0])
        } else {
            [0.0, 0.0, 0.0]
        };

        if let Some(roll) = self.relative_roll_from_points(decoded, has_user) {
            roll_deg = roll;
        }

        let raw_angles = [yaw_deg, pitch_deg, roll_deg];
        let (translation, angles) = if self.send_translation {
            let angle_comp = scale_matrix(
                self.angle_translation_comp,
                self.angle_translation_comp_scale,
            );
            let rotation_comp = self.rotation_compensation(raw_translation, raw_angles, origin.is_some());
            match self.coupling_mode {
                CouplingMode::Rotation => (
                    [
                        raw_translation[0] - dot3(rotation_comp[0], raw_angles),
                        raw_translation[1] - dot3(rotation_comp[1], raw_angles),
                        raw_translation[2] - dot3(rotation_comp[2], raw_angles),
                    ],
                    raw_angles,
                ),
                CouplingMode::Translation => (
                    raw_translation,
                    angles_from_translation(raw_angles, raw_translation, angle_comp),
                ),
                CouplingMode::Hybrid => self.choose_pose_hypothesis_hybrid(
                    raw_translation,
                    raw_angles,
                    angle_comp,
                    rotation_comp,
                ),
                CouplingMode::Auto => choose_pose_hypothesis(
                    raw_translation,
                    raw_angles,
                    rotation_comp,
                    angle_comp,
                    self.angle_translation_deadzone,
                ),
            }
        } else {
            ([0.0, 0.0, 0.0], raw_angles)
        };
        let [x_cm, y_cm, z_cm] = translation;
        let [yaw_deg, pitch_deg, roll_deg] = [
            angles[0].clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            angles[1].clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            angles[2].clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        ];

        // OpenTrack "UDP over network" consumes pose axes in the same order
        // as its plugin API: TX, TY, TZ, Yaw, Pitch, Roll.
        let values = self.filter_pose([x_cm, y_cm, z_cm, yaw_deg, pitch_deg, roll_deg]);
        let mut packet = [0u8; 48];
        for (i, value) in values.iter().enumerate() {
            packet[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }

        self.socket
            .send_to(&packet, self.target)
            .context("failed to send OpenTrack UDP packet")?;

        Ok(Some(values))
    }

    pub(crate) fn choose_pose_hypothesis_hybrid(
        &mut self,
        raw_translation: [f64; 3],
        raw_angles: [f64; 3],
        angle_comp: [[f64; 3]; 3],
        rotation_comp: [[f64; 3]; 3],
    ) -> ([f64; 3], [f64; 3]) {
        let rotation_pose = pose_for_coupling_mode(
            CouplingMode::Rotation,
            raw_translation,
            raw_angles,
            rotation_comp,
            angle_comp,
        );
        let translation_pose = pose_for_coupling_mode(
            CouplingMode::Translation,
            raw_translation,
            raw_angles,
            rotation_comp,
            angle_comp,
        );
        let rotation_score = norm3(rotation_pose.0) / (norm3(raw_translation) + 1.0);
        let translation_score = norm3(translation_pose.1) / (norm3(raw_angles) + 1.0);
        let desired =
            if dot3(raw_translation, raw_translation).sqrt() <= self.angle_translation_deadzone {
                CouplingMode::Rotation
            } else if translation_score + OPENTRACK_COUPLING_SWITCH_MARGIN < rotation_score {
                CouplingMode::Translation
            } else if rotation_score + OPENTRACK_COUPLING_SWITCH_MARGIN < translation_score {
                CouplingMode::Rotation
            } else {
                self.coupling_state
            };

        if desired == self.coupling_state {
            self.coupling_candidate = desired;
            self.coupling_candidate_count = 0;
        } else if desired == self.coupling_candidate {
            self.coupling_candidate_count += 1;
            if self.coupling_candidate_count >= OPENTRACK_COUPLING_SWITCH_FRAMES {
                self.coupling_state = desired;
                self.coupling_candidate_count = 0;
            }
        } else {
            self.coupling_candidate = desired;
            self.coupling_candidate_count = 1;
        }

        match self.coupling_state {
            CouplingMode::Translation => translation_pose,
            _ => rotation_pose,
        }
    }

    pub(crate) fn filter_pose(&mut self, mut values: [f64; 6]) -> [f64; 6] {
        for value in &mut values[3..] {
            if value.abs() < self.angle_deadzone {
                *value = 0.0;
            }
        }

        let Some(previous) = self.last_pose else {
            self.last_pose = Some(values);
            return values;
        };

        let mut filtered = values;
        for i in 0..6 {
            filtered[i] = previous[i] + self.smoothing_alpha * (values[i] - previous[i]);
        }
        self.last_pose = Some(filtered);
        filtered
    }

    pub(crate) fn calibrate_translation_origin(
        &mut self,
        head: [f64; 3],
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        calibrate_origin(
            &mut self.origin,
            &mut self.origin_sum,
            &mut self.origin_count,
            self.origin_samples,
            head,
            can_calibrate,
        )
    }

    pub(crate) fn relative_angles(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> [f64; 3] {
        // At extreme head poses the tracker briefly drops the head-angle
        // occurrence. Recentering to zero on those frames makes the model snap
        // to center and back (a visible jump), so hold the last good angle
        // until a fresh one arrives instead.
        match self.relative_angles_fresh(decoded, can_calibrate) {
            Some(angles) => {
                self.last_angles = Some(angles);
                angles
            }
            None => self.last_angles.unwrap_or([0.0, 0.0, 0.0]),
        }
    }

    /// Head rotation from the learned [`HEAD_ROT_MODEL`]: gather the occ 0-9
    /// landmark positions, subtract their centroid (making this invariant to a
    /// pure head translation), apply the linear map, then zero against the
    /// calibrated rest pose. Returns yaw/pitch/roll in degrees, or `None` when
    /// any landmark is missing (so the caller holds the last angle).
    fn model_angles(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        let mut points = [[0.0; 3]; 10];
        let mut centroid = [0.0; 3];
        for (occurrence, point) in points.iter_mut().enumerate() {
            for (component, value) in point.iter_mut().enumerate() {
                *value = field_value(
                    decoded,
                    LiveField::new("", 0x00031f41, occurrence, component),
                )?;
                centroid[component] += *value;
            }
        }
        for value in &mut centroid {
            *value /= points.len() as f64;
        }

        // Linear map over the centroid-centred coordinates plus bias.
        let mut model = HEAD_ROT_MODEL[30];
        let mut feature = 0;
        for point in &points {
            for (component, value) in point.iter().enumerate() {
                let centred = value - centroid[component];
                for (axis, weight) in HEAD_ROT_MODEL[feature].iter().enumerate() {
                    model[axis] += weight * centred;
                }
                let _ = component;
                feature += 1;
            }
        }

        // Model columns are [pitch, yaw, roll] in radians; OpenTrack wants
        // [yaw, pitch, roll] in degrees.
        let raw = [
            model[1].to_degrees(),
            model[0].to_degrees(),
            model[2].to_degrees(),
        ];

        let origin = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            raw,
            can_calibrate,
        )?;

        Some([
            (raw[0] - origin[0]).clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            (raw[1] - origin[1]).clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            (raw[2] - origin[2]).clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        ])
    }

    fn relative_angles_fresh(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        if let Some(angles) = self.relative_angles_from_points(decoded, can_calibrate) {
            return Some(angles);
        }

        if let AngleSource::Model = self.angle_source {
            return self.model_angles(decoded, can_calibrate);
        }

        if let AngleSource::Gaze = self.angle_source {
            if let Some(angles) = self.relative_angles_from_gaze(decoded, can_calibrate) {
                return Some(angles);
            }
        };

        let occurrence = self.angle_occurrence?;
        let mut angles = [0.0; 3];
        for (out_axis, source) in self.angle_map.iter().enumerate() {
            if source.is_disabled() {
                continue;
            }

            let value = field_value(
                decoded,
                LiveField::new("", 0x00031f41, occurrence, source.component),
            )?;
            angles[out_axis] = source.sign * value;
        }

        let origin = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            angles,
            can_calibrate,
        )?;

        Some([
            ((angles[0] - origin[0]) / self.angle_scale[0])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            ((angles[1] - origin[1]) / self.angle_scale[1])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            ((angles[2] - origin[2]) / self.angle_scale[2])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        ])
    }

    pub(crate) fn relative_angles_from_gaze(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        let yaw = mean_keys(
            decoded,
            &[
                LiveField::new("", 0x00021f40, 0, 0),
                LiveField::new("", 0x00021f40, 5, 0),
            ],
        )?;
        let pitch = mean_keys(
            decoded,
            &[
                LiveField::new("", 0x00021f40, 0, 1),
                LiveField::new("", 0x00021f40, 5, 1),
            ],
        )?;
        let angles = [yaw, -pitch, 0.0];

        let origin = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            angles,
            can_calibrate,
        )?;

        Some([
            ((angles[0] - origin[0]) / self.angle_scale[0])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            ((angles[1] - origin[1]) / self.angle_scale[1])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            0.0,
        ])
    }

    pub(crate) fn relative_angles_from_points(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<[f64; 3]> {
        let (a, b) = self.angle_points?;
        let pa = head_point(decoded, a)?;
        let pb = head_point(decoded, b)?;
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let horizontal = (dx * dx + dz * dz).sqrt();
        if horizontal <= f64::EPSILON {
            return None;
        }

        let angles = [
            dx.atan2(dz).to_degrees(),
            (-dy).atan2(horizontal).to_degrees(),
            0.0,
        ];

        let origin = calibrate_origin(
            &mut self.angle_origin,
            &mut self.angle_origin_sum,
            &mut self.angle_origin_count,
            self.origin_samples,
            angles,
            can_calibrate,
        )?;

        Some([
            normalize_angle_deg(angles[0] - origin[0])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            normalize_angle_deg(angles[1] - origin[1])
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
            0.0,
        ])
    }

    pub(crate) fn relative_roll_from_points(
        &mut self,
        decoded: &BTreeMap<(u32, usize, usize), f64>,
        can_calibrate: bool,
    ) -> Option<f64> {
        let (a, b) = self.roll_points?;
        let pa = head_point(decoded, a)?;
        let pb = head_point(decoded, b)?;
        let roll = (pb[1] - pa[1]).atan2(pb[0] - pa[0]).to_degrees();
        let origin = calibrate_scalar_origin(
            &mut self.roll_origin,
            &mut self.roll_origin_sum,
            &mut self.roll_origin_count,
            self.origin_samples,
            roll,
            can_calibrate,
        )?;

        Some(
            normalize_angle_deg(roll - origin)
                .clamp(-OPENTRACK_MAX_HEAD_ANGLE_DEG, OPENTRACK_MAX_HEAD_ANGLE_DEG),
        )
    }
}

fn is_zero_matrix(matrix: [[f64; 3]; 3]) -> bool {
    matrix.iter().flatten().all(|value| *value == 0.0)
}

pub(crate) fn pose_for_coupling_mode(
    mode: CouplingMode,
    raw_translation: [f64; 3],
    raw_angles: [f64; 3],
    rotation_comp: [[f64; 3]; 3],
    angle_translation_comp: [[f64; 3]; 3],
) -> ([f64; 3], [f64; 3]) {
    match mode {
        CouplingMode::Translation => (
            raw_translation,
            angles_from_translation(raw_angles, raw_translation, angle_translation_comp),
        ),
        _ => (
            [
                raw_translation[0] - dot3(rotation_comp[0], raw_angles),
                raw_translation[1] - dot3(rotation_comp[1], raw_angles),
                raw_translation[2] - dot3(rotation_comp[2], raw_angles),
            ],
            raw_angles,
        ),
    }
}
