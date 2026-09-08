//! `OpenTrack` "UDP over network" sink for the replay/track CLI paths: turns a
//! decoded `TrackingFrame` into the 6-double `x,y,z,yaw,pitch,roll` packet,
//! with origin calibration, rotation/translation decoupling and smoothing.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use tracing::info;

use crate::math::{
    angles_from_translation, calibrate_origin, calibrate_scalar_origin, choose_pose_hypothesis,
    dot3, norm3, normalize_angle_deg, scale_matrix, solve_3x3,
};
use tobii_proto::decode::{LiveField, TrackingFrame, field_value, head_point, mean_keys};

/// Device head-position units per `OpenTrack` centimetre.
pub(crate) const OPENTRACK_HEAD_SCALE: f64 = 1000.0;

/// Default `angle_scale` for [`AngleSource::Gaze`].
pub(crate) const DEFAULT_OPENTRACK_GAZE_ANGLE_SCALE: [f64; 3] = [40.0, 60.0, 1.0];

/// Default `angle_scale` for [`AngleSource::Head`] / [`AngleSource::Model`].
pub(crate) const DEFAULT_OPENTRACK_HEAD_ANGLE_SCALE: [f64; 3] = [2.0, 1.0, 2.0];

/// Default head-angle occurrence of 0x00031f41.
pub(crate) const DEFAULT_OPENTRACK_ANGLE_OCC: usize = 7;

/// Default roll axis: the inter-eye line between the two eyeball rotation
/// centers (occ4 = left, occ9 = right of 0x00031f41). These points are
/// gaze-invariant, so the roll comes out clean (idle σ ≈ 0.16°) — unlike the
/// stream's positional "yaw"/"pitch", which are translation artifacts.
/// Override on the replay path with `--opentrack-roll-points`/`--opentrack-no-roll-points`.
pub(crate) const DEFAULT_OPENTRACK_ROLL_POINTS: (usize, usize) = (4, 9);

/// Output angles are clamped to +/- this many degrees.
pub(crate) const OPENTRACK_MAX_HEAD_ANGLE_DEG: f64 = 45.0;

/// Frames averaged into each rest-pose origin by default.
pub(crate) const DEFAULT_OPENTRACK_ORIGIN_SAMPLES: usize = 30;

/// Default exponential smoothing factor.
pub(crate) const DEFAULT_OPENTRACK_SMOOTHING: f64 = 0.35;

/// Default angle deadzone (degrees).
pub(crate) const DEFAULT_OPENTRACK_ANGLE_DEADZONE_DEG: f64 = 0.35;

/// Default rotation -> translation lever arm (none).
pub(crate) const DEFAULT_OPENTRACK_ROTATION_COMP: [[f64; 3]; 3] = [[0.0; 3]; 3];

/// Default per-axis translation multiplier.
pub(crate) const DEFAULT_OPENTRACK_TRANSLATION_SCALE: [f64; 3] = [1.0, 1.0, 1.0];

/// Default translation -> angle compensation (none).
pub(crate) const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP: [[f64; 3]; 3] = [[0.0; 3]; 3];

/// Default scalar on the translation -> angle compensation.
pub(crate) const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_COMP_SCALE: f64 = 1.0;

/// Default translation deadzone (cm) for the hybrid/auto hypotheses.
pub(crate) const DEFAULT_OPENTRACK_ANGLE_TRANSLATION_DEADZONE_CM: f64 = 0.0;

/// Consecutive frames a hybrid-mode switch must be requested before it happens.
pub(crate) const OPENTRACK_COUPLING_SWITCH_FRAMES: usize = 4;

/// Score margin one hypothesis must win by to trigger a hybrid-mode switch.
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

/// Default raw component -> output axis map: yaw = comp 0, pitch = -comp 1, roll off.
pub(crate) const DEFAULT_OPENTRACK_ANGLE_MAP: [AngleComponent; 3] = [
    AngleComponent::new(0, 1.0),
    AngleComponent::new(1, -1.0),
    AngleComponent::disabled(),
];

/// Where the yaw/pitch signal sent to `OpenTrack` comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AngleSource {
    /// Mean gaze direction of both eyes (0x00021f40 occ 0/5).
    Gaze,
    /// The stream's positional head "angles" (0x00031f41, `--opentrack-angle-occ`).
    Head,
    /// Learned translation-invariant rigid-rotation model (occ 0-9 landmarks).
    Model,
}

/// How rotation- and translation-induced pose components are separated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CouplingMode {
    /// Subtract `rotation_comp * angles` from the translation.
    Rotation,
    /// Subtract `angle_translation_comp * translation` from the angles.
    Translation,
    /// Pick per frame between the two hypotheses with hysteresis.
    Hybrid,
    /// Pick per frame by [`choose_pose_hypothesis`].
    Auto,
}

/// One output angle axis: which stream component feeds it and with what sign.
/// A zero sign disables the axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AngleComponent {
    /// Component index (0..3) inside the head-angle occurrence.
    pub(crate) component: usize,
    /// Multiplier applied to the raw value; `0.0` means disabled.
    pub(crate) sign: f64,
}

impl AngleComponent {
    /// Axis fed by `component`, scaled by `sign`.
    #[must_use]
    pub(crate) const fn new(component: usize, sign: f64) -> Self {
        Self { component, sign }
    }

    /// Axis that always outputs zero.
    #[must_use]
    pub(crate) const fn disabled() -> Self {
        Self {
            component: 0,
            sign: 0.0,
        }
    }

    /// `true` when this axis is switched off (exact-zero sign is the sentinel).
    #[must_use]
    pub(crate) fn is_disabled(self) -> bool {
        self.sign == 0.0
    }
}

/// Tuning knobs for [`OpentrackUdp`], filled from the CLI options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OpentrackConfig {
    /// Send head translation (otherwise x/y/z are always zero).
    pub(crate) send_translation: bool,
    /// Source of the yaw/pitch signal.
    pub(crate) angle_source: AngleSource,
    /// Head-angle occurrence of 0x00031f41 used by [`AngleSource::Head`].
    pub(crate) angle_occurrence: Option<usize>,
    /// Divisor per output axis (raw units per degree).
    pub(crate) angle_scale: [f64; 3],
    /// Raw component -> output axis mapping for [`AngleSource::Head`].
    pub(crate) angle_map: [AngleComponent; 3],
    /// Frames averaged for each rest-pose origin.
    pub(crate) origin_samples: usize,
    /// Landmark pair whose direction gives yaw/pitch, if any.
    pub(crate) angle_points: Option<(usize, usize)>,
    /// Landmark pair whose in-plane direction gives roll, if any.
    pub(crate) roll_points: Option<(usize, usize)>,
    /// Exponential smoothing factor for the output pose (1.0 = no smoothing).
    pub(crate) smoothing_alpha: f64,
    /// Angles below this magnitude (degrees) are zeroed.
    pub(crate) angle_deadzone: f64,
    /// Fixed rotation -> translation lever arm (zero = learn online when enabled).
    pub(crate) rotation_comp: [[f64; 3]; 3],
    /// Per-axis multiplier on the translation.
    pub(crate) translation_scale: [f64; 3],
    /// Translation -> angle compensation matrix.
    pub(crate) angle_translation_comp: [[f64; 3]; 3],
    /// Scalar applied to `angle_translation_comp`.
    pub(crate) angle_translation_comp_scale: f64,
    /// Translation magnitude (cm) below which the rotation hypothesis wins.
    pub(crate) angle_translation_deadzone: f64,
    /// Rotation/translation separation strategy.
    pub(crate) coupling_mode: CouplingMode,
    /// Learn the rotation -> translation lever arm online.
    pub(crate) auto_decouple: bool,
}

/// Live `OpenTrack` UDP sender with its calibration and filter state.
#[derive(Debug)]
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
    /// Resolve `host:port`, bind a UDP socket and build the sender from `config`.
    ///
    /// # Errors
    /// Fails when the target does not resolve or the socket cannot be bound.
    pub(crate) fn new(host: &str, port: u16, config: &OpentrackConfig) -> Result<Self> {
        let OpentrackConfig {
            send_translation,
            angle_source,
            angle_occurrence,
            angle_scale,
            angle_map,
            origin_samples,
            angle_points,
            roll_points,
            smoothing_alpha,
            angle_deadzone,
            rotation_comp,
            translation_scale,
            angle_translation_comp,
            angle_translation_comp_scale,
            angle_translation_deadzone,
            coupling_mode,
            auto_decouple,
        } = *config;
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
        info!(
            %target,
            angle_source = source,
            auto_decouple = decouple,
            "sending OpenTrack UDP frames"
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
            auto_decouple: decouple,
            decouple_ata: [[0.0; 3]; 3],
            decouple_atb: [[0.0; 3]; 3],
            decouple_weight: 0.0,
            learned_rotation_comp: [[0.0; 3]; 3],
            last_angles: None,
        })
    }

    /// Positional form of [`OpentrackUdp::new`], kept for the existing
    /// `device::replay_and_read_stream` call site.
    ///
    /// # Errors
    /// See [`OpentrackUdp::new`].
    #[allow(clippy::too_many_arguments)] // reason: legacy call site; prefer `new` + `OpentrackConfig`
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
        Self::new(
            host,
            port,
            &OpentrackConfig {
                send_translation,
                angle_source,
                angle_occurrence,
                angle_scale,
                angle_map,
                origin_samples,
                angle_points,
                roll_points,
                smoothing_alpha,
                angle_deadzone,
                rotation_comp,
                translation_scale,
                angle_translation_comp,
                angle_translation_comp_scale,
                angle_translation_deadzone,
                coupling_mode,
                auto_decouple,
            },
        )
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
            for (ata_row, angle_r) in self.decouple_ata.iter_mut().zip(angles) {
                for (cell, angle_c) in ata_row.iter_mut().zip(angles) {
                    *cell = *cell * OPENTRACK_DECOUPLE_FORGET + angle_r * angle_c;
                }
            }
            for (atb_row, shift) in self.decouple_atb.iter_mut().zip(translation) {
                for (cell, angle_c) in atb_row.iter_mut().zip(angles) {
                    *cell = *cell * OPENTRACK_DECOUPLE_FORGET + shift * angle_c;
                }
            }
            self.decouple_weight = self.decouple_weight * OPENTRACK_DECOUPLE_FORGET + 1.0;

            if self.decouple_weight >= OPENTRACK_DECOUPLE_MIN_WEIGHT {
                // Ridge keeps unexcited axes (e.g. roll with no roll motion)
                // from inventing a compensation; it shrinks toward zero there.
                let ridge = OPENTRACK_DECOUPLE_RIDGE * self.decouple_weight;
                let mut ata = self.decouple_ata;
                for (i, row) in ata.iter_mut().enumerate() {
                    row[i] += ridge;
                }
                for (learned_row, atb_row) in
                    self.learned_rotation_comp.iter_mut().zip(self.decouple_atb)
                {
                    if let Some(row) = solve_3x3(ata, atb_row) {
                        for (cell, solved) in learned_row.iter_mut().zip(row) {
                            *cell = *cell * (1.0 - OPENTRACK_DECOUPLE_BLEND)
                                + solved * OPENTRACK_DECOUPLE_BLEND;
                        }
                    }
                }
            }
        }

        self.learned_rotation_comp
    }

    /// Convert one frame to an `OpenTrack` pose and send it. Returns the pose
    /// that went on the wire, or `None` when the frame carries no head position.
    ///
    /// # Errors
    /// Fails when the UDP send fails.
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
            let rotation_comp =
                self.rotation_compensation(raw_translation, raw_angles, origin.is_some());
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
        for (chunk, value) in packet.chunks_exact_mut(8).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        self.socket
            .send_to(&packet, self.target)
            .context("failed to send OpenTrack UDP packet")?;

        Ok(Some(values))
    }

    /// Hybrid coupling: score both hypotheses and switch state only after
    /// [`OPENTRACK_COUPLING_SWITCH_FRAMES`] consecutive frames agree.
    fn choose_pose_hypothesis_hybrid(
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
            CouplingMode::Rotation | CouplingMode::Hybrid | CouplingMode::Auto => rotation_pose,
        }
    }

    /// Apply the angle deadzone, then exponentially smooth against the last
    /// sent pose.
    fn filter_pose(&mut self, mut values: [f64; 6]) -> [f64; 6] {
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
        for ((out, prev), value) in filtered.iter_mut().zip(previous).zip(values) {
            *out = prev + self.smoothing_alpha * (value - prev);
        }
        self.last_pose = Some(filtered);
        filtered
    }

    /// Average the first `origin_samples` head positions (while a user is
    /// present) into the translation rest pose.
    fn calibrate_translation_origin(
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

    /// Yaw/pitch/roll relative to the calibrated rest pose, in degrees.
    fn relative_angles(
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

        if self.angle_source == AngleSource::Model {
            return self.model_angles(decoded, can_calibrate);
        }

        if self.angle_source == AngleSource::Gaze
            && let Some(angles) = self.relative_angles_from_gaze(decoded, can_calibrate)
        {
            return Some(angles);
        }

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

    /// Yaw/pitch from the mean binocular gaze direction (roll is always 0).
    fn relative_angles_from_gaze(
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

    /// Yaw/pitch from the direction between two head landmarks
    /// (`angle_points`); roll is always 0.
    fn relative_angles_from_points(
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

    /// Roll from the in-plane direction between two head landmarks
    /// (`roll_points`).
    fn relative_roll_from_points(
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

/// `true` when every entry is exactly zero (the "not configured" sentinel).
fn is_zero_matrix(matrix: [[f64; 3]; 3]) -> bool {
    matrix.iter().flatten().all(|value| *value == 0.0)
}

/// `(translation, angles)` under one coupling hypothesis; `Hybrid`/`Auto`
/// evaluate as `Rotation` here, the caller does the selection.
#[must_use]
fn pose_for_coupling_mode(
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
        CouplingMode::Rotation | CouplingMode::Hybrid | CouplingMode::Auto => (
            [
                raw_translation[0] - dot3(rotation_comp[0], raw_angles),
                raw_translation[1] - dot3(rotation_comp[1], raw_angles),
                raw_translation[2] - dot3(rotation_comp[2], raw_angles),
            ],
            raw_angles,
        ),
    }
}
