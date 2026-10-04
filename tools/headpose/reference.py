"""The executable specification of tobii-pose's head pose estimator (head.rs): one face fit of a
0x50e image in, one Stream Engine head pose (tobii_head_pose_t) out.

A port of the head pose study's streaming reference: the same functions, the same order of
operations and so the same numbers, bit for bit (make_vectors.py regenerates the study's test
vectors from it). Plain scalar and 3x3 code on purpose, one call per image; no session data, no
pixels. The vectorised twins in common.py are for the fits over whole sessions.

Frames: C camera (x right in the image, y down, z forward); S tracker (p_S = D p_C,
D = diag(-1, -1, 1)); T display, from the display area (TL, TR, BL in S, mm): the rows of R_TS
are x^ = unit(TR - TL), y^ = unit((TL - BL) - ((TL - BL).x^) x^), z^ = x^ x y^; c = (TR + BL) / 2;
p_T = R_TS (p_S - c) (tobii_ipc::geometry::DisplayFrame).

Per image, the input is the face fit (face, score, R_cam object -> camera, t_cam_mm the canonical
mesh's origin in C, lm280 the 468 landmarks in the 280-px frame's continuous pixels) and t_us, the
image's time (offline the image's device time, live its host time).

  1. validity G3: face AND score >= 0 AND edge(centroid of the 468 landmarks) >= 6 px AND
     edge(landmark 1, the nose tip) >= -4 px, edge(u, v) = min(u, 280 - u, v, 280 - v), signed
     (+ inside), NaN (so failing) when u or v is. No hysteresis. All four of the pose's validity
     flags take this one value.
  2. rotation: H_S = D R_cam D Q (head -> S), R_T = R_TS H_S, (x, y, z) = yxz(R_T):
     x = asin(-R_T[1][2]), y = atan2(R_T[0][2], R_T[2][2]), z = atan2(R_T[1][0], R_T[1][1]).
  3. position (S, mm):
       PnP branch  p0 = s D t_cam + H_S d;
                   p_pnp = |p0| unit(g_p p0x/p0z + ox_p, g_p p0y/p0z + oy_p, 1)
       eye branch  a, b = centroids of the 16 eye-contour landmarks (image-left, image-right), rays
                   r = unit(((u - cu)/fu, (v - cv)/fv, 1)) with the 0x50e intrinsics;
                   m = unit(r_a + r_b), ang = acos(r_a . r_b), fs = sqrt(max(1 - (H_S[:, 0] . m)^2,
                   1e-6)), range = K fs / sin(ang), p_eye = range unit(g_e mx/mz + ox_e,
                   g_e my/mz + oy_e, 1)
       correction  c = w (p_eye - p_pnp) (w = 0: the PnP branch alone; a non-finite eye branch,
                   such as two eyes on one ray (sin(ang) = 0), leaves the correction as it was)
  4. filters, one step per valid image; dt is the time since the last VALID image:
       correction  c_lp += beta(dt) (c - c_lp),           beta(dt) = dt / (dt + TAU_BETA)
       position    y += alpha(dt) (p_pnp + c_lp - y),     alpha(dt) = dt / (dt + TAU_POS)   (in S)
       rotation    a one-euro filter per yxz angle, degrees: d = wrap(x - x_prev) / dt,
                   dx += a(DCUT, dt) (d - dx), fc = fmin + beta |dx|,
                   out = wrap(out + a(fc, dt) wrap(x - out)), a(f, dt) = 1 / (1 + 1 / (2 pi f dt))
     Hard reset (state = input, dx = 0): at the first valid image, when dt <= 0 or dt > 1 s, on
     set_display_area() / set_display_frame(), after a non-finite result, and -- with rule
     'reset' only -- at the first valid image after an invalid one. Rule 'timeaware' (the
     default, and head.rs's) keeps the state across invalid images: the longer the gap, the
     larger dt and the closer the first step comes to the new input.
  5. output: valid -> flags 1, position_T = R_TS (y - c) mm, rotation = the filtered angles in
     radians; invalid -> flags 0 and the last valid values (zeros before the first valid image).

The constants are a dict P (keys as in PROTOTYPE); head_params() and params_from_head_params()
convert it to and from the field names of head.rs's HeadParams, which fit.py writes.
"""

import math

import numpy as np

D = np.diag([-1.0, -1.0, 1.0])
FRAME_PX = 280.0
NOSE_TIP = 1
CONTOUR_IMAGE_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
CONTOUR_IMAGE_RIGHT = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
]
INTRINSICS = dict(fu=-375.9, cu=140.9, fv=-383.3, cv=139.8)

# The display area of the three Windows sessions the constants were fitted on (TL, TR, BL in S,
# mm): the area of their init, which the gaze origins 0x22/0x24 of every frame confirm.
WINDOWS_AREA = (
    (-315.7917175292969, 324.4114685058594, 111.23909759521484),
    (317.7962341308594, 324.4114685058594, 111.23909759521484),
    (-315.7917175292969, 10.267330169677734, -3.100020170211792),
)

# The image interval of the 0x50e stream, s: the time constants are given as the share of the way
# a filter moves in one interval.
DT_NOMINAL_S = 0.030208
ONE_EURO_PER_AXIS = {"x": (1.0, 0.4), "y": (0.7, 0.2), "z": (1.5, 0.1)}
ONE_EURO_SHARED = {"x": (1.0, 0.2), "y": (1.0, 0.2), "z": (1.0, 0.2)}

# The study's constants (2026-09-29): fitted on all three Windows sessions with the Python port of
# the tracker, BLlp position (PnP branch + eye correction), a one-euro filter per axis. The test
# vectors are made with them. fit_cost_* are the robust least-squares costs of the two position
# fits (provenance only).
PROTOTYPE = {
    "K": 64.2916489241778,
    "g_e": 1.0208289329512754,
    "ox_e": -0.004033985018082697,
    "oy_e": -0.00026239095608242275,
    "s": 0.9714295539449831,
    "d": [-2.8243437361896615, 27.404881620713358, -36.48321259035155],
    "g_p": 1.0135567464173814,
    "ox_p": 0.0018157214539677182,
    "oy_p": -0.0074094098763837655,
    "w": 0.38341850370665953,
    "fit_cost_eye": 1690209.3639196542,
    "fit_cost_pnp": 1360319.5414331947,
    "Q": [
        [0.9998600753746771, -0.007035297618850274, -0.015176767085183904],
        [0.006109694286072543, 0.9981679245204076, -0.06019523315306708],
        [0.0155724534828155, 0.060094084950480346, 0.998071239765223],
    ],
    "tau_pos_s": 0.07048533333333333,
    "tau_beta_s": 0.07048533333333333,
    "beta_a": 0.3,
    "one_euro": {"x": [1.0, 0.4], "y": [0.7, 0.2], "z": [1.5, 0.1]},
    "one_euro_dcut_hz": 1.0,
    "reset_gap_s": 1.0,
    "g3_centroid_min_px": 6.0,
    "g3_nose_min_px": -4.0,
}


# ------------------------------------------------------------------ geometry
def unit(v):
    v = np.asarray(v, float)
    return v / math.sqrt(float(v @ v))


def display_frame(TL, TR, BL):
    """(R_TS, c): p_T = R_TS (p_S - c). Gram-Schmidt, so that a sheared area still gives a
    rotation."""
    TL, TR, BL = (np.asarray(p, float) for p in (TL, TR, BL))
    x = unit(TR - TL)
    up = TL - BL
    y = unit(up - (up @ x) * x)
    z = np.cross(x, y)
    return np.stack([x, y, z]), 0.5 * (TR + BL)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def compose_yxz(x, y, z):
    """R = Ry(y) Rx(x) Rz(z) (radians)."""
    return ry(y) @ rx(x) @ rz(z)


def euler_yxz(R):
    """The inverse of compose_yxz: (x, y, z), radians."""
    return (
        math.asin(max(-1.0, min(1.0, -R[1][2]))),
        math.atan2(R[0][2], R[2][2]),
        math.atan2(R[1][0], R[1][1]),
    )


def euler_zyx(R):
    """The WRONG form, for the broken-variant checks only: R = Rz(z) Ry(y) Rx(x) read as
    (x, y, z)."""
    return (
        math.atan2(R[2][1], R[2][2]),
        math.asin(max(-1.0, min(1.0, -R[2][0]))),
        math.atan2(R[1][0], R[0][0]),
    )


def wrap_deg(a):
    """To [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ validity
def edge_px(u, v):
    """The signed distance of (u, v) from the frame's nearest edge, px (+ inside). NaN when u or v
    is: min() would pass over a NaN that does not come first, and an unusable point must fail G3
    (head.rs's edge_px, common.edge_v)."""
    if math.isnan(u) or math.isnan(v):
        return math.nan
    return min(u, FRAME_PX - u, v, FRAME_PX - v)


def g3(face, score, lm280, centroid_min=6.0, nose_min=-4.0):
    """(valid, centroid edge px, nose edge px)."""
    if not face or not (score >= 0.0):
        return False, float("nan"), float("nan")
    lm = np.asarray(lm280, float)
    cu, cv = float(lm[:, 0].mean()), float(lm[:, 1].mean())
    ce = edge_px(cu, cv)
    ne = edge_px(float(lm[NOSE_TIP, 0]), float(lm[NOSE_TIP, 1]))
    return (ce >= centroid_min) and (ne >= nose_min), ce, ne


def g3_points(face, score, centroid, nose_tip, centroid_min=6.0, nose_min=-4.0):
    """g3() from the landmarks' centroid and the nose tip, (u, v) each."""
    if not face or not (score >= 0.0):
        return False, float("nan"), float("nan")
    ce = edge_px(float(centroid[0]), float(centroid[1]))
    ne = edge_px(float(nose_tip[0]), float(nose_tip[1]))
    return (ce >= centroid_min) and (ne >= nose_min), ce, ne


# ------------------------------------------------------------------ per-image models
def ray_S(u, v, K=INTRINSICS):
    return unit(np.array([(u - K["cu"]) / K["fu"], (v - K["cv"]) / K["fv"], 1.0]))


def dir_map(p, g, ox, oy):
    """Keep |p|, map the direction in tan space: |p| unit(g px/pz + ox, g py/pz + oy, 1)."""
    return math.sqrt(float(p @ p)) * unit(
        np.array([g * p[0] / p[2] + ox, g * p[1] / p[2] + oy, 1.0])
    )


def head_S(R_cam, Q):
    return D @ np.asarray(R_cam, float) @ D @ np.asarray(Q, float)


def p_pnp_S(t_cam_mm, H, P):
    p0 = P["s"] * (D @ np.asarray(t_cam_mm, float)) + H @ np.asarray(P["d"], float)
    return dir_map(p0, P["g_p"], P["ox_p"], P["oy_p"])


def eye_points(lm280):
    lm = np.asarray(lm280, float)
    return lm[CONTOUR_IMAGE_LEFT].mean(0), lm[CONTOUR_IMAGE_RIGHT].mean(0)


def p_eye_S(eye_a, eye_b, H, P):
    ra, rb = ray_S(*eye_a), ray_S(*eye_b)
    m = unit(ra + rb)
    # As head.rs's f64: the clamp keeps a NaN, and a division by sin(0) (the two rays one) gives
    # inf, not an exception. Either way the eye branch is not finite, which leaves the correction
    # as it was.
    cos = float(ra @ rb)
    ang = math.acos(cos if math.isnan(cos) else max(-1.0, min(1.0, cos)))
    fs = math.sqrt(max(1.0 - float(H[:, 0] @ m) ** 2, 1e-6))
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = float(np.float64(P["K"] * fs) / math.sin(ang))
        return rng * unit(
            np.array([P["g_e"] * m[0] / m[2] + P["ox_e"], P["g_e"] * m[1] / m[2] + P["oy_e"], 1.0])
        )


# ------------------------------------------------------------------ filters
def a_tau(dt, tau):
    """The time-aware EMA coefficient dt / (dt + tau) (0.30 at dt 30.208 ms for tau 70.4853 ms)."""
    return dt / (dt + tau)


def a_fc(f, dt):
    """The one-euro smoothing coefficient 1 / (1 + 1 / (2 pi f dt))."""
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * f * dt))


def tau_of_a(a, dt=DT_NOMINAL_S):
    """The time constant of an EMA that moves `a` of the way in one image interval."""
    return dt * (1.0 / a - 1.0)


class OneEuro:
    """One channel, degrees; reset() then step(x, dt)."""

    def __init__(self, fmin, beta, dcut=1.0):
        self.fmin, self.beta, self.dcut = fmin, beta, dcut
        self.y = self.xp = None
        self.dx = 0.0

    def reset(self, x):
        self.y = self.xp = x
        self.dx = 0.0
        return self.y

    def step(self, x, dt):
        d = wrap_deg(x - self.xp) / dt
        self.dx += a_fc(self.dcut, dt) * (d - self.dx)
        fc = self.fmin + self.beta * abs(self.dx)
        self.y = wrap_deg(self.y + a_fc(fc, dt) * wrap_deg(x - self.y))
        self.xp = x
        return self.y


class Ema3:
    """Three channels (S, mm); reset() then step(x, dt)."""

    def __init__(self, tau):
        self.tau = tau
        self.y = None

    def reset(self, x):
        self.y = np.array(x, float)
        return self.y.copy()

    def step(self, x, dt):
        self.y = self.y + a_tau(dt, self.tau) * (np.asarray(x, float) - self.y)
        return self.y.copy()


# ------------------------------------------------------------------ the filter bank, the reset rule
class Smoother:
    """All the filter state of one estimator: the correction's low-pass and the position's EMA
    (3 channels each, S, mm) and the one-euro filters of the three yxz angles (degrees). One call
    per processed image: invalid() for an image without a valid pose, valid(...) for one with;
    hard_reset() drops the state (a display area change, a non-finite result)."""

    def __init__(
        self, tau_pos_s, tau_beta_s, one_euro, dcut_hz=1.0, reset_gap_s=1.0, rule="timeaware"
    ):
        assert rule in ("timeaware", "reset")
        self.rule = rule
        self.reset_gap_s = reset_gap_s
        self.oe = [OneEuro(one_euro[k][0], one_euro[k][1], dcut_hz) for k in "xyz"]
        self.ema_pos = Ema3(tau_pos_s)
        self.ema_cor = Ema3(tau_beta_s)
        self.have = False  # filter state exists
        self.t_last = None  # time of the last valid sample (us)
        self.prev_valid = False  # the previous processed image was valid

    def hard_reset(self):
        self.have = False

    def invalid(self):
        self.prev_valid = False

    def valid(self, t_us, pnp_S, cor_S, ang_deg):
        """pnp_S: the PnP branch's position (S, mm); cor_S: w (p_eye - p_pnp), or None (the PnP
        branch alone, or a non-finite eye branch); ang_deg: the yxz angles (x, y, z), degrees.
        Returns (filtered position S, filtered angles deg, dt s or None, reset flag)."""
        dt = (t_us - self.t_last) * 1e-6 if self.have else None
        reset = (
            dt is None
            or dt <= 0.0
            or dt > self.reset_gap_s
            or (self.rule == "reset" and not self.prev_valid)
        )
        if reset:
            cl = self.ema_cor.reset(cor_S if cor_S is not None else np.zeros(3))
            y = self.ema_pos.reset(np.asarray(pnp_S, float) + cl)
            rot = np.array([f.reset(float(a)) for f, a in zip(self.oe, ang_deg)])
        else:
            cl = self.ema_cor.step(cor_S, dt) if cor_S is not None else self.ema_cor.y.copy()
            y = self.ema_pos.step(np.asarray(pnp_S, float) + cl, dt)
            rot = np.array([f.step(float(a), dt) for f, a in zip(self.oe, ang_deg)])
        self.have = True
        self.t_last = t_us
        self.prev_valid = True
        return y, rot, dt, reset


# ------------------------------------------------------------------ the estimator
class HeadPoseEstimator:
    def __init__(self, P, area=None, rule="timeaware", rot_filter=None, decomposition="yxz"):
        """P: the constants (keys as in PROTOTYPE); area: (TL, TR, BL) in S, mm. rule: 'timeaware'
        (the state is kept, dt since the last valid image) or 'reset' (reset at the first valid
        image after an invalid one). rot_filter: {'x': (fmin, beta), 'y': ..., 'z': ...} in place
        of P['one_euro']; decomposition: 'yxz' (right) or 'zyx' (the broken variant, checks
        only)."""
        self.P = P
        self.dec = decomposition
        self.sm = Smoother(
            P["tau_pos_s"],
            P["tau_beta_s"],
            rot_filter or P["one_euro"],
            P["one_euro_dcut_hz"],
            P["reset_gap_s"],
            rule,
        )
        self.last_out = (np.zeros(3), np.zeros(3))
        self.R_TS = self.c = None
        if area is not None:
            self.set_display_area(*area)

    def set_display_area(self, TL, TR, BL):
        self.set_display_frame(*display_frame(TL, TR, BL))

    def set_display_frame(self, R_TS, c):
        """The display frame itself (p_T = R_TS (p_S - c)), e.g. one fitted to a log's gaze
        origins; resets the filters like a new display area."""
        self.R_TS, self.c = np.asarray(R_TS, float), np.asarray(c, float)
        self.sm.hard_reset()

    def _invalid(self, extra):
        self.sm.invalid()
        pos, rot = self.last_out
        return dict(valid=False, position_mm=pos.copy(), rotation_rad=rot.copy(), **extra)

    def step(self, t_us, face, score, R_cam, t_cam_mm, lm280):
        """One image: its time (us) and its face fit, lm280 the 468 landmarks."""
        P = self.P
        ok, ce, ne = g3(face, score, lm280, P["g3_centroid_min_px"], P["g3_nose_min_px"])
        return self._step(t_us, ok, ce, ne, R_cam, t_cam_mm, lambda: eye_points(lm280))

    def step_points(self, t_us, face, score, R_cam, t_cam_mm, centroid, nose_tip, eye_a, eye_b):
        """step() for a fit given by the points of its landmarks the estimator reads, as the fits
        CSV has them: the centroid of the 468, the nose tip and the two eye-contour centroids."""
        P = self.P
        ok, ce, ne = g3_points(
            face, score, centroid, nose_tip, P["g3_centroid_min_px"], P["g3_nose_min_px"]
        )
        return self._step(t_us, ok, ce, ne, R_cam, t_cam_mm, lambda: (eye_a, eye_b))

    def _step(self, t_us, ok, ce, ne, R_cam, t_cam_mm, eyes):
        P = self.P
        extra = dict(g3_centroid_edge_px=ce, g3_nose_edge_px=ne)
        if not ok or self.R_TS is None:
            return self._invalid(extra)
        H = head_S(R_cam, P["Q"])
        RT = self.R_TS @ H
        ang = np.degrees((euler_yxz if self.dec == "yxz" else euler_zyx)(RT))
        pp = p_pnp_S(t_cam_mm, H, P)
        cor = None
        if P["w"] != 0.0:
            ea, eb = eyes()
            pe = p_eye_S(ea, eb, H, P)
            if np.isfinite(pe).all():
                cor = P["w"] * (pe - pp)
        if not (np.isfinite(ang).all() and np.isfinite(pp).all()):
            self.sm.hard_reset()
            return self._invalid(extra)
        y, rot, dt, reset = self.sm.valid(t_us, pp, cor, ang)
        pos_T = self.R_TS @ (y - self.c)
        rot_rad = np.radians(rot)
        self.last_out = (pos_T.copy(), rot_rad.copy())
        extra.update(
            raw_rot_deg=ang, raw_pnp_S=pp, raw_cor_S=cor, filt_S=y, dt_s=dt, filter_reset=reset
        )
        return dict(valid=True, position_mm=pos_T, rotation_rad=rot_rad, **extra)


# ------------------------------------------------------------------ the constants
def make_params(Q, pos, beta_a=0.2, tau_pos_s=None, one_euro=None, g3=(6.0, -4.0)):
    """P from a rotation offset Q and a position fit `pos` (K, g_e, ox_e, oy_e, s, d, g_p, ox_p,
    oy_p, w, ...), the correction's per-image share beta_a and the position's time constant
    (default: 0.30 of the way per image)."""
    P = dict(pos)
    P.update(
        Q=np.asarray(Q, float),
        tau_pos_s=float(tau_of_a(0.30) if tau_pos_s is None else tau_pos_s),
        tau_beta_s=float(tau_of_a(beta_a)),
        beta_a=float(beta_a),
        one_euro=dict(one_euro or ONE_EURO_PER_AXIS),
        one_euro_dcut_hz=1.0,
        reset_gap_s=1.0,
        g3_centroid_min_px=float(g3[0]),
        g3_nose_min_px=float(g3[1]),
    )
    return P


def shipped(P):
    """P with the two choices head.rs ships: the PnP branch alone (w = 0; the eye branch keeps its
    constants, for comparison) and one one-euro filter (1.0 Hz, 0.2 Hz per deg/s) shared by the
    three angles. shipped(PROTOTYPE) is the prototype fit's HeadParams::FITTED."""
    return dict(P, w=0.0, one_euro={k: list(v) for k, v in ONE_EURO_SHARED.items()})


# head.rs's HeadParams::FITTED until the constants are fitted again on the Rust tracker's fits.
FITTED = shipped(PROTOTYPE)


def _direction(gain, ox, oy):
    return {"gain": float(gain), "offset": [float(ox), float(oy)]}


def head_params(P):
    """P under the field names of head.rs's HeadParams (a JSON-ready dict)."""
    return {
        "rotation_offset": np.asarray(P["Q"], float).tolist(),
        "pnp_scale": float(P["s"]),
        "pnp_point_mm": [float(v) for v in P["d"]],
        "pnp_direction": _direction(P["g_p"], P["ox_p"], P["oy_p"]),
        "eye_rays": dict(INTRINSICS),
        "eye_contours": [list(CONTOUR_IMAGE_LEFT), list(CONTOUR_IMAGE_RIGHT)],
        "eye_range_mm": float(P["K"]),
        "eye_direction": _direction(P["g_e"], P["ox_e"], P["oy_e"]),
        "eye_weight": float(P["w"]),
        "position_tau_s": float(P["tau_pos_s"]),
        "correction_tau_s": float(P["tau_beta_s"]),
        "rotation_filters": [
            {"min_cutoff_hz": float(P["one_euro"][k][0]), "beta": float(P["one_euro"][k][1])}
            for k in "xyz"
        ],
        "rotation_derivative_cutoff_hz": float(P["one_euro_dcut_hz"]),
        "reset_gap_s": float(P["reset_gap_s"]),
        "min_centroid_edge_px": float(P["g3_centroid_min_px"]),
        "min_nose_tip_edge_px": float(P["g3_nose_min_px"]),
    }


def params_from_head_params(H):
    """The inverse of head_params(). The eye rays and contours are this module's constants:
    the reference has no other ones, so a HeadParams with others is refused."""
    contours = [CONTOUR_IMAGE_LEFT, CONTOUR_IMAGE_RIGHT]
    if H["eye_rays"] != INTRINSICS or H["eye_contours"] != contours:
        raise ValueError(
            "the reference casts the eye rays with INTRINSICS and the CONTOUR_* landmarks only"
        )
    tau_beta = float(H["correction_tau_s"])
    return {
        "K": H["eye_range_mm"],
        "g_e": H["eye_direction"]["gain"],
        "ox_e": H["eye_direction"]["offset"][0],
        "oy_e": H["eye_direction"]["offset"][1],
        "s": H["pnp_scale"],
        "d": list(H["pnp_point_mm"]),
        "g_p": H["pnp_direction"]["gain"],
        "ox_p": H["pnp_direction"]["offset"][0],
        "oy_p": H["pnp_direction"]["offset"][1],
        "w": H["eye_weight"],
        "Q": [list(r) for r in H["rotation_offset"]],
        "tau_pos_s": H["position_tau_s"],
        "tau_beta_s": tau_beta,
        "beta_a": a_tau(DT_NOMINAL_S, tau_beta),
        "one_euro": {
            k: [f["min_cutoff_hz"], f["beta"]] for k, f in zip("xyz", H["rotation_filters"])
        },
        "one_euro_dcut_hz": H["rotation_derivative_cutoff_hz"],
        "reset_gap_s": H["reset_gap_s"],
        "g3_centroid_min_px": H["min_centroid_edge_px"],
        "g3_nose_min_px": H["min_nose_tip_edge_px"],
    }
