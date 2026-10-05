#!/usr/bin/env python3
"""Evaluate the head pose against the Stream Engine's own: run reference.py's estimator over the
face fits of recorded sessions, or take the daemon's own poses of them, and print the acceptance
metrics, the filters' lag and rest jitter among them, and the gates.

    evaluate.py --session LOG JSONL FITS [--session ...] [--area A]
                [--params FIT.json [--loso]] [--blend] [--rotation-filters params|shared|per-axis]
                [--rule timeaware|reset] [--reference-fits REF ...] [--ours-csv CSV ...]
                [--broken zyx|q-transposed|no-q|no-filter] [--gates GATES.json|none]

Ours: the pose reference.py's estimator (the Python twin of head.rs) makes of the fits, unless
--ours-csv gives, one per session in order, the --csv of `tobii5-init-replay compare-dll LOG JSONL
--head`: the poses of the daemon's own pipeline (HeadStep, with the HeadParams it was built with),
one row per image with a DLL pose. Those are then what every metric measures, the lag and the rest
jitter included; the twin, run with the constants given here, gives the unfiltered rows only (the
CSV has no unfiltered pose) and is compared with the CSV, image by image. Give the constants the
daemon was built with and the fits of its tracker, or the comparison says they differ, and the
lag reference (below) is mapped with other constants than the poses it is the reference of.

The constants: --params gives fit.py's JSON, its head_params or with --loso each session's fold
(fitted without it; the fold is found by the session's clock offset and image count, which the
JSON records, so the sessions may come in any order and any subset of the fitted ones). Without
it, reference.FITTED: the study's constants with the choices head.rs ships, which HeadParams::FITTED
held until the constants were fitted again on the Rust tracker's fits (2026-10-05, README step 3):
give --params with that fit's JSON for the constants head.rs ships now.
--blend turns the eye correction on with the weight the fit found (without --params, the study's)
and changes nothing else; --rotation-filters swaps the one-euro filters. --broken runs a
deliberately broken pipeline, to see which gates catch it.

Images: every 0x50e image of the log. DLL-valid: the DLL's four flags set; ours: the estimator's
validity (G3). Subsets of the DLL-valid images (the errors are taken where ours is valid too):
ALL; BOTH / NONE (both / neither of the device's eyeball centres 0x17/0x18 in the last gaze frame
before the image); YAW20 (|DLL yaw| >= 20 deg); REACQ (the first 10 images of every DLL-valid
run) and REACQgap (the same but the session start's); COMB (|yaw| > 15 and (|pitch| > 8 or
|roll| > 10), DLL angles).
  validity  coverage = ours valid among the DLL-valid; over the images with a DLL pose:
            agreement, FV (DLL invalid, ours valid), FI (the reverse; no face or gated by G3); DLL
            loss events (maximal runs of DLL-invalid images) detected = overlapped by one of our
            invalid runs, onset / offset = our first / last invalid image - the DLL's.
  rotation  |wrap(ours - DLL)| per yxz angle, deg: median, p95; the median signed pitch error; the
            geodesic angle between the two rotations, median, p95. raw = unfiltered, final =
            filtered.
  position  the display frame, mm: median and p95 |ours - DLL| per axis, 3-D median; jitter =
            second-difference RMS ours / DLL.
  lag       the coherence-weighted phase delay over 0.4-2.6 Hz (128-image Hann segments in
            steps of 32, linear detrend, coherence > 0.2) behind a per-image reference, of ours
            and of the DLL's, ms; dlag = ours - DLL.
  rest jitter  the RMS of the residual of a 2-Hz zero-phase low-pass (Butterworth, order 4) over
            the stillest quarter of the 32-image windows, chosen by the reference's speed
            (3-Hz low-pass); ours / DLL.
The reference is the unfiltered pose (rotation, and the PnP branch's position) of a fit of each
image that the evaluated fits do not feed: --reference-fits, one per session in order, the fits of
a stateless run of a tracker (every image fitted on its own, the crop iterated on the image
itself), as a fits CSV or as the head pose study's .npz (devts, face, R3, t3_cm; its
work2/filter-identify/runs/sl_sN.npz). Without it the reference is the evaluated fits' own
unfiltered pose. Our filter then delays the reference's own noise too, which adds to our lag
(enough to fail G15 where the stateless reference passes), so the lag and the rest jitter are
printed for information only and their gates (G15-G18) are not checked.

The gates (gates.json, the file compare-dll --head checks) apply to a session whose clock offset
they name (and image count, where they give one); evaluate.py checks every gate, those marked as
compare-dll's too. A metric that could not be measured (NaN, on any axis) fails its gate. Exits 0
when every gate of every session was checked and passed, 1 when one fails, 3 when none fails but
some went unchecked (G15-G18 without --reference-fits, a session the gates have no thresholds
for); with --gates none, 0.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import reference as ref  # noqa: E402

GATES = os.path.join(HERE, "gates.json")
# The gates that need an independent lag reference.
FILTER_METRICS = (
    "rotation_lag_max_ms",
    "position_lag_max_ms",
    "rotation_rest_jitter_ratio",
    "position_rest_jitter_ratio",
)
GATE_KINDS = ("min", "max", "abs", "range")
# The exit status of each verdict on the gates: 2 is argparse's, for a bad command line.
EXIT = {"pass": 0, "fail": 1, "incomplete": 3}
SUBSETS = ("ALL", "BOTH", "NONE", "YAW20", "REACQ", "REACQgap")
CHANNELS = [("rot", 0), ("rot", 1), ("rot", 2), ("pos", 0), ("pos", 1), ("pos", 2)]


# ------------------------------------------------------------------ lag and rest jitter
def lowpass0(x, fc, ok=None):
    """A zero-phase Butterworth (order 4) low-pass over every finite run longer than 30 images."""
    bb, aa = butter(4, fc / (C.FS / 2))
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) if ok is None else ok
    for a, ln in zip(*C.runs_of(ok)):
        if ln > 30:
            out[a : a + ln] = filtfilt(bb, aa, x[a : a + ln])
    return out


def still_mask(reference, ok):
    """The stillest quarter of the 32-image windows, by the median speed of the reference."""
    lp = lowpass0(np.where(ok, reference, np.nan), 3.0, ok)
    sp = np.abs(np.gradient(lp)) * C.FS
    n = len(reference)
    wins = []
    # Every whole window, the last one included. The study's code stopped short of [n - 32, n)
    # when n is a multiple of 32, which none of the Windows sessions is (n % 32 = 4, 20, 10): the
    # gates set from them hold either way.
    for a in range(0, n - 31, 32):
        if ok[a : a + 32].all() and np.isfinite(sp[a : a + 32]).all():
            wins.append((np.median(sp[a : a + 32]), a))
    wins.sort()
    m = np.zeros(n, bool)
    for _, a in wins[: max(1, len(wins) // 4)]:
        m[a : a + 32] = True
    return m


def jitter(x, m):
    """The RMS of x's 2-Hz high-pass on the images of m (NaN with 50 or fewer)."""
    ok = np.isfinite(x)
    r = x - lowpass0(np.where(ok, x, np.nan), 2.0, ok)
    mm = m & np.isfinite(r)
    return np.sqrt(np.mean(r[mm] ** 2)) if mm.sum() > 50 else np.nan


def seg_ffts(sigs, ok, nseg=128, step=32):
    """The FFTs of the Hann-windowed, linearly detrended 128-image segments in steps of 32 inside
    the runs of ok: X (segments, signals, frequencies), the frequencies, the starts."""
    st = []
    for a, ln in zip(*C.runs_of(ok)):
        k = int(a)
        while k + nseg <= a + ln:
            st.append(k)
            k += step
    w = np.hanning(nseg)
    t = np.arange(nseg)
    f = np.fft.rfftfreq(nseg, 1 / C.FS)
    X = np.zeros((len(st), len(sigs), len(f)), complex)
    for i, a in enumerate(st):
        for j, x in enumerate(sigs):
            v = np.asarray(x[a : a + nseg], float)
            c = np.polyfit(t, v, 1)
            X[i, j] = np.fft.rfft((v - np.polyval(c, t)) * w)
    return X, f, st


def phase_delay(X, f, i_ref, i_sig, band=(0.4, 2.6), cmin=0.2):
    """The coherence-weighted phase delay (ms) of signal i_sig behind i_ref, and the mean
    coherence."""
    Sxy = np.sum(X[:, i_sig, :] * np.conj(X[:, i_ref, :]), 0)
    Sxx = np.sum(X[:, i_ref, :] * np.conj(X[:, i_ref, :]), 0).real
    Syy = np.sum(X[:, i_sig, :] * np.conj(X[:, i_sig, :]), 0).real
    c = np.abs(Sxy) ** 2 / (Sxx * Syy)
    m = (f >= band[0]) & (f <= band[1]) & (c > cmin)
    if m.sum() == 0:
        return np.nan, np.nan
    w = c[m] / (1 - np.minimum(c[m], 0.99))
    ph = -np.angle(Sxy[m])
    tau = np.sum(w * ph * f[m]) / np.sum(w * f[m] ** 2) / (2 * np.pi)
    return 1000 * tau, float(np.mean(c[m]))


def lag(reference, x):
    ok = np.isfinite(reference) & np.isfinite(x)
    X, f, st = seg_ffts([reference, x], ok)
    return phase_delay(X, f, 0, 1)[0] if len(st) >= 3 else np.nan


def read_reference(path):
    """A lag reference: a fits CSV, or the head pose study's stateless run (.npz: devts, face, R3,
    t3_cm)."""
    if not path.endswith(".npz"):
        return C.read_fits(path)
    z = np.load(path)
    return dict(
        devts=z["devts"].astype(np.int64),
        face=z["face"].astype(bool),
        R_cam=z["R3"],
        t_cam_mm=10.0 * z["t3_cm"],
    )


def per_image_reference(fits, o, P):
    """The unfiltered pose of each image's fit: yxz angles (deg) and the PnP branch's position (T,
    mm); NaN without a face."""
    n = o["n"]
    f = fits["face"]
    rot = np.full((n, 3), np.nan)
    pos = np.full((n, 3), np.nan)
    H = C.head_S_v(fits["R_cam"][f], np.asarray(P["Q"], float))
    rot[f] = np.degrees(C.euler_yxz_v(o["R_TS"] @ H))
    pos[f] = C.to_T(
        o, C.p_pnp_v(fits["t_cam_mm"][f], H, P["s"], P["d"], P["g_p"], P["ox_p"], P["oy_p"])
    )
    return rot, pos


def lag_jitter(o, ours, reference):
    """Per channel: the DLL's lag, ours, ours - DLL and the rest jitter ratio."""
    out = {}
    for kind, ax in CHANNELS:
        ref_ = reference[kind][:, ax]
        dll = (o["dll_rot"] if kind == "rot" else o["dll_pos"])[:, ax]
        m = still_mask(ref_, np.isfinite(ref_) & np.isfinite(dll))
        y = ours[kind][:, ax]
        ld, lo = lag(ref_, dll), lag(ref_, y)
        out[(kind, ax)] = dict(lag_dll=ld, lag=lo, dlag=lo - ld, jit=jitter(y, m) / jitter(dll, m))
    return out


# ------------------------------------------------------------------ ours, from compare-dll --head
# The columns of compare-dll --head's --csv read here (it has more).
OURS_COLUMNS = (
    "image",
    "device_ts_us",
    "dll_valid",
    "ours_valid",
    *(f"ours_pos_{c}_mm" for c in "xyz"),
    *(f"ours_rot_{c}_rad" for c in "xyz"),
)


def read_ours(path, o):
    """compare-dll --head's --csv of session o as per-image arrays: valid, pos (T, mm) and rot
    (yxz, deg), NaN where our pose is invalid or the image has no row; has, the images with a
    row. Its rows must be this log's images with a DLL pose, by index and device time, and the
    DLL's validity theirs: the same session, paired at the same clock offset."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        missing = [c for c in OURS_COLUMNS if c not in (rd.fieldnames or [])]
        if missing:
            C.fail(f"{path}: not compare-dll --head's CSV (no {', '.join(missing)})")
        rows = list(rd)
    n = o["n"]
    idx = np.array([int(r["image"]) for r in rows], np.int64)
    devts = np.array([int(r["device_ts_us"]) for r in rows], np.int64)
    if ((idx < 0) | (idx >= n)).any() or not np.array_equal(o["devts"][idx], devts):
        C.fail(f"{path}: its rows are not images of {o['paths']['log']} (index, device time)")
    has = np.zeros(n, bool)
    has[idx] = True
    if len(np.unique(idx)) != len(idx) or not np.array_equal(has, o["dll_have"]):
        C.fail(
            f"{path}: its {len(idx)} rows are not the {int(o['dll_have'].sum())} images with a "
            "DLL pose (another clock offset?)"
        )
    if not np.array_equal(np.array([r["dll_valid"] == "1" for r in rows]), o["dll_valid"][idx]):
        C.fail(f"{path}: the DLL's validity in it is not this session's")
    ok = np.array([r["ours_valid"] == "1" for r in rows], bool)
    pos = np.array([[float(r[f"ours_pos_{c}_mm"]) for c in "xyz"] for r in rows]).reshape(-1, 3)
    rot = np.array([[float(r[f"ours_rot_{c}_rad"]) for c in "xyz"] for r in rows]).reshape(-1, 3)
    valid = np.zeros(n, bool)
    valid[idx] = ok
    out = dict(valid=valid, pos=np.full((n, 3), np.nan), rot=np.full((n, 3), np.nan), has=has)
    out["pos"][idx[ok]] = pos[ok]
    out["rot"][idx[ok]] = np.degrees(rot[ok])
    return out


def twin_difference(r, ours):
    """The Python twin's run r against the CSV's poses, on the images with a row: on how many
    their validity differs, and where both are valid the largest |difference| of the position
    (mm) and of each angle (deg, wrapped)."""
    both = r["valid"] & ours["valid"]
    dpos = np.abs(r["pos"][both] - ours["pos"][both])
    drot = np.abs(C.wrap_deg(r["rot"][both] - ours["rot"][both]))
    return dict(
        validity=int(((r["valid"] != ours["valid"]) & ours["has"]).sum()),
        rows=int(ours["has"].sum()),
        both=int(both.sum()),
        pos_mm=float(dpos.max()) if len(dpos) else np.nan,
        rot_deg=float(drot.max()) if len(drot) else np.nan,
    )


# ------------------------------------------------------------------ validity
def validity(pred, o):
    """Agreement with the DLL's validity on the images with a DLL pose, and its loss events."""
    state = np.where(o["dll_valid"], 1, np.where(o["dll_have"], 0, -1))
    face = o["face"]
    has = state >= 0
    V = (state == 1) & has
    inv = (state == 0) & has
    fv = int((pred & inv).sum())
    fi = int((~pred & V).sum())
    fi_face = int((~pred & V & ~face).sum())
    nh = int(has.sum())
    ds, dl = C.runs_of(inv)
    ps, pl = C.runs_of(~pred & has)
    pe = ps + pl
    ev = []
    for a, ln in zip(ds, dl):
        b = a + ln
        ov = np.flatnonzero((ps < b) & (pe > a))
        if len(ov) == 0:
            ev.append((int(a), int(ln), None, None, 0))
            continue
        ev.append(
            (int(a), int(ln), int(ps[ov[0]] - a), int(pe[ov[-1]] - b), int((~pred[a:b]).sum()))
        )
    false_ev = [int(ln) for a, ln in zip(ps, pl) if not inv[a : a + ln].any()]
    return dict(
        n=nh,
        nV=int(V.sum()),
        nI=int(inv.sum()),
        FV=fv,
        FI=fi,
        FI_face=fi_face,
        FI_gate=fi - fi_face,
        agree=1 - (fv + fi) / nh if nh else np.nan,
        events=ev,
        n_det=sum(1 for e in ev if e[2] is not None),
        false_ev=len(false_ev),
        false_ev3=sum(1 for ln in false_ev if ln >= 3),
    )


# ------------------------------------------------------------------ one session
def evaluate(o, P, args, ref_fits, ours=None):
    """Every metric of one session: of the Python twin's poses, or of ours, the poses of
    compare-dll --head's CSV (read_ours), when given."""
    Pr = dict(P)
    dec = "yxz"
    if args.broken == "zyx":
        dec = "zyx"
    elif args.broken == "q-transposed":
        Pr["Q"] = np.asarray(P["Q"], float).T
    elif args.broken == "no-q":
        Pr["Q"] = np.eye(3)
    r = C.run_reference(o, Pr, rule=args.rule, decomposition=dec)
    stage = "raw" if args.broken == "no-filter" else "final"
    if ours is not None:
        rot, pos, val = ours["rot"], ours["pos"], ours["valid"]
    elif stage == "final":
        rot, pos, val = r["rot"], r["pos"], r["valid"]
    else:
        rot, pos, val = r["raw_rot"], r["raw_pos"], r["valid"]
    res = dict(stage=stage, independent_reference=ref_fits is not None, csv=ours is not None)
    if ours is not None:
        res["twin"] = twin_difference(r, ours)
    vd = o["dll_valid"]
    res["coverage"] = (val & vd).sum() / max(vd.sum(), 1)
    res["validity"] = validity(val, o)
    # rotation; raw is always the twin's, on its own validity
    rows = {}
    for sub in ("ALL", "COMB"):
        for st, ang, v in (("raw", r["raw_rot"], r["valid"]), ("final", rot, val)):
            m = o["masks"][sub] & v
            e = C.wrap_deg(ang[m] - o["dll_rot"][m])
            g = C.geo_deg(C.compose_yxz_v(np.radians(ang[m])), o["R_dll"][m])
            ok = len(e) > 0
            rows[(sub, st)] = dict(
                n=int(m.sum()),
                med=np.median(np.abs(e), 0) if ok else np.full(3, np.nan),
                p95=np.percentile(np.abs(e), 95, 0) if ok else np.full(3, np.nan),
                bias=float(np.median(e[:, 0])) if ok else np.nan,
                geo_med=float(np.median(g)) if ok else np.nan,
                geo_p95=float(np.percentile(g, 95)) if ok else np.nan,
            )
    res["rotation"] = rows
    # position
    prow = {}
    for sub in SUBSETS:
        M = o["masks"][sub]
        m = M & val
        e = pos[m] - o["dll_pos"][m]
        ok = len(e) > 0
        prow[sub] = dict(
            n=int(M.sum()),
            cov=float(m.sum() / M.sum()) if M.sum() else np.nan,
            med=np.median(np.abs(e), 0) if ok else np.full(3, np.nan),
            p95=np.percentile(np.abs(e), 95, 0) if ok else np.full(3, np.nan),
            med3=float(np.median(np.linalg.norm(e, axis=1))) if ok else np.nan,
        )
    both = val & vd
    prow["jitter"] = C.d2rms(pos, both) / C.d2rms(o["dll_pos"], both)
    res["position"] = prow
    # lag and rest jitter, against the reference made with the good constants
    rr, rp = per_image_reference(ref_fits if ref_fits is not None else o, o, P)
    reference = dict(rot=rr, pos=rp)
    res["lag"] = lag_jitter(o, dict(rot=rot, pos=pos), reference)
    res["lag_raw"] = lag_jitter(o, dict(rot=r["raw_rot"], pos=r["raw_pos"]), reference)
    # the gates' metrics
    a = rows[("ALL", stage)]
    c = rows[("COMB", stage)]
    p = prow["ALL"]
    lj = res["lag"]
    # Per axis; np.max and np.min keep a NaN (a lag or a jitter that could not be measured), so
    # that the gate fails, where max() and min() would skip it unless it came first.
    dlag = {kind: np.array([lj[(kind, k)]["dlag"] for k in range(3)]) for kind in ("rot", "pos")}
    jit = {kind: np.array([lj[(kind, k)]["jit"] for k in range(3)]) for kind in ("rot", "pos")}
    res["metrics"] = {
        "coverage_pct": 100.0 * res["coverage"],
        "agreement_pct": 100.0 * res["validity"]["agree"],
        "loss_events_found": res["validity"]["n_det"],
        "rotation_median_abs_x_deg": float(a["med"][0]),
        "rotation_median_abs_y_deg": float(a["med"][1]),
        "rotation_median_abs_z_deg": float(a["med"][2]),
        "rotation_median_signed_x_deg": a["bias"],
        "rotation_geodesic_p95_deg": a["geo_p95"],
        "combined_rotation_p95_abs_x_deg": float(c["p95"][0]),
        "combined_rotation_geodesic_p95_deg": c["geo_p95"],
        "position_median_abs_x_mm": float(p["med"][0]),
        "position_median_abs_y_mm": float(p["med"][1]),
        "position_median_abs_z_mm": float(p["med"][2]),
        "position_p95_abs_z_mm": float(p["p95"][2]),
        "rotation_lag_max_ms": float(np.max(np.abs(dlag["rot"]))),
        "position_lag_max_ms": float(np.max(np.abs(dlag["pos"]))),
        "rotation_rest_jitter_ratio": [float(np.min(jit["rot"])), float(np.max(jit["rot"]))],
        "position_rest_jitter_ratio": [float(np.min(jit["pos"])), float(np.max(jit["pos"]))],
    }
    return res


# ------------------------------------------------------------------ the gates
def check(kind, gate, val):
    if kind == "min":
        return val >= gate
    if kind == "max":
        return val <= gate
    if kind == "abs":
        return abs(val) <= gate
    return gate[0] <= val[0] and val[1] <= gate[1]


def fmt(val):
    if isinstance(val, (list, tuple)):
        return f"{val[0]:.2f}..{val[1]:.2f}"
    return f"{val:.2f}" if isinstance(val, float) else str(val)


def headroom(kind, gate, val):
    if kind == "min":
        return f"{val - gate:+.2f}"
    if kind == "max":
        return f"{gate - val:+.2f}"
    if kind == "abs":
        return f"{gate - abs(val):+.2f}"
    return f"{val[0] - gate[0]:+.2f} / {gate[1] - val[1]:+.2f}"


def gate_session(gates, o):
    """The gates' name for session o, matched by clock offset (and image count, where the file
    gives one), or None."""
    for s in gates["sessions"]:
        if s["clock_offset_us"] == o["k"] and s.get("images", o["n"]) == o["n"]:
            return s["name"]
    return None


def measurable(val):
    """Whether a metric's value is a number (both ends of a range)."""
    return bool(np.isfinite(np.asarray(val, float)).all())


def print_gates(gates, sessions, results):
    """Check the gates file (compare-dll --head's: the sessions by clock offset, per gate its
    metric, kind and thresholds per session; evaluate.py checks every gate, whoever its "by"
    names) and print the table. Returns "fail" when a gate fails (a value that is not a number
    fails), "incomplete" when none fails but a gate or a whole session went unchecked, else
    "pass"."""
    names = [gate_session(gates, o) for o in sessions]
    no_gates = [o["name"] for o, g in zip(sessions, names) if g is None]
    for o, g in zip(sessions, names):
        if g is None:
            print(f"\n{o['name']}: no gates (none for clock offset {o['k']} and {o['n']} images)")
    cols = [(o, g, res) for o, g, res in zip(sessions, names, results) if g is not None]
    rows, failed, unchecked = [], [], []
    for gt in gates["gates"] if cols else ():
        if gt["kind"] not in GATE_KINDS:
            C.fail(f"gate {gt['id']}: unknown kind {gt['kind']!r}")
        if gt["metric"] not in results[0]["metrics"]:
            C.fail(f"gate {gt['id']}: no metric {gt['metric']!r} here")
        cells = [f"{gt['id']} {gt['name']}", gt["kind"]]
        for o, g, res in cols:
            val, th = res["metrics"][gt["metric"]], gt["thresholds"].get(g)
            if th is None:
                unchecked.append(f"{gt['id']} {o['name']}")
                cells.append(f"{fmt(val)}: not checked (no threshold for {g})")
                continue
            if gt["metric"] in FILTER_METRICS and not res["independent_reference"]:
                unchecked.append(f"{gt['id']} {o['name']}")
                cells.append(f"{fmt(val)}: not checked (no --reference-fits)")
                continue
            ok = measurable(val) and check(gt["kind"], th, val)
            if not ok:
                failed.append(f"{gt['id']} {o['name']}")
            th_s = f"{th[0]}..{th[1]}" if isinstance(th, list) else f"{th}"
            if not measurable(val):
                cells.append(f"{fmt(val)} vs {th_s}: FAIL (not measurable)")
                continue
            word = "pass" if ok else "FAIL"
            cells.append(f"{fmt(val)} vs {th_s}: {word} ({headroom(gt['kind'], th, val)})")
        rows.append(cells)
    if cols:
        print("\n## Gates: value (headroom, + = passes)")
        heads = [o["name"] if g == o["name"] else f"{o['name']} (gates' {g})" for o, g, _ in cols]
        C.table(["gate", "kind", *heads], rows)
    if failed:
        verdict = "fail"
        line = "FAIL " + ", ".join(failed)
    elif unchecked or no_gates:
        verdict = "incomplete"
        line = "INCOMPLETE (none failed, but not every gate was checked)"
    else:
        verdict = "pass"
        line = "all pass"
    print(
        "\ngates: "
        + line
        + (f"; not checked: {', '.join(unchecked)}" if unchecked else "")
        + (f"; no gates for {', '.join(no_gates)}" if no_gates else "")
    )
    return verdict


# ------------------------------------------------------------------ the report
def event_stat(v, i):
    det = [e for e in v["events"] if e[2] is not None]
    if not det:
        return "-"
    return f"{np.median([abs(e[i]) for e in det]):.1f}/{max(abs(e[i]) for e in det)}"


def report(sessions, results):
    names = " / ".join(o["name"] for o in sessions)
    V = [res["validity"] for res in results]
    if results[0]["csv"]:
        print(
            "\nours: compare-dll --head's poses (--ours-csv); raw and unfiltered: the Python "
            "twin's (the CSV has no unfiltered pose)"
        )
    print("\n## Validity (G3) against the DLL's flags")
    C.table(
        ["quantity", names],
        [
            [
                "coverage of the DLL-valid images",
                C.tri([100 * res["coverage"] for res in results], "{:.2f}%"),
            ],
            [
                "agreement on the images with a DLL pose",
                C.tri([100 * v["agree"] for v in V], "{:.2f}%"),
            ],
            [
                "FV: ours valid / DLL-invalid images",
                C.tri([f"{v['FV']}/{v['nI']}" for v in V], "{}"),
            ],
            [
                "FI: ours invalid / DLL-valid images (no face, gated)",
                C.tri([f"{v['FI']}/{v['nV']} ({v['FI_face']}, {v['FI_gate']})" for v in V], "{}"),
            ],
            [
                "DLL loss events detected",
                C.tri([f"{v['n_det']}/{len(v['events'])}" for v in V], "{}"),
            ],
            [
                "onset, |ours - DLL| median / max (images)",
                C.tri([event_stat(v, 2) for v in V], "{}"),
            ],
            [
                "offset, |ours - DLL| median / max (images)",
                C.tri([event_stat(v, 3) for v in V], "{}"),
            ],
            [
                "false-invalid runs (3 images or more)",
                C.tri([f"{v['false_ev']} ({v['false_ev3']})" for v in V], "{}"),
            ],
        ],
    )
    for o, v in zip(sessions, V):
        for a, ln, on, off, cov in v["events"]:
            if on is None:
                what = "NOT detected"
            else:
                what = f"onset {on:+d}, offset {off:+d}, ours invalid on {cov}/{ln}"
            inside = slice(a, a + ln)
            print(
                f"  {o['name']} DLL-invalid run [{a}, {a + ln}), {ln} images: {what}; faces inside "
                f"{int(o['face'][inside].sum())}, G3-valid inside {int(o['g3'][inside].sum())}"
            )

    print(f"\n## Rotation against the DLL's, deg ({names})")
    rows = []
    stages = ("raw", "final") if results[0]["stage"] == "final" else ("raw",)
    for sub in ("ALL", "COMB"):
        for st in stages:
            R = [res["rotation"][(sub, st)] for res in results]
            rows.append(
                [sub, st, C.tri([r["n"] for r in R], "{}")]
                + [C.tri([r["med"][k] for r in R]) for k in range(3)]
                + [C.tri([r["p95"][k] for r in R]) for k in range(3)]
                + [C.tri([r["geo_med"] for r in R]), C.tri([r["geo_p95"] for r in R])]
                + [C.tri([r["bias"] for r in R], "{:+.2f}")]
            )
    errs = ["median |e| x", "median |e| y", "median |e| z", "p95 x", "p95 y", "p95 z"]
    C.table(
        ["subset", "stage", "n", *errs, "geodesic median", "geodesic p95", "median signed x"], rows
    )

    print(f"\n## Position against the DLL's, display frame, mm, {results[0]['stage']} ({names})")
    rows = []
    for sub in SUBSETS:
        R = [res["position"][sub] for res in results]
        rows.append(
            [sub, C.tri([r["n"] for r in R], "{}"), C.tri([100 * r["cov"] for r in R], "{:.1f}%")]
            + [C.tri([r["med"][k] for r in R], "{:.1f}") for k in range(3)]
            + [C.tri([r["p95"][k] for r in R], "{:.1f}") for k in range(3)]
            + [C.tri([r["med3"] for r in R], "{:.1f}")]
        )
    C.table(["subset", "n", "coverage", *errs, "3-D median"], rows)
    J = [res["position"]["jitter"] for res in results]
    print(
        "second-difference jitter, ours / DLL, x ; y ; z: "
        + " ; ".join(C.tri([j[k] for j in J]) for k in range(3))
    )

    if all(r["independent_reference"] for r in results):
        which = "an independent reference (--reference-fits)"
    else:
        which = "the fits' own unfiltered pose (biased: see --help)"
    print(f"\n## Lag (ms) behind {which}, and rest jitter, ours / DLL ({names})")
    rows = []
    for kind, ax in CHANNELS:
        A = [res["lag"][(kind, ax)] for res in results]
        B = [res["lag_raw"][(kind, ax)] for res in results]
        rows.append(
            [
                ("rotation " if kind == "rot" else "position ") + "xyz"[ax],
                C.tri([a["lag_dll"] for a in A], "{:.1f}"),
                C.tri([a["lag"] for a in A], "{:.1f}"),
                C.tri([a["dlag"] for a in A], "{:+.1f}"),
                C.tri([a["jit"] for a in A], "x{:.2f}"),
                C.tri([b["dlag"] for b in B], "{:+.1f}"),
                C.tri([b["jit"] for b in B], "x{:.2f}"),
            ]
        )
    C.table(
        [
            "channel",
            "DLL lag",
            "ours: lag",
            "dlag (ours - DLL)",
            "rest jitter",
            "unfiltered: dlag",
            "rest jitter",
        ],
        rows,
    )


# ------------------------------------------------------------------ main
def fit_names(doc, o):
    """The names fit.py gave session o in its JSON (its own s1, s2, ... by its argument order,
    which need not be this one), found as the gates are: by clock offset and image count."""
    return [
        name
        for name, s in doc.get("sessions", {}).items()
        if s.get("clock_offset_us") == o["k"] and s.get("images") == o["n"]
    ]


def constants(args, o):
    """The constants for session o, and a line saying what they are."""
    if args.params:
        P, doc = C.load_params(args.params)
        what = os.path.basename(args.params)
        fit = doc.get("fit", {})
        names = fit_names(doc, o)
        if args.loso:
            if not names:
                C.fail(
                    f"{args.params} was not fitted on {o['paths']['log']} (clock offset "
                    f"{o['k']} us, {o['n']} images): none of its folds leaves this session out"
                )
            fold = doc.get("folds", {}).get(names[0])
            if fold is None:
                C.fail(f"{args.params} has no fold that leaves out its {names[0]}")
            if any(t in names for t in fold["trained_on"]):
                C.fail(
                    f"{args.params}: fold {names[0]} was fitted on this session too (fit.py was "
                    f"given it as {' and '.join(names)})"
                )
            P = ref.params_from_head_params(fold["head_params"])
            fit = fold.get("fit", {})
            what += (
                f" fold {names[0]} (leaves out this session, the fit's {names[0]}; fitted on "
                f"{', '.join(fold['trained_on'])})"
            )
        elif "sessions" in doc:
            what += (
                f" (fitted on this session too, the fit's {names[0]})"
                if names
                else " (not fitted on this session)"
            )
        if args.blend:
            if "eye_weight" not in fit:
                C.fail("--blend needs fit.py's JSON: it takes the eye weight the fit found")
            P["w"] = fit["eye_weight"]
    else:
        # PROTOTYPE differs from FITTED in its rotation filters too: only the weight is taken.
        P = dict(ref.FITTED)
        what = "reference.FITTED (the study's constants, HeadParams::FITTED before the refit)"
        if args.blend:
            P["w"] = ref.PROTOTYPE["w"]
            what += " with the study's eye weight"
    if args.rotation_filters != "params":
        src = ref.ONE_EURO_PER_AXIS if args.rotation_filters == "per-axis" else ref.ONE_EURO_SHARED
        P["one_euro"] = {k: list(v) for k, v in src.items()}
    filters = ", ".join(f"{k} ({v[0]:g}, {v[1]:g})" for k, v in P["one_euro"].items())
    return P, f"{what}: eye weight {P['w']:.4f}, one-euro {filters}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_session_args(ap)
    ap.add_argument(
        "--params",
        help="fit.py's JSON; default: reference.FITTED, the study's constants (HeadParams::FITTED "
        "before the refit on the Rust tracker)",
    )
    ap.add_argument(
        "--loso",
        action="store_true",
        help="each session with its fold of --params (fitted without it)",
    )
    ap.add_argument(
        "--blend",
        action="store_true",
        help="the eye correction on, with the weight the fit found (without --params, the "
        "study's); nothing else changes",
    )
    ap.add_argument(
        "--rotation-filters",
        choices=("params", "shared", "per-axis"),
        default="params",
        help="the one-euro filters: the constants', one shared by the angles (1.0, 0.2), or the "
        "study's per-axis ones",
    )
    ap.add_argument(
        "--rule",
        choices=("timeaware", "reset"),
        default="timeaware",
        help="the filters' reset rule (head.rs: timeaware)",
    )
    ap.add_argument(
        "--reference-fits",
        action="append",
        metavar="REF",
        help="the lag reference: a stateless run's fits, one per session in order (a fits CSV or "
        "the study's .npz); without it the lag gates are not checked",
    )
    ap.add_argument(
        "--ours-csv",
        action="append",
        metavar="CSV",
        help="our poses: compare-dll --head's --csv, one per session in order (the daemon's own "
        "pipeline), evaluated in place of the Python twin's, which is compared with them",
    )
    ap.add_argument(
        "--broken",
        choices=("zyx", "q-transposed", "no-q", "no-filter"),
        help="evaluate a broken pipeline instead (the angles read as zyx, Q transposed, Q left "
        "out, the unfiltered pose)",
    )
    ap.add_argument(
        "--gates", default=GATES, help="the gates' JSON, or 'none' (default: gates.json here)"
    )
    args = ap.parse_args()
    if args.loso and not args.params:
        C.fail("--loso needs --params")
    if args.reference_fits and len(args.reference_fits) != len(args.session):
        C.fail(f"{len(args.reference_fits)} --reference-fits for {len(args.session)} sessions")
    if args.ours_csv:
        if len(args.ours_csv) != len(args.session):
            C.fail(f"{len(args.ours_csv)} --ours-csv for {len(args.session)} sessions")
        # The CSV is what the daemon made with its own constants: neither a fold's nor broken.
        if args.loso or args.broken:
            C.fail("--ours-csv evaluates the daemon's poses: no --loso or --broken with it")
    sessions = C.load_sessions(args)
    results = []
    for i, o in enumerate(sessions):
        P, what = constants(args, o)
        ref_fits = None
        if args.reference_fits:
            ref_fits = read_reference(args.reference_fits[i])
            if not np.array_equal(ref_fits["devts"], o["devts"]):
                C.fail(f"{args.reference_fits[i]}: not the images of {o['paths']['log']}")
        lag_ref = (
            os.path.basename(args.reference_fits[i])
            if ref_fits is not None
            else "its own fits, unfiltered"
        )
        ours = read_ours(args.ours_csv[i], o) if args.ours_csv else None
        print(
            f"{o['name']}: {what}; rule {args.rule}"
            + (f"; BROKEN: {args.broken}" if args.broken else "")
            + f"; lag reference: {lag_ref}"
            + (f"; ours: {os.path.basename(args.ours_csv[i])}" if ours is not None else ""),
            flush=True,
        )
        res = evaluate(o, P, args, ref_fits, ours)
        if ours is not None:
            t = res["twin"]
            # The twin and head.rs agree to ~1e-12 on the same fits and constants.
            same = t["validity"] == 0 and t["pos_mm"] <= 1e-6 and t["rot_deg"] <= 1e-6
            print(
                "  the Python twin with these constants against the CSV: validity differs on "
                f"{t['validity']} of its {t['rows']} images; where both are "
                f"valid ({t['both']}), |position| up to {t['pos_mm']:.1e} mm, |angle| up to "
                f"{t['rot_deg']:.1e} deg"
                + (
                    ""
                    if same
                    else "; WARNING: not the CSV's pipeline (other constants, fits or display "
                    "frame), so the lag reference is not mapped as its poses are"
                ),
                flush=True,
            )
        results.append(res)
    report(sessions, results)
    verdict = "pass"
    if args.gates != "none":
        with open(args.gates) as f:
            gates = json.load(f)
        if not (isinstance(gates.get("sessions"), list) and isinstance(gates.get("gates"), list)):
            C.fail(f'{args.gates}: not a gates file (a "sessions" and a "gates" list)')
        verdict = print_gates(gates, sessions, results)
    sys.exit(EXIT[verdict])


if __name__ == "__main__":
    main()
