#!/usr/bin/env python3
"""Fit the head pose constants to the Stream Engine's own head pose: leave-one-session-out (LOSO)
and on all the sessions given.

    fit.py --session LOG JSONL FITS [--session ...] [--area A] [--out FIT.json]
           [--in-sample] [--blend] [--rotation-filters shared|per-axis]

The images fitted on are those where the DLL's pose is valid and G3 holds for the fit (where both
give a pose). In each fold (one per session, fitted on the others; and one on all of them):
  Q         B = argmin sum |R_dll - A R_cam B|_F (Procrustes), A = R_TS D; Q = D B.
  position  the eye branch (K, g_e, ox_e, oy_e) and the PnP branch (s, d, g_p, ox_p, oy_p), each by
            robust least squares (soft_l1, f_scale 5 mm, residuals in T) of the unfiltered
            per-image position against the DLL's; then the eye weight w by linear least squares
            on the unfiltered positions.
  beta      the eye correction's low-pass, as the share of the way it moves per image (30.208 ms):
            the largest in BETA_GRID whose jitter on the training sessions (second-difference RMS,
            ours / DLL, mean over x, y, z and the sessions) is at most 1.05 x the PnP branch's
            alone. Accuracy hardly depends on it; jitter does.
The one-euro filters and the position's time constant are not fitted here (the filter study chose
them); nor are the G3 thresholds.

Prints each fold's constants and its beta grid, then every session's errors against the DLL with
the constants fitted on the others (with --in-sample also with those fitted on all). Writes
FIT.json: "head_params", the all-session constants under the field names of head.rs's HeadParams,
with the choices it ships (the PnP branch alone, w = 0, unless --blend; one one-euro filter shared
by the three angles unless --rotation-filters per-axis): what HeadParams::FITTED takes. "fit" has
the fit's own numbers (the eye weight found, beta, the costs), "folds" the same per held-out
session.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import reference as ref  # noqa: E402

BETA_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0)
JIT_TOL = 1.05


# ------------------------------------------------------------------ the fits
def fit_Q(S_list, use_key="g3"):
    """Procrustes: B = argmin sum |R_dll - A R_cam B| over the DLL-valid and our-valid images;
    Q = D B."""
    M = np.zeros((3, 3))
    for o in S_list:
        m = o["dll_valid"] & o[use_key]
        A = o["R_TS"] @ C.D_
        M += np.einsum("nji,njk->ik", A @ o["R_cam"][m], o["R_dll"][m])
    return C.D_ @ C.procrustes(M)


def fit_position(S_list, Q, use_key="g3"):
    """The two position branches by robust least squares against the DLL's position (T, mm), then
    the eye weight w by linear least squares (S)."""
    pre = []
    for o in S_list:
        m = o["dll_valid"] & o[use_key]
        H = C.head_S_v(o["R_cam"][m], Q)
        pre.append(
            (o, o["t_cam_mm"][m], H, C.eye_geom(o["eye_a"][m], o["eye_b"][m], H), o["dll_pos"][m])
        )

    def res_eye(x):
        return np.concatenate(
            [(C.to_T(o, C.p_eye_v(gm, *x)) - y).ravel() for o, t, H, gm, y in pre]
        )

    def res_pnp(x):
        return np.concatenate(
            [
                (C.to_T(o, C.p_pnp_v(t, H, x[0], x[1:4], x[4], x[5], x[6])) - y).ravel()
                for o, t, H, gm, y in pre
            ]
        )

    re = least_squares(
        res_eye, np.array([65.5, 1.0, 0.0, 0.0]), loss="soft_l1", f_scale=5.0, x_scale="jac"
    )
    rp = least_squares(
        res_pnp,
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        loss="soft_l1",
        f_scale=5.0,
        x_scale="jac",
    )
    K, g_e, ox_e, oy_e = re.x
    s_, d0, d1, d2, g_p, ox_p, oy_p = rp.x
    num = den = 0.0
    for o, t, H, gm, y in pre:
        pe = C.p_eye_v(gm, *re.x)
        pp = C.p_pnp_v(t, H, *rp.x[:1], rp.x[1:4], *rp.x[4:])
        dl = pe - pp
        r = C.to_S(o, y) - pp
        num += np.sum(dl * r)
        den += np.sum(dl * dl)
    w = num / den
    return dict(
        K=float(K),
        g_e=float(g_e),
        ox_e=float(ox_e),
        oy_e=float(oy_e),
        s=float(s_),
        d=[float(d0), float(d1), float(d2)],
        g_p=float(g_p),
        ox_p=float(ox_p),
        oy_p=float(oy_p),
        w=float(w),
        fit_cost_eye=float(re.cost),
        fit_cost_pnp=float(rp.cost),
    )


def pos_chain(o, Q, P, beta_a, rule="timeaware"):
    """The reference's position chain over a session, vectorised: the filtered and the unfiltered
    position (T, mm; NaN where G3 fails)."""
    v = o["g3"]
    n = o["n"]
    H = np.full((n, 3, 3), np.nan)
    H[v] = C.head_S_v(o["R_cam"][v], Q)
    pp = np.full((n, 3), np.nan)
    pp[v] = C.p_pnp_v(o["t_cam_mm"][v], H[v], P["s"], P["d"], P["g_p"], P["ox_p"], P["oy_p"])
    cor = np.zeros((n, 3))
    if P["w"] != 0:
        pe = np.full((n, 3), np.nan)
        pe[v] = C.p_eye_v(
            C.eye_geom(o["eye_a"][v], o["eye_b"][v], H[v]), P["K"], P["g_e"], P["ox_e"], P["oy_e"]
        )
        cor = P["w"] * (pe - pp)
    cl = C.filt_series(
        o["t_us"], v, np.where(v[:, None], cor, 0.0), "ema", ref.tau_of_a(beta_a), rule
    )
    y = C.filt_series(o["t_us"], v, pp + np.nan_to_num(cl), "ema", ref.tau_of_a(0.30), rule)
    return C.to_T(o, y), C.to_T(o, pp + cor)


def score(o, pT):
    """The median |error| per axis of pT against the DLL's position, and its jitter ratio."""
    m = o["dll_valid"] & np.isfinite(pT).all(1)
    e = pT[m] - o["dll_pos"][m]
    return np.median(np.abs(e), 0), C.d2rms(pT, m) / C.d2rms(o["dll_pos"], m)


def fit_fold(S):
    """One fold: Q, the position constants and beta, fitted on the sessions S."""
    Q = fit_Q(S)
    pos = fit_position(S, Q)
    grid = []
    P0 = dict(pos, w=0.0)
    j0 = np.mean([score(o, pos_chain(o, Q, P0, 1.0)[0])[1] for o in S], 0)
    for b in BETA_GRID:
        acc, jit = [], []
        for o in S:
            a, j = score(o, pos_chain(o, Q, pos, b)[0])
            acc.append(a)
            jit.append(j)
        grid.append(dict(beta=b, med=np.mean(acc, 0).tolist(), jit=np.mean(jit, 0).tolist()))
    ok = [g for g in grid if np.mean(g["jit"]) <= JIT_TOL * np.mean(j0)]
    beta = max(g["beta"] for g in ok) if ok else min(BETA_GRID)
    return dict(
        train=[o["name"] for o in S],
        Q=Q.tolist(),
        Q_yxz_deg_xyz=np.degrees(ref.euler_yxz(Q)).tolist(),
        pos=pos,
        beta=beta,
        pnp_only_jit=j0.tolist(),
        beta_grid=grid,
        n_frames=int(sum((o["dll_valid"] & o["g3"]).sum() for o in S)),
    )


def params_of(fold, blend, per_axis):
    """The fold's constants as the reference takes them, with the choices to ship."""
    filters = ref.ONE_EURO_PER_AXIS if per_axis else ref.ONE_EURO_SHARED
    P = ref.make_params(
        np.array(fold["Q"]),
        fold["pos"],
        beta_a=fold["beta"],
        one_euro={k: list(v) for k, v in filters.items()},
    )
    if not blend:
        P["w"] = 0.0
    return P


# ------------------------------------------------------------------ the held-out errors
def rotation_errors(o, r, stage):
    m = o["masks"]["ALL"] & r["valid"]
    ang = r["rot"] if stage == "final" else r["raw_rot"]
    e = np.abs(C.wrap_deg(ang[m] - o["dll_rot"][m]))
    g = C.geo_deg(C.compose_yxz_v(np.radians(ang[m])), o["R_dll"][m])
    return dict(
        n=int(m.sum()),
        med=np.median(e, 0),
        p95=np.percentile(e, 95, 0),
        geo_med=float(np.median(g)),
        geo_p95=float(np.percentile(g, 95)),
    )


def position_errors(o, r):
    M = o["masks"]["ALL"]
    m = M & r["valid"]
    e = r["pos"][m] - o["dll_pos"][m]
    ok = r["valid"] & o["dll_valid"]
    return dict(
        n=int(M.sum()),
        cov=float(m.sum() / max(M.sum(), 1)),
        med=np.median(np.abs(e), 0),
        p95=np.percentile(np.abs(e), 95, 0),
        med3=float(np.median(np.linalg.norm(e, axis=1))),
        jit=C.d2rms(r["pos"], ok) / C.d2rms(o["dll_pos"], ok),
    )


def held_out_tables(sessions, fold_of, labels, per_axis):
    """Rotation and position errors of every session with the constants fold_of[label](session)."""
    rows_r, rows_p = [], []
    for label in labels:
        R = {"raw": [], "final": []}
        Pos = {"PnP-only": [], "BLlp": []}
        for o in sessions:
            fold = fold_of[label](o)
            for model in ("PnP-only", "BLlp"):
                P = params_of(fold, model == "BLlp", per_axis)
                r = C.run_reference(o, P)
                Pos[model].append(position_errors(o, r))
                if model == "BLlp":
                    for stage in ("raw", "final"):
                        R[stage].append(rotation_errors(o, r, stage))
        for stage in ("raw", "final"):
            rr = R[stage]
            rows_r.append(
                [label, stage, C.tri([x["n"] for x in rr], "{}")]
                + [C.tri([x["med"][k] for x in rr]) for k in range(3)]
                + [C.tri([x["p95"][k] for x in rr]) for k in range(3)]
                + [C.tri([x["geo_med"] for x in rr]), C.tri([x["geo_p95"] for x in rr])]
            )
        for model, pp in Pos.items():
            rows_p.append(
                [model, label, C.tri([x["n"] for x in pp], "{}")]
                + [C.tri([100 * x["cov"] for x in pp], "{:.1f}%")]
                + [C.tri([x["med"][k] for x in pp], "{:.1f}") for k in range(3)]
                + [C.tri([x["p95"][k] for x in pp], "{:.1f}") for k in range(3)]
                + [C.tri([x["med3"] for x in pp], "{:.1f}")]
                + [" ; ".join(C.tri([x["jit"][k] for x in pp]) for k in range(3))]
            )
    names = " / ".join(o["name"] for o in sessions)
    errs = ["median |e| x", "median |e| y", "median |e| z", "p95 x", "p95 y", "p95 z"]
    print(
        f"\n## Rotation against the DLL's (deg, {names}): the pipeline's yxz angles on the images "
        "where both are valid; raw = unfiltered, final = filtered"
    )
    C.table(["constants", "stage", "n", *errs, "geodesic median", "geodesic p95"], rows_r)
    print(
        f"\n## Position against the DLL's (display frame, mm, {names}), filtered; jitter = "
        "second-difference RMS, ours / DLL, x ; y ; z"
    )
    C.table(["model", "constants", "n", "coverage", *errs, "3-D median", "jitter"], rows_p)


# ------------------------------------------------------------------ main
def fold_json(fold, blend, per_axis):
    p = fold["pos"]
    return dict(
        trained_on=fold["train"],
        frames=fold["n_frames"],
        head_params=ref.head_params(params_of(fold, blend, per_axis)),
        fit=dict(
            eye_weight=p["w"],
            correction_share=fold["beta"],
            rotation_offset_yxz_deg=fold["Q_yxz_deg_xyz"],
            fit_cost_eye=p["fit_cost_eye"],
            fit_cost_pnp=p["fit_cost_pnp"],
            pnp_only_jitter=fold["pnp_only_jit"],
            beta_grid=fold["beta_grid"],
        ),
    )


def print_fold(label, r):
    q = r["Q_yxz_deg_xyz"]
    p = r["pos"]
    train = ", ".join(r["train"])
    print(
        f"== {label} (train {train}, {r['n_frames']} images): Q yxz (x, y, z) deg = "
        f"({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f})"
    )
    d = p["d"]
    print(
        f"   eye K {p['K']:.4f} g {p['g_e']:.5f} ox {p['ox_e']:+.6f} oy {p['oy_e']:+.6f} | "
        f"pnp s {p['s']:.5f} d ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}) g {p['g_p']:.5f} "
        f"ox {p['ox_p']:+.6f} oy {p['oy_p']:+.6f} | w {p['w']:.4f}"
    )
    print(
        "   beta grid (train mean of median |e| x/y/z mm; jitter ratio x/y/z); PnP-only jitter "
        + "/".join(f"{x:.3f}" for x in r["pnp_only_jit"])
        + f"; tolerance x{JIT_TOL} on the mean"
    )
    for g in r["beta_grid"]:
        mark = " <- chosen" if g["beta"] == r["beta"] else ""
        print(
            f"     beta {g['beta']:.2f}: med "
            + "/".join(f"{x:.2f}" for x in g["med"])
            + " | jit "
            + "/".join(f"{x:.3f}" for x in g["jit"])
            + f" (mean {np.mean(g['jit']):.3f}){mark}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_session_args(ap)
    ap.add_argument("--out", help="the JSON to write (head_params, fit, folds)")
    ap.add_argument(
        "--in-sample",
        action="store_true",
        help="also print every session's errors with the constants fitted on all the sessions",
    )
    ap.add_argument(
        "--blend",
        action="store_true",
        help="ship the eye correction with the weight the fit found (BLlp) instead of w = 0",
    )
    ap.add_argument(
        "--rotation-filters",
        choices=("shared", "per-axis"),
        default="shared",
        help="the one-euro filters to ship and to filter with: one shared by the three angles "
        "(1.0 Hz, 0.2 Hz per deg/s) or the study's per-axis ones",
    )
    args = ap.parse_args()
    per_axis = args.rotation_filters == "per-axis"
    sessions = C.load_sessions(args)
    print()
    folds = {}
    if len(sessions) > 1:
        for o in sessions:
            train = [t for t in sessions if t is not o]
            folds[o["name"]] = fit_fold(train)
            print_fold(f"fold {o['name']}", folds[o["name"]])
    every = fit_fold(sessions)
    print_fold("all", every)
    fold_of, labels = {}, []
    if folds:
        fold_of["LOSO"] = lambda o: folds[o["name"]]
        labels.append("LOSO")
    if args.in_sample or not folds:
        fold_of["in-sample"] = lambda o: every
        labels.append("in-sample")
    held_out_tables(sessions, fold_of, labels, per_axis)
    if args.out:
        doc = dict(
            about="Head pose constants fitted by tools/headpose/fit.py to the Stream Engine's "
            "head pose. head_params: fitted on all the sessions, under head.rs's HeadParams field "
            "names, with the choices to ship (options); fit: the fit's own numbers; folds: the "
            "same fitted without each session.",
            options=dict(
                blend=args.blend,
                rotation_filters=args.rotation_filters,
                area="given" if args.area is not None else "fitted to each log's gaze origins",
            ),
            sessions={
                o["name"]: dict(
                    log=os.path.basename(o["paths"]["log"]),
                    dll=os.path.basename(o["paths"]["dll"]),
                    fits=os.path.basename(o["paths"]["fits"]),
                    clock_offset_us=int(o["k"]),
                    images=int(o["n"]),
                    fit_images=int((o["dll_valid"] & o["g3"]).sum()),
                )
                for o in sessions
            },
            **fold_json(every, args.blend, per_axis),
            folds={name: fold_json(f, args.blend, per_axis) for name, f in folds.items()},
        )
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
