#!/usr/bin/env python3
"""Set the thresholds of the acceptance gates (gates.json) from the Rust pipeline's
leave-one-session-out values, by the rule below, and check them against the broken pipelines.

    gates.py --session LOG JSONL FITS [--session ...] [--area A] --params FIT.json
             --reference-fits REF [--reference-fits ...] [--gates GATES.json] [--out OUT.json]

The values are those evaluate.py measures of the Python twin of head.rs (which agrees with it to
~1e-12 on the same fits and constants) over the fits: with each session's fold of fit.py's JSON,
fitted without it (evaluate.py --params FIT.json --loso), and with the constants fitted on all of
them (--params FIT.json); for the good pipeline and for the four broken ones (--broken); the lag
and the rest jitter against --reference-fits, the stateless reference, one per session in order.
Each threshold is the good pipeline's leave-one-session-out value of the gate's metric, moved by:

  G1, G2    coverage and agreement: 0.5 percentage points down, rounded down to 0.1;
  G3        loss events found: one fewer, but none fewer in the sessions of EVERY_LOSS;
  G4-G6     rotation medians: 30 % of the value up, at least 0.25 deg, rounded up to 0.1;
  G7        |median signed pitch error|: as G4-G6; but in a session where the pipeline without Q
            would pass every gate so set (no other gate catches it there), midway between the two
            values, rounded to 0.1;
  G8-G10    rotation p95s: 20 % up, at least 1 deg, rounded up to 0.5;
  G11-G13   position medians: 30 % up, at least 0.5 mm, rounded up to 0.1;
  G14       position p95 z: 20 % up, at least 5 mm, rounded up to 1;
  G15, G16  lags: 5 ms up, rounded up to 1;
  G17, G18  rest jitter ratios: from 0.2 below the smallest to 0.3 above the largest, rounded out
            to 0.1, but from 0.8 or less to 1.2 or more: a ratio is ours over the DLL's, and 1,
            the DLL's own jitter, lies inside every range with room on both sides.

Then the check: the good pipeline must pass every gate in every session, and each broken one fail
at least one in every session, with either constants. It prints each gate's values and thresholds
(GATES.json's own in parentheses), then the gates each pipeline fails. When the check fails it
exits 1 and writes nothing; else, with --out, it writes there GATES.json (by default gates.json
here) with the new thresholds and nothing else changed, in its layout. The sessions are
GATES.json's, found by clock offset as evaluate.py and compare-dll find them, and must be all of
them.
"""

import argparse
import json
import os
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import evaluate as E  # noqa: E402

MODES = ("loso", "in-sample")
VARIANTS = ("good", "zyx", "q-transposed", "no-q", "no-filter")
# The sessions (as gates.json names them) where every loss event found must stay found: s3, whose
# two long losses the pipeline finds (the third, of 5 images, it does not see).
EVERY_LOSS = ("s3",)
# The metrics of each part of the rule.
VALIDITY_SHARES = ("coverage_pct", "agreement_pct")
LOSS_EVENTS = "loss_events_found"
PITCH_BIAS = "rotation_median_signed_x_deg"
ROTATION_MEDIANS = (
    "rotation_median_abs_x_deg",
    "rotation_median_abs_y_deg",
    "rotation_median_abs_z_deg",
    PITCH_BIAS,
)
ROTATION_P95S = (
    "rotation_geodesic_p95_deg",
    "combined_rotation_p95_abs_x_deg",
    "combined_rotation_geodesic_p95_deg",
)
POSITION_MEDIANS = (
    "position_median_abs_x_mm",
    "position_median_abs_y_mm",
    "position_median_abs_z_mm",
)
POSITION_P95 = "position_p95_abs_z_mm"
LAGS = ("rotation_lag_max_ms", "position_lag_max_ms")
JITTERS = ("rotation_rest_jitter_ratio", "position_rest_jitter_ratio")
# The least a jitter range holds: a ratio is ours over the DLL's, so 1 is the DLL's own jitter,
# which a filter that matches it must pass with room.
JITTER_PARITY = (0.8, 1.2)


# ------------------------------------------------------------------ the rule
def rounded(x, step, mode):
    """x to a multiple of step (a decimal string), by mode (decimal's ROUND_*)."""
    q = (Decimal(repr(float(x))) / Decimal(step)).to_integral_value(rounding=mode)
    return float(q * Decimal(step))


def raised(v, share, least, step):
    """|v| up by share of it, at least by least, rounded up to step."""
    a = abs(float(v))
    return rounded(a + max(least, share * a), step, ROUND_CEILING)


def rule(metric, v, session):
    """The threshold of the gate on metric in session whose leave-one-session-out value is v (but
    for G7's exception, which thresholds() makes)."""
    if metric in VALIDITY_SHARES:
        return rounded(v - 0.5, "0.1", ROUND_FLOOR)
    if metric == LOSS_EVENTS:
        return int(v) if session in EVERY_LOSS else max(0, int(v) - 1)
    if metric in ROTATION_MEDIANS:
        return raised(v, 0.3, 0.25, "0.1")
    if metric in ROTATION_P95S:
        return raised(v, 0.2, 1.0, "0.5")
    if metric in POSITION_MEDIANS:
        return raised(v, 0.3, 0.5, "0.1")
    if metric == POSITION_P95:
        return raised(v, 0.2, 5.0, "1")
    if metric in LAGS:
        return rounded(v + 5.0, "1", ROUND_CEILING)
    if metric in JITTERS:
        lo, hi = JITTER_PARITY
        return [
            min(rounded(v[0] - 0.2, "0.1", ROUND_FLOOR), lo),
            max(rounded(v[1] + 0.3, "0.1", ROUND_CEILING), hi),
        ]
    C.fail(f"no rule for the metric {metric!r}")


def failed(gates, th, name, metrics):
    """The ids of the gates that the metrics of session name fail with the thresholds th."""
    return [
        g["id"]
        for g in gates["gates"]
        if not (
            E.measurable(metrics[g["metric"]])
            and E.check(g["kind"], th[g["id"]][name], metrics[g["metric"]])
        )
    ]


def thresholds(gates, names, M):
    """{gate id: {session: threshold}} by the rule, from M[mode][variant], the metrics of each
    session in the order of names."""
    good = M["loso"]["good"]
    th = {
        g["id"]: {name: rule(g["metric"], m[g["metric"]], name) for name, m in zip(names, good)}
        for g in gates["gates"]
    }
    g7 = [g["id"] for g in gates["gates"] if g["metric"] == PITCH_BIAS]
    if len(g7) != 1:
        C.fail(f"{len(g7)} gates on {PITCH_BIAS}, not one")
    for i, name in enumerate(names):
        no_q = M["loso"]["no-q"][i]
        if failed(gates, th, name, no_q):
            continue
        ours, theirs = abs(good[i][PITCH_BIAS]), abs(no_q[PITCH_BIAS])
        mid = rounded((ours + theirs) / 2, "0.1", ROUND_HALF_EVEN)
        if not ours < mid < theirs:
            C.fail(
                f"{name}: the pipeline without Q passes every gate, and no {g7[0]} threshold to "
                f"0.1 lies between ours ({ours:.4f}) and its ({theirs:.4f})"
            )
        th[g7[0]][name] = mid
    return th


# ------------------------------------------------------------------ the runs
def run_all(sessions, refs, params):
    """{mode: {variant: [the metrics of each session]}} of the Python twin, and {(mode, i): what the
    constants of session i are}."""
    M, what = {}, {}
    for mode in MODES:
        M[mode] = {}
        for variant in VARIANTS:
            args = argparse.Namespace(
                params=params,
                loso=mode == "loso",
                blend=False,
                rotation_filters="params",
                rule="timeaware",
                broken=None if variant == "good" else variant,
            )
            M[mode][variant] = []
            for i, (o, rf) in enumerate(zip(sessions, refs)):
                P, what[(mode, i)] = E.constants(args, o)
                M[mode][variant].append(E.evaluate(o, P, args, rf)["metrics"])
            print(f"evaluated {mode}: {variant}", flush=True)
    return M, what


# ------------------------------------------------------------------ printing and writing
def compact(ids):
    """Gate ids (G and a number), three or more in a row as a range: G4, G5, G8, G9, G10 ->
    'G4, G5, G8-G10'."""
    runs = []  # [first id, last id, last number, length]
    for i in ids:
        n = int(i[1:])
        if runs and runs[-1][2] == n - 1:
            runs[-1][1:] = [i, n, runs[-1][3] + 1]
        else:
            runs.append([i, i, n, 1])
    parts = []
    for first, last, _, length in runs:
        parts += [f"{first}-{last}"] if length >= 3 else [first, last][:length]
    return ", ".join(parts)


def value_str(v):
    if isinstance(v, (list, tuple)):
        return f"{v[0]:.4f}..{v[1]:.4f}"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def threshold_str(t):
    if t is None:
        return "-"
    return f"{t[0]:g}..{t[1]:g}" if isinstance(t, list) else f"{t:g}"


def dump(doc):
    """gates.json's layout: each top-level key, and each key of a session or a gate, on a line of
    its own; a gate's thresholds on one line."""

    def text(key, v):
        if key != "thresholds":
            return json.dumps(v, ensure_ascii=False)
        return "{ " + ", ".join(f"{json.dumps(s)}: {json.dumps(t)}" for s, t in v.items()) + " }"

    def fields(d, pad):
        return [f"{pad}{json.dumps(k)}: {text(k, v)}" for k, v in d.items()]

    top = []
    for k, v in doc.items():
        if k in ("sessions", "gates"):
            items = ["    {\n" + ",\n".join(fields(item, "      ")) + "\n    }" for item in v]
            top.append(f"  {json.dumps(k)}: [\n" + ",\n".join(items) + "\n  ]")
        else:
            top.append(f"  {json.dumps(k)}: {text(k, v)}")
    return "{\n" + ",\n".join(top) + "\n}\n"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_session_args(ap)
    ap.add_argument(
        "--params",
        required=True,
        help="fit.py's JSON: its folds give the leave-one-session-out values, its head_params "
        "the in-sample check",
    )
    ap.add_argument(
        "--reference-fits",
        action="append",
        metavar="REF",
        required=True,
        help="the lag reference: a stateless run's fits, one per session in order (a fits CSV or "
        "the study's .npz)",
    )
    ap.add_argument(
        "--gates",
        default=E.GATES,
        help="the gates file whose thresholds to set (default: gates.json here)",
    )
    ap.add_argument("--out", help="where to write it with the new thresholds")
    args = ap.parse_args()
    if len(args.reference_fits) != len(args.session):
        C.fail(f"{len(args.reference_fits)} --reference-fits for {len(args.session)} sessions")
    with open(args.gates) as f:
        gates = json.load(f)
    if not (isinstance(gates.get("sessions"), list) and isinstance(gates.get("gates"), list)):
        C.fail(f'{args.gates}: not a gates file (a "sessions" and a "gates" list)')
    sessions = C.load_sessions(args)
    names = []
    for o in sessions:
        name = E.gate_session(gates, o)
        if name is None:
            C.fail(f"{args.gates} has no session of clock offset {o['k']} and {o['n']} images")
        names.append(name)
    missing = [s["name"] for s in gates["sessions"] if s["name"] not in names]
    if missing or len(set(names)) != len(names):
        C.fail(f"the sessions given are not {args.gates}'s, each once (missing: {missing})")
    refs = []
    for o, path in zip(sessions, args.reference_fits):
        rf = E.read_reference(path)
        if not np.array_equal(rf["devts"], o["devts"]):
            C.fail(f"{path}: not the images of {o['paths']['log']}")
        refs.append(rf)
    print()
    M, what = run_all(sessions, refs, args.params)
    for (mode, i), line in what.items():
        print(f"{mode} {names[i]}: {line}")
    th = thresholds(gates, names, M)

    print(
        "\n## Thresholds: the leave-one-session-out value -> the threshold "
        f"({os.path.basename(args.gates)}'s now)"
    )
    order = [s["name"] for s in gates["sessions"]]
    col = {name: i for i, name in enumerate(names)}
    rows = []
    for g in gates["gates"]:
        cells = [f"{g['id']} {g['name']}"]
        for name in order:
            v = M["loso"]["good"][col[name]][g["metric"]]
            new, old = th[g["id"]][name], g["thresholds"].get(name)
            cells.append(f"{value_str(v)} -> {threshold_str(new)} ({threshold_str(old)})")
        rows.append(cells)
    C.table(["gate", *order], rows)

    print("\n## The gates each pipeline fails")
    rows, problems = [], []
    for variant in VARIANTS:
        for mode in MODES:
            cells = []
            for name in order:
                f = failed(gates, th, name, M[mode][variant][col[name]])
                cells.append(compact(f) or "none")
                if variant == "good" and f:
                    problems.append(f"the pipeline fails {compact(f)} in {name} ({mode})")
                elif variant != "good" and not f:
                    problems.append(f"the pipeline {variant} passes every gate in {name} ({mode})")
            rows.append([variant, mode, *cells])
    C.table(["pipeline", "constants", *order], rows)
    if problems:
        print("\ncheck FAILED: " + "; ".join(problems) + "; nothing written")
        sys.exit(1)
    print(
        "\nchecked: the good pipeline passes every gate, each broken one fails one in each session"
    )
    if args.out:
        for g in gates["gates"]:
            g["thresholds"] = {name: th[g["id"]][name] for name in order}
        with open(args.out, "w") as f:
            f.write(dump(gates))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
