"""What fit.py and evaluate.py share: the recorded sessions and numpy twins of reference.py.

A session is three files, none of them in the repository (they hold the user's face):
  - the TBI5LOG1 log of the tracker's EP 0x83 stream (tobii5-init-replay's packet log, or
    `import-tsv` of a USBPcap capture): the 0x500 gaze frames and the 0x50e images;
  - the Stream Engine's JSONL of the same session (one callback per line: gazePoint,
    gazeOrigin, headPose), recorded on Windows;
  - the face fits `tobii5-init-replay image83-replay LOG --fits FITS.csv` made of the log's
    images (one row per image, in log order).
load_session() reads all three, finds the clock offset K between the DLL and the device
(timestamp_us + K = device time), pairs every DLL head pose with its image, and takes the display
frame from the log (a rigid fit of the gaze origins 0x02/0x08 in the tracker frame onto
0x22/0x24 in the display frame) unless an area is given. No pixels are read.
"""

import argparse
import csv
import json
import math
import os
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import reference as ref

D_ = np.diag([-1.0, -1.0, 1.0])
FS = 1.0 / ref.DT_NOMINAL_S


def fail(msg):
    sys.exit(f"error: {msg}")


# ------------------------------------------------------------------ the TBI5LOG1 log
LOG_MAGIC = b"TBI5LOG1"
EP_STREAM = 0x83
MARKER_STREAM = 0x53
STREAM_GAZE = 0x500
STREAM_IMAGE = 0x50E
TLV_OFFSET = 34  # 32-byte header, then 00 00
KEY_FIELD_ID = 0x20BB9
UNITS_PER_MM = 1024.0
# The 0x500 fields read here: the gaze origins in the tracker frame (0x02, 0x08) and in the
# display frame (0x22, 0x24), the eyeball centres (0x17, 0x18), all 1/1024 mm; the combined gaze
# point (0x1c, x1024) and its validity (0x1b).
GAZE_VECTORS = (0x02, 0x08, 0x17, 0x18, 0x22, 0x24, 0x1C)
GAZE_SCALARS = (0x1B,)


def log_messages(path):
    """Yield (host time us, message bytes) for every message of EP 0x83 in a TBI5LOG1 log, put
    together from the transfers by the length at +4 (little-endian, the 8-byte prefix included).
    A record is `[0, ep, 0, 0]`, the host time (u64 LE, us), the length (u32 LE), the bytes."""
    buf = bytearray()
    with open(path, "rb") as f:
        if f.read(len(LOG_MAGIC)) != LOG_MAGIC:
            fail(f"{path}: not a TBI5LOG1 log")
        while True:
            head = f.read(16)
            if len(head) < 16:
                return
            ep = head[1]
            host_us = struct.unpack_from("<Q", head, 4)[0]
            ln = struct.unpack_from("<I", head, 12)[0]
            data = f.read(ln)
            if len(data) < ln:
                return
            if ep != EP_STREAM:
                continue
            buf += data
            while len(buf) >= 12:
                mlen = struct.unpack_from("<I", buf, 4)[0]
                if mlen < 12 or mlen > 4_000_000 or struct.unpack_from("<I", buf, 0)[0] > 1:
                    del buf[0]  # lost sync: look for the next message start
                    continue
                if len(buf) < mlen:
                    break
                yield host_us, bytes(buf[:mlen])
                del buf[:mlen]


def tlv(payload, start=TLV_OFFSET):
    """The (type, value) entries of a message's TLV list."""
    off, n = start, len(payload)
    while off + 5 <= n:
        t = payload[off]
        ln = struct.unpack_from(">I", payload, off + 1)[0]
        off += 5
        if off + ln > n:
            return
        yield t, payload[off : off + ln]
        off += ln


def keyed(payload):
    """A stream message's keyed fields: {key: ('vec', [values]) | ('scalar', int) | ('blob',
    bytes)}. Vector components are the raw 32.32 values (lengths in 1/1024 mm)."""
    out = {}
    ents = list(tlv(payload))
    i = 0
    cur = None
    while i < len(ents):
        t, v = ents[i]
        if t == 5 and len(v) == 4 and struct.unpack(">I", v)[0] == KEY_FIELD_ID:
            if i + 1 < len(ents) and ents[i + 1][0] == 2:
                cur = struct.unpack(">I", ents[i + 1][1])[0]
                i += 2
                if i < len(ents):
                    t2, v2 = ents[i]
                    if t2 in (2, 3) and len(v2) == 4:
                        out[cur] = ("scalar", struct.unpack(">I", v2)[0])
                        i += 1
                    elif t2 == 6 and len(v2) == 8:
                        out[cur] = ("scalar", struct.unpack(">Q", v2)[0])
                        i += 1
                    elif t2 == 0x15:
                        out[cur] = ("blob", v2)
                        i += 1
                continue
            i += 1
            continue
        if t == 5:
            vals = []
            j = i + 1
            while j < len(ents) and ents[j][0] == 4:
                vals.append(struct.unpack(">q", ents[j][1])[0] / 4294967296.0)
                j += 1
            if cur is not None:
                out[cur] = ("vec", vals)
            i = j
            continue
        i += 1
    return out


def read_log(path):
    """The images' device times and the gaze frames of a log: dict(image_devts (n,), gaze_devts
    (m,), gaze {key: (m, k) raw values, NaN where the frame lacks the key}, gaze_scalars {key: (m,)
    int, -1 where absent}). Gaze frames without a device time are dropped."""
    img, gdev, gvec, gsc = [], [], {k: [] for k in GAZE_VECTORS}, {k: [] for k in GAZE_SCALARS}
    for _, m in log_messages(path):
        if len(m) < 24 or struct.unpack_from(">I", m, 8)[0] != MARKER_STREAM:
            continue
        sid = struct.unpack_from(">I", m, 20)[0]
        if sid == STREAM_IMAGE:
            kv = keyed(m)
            if 1 in kv and kv[1][0] == "scalar":
                img.append(kv[1][1])
        elif sid == STREAM_GAZE:
            kv = keyed(m)
            if 1 not in kv or kv[1][0] != "scalar":
                continue
            gdev.append(kv[1][1])
            for k in GAZE_VECTORS:
                x = kv.get(k)
                gvec[k].append(x[1] if x and x[0] == "vec" else None)
            for k in GAZE_SCALARS:
                x = kv.get(k)
                gsc[k].append(x[1] if x and x[0] == "scalar" else -1)
    gaze = {}
    for k, rows in gvec.items():
        width = max((len(r) for r in rows if r is not None), default=0)
        a = np.full((len(rows), width), np.nan)
        for i, r in enumerate(rows):
            if r is not None:
                a[i, : len(r)] = r
        gaze[k] = a
    return dict(
        image_devts=np.array(img, np.int64),
        gaze_devts=np.array(gdev, np.int64),
        gaze=gaze,
        gaze_scalars={k: np.array(v, np.int64) for k, v in gsc.items()},
    )


def sentinel_nan(raw):
    """Raw components that are the device's 'no value' (exactly 0 or +-1024) as NaN."""
    a = np.array(raw, float)
    a[(a == 0.0) | (np.abs(a) == UNITS_PER_MM)] = np.nan
    return a


# ------------------------------------------------------------------ the Stream Engine's JSONL
def read_dll(path):
    """The DLL's gazePoint and headPose callbacks as arrays (head pose angles in radians, as the
    DLL gives them)."""
    gp, hp = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "gazePoint" in d:
                g = d["gazePoint"]
                gp.append((g["timestamp_us"], g["validity"], *g["position_xy"]))
            elif "headPose" in d:
                g = d["headPose"]
                hp.append(
                    (
                        g["timestamp_us"],
                        d.get("hostUs", -1),
                        g["position_validity"],
                        *g["position_xyz"],
                        *g["rotation_validity_xyz"],
                        *g["rotation_xyz"],
                    )
                )
    gp = np.array(gp, dtype=np.float64).reshape(-1, 4)
    hp = np.array(hp, dtype=np.float64).reshape(-1, 12)
    return dict(
        gp_ts=gp[:, 0].astype(np.int64),
        gp_valid=gp[:, 1].astype(int),
        gp_xy=gp[:, 2:4],
        hp_ts=hp[:, 0].astype(np.int64),
        hp_host=hp[:, 1].astype(np.int64),
        hp_pvalid=hp[:, 2].astype(int),
        hp_pos=hp[:, 3:6],
        hp_rvalid=hp[:, 6:9].astype(int),
        hp_rot=hp[:, 9:12],
    )


def clock_offset(log, dll):
    """K (us) with DLL timestamp_us + K = device time, as compare-dll takes it: from the valid DLL
    gaze points equal, as f32, to a frame's combined gaze point (key 0x1c / 1024); the median of
    the device - DLL differences. Returns (K, matches agreeing on it, matches)."""
    xy = log["gaze"][0x1C]
    if xy.shape[1] < 2:
        fail("the log has no combined gaze points (key 0x1c)")
    by_xy = {}
    ok = np.isfinite(xy).all(1)
    bits = (xy[ok] / UNITS_PER_MM).astype(np.float32).view(np.uint32)
    for (bx, by), ts in zip(bits.tolist(), log["gaze_devts"][ok].tolist()):
        by_xy[(bx, by)] = ts
    v = dll["gp_valid"] != 0
    dbits = dll["gp_xy"][v].astype(np.float32).view(np.uint32)
    offsets = sorted(
        by_xy[k] - ts
        for k, ts in zip(map(tuple, dbits.tolist()), dll["gp_ts"][v].tolist())
        if k in by_xy
    )
    if not offsets:
        fail("no DLL gaze point matches a gaze frame of the log: not the same session?")
    k = offsets[len(offsets) // 2]
    return k, sum(1 for o in offsets if o == k), len(offsets)


def pair_head_poses(image_devts, dll, k):
    """One row per image: dll_have (a DLL head pose of this image), dll_valid (its four flags
    set), dll_pos (mm), dll_rot (deg), NaN where not valid; and the counts of the DLL poses that
    match no image and of those whose flags disagree (the tools take a pose valid only when all
    four are)."""
    n = len(image_devts)
    t = dll["hp_ts"] + k
    idx = np.searchsorted(image_devts, t)
    hit = (idx < n) & (image_devts[np.minimum(idx, n - 1)] == t)
    idx = idx[hit]
    pv = dll["hp_pvalid"][hit].astype(bool)
    rv = dll["hp_rvalid"][hit].astype(bool)
    allv = pv & rv.all(1)
    have = np.zeros(n, bool)
    valid = np.zeros(n, bool)
    have[idx] = True
    valid[idx] = allv
    pos = np.full((n, 3), np.nan)
    rot = np.full((n, 3), np.nan)
    pos[idx[allv]] = dll["hp_pos"][hit][allv]
    rot[idx[allv]] = np.degrees(dll["hp_rot"][hit][allv])
    flags = np.column_stack([pv, rv])
    mixed = int((flags.any(1) & ~flags.all(1)).sum())
    return dict(
        dll_have=have,
        dll_valid=valid,
        dll_pos=pos,
        dll_rot=rot,
        unpaired=int((~hit).sum()),
        mixed_flags=mixed,
    )


# ------------------------------------------------------------------ the display frame
def kabsch(P, Q):
    """The rigid Q ~ R P + t closest in least squares (rows are points): R, t, residuals."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc
    return R, t, Q - (P @ R.T + t)


def fit_display_frame(log):
    """(R_TS, c, description) from the gaze origins the device reports in both frames: 0x02 ->
    0x22 and 0x08 -> 0x24, components that are no value dropped. The device computes 0x22/0x24
    from its display area, so this is the area's frame to ~1e-4 mm."""
    P, Q = [], []
    for ks, kt in ((0x02, 0x22), (0x08, 0x24)):
        a, b = sentinel_nan(log["gaze"][ks]), sentinel_nan(log["gaze"][kt])
        if a.shape[1] != 3 or b.shape[1] != 3:
            continue
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        P.append(a[ok] / UNITS_PER_MM)
        Q.append(b[ok] / UNITS_PER_MM)
    P = np.concatenate(P) if P else np.zeros((0, 3))
    Q = np.concatenate(Q) if Q else np.zeros((0, 3))
    if len(P) < 10:
        fail(f"only {len(P)} gaze origins in both frames: pass the display area (--area)")
    R, t, res = kabsch(P, Q)
    tilt = math.degrees(math.atan2(R[1][2], R[1][1]))
    rms = np.sqrt((res**2).mean(0))
    desc = (
        f"rigid fit of {len(P)} gaze origins 0x02/0x08 -> 0x22/0x24: tilt {tilt:.5f} deg, t ("
        + ", ".join(f"{v:+.5f}" for v in t)
        + ") mm, residual rms "
        + "/".join(f"{v:.1e}" for v in rms)
        + " mm"
    )
    return R, -R.T @ t, desc


def parse_area(text):
    """--area: 'windows' (the Windows sessions' area) or nine numbers TLx,TLy,TLz,TRx,...,BLz
    (mm). An ArgumentTypeError's message is what argparse shows (a ValueError's is not)."""
    if text == "windows":
        return ref.WINDOWS_AREA
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError:
        v = []
    if len(v) != 9:
        raise argparse.ArgumentTypeError(
            f"{text!r} is neither 'windows' nor 9 comma-separated numbers: TL, TR, BL (x, y, z mm)"
        )
    return tuple(tuple(v[i : i + 3]) for i in (0, 3, 6))


# ------------------------------------------------------------------ the face fits
FITS_COLUMNS = (
    "image_idx,device_ts_us,face,score,found_by_detector,"
    "r_cam_00,r_cam_01,r_cam_02,r_cam_10,r_cam_11,r_cam_12,r_cam_20,r_cam_21,r_cam_22,"
    "t_cam_x_mm,t_cam_y_mm,t_cam_z_mm,"
    "centroid_u,centroid_v,nose_tip_u,nose_tip_v,"
    "eye_image_left_u,eye_image_left_v,eye_image_right_u,eye_image_right_v,"
    "corner_33_u,corner_33_v,corner_263_u,corner_263_v"
).split(",")


def read_fits(path):
    """image83-replay's --fits CSV (one row per image, in log order) as arrays; NaN where the
    image has no face."""
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header != FITS_COLUMNS:
            fail(f"{path}: not an image83-replay --fits CSV (header {(header or [])[:4]}...)")
        rows = [[float(x) if x != "" else math.nan for x in r] for r in rd]
    a = np.array(rows, float).reshape(-1, len(FITS_COLUMNS))
    col = {name: a[:, i] for i, name in enumerate(FITS_COLUMNS)}

    def pt(name):
        return np.stack([col[f"{name}_u"], col[f"{name}_v"]], 1)

    if not (col["image_idx"] == np.arange(len(a))).all():
        fail(f"{path}: image_idx does not run 0, 1, 2, ...")
    R_cam = np.stack([col[f"r_cam_{i}{j}"] for i in range(3) for j in range(3)], 1)
    return dict(
        devts=col["device_ts_us"].astype(np.int64),
        face=col["face"] == 1,
        score=col["score"],
        found_by_detector=col["found_by_detector"] == 1,
        R_cam=R_cam.reshape(-1, 3, 3),
        t_cam_mm=np.stack([col[f"t_cam_{c}_mm"] for c in "xyz"], 1),
        centroid=pt("centroid"),
        nose_tip=pt("nose_tip"),
        eye_a=pt("eye_image_left"),
        eye_b=pt("eye_image_right"),
        corner_33=pt("corner_33"),
        corner_263=pt("corner_263"),
    )


# ------------------------------------------------------------------ the session
def runs_of(mask):
    """(start, length) of the runs of True."""
    m = np.concatenate([[0], np.asarray(mask, int), [0]])
    d = np.diff(m)
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0]
    return st, en - st


def edge_v(uv):
    u, v = uv[..., 0], uv[..., 1]
    return np.minimum(np.minimum(u, ref.FRAME_PX - u), np.minimum(v, ref.FRAME_PX - v))


def apply_fits(o, fits):
    """Put a fits table into session o: the fit of every image and G3 (with FITTED's thresholds,
    which the fit does not change)."""
    rows = len(fits["devts"])
    if rows != o["n"] or not (fits["devts"] == o["devts"]).all():
        fail(f"{o['name']}: the fits ({rows} rows) are not of this log's {o['n']} images")
    cmin, nmin = ref.FITTED["g3_centroid_min_px"], ref.FITTED["g3_nose_min_px"]
    keys = ("face", "score", "found_by_detector", "R_cam", "t_cam_mm")
    o.update({k: fits[k] for k in keys + ("centroid", "nose_tip", "eye_a", "eye_b")})
    ce = edge_v(fits["centroid"])
    ne = edge_v(fits["nose_tip"])
    o["g3_ce"] = np.where(o["face"], ce, np.nan)
    o["g3_ne"] = np.where(o["face"], ne, np.nan)
    o["g3"] = (
        o["face"]
        & (o["score"] >= 0)
        & (np.nan_to_num(ce, nan=-1e9) >= cmin)
        & (np.nan_to_num(ne, nan=-1e9) >= nmin)
    )
    return o


def load_session(name, log_path, dll_path, fits_path, area=None):
    """Everything the fit and the evaluation read of one session, one row per 0x50e image:
    devts / t_us; the paired DLL pose (dll_have, dll_valid, dll_pos T mm, dll_rot deg x y z, R_dll);
    dev_ok / dev_none: both / neither eyeball centre 0x17/0x18 in the last gaze frame before the
    image (less than 100 ms before it); the fit (face, score, R_cam, t_cam_mm, the points, G3);
    the display frame (R_TS, c); masks of the DLL-valid images: ALL, BOTH, NONE, YAW20 (|DLL yaw|
    >= 20), REACQ (the first 10 of every DLL-valid run), REACQgap (the same but the session
    start's), COMB (|yaw| > 15 and (|pitch| > 8 or |roll| > 10))."""
    log = read_log(log_path)
    dll = read_dll(dll_path)
    fits = read_fits(fits_path)
    devts = fits["devts"]
    n = len(devts)
    if n != len(log["image_devts"]) or not (devts == log["image_devts"]).all():
        fail(
            f"{fits_path}: its {n} rows are not the {len(log['image_devts'])} images of {log_path}"
        )
    k, agree, matches = clock_offset(log, dll)
    pr = pair_head_poses(devts, dll, k)
    o = dict(
        name=name,
        n=n,
        devts=devts,
        t_us=devts.copy(),
        k=k,
        k_agree=agree,
        k_matches=matches,
        paths=dict(log=log_path, dll=dll_path, fits=fits_path),
        dll_records=len(dll["hp_ts"]),
        gaze_frames=len(log["gaze_devts"]),
        **pr,
    )
    rot = o["dll_rot"]
    valid = o["dll_valid"]
    o["R_dll"] = np.full((n, 3, 3), np.nan)
    o["R_dll"][valid] = compose_yxz_v(np.radians(rot[valid]))
    if area is None:
        o["R_TS"], o["c"], o["frame"] = fit_display_frame(log)
    else:
        o["R_TS"], o["c"] = ref.display_frame(*area)
        o["frame"] = "display area " + " ".join(
            "(" + ", ".join(f"{v:g}" for v in p) + ")" for p in area
        )
    # the device's eyes in the last gaze frame before each image
    gdev = log["gaze_devts"]
    oo = np.argsort(gdev, kind="stable")
    gdev = gdev[oo]
    L = sentinel_nan(log["gaze"][0x17])[oo] / UNITS_PER_MM
    R = sentinel_nan(log["gaze"][0x18])[oo] / UNITS_PER_MM
    j = np.searchsorted(gdev, devts, side="left") - 1
    okj = j >= 0
    j = np.clip(j, 0, max(len(gdev) - 1, 0))
    if len(gdev):
        okj &= (devts - gdev[j]) / 1e3 < 100.0
        Lf = okj & np.isfinite(L[j]).all(1)
        Rf = okj & np.isfinite(R[j]).all(1)
    else:
        Lf = Rf = np.zeros(n, bool)
    o["dev_ok"] = Lf & Rf
    o["dev_none"] = ~Lf & ~Rf
    apply_fits(o, fits)
    # subsets of the DLL-valid images
    yaw, pit, rol = (np.abs(np.nan_to_num(rot[:, i])) for i in (1, 0, 2))
    reacq = np.zeros(n, bool)
    reacq_gap = np.zeros(n, bool)
    st, ln = runs_of(valid)
    for i, (a, k) in enumerate(zip(st, ln)):
        reacq[a : a + min(10, k)] = True
        if i > 0:
            reacq_gap[a : a + min(10, k)] = True
    o["masks"] = dict(
        ALL=valid,
        BOTH=valid & o["dev_ok"],
        NONE=valid & o["dev_none"],
        YAW20=valid & (yaw >= 20.0),
        REACQ=valid & reacq,
        REACQgap=valid & reacq_gap,
        COMB=valid & (yaw > 15.0) & ((pit > 8.0) | (rol > 10.0)),
    )
    return o


def describe(o):
    """A few lines on what was read and paired."""
    v, inv = o["dll_valid"], o["dll_have"] & ~o["dll_valid"]
    log, dll, fits = (os.path.basename(o["paths"][k]) for k in ("log", "dll", "fits"))
    paired = o["dll_records"] - o["unpaired"]
    counts = f"{int(v.sum())} / {int(inv.sum())} / {int((~o['dll_have']).sum())}"
    mixed = o["mixed_flags"]
    face, g3 = o["face"], o["g3"]
    detected = int((face & o["found_by_detector"]).sum())
    lines = [
        f"{o['name']}: {log}: {o['n']} images, {o['gaze_frames']} gaze frames; "
        f"{dll}: {o['dll_records']} head poses",
        f"  clock offset K {o['k']} us (agreed by {o['k_agree']} of {o['k_matches']} matching gaze "
        f"points); head poses paired with an image {paired} (unpaired {o['unpaired']}); DLL "
        f"valid / invalid / no pose {counts}"
        + (f"; {mixed} poses with flags that disagree (taken invalid)" if mixed else ""),
        f"  display frame: {o['frame']}",
        f"  {fits}: faces {int(face.sum())} (found by the detector {detected}), G3-valid "
        f"{int(g3.sum())}; at DLL-valid images: faces {int((face & v).sum())}, G3-valid "
        f"{int((g3 & v).sum())} of {int(v.sum())}",
    ]
    return "\n".join(lines)


def add_session_args(parser):
    parser.add_argument(
        "--session",
        nargs=3,
        action="append",
        metavar=("LOG", "JSONL", "FITS"),
        required=True,
        help="a session: the TBI5LOG1 log, the Stream Engine's JSONL of it, and image83-replay's "
        "--fits CSV of the log (repeat; named s1, s2, ... in the order given)",
    )
    parser.add_argument(
        "--area",
        type=parse_area,
        default=None,
        help="the display area, 'windows' or TLx,TLy,TLz,TRx,TRy,TRz,BLx,BLy,BLz (mm); "
        "default: fitted to each log's gaze origins",
    )


def load_sessions(args):
    out = []
    for i, (log, dll, fits) in enumerate(args.session):
        o = load_session(f"s{i + 1}", log, dll, fits, args.area)
        print(describe(o), flush=True)
        out.append(o)
    return out


# ------------------------------------------------------------------ printing
def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def tri(vals, f="{:.2f}"):
    """The sessions' values of one quantity, s1 / s2 / ...; '-' for none."""
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append("-")
        else:
            out.append(f.format(v))
    return " / ".join(out)


# ------------------------------------------------------------------ twins of reference.py
def compose_yxz_v(ang_rad):
    a = np.asarray(ang_rad, float)
    return Rotation.from_euler("YXZ", np.stack([a[:, 1], a[:, 0], a[:, 2]], 1)).as_matrix()


def euler_yxz_v(R):
    a = Rotation.from_matrix(R).as_euler("YXZ")
    return np.stack([a[:, 1], a[:, 0], a[:, 2]], 1)


def geo_deg(Ra, Rb):
    M = np.einsum("nji,njk->nik", Ra, Rb)
    tr = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def procrustes(M):
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def head_S_v(R_cam, Q):
    return D_ @ R_cam @ D_ @ Q


def dir_map_v(p, g, ox, oy):
    r = np.linalg.norm(p, axis=1)
    v = np.stack([g * p[:, 0] / p[:, 2] + ox, g * p[:, 1] / p[:, 2] + oy, np.ones(len(p))], 1)
    return r[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)


def p_pnp_v(t_cam_mm, H, s, d, g, ox, oy):
    p0 = s * (t_cam_mm @ D_.T) + np.einsum("nij,j->ni", H, np.asarray(d, float))
    return dir_map_v(p0, g, ox, oy)


def ray_v(uv):
    K = ref.INTRINSICS
    r = np.stack(
        [(uv[:, 0] - K["cu"]) / K["fu"], (uv[:, 1] - K["cv"]) / K["fv"], np.ones(len(uv))], 1
    )
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def eye_geom(eye_a, eye_b, H):
    ra, rb = ray_v(eye_a), ray_v(eye_b)
    m = ra + rb
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    ang = np.arccos(np.clip(np.sum(ra * rb, 1), -1, 1))
    fs = np.sqrt(np.maximum(1 - np.sum(H[:, :, 0] * m, 1) ** 2, 1e-6))
    return m, ang, fs


def p_eye_v(geom, K, g, ox, oy):
    m, ang, fs = geom
    rng = K * fs / np.sin(ang)
    v = np.stack([g * m[:, 0] / m[:, 2] + ox, g * m[:, 1] / m[:, 2] + oy, np.ones(len(m))], 1)
    return rng[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)


def to_T(o, pS):
    return (pS - o["c"]) @ o["R_TS"].T


def to_S(o, pT):
    return pT @ o["R_TS"] + o["c"]


def ema_series(t_us, valid, X, tau, rule="timeaware", reset_gap_s=1.0):
    """reference.Ema3 over a whole series, X (n, k), with tau in s: (n, k), NaN where not valid;
    the reset rule of reference.Smoother, but its reset after a result that is not finite, which
    the finite inputs here never give."""
    n, k = X.shape
    Y = np.full((n, k), np.nan)
    have = False
    t_last = None
    prev_valid = False
    y = None
    for i in range(n):
        if not valid[i]:
            prev_valid = False
            continue
        x = X[i]
        dt = None if not have else (t_us[i] - t_last) * 1e-6
        reset = dt is None or dt <= 0 or dt > reset_gap_s or (rule == "reset" and not prev_valid)
        if reset:
            y = x.copy()
        else:
            y = y + dt / (dt + tau) * (x - y)
        have = True
        t_last = t_us[i]
        prev_valid = True
        Y[i] = y
    return Y


def d2rms(p, ok):
    """RMS of the second difference per column over the triples of ok images."""
    m = ok[2:] & ok[1:-1] & ok[:-2]
    return np.sqrt(np.mean((p[2:] - 2 * p[1:-1] + p[:-2])[m] ** 2, 0))


def run_reference(o, P, rule="timeaware", rot_filter=None, decomposition="yxz"):
    """reference.HeadPoseEstimator over a session, one step per image (time: the image's device
    time). Per-image arrays, NaN where the pose is invalid: valid, pos (T mm), rot (deg), and the
    unfiltered raw_rot (deg) and raw_pos (T mm, the PnP branch plus the unfiltered correction)."""
    est = ref.HeadPoseEstimator(P, rule=rule, rot_filter=rot_filter, decomposition=decomposition)
    est.set_display_frame(o["R_TS"], o["c"])
    n = o["n"]
    valid = np.zeros(n, bool)
    pos = np.full((n, 3), np.nan)
    rot = np.full((n, 3), np.nan)
    raw_rot = np.full((n, 3), np.nan)
    raw_pos = np.full((n, 3), np.nan)
    for i in range(n):
        f = bool(o["face"][i])
        r = est.step_points(
            int(o["t_us"][i]),
            f,
            float(o["score"][i]) if f else math.nan,
            o["R_cam"][i],
            o["t_cam_mm"][i],
            o["centroid"][i],
            o["nose_tip"][i],
            o["eye_a"][i],
            o["eye_b"][i],
        )
        if r["valid"]:
            valid[i] = True
            pos[i] = r["position_mm"]
            rot[i] = np.degrees(r["rotation_rad"])
            raw_rot[i] = r["raw_rot_deg"]
            cor = r["raw_cor_S"]
            raw_pos[i] = to_T(o, (r["raw_pnp_S"] + (cor if cor is not None else 0.0))[None])[0]
    return dict(valid=valid, pos=pos, rot=rot, raw_rot=raw_rot, raw_pos=raw_pos)


def load_params(path):
    """A fit.py JSON (or a bare HeadParams dict) as (P, document)."""
    with open(path) as f:
        doc = json.load(f)
    return ref.params_from_head_params(doc.get("head_params", doc)), doc
