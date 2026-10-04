# Head pose: fitting and acceptance

The head pose libtobii hands out is the Stream Engine's (`tobii_head_pose_t`): absolute, in the
display frame, filtered as the DLL filters it. Its constants (`HeadParams::FITTED` in
`crates/tobii-pose/src/head.rs`) were fitted to what the DLL itself reported in three Windows
sessions, and the replay of those sessions against the DLL is the acceptance test. These scripts
redo both from the repository: fit the constants to the face fits of the Rust tracker, write them
under `HeadParams`' field names, measure the result against the DLL, and regenerate the unit
tests' vectors. Python 3 with NumPy and SciPy, nothing else.

| file | what it is |
|---|---|
| `reference.py` | The executable specification of `head.rs`: validity, rotation, position, filters, one call per image. A port of the head pose study's reference: the same operations, so the same numbers bit for bit. Also the study's constants (`PROTOTYPE`), the same with the choices `head.rs` ships (`FITTED`, which `HeadParams::FITTED` holds until the constants are fitted again) and the conversion to and from `HeadParams`' field names. |
| `make_vectors.py` | Writes the synthetic test vectors `head.rs`'s tests carry, from `reference.py` and the canonical mesh of `canonical.rs`. No session data. |
| `common.py` | Reads a session: the TBI5LOG1 log, the DLL's JSONL, the face fits; finds the clock offset, pairs every DLL pose with its image, fits the display frame. Vectorised twins of `reference.py` for whole sessions. |
| `fit.py` | Leave-one-session-out refit of the constants; prints the folds and every session's errors; writes the constants as JSON. |
| `evaluate.py` | Runs `reference.py` over the fits with a set of constants, or takes the daemon's own poses from `compare-dll --head`'s CSV, and prints every acceptance metric, lag and rest jitter included, against `gates.json`. |
| `gates.json` | The acceptance gates, per session: the file `compare-dll --head` checks, which `evaluate.py` reads too. |
| `blaze_face_to_onnx.py` | The conversion of MediaPipe's BlazeFace detector to the ONNX model tobii-pose embeds (its own docstring; needs tflite2onnx). |

## The data, which never goes into the repository

A session is three files:

- **the TBI5LOG1 log** of the tracker's EP 0x83 stream: the 0x500 gaze frames and the 0x50e
  images, the user's face in every one of them. The Windows sessions were captured with USBPcap
  while the DLL ran, and imported with `tobii5-init-replay import-tsv` (a tshark TSV of
  timestamp, endpoint and data);
- **the DLL's JSONL**: one line per callback it made in the same session (`gazePoint`,
  `gazeOrigin`, `headPose`), written by a logger on Windows;
- **the face fits** the Rust tracker makes of the log's images:
  `tobii5-init-replay image83-replay LOG --fits FITS.csv`, one row per image (the layout is in
  `image83-replay -h`).

The three Windows sessions live outside the tracked tree: the logs in
`fixtures/analysis/headpose-2026-09-27/data/session{1,2,3}.bin` (gitignored, with the
captures in `fixtures/captures/`), the JSONL in `~/Work/tobii-sys/session{1,2,3}.jsonl`. The fits,
the landmarks and everything the scripts print per image are derived from images of a face: keep
them under `fixtures/` too, and never commit them. What does go into the repository is code, the
fitted constants, the gate thresholds and the synthetic vectors.

The lag and the rest jitter need a per-image reference that the evaluated fits do not feed: the
fits of a stateless run of a tracker, every image fitted on its own. The head pose study's are
`fixtures/analysis/headpose-2026-09-27/work2/filter-identify/runs/sl_s{1,2,3}.npz`;
`evaluate.py --reference-fits` reads them as they are, or a fits CSV.

## Refitting the constants and accepting them

The commands run from the repository's root and write under `fixtures/` (gitignored), in a
directory of the refit's own: the fits, and whatever is printed per image, come from images of a
face.

    OUT=fixtures/analysis/headpose-refit
    mkdir -p $OUT

1. Replay every session through the tracker as it is now (release build), for each N:

       target/release/tobii5-init-replay image83-replay \
           fixtures/analysis/headpose-2026-09-27/data/sessionN.bin --fits $OUT/fits_sN.csv

2. Fit, leave one session out (fit.py names the sessions s1, s2, ... in the order given, and
   `fit.json` records each one's clock offset and image count, by which `evaluate.py --loso`
   finds the fold that leaves a session out, whatever the order of its own arguments):

       tools/headpose/fit.py --session LOG1 JSONL1 $OUT/fits_s1.csv \
           --session LOG2 JSONL2 $OUT/fits_s2.csv --session LOG3 JSONL3 $OUT/fits_s3.csv \
           --in-sample --out $OUT/fit.json

   It prints each fold's constants and the beta grid it chose from, then every session's errors
   with the constants fitted without it (LOSO) and, with `--in-sample`, with those fitted on all.
   Last comes the study's search of the G3 thresholds: each fold's, tested on the session it
   leaves out, and the optimum on all the sessions next to the thresholds `head_params` keep
   (FITTED's, 6 and -4 px), which the fit does not change.
   `fit.json`'s `head_params` are the constants fitted on all the sessions, with the choices
   `head.rs` ships: the PnP branch alone (eye weight 0; `--blend` keeps the weight found) and
   one one-euro filter for the three angles (`--rotation-filters per-axis` for the study's).
3. Copy `head_params` into `HeadParams::FITTED`, field for field, and say in its doc comment
   what it was fitted on (date, sessions, tracker). `reference.FITTED`, `evaluate.py`'s default,
   stays the study's: give the scripts `--params $OUT/fit.json` from here on.
4. Regenerate the vectors with the new constants and update the literals in `head.rs`'s tests.
   The vectors keep the eye branch on (the weight the fit found) and a different filter per
   angle, so that both stay covered:

       tools/headpose/make_vectors.py $OUT/vectors.json --params $OUT/fit.json \
           --eye-weight fitted --rotation-filters per-axis

   Without options it writes the study's vectors, the ones the tests carry now, byte for byte.
5. Accept: replay every session through the daemon's own pipeline (`HeadStep` with the new
   `HeadParams::FITTED`), which checks the gates it can (G1-G14) and writes its pose of every
   image with a DLL pose, for each N:

       target/release/tobii5-init-replay compare-dll LOGN JSONLN --head \
           --gates tools/headpose/gates.json --csv $OUT/head_sN.csv

   then measure those poses' lag and rest jitter (G15-G18), and every other gate again, with
   `evaluate.py`, given the same fits and the constants the daemon was built with:

       tools/headpose/evaluate.py --session LOG1 JSONL1 $OUT/fits_s1.csv ... \
           --params $OUT/fit.json --ours-csv $OUT/head_s1.csv --ours-csv $OUT/head_s2.csv \
           --ours-csv $OUT/head_s3.csv --reference-fits .../sl_s1.npz \
           --reference-fits .../sl_s2.npz --reference-fits .../sl_s3.npz

   It also compares the Python twin of those constants with the CSV, image by image: with the
   daemon's constants and fits they agree on every image's validity and to ~1e-12 mm and deg, and
   a WARNING says when they do not. Every gate must pass in every session (exit 0); the numbers
   go into the commit message. Without `--ours-csv` `evaluate.py` evaluates the Python twin:
   `--params $OUT/fit.json --loso` gives the leave-one-session-out numbers, and
   `--broken zyx|q-transposed|no-q|no-filter` shows which gates catch a broken pipeline.

## What is measured

On every 0x50e image of a session. A DLL pose is valid when its four flags are set (they always
are or are not together); ours is valid when G3 holds. The subsets are of the DLL-valid images,
and the errors are taken where ours is valid too: ALL; BOTH and NONE (both or neither of the
device's eyeball centres 0x17/0x18 in the last gaze frame before the image, under 100 ms old);
YAW20 (|DLL yaw| >= 20°); REACQ (the first 10 images of every run of DLL-valid images) and
REACQgap (the same without the session start's); COMB (|yaw| > 15° and (|pitch| > 8° or
|roll| > 10°), the combined poses on which a wrong Euler order shows).

- **Validity**: coverage = our valid share of the DLL-valid images; agreement over the images
  with a DLL pose; DLL loss events (maximal runs of DLL-invalid images) detected = overlapped by
  a run of ours.
- **Rotation**: |wrap(ours - DLL)| per yxz angle, median and p95; the median signed pitch error;
  the geodesic angle between the two rotations, median and p95.
- **Position**: |ours - DLL| per axis in the display frame, median and p95, and the 3-D median.
- **Lag**: the phase delay over 0.4-2.6 Hz behind the reference, weighted by coherence, from
  128-image Hann segments in steps of 32 (linear detrend, coherence > 0.2), of ours and of the
  DLL's; the gate takes the largest |ours - DLL| of the three axes.
- **Rest jitter**: the RMS of the residual of a 2-Hz zero-phase low-pass over the stillest
  quarter of the 32-image windows, chosen by the reference's speed; ours / DLL, all three axes
  within the gate's range.

`gates.json` names the sessions by their clock offset, so the gates follow the capture, not the
order of the arguments (a session may also give its number of images, which must then match). It
has, per gate, the metric, the kind (`min`: the value is at least the gate, `max`: at most, `abs`:
its magnitude at most, `range`: every axis inside), `by` (`compare-dll` for the gates
`compare-dll --head` checks, `python` for the lag and the rest jitter, which only `evaluate.py`
measures; `evaluate.py` checks them all) and a threshold per session (`thresholds`). The
thresholds are the study's leave-one-session-out values plus headroom for the port; each was
checked against the broken pipelines, which fail at least one gate in every session.

## Details

- **Time.** Offline, an image's time is its device timestamp; the filters run on it, as the
  daemon runs them on the image's host time.
- **Clock offset.** The DLL's timestamps are the device's minus a constant per session, taken as
  `compare-dll` takes it: from the valid DLL gaze points equal (as f32) to a frame's combined
  gaze point.
- **Display frame.** Fitted to the gaze origins the device reports in both frames (0x02/0x08 in
  the tracker frame, 0x22/0x24 in the display frame), which agrees with the display area of the
  Windows sessions to 1e-4 mm; `--area windows` (or nine numbers) takes an area instead.
- **`--ours-csv`**: `compare-dll --head` writes a row per image with a DLL pose only, so the lag
  and the rest jitter of its poses are taken over the DLL's span of each session. On the
  Windows sessions, against the same poses over every image, that moves an axis's lag by 0.8 ms
  at most (s2's pitch), the lag gates' values by 0.23 ms and the jitter ratios by 0.0002.
- **Without `--reference-fits`** the lag reference is the evaluated fits' own unfiltered pose.
  Our filter then also delays that reference's noise, which adds to our lag, so `evaluate.py`
  prints the lag and the jitter but leaves G15-G18 unchecked.
- **The verdict.** `evaluate.py` exits 0 only when every gate of every session was checked and
  passed: 1 when a gate fails (a lag or a jitter that could not be measured on one of the axes
  fails it too), 3 when none fails but some were not checked (G15-G18 without
  `--reference-fits`, or a session `gates.json` has no thresholds for). The last line says which.
