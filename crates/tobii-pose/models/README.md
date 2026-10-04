# tobii-pose third-party material

`tobii-pose` embeds three pieces of MediaPipe
(<https://github.com/google-ai-edge/mediapipe>) at build time: the
face-landmark and face-detection models in this directory (`include_bytes!`
in `src/track.rs` and `src/detect.rs`) and the canonical face mesh in
`src/canonical.rs`. All three are Apache-2.0, not MIT like the rest of the
repository, so the crate is `MIT AND Apache-2.0`. The binaries that embed
it, `tobiid` and `tobii5-init-replay`, carry all three (`libtobii.so` does
not).

- `LICENSE` is the Apache License 2.0, verbatim from
  <https://www.apache.org/licenses/LICENSE-2.0.txt> (SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
- `NOTICE` names the material and says how it was changed (Apache-2.0
  §4(b)). MediaPipe has no NOTICE file of its own to carry over (§4(d)):
  there is none at the root of its repository or in the model bundle.

Below, per item: what it is, where it comes from, what was changed, what was
checked (2026-10-04) and what is not known.

## `face_landmarks.onnx`

| | |
|---|---|
| What | MediaPipe Face Mesh V2: 478 face landmarks in 3-D (the 468 mesh points and 10 iris points), a face-presence score and a tongue-out score, from a 256x256 face crop |
| Committed file | 4,920,998 bytes, SHA-256 `134ea2dc5d896849d4883cce73d794bb63bd7dfad9b5d032450065b960acfc6c` |
| Upstream | `face_landmarks_detector.tflite` (2,553,590 bytes, SHA-256 `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5`) inside the Face Landmarker bundle `face_landmarker.task`, a zip (3,758,596 bytes, SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`, Last-Modified 2023-05-03): <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>. `float16/latest`, which MediaPipe's Face Landmarker page links, serves the same file |
| Model card | "MediaPipe Face Mesh V2", <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf>, linked from the same page: model date September 15, 2022; authors Geng Yan and Ivan Grishchenko, Google; "LICENSED UNDER Apache License, Version 2.0" |
| Licence | Apache-2.0 |
| Changed | converted from TensorFlow Lite to ONNX with tf2onnx 1.17.0, which also stores the float16 weights as float32 (same values); no other change is known |

### Inputs and outputs, as `src/track.rs` uses them

The names are the `.tflite`'s; its metadata calls them `image`,
`face_landmarks`, `presence` and `tongue_out`. The batch dimension N is 1 in
the `.tflite` and symbolic in the ONNX; `track.rs` passes 1.

| Tensor | Shape | Content and use |
|---|---|---|
| `input_12` | N x 256 x 256 x 3, float32 (NHWC) | the face crop, RGB in [0, 1] (the `.tflite`'s metadata: mean 0, std 255). `FaceModel::landmarks` resamples a square crop of the grey frame, turned by the face's eye line, bilinearly and puts each value in all three channels |
| `Identity` | N x 1 x 1 x 1434 | 478 landmarks x (x, y, z): x and y in pixels of the 256x256 input, z relative to the face's centre of mass and scaled with the face's width (model card). The code keeps the first 468 (`NLM`, the points of the canonical mesh) and drops the 10 iris points |
| `Identity_1` | N x 1 x 1 x 1 | face presence as a logit (no sigmoid in the graph). The code reads below 0, a probability below the model card's default threshold of 0.5, as no face, and a NaN too |
| `Identity_2` | N x 1 | tongue-out score, after a sigmoid; unused |

### ONNX metadata

- IR version 8; opsets `ai.onnx` 17 and `ai.onnx.ml` 2.
- `producer_name` "tf2onnx", `producer_version` "1.17.0 None"; graph name
  "tf2onnx", graph doc string "converted from
  /m/face_landmarks_detector.tflite"; empty model doc string, domain and
  `metadata_props`; `model_version` 0. The `.tflite`'s own metadata (name
  "Face mesh detection model v2", version "1", author "MediaPipe", no licence
  field) did not carry over.
- 223 nodes (106 Conv, 69 PRelu, 34 Add, 6 MaxPool, 3 Pad, 3 Reshape, 1
  Transpose, 1 Sigmoid) and 257 initializers (251 float32, 6 int64);
  `onnx.checker` accepts it.

### Checked

- The bundle as downloaded on 2026-10-04 is byte-identical to the copy this
  repository committed in f0deb29 (`pose.py` still gives its URL), and the
  `.tflite` in it to `mp_models/face_landmarks_detector.tflite`, committed in
  ef03ee5 next to this ONNX (then also `mp_models/landmarks.onnx`, the same
  git blob). d8aa2d4 removed the bundle and `mp_models/`; both are still in
  the history.
- Each of the ONNX's 251 float32 initializers (1,199,316 values) is exactly
  representable in float16 and holds the same values as one of the
  `.tflite`'s 251 float16 constants. The operators match in number: the 106
  Conv are the `.tflite`'s 72 CONV_2D and 34 DEPTHWISE_CONV_2D, and its 69
  PRELU, 34 ADD, 6 MAX_POOL_2D, 3 PAD and 1 LOGISTIC are the 69 PRelu, 34
  Add, 6 MaxPool, 3 Pad and 1 Sigmoid.
- The `.tflite` graph evaluated op by op (NumPy, float32) and the ONNX in
  ONNX Runtime agree on the same inputs: landmarks within 2.3e-4 px,
  presence logits within 2.3e-4, tongue-out within 1.2e-6, over four
  synthetic images (noise, gradients, flat grey) and four face crops from a
  recorded session (presence logits -1.3 to +11.2).

### Not known

- When the conversion was made, other than before 2026-06-03 (ef03ee5), and
  how: the tf2onnx command line and options, and the TensorFlow and Python
  versions it ran with. The version string names no tf2onnx commit, and
  `/m/` in the doc string is a path on the machine that converted it.
- Whether anything edited the graph afterwards. Nothing suggests it (the node
  names are tf2onnx's throughout, and the weights and outputs match the
  `.tflite`), but there is no record.
- The licence of the files themselves: neither the bundle nor the `.tflite`
  carries a licence field or notice. Apache-2.0 is what the model card
  states.

## `blaze_face_short_range.onnx`

| | |
|---|---|
| What | MediaPipe's BlazeFace short-range face detector: the faces in a 128x128 image, each a box, six keypoints (the two eyes, the nose tip, the mouth, the two tragions) and a score, as offsets from 896 SSD anchors |
| Committed file | 425,659 bytes, SHA-256 `efdf0e80235a9cff016232f59c0dadd5135b7941e408de262ad5e9d1bbde292e` |
| Upstream | `blaze_face_short_range.tflite`, the MediaPipe Face Detector's float16 model, version 1 (229,746 bytes, SHA-256 `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`, Last-Modified 2023-04-26): <https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite>. `float16/latest` serves the same file. The `mediapipe` Python package (0.10.35) does not ship it |
| Model card | "MediaPipe BlazeFace Model Card (Short Range)", <https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf>: date June 9, 2021; author Valentin Bazarevsky, Google; "LICENSED UNDER Apache License, Version 2.0" |
| Licence | Apache-2.0 |
| Changed | converted from TensorFlow Lite to ONNX with tflite2onnx 0.4.1, default options: the NHWC graph becomes an NCHW one, and the float16 weights and their DEQUANTIZE operators become float32 initializers of the same values. `tools/headpose/blaze_face_to_onnx.py` converts the upstream file the same way and checks the result against this one |

### Inputs and outputs, as `src/detect.rs` uses them

The names are the `.tflite`'s; its metadata calls them `image`, `raw
boxes/keypoints` and `scores`.

| Tensor | Shape | Content and use |
|---|---|---|
| `input` | 1 x 3 x 128 x 128, float32 (NCHW) | the image scaled to [-1, 1] (the `.tflite`'s metadata: mean 127.5, std 127.5). `FaceDetector::detect` resizes the camera's own grey frame (280x280 for the 0x50e stream) to 128x128 as OpenCV's `cv2.resize(..., interpolation=INTER_LINEAR)` does for an 8-bit image, bit for bit, as the prototype did; takes x / 127.5 - 1 of each value and puts it in all three channels |
| `regressors` | 1 x 896 x 16 | per anchor, in input pixels: the box centre's offset from the anchor (x, y), the box's width and height, then x and y of the six keypoints, also from the anchor. The code divides them by 128 for fractions of the image |
| `classificators` | 1 x 896 x 1 | per anchor, the face score as a logit. The code clamps it to ±100 and takes its sigmoid |

The anchors and the decoding are those of MediaPipe's
`face_detection_short_range` graph: four layers of strides 8, 16, 16 and 16,
two anchors per layer and cell, all of unit size, centred in their cells
(512 on the 16x16 grid, then 384 on the 8x8 one); weighted non-maximum
suppression, merging the candidates that overlap the best remaining one by
an intersection over union above 0.3, their boxes and keypoints averaged
with their scores as weights. The score threshold is 0.3, the
`min_detection_confidence` the prototype ran MediaPipe's own detector with
(MediaPipe's default is 0.5). The tracker looks for a lost face in one
detection: the one nearest the last face it found, or the best-scoring one
before it has found any (the prototype took the best-scoring one every
time). Of that detection's keypoints it uses only the two eyes.

### ONNX metadata

- IR version 6; opset `ai.onnx` 11.
- `producer_name` "tflite2onnx", empty `producer_version`; graph name
  "pre-alpha" (tflite2onnx's own placeholder); empty doc strings, domain and
  `metadata_props`; `model_version` 0. The `.tflite`'s own metadata (name
  "Short Range Face Detection", description "Detects human face with
  frontal camera", version "1", author "MediaPipe", no licence field) did not
  carry over.
- 94 nodes (37 Conv, 17 Relu, 16 Add, 11 Pad, 4 Transpose, 4 Reshape, 3
  MaxPool, 2 Concat) and 87 initializers (74 float32, 13 int64);
  `onnx.checker` accepts it.

### Checked

- The `.tflite` as downloaded on 2026-10-04 is byte-identical to the copy
  the Python prototype of the head-pose study ran (in the gitignored
  `fixtures/`).
- Each of the ONNX's 74 float32 initializers (101,390 values) is exactly
  representable in float16 and holds the same values as one of the
  `.tflite`'s 74 float16 constants. The operators match in number: the 37
  Conv are the `.tflite`'s 21 CONV_2D and 16 DEPTHWISE_CONV_2D, and its 17
  RELU, 16 ADD, 11 PAD, 4 RESHAPE, 3 MAX_POOL_2D and 2 CONCATENATION are the
  17 Relu, 16 Add, 11 Pad, 4 Reshape, 3 MaxPool and 2 Concat; the 4
  Transpose turn the NCHW feature maps back to NHWC before the reshapes into
  the outputs.
- Converting the `.tflite` again with tflite2onnx 0.4.1
  (`tools/headpose/blaze_face_to_onnx.py`) gives the same graph: the same
  nodes, inputs and outputs, and every initializer and `value_info` entry the
  same by name. The bytes differ only in the order of those two lists, which
  tflite2onnx takes from a Python set of tensor objects and which changes
  from run to run; sorted by name, the two files are byte-identical.
- Against MediaPipe's own detector (the Tasks API's `FaceDetector` on the
  `.tflite`) on 400 frames of a recorded session, the ONNX in ONNX Runtime
  with the decoding above: both found a face on 393 frames, MediaPipe alone
  on 1, neither on 6. Where both did, in the 280x280 image: box centres
  0.89 px apart in the median (2.78 at p95), the widths' ratio 1.006, the
  largest difference in an eye keypoint's coordinates 1.34 px (3.67 at
  p95), the scores the same (median difference 0.000).
- In the prototype's tracker, which runs the same code as before on every
  frame the detector is not involved in (its replay of those frames
  reproduces the earlier runs exactly), putting this model in place of
  MediaPipe's own detector loses the face on 1 and 2 more of the frames on
  which the Stream Engine had a head pose, in two of three recorded sessions
  (96.86 and 99.98 % of them kept, against 96.88 and 100 %).

### Not known

- The command that made the committed file, on 2026-09-27 (its date) for the
  head-pose study's prototype: no record of it was kept but the converter,
  tflite2onnx 0.4.1. That it ran with the default options follows from the
  reconversion above.
- The licence of the files themselves: neither the `.tflite` nor the ONNX
  carries a licence field or notice. Apache-2.0 is what the model card
  states.

## Canonical face mesh (`src/canonical.rs`)

| | |
|---|---|
| What | the 468 vertices (cm) of MediaPipe's canonical face model, the neutral 3-D face that MediaPipe's face-geometry module aligns to the landmarks. `track.rs` fits it to the landmarks (Kabsch, then PnP) to get the head pose |
| Committed form | `CANONICAL_FACE: [[f32; 3]; 468]` |
| Upstream | `mediapipe/modules/face_geometry/data/canonical_face_model.obj` in google-ai-edge/mediapipe, 45,999 bytes, SHA-256 `8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618`, added in upstream commit a908d668 (2020-09-16) and unchanged since. The copy under `mediapipe/tasks/cc/vision/face_geometry/data/` (2023) is the same file |
| Licence | Apache-2.0: the repository's licence. The `.obj` has no header; the `BUILD` file beside it says "Copyright 2020 The MediaPipe Authors" |
| Changed | only the 468 vertex positions (`v` lines) are kept, without the texture coordinates (`vt`) and faces (`f`). y and z are negated, a 180° turn about x from the `.obj`'s frame (y up, the face looking along +z) to the camera's (y down, the nose towards -z). Each value is rounded to five significant digits, at most 5e-5 cm from the `.obj` |

Checked: printing the `.obj`'s vertices that way (`%.5g`, y and z negated)
reproduces all 1404 values in `canonical.rs`. The `.obj` committed next to it
in ef03ee5 (`mp_models/canonical_face_model.obj`, removed in d8aa2d4) is
byte-identical to upstream's.

Not known: the script that wrote `canonical.rs`; ef03ee5 does not contain
it.
