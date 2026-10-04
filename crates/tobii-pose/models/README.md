# tobii-pose third-party material

`tobii-pose` embeds two pieces of MediaPipe
(<https://github.com/google-ai-edge/mediapipe>) at build time: the
face-landmark model in this directory (`include_bytes!` in `src/track.rs`)
and the canonical face mesh in `src/canonical.rs`. Both are Apache-2.0, not
MIT like the rest of the repository, so the crate is `MIT AND Apache-2.0`.
The binaries that embed it, `tobiid` and `tobii5-init-replay`, carry both
(`libtobii.so` does not).

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
