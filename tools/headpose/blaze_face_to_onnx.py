#!/usr/bin/env python3
"""Convert MediaPipe's BlazeFace short-range face detector to the ONNX model
tobii-pose embeds (crates/tobii-pose/models/blaze_face_short_range.onnx), and
check a converted model against the committed one.

The upstream model is the MediaPipe Face Detector's float16 model, version 1
(229,746 bytes, SHA-256 below):

  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

The conversion is tflite2onnx 0.4.1 with its default options (no explicit
layouts): it turns the NHWC graph into an NCHW one (input `input`
1x3x128x128) and folds the float16 weights and their DEQUANTIZE operators into
float32 initializers of the same values. The result's graph is the committed
model's; its bytes are not, because tflite2onnx emits the initializers and
value_info in the order of a Python set of tensor objects, which changes from
run to run. `--check` therefore compares the graphs with those two lists
sorted by name.

    blaze_face_to_onnx.py convert TFLITE OUT.onnx
    blaze_face_to_onnx.py check OUT.onnx crates/tobii-pose/models/blaze_face_short_range.onnx

Needs tflite2onnx 0.4.1 (which pulls in onnx and tflite) and numpy. Nothing
is downloaded; fetch the .tflite yourself. Scratch output belongs in
fixtures/, which is gitignored.
"""

import argparse
import hashlib
import sys

UPSTREAM_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
COMMITTED_SHA256 = "efdf0e80235a9cff016232f59c0dadd5135b7941e408de262ad5e9d1bbde292e"
CONVERTER_VERSION = "0.4.1"
# What tobii-pose's detect.rs reads: names, element type FLOAT, shapes.
SIGNATURE = (
    [("input", [1, 3, 128, 128])],
    [("regressors", [1, 896, 16]), ("classificators", [1, 896, 1])],
)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def signature(model):
    """(inputs, outputs) as [(name, [dims])], every one a FLOAT tensor."""
    import onnx

    def io(values):
        out = []
        for v in values:
            t = v.type.tensor_type
            if t.elem_type != onnx.TensorProto.FLOAT:
                sys.exit(f"{v.name}: not a float tensor")
            out.append((v.name, [d.dim_value for d in t.shape.dim]))
        return out

    return io(model.graph.input), io(model.graph.output)


def normalised(model):
    """The model's bytes with the initializers and value_info sorted by name."""
    for field in (model.graph.initializer, model.graph.value_info):
        items = sorted(field, key=lambda t: t.name)
        del field[:]
        field.extend(items)
    return model.SerializeToString()


def convert(args):
    import tflite2onnx

    if tflite2onnx.__version__ != CONVERTER_VERSION:
        sys.exit(f"tflite2onnx {tflite2onnx.__version__}, not {CONVERTER_VERSION}")
    digest = sha256(args.tflite)
    print(f"{args.tflite}: SHA-256 {digest}")
    if digest != UPSTREAM_SHA256:
        print(f"  not the upstream float16/1 model ({UPSTREAM_SHA256})", file=sys.stderr)
        if not args.force:
            sys.exit("refusing to convert it (--force to go on)")
    tflite2onnx.convert(args.tflite, args.onnx)
    import onnx

    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)
    got = signature(model)
    if got != SIGNATURE:
        sys.exit(f"inputs/outputs {got}, expected {SIGNATURE}")
    print(f"{args.onnx}: SHA-256 {sha256(args.onnx)}; inputs/outputs {got}")


def check(args):
    import onnx

    a, b = onnx.load(args.converted), onnx.load(args.committed)
    digest = sha256(args.committed)
    print(f"{args.committed}: SHA-256 {digest}"
          + ("" if digest == COMMITTED_SHA256 else f" (the committed model is {COMMITTED_SHA256})"))
    for name, model in (("converted", a), ("committed", b)):
        if signature(model) != SIGNATURE:
            sys.exit(f"{name}: inputs/outputs {signature(model)}, expected {SIGNATURE}")
    if normalised(a) != normalised(b):
        sys.exit("the graphs differ (initializers and value_info compared by name)")
    print("same graph: nodes, inputs, outputs, and initializers and value_info by name")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="convert the upstream .tflite with tflite2onnx")
    c.add_argument("tflite")
    c.add_argument("onnx")
    c.add_argument("--force", action="store_true", help="convert a .tflite that is not upstream's")
    c.set_defaults(func=convert)
    k = sub.add_parser("check", help="compare a converted model with the committed one")
    k.add_argument("converted")
    k.add_argument("committed")
    k.set_defaults(func=check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
