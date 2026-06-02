#!/usr/bin/env python3
"""
Head pose from the Tobii IR camera -> OpenTrack UDP (gaze-independent).

Reads raw 560x560 8-bit frames from a FIFO (fed by `tobii camera --fifo`),
runs MediaPipe FaceLandmarker (Tasks API), takes head rotation from the facial
transformation matrix (face geometry, NOT eye direction), and sends OpenTrack
"UDP over network" packets (TX,TY,TZ,Yaw,Pitch,Roll as 6 little-endian f64).

Setup:
  pip install mediapipe opencv-python numpy
  curl -L -o face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Run:
  mkfifo /tmp/tobii_cam
  sudo ./target/release/tobii5-init-replay camera --fifo /tmp/tobii_cam &
  python3 pose.py /tmp/tobii_cam
"""
import sys, socket, struct, time
import numpy as np

W = H = 560
FRAME_BYTES = W * H
OPENTRACK_ADDR = ("127.0.0.1", 4242)
MODEL = "face_landmarker.task"

# Tune after a first look (flip a sign if an axis goes the wrong way).
ANGLE_SIGN = (1.0, 1.0, 1.0)   # (pitch, yaw, roll)
ANGLE_GAIN = (1.0, 1.0, 1.0)
# Translation from the rotation-decoupled facial transformation matrix (~cm).
SEND_TRANSLATION = True
TRANS_SIGN = (-1.0, 1.0, -1.0)  # camera Y is down; flip if an axis is reversed
TRANS_GAIN = (1.2, 1.2, 1.2)
SMOOTH = 0.5                   # EMA on output (0..1, lower = smoother)
CALIB_FRAMES = 30             # frames averaged for the rest pose
CLAMP_DEG = 45.0


def euler_deg(M):
    R = M[:3, :3]
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: pose.py <fifo-or-file> [model.task]")
    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
    except ImportError as e:
        sys.exit(f"missing dep ({e}); pip install mediapipe opencv-python numpy")

    # IMAGE mode: treat every frame independently. The device interleaves a
    # full-face image with a non-face one; VIDEO mode's temporal tracker gets
    # corrupted by that, so per-frame detection is more reliable here.
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_facial_transformation_matrixes=True,
    )
    lmk = vision.FaceLandmarker.create_from_options(opts)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    origin = None
    accum = []
    out = np.zeros(6)
    have = False
    frames = 0
    faces = 0
    last_raw = None
    rawmin = np.array([1e9, 1e9, 1e9])
    rawmax = np.array([-1e9, -1e9, -1e9])
    tmin = np.array([1e9, 1e9, 1e9])
    tmax = np.array([-1e9, -1e9, -1e9])
    t0 = time.time()

    print(f"opening {path} ...", file=sys.stderr)
    with open(path, "rb") as f:
        while True:
            data = f.read(FRAME_BYTES)
            if len(data) < FRAME_BYTES:
                break
            gray = np.frombuffer(data, np.uint8).reshape(H, W)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = lmk.detect(mpimg)
            frames += 1

            pose = None
            raw_ang = None
            if res.facial_transformation_matrixes:
                M = np.array(res.facial_transformation_matrixes[0])
                ang = euler_deg(M)             # pitch, yaw, roll (deg)
                raw_ang = ang
                last_raw = ang
                rawmin = np.minimum(rawmin, ang)
                rawmax = np.maximum(rawmax, ang)
                tr = M[:3, 3]                  # face origin in camera space (~cm)
                tmin = np.minimum(tmin, tr)
                tmax = np.maximum(tmax, tr)
                pose = np.array([tr[0], tr[1], tr[2], ang[0], ang[1], ang[2]])
                faces += 1

            if pose is not None:
                if origin is None:
                    accum.append(pose)
                    if len(accum) >= CALIB_FRAMES:
                        origin = np.mean(accum, axis=0)
                        print("\ncalibrated rest pose", file=sys.stderr)
                    continue
                rel = pose - origin
                tx, ty, tz = (rel[0] * TRANS_SIGN[0] * TRANS_GAIN[0],
                              rel[1] * TRANS_SIGN[1] * TRANS_GAIN[1],
                              rel[2] * TRANS_SIGN[2] * TRANS_GAIN[2])
                pitch = np.clip(rel[3] * ANGLE_SIGN[0] * ANGLE_GAIN[0], -CLAMP_DEG, CLAMP_DEG)
                yaw = np.clip(rel[4] * ANGLE_SIGN[1] * ANGLE_GAIN[1], -CLAMP_DEG, CLAMP_DEG)
                roll = np.clip(rel[5] * ANGLE_SIGN[2] * ANGLE_GAIN[2], -CLAMP_DEG, CLAMP_DEG)
                if not SEND_TRANSLATION:
                    tx = ty = tz = 0.0
                target = np.array([tx, ty, tz, yaw, pitch, roll])  # OpenTrack order
                out = target if not have else out + SMOOTH * (target - out)
                have = True

            if have:
                sock.sendto(struct.pack("<6d", *out), OPENTRACK_ADDR)

            if frames % 16 == 0:
                fps = frames / (time.time() - t0)
                if last_raw is not None:
                    trng = tmax - tmin
                    lr = (f"yaw/pit/roll={out[3]:+5.1f}/{out[4]:+5.1f}/{out[5]:+5.1f}  "
                          f"tx/ty/tz={out[0]:+5.1f}/{out[1]:+5.1f}/{out[2]:+5.1f}  "
                          f"trange={trng[0]:4.1f}/{trng[1]:4.1f}/{trng[2]:4.1f}")
                else:
                    lr = "no face yet"
                print(f"\r{fps:4.0f}fps hit={100*faces//max(frames,1):3d}%  {lr}   ",
                      end="", file=sys.stderr)


if __name__ == "__main__":
    main()
