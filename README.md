# tobii-linux

A Linux driver stack for the **Tobii Eye Tracker 5** (USB `2104:0313`), built
by reverse-engineering the device's USB protocol. It gives you gaze, user
presence and gaze-independent 6-DOF head pose — all at 33 Hz, all at the same
time — plus calibration, through a daemon, a drop-in `libtobii.so` for Tobii's
Stream Engine 4.1 C API, and a few thin clients.

Tobii ships no Linux support for this device. Everything here was recovered
from USB captures of the Windows Stream Engine.

> Status: works daily on the author's desk. Independently reimplemented from
> USB captures; not affiliated with or endorsed by Tobii.

## What it does

| Signal | Rate | Source |
|---|---|---|
| Gaze point + validity | 33 Hz | the device's processed `0x500` stream |
| Per-eye gaze origin, eye position, gaze data | 33 Hz | the same stream |
| User presence | on change | the `0x504` stream |
| Head pose, 6 DOF | 33 Hz | face landmarks on the device's own IR frames |
| IR camera image, 280×280 | 33 Hz | the `0x50e` stream |
| Device info, track box, display area, notifications | on request / change | the device's own answers |
| Calibration | on demand | `tobii-calibrate`, saved per user |

Everything the Stream Engine reports is reproduced bit for bit: replaying a
captured Windows session through the decoder gives exactly the gaze points
and gaze origins the Stream Engine delivered for it (5 284 of 5 284 frames,
error 0; `tobii5-init-replay compare-dll`).

The head pose is **gaze-independent**: turning your eyes does not move it.

### The interesting part

The device multiplexes a second stream, `0x50e` `primary_camera_image`, on the
same bulk endpoint as gaze — a 280×280 8-bit IR frame of your face at 33 Hz.
The Windows driver subscribes to it with command 1220 right after init; nothing
in the documented API hints it exists.

That matters because the eye-position fields in the gaze stream **cannot** give
you a signed head yaw. Both eyes are reconstructed at equal range from the
camera, so the per-eye depth is synthetic and any "yaw" computed from it is
lateral head position in disguise. Feeding the IR frames to a face-landmark
model instead gives a real signed yaw at ~2° resolution, and it costs about
6 ms of CPU per frame. That conclusion came out of a field-by-field study of
11 800 recorded frames, cross-checked against head poses read off the IR images
themselves; the tooling for it is in `tobii-tools`.

## Quick start

```bash
cargo build --release --workspace     # or: make build
make install                          # binaries + libtobii.so + headers + udev rules + user units
make enable                           # start the daemon now
tobii-calibrate                       # calibrate for your eyes (once)
```

Then either link against `libtobii.so`, or run a client:

```bash
tobii-opentrack                       # head pose -> OpenTrack UDP 127.0.0.1:4242
tobii-gaze-keys                       # look at a screen edge + hold Super -> arrow keys
```

Full instructions, tuning knobs and troubleshooting are in
**[INSTALL.md](INSTALL.md)**.

Requirements: a stable Rust toolchain, libusb, and the tracker plugged in. The
face model and the init capture are embedded at build time, so the installed
binaries need no runtime data files.

## Architecture

```
libtobii.so ─────┐
tobii-opentrack ─┤
tobii-gaze-keys ─┼── unix socket ──► tobiid ──USB──► Tobii ET5
tobii-calibrate ─┘                      │
                              EP 0x83: gaze 0x500, presence 0x504, IR image 0x50e
```

`tobiid` is the only process that claims the device; everyone else is a client,
so several consumers can read gaze and head pose at once.

`libtobii.so` exports **all 153 entry points** of Tobii's
`tobii_stream_engine.dll` 4.1.0.3, with its signatures, error numbering and
struct layouts, plus a `tobii_recenter` extension — so any Stream Engine client
links and runs. The ABI was recovered from the DLL itself (`tools/abi/`) and
the archived 4.1.0 reference. What stands behind the entry points:

| | Entry points |
|---|---|
| Implemented (daemon-backed) | device lifetime and callbacks; gaze point, gaze origin, eye position, user position guide, presence, head pose, gaze data, IR image and notification streams; device info, track box, display area (get/set), mounting, states; 2-D calibration |
| Answered locally | API version, system clock, output frequency (33 Hz), enabled eye, capabilities, feature group, license validation, display-area calculation, calibration parsing |
| `TOBII_ERROR_NOT_SUPPORTED` | what the ET5 was never observed doing: wearable, face id, illumination, power and pause, firmware, diagnostics, extensions, custom streams, 3-D and per-eye calibration |

The headers are in `crates/tobii-ffi/include/tobii/` (installed to
`/usr/local/include/tobii/`); OpenTrack's `tracker-tobii` plugin builds against
them unchanged (INSTALL.md §8).

### Workspace

Eleven crates, split so the driver never compiles the research tooling and
`libtobii.so` links neither ONNX Runtime nor the embedded assets (it is 0.45 MB
rather than 20 MB).

| Crate | Holds | Heavy deps |
|---|---|---|
| `tobii-proto` | wire formats: framing, TLV, commands, the gaze/presence/image streams, device facts, the capture log | none |
| `tobii-pose` | face landmarks and the head-pose fit; owns the model | `ort` |
| `tobii-usb` | USB transport and the live `0x83` engine; owns the init capture | `rusb` |
| `tobii-ipc` | the daemon protocol and the display geometry | none (std only) |
| `tobii-calib` | the calibration blob format and the per-user store | none (std only) |
| `tobii-log` | shared `tracing` setup | — |
| `tobiid` | the daemon | — |
| `tobii-ffi` | `libtobii.so` (cdylib) | — |
| `tobii-clients` | `tobii-opentrack`, `tobii-gaze-keys` | — |
| `tobii-calibrate` | the calibration window | `winit`, `softbuffer` |
| `tobii-tools` | `tobii5-init-replay`: log analysis, UVC camera, diagnostics | all of the above |

`make check` runs fmt, clippy with `-D warnings`, the tests and rustdoc.
`make verify-abi` asserts `libtobii.so` exports exactly the 154 symbols in
`crates/tobii-ffi/abi-symbols.txt`, then compiles and runs
`crates/tobii-ffi/abi-smoke.c` against the headers: it takes the address of
every symbol through them and checks versions, rejections and struct layouts.

## Research tooling

`tobii5-init-replay` is built but not installed — none of it is needed to run
the driver. It is the bench the protocol was worked out on, and it is still the
regression harness:

```bash
target/release/tobii5-init-replay image83 --secs 10 --pose   # live A/B of both streams
target/release/tobii5-init-replay probe                      # 0x83 vs UVC concurrency
target/release/tobii5-init-replay image83-replay log.bin --csv out.csv
target/release/tobii5-init-replay analyze-log log.bin        # blind field scan
target/release/tobii5-init-replay head-axes yaw:a.bin roll:b.bin
target/release/tobii5-init-replay compare-dll session.bin session.jsonl  # decoder vs the DLL
target/release/tobii5-init-replay ipc-probe --secs 5         # ask the running daemon everything
tools/abi/dll_abi.py headers                                 # headers vs tobii_stream_engine.dll
```

## Extras

- **[paperwm-gaze/](paperwm-gaze/)** — a GNOME Shell extension that lets you
  pick a window on the PaperWM Alt+Tab minimap by looking at it.
- **[tools/splice_calibration.py](tools/splice_calibration.py)** — injects a
  calibration blob into a captured init sequence (superseded by
  `tobii-calibrate`, kept for research).

## Limitations

- One process owns the device. `tobiid` arbitrates; run everything else as a
  client.
- The UVC camera interface and the `0x83` streams are mutually exclusive in
  firmware. Nothing in the driver uses UVC any more, but keep the shipped
  `99-tobii-no-uvcvideo.rules` in place so nothing else grabs it.
- Head pose needs a face in frame; it reports nothing when you look away.
- Until you run `tobii-calibrate`, the calibration in use is the one embedded
  in `init_packets_ep.txt` — the author's. Likewise the display area is the
  author's 27" monitor until you set yours: `tobii-calibrate` does it first
  (you line two ticks up with the marks on the tracker), and the daemon keeps
  whatever display area is set (`~/.config/tobii/display-area`).
- The calibration sequence mirrors the one captured from Windows; 3-D and
  per-eye calibration were never captured and are not supported.
- The ET5 reports no pupil diameter; `tobii_gaze_data_t.pupil_validity` is
  always invalid.

## License

MIT.
