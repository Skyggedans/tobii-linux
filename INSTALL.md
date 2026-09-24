# Install & Run

Linux stack for the **Tobii Eye Tracker 5** (USB `2104:0313`): a daemon that
claims the device once and serves head pose / gaze / presence (and everything
else the Stream Engine offers) to multiple clients, a drop-in `libtobii.so` for
the Stream Engine 4.1 C API, a calibration tool and an OpenTrack bridge.

```
libtobii.so / tobii-opentrack ──unix socket──▶ tobiid ──USB──▶ Tobii ET5
                                                 │
                              EP 0x83: gaze + presence (stream 0x500, 33 Hz)
                                     + 280x280 IR face image (stream 0x50e, 33 Hz)
                                       → 6DOF head pose (MediaPipe landmarks + PnP)
                              all served concurrently by one engine
```

Head pose does not use the UVC camera: the device multiplexes its own IR image
stream on the same endpoint as gaze (what the Windows Stream Engine uses), so
head pose, gaze and presence work at the same time. The UVC camera survives
only in the standalone research subcommands (`camera`, `track`, `probe`).

---

## 1. Prerequisites

- Rust toolchain (stable) + Cargo.
- The repo includes the model (`crates/tobii-pose/models/face_landmarks.onnx`)
  and the init capture (`crates/tobii-usb/init_packets_ep.txt`); both are
  embedded at **build** time, so no runtime
  data files are needed.
- A Tobii Eye Tracker 5 plugged in.

## 2. Build

```bash
cargo build --release --workspace     # or: make build
```

Produces in `target/release/`:

| Artifact | What it is | Installed? |
|---|---|---|
| `tobiid` | the daemon (claims the device, serves clients) | yes |
| `tobii-opentrack` | thin client: head pose → OpenTrack UDP | yes |
| `tobii-gaze-keys` | thin client: gaze at screen edge → Left/Right arrow key | yes |
| `tobii-calibrate` | fullscreen calibration (Wayland and X11) | yes |
| `libtobii.so` | the Stream Engine 4.1 C ABI, all 153 entry points; a daemon client | yes |
| `tobii/*.h` | the C headers for `libtobii.so` (`$PREFIX/include/tobii/`) | yes |
| `tobii5-init-replay` | log analysis, UVC camera and diagnostics | no — run it from `target/release/` |

### Workspace layout

The driver and the research tooling are separate crates, so the daemon never
compiles the analysis code and `libtobii.so` links neither ONNX Runtime nor the
embedded assets (it is ~0.45 MB rather than ~20 MB).

| Crate | What it holds | Heavy deps |
|---|---|---|
| `tobii-proto` | wire formats: framing, TLV, commands and responses, the 0x500/0x504/0x50e streams, device facts, the `TBI5LOG1` log | none |
| `tobii-pose` | face landmarks and the head-pose fit; owns `models/` | `ort` |
| `tobii-usb` | USB transport and the live 0x83 engine; owns `init_packets_ep.txt` | `rusb` |
| `tobii-ipc` | the daemon protocol and the display geometry | none (std only) |
| `tobii-calib` | the calibration blob format and the per-user store | none (std only) |
| `tobii-log` | shared `tracing` subscriber setup | — |
| `tobiid` | the daemon binary | — |
| `tobii-ffi` | `libtobii.so` (cdylib) | — |
| `tobii-clients` | `tobii-opentrack`, `tobii-gaze-keys` | — |
| `tobii-calibrate` | the calibration window | `winit`, `softbuffer` |
| `tobii-tools` | `tobii5-init-replay`: analysis, UVC camera, diagnostics | all of the above |

`make check` runs what CI would: `cargo fmt --all --check`, clippy with
`-D warnings` over all targets, the tests and `cargo doc`. `make verify-abi`
asserts `libtobii.so` exports exactly the 154 symbols listed in
`crates/tobii-ffi/abi-symbols.txt` (every export of the reference DLL plus
`tobii_recenter`), then compiles and runs `crates/tobii-ffi/abi-smoke.c`
against the headers so the C declarations and the library cannot drift apart.
With the reference DLL at hand, `tools/abi/dll_abi.py headers` checks every
prototype against the DLL's machine code (`tools/abi/README.md`).

## 2b. One-shot install (Makefile)

The fastest path — installs binaries + `libtobii.so` to `/usr/local` (plus
`/etc/ld.so.conf.d/tobii.conf`, so the dynamic loader finds the library
where it does not search `/usr/local/lib` by itself, e.g. Fedora), the udev
rule, and the systemd **user** units (with `ExecStart` rewritten to the
installed binary). Run it as **your user**; it invokes `sudo` for the system
parts itself:

```bash
make install      # build + binaries + lib + udev rule + user units
make enable       # start the always-on user service now
```

Other targets: `make build`, `make disable`, `make uninstall`, `make clean`.
Override locations with e.g. `make install PREFIX=/usr`.

After installing this way you can skip §3 (udev) and §4b/§4c (units) below — they
were done for you. Continue at §6 (logs) / §8 (clients). The rest of this
document covers the manual steps and details.

## 3. Device permissions (run without root)

Install the udev rule so the logged-in user can claim the device:

```bash
sudo cp systemd/99-tobii-uaccess.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
# then re-plug the tracker (or it applies on next connect)
```

Check: `ls -l /dev/bus/usb/$(lsusb | awk '/2104:0313/{printf "%03d/%03d",$2,$4}')`
should be group-readable/writable for you.

**Recommended:** also install the rule that keeps the kernel `uvcvideo` driver
off this device. We drive the IR camera over libusb, not V4L2; uvcvideo only
races our init (a source of "works every other start" flakiness) and the device
isn't a usable webcam anyway:

```bash
sudo cp systemd/99-tobii-no-uvcvideo.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger   # then re-plug
```

(`make install` installs both udev rules for you.) Verify it took effect:
`ls /dev/video*` should no longer show a node for the tracker, and
`lsusb -t` should list interface 2 of `2104:0313` with no `uvcvideo` driver.

**Only for `tobii-gaze-keys`:** it injects key presses through `/dev/uinput`,
which is root-only by default. To run it without `sudo`, hand the device to the
`input` group and join that group:

```bash
sudo cp systemd/99-tobii-uinput.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger /dev/uinput
sudo usermod -aG input "$USER"        # then log out/in for the group to apply
```

Check: `ls -l /dev/uinput` should show group `input` with mode `crw-rw----`, and
`id -nG` should list `input`. (Skip this if you'll just run the tool with `sudo`.)

---

## 4. Running

You have three options. **Pick one.**

### 4a. Quick start (no systemd)

Just run a client — it auto-spawns `tobiid` if none is listening:

```bash
./target/release/tobii-opentrack            # head pose → OpenTrack 127.0.0.1:4242
./target/release/tobii-opentrack --host 127.0.0.1 --port 4242
```

> After a rebuild, kill the old daemon so a fresh one spawns:
> `pkill -f release/tobiid`.

### 4b. systemd user service (always-on) — recommended

```bash
mkdir -p ~/.config/systemd/user
cp systemd/tobiid.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tobiid
systemctl --user status tobiid
```

Now clients just connect (no auto-spawn, no "kill old daemon" dance).
The `ExecStart` path in the unit is the dev build (`%h/Work/tobii/target/...`);
change it to `/usr/local/bin/tobiid` after installing the binary there.

### 4c. systemd socket activation (start on first use)

`tobiid` supports the `sd_listen_fds(3)` protocol (uses the passed socket at
fd 3). systemd creates the socket up front and starts the daemon on the first
connection:

```bash
cp systemd/tobiid.service systemd/tobiid.socket ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tobiid.socket   # enable the SOCKET, not the service
```

The socket lives at `%t/tobiid.sock` = `$XDG_RUNTIME_DIR/tobiid.sock`, matching
what clients connect to. (The daemon does not auto-stop when idle; it does not
hold the device until a client subscribes.)

---

## 5. After a rebuild

```bash
cargo build --release --workspace
systemctl --user restart tobiid        # 4b
# or: systemctl --user restart tobiid.socket   # 4c
# or: pkill -f release/tobiid                   # 4a (next client respawns it)
```

## 6. Logs & debug

All binaries log through `tracing` to **stderr**: one line per event with a
timestamp, level and structured `key=value` fields (colour only on a
terminal). The level is set with `RUST_LOG` (default `info`; ONNX Runtime's
own chatter is capped at `warn` unless you override it), e.g.
`RUST_LOG=debug` to also see the per-packet USB handshake trace, or
`RUST_LOG=tobii=debug,ort=warn`. With a systemd unit it all lands in the
journal:

```bash
journalctl --user -u tobiid -f
```

Diagnostic subcommands (`image83`, `probe`, `head83`, the analysis tools)
print their **reports on stdout**; only status/warnings go to stderr, so
the reports can be piped or redirected cleanly.

(Auto-spawned daemons — option 4a — have their stdio sent to `/dev/null`; run
`tobiid` by hand in a terminal if you want to see its output there.)

---

## 7. Tuning (environment variables on the daemon)

The head-pose tracker reads these at start. Set them on **`tobiid`** (the
tracker runs in the daemon, not in the client):

| Var | Default | Meaning |
|---|---|---|
| `TOBII_PIVOT_DOWN` | `11.0` | neck pivot below the face origin (cm) — fixes yaw sliding |
| `TOBII_PIVOT_BACK` | `6.0` | neck pivot behind the face origin (cm) — fixes pitch sliding |
| `TOBII_POSE_DEBUG` | unset | log raw `t` vs pivoted `t'` and angles |
| `TOBII_ROLL_EYELINE` | unset | `1` measures roll directly from the eye line (decoupled from yaw/pitch, symmetric by construction) instead of from the euler solve |
| `TOBII_PREWARM` | unset | `1` (or the historical `head` / `gaze`): init the device at daemon start and keep it warm, so client connects are instant (IR illuminator stays on while the service runs). |
| `TOBII_NO_RESET` | unset | `1` skips the USB reset at init (a couple seconds faster; the reset rarely helps now that uvcvideo is kept off the device) |
| `TOBII_NO_IMAGE` | unset | `1` does not start the 0x50e image stream (gaze/presence only; no head pose from the gaze engine) |
| `TOBII_IMAGE83_DEBUG` | unset | `1` logs the image head-pose worker's frame/pose rate and inference time every 5 s (an `info` event with `frames_per_s`/`poses_per_s`/`mean_ms` fields) |
| `RUST_LOG` | `info` | log filter for all binaries (`debug`, `tobii=debug,ort=warn`, …); see §6 |
| `TOBII_DISPLAY_MM` | unset | your monitor as `<width>x<height>[+<offset_x>]` in mm, e.g. `597x336`: once the tracker reports its mounting, the daemon computes the display area (the Stream Engine's `tobii_calculate_display_area_basic`) and writes it, replacing the capture author's monitor that the init replay configures. Gaze coordinates are relative to this area. `offset_x` is how far right of the tracker the screen centre is. Usually unneeded: `tobii-calibrate` sets the display area and the daemon keeps it (§8a); a saved display area wins over this variable, which only fills in while nothing is saved (delete `~/.config/tobii/display-area` to use it). |
| `TOBII_CALIBRATION` | unset | where the calibration is kept (default `$XDG_CONFIG_HOME/tobii/calibration.bin`), or `embedded` to use the built-in one (§8a) |
| `TOBII_CAMERA_TILT_DEG` | `20` | upward tilt of the tracker camera; head angles are reported in the upright frame (yaw about the true vertical), so a turn does not leak into roll |

Head-pose inference on the image stream runs only while some client subscribes
to head pose (about 6 ms per frame at 33 Hz on a desktop CPU); gaze-only
clients don't pay for it.

Raise the pivots if pure rotations still **slide**; lower them if they
**over-shoot** (slide the other way). For a systemd unit:

```bash
systemctl --user edit tobiid
# add:
#   [Service]
#   Environment=TOBII_PIVOT_DOWN=14 TOBII_PIVOT_BACK=8
#   Environment=TOBII_POSE_DEBUG=1
systemctl --user restart tobiid
```

For option 4a, env on the client propagates to the daemon **only if the client
spawns it** (so `pkill -f release/tobiid` first):

```bash
pkill -f release/tobiid
TOBII_PIVOT_DOWN=14 TOBII_PIVOT_BACK=8 ./target/release/tobii-opentrack
```

---

## 8. Clients

- **`tobii-opentrack`** — subscribes head pose, sends the OpenTrack UDP packet
  (`x,y,z,yaw,pitch,roll`; translation in **cm**, angles in **degrees**). In
  OpenTrack pick *Input = UDP over network*, port 4242.
- **`libtobii.so`** — the Stream Engine 4.1 C API: every one of the 153
  entry points of `tobii_stream_engine.dll` 4.1.0.3, with its signatures,
  `tobii_error_t` numbering and struct layouts, plus the `tobii_recenter`
  extension. Headers: `/usr/local/include/tobii/` (`tobii.h`,
  `tobii_streams.h`, `tobii_config.h`, `tobii_licensing.h`,
  `tobii_advanced.h`, `tobii_wearable.h`, and `tobii_internal.h` for the
  exports Tobii never documented). Link against it and it talks to the daemon
  for you; several processes can use the tracker at once.

  Implemented: gaze point (the device's filtered combined gaze, unclamped —
  bit-identical to what the Windows Stream Engine delivers), gaze origin
  (display frame, mm), eye position and user position guide (track-box
  normalised), gaze data (per eye, tracker frame; no pupil diameter on this
  device), presence (on change), head pose (mm, radians about x/y/z), the IR
  image (280×280, `tobii_image_subscribe`), notifications (display area,
  calibration), device info, track box, display area (get and set — kept
  across re-inits and, like the Stream Engine, across sessions), mounting,
  states, capabilities, and 2-D calibration.
  Timestamps are the device clock. Everything the ET5 was never observed doing
  (wearable, face id, illumination, power, firmware, diagnostics, 3-D and
  per-eye calibration) returns `TOBII_ERROR_NOT_SUPPORTED`.

  #### OpenTrack's `tracker-tobii` plugin

  The Windows plugin builds against `libtobii.so` as-is once OpenTrack's
  `tracker-tobii/CMakeLists.txt` is allowed to configure on Linux:

  ```bash
  cmake -S . -B build -DSDK_TOBII=/usr/local     # wherever `make install` put it
  cmake --build build --target opentrack-tracker-tobii
  ```

  Then pick *Tobii Eye Tracker* as OpenTrack's tracker. This is the alternative
  to the `tobii-opentrack` UDP bridge above; the bridge needs no plugin at all.
- **`tobii-gaze-keys`** — subscribes to **gaze**. While you look at the left/right edge
  of the screen **and hold a Super/Meta key**, it taps the **Left**/**Right**
  arrow key once per second, via `/dev/uinput` (works under both X11 and Wayland).
  Because Super stays physically held, the app sees **Super+Left / Super+Right** —
  e.g. switching tiles/workspaces. Needs `/dev/uinput` + `/dev/input` access (§3).

  ```bash
  tobii-gaze-keys                                   # auto-calibrate, gated by Super
  tobii-gaze-keys --margin 0.3 --interval-ms 800
  tobii-gaze-keys --no-super                        # don't require Super
  tobii-gaze-keys --left 0.2 --right 0.85           # fixed absolute thresholds
  ```

  By default it **auto-calibrates**: it learns the gaze-X range you actually reach
  (look fully left and right once) and fires when gaze is within `--margin` of
  either observed extreme — robust to a gaze stream that isn't centered/symmetric.
  `--left`/`--right` switch to fixed absolute thresholds (`0`=left .. `1`=right);
  `--interval-ms` is the min gap between repeated taps while gaze stays at an edge;
  `--no-super` drops the Super requirement.
- **`tobii5-init-replay track`** — standalone head→OpenTrack without the daemon
  (claims the device directly). Handy for isolating issues; same tracker code.
- Diagnostics: `tobii5-init-replay image83 [--secs 10] [--pose] [--log f.bin] [--no-image]`
  (starts gaze + the 0x50e image stream, reports per-stream rates and gaze
  validity, saves the first frames as PGM, `--pose` runs the head tracker live;
  `--no-image` is the gaze-only baseline), `… image83-replay <log.bin> [--csv out.csv]`
  (runs the tracker over a logged capture and pairs poses with the 0x83 head
  anchors), `… probe` (UVC-camera-vs-0x83 concurrency), `… head83 <log.bin>`
  (research: head pose from 0x83 points).

### 8a. Calibration

The tracker needs calibrating once per user. Until then it runs on the
calibration embedded in the init capture, which is the author's, and on the
author's monitor.

```bash
tobii-calibrate --list-monitors         # which monitor is the tracker on?
tobii-calibrate --monitor 0             # display setup, then 14 points; Esc cancels
```

First the display setup: two white ticks at the bottom edge of the screen,
to be lined up with the two white marks on the tracker's front (drag them,
or Left/Right to move them, Shift for bigger steps; Enter when they line
up). The monitor's size comes from its EDID, so the ticks keep the marks'
distance apart and only tell where the tracker sits under the screen; for a
monitor without a believable EDID (some TVs and projectors) Up/Down spread
them too, and their spacing measures the screen. The ticks start where the
current setting puts them, so usually Enter is all it takes. A calibration
only holds for the display area it is made on, so the setup runs inside the
calibration session, before the points: the daemon writes the display area
to the tracker at once, and saves it (in `~/.config/tobii/display-area`, for
every later start) together with the first calibration computed on it. If
the session ends without a calibration (Esc, a failure, the client dying),
the tracker gets the previous display area back along with the previous
calibration. Outside a calibration session, a display area set through
`tobii_set_display_area` is saved at once. `--no-display-setup`
skips the setup, and so does `--windowed` (it needs the whole monitor).

Then a dot travels through the 7-point pattern twice; look at its centre
until the ring closes and the spinner finishes. The daemon has the tracker
compute the calibration after each batch, and when the session ends normally
saves the last one to `~/.config/tobii/calibration.bin` (the previous one is
kept as `calibration.bin.prev`), with the display area it was made on, and
uploads it at every later start. A session that does not run to its end
(Esc, a failure, the client dying, the daemon or tracker going away) leaves
nothing behind: the calibration and display area it started from stay. The
result screen shows the targets and your live gaze to check it.

- `--rounds 1` for a quick 7-point pass (half the tracker's 14 stored points
  stay from the previous calibration); `--dwell-ms` to linger longer per point.
- `--reset` goes back to the built-in calibration; or delete the file, or set
  `TOBII_CALIBRATION=embedded`. `mv calibration.bin.prev calibration.bin` (and
  a daemon restart) restores the previous one.
- `--dry-run --windowed` shows the screens without a tracker (`--windowed`
  only goes with `--dry-run`: a calibration needs the whole monitor).
- Only one client can calibrate at a time; a second one is told the tracker is
  busy. If the calibrating client dies, the daemon stops the session and puts
  the previous calibration back.
- Stream Engine applications can calibrate too, through `tobii_calibration_*`
  in `libtobii.so`; the result is saved the same way.

If the compositor opens the window on another monitor, or not fullscreen
(PaperWM puts new windows on the monitor in use), `tobii-calibrate` asks for
the right one again; if the window stays off it, it says so on screen and
will not take the display setup until the window is fullscreen on the right
monitor (with PaperWM, Super+Shift+Ctrl+Left/Right moves it); the points
wait for it too.

### Recenter (recalibrate the head rest pose)

Sit in your neutral pose and trigger a recenter — the daemon recalibrates
without restarting the device/stream. Three ways:

```bash
tobii-opentrack --recenter                 # one-shot client (bind to a hotkey)
systemctl --user kill -s SIGUSR1 tobiid    # or signal the service
kill -USR1 $(pgrep -x tobiid)              # or signal the process
```

Via `libtobii.so`: call `tobii_recenter(device)`. (Affects the head tracker;
gaze has no rest pose.)

---

## 9. Notes & troubleshooting

- **Head pose + gaze together.** One engine serves head pose (from the device's
  0x50e IR image stream), gaze and presence concurrently; verified at 33 Hz each
  with gaze validity unchanged. Never stream the UVC camera while the daemon
  runs (the `camera`/`track`/`probe` research commands, or any app opening it
  through uvcvideo): that throttles the 0x83 streams to <1 Hz.
- **Head pose arrives ~1 s after the first frame.** The tracker averages the
  first 30 frames as the rest pose (recenter to redo it); a cold device may
  additionally take one re-open (~10 s) before any stream arms.
- **Lazy claim.** An idle daemon holds no device; it opens the tracker only on
  the first subscription and releases it when the last client disconnects.
- **"It flies around."** You're talking to an **old daemon** (pre-rebuild) — it
  still has the previous code/units. Restart it (§5).
- **"Rotations slide."** Tune `TOBII_PIVOT_DOWN` / `TOBII_PIVOT_BACK` (§7).
- **Tracking after logout.** User services stop at logout unless you enable
  lingering: `loginctl enable-linger $USER`.
- **Permission denied on the device.** Re-check §3 (udev rule + re-plug).
- **Checking the daemon end to end.** `target/release/tobii5-init-replay
  ipc-probe` asks the running daemon for everything (device info, track box,
  mounting, display area, calibration id, clock) and reports the rate of every
  stream; `--set-display 597,336` also writes a display area.
