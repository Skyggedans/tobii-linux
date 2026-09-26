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
| `tobii-ipc` | the daemon protocol, the display geometry and the host clock | none (`libc` only) |
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
systemctl --user restart tobiid        # 4b and 4c
# or: pkill -f release/tobiid          # 4a (next client respawns it)
```

Under socket activation (4c) restart the service, not `tobiid.socket`:
restarting the socket does not restart a running daemon, so the old build
stays up, and clients that connect after it are not served until it exits.
`tobiid` removes at shutdown only a socket file it bound itself, never
systemd's, so clients reach the restarted daemon through the same socket.

**Once, under 4c, when the running `tobiid` was built without the commit
"daemon: leave the socket file to systemd under socket activation"** (any
build before it, whatever its date; `strings "$(command -v tobiid)" | grep -q
'did not bind'` fails for such a build): that daemon removes systemd's
`$XDG_RUNTIME_DIR/tobiid.sock` at shutdown, and
a restart runs its shutdown, so the first restart onto a newer build still
loses the file. A client that then finds no socket spawns a `tobiid` of its own
(as in 4a), which binds the path and may claim the tracker. For that one
upgrade, and whenever the file is missing, do this instead of the restart:

```bash
systemctl --user stop tobiid.service tobiid.socket
pkill -x tobiid                        # a daemon a client spawned meanwhile
while pgrep -x tobiid >/dev/null; do sleep 0.2; done
systemctl --user start tobiid.socket
systemctl --user start tobii-gaze-keys.service   # if enabled: stopping tobiid.service stopped it
```

`test -S "$XDG_RUNTIME_DIR/tobiid.sock"` should then succeed, and
`pgrep -a tobiid` list at most one daemon (none until a client connects).
Check the file this way rather than with a client: `ipc-probe` and the other
clients spawn a daemon of their own when they find no socket, which hides a
missing file.

**Once, when moving to host-clock timestamps.** A `tobiid` built without the
commit "daemon: send sample timestamps on the host clock" (any build before
it, whatever its date) sends sample timestamps on the tracker's clock, which
a `libtobii.so` with that commit hands to its callbacks as
`tobii_system_clock` times; an application still running an older
`libtobii.so` gets the new daemon's host times where it expects the
tracker's. Nothing detects the mismatch: after installing the first build
with that commit, restart the daemon, and restart the applications that read
timestamps.

**Once, when moving to a `libtobii.so` that reports a lost connection.** An
application that loaded `libtobii.so` before `make install` keeps the old
library in memory. A library built without the commit "ffi: report a lost
daemon connection" never notices the daemon going away, so after the restart
that application gets no samples and cannot recover by reconnecting. Restart
every `libtobii.so` application after the first daemon restart onto such a
build; in OpenTrack, stop and start tracking.

**Running clients.** A restart, like a crash, closes every client's
connection:

- `paperwm-gaze` connects again by itself, once a second.
- `tobii-opentrack`, and `tobii-gaze-keys` run by hand, exit with an error
  about the lost connection; start them again. The `tobii-gaze-keys`
  service (`make enable-keys`) needs nothing: systemd restarts it along with
  `tobiid`, and 2 s after it exits on a crash.
- `tobii-calibrate` fails a calibration under way and shows the error in its
  window until Esc; run it again. Once a calibration has finished (and is
  saved), its verification view only stops showing gaze.
- A `libtobii.so` application is told by `tobii_device_process_callbacks`
  (`TOBII_ERROR_CONNECTION_FAILED`, §8); libtobii does not reconnect by
  itself. The application recovers when a `tobii_device_reconnect` succeeds
  (it fails until a daemon is back, so retry it), or when it destroys the
  device and creates it again, which, unlike a reconnect, spawns a daemon if
  none listens. A reconnect connects to a running daemon and never spawns
  one, so under 4b and 4c it succeeds once the restarted service is up, and
  under 4a only once some client has spawned a new daemon. It restores the
  application's subscriptions; a calibration session or pause the old
  connection held has ended. An application that does neither gets no
  samples again. OpenTrack's `tracker-tobii` plugin does neither by itself:
  it recovers when tracking is stopped and started, which creates the device
  again (§8).

After a crash under 4c, systemd waits `RestartSec=2` before it starts the
daemon again. A reconnect with subscriptions meanwhile connects into the
socket's backlog, gets no answer within its ~500 ms and gives up. The
restarted daemon still serves each such abandoned connection (it starts the
tracker's engine for it, then drops it again unless another client wants
it), which can make the first attempts after it is up miss their ~500 ms
too. Expect a few seconds and some retries before samples resume. A
reconnect with no subscriptions has nothing to wait for and succeeds as soon
as it connects; the application's first requests then wait for the daemon to
start, and can time out.

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

`libtobii.so` prints nothing, whatever `RUST_LOG` says: inside an application
its diagnostics go only to the `tobii_custom_log_t` the application hands
`tobii_api_create` (§8).

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
| `TOBII_NO_RESET` | unset | `1` skips the USB reset the engine tries once opens keep failing, after two failures in a row: opens whose init fails, that do not arm the stream (bar the first after a start or a lost stream: the tracker needs that one to arm), or that lose it within 30 s (until the engine waited out a tracker starting its sensor, every open right after the reset failed its init on a 2 s write timeout; whether the reset helps is unconfirmed on hardware) |
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
  normalised), gaze data (per eye, tracker frame, with the pupil diameter),
  presence (on change), head pose (mm, radians about x/y/z), the IR
  image (280×280, `tobii_image_subscribe`), notifications (display area,
  calibration, pause), device info, track box, display area (get and set —
  kept across re-inits and, like the Stream Engine, across sessions),
  mounting, states, capabilities, 2-D calibration (discarding a point too),
  a device/host clock pair (`tobii_timesync`), the tracker's stream
  catalogue (`tobii_enumerate_stream_types`), pause and resume, and the
  device name. `tobii_hardware_configuration_get` is provisional: the ET5
  has reported no hardware configuration on Linux, so it returns
  `TOBII_ERROR_NOT_SUPPORTED`.
  Timestamps are the host clock `tobii_system_clock` reads (`CLOCK_MONOTONIC`),
  onto which the daemon maps the tracker's with an offset it estimates from
  the arrivals, afresh at every tracker init (README.md, Architecture,
  "Timestamps"); gaze data and `tobii_timesync` keep the tracker's time too.
  Everything the ET5 was never observed doing (wearable, face id,
  illumination, power, firmware, diagnostics, 3-D and per-eye calibration,
  calibration stimulus points) returns `TOBII_ERROR_NOT_SUPPORTED`; for the
  stimulus points that is the Stream Engine's own answer without Tobii's
  service (its in-process legacy TTP module), and its answer behind the
  service was never captured. No licence is checked: what the Stream Engine
  reserves for its professional, config or internal feature groups, or for
  an additional-features licence (the IR image), works too.

  A name set with `tobii_set_device_name` is kept by the daemon, not the
  tracker, in `~/.config/tobii/device-name` (`$XDG_CONFIG_HOME/tobii`): the
  name on one line, at most 63 bytes. The daemon reads it at start, so
  `echo Desk > ~/.config/tobii/device-name` and a daemon restart (§5) name
  the device too; delete the file (and restart) to get the model back.

  A pause (`tobii_pause_device`) holds for every client until any client
  resumes, the client that paused last disconnects, or the tracker
  re-initialises; a calibration cannot start while the tracker is paused.

  A lost daemon connection (a restart or crash, §5) is reported, not
  repaired: `tobii_device_process_callbacks` delivers what had arrived and
  then returns `TOBII_ERROR_CONNECTION_FAILED` on every call until a
  `tobii_device_reconnect` succeeds (retry it: it fails while the daemon is
  not back), and `tobii_wait_for_callbacks` wakes once for it. The same
  error from a request (a clock pair, a pause, a calibration, a display-area
  write) can instead mean the daemon has no tracker, over a connection that
  is fine; a reconnect then succeeds without bringing the tracker back. An
  unplugged tracker never shows in `tobii_device_process_callbacks`: its
  samples stop and resume on the same connection once it is back (§9, *Lazy
  claim*).

  Logging: `libtobii.so` prints nothing; an application sees its diagnostics
  only through the `tobii_custom_log_t` it hands `tobii_api_create`. They are
  libtobii's own, not a line per failing call as the Stream Engine writes:
  `TOBII_LOG_LEVEL_ERROR` for a refused `field_of_use`, a failed connect or
  reconnect, a lost daemon connection when `tobii_device_process_callbacks`
  reports it (once per loss) or a daemon reply that does not decode, and
  `TOBII_LOG_LEVEL_INFO` for each connect and reconnect.
  The logger is called on the thread inside the `tobii_*` call that logs,
  with no lock of libtobii's held; a call from inside it, on that thread,
  that a callback could not make either returns
  `TOBII_ERROR_CALLBACK_IN_PROGRESS`. A `tobii_custom_alloc_t` is checked as
  in the Stream Engine and never called. Unlike the Stream Engine, libtobii
  does not serialise calls on a device: one device must not be used from
  two threads at once (README.md, Architecture, "Threads").

  #### OpenTrack's `tracker-tobii` plugin

  The Windows plugin builds against `libtobii.so` as-is once OpenTrack's
  `tracker-tobii/CMakeLists.txt` is allowed to configure on Linux:

  ```bash
  cmake -S . -B build -DSDK_TOBII=/usr/local     # wherever `make install` put it
  cmake --build build --target opentrack-tracker-tobii
  ```

  Then pick *Tobii Eye Tracker* as OpenTrack's tracker. This is the alternative
  to the `tobii-opentrack` UDP bridge above; the bridge needs no plugin at all.

  The plugin needs a reconnect call to survive a daemon restart or crash
  (§5). As it stands it never calls `tobii_device_reconnect`, and on any
  error from `tobii_device_process_callbacks` it leaves the pose unset,
  which OpenTrack takes as all zeros, the head at the tracker's origin. So
  from the restart on, the view jumps away (with OpenTrack's centering on,
  by the pose it was centered at) and stays there until you stop and start
  tracking in OpenTrack, which creates the device again. A plugin that, on
  `TOBII_ERROR_CONNECTION_FAILED`, calls `tobii_device_reconnect` at most
  once a second and keeps reporting the last pose meanwhile picks up again
  by itself once the daemon is back; that change to the plugin is a separate
  patch, the commit "tracker/tobii: reconnect when the connection is lost"
  on the `tracker-tobii-linux` branch of the OpenTrack fork, which also
  carries the Linux build change above. An unplugged tracker needs none of
  this: the pose holds, then resumes once the tracker is back.
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
every later start) with the calibration made on it when the session is
kept. If the session ends without a calibration (Esc, a failure, the client
dying, the daemon or tracker going away, or the tracker re-initialising),
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
(Esc, a failure, the client dying, the daemon or tracker going away, or the
tracker re-initialising) leaves nothing behind: the calibration and display
area it started from stay. A stop that has saved the calibration keeps it
even if the tracker then refuses it or goes away before taking it: the stop
reports the failure, but the tracker loads the saved calibration and display
area at its next init. The result screen shows the targets and your live
gaze to check it.

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
  in `libtobii.so`; the result is saved the same way, once the application
  calls `tobii_calibration_stop`. A session that ends before that (the
  application exiting, the daemon or tracker going away, or the tracker
  re-initialising) leaves nothing behind, as above. When the tracker went
  away or re-initialised, the application's later calls in it,
  `tobii_calibration_stop` included, are
  `TOBII_ERROR_CALIBRATION_NOT_STARTED`.

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
- **A stalled tracker is re-opened in place.** No gaze for 5 s (10 s after a
  resume) outside a pause, or a USB error, makes the engine re-open and
  re-init the tracker. Only failures in a row add up: an open whose init
  fails, one that does not arm the stream, or one that loses it within 30 s.
  The first open that does not arm after a start or a lost stream is not a
  failure: the tracker arms only on the open after it (a firmware quirk).
  After two failures in a row the tracker is USB-reset, and after five the
  engine stops; the daemon starts a new one if a client still needs the
  tracker (or pre-warm is on) and it is plugged in.
- **A tracker starting its sensor is waited out.** For about 3.6 s after it
  starts its sensor (on a cold start, or in the open after a USB reset) the
  tracker takes no commands, which lands inside the init's calibration
  upload. The engine retries each 4 KB piece of a write the tracker refuses
  for up to 6 s (per piece), reading what it sends meanwhile, before it
  calls the init or command failed.
- **Lazy claim.** An idle daemon holds no device; it opens the tracker on the
  first subscription or the first request that needs it (device info, a
  clock pair, a pause, …) and releases it when the last such client
  disconnects. Facts from the last init (device info, track box, the stream
  catalogue, …) are answered even while the tracker is unplugged. Requests
  that need it live (a clock pair, a pause, starting, retrieving or applying
  a calibration, a display-area write) then fail at once with
  `TOBII_ERROR_CONNECTION_FAILED`, from when the daemon has given the tracker
  up (a few seconds after the unplug); the daemon logs once that no tracker
  is on the bus and opens it once it is plugged back in, for a client still
  connected. A calibration session under way ends then, saving nothing,
  unless its client is already stopping it, and the client's later calls in
  it are `TOBII_ERROR_CALIBRATION_NOT_STARTED` (§8a).
- **"It flies around."** You're talking to an **old daemon** (pre-rebuild) — it
  still has the previous code/units. Restart it (§5).
- **"Rotations slide."** Tune `TOBII_PIVOT_DOWN` / `TOBII_PIVOT_BACK` (§7).
- **Tracking after logout.** User services stop at logout unless you enable
  lingering: `loginctl enable-linger $USER`.
- **Permission denied on the device.** Re-check §3 (udev rule + re-plug).
- **Checking the daemon end to end.** `target/release/tobii5-init-replay
  ipc-probe` asks the running daemon for everything (device info, track box,
  mounting, display area, stream types, hardware configuration, device name,
  calibration id, calibrating and paused states, fault and warning lists,
  clock) and reports the rate of every stream; `--set-display 597,336` also
  writes a display area.
