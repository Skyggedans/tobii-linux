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
only in the standalone research subcommands (`camera`, `probe`).

---

## 1. Prerequisites

- Rust toolchain (stable) + Cargo.
- The repo includes the models (`face_landmarks.onnx` and
  `blaze_face_short_range.onnx` in `crates/tobii-pose/models/`) and the init
  capture (`crates/tobii-usb/init_packets_ep.txt`); all are embedded at
  **build** time, so no runtime data files are needed.
- A Tobii Eye Tracker 5 plugged in.

## 2. Build

```bash
cargo build --release --workspace     # or: make build
```

Produces in `target/release/`:

| Artifact | What it is | Installed? |
|---|---|---|
| `tobiid` | the daemon (claims the device, serves clients) | yes |
| `tobii-opentrack` | thin client: the Stream Engine's head pose → OpenTrack UDP | yes |
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
| `tobii-pose` | face landmarks, face detection and the Stream Engine's head pose; owns `models/` | `ort` |
| `tobii-usb` | USB transport and the live 0x83 engine; owns `init_packets_ep.txt` | `rusb` |
| `tobii-ipc` | the daemon protocol and its deadlines, the display geometry and the host clock | none (`libc` only) |
| `tobii-calib` | the calibration blob format and the per-user store | none (std only) |
| `tobii-log` | shared `tracing` subscriber setup | — |
| `tobiid` | the daemon binary | — |
| `tobii-ffi` | `libtobii.so` (cdylib) | — |
| `tobii-clients` | `tobii-opentrack`, `tobii-gaze-keys` | — |
| `tobii-calibrate` | the calibration window | `winit`, `softbuffer` |
| `tobii-tools` | `tobii5-init-replay`: analysis, UVC camera, diagnostics | all of the above |

`make check` runs what CI would: `cargo fmt --all --check`, clippy with
`-D warnings` over all targets, the tests and `cargo doc`. `make verify-abi`
asserts `libtobii.so` exports exactly the 153 symbols listed in
`crates/tobii-ffi/abi-symbols.txt` (the exports of the reference DLL, and
none of its own), then compiles and runs `crates/tobii-ffi/abi-smoke.c`
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

**Once, when moving to the Stream Engine's head pose.** A `tobiid` built
with the commit "tobiid, engine: stop publishing the legacy head pose" makes
one head pose, the Stream Engine's, which `libtobii.so` delivers from the
commit "ffi: deliver the Stream Engine's head pose, validity included" on,
and `tobii-opentrack` sends from "clients: send OpenTrack the Stream
Engine's head pose" on. Before those commits both read the daemon's own
head pose, relative to a rest pose, which the new daemon no longer makes.
With an older build on either side of the socket, the client gets no head
pose, and no error:

- A `tobiid` built without the commit "engine, tobiid: publish a head pose
  for every image" (any build before it, whatever its date) acknowledges
  the new pose's subscription and never sends it: `libtobii.so` logs one
  WARN line after 20 s (§8; OpenTrack gives libtobii no logger), and
  `tobii-opentrack` prints one on its stderr.
- A newer `tobiid` acknowledges the old pose's subscription, from an
  application still running an older `libtobii.so`, from an older
  `tobii-opentrack`, or from a client with its own copy of the protocol
  that asks for stream bit 0, such as the `tobii-hub` Flutter plugin for
  its `head` stream, and never sends it; its log says so, once. Such a
  client gets no head pose until it is ported to the Stream Engine's,
  `HEAD_POSE` (bit 9 of the `u32` SUBSCRIBE mask, frames tagged `0x29`;
  `crates/tobii-ipc/src/lib.rs`), which it centres itself: the daemon
  ignores the RECENTER frame such a client may send (`tobii-hub`'s
  `recenter()`), logging the first.

`make install` replaces the binaries, not the daemon that is running nor
the library an application has loaded: after installing the first build
with those commits, restart the daemon with `systemctl --user restart
tobiid.service` (4b and 4c; under 4c `pkill -x tobiid` does too, as the
socket starts the new daemon at the next connection; under 4a
`pkill -x tobiid`, and the next client spawns it), then `tobii-opentrack`
and the `libtobii.so` applications that were running. OpenTrack loads its
tracker plugins, and `libtobii.so` with them, when it starts: quit it and
start it again, as stopping and starting tracking keeps the library it has.

In OpenTrack the pose then behaves much as the Stream Engine's does on
Windows (§8 has every axis), and not as the old one did; the plugin with a
`libtobii.so` that already delivered the Stream Engine's pose sees no
change:

- **Centring.** Nothing in the daemon centres the pose any more: OpenTrack
  does, at the first pose and on its *Center* shortcut, so bind a recenter
  hotkey to that shortcut. The daemon ignores SIGUSR1 (`systemctl --user
  kill -s SIGUSR1 tobiid`), logging the first, `tobii-opentrack --recenter`
  is refused with an error that says so, and `libtobii.so` no longer
  exports `tobii_recenter`, its one entry point beyond the DLL's: a program
  that calls it must be rebuilt without the call.
- **Signs.** With the `tracker-tobii` plugin, TZ, Pitch and Roll now move
  as on Windows, the other way from before: undo any inversion you set on
  those axes in OpenTrack's mapping. With `tobii-opentrack`, TX, TZ, Yaw,
  Pitch and Roll move the other way and only TY as before: to keep a
  profile made for the old bridge moving each axis the way it did, toggle
  *Pre-invert* on those five axes (OpenTrack's *Options*, *Output* tab). A
  profile made for the plugin on the Stream Engine's pose, here or on
  Windows, fits both unchanged.
- **Range.** No axis stops at ±45° any more, and the position is a point
  between the eyes, no longer a pivot at the neck, so turning the head
  moves it too. How much further each axis moves than it did depends on
  the `tobiid` you come from (§8 has each axis against the Stream
  Engine's):
  - One built without the commit "pose: rotate the face crop by the eye
    line, size it from the landmarks" (any build before it, whatever its
    date): every axis but pitch moves further, pitch about as far. TZ,
    which hardly followed your distance from the screen, moves 2 to 8
    times as far, yaw 1.3 to 2.8 times, roll 1.2 to 1.4 times, and TX and
    TY 1.1 to 1.4 times: a gain you raised to make up for a small TZ, yaw
    or roll may now be too much.
  - One built with it: TX moves 1.5 to 1.9 times as far, TY 1.4 to 1.8
    times, TZ 1.1 to 1.3 times and pitch 1.0 to 1.15 times, yaw and roll
    about as far: TX and TY may want a third to a half less gain.
- **Invalid poses.** Near the edges of the camera's view, and when the face
  is lost, the pose now comes marked invalid, where nothing came before:
  `tobii-opentrack` sends nothing then, as before, but the plugin as it
  stands jumps the view (§8, *Invalid poses*).
- **Variables.** The daemon no longer reads the variables that shaped its
  own pose (the neck pivot, the camera tilt, the eye-line roll and a debug
  log of that pose): a unit override may drop them.

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
socket's backlog, gets no answer within its ~500 ms and gives up, closing
the connection. The restarted daemon still reads each such abandoned
connection, but skips a subscription whose client has hung up by the time it
would act on it (the journal says "skipping a subscription from a client
that has hung up", once per connection): it starts no engine for it, so a
later attempt's answer does not wait while one starts and stops again. The
same holds for an attempt that gives up while a running daemon is busy.
Expect a few seconds and some retries before samples resume: attempts fail
until the daemon is up, and the tracker then has to start. A reconnect with
no subscriptions has nothing to wait for and succeeds as soon as it
connects; the application's first requests then wait for the daemon to
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

Set these on **`tobiid`** (the head-pose tracker runs in the daemon, not in
the client). The head pose, which `libtobii.so` delivers (to OpenTrack's
`tracker-tobii` plugin, among others) and `tobii-opentrack` sends (§8), has
no settings of its own: its constants are fitted to the Stream Engine's
output, and its frame is the display area's (`TOBII_DISPLAY_MM` below,
§8a; README.md, Architecture, "Head pose").

| Var | Default | Meaning |
|---|---|---|
| `TOBII_PREWARM` | unset | `1` (or the historical `head` / `gaze`): init the device at daemon start and keep it warm, so client connects are instant (IR illuminator stays on while the service runs). |
| `TOBII_NO_RESET` | unset | `1` skips the USB reset the engine tries once opens keep failing, after two failures in a row: opens whose init fails, that do not arm the stream (bar the first after a start or a lost stream: the tracker needs that one to arm), or that lose it within 30 s (until the engine waited out a tracker starting its sensor, every open right after the reset failed its init on a 2 s write timeout; whether the reset helps is unconfirmed on hardware) |
| `TOBII_NO_IMAGE` | unset | `1` does not start the 0x50e image stream (gaze/presence only; no head pose from the gaze engine) |
| `TOBII_IMAGE83_DEBUG` | unset | `1` logs the image head-pose worker's statistics every 5 s while a client wants a head pose (an `info` event): the images it made a pose of and the valid poses, per second (`frames_per_s`/`valid_per_s`), the images dropped before it could take them (`overwritten`, 0 when it keeps up), the time per image (`mean_ms`/`p99_ms`/`max_ms`) and the face-detector rate (`detector_runs_per_s`: how often the tracker looked for a lost face) |
| `RUST_LOG` | `info` | log filter for all binaries (`debug`, `tobii=debug,ort=warn`, …); see §6 |
| `TOBII_DISPLAY_MM` | unset | your monitor as `<width>x<height>[+<offset_x>]` in mm, e.g. `597x336`: once the tracker reports its mounting, the daemon computes the display area (the Stream Engine's `tobii_calculate_display_area_basic`) and writes it, replacing the capture author's monitor that the init replay configures. Gaze coordinates, and the Stream Engine's head pose, are relative to this area. `offset_x` is how far right of the tracker the screen centre is. Usually unneeded: `tobii-calibrate` sets the display area and the daemon keeps it (§8a); a saved display area wins over this variable, which only fills in while nothing is saved (delete `~/.config/tobii/display-area` to use it). |
| `TOBII_CALIBRATION` | unset | where the calibration is kept (default `$XDG_CONFIG_HOME/tobii/calibration.bin`), or `embedded` to use the built-in one (§8a) |

Head-pose inference on the image stream runs only while some client subscribes
to the head pose (about 6 ms per frame at 33 Hz on a desktop CPU); gaze-only
clients don't pay for it.

For a systemd unit:

```bash
systemctl --user edit tobiid
# add:
#   [Service]
#   Environment=TOBII_IMAGE83_DEBUG=1
systemctl --user restart tobiid
```

For option 4a, env on the client propagates to the daemon **only if the client
spawns it** (so `pkill -f release/tobiid` first):

```bash
pkill -f release/tobiid
TOBII_IMAGE83_DEBUG=1 ./target/release/tobii-opentrack
```

---

## 8. Clients

- **`tobii-opentrack`** — subscribes to the Stream Engine's head pose, the
  one `libtobii.so` delivers (below), and sends the OpenTrack UDP packet
  (`x,y,z,yaw,pitch,roll`; translation in **cm**, angles in **degrees**). In
  OpenTrack pick *Input = UDP over network*, port 4242. Each axis goes out
  as OpenTrack's `tracker-tobii` plugin hands it over (TX is -x, TY y and TZ
  z of the position in the display frame; yaw is minus the angle about y,
  pitch the angle about x and roll the angle about z), so the bridge and the
  plugin move alike while the pose is valid, and one OpenTrack profile fits
  both. A pose the daemon marks invalid (no face, or one at the edge of the
  camera's view) is not sent, and OpenTrack holds the last one, where the
  plugin as it stands jumps the view (below, *Invalid poses*). The pose is
  absolute: OpenTrack centres it, at the first pose (*Center at startup*,
  on by default) and on its *Center* shortcut. What the plugin section
  below says of the pose (translation with rotation, range) holds for the
  bridge too. From a `tobiid` too old to send the Stream Engine's head pose
  (§5) it gets nothing, and says so once, after 20 s.
- **`libtobii.so`** — the Stream Engine 4.1 C API: every one of the 153
  entry points of `tobii_stream_engine.dll` 4.1.0.3, with its signatures,
  `tobii_error_t` numbering and struct layouts, and no others. Headers:
  `/usr/local/include/tobii/` (`tobii.h`, `tobii_streams.h`,
  `tobii_config.h`, `tobii_licensing.h`, `tobii_advanced.h`,
  `tobii_wearable.h`, and `tobii_internal.h` for the exports Tobii never
  documented). Link against it and it talks to the daemon for you; several
  processes can use the tracker at once.

  Implemented: gaze point (the device's filtered combined gaze, unclamped —
  bit-identical to what the Windows Stream Engine delivers), gaze origin
  (display frame, mm), eye position and user position guide (track-box
  normalised), gaze data (per eye, tracker frame, with the pupil diameter),
  raw gaze (the Stream Engine's own record of each gaze frame, every value
  as the tracker sent it, `tobii_gaze_raw_subscribe`), presence (on change),
  head pose (the Stream Engine's: absolute, in the display frame, one for
  every IR image, valid or not; README.md, Architecture, "Head pose"), the
  IR image (280×280, `tobii_image_subscribe`), notifications (display area,
  calibration, pause, faults and warnings), device info, track box, display
  area (get and set — kept across re-inits and, like the Stream Engine,
  across sessions), mounting, states, capabilities, 2-D calibration
  (discarding a point too), a device/host clock pair (`tobii_timesync`), the
  tracker's stream catalogue (`tobii_enumerate_stream_types`), pause and
  resume, and the device name. `tobii_hardware_configuration_get` is
  provisional: the ET5 has reported no hardware configuration on Linux, so
  it returns `TOBII_ERROR_NOT_SUPPORTED`.
  Timestamps are the host clock `tobii_system_clock` reads (`CLOCK_MONOTONIC`),
  onto which the daemon maps the tracker's with an offset it estimates from
  the arrivals, afresh at every tracker init (README.md, Architecture,
  "Timestamps"); gaze data and `tobii_timesync` keep the tracker's time too,
  and raw gaze has only the tracker's time.
  Everything the ET5 was never observed doing (wearable, face id,
  illumination, power, firmware, diagnostics, 3-D and per-eye calibration,
  calibration stimulus points, the internal low-frequency head, multiple
  faces, wearable limited image and secondary camera image streams) returns
  `TOBII_ERROR_NOT_SUPPORTED`; for the stimulus points and those five
  streams that is the Stream Engine's own answer without Tobii's service,
  for a tracker it drives itself through its in-process legacy TTP module,
  and its answer behind the service was never captured. No licence is
  checked: what the Stream Engine reserves for its professional, config or
  internal feature groups, or for an additional-features licence (the IR
  image), works too. Raw gaze is one such: the Stream Engine serves it only
  to its internal feature group, and only without Tobii's service (for any
  URL, libtobii's `tobii-ffi://` included, but `tobii-prp://` and
  `tprp-tcp://`). A `tobiid` built before raw gaze takes the subscription
  and never sends the stream, and one built before the Stream Engine's head
  pose does the same with the head pose: restart the daemon after
  installing (§5).

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
  reports it (once per loss), a daemon reply that does not decode, or a
  `tobii_calibration_stop` that failed after the daemon saved the
  calibration (saved but perhaps not applied: the tracker loads it at its
  next init), `TOBII_LOG_LEVEL_WARN`, once per device, for a head pose the
  daemon has not sent in the 20 s since it was subscribed (a `tobiid` older
  than the library sends none: restart it after installing), and
  `TOBII_LOG_LEVEL_INFO` for each connect and reconnect.
  The logger is called on the thread inside the `tobii_*` call that logs,
  with none of that call's locks held, and from several threads at once if
  they log at once, so lines from different threads may interleave; a call
  from inside it, on that thread, that a callback could not make either
  returns `TOBII_ERROR_CALLBACK_IN_PROGRESS`. A `tobii_custom_alloc_t` is
  checked as in the Stream Engine and never called.

  Threads may share a device, as the Stream Engine promises: calls on it from
  several threads at once are safe. Its requests, subscription changes and
  reconnects run one at a time, in the order they are called (a slow one
  delays the others: a calibration start may take ~3 min at worst, a pause
  a minute), its callbacks run one at a time on whichever thread processes
  it, and a process call that finds another thread at it (processing it, a
  wait or a clear at its queue for a moment, or a reconnect waiting for
  tobiid to take the subscriptions back, then closing the old connection)
  returns at once, delivering nothing.
  `tobii_device_destroy` and `tobii_api_destroy` must not overlap any other
  call on the handle, and nothing may use it afterwards: join the thread
  that processes a device before destroying it. Threads that create devices
  at once while no daemon runs spawn one `tobiid` between them (processes
  that do so can still spawn one each). A callback, or the logger,
  must not block on another thread's call into any device (nor on a thread
  that waits for one), which can deadlock, as in the Stream Engine.
  `tobii_calibration_retrieve`'s receiver is refused on its thread the calls
  a callback is, but holds no lock, so it may wait for other threads' calls.
  README.md (Architecture, "Threads") has the rest, and where it differs
  from Windows.

  #### OpenTrack's `tracker-tobii` plugin

  The Windows plugin builds against `libtobii.so` as-is once OpenTrack's
  `tracker-tobii/CMakeLists.txt` is allowed to configure on Linux:

  ```bash
  cmake -S . -B build -DSDK_TOBII=/usr/local     # wherever `make install` put it
  cmake --build build --target opentrack-tracker-tobii
  ```

  Then pick *Tobii Eye Tracker* as OpenTrack's tracker. This is the alternative
  to the `tobii-opentrack` UDP bridge above; the bridge needs no plugin at all.

  Through `libtobii.so` the plugin gets the Stream Engine's head pose, the
  pose it was written for, as the bridge does (above); §5 says what changed
  from earlier builds. Compared with the DLL on Windows:

  - **Centring.** The pose is absolute, and OpenTrack centres it itself, as
    on Windows: at the first valid pose (*Center at startup*, on by
    default) and on its *Center* shortcut.
  - **Translation with rotation.** The position is a point between the
    eyes, so turning the head moves TX, TY and TZ too, some 5 cm sideways
    for a 30° turn, as with the DLL. OpenTrack's *Relative translation*
    options, with *Neck displacement*, can make up for part of that
    (untried here).
  - **Range.** Most axes move as far as the Stream Engine's. As multiples
    of the Stream Engine's pose of the same motion (the slope of a fit on
    each of the three Windows sessions):

    | Axis | Slope |
    |---|---|
    | TX | 0.98 to 1.00 |
    | TY | 0.84 to 0.96 |
    | TZ | 0.98 to 1.02 |
    | Yaw | 1.01 to 1.15 |
    | Pitch | 0.79 to 0.99 |
    | Roll | 0.97 to 1.00 |

    Yaw and pitch are where it differs from Windows: in two of the sessions
    yaw turns about 1.2 times as far as the Stream Engine's at 10 to 30° and
    pitch 0.8 to 0.9 times as far, in the third both within 10 %. So a
    profile whose curves you tuned on Windows may turn the view some 20 %
    further in yaw and 10 to 20 % less in pitch here.
  - **Invalid poses.** Near the edges of the camera's view, and when the face
    is lost, the pose comes marked invalid, as the DLL's does. The plugin as
    it stands then sets no axis, which OpenTrack takes as zeros: until the
    face is back, the view jumps to minus the pose OpenTrack centred on (in
    TZ by about your distance from the screen), and a centring meanwhile
    takes the zeros as the centre. It does the same with the DLL on Windows.
    The commit "tracker/tobii: hold each axis at its last valid value" on
    the `tracker-tobii-linux` branch of the OpenTrack fork has the plugin
    hold each axis's last valid value instead.

  The plugin needs a reconnect call to survive a daemon restart or crash
  (§5). As it stands it never calls `tobii_device_reconnect`, and on any
  error from `tobii_device_process_callbacks` it leaves the pose unset,
  which OpenTrack takes as all zeros, the head at the display's centre. So
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
- Diagnostics: `tobii5-init-replay image83 [--secs 10] [--pose] [--log f.bin] [--no-image]`
  (starts gaze + the 0x50e image stream, reports per-stream rates and gaze
  validity, saves the first frames as PGM; `--pose` makes the Stream Engine's
  head pose of every image as the daemon does, but without it, which is handy
  for isolating issues, in the display area the init capture writes: the
  author's monitor for the shipped `init_packets_ep.txt`, not yours;
  `--no-image` is the gaze-only baseline),
  `… image83-replay <log.bin> [--csv out.csv] [--fits fits.csv]
  [--landmarks landmarks.f32]` (runs the tracker over a logged capture;
  `--csv` gives each image's face and the head anchors of the last 0x83 gaze
  frame, `--fits` every image's face fit at full precision and `--landmarks`
  its 468 landmarks as f32, layouts in `--help`), `… probe`
  (UVC-camera-vs-0x83 concurrency), `… head83 <log.bin>` (research: head pose
  from 0x83 points).

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
uploads it at every later start. The result screen then shows the targets
and your live gaze to check it. A session that does not run to its end
(Esc, a failure, the client dying, the daemon or tracker going away, or the
tracker re-initialising) leaves nothing behind: the calibration and display
area it started from stay. A stop that has saved the calibration keeps it
even if the tracker then refuses it, goes away before taking it or does not
answer in time: the stop reports the failure, but the tracker loads the
saved calibration and display area at its next init. `tobii-calibrate` then
says the calibration was saved and the tracker takes it at its next start
(it may not have taken it at once), on its failure screen rather than the
result screen, still writes `--export`, and exits with status 2 (1 for any
other failure; with a daemon from before this was told, such a stop reads
as not kept).

Each step waits for the tracker as long as the daemon may take over it, so
that `tobii-calibrate` never gives up on a step the daemon still carries
out. That is usually a second or two, but a tracker starting up or
re-opened after a stall can hold a step for a minute or more (a start up to
about 3). The start, the display-area write and the save or discard then
say they are waiting for the tracker; a point or a compute keeps its
spinner turning. Esc closes the window at once; `tobii-calibrate` then
waits up to 25 s for the step under way (saying so in the terminal when it
takes more than a second), and if it has not finished by then, says
whether the daemon still saves the calibration (the save was under way;
`--export` then writes nothing) or discards it.

- `--rounds 1` for a quick 7-point pass (half the tracker's 14 stored points
  stay from the previous calibration); `--dwell-ms` to linger longer per point.
- `--reset` goes back to the built-in calibration; or delete the file, or set
  `TOBII_CALIBRATION=embedded`. The daemon deletes the file before it
  writes the built-in calibration to the tracker, so should the tracker not
  take the write (it times out, or the tracker goes away), it still gets the
  built-in one at its next start. `mv calibration.bin.prev calibration.bin`
  (and a daemon restart) restores the previous one.
- `--dry-run --windowed` shows the screens without a tracker (`--windowed`
  only goes with `--dry-run`: a calibration needs the whole monitor).
- Only one client can calibrate at a time; a second one is told the tracker is
  busy. If the calibrating client dies, the daemon stops the session and puts
  the previous calibration back.
- Stream Engine applications can calibrate too, through `tobii_calibration_*`
  in `libtobii.so`; the result is saved the same way, once the application
  calls `tobii_calibration_stop` (whose failure after the save only its
  logger hears of, as an ERROR line). A session that ends before that (the
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

---

## 9. Notes & troubleshooting

- **Head pose + gaze together.** One engine serves head pose (from the device's
  0x50e IR image stream), gaze and presence concurrently; verified at 33 Hz each
  with gaze validity unchanged. Never stream the UVC camera while the daemon
  runs (the `camera`/`probe` research commands, or any app opening it
  through uvcvideo): that throttles the 0x83 streams to <1 Hz.
- **When the head pose starts.** `libtobii.so`'s comes with the first IR
  image, invalid until the tracker has a face clear of the image's edges;
  `tobii-opentrack` sends the first valid one. A cold device may
  additionally take one re-open (~10 s) before any stream arms.
- **A stalled tracker is re-opened in place.** No gaze for 5 s (10 s after a
  resume) outside a pause, or a USB error, makes the engine re-open and
  re-init the tracker. Only failures in a row add up: an open whose init
  fails, one that does not arm the stream, or one that loses it within 30 s.
  The first open that does not arm after a start or a lost stream is not a
  failure: the tracker arms only on the open after it (a firmware quirk).
  After two failures in a row the tracker is USB-reset, and after five the
  engine stops; the daemon starts a new one if a client still needs the
  tracker (or pre-warm is on) and it is plugged in. An engine that stops
  before the tracker was ever ready (no init went through) is replaced no
  sooner than 3 s after it started, the next one that does the same 6 s
  after it started, then 12, 24, 48 s and at most once a minute (the daemon
  checks every 3 s); the journal says so once per step, with the reason
  when there is one. A tracker ready again, a re-plug (the daemon finds the
  tracker at a new USB address, or back after it found it gone), no client
  wanting the tracker any more, or a daemon restart ends the backoff, and a
  client that subscribes or makes a request gets an engine at once whatever
  the backoff.
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
- **"It flies around."** With OpenTrack's `tracker-tobii` plugin as it
  stands, a pose marked invalid (the face lost, or at the edge of the
  camera's view) and a lost daemon connection both leave the pose unset,
  which OpenTrack takes as zeros, so the view jumps away; §8 has both, and
  the plugin's commits that hold the last valid pose and reconnect.
  `tobii-opentrack` sends no invalid pose, so OpenTrack holds its last one.
  A daemon too old or too new for the client no longer makes it fly: the
  client gets no head pose at all (below, and §5).
- **No head pose in a Stream Engine application.** After an install, the
  running `tobiid` is likely older than the library: it takes the
  subscription and sends nothing, and libtobii logs one WARN line after
  20 s, to an application that gave it a logger. Restart the daemon, then
  the application (§5). A daemon whose tracker sends no IR images (unplugged,
  paused, or with `TOBII_NO_IMAGE` set) sends none either.
- **"Rotations slide."** In `tobii-opentrack` and OpenTrack's
  `tracker-tobii` plugin a turn moves the position by design, as on Windows:
  the position is a point between the eyes (§8).
- **Tracking after logout.** User services stop at logout unless you enable
  lingering: `loginctl enable-linger $USER`. The shipped udev rule grants
  the tracker through `uaccess`, to the user of the active local session
  only: after logout the daemon keeps a tracker it has open, but a re-open
  (a stall, a re-plug) is refused (`no permission …`, below) until you log
  back in. Tracking that must survive that needs a rule granting the device
  to a group you are in instead.
- **Permission denied on the device.** The journal says `no permission to
  open the tracker; check the udev rule (INSTALL §3) and that this user's
  session is the active one`: the engine tries once more 0.7 s later (a
  tracker just plugged in refuses for a moment, until udev applies the
  rule), then stops, without a USB reset (it cannot fix a permission), and
  the daemon retries as above. Re-check §3 (udev rule, then `udevadm
  trigger` or a re-plug), and that your session is the active one on the
  seat (not another user's, not an SSH-only login): `uaccess` grants the
  device to that session's user alone. The tracker is taken at the next
  retry, or sooner: within 3 s of a re-plug, when a client next subscribes
  or asks for something (reconnect the application), or on
  `systemctl --user restart tobiid`. While the tracker still refuses,
  requests that need it live wait out their timeouts (a clock pair 25 s,
  ending `TOBII_ERROR_TIMED_OUT`).
- **"The tracker is in use by another process".** Something else has
  claimed the tracker's interface 0: usually a second `tobiid` (one started
  by hand while the service runs: `pgrep -a tobiid`), or a
  `tobii5-init-replay` command that opens the tracker (the replay, run
  without a subcommand, `camera`, `image83`, `probe`, …). The
  engine tries twice more, 0.7 s apart, in case it is a daemon handing
  over, then stops, and the daemon retries as above; stop the other process
  and the tracker is taken at the next retry, or at once when a client next
  subscribes or asks for something.
- **Checking the daemon end to end.** `target/release/tobii5-init-replay
  ipc-probe` asks the running daemon for everything (device info, track box,
  mounting, display area, stream types, hardware configuration, device name,
  calibration id, calibrating and paused states, fault and warning lists,
  clock) and reports the rate of every stream; `--set-display 597,336` also
  writes a display area.
