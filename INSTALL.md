# Install & Run

Linux stack for the **Tobii Eye Tracker 5** (USB `2104:0313`): a daemon that
claims the device once and serves head pose / gaze / presence to multiple
clients, plus an OpenTrack bridge and a Stream-Engine-like `libtobii.so`.

```
libtobii.so / tobii-opentrack ──unix socket──▶ tobiid ──USB──▶ Tobii ET5
                                                 │
                                  camera 6DOF head pose  OR  0x83 gaze+presence
                                  (mutually exclusive — one mode at a time)
```

---

## 1. Prerequisites

- Rust toolchain (stable) + Cargo.
- The repo includes the model (`models/face_landmarks.onnx`) and init capture
  (`init_packets_ep.txt`); both are embedded at **build** time, so no runtime
  data files are needed.
- A Tobii Eye Tracker 5 plugged in.

## 2. Build

```bash
cargo build --release
```

Produces in `target/release/`:

| Artifact | What it is |
|---|---|
| `tobiid` | the daemon (claims the device, serves clients) |
| `tobii-opentrack` | thin client: head pose → OpenTrack UDP |
| `tobii-gaze-keys` | thin client: gaze at screen edge → Left/Right arrow key |
| `tobii5-init-replay` | the CLI / analysis & diagnostics tool |
| `libtobii.so` | C ABI (`tobii_*`) in the shape of the Tobii Stream Engine; a daemon client |

## 2b. One-shot install (Makefile)

The fastest path — installs binaries + `libtobii.so` to `/usr/local`, the udev
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
cargo build --release
systemctl --user restart tobiid        # 4b
# or: systemctl --user restart tobiid.socket   # 4c
# or: pkill -f release/tobiid                   # 4a (next client respawns it)
```

## 6. Logs & debug

With a systemd unit the daemon's output (incl. `tracker pivot=…` and pose
debug) goes to the journal:

```bash
journalctl --user -u tobiid -f
```

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
| `TOBII_PREWARM` | unset | `head` or `gaze`: init the device at daemon start and keep it warm, so client connects are instant (IR illuminator stays on while the service runs). Recommended for an always-on head-tracking service. |
| `TOBII_NO_RESET` | unset | `1` skips the USB reset at init (a couple seconds faster; the reset rarely helps now that uvcvideo is kept off the device) |

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
- **`libtobii.so`** — C ABI: `tobii_api_create`, `tobii_device_create`,
  `tobii_head_pose_subscribe`, `tobii_gaze_point_subscribe`,
  `tobii_user_presence_subscribe`, `tobii_wait_for_callbacks`,
  `tobii_device_process_callbacks`. Position in **mm**, rotation in **radians**
  (`[pitch, yaw, roll]`), gaze normalized `0..1`. Link against it and it talks
  to the daemon for you.
- **`tobii-gaze-keys`** — subscribes to **gaze** (so it runs the daemon in gaze
  mode, mutually exclusive with head pose). While you look at the left/right edge
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
- Diagnostics: `tobii5-init-replay probe` (camera-vs-0x83 concurrency),
  `… head83 <log.bin>` (research: head pose from 0x83 points).

### Recenter (recalibrate the head rest pose)

Sit in your neutral pose and trigger a recenter — the daemon recalibrates
without restarting the device/stream. Three ways:

```bash
tobii-opentrack --recenter                 # one-shot client (bind to a hotkey)
systemctl --user kill -s SIGUSR1 tobiid    # or signal the service
kill -USR1 $(pgrep -x tobiid)              # or signal the process
```

Via `libtobii.so`: call `tobii_recenter(device)`. (Head mode only; gaze has no
rest pose.)

---

## 9. Notes & troubleshooting

- **One mode at a time.** The hardware can't stream the IR camera and the 0x83
  processed stream together. Head pose (camera) and gaze/presence (0x83) are
  *mutually exclusive*: while one client holds head pose, a gaze subscription
  gets `TOBII_ERROR_CONFLICTING_API` / a busy reply, and vice-versa.
- **Lazy claim.** An idle daemon holds no device; it opens the tracker only on
  the first subscription and releases it when the last client disconnects.
- **"It flies around."** You're talking to an **old daemon** (pre-rebuild) — it
  still has the previous code/units. Restart it (§5).
- **"Rotations slide."** Tune `TOBII_PIVOT_DOWN` / `TOBII_PIVOT_BACK` (§7).
- **Tracking after logout.** User services stop at logout unless you enable
  lingering: `loginctl enable-linger $USER`.
- **Permission denied on the device.** Re-check §3 (udev rule + re-plug).
