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
- **`tobii5-init-replay track`** — standalone head→OpenTrack without the daemon
  (claims the device directly). Handy for isolating issues; same tracker code.
- Diagnostics: `tobii5-init-replay probe` (camera-vs-0x83 concurrency),
  `… head83 <log.bin>` (research: head pose from 0x83 points).

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
