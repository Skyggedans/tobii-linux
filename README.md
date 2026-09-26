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
| Device info, track box, display area | on request | the device's own answers |
| Notifications: display area, calibration, pause, faults, warnings | on change | the device's own, and the daemon's for calibration and pause |
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
| Implemented (daemon-backed) | device lifetime and callbacks; gaze point, gaze origin, eye position, user position guide, presence, head pose, gaze data and IR image streams; notifications (display area, calibration state and id, pause, faults and warnings); device info, track box, display area (get/set), mounting, states; 2-D calibration, including discarding a point; the device/host clock pair (`tobii_timesync`); the tracker's stream catalogue; device pause and resume; the device name (kept by the host); the hardware configuration (provisional, see below) |
| Answered locally | API version, error texts (the DLL's), system clock, output frequency (33 Hz), enabled eye, capabilities, feature group, license validation, display-area calculation, calibration parsing, internal-stream support (the IR image only), internal-capability support (eyeball centres only), lens-configuration writability (never), the internal low-frequency head rotation and position, multiple faces position, wearable limited image and secondary camera image streams (never, as the DLL answers for a tracker it drives itself; behind Tobii's service, never captured) |
| `TOBII_ERROR_NOT_SUPPORTED` | what the ET5 was never observed doing: wearable, face id, illumination, power, firmware, diagnostics, extensions, custom streams, 3-D and per-eye calibration, calibration stimulus points (as the DLL's in-process legacy TTP module answers for an ET5; behind Tobii's service, never captured) |

Where the answers come from, and where they differ from Windows:

- **No licences.** Every key validates and the feature group is consumer, but
  nothing checks it: gaze data, timesync, calibration, display-area and name
  writes, the IR image, the stream catalogue and pause all work, though the
  Stream Engine reserves them for higher feature groups or, for the IR
  image, an additional-features licence.
- **Facts from the last init.** Device info, track box, display area,
  mounting, the stream catalogue, the hardware configuration and the fault
  and warning lists (`tobii_get_state_string`) are what the tracker reported
  at its last init, so they are answered even while it is unplugged
  (`TOBII_ERROR_TIMED_OUT` only if the daemon has not seen a tracker yet). A
  fault or warning list the tracker announces later (3200/3210, a layout
  taken from the Stream Engine and never yet seen from an ET5) replaces the
  one its init reported, and reaches notification subscribers either way, as
  in the Stream Engine. The first call that needs the tracker starts it, and
  it then stays on (IR illuminator lit) for as long as that connection is
  open, as for an open device in the Stream Engine. While it is unplugged,
  such a call (a clock pair, a pause, starting, retrieving or applying a
  calibration, a display-area write) is `TOBII_ERROR_CONNECTION_FAILED` at
  once from when the daemon has given the tracker up, a few seconds after the
  unplug (until then the call waits on the daemon's re-opens), and the daemon
  starts the tracker once it is plugged back in, for a connection still open.
  A calibration session its owner is not already stopping ends, saving
  nothing, when the daemon loses the tracker or the tracker re-initialises:
  its owner's later calls in it, `tobii_calibration_stop` included, are
  `TOBII_ERROR_CALIBRATION_NOT_STARTED`. A stop under way finishes, and a
  calibration it has saved stays saved even if the stop then fails
  (`TOBII_ERROR_CONNECTION_FAILED` when the tracker went away); the tracker
  loads it at its next init.
- **Lost daemon.** When the connection to `tobiid` is lost (the daemon
  stopped, crashed or was restarted), `tobii_device_process_callbacks`
  delivers what had arrived and then returns `TOBII_ERROR_CONNECTION_FAILED`
  on every call; the DLL returns it without delivering its queue on that
  call. `tobii_wait_for_callbacks` wakes once for the loss and never returns
  `TOBII_ERROR_CONNECTION_FAILED`. Nothing reconnects by itself: as the
  Stream Engine documentation says, the application calls
  `tobii_device_reconnect`. That only connects to a running daemon (it never
  spawns one, unlike `tobii_device_create`), gives up with
  `TOBII_ERROR_CONNECTION_FAILED` within ~500 ms when it cannot, and
  restores the subscriptions, not a calibration session or a pause the lost
  connection held. A tracker unplug is not a lost connection and never shows
  in `tobii_device_process_callbacks`: the daemon keeps every connection, and
  the samples resume on it once the tracker is back. Requests that need the
  tracker get `TOBII_ERROR_CONNECTION_FAILED` meanwhile (above), but that is
  the daemon's answer over a live connection, and a reconnect then succeeds
  without bringing the tracker back. OpenTrack's `tracker-tobii` plugin
  needs a reconnect call to recover by itself; as it stands, stopping and
  starting tracking recovers it (INSTALL.md §8).
- **Pause.** One state for the tracker, shared by every client, as in the
  Stream Engine: the last call wins and any client may resume.
  `TOBII_STATE_DEVICE_PAUSED` and a `DEVICE_PAUSED_STATE_CHANGED`
  notification change as soon as the tracker accepts the call. A pause ends
  when the client that paused last disconnects, and when the tracker
  re-initialises (after a USB failure or an unplug). A resume the tracker
  does not answer still succeeds: the daemon re-opens the silent tracker, and
  its init resumes it. Pausing during a calibration session is
  `TOBII_ERROR_CALIBRATION_BUSY`; starting one, or asking for
  `tobii_timesync`, while paused is `TOBII_ERROR_NOT_AVAILABLE`.
- **Notifications.** Six of the thirteen types are delivered:
  `CALIBRATION_STATE_CHANGED` and `DEVICE_PAUSED_STATE_CHANGED` from the
  daemon itself, when a calibration session starts or ends and when the
  tracker accepts a pause or resume (the Stream Engine waits for the
  tracker's own messages, of which the ET5 was seen sending only the
  pause's); `DISPLAY_AREA_CHANGED`, `CALIBRATION_ID_CHANGED`,
  `FAULTS_CHANGED` and `WARNINGS_CHANGED` from the tracker's, one for each
  (as the Stream Engine does for faults and warnings, and as far as traced
  for the other two). The others never come: the output frequency is
  fixed at 33 Hz, both eyes are always used, the ET5 was never seen changing
  its track box, entering power save or reporting a face type, and the
  Stream Engine has no source for `CALIBRATION_ENABLED_EYE_CHANGED`.
  Exclusive mode is not reported either: the ET5 sends what the Stream
  Engine reads as exclusive mode on and off only on Linux, whenever an open
  starts the sensor (a cold engine start, or a re-open after the stream
  was lost): on with that open, off with the re-open that primes the
  stream about 10 s later. It never does so on Windows, so it says nothing
  about another application; `TOBII_STATE_EXCLUSIVE_MODE` stays false. The
  Stream Engine most likely also delivers one
  `COMBINED_GAZE_EYE_SELECTION_CHANGED` (both eyes) after
  `tobii_device_create` or `tobii_device_reconnect`, from the tracker's
  answer to its init; libtobii does not. `tools/abi/README.md` has the
  Stream Engine's side.
- **Device name.** A name set with `tobii_set_device_name` is kept by the
  daemon in `~/.config/tobii/device-name`, for every client and later
  sessions; nothing is written to the tracker. Until one is set,
  `tobii_get_device_name` gives the model.
- **Calibration parsing.** `tobii_calibration_parse` checks the whole blob
  before it hands out a point: data that is not a valid calibration is
  `TOBII_ERROR_OPERATION_FAILED`, as the Stream Engine documentation says.
  That is a blob shorter than its 44-byte header or over 4 MiB, a point list
  that lies outside the blob, holds more than 256 points or does not end
  exactly at `data_size`, a status word other than -1, 0, 1 or 2 (its low
  32 bits read as a signed int, as the Stream Engine reads them, with the
  upper 32 zero, or all ones with -1), and a value that is not finite or
  lies more than half a display off it, but for the mapping of an eye
  marked failed (-1). The Stream Engine returns that only for a negative
  point count; it reads everything else as given, past `data_size` if the
  blob says so, and returns `TOBII_ERROR_NO_ERROR` (8 zero bytes are an
  empty calibration there). Both report an eye whose status word is -1 as
  `FAILED_OR_INVALID` and pass its mapping on as the blob holds it.
- **Timestamps.** Every callback timestamp is on `tobii_system_clock`'s
  clock, as in the Stream Engine; the tracker's clock is left only in gaze
  data's `timestamp_tracker_us` and `tobii_timesync`'s `tracker_us`. A head
  pose carries the time of the IR image it was made from, as there. The
  Stream Engine adds one offset per connection, from round trips to its
  service, and refreshes it only in `tobii_update_timesync` and
  `tobii_timesync`. Here the daemon's USB engine estimates the offset from
  the arrivals: the smallest receipt-minus-device time of the gaze frames and
  IR images of the last 120 s (the gaze frames alone when the image stream is
  off), started afresh at every tracker init, which may restart the
  tracker's clock. So it follows the drift between the clocks (5 to 13 ppm
  in the captures, some 50 ms an hour for a fixed offset), and every client
  gets the same stamps. The interval between two stamps can then differ
  from the tracker's by a couple of ms when the estimate moves (the Stream
  Engine's are exact within a connection), though never below 1 µs within a
  stream while the tracker stays open; across an init the stamps run on with
  the host clock instead of jumping back, and a presence reported again to a
  new subscriber keeps the stamp it was last reported with. A stamp is no
  later than the daemon's read of its sample (1 µs past it when two samples
  of a stream are read together), but for a sample read during the
  tracker's init, which gets the time it is delivered, a few ms late.
- **Clock pair.** `tobii_timesync` pairs the device timestamp of the next gaze
  frame with the host clock (`CLOCK_MONOTONIC`, as `tobii_system_clock`) in a
  fixed 30 ms bracket; the Stream Engine times a round trip instead, on
  `QueryPerformanceCounter`. With no tracker plugged in it is
  `TOBII_ERROR_CONNECTION_FAILED` rather than a wait, and a tracker unplugged
  shortly before or during the call makes it so about a second after the
  daemon gives the tracker up.
- **Hardware configuration.** Its layout is the DLL's, but what the fields
  hold is inferred from one Windows capture, and on Linux the ET5 has
  answered its command (2120) with no data, so
  `tobii_hardware_configuration_get` is `TOBII_ERROR_NOT_SUPPORTED` until it
  reports one.
- **Threads.** As the Stream Engine promises, the functions may be called
  from several threads at once, on one device too: a device's state is
  locked by concern, as the DLL's is. Its requests, subscription changes and
  reconnects run one at a time, in the order they are called, each for its
  whole round trip to tobiid (a pause may take up to a minute), so one
  thread's calls made back to back hold another thread's up for one of them
  at most. They do not hold up its callbacks, processing or waiting, but for
  a reconnect's round trip (~500 ms at most), and a recenter, a write with
  no reply, waits for those under way or called before it; a reconnect's
  ~500 ms counts from when the calls ahead of it, and then a process call
  another thread is making, have finished. Its callbacks run one at a time,
  on whichever thread processes it, and a subscribe, an unsubscribe, a clear
  or a reconnect waits for one running on another thread; once an
  unsubscribe returns, its callback is not running and never runs again.
  `tobii_device_destroy` and `tobii_api_destroy` take no lock, as in the
  Stream Engine (whose documentation says so for `tobii_device_destroy`): no
  other thread may be inside a call on the handle, or use it afterwards.
  Threads that create devices at once while no daemon runs spawn one
  `tobiid` between them: the others wait for that spawn, then connect to its
  daemon or fail as it did (two processes doing so can still spawn one
  each). `TOBII_ERROR_CALLBACK_IN_PROGRESS` guards only the thread a
  callback, the logger or `tobii_calibration_retrieve`'s receiver runs on. A
  callback, or the logger, must not block on another thread's call into any
  device (nor on a thread that waits for one), which can deadlock, as in the
  Stream Engine: only processing and waiting are sure to go on while a
  callback runs. The retrieve receiver holds no lock, so it may wait for
  other threads' calls. Unlike the DLL, a device's requests, subscription
  changes and reconnects run in the order they are called, where the DLL's
  critical section promises no order among its waiters (Windows semantics,
  not read from the DLL), so there one thread's calls made back to back may
  keep another thread's out for long; `tobii_wait_for_callbacks` waits on a
  device another thread is processing as on any other, where the DLL skips
  such a device, returning at once when it was the only one, so a
  wait-and-process loop on that device alone spins; a subscribe lets the
  device's other callbacks run during its round trip, where the DLL holds
  them back, so a subscribe that fails may have had its callback called
  before it returned; a `tobii_device_process_callbacks` that finds another
  thread processing returns at once, as there, but with
  `TOBII_ERROR_CONNECTION_FAILED` once the loss has been reported, and
  delivers nothing (the DLL first delivers the device's queued
  notifications); a clear waits for another thread's processing, never for a
  request (both wait for a reconnect's round trip);
  `tobii_calibration_retrieve` calls its receiver with no lock held, where
  the DLL holds the device's API mutex; and the logger is never called under
  a lock of the call that logs.
- **Logging and allocation.** The `tobii_custom_log_t` logger gets
  libtobii's own few lines (a refused `field_of_use`, a failed connect or
  reconnect, a lost daemon connection once per loss, a daemon reply that
  does not decode, each connect and reconnect), not the line per failing
  call the Stream Engine writes: the returned status says that. It is
  called on the thread inside the call that logs, from several threads at
  once if they log at once (their lines may interleave), and a device keeps
  logging through it after `tobii_api_destroy`. A `tobii_custom_alloc_t` is
  checked as in the Stream Engine and never called: libtobii allocates with
  Rust's allocator, where the DLL allocates the API handle, each device and
  long log lines through it (INSTALL.md §8).

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
| `tobii-ipc` | the daemon protocol, the display geometry and the host clock | none (`libc` only) |
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
  per-eye calibration were never captured and are not supported. Discarding a
  2-D point and pausing the tracker send the commands the DLL sends for them
  (1080, and 3100 with 1), which were never captured.

## License

MIT.
