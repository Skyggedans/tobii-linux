# PaperWM gaze switcher — design

**Date:** 2026-06-28
**Status:** approved (design); pending implementation plan
**Author:** brainstormed with Claude

## Goal

While the user holds **Alt+Tab** (the PaperWM navigator is open and showing its
minimap "ribbon"), let **gaze** pick the application tile: looking at a minimap
tile makes that window the current selection ("hover it into focus"). Releasing
Alt activates the selected window — natively, via PaperWM's existing
`_finish`. The utility never opens or closes the switcher and never injects the
activation; it only steers the selection while Alt is held.

This is a sibling capability to the existing `tobii-gaze-keys` (look at a screen
edge → arrow key), but for a different interaction: discrete app selection on the
navigator ribbon.

## Environment (verified)

- GNOME Shell **50.2** on **Wayland**, single monitor (`HDMI-1`).
- **PaperWM** v148 enabled; `focus-mode = 'click'` (no sloppy/focus-follows-mouse).
- Standard mutter `switch-applications` binding is cleared by PaperWM; Alt+Tab is
  served by PaperWM's own navigator (`navigator.js` → `preview_navigate`).
- The repo already carries an in-place PaperWM patch (`// --- gaze-keys patch ---`
  in `keybindings.js`), so editing the extension in place is an established
  pattern here.
- `tobiid` daemon serves a gaze stream over a Unix socket (see
  `tobii/src/ipc.rs`); `tobii-gaze-keys` is an existing thin client of it.

## Approach decision

Chosen: **B — gaze integration inside PaperWM (GJS).**

The PaperWM navigator's `ActionDispatcher` grabs a modal and listens to
**key events only** — it does not select on pointer motion/hover
(`navigator.js:91-92`, `:164-214`). Selection is `space.selectedWindow`, moved
by keybinding actions, and on Alt release `_finish` activates it
(`navigator.js:253-265`). `focus-mode` is `'click'`, so there is no reliable
hover-to-select to lean on.

Therefore we do **not** rely on any hover behavior. The module runs inside the
extension, reads gaze, and **directly** sets the selection by calling PaperWM's
own `ensureViewport(window, space, { moveto: false })` for the window whose
minimap tile is under the gaze point. This is the most faithful to "look at it →
it's selected", uses exact known geometry (no pointer-warp or compositor
quirks), and needs no calibration to be usable because the live highlight gives
the user closed-loop feedback.

Rejected alternatives:
- **A — gaze warps the mouse pointer (pure Rust, uinput abs device):** depends on
  hover-select existing during the navigator grab (it does not, given
  `focus-mode='click'` and no PaperWM hover handler) and on reliable absolute
  cursor positioning on Wayland. Higher risk for this setup.
- **C — relative "gaze joystick" (Rust, emit switch-left/right):** simplest, but
  it is "scrub toward it and stop", not "point at it". Kept only as a fallback.

## Key PaperWM internals this depends on

All paths are in the PaperWM extension dir
(`~/.local/share/gnome-shell/extensions/paperwm@paperwm.github.com`).

- **Navigation state:** `navigator.js` exports `navigating` (bool) and
  `navigator` (the current `Navigator` instance). The instance holds
  `minimaps` — a `Map(space → Minimap | timeoutId)`. The minimap is created
  lazily ~200ms after navigation starts (`navigator.js:330-344`); until then the
  map value is a number (pending timeout) and must be skipped.
- **Select API:** `Tiling.ensureViewport(metaWindow, space, options)`
  (`tiling.js:4355`). With `{ moveto: false }` it sets
  `space.selectedWindow = metaWindow`, raises the clone, calls `updateSelection`,
  and emits `space.emit('select')` — **without scrolling** the live space
  (`tiling.js:4373-4403`). The minimap listens to `'select'`
  (`minimap.js:70`) and moves its highlight/label.
- **Active space:** `Tiling.spaces.selectedSpace` (matches what the navigator
  uses in `_doAction`/`_finish`). `space.monitor` carries `{x, y, width, height}`.
- **Minimap geometry** (`minimap.js`):
  - `Minimap extends Array` of columns; each column is an array of per-window
    tile actors. Each tile `c` carries `c.meta_window` and `c.clone.meta_window`
    (`minimap.js:158-163`).
  - The minimap `actor` is added to `Main.uiGroup` and centered on the monitor:
    `actor.x = monitor.x + floor((monitor.width - actor.width)/2)`,
    `actor.y = monitor.y + floor((monitor.height - actor.height)/2)`
    (`minimap.js:198-200`). `actor.height = space.height * 0.20`.
  - Inside `actor`: `clip` at `clip.set_position(12 + gap, 12 + round(1.5*gap))`
    (`minimap.js:63`), then `container` inside `clip`.
  - Tiles are laid out left→right by column, top→bottom within a column:
    `c.set_position(x, y)`, `x += colWidth + gap` (`minimap.js:177-191`).
  - `container.x` is the ribbon's horizontal scroll offset; it is pinned to 0
    while all tiles fit the clip, and only shifts when the selection nears a clip
    edge with an overflowing ribbon (`minimap.js:239-254`).
  - Therefore a tile's **global rect** is:
    `X0 = actor.x + clip.x + container.x + c.x`,
    `Y0 = actor.y + clip.y + container.y + c.y` (container.y = 0),
    size `c.width × c.height`.
- **Module wiring** (`extension.js:49-80`, `imports.js`): modules are listed in a
  `modules` array; PaperWM calls `enable(extension)` in order and `disable()` in
  reverse. Adding a module = new `gaze.js` + one line in `imports.js`
  (`export * as Gaze from './gaze.js'`) + one entry in the `modules` array.
  (The `~/.config/paperwm/user.js` hook is disabled in this version —
  `extension.js:162-177` — so in-place wiring is required.)

## tobiid IPC protocol (re-implemented in GJS)

From `tobii/src/ipc.rs`:

- **Socket:** `$XDG_RUNTIME_DIR/tobiid.sock`, else `/tmp/tobiid.sock`.
- **Framing:** each frame = `u32 LE length` + body.
- **Subscribe (client→daemon):** body = `[0x01, streams]`, `STREAM_GAZE = 0x02`.
- **Subscribed reply (daemon→client):** body = `[0x10, ok]` (`ok` 1/0).
- **Gaze frame (daemon→client):** body (18 bytes) =
  `[0x21, i64 ts_us (8, LE), u8 valid, f32 x (4, LE), f32 y (4, LE)]`,
  with `x, y ∈ 0..1`. Offsets: tag@0, ts@1, valid@9, x@10, y@14.

GJS decode: read with `Gio` async streams, accumulate bytes, parse the 4-byte
length, then the body via a `DataView` (`getFloat32(offset, /*littleEndian=*/true)`).

## Components / data flow

```
tobiid (gaze mode) ──unix socket──> gaze.js (in PaperWM)
                                       │  async Gio read, decode TAG_GAZE
                                       │  → lastGaze {x,y,valid}, EMA-smoothed
                                       ▼
                         on each gaze sample, IF Navigator.navigating
                         AND live Minimap exists for selectedSpace:
                                       │  map (x,y)→monitor px
                                       │  hit-test minimap tiles → metaWindow
                                       │  (hysteresis at tile borders)
                                       ▼
                         Tiling.ensureViewport(win, space, {moveto:false})
                                       │  → space.emit('select')
                                       ▼
                         minimap highlight + label follow (PaperWM)
                                       │
                user releases Alt ─────┘
                                       ▼
                         navigator _finish → activates space.selectedWindow
                         (unchanged PaperWM code)
```

## Detailed design

### gaze.js module shape

- `enable(extension)`:
  - Resolve socket path; open a persistent connection via `Gio.SocketClient`
    + `Gio.UnixSocketAddress`, fully async. Persistent (not lazy-per-navigation)
    for low latency when Alt+Tab opens; this matches `tobii-gaze-keys`, which
    already holds a gaze subscription, so the daemon is expected to be in gaze
    mode during gaze use.
  - On connect: write the subscribe frame; start the async read loop.
  - If connect fails or the socket EOFs (daemon restart), retry on a GLib timeout
    with backoff; stop retrying once disabled.
- Read loop: async-read into a buffer, parse frames, on `TAG_GAZE` update
  `lastGaze` and apply EMA smoothing; then call `maybeSelect()`.
- `maybeSelect()` (the only place that touches PaperWM selection):
  - Guard: `Navigator.navigating` true, `lastGaze.valid`, and a live `Minimap`
    instance exists for `Tiling.spaces.selectedSpace`
    (`Navigator.navigator.minimaps.get(space)` is a `Minimap`, not a number).
  - Map smoothed `(x,y)` → monitor pixel using `space.monitor` rect.
  - Hit-test tiles (global rects above); pick the containing tile's
    `meta_window`. Apply border **hysteresis**: only switch when the point is
    inside the candidate tile by a small margin, so jitter at a shared border
    doesn't flip-flop.
  - If the chosen window differs from `space.selectedWindow`, call
    `Tiling.ensureViewport(win, space, { moveto: false })`.
- `disable()`:
  - Cancel the async read (`Gio.Cancellable.cancel()`), close streams/socket,
    remove any GLib timeout sources, disconnect signals, null all references.
    Robust teardown is mandatory — a leaked read or source destabilizes the
    Shell.

### Gaze → pixel mapping / calibration (v1)

- Linear map of smoothed gaze `x,y ∈ 0..1` to the monitor rect (single monitor),
  then hit-test against the centered ribbon. Because the ribbon is centered and
  selection is discrete with a live highlight, exact calibration is not required
  — the user corrects with their eyes.
- Expose minimal knobs (constants first, settings later): EMA factor, optional
  scale/offset per axis, border-hysteresis margin.
- The adaptive-envelope auto-calibration used by `tobii-gaze-keys` is a possible
  later enhancement, not v1.

### Activation

Unchanged. On Alt release the navigator's `_finish` calls `nav.accept()` and
activates `space.selectedWindow` (`navigator.js:253-265`). The module does
nothing on release.

## Source layout, install & maintenance

- Source of truth lives in the **tobii repo**, e.g. `paperwm-gaze/gaze.js`, plus
  a Makefile target `install-paperwm-gaze` that:
  1. copies `gaze.js` into the PaperWM extension dir,
  2. idempotently injects the `imports.js` export and the `extension.js`
     `modules` entry, guarded by marker comments
     (`// --- tobii gaze switcher (begin/end) ---`), mirroring the existing
     `// --- gaze-keys patch ---` convention.
- Because PaperWM updates overwrite extension files, the install step must be
  re-runnable after each PaperWM upgrade. This is documented as the maintenance
  story (same caveat already accepted for the gaze-keys patch).

## Risks & spike-first plan

Riskiest assumptions, validated by a throwaway spike **before** wiring selection:

1. **Async framed socket read in GJS** — `Gio` async read of the length-prefixed
   protocol without blocking the Shell. Spike: log decoded gaze `(x,y,valid)`.
2. **Live minimap reachability + tile global coords** — `Navigator.navigator
   .minimaps.get(space)` returns a usable `Minimap`, and the computed global tile
   rects match what's on screen. Spike: while navigating, log the title of the
   tile under gaze; eyeball that it tracks the right tile.
3. **`ensureViewport(moveto:false)` side-effect-free selection** — moves
   highlight + activates correctly on release, without fighting PaperWM's own
   keyboard navigation or the existing gaze-keys patch.

Only after the spike confirms (1) and (2) do we wire `ensureViewport` and add
smoothing/hysteresis.

Known edge case: with an overflowing ribbon, `container.x` scrolls when the
selection nears a clip edge (`minimap.js:239-254`), shifting the gaze→tile
mapping under a fixed gaze. v1 mitigations: border hysteresis + EMA; optionally
pin `container.x` during gaze hover. Full edge-pan ("look past the edge to bring
off-ribbon windows in") is out of scope for v1.

## Testing

- **Unit-testable pure logic**, factored out of GJS side effects:
  - frame decoder (length-prefix + `TAG_GAZE` byte layout),
  - tile hit-test given a list of `{x0,y0,w,h,window}` rects + a point + the
    hysteresis rule.
  These can be exercised with a small GJS test harness or by mirroring the logic
  where convenient.
- **Integration** (navigator selection, activation on release, teardown) is
  verified live via the spike and manual runs; there is no automated GNOME Shell
  test harness here.

## Out of scope (v1) / future

- Edge-pan to reach windows beyond the visible ribbon.
- Vertical/column selection beyond what `getWindowAtPoint`-style row hit-testing
  already gives for stacked columns.
- Multi-monitor mapping (single monitor confirmed).
- Adaptive gaze auto-calibration and a GUI for the knobs.
- Dwell-to-confirm or blink gestures.

## Dependencies

- `tobiid` running and in **gaze mode** (same requirement as `tobii-gaze-keys`;
  gaze mode is mutually exclusive with the head/camera mode at the device).
- PaperWM enabled (the module is part of it).
