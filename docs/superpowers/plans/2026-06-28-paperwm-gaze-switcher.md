# PaperWM gaze switcher — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While Alt+Tab holds the PaperWM navigator open, let gaze pick the app tile on the minimap ribbon; releasing Alt activates it via PaperWM's own `_finish`.

**Architecture:** An in-place GJS module (`gaze.js`) added to the PaperWM extension reads the `tobiid` gaze stream over a Unix socket (async Gio), maps the smoothed gaze point to the minimap's tile geometry, and sets the selection through PaperWM's `Tiling.ensureViewport(win, space, {moveto:false})`. Pure, gi-free logic (frame decode, gaze→pixel, hit-test) lives in `gazelib.js` and is unit-tested with standalone `gjs`. Activation is unchanged PaperWM behavior.

**Tech Stack:** GJS (GNOME Shell 50.2, ESM modules), Gio/GLib, PaperWM v148 internals (`Tiling`, `Navigator`, `Minimap`), the `tobiid` Rust daemon's socket protocol. Build/install via `make`.

## Global Constraints

- Target: GNOME Shell **50.2**, **Wayland**, single monitor, PaperWM **v148** at `~/.local/share/gnome-shell/extensions/paperwm@paperwm.github.com`.
- GJS files are **ES modules**. Pure logic (`gazelib.js`) must import **no** `gi://` / `resource://` so it runs under `gjs -m` for tests.
- **No blocking I/O on the Shell main loop.** All socket work in `gaze.js` uses async Gio (`connect_async`, `read_bytes_async`) driven by the Shell's main loop. (Standalone probe/test scripts may block — they are not in the Shell.)
- **Mandatory teardown** on `disable()`: cancel the `Gio.Cancellable`, remove any GLib timeout source, close the connection, null references. A leaked async read or source destabilizes the Shell.
- tobiid IPC (verbatim from `tobii/src/ipc.rs`): socket `$XDG_RUNTIME_DIR/tobiid.sock` else `/tmp/tobiid.sock`; frame = `u32 LE length` + body; SUBSCRIBE body = `[0x01, streams]` with `STREAM_GAZE = 0x02`; SUBSCRIBED body = `[0x10, ok]`; GAZE body (18 bytes) = `[0x21, i64 ts_us (LE), u8 valid, f32 x (LE), f32 y (LE)]`, offsets tag@0 ts@1 valid@9 x@10 y@14, with `x,y ∈ 0..1`.
- Selection API: `Tiling.ensureViewport(metaWindow, space, { moveto: false })`. **Never** inject the activation — releasing Alt is handled by PaperWM `_finish`.
- Install is **in place** and **re-runnable** (idempotent, marker-guarded) because PaperWM upgrades overwrite extension files. Source of truth lives in the tobii repo under `paperwm-gaze/`.
- Module wiring: `extension.js` imports `Gaze` directly from `./gaze.js` and adds it to the `modules` array; `imports.js` is **not** modified.

---

### Task 1: Pure frame decoder + EMA + gaze→pixel + hit-test (`gazelib.js`)

**Files:**
- Create: `paperwm-gaze/gazelib.js`
- Test: `paperwm-gaze/tests/test_gazelib.js`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TAG_SUBSCRIBE=0x01`, `TAG_SUBSCRIBED=0x10`, `TAG_GAZE=0x21`, `STREAM_GAZE=0x02` (consts)
  - `frameWithLen(body: Uint8Array): Uint8Array`
  - `subscribeFrame(streams=STREAM_GAZE): Uint8Array`
  - `parseFrames(buf: Uint8Array): { gaze: {tsUs,valid,x,y}[], subscribedOk: bool|null, rest: Uint8Array }`
  - `ema(prev: number|null, cur: number, alpha: number): number`
  - `gazeToPixel(gx, gy, monitor: {x,y,width,height}): { px, py }`
  - `hitTest(tiles: {x0,y0,w,h,id}[], px, py, prevId, margin): id|null`

- [ ] **Step 1: Verify gjs is available**

Run: `command -v gjs && gjs --version`
Expected: a path and a version line (e.g. `1.82.x`). If missing, install `gjs` before continuing.

- [ ] **Step 2: Write the failing test**

Create `paperwm-gaze/tests/test_gazelib.js`:

```javascript
import system from 'system';
import {
    parseFrames, frameWithLen, subscribeFrame, ema, gazeToPixel, hitTest,
    TAG_GAZE, TAG_SUBSCRIBE, STREAM_GAZE,
} from '../gazelib.js';

let failures = 0;
function ok(cond, msg) {
    print(`${cond ? 'ok' : 'NOT OK'} - ${msg}`);
    if (!cond) failures++;
}
const approx = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;

function gazeFrame(tsUs, valid, x, y) {
    const body = new Uint8Array(18);
    const dv = new DataView(body.buffer);
    body[0] = TAG_GAZE;
    dv.setBigInt64(1, BigInt(tsUs), true);
    body[9] = valid ? 1 : 0;
    dv.setFloat32(10, x, true);
    dv.setFloat32(14, y, true);
    return frameWithLen(body);
}
function cat(...arrs) {
    const out = new Uint8Array(arrs.reduce((n, a) => n + a.length, 0));
    let o = 0;
    for (const a of arrs) { out.set(a, o); o += a.length; }
    return out;
}

// subscribe frame layout
{
    const f = subscribeFrame(STREAM_GAZE);
    ok(f.length === 6, 'subscribe frame is 6 bytes');
    ok(f[4] === TAG_SUBSCRIBE && f[5] === STREAM_GAZE, 'subscribe body bytes');
}
// one gaze frame
{
    const { gaze, rest } = parseFrames(gazeFrame(123, true, 0.25, 0.75));
    ok(gaze.length === 1, 'one gaze frame parsed');
    ok(gaze[0].valid && approx(gaze[0].x, 0.25) && approx(gaze[0].y, 0.75), 'gaze decoded');
    ok(rest.length === 0, 'no leftover');
}
// two frames + partial third held back as rest
{
    const full = cat(gazeFrame(1, true, 0.1, 0.2), gazeFrame(2, false, 0.3, 0.4));
    const partial = gazeFrame(3, true, 0.5, 0.6).subarray(0, 10);
    const { gaze, rest } = parseFrames(cat(full, partial));
    ok(gaze.length === 2, 'two complete frames parsed');
    ok(rest.length === 10, 'partial retained as rest');
}
// ema
ok(ema(null, 5, 0.5) === 5, 'ema seeds first value');
ok(approx(ema(0, 10, 0.5), 5), 'ema halfway');
// gazeToPixel
{
    const p = gazeToPixel(0.5, 0.5, { x: 0, y: 0, width: 1000, height: 800 });
    ok(approx(p.px, 500) && approx(p.py, 400), 'gazeToPixel center');
}
// hitTest + hysteresis
{
    const tiles = [
        { x0: 0, y0: 0, w: 100, h: 100, id: 'a' },
        { x0: 100, y0: 0, w: 100, h: 100, id: 'b' },
    ];
    ok(hitTest(tiles, 50, 50, null, 10) === 'a', 'hits a');
    ok(hitTest(tiles, 150, 50, 'a', 10) === 'b', 'moves to b when inside');
    ok(hitTest(tiles, 105, 50, 'a', 10) === 'a', 'hysteresis keeps a near border');
    ok(hitTest(tiles, 300, 50, 'b', 10) === 'b', 'outside keeps previous');
}

print(failures === 0 ? 'ALL PASS' : `FAILED ${failures}`);
if (failures > 0) system.exit(1);
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `gjs -m paperwm-gaze/tests/test_gazelib.js`
Expected: FAIL — error like `Error: Module './../gazelib.js' not found` (file doesn't exist yet).

- [ ] **Step 4: Write `gazelib.js`**

Create `paperwm-gaze/gazelib.js`:

```javascript
// Pure, gi-free helpers for the tobii gaze switcher. Imported by both
// gaze.js (inside GNOME Shell) and the standalone gjs test/probe scripts.
// Keep this file free of gi:// / resource:// imports so it runs under `gjs -m`.

// tobiid IPC constants — mirror tobii/src/ipc.rs.
export const TAG_SUBSCRIBE = 0x01;
export const TAG_SUBSCRIBED = 0x10;
export const TAG_GAZE = 0x21;
export const STREAM_GAZE = 0x02;

// Prefix a frame body with its u32 little-endian length.
export function frameWithLen(body) {
    const out = new Uint8Array(4 + body.length);
    new DataView(out.buffer).setUint32(0, body.length, true);
    out.set(body, 4);
    return out;
}

// A complete SUBSCRIBE frame (length-prefixed) for the given stream bitmask.
export function subscribeFrame(streams = STREAM_GAZE) {
    return frameWithLen(Uint8Array.from([TAG_SUBSCRIBE, streams]));
}

// Parse all complete frames in `buf`. Returns decoded gaze samples, the last
// SUBSCRIBED ack seen (or null), and the unconsumed tail bytes (`rest`).
export function parseFrames(buf) {
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const gaze = [];
    let subscribedOk = null;
    let off = 0;
    while (buf.length - off >= 4) {
        const len = dv.getUint32(off, true);
        if (buf.length - off - 4 < len)
            break; // body not fully arrived yet
        const b = off + 4;
        const tag = buf[b];
        if (tag === TAG_GAZE && len >= 18) {
            gaze.push({
                tsUs: Number(dv.getBigInt64(b + 1, true)),
                valid: buf[b + 9] !== 0,
                x: dv.getFloat32(b + 10, true),
                y: dv.getFloat32(b + 14, true),
            });
        } else if (tag === TAG_SUBSCRIBED && len >= 2) {
            subscribedOk = buf[b + 1] !== 0;
        }
        off = b + len;
    }
    return { gaze, subscribedOk, rest: buf.subarray(off) };
}

// Exponential moving average for a scalar; seeds with `cur` on first call.
export function ema(prev, cur, alpha) {
    if (prev === null || prev === undefined)
        return cur;
    return prev + alpha * (cur - prev);
}

// Map gaze (gx,gy in 0..1) to an absolute pixel on `monitor`.
export function gazeToPixel(gx, gy, monitor) {
    return {
        px: monitor.x + gx * monitor.width,
        py: monitor.y + gy * monitor.height,
    };
}

// Find the id of the tile under (px,py). Border hysteresis: when crossing into
// a new tile, require being inside it by `margin` before switching; while
// outside every tile, keep the previous id. `tiles`: [{x0,y0,w,h,id}].
export function hitTest(tiles, px, py, prevId, margin) {
    let hit = null;
    for (const t of tiles) {
        if (px >= t.x0 && px <= t.x0 + t.w && py >= t.y0 && py <= t.y0 + t.h) {
            hit = t.id;
            break;
        }
    }
    if (hit !== null && hit !== prevId && prevId !== null && prevId !== undefined) {
        const t = tiles.find(t => t.id === hit);
        const inset = px >= t.x0 + margin && px <= t.x0 + t.w - margin &&
                      py >= t.y0 + margin && py <= t.y0 + t.h - margin;
        if (!inset)
            return prevId; // not clearly inside the new tile yet
    }
    return hit !== null ? hit : (prevId ?? null);
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `gjs -m paperwm-gaze/tests/test_gazelib.js`
Expected: each line `ok - ...`, final line `ALL PASS`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add paperwm-gaze/gazelib.js paperwm-gaze/tests/test_gazelib.js
git commit -m "feat(gaze-switch): pure frame decode + gaze map + hit-test with tests"
```

---

### Task 2: Standalone socket probe (de-risk async Gio framing outside the Shell)

**Files:**
- Create: `paperwm-gaze/tests/probe_socket.js`

**Interfaces:**
- Consumes: `subscribeFrame`, `parseFrames`, `STREAM_GAZE` from `gazelib.js`.
- Produces: nothing imported elsewhere (manual diagnostic).

This proves the Gio async read + length-prefix framing against the live daemon **before** that code runs inside the Shell (spec risk #1).

- [ ] **Step 1: Write the probe**

Create `paperwm-gaze/tests/probe_socket.js`:

```javascript
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import system from 'system';
import { subscribeFrame, parseFrames, STREAM_GAZE } from '../gazelib.js';

const runtime = GLib.getenv('XDG_RUNTIME_DIR');
const path = runtime ? `${runtime}/tobiid.sock` : '/tmp/tobiid.sock';

const client = new Gio.SocketClient();
let conn;
try {
    conn = client.connect(new Gio.UnixSocketAddress({ path }), null);
} catch (e) {
    printerr(`connect failed: ${e}`);
    system.exit(1);
}
conn.get_output_stream().write_all(subscribeFrame(STREAM_GAZE), null);
const istream = conn.get_input_stream();

let buf = new Uint8Array(0);
let count = 0;
const loop = GLib.MainLoop.new(null, false);

function readChunk() {
    istream.read_bytes_async(4096, GLib.PRIORITY_DEFAULT, null, (s, res) => {
        const bytes = s.read_bytes_finish(res);
        const arr = bytes.get_data();
        if (!arr || arr.length === 0) { print('EOF'); loop.quit(); return; }
        const merged = new Uint8Array(buf.length + arr.length);
        merged.set(buf, 0); merged.set(arr, buf.length);
        const { gaze, subscribedOk, rest } = parseFrames(merged);
        buf = Uint8Array.from(rest);
        if (subscribedOk !== null) print(`subscribed ok=${subscribedOk}`);
        for (const g of gaze) {
            print(`gaze valid=${g.valid} x=${g.x.toFixed(3)} y=${g.y.toFixed(3)}`);
            if (++count >= 10) { loop.quit(); return; }
        }
        readChunk();
    });
}
readChunk();
loop.run();
print('done');
```

- [ ] **Step 2: Ensure the daemon is up in gaze mode**

Run: `test -S "$XDG_RUNTIME_DIR/tobiid.sock" && echo socket-present || echo no-socket`
If `no-socket`, start the daemon as you do for `tobii-gaze-keys` (e.g. `systemctl --user start tobiid` or run `target/release/tobiid`). The daemon must be in **gaze** mode (not head/camera).

- [ ] **Step 3: Run the probe and verify gaze output**

Run: `gjs -m paperwm-gaze/tests/probe_socket.js`
Expected: `subscribed ok=true`, then ~10 lines `gaze valid=true x=0.xxx y=0.xxx` while you look around, then `done`.
If you see `subscribed ok=false`, the daemon is busy in the other (head) mode — switch it to gaze mode and retry. This confirms the framing/decoder works over a real socket.

- [ ] **Step 4: Commit**

```bash
git add paperwm-gaze/tests/probe_socket.js
git commit -m "test(gaze-switch): standalone gjs socket probe against tobiid"
```

---

### Task 3: `gaze.js` lifecycle + socket client (connects, subscribes, logs gaze)

**Files:**
- Create: `paperwm-gaze/gaze.js`

**Interfaces:**
- Consumes: `subscribeFrame`, `parseFrames`, `ema`, `STREAM_GAZE` from `./gazelib.js`; PaperWM `Tiling`, `Navigator` from `./imports.js` (used in later tasks).
- Produces: `enable()`, `disable()` (PaperWM module contract); class `GazeSwitcher` with async-safe teardown.

This builds the real in-Shell socket client (async Gio, reconnect, Cancellable teardown). At this stage `_onGaze` only smooths + logs, so we can verify the Shell connects and streams without yet touching PaperWM selection.

- [ ] **Step 1: Write `gaze.js` (log-only gaze)**

Create `paperwm-gaze/gaze.js`:

```javascript
// tobii gaze switcher: while the PaperWM navigator is open (Alt+Tab held),
// pick the minimap tile under the user's gaze. Reads tobiid's gaze stream over
// a Unix socket (async Gio) and drives selection via Tiling.ensureViewport.
// Source of truth lives in the tobii repo (paperwm-gaze/); installed in place.

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

import { Tiling, Navigator } from './imports.js';
import { subscribeFrame, parseFrames, ema, STREAM_GAZE } from './gazelib.js';

const EMA_ALPHA = 0.4;        // gaze smoothing (0..1; higher = snappier)
const HYSTERESIS_PX = 24;     // border deadband before switching tiles
const RECONNECT_MS = 1000;    // backoff between reconnect attempts

let state = null;

class GazeSwitcher {
    constructor() {
        this._cancellable = new Gio.Cancellable();
        this._conn = null;
        this._istream = null;
        this._buf = new Uint8Array(0);
        this._sx = null;          // smoothed gaze x (0..1)
        this._sy = null;          // smoothed gaze y (0..1)
        this._reconnectId = 0;
        this._prevId = null;      // previously selected tile id
        this._connect();
    }

    _socketPath() {
        const r = GLib.getenv('XDG_RUNTIME_DIR');
        return r ? `${r}/tobiid.sock` : '/tmp/tobiid.sock';
    }

    _connect() {
        const client = new Gio.SocketClient();
        const addr = new Gio.UnixSocketAddress({ path: this._socketPath() });
        client.connect_async(addr, this._cancellable, (c, res) => {
            let conn;
            try {
                conn = c.connect_finish(res);
            } catch (e) {
                if (!this._cancellable.is_cancelled()) this._scheduleReconnect();
                return;
            }
            this._conn = conn;
            this._istream = conn.get_input_stream();
            try {
                conn.get_output_stream().write_all(
                    subscribeFrame(STREAM_GAZE), this._cancellable);
            } catch (e) {
                this._scheduleReconnect();
                return;
            }
            this._buf = new Uint8Array(0);
            this._readChunk();
        });
    }

    _scheduleReconnect() {
        if (this._cancellable.is_cancelled() || this._reconnectId) return;
        this._reconnectId = GLib.timeout_add(
            GLib.PRIORITY_DEFAULT, RECONNECT_MS, () => {
                this._reconnectId = 0;
                if (this._cancellable.is_cancelled()) return GLib.SOURCE_REMOVE;
                this._connect();
                return GLib.SOURCE_REMOVE;
            });
    }

    _readChunk() {
        if (!this._istream || this._cancellable.is_cancelled()) return;
        this._istream.read_bytes_async(
            4096, GLib.PRIORITY_DEFAULT, this._cancellable, (s, res) => {
                let bytes;
                try {
                    bytes = s.read_bytes_finish(res);
                } catch (e) {
                    if (!this._cancellable.is_cancelled()) this._reconnect();
                    return;
                }
                const arr = bytes.get_data();
                if (!arr || arr.length === 0) { this._reconnect(); return; }
                const merged = new Uint8Array(this._buf.length + arr.length);
                merged.set(this._buf, 0);
                merged.set(arr, this._buf.length);
                const { gaze, rest } = parseFrames(merged);
                this._buf = Uint8Array.from(rest);
                for (const g of gaze) this._onGaze(g);
                this._readChunk();
            });
    }

    _reconnect() {
        try { this._conn?.close(null); } catch (e) {}
        this._conn = null;
        this._istream = null;
        this._scheduleReconnect();
    }

    _onGaze(g) {
        if (!g.valid) return;
        this._sx = ema(this._sx, g.x, EMA_ALPHA);
        this._sy = ema(this._sy, g.y, EMA_ALPHA);
        // Task 5 replaces this log with selection logic.
        if (Navigator.navigating)
            console.log(`#tobii-gaze x=${this._sx.toFixed(3)} y=${this._sy.toFixed(3)}`);
    }

    destroy() {
        this._cancellable.cancel();
        if (this._reconnectId) {
            GLib.source_remove(this._reconnectId);
            this._reconnectId = 0;
        }
        try { this._conn?.close(null); } catch (e) {}
        this._conn = null;
        this._istream = null;
        this._buf = null;
    }
}

export function enable() {
    state = new GazeSwitcher();
    console.log('#tobii-gaze switcher enabled');
}

export function disable() {
    state?.destroy();
    state = null;
    console.log('#tobii-gaze switcher disabled');
}
```

- [ ] **Step 2: Syntax-check the module**

Run: `gjs -c "void 0" && node --check paperwm-gaze/gaze.js 2>/dev/null || echo "skip node check"`
Expected: no syntax error printed. (The `gi://` / `./imports.js` imports cannot resolve outside the Shell — this step only catches gross syntax errors; full validation happens in Task 4.)

- [ ] **Step 3: Commit**

```bash
git add paperwm-gaze/gaze.js
git commit -m "feat(gaze-switch): GJS module skeleton with async tobiid socket client"
```

---

### Task 4: Install target + in-Shell verification (connect & log)

**Files:**
- Create: `paperwm-gaze/Makefile` (self-contained; does NOT modify the repo root Makefile)

**Interfaces:**
- Consumes: `paperwm-gaze/gaze.js`, `paperwm-gaze/gazelib.js`.
- Produces: `make -C paperwm-gaze install`, `make -C paperwm-gaze uninstall`, `make -C paperwm-gaze test`.

> Note: the install target lives in its own `paperwm-gaze/Makefile` (not the repo
> root Makefile) so the gaze switcher stays self-contained and the install never
> entangles unrelated root-Makefile changes. Recipes run with cwd = `paperwm-gaze/`,
> so source files are referenced as `gaze.js`/`gazelib.js`.

- [ ] **Step 1: Create the self-contained Makefile**

Create `paperwm-gaze/Makefile` (use **tab** indentation for recipe lines, as Make requires):

```makefile
# tobii gaze switcher — self-contained install into the PaperWM extension.
PAPERWM_EXT ?= $(HOME)/.local/share/gnome-shell/extensions/paperwm@paperwm.github.com

.PHONY: install uninstall test
install:
	@test -d "$(PAPERWM_EXT)" || { echo "PaperWM not found at $(PAPERWM_EXT)"; exit 1; }
	cp gaze.js gazelib.js "$(PAPERWM_EXT)/"
	@grep -q '// tobii-gaze' "$(PAPERWM_EXT)/extension.js" || { \
	  sed -i "/} from '.\/imports.js';/a import * as Gaze from './gaze.js'; // tobii-gaze" "$(PAPERWM_EXT)/extension.js"; \
	  sed -i "/modules = \[/a\\        Gaze, // tobii-gaze" "$(PAPERWM_EXT)/extension.js"; \
	  echo "patched extension.js"; \
	}
	@echo "Installed. Log out and back in (Wayland) to reload PaperWM."

uninstall:
	-sed -i '/\/\/ tobii-gaze/d' "$(PAPERWM_EXT)/extension.js"
	-rm -f "$(PAPERWM_EXT)/gaze.js" "$(PAPERWM_EXT)/gazelib.js"
	@echo "Removed tobii gaze switcher. Log out and back in to reload PaperWM."

test:
	gjs -m tests/test_gazelib.js
```

- [ ] **Step 2: Install and verify the patch is idempotent**

Run: `make -C paperwm-gaze install && make -C paperwm-gaze install`
Expected: first run prints `patched extension.js` then `Installed...`; second run skips patching (no second `patched extension.js`). Confirm exactly one occurrence each:
Run: `grep -c '// tobii-gaze' "$HOME/.local/share/gnome-shell/extensions/paperwm@paperwm.github.com/extension.js"`
Expected: `2` (the import line + the modules entry).

- [ ] **Step 3: Reload PaperWM and verify enable + socket connect**

Log out and back in (Wayland cannot hot-reload changed extension files). Then:
Run: `journalctl --user -b 0 --no-pager | grep -i '#tobii-gaze' | tail -20`
Expected: a line `#tobii-gaze switcher enabled`. Now hold **Alt+Tab** and look around, release, and re-run the grep.
Expected: lines `#tobii-gaze x=0.xxx y=0.xxx` appearing only while Alt+Tab was held (spec risk #1 confirmed in the Shell). If you instead see repeated connect failures, ensure `tobiid` is running in gaze mode.

- [ ] **Step 4: Verify clean teardown**

Run: `gnome-extensions disable paperwm@paperwm.github.com && sleep 1 && journalctl --user -b 0 --no-pager | grep -i '#tobii-gaze' | tail -3`
Expected: a `#tobii-gaze switcher disabled` line and no errors/backtraces after it. Re-enable:
Run: `gnome-extensions enable paperwm@paperwm.github.com`

- [ ] **Step 5: Commit**

```bash
git add paperwm-gaze/Makefile
git commit -m "build(gaze-switch): idempotent in-place install target for PaperWM"
```

---

### Task 5: Map gaze to minimap tiles, log the window under gaze (no selection yet)

**Files:**
- Modify: `paperwm-gaze/gaze.js` (add `_tilesOf`, `_maybeSelect`; call from `_onGaze`)

**Interfaces:**
- Consumes: `gazeToPixel`, `hitTest` from `./gazelib.js`; `Navigator.navigating`, `Navigator.navigator`, `Tiling.spaces.selectedSpace`, the live `Minimap` (`extends Array` of columns of tile actors with `.meta_window`, `.x/.y/.width/.height`, and `.actor/.clip/.container`).
- Produces: `_tilesOf(minimap) → {x0,y0,w,h,id,window}[]`; `_maybeSelect()` (logs the hit window title at this stage).

This validates the minimap geometry + hit-test live (spec risk #2) before wiring `ensureViewport`.

- [ ] **Step 1: Add the imports**

In `paperwm-gaze/gaze.js`, change the gazelib import line:

```javascript
import { subscribeFrame, parseFrames, ema, gazeToPixel, hitTest, STREAM_GAZE } from './gazelib.js';
```

- [ ] **Step 2: Add `_tilesOf` and `_maybeSelect` (log-only) and call them**

In `paperwm-gaze/gaze.js`, replace the `_onGaze` method with:

```javascript
    _onGaze(g) {
        if (!g.valid) return;
        this._sx = ema(this._sx, g.x, EMA_ALPHA);
        this._sy = ema(this._sy, g.y, EMA_ALPHA);
        this._maybeSelect();
    }

    _tilesOf(minimap) {
        const baseX = minimap.actor.x + minimap.clip.x + minimap.container.x;
        const baseY = minimap.actor.y + minimap.clip.y + minimap.container.y;
        const tiles = [];
        for (const column of minimap) {
            for (const c of column) {
                if (!c.meta_window) continue;
                tiles.push({
                    x0: baseX + c.x, y0: baseY + c.y,
                    w: c.width, h: c.height,
                    id: c.meta_window.get_id(),
                    window: c.meta_window,
                });
            }
        }
        return tiles;
    }

    _maybeSelect() {
        if (!Navigator.navigating) { this._prevId = null; return; }
        const nav = Navigator.navigator;
        if (!nav) return;
        const space = Tiling.spaces.selectedSpace;
        const minimap = nav.minimaps?.get(space);
        if (!minimap || typeof minimap === 'number') return; // pending/absent
        const tiles = this._tilesOf(minimap);
        if (tiles.length === 0) return;
        const { px, py } = gazeToPixel(this._sx, this._sy, space.monitor);
        const id = hitTest(tiles, px, py, this._prevId, HYSTERESIS_PX);
        if (id === null || id === this._prevId) return;
        const t = tiles.find(t => t.id === id);
        // Task 6 replaces this log with the actual selection call.
        if (t) console.log(`#tobii-gaze over: ${t.window.title}`);
        this._prevId = id;
    }
```

- [ ] **Step 3: Reinstall and reload**

Run: `make -C paperwm-gaze install`
Then log out and back in.

- [ ] **Step 4: Verify the logged window tracks your gaze**

Hold **Alt+Tab** to open the ribbon, move your gaze across the minimap tiles, then release. Then:
Run: `journalctl --user -b 0 --no-pager | grep '#tobii-gaze over:' | tail -20`
Expected: `#tobii-gaze over: <window title>` lines whose titles match the tiles you looked at (left tiles → left apps, etc.). This confirms the global tile-rect math and hit-test are correct. If titles are consistently offset, note it — it will be addressed via the calibration knobs in Task 6, but typically the map is correct for a single monitor.

- [ ] **Step 5: Commit**

```bash
git add paperwm-gaze/gaze.js
git commit -m "feat(gaze-switch): map gaze to minimap tiles (log-only verification)"
```

---

### Task 6: Drive the selection via `ensureViewport`; final verification

**Files:**
- Modify: `paperwm-gaze/gaze.js` (replace the log in `_maybeSelect` with the selection call)

**Interfaces:**
- Consumes: `Tiling.ensureViewport(metaWindow, space, { moveto: false })`.
- Produces: working gaze selection; release-to-activate via unchanged PaperWM `_finish`.

- [ ] **Step 1: Replace the log with the selection call**

In `paperwm-gaze/gaze.js`, in `_maybeSelect`, replace the two lines:

```javascript
        const t = tiles.find(t => t.id === id);
        // Task 6 replaces this log with the actual selection call.
        if (t) console.log(`#tobii-gaze over: ${t.window.title}`);
        this._prevId = id;
```

with:

```javascript
        const t = tiles.find(t => t.id === id);
        if (t && t.window !== space.selectedWindow) {
            try {
                Tiling.ensureViewport(t.window, space, { moveto: false });
            } catch (e) {
                console.log(`#tobii-gaze ensureViewport failed: ${e}`);
            }
        }
        this._prevId = id;
```

- [ ] **Step 2: Reinstall and reload**

Run: `make -C paperwm-gaze install`
Then log out and back in.

- [ ] **Step 3: Verify end-to-end selection + activation**

Hold **Alt+Tab**; without moving the keyboard, move your gaze to a different app's minimap tile and watch the highlight follow your gaze. Release Alt.
Expected: the highlighted (gaze-selected) window is the one that gets focused/raised. Repeat for several tiles and directions.

- [ ] **Step 4: Verify no regressions to normal navigation**

With **Alt+Tab**, press Tab/Shift+Tab (and your usual PaperWM nav keys) as before, ignoring gaze.
Expected: keyboard navigation still moves the selection normally; gaze and keys don't fight destructively (gaze updates selection on look, keys on press; last action wins). Also confirm the existing `tobii-gaze-keys` (Super+edges) still behaves as before.

- [ ] **Step 5: Verify teardown once more**

Run: `gnome-extensions disable paperwm@paperwm.github.com && sleep 1 && journalctl --user -b 0 --no-pager | grep -iE 'tobii-gaze|Gjs|paperwm' | tail -10`
Expected: `#tobii-gaze switcher disabled`, no async-read errors or backtraces afterward. Re-enable:
Run: `gnome-extensions enable paperwm@paperwm.github.com`

- [ ] **Step 6: Commit**

```bash
git add paperwm-gaze/gaze.js
git commit -m "feat(gaze-switch): select minimap tile under gaze during Alt+Tab"
```

---

### Task 7: README + tuning knobs documentation

**Files:**
- Create: `paperwm-gaze/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: install/uninstall/test instructions and the meaning of `EMA_ALPHA`, `HYSTERESIS_PX`, `RECONNECT_MS`.

- [ ] **Step 1: Write the README**

Create `paperwm-gaze/README.md`:

```markdown
# tobii gaze switcher (PaperWM)

Pick the app on the PaperWM Alt+Tab minimap ribbon with your gaze: hold
Alt+Tab, look at a tile to select it, release Alt to activate it (PaperWM
activates the selected window natively).

## Requirements
- GNOME Shell on Wayland with PaperWM enabled.
- `tobiid` running in **gaze** mode (same as `tobii-gaze-keys`).
- `gjs` (for the tests).

## Install
    make -C paperwm-gaze install
    # then log out and back in (Wayland can't hot-reload extension files)

Re-run after every PaperWM upgrade — upgrades overwrite extension files. The
target is idempotent (marker-guarded) and copies `gaze.js` + `gazelib.js` into
the extension and patches `extension.js`.

## Uninstall
    make -C paperwm-gaze uninstall
    # then log out and back in

## Test
    gjs -m paperwm-gaze/tests/test_gazelib.js     # pure logic unit tests
    gjs -m paperwm-gaze/tests/probe_socket.js     # live socket probe (needs tobiid)

## Tuning (constants at the top of gaze.js)
- `EMA_ALPHA` (0..1): gaze smoothing. Higher = snappier but jumpier. Default 0.4.
- `HYSTERESIS_PX`: border deadband before switching tiles; raise if selection
  flickers between adjacent tiles. Default 24.
- `RECONNECT_MS`: backoff between reconnect attempts to tobiid. Default 1000.

## Troubleshooting
- Watch logs: `journalctl --user -b 0 -f | grep '#tobii-gaze'`.
- No gaze lines while holding Alt+Tab → tobiid not in gaze mode, or busy in
  head/camera mode (mutually exclusive at the device).
```

- [ ] **Step 2: Commit**

```bash
git add paperwm-gaze/README.md
git commit -m "docs(gaze-switch): README with install, test, and tuning notes"
```

---

## Self-Review

**Spec coverage:**
- Goal (gaze selects ribbon tile, release activates) → Tasks 5–6.
- Approach B / in-place GJS module → Tasks 3–4.
- tobiid IPC protocol decode → Task 1; verified live → Task 2, Task 4.
- Minimap geometry / global tile rects → Task 5 (`_tilesOf`).
- `ensureViewport(moveto:false)` selection, activation untouched → Task 6.
- Persistent async socket + reconnect + Cancellable teardown → Task 3, verified Task 4/6.
- Calibration knobs (EMA, scale via mapping, hysteresis) → Tasks 1, 3, 7.
- Source layout + idempotent install + maintenance story → Task 4, Task 7.
- Spike-first risks (1 socket framing, 2 tile geometry, 3 ensureViewport) → Task 2, Task 5, Task 6 respectively.
- Unit-testable pure logic vs live integration → Task 1 (unit), Tasks 2/4/5/6 (live).
- Out-of-scope items (edge-pan, multimonitor, auto-cal) correctly absent.

**Placeholder scan:** No TBD/TODO-as-work; the two in-code comments ("Task 4/6 replaces this…") are intentional staged-verification markers, each replaced by a concrete code block in the named task.

**Type consistency:** `parseFrames` returns `{gaze, subscribedOk, rest}` — used consistently in probe and `_readChunk`. `hitTest(tiles, px, py, prevId, margin)` signature matches all call sites. Tile objects carry `{x0,y0,w,h,id}` everywhere (`_tilesOf` adds `window`; `hitTest` ignores extras). `ensureViewport(window, space, {moveto:false})` matches the PaperWM signature at `tiling.js:4355`. Staged-verification comments name the task that replaces them (Task 3 → Task 5, Task 5 → Task 6).
