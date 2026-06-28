// tobii gaze switcher: while the PaperWM navigator is open (Alt+Tab held),
// pick the minimap tile under the user's gaze. Reads tobiid's gaze stream over
// a Unix socket (async Gio) and drives selection via Tiling.ensureViewport.
// Source of truth lives in the tobii repo (paperwm-gaze/); installed in place.

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

import { Tiling, Navigator } from './imports.js';
import { subscribeFrame, parseFrames, ema, gazeToPixel, hitTest, STREAM_GAZE } from './gazelib.js';

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
                this._reconnect();
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
                if (!arr || arr.length === 0) {
                    if (!this._cancellable.is_cancelled()) this._reconnect();
                    return;
                }
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
