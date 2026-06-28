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
