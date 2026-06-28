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
