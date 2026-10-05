# tobii gaze switcher (PaperWM)

Pick the app on the PaperWM Alt+Tab minimap ribbon with your gaze: hold
Alt+Tab, look at a tile to select it, release Alt to activate it (PaperWM
activates the selected window natively).

## Requirements
- GNOME Shell on Wayland with PaperWM enabled.
- `tobiid` running, as for `tobii-gaze-keys`.
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
- No tile selected while holding Alt+Tab → check that tobiid runs and sends
  gaze: `gjs -m paperwm-gaze/tests/probe_socket.js` prints `subscribed ok=true`,
  then ten `gaze valid=…` lines.
