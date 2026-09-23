# tools/abi — recovering the Stream Engine ABI

`libtobii.so` exports every entry point of Tobii's `tobii_stream_engine.dll`
**4.1.0.3** (the version resource of the reference DLL). No 4.x header was ever
published outside Tobii's registration wall, so the ABI is reconstructed from
three sources, in order of authority:

1. **The DLL itself** — `dll_abi.py` reads its export table and derives, per
   export, which argument registers and stack slots are read, at what width,
   and which constant status codes are returned.
2. **The 4.1.0 reference documentation** (formerly
   `tobiitech.github.io/stream-engine-docs`, now offline; the Wayback snapshot
   `web.archive.org/web/20241017170526id_/https://tobiitech.github.io/stream-engine-docs/`
   is complete). It documents 80 of the 153 exports and their types.
3. **The Stream Engine 1.2.1 / 2.2.2 public headers** for types the 4.1 docs
   only describe in prose.

None of these is vendored. Put the DLL anywhere and pass `--dll` (or set
`TOBII_DLL`); scratch output goes to `fixtures/abi/`, which is gitignored.

## Usage

    tools/abi/dll_abi.py exports --check crates/tobii-ffi/abi-symbols.txt
    tools/abi/dll_abi.py args --out fixtures/abi/args-4.1.0.3.tsv
    tools/abi/dll_abi.py headers --args fixtures/abi/args-4.1.0.3.tsv
    tools/abi/dll_abi.py dis tobii_calibration_collect_data_2d
    tools/abi/dll_abi.py callsites 0xeb70

`args` reports the *first read* of each argument location before anything
writes it, walking linearly from the entry and stopping at an exit no earlier
branch jumps past. It follows thunks, recognises MSVC's hot-patch `rex push`
and `__chkstk` large-frame probes, and restores the frame after each early
epilogue. An argument the callee never reads is missed, so the result is a
lower bound on arity; `headers` fails only when the DLL reads *more* than a
prototype declares or a float/integer position disagrees.

## What it established

- 153 exports: 80 documented in 4.1.0, 73 undocumented.
- `tobii_device_create(api, url, field_of_use, device)` — four arguments,
  `field_of_use` read as 32 bits and validated to 1..2. A caller compiled
  against the three-argument pre-4.0 header lands its `device` pointer there.
- `tobii_device_create_ex` takes seven (stack slots 4..6);
  `tobii_calculate_display_area_basic` six (floats in xmm1..3, pointers on the
  stack); `tobii_calibration_collect_data_per_eye_2d` five;
  `tobii_send_custom_command` six.
- `tobii_error_t` numbering 0..20 comes from the jump table in
  `tobii_error_message`.

Layouts, and where each comes from:

| Type | Size | Source |
|---|---|---|
| `tobii_device_info_t` | 2048 | DLL (strncpy sizes and offsets in `tobii_get_device_info`) |
| `tobii_notification_t` | 520, align 4 | DLL (notification dispatcher: 512-byte string union at +8) |
| `tobii_device_name_t` | `char[64]` | DLL (`tobii_get_device_name` writes 0x40 bytes) |
| `tobii_state_string_t` | `char[512]` | DLL |
| `tobii_version_t` | 16 | DLL (`tobii_get_api_version` writes {4,1,0,3}) |
| `tobii_gaze_data_t` | 168 (8+8+76+76) | 4.1 docs field list |
| `tobii_geometry_mounting_t` | 36 | 4.1 docs + the device's 2110 response shape |
| `tobii_calibration_point_data_t` | 32 | 4.1 docs |
| everything else in `tobii_streams.h` / `tobii_wearable.h` | — | 4.1 docs, unchanged since 1.x |

Undocumented exports are declared in `tobii_internal.h` with the arity the DLL
shows and best-effort types; all but a handful return
`TOBII_ERROR_NOT_SUPPORTED` without reading their arguments, so the exact
parameter types cannot matter at runtime.
