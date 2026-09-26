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
- `tobii_stream_t` numbering 0..11 comes from the DLL; the 4.1 docs still list
  the pre-4.0 names, among them `WEARABLE` and `CUSTOM`, which have no value
  in the DLL. `tobii_stream_supported` (0x180141820) passes the caller's value
  unchanged to helper 0x1801591b0, which the ten stream subscribes other than
  user presence and digital syncport call with their own number. The helper's
  map at 0x180153a50 sends 0..2, 4, 5, 7 and 11 to tracker streams named in
  the DLL's `PRP_STREAM_ENUM` table: 0..2 `GAZE_POINT`, `GAZE_ORIGIN`,
  `EYE_POSITION_NORMALIZED`, 4 `HEADPOSE`, 5 `ADVANCED_GAZE` (gaze data),
  7 `DIAGNOSTICS_IMAGE`, 11 `WEARABLE_FOVEATED`; the rest go to `INVALID`.
  8..10 are checked against the compound streams `USER_POSITION_GUIDE_XYZ`,
  `WEARABLE_CONSUMER` and `WEARABLE_ADVANCED` (from 0x1801592f4), and 9 and
  10 against `WEARABLE` as well. 3 and 6 are identified by the checks the
  helper shares with the user presence and digital syncport subscribes
  (property 0xb `USER_PRESENCE`, `[+0xa84c] != 2`). A value of 12 or more is
  not supported, without an error; a negative one is
  `TOBII_ERROR_INVALID_PARAMETER`.
- The other public enums agree with the DLL wherever it shows them. Matching:
  capability 0..18 (range check at 0x180141b81), state 0..7 (state-to-property
  map at 0x180142ba0), notification type and value type (dispatchers at
  0x180153de0 and 0x18016c590), field of use (to-string at 0x1800314d0),
  licence validation results 4..9 (`licensekey_validate_license`), lens
  configuration writable (`tobii_lens_configuration_writable` writes 1 at
  0x1801522a9 when the tracker lists property 0xa `LENS_CONFIGURATION`),
  state bool and supported (`setne` gives 1 for true at 0x180153eaf and
  0x180141972).
  Consistent, with the names only from the docs: feature group (the mapping
  at 0x18015026e in `tobii_get_feature_group`), enabled eye
  (`tobii_set_enabled_eye` at 0x180149c40), user presence status (the
  presence notification at 0x180153f6e), log level (error paths pass 0, and
  "Connected to platform module" passes 2, INFO, at 0x180153c19), validity
  (the real DLL's Windows session output: 0 with position (-1,-1), 1
  otherwise).
  Calibration point status: values from the DLL, names from the SDK header
  (the 4.1 docs never define it). `tobii_calibration_parse` maps each eye's
  status word in the blob, reading its low 32 bits, at 0x180147910 (left) and
  0x180147950 (right): 1 gives 2 (`VALID_AND_USED_IN_CALIBRATION`), 0 gives 1
  (`VALID_BUT_NOT_USED_IN_CALIBRATION`), anything else 0
  (`FAILED_OR_INVALID`; it tests for -1 explicitly).
  No DLL evidence: wearable foveated tracking state, device generation.
- What `tobii_device_info_t` holds for a tracker the DLL drives over USB.
  `tobii_get_device_info` (0x180142e00) copies every field from the device's
  cached copy, which the connect callback (0x180153bf0) fills with one
  memcpy (0x180153c1e) on create and on reconnect. For a URL that is neither
  `tobii-prp://` nor `tprp-tcp://` the DLL runs its own tracker module
  in-process: `setup_device_info` (0x18016e250) takes serial, model,
  generation and firmware from command 1420 and the properties from 1330
  (`tracker_get_properties`, 0x1801a22a0), keeping property 0, the
  integration type (0x18016e33a; `Peripheral` on the ET5). `platmod_start`
  (0x18016ac30) passes on those five strings and properties 3 and 4, and no
  other device string; the host copies them into the connect message
  (0x18001ff1f; the serialiser writes the integration type at 0x180040425),
  so `integration_id`, `hw_calibration_version`, `hw_calibration_date` and
  `lot_id` arrive empty. `runtime_build_version` is the module's
  `Legacy TTP (4.1.0/3)` (sprintf at 0x18016aad0, copied at 0x180032a3e);
  libtobii names itself there instead. `tobii_get_device_info_internal`
  reads properties 3 and 4 (0x18014ffe2..0x18015003d). Not traced: only the
  transport between the in-process server and the host's deserialiser
  (0x180043df0).

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
| `tobii_calibration_point_data_t` | 32 | SDK header; DLL (`tobii_calibration_parse` fills +8/+0xc/+0x14/+0x18 at 0x1801478ee..0x180147972) |
| `tobii_timesync_data_t` | 24 | DLL (`tobii_timesync` writes three qwords at +0/+8/+16); which field is which, and the names, inferred |
| `tobii_stream_type_t` | 136 | DLL (offsets 0/4/8/72 in `tobii_enumerate_stream_types`); size inferred, field names ours |
| `tobii_hardware_configuration_t` | 2472, align 8 | DLL (its copy at 0x18014abe0 and the PRP deserialiser at 0x180045056); **provisional**: the values decoded from one Windows 2120 answer, field names and units ours |
| everything else in `tobii_streams.h` / `tobii_wearable.h` | — | 4.1 docs, unchanged since 1.x |

Undocumented exports are declared in `tobii_internal.h` with the arity the DLL
shows. The 11 implemented ones (field of use, IR image, internal-stream
support, timesync, the stream catalogue, pause and resume, hardware
configuration) have real types; the other 62 return
`TOBII_ERROR_NOT_SUPPORTED` without reading their arguments, so their
best-effort parameter types cannot matter at runtime.
