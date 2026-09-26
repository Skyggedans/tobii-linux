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
  `tobii_error_message` (0x180144cb0, table 0x180144df4), and so do the texts
  libtobii returns, verbatim. Each case is a `lea rax,[rip+d]` to a `.rdata`
  string in 0x180220b90..0x180220f30; 2 and 19 share one case (0x180144dac),
  "Insufficient permissions when using a restricted feature.". The compare at
  0x180144cb4 is unsigned, so any other value, a negative one included, takes
  the default case: `snprintf` of "Undefined error (0x%x). Please contact
  support." (0x180220f60) into one process-global 64-byte buffer (0x18024ee60,
  in `.bss`), which is returned. Nothing in the DLL calls the function, and
  only it touches the buffer; the DLL's own log lines name an error from a
  separate table of `TOBII_ERROR_*` names (file offset 0x21f070) or
  "Undefined tobii error (0x%x).". To re-derive the texts,
  `objdump -s --start-address=0x180144df4 --stop-address=0x180144e48` dumps
  the table: 21 little-endian 32-bit RVAs in code order (`d44c1400` is
  0x144cd4, so case 0 is at 0x180144cd4 once the image base 0x180000000 is
  added). `objdump -d -M intel --start-address=0x180144cb0
  --stop-address=0x180144df4` prints each case's `lea` with its target as a
  comment (`# 0x180220b90`), and `strings -a -t x` finds that string at
  file offset target minus the image base minus 0x1800 (`.rdata` is at RVA
  0x1a8000, file offset 0x1a6800): 0x21f390..0x21f730, and the format at
  0x21f760.
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
  presence notification at 0x180153f6e), log level (below), validity (the
  real DLL's Windows session output: 0 with position (-1,-1), 1 otherwise).
  Log level: the DLL's own lines go through one helper, 0x18015e360. Its 810
  direct call sites, plus 120 through two error-name helpers (0x1800010f0
  and 0x180157440, which call it at 0x180001266 and 0x18015754a with the
  level they are given, always 0), pass ERROR, 0, but for three at INFO, 2:
  "Connected to platform module" at 0x180153c19 and a firmware upgrade in
  progress at 0x18014467e and 0x1801588e3. That helper is not the only path:
  thunks forward the lines of the DLL's sub-libraries to `log_func`
  directly, at DEBUG, 3, and TRACE, 4, from enumeration (0x18015b070), at
  their own level, 0..4 unchanged, from the legacy TTP layer (0x1801706c0),
  and at a level taken from the message (0x18015d930).
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
- What `tobii_calibration_stimulus_points_get` (0x180149d10) answers. It is
  an internal API (its source file, 0x180149d22, is
  `src\api\tobii_internal.cpp`) that only reads a property: it calls the
  DLL's `tobii_property_get` (0x18015d300, `core\internal.cpp`) with PRP
  property 0x13 and a copier (0x180001800). 0x13 is
  `PRP_PROPERTY_ENUM_CALIBRATION_STIMULUS_POINTS` by the DLL's id-to-name
  switch (0x18002e8b0, table 0x18002ea00; case 19 is at 0x18002e9b8). Only
  the switch names an id: the names in `.rdata` are not in enum order (the
  switch gives 13 `FACE_ID_STATE` and 14 `FACE_ID_PARAMETERS`, the ids
  `tobii_get_face_id_state` and `tobii_get_face_id_parameters` pass at
  0x18014a639 and 0x18014a589, and the strings stand the other way round).
  `tobii_property_get` answers `TOBII_ERROR_INVALID_PARAMETER` for a null
  device (unlogged) or output, then `TOBII_ERROR_CALLBACK_IN_PROGRESS` inside
  a callback, then `TOBII_ERROR_NOT_SUPPORTED` (0x18015d450), before any
  request, when the device's property list (host device +0x8704, count
  +0x8754) lacks the id; nothing checks for a calibration session. That list
  is what the tracker module reported at connect. The legacy TTP
  `platmod_start` fills its readable list with module ids 9, 0, 2, 3, 5, 6,
  7, 8, 0xe, 4 and 0xb only (0x18016b536..0x18016b8d2); the server maps them
  through the table at 0x18001f9b4 to PRP ids 9, 1, 4, 8, 7, 6, 3, 2, 5, 0xa
  and 0x10 (module id 17 would be 0x13, 0x18001f288) and adds field of use,
  0x11 (0x18001f2bc). So on the DLL's own path an ET5 always gets
  `TOBII_ERROR_NOT_SUPPORTED`, which libtobii answers too; behind Tobii's
  service the answer is the service's, never captured, and no capture has a
  tracker command for the points. The copier writes an `int` count and that
  many 36-byte records, nine 32-bit words each, with no bound; the PRP body is
  a fixed 0x488 bytes (serialiser 0x180042517, deserialiser 0x18004563e): a
  size word, the count and 32 records. Nothing in the DLL reads a record's
  fields. Other exports reach `tobii_property_get` with PRP ids that mapped
  list does not hold either: 0xc (`tobii_hardware_configuration_get`,
  0x18014ab79), 0xd and 0xe (the face id state and parameters, above), 0xf
  (`tobii_get_combined_gaze_hid_track_box`, 0x180149fd2) and 0x12
  (`tobii_get_display_id`, 0x1801467b9); their module ids would be 10, 12, 13,
  1 and 16.

Layouts, and where each comes from:

| Type | Size | Source |
|---|---|---|
| `tobii_device_info_t` | 2048 | DLL (strncpy sizes and offsets in `tobii_get_device_info`) |
| `tobii_notification_t` | 520, align 4 | DLL (notification dispatcher: 512-byte string union at +8) |
| `tobii_device_name_t` | `char[64]` | DLL (`tobii_get_device_name` writes 0x40 bytes) |
| `tobii_state_string_t` | `char[512]` | DLL |
| `tobii_version_t` | 16 | DLL (`tobii_get_api_version` writes {4,1,0,3}) |
| `tobii_custom_alloc_t` | 24 | DLL (`tobii_api_create` checks +8 and +0x10 at 0x180144ac7..0x180144ad2 and copies 24 bytes at 0x180144b36) |
| `tobii_custom_log_t` | 16 | DLL (`tobii_api_create` checks +8 at 0x180144ae2 and copies 16 bytes at 0x180144b78) |
| `tobii_gaze_data_t` | 168 (8+8+76+76) | 4.1 docs field list |
| `tobii_geometry_mounting_t` | 36 | 4.1 docs + the device's 2110 response shape |
| `tobii_calibration_point_data_t` | 32 | SDK header; DLL (`tobii_calibration_parse` fills +8/+0xc/+0x14/+0x18 at 0x1801478ee..0x180147972) |
| `tobii_timesync_data_t` | 24 | DLL (`tobii_timesync` writes three qwords at +0/+8/+16); which field is which, and the names, inferred |
| `tobii_stream_type_t` | 136 | DLL (offsets 0/4/8/72 in `tobii_enumerate_stream_types`); size inferred, field names ours |
| `tobii_hardware_configuration_t` | 2472, align 8 | DLL (its copy at 0x18014abe0 and the PRP deserialiser at 0x180045056); **provisional**: the values decoded from one Windows 2120 answer, field names and units ours |
| `tobii_calibration_stimulus_points_t` | 1156, align 4 | DLL (the copier at 0x180001800, the PRP serialiser at 0x180042517 and deserialiser at 0x18004563e, the property cache copy at 0x180031c44); record contents unknown, names ours |
| everything else in `tobii_streams.h` / `tobii_wearable.h` | — | 4.1 docs, unchanged since 1.x |

Undocumented exports are declared in `tobii_internal.h` with the arity the DLL
shows. The 12 implemented ones (field of use, IR image, internal-stream
and internal-capability support, timesync, the stream catalogue, pause and
resume, hardware configuration) have real types, and so does
`tobii_calibration_stimulus_points_get`, which checks its arguments and then
answers `TOBII_ERROR_NOT_SUPPORTED` as the DLL's in-process tracker module
does for an ET5; the other 60 return `TOBII_ERROR_NOT_SUPPORTED` without
reading their arguments, so their best-effort parameter types cannot matter
at runtime.
