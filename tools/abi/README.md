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
- What the subscribes and unsubscribes of internal streams 3, 4, 5, 7 and 8
  answer, streams the DLL never serves for a tracker it drives itself. Each
  export passes an internal stream id in `edx` to one helper, 0x18015bbf0 to
  subscribe (`tobii_internal_stream_subscribe` in its log, `internal.cpp`)
  and 0x18015bee0 to unsubscribe: 3 low-frequency head
  rotation (0x18014b7ad, 0x18014b707), 4 low-frequency head position
  (0x18014b8ed, 0x18014b847), 5 multiple faces position (0x18014b66d,
  0x18014b5c7), 7 wearable limited image (0x18014b24d, 0x18014b1a7) and 8
  secondary camera image (0x180150fdd, 0x180150f37). The subscribe helper
  answers `TOBII_ERROR_INVALID_PARAMETER` for a null device (unlogged) or
  callback, `TOBII_ERROR_CALLBACK_IN_PROGRESS` inside a callback, then
  `TOBII_ERROR_NOT_SUPPORTED` (0x18015bd29) when its support check
  0x180157860 fails; the unsubscribe helper checks for a null device, the
  callback flag and then the same support check (0x18015bf2d) before it
  looks for a subscription, so it never answers `NOT_SUBSCRIBED` for them.
  The check has two paths. When the DLL runs its own tracker module
  (`[device+0x4e8]`, set for any URL but `tobii-prp://` and `tprp-tcp://`,
  0x180157d47..0x180157fee), only ids 0, 1, 2 and 6 have a case, each asking
  the module's TTP stream list for its stream (2, 3, 7 and 0xb through
  0x180170260); 3, 4, 5, 7 and 8 fall through to `false` (0x1801578c0), and
  so does every id above 8. Otherwise the id goes through the map at
  0x180153b60 (2 to 1, 3 to 9, 4 to 8, 5 to 10, 6 to 7, 7 to 11, 8 to 0x17; 0
  and 1 to 0, never supported) to the service's stream list (`+0x8854`,
  count `+0x88bc`), never captured for an ET5. The map's targets are
  `PRP_STREAM_ENUM` values, named by the to-string switch at 0x1800232a3
  (table at RVA 0x237d8): 1 `CUSTOM`, 7 `IMAGE_COLLECTION`, 8
  `LOW_FREQUENCY_HEAD_POSITION`, 9 `LOW_FREQUENCY_HEAD_ROTATION`, 10
  `MULTIPLE_FACES_POSITION`, 11 `WEARABLE_LIMITED_IMAGE`, 0x17
  `SECONDARY_CAMERA_IMAGE`. `tobii_internal_stream_supported` calls the same
  check (0x18014cc59), so it reports 3, 4, 5, 7 and 8 unsupported on that
  path, as libtobii does. Below the check, the legacy TTP module
  (`platmod_legacy_ttp.cpp`) refuses them again: its unsubscribe and
  subscribe for each (rotation 0x180163150 and 0x1801631e0, position
  0x180163270 and 0x180163300, multiple faces 0x18015feb0 and 0x18015ff40,
  wearable limited image 0x18015fc70 and 0x18015fd00, secondary camera image
  0x18015fa30 and 0x18015fac0) log `PLATMOD_ERROR_NOT_SUPPORTED`
  (0x180222bd8, format 0x180220b60) and return 3 whatever they are given.
  Behind the service the DLL calls a low-frequency head callback with a
  24-byte record (dispatch at 0x180154bdc for PRP stream 8 and 0x180154c5d
  for 9, again at 0x180156d70 and 0x180156cc0; callback slots `+0x15de8` and
  `+0x15e00`, from the registration at 0x1801536d0): an `int64` at +0, the
  service's package timestamp plus `[device+0x15bd0]`, which the DLL only
  ever sets to 0 (0x180153c7c); an `int32` validity at +8, the package's
  flag tested for non-zero; and three `float`s at +0xc, copied unchanged.
  The platform side's `low_frequency_head_position_callback` (0x1800279f0)
  and `_rotation_callback` (0x180027c50) pack what a tracker module hands
  them unchanged too (the timestamp, validity 1 as the flag, the three
  words at 0x180027acd..0x180027adc), so the units, frame and rate are the
  service's, and nothing in the DLL produces or decimates the values.
  libtobii answers as the DLL does for a tracker it drives itself and does
  not re-publish its head pose.
- Which notifications the DLL delivers for an ET5, and which libtobii
  sends. On the DLL's own tracker module (`[device+0x4e8]`, above),
  `tobii_notifications_subscribe` (0x180151440) stores the application's
  callback twice: in the platmod's generic slot `+0xeb90` (0x18015175a; a
  second subscribe is `TOBII_ERROR_ALREADY_SUBSCRIBED`, 0x1801516d4) and,
  once it has subscribed the tracker's PRP properties but those in mask
  0x66800 (0x18015179f), at `[device+0x15bd8]` (0x1801518f8). A tracker
  notification is classified by its id (0x18017b7ad..0x18017babf), built
  into a 520-byte record (0x180188100, table at RVA 0x1883c8) and handed to
  0x18016efa0, which updates the platmod's caches and queues the record in
  a 16-entry ring at `+0x32948`. `tobii_device_process_callbacks` drains
  the ring (0x180143b30 → 0x1801593f0 → 0x180171400, which calls the TTP
  dispatcher 0x18016c590 at 0x180172285; table at RVA 0x16ca2c), and
  `tobii_device_clear_callback_buffers` flushes it (0x180143a8d →
  0x180158a20, 0x18016e720 at 0x180158a47) before it runs the same drain
  (0x180158ab2). The TTP dispatcher calls the generic slot itself for
  types 2, 4, 5, 9, 10, 11 and 12, and the platmod's property slots for
  the others, whose events reach the application through the PRP
  dispatcher 0x180153de0 (table at RVA 0x15415c, by `PRP_PROPERTY` id).
  That one has notification cases only for properties 1, 2, 4, 5, 6 and 7
  and other callbacks for 11, 13, 14, 17 and 18; every other id (0, 3
  `REMOTE_WAKE_ACTIVE`, 8 `DEVICE_NAME`, 9, 10 `LENS_CONFIGURATION`, 12,
  15, 16 `ENABLED_EYE`, 19 and 20) goes to its exit at 0x180153f09. The
  TTP dispatcher compares no value with the last one; the hop from a
  property slot to the PRP dispatcher (the transport around 0x180043df0)
  is not traced. By tracker id:
  - 1040/1050: type 0 `CALIBRATION_STATE_CHANGED`, true/false (slot
    `+0xea40`, property 7 at 0x180153f58);
  - 1271/1272: type 1 `EXCLUSIVE_MODE_STATE_CHANGED`, true/false
    (0x18018817f, 0x18018818c; slot `+0xea70`, property 5 at 0x180153f2e);
  - 1410: type 2 `TRACK_BOX_CHANGED`, no value (0x18016c70e);
  - 1450: type 3 `DISPLAY_AREA_CHANGED` (slot `+0xea30`, property 1 at
    0x180153e2d);
  - 1640 and 1680: type 4 `FRAMERATE_CHANGED`, a float (0x180188271,
    0x180188240; 0x18016c7fd);
  - 3020/3030: type 5 `POWER_SAVE_STATE_CHANGED`, true/false (0x18016c84d,
    0x18016c897);
  - 3110: type 6 `DEVICE_PAUSED_STATE_CHANGED` from a `u32` 0 or 1,
    anything above 1 dropped (0x1801882b4; slot `+0xea20`, property 4 at
    0x180153f21);
  - 3220: type 8 `CALIBRATION_ID_CHANGED` (0x1801882ee; slot `+0xea50`,
    property 6 at 0x180153f3c);
  - 3180: type 9 `COMBINED_GAZE_EYE_SELECTION_CHANGED`, 1 left, 2 right,
    anything else both (0x18018830b, 0x18016c933);
  - 3200 and 3210: types 10 `FAULTS_CHANGED` and 11 `WARNINGS_CHANGED`,
    the new list as a string of at most 511 bytes (0x180188339,
    0x180188343; 0x18016c9a3, 0x18016c9b7), from one string parameter
    (schema 0x1801810d2, 0x1801810bd) whose reader takes TLV type 0x14
    only (0x180003be3);
  - 3330: type 12 `FACE_TYPE_CHANGED`, a string (0x18016c9cb).

  Nothing produces type 7 `CALIBRATION_ENABLED_EYE_CHANGED`. 3200 and 3210
  also replace the text of the fault or warning cache (0x18016f0e9,
  0x18016f108) but not its "present" flag, which only a 1490 that reported
  the list sets (0x18016dce1, 0x18016dd8b) and `tobii_get_state_string`
  checks (0x1801426fd, 0x180142563). 1271 also queues presence status 0,
  `UNKNOWN` (0x18016f005..0x18016f034), which reaches the platmod's
  presence slot `+0xeaf0` when it differs from the last
  (0x1801720d8..0x1801720f2); the application's presence callback hangs
  off PRP property 11 (0x180153f63), past the same untraced hop. The
  platmod's subscribes for the six properties the PRP dispatcher turns
  into notifications each call their slot once, straight away, with a
  constant rather than the tracker's state: power save with 0 (0x180165012;
  nothing else calls that slot), exclusive mode with 0 (0x180164a52),
  display area with an all-zero area (0x18016601a..0x180166048),
  calibration active and calibration id with 0 (0x180165972, 0x1801656e2)
  and device paused with 1 (0x18016752f). So a new subscriber may get
  initial notifications; whether they reach it is not traced either.

  What the ET5 sends: the Windows captures hold 3180 (`u32 3`, once per
  init, right after the tracker's answer to the init's 3160), 3220 (five in
  the calibration) and 1450 (one in the display change), and nothing else.
  On Linux it also sends, without a body and early in an open's init, 1271
  when the open starts the sensor (a cold engine start, or a re-open after
  the stream was lost; such an open has never armed) or 1272 when it finds
  it running (the re-open that primes the stream about 10 s later, or an
  engine start while the sensor still runs, which arms without a prime);
  some opens bring neither, and opens after a USB reset have brought both.
  It sends 3110 just before its answer when a 3100 changes its state (a
  pause, or a resume of a paused tracker; seen 2026-09-24); an init's
  resume of a tracker that was not paused brings none (one of 34 init
  resumes in the Linux logs brought one, most likely for a tracker left
  paused). 1040, 1050, 1410, 1640, 1680, 3020, 3030, 3200, 3210 and 3330
  were never seen.

  libtobii sends six types. `CALIBRATION_STATE_CHANGED` and
  `DEVICE_PAUSED_STATE_CHANGED` come from the daemon itself, when a
  calibration session starts or ends and when the tracker accepts a pause
  or resume: the ET5 was never seen sending 1040/1050, not even in the
  Windows calibration capture, and the daemon only logs 3110.
  `DISPLAY_AREA_CHANGED` (1450), `CALIBRATION_ID_CHANGED` (3220),
  `FAULTS_CHANGED` (3200) and `WARNINGS_CHANGED` (3210) come from the
  tracker's, one for each, changed or not: as in the DLL for 3200 and
  3210, and for 1450 and 3220 as far as traced, since the property hop
  above is not. The rest are skipped:
  - exclusive mode: 1271/1272 are a Linux cold-start artifact that no
    Windows capture shows, and would turn it on whenever an open starts
    the sensor and off again with the prime about 10 s later.
    `TOBII_STATE_EXCLUSIVE_MODE` stays false, where the DLL answers it
    from status string 2 of the 1490 and from 1271/1272 (`+0xe60a`,
    0x1801648f3), and the presence side effect is not modelled either;
  - the track box, power save and face type, which the ET5 never sent and
    libtobii cannot change (`tobii_power_save_activate` / `_deactivate`
    and the face-type exports are `TOBII_ERROR_NOT_SUPPORTED`), nor the
    initial power-save value above;
  - the frame rate, fixed at 33 Hz;
  - the combined-gaze eye selection, always both. After
    `tobii_device_create` or `tobii_device_reconnect` the DLL most likely
    hands one `BOTH` to an application that subscribes before its next
    `tobii_device_process_callbacks`, unless it clears the callback buffers
    first: its create/reconnect routine flushes the ring at its start
    (0x18016e720, called at 0x18016db7c), before the init's 3160, so the
    3180 answering that 3160 stays queued until one of those two calls. The
    value is constant, and the daemon's init is not tied to a client;
  - type 7, which has no producer. There is no device-name notification
    type.
- How the DLL keeps the 4.1 docs' promise of "full thread safety across
  all API functions" (and why libtobii locks a device as it does). It
  imports critical sections and no SRW locks; its wrappers are create
  0x18003b1f0, lock 0x18003b2b0 (`EnterCriticalSection`), try-lock
  0x18003b2c0 (`TryEnterCriticalSection`), unlock 0x18003b2e0 and delete
  0x18003b270. A critical section is reentrant on its own thread (Windows
  semantics, not read from the DLL). The callback flag is a TLS slot per API
  instance, so `TOBII_ERROR_CALLBACK_IN_PROGRESS` refuses only the
  callback's own thread: `tobii_api_create` allocates it (`TlsAlloc` at
  0x180144be1, kept at api+0x130), `tobii_api_destroy` frees it (`TlsFree`
  at 0x180144a00) and takes no lock, and the device entry points check it
  before they take any lock, right after their first argument checks. Only
  a null device has to come before it, as the slot index is read through
  the device (`mov rax,[rcx]`, then `mov ecx,[rax+0x130]` in process).
  Process tests its handle at 0x180143b39, then the flag at 0x180143b55,
  and likewise the subscribe helper (a null device at 0x18015cc1b, a null
  callback at 0x18015cc20, then the flag at 0x18015cc9c), wait (its count,
  array and null handles at 0x180143f00..0x180143f24 and whether they share
  an API instance at 0x180143f45, `TOBII_ERROR_CONFLICTING_API_INSTANCES`
  if not, then the flag at 0x180143f58), reconnect (0x180143829, then
  0x180143845) and destroy (0x1801440e9, then 0x180144100).
  Device create makes three critical sections, and the platform module
  three more (+0x4620, +0x4628, +0x4630: 0x18000e580, 0x18000e5ce,
  0x18000e61c):
  - dev+0x4e0, an API mutex held for a call's whole tracker round trip:
    subscribe 0x18015ce3d..0x18015ceaf, device info 0x180142ee1, track box
    0x180142cd1, timesync 0x180145c08, calibration start 0x180149a60,
    reconnect 0x180143873..0x180143a0f, clear 0x180143a85. Process, wait
    and destroy never take it. The calls that take it generally log under
    it, their error lines included: for example subscribe (the error helper
    0x1800010f0 at 0x18015cea0), device info (0x18014327b), track box
    (0x180142d5a, 0x180142dd1), reconnect (0x180143943, 0x1801439fe) and
    `tobii_calibration_retrieve` (0x180147c57); retrieve also calls its
    receiver under it with the callback flag set (0x180147bbc..0x180147c79).
  - dev+0x4d8, held around every user callback (streams 0x1801546a0 to
    0x180155091, notifications 0x180153ed7 to 0x180153f04), by an
    unsubscribe (0x180153580) and by the subscribe worker across the
    platform module's subscribe (0x18015371b..0x180153845), which stores
    the callback only once that has succeeded (0x1801537cc..0x1801537de).
    So an unsubscribe on another thread waits for a callback that is
    running, and every callback of the device waits out a subscribe's
    round trip.
  - dev+0x9818, around the device's notification queue.

  `tobii_device_process_callbacks` (0x180143b30 → 0x1801593f0) sets the
  flag (0x1801594ff), swaps the notification queue out under dev+0x9818 and
  delivers it (0x180159515..0x180159566), and only then try-enters the
  platform module's +0x4628 (0x18000e9d9). When another thread holds that,
  it returns 0 (0x18000e9ea), which the export's jump table turns into
  `TOBII_ERROR_NO_ERROR` (0x180143eb8, case 0 at 0x180143c06), after a loss
  too. So two threads can each deliver notifications of one device, one
  after the other under dev+0x4d8, while one processes the rest.
  `tobii_wait_for_callbacks` (0x1801596e0) returns at once if a device has
  data queued (0x180157030, called at 0x18015972a); otherwise it try-enters
  each device's +0x4628 (0x18000e940) and holds it through the whole wait,
  100 ms at most (0x180159a54; released at 0x180159b17). A device another
  thread holds gets no wait handle, and with no handle at all the wait
  returns 0, `TOBII_ERROR_NO_ERROR`, at once (read from the control flow,
  not observed). `tobii_device_reconnect` enters +0x4620, +0x4628 and
  +0x4630 with no timeout (0x18000eb5f, 0x18000eb88, 0x18000ebb1), after
  dev+0x4e0, so it waits for a process or a wait on another thread, and
  holds them through the platform module's reconnect (let go at
  0x18000ed63..0x18000ed7f).
  `tobii_device_clear_callback_buffers` swaps the callback table out under
  dev+0x4d8 and runs the internal process (0x180158a20, at 0x180158ab2),
  whose try-enter fails while another thread processes, so only the
  notifications are dropped then. `tobii_device_destroy` (0x1801440e0 →
  0x18015a320) takes no lock and deletes the device's three critical
  sections (0x18015a395, 0x18015a3a6, 0x18015a3b7), so another thread's
  call on the device meanwhile is undefined, as the docs say ("Make sure
  that no background thread is using the device"). `tobii_system_clock`
  (0x1801432d0) reads the clock and checks no flag.

  libtobii splits a device the same way, with a lock per concern (`Device`
  in `crates/tobii-ffi/src/device.rs`: `command`, a first-come,
  first-served ticket lock, for dev+0x4e0, and std `Mutex`es, `dispatch`
  for +0x4628 and dev+0x9818 and `callbacks` for dev+0x4d8), and
  keeps the DLL's rule for destroy. A reconnect holds `dispatch` for its
  round trip, as the DLL's holds +0x4628 through the platform module's
  reconnect: from before its new connection asks for the streams until it
  has swapped connections or failed, so that nothing tobiid sends both
  connections is delivered twice. Where it differs, it does on purpose:
  the command lock lets its waiters in in the order they asked, where a
  critical section promises no order (Windows semantics, not read from the
  DLL), so requests made back to back on one thread cannot keep another
  thread's subscribe out, as they did for up to 5 s behind a std `Mutex`
  on Linux; a wait holds no lock while it sleeps, and waits on a device
  another thread holds rather than skip it; a subscribe lets the callbacks
  lock go for its round trip; a busy process delivers nothing, and answers
  `TOBII_ERROR_CONNECTION_FAILED` once a loss has been reported; a clear
  waits for another thread's process, never for a request (both wait for
  a reconnect, the DLL's under dev+0x4e0); the callback flag is one per thread, for every API instance, so a
  callback may call into no device at all; `tobii_calibration_retrieve`'s
  receiver runs under that flag, as the DLL's does, but under no lock,
  where the DLL's runs under dev+0x4e0; and the logger never runs under a
  lock of the call that logs. libtobii's locks are not reentrant, which that
  flag makes safe: it refuses a callback's call before any lock is taken.

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
does for an ET5. The 10 subscribes and unsubscribes of internal streams 3,
4, 5, 7 and 8 answer the same way, as the DLL does for a tracker it drives
itself; they only compare their device and callback with null, so the
callback's `void const*` cannot matter. The other 50 return
`TOBII_ERROR_NOT_SUPPORTED` without reading their arguments, so their
best-effort parameter types cannot matter at runtime.
