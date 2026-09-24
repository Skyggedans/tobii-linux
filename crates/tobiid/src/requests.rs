//! Client requests: what the device reported about itself (identity,
//! geometry, stream catalogue, hardware configuration), its display area,
//! states and clock.
//! Calibration requests are handed to [`crate::calibration`], the device
//! name to [`crate::name`], pause and resume to [`crate::pause`].
//!
//! Identity and geometry come from the facts collected during the engine's
//! init, so answering them costs no USB traffic; a request made while the
//! engine is still starting waits for them.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tobii_ipc::geometry::{DisplayArea, display_area_basic};
use tobii_ipc::request::{
    self, DeviceInfo, Request, Timesync, decode_display_area, encode_device_info,
    encode_display_area, encode_geometry_mounting, encode_hardware_configuration,
    encode_stream_types, encode_timesync, encode_track_box, encode_u32, kind, state, status,
};
use tobii_proto::facts::{DEFAULT_DISPLAY_ID, DeviceFacts, display_area_set_payload};
use tobii_proto::protocol::cmd;
use tracing::{info, warn};

use crate::daemon::{State, lock_state};

/// A reply: Stream Engine status code and kind-specific payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reply {
    pub(crate) status: u8,
    pub(crate) payload: Vec<u8>,
}

impl Reply {
    pub(crate) fn ok(payload: Vec<u8>) -> Self {
        Self {
            status: status::OK,
            payload,
        }
    }

    pub(crate) fn err(status: u8) -> Self {
        Self {
            status,
            payload: Vec::new(),
        }
    }
}

impl From<Result<Vec<u8>, u8>> for Reply {
    fn from(r: Result<Vec<u8>, u8>) -> Self {
        r.map_or_else(Self::err, Self::ok)
    }
}

/// How long a request waits for the device's init to report its facts.
const FACTS_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the device may take to acknowledge a display area.
const DISPLAY_AREA_TIMEOUT: Duration = Duration::from_secs(5);
/// Host-side latency bound between the device stamping a gaze frame and the
/// daemon reading it: the lower edge of the TIMESYNC window.
const FRAME_LATENCY_US: i64 = 30_000;
/// How long a TIMESYNC waits for a gaze frame newer than the request: a
/// cold engine takes as long as its facts.
#[cfg(not(test))]
const TIMESYNC_TIMEOUT: Duration = FACTS_TIMEOUT;
#[cfg(test)]
const TIMESYNC_TIMEOUT: Duration = Duration::from_millis(300);
/// How often a TIMESYNC looks for that frame (the tracker sends ~33 a second).
const TIMESYNC_POLL: Duration = Duration::from_millis(5);

/// Answer one request from `client`.
pub(crate) fn handle(state: &Mutex<State>, client: u64, req: &Request<'_>) -> Reply {
    match req.kind {
        kind::DEVICE_INFO => facts(state, client, |f| Some(encode_device_info(&device_info(f)))),
        kind::TRACK_BOX => facts(state, client, |f| {
            f.track_box.as_ref().map(encode_track_box)
        }),
        kind::DISPLAY_AREA_GET => facts(state, client, |f| {
            f.display_area.as_ref().map(encode_display_area)
        }),
        kind::GEOMETRY_MOUNTING => facts(state, client, |f| {
            f.mounting.as_ref().map(encode_geometry_mounting)
        }),
        // From the init's 1200, not a new one per call as the DLL sends: the
        // catalogue is the firmware's.
        kind::STREAM_TYPES => facts(state, client, |f| {
            (!f.streams.is_empty()).then(|| encode_stream_types(&f.streams))
        }),
        // Provisional, and NOT_SUPPORTED so far: the ET5 answers the init's
        // 2120 with an empty payload on Linux.
        kind::HARDWARE_CONFIGURATION => facts(state, client, |f| {
            f.hardware.as_ref().map(encode_hardware_configuration)
        }),
        kind::DISPLAY_AREA_SET => match decode_display_area(req.payload) {
            Some(area) if is_finite(&area) => set_display_area(state, client, area),
            _ => Reply::err(status::INVALID_PARAMETER),
        },
        kind::STATE => match request::decode_u32(req.payload) {
            Some(state::CALIBRATION_ID) => {
                let id = lock_state(state).calibration.id;
                id.map_or_else(
                    || Reply::err(status::NOT_AVAILABLE),
                    |id| Reply::ok(encode_u32(id)),
                )
            }
            Some(state::CALIBRATION_ACTIVE) => {
                let active = lock_state(state).calibration.is_active();
                Reply::ok(vec![u8::from(active)])
            }
            Some(state::DEVICE_PAUSED) => Reply::ok(vec![u8::from(lock_state(state).paused)]),
            Some(_) => Reply::err(status::NOT_SUPPORTED),
            None => Reply::err(status::INVALID_PARAMETER),
        },
        kind::TIMESYNC => timesync(state, client),
        kind::DEVICE_NAME_GET => crate::name::get(state, client),
        kind::DEVICE_NAME_SET => crate::name::set(state, req.payload),
        kind::DEVICE_PAUSE => crate::pause::handle(state, client, req.payload),
        k if (kind::CALIBRATION_START..=kind::CALIBRATION_CLEAR).contains(&k) => {
            crate::calibration::handle(state, client, k, req.payload)
        }
        _ => Reply::err(status::NOT_SUPPORTED),
    }
}

fn device_info(f: &DeviceFacts) -> DeviceInfo {
    f.info.clone()
}

fn is_finite(area: &DisplayArea) -> bool {
    [area.top_left_mm, area.top_right_mm, area.bottom_left_mm]
        .iter()
        .flatten()
        .all(|v| v.is_finite())
}

/// Wait for the device's facts (starting the engine if needed) and answer
/// from them; `NOT_SUPPORTED` when the device did not report that fact.
pub(crate) fn facts(
    state: &Mutex<State>,
    client: u64,
    answer: impl Fn(&DeviceFacts) -> Option<Vec<u8>>,
) -> Reply {
    let started = Instant::now();
    let _ = lock_state(state).device_for(client);
    loop {
        if let Some(facts) = lock_state(state).facts.clone() {
            return answer(&facts).map_or_else(|| Reply::err(status::NOT_SUPPORTED), Reply::ok);
        }
        if started.elapsed() > FACTS_TIMEOUT {
            return Reply::err(status::TIMED_OUT);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

/// A device/host clock pair from the first gaze frame seen after the
/// request, so that it is fresh and belongs to the running device session.
/// Frames are counted rather than compared with the wall clock, which may
/// step back meanwhile. Starts the engine if needed and, as the facts do,
/// keeps it up while `client` stays connected. `TIMED_OUT` if no such frame
/// arrives in time; `NOT_AVAILABLE` while the device is paused, since it
/// sends none.
fn timesync(state: &Mutex<State>, client: u64) -> Reply {
    let started = Instant::now();
    let asked = {
        let mut st = lock_state(state);
        let _ = st.device_for(client);
        st.gaze_frames
    };
    loop {
        // Copied out, so that no guard is held across the sleep.
        let (frames, clock, paused) = {
            let st = lock_state(state);
            (st.gaze_frames, st.clock, st.paused)
        };
        if paused {
            return Reply::err(status::NOT_AVAILABLE);
        }
        // A stop clears `clock` after the count moved: wait for the next one.
        if frames != asked
            && let Some((device_us, host_us)) = clock
        {
            return Reply::ok(encode_timesync(&Timesync {
                host_start_us: host_us - FRAME_LATENCY_US,
                device_us,
                host_end_us: host_us,
            }));
        }
        if started.elapsed() > TIMESYNC_TIMEOUT {
            return Reply::err(status::TIMED_OUT);
        }
        thread::sleep(TIMESYNC_POLL);
    }
}

/// Write a display area to the device and, once it accepts it, keep it
/// across re-inits and sessions. Refused while another client calibrates
/// (its calibration is being made on the current area). During the caller's
/// own session it is saved only with the calibration computed on it, and
/// what it replaces is remembered, so that a session ending without a
/// calibration can put it back (see [`crate::calibration`]).
fn set_display_area(state: &Mutex<State>, client: u64, area: DisplayArea) -> Reply {
    let (device, display_id) = {
        let mut st = lock_state(state);
        if let Err(code) = st.calibration.display_access(client) {
            return Reply::err(code);
        }
        // A client's area supersedes TOBII_DISPLAY_MM for this run: do not
        // let the variable's write, queued at the next device init, undo it.
        st.display_request = None;
        let before = crate::calibration::DisplayBefore {
            device: st.facts.as_ref().and_then(|f| f.display_area),
            configured: st.display_override,
        };
        st.calibration.note_display_before(client, before);
        let display_id = st
            .facts
            .as_ref()
            .and_then(|f| f.display_id)
            .unwrap_or(DEFAULT_DISPLAY_ID);
        (st.device_for(client), display_id)
    };
    let Some(device) = device else {
        return Reply::err(status::CONNECTION_FAILED);
    };
    let result = crate::device::run(
        device.as_ref(),
        cmd::DISPLAY_AREA_SET,
        display_area_set_payload(&area, display_id),
        DISPLAY_AREA_TIMEOUT,
    );
    if result.is_ok() {
        let file = {
            let mut st = lock_state(state);
            let save_now = st.calibration.note_display_set(client, area);
            st.display_override = Some(area);
            if let Some(engine) = st.engine.as_ref() {
                engine.set_display_area_override(Some(area));
            }
            if let Some(facts) = st.facts.as_ref() {
                let mut facts = (**facts).clone();
                facts.display_area = Some(area);
                st.facts = Some(Arc::new(facts));
            }
            st.display_file.clone().filter(|_| save_now)
        };
        // Kept across sessions, as the Stream Engine does.
        if let Some(path) = file {
            save_display_area(&path, &area);
        }
    }
    result.map(|_| Vec::new()).into()
}

/// Save `area` at `path`. The device has it already, so failing only costs
/// it at the next start. The same area again is not written (it would push
/// the rollback copy out).
pub(crate) fn save_display_area(path: &std::path::Path, area: &DisplayArea) {
    if crate::display::load(path).ok().flatten() == Some(*area) {
        return;
    }
    match crate::display::save(path, area) {
        Ok(()) => info!(
            path = %path.display(),
            width_mm = width_mm(area),
            "display area saved"
        ),
        Err(e) => warn!(path = %path.display(), error = %e, "could not save the display area"),
    }
}

/// Put the display area back as it was before a calibration session that
/// did not commit changed it: on `device` (when there is one) and for later
/// inits. The saved file never had the session's area.
pub(crate) fn put_display_back(
    state: &Mutex<State>,
    device: Option<&dyn crate::device::DeviceCommands>,
    before: &crate::calibration::DisplayBefore,
) {
    if let (Some(device), Some(area)) = (device, before.device) {
        let display_id = lock_state(state)
            .facts
            .as_ref()
            .and_then(|f| f.display_id)
            .unwrap_or(DEFAULT_DISPLAY_ID);
        if let Err(code) = crate::device::run(
            device,
            cmd::DISPLAY_AREA_SET,
            display_area_set_payload(&area, display_id),
            DISPLAY_AREA_TIMEOUT,
        ) {
            warn!(
                status = code,
                "could not put the display area back on the device"
            );
        }
    }
    put_display_configuration_back(&mut lock_state(state), before);
}

/// The configuration half of [`put_display_back`]: what later inits write.
pub(crate) fn put_display_configuration_back(
    st: &mut State,
    before: &crate::calibration::DisplayBefore,
) {
    st.display_override = before.configured;
    if let Some(engine) = st.engine.as_ref() {
        engine.set_display_area_override(before.configured);
    }
    if let Some(facts) = st.facts.as_ref() {
        let mut facts = (**facts).clone();
        facts.display_area = before.device;
        st.facts = Some(Arc::new(facts));
    }
    info!("display area put back: the calibration session did not commit");
}

/// Width of a display area, for the log.
fn width_mm(area: &DisplayArea) -> f64 {
    (0..3)
        .map(|i| (area.top_right_mm[i] - area.top_left_mm[i]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// At start: configure the saved display area, so the engine's first init
/// already writes it. It wins over `TOBII_DISPLAY_MM`: it was set on the
/// device later (the calibration saved with it was made on it), and the
/// variable only fills in while nothing is saved.
pub(crate) fn restore_saved_display_area(st: &mut State) {
    let Some(path) = st.display_file.clone() else {
        return;
    };
    match crate::display::load(&path) {
        Ok(Some(area)) => {
            info!(
                path = %path.display(),
                width_mm = width_mm(&area),
                "display area: using the saved one"
            );
            if st.display_request.is_some() {
                info!(
                    path = %path.display(),
                    "TOBII_DISPLAY_MM is ignored while a display area is saved; delete the file to use it"
                );
            }
            st.display_override = Some(area);
        }
        Ok(None) => {}
        Err(e) => warn!(path = %path.display(), error = %e, "ignoring the saved display area"),
    }
}

/// A monitor size from `TOBII_DISPLAY_MM=<width>x<height>[+<offset_x>]` (mm):
/// the display area is computed from it and the tracker's mounting once the
/// device reports the mounting, replacing the capture author's monitor that
/// the init replay would otherwise leave configured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DisplaySize {
    width_mm: f64,
    height_mm: f64,
    offset_x_mm: f64,
}

impl DisplaySize {
    pub(crate) fn from_env() -> Option<Self> {
        let value = std::env::var("TOBII_DISPLAY_MM").ok()?;
        let parsed = Self::parse(&value);
        if parsed.is_none() {
            warn!(
                value,
                "TOBII_DISPLAY_MM: expected <width>x<height>[+<offset_x>] in mm"
            );
        }
        parsed
    }

    fn parse(s: &str) -> Option<Self> {
        let (size, offset) = s.split_once('+').map_or((s, "0"), |(a, b)| (a, b));
        let (w, h) = size.split_once(['x', 'X'])?;
        let parsed = Self {
            width_mm: w.trim().parse().ok()?,
            height_mm: h.trim().parse().ok()?,
            offset_x_mm: offset.trim().parse().ok()?,
        };
        (parsed.width_mm > 0.0 && parsed.height_mm > 0.0 && parsed.offset_x_mm.is_finite())
            .then_some(parsed)
    }
}

/// After a device init: if `TOBII_DISPLAY_MM` is set and no client has set a
/// display area, compute one from the reported mounting and write it. The
/// write runs on its own thread (the caller is the fan-out pump); later
/// re-inits pick the area up from the override.
pub(crate) fn apply_display_request(st: &mut State) {
    let (Some(size), None) = (st.display_request, st.display_override) else {
        return;
    };
    let Some(mounting) = st.facts.as_ref().and_then(|f| f.mounting) else {
        return;
    };
    let area = display_area_basic(size.width_mm, size.height_mm, size.offset_x_mm, &mounting);
    st.display_override = Some(area);
    if let Some(engine) = st.engine.as_ref() {
        engine.set_display_area_override(Some(area));
        let commands = engine.commands();
        let payload = display_area_set_payload(&area, DEFAULT_DISPLAY_ID);
        info!(
            width_mm = size.width_mm,
            height_mm = size.height_mm,
            "configuring the display area"
        );
        thread::spawn(move || {
            let device: &dyn crate::device::DeviceCommands = &commands;
            if let Err(status) =
                crate::device::run(device, cmd::DISPLAY_AREA_SET, payload, DISPLAY_AREA_TIMEOUT)
            {
                warn!(status, "could not set the display area");
            }
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
    use tobii_proto::time::now_us;

    #[test]
    #[allow(clippy::float_cmp)] // reason: parsed literals
    fn parses_display_sizes() {
        let s = DisplaySize::parse("597x336").expect("size");
        assert_eq!(
            (s.width_mm, s.height_mm, s.offset_x_mm),
            (597.0, 336.0, 0.0)
        );
        let s = DisplaySize::parse(" 633.6 X 334.3 + 1.0 ").expect("size");
        assert_eq!(
            (s.width_mm, s.height_mm, s.offset_x_mm),
            (633.6, 334.3, 1.0)
        );
        assert_eq!(DisplaySize::parse("597"), None);
        assert_eq!(DisplaySize::parse("-1x2"), None);
    }

    #[test]
    fn state_requests_answer_from_the_daemon() {
        let state = Mutex::new(crate::daemon::tests::state_with_client(1));
        lock_state(&state).calibration.id = Some(1_904_654_973);
        let ask = |id| {
            handle(
                &state,
                1,
                &Request {
                    id: 9,
                    kind: kind::STATE,
                    payload: &encode_u32(id),
                },
            )
        };
        assert_eq!(
            ask(state::CALIBRATION_ID),
            Reply::ok(encode_u32(1_904_654_973))
        );
        assert_eq!(ask(state::CALIBRATION_ACTIVE), Reply::ok(vec![0]));
        assert_eq!(ask(state::DEVICE_PAUSED), Reply::ok(vec![0]));
        lock_state(&state).paused = true;
        assert_eq!(ask(state::DEVICE_PAUSED), Reply::ok(vec![1]));
        assert_eq!(ask(3), Reply::err(status::NOT_SUPPORTED));
    }

    #[test]
    fn stream_types_answer_from_the_facts() {
        let mut st = crate::daemon::tests::state_with_client(1);
        st.fake_device = Some(Arc::new(Answering(1)));
        let state = Mutex::new(st);
        let req = Request {
            id: 1,
            kind: kind::STREAM_TYPES,
            payload: &[],
        };
        let streams = vec![
            request::StreamType {
                id: 0x500,
                name: "gaze".into(),
                ..request::StreamType::default()
            },
            request::StreamType {
                id: 0x508,
                name: "image_collection".into(),
                text: String::new(),
                value: 1000,
            },
        ];
        lock_state(&state).facts = Some(Arc::new(DeviceFacts {
            streams: streams.clone(),
            ..DeviceFacts::default()
        }));

        let reply = handle(&state, 1, &req);

        assert_eq!(reply, Reply::ok(encode_stream_types(&streams)));
        assert!(lock_state(&state).clients[0].holds_device);

        // An init that reported no catalogue.
        lock_state(&state).facts = Some(Arc::new(DeviceFacts::default()));
        assert_eq!(handle(&state, 1, &req), Reply::err(status::NOT_SUPPORTED));
    }

    #[test]
    fn the_hardware_configuration_answers_from_the_facts() {
        let mut st = crate::daemon::tests::state_with_client(1);
        st.fake_device = Some(Arc::new(Answering(1)));
        st.facts = Some(Arc::new(DeviceFacts::default()));
        let state = Mutex::new(st);
        let req = Request {
            id: 1,
            kind: kind::HARDWARE_CONFIGURATION,
            payload: &[],
        };

        // What the ET5 reports on Linux: an empty 2120.
        assert_eq!(handle(&state, 1, &req), Reply::err(status::NOT_SUPPORTED));
        assert!(lock_state(&state).clients[0].holds_device);

        let hardware = request::HardwareConfiguration {
            entries: vec![request::HardwareEntry {
                param_a: 16.0,
                width: 2240,
                ..request::HardwareEntry::default()
            }],
            points_mm: vec![[130.0, 0.76, 1.62]],
            mode: 1,
        };
        lock_state(&state).facts = Some(Arc::new(DeviceFacts {
            hardware: Some(hardware.clone()),
            ..DeviceFacts::default()
        }));
        let reply = handle(&state, 1, &req);

        assert_eq!(reply.status, status::OK);
        assert_eq!(
            request::decode_hardware_configuration(&reply.payload),
            Some(hardware)
        );
    }

    /// Ask for a clock pair from a state that has seen a gaze frame with
    /// `clock`, with a stand-in device so that no engine starts; `feed` runs
    /// beside the request as the pump would, until it is answered.
    fn timesync_from(
        clock: Option<(i64, i64)>,
        feed: impl Fn(&Mutex<State>) + Send + Sync,
    ) -> (Reply, Mutex<State>) {
        let mut st = crate::daemon::tests::state_with_client(1);
        st.fake_device = Some(Arc::new(Answering(1)));
        if let Some((device_us, host_us)) = clock {
            st.note_clock(device_us, host_us);
        }
        let state = Mutex::new(st);
        let req = Request {
            id: 1,
            kind: kind::TIMESYNC,
            payload: &[],
        };
        let answered = AtomicBool::new(false);
        let reply = thread::scope(|s| {
            s.spawn(|| {
                while !answered.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(10));
                    feed(&state);
                }
            });
            let reply = handle(&state, 1, &req);
            answered.store(true, Ordering::Relaxed);
            reply
        });
        (reply, state)
    }

    #[test]
    fn timesync_serves_the_first_gaze_frame_after_the_request() {
        let asked = i64::try_from(now_us()).expect("now");
        let device_us = AtomicI64::new(5_000_000);
        // The gaze stream: a newer frame every 10 ms.
        let (reply, state) = timesync_from(Some((4_000_000, 1)), |state| {
            let device = device_us.fetch_add(10_000, Ordering::Relaxed);
            let host = i64::try_from(now_us()).expect("now");
            lock_state(state).note_clock(device, host);
        });

        assert_eq!(reply.status, status::OK);
        let sync = request::decode_timesync(&reply.payload).expect("timesync");
        assert!(sync.device_us >= 5_000_000, "not the stale pair");
        assert!(sync.host_end_us >= asked);
        assert_eq!(sync.host_end_us - sync.host_start_us, FRAME_LATENCY_US);
        assert!(
            lock_state(&state).clients[0].holds_device,
            "the device stays up for the client"
        );
    }

    #[test]
    fn timesync_serves_a_new_frame_across_a_wall_clock_step_back() {
        // The host clock went back an hour: new frames look older than the
        // request, but they are still new.
        let host_us = i64::try_from(now_us()).expect("now") - 3_600_000_000;
        let (reply, _) = timesync_from(Some((4_000_000, 1)), |state| {
            lock_state(state).note_clock(5_000_000, host_us);
        });

        assert_eq!(reply.status, status::OK);
        let sync = request::decode_timesync(&reply.payload).expect("timesync");
        assert_eq!((sync.device_us, sync.host_end_us), (5_000_000, host_us));
    }

    #[test]
    fn timesync_times_out_on_a_stale_pair() {
        let stale = |_: &Mutex<State>| {};
        let (reply, _) = timesync_from(Some((5_000_000, 1)), stale);
        assert_eq!(reply, Reply::err(status::TIMED_OUT));
        let (reply, _) = timesync_from(None, stale);
        assert_eq!(reply, Reply::err(status::TIMED_OUT));
    }

    /// Answers every command with `status`.
    pub(crate) struct Answering(pub(crate) u32);

    impl crate::device::DeviceCommands for Answering {
        fn run(
            &self,
            _cmd: u32,
            _payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<tobii_usb::engine::CommandResponse, tobii_usb::engine::CommandError> {
            Ok(tobii_usb::engine::CommandResponse {
                status: self.0,
                ..tobii_usb::engine::CommandResponse::ok(Vec::new())
            })
        }
    }

    /// Set `area` on a device answering `device_status`; the reply and
    /// what the daemon now configures at init.
    fn set_area(
        device_status: u32,
        path: &std::path::Path,
        area: &DisplayArea,
    ) -> (Reply, Option<DisplayArea>) {
        let mut st = crate::daemon::tests::state_with_client(1);
        st.fake_device = Some(Arc::new(Answering(device_status)));
        st.display_file = Some(path.to_path_buf());
        let state = Mutex::new(st);
        let payload = encode_display_area(area);
        let reply = handle(
            &state,
            1,
            &Request {
                id: 1,
                kind: kind::DISPLAY_AREA_SET,
                payload: &payload,
            },
        );
        (reply, lock_state(&state).display_override)
    }

    #[test]
    fn a_display_area_the_device_accepts_is_saved_and_restored_at_start() {
        let dir = std::env::temp_dir().join(format!("tobiid-area-{}", std::process::id()));
        let path = dir.join("display-area");
        let area = display_area_basic(
            597.0,
            336.0,
            1.0,
            &tobii_ipc::geometry::GeometryMounting {
                guides: 2,
                width_mm: 184.0,
                angle_deg: 20.0,
                external_offset_mm: [0.0, -0.16, 13.85],
                internal_offset_mm: [0.0, 5.38, 9.86],
            },
        );

        let (reply, configured) = set_area(2, &path, &area);
        assert_eq!(reply, Reply::err(status::OPERATION_FAILED));
        assert!(!path.exists(), "a refused area is not saved");
        assert_eq!(configured, None, "nor configured for the next init");
        // Saved as received: the wire carries f32.
        let (reply, configured) = set_area(1, &path, &area);
        let area = decode_display_area(&encode_display_area(&area)).expect("wire");
        assert_eq!((reply, configured), (Reply::ok(Vec::new()), Some(area)));
        assert_eq!(crate::display::load(&path).expect("load"), Some(area));

        let mut st = crate::daemon::tests::state_with_client(1);
        st.display_file = Some(path.clone());
        st.display_request = None;
        restore_saved_display_area(&mut st);
        assert_eq!(st.display_override, Some(area));

        let mut st = crate::daemon::tests::state_with_client(1);
        st.display_file = Some(path.clone());
        st.display_request = DisplaySize::parse("600x340");
        restore_saved_display_area(&mut st);
        assert_eq!(st.display_override, Some(area), "the saved area wins");

        // The same area again leaves the rollback copy alone.
        let prev = tobii_calib::store::previous_path(&path);
        let _ = std::fs::remove_file(&prev);
        assert_eq!(set_area(1, &path, &area).0, Reply::ok(Vec::new()));
        assert!(!prev.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_non_finite_display_area_is_rejected() {
        let state = Mutex::new(crate::daemon::tests::state_with_client(1));
        let area = DisplayArea {
            top_left_mm: [f64::NAN, 0.0, 0.0],
            ..DisplayArea::default()
        };
        let payload = encode_display_area(&area);
        let req = Request {
            id: 1,
            kind: kind::DISPLAY_AREA_SET,
            payload: &payload,
        };
        assert_eq!(
            handle(&state, 1, &req),
            Reply::err(status::INVALID_PARAMETER)
        );
    }
}
