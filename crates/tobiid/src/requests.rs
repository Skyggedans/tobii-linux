//! Client requests: what the device reported about itself, its display area,
//! states and clock. Calibration requests are handed to [`crate::calibration`].
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
    encode_display_area, encode_geometry_mounting, encode_timesync, encode_track_box, encode_u32,
    kind, state, status,
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
            Some(_) => Reply::err(status::NOT_SUPPORTED),
            None => Reply::err(status::INVALID_PARAMETER),
        },
        kind::TIMESYNC => match lock_state(state).clock {
            Some((device_us, host_us)) => Reply::ok(encode_timesync(&Timesync {
                host_start_us: host_us - FRAME_LATENCY_US,
                device_us,
                host_end_us: host_us,
            })),
            None => Reply::err(status::NOT_AVAILABLE),
        },
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
fn facts(
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

/// Write a display area to the device and keep it across re-inits.
fn set_display_area(state: &Mutex<State>, client: u64, area: DisplayArea) -> Reply {
    let (device, display_id) = {
        let mut st = lock_state(state);
        st.display_override = Some(area);
        if let Some(engine) = st.engine.as_ref() {
            engine.set_display_area_override(Some(area));
        }
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
        let mut st = lock_state(state);
        if let Some(facts) = st.facts.as_ref() {
            let mut facts = (**facts).clone();
            facts.display_area = Some(area);
            st.facts = Some(Arc::new(facts));
        }
    }
    result.map(|_| Vec::new()).into()
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
mod tests {
    use super::*;

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
        assert_eq!(ask(3), Reply::err(status::NOT_SUPPORTED));
    }

    #[test]
    fn timesync_needs_a_gaze_frame() {
        let state = Mutex::new(crate::daemon::tests::state_with_client(1));
        let req = Request {
            id: 1,
            kind: kind::TIMESYNC,
            payload: &[],
        };
        assert_eq!(handle(&state, 1, &req), Reply::err(status::NOT_AVAILABLE));
        lock_state(&state).clock = Some((5_000_000, 1_700_000_000_000_000));
        let reply = handle(&state, 1, &req);
        let sync = request::decode_timesync(&reply.payload).expect("timesync");
        assert_eq!(sync.device_us, 5_000_000);
        assert!(sync.host_start_us < sync.host_end_us);
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
