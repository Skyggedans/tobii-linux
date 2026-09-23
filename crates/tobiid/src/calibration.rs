//! The calibration session, owned by one client at a time.
//!
//! Commands follow the Windows engine's captured sequence (see
//! `tobii_proto::calibration`): start = 1010, 1060, then 1110 with the active
//! calibration to seed the device's sample ring; collect = 1030; compute =
//! 1070, then 1100 to read the result back. Stop = 1020, then 1110 with the
//! calibration to run from then on: the device never stays in the empty
//! state a start leaves it in, and what it runs after a session does not
//! depend on whether 1020 keeps a computed calibration (never established
//! from the captures).
//!
//! A session commits only when its owner stops it asking to keep the result
//! ([`STOP_KEEP`], what `tobii_calibration_stop` sends): the last computed
//! calibration then stays on the device and is saved as the user's, uploaded
//! at every later init. Stopped with [`STOP_DISCARD`], by its owner going
//! away, or by the engine dying, a session leaves nothing behind: the
//! calibration it started from goes back on the device and nothing is saved.
//! A session stopped part way through would otherwise keep a calibration
//! that mixes its points with the previous session's (the device keeps the
//! last 14).
//!
//! A session belongs to the client that started it: others get
//! `CALIBRATION_BUSY`. The owner may set the display area during the session
//! (a calibration only holds for the display area it is made on): it goes to
//! the device at once and is saved with the calibration when the session
//! commits; otherwise the previous area goes back too.

use std::sync::Mutex;
use std::time::Duration;

use tobii_calib::store::{self, Location};
use tobii_ipc::geometry::DisplayArea;
use tobii_ipc::request::{STOP_DISCARD, STOP_KEEP, decode_point_2d, encode_u32, kind, status};
use tobii_ipc::{
    Notification, NotificationValue, STREAM_NOTIFICATIONS, encode_notification, notification,
};
use tobii_proto::calibration::{EYES_BOTH, blob_from_payload, cmd, collect_payload, write_payload};
use tracing::{info, warn};

use crate::daemon::{State, lock_state};
use crate::device::{DeviceCommands, run};
use crate::requests::Reply;

/// Commands the device answers at once (and collect, which takes ~0.9 s).
const QUICK: Duration = Duration::from_secs(5);
/// Compute (~2 s) and uploading a ~660 KB calibration.
const SLOW: Duration = Duration::from_secs(10);

/// `tobii_enabled_eye_t`: both eyes.
const ENABLED_EYE_BOTH: u8 = 2;

/// The running session and the active calibration id.
#[derive(Debug)]
pub(crate) struct Calibration {
    session: Option<Session>,
    /// The owner of a session the engine took down: it may not set the
    /// display area until it stops or goes away (nothing would put it back).
    orphaned: Option<u64>,
    /// The calibration id the device reports.
    pub(crate) id: Option<u32>,
    /// Where computed calibrations are saved.
    pub(crate) location: Location,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            session: None,
            orphaned: None,
            id: None,
            location: store::configured(),
        }
    }
}

impl Calibration {
    pub(crate) fn is_active(&self) -> bool {
        self.session.is_some()
    }

    /// The client whose session is running.
    pub(crate) fn owner(&self) -> Option<u64> {
        self.session.as_ref().map(|s| s.owner)
    }

    /// Whether `client` may set the display area now: not while another
    /// client calibrates, nor after its own session died with the engine.
    pub(crate) fn display_access(&self, client: u64) -> Result<(), u8> {
        match &self.session {
            Some(s) if s.owner != client => Err(status::CALIBRATION_BUSY),
            None if self.orphaned == Some(client) => Err(status::CALIBRATION_NOT_STARTED),
            _ => Ok(()),
        }
    }

    /// `client` is about to set the display area: inside its session,
    /// remember the area it replaces (the first change only), to put back if
    /// the session does not commit. Done before the command goes out, so an
    /// answer that never comes still leaves the way back.
    pub(crate) fn note_display_before(&mut self, client: u64, before: DisplayBefore) {
        if let Some(s) = self.session.as_mut()
            && s.owner == client
        {
            s.display_before.get_or_insert(before);
        }
    }

    /// `client` set the display area to `area`: inside its session it is
    /// saved when the session commits (`false`), otherwise now (`true`).
    pub(crate) fn note_display_set(&mut self, client: u64, area: DisplayArea) -> bool {
        match self.session.as_mut() {
            Some(s) if s.owner == client => {
                s.unsaved_display = Some(area);
                false
            }
            _ => true,
        }
    }
}

/// The display area as it was before a session's owner changed it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DisplayBefore {
    /// What the device had; written back.
    pub(crate) device: Option<DisplayArea>,
    /// What later inits wrote then (the daemon's override).
    pub(crate) configured: Option<DisplayArea>,
}

#[derive(Debug, Clone, PartialEq)]
struct Session {
    owner: u64,
    /// The calibration active when the session started.
    previous: Vec<u8>,
    /// The calibration this session computed last, once it has.
    computed: Option<Vec<u8>>,
    /// The display area the owner replaced.
    display_before: Option<DisplayBefore>,
    /// The display area the owner set, saved when the session commits.
    unsaved_display: Option<DisplayArea>,
}

fn broadcast_state(st: &mut State, active: bool) {
    let body = encode_notification(&Notification {
        kind: notification::CALIBRATION_STATE_CHANGED,
        value: NotificationValue::State(active),
    });
    st.broadcast(STREAM_NOTIFICATIONS, &body);
}

/// The calibration to restore: the saved one when valid, else the one built
/// into the init replay.
fn current_blob(location: &Location) -> Result<Vec<u8>, u8> {
    if let Location::File(path) = location
        && let Ok(Some((blob, _))) = store::load(path)
    {
        return Ok(blob);
    }
    tobii_usb::calibration::embedded_blob().map_err(|e| {
        warn!(error = %e, "no calibration to restore");
        status::OPERATION_FAILED
    })
}

/// Save a blob the device produced or a client supplied; failing to save is
/// reported but does not fail the request (the device has it either way).
fn save(location: &Location, blob: &[u8]) {
    match location {
        Location::File(path) => match store::save(path, blob) {
            Ok(info) => {
                info!(path = %path.display(), id = info.id, points = info.points, "calibration saved");
            }
            Err(e) => warn!(path = %path.display(), error = %e, "calibration not saved"),
        },
        Location::Embedded => info!("calibration not saved (TOBII_CALIBRATION=embedded)"),
    }
}

/// Read the active calibration from the device.
fn read_blob(device: &dyn DeviceCommands) -> Result<Vec<u8>, u8> {
    let payload = run(device, cmd::READ, Vec::new(), QUICK)?;
    blob_from_payload(&payload)
        .map(<[u8]>::to_vec)
        .ok_or(status::OPERATION_FAILED)
}

/// What a request may do given the session.
enum Access {
    /// Only the session's owner.
    Owner,
    /// Anyone, unless another client owns the session.
    NotBusy,
    /// Anyone, any time.
    Any,
}

/// The device, the session as it was, and where calibrations are saved.
type Prepared = (
    std::sync::Arc<dyn DeviceCommands>,
    Option<Session>,
    Location,
);

/// Check `access` for `client` and fetch the device; the session and its
/// location for the caller to use once the lock is released.
fn prepare(state: &Mutex<State>, client: u64, access: &Access) -> Result<Prepared, u8> {
    let mut st = lock_state(state);
    let session = st.calibration.session.clone();
    match (access, &session) {
        (Access::Owner, Some(s)) if s.owner == client => {}
        (Access::Owner, _) => return Err(status::CALIBRATION_NOT_STARTED),
        (Access::NotBusy, Some(s)) if s.owner != client => return Err(status::CALIBRATION_BUSY),
        _ => {}
    }
    let location = st.calibration.location.clone();
    let device = st.device_for(client).ok_or(status::CONNECTION_FAILED)?;
    Ok((device, session, location))
}

/// Answer a calibration request from `client`.
pub(crate) fn handle(state: &Mutex<State>, client: u64, request: u8, payload: &[u8]) -> Reply {
    match request {
        kind::CALIBRATION_START => start(state, client, payload),
        kind::CALIBRATION_STOP => match payload {
            STOP_KEEP => stop(state, client, true).into(),
            STOP_DISCARD => stop(state, client, false).into(),
            _ => Reply::err(status::INVALID_PARAMETER),
        },
        kind::CALIBRATION_COLLECT_2D => match decode_point_2d(payload) {
            Some((x, y)) if (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) => {
                prepare(state, client, &Access::Owner)
                    .and_then(|(device, _, _)| {
                        run(
                            device.as_ref(),
                            cmd::COLLECT_2D,
                            collect_payload(x, y, EYES_BOTH),
                            QUICK,
                        )
                    })
                    .map(|_| Vec::new())
                    .into()
            }
            _ => Reply::err(status::INVALID_PARAMETER),
        },
        kind::CALIBRATION_CLEAR => prepare(state, client, &Access::Owner)
            .and_then(|(device, _, _)| run(device.as_ref(), cmd::CLEAR, Vec::new(), QUICK))
            .map(|_| Vec::new())
            .into(),
        kind::CALIBRATION_COMPUTE => compute(state, client).into(),
        kind::CALIBRATION_RETRIEVE => prepare(state, client, &Access::Any)
            .and_then(|(device, _, _)| read_blob(device.as_ref()))
            .into(),
        kind::CALIBRATION_APPLY => apply(state, client, payload).into(),
        // Discarding a point was never captured; its command is unknown.
        _ => Reply::err(status::NOT_SUPPORTED),
    }
}

fn start(state: &Mutex<State>, client: u64, payload: &[u8]) -> Reply {
    match payload {
        [ENABLED_EYE_BOTH] => {}
        // Per-eye calibration was never captured.
        [0 | 1] => return Reply::err(status::NOT_SUPPORTED),
        _ => return Reply::err(status::INVALID_PARAMETER),
    }
    let location = lock_state(state).calibration.location.clone();
    let previous = match current_blob(&location) {
        Ok(blob) => blob,
        Err(code) => return Reply::err(code),
    };
    let device = {
        let mut st = lock_state(state);
        match &st.calibration.session {
            Some(s) if s.owner == client => return Reply::err(status::CALIBRATION_ALREADY_STARTED),
            Some(_) => return Reply::err(status::CALIBRATION_BUSY),
            None => {}
        }
        let Some(device) = st.device_for(client) else {
            return Reply::err(status::CONNECTION_FAILED);
        };
        // Claimed before the device work so a second client is turned away.
        st.calibration.session = Some(Session {
            owner: client,
            previous: previous.clone(),
            computed: None,
            display_before: None,
            unsaved_display: None,
        });
        st.calibration.orphaned = None;
        device
    };
    let seeded = run(device.as_ref(), cmd::START, Vec::new(), QUICK)
        .and_then(|_| run(device.as_ref(), cmd::CLEAR, Vec::new(), QUICK))
        .and_then(|_| run(device.as_ref(), cmd::WRITE, write_payload(&previous), SLOW));
    let mut st = lock_state(state);
    match seeded {
        Ok(_) => {
            info!(client, "calibration started");
            broadcast_state(&mut st, true);
            Reply::ok(Vec::new())
        }
        Err(code) => {
            st.calibration.session = None;
            drop(st);
            // A start or clear may have gone through: never leave the device
            // without its calibration.
            let _ = run(device.as_ref(), cmd::STOP, Vec::new(), QUICK);
            let _ = run(device.as_ref(), cmd::WRITE, write_payload(&previous), SLOW);
            Reply::err(code)
        }
    }
}

fn compute(state: &Mutex<State>, client: u64) -> Result<Vec<u8>, u8> {
    let (device, _, _) = prepare(state, client, &Access::Owner)?;
    run(device.as_ref(), cmd::COMPUTE, Vec::new(), SLOW)?;
    let blob = read_blob(device.as_ref())?;
    let id = tobii_calib::blob::calibration_id(&blob);
    let mut st = lock_state(state);
    if let Some(s) = st.calibration.session.as_mut() {
        // Kept for the stop: only a valid calibration may be committed.
        match tobii_calib::blob::validate(&blob) {
            Ok(_) => s.computed = Some(blob),
            Err(e) => warn!(error = %e, "computed calibration does not validate; not kept"),
        }
    }
    if id.is_some() {
        st.calibration.id = id;
    }
    drop(st);
    info!(client, id, "calibration computed");
    Ok(encode_u32(id.unwrap_or(0)))
}

fn apply(state: &Mutex<State>, client: u64, payload: &[u8]) -> Result<Vec<u8>, u8> {
    let (device, _, location) = prepare(state, client, &Access::NotBusy)?;
    let blob = if payload.is_empty() {
        // Revert to the calibration built into the init replay.
        if let Location::File(path) = &location
            && let Err(e) = store::remove(path)
        {
            warn!(error = %e, "could not remove the saved calibration");
        }
        tobii_usb::calibration::embedded_blob().map_err(|_| status::OPERATION_FAILED)?
    } else {
        tobii_calib::blob::validate(payload).map_err(|_| status::INVALID_PARAMETER)?;
        payload.to_vec()
    };
    run(device.as_ref(), cmd::WRITE, write_payload(&blob), SLOW)?;
    if !payload.is_empty() {
        save(&location, &blob);
    }
    let mut st = lock_state(state);
    if let Some(id) = tobii_calib::blob::calibration_id(&blob) {
        st.calibration.id = Some(id);
    }
    Ok(Vec::new())
}

/// End `client`'s session: commit what it computed when `keep` (and it
/// computed something), else put back what it started from.
fn stop(state: &Mutex<State>, client: u64, keep: bool) -> Result<Vec<u8>, u8> {
    if lock_state(state).calibration.orphaned == Some(client) {
        // Its session died with the engine; the stop only closes that out.
        lock_state(state).calibration.orphaned = None;
        return Err(status::CALIBRATION_NOT_STARTED);
    }
    let (device, session, location) = prepare(state, client, &Access::Owner)?;
    let stopped = run(device.as_ref(), cmd::STOP, Vec::new(), QUICK);
    let Some(session) = session else {
        return Err(status::CALIBRATION_NOT_STARTED);
    };
    let commit = session.computed.clone().filter(|_| keep);
    let active = commit.as_ref().unwrap_or(&session.previous);
    if let Err(code) = run(device.as_ref(), cmd::WRITE, write_payload(active), SLOW) {
        warn!(
            status = code,
            "could not write the calibration back after stopping"
        );
    }
    match &commit {
        Some(blob) => {
            save(&location, blob);
            let file = lock_state(state).display_file.clone();
            if let (Some(area), Some(path)) = (session.unsaved_display, file) {
                crate::requests::save_display_area(&path, &area);
            }
        }
        None => {
            if let Some(before) = &session.display_before {
                crate::requests::put_display_back(state, Some(device.as_ref()), before);
            }
        }
    }
    let mut st = lock_state(state);
    if let Some(id) = tobii_calib::blob::calibration_id(active) {
        st.calibration.id = Some(id);
    }
    st.calibration.session = None;
    broadcast_state(&mut st, false);
    info!(client, kept = commit.is_some(), "calibration stopped");
    stopped.map(|_| Vec::new())
}

/// A client disconnected: a session it owned is discarded (it never said to
/// keep it).
pub(crate) fn on_client_gone(state: &Mutex<State>, client: u64) {
    let (owns, orphaned) = {
        let st = lock_state(state);
        (
            st.calibration.owner() == Some(client),
            st.calibration.orphaned == Some(client),
        )
    };
    if orphaned {
        lock_state(state).calibration.orphaned = None;
    }
    if owns {
        warn!(
            client,
            "calibrating client disconnected; discarding its session"
        );
        let _ = stop(state, client, false);
    }
}

/// The engine went away: any session died with it, uncommitted. The next
/// init uploads the saved calibration, which is the one the session started
/// from, and the display area it was made on is configured again.
pub(crate) fn on_engine_lost(st: &mut State) {
    if let Some(session) = st.calibration.session.take() {
        if let Some(before) = &session.display_before {
            crate::requests::put_display_configuration_back(st, before);
        }
        if let Some(id) = tobii_calib::blob::calibration_id(&session.previous) {
            st.calibration.id = Some(id);
        }
        st.calibration.orphaned = Some(session.owner);
        broadcast_state(st, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tobii_ipc::request::decode_u32;
    use tobii_usb::engine::{CommandError, CommandResponse};

    /// Answers every command, remembers them, and hands back `blob` for a read.
    struct FakeDevice {
        blob: Vec<u8>,
        log: Mutex<Vec<u32>>,
    }

    impl DeviceCommands for FakeDevice {
        fn run(
            &self,
            cmd: u32,
            _payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<CommandResponse, CommandError> {
            self.log.lock().expect("log").push(cmd);
            let payload = if cmd == cmd::READ {
                write_payload(&self.blob)
            } else {
                Vec::new()
            };
            Ok(CommandResponse { status: 1, payload })
        }
    }

    struct Setup {
        state: Mutex<State>,
        device: Arc<FakeDevice>,
        dir: std::path::PathBuf,
    }

    impl Drop for Setup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn setup(tag: &str) -> Setup {
        let dir = std::env::temp_dir().join(format!("tobiid-calib-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut blob = tobii_usb::calibration::embedded_blob().expect("blob");
        blob[20..24].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let device = Arc::new(FakeDevice {
            blob,
            log: Mutex::new(Vec::new()),
        });
        let mut st = crate::daemon::tests::state_with_client(1);
        st.clients
            .append(&mut crate::daemon::tests::state_with_client(2).clients);
        st.fake_device = Some(device.clone());
        st.calibration.location = Location::File(dir.join("calibration.bin"));
        Setup {
            state: Mutex::new(st),
            device,
            dir,
        }
    }

    fn ask(s: &Setup, client: u64, k: u8, payload: &[u8]) -> Reply {
        handle(&s.state, client, k, payload)
    }

    #[test]
    fn a_session_runs_the_captured_command_sequence_and_saves_the_result() {
        let s = setup("session");

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]),
            Reply::ok(vec![])
        );
        let point = tobii_ipc::request::encode_point_2d(0.5, 0.1);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &point),
            Reply::ok(vec![])
        );
        let computed = ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]);
        assert_eq!(decode_u32(&computed.payload), Some(0x1234_5678));
        assert_eq!(ask(&s, 1, kind::CALIBRATION_STOP, &[]), Reply::ok(vec![]));

        assert_eq!(
            *s.device.log.lock().expect("log"),
            vec![
                cmd::START,
                cmd::CLEAR,
                cmd::WRITE,
                cmd::COLLECT_2D,
                cmd::COMPUTE,
                cmd::READ,
                cmd::STOP,
                cmd::WRITE
            ],
            "the computed calibration is written back after stop"
        );
        let (saved, info) = store::load(&s.dir.join("calibration.bin"))
            .expect("load")
            .expect("saved");
        assert_eq!((info.id, saved), (0x1234_5678, s.device.blob.clone()));
        assert_eq!(lock_state(&s.state).calibration.id, Some(0x1234_5678));
        assert!(!lock_state(&s.state).calibration.is_active());
    }

    #[test]
    fn one_owner_at_a_time() {
        let s = setup("owner");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::CALIBRATION_ALREADY_STARTED
        );
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::CALIBRATION_BUSY
        );
        let point = tobii_ipc::request::encode_point_2d(0.5, 0.5);
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_COLLECT_2D, &point).status,
            status::CALIBRATION_NOT_STARTED
        );
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_APPLY, &s.device.blob.clone()).status,
            status::CALIBRATION_BUSY
        );
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_RETRIEVE, &[]).status,
            status::OK,
            "reading is always allowed"
        );
    }

    #[test]
    fn an_owner_that_disconnects_is_stopped_and_the_calibration_restored() {
        let s = setup("gone");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
        s.device.log.lock().expect("log").clear();

        on_client_gone(&s.state, 1);

        assert_eq!(
            *s.device.log.lock().expect("log"),
            vec![cmd::STOP, cmd::WRITE]
        );
        assert!(!lock_state(&s.state).calibration.is_active());
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
    }

    #[test]
    fn apply_validates_saves_and_reverts() {
        let s = setup("apply");
        let path = s.dir.join("calibration.bin");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_APPLY, &[0u8; 64]).status,
            status::INVALID_PARAMETER
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_APPLY, &s.device.blob.clone()).status,
            status::OK
        );
        assert!(path.exists());
        assert_eq!(ask(&s, 1, kind::CALIBRATION_APPLY, &[]).status, status::OK);
        assert!(
            !path.exists(),
            "an empty apply reverts to the built-in calibration"
        );
        assert_eq!(lock_state(&s.state).calibration.id, Some(1_904_654_973));
    }

    #[test]
    fn unsupported_and_malformed_requests() {
        let s = setup("reject");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[0]).status,
            status::NOT_SUPPORTED
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[]).status,
            status::INVALID_PARAMETER
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_DISCARD_2D, &[]).status,
            status::NOT_SUPPORTED
        );
        let outside = tobii_ipc::request::encode_point_2d(1.5, 0.5);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &outside).status,
            status::INVALID_PARAMETER
        );
    }

    fn area(width_mm: f64) -> DisplayArea {
        let mounting = tobii_ipc::geometry::GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 20.0,
            external_offset_mm: [0.0, -0.16, 13.85],
            internal_offset_mm: [0.0, 5.38, 9.86],
        };
        // As the wire carries it (f32), so it compares equal after a set.
        let a = tobii_ipc::geometry::display_area_basic(width_mm, 336.0, 0.0, &mounting);
        tobii_ipc::request::decode_display_area(&tobii_ipc::request::encode_display_area(&a))
            .expect("area")
    }

    fn set_area(s: &Setup, client: u64, a: &DisplayArea) -> Reply {
        let payload = tobii_ipc::request::encode_display_area(a);
        crate::requests::handle(
            &s.state,
            client,
            &tobii_ipc::request::Request {
                id: 1,
                kind: kind::DISPLAY_AREA_SET,
                payload: &payload,
            },
        )
    }

    /// A session's owner has an older area configured and on the device,
    /// nothing saved, and a display file to save to.
    fn display_setup(tag: &str) -> (Setup, std::path::PathBuf, DisplayArea) {
        let s = setup(tag);
        let file = s.dir.join("display-area");
        let old = area(633.6);
        {
            let mut st = lock_state(&s.state);
            st.display_file = Some(file.clone());
            st.display_override = Some(old);
            st.facts = Some(Arc::new(tobii_proto::facts::DeviceFacts {
                display_area: Some(old),
                ..Default::default()
            }));
        }
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        (s, file, old)
    }

    fn display_writes(s: &Setup) -> usize {
        s.device
            .log
            .lock()
            .expect("log")
            .iter()
            .filter(|c| **c == tobii_proto::protocol::cmd::DISPLAY_AREA_SET)
            .count()
    }

    #[test]
    fn a_session_that_computes_nothing_puts_the_display_area_back() {
        let (s, file, old) = display_setup("area-back");
        assert_eq!(
            set_area(&s, 2, &area(597.0)).status,
            status::CALIBRATION_BUSY,
            "only the owner changes the area during its session"
        );
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
        assert!(!file.exists(), "saved only with a calibration made on it");

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, &[]),
            Reply::ok(Vec::new())
        );

        assert_eq!(display_writes(&s), 2, "set, then put back");
        let st = lock_state(&s.state);
        assert_eq!(st.display_override, Some(old));
        assert_eq!(st.facts.as_ref().and_then(|f| f.display_area), Some(old));
        assert!(!file.exists());
    }

    #[test]
    fn a_session_that_computed_keeps_its_display_area() {
        let (s, file, _) = display_setup("area-kept");
        let new = area(597.0);
        assert_eq!(set_area(&s, 1, &new), Reply::ok(Vec::new()));
        assert!(!file.exists());
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );
        assert!(
            !file.exists(),
            "nothing is saved before the session commits"
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, &[]),
            Reply::ok(Vec::new())
        );

        assert_eq!(display_writes(&s), 1);
        assert_eq!(lock_state(&s.state).display_override, Some(new));
        assert_eq!(crate::display::load(&file).expect("load"), Some(new));
    }

    #[test]
    fn losing_the_engine_mid_session_puts_the_display_configuration_back() {
        let (s, file, old) = display_setup("area-engine");
        crate::display::save(&file, &area(520.0)).expect("save");
        // The saved area before the change is what goes back to disk.
        let saved_before = crate::display::load(&file).expect("load");
        // A new session, with that file in place.
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, &[]),
            Reply::ok(Vec::new())
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));

        on_engine_lost(&mut lock_state(&s.state));

        assert_eq!(lock_state(&s.state).display_override, Some(old));
        assert_eq!(crate::display::load(&file).expect("load"), saved_before);
    }

    #[test]
    fn a_discarded_session_leaves_nothing_behind() {
        let (s, file, old) = display_setup("discard");
        let calibration = s.dir.join("calibration.bin");
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD),
            Reply::ok(Vec::new())
        );

        assert!(!calibration.exists() && !file.exists());
        let st = lock_state(&s.state);
        assert_eq!(st.display_override, Some(old));
        // The calibration it started from (the built-in one) is back in use.
        let built_in = tobii_usb::calibration::embedded_blob().expect("blob");
        assert_eq!(
            st.calibration.id,
            tobii_calib::blob::calibration_id(&built_in)
        );
        assert!(!st.calibration.is_active());
    }

    #[test]
    fn a_client_going_away_discards_its_session() {
        let (s, file, old) = display_setup("gone");
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );

        on_client_gone(&s.state, 1);

        assert!(!s.dir.join("calibration.bin").exists() && !file.exists());
        assert_eq!(lock_state(&s.state).display_override, Some(old));
    }

    #[test]
    fn the_owner_of_a_session_lost_with_the_engine_must_close_it_first() {
        let (s, _, _) = display_setup("orphan");
        on_engine_lost(&mut lock_state(&s.state));

        assert_eq!(
            set_area(&s, 1, &area(597.0)).status,
            status::CALIBRATION_NOT_STARTED,
            "nothing would put this area back"
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD).status,
            status::CALIBRATION_NOT_STARTED
        );
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
    }

    #[test]
    fn a_stop_payload_is_keep_or_discard() {
        let s = setup("stop-payload");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, &[7]).status,
            status::INVALID_PARAMETER
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(Vec::new())
        );
    }
}
