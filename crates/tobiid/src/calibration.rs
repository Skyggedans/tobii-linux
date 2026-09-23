//! The calibration session, owned by one client at a time.
//!
//! Commands follow the Windows engine's captured sequence (see
//! `tobii_proto::calibration`): start = 1010, 1060, then 1110 with the active
//! calibration to seed the device's sample ring; collect = 1030; compute =
//! 1070, then 1100 to read the result back, which is saved as the user's
//! calibration and uploaded at every later init. Stop = 1020, then 1110 with
//! the session's result, or the previous calibration when nothing was
//! computed: the device never stays in the empty state a start leaves it in,
//! and what it runs after a session does not depend on whether 1020 keeps a
//! computed calibration (never established from the captures).
//!
//! A session belongs to the client that started it: others get
//! `CALIBRATION_BUSY`, and when the owner disconnects the session is stopped
//! as if it had asked.

use std::sync::Mutex;
use std::time::Duration;

use tobii_calib::store::{self, Location};
use tobii_ipc::request::{decode_point_2d, encode_u32, kind, status};
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
    /// The calibration id the device reports.
    pub(crate) id: Option<u32>,
    /// Where computed calibrations are saved.
    pub(crate) location: Location,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            session: None,
            id: None,
            location: store::configured(),
        }
    }
}

impl Calibration {
    pub(crate) fn is_active(&self) -> bool {
        self.session.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Session {
    owner: u64,
    /// The calibration this session computed, once it has.
    computed: Option<Vec<u8>>,
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
        kind::CALIBRATION_STOP => stop(state, client).into(),
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
    let (device, location) = {
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
            computed: None,
        });
        (device, st.calibration.location.clone())
    };
    let seeded = run(device.as_ref(), cmd::START, Vec::new(), QUICK)
        .and_then(|_| run(device.as_ref(), cmd::CLEAR, Vec::new(), QUICK))
        .and_then(|_| current_blob(&location))
        .and_then(|blob| run(device.as_ref(), cmd::WRITE, write_payload(&blob), SLOW));
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
            let _ = run(device.as_ref(), cmd::STOP, Vec::new(), QUICK);
            Reply::err(code)
        }
    }
}

fn compute(state: &Mutex<State>, client: u64) -> Result<Vec<u8>, u8> {
    let (device, _, location) = prepare(state, client, &Access::Owner)?;
    run(device.as_ref(), cmd::COMPUTE, Vec::new(), SLOW)?;
    let blob = read_blob(device.as_ref())?;
    let id = tobii_calib::blob::calibration_id(&blob);
    match tobii_calib::blob::validate(&blob) {
        Ok(_) => save(&location, &blob),
        Err(e) => warn!(error = %e, "computed calibration does not validate; not saved"),
    }
    let mut st = lock_state(state);
    if let Some(s) = st.calibration.session.as_mut() {
        s.computed = Some(blob);
    }
    if id.is_some() {
        st.calibration.id = id;
    }
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

fn stop(state: &Mutex<State>, client: u64) -> Result<Vec<u8>, u8> {
    let (device, session, location) = prepare(state, client, &Access::Owner)?;
    let stopped = run(device.as_ref(), cmd::STOP, Vec::new(), QUICK);
    let active = match session.and_then(|s| s.computed) {
        Some(blob) => Ok(blob),
        None => current_blob(&location),
    };
    if let Ok(blob) = active
        && let Err(code) = run(device.as_ref(), cmd::WRITE, write_payload(&blob), SLOW)
    {
        warn!(
            status = code,
            "could not write the calibration back after stopping"
        );
    }
    let mut st = lock_state(state);
    st.calibration.session = None;
    broadcast_state(&mut st, false);
    info!(client, "calibration stopped");
    stopped.map(|_| Vec::new())
}

/// The owner disconnected: stop its session.
pub(crate) fn on_client_gone(state: &Mutex<State>, client: u64) {
    let owns = lock_state(state)
        .calibration
        .session
        .as_ref()
        .is_some_and(|s| s.owner == client);
    if owns {
        warn!(
            client,
            "calibrating client disconnected; stopping its session"
        );
        let _ = stop(state, client);
    }
}

/// The engine went away: any session died with it (the next init uploads the
/// saved calibration, which is the pre-session one unless a compute saved a
/// new one).
pub(crate) fn on_engine_lost(st: &mut State) {
    if st.calibration.session.take().is_some() {
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
}
