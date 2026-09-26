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
//! Discard = 1080 with a collect's payload, taken from the DLL (never
//! captured). It leaves the session as it is. A collect or discard point
//! outside the display is refused before the session is looked at; the DLL
//! checks the session first and sends any point unchecked.
//!
//! Collect, discard, clear and compute need the device's session: one the
//! device refuses for a bad state is `CALIBRATION_NOT_STARTED`, as the Stream
//! Engine documents for them. The daemon's session is left to its owner, who
//! may still stop it (a device init ends it, see [`on_device_ready`]). A bad
//! state anywhere else (the start, the stop's write-back, reading or writing
//! the calibration) is `OPERATION_FAILED`. Provisional: the ET5 was never
//! captured refusing a command.
//!
//! A session commits only when its owner stops it asking to keep the result
//! ([`STOP_KEEP`], what `tobii_calibration_stop` sends): the last computed
//! calibration is then saved as the user's, stays on the device and is
//! uploaded at every later init. One that cannot be saved is not kept either
//! (the stop is `OPERATION_FAILED`). One saved is kept even if the device
//! then refuses it or goes away before taking it: the stop answers that
//! failure, and the next init uploads it. With `TOBII_CALIBRATION=embedded`
//! none is saved, and the one kept runs until the next init. Stopped with
//! [`STOP_DISCARD`], by its owner going away, by the engine dying or by the
//! device re-initialising (the engine re-opening it after a stall or a USB
//! error), a session leaves nothing behind: the calibration it started from
//! goes back on the device and nothing is saved.
//! A session stopped part way through would otherwise keep a calibration
//! that mixes its points with the previous session's (the device keeps the
//! last 14). A stop under way is left to finish by a device init, and by a
//! lost engine when it saves (see [`on_device_ready`] and
//! [`on_engine_lost`]).
//!
//! A session belongs to the client that started it: others get
//! `CALIBRATION_BUSY`. None starts while the device is paused or a pause is
//! on its way (`NOT_AVAILABLE`), and a pause waits for no session (see
//! [`crate::pause`]). The owner may set the display area during the session
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
use tobii_proto::calibration::{
    EYES_BOTH, blob_from_payload, cmd, collect_payload, discard_payload, write_payload,
};
use tracing::{info, warn};

use crate::daemon::{State, lock_state, look_for_tracker};
use crate::device::{DeviceCommands, run, run_in_session};
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
    /// The owner of a session that ended without it (the engine lost, or
    /// the device re-initialised): it may not set the display area until it
    /// stops or goes away (nothing would put it back).
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
    /// client calibrates, nor after its own session ended with the engine
    /// or a device init.
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
    /// The device took the start (1010, 1060, 1110) and clients were told.
    /// Until then a device init or a lost engine leaves the session to the
    /// start in flight (see [`on_device_ready`] and [`on_engine_lost`]).
    started: bool,
    /// Its owner's stop, once it runs: a device init leaves the session to
    /// it, and so does a lost engine when it saves (see [`on_device_ready`]
    /// and [`on_engine_lost`]).
    stop: Option<Stopping>,
}

/// What a stop under way does with the session's calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stopping {
    /// Saves none: it discards, or keeps with none computed or none saved
    /// (`TOBII_CALIBRATION=embedded`).
    SavesNothing,
    /// Saves the one the session computed, with the display area its owner
    /// set.
    Saves,
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

/// Save a blob the device produced or a client supplied. Whether it went to
/// disk: `false` when it could not be written or nothing is saved
/// (`TOBII_CALIBRATION=embedded`).
fn save(location: &Location, blob: &[u8]) -> bool {
    match location {
        Location::File(path) => match store::save(path, blob) {
            Ok(info) => {
                info!(path = %path.display(), id = info.id, points = info.points, "calibration saved");
                true
            }
            Err(e) => {
                warn!(path = %path.display(), error = %e, "calibration not saved");
                false
            }
        },
        Location::Embedded => {
            info!("calibration not saved (TOBII_CALIBRATION=embedded)");
            false
        }
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

/// Check `access` for `client`.
fn check(st: &mut State, client: u64, access: &Access) -> Result<(), u8> {
    match (access, st.calibration.owner()) {
        (Access::Owner, Some(owner)) if owner == client => Ok(()),
        (Access::Owner, _) => {
            // A client whose session ended without it now knows it.
            if st.calibration.orphaned == Some(client) {
                st.calibration.orphaned = None;
            }
            Err(status::CALIBRATION_NOT_STARTED)
        }
        (Access::NotBusy, Some(owner)) if owner != client => Err(status::CALIBRATION_BUSY),
        _ => Ok(()),
    }
}

/// Check `access` for `client` and fetch the device; the session and its
/// location for the caller to use once the lock is released.
fn prepare(state: &Mutex<State>, client: u64, access: &Access) -> Result<Prepared, u8> {
    let on_bus = look_for_tracker(state);
    prepare_locked(&mut lock_state(state), client, access, on_bus)
}

/// [`prepare`] under the state lock the caller holds, with what
/// [`look_for_tracker`] found before it was taken. Checked again once the
/// device is fetched, and before a missing one is `CONNECTION_FAILED`: a
/// dead engine the fetch drops takes the session along unless its stop
/// saves (see [`on_engine_lost`]), whether or not another starts. A request
/// refused before the fetch starts no engine.
fn prepare_locked(
    st: &mut State,
    client: u64,
    access: &Access,
    tracker_on_bus: bool,
) -> Result<Prepared, u8> {
    check(st, client, access)?;
    let device = st.device_for(client, tracker_on_bus);
    check(st, client, access)?;
    let device = device.ok_or(status::CONNECTION_FAILED)?;
    let session = st.calibration.session.clone();
    let location = st.calibration.location.clone();
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
        kind::CALIBRATION_COLLECT_2D => {
            point_2d(state, client, payload, cmd::COLLECT_2D, collect_payload)
        }
        kind::CALIBRATION_DISCARD_2D => {
            point_2d(state, client, payload, cmd::DISCARD_2D, discard_payload)
        }
        kind::CALIBRATION_CLEAR => prepare(state, client, &Access::Owner)
            .and_then(|(device, _, _)| {
                run_in_session(device.as_ref(), cmd::CLEAR, Vec::new(), QUICK)
            })
            .map(|_| Vec::new())
            .into(),
        kind::CALIBRATION_COMPUTE => compute(state, client).into(),
        kind::CALIBRATION_RETRIEVE => prepare(state, client, &Access::Any)
            .and_then(|(device, _, _)| read_blob(device.as_ref()))
            .into(),
        kind::CALIBRATION_APPLY => apply(state, client, payload).into(),
        _ => Reply::err(status::NOT_SUPPORTED),
    }
}

/// Send the point in `payload` to the device as `command` for the session's
/// owner, both eyes. The point must lie on the display (0..=1 each way).
fn point_2d(
    state: &Mutex<State>,
    client: u64,
    payload: &[u8],
    command: u32,
    build: fn(f32, f32, u32) -> Vec<u8>,
) -> Reply {
    let Some((x, y)) = decode_point_2d(payload)
        .filter(|(x, y)| (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y))
    else {
        return Reply::err(status::INVALID_PARAMETER);
    };
    prepare(state, client, &Access::Owner)
        .and_then(|(device, _, _)| {
            run_in_session(device.as_ref(), command, build(x, y, EYES_BOTH), QUICK)
        })
        .map(|_| Vec::new())
        .into()
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
    let on_bus = look_for_tracker(state);
    let (device, losses) = {
        let mut st = lock_state(state);
        match &st.calibration.session {
            Some(s) if s.owner == client => return Reply::err(status::CALIBRATION_ALREADY_STARTED),
            Some(_) => return Reply::err(status::CALIBRATION_BUSY),
            None => {}
        }
        // A paused device sends no gaze to calibrate with.
        if st.paused || st.pausing {
            return Reply::err(status::NOT_AVAILABLE);
        }
        let Some(device) = st.device_for(client, on_bus) else {
            return Reply::err(status::CONNECTION_FAILED);
        };
        // Claimed before the device work so a second client is turned away.
        st.calibration.session = Some(Session {
            owner: client,
            previous: previous.clone(),
            computed: None,
            display_before: None,
            unsaved_display: None,
            started: false,
            stop: None,
        });
        if st.calibration.orphaned == Some(client) {
            st.calibration.orphaned = None;
        }
        (device, st.engine_losses)
    };
    // Inits count from the device's answer to 1010. One before the engine
    // sends it (a cold engine inits twice before it runs a queued command)
    // came before the session did; one while it is on the device fails it.
    let mut inits = None;
    let seeded = run(device.as_ref(), cmd::START, Vec::new(), QUICK)
        .and_then(|_| {
            inits = Some(lock_state(state).device_inits);
            run(device.as_ref(), cmd::CLEAR, Vec::new(), QUICK)
        })
        .and_then(|_| run(device.as_ref(), cmd::WRITE, write_payload(&previous), SLOW));
    let mut st = lock_state(state);
    let seeded = match seeded {
        // The engine that took the start was lost before it could be
        // announced: its successor's device has no session.
        Ok(_) if st.engine_losses != losses => {
            warn!(client, "the engine was lost as the calibration started");
            Err(status::CONNECTION_FAILED)
        }
        // The engine re-opened the device once it had answered the 1010
        // (only the command in flight fails): the init ended the session,
        // and what the start sent after it went to a device outside one.
        // Answered as an init that cuts into a command is, whatever the
        // device made of the rest.
        _ if inits.is_some_and(|n| st.device_inits != n) => {
            warn!(
                client,
                "the device re-initialised as the calibration started"
            );
            Err(status::CONNECTION_FAILED)
        }
        seeded => seeded,
    };
    match seeded {
        Ok(_) => {
            if let Some(s) = st.calibration.session.as_mut()
                && s.owner == client
            {
                s.started = true;
            }
            info!(client, "calibration started");
            broadcast_state(&mut st, true);
            Reply::ok(Vec::new())
        }
        Err(code) => {
            // Its own session only: nothing else may be cleared here.
            if st.calibration.owner() == Some(client) {
                st.calibration.session = None;
            }
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
    run_in_session(device.as_ref(), cmd::COMPUTE, Vec::new(), SLOW)?;
    // The read-back needs no session: a bad state there is a failed read.
    let blob = read_blob(device.as_ref())?;
    let id = tobii_calib::blob::calibration_id(&blob);
    let mut st = lock_state(state);
    // The session may have ended while the device computed (the device
    // re-initialised, or the engine was lost): what it computed belongs to
    // no session then, and the calibration the session started from stays
    // the active one.
    check(&mut st, client, &Access::Owner).inspect_err(|_| {
        warn!(
            client,
            "the calibration session ended as it computed; not kept"
        );
    })?;
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
    // Applied inside the caller's own session: that is now what the
    // session goes back to, and what it computed before is superseded.
    if let Some(s) = st.calibration.session.as_mut()
        && s.owner == client
    {
        s.previous = blob;
        s.computed = None;
    }
    Ok(Vec::new())
}

/// End `client`'s session: commit what it computed when `keep` (and it
/// computed something, and it could be saved), else put back what it
/// started from.
fn stop(state: &Mutex<State>, client: u64, keep: bool) -> Result<Vec<u8>, u8> {
    if lock_state(state).calibration.orphaned == Some(client) {
        // Its session ended without it (the engine lost, or the device
        // re-initialised); the stop only closes that out.
        lock_state(state).calibration.orphaned = None;
        return Err(status::CALIBRATION_NOT_STARTED);
    }
    let on_bus = look_for_tracker(state);
    let (device, session, location, losses) = {
        let mut st = lock_state(state);
        let (device, session, location) = prepare_locked(&mut st, client, &Access::Owner, on_bus)?;
        // Under the same lock: a device init from now on leaves the session
        // to this stop, whose commands still reach the device; so does a
        // lost engine when the stop saves, as the owner's display area then
        // stays configured.
        if let Some(s) = st.calibration.session.as_mut() {
            let saves = keep && s.computed.is_some() && matches!(location, Location::File(_));
            s.stop = Some(if saves {
                Stopping::Saves
            } else {
                Stopping::SavesNothing
            });
        }
        (device, session, location, st.engine_losses)
    };
    let Some(session) = session else {
        return Err(status::CALIBRATION_NOT_STARTED);
    };
    let commit = session.computed.clone().filter(|_| keep);
    // Saved before the device gets it: an init from now on (the device
    // re-opened, or an engine started in place of a lost one) uploads what
    // this stop writes. The display area only goes to disk with the
    // calibration made on it.
    let saved = commit.as_ref().is_some_and(|blob| save(&location, blob));
    if saved {
        let file = lock_state(state).display_file.clone();
        if let (Some(area), Some(path)) = (session.unsaved_display, file) {
            crate::requests::save_display_area(&path, &area);
        }
    }
    // One that could not be saved is not kept either: later inits would
    // not upload it. With TOBII_CALIBRATION=embedded none is saved, and the
    // one kept runs until the next init.
    let unsaved = commit.is_some() && !saved && matches!(location, Location::File(_));
    let kept = commit.filter(|_| !unsaved);
    let active = kept.as_ref().unwrap_or(&session.previous);
    // What 1020 leaves the device running was never established; the write
    // that follows decides, so a refused 1020 alone does not fail the stop.
    if let Err(code) = run(device.as_ref(), cmd::STOP, Vec::new(), QUICK) {
        warn!(
            status = code,
            "the device did not take the calibration stop"
        );
    }
    let mut written = run(device.as_ref(), cmd::WRITE, write_payload(active), SLOW);
    if let Err(code) = written {
        warn!(
            status = code,
            "could not write the calibration after stopping; the device takes the saved one at its next start"
        );
    }
    // An engine started in place of one lost during the stop may have read
    // the calibration file before the save replaced it: it gets the saved
    // one too, and its answer is the one that counts.
    let successor = {
        let st = lock_state(state);
        st.commands()
            .filter(|_| saved && st.engine_losses != losses)
    };
    if let Some(successor) = successor {
        written = run(successor.as_ref(), cmd::WRITE, write_payload(active), SLOW);
        if let Err(code) = written {
            warn!(
                status = code,
                "could not write the saved calibration to the engine started in place of the lost one; it takes it at its next start"
            );
        }
    }
    if kept.is_none()
        && let Some(before) = &session.display_before
    {
        crate::requests::put_display_back(state, before);
    }
    let mut st = lock_state(state);
    st.calibration.id = if st.engine_losses != losses && !saved {
        // What this stop wrote went with the lost engine: the next init
        // uploads the calibration the session started from.
        tobii_calib::blob::calibration_id(&session.previous)
    } else if written.is_ok() {
        tobii_calib::blob::calibration_id(active)
    } else {
        // Unknown when the write failed.
        None
    };
    // A lost engine may have ended the session under this stop, and told
    // every client: its owner hears of it from this answer.
    if st
        .calibration
        .session
        .take_if(|s| s.owner == client)
        .is_some()
    {
        broadcast_state(&mut st, false);
    } else if st.calibration.orphaned == Some(client) {
        st.calibration.orphaned = None;
    }
    info!(client, kept = kept.is_some(), "calibration stopped");
    if unsaved {
        return Err(status::OPERATION_FAILED);
    }
    written.map(|_| Vec::new())
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

/// The engine went away: a started session died with it, uncommitted. The
/// next init uploads the saved calibration, which is the one the session
/// started from, and the display area it was made on is configured again,
/// before a new engine starts and writes it. A start in flight is left to
/// its own error path: its commands went to the lost engine. A stop that
/// saves is left to finish, as by a device init, and settles the
/// calibration id: the display area its owner set stays configured, for
/// the next init to write with the calibration made on it. The stop saves
/// that calibration before it writes it, writes it again to an engine
/// started in place of this one (whose init may have read the file
/// first), and puts the area back itself if the save fails. Any other stop
/// is not waited for: the area the session replaced must be configured
/// before a new engine starts.
pub(crate) fn on_engine_lost(st: &mut State) {
    let Some(session) = st
        .calibration
        .session
        .take_if(|s| s.started && s.stop != Some(Stopping::Saves))
    else {
        return;
    };
    warn!(
        client = session.owner,
        "the engine stopped; discarding the calibration session"
    );
    if let Some(before) = &session.display_before {
        crate::requests::put_display_configuration_back(st, before);
    }
    orphan(st, &session);
}

/// The device finished an init, which leaves it outside any session: a
/// started session ends as with a lost engine. The init uploaded the saved
/// calibration, the one the session started from, so none is written; but
/// it wrote the display area configured then, the session's own once its
/// owner set one, so the area the session replaced goes back on the device
/// too. A start still in flight is left to its own error path. An init
/// before the engine sends the start's 1010 is harmless: a cold engine
/// inits twice before it runs a queued command. One while the 1010 is on
/// the device fails it, as a re-open fails any command in flight. One
/// after the device answered it fails the start too: the start's later
/// commands go through the engine's queue, which outlasts the re-open, to
/// a device outside any session, so the start counts the inits from that
/// answer. A stop that is running is left to finish: the engine keeps its
/// command queue across the re-open, so the stop's commands not yet sent
/// still reach the device. The one on the device at the re-open fails, and
/// the stop answers that failure, but an init after the stop's save uploads
/// what it saved.
///
/// Unverified on hardware: that a re-open takes the device out of its
/// calibration session. The init replay sends no 1020, so a device that
/// stayed in it would be left in a session nobody owns.
pub(crate) fn on_device_ready(st: &mut State) {
    let Some(session) = st
        .calibration
        .session
        .take_if(|s| s.started && s.stop.is_none())
    else {
        return;
    };
    warn!(
        client = session.owner,
        "the device re-initialised; discarding the calibration session"
    );
    if let Some(before) = &session.display_before {
        crate::requests::put_display_back_detached(st, before);
    }
    orphan(st, &session);
}

/// End a session its owner did not stop: the calibration it started from
/// is the active one again, its owner is told on its next request, and
/// every client hears the session ended.
fn orphan(st: &mut State, session: &Session) {
    if let Some(id) = tobii_calib::blob::calibration_id(&session.previous) {
        st.calibration.id = Some(id);
    }
    st.calibration.orphaned = Some(session.owner);
    broadcast_state(st, false);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, Weak};
    use std::thread;
    use std::time::Instant;
    use tobii_ipc::request::decode_u32;
    use tobii_proto::facts::{DEFAULT_DISPLAY_ID, DeviceFacts, display_area_set_payload};
    use tobii_proto::protocol::cmd::DISPLAY_AREA_SET;
    use tobii_proto::protocol::ttp_error;
    use tobii_usb::engine::{CommandError, CommandResponse, Sample};

    /// What befalls the device while a command is on its way.
    #[derive(Debug, Clone, Copy)]
    enum Mishap {
        /// The device re-initialises first: the pump takes in a
        /// `DeviceReady` before the command reaches it.
        Reinit,
        /// The device re-initialises, and then the client starts a session
        /// of its own, before the command reaches the device.
        ReinitThenStart(u64),
        /// The engine is lost, and the command with it.
        EngineLost,
        /// The engine is lost once the device has answered.
        EngineLostAfter,
        /// The engine is lost once the device has answered, and another
        /// client's request starts the next one.
        EngineReplacedAfter,
        /// The engine is lost, and the command with it, and none runs in
        /// its place (the tracker was unplugged).
        Unplugged,
    }

    /// Answers every command, remembers them, and hands back `blob` for a read.
    struct FakeDevice {
        blob: Vec<u8>,
        log: Mutex<Vec<u32>>,
        /// Every command with its payload.
        payloads: Mutex<Vec<(u32, Vec<u8>)>>,
        /// A command the device refuses.
        refuse: Mutex<Option<u32>>,
        /// A command the device refuses for a bad state (TTP `BAD_STATE`).
        bad_state: Mutex<Option<u32>>,
        /// What befalls the device the next time it gets that command.
        mishap: Mutex<Option<(u32, Mishap)>>,
        /// The daemon's state, for a mishap to reach.
        state: Weak<Mutex<State>>,
    }

    impl FakeDevice {
        /// Lose the engine, as the daemon drops one.
        fn lose_engine(&self) {
            if let Some(state) = self.state.upgrade() {
                lock_state(&state).drop_engine();
            }
        }

        /// Lose the engine and start the next one, as a request that finds
        /// it dead does (the stand-in stands for the new one).
        fn replace_engine(&self) {
            if let Some(state) = self.state.upgrade() {
                lock_state(&state).ensure_engine(true);
            }
        }

        /// Lose the engine with none in its place: the stand-in goes too.
        fn unplug(&self) {
            if let Some(state) = self.state.upgrade() {
                let mut st = lock_state(&state);
                st.drop_engine();
                st.fake_device = None;
            }
        }

        /// Have the pump take in a `DeviceReady`.
        fn reinit(&self) {
            if let Some(state) = self.state.upgrade() {
                lock_state(&state).observe(&Sample::DeviceReady(Arc::new(DeviceFacts::default())));
            }
        }
    }

    impl DeviceCommands for FakeDevice {
        fn run(
            &self,
            cmd: u32,
            payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<CommandResponse, CommandError> {
            let mishap = self
                .mishap
                .lock()
                .expect("mishap")
                .take_if(|(on, _)| *on == cmd)
                .map(|(_, mishap)| mishap);
            match mishap {
                Some(Mishap::Reinit) => self.reinit(),
                Some(Mishap::ReinitThenStart(client)) => {
                    self.reinit();
                    if let Some(state) = self.state.upgrade() {
                        let _ = start(&state, client, &[ENABLED_EYE_BOTH]);
                    }
                }
                Some(Mishap::EngineLost) => {
                    self.lose_engine();
                    return Err(CommandError::EngineGone);
                }
                Some(Mishap::Unplugged) => {
                    self.unplug();
                    return Err(CommandError::EngineGone);
                }
                Some(Mishap::EngineLostAfter | Mishap::EngineReplacedAfter) | None => {}
            }
            self.log.lock().expect("log").push(cmd);
            self.payloads.lock().expect("payloads").push((cmd, payload));
            if *self.refuse.lock().expect("refuse") == Some(cmd) {
                return Ok(CommandResponse {
                    status: 2,
                    ..CommandResponse::ok(Vec::new())
                });
            }
            if *self.bad_state.lock().expect("bad state") == Some(cmd) {
                return Ok(CommandResponse {
                    error: ttp_error::BAD_STATE,
                    ..CommandResponse::ok(Vec::new())
                });
            }
            let payload = if cmd == cmd::READ {
                write_payload(&self.blob)
            } else {
                Vec::new()
            };
            match mishap {
                Some(Mishap::EngineLostAfter) => self.lose_engine(),
                Some(Mishap::EngineReplacedAfter) => self.replace_engine(),
                Some(
                    Mishap::Reinit
                    | Mishap::ReinitThenStart(_)
                    | Mishap::EngineLost
                    | Mishap::Unplugged,
                )
                | None => {}
            }
            Ok(CommandResponse::ok(payload))
        }
    }

    struct Setup {
        state: Arc<Mutex<State>>,
        device: Arc<FakeDevice>,
        dir: std::path::PathBuf,
    }

    impl Drop for Setup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn setup(tag: &str) -> Setup {
        setup_with(tag, device_blob())
    }

    /// The calibration the device computes: the one built into the init
    /// replay, with an id of its own.
    pub(crate) fn device_blob() -> Vec<u8> {
        let mut blob = tobii_usb::calibration::embedded_blob().expect("blob");
        blob[20..24].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        blob
    }

    /// As [`setup`], with the device computing `blob`.
    fn setup_with(tag: &str, blob: Vec<u8>) -> Setup {
        let dir = std::env::temp_dir().join(format!("tobiid-calib-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut st = crate::daemon::tests::state_with_client(1);
        st.clients
            .append(&mut crate::daemon::tests::state_with_client(2).clients);
        st.calibration.location = Location::File(dir.join("calibration.bin"));
        let mut device = None;
        let state = Arc::new_cyclic(|weak| {
            let fake = Arc::new(FakeDevice {
                blob,
                log: Mutex::new(Vec::new()),
                payloads: Mutex::new(Vec::new()),
                refuse: Mutex::new(None),
                bad_state: Mutex::new(None),
                mishap: Mutex::new(None),
                state: weak.clone(),
            });
            st.fake_device = Some(fake.clone());
            device = Some(fake);
            Mutex::new(st)
        });
        Setup {
            state,
            device: device.expect("device"),
            dir,
        }
    }

    /// Have `mishap` befall the device the next time it gets `command`.
    fn befall(s: &Setup, command: u32, mishap: Mishap) {
        *s.device.mishap.lock().expect("mishap") = Some((command, mishap));
    }

    /// The device re-initialised.
    fn device_ready(s: &Setup) {
        lock_state(&s.state).observe(&Sample::DeviceReady(Arc::new(DeviceFacts::default())));
    }

    /// Forget the commands the device got so far.
    fn forget_sent(s: &Setup) {
        s.device.log.lock().expect("log").clear();
        s.device.payloads.lock().expect("payloads").clear();
    }

    /// Whether the device gets `command` with `payload` within two seconds
    /// (from a thread of its own).
    fn gets_soon(s: &Setup, command: u32, payload: &[u8]) -> bool {
        gets_soon_times(s, command, payload, 1)
    }

    /// Whether the device has got `command` with `payload` `times` times
    /// within two seconds.
    fn gets_soon_times(s: &Setup, command: u32, payload: &[u8], times: usize) -> bool {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if s.device
                .payloads
                .lock()
                .expect("payloads")
                .iter()
                .filter(|(c, p)| *c == command && p == payload)
                .count()
                >= times
            {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Whether the device gets no further command for 100 ms, time enough
    /// for one sent from a thread of its own to arrive.
    fn stays_quiet(s: &Setup) -> bool {
        let sent = s.device.log.lock().expect("log").len();
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if s.device.log.lock().expect("log").len() != sent {
                return false;
            }
            thread::sleep(Duration::from_millis(5));
        }
        true
    }

    /// The payload of the last display area the device got.
    fn last_display_write(s: &Setup) -> Option<Vec<u8>> {
        s.device
            .payloads
            .lock()
            .expect("payloads")
            .iter()
            .rev()
            .find(|(c, _)| *c == DISPLAY_AREA_SET)
            .map(|(_, p)| p.clone())
    }

    /// The session states announced to `client` so far.
    fn announced(s: &Setup, client: u64) -> Vec<bool> {
        crate::daemon::tests::outbox(&lock_state(&s.state), client)
            .iter()
            .filter_map(|body| match tobii_ipc::decode_server(body) {
                Some(tobii_ipc::ServerMsg::Notification(Notification {
                    kind: notification::CALIBRATION_STATE_CHANGED,
                    value: NotificationValue::State(active),
                })) => Some(active),
                _ => None,
            })
            .collect()
    }

    /// The id of the calibration built into the init replay.
    fn built_in_id() -> Option<u32> {
        tobii_calib::blob::calibration_id(&tobii_usb::calibration::embedded_blob().expect("blob"))
    }

    fn ask(s: &Setup, client: u64, k: u8, payload: &[u8]) -> Reply {
        handle(&s.state, client, k, payload)
    }

    /// Have the device refuse `command` for a bad state from now on (`None`:
    /// no command).
    fn refuse_for_bad_state(s: &Setup, command: Option<u32>) {
        *s.device.bad_state.lock().expect("bad state") = command;
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
    fn a_computed_calibration_with_a_failed_eye_is_kept_and_saved() {
        // The tracker lost the first eye at record 3: its status -1, as a
        // 32-bit int, and a mapping that is not a number.
        let mut blob = device_blob();
        let total =
            usize::try_from(tobii_calib::blob::header(&blob).expect("header").total).expect("fits");
        let record = total + 8 + 3 * tobii_calib::blob::RECORD_LEN;
        blob[record + 8..record + 12].copy_from_slice(&f32::NAN.to_le_bytes());
        blob[record + 16..record + 24].copy_from_slice(&0xffff_ffffu64.to_le_bytes());
        let s = setup_with("failed-eye", blob.clone());
        let path = s.dir.join("calibration.bin");

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]),
            Reply::ok(vec![])
        );
        let computed = ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]);
        assert_eq!(decode_u32(&computed.payload), Some(0x1234_5678));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(vec![])
        );

        // What the device computed is written back, not the calibration the
        // session started from, and saved.
        assert_eq!(
            s.device.payloads.lock().expect("payloads").last(),
            Some(&(cmd::WRITE, write_payload(&blob)))
        );
        let (saved, _) = store::load(&path).expect("load").expect("saved");
        assert_eq!(saved, blob);
        assert_eq!(lock_state(&s.state).calibration.id, Some(0x1234_5678));
        // And it may be applied again.
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_APPLY, &blob).status,
            status::OK
        );
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
        let outside = tobii_ipc::request::encode_point_2d(1.5, 0.5);
        let nan = tobii_ipc::request::encode_point_2d(0.5, f32::NAN);
        for k in [kind::CALIBRATION_COLLECT_2D, kind::CALIBRATION_DISCARD_2D] {
            for payload in [&[][..], &outside, &nan] {
                assert_eq!(
                    ask(&s, 1, k, payload).status,
                    status::INVALID_PARAMETER,
                    "kind {k}, {payload:?}"
                );
            }
        }
        assert!(s.device.log.lock().expect("log").is_empty());
    }

    #[test]
    fn a_discard_sends_1080_with_the_collected_point() {
        let s = setup("discard-point");
        let point = tobii_ipc::request::encode_point_2d(0.3, 0.3);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &point),
            Reply::ok(Vec::new())
        );

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_DISCARD_2D, &point),
            Reply::ok(Vec::new())
        );

        assert_eq!(
            *s.device.log.lock().expect("log"),
            vec![
                cmd::START,
                cmd::CLEAR,
                cmd::WRITE,
                cmd::COLLECT_2D,
                cmd::DISCARD_2D
            ]
        );
        let payloads = s.device.payloads.lock().expect("payloads").clone();
        let expected = collect_payload(0.3, 0.3, EYES_BOTH);
        assert_eq!(payloads[3], (cmd::COLLECT_2D, expected.clone()));
        assert_eq!(payloads[4], (cmd::DISCARD_2D, expected));
        assert_eq!(
            lock_state(&s.state).calibration.owner(),
            Some(1),
            "the session goes on"
        );
    }

    #[test]
    fn a_discard_needs_its_clients_session() {
        let s = setup("discard-session");
        let point = tobii_ipc::request::encode_point_2d(0.3, 0.3);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_DISCARD_2D, &point).status,
            status::CALIBRATION_NOT_STARTED
        );
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_DISCARD_2D, &point).status,
            status::CALIBRATION_NOT_STARTED,
            "another client's session"
        );
        assert!(!s.device.log.lock().expect("log").contains(&cmd::DISCARD_2D));
    }

    #[test]
    fn a_discard_accepted_or_refused_keeps_the_sessions_result() {
        let s = setup("discard-result");
        let point = tobii_ipc::request::encode_point_2d(0.3, 0.3);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_DISCARD_2D, &point),
            Reply::ok(Vec::new())
        );
        *s.device.refuse.lock().expect("refuse") = Some(cmd::DISCARD_2D);

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_DISCARD_2D, &point).status,
            status::OPERATION_FAILED
        );

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(Vec::new())
        );
        assert!(
            s.dir.join("calibration.bin").exists(),
            "what the session computed is still committed"
        );
    }

    #[test]
    fn a_session_request_refused_for_a_bad_state_is_not_started_and_the_session_kept() {
        let s = setup("bad-state-session");
        let point = tobii_ipc::request::encode_point_2d(0.3, 0.3);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
            status::OK
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );
        forget_sent(&s);

        for (k, payload, command) in [
            (kind::CALIBRATION_COLLECT_2D, &point[..], cmd::COLLECT_2D),
            (kind::CALIBRATION_DISCARD_2D, &point[..], cmd::DISCARD_2D),
            (kind::CALIBRATION_CLEAR, &[][..], cmd::CLEAR),
            (kind::CALIBRATION_COMPUTE, &[][..], cmd::COMPUTE),
        ] {
            refuse_for_bad_state(&s, Some(command));
            assert_eq!(
                ask(&s, 1, k, payload).status,
                status::CALIBRATION_NOT_STARTED,
                "kind {k}"
            );
            let st = lock_state(&s.state);
            assert_eq!(
                st.calibration.owner(),
                Some(1),
                "kind {k}: the session goes on"
            );
            assert_eq!(st.calibration.orphaned, None, "kind {k}: no orphan mark");
        }

        assert_eq!(
            *s.device.log.lock().expect("log"),
            vec![cmd::COLLECT_2D, cmd::DISCARD_2D, cmd::CLEAR, cmd::COMPUTE],
            "nothing ends the session on the device"
        );
        refuse_for_bad_state(&s, None);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(Vec::new()),
            "its owner may still stop it"
        );
        assert!(
            s.dir.join("calibration.bin").exists(),
            "what the session computed before is committed"
        );
    }

    #[test]
    fn a_session_request_the_reinitialised_device_refuses_tells_the_owner_again() {
        let s = setup("bad-state-reinit");
        let point = tobii_ipc::request::encode_point_2d(0.3, 0.3);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]),
            Reply::ok(Vec::new())
        );
        forget_sent(&s);
        // The device re-initialises while the collect waits its turn, and
        // then has no session for it.
        befall(&s, cmd::COLLECT_2D, Mishap::Reinit);
        refuse_for_bad_state(&s, Some(cmd::COLLECT_2D));

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &point).status,
            status::CALIBRATION_NOT_STARTED
        );

        {
            let st = lock_state(&s.state);
            assert!(!st.calibration.is_active(), "the init ended the session");
            assert_eq!(
                st.calibration.orphaned,
                Some(1),
                "the device's answer leaves the init's orphan mark"
            );
        }
        assert!(stays_quiet(&s));
        assert_eq!(
            *s.device.log.lock().expect("log"),
            vec![cmd::COLLECT_2D],
            "no stop or write follows the refused collect"
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD).status,
            status::CALIBRATION_NOT_STARTED,
            "the owner is told again"
        );
        assert_eq!(lock_state(&s.state).calibration.orphaned, None);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]),
            Reply::ok(Vec::new()),
            "and may start again"
        );
    }

    #[test]
    fn a_bad_state_where_no_session_is_needed_is_a_failed_operation() {
        let s = setup("bad-state-elsewhere");
        let path = s.dir.join("calibration.bin");
        // Each of the start's own commands: 1060 and 1110 too.
        for command in [cmd::START, cmd::CLEAR, cmd::WRITE] {
            refuse_for_bad_state(&s, Some(command));
            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]).status,
                status::OPERATION_FAILED,
                "cmd {command}"
            );
            assert!(
                !lock_state(&s.state).calibration.is_active(),
                "cmd {command}"
            );
        }
        refuse_for_bad_state(&s, Some(cmd::READ));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_RETRIEVE, &[]).status,
            status::OPERATION_FAILED
        );
        refuse_for_bad_state(&s, Some(cmd::WRITE));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_APPLY, &s.device.blob.clone()).status,
            status::OPERATION_FAILED
        );
        assert!(
            !path.exists(),
            "a calibration the device refused is not saved"
        );

        refuse_for_bad_state(&s, None);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[ENABLED_EYE_BOTH]),
            Reply::ok(Vec::new())
        );
        // Only the compute itself needs the session, not its read-back.
        refuse_for_bad_state(&s, Some(cmd::READ));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OPERATION_FAILED
        );
        assert_eq!(lock_state(&s.state).calibration.owner(), Some(1));
        refuse_for_bad_state(&s, Some(cmd::WRITE));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD).status,
            status::OPERATION_FAILED,
            "the stop ended a session: its write-back failed"
        );
        assert!(!lock_state(&s.state).calibration.is_active());
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
        let (s, file, old) = display_setup("gone-discard");
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

    #[test]
    fn an_apply_inside_the_session_is_what_it_goes_back_to() {
        let s = setup("apply-in-session");
        let path = s.dir.join("calibration.bin");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        let applied = s.device.blob.clone();
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_APPLY, &applied).status,
            status::OK
        );

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD),
            Reply::ok(Vec::new())
        );

        // Device and disk agree on the applied calibration.
        assert_eq!(lock_state(&s.state).calibration.id, Some(0x1234_5678));
        let (saved, _) = store::load(&path).expect("load").expect("saved");
        assert_eq!(saved, applied);
    }

    #[test]
    fn a_refused_stop_command_alone_does_not_fail_a_kept_session() {
        let s = setup("stop-refused");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );
        *s.device.refuse.lock().expect("refuse") = Some(cmd::STOP);

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(Vec::new())
        );
        assert!(s.dir.join("calibration.bin").exists());
    }

    #[test]
    fn a_failed_write_at_stop_is_reported_and_the_id_unknown() {
        let s = setup("write-refused");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        *s.device.refuse.lock().expect("refuse") = Some(cmd::WRITE);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD).status,
            status::OPERATION_FAILED
        );
        assert_eq!(lock_state(&s.state).calibration.id, None);
    }

    #[test]
    fn a_calibration_that_cannot_be_saved_is_not_kept() {
        // Without a lost engine, and with one lost once the device took the
        // write and the next started before the stop ends.
        for (tag, mishap, started) in [
            ("unsavable", None, 0),
            ("unsavable-lost", Some(Mishap::EngineReplacedAfter), 1),
        ] {
            let (s, file, old) = display_setup(tag);
            // The calibration cannot be written: its path is a directory.
            std::fs::create_dir_all(s.dir.join("calibration.bin")).expect("dir");
            let new = area(597.0);
            assert_eq!(set_area(&s, 1, &new), Reply::ok(Vec::new()), "{tag}");
            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
                status::OK,
                "{tag}"
            );
            if let Some(mishap) = mishap {
                befall(&s, cmd::WRITE, mishap);
            }

            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP).status,
                status::OPERATION_FAILED,
                "{tag}"
            );

            assert!(!file.exists(), "{tag}: saved only with its calibration");
            let built_in = tobii_usb::calibration::embedded_blob().expect("blob");
            let last_write = s
                .device
                .payloads
                .lock()
                .expect("payloads")
                .iter()
                .rev()
                .find(|(c, _)| *c == cmd::WRITE)
                .map(|(_, p)| p.clone());
            assert_eq!(
                last_write,
                Some(write_payload(&built_in)),
                "{tag}: the device goes back to what the session started from"
            );
            assert_eq!(
                last_display_write(&s),
                Some(display_area_set_payload(&old, DEFAULT_DISPLAY_ID)),
                "{tag}: and so does the area, on the engine running now"
            );
            let st = lock_state(&s.state);
            assert_eq!(
                st.engines_started,
                vec![Some(new); started],
                "{tag}: an engine started while the stop saved inits with the owner's area"
            );
            assert_eq!(st.display_override, Some(old), "{tag}");
            assert_eq!(
                st.facts.as_ref().and_then(|f| f.display_area),
                Some(old),
                "{tag}"
            );
            assert_eq!(st.calibration.id, built_in_id(), "{tag}");
            assert!(!st.calibration.is_active(), "{tag}");
            assert_eq!(
                st.calibration.display_access(1),
                Ok(()),
                "{tag}: no orphan mark"
            );
        }
    }

    #[test]
    fn the_orphan_mark_lasts_until_its_client_is_told() {
        let (s, _, _) = display_setup("orphan-told");
        on_engine_lost(&mut lock_state(&s.state));
        // Another client's session does not lift it.
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_STOP, STOP_DISCARD),
            Reply::ok(Vec::new())
        );
        assert_eq!(
            set_area(&s, 1, &area(597.0)).status,
            status::CALIBRATION_NOT_STARTED
        );
        // Its own request answered "not started" does.
        let point = tobii_ipc::request::encode_point_2d(0.5, 0.5);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &point).status,
            status::CALIBRATION_NOT_STARTED
        );
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
    }

    #[test]
    fn a_device_init_ends_a_started_session_and_puts_its_display_area_back() {
        let (s, file, old) = display_setup("reinit");
        lock_state(&s.state).clients[1].streams = STREAM_NOTIFICATIONS;
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
        forget_sent(&s);

        lock_state(&s.state).observe(&Sample::DeviceReady(Arc::new(DeviceFacts {
            calibration_id: Some(7),
            ..DeviceFacts::default()
        })));

        {
            let st = lock_state(&s.state);
            assert!(!st.calibration.is_active());
            assert_eq!(
                st.calibration.id,
                built_in_id(),
                "the one the init uploaded, which the session started from"
            );
            assert_eq!(st.display_override, Some(old));
            assert_eq!(st.facts.as_ref().and_then(|f| f.display_area), Some(old));
        }
        assert!(
            gets_soon(
                &s,
                DISPLAY_AREA_SET,
                &display_area_set_payload(&old, DEFAULT_DISPLAY_ID)
            ),
            "the init wrote the session's area: the old one goes back"
        );
        assert!(stays_quiet(&s), "nothing more");
        assert!(
            !s.device
                .log
                .lock()
                .expect("log")
                .iter()
                .any(|c| (cmd::START..=cmd::WRITE).contains(c)),
            "the init restored the calibration"
        );
        assert!(!file.exists());
        assert_eq!(announced(&s, 2), [false]);
        // Its owner is told as after a lost engine.
        assert_eq!(
            set_area(&s, 1, &area(597.0)).status,
            status::CALIBRATION_NOT_STARTED
        );
        let point = tobii_ipc::request::encode_point_2d(0.5, 0.5);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &point).status,
            status::CALIBRATION_NOT_STARTED
        );
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
        // The device's own word on the id then stands.
        lock_state(&s.state).observe(&Sample::Notification(
            tobii_proto::facts::DeviceNotification::CalibrationIdChanged(7),
        ));
        assert_eq!(lock_state(&s.state).calibration.id, Some(7));
    }

    #[test]
    fn a_device_init_with_no_display_change_to_undo_sends_nothing() {
        let s = setup("reinit-quiet");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        forget_sent(&s);

        device_ready(&s);

        assert!(!lock_state(&s.state).calibration.is_active());
        assert!(stays_quiet(&s));
        assert!(s.device.log.lock().expect("log").is_empty());
    }

    #[test]
    fn a_device_init_before_the_start_reaches_the_device_leaves_the_session() {
        let s = setup("reinit-start");
        lock_state(&s.state).clients[0].streams = STREAM_NOTIFICATIONS;
        // A cold tracker inits a second time while the 1010 waits its turn.
        befall(&s, cmd::START, Mishap::Reinit);

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );

        let point = tobii_ipc::request::encode_point_2d(0.5, 0.5);
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COLLECT_2D, &point),
            Reply::ok(Vec::new()),
            "the session goes on"
        );
        assert_eq!(announced(&s, 1), [true]);
    }

    #[test]
    fn a_device_init_after_the_device_took_the_start_fails_it() {
        // The engine re-opens the device once 1010 is answered, before 1060
        // leaves its queue, or once 1060 is answered, before 1110 does: the
        // rest of the start reaches a device the init took out of the
        // session.
        for (tag, command) in [("reinit-clear", cmd::CLEAR), ("reinit-seed", cmd::WRITE)] {
            let s = setup(tag);
            lock_state(&s.state).clients[0].streams = STREAM_NOTIFICATIONS;
            befall(&s, command, Mishap::Reinit);

            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_START, &[2]).status,
                status::CONNECTION_FAILED,
                "{tag}"
            );

            assert_eq!(
                *s.device.log.lock().expect("log"),
                vec![cmd::START, cmd::CLEAR, cmd::WRITE, cmd::STOP, cmd::WRITE],
                "{tag}: the calibration it started from goes back"
            );
            {
                let st = lock_state(&s.state);
                assert!(!st.calibration.is_active(), "{tag}");
                assert_eq!(
                    st.calibration.display_access(1),
                    Ok(()),
                    "{tag}: no orphan mark"
                );
            }
            assert!(announced(&s, 1).is_empty(), "{tag}: nothing to announce");
            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_START, &[2]),
                Reply::ok(Vec::new()),
                "{tag}"
            );
        }
    }

    #[test]
    fn a_refusal_after_a_device_init_fails_the_start_as_the_init_does() {
        let s = setup("reinit-refused");
        // The re-initialised device, outside any session, refuses the 1060.
        befall(&s, cmd::CLEAR, Mishap::Reinit);
        refuse_for_bad_state(&s, Some(cmd::CLEAR));

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]).status,
            status::CONNECTION_FAILED
        );
        assert_eq!(
            *s.device.log.lock().expect("log"),
            vec![cmd::START, cmd::CLEAR, cmd::STOP, cmd::WRITE],
            "the calibration it started from goes back"
        );
        assert!(!lock_state(&s.state).calibration.is_active());
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]).status,
            status::OPERATION_FAILED,
            "the same refusal with no init is the device's own"
        );
    }

    #[test]
    fn a_device_init_during_a_kept_stop_leaves_the_session_to_it() {
        let (s, file, _) = display_setup("reinit-keep");
        lock_state(&s.state).clients[1].streams = STREAM_NOTIFICATIONS;
        let new = area(597.0);
        assert_eq!(set_area(&s, 1, &new), Reply::ok(Vec::new()));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::OK
        );
        befall(&s, cmd::STOP, Mishap::Reinit);

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(Vec::new())
        );

        {
            let st = lock_state(&s.state);
            assert_eq!(st.display_override, Some(new), "the kept area stays");
            assert_eq!(st.facts.as_ref().and_then(|f| f.display_area), Some(new));
            assert_eq!(st.calibration.id, Some(0x1234_5678));
            assert_eq!(st.calibration.display_access(1), Ok(()), "no orphan mark");
        }
        assert_eq!(crate::display::load(&file).expect("load"), Some(new));
        assert!(stays_quiet(&s));
        assert_eq!(display_writes(&s), 1, "the old area does not go back");
        assert_eq!(announced(&s, 2), [false]);
    }

    #[test]
    fn a_device_init_during_a_discarding_stop_leaves_the_session_to_it() {
        let (s, file, old) = display_setup("reinit-discard");
        lock_state(&s.state).clients[1].streams = STREAM_NOTIFICATIONS;
        assert_eq!(set_area(&s, 1, &area(597.0)), Reply::ok(Vec::new()));
        befall(&s, cmd::STOP, Mishap::Reinit);

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD),
            Reply::ok(Vec::new())
        );

        {
            let st = lock_state(&s.state);
            assert_eq!(st.display_override, Some(old));
            assert_eq!(st.facts.as_ref().and_then(|f| f.display_area), Some(old));
            assert_eq!(st.calibration.display_access(1), Ok(()), "no orphan mark");
        }
        assert!(!file.exists());
        assert!(stays_quiet(&s));
        assert_eq!(display_writes(&s), 2, "set, then put back by the stop");
        assert_eq!(announced(&s, 2), [false]);
    }

    #[test]
    fn an_area_set_whose_session_ends_on_the_way_is_neither_kept_nor_saved() {
        let (s, file, old) = display_setup("reinit-area");
        befall(&s, DISPLAY_AREA_SET, Mishap::Reinit);

        assert_eq!(
            set_area(&s, 1, &area(597.0)).status,
            status::CALIBRATION_NOT_STARTED
        );

        {
            let st = lock_state(&s.state);
            assert_eq!(st.display_override, Some(old));
            assert_eq!(st.facts.as_ref().and_then(|f| f.display_area), Some(old));
        }
        assert!(!file.exists());
        let back = display_area_set_payload(&old, DEFAULT_DISPLAY_ID);
        // Put back by the init's thread and again after the refused area,
        // so the tracker ends on the old one whichever went first.
        assert!(gets_soon_times(&s, DISPLAY_AREA_SET, &back, 2));
        assert!(stays_quiet(&s));
        assert_eq!(display_writes(&s), 3, "the refused area, the old one twice");
        assert_eq!(last_display_write(&s), Some(back));
    }

    #[test]
    fn a_compute_whose_session_ends_on_the_way_is_not_kept() {
        let s = setup("reinit-compute");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        // Client 2 claims the device as soon as client 1's session ends.
        befall(&s, cmd::READ, Mishap::ReinitThenStart(2));

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
            status::CALIBRATION_NOT_STARTED
        );

        {
            let st = lock_state(&s.state);
            assert_eq!(
                st.calibration.id,
                built_in_id(),
                "the one client 1's session started from"
            );
            assert_eq!(st.calibration.orphaned, None, "client 1 has been told");
            assert_eq!(st.calibration.owner(), Some(2));
            assert!(
                st.calibration
                    .session
                    .as_ref()
                    .is_some_and(|session| session.computed.is_none()),
                "client 2's session does not get what client 1 computed"
            );
        }
        assert_eq!(
            ask(&s, 2, kind::CALIBRATION_STOP, STOP_KEEP),
            Reply::ok(Vec::new())
        );
        assert!(!s.dir.join("calibration.bin").exists(), "nothing to commit");
    }

    #[test]
    fn an_engine_lost_during_the_stop_ends_the_session_once() {
        let s = setup("lost-stop");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        lock_state(&s.state).clients[1].streams = STREAM_NOTIFICATIONS;
        befall(&s, cmd::STOP, Mishap::EngineLostAfter);

        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_STOP, STOP_DISCARD),
            Reply::ok(Vec::new())
        );

        assert_eq!(announced(&s, 2), [false]);
        assert_eq!(
            lock_state(&s.state).calibration.display_access(1),
            Ok(()),
            "its stop answered: no orphan mark"
        );
    }

    #[test]
    fn an_engine_lost_during_a_kept_stop_leaves_the_session_to_it() {
        // Lost once the device took the write, with the next engine started
        // before the stop ends; lost with the write, with one running in its
        // place by then (the stand-in); and lost with the write, with none.
        // An engine running at the end gets the saved calibration from the
        // stop: its init may have read the file before the save.
        for (tag, mishap, answer, id, started, writes) in [
            (
                "lost-keep",
                Mishap::EngineReplacedAfter,
                status::OK,
                Some(0x1234_5678),
                1,
                2,
            ),
            (
                "lost-keep-write",
                Mishap::EngineLost,
                status::OK,
                Some(0x1234_5678),
                0,
                1,
            ),
            (
                "lost-keep-unplugged",
                Mishap::Unplugged,
                status::CONNECTION_FAILED,
                None,
                0,
                0,
            ),
        ] {
            let (s, file, _) = display_setup(tag);
            lock_state(&s.state).clients[1].streams = STREAM_NOTIFICATIONS;
            let new = area(597.0);
            assert_eq!(set_area(&s, 1, &new), Reply::ok(Vec::new()), "{tag}");
            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
                status::OK,
                "{tag}"
            );
            befall(&s, cmd::WRITE, mishap);

            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_STOP, STOP_KEEP).status,
                answer,
                "{tag}"
            );

            let (saved, _) = store::load(&s.dir.join("calibration.bin"))
                .expect("load")
                .expect("saved");
            assert_eq!(saved, s.device.blob, "{tag}");
            assert_eq!(
                crate::display::load(&file).expect("load"),
                Some(new),
                "{tag}"
            );
            assert_eq!(
                s.device
                    .payloads
                    .lock()
                    .expect("payloads")
                    .iter()
                    .filter(|(c, p)| *c == cmd::WRITE && *p == write_payload(&saved))
                    .count(),
                writes,
                "{tag}: written by the stop, and again to the engine running at its end"
            );
            {
                let st = lock_state(&s.state);
                assert_eq!(
                    st.display_override,
                    Some(new),
                    "{tag}: what later inits write is what was saved"
                );
                assert_eq!(
                    st.facts.as_ref().and_then(|f| f.display_area),
                    Some(new),
                    "{tag}"
                );
                assert_eq!(
                    st.engines_started,
                    vec![Some(new); started],
                    "{tag}: the next engine's first init writes it too"
                );
                assert_eq!(st.calibration.id, id, "{tag}: unknown if unwritten");
                assert!(!st.calibration.is_active(), "{tag}");
                assert_eq!(
                    st.calibration.display_access(1),
                    Ok(()),
                    "{tag}: no orphan mark"
                );
            }
            assert_eq!(announced(&s, 2), [false], "{tag}");
        }
    }

    #[test]
    fn an_engine_lost_during_a_stop_that_saves_nothing_puts_the_area_back_before_the_next_starts() {
        // A discard; a keep with nothing computed; a keep with
        // TOBII_CALIBRATION=embedded, which saves nothing.
        for (tag, payload, computes, location) in [
            ("lost-discard", STOP_DISCARD, true, None),
            ("lost-keep-none", STOP_KEEP, false, None),
            (
                "lost-keep-embedded",
                STOP_KEEP,
                true,
                Some(Location::Embedded),
            ),
        ] {
            let (s, file, old) = display_setup(tag);
            lock_state(&s.state).clients[1].streams = STREAM_NOTIFICATIONS;
            if let Some(location) = location {
                lock_state(&s.state).calibration.location = location;
            }
            assert_eq!(
                set_area(&s, 1, &area(597.0)),
                Reply::ok(Vec::new()),
                "{tag}"
            );
            if computes {
                assert_eq!(
                    ask(&s, 1, kind::CALIBRATION_COMPUTE, &[]).status,
                    status::OK,
                    "{tag}"
                );
            }
            befall(&s, cmd::WRITE, Mishap::EngineReplacedAfter);

            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_STOP, payload),
                Reply::ok(Vec::new()),
                "{tag}"
            );

            assert!(
                !s.dir.join("calibration.bin").exists() && !file.exists(),
                "{tag}"
            );
            {
                let st = lock_state(&s.state);
                assert_eq!(
                    st.engines_started,
                    [Some(old)],
                    "{tag}: the next engine's first init writes the area the session replaced"
                );
                assert_eq!(st.display_override, Some(old), "{tag}");
                assert_eq!(
                    st.facts.as_ref().and_then(|f| f.display_area),
                    Some(old),
                    "{tag}"
                );
                assert_eq!(
                    st.calibration.id,
                    built_in_id(),
                    "{tag}: the one the session started from, which that init uploads"
                );
                assert!(!st.calibration.is_active(), "{tag}");
                assert_eq!(
                    st.calibration.display_access(1),
                    Ok(()),
                    "{tag}: its stop answered: no orphan mark"
                );
            }
            assert_eq!(announced(&s, 2), [false], "{tag}");
        }
    }

    #[test]
    fn a_request_that_replaces_a_dead_engine_finds_its_session_gone() {
        // A new engine starts for a tracker on the bus and none without one:
        // either way the session went with the old engine.
        for (tag, on_bus) in [("lost-on-fetch", true), ("lost-unplugged", false)] {
            let s = setup(tag);
            lock_state(&s.state).fake_present = on_bus;
            let point = tobii_ipc::request::encode_point_2d(0.5, 0.5);
            for (k, payload) in [
                (kind::CALIBRATION_COLLECT_2D, &point[..]),
                (kind::CALIBRATION_COMPUTE, &[][..]),
                (kind::CALIBRATION_STOP, STOP_KEEP),
            ] {
                // Without a tracker the stand-in went with the engine.
                lock_state(&s.state).fake_device = Some(s.device.clone());
                assert_eq!(
                    ask(&s, 1, kind::CALIBRATION_START, &[2]),
                    Reply::ok(Vec::new()),
                    "{tag}, kind {k}"
                );
                forget_sent(&s);
                lock_state(&s.state).lose_engine_on_fetch = true;

                assert_eq!(
                    ask(&s, 1, k, payload).status,
                    status::CALIBRATION_NOT_STARTED,
                    "{tag}, kind {k}"
                );

                assert!(
                    s.device.log.lock().expect("log").is_empty(),
                    "{tag}, kind {k}: nothing reaches the device"
                );
                let st = lock_state(&s.state);
                assert!(!st.calibration.is_active(), "{tag}, kind {k}");
                assert_eq!(
                    st.calibration.display_access(1),
                    Ok(()),
                    "{tag}, kind {k}: its owner has been told"
                );
                assert_eq!(
                    st.fake_device.is_some(),
                    on_bus,
                    "{tag}, kind {k}: an engine runs only for a tracker"
                );
            }
        }
    }

    #[test]
    fn losing_the_engine_as_a_start_is_on_its_way_leaves_it_to_the_start() {
        for (tag, command, mishap) in [
            ("lost-start", cmd::START, Mishap::EngineLost),
            ("lost-seeded", cmd::WRITE, Mishap::EngineLostAfter),
        ] {
            let s = setup(tag);
            lock_state(&s.state).clients[0].streams = STREAM_NOTIFICATIONS;
            befall(&s, command, mishap);

            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_START, &[2]).status,
                status::CONNECTION_FAILED,
                "{tag}"
            );

            {
                let st = lock_state(&s.state);
                assert!(!st.calibration.is_active(), "{tag}");
                assert_eq!(
                    st.calibration.display_access(1),
                    Ok(()),
                    "{tag}: no orphan mark"
                );
            }
            assert!(announced(&s, 1).is_empty(), "{tag}: nothing to announce");
            assert_eq!(
                ask(&s, 1, kind::CALIBRATION_START, &[2]),
                Reply::ok(Vec::new()),
                "{tag}"
            );
        }
    }
}
