//! A minimal calibration session, run directly against the engine's
//! [`Commands`] handle: the captured sequence start/clear/write, a collect
//! per point, compute, read, stop/write
//! ([`tobii_proto::calibration`](tobii_proto::calibration)'s module docs).
//! Kotlin drives one point at a time; there is no multi-client ownership to
//! arbitrate, unlike the daemon's own session (`tobiid::calibration`), since
//! this app is the device's only user.
//!
//! The computed blob is saved under the directory Kotlin passes in (the
//! app's own files directory — there is no `$HOME` to default to on
//! Android), via [`tobii_calib::store`], and loaded from there as the
//! session's starting point, same as the daemon does for its init replay.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tobii_ipc::deadline::{CALIBRATION_QUICK as QUICK, CALIBRATION_SLOW as SLOW};
use tobii_proto::calibration::{EYES_BOTH, collect_payload, cmd, write_payload};
use tobii_proto::protocol::{RESPONSE_STATUS_OK, ttp_error};
use tobii_usb::engine::{Commands, DeviceCommand};

struct Session {
    path: PathBuf,
    /// The calibration active before this session; written back if it ends
    /// without being kept.
    previous: Vec<u8>,
    /// What the last compute produced, if any.
    computed: Option<Vec<u8>>,
}

fn session_slot() -> &'static Mutex<Option<Session>> {
    static SESSION: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
    SESSION.get_or_init(|| Mutex::new(None))
}

/// Run `cmd` and reduce its outcome to the payload or a one-line reason: an
/// answer succeeds only with the OK status and no TTP error, same test as
/// the daemon's own `run` (`tobiid::device::run`).
fn run(commands: &Commands, cmd: u32, payload: Vec<u8>, timeout: Duration) -> Result<Vec<u8>, String> {
    let response = commands
        .run(DeviceCommand { cmd, payload, timeout })
        .map_err(|e| format!("command {cmd}: {e}"))?;
    if response.status == RESPONSE_STATUS_OK && response.error == ttp_error::NONE {
        Ok(response.payload)
    } else {
        Err(format!(
            "command {cmd} refused: status={} error={:#x}",
            response.status, response.error
        ))
    }
}

/// The calibration to start from: the one saved at `path`, else the one
/// built into the init replay.
fn starting_blob(path: &std::path::Path) -> Result<Vec<u8>, String> {
    if let Ok(Some((blob, _))) = tobii_calib::store::load(path) {
        return Ok(blob);
    }
    tobii_usb::calibration::embedded_blob().map_err(|e| e.to_string())
}

/// Starts a session: 1010, 1060, 1110 with the starting blob. `storage_dir`
/// is where the computed calibration is saved and loaded from (Kotlin's
/// `context.filesDir`).
pub(crate) fn start(commands: &Commands, storage_dir: &str) -> Result<(), String> {
    let path = PathBuf::from(storage_dir).join("calibration.bin");
    let previous = starting_blob(&path)?;
    run(commands, cmd::START, Vec::new(), QUICK)?;
    run(commands, cmd::CLEAR, Vec::new(), QUICK)?;
    run(commands, cmd::WRITE, write_payload(&previous), SLOW)?;
    *session_slot().lock().unwrap() = Some(Session {
        path,
        previous,
        computed: None,
    });
    Ok(())
}

/// Collects one point (1030), normalised display coordinates, both eyes.
pub(crate) fn collect(commands: &Commands, x: f32, y: f32) -> Result<(), String> {
    run(commands, cmd::COLLECT_2D, collect_payload(x, y, EYES_BOTH), QUICK)?;
    Ok(())
}

/// Computes (1070) and reads the result back (1100); the blob must validate
/// to be kept. Returns the new calibration's id.
pub(crate) fn compute(commands: &Commands) -> Result<u32, String> {
    run(commands, cmd::COMPUTE, Vec::new(), SLOW)?;
    let payload = run(commands, cmd::READ, Vec::new(), QUICK)?;
    let blob = tobii_proto::calibration::blob_from_payload(&payload)
        .ok_or("compute: no blob in the read-back")?
        .to_vec();
    let info = tobii_calib::blob::validate(&blob).map_err(|e| format!("computed calibration: {e}"))?;
    let mut guard = session_slot().lock().unwrap();
    let session = guard.as_mut().ok_or("compute: no session running")?;
    session.computed = Some(blob);
    Ok(info.id)
}

/// Ends the session: 1020, then 1110 with the computed blob if `keep` (and
/// one was computed and validated), else the blob the session started from.
/// A kept blob is saved to the session's path. Returns whether one was
/// saved.
pub(crate) fn stop(commands: &Commands, keep: bool) -> Result<bool, String> {
    let Session {
        path,
        previous,
        computed,
    } = session_slot().lock().unwrap().take().ok_or("stop: no session running")?;
    let kept = keep.then_some(computed).flatten();
    let active = kept.as_ref().unwrap_or(&previous);
    run(commands, cmd::STOP, Vec::new(), QUICK)?;
    run(commands, cmd::WRITE, write_payload(active), SLOW)?;
    let Some(blob) = kept else { return Ok(false) };
    Ok(tobii_calib::store::save(&path, &blob).is_ok())
}
