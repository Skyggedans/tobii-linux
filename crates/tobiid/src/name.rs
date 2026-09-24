//! The device name: a nickname a client gives the tracker, kept by the host.
//!
//! The Stream Engine writes a name set through `tobii_set_device_name` to the
//! tracker (command 1710), which was never captured on the ET5; the daemon
//! keeps it instead, in `device-name` in the per-user Tobii directory, and
//! nothing is sent to the device. Until a name is set, the device's model
//! stands in for it.
//!
//! The file holds the name's bytes, at most [`DEVICE_NAME_MAX`] of them and
//! not necessarily UTF-8, then a newline. One trailing newline is dropped on
//! load, so a file written with `echo` or an editor names the device without
//! it, and a name that itself ends in a newline survives the round trip. A
//! name is set, saved, then answered under one lock, so two clients naming
//! the device at once leave the file and the answer agreeing.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use tobii_ipc::request::{DEVICE_NAME_MAX, status};
use tracing::{info, warn};

use crate::daemon::{State, lock_state};
use crate::requests::Reply;

/// File name in the per-user Tobii directory.
const FILE_NAME: &str = "device-name";

/// Where the name is saved, if there is a per-user directory.
pub(crate) fn default_path() -> Option<PathBuf> {
    tobii_calib::store::config_dir().map(|dir| dir.join(FILE_NAME))
}

/// A name as the device would keep it: what comes before the first NUL, at
/// most [`DEVICE_NAME_MAX`] bytes. The cut may split a UTF-8 character, as
/// the Stream Engine's does.
pub(crate) fn normalise(name: &[u8]) -> &[u8] {
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    &name[..end.min(DEVICE_NAME_MAX)]
}

/// Read the name saved at `path`; `Ok(None)` when there is none. One
/// trailing newline is dropped; an empty file is an empty name.
///
/// # Errors
///
/// When the file exists but cannot be read.
pub(crate) fn load(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => {
            let name = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
            Ok(Some(normalise(name).to_vec()))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Save `name` and a newline at `path` atomically, keeping the previous file
/// as `.prev`.
///
/// # Errors
///
/// When the directory or file cannot be written.
pub(crate) fn save(path: &Path, name: &[u8]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(name.len() + 1);
    bytes.extend_from_slice(name);
    bytes.push(b'\n');
    tobii_calib::store::write_atomic(path, &bytes)
}

/// At start: use the saved name, if any.
pub(crate) fn restore(st: &mut State) {
    let Some(path) = st.name_file.clone() else {
        return;
    };
    match load(&path) {
        Ok(Some(name)) => {
            info!(path = %path.display(), len = name.len(), "device name: using the saved one");
            st.device_name = Some(name);
        }
        Ok(None) => {}
        Err(e) => warn!(path = %path.display(), error = %e, "ignoring the saved device name"),
    }
}

/// The name a client set, at once; else the model the device reported,
/// waiting for its facts as the identity requests do.
pub(crate) fn get(state: &Mutex<State>, client: u64) -> Reply {
    if let Some(name) = lock_state(state).device_name.clone() {
        return Reply::ok(name);
    }
    crate::requests::facts(state, client, |f| {
        Some(normalise(f.info.model.as_bytes()).to_vec())
    })
}

/// Name the device. The name is saved first and answered only once saved:
/// a save failure is `OPERATION_FAILED` and leaves the old name. The same
/// name again is not written (it would push the rollback copy out).
pub(crate) fn set(state: &Mutex<State>, name: &[u8]) -> Reply {
    let name = normalise(name);
    let (lock, file) = {
        let st = lock_state(state);
        (Arc::clone(&st.name_lock), st.name_file.clone())
    };
    // Held from the save to the update below, never with the state lock
    // across the write.
    let _named = lock.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(path) = file.filter(|p| load(p).ok().flatten().as_deref() != Some(name)) {
        if let Err(e) = save(&path, name) {
            warn!(path = %path.display(), error = %e, "could not save the device name");
            return Reply::err(status::OPERATION_FAILED);
        }
        info!(path = %path.display(), len = name.len(), "device name saved");
    }
    lock_state(state).device_name = Some(name.to_vec());
    Reply::ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::handle;
    use tobii_ipc::request::{Request, kind};
    use tobii_proto::facts::DeviceFacts;

    const MODEL: &str = "IS5_Large_Eyetracker_5";

    /// A fresh directory for one test.
    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tobiid-name-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// A state whose device reported `MODEL`, with a stand-in device so that
    /// no engine starts, saving names at `file`.
    fn state_with_model(file: Option<PathBuf>) -> Mutex<State> {
        let mut st = crate::daemon::tests::state_with_client(1);
        st.fake_device = Some(Arc::new(crate::requests::tests::Answering(1)));
        st.facts = Some(Arc::new(DeviceFacts {
            info: tobii_ipc::request::DeviceInfo {
                model: MODEL.into(),
                ..Default::default()
            },
            ..DeviceFacts::default()
        }));
        st.name_file = file;
        Mutex::new(st)
    }

    fn ask(state: &Mutex<State>, kind: u8, payload: &[u8]) -> Reply {
        handle(
            state,
            1,
            &Request {
                id: 1,
                kind,
                payload,
            },
        )
    }

    #[test]
    fn a_name_is_cut_at_its_nul_and_to_63_bytes() {
        assert_eq!(normalise(b"Desk"), b"Desk");
        assert_eq!(normalise(b"Desk\0junk"), b"Desk");
        assert_eq!(normalise(b"\0Desk"), b"");
        assert_eq!(normalise(&[b'x'; 100]), &[b'x'; 63][..]);
        assert_eq!(normalise(&[0xff, 0xfe]), &[0xff, 0xfe]);
    }

    #[test]
    fn saves_and_loads_through_a_file() {
        let dir = temp_dir("file");
        let path = dir.join(FILE_NAME);
        assert_eq!(load(&path).expect("absent is fine"), None);

        fs::create_dir_all(&dir).expect("dir");
        fs::write(&path, b"").expect("write");
        assert_eq!(load(&path).expect("empty"), Some(Vec::new()));

        // Raw bytes, not necessarily UTF-8; the one before stays as `.prev`.
        save(&path, b"Desk").expect("save");
        save(&path, &[0xff, 0xfe, b'x']).expect("save");
        assert_eq!(load(&path).expect("load"), Some(vec![0xff, 0xfe, b'x']));
        let prev = tobii_calib::store::previous_path(&path);
        assert_eq!(load(&prev).expect("prev"), Some(b"Desk".to_vec()));

        // The file ends in a newline; a name that does keeps its own.
        assert_eq!(fs::read(&path).expect("read"), [0xff, 0xfe, b'x', b'\n']);
        save(&path, b"Desk\n").expect("save");
        assert_eq!(load(&path).expect("load"), Some(b"Desk\n".to_vec()));

        // A hand-made file is read as a set name would be kept: its newline
        // (`echo`, an editor) dropped, cut to 63 bytes.
        fs::write(&path, b"Desk\n").expect("write");
        assert_eq!(load(&path).expect("echoed"), Some(b"Desk".to_vec()));
        fs::write(&path, b"Desk").expect("write");
        assert_eq!(load(&path).expect("printed"), Some(b"Desk".to_vec()));
        fs::write(&path, [b'y'; 100]).expect("write");
        assert_eq!(load(&path).expect("long"), Some(vec![b'y'; 63]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_model_stands_in_until_a_name_is_set() {
        let dir = temp_dir("set");
        let path = dir.join(FILE_NAME);
        let state = state_with_model(Some(path.clone()));

        assert_eq!(
            ask(&state, kind::DEVICE_NAME_GET, &[]),
            Reply::ok(MODEL.into())
        );
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Desk"),
            Reply::ok(Vec::new())
        );
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_GET, &[]),
            Reply::ok(b"Desk".to_vec())
        );
        assert_eq!(load(&path).expect("load"), Some(b"Desk".to_vec()));

        // The same name again leaves the rollback copy alone.
        let prev = tobii_calib::store::previous_path(&path);
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Lab").status,
            status::OK
        );
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Lab").status,
            status::OK
        );
        assert_eq!(load(&path).expect("load"), Some(b"Lab".to_vec()));
        assert_eq!(load(&prev).expect("prev"), Some(b"Desk".to_vec()));

        // Kept for the next start.
        let mut st = crate::daemon::tests::state_with_client(1);
        st.name_file = Some(path.clone());
        restore(&mut st);
        assert_eq!(st.device_name, Some(b"Lab".to_vec()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_set_name_is_kept_as_the_device_would_keep_it() {
        let dir = temp_dir("cut");
        let state = state_with_model(Some(dir.join(FILE_NAME)));
        let get = || ask(&state, kind::DEVICE_NAME_GET, &[]);

        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, &[b'x'; 100]).status,
            status::OK
        );
        assert_eq!(get(), Reply::ok(vec![b'x'; 63]));
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Desk\0junk").status,
            status::OK
        );
        assert_eq!(get(), Reply::ok(b"Desk".to_vec()));
        // An empty name is a name, not the model.
        assert_eq!(ask(&state, kind::DEVICE_NAME_SET, &[]).status, status::OK);
        assert_eq!(get(), Reply::ok(Vec::new()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_name_that_cannot_be_saved_is_refused_and_the_old_one_stays() {
        let dir = temp_dir("fail");
        fs::create_dir_all(&dir).expect("dir");
        // The directory to save in cannot be made: a file has its name.
        let blocker = dir.join("not-a-directory");
        fs::write(&blocker, b"").expect("write");
        let state = state_with_model(Some(blocker.join(FILE_NAME)));

        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Desk"),
            Reply::err(status::OPERATION_FAILED)
        );
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_GET, &[]),
            Reply::ok(MODEL.into())
        );
        // A name set before stays too.
        lock_state(&state).device_name = Some(b"Old".to_vec());
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Desk"),
            Reply::err(status::OPERATION_FAILED)
        );
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_GET, &[]),
            Reply::ok(b"Old".to_vec())
        );

        // With no per-user directory the name lives in memory only.
        let state = state_with_model(None);
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_SET, b"Desk").status,
            status::OK
        );
        assert_eq!(
            ask(&state, kind::DEVICE_NAME_GET, &[]),
            Reply::ok(b"Desk".to_vec())
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
