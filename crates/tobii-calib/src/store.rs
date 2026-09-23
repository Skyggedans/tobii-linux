//! The per-user calibration file.
//!
//! The daemon saves every calibration it computes or is given, and the engine
//! uploads the saved one at init instead of the calibration embedded in its
//! replay (which is the author's). Where it lives:
//!
//! 1. `$TOBII_CALIBRATION` — a path, or `embedded` to ignore any saved file;
//! 2. `$XDG_CONFIG_HOME/tobii/calibration.bin`;
//! 3. `$HOME/.config/tobii/calibration.bin`.
//!
//! Each save keeps the previous file as `calibration.bin.prev`, so a bad
//! calibration is one `mv` away from being undone.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::blob::{self, BlobError, BlobInfo};

/// Environment variable overriding the location (`embedded` disables it).
pub const ENV_OVERRIDE: &str = "TOBII_CALIBRATION";
/// Value of [`ENV_OVERRIDE`] that means "use the calibration built into the
/// init replay".
pub const EMBEDDED: &str = "embedded";

/// Where the calibration comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// The calibration embedded in the init replay.
    Embedded,
    /// A user file (which may not exist yet).
    File(PathBuf),
}

/// The configured location, from the environment.
#[must_use]
pub fn configured() -> Location {
    locate(
        std::env::var_os(ENV_OVERRIDE),
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    )
}

fn locate(overridden: Option<OsString>, xdg: Option<OsString>, home: Option<OsString>) -> Location {
    let non_empty = |v: Option<OsString>| v.filter(|s| !s.is_empty());
    if let Some(value) = non_empty(overridden) {
        return if value == EMBEDDED {
            Location::Embedded
        } else {
            Location::File(PathBuf::from(value))
        };
    }
    let config = non_empty(xdg)
        .map(PathBuf::from)
        .or_else(|| non_empty(home).map(|h| PathBuf::from(h).join(".config")));
    config.map_or(Location::Embedded, |dir| {
        Location::File(dir.join("tobii").join("calibration.bin"))
    })
}

/// The rollback copy kept next to `path`.
#[must_use]
pub fn previous_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(OsString::from).unwrap_or_default();
    name.push(".prev");
    path.with_file_name(name)
}

/// Why the calibration file could not be used.
#[derive(Debug)]
#[non_exhaustive]
pub enum StoreError {
    /// The file could not be read or written.
    Io(io::Error),
    /// The file holds something that is not a calibration.
    Invalid(BlobError),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "calibration file: {e}"),
            Self::Invalid(e) => write!(f, "calibration file: {e}"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Invalid(e) => Some(e),
        }
    }
}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<BlobError> for StoreError {
    fn from(e: BlobError) -> Self {
        Self::Invalid(e)
    }
}

/// Read and validate the file at `path`; `Ok(None)` when there is none.
///
/// # Errors
///
/// [`StoreError::Io`] when it exists but cannot be read,
/// [`StoreError::Invalid`] when it is not a calibration blob.
pub fn load(path: &Path) -> Result<Option<(Vec<u8>, BlobInfo)>, StoreError> {
    let blob = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let info = blob::validate(&blob)?;
    Ok(Some((blob, info)))
}

/// Save `blob` to `path` atomically, keeping the file it replaces as
/// [`previous_path`]. The blob is validated first, so a bad one never
/// displaces a good one.
///
/// # Errors
///
/// [`StoreError::Invalid`] for a bad blob, [`StoreError::Io`] when the
/// directory or file cannot be written.
pub fn save(path: &Path, blob: &[u8]) -> Result<BlobInfo, StoreError> {
    let info = blob::validate(blob)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let mut tmp_name = path.file_name().map(OsString::from).unwrap_or_default();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(blob)?;
        file.sync_all()?;
    }
    if path.exists() {
        fs::rename(path, previous_path(path))?;
    }
    fs::rename(&tmp, path)?;
    // Make the renames durable; failing to sync a directory is not fatal.
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(info)
}

/// Remove the calibration at `path` (the `.prev` copy stays). Absent is fine.
///
/// # Errors
///
/// [`StoreError::Io`] when it exists and cannot be removed.
pub fn remove(path: &Path) -> Result<(), StoreError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> Option<OsString> {
        Some(OsString::from(s))
    }

    #[test]
    fn location_precedence() {
        assert_eq!(
            locate(os("embedded"), os("/x"), os("/h")),
            Location::Embedded
        );
        assert_eq!(
            locate(os("/tmp/c.bin"), os("/x"), os("/h")),
            Location::File("/tmp/c.bin".into())
        );
        assert_eq!(
            locate(None, os("/x"), os("/h")),
            Location::File("/x/tobii/calibration.bin".into())
        );
        assert_eq!(
            locate(os(""), os(""), os("/h")),
            Location::File("/h/.config/tobii/calibration.bin".into())
        );
        assert_eq!(locate(None, None, None), Location::Embedded);
    }

    /// A scratch directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("tobii-calib-{tag}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn save_load_rotate_remove() {
        let scratch = Scratch::new("store");
        let path = scratch.0.join("tobii").join("calibration.bin");
        let first = crate::blob::tests::embedded_blob();
        let mut second = first.clone();
        second[20..24].copy_from_slice(&42u32.to_le_bytes());

        assert!(load(&path).expect("absent is fine").is_none());
        assert_eq!(save(&path, &first).expect("save").id, 1_904_654_973);
        assert_eq!(save(&path, &second).expect("save").id, 42);

        let (loaded, info) = load(&path).expect("load").expect("present");
        assert_eq!((loaded, info.id), (second, 42));
        assert_eq!(fs::read(previous_path(&path)).expect("prev"), first);

        remove(&path).expect("remove");
        assert!(load(&path).expect("load").is_none());
        assert!(previous_path(&path).exists(), "the rollback copy stays");
    }

    #[test]
    fn a_bad_blob_neither_saves_nor_loads() {
        let scratch = Scratch::new("bad");
        let path = scratch.0.join("calibration.bin");
        assert!(matches!(
            save(&path, &[0u8; 100]),
            Err(StoreError::Invalid(_))
        ));
        assert!(!path.exists());
        fs::create_dir_all(&scratch.0).expect("dir");
        fs::write(&path, [0u8; 100]).expect("write");
        assert!(matches!(load(&path), Err(StoreError::Invalid(_))));
    }
}
