//! The saved display area: the monitor the tracker is mounted on.
//!
//! The Stream Engine keeps a display area set through
//! `tobii_set_display_area` across sessions, so the daemon does too: every
//! area a client sets and the device accepts is written to
//! `display-area` in the per-user Tobii directory, and loaded at start as the
//! area the init replay configures (instead of the capture author's
//! monitor). `TOBII_DISPLAY_MM` only applies while nothing is saved.
//!
//! The file is text, three corners in millimetres in the tracker's frame:
//!
//! ```text
//! top_left -298.5 324.4 111.2
//! top_right 298.5 324.4 111.2
//! bottom_left -298.5 10.3 -3.1
//! ```

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use tobii_ipc::geometry::DisplayArea;

/// File name in the per-user Tobii directory.
const FILE_NAME: &str = "display-area";

/// Where the display area is saved, if there is a per-user directory.
pub(crate) fn default_path() -> Option<PathBuf> {
    tobii_calib::store::config_dir().map(|dir| dir.join(FILE_NAME))
}

/// The file's text for `area`. Values print in their shortest exact form, so
/// they read back identical.
pub(crate) fn to_text(area: &DisplayArea) -> String {
    let mut s = String::from(
        "# The monitor the eye tracker is mounted on: corners in mm, tracker frame.\n\
         # Written by tobiid whenever a display area is set.\n",
    );
    for (name, p) in [
        ("top_left", area.top_left_mm),
        ("top_right", area.top_right_mm),
        ("bottom_left", area.bottom_left_mm),
    ] {
        let _ = writeln!(s, "{name} {} {} {}", p[0], p[1], p[2]);
    }
    s
}

/// Parse the file's text: each corner exactly once, finite, and a display
/// with a width and a height.
pub(crate) fn from_text(text: &str) -> Option<DisplayArea> {
    let (mut tl, mut tr, mut bl) = (None, None, None);
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut words = line.split_whitespace();
        let slot = match words.next()? {
            "top_left" => &mut tl,
            "top_right" => &mut tr,
            "bottom_left" => &mut bl,
            _ => return None,
        };
        let mut p = [0.0; 3];
        for v in &mut p {
            *v = words
                .next()?
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())?;
        }
        if words.next().is_some() || slot.replace(p).is_some() {
            return None;
        }
    }
    let area = DisplayArea {
        top_left_mm: tl?,
        top_right_mm: tr?,
        bottom_left_mm: bl?,
    };
    let span = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>();
    (span(area.top_left_mm, area.top_right_mm) > 1.0
        && span(area.top_left_mm, area.bottom_left_mm) > 1.0)
        .then_some(area)
}

/// Read the area saved at `path`; `Ok(None)` when there is none.
///
/// # Errors
///
/// When the file exists but cannot be read, or does not hold a display area.
pub(crate) fn load(path: &Path) -> io::Result<Option<DisplayArea>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    from_text(&text).map(Some).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "not a display area (three corners in mm)",
        )
    })
}

/// Save `area` at `path` atomically, keeping the previous file as `.prev`.
///
/// # Errors
///
/// When the directory or file cannot be written.
pub(crate) fn save(path: &Path, area: &DisplayArea) -> io::Result<()> {
    tobii_calib::store::write_atomic(path, to_text(area).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_ipc::geometry::{GeometryMounting, display_area_basic};

    fn area() -> DisplayArea {
        let mounting = GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 20.0,
            external_offset_mm: [0.0, -0.16, 13.85],
            internal_offset_mm: [0.0, 5.38, 9.86],
        };
        display_area_basic(597.0, 336.0, 1.002_26, &mounting)
    }

    #[test]
    fn the_text_reads_back_exactly() {
        let a = area();
        assert_eq!(from_text(&to_text(&a)), Some(a));
    }

    #[test]
    fn anything_but_three_finite_corners_is_rejected() {
        let good = "top_left 0 10 0\ntop_right 100 10 0\nbottom_left 0 0 0\n";
        assert!(from_text(good).is_some());
        for bad in [
            "top_left 0 10 0\ntop_right 100 10 0\n",
            "top_left 0 10 0\ntop_right 100 10 0\nbottom_left 0 0\n",
            "top_left 0 10 0\ntop_right 100 10 0\nbottom_left 0 0 0 1\n",
            "top_left 0 10 0\ntop_right 100 10 0\nbottom_left 0 0 NaN\n",
            "top_left 0 10 0\ntop_left 0 10 0\ntop_right 100 10 0\nbottom_left 0 0 0\n",
            "top_left 0 10 0\ntop_right 100 10 0\nbottom_left 0 0 0\ncentre 1 2 3\n",
            "top_left 0 0 0\ntop_right 0 0 0\nbottom_left 0 0 0\n",
        ] {
            assert_eq!(from_text(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn saves_and_loads_through_a_file() {
        let dir = std::env::temp_dir().join(format!("tobiid-display-{}", std::process::id()));
        let path = dir.join(FILE_NAME);
        assert_eq!(load(&path).expect("absent is fine"), None);
        save(&path, &area()).expect("save");
        assert_eq!(load(&path).expect("load"), Some(area()));
        fs::write(&path, "garbage").expect("write");
        assert!(load(&path).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
