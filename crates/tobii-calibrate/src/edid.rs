//! A monitor's physical size from its EDID, read from the kernel's DRM
//! connectors in sysfs: the starting point of the display setup, so the
//! ticks usually already sit on the tracker's marks.

use std::fs;

/// Where the kernel lists display connectors, as `card<N>-<connector>`.
const DRM_DIR: &str = "/sys/class/drm";

/// The size in millimetres (width, height) of the monitor on the connector
/// the window system calls `name`, among `monitors`. When the names cannot
/// be matched, the size is still known if every one of the monitors reports
/// the same one.
pub(crate) fn monitor_size_mm(name: Option<&str>, monitors: usize) -> Option<(f64, f64)> {
    let mut found = Vec::new();
    for entry in fs::read_dir(DRM_DIR).ok()?.flatten() {
        let file_name = entry.file_name();
        let Some((card, connector)) = file_name.to_str().and_then(|n| n.split_once('-')) else {
            continue;
        };
        if !card.starts_with("card") {
            continue;
        }
        // Disconnected connectors have an empty EDID.
        if let Some(size) = fs::read(entry.path().join("edid"))
            .ok()
            .as_deref()
            .and_then(size_from_edid)
        {
            found.push((connector.to_owned(), size));
        }
    }
    pick(name, &found, monitors)
}

/// The kernel names HDMI connectors `HDMI-A-<n>`, GNOME `HDMI-<n>`; the
/// other connector types are named alike.
fn same_connector(kernel: &str, name: &str) -> bool {
    kernel == name || kernel.replacen("HDMI-A-", "HDMI-", 1) == name
}

fn pick(name: Option<&str>, found: &[(String, (f64, f64))], monitors: usize) -> Option<(f64, f64)> {
    if let Some(name) = name
        && let Some((_, size)) = found.iter().find(|(c, _)| same_connector(c, name))
    {
        return Some(*size);
    }
    // A monitor with no EDID here could be the one: then nothing is known.
    if found.len() < monitors.max(1) {
        return None;
    }
    let (_, first) = found.first()?;
    found
        .iter()
        .all(|(_, s)| (s.0 - first.0).abs() < 1.0 && (s.1 - first.1).abs() < 1.0)
        .then_some(*first)
}

/// Sizes TVs and projectors put in their EDID for an aspect ratio, not a
/// size.
const PLACEHOLDERS: [(u16, u16); 2] = [(1600, 900), (1600, 1000)];

/// The image size an EDID declares: the preferred timing's size in mm, else
/// the basic block's size in cm; none for a known placeholder.
fn size_from_edid(edid: &[u8]) -> Option<(f64, f64)> {
    const HEADER: [u8; 8] = [0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0];
    if edid.get(..8)? != HEADER {
        return None;
    }
    let timing = edid.get(54..72)?;
    let is_timing = timing[0] != 0 || timing[1] != 0;
    let w = u16::from(timing[12]) | (u16::from(timing[14] & 0xf0) << 4);
    let h = u16::from(timing[13]) | (u16::from(timing[14] & 0x0f) << 8);
    let (w, h) = if is_timing && w > 0 && h > 0 {
        (w, h)
    } else {
        (u16::from(edid[21]) * 10, u16::from(edid[22]) * 10)
    };
    (w > 0 && h > 0 && !PLACEHOLDERS.contains(&(w, h))).then(|| (f64::from(w), f64::from(h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal EDID: header, basic size in cm, a preferred timing's size.
    fn edid(cm: (u8, u8), mm: (u16, u16)) -> Vec<u8> {
        let mut e = vec![0u8; 128];
        e[..8].copy_from_slice(&[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
        (e[21], e[22]) = cm;
        e[54] = 0x56; // a non-zero pixel clock: a timing, not a descriptor
        e[66] = u8::try_from(mm.0 & 0xff).expect("low byte");
        e[67] = u8::try_from(mm.1 & 0xff).expect("low byte");
        e[68] = u8::try_from(((mm.0 >> 4) & 0xf0) | ((mm.1 >> 8) & 0x0f)).expect("high bits");
        e
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: whole millimetres
    fn reads_the_preferred_timing_size_then_the_basic_one() {
        assert_eq!(
            size_from_edid(&edid((60, 34), (597, 336))),
            Some((597.0, 336.0))
        );
        assert_eq!(
            size_from_edid(&edid((60, 34), (0, 0))),
            Some((600.0, 340.0))
        );
        assert_eq!(size_from_edid(&edid((0, 0), (0, 0))), None);
        // Aspect ratios in the size fields.
        assert_eq!(size_from_edid(&edid((160, 90), (0, 0))), None);
        assert_eq!(size_from_edid(&edid((60, 34), (1600, 900))), None);
        assert_eq!(size_from_edid(&[0; 64]), None);
        assert_eq!(size_from_edid(&[]), None);
    }

    #[test]
    fn matches_gnome_names_to_kernel_connectors() {
        let found = vec![
            ("DP-3".to_owned(), (597.0, 336.0)),
            ("HDMI-A-4".to_owned(), (527.0, 296.0)),
        ];
        assert_eq!(pick(Some("HDMI-4"), &found, 2), Some((527.0, 296.0)));
        assert_eq!(pick(Some("HDMI-A-4"), &found, 2), Some((527.0, 296.0)));
        assert_eq!(pick(Some("DP-3"), &found, 2), Some((597.0, 336.0)));
        // Unknown name, monitors that differ: not guessed.
        assert_eq!(pick(Some("DP-1"), &found, 2), None);
        assert_eq!(pick(None, &found, 2), None);
        // Unknown name, all the same: that size.
        let same = vec![
            ("DP-3".to_owned(), (597.0, 336.0)),
            ("HDMI-A-4".to_owned(), (597.0, 336.0)),
        ];
        assert_eq!(pick(Some("XWAYLAND0"), &same, 2), Some((597.0, 336.0)));
        // ...unless some monitor has no EDID here: it could be the one.
        assert_eq!(pick(Some("HDMI-0"), &same[..1], 2), None);
        assert_eq!(pick(Some("DP-3"), &[], 1), None);
    }
}
