//! Tracker and display geometry shared by the daemon, `libtobii.so` and the
//! protocol decoders.
//!
//! Two frames appear on the wire. The **tracker frame** (S) has its origin at
//! the tracker, x to the user's right, y up the camera's tilted axis and z
//! towards the user; everything the device reports about itself (track box,
//! mounting, display area) is in S. The **display frame** (T) has its origin
//! at the centre of the screen, x right, y up the screen, z out of the screen
//! towards the user; the Stream Engine reports gaze origins in T.
//!
//! All lengths are millimetres. The device's own unit is 1/1024 mm; the
//! conversion happens in the protocol decoder, never here.

/// The screen rectangle in the tracker frame: three corners, in the order and
/// meaning of the Stream Engine's `tobii_display_area_t`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DisplayArea {
    /// Top-left corner, mm.
    pub top_left_mm: [f64; 3],
    /// Top-right corner, mm.
    pub top_right_mm: [f64; 3],
    /// Bottom-left corner, mm.
    pub bottom_left_mm: [f64; 3],
}

/// The volume the tracker can see the user in: eight corners in the order of
/// the Stream Engine's `tobii_track_box_t` (front face, then back face; each
/// upper-right, upper-left, lower-left, lower-right). Tracker frame, mm.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TrackBox {
    /// The corners, front face first.
    pub corners_mm: [[f64; 3]; 8],
}

/// How the tracker is mounted on the display: the Stream Engine's
/// `tobii_geometry_mounting_t`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeometryMounting {
    /// Number of mounting guides.
    pub guides: i32,
    /// Width of the tracker, mm.
    pub width_mm: f64,
    /// Tilt of the tracker's camera axis relative to the screen, degrees.
    pub angle_deg: f64,
    /// Offset of the mounting point, as seen from outside the tracker, mm.
    pub external_offset_mm: [f64; 3],
    /// Offset of the camera origin inside the tracker, mm.
    pub internal_offset_mm: [f64; 3],
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    if n > 0.0 { scale(a, 1.0 / n) } else { a }
}

/// The display rectangle a screen of `width_mm` x `height_mm`, centred
/// `offset_x_mm` to the right of the tracker, occupies in the tracker frame:
/// the Stream Engine's `tobii_calculate_display_area_basic`.
///
/// The screen's bottom edge sits at `internal - Rx(angle) * external` (the
/// camera origin minus the rotated mounting offset), and the screen rises
/// along the camera's tilted y axis. Both displays in the Windows captures are
/// reproduced by this to well under a micrometre.
#[must_use]
pub fn display_area_basic(
    width_mm: f64,
    height_mm: f64,
    offset_x_mm: f64,
    mounting: &GeometryMounting,
) -> DisplayArea {
    let (sin, cos) = mounting.angle_deg.to_radians().sin_cos();
    let e = mounting.external_offset_mm;
    let rotated_external = [e[0], cos * e[1] - sin * e[2], sin * e[1] + cos * e[2]];
    let bottom_centre = sub(mounting.internal_offset_mm, rotated_external);
    let right = [1.0, 0.0, 0.0];
    let up = [0.0, cos, sin];
    let bottom_left = add(bottom_centre, scale(right, offset_x_mm - width_mm / 2.0));
    let top_left = add(bottom_left, scale(up, height_mm));
    DisplayArea {
        top_left_mm: top_left,
        top_right_mm: add(top_left, scale(right, width_mm)),
        bottom_left_mm: bottom_left,
    }
}

/// Orthonormal basis of the display frame, expressed in the tracker frame:
/// `(centre, right, up, normal)`.
fn display_basis(area: &DisplayArea) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let right = normalize(sub(area.top_right_mm, area.top_left_mm));
    let up = normalize(sub(area.top_left_mm, area.bottom_left_mm));
    let normal = cross(right, up);
    let centre = scale(add(area.bottom_left_mm, area.top_right_mm), 0.5);
    (centre, right, up, normal)
}

/// Map a point from the display frame (T) to the tracker frame (S).
#[must_use]
pub fn display_to_tracker(area: &DisplayArea, p: [f64; 3]) -> [f64; 3] {
    let (centre, right, up, normal) = display_basis(area);
    add(
        centre,
        add(
            scale(right, p[0]),
            add(scale(up, p[1]), scale(normal, p[2])),
        ),
    )
}

/// Map a point from the tracker frame (S) to the display frame (T).
#[must_use]
pub fn tracker_to_display(area: &DisplayArea, p: [f64; 3]) -> [f64; 3] {
    let (centre, right, up, normal) = display_basis(area);
    let d = sub(p, centre);
    [dot(d, right), dot(d, up), dot(d, normal)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signed 32.32 fixed point in the device's 1/1024 mm unit -> mm. Exact:
    /// every value here fits in 53 bits.
    #[allow(clippy::cast_precision_loss)] // reason: exact for these magnitudes
    fn mm(raw: u64) -> f64 {
        i64::from_ne_bytes(raw.to_ne_bytes()) as f64 / 4_294_967_296.0 / 1024.0
    }

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// The mounting the tracker reports (command 2110 in init.pcapng).
    fn captured_mounting() -> GeometryMounting {
        GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 20.0,
            external_offset_mm: [0.0, mm(0xffff_ff5c_28f5_c290), mm(0x0000_3766_6666_6666)],
            internal_offset_mm: [0.0, mm(0x0000_1585_1eb8_51eb), mm(0x0000_2770_a3d7_0a3d)],
        }
    }

    #[test]
    fn an_untilted_mount_gives_an_upright_rectangle() {
        let flat = GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 0.0,
            external_offset_mm: [0.0; 3],
            internal_offset_mm: [0.0; 3],
        };
        let area = display_area_basic(600.0, 300.0, 0.0, &flat);
        assert!(close(area.bottom_left_mm, [-300.0, 0.0, 0.0], 1e-12));
        assert!(close(area.top_left_mm, [-300.0, 300.0, 0.0], 1e-12));
        assert!(close(area.top_right_mm, [300.0, 300.0, 0.0], 1e-12));
    }

    /// The display area the Windows engine sent at init (command 1440, the
    /// author's monitor): the formula must reproduce all nine coordinates
    /// from the monitor's size, the reported mounting and the one x offset
    /// both captured monitors share.
    #[test]
    fn reproduces_the_captured_display_area() {
        let captured = DisplayArea {
            top_left_mm: [
                mm(0xfffb_10d5_4800_0000),
                mm(0x0005_11a5_5800_0000),
                mm(0x0001_bcf4_d600_0000),
            ],
            top_right_mm: [
                mm(0x0004_f72f_5800_0000),
                mm(0x0005_11a5_5800_0000),
                mm(0x0001_bcf4_d600_0000),
            ],
            bottom_left_mm: [
                mm(0xfffb_10d5_4800_0000),
                mm(0x0000_2911_bf00_0000),
                mm(0xffff_f399_9450_0000),
            ],
        };
        let width = captured.top_right_mm[0] - captured.top_left_mm[0];
        let height = dot(
            sub(captured.top_left_mm, captured.bottom_left_mm),
            sub(captured.top_left_mm, captured.bottom_left_mm),
        )
        .sqrt();
        let offset_x = 1026.3125 / 1024.0;

        let area = display_area_basic(width, height, offset_x, &captured_mounting());

        assert!(
            close(area.top_left_mm, captured.top_left_mm, 1e-4),
            "{area:?}"
        );
        assert!(close(area.top_right_mm, captured.top_right_mm, 1e-4));
        assert!(close(area.bottom_left_mm, captured.bottom_left_mm, 1e-4));
    }

    #[test]
    fn display_and_tracker_frames_are_inverse() {
        let area = display_area_basic(597.0, 336.0, 1.0, &captured_mounting());
        let p = [-57.9, 112.5, 618.3];
        let back = tracker_to_display(&area, display_to_tracker(&area, p));
        assert!(close(back, p, 1e-9), "{back:?}");
        // The screen centre is the display frame's origin.
        let centre = scale(add(area.bottom_left_mm, area.top_right_mm), 0.5);
        assert!(close(tracker_to_display(&area, centre), [0.0; 3], 1e-9));
    }
}
