//! Tracker and display geometry shared by the daemon, `libtobii.so` and the
//! protocol decoders.
//!
//! Two frames appear on the wire. The **tracker frame** (S) has its origin at
//! the tracker, x to the user's right, y up the camera's tilted axis and z
//! towards the user; everything the device reports about itself (track box,
//! mounting, display area) is in S. The **display frame** (T) has its origin
//! at the centre of the screen, x right, y up the screen, z out of the screen
//! towards the user; the Stream Engine reports gaze origins in T. A display
//! area fixes T: [`DisplayFrame`] builds it and maps points between the two.
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

/// The top edge of a display area, and its height across that edge, must be
/// longer than this, mm, as tobiid requires of a saved area it loads.
const MIN_SIDE_MM: f64 = 1.0;

/// `side` scaled to unit length; `None` when it is `MIN_SIDE_MM` long or
/// shorter, not finite, or too long to square.
fn side_direction(side: [f64; 3]) -> Option<[f64; 3]> {
    let square = dot(side, side);
    // A NaN fails the comparison, an infinity the check.
    (square > MIN_SIDE_MM * MIN_SIDE_MM && square.is_finite())
        .then(|| scale(side, 1.0 / square.sqrt()))
}

/// The display frame (T) a display area fixes, and the rigid map between it
/// and the tracker frame (S).
///
/// The origin is the centre of the area, half-way from the bottom-left corner
/// to the top-right one. x runs along the top edge, from the top-left corner
/// to the top-right one; y is the left edge, from the bottom-left corner up to
/// the top-left one, less its part along x (Gram–Schmidt); z = x × y points
/// out of the screen, towards the user. A point maps as
/// `p_T = R * (p_S - centre)`, where the rows of R are those axes in S.
///
/// This is how the tracker converts its gaze origins: on the gaze frames of
/// the Windows captures and the Linux logs, made on two different areas, it
/// gives the display-frame origins the tracker sent (keys 0x22/0x24) from the
/// tracker-frame ones (0x02/0x08) to 0.0002 mm. Every area seen so far was a
/// rectangle. On a sheared one, taking the part along x out of the left edge
/// keeps the frame orthonormal; how the tracker treats such an area is not
/// known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayFrame {
    /// `R_T←S`: the display's x, y and z axes in the tracker frame, as rows.
    rotation: [[f64; 3]; 3],
    /// The centre of the area in the tracker frame, mm.
    centre: [f64; 3],
}

impl DisplayFrame {
    /// The frame `area` fixes, or `None` when it fixes none: a corner is not
    /// finite, or the area has no width or height to speak of, its top edge
    /// or its height across that edge being 1 mm or less. For a rectangle,
    /// that is the check tobiid makes of a display area it loads.
    #[must_use]
    pub fn new(area: &DisplayArea) -> Option<Self> {
        let DisplayArea {
            top_left_mm: tl,
            top_right_mm: tr,
            bottom_left_mm: bl,
        } = *area;
        // A corner that is not finite leaves an edge that is not finite
        // either, which side_direction refuses.
        let x = side_direction(sub(tr, tl))?;
        let left = sub(tl, bl);
        let y = side_direction(sub(left, scale(x, dot(left, x))))?;
        Some(Self {
            rotation: [x, y, cross(x, y)],
            // (tr + bl) / 2, halved first so that finite corners cannot
            // overflow.
            centre: add(scale(tr, 0.5), scale(bl, 0.5)),
        })
    }

    /// Map a point from the tracker frame (S) to the display frame (T), mm.
    #[must_use]
    pub fn to_display(self, p: [f64; 3]) -> [f64; 3] {
        let d = sub(p, self.centre);
        self.rotation.map(|axis| dot(axis, d))
    }

    /// Map a point from the display frame (T) to the tracker frame (S), mm:
    /// the inverse of [`to_display`](Self::to_display).
    #[must_use]
    pub fn to_tracker(self, p: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = self.rotation;
        add(
            self.centre,
            add(scale(x, p[0]), add(scale(y, p[1]), scale(z, p[2]))),
        )
    }

    /// The rotation from the tracker frame to the display frame, `R_T←S`. Its
    /// rows are the display's x, y and z axes in the tracker frame, so `R * v`
    /// turns a direction `v` from S into T.
    #[must_use]
    pub const fn rotation(&self) -> [[f64; 3]; 3] {
        self.rotation
    }

    /// The centre of the area in the tracker frame, mm: the display frame's
    /// origin.
    #[must_use]
    pub const fn centre(&self) -> [f64; 3] {
        self.centre
    }
}

/// Map a point from the display frame (T) of `area` to the tracker frame (S):
/// [`DisplayFrame::to_tracker`]. Every coordinate is NaN when the area fixes
/// no frame.
#[must_use]
pub fn display_to_tracker(area: &DisplayArea, p: [f64; 3]) -> [f64; 3] {
    DisplayFrame::new(area).map_or([f64::NAN; 3], |frame| frame.to_tracker(p))
}

/// Map a point from the tracker frame (S) to the display frame (T) of `area`:
/// [`DisplayFrame::to_display`]. Every coordinate is NaN when the area fixes
/// no frame.
#[must_use]
pub fn tracker_to_display(area: &DisplayArea, p: [f64; 3]) -> [f64; 3] {
    DisplayFrame::new(area).map_or([f64::NAN; 3], |frame| frame.to_display(p))
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

    /// The display area the Windows engine sent at init (command 1440, the
    /// author's monitor).
    fn captured_area() -> DisplayArea {
        DisplayArea {
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
        }
    }

    /// A 597 x 336 mm area tobiid saved on Linux, centred 0.65 mm left of
    /// the captured ones. No gaze frame was ever streamed on it.
    fn linux_area() -> DisplayArea {
        DisplayArea {
            top_left_mm: [
                -298.152_252_197_265_6,
                326.004_058_837_890_6,
                111.818_748_474_121_1,
            ],
            top_right_mm: [
                298.847_747_802_734_4,
                326.004_058_837_890_6,
                111.818_748_474_121_1,
            ],
            bottom_left_mm: [
                -298.152_252_197_265_6,
                10.267_330_169_677_734,
                -3.100_020_170_211_792,
            ],
        }
    }

    /// An area rolled 3° in its plane and sheared: its left edge is 1.1° off
    /// square to its top edge.
    fn sheared_area() -> DisplayArea {
        DisplayArea {
            top_left_mm: [-300.0, 330.0, 110.0],
            top_right_mm: [298.6, 361.3, 110.0],
            bottom_left_mm: [-290.0, 10.0, -3.0],
        }
    }

    /// `R_T←S` of a display turned about x by the angle of `cos` and `sin`.
    fn tilted(cos: f64, sin: f64) -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, cos, sin], [0.0, -sin, cos]]
    }

    fn close_rows(a: [[f64; 3]; 3], b: [[f64; 3]; 3], tol: f64) -> bool {
        a.into_iter().zip(b).all(|(x, y)| close(x, y, tol))
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

    /// The formula must reproduce all nine coordinates of the captured area
    /// from the monitor's size, the reported mounting and the one x offset
    /// both captured monitors share.
    #[test]
    fn reproduces_the_captured_display_area() {
        let captured = captured_area();
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

    /// The frame the tracker converted the Windows sessions' gaze origins
    /// with: turned 20° about x, and the tracker 175.7 mm below the screen's
    /// centre and 6.4 mm in front of it.
    #[test]
    fn the_captured_area_gives_the_frame_the_tracker_used() {
        let frame = DisplayFrame::new(&captured_area()).expect("a display frame");

        let (cos, sin) = (0.939_692_618_726_777_8, 0.342_020_148_983_083_36);
        assert!(
            close_rows(frame.rotation(), tilted(cos, sin), 1e-12),
            "{frame:?}"
        );
        // t = -R c, where the tracker is in the display frame.
        let t = frame.to_display([0.0; 3]);
        let want = [
            -1.002_258_300_781_25,
            -175.740_470_065_770_13,
            6.424_699_866_143_832,
        ];
        assert!(close(t, want, 1e-9), "{t:?}");
        // A head 60 cm in front of the tracker.
        let head = frame.to_display([0.0, 150.0, 600.0]);
        let want = [
            -1.002_258_300_781_25,
            170.425_512_133_096_56,
            518.937_248_754_748,
        ];
        assert!(close(head, want, 1e-9), "{head:?}");
    }

    #[test]
    fn the_linux_area_keeps_the_tilt_and_moves_the_centre() {
        let frame = DisplayFrame::new(&linux_area()).expect("a display frame");

        let (cos, sin) = (0.939_692_623_134_647_4, 0.342_020_136_872_561_05);
        assert!(
            close_rows(frame.rotation(), tilted(cos, sin), 1e-12),
            "{frame:?}"
        );
        let t = frame.to_display([0.0; 3]);
        let want = [
            -0.347_747_802_734_375,
            -176.587_868_978_383_6,
            6.424_699_755_465_593,
        ];
        assert!(close(t, want, 1e-9), "{t:?}");
    }

    /// Taking the left edge as y would leave it 1.1° off square to x here.
    #[test]
    fn a_sheared_area_still_gives_an_orthonormal_frame() {
        let frame = DisplayFrame::new(&sheared_area()).expect("a display frame");

        let r = frame.rotation();
        for (i, a) in r.into_iter().enumerate() {
            for (j, b) in r.into_iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot(a, b) - want).abs() < 1e-12, "rows {i}, {j}: {r:?}");
            }
        }
        let det = dot(cross(r[0], r[1]), r[2]);
        assert!((det - 1.0).abs() < 1e-12, "{det}");
        // As the study's reference implementation gives it.
        let want = [
            [0.998_635_744_185_910_7, 0.052_217_338_444_736_076, 0.0],
            [
                -0.049_239_064_085_348_65,
                0.941_677_436_469_319_3,
                0.332_895_058_858_749_04,
            ],
            [
                0.017_382_893_955_007_635,
                -0.332_440_904_839_219_4,
                0.942_963_880_425_666,
            ],
        ];
        assert!(close_rows(r, want, 1e-12), "{r:?}");
        let t = frame.to_display([0.0; 3]);
        let want = [
            -13.988_282_582_264_68,
            -192.420_573_753_905_22,
            11.194_339_936_621_416,
        ];
        assert!(close(t, want, 1e-9), "{t:?}");
    }

    /// In its own frame an area lies in the plane z = 0, centred on the
    /// origin, with its top edge along x and its top above its bottom; and
    /// the frame maps points both ways.
    #[test]
    fn an_area_lies_flat_in_its_frame() {
        let basic = display_area_basic(597.0, 336.0, 1.0, &captured_mounting());
        for area in [captured_area(), linux_area(), sheared_area(), basic] {
            let frame = DisplayFrame::new(&area).expect("a display frame");

            let centre = scale(add(area.top_right_mm, area.bottom_left_mm), 0.5);
            assert!(close(frame.centre(), centre, 1e-12), "{area:?}");
            let [tl, tr, bl] = [area.top_left_mm, area.top_right_mm, area.bottom_left_mm]
                .map(|corner| frame.to_display(corner));
            for corner in [tl, tr, bl] {
                assert!(
                    corner[2].abs() < 1e-9,
                    "{corner:?} off the plane of {area:?}"
                );
            }
            assert!((tl[1] - tr[1]).abs() < 1e-9 && tl[0] < tr[0], "{area:?}");
            assert!(bl[1] < tl[1], "{area:?}");
            for p in [[-57.9, 112.5, 618.3], [0.0; 3], [250.0, -40.0, 1200.0]] {
                let back = frame.to_tracker(frame.to_display(p));
                assert!(close(back, p, 1e-9), "{back:?} for {p:?} on {area:?}");
                let back = frame.to_display(frame.to_tracker(p));
                assert!(close(back, p, 1e-9), "{back:?} for {p:?} on {area:?}");
            }
        }
    }

    #[test]
    fn an_area_without_width_or_height_fixes_no_frame() {
        let good = DisplayArea {
            top_left_mm: [-50.0, 60.0, 0.0],
            top_right_mm: [50.0, 60.0, 0.0],
            bottom_left_mm: [-50.0, 0.0, 0.0],
        };
        let with = |change: fn(&mut DisplayArea)| {
            let mut area = good;
            change(&mut area);
            area
        };
        assert!(DisplayFrame::new(&good).is_some());

        for (what, area) in [
            ("all zeros", DisplayArea::default()),
            ("no width", with(|a| a.top_right_mm = a.top_left_mm)),
            ("1 mm wide", with(|a| a.top_right_mm = [-49.0, 60.0, 0.0])),
            ("no height", with(|a| a.bottom_left_mm = a.top_left_mm)),
            ("1 mm high", with(|a| a.bottom_left_mm = [-50.0, 59.0, 0.0])),
            // Corners in a line: a left edge 30 mm long, and no height.
            ("flat", with(|a| a.bottom_left_mm = [-80.0, 60.0, 0.0])),
            // Sheared: a left edge 30 mm long, 1 mm high across the top.
            (
                "1 mm high, sheared",
                with(|a| a.bottom_left_mm = [-80.0, 59.0, 0.0]),
            ),
            ("too wide to square", with(|a| a.top_right_mm[0] = 1e300)),
        ] {
            assert!(DisplayFrame::new(&area).is_none(), "{what}: {area:?}");
        }
        // A NaN or an infinity in any coordinate.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for i in 0..9 {
                let mut corners = [good.top_left_mm, good.top_right_mm, good.bottom_left_mm];
                corners.as_flattened_mut()[i] = bad;
                let [top_left_mm, top_right_mm, bottom_left_mm] = corners;
                let area = DisplayArea {
                    top_left_mm,
                    top_right_mm,
                    bottom_left_mm,
                };
                assert!(DisplayFrame::new(&area).is_none(), "{area:?}");
            }
        }

        // Just over 1 mm either way is enough.
        assert!(DisplayFrame::new(&with(|a| a.top_right_mm = [-48.99, 60.0, 0.0])).is_some());
        assert!(DisplayFrame::new(&with(|a| a.bottom_left_mm = [-50.0, 58.99, 0.0])).is_some());
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
        // An area that fixes no frame maps nowhere.
        let none = DisplayArea::default();
        assert!(tracker_to_display(&none, p).iter().all(|v| v.is_nan()));
        assert!(display_to_tracker(&none, p).iter().all(|v| v.is_nan()));
    }
}
