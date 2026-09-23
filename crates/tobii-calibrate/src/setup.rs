//! The display setup: two ticks on screen, lined up with the two guide marks
//! on the tracker's front, give the monitor's size and the tracker's offset
//! from its centre — the inputs of the Stream Engine's
//! `tobii_calculate_display_area_basic`.
//!
//! The ticks' midpoint is the tracker's centre, which gives the offset. The
//! monitor's size comes from its EDID when that matches the window (its
//! aspect ratio, so not a letterboxed mode): the ticks are then held the
//! marks' distance apart at the EDID's pixel pitch, and only move together.
//! Lining them up by eye would do worse: the marks sit in front of the screen
//! and look further apart than they are. Without a usable EDID the spacing is
//! set by hand too: the marks are the mounting's `width_mm` apart, so the
//! spacing in pixels is the pixel pitch, and the fullscreen window's size
//! times the pitch is the monitor's (square pixels).

use tobii_ipc::geometry::DisplayArea;

/// Width to start from when neither the EDID nor the tracker knows one: a
/// 24" 16:9 monitor.
const FALLBACK_WIDTH_MM: f64 = 531.0;

/// The ticks never come closer than this fraction of the window width.
const MIN_SPACING: f64 = 0.05;

/// How far the EDID's aspect ratio may be from the window's for its size to
/// be the window's.
const ASPECT_TOLERANCE: f64 = 0.02;

/// A believable monitor is wider than this many times the distance between
/// the tracker's marks (a 15" laptop is twice as wide as the Eye Tracker 5's
/// marks), and narrower than [`MAX_MONITOR_MM`]. EDIDs of TVs and projectors
/// often hold an aspect ratio (160 x 90) or a size in cm instead.
const MIN_MONITOR_OVER_GUIDE: f64 = 1.5;
const MAX_MONITOR_MM: f64 = 3000.0;

/// What the setup starts from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Start {
    /// Distance between the tracker's two guide marks, mm.
    pub(crate) guide_mm: f64,
    /// The monitor's size (width, height) from its EDID, as it reports it.
    pub(crate) monitor_mm: Option<(f64, f64)>,
    /// The display area the tracker has now.
    pub(crate) current: Option<DisplayArea>,
}

/// The monitor and where the tracker sits under it, as
/// `tobii_calculate_display_area_basic` takes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Choice {
    pub(crate) width_mm: f64,
    pub(crate) height_mm: f64,
    /// How far right of the tracker's centre the monitor's centre is.
    pub(crate) offset_x_mm: f64,
}

/// The two ticks, in window pixels from the left edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ticks {
    pub(crate) left: f64,
    pub(crate) right: f64,
}

/// Orient an EDID size like a `width_px × height_px` window (a monitor
/// turned to portrait still reports its landscape size).
pub(crate) fn oriented(size_mm: (f64, f64), width_px: f64, height_px: f64) -> (f64, f64) {
    let (w, h) = size_mm;
    if (w > h) == (width_px > height_px) {
        (w, h)
    } else {
        (h, w)
    }
}

fn span(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}

impl Start {
    /// The EDID's size if it is the size of a `width_px × height_px`
    /// fullscreen window: believable, oriented like it, and of the same
    /// aspect ratio.
    pub(crate) fn monitor_for(&self, width_px: f64, height_px: f64) -> Option<(f64, f64)> {
        let (w, h) = oriented(self.monitor_mm?, width_px, height_px);
        let aspect_px = width_px / height_px;
        let wide = w.max(h);
        let believable = wide >= self.guide_mm * MIN_MONITOR_OVER_GUIDE && wide <= MAX_MONITOR_MM;
        (believable && (w / h - aspect_px).abs() <= aspect_px * ASPECT_TOLERANCE).then_some((w, h))
    }

    /// The ticks keep the marks' distance and only move together: the
    /// EDID gives the size.
    pub(crate) fn spacing_fixed(&self, width_px: f64, height_px: f64) -> bool {
        self.monitor_for(width_px, height_px).is_some()
    }

    /// Where the ticks start: spaced by the EDID's pixel pitch (or the
    /// tracker's current area), centred where the current area puts the
    /// tracker.
    pub(crate) fn ticks(&self, width_px: f64, height_px: f64) -> Ticks {
        let width_mm = self
            .monitor_for(width_px, height_px)
            .map(|(w, _)| w)
            .or_else(|| {
                self.current
                    .map(|a| span(a.top_left_mm, a.top_right_mm))
                    .filter(|w| *w > 0.0)
            })
            .unwrap_or(FALLBACK_WIDTH_MM);
        let pitch = width_mm / width_px;
        // The current area's horizontal centre is the monitor's offset from
        // the tracker (the tracker sits at x = 0).
        let offset_mm = self
            .current
            .map_or(0.0, |a| (a.top_left_mm[0] + a.top_right_mm[0]) / 2.0);
        let centre = width_px / 2.0 - offset_mm / pitch;
        let half = self.guide_mm / pitch / 2.0;
        self.fit(
            Ticks {
                left: centre - half,
                right: centre + half,
            },
            width_px,
            height_px,
        )
    }

    /// `ticks` made valid for a `width_px × height_px` window: with the
    /// EDID's size, the marks' distance apart around their middle (so a
    /// window that changed monitor or orientation gets the right spacing
    /// back), and the pair kept whole on screen; otherwise just on screen, in
    /// order and apart.
    pub(crate) fn fit(&self, ticks: Ticks, width_px: f64, height_px: f64) -> Ticks {
        match self.monitor_for(width_px, height_px) {
            Some((w, _)) => {
                let half = self.guide_mm / (w / width_px) / 2.0;
                Ticks::centred((ticks.left + ticks.right) / 2.0, half, width_px)
            }
            None => ticks.kept_in(width_px),
        }
    }

    /// What the ticks say, on a `width_px × height_px` fullscreen window.
    pub(crate) fn choice(&self, ticks: Ticks, width_px: f64, height_px: f64) -> Option<Choice> {
        let (width_mm, height_mm, pitch) = match self.monitor_for(width_px, height_px) {
            Some((w, h)) => (w, h, w / width_px),
            None => {
                let spacing = ticks.right - ticks.left;
                if spacing < 1.0 || !self.guide_mm.is_finite() || self.guide_mm <= 0.0 {
                    return None;
                }
                let pitch = self.guide_mm / spacing;
                (width_px * pitch, height_px * pitch, pitch)
            }
        };
        Some(Choice {
            width_mm,
            height_mm,
            offset_x_mm: (width_px / 2.0 - (ticks.left + ticks.right) / 2.0) * pitch,
        })
    }
}

impl Ticks {
    /// A pair `half` either side of `middle`, moved whole inside the window.
    fn centred(middle: f64, half: f64, width_px: f64) -> Self {
        let half = half.min(width_px / 2.0);
        let middle = middle.clamp(half, width_px - half);
        Self {
            left: middle - half,
            right: middle + half,
        }
    }

    /// Both moved by `dx`, as far as the window allows.
    pub(crate) fn shifted(self, dx: f64, width_px: f64) -> Self {
        let dx = dx.clamp(-self.left, width_px - self.right);
        Self {
            left: self.left + dx,
            right: self.right + dx,
        }
    }

    /// Spread apart by `d` (closer for a negative `d`), about their middle.
    pub(crate) fn widened(self, d: f64, width_px: f64) -> Self {
        Self {
            left: self.left - d / 2.0,
            right: self.right + d / 2.0,
        }
        .kept_in(width_px)
    }

    /// The left tick moved to `x`.
    pub(crate) fn with_left(self, x: f64, width_px: f64) -> Self {
        Self { left: x, ..self }.kept_in(width_px)
    }

    /// The right tick moved to `x`.
    pub(crate) fn with_right(self, x: f64, width_px: f64) -> Self {
        Self { right: x, ..self }.kept_in(width_px)
    }

    /// Inside the window, in order, and not closer than [`MIN_SPACING`].
    fn kept_in(self, width_px: f64) -> Self {
        let min = width_px * MIN_SPACING;
        let mut left = self.left.clamp(0.0, width_px - min);
        let mut right = self.right.clamp(min, width_px);
        if right - left < min {
            let middle = ((left + right) / 2.0).clamp(min / 2.0, width_px - min / 2.0);
            left = middle - min / 2.0;
            right = middle + min / 2.0;
        }
        Self { left, right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_ipc::geometry::{GeometryMounting, display_area_basic};

    const W: f64 = 3840.0;
    const H: f64 = 2160.0;

    fn mounting() -> GeometryMounting {
        GeometryMounting {
            guides: 2,
            width_mm: 184.0,
            angle_deg: 20.0,
            external_offset_mm: [0.0, -0.16, 13.85],
            internal_offset_mm: [0.0, 5.38, 9.86],
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn with_edid(monitor_mm: (f64, f64)) -> Start {
        Start {
            guide_mm: 184.0,
            monitor_mm: Some(monitor_mm),
            current: None,
        }
    }

    #[test]
    fn untouched_ticks_give_back_the_monitor_and_the_current_offset() {
        let start = Start {
            current: Some(display_area_basic(633.6, 334.3, 1.0, &mounting())),
            ..with_edid((597.0, 336.0))
        };
        let t = start.ticks(W, H);
        let c = start.choice(t, W, H).expect("choice");
        assert!(
            close(c.width_mm, 597.0) && close(c.height_mm, 336.0),
            "{c:?}"
        );
        assert!(close(c.offset_x_mm, 1.0), "{c:?}");
        // The ticks are the guides' 184 mm apart at the EDID's pitch.
        assert!(close((t.right - t.left) * 597.0 / W, 184.0), "{t:?}");
    }

    #[test]
    fn with_an_edid_the_ticks_move_the_offset_and_the_size_stays() {
        let start = with_edid((597.0, 336.0));
        assert!(start.spacing_fixed(W, H));
        let t = start.ticks(W, H);
        let pitch = 597.0 / W;
        // The tracker 100 px left of the centre: the monitor's centre is
        // 100 px right of the tracker.
        let c = start.choice(t.shifted(-100.0, W), W, H).expect("choice");
        assert!(close(c.offset_x_mm, 100.0 * pitch), "{c:?}");
        assert!(
            close(c.width_mm, 597.0) && close(c.height_mm, 336.0),
            "{c:?}"
        );
        // Spread ticks (should the window let them be) do not resize it.
        let spread = start.choice(t.widened(40.0, W), W, H).expect("choice");
        assert!(close(spread.width_mm, 597.0), "{spread:?}");
    }

    #[test]
    fn without_a_usable_edid_the_spacing_sets_the_size_in_square_pixels() {
        // A 16:10 panel running a 16:9 mode (letterboxed): its EDID size is
        // not the picture's.
        let start = with_edid((518.0, 324.0));
        assert!(!start.spacing_fixed(W, H));
        let t = start.ticks(W, H);
        let c = start.choice(t, W, H).expect("choice");
        assert!(close(c.height_mm, c.width_mm * H / W), "{c:?}");
        let wider = start.choice(t.widened(40.0, W), W, H).expect("choice");
        assert!(wider.width_mm < c.width_mm && wider.height_mm < c.height_mm);
        assert!(close(wider.width_mm / wider.height_mm, W / H));
    }

    #[test]
    fn without_an_edid_the_current_area_places_the_ticks() {
        let start = Start {
            guide_mm: 184.0,
            monitor_mm: None,
            current: Some(display_area_basic(633.6, 334.3, 0.0, &mounting())),
        };
        let c = start.choice(start.ticks(W, H), W, H).expect("choice");
        assert!(close(c.width_mm, 633.6), "{c:?}");
        assert!(close(c.height_mm, 633.6 * H / W), "{c:?}");
        let blind = Start {
            current: None,
            ..start
        };
        let c = blind.choice(blind.ticks(W, H), W, H).expect("choice");
        assert!(close(c.width_mm, FALLBACK_WIDTH_MM) && close(c.offset_x_mm, 0.0));
    }

    #[test]
    fn a_portrait_window_turns_the_edid_size() {
        let start = with_edid((597.0, 336.0));
        assert_eq!(start.monitor_for(H, W), Some((336.0, 597.0)));
        let c = start.choice(start.ticks(H, W), H, W).expect("choice");
        assert!(
            close(c.width_mm, 336.0) && close(c.height_mm, 597.0),
            "{c:?}"
        );
    }

    #[test]
    fn the_ticks_stay_on_screen_in_order_and_apart() {
        let t = with_edid((597.0, 336.0)).ticks(W, H);
        let spacing = t.right - t.left;
        let far = t.shifted(-10_000.0, W);
        assert!(close(far.left, 0.0) && close(far.right - far.left, spacing));
        let crossed = t.with_left(W, W);
        assert!(crossed.left < crossed.right && crossed.right <= W);
        assert!(crossed.right - crossed.left >= W * MIN_SPACING - 1e-9);
        let squeezed = t.widened(-10_000.0, W);
        assert!(squeezed.right - squeezed.left >= W * MIN_SPACING - 1e-9);
    }

    #[test]
    fn an_unbelievable_edid_leaves_the_spacing_free() {
        for bogus in [(160.0, 90.0), (60.0, 34.0), (16_000.0, 9_000.0)] {
            let start = with_edid(bogus);
            assert!(!start.spacing_fixed(W, H), "{bogus:?}");
            let t = start.ticks(W, H);
            assert!(t.left > 0.0 && t.right < W, "{bogus:?}: {t:?}");
        }
    }

    #[test]
    fn a_fixed_pair_stays_whole_at_the_edge() {
        // The current area puts the tracker far left: the pair is moved in,
        // not squeezed.
        let start = Start {
            current: Some(display_area_basic(597.0, 336.0, 400.0, &mounting())),
            ..with_edid((597.0, 336.0))
        };
        let t = start.ticks(W, H);
        assert!(close(t.left, 0.0), "{t:?}");
        assert!(close((t.right - t.left) * 597.0 / W, 184.0), "{t:?}");
    }

    #[test]
    fn a_window_that_turned_gets_the_spacing_of_its_new_shape() {
        let start = with_edid((597.0, 336.0));
        // Placed on a portrait window, then fitted to the landscape one.
        let portrait = start.ticks(H, W);
        let scaled = Ticks {
            left: portrait.left * W / H,
            right: portrait.right * W / H,
        };
        let t = start.fit(scaled, W, H);
        assert!(close((t.right - t.left) * 597.0 / W, 184.0), "{t:?}");
    }

    #[test]
    #[allow(clippy::float_cmp)] // reason: swapped literals
    fn edid_sizes_follow_the_window_orientation() {
        assert_eq!(oriented((597.0, 336.0), W, H), (597.0, 336.0));
        assert_eq!(oriented((597.0, 336.0), H, W), (336.0, 597.0));
    }
}
