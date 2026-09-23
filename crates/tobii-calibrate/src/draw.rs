//! Software drawing into a `0RGB` `u32` framebuffer: just what the
//! calibration screen needs — fills, anti-aliased discs and rings, and 8x8
//! bitmap text scaled up.

use font8x8::UnicodeFonts;

/// A framebuffer: row-major pixels, `width * height` long.
pub(crate) struct Canvas<'a> {
    pub(crate) pixels: &'a mut [u32],
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// `0RGB` from components.
pub(crate) const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

/// Mix `over` onto `under` with coverage `alpha` in `0..=1`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // reason: channels stay in 0..=255
fn blend(under: u32, over: u32, alpha: f32) -> u32 {
    let a = alpha.clamp(0.0, 1.0);
    let mix = |shift: u32| {
        let u = ((under >> shift) & 0xff) as f32;
        let o = ((over >> shift) & 0xff) as f32;
        ((u + (o - u) * a).round() as u32) << shift
    };
    mix(16) | mix(8) | mix(0)
}

impl Canvas<'_> {
    pub(crate) fn fill(&mut self, color: u32) {
        self.pixels.fill(color);
    }

    fn put(&mut self, x: usize, y: usize, color: u32, alpha: f32) {
        if x < self.width && y < self.height {
            let p = &mut self.pixels[y * self.width + x];
            *p = blend(*p, color, alpha);
        }
    }

    /// Visit the pixels around `(cx, cy)` within `reach`, with their distance
    /// from the centre.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    // reason: pixel coordinates are small and clamped to the canvas
    fn around(
        &mut self,
        cx: f32,
        cy: f32,
        reach: f32,
        mut paint: impl FnMut(&mut Self, usize, usize, f32),
    ) {
        let x0 = (cx - reach).floor().max(0.0) as usize;
        let y0 = (cy - reach).floor().max(0.0) as usize;
        let x1 = ((cx + reach).ceil().max(0.0) as usize).min(self.width.saturating_sub(1));
        let y1 = ((cy + reach).ceil().max(0.0) as usize).min(self.height.saturating_sub(1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                paint(self, x, y, d);
            }
        }
    }

    /// A filled disc with a one-pixel anti-aliased edge.
    pub(crate) fn disc(&mut self, cx: f32, cy: f32, radius: f32, color: u32) {
        self.around(cx, cy, radius + 1.0, |c, x, y, d| {
            let alpha = radius + 0.5 - d;
            if alpha > 0.0 {
                c.put(x, y, color, alpha);
            }
        });
    }

    /// A ring of the given thickness, anti-aliased on both edges.
    pub(crate) fn ring(&mut self, cx: f32, cy: f32, radius: f32, thickness: f32, color: u32) {
        let half = thickness / 2.0;
        self.around(cx, cy, radius + half + 1.0, |c, x, y, d| {
            let alpha = half + 0.5 - (d - radius).abs();
            if alpha > 0.0 {
                c.put(x, y, color, alpha);
            }
        });
    }

    /// An arc of a ring from 12 o'clock clockwise over `fraction` of the turn.
    pub(crate) fn arc(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        thickness: f32,
        fraction: f32,
        color: u32,
    ) {
        let half = thickness / 2.0;
        let end = fraction.clamp(0.0, 1.0) * std::f32::consts::TAU;
        self.around(cx, cy, radius + half + 1.0, |c, x, y, d| {
            #[allow(clippy::cast_precision_loss)] // reason: small pixel coordinates
            let angle = (x as f32 + 0.5 - cx)
                .atan2(cy - (y as f32 + 0.5))
                .rem_euclid(std::f32::consts::TAU);
            let alpha = half + 0.5 - (d - radius).abs();
            if alpha > 0.0 && angle <= end {
                c.put(x, y, color, alpha);
            }
        });
    }

    /// `text` in the 8x8 font, each font pixel `scale` screen pixels, with its
    /// top-left corner at `(x, y)`. Characters the font lacks are skipped.
    pub(crate) fn text(&mut self, x: usize, y: usize, scale: usize, color: u32, text: &str) {
        for (i, ch) in text.chars().enumerate() {
            let Some(glyph) = font8x8::BASIC_FONTS.get(ch) else {
                continue;
            };
            let ox = x + i * 8 * scale;
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if bits >> col & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                self.put(ox + col * scale + dx, y + row * scale + dy, color, 1.0);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Width in pixels of `text` at `scale`.
    pub(crate) fn text_width(text: &str, scale: usize) -> usize {
        text.chars().count() * 8 * scale
    }

    /// `text` centred horizontally at row `y`.
    pub(crate) fn text_centered(&mut self, y: usize, scale: usize, color: u32, text: &str) {
        let x = self.width.saturating_sub(Self::text_width(text, scale)) / 2;
        self.text(x, y, scale, color, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(pixels: &[u32]) -> usize {
        pixels.iter().filter(|p| **p != 0).count()
    }

    #[test]
    fn a_disc_covers_about_its_area() {
        let mut px = vec![0u32; 100 * 100];
        let mut c = Canvas {
            pixels: &mut px,
            width: 100,
            height: 100,
        };
        c.disc(50.0, 50.0, 10.0, rgb(255, 255, 255));
        let area = lit(&px);
        // pi * r^2 = 314, plus a partly covered edge.
        assert!((300..=380).contains(&area), "{area}");
    }

    #[test]
    fn shapes_clip_at_the_edges() {
        let mut px = vec![0u32; 20 * 20];
        let mut c = Canvas {
            pixels: &mut px,
            width: 20,
            height: 20,
        };
        c.disc(0.0, 0.0, 30.0, rgb(1, 2, 3));
        c.ring(19.0, 19.0, 50.0, 4.0, rgb(1, 2, 3));
        assert_eq!(px.len(), 400);
    }

    #[test]
    fn text_draws_known_glyphs_only() {
        let mut px = vec![0u32; 64 * 16];
        let mut c = Canvas {
            pixels: &mut px,
            width: 64,
            height: 16,
        };
        c.text(0, 0, 1, rgb(255, 255, 255), "A\u{1F600}");
        assert!(lit(&px) > 0);
        assert_eq!(Canvas::text_width("abc", 2), 48);
    }

    #[test]
    fn half_an_arc_lights_about_half_a_ring() {
        let mut full = vec![0u32; 100 * 100];
        let mut half = vec![0u32; 100 * 100];
        Canvas {
            pixels: &mut full,
            width: 100,
            height: 100,
        }
        .ring(50.0, 50.0, 30.0, 4.0, 1);
        Canvas {
            pixels: &mut half,
            width: 100,
            height: 100,
        }
        .arc(50.0, 50.0, 30.0, 4.0, 0.5, 1);
        let ratio = lit(&half) as f64 / lit(&full) as f64;
        assert!((0.45..=0.56).contains(&ratio), "{ratio}");
    }
}
