//! The calibration window: fullscreen on the chosen monitor, drawn in
//! software. It only displays what the worker reports (targets, progress,
//! the result) and turns Esc into an abort.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tobii_calib::STIMULUS_POINTS;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::monitor::MonitorHandle;
use winit::window::{Fullscreen, Window, WindowId};

use crate::draw::{Canvas, rgb};
use crate::sequence::{Phase, Summary, Timing, UiEvent};

const BACKGROUND: u32 = rgb(22, 22, 26);
const FOREGROUND: u32 = rgb(230, 230, 235);
const MUTED: u32 = rgb(130, 130, 140);
const ACCENT: u32 = rgb(80, 200, 120);
const WARNING: u32 = rgb(230, 90, 80);
const GAZE: u32 = rgb(90, 160, 255);

/// Which monitor to calibrate on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MonitorChoice {
    /// The primary monitor, or the first one.
    Default,
    /// By position in the list `--list-monitors` prints.
    Index(usize),
    /// The first monitor whose name contains this.
    Name(String),
}

/// What is on screen.
#[derive(Debug, Clone, PartialEq)]
enum Screen {
    Status(String),
    Target {
        at: [f32; 2],
        from: [f32; 2],
        phase: Phase,
        step: usize,
        total: usize,
    },
    Computing,
    Finished(Summary),
    Failed(String),
}

/// The winit application.
pub(crate) struct Ui {
    abort: Arc<AtomicBool>,
    monitor: MonitorChoice,
    list_only: bool,
    windowed: bool,
    timing: Timing,
    verify: Duration,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    screen: Screen,
    since: Instant,
    gaze: Option<([f32; 2], bool)>,
    /// Set when the session ended badly, for the exit code.
    pub(crate) failed: Option<String>,
}

impl Ui {
    pub(crate) fn new(
        abort: Arc<AtomicBool>,
        monitor: MonitorChoice,
        list_only: bool,
        windowed: bool,
        timing: Timing,
        verify: Duration,
    ) -> Self {
        Self {
            abort,
            monitor,
            list_only,
            windowed,
            timing,
            verify,
            window: None,
            surface: None,
            screen: Screen::Status("starting...".into()),
            since: Instant::now(),
            gaze: None,
            failed: None,
        }
    }

    fn pick_monitor(&self, event_loop: &ActiveEventLoop) -> Option<MonitorHandle> {
        let monitors: Vec<MonitorHandle> = event_loop.available_monitors().collect();
        match &self.monitor {
            MonitorChoice::Index(i) => monitors.get(*i).cloned(),
            MonitorChoice::Name(part) => monitors
                .iter()
                .find(|m| m.name().is_some_and(|n| n.contains(part.as_str())))
                .cloned(),
            MonitorChoice::Default => event_loop
                .primary_monitor()
                .or_else(|| monitors.first().cloned()),
        }
    }

    fn draw(&mut self) {
        let (Some(window), Some(surface)) = (self.window.as_ref(), self.surface.as_mut()) else {
            return;
        };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return;
        };
        if surface.resize(w, h).is_err() {
            return;
        }
        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };
        let (width, height) = (size.width as usize, size.height as usize);
        let mut c = Canvas {
            pixels: &mut buffer,
            width,
            height,
        };
        paint(
            &mut c,
            &self.screen,
            self.since.elapsed(),
            self.timing,
            self.gaze,
        );
        let _ = buffer.present();
    }

    fn animating(&self) -> bool {
        !matches!(self.screen, Screen::Status(_) | Screen::Failed(_))
    }
}

/// A point in normalised display coordinates, in pixels.
#[allow(clippy::cast_precision_loss)] // reason: screen sizes are far below 2^24
pub(crate) fn to_pixels(at: [f32; 2], width: usize, height: usize) -> (f32, f32) {
    (at[0] * width as f32, at[1] * height as f32)
}

fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn fraction(elapsed: Duration, of: Duration) -> f32 {
    if of.is_zero() {
        1.0
    } else {
        (elapsed.as_secs_f32() / of.as_secs_f32()).min(1.0)
    }
}

/// Text scale for the screen size.
fn scale_for(height: usize) -> usize {
    (height / 360).clamp(2, 6)
}

fn paint(
    c: &mut Canvas<'_>,
    screen: &Screen,
    elapsed: Duration,
    timing: Timing,
    gaze: Option<([f32; 2], bool)>,
) {
    c.fill(BACKGROUND);
    let (w, h) = (c.width, c.height);
    let s = scale_for(h);
    #[allow(clippy::cast_precision_loss)] // reason: small sizes
    let unit = h as f32 / 100.0;
    match screen {
        Screen::Status(text) => c.text_centered(h / 2, s, FOREGROUND, text),
        Screen::Target {
            at,
            from,
            phase,
            step,
            total,
        } => {
            let t = match phase {
                Phase::Travel => ease(fraction(elapsed, timing.travel)),
                _ => 1.0,
            };
            let pos = [
                from[0] + (at[0] - from[0]) * t,
                from[1] + (at[1] - from[1]) * t,
            ];
            let (x, y) = to_pixels(pos, w, h);
            match phase {
                Phase::Travel => c.disc(x, y, unit * 1.2, FOREGROUND),
                Phase::Dwell => {
                    let shrink = 1.0 - fraction(elapsed, timing.dwell);
                    c.ring(x, y, unit * (1.2 + 4.0 * shrink), unit * 0.35, MUTED);
                    c.disc(x, y, unit * 1.2, FOREGROUND);
                }
                Phase::Collecting => {
                    c.arc(
                        x,
                        y,
                        unit * 2.4,
                        unit * 0.4,
                        (elapsed.as_secs_f32() * 1.5).fract(),
                        ACCENT,
                    );
                    c.disc(x, y, unit * 1.2, FOREGROUND);
                }
            }
            c.disc(x, y, unit * 0.3, BACKGROUND);
            let hint = format!("look at the dot   {step} / {total}   Esc to cancel");
            c.text_centered(h - 3 * 8 * s, s, MUTED, &hint);
        }
        Screen::Computing => {
            #[allow(clippy::cast_precision_loss)] // reason: small sizes
            let (x, y) = (w as f32 / 2.0, h as f32 / 2.0);
            c.arc(
                x,
                y,
                unit * 4.0,
                unit * 0.5,
                (elapsed.as_secs_f32() * 1.5).fract(),
                ACCENT,
            );
            c.text_centered(h / 2 + h * 8 / 100, s, FOREGROUND, "computing...");
        }
        Screen::Finished(summary) => {
            for target in STIMULUS_POINTS {
                let (x, y) = to_pixels(target, w, h);
                c.ring(x, y, unit * 1.5, unit * 0.3, MUTED);
            }
            if let Some((g, valid)) = gaze {
                let (x, y) = to_pixels(g, w, h);
                c.disc(x, y, unit * 0.9, if valid { GAZE } else { MUTED });
            }
            let error = summary.mean_error.map_or(String::new(), |e| {
                format!(", mean offset {:.1}% of the screen", e * 100.0)
            });
            let line = format!(
                "calibration {:08x} saved: {} points{error}",
                summary.id, summary.points
            );
            c.text_centered(h / 3, s, ACCENT, &line);
            c.text_centered(
                h / 3 + 12 * s,
                s,
                MUTED,
                "look at the circles to check it; Esc to close",
            );
        }
        Screen::Failed(text) => {
            c.text_centered(h / 2, s, WARNING, text);
            c.text_centered(
                h / 2 + 12 * s,
                s,
                MUTED,
                "the previous calibration is kept; Esc to close",
            );
        }
    }
}

impl ApplicationHandler<UiEvent> for Ui {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.list_only {
            for (i, m) in event_loop.available_monitors().enumerate() {
                let size = m.size();
                println!(
                    "{i}: {} ({}x{})",
                    m.name().unwrap_or_else(|| "unnamed".into()),
                    size.width,
                    size.height
                );
            }
            event_loop.exit();
            return;
        }
        if self.window.is_some() {
            return;
        }
        let mut attributes = Window::default_attributes().with_title("tobii-calibrate");
        if self.windowed {
            attributes = attributes.with_inner_size(LogicalSize::new(1280.0, 800.0));
        } else {
            let Some(monitor) = self.pick_monitor(event_loop) else {
                self.failed = Some(format!("no monitor matches {:?}", self.monitor));
                event_loop.exit();
                return;
            };
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(Some(monitor))));
        }
        let window = match event_loop.create_window(attributes) {
            Ok(w) => Rc::new(w),
            Err(e) => {
                self.failed = Some(format!("could not open a window: {e}"));
                event_loop.exit();
                return;
            }
        };
        window.set_cursor_visible(false);
        let surface = softbuffer::Context::new(Rc::clone(&window))
            .and_then(|context| softbuffer::Surface::new(&context, Rc::clone(&window)));
        match surface {
            Ok(surface) => self.surface = Some(surface),
            Err(e) => {
                self.failed = Some(format!("could not draw into the window: {e}"));
                event_loop.exit();
                return;
            }
        }
        window.request_redraw();
        self.window = Some(window);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UiEvent) {
        let next = match event {
            UiEvent::Gaze(at, valid) => {
                self.gaze = Some((at, valid));
                None
            }
            UiEvent::Status(text) => Some(Screen::Status(text)),
            UiEvent::Target {
                at,
                from,
                phase,
                step,
                total,
            } => Some(Screen::Target {
                at,
                from,
                phase,
                step,
                total,
            }),
            UiEvent::Computing => Some(Screen::Computing),
            UiEvent::Finished(summary) => Some(Screen::Finished(summary)),
            UiEvent::Failed(text) => {
                self.failed = Some(text.clone());
                Some(Screen::Failed(text))
            }
        };
        if let Some(screen) = next {
            self.screen = screen;
            self.since = Instant::now();
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Relaxed: a pure signal to the worker.
                self.abort.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                // Relaxed: a pure signal to the worker.
                self.abort.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::Resized(_) => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self.screen, Screen::Finished(_)) && self.since.elapsed() > self.verify {
            event_loop.exit();
            return;
        }
        if self.animating() {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(16),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // reason: exact products of small values
    fn normalised_points_map_onto_the_surface() {
        assert_eq!(to_pixels([0.5, 0.5], 2560, 1440), (1280.0, 720.0));
        assert_eq!(to_pixels([0.1, 0.9], 1000, 500), (100.0, 450.0));
    }

    #[test]
    fn every_screen_paints_without_panicking() {
        let timing = Timing {
            travel: Duration::from_millis(300),
            dwell: Duration::from_millis(1000),
        };
        let screens = [
            Screen::Status("waiting".into()),
            Screen::Target {
                at: [0.9, 0.1],
                from: [0.5, 0.5],
                phase: Phase::Dwell,
                step: 2,
                total: 14,
            },
            Screen::Computing,
            Screen::Finished(Summary {
                id: 7,
                points: 14,
                mean_error: Some(0.02),
                blob: Vec::new(),
            }),
            Screen::Failed("no".into()),
        ];
        let mut px = vec![0u32; 320 * 200];
        for screen in &screens {
            let mut c = Canvas {
                pixels: &mut px,
                width: 320,
                height: 200,
            };
            paint(
                &mut c,
                screen,
                Duration::from_millis(150),
                timing,
                Some(([0.5, 0.5], true)),
            );
        }
        assert!(px.iter().any(|p| *p != BACKGROUND));
    }
}
