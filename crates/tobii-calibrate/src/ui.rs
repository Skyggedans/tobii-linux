//! The calibration window: fullscreen on the chosen monitor, drawn in
//! software. It displays what the worker reports (targets, progress, the
//! result), runs the display setup (ticks moved by mouse and keyboard, the
//! answer sent back to the worker) and turns Esc into an abort.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use tobii_calib::STIMULUS_POINTS;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::monitor::MonitorHandle;
use winit::window::{Fullscreen, Window, WindowId};

use crate::draw::{Canvas, rgb};
use crate::sequence::{Phase, Summary, Timing, UiEvent};
use crate::setup::{Choice, Start, Ticks};

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
    DisplaySetup(Setup),
}

/// The display setup on screen: what it started from and where the ticks are.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Setup {
    start: Start,
    ticks: Ticks,
    /// The window width `ticks` are measured in.
    width_px: f64,
}

impl Setup {
    /// The ticks on a `width_px × height_px` window: they scale with its
    /// width, and are refitted to it (see [`Start::fit`]).
    fn ticks_for(&self, width_px: f64, height_px: f64) -> Ticks {
        let k = width_px / self.width_px;
        let scaled = Ticks {
            left: self.ticks.left * k,
            right: self.ticks.right * k,
        };
        self.start.fit(scaled, width_px, height_px)
    }
}

/// What a mouse drag on the setup screen moves.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    Left,
    Right,
    /// Both ticks, as they were when the drag started at `grabbed_x`.
    Both {
        grabbed_x: f64,
        from: Ticks,
    },
}

/// What the window and the calibration worker signal each other.
pub(crate) struct Signals {
    /// Set by the window (Esc, closed): the worker stops the session.
    pub(crate) abort: Arc<AtomicBool>,
    /// Set by the window while it is fullscreen on its monitor: the worker
    /// shows points only then.
    pub(crate) placed: Arc<AtomicBool>,
}

/// The winit application.
pub(crate) struct Ui {
    abort: Arc<AtomicBool>,
    placed: Arc<AtomicBool>,
    monitor: MonitorChoice,
    list_only: bool,
    windowed: bool,
    timing: Timing,
    verify: Duration,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    screen: Screen,
    /// Counts screen changes, so a frame knows whether it continues the
    /// last one (and only the moving parts need repainting on screen).
    epoch: u64,
    since: Instant,
    gaze: Option<([f32; 2], bool)>,
    /// When the next animation frame is due.
    next_frame: Instant,
    /// A frame was asked for and has not been drawn yet.
    redraw_pending: bool,
    /// What the last presented frame showed.
    presented: Option<Presented>,
    /// Something changed on a still screen; draw it at the next paced frame.
    dirty: bool,
    /// The display setup's answer, for the worker.
    answers: Sender<Choice>,
    /// Name of the monitor the window is on, to find its EDID.
    monitor_name: Option<String>,
    /// How many monitors there are.
    monitor_count: usize,
    /// The monitor the fullscreen window must stay on.
    target: Option<MonitorHandle>,
    /// When the window was made: the compositor gets a moment to place it.
    created: Option<Instant>,
    /// Tries so far at moving the window back to `target`, and the last.
    placement_tries: u32,
    placement_last: Option<Instant>,
    /// The window stayed on another monitor: the one it belongs on.
    misplaced: Option<String>,
    /// Last pointer position across the window, physical pixels.
    pointer_x: Option<f64>,
    drag: Option<Drag>,
    shift: bool,
    /// Set when the session ended badly, for the exit code.
    pub(crate) failed: Option<String>,
    /// The user closed the window (Esc or the window's close).
    pub(crate) escaped: bool,
}

/// The last presented frame, for working out what the next one changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Presented {
    epoch: u64,
    size: (usize, usize),
    moving: Option<Bounds>,
}

/// What to hand the compositor as damage: `None` for the whole surface (a
/// new screen or size, or a redraw the window did not ask for), else what
/// moved since the last frame (`Some(None)`: nothing did).
fn damage(
    last: Option<Presented>,
    epoch: u64,
    size: (usize, usize),
    moving: Option<Bounds>,
    asked: bool,
) -> Option<Option<Bounds>> {
    match last {
        Some(last) if asked && last.epoch == epoch && last.size == size => {
            Some(Bounds::union(last.moving, moving))
        }
        _ => None,
    }
}

/// Animation frame interval (60 Hz). On Wayland the compositor's frame
/// callbacks pace drawing too; this also bounds the other backends. Each
/// frame of a 4K window is 33 MB for the compositor to take in, so drawing
/// unpaced (hundreds of frames a second) stalls the whole desktop.
const FRAME_INTERVAL: Duration = Duration::from_micros(16_667);

/// How often the loop looks again while a frame it asked for has not been
/// drawn (a hidden window gets no frame callbacks on Wayland).
const PENDING_CHECK: Duration = Duration::from_millis(100);

/// How many times the window is sent back to the monitor it was opened on
/// when the compositor puts it on another one (`PaperWM` moves a new window to
/// the monitor in use), and the pause between tries.
const PLACEMENT_TRIES: u32 = 5;
const PLACEMENT_PAUSE: Duration = Duration::from_millis(500);

/// Whether to ask for a frame now, and when to wake up next: frames at most
/// every [`FRAME_INTERVAL`], never a wake-up in the past.
fn pace(now: Instant, next_frame: Instant, pending: bool) -> (bool, Instant) {
    if pending {
        (false, now + PENDING_CHECK)
    } else if now >= next_frame {
        (true, now + PENDING_CHECK)
    } else {
        (false, next_frame)
    }
}

/// A pixel rectangle `[x0, x1) × [y0, y1)` inside the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Bounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Bounds {
    /// The square of half-size `half` around `(cx, cy)`, clipped to a
    /// `width × height` surface; `None` when it lies outside.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // reason: clamped to the surface first
    fn around(cx: f32, cy: f32, half: f32, width: usize, height: usize) -> Option<Self> {
        #[allow(clippy::cast_precision_loss)] // reason: screen sizes are far below 2^24
        let clamp = |v: f32, max: usize| v.clamp(0.0, max as f32) as usize;
        let b = Self {
            x0: clamp((cx - half).floor(), width),
            y0: clamp((cy - half).floor(), height),
            x1: clamp((cx + half).ceil() + 1.0, width),
            y1: clamp((cy + half).ceil() + 1.0, height),
        };
        (b.x0 < b.x1 && b.y0 < b.y1).then_some(b)
    }

    /// The smallest rectangle holding both.
    fn union(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        match (a, b) {
            (Some(a), Some(b)) => Some(Self {
                x0: a.x0.min(b.x0),
                y0: a.y0.min(b.y0),
                x1: a.x1.max(b.x1),
                y1: a.y1.max(b.y1),
            }),
            (a, None) => a,
            (None, b) => b,
        }
    }

    fn rect(self) -> Option<softbuffer::Rect> {
        Some(softbuffer::Rect {
            x: u32::try_from(self.x0).ok()?,
            y: u32::try_from(self.y0).ok()?,
            width: NonZeroU32::new(u32::try_from(self.x1 - self.x0).ok()?)?,
            height: NonZeroU32::new(u32::try_from(self.y1 - self.y0).ok()?)?,
        })
    }
}

impl Ui {
    pub(crate) fn new(
        signals: Signals,
        monitor: MonitorChoice,
        list_only: bool,
        windowed: bool,
        timing: Timing,
        verify: Duration,
        answers: Sender<Choice>,
    ) -> Self {
        Self {
            abort: signals.abort,
            placed: signals.placed,
            monitor,
            list_only,
            windowed,
            timing,
            verify,
            window: None,
            surface: None,
            screen: Screen::Status("starting...".into()),
            epoch: 0,
            since: Instant::now(),
            gaze: None,
            next_frame: Instant::now(),
            redraw_pending: false,
            presented: None,
            dirty: false,
            answers,
            monitor_name: None,
            monitor_count: 0,
            target: None,
            created: None,
            placement_tries: 0,
            placement_last: None,
            misplaced: None,
            pointer_x: None,
            drag: None,
            shift: false,
            failed: None,
            escaped: false,
        }
    }

    /// The display setup, while it is on screen.
    fn setup(&self) -> Option<Setup> {
        match self.screen {
            Screen::DisplaySetup(setup) => Some(setup),
            _ => None,
        }
    }

    /// Where the window is, if it is not fullscreen on the monitor it was
    /// opened on (and the compositor has said where it is), with that
    /// monitor. A mirror of the monitor (same place and size) counts as it;
    /// fullscreen means covering the monitor, whatever the window system
    /// believes it asked for (X11 reports the request, not the outcome).
    fn misplaced_now(&self) -> Option<(String, MonitorHandle)> {
        let target = self.target.clone()?;
        let window = self.window.as_ref()?;
        let Some(current) = window.current_monitor() else {
            return Some(("no monitor".to_owned(), target));
        };
        let same_place = current.position() == target.position() && current.size() == target.size();
        if current.name() != target.name() && !same_place {
            let on = current.name().unwrap_or_else(|| "another monitor".into());
            return Some((on, target));
        }
        let (size, full) = (window.inner_size(), target.size());
        let covers = |w: u32, h: u32| {
            f64::from(size.width) >= f64::from(w) * 0.98
                && f64::from(size.height) >= f64::from(h) * 0.98
        };
        // Wayland reports a turned monitor's mode unturned.
        let fullscreen = window.fullscreen().is_some()
            && (covers(full.width, full.height) || covers(full.height, full.width));
        (!fullscreen).then(|| ("a window, not fullscreen".to_owned(), target))
    }

    /// Keep the fullscreen window on its monitor: send it back a few times
    /// if the compositor moved it, then warn on screen (a calibration shown
    /// on another monitor would be saved all the same). Returns when to look
    /// again.
    fn keep_on_target(&mut self) -> Option<Instant> {
        let now = Instant::now();
        self.target.as_ref()?;
        if let Some(created) = self.created
            && now < created + 2 * PLACEMENT_PAUSE
        {
            return Some(created + 2 * PLACEMENT_PAUSE);
        }
        let misplaced = self.misplaced_now();
        // Relaxed: a pure signal to the worker.
        self.placed.store(misplaced.is_none(), Ordering::Relaxed);
        let Some((on, target)) = misplaced else {
            if self.misplaced.take().is_some() {
                tracing::info!("the window is back on its monitor");
                self.epoch += 1;
                self.request_frame();
            }
            return None;
        };
        if let Some(last) = self.placement_last
            && now < last + PLACEMENT_PAUSE
        {
            return Some(last + PLACEMENT_PAUSE);
        }
        if self.placement_tries < PLACEMENT_TRIES {
            self.placement_tries += 1;
            self.placement_last = Some(now);
            tracing::warn!(
                on,
                wanted = target.name().as_deref().unwrap_or("?"),
                attempt = self.placement_tries,
                "the window is not fullscreen on its monitor: asking again"
            );
            if let Some(window) = &self.window {
                // Out of fullscreen first: asking for the state the window
                // is believed to have already is a no-op (winit on X11).
                window.set_fullscreen(None);
                window.set_fullscreen(Some(Fullscreen::Borderless(Some(target))));
            }
            return Some(now + PLACEMENT_PAUSE);
        }
        if self.misplaced.is_none() {
            let wanted = target.name().unwrap_or_else(|| "its monitor".into());
            tracing::warn!(wanted, "the window stays off its monitor");
            self.misplaced = Some(wanted);
            self.epoch += 1;
            self.request_frame();
        }
        None
    }

    /// The window's size in physical pixels.
    fn window_size(&self) -> Option<(f64, f64)> {
        let size = self.window.as_ref()?.inner_size();
        Some((f64::from(size.width), f64::from(size.height)))
    }

    /// Put `screen` up: drawn at once, the pointer shown only for the setup.
    fn show(&mut self, screen: Screen) {
        tracing::debug!(screen = screen_name(&screen), "screen");
        if let Some(w) = &self.window {
            w.set_cursor_visible(matches!(screen, Screen::DisplaySetup(_)));
        }
        self.screen = screen;
        self.drag = None;
        self.epoch += 1;
        self.since = Instant::now();
        self.request_frame();
    }

    /// The setup screen for `req`, its ticks placed from the monitor's EDID
    /// (or the tracker's current area).
    fn setup_screen(&self, req: crate::sequence::SetupRequest) -> Option<Screen> {
        let (width, height) = self.window_size()?;
        let monitor_mm =
            crate::edid::monitor_size_mm(self.monitor_name.as_deref(), self.monitor_count);
        tracing::info!(
            monitor = self.monitor_name.as_deref().unwrap_or("?"),
            ?monitor_mm,
            guide_mm = req.guide_mm,
            "display setup"
        );
        let start = Start {
            guide_mm: req.guide_mm,
            monitor_mm,
            current: req.current,
        };
        Some(Screen::DisplaySetup(Setup {
            start,
            ticks: start.ticks(width, height),
            width_px: width,
        }))
    }

    /// Move the setup's ticks to `ticks` (measured on a window `width_px`
    /// wide) and have them drawn.
    fn set_ticks(&mut self, ticks: Ticks, width_px: f64) {
        if let Screen::DisplaySetup(setup) = &mut self.screen {
            setup.ticks = ticks;
            setup.width_px = width_px;
            self.dirty = true;
        }
    }

    /// A key on the setup screen: arrows move the ticks together, up and
    /// down spread them (unless the EDID fixes their spacing), Shift makes
    /// the steps ten pixels; Enter answers.
    fn setup_key(&mut self, key: &Key) {
        let (Some(setup), Some((width, height))) = (self.setup(), self.window_size()) else {
            return;
        };
        let step = if self.shift { 10.0 } else { 1.0 };
        let ticks = setup.ticks_for(width, height);
        let spreads = !setup.start.spacing_fixed(width, height);
        let moved = match key {
            Key::Named(NamedKey::ArrowLeft) => ticks.shifted(-step, width),
            Key::Named(NamedKey::ArrowRight) => ticks.shifted(step, width),
            Key::Named(NamedKey::ArrowUp) if spreads => ticks.widened(step, width),
            Key::Named(NamedKey::ArrowDown) if spreads => ticks.widened(-step, width),
            Key::Named(NamedKey::Enter) => return self.confirm_setup(),
            _ => return,
        };
        self.set_ticks(moved, width);
    }

    /// The pointer moved: carry what is being dragged.
    fn setup_pointer(&mut self, x: f64) {
        self.pointer_x = Some(x);
        let (Some(drag), Some(setup), Some((width, height))) =
            (self.drag, self.setup(), self.window_size())
        else {
            return;
        };
        let ticks = setup.ticks_for(width, height);
        let moved = match drag {
            Drag::Left => ticks.with_left(x, width),
            Drag::Right => ticks.with_right(x, width),
            Drag::Both { grabbed_x, from } => from.shifted(x - grabbed_x, width),
        };
        self.set_ticks(moved, width);
    }

    /// A press on a tick drags that tick (when their spacing is free);
    /// anywhere else, both.
    fn setup_button(&mut self, state: ElementState) {
        if state == ElementState::Released {
            self.drag = None;
            return;
        }
        let (Some(x), Some(setup), Some((width, height))) =
            (self.pointer_x, self.setup(), self.window_size())
        else {
            return;
        };
        let ticks = setup.ticks_for(width, height);
        let single = !setup.start.spacing_fixed(width, height);
        let reach = (height / 100.0 * 1.5).max(12.0);
        let (to_left, to_right) = ((x - ticks.left).abs(), (x - ticks.right).abs());
        self.drag = Some(if single && to_left <= reach && to_left <= to_right {
            Drag::Left
        } else if single && to_right <= reach {
            Drag::Right
        } else {
            Drag::Both {
                grabbed_x: x,
                from: ticks,
            }
        });
    }

    /// Enter on the setup screen: send the answer and wait for the worker.
    /// Not while the window is on another monitor: the ticks would measure
    /// that one.
    fn confirm_setup(&mut self) {
        if let Some((on, _)) = self.misplaced_now() {
            tracing::warn!(on, "not confirming the display setup off its monitor");
            return;
        }
        let (Some(setup), Some((width, height))) = (self.setup(), self.window_size()) else {
            return;
        };
        let Some(choice) = setup
            .start
            .choice(setup.ticks_for(width, height), width, height)
        else {
            return;
        };
        tracing::info!(?choice, "display setup confirmed");
        if self.answers.send(choice).is_err() {
            tracing::warn!("the calibration worker is gone");
        }
        self.show(Screen::Status("setting the display area...".into()));
    }

    /// Ask for a frame as soon as the backend allows.
    fn request_frame(&mut self) {
        if let Some(w) = &self.window {
            w.request_redraw();
            self.redraw_pending = true;
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
        // A redraw this window did not ask for (an X11 expose, a configure)
        // may need the whole surface back, not just what moved.
        let asked = std::mem::take(&mut self.redraw_pending);
        self.dirty = false;
        let (Some(window), Some(surface)) = (self.window.as_ref(), self.surface.as_mut()) else {
            return;
        };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            tracing::debug!(?size, "frame skipped: empty window");
            return;
        };
        let started = Instant::now();
        self.next_frame = started + FRAME_INTERVAL;
        if let Err(e) = surface.resize(w, h) {
            tracing::warn!(error = %e, ?size, "frame skipped: resize failed");
            return;
        }
        let mut buffer = match surface.buffer_mut() {
            Ok(buffer) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, "frame skipped: no buffer");
                return;
            }
        };
        let waited = started.elapsed();
        let (width, height) = (size.width as usize, size.height as usize);
        let elapsed = self.since.elapsed();
        let mut c = Canvas {
            pixels: &mut buffer,
            width,
            height,
        };
        paint(&mut c, &self.screen, elapsed, self.timing, self.gaze);
        if let Some(wanted) = &self.misplaced {
            paint_misplaced(&mut c, wanted);
        }
        let painted = started.elapsed();

        // The whole frame is repainted, but only what moved since the last
        // one is handed to the compositor as damage: a few KB instead of
        // the whole surface. A new screen or size is damaged whole.
        let moving = moving_bounds(&self.screen, elapsed, self.timing, self.gaze, width, height);
        let damage = damage(self.presented, self.epoch, (width, height), moving, asked);
        self.presented = Some(Presented {
            epoch: self.epoch,
            size: (width, height),
            moving,
        });
        // Wayland: pace the next redraw by the compositor's frame callback.
        window.pre_present_notify();
        let result = match damage {
            None => buffer.present(),
            Some(changed) => {
                let rects: Vec<softbuffer::Rect> =
                    changed.and_then(Bounds::rect).into_iter().collect();
                buffer.present_with_damage(&rects)
            }
        };
        if let Err(e) = result {
            tracing::warn!(error = %e, "frame not presented");
        }
        tracing::trace!(
            screen = screen_name(&self.screen),
            width,
            height,
            damage = ?damage,
            buffer_ms = waited.as_millis(),
            paint_ms = (painted - waited).as_millis(),
            total_ms = started.elapsed().as_millis(),
            "frame"
        );
    }

    /// Screens that change by themselves; the rest are drawn on events.
    fn animating(&self) -> bool {
        !matches!(
            self.screen,
            Screen::Status(_) | Screen::Failed(_) | Screen::DisplaySetup(_)
        )
    }
}

/// A short name of a screen, for the log.
fn screen_name(screen: &Screen) -> &'static str {
    match screen {
        Screen::Status(_) => "status",
        Screen::Target { .. } => "target",
        Screen::Computing => "computing",
        Screen::Finished(_) => "finished",
        Screen::Failed(_) => "failed",
        Screen::DisplaySetup(_) => "display setup",
    }
}

/// Where the setup screen draws: the ticks from `tick_top` to the bottom
/// edge (right above the tracker), the live reading at `reading_y` above.
fn setup_layout(height: usize) -> (usize, usize) {
    let tick_top = height - height / 10;
    (tick_top, tick_top.saturating_sub(3 * 8 * scale_for(height)))
}

/// The live reading of the setup screen.
fn describe(choice: &Choice) -> String {
    let side = if choice.offset_x_mm.abs() < 0.05 {
        "under its centre".to_owned()
    } else {
        // The monitor's centre right of the tracker: the tracker is left.
        let side = if choice.offset_x_mm > 0.0 {
            "left"
        } else {
            "right"
        };
        format!("{:.1} mm {side} of its centre", choice.offset_x_mm.abs())
    };
    format!(
        "screen {:.0} x {:.0} mm, tracker {side}",
        choice.width_mm, choice.height_mm
    )
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

/// Where a target is drawn: travelling from `from` to `at`, then resting.
fn target_position(
    at: [f32; 2],
    from: [f32; 2],
    phase: Phase,
    elapsed: Duration,
    timing: Timing,
) -> [f32; 2] {
    let t = match phase {
        Phase::Travel => ease(fraction(elapsed, timing.travel)),
        _ => 1.0,
    };
    [
        from[0] + (at[0] - from[0]) * t,
        from[1] + (at[1] - from[1]) * t,
    ]
}

/// Everything on `screen` that changes from frame to frame, in pixels: the
/// target and its ring, the spinner, the gaze dot. The rest of a screen is
/// the same in every frame. Sized by the largest shape [`paint`] draws
/// there, plus antialiasing.
fn moving_bounds(
    screen: &Screen,
    elapsed: Duration,
    timing: Timing,
    gaze: Option<([f32; 2], bool)>,
    width: usize,
    height: usize,
) -> Option<Bounds> {
    #[allow(clippy::cast_precision_loss)] // reason: small sizes
    let unit = height as f32 / 100.0;
    match screen {
        Screen::Target {
            at, from, phase, ..
        } => {
            let (x, y) = to_pixels(
                target_position(*at, *from, *phase, elapsed, timing),
                width,
                height,
            );
            // The dwell ring at its widest: radius 5.2, thickness 0.35.
            Bounds::around(x, y, unit * 5.4 + 2.0, width, height)
        }
        Screen::Computing => {
            #[allow(clippy::cast_precision_loss)] // reason: small sizes
            let (x, y) = (width as f32 / 2.0, height as f32 / 2.0);
            // The spinner: radius 4, thickness 0.5.
            Bounds::around(x, y, unit * 4.3 + 2.0, width, height)
        }
        Screen::Finished(_) => gaze.and_then(|(g, _)| {
            let (x, y) = to_pixels(g, width, height);
            Bounds::around(x, y, unit * 0.9 + 2.0, width, height)
        }),
        // The ticks move anywhere along the bottom; the reading changes too.
        Screen::DisplaySetup(_) => Some(Bounds {
            x0: 0,
            y0: setup_layout(height).1,
            x1: width,
            y1: height,
        }),
        Screen::Status(_) | Screen::Failed(_) => None,
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
            let pos = target_position(*at, *from, *phase, elapsed, timing);
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
            c.text_centered(h / 2 + 12 * s, s, MUTED, "Esc to close");
        }
        Screen::DisplaySetup(setup) => paint_setup(c, setup, s, unit),
    }
}

/// The warning over every screen while the window is on the wrong monitor.
fn paint_misplaced(c: &mut Canvas<'_>, wanted: &str) {
    let s = scale_for(c.height);
    c.text_centered(
        2 * 8 * s,
        s,
        WARNING,
        &format!("this belongs fullscreen on {wanted}: move it there, or press Esc"),
    );
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // reason: screen-sized values
fn paint_setup(c: &mut Canvas<'_>, setup: &Setup, s: usize, unit: f32) {
    let (w, h) = (c.width, c.height);
    let ticks = setup.ticks_for(w as f64, h as f64);
    c.text_centered(
        h / 3,
        s,
        FOREGROUND,
        "line the two ticks up with the two white marks on the eye tracker",
    );
    let hint = if setup.start.spacing_fixed(w as f64, h as f64) {
        "drag them, or Left/Right to move them (Shift: faster)"
    } else {
        "drag them, or Left/Right: move, Up/Down: spread, Shift: faster"
    };
    c.text_centered(h / 3 + 12 * s, s, MUTED, hint);
    c.text_centered(
        h / 3 + 24 * s,
        s,
        MUTED,
        "Enter when they line up, Esc to cancel",
    );
    let (tick_top, reading_y) = setup_layout(h);
    if let Some(choice) = setup.start.choice(ticks, w as f64, h as f64) {
        c.text_centered(reading_y, s, ACCENT, &describe(&choice));
    }
    let half = (unit * 0.1).max(1.0);
    for x in [ticks.left, ticks.right] {
        c.vbar(x as f32, half, tick_top, h, FOREGROUND);
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
            self.monitor_name = monitor.name();
            self.monitor_count = event_loop.available_monitors().count();
            self.target = Some(monitor.clone());
            self.created = Some(Instant::now());
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
        self.window = Some(window);
        self.request_frame();
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
            UiEvent::DisplaySetup(req) => self.setup_screen(req),
        };
        // A gaze sample only shows on an animating screen, whose next paced
        // frame picks it up; a new screen is drawn at once.
        if let Some(screen) = next {
            self.show(screen);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Relaxed: a pure signal to the worker.
                self.abort.store(true, Ordering::Relaxed);
                self.escaped = true;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                // Relaxed: a pure signal to the worker.
                self.abort.store(true, Ordering::Relaxed);
                self.escaped = true;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.setup_key(&event.logical_key);
            }
            WindowEvent::ModifiersChanged(m) => self.shift = m.state().shift_key(),
            WindowEvent::CursorMoved { position, .. } => self.setup_pointer(position.x),
            // A button release may never come: stop dragging.
            WindowEvent::CursorLeft { .. } | WindowEvent::Focused(false) => self.drag = None,
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.setup_button(state),
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::Resized(_) => {
                // Whatever the window held may be gone: repaint all of it.
                self.presented = None;
                self.request_frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self.screen, Screen::Finished(_)) && self.since.elapsed() > self.verify {
            event_loop.exit();
            return;
        }
        let placement = self.keep_on_target();
        if self.animating() || self.dirty {
            // Asking for a redraw wakes the loop at once, so frames must be
            // paced here rather than by the wake-up time alone.
            let (request, wake) = pace(Instant::now(), self.next_frame, self.redraw_pending);
            if request {
                self.request_frame();
            }
            let wake = placement.map_or(wake, |p| p.min(wake));
            event_loop.set_control_flow(ControlFlow::WaitUntil(wake));
        } else {
            event_loop
                .set_control_flow(placement.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
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
            Screen::DisplaySetup(setup_screen(320.0)),
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

    #[test]
    fn a_loop_woken_by_every_redraw_request_still_draws_at_most_60_frames_a_second() {
        // Asking for a redraw wakes the loop at once: model the loop running
        // flat out, each requested frame drawn straight away.
        let start = Instant::now();
        let (mut now, mut next_frame, mut frames) = (start, start, 0);
        while now < start + Duration::from_secs(1) {
            let (request, wake) = pace(now, next_frame, false);
            assert!(wake > now, "the loop must not wake in the past");
            if request {
                frames += 1;
                next_frame = now + FRAME_INTERVAL;
                now += Duration::from_micros(100);
            } else {
                now = wake;
            }
        }
        assert!((59..=61).contains(&frames), "{frames} frames");

        // A frame asked for and not drawn yet (no frame callback from the
        // compositor): look again later, do not ask again or spin.
        let (request, wake) = pace(start, start, true);
        assert!(!request && wake > start);
    }

    const TIMING: Timing = Timing {
        travel: Duration::from_millis(300),
        dwell: Duration::from_millis(1000),
    };

    type Frame = (Duration, Option<([f32; 2], bool)>);

    const W: usize = 640;
    const H: usize = 360;

    /// Paint two frames of `screen`; every pixel that differs must lie in the
    /// damage the second frame reports.
    fn assert_damage_covers_changes(screen: &Screen, a: Frame, b: Frame) {
        assert_damage_covers_changes_between((screen, a), (screen, b));
    }

    /// The same for two frames of one screen whose state changed in between
    /// (the setup's ticks moved).
    fn assert_damage_covers_changes_between(a: (&Screen, Frame), b: (&Screen, Frame)) {
        let screen = b.0;
        let frame = |(screen, (elapsed, gaze)): (&Screen, Frame)| {
            let mut px = vec![0u32; W * H];
            let mut c = Canvas {
                pixels: &mut px,
                width: W,
                height: H,
            };
            paint(&mut c, screen, elapsed, TIMING, gaze);
            px
        };
        let (pa, pb) = (frame(a), frame(b));
        let damage = Bounds::union(
            moving_bounds(a.0, a.1.0, TIMING, a.1.1, W, H),
            moving_bounds(b.0, b.1.0, TIMING, b.1.1, W, H),
        );
        let mut changed = 0;
        for y in 0..H {
            for x in 0..W {
                if pa[y * W + x] != pb[y * W + x] {
                    changed += 1;
                    let d = damage.expect("pixels changed but nothing is damaged");
                    assert!(
                        (d.x0..d.x1).contains(&x) && (d.y0..d.y1).contains(&y),
                        "pixel ({x}, {y}) changed outside {d:?} on {screen:?}"
                    );
                }
            }
        }
        assert!(changed > 0, "the two frames of {screen:?} should differ");
    }

    #[test]
    fn damage_covers_everything_that_moves() {
        let ms = Duration::from_millis;
        let target = |phase| Screen::Target {
            at: [0.9, 0.1],
            from: [0.5, 0.5],
            phase,
            step: 2,
            total: 14,
        };
        let seen = Some(([0.3, 0.4], true));
        for (screen, a, b) in [
            (target(Phase::Travel), (ms(100), None), (ms(117), None)),
            (target(Phase::Dwell), (ms(0), None), (ms(17), None)),
            (target(Phase::Dwell), (ms(500), None), (ms(517), None)),
            (target(Phase::Collecting), (ms(100), None), (ms(117), None)),
            (Screen::Computing, (ms(100), None), (ms(117), None)),
        ] {
            assert_damage_covers_changes(&screen, a, b);
        }
        let finished = Screen::Finished(Summary {
            id: 7,
            points: 14,
            mean_error: Some(0.02),
            blob: Vec::new(),
        });
        for next in [
            Some(([0.32, 0.45], false)),
            None,
            Some(([-1.0, -1.0], false)),
            Some(([0.95, 0.99], true)),
        ] {
            assert_damage_covers_changes(&finished, (ms(100), seen), (ms(117), next));
        }
    }

    /// A setup screen; `monitor_mm` fixes the spacing (a believable EDID).
    fn setup_with(width_px: f64, monitor_mm: Option<(f64, f64)>) -> Setup {
        let start = Start {
            guide_mm: 184.0,
            monitor_mm,
            current: None,
        };
        Setup {
            start,
            ticks: start.ticks(width_px, width_px * 9.0 / 16.0),
            width_px,
        }
    }

    fn setup_screen(width_px: f64) -> Setup {
        setup_with(width_px, Some((597.0, 336.0)))
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // reason: small sizes
    fn moving_the_setup_ticks_damages_only_the_bottom_band() {
        let w = W as f64;
        let frame = (Duration::ZERO, None);
        let fixed = setup_screen(w);
        let free = setup_with(w, None);
        for (before, ticks) in [
            (fixed, fixed.ticks.shifted(-37.0, w)),
            (fixed, fixed.ticks.shifted(80.0, w)),
            (free, free.ticks.widened(25.0, w)),
            (free, free.ticks.with_right(w, w)),
        ] {
            let after = Screen::DisplaySetup(Setup { ticks, ..before });
            assert_damage_covers_changes_between(
                (&Screen::DisplaySetup(before), frame),
                (&after, frame),
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // reason: small sizes
    fn a_fixed_pair_cannot_be_spread() {
        let w = W as f64;
        let h = H as f64;
        let fixed = setup_screen(w);
        let spread = Setup {
            ticks: fixed.ticks.widened(25.0, w),
            ..fixed
        };
        assert_eq!(spread.ticks_for(w, h), fixed.ticks_for(w, h));
    }

    #[test]
    fn the_reading_says_where_the_tracker_is() {
        let at = |offset_x_mm| {
            describe(&Choice {
                width_mm: 597.0,
                height_mm: 336.0,
                offset_x_mm,
            })
        };
        assert_eq!(
            at(1.0),
            "screen 597 x 336 mm, tracker 1.0 mm left of its centre"
        );
        assert_eq!(
            at(-12.34),
            "screen 597 x 336 mm, tracker 12.3 mm right of its centre"
        );
        assert_eq!(at(0.01), "screen 597 x 336 mm, tracker under its centre");
    }

    #[test]
    fn a_redraw_the_window_did_not_ask_for_damages_everything() {
        let last = Some(Presented {
            epoch: 3,
            size: (640, 360),
            moving: Bounds::around(100.0, 100.0, 5.0, 640, 360),
        });
        let moving = Bounds::around(120.0, 100.0, 5.0, 640, 360);
        assert_eq!(damage(last, 3, (640, 360), moving, false), None);
        assert_eq!(damage(last, 4, (640, 360), moving, true), None);
        assert_eq!(damage(last, 3, (641, 360), moving, true), None);
        assert_eq!(damage(None, 3, (640, 360), moving, true), None);
        let partial = damage(last, 3, (640, 360), moving, true).expect("partial");
        assert_eq!(partial, Bounds::union(last.and_then(|l| l.moving), moving));
    }

    #[test]
    fn bounds_are_clipped_to_the_surface() {
        assert_eq!(Bounds::around(-50.0, -50.0, 10.0, 640, 360), None);
        assert_eq!(
            Bounds::around(5.0, 355.0, 10.0, 640, 360),
            Some(Bounds {
                x0: 0,
                y0: 345,
                x1: 16,
                y1: 360
            })
        );
    }
}
