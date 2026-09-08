//! Background engine that owns the USB device and produces tracking samples:
//! gaze point + user presence from the 0x83 gaze stream (~33 Hz) plus
//! gaze-independent 6DOF head pose from the device's own 280x280 IR image
//! stream (0x50e), which the firmware multiplexes on the same endpoint
//! concurrently. Head-pose inference only runs while a client wants it
//! (`set_head_wanted`).
//!
//! The UVC camera (interface 2) is never used by the engine: streaming it
//! throttles the 0x83 streams from ~33 Hz to <1 Hz (firmware mode, not
//! bandwidth — verified with `probe`). The UVC path survives only in the
//! standalone research subcommands (`camera`, `track`, `probe`).
//!
//! The `stop` / `recenter` / `head_wanted` flags are pure signals (no data is
//! published alongside them), so every access uses `Ordering::Relaxed`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::error;

/// One 6DOF head-pose estimate from the IR image stream.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PoseSample {
    /// Device timestamp of the source image frame, microseconds.
    pub timestamp_us: i64,
    /// Head translation `[tx, ty, tz]` in centimetres.
    pub pos_cm: [f64; 3],
    /// Head rotation `[yaw, pitch, roll]` in degrees.
    pub rot_deg: [f64; 3],
}

/// One gaze + presence sample from the 0x83 gaze stream.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GazeSample {
    /// Device timestamp, microseconds.
    pub timestamp_us: i64,
    /// Whether `gaze_norm` holds a usable gaze point.
    pub gaze_valid: bool,
    /// Gaze point in normalised screen coordinates, roughly `[-1, 1]` per axis.
    pub gaze_norm: [f64; 2],
    /// User presence as reported by the device.
    pub present: bool,
    /// Pupil diameter `[left, right]` in millimetres; `NaN` if unavailable.
    pub pupil_mm: [f64; 2],
}

/// One item out of the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Sample {
    /// Head pose (only while head pose is wanted, see [`Engine::set_head_wanted`]).
    Pose(PoseSample),
    /// Gaze point and presence.
    Gaze(GazeSample),
}

/// Owns the device thread. The daemon drives `wait`/`drain` from a single
/// consumer thread (the fan-out pump), so `&mut self` access is sufficient
/// and lock-free.
#[derive(Debug)]
pub struct Engine {
    stop: Arc<AtomicBool>,
    recenter: Arc<AtomicBool>,
    head_wanted: Arc<AtomicBool>,
    rx: Receiver<Sample>,
    pending: VecDeque<Sample>,
    handle: Option<JoinHandle<()>>,
}

impl Engine {
    /// Spawn the device thread and start streaming. Failures inside the thread
    /// are logged once; [`Engine::is_alive`] turns false when it has exited.
    #[must_use]
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let recenter = Arc::new(AtomicBool::new(false));
        let head_wanted = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let s = Arc::clone(&stop);
        let rc = Arc::clone(&recenter);
        let hw = Arc::clone(&head_wanted);
        let handle = thread::spawn(move || {
            if let Err(e) = crate::device::run_gaze_engine(&s, &rc, &hw, &tx) {
                error!(error = %format_args!("{e:#}"), "tobii engine stopped");
            }
        });
        Engine {
            stop,
            recenter,
            head_wanted,
            rx,
            pending: VecDeque::new(),
            handle: Some(handle),
        }
    }

    /// Ask the head tracker to recalibrate its rest pose on the next frames.
    pub fn request_recenter(&self) {
        // Relaxed: a pure signal, no data is published with it.
        self.recenter.store(true, Ordering::Relaxed);
    }

    /// Run head-pose inference on the image stream (costs CPU) while some
    /// client consumes head pose; frames are dropped otherwise.
    pub fn set_head_wanted(&self, wanted: bool) {
        // Relaxed: a pure signal, no data is published with it.
        self.head_wanted.store(wanted, Ordering::Relaxed);
    }

    /// Is the device thread still running (false once it has exited/failed)?
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Block up to `timeout` until at least one sample is available.
    pub fn wait(&mut self, timeout: Duration) -> bool {
        if !self.pending.is_empty() {
            return true;
        }
        match self.rx.recv_timeout(timeout) {
            Ok(sample) => {
                self.pending.push_back(sample);
                true
            }
            Err(_) => false,
        }
    }

    /// Take all queued samples (for the callback pump).
    pub fn drain(&mut self) -> Vec<Sample> {
        let mut out = Vec::new();
        self.drain_into(&mut out);
        out
    }

    /// Append all queued samples to `out`, reusing its allocation (for the
    /// per-tick fan-out loop).
    pub fn drain_into(&mut self, out: &mut Vec<Sample>) {
        while let Ok(sample) = self.rx.try_recv() {
            self.pending.push_back(sample);
        }
        out.extend(self.pending.drain(..));
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Relaxed: a pure signal; the join below is the synchronisation point.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // A panicked device thread has already logged; nothing to recover.
            let _ = h.join();
        }
    }
}
