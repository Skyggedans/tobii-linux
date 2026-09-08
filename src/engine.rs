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

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Clone, Copy)]
pub struct PoseSample {
    pub timestamp_us: i64,
    pub pos_cm: [f64; 3],  // tx, ty, tz
    pub rot_deg: [f64; 3], // yaw, pitch, roll
}

#[derive(Clone, Copy)]
pub struct GazeSample {
    pub timestamp_us: i64,
    pub gaze_valid: bool,
    pub gaze_norm: [f64; 2], // normalized screen, ~[-1, 1]
    pub present: bool,       // user presence
    pub pupil_mm: [f64; 2],  // left, right diameter; NaN if unavailable
}

/// One item out of the engine.
#[derive(Clone, Copy)]
pub enum Sample {
    Pose(PoseSample),
    Gaze(GazeSample),
}

/// Owns the device thread. The FFI layer drives `wait`/`drain` from a single
/// consumer thread (the Stream-Engine-style callback pump), so `&mut self`
/// access is sufficient and lock-free.
pub struct Engine {
    stop: Arc<AtomicBool>,
    recenter: Arc<AtomicBool>,
    head_wanted: Arc<AtomicBool>,
    rx: Receiver<Sample>,
    pending: VecDeque<Sample>,
    handle: Option<JoinHandle<()>>,
}

impl Engine {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let recenter = Arc::new(AtomicBool::new(false));
        let head_wanted = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let s = stop.clone();
        let rc = recenter.clone();
        let hw = head_wanted.clone();
        let handle = thread::spawn(move || {
            if let Err(e) = crate::device::run_gaze_engine(&s, &rc, &hw, &tx) {
                eprintln!("tobii engine stopped: {e:?}");
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
        self.recenter.store(true, Ordering::Relaxed);
    }

    /// Run head-pose inference on the image stream (costs CPU) while some
    /// client consumes head pose; frames are dropped otherwise.
    pub fn set_head_wanted(&self, wanted: bool) {
        self.head_wanted.store(wanted, Ordering::Relaxed);
    }

    /// Is the device thread still running (false once it has exited/failed)?
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
        while let Ok(sample) = self.rx.try_recv() {
            self.pending.push_back(sample);
        }
        self.pending.drain(..).collect()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
