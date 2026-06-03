//! Background engine that owns the USB device and produces tracking samples.
//!
//! The device cannot run the processed (0x83) pipeline and the raw camera at
//! the same time: pulling the UVC camera throttles 0x83 from ~33 Hz to <1 Hz
//! (firmware mode, not bandwidth — verified with `probe`). So an `Engine` runs
//! in exactly **one** mode at a time:
//!   * `HeadCamera` — gaze-independent 6DOF head pose from the IR camera (~8 fps)
//!   * `Gaze`       — gaze point + user presence from the 0x83 stream (~33 Hz)
//! The FFI device picks the mode from the first subscription; mixing the two on
//! one physical device is rejected.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EngineMode {
    HeadCamera,
    Gaze,
}

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
}

/// One item out of the engine; which variant arrives depends on the mode.
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
    rx: Receiver<Sample>,
    pending: VecDeque<Sample>,
    handle: Option<JoinHandle<()>>,
    mode: EngineMode,
}

impl Engine {
    pub fn start(mode: EngineMode) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let s = stop.clone();
        let handle = thread::spawn(move || {
            let result = match mode {
                EngineMode::HeadCamera => crate::device::run_pose_engine(&s, &tx),
                EngineMode::Gaze => crate::device::run_gaze_engine(&s, &tx),
            };
            if let Err(e) = result {
                eprintln!("tobii engine ({mode:?}) stopped: {e:?}");
            }
        });
        Engine {
            stop,
            rx,
            pending: VecDeque::new(),
            handle: Some(handle),
            mode,
        }
    }

    pub fn mode(&self) -> EngineMode {
        self.mode
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
