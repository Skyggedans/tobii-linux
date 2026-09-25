//! Owning the Tobii Eye Tracker 5 over USB.
//!
//! [`device`] is the transport and the live pipeline: open and claim the
//! device, replay the captured init sequence, then demultiplex the gaze
//! (0x500) and IR image (0x50e) streams that share bulk endpoint 0x83, running
//! head-pose inference on the images. [`engine`] wraps that in a background
//! thread producing [`engine::Sample`]s, which is what the daemon consumes.
//! Each sample gives its device timestamp on the host clock too, as the
//! private `time_map` estimates it from the arrivals.
//!
//! [`calibration`] adapts the embedded init capture to the user: their saved
//! calibration and display area replace the capture author's.
//!
//! The init capture is embedded, so the binaries need no runtime data files.

pub mod calibration;
pub mod device;
pub mod engine;
mod time_map;
