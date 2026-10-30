//! Owning the Tobii Eye Tracker 5 over USB.
//!
//! [`device`] is the transport and the live pipeline: open and claim the
//! device, replay the captured init sequence and ask the tracker the columns
//! of its streams, then demultiplex the gaze (0x500), presence (0x504), IR
//! image (0x50e) and, while it runs, custom (0x1770) streams that share bulk
//! endpoint 0x83, running head-pose inference on the images in the private
//! `pose` worker. [`engine`]
//! wraps that in a background thread producing [`engine::Sample`]s, which is
//! what the daemon consumes, and running the daemon's device commands, the
//! starts and stops of the tracker's catalogue streams among them.
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
#[cfg(feature = "head-pose")]
mod pose;
#[cfg(not(feature = "head-pose"))]
#[path = "pose_stub.rs"]
mod pose;
mod time_map;
