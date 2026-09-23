//! The calibration blob of the Tobii Eye Tracker 5, and where a user's own
//! calibration is kept.
//!
//! The device computes the blob (command 1070) and hands it out on request
//! (1100); the host stores it and uploads it again at every init (1110). Most
//! of it is opaque. What is understood — the header and the trailing list of
//! calibration points — is in [`blob`]; [`store`] keeps one blob per user so
//! the daemon can replace the calibration embedded in its init replay with
//! the user's own.

pub mod blob;
pub mod store;

pub use blob::{BlobError, BlobHeader, BlobInfo, PointRecord, STIMULUS_POINTS};
