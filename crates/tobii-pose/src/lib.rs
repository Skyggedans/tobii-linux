//! Head pose from the tracker's own IR frames.
//!
//! [`track`] wraps the `MediaPipe` face-landmark model (ONNX Runtime), fits
//! the canonical mesh to the landmarks and reports a 6-DOF pose relative to a
//! calibrated rest position. The model and the mesh are embedded at build
//! time, so nothing has to be shipped alongside the binaries.

mod canonical;
pub mod track;
