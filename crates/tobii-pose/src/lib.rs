//! Head pose from the tracker's own IR frames.
//!
//! [`track`] wraps the `MediaPipe` face-landmark model (ONNX Runtime), fits
//! the canonical mesh to the landmarks and reports a 6-DOF pose relative to a
//! calibrated rest position. The model and the mesh are embedded at build
//! time, so no data files have to be installed alongside the binaries.
//!
//! Both are `MediaPipe`'s and Apache-2.0, not MIT like the rest of the
//! workspace: `models/LICENSE` and `models/NOTICE` cover them, and
//! `models/README.md` says where they come from.

mod canonical;
pub mod track;
