//! Head pose from the tracker's own IR frames.
//!
//! [`track`] wraps the `MediaPipe` face-landmark model (ONNX Runtime) and
//! fits the canonical mesh to the landmarks, one [`track::FaceFit`] per frame;
//! when it loses the face, `MediaPipe`'s `BlazeFace` face detector finds it
//! again. [`track::RestPose`] turns the fits into the legacy 6-DOF pose
//! relative to a calibrated rest position. The two models and the mesh are
//! embedded at build time, so no data files have to be installed alongside
//! the binaries.
//!
//! All three are `MediaPipe`'s and Apache-2.0, not MIT like the rest of the
//! workspace: `models/LICENSE` and `models/NOTICE` cover them, and
//! `models/README.md` says where they come from.

mod canonical;
mod detect;
pub mod track;
