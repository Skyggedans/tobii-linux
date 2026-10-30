//! Stand-in for [`pose`](super::pose) when built without the `head-pose`
//! feature: the same single-slot [`Mailbox`] the USB reader hands images to
//! (so `device.rs` needs no `#[cfg]` of its own), but a [`run_worker`] that
//! never starts an ONNX Runtime session and produces no [`PoseSample`].
//! Built for targets where linking the prebuilt ONNX Runtime binaries is
//! impractical (e.g. Android) and gaze (stream 0x500) is all that's wanted.

use std::sync::mpsc::Sender;
use std::sync::{Condvar, Mutex, PoisonError};

use tracing::info;

use crate::engine::{ImageSample, Sample, Shared};

/// The single-slot hand-off of the newest image from the USB reader to the
/// pose worker. With no worker reading it here, a put image is simply
/// dropped (replaced, in turn, by the next one).
#[derive(Debug, Default)]
pub(crate) struct Mailbox {
    slot: Mutex<Option<ImageSample>>,
    wake: Condvar,
}

impl Mailbox {
    /// Hand `image` over; with no worker to take it, it is just dropped at
    /// the next `put` or when the mailbox itself is.
    pub(crate) fn put(&self, image: ImageSample) {
        *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(image);
    }

    /// Wake the worker, so that it sees a stop at once. No-op: there is no
    /// worker thread in this build.
    pub(crate) fn wake(&self) {
        self.wake.notify_all();
    }
}

/// Logs that this build has no head-pose worker and returns at once, the
/// same as the real worker does when its models fail to load: the engine
/// runs on without head pose.
pub(crate) fn run_worker(_mailbox: &Mailbox, _shared: &Shared, _tx: &Sender<Sample>) {
    info!("image83 pose worker disabled: built without the head-pose feature");
}
