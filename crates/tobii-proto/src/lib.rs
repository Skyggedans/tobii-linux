//! Wire formats of the Tobii Eye Tracker 5, with no device attached.
//!
//! The device multiplexes several streams on bulk endpoint 0x83; this crate
//! turns those bytes into values and back:
//!
//! - [`protocol`] — message framing, the `init_packets` text format, the
//!   bulk-read reassembler and the stream start/stop commands.
//! - [`decode`] — the 0x500 gaze stream: TLV fields and the derived
//!   [`decode::TrackingFrame`].
//! - [`image83`] — the 0x50e IR image stream (280x280 8-bit frames).
//! - [`log`] — the `TBI5LOG1` capture format, written live and replayed offline.
//! - [`time`] — the wall clock those frames are stamped with.

pub mod decode;
pub mod image83;
pub mod log;
pub mod protocol;
pub mod time;
