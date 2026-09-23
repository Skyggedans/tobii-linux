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
//! - [`tlv`] — the TLV encoding inside every message, including the keyed
//!   fields of the stream messages.
//! - [`log`] — the `TBI5LOG1` capture format, written live and replayed offline.
//! - [`time`] — the wall clock those frames are stamped with.

pub mod decode;
pub mod image83;
pub mod log;
pub mod protocol;
pub mod time;
pub mod tlv;

/// A message captured from the device, as bytes: `fixtures/<name>.hex`. See
/// `fixtures/README.md` for where each one comes from.
#[cfg(test)]
macro_rules! fixture {
    ($name:literal) => {
        $crate::protocol::hex_to_bytes(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/",
            $name,
            ".hex"
        )))
        .expect("fixture is valid hex")
    };
}
#[cfg(test)]
pub(crate) use fixture;

/// The Windows engine's init replay, shipped by `tobii-usb`.
#[cfg(test)]
pub(crate) const INIT_PACKETS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tobii-usb/init_packets_ep.txt"
));
