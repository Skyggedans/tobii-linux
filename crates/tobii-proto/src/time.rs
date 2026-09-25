//! Wall-clock helper shared by the decoder and the packet log. The USB engine
//! stamps its reads with `tobii_ipc::host_clock_us` (`CLOCK_MONOTONIC`)
//! instead.
//!
//! It lives on its own so the pure codec modules do not have to depend on the
//! CLI's file sinks just to stamp a frame.

use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock microseconds since the Unix epoch (0 before the epoch,
/// saturating at `u64::MAX`).
#[must_use]
pub fn now_us() -> u64 {
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    u64::try_from(micros).unwrap_or(u64::MAX)
}
