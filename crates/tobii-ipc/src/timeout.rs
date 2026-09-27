//! How long a client waits for tobiid's answer to a request: longer than the
//! daemon may take over it ([`deadline`]), so that the client reports what
//! the daemon did. libtobii, `tobii-calibrate` and `ipc-probe` take their
//! timeouts from here. A client that gave up first would report a timeout
//! while the daemon carried the request out (a calibration started or saved,
//! a display area set), and its next request would wait behind it: the
//! daemon runs a connection's requests one at a time, and runs those it has
//! read even after the connection hangs up. A timeout is then the daemon's
//! own answer ([`status::TIMED_OUT`](crate::request::status::TIMED_OUT)),
//! or a daemon held up past its deadlines by what they leave out (its state
//! lock held while it stops an engine no client wants any more, whose thread
//! it joins, say), which may still carry the request out.
//!
//! A request answered from the facts the device reported at its last init
//! waits for its first init. One that needs the device live runs device
//! commands, which the daemon's engine takes only once the device streams (a
//! cold start takes ~12 s: an init, a re-open to prime the stream, a second
//! init): each may wait 30 s for the engine to take and write it, then its
//! own deadline for the answer ([`deadline::command`]). A request that runs
//! several commands waits for each in turn, the whole 30 s each at worst:
//! the engine may re-open the device between two of them (after a stall),
//! and the next waits for the stream to arm again. The worst cases are
//! [`worst`]'s, which tobiid's tests hold its requests' device commands to,
//! and each timeout is at least [`MARGIN`] past its request's, which is
//! checked at compile time:
//!
//! | Timeout | Requests ([`kind`]) | The daemon waits for | Worst case |
//! |---|---|---|---|
//! | [`FACTS`] 12 s | the facts reads (below) | the facts, at the device's first init | 10 s |
//! | [`FACTS`] 12 s | `DEVICE_NAME_SET` | nothing: it saves the name in a file | — |
//! | [`STATE`] 3 s | `STATE` of a flag or a number (`DEVICE_PAUSED`, `CALIBRATION_ID`, `CALIBRATION_ACTIVE`) | nothing: its own state | — |
//! | [`TIMESYNC`] 27 s | `TIMESYNC` | a gaze frame newer than the request | 25 s |
//! | [`DEVICE_PAUSE`] 60 s | `DEVICE_PAUSE` | another pause or resume to finish (20 s), then 3100 (6 s) | 20 + 36 = 56 s |
//! | [`DISPLAY_AREA_SET`] 75 s | `DISPLAY_AREA_SET` | 1440 (5 s); when the caller's calibration session ended meanwhile, 1440 again with the area that end put back | 35 + 35 = 70 s |
//! | [`CALIBRATION_START`] 190 s | `CALIBRATION_START` | 1010, 1060 (5 s each), 1110 (10 s); when one fails, 1020 (5 s) and 1110 (10 s) to put the calibration back | 35 + 35 + 40 + 35 + 40 = 185 s |
//! | [`CALIBRATION_STOP`] 120 s | `CALIBRATION_STOP` | 1020 (5 s), 1110 (10 s), then 1110 again for an engine started in place of a lost one, or 1440 (5 s) to put the display area back | 35 + 40 + 40 = 115 s |
//! | [`CALIBRATION_COLLECT_2D`] 40 s | `CALIBRATION_COLLECT_2D` | 1030 (5 s) | 35 s |
//! | [`CALIBRATION_DISCARD_2D`] 40 s | `CALIBRATION_DISCARD_2D` | 1080 (5 s) | 35 s |
//! | [`CALIBRATION_CLEAR`] 40 s | `CALIBRATION_CLEAR` | 1060 (5 s) | 35 s |
//! | [`CALIBRATION_COMPUTE`] 80 s | `CALIBRATION_COMPUTE` | 1070 (10 s), then 1100 (5 s) to read the result back | 40 + 35 = 75 s |
//! | [`CALIBRATION_RETRIEVE`] 40 s | `CALIBRATION_RETRIEVE` | 1100 (5 s) | 35 s |
//! | [`CALIBRATION_APPLY`] 45 s | `CALIBRATION_APPLY`, the built-in calibration (an empty payload) included | 1110 (10 s) | 40 s |
//!
//! The facts reads are `DEVICE_INFO`, `TRACK_BOX`, `DISPLAY_AREA_GET`,
//! `GEOMETRY_MOUNTING`, `DEVICE_NAME_GET` (before a name is set), `STATE` of
//! a string (`FAULT`, `WARNING`), `STREAM_TYPES` and
//! `HARDWARE_CONFIGURATION`. A SUBSCRIBE is not a request: the daemon
//! acknowledges it without the device.
//!
//! [`deadline`]: crate::deadline
//! [`deadline::command`]: crate::deadline::command
//! [`kind`]: crate::request::kind

use std::time::Duration;

use crate::deadline::worst;

/// How much longer than its request's worst case in the daemon a timeout is
/// at least: for what the table leaves out, which takes milliseconds (a bus
/// scan when no engine runs, the daemon's state lock but while it stops an
/// engine, a file saved, the socket both ways, the daemon's poll steps of 5
/// and 50 ms).
pub const MARGIN: Duration = Duration::from_secs(2);

/// Device facts are ready once the daemon's engine has initialised the
/// device, which a cold start can take most of 10 s to do.
pub const FACTS: Duration = Duration::from_secs(12);
/// A state the daemon keeps itself.
pub const STATE: Duration = Duration::from_secs(3);
/// A clock pair: the daemon waits 25 s for a gaze frame, since a cold
/// device streams gaze only after its second init.
pub const TIMESYNC: Duration = Duration::from_secs(27);
/// A pause or resume.
pub const DEVICE_PAUSE: Duration = Duration::from_secs(60);
/// A display-area write.
pub const DISPLAY_AREA_SET: Duration = Duration::from_secs(75);
/// A calibration start.
pub const CALIBRATION_START: Duration = Duration::from_secs(190);
/// A calibration stop, whether it keeps the session or discards it.
pub const CALIBRATION_STOP: Duration = Duration::from_secs(120);
/// Collecting a 2-D point.
pub const CALIBRATION_COLLECT_2D: Duration = Duration::from_secs(40);
/// Discarding a 2-D point.
pub const CALIBRATION_DISCARD_2D: Duration = Duration::from_secs(40);
/// Clearing the points collected.
pub const CALIBRATION_CLEAR: Duration = Duration::from_secs(40);
/// Computing a calibration.
pub const CALIBRATION_COMPUTE: Duration = Duration::from_secs(80);
/// Reading the active calibration.
pub const CALIBRATION_RETRIEVE: Duration = Duration::from_secs(40);
/// Writing a calibration, or going back to the built-in one.
pub const CALIBRATION_APPLY: Duration = Duration::from_secs(45);

/// Whether `timeout` outlasts a daemon that takes `worst` by [`MARGIN`]:
/// what a client's timeout for a request must do (see the module docs).
#[must_use]
pub const fn outlasts(timeout: Duration, worst: Duration) -> bool {
    timeout.as_nanos() >= worst.as_nanos() + MARGIN.as_nanos()
}

// The table, row by row.
const _: () = {
    assert!(outlasts(FACTS, worst::FACTS));
    assert!(outlasts(STATE, Duration::ZERO));
    assert!(outlasts(TIMESYNC, worst::TIMESYNC));
    assert!(outlasts(DEVICE_PAUSE, worst::DEVICE_PAUSE));
    assert!(outlasts(DISPLAY_AREA_SET, worst::DISPLAY_AREA_SET));
    assert!(outlasts(CALIBRATION_START, worst::CALIBRATION_START));
    assert!(outlasts(CALIBRATION_STOP, worst::CALIBRATION_STOP));
    assert!(outlasts(
        CALIBRATION_COLLECT_2D,
        worst::CALIBRATION_COLLECT_2D
    ));
    assert!(outlasts(
        CALIBRATION_DISCARD_2D,
        worst::CALIBRATION_DISCARD_2D
    ));
    assert!(outlasts(CALIBRATION_CLEAR, worst::CALIBRATION_CLEAR));
    assert!(outlasts(CALIBRATION_COMPUTE, worst::CALIBRATION_COMPUTE));
    assert!(outlasts(CALIBRATION_RETRIEVE, worst::CALIBRATION_RETRIEVE));
    assert!(outlasts(CALIBRATION_APPLY, worst::CALIBRATION_APPLY));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timeout_outlasts_a_worst_case_only_by_the_margin_or_more() {
        let worst = worst::CALIBRATION_RETRIEVE;

        assert!(outlasts(worst + MARGIN, worst));
        assert!(outlasts(worst + MARGIN * 2, worst));
        assert!(!outlasts(worst + MARGIN - Duration::from_nanos(1), worst));
        assert!(!outlasts(worst, worst), "equal gives up at the same time");
    }
}
