//! When the watchdog may start the engine again after engines that ended
//! before the tracker was ready.
//!
//! An engine the tracker never got ready for (no init went through: the
//! tracker may not be opened, or no open worked) would end the same way if
//! it were started again at once, so the watchdog spaces the next out: no
//! sooner than [`FIRST_WAIT`] after the first such engine started, twice as
//! long after each one after it in a row, and never more than [`MAX_WAIT`].
//! Counting from the start rather than the end keeps the first step at the
//! watchdog's own pace (the pass after the one that started an engine that
//! failed at once starts the next) and bounds how often engines start,
//! however long each took to fail. Without it, every watchdog pass (3 s
//! apart) that found no engine running started one, for as long as one was
//! wanted, and each logged its failed opens.
//!
//! The backoff starts afresh when an engine gets the tracker ready (see
//! [`crate::daemon::State::observe`]), when the tracker is plugged back in
//! (the watchdog finds it at another address than before, or back after it
//! found it gone; say after installing the udev rule), when no engine is
//! wanted any more, and with the daemon. It holds the watchdog alone: a
//! client's subscription or request still starts an engine at once (see
//! [`crate::daemon::State::ensure_engine`]).

use std::time::{Duration, Instant};

use tobii_usb::device::BusAddress;

/// How long after the first engine in a row that ended before the tracker
/// was ready started the watchdog waits: its own period, so that the pass
/// after the one that started an engine that failed at once starts the next.
const FIRST_WAIT: Duration = Duration::from_secs(3);

/// The longest the watchdog waits between engine starts.
const MAX_WAIT: Duration = Duration::from_secs(60);

/// Engines in a row that ended before the tracker was ready, when the
/// watchdog may start the next, and where the tracker was last seen. Time is
/// passed in, so that it is tested without waiting.
#[derive(Debug, Default)]
pub(crate) struct Backoff {
    /// Engines in a row that ended before the tracker was ready.
    in_a_row: u32,
    /// The watchdog starts no engine before this.
    until: Option<Instant>,
    /// What the last look at the bus found since an engine last ended or the
    /// backoff last started afresh.
    seen: Seen,
}

/// What a look at the bus found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Seen {
    /// No look since an engine last ended or the backoff started afresh: an
    /// engine may have re-enumerated the tracker (a USB reset), so the next
    /// look only notes where it is.
    #[default]
    Nothing,
    /// No tracker on the bus.
    Absent,
    /// The tracker, at this address.
    At(BusAddress),
}

impl Backoff {
    /// The wait after `in_a_row` engines in a row that ended before the
    /// tracker was ready (none after none).
    pub(crate) fn wait_after(in_a_row: u32) -> Duration {
        let Some(doublings) = in_a_row.checked_sub(1) else {
            return Duration::ZERO;
        };
        // From 2^5 on (96 s) the cap applies; capping the shift keeps it in
        // range.
        FIRST_WAIT
            .saturating_mul(1 << doublings.min(5))
            .min(MAX_WAIT)
    }

    /// Engines in a row that ended before the tracker was ready.
    pub(crate) fn in_a_row(&self) -> u32 {
        self.in_a_row
    }

    /// An engine started at `started` ended without getting the tracker
    /// ready: back off one step. The wait, from `started`, before the
    /// watchdog may start the next. The look at the bus before is forgotten:
    /// the tracker is back if it was found gone, since the engine found it,
    /// and may have moved if the engine reset it.
    pub(crate) fn engine_failed(&mut self, started: Instant) -> Duration {
        self.in_a_row = self.in_a_row.saturating_add(1);
        let wait = Self::wait_after(self.in_a_row);
        self.until = started.checked_add(wait);
        self.seen = Seen::Nothing;
        wait
    }

    /// Start afresh: an engine got the tracker ready, or none is wanted any
    /// more. Whether an engine had ended before the tracker was ready since
    /// the backoff last did.
    pub(crate) fn reset(&mut self) -> bool {
        let backed_off = self.in_a_row > 0;
        *self = Self::default();
        backed_off
    }

    /// What a look at the bus `found`: where the tracker is, if on it. The
    /// tracker at another address than the last look found, or back after
    /// it found it gone, was plugged back in: the backoff starts afresh.
    /// Whether that stopped one under way.
    pub(crate) fn saw_bus(&mut self, found: Option<BusAddress>) -> bool {
        let now = found.map_or(Seen::Absent, Seen::At);
        let before = std::mem::replace(&mut self.seen, now);
        let replugged = match (before, now) {
            (Seen::Absent, Seen::At(_)) => true,
            (Seen::At(was), Seen::At(is)) => was != is,
            _ => false,
        };
        if !replugged {
            return false;
        }
        let backed_off = self.reset();
        self.seen = now;
        backed_off
    }

    /// How long the watchdog must still wait at `now` before it starts an
    /// engine; `None` once it may.
    pub(crate) fn wait_left(&self, now: Instant) -> Option<Duration> {
        self.until
            .and_then(|until| until.checked_duration_since(now))
            .filter(|left| !left.is_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    /// The tracker at `address` on bus 1.
    fn at(address: u8) -> Option<BusAddress> {
        Some(BusAddress { bus: 1, address })
    }

    #[test]
    fn the_wait_doubles_from_three_seconds_up_to_a_minute() {
        let waits: Vec<u64> = (0..=9).map(|n| Backoff::wait_after(n).as_secs()).collect();
        assert_eq!(waits, [0, 3, 6, 12, 24, 48, 60, 60, 60, 60]);
        assert_eq!(Backoff::wait_after(u32::MAX), MAX_WAIT);
    }

    #[test]
    fn each_failed_engine_holds_the_watchdog_off_for_the_next_wait_from_its_start() {
        let t0 = Instant::now();
        let mut backoff = Backoff::default();
        assert_eq!(backoff.wait_left(t0), None, "nothing to wait for yet");

        assert_eq!(backoff.engine_failed(t0), 3 * S);
        assert_eq!(backoff.wait_left(t0), Some(3 * S));
        assert_eq!(backoff.wait_left(t0 + 2 * S), Some(S));
        assert_eq!(backoff.wait_left(t0 + 3 * S), None);

        let t1 = t0 + 4 * S;
        assert_eq!(backoff.engine_failed(t1), 6 * S);
        assert_eq!(backoff.wait_left(t1 + 5 * S), Some(S));
        assert_eq!(backoff.wait_left(t1 + 6 * S), None);
        assert_eq!(backoff.in_a_row(), 2);
    }

    #[test]
    fn a_ready_tracker_starts_the_backoff_afresh() {
        let t0 = Instant::now();
        let mut backoff = Backoff::default();
        assert!(!backoff.reset(), "nothing was backed off");
        for _ in 0..4 {
            let _ = backoff.engine_failed(t0);
        }

        assert!(backoff.reset());

        assert_eq!(backoff.wait_left(t0), None);
        assert_eq!(backoff.engine_failed(t0), 3 * S, "the first step again");
    }

    #[test]
    fn the_tracker_back_on_the_bus_after_an_absence_starts_the_backoff_afresh() {
        let t0 = Instant::now();
        let mut backoff = Backoff::default();
        let _ = backoff.engine_failed(t0);
        let _ = backoff.engine_failed(t0);
        assert!(!backoff.saw_bus(at(7)), "never seen gone: no reset");
        assert_eq!(backoff.in_a_row(), 2);

        assert!(!backoff.saw_bus(None));
        assert_eq!(backoff.wait_left(t0), Some(6 * S), "gone: still held off");
        assert!(backoff.saw_bus(at(8)), "back");

        assert_eq!(backoff.wait_left(t0), None);
        assert_eq!(backoff.in_a_row(), 0);
        assert!(!backoff.saw_bus(at(8)), "once per re-plug");
    }

    #[test]
    fn the_tracker_at_another_address_was_plugged_back_in_however_briefly() {
        let t0 = Instant::now();
        let mut backoff = Backoff::default();
        let _ = backoff.engine_failed(t0);
        assert!(!backoff.saw_bus(at(7)));
        assert!(!backoff.saw_bus(at(7)), "where it was");

        assert!(backoff.saw_bus(at(9)), "re-plugged between two looks");

        assert_eq!(backoff.wait_left(t0), None);
        assert_eq!(backoff.in_a_row(), 0);
        let _ = backoff.engine_failed(t0);
        assert!(!backoff.saw_bus(at(9)), "noted, after the engine ended");
        assert!(
            backoff.saw_bus(Some(BusAddress { bus: 2, address: 9 })),
            "another port"
        );
    }

    #[test]
    fn an_engine_that_ended_leaves_no_look_to_compare_with() {
        let t0 = Instant::now();
        let mut backoff = Backoff::default();

        // Seen gone, then plugged in: the engine that failed was started on
        // the tracker already back.
        assert!(!backoff.saw_bus(None));
        let _ = backoff.engine_failed(t0);
        assert!(!backoff.saw_bus(at(7)));
        assert_eq!(backoff.wait_left(t0), Some(3 * S), "still held off");

        // Seen at one address, then moved by the engine's USB reset.
        let _ = backoff.engine_failed(t0);
        assert!(!backoff.saw_bus(at(8)));
        assert_eq!(backoff.in_a_row(), 2);
    }
}
