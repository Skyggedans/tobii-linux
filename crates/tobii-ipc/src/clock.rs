//! The host clock, shared by the daemon and `libtobii.so` so that the host
//! timestamps one process takes mean the same in every other on the host.
//! That holds within one time namespace: `CLOCK_MONOTONIC` is offset per
//! namespace (Linux 5.6+), so a client in a container with its own, talking
//! to the host's daemon through a mounted socket, reads a shifted clock.

/// The host clock, `CLOCK_MONOTONIC`, in microseconds.
///
/// `tobii_system_clock` returns it, every sample timestamp the daemon sends
/// is on it (bar gaze data's tracker time and raw gaze's time, which stay on
/// the device clock), and the engine stamps each gaze frame's receipt with
/// it, which is where the TIMESYNC clock pair takes its host time from. Like
/// the DLL's `QueryPerformanceCounter` it never steps back and its epoch is
/// undefined (on Linux it counts from boot and stops while suspended).
///
/// `clock_gettime` fails only for a clock the kernel lacks (`EINVAL`) or a
/// result it cannot write (`EFAULT`). Every Linux since 2.6 has
/// `CLOCK_MONOTONIC`, and the result is a local, so it cannot fail here;
/// were it to, the answer would be 0.
#[must_use]
#[allow(clippy::useless_conversion)] // reason: `time_t` and `c_long` are i32 on some targets
pub fn host_clock_us() -> i64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a live local, valid for the one `timespec` write, and
    // the clock id is a constant the kernel knows.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) } != 0 {
        return 0;
    }
    i64::from(now.tv_sec)
        .saturating_mul(1_000_000)
        .saturating_add(i64::from(now.tv_nsec) / 1_000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// How far the host clock reads behind the wall clock at least: 1e15 µs,
    /// some 31.7 years, where the wall clock is some 1.8e15 µs past its epoch
    /// and the monotonic one counts from boot.
    const FAR_FROM_THE_WALL_CLOCK_US: i64 = 1_000_000_000_000_000;

    /// `CLOCK_MONOTONIC` in microseconds, read without `host_clock_us`.
    #[allow(clippy::useless_conversion)] // reason: `time_t` and `c_long` are i32 on some targets
    fn monotonic_us() -> i64 {
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `now` is a live local, valid for one `timespec` write.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) };
        assert_eq!(rc, 0);
        i64::from(now.tv_sec) * 1_000_000 + i64::from(now.tv_nsec) / 1_000
    }

    #[test]
    fn the_host_clock_is_clock_monotonic_not_the_wall_clock() {
        let before = monotonic_us();
        let host = host_clock_us();
        let after = monotonic_us();
        assert!(
            before <= host && host <= after,
            "{host} is not between {before} and {after}"
        );

        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("after the epoch")
            .as_micros();
        let wall = i64::try_from(wall).expect("fits");
        assert!(
            (wall - host).abs() > FAR_FROM_THE_WALL_CLOCK_US,
            "{host} reads like the wall clock ({wall})"
        );
    }
}
