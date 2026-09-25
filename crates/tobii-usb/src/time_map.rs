//! Device time to host time, for the samples the engine hands out.
//!
//! The tracker stamps every stream message with its own clock (key 1, in
//! microseconds), and the host reads the message some time later: the
//! transfer, the controller's polling and the reader's own schedule. The
//! Stream Engine adds one offset to every tracker timestamp, taken from round
//! trips to its service; nothing here reads the tracker's clock on demand, so
//! the offset is estimated from the arrivals themselves instead. The smallest
//! `host_rx - device` over the last 120 s (twelve buckets of 10 s) belongs to
//! the fastest delivery, the one nearest the true offset plus the fixed part
//! of the latency. The window follows the drift between the two clocks (5 to
//! 13 ppm in the captures, some 50 ms an hour for a fixed offset) at the cost
//! of up to that drift over 120 s, under 2 ms.
//!
//! Gaze frames and IR images both feed it: images reach the host 3 to 5.5 ms
//! closer to their device timestamp than gaze frames do (the session1-3
//! captures), so a map fed with gaze alone would put every image, and every
//! head pose made from one, after it arrived. With images off
//! (`TOBII_NO_IMAGE`, or a stream start that failed) gaze alone feeds it.
//!
//! One map serves one open of the device: a new open may restart the
//! tracker's clock (the Linux logs all begin at a device time of 12 to 26 s),
//! and a new map is the reset. Should the device time of a stream go back
//! within an open all the same, the map starts over from that arrival.

use std::collections::VecDeque;

use tracing::{debug, warn};

/// How long one bucket of the sliding minimum spans, host microseconds.
const BUCKET_US: i64 = 10_000_000;

/// How far back the minimum looks: twelve buckets.
const WINDOW_US: i64 = 12 * BUCKET_US;

/// A stream whose timestamps the map converts. Each keeps its own order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stream {
    /// 0x500 gaze frames, which feed the map.
    Gaze,
    /// 0x504 presence, which does not (it comes only on change).
    Presence,
    /// 0x50e IR images, which feed the map.
    Image,
}

/// The smallest `host_rx - device` seen in one bucket of the window.
#[derive(Debug, Clone, Copy)]
struct Bucket {
    /// Host time of the arrival that opened the bucket.
    opened_us: i64,
    /// The smallest offset among the arrivals in it.
    min_offset_us: i64,
}

/// What the map remembers of one stream.
#[derive(Debug, Clone, Copy, Default)]
struct StreamState {
    /// Device time of the stream's last arrival: one earlier than this means
    /// the device clock restarted.
    last_device_us: Option<i64>,
    /// The last host time handed out for the stream.
    last_stamp_us: Option<i64>,
}

/// Maps device timestamps to the host clock ([`tobii_ipc::host_clock_us`])
/// for one open of the device (see the module doc).
#[derive(Debug, Default)]
pub(crate) struct TimeMap {
    /// The window, oldest bucket first; at most twelve.
    buckets: VecDeque<Bucket>,
    /// Each stream's order and last device time, one field per [`Stream`]
    /// (see [`TimeMap::state`]).
    gaze: StreamState,
    presence: StreamState,
    image: StreamState,
}

impl TimeMap {
    /// Learn from a message of `stream` stamped `device_us` by the device and
    /// read by the host at `rx_us`. A device time before the stream's last
    /// starts the map over from this arrival.
    pub(crate) fn observe(&mut self, stream: Stream, device_us: i64, rx_us: i64) {
        let last_device_us = self.state(stream).last_device_us;
        if let Some(last_device_us) = last_device_us.filter(|&last| device_us < last) {
            warn!(
                ?stream,
                device_us, last_device_us, "device clock went back; host time map starts over"
            );
            self.restart();
        }
        self.state(stream).last_device_us = Some(device_us);

        self.expire(rx_us);
        let offset_us = rx_us.saturating_sub(device_us);
        match self.buckets.back_mut() {
            Some(newest) if rx_us < newest.opened_us.saturating_add(BUCKET_US) => {
                newest.min_offset_us = newest.min_offset_us.min(offset_us);
            }
            _ => {
                self.buckets.push_back(Bucket {
                    opened_us: rx_us,
                    min_offset_us: offset_us,
                });
                debug!(
                    offset_us = ?self.offset_us(),
                    buckets = self.buckets.len(),
                    "host time map: bucket opened"
                );
            }
        }
    }

    /// The current estimate of `host - device`, microseconds: the smallest
    /// offset in the window. `None` before the first arrival, and once the
    /// window has passed the last.
    pub(crate) fn offset_us(&self) -> Option<i64> {
        self.buckets.iter().map(|b| b.min_offset_us).min()
    }

    /// The host time of a message of `stream` stamped `device_us` by the
    /// device and read by the host at `rx_us`: the device time plus the
    /// offset over the 120 s before `rx_us`, or `rx_us` while there is none
    /// (no arrival yet, or none in that window: a paused device sends
    /// nothing, and the offset from before a long pause is off by the drift
    /// since).
    ///
    /// It is no later than `rx_us`: the message cannot have been sent after
    /// it was read. A gaze frame or image observed first always maps at or
    /// before its read time anyway; presence, which is not observed, might
    /// not.
    ///
    /// Within a stream the host times strictly increase, as the device times
    /// do: when the estimate falls (a faster arrival, the device clock
    /// restarting) by more than the time between two messages, the next one
    /// gets its predecessor's host time plus 1 us rather than an earlier one.
    /// The order wins over the read time, which it passes only should two
    /// messages of a stream share a read time (the messages of one transfer
    /// do), and then by one.
    /// The steps seen in the captures, under 10 ms and mostly under 1 ms, never
    /// get near the 30 ms between two frames. The order survives a restart of
    /// the map, since host time runs on.
    pub(crate) fn stamp(&mut self, stream: Stream, device_us: i64, rx_us: i64) -> i64 {
        self.expire(rx_us);
        let mapped = self
            .offset_us()
            .map_or(rx_us, |offset| device_us.saturating_add(offset))
            .min(rx_us);
        self.in_order(stream, mapped)
    }

    /// The host time of a message of `stream` that carries no device time,
    /// read by the host at `rx_us`: its read time, in the stream's order (see
    /// [`TimeMap::stamp`]).
    pub(crate) fn stamp_read(&mut self, stream: Stream, rx_us: i64) -> i64 {
        self.in_order(stream, rx_us)
    }

    /// `stamp`, or the stream's last host time plus 1 us should that be
    /// later; recorded as the stream's last.
    fn in_order(&mut self, stream: Stream, stamp: i64) -> i64 {
        let state = self.state(stream);
        let stamp = state
            .last_stamp_us
            .map_or(stamp, |last| stamp.max(last.saturating_add(1)));
        state.last_stamp_us = Some(stamp);
        stamp
    }

    /// Drop the buckets opened 120 s or more before `now_us`.
    fn expire(&mut self, now_us: i64) {
        let horizon = now_us.saturating_sub(WINDOW_US);
        while self.buckets.front().is_some_and(|b| b.opened_us <= horizon) {
            self.buckets.pop_front();
        }
    }

    /// Forget the window and the device times, after the device clock went
    /// back. The streams' last host times stay: host time runs on.
    ///
    /// A message of another stream that the device stamped before its clock
    /// went back but that is read after this restart counts as an arrival on
    /// the new clock: its offset, short by the size of the jump, becomes the
    /// minimum, and until that stream's next message (itself on the new
    /// clock, so earlier on the device: another restart) every stamp is its
    /// stream's last plus 1 us. That is one frame, some 30 ms. Fencing the
    /// other streams off until they too go back would instead shut one out
    /// for the rest of the open should the restart come from a single bad
    /// timestamp rather than the clock; and the clock has only been seen to
    /// restart between opens.
    fn restart(&mut self) {
        self.buckets.clear();
        for state in [&mut self.gaze, &mut self.presence, &mut self.image] {
            state.last_device_us = None;
        }
    }

    /// What the map remembers of `stream`.
    fn state(&mut self, stream: Stream) -> &mut StreamState {
        match stream {
            Stream::Gaze => &mut self.gaze,
            Stream::Presence => &mut self.presence,
            Stream::Image => &mut self.image,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000;
    const S: i64 = 1_000_000;
    /// Time between two frames of a 33 Hz stream.
    const FRAME: i64 = 30 * MS;
    /// One frame in this many (7 s at 33 Hz) arrives at the latency floor.
    const FLOOR_EVERY: i64 = 233;
    /// The latest a frame arrives after the latency floor.
    const MAX_LATE: i64 = 12 * MS;

    /// How long after the latency floor frame `i` arrives: on it for the last
    /// frame of every run of `FLOOR_EVERY`, else a deterministic spread of
    /// 1 ms up to `MAX_LATE`.
    fn late_us(i: i64) -> i64 {
        if i % FLOOR_EVERY == FLOOR_EVERY - 1 {
            0
        } else {
            MS + (i * 7_919) % (MAX_LATE - MS)
        }
    }

    /// Host time the map would give each of `n` frames of `stream` whose
    /// device times start at `device0` and follow `FRAME` apart, each read
    /// `latency(i)` after `device + offset`. Returns (device, rx, stamp).
    fn run(
        map: &mut TimeMap,
        stream: Stream,
        device0: i64,
        offset: i64,
        n: i64,
        latency: impl Fn(i64) -> i64,
    ) -> Vec<(i64, i64, i64)> {
        (0..n)
            .map(|i| {
                let device = device0 + i * FRAME;
                let rx = device + offset + latency(i);
                map.observe(stream, device, rx);
                (device, rx, map.stamp(stream, device, rx))
            })
            .collect()
    }

    #[test]
    fn before_any_arrival_a_message_is_stamped_when_it_was_read() {
        let mut map = TimeMap::default();
        assert_eq!(map.offset_us(), None);
        assert_eq!(map.stamp(Stream::Presence, 5 * S, 90 * S), 90 * S);
    }

    #[test]
    fn the_first_arrival_is_stamped_when_it_was_read() {
        let mut map = TimeMap::default();
        map.observe(Stream::Gaze, 5 * S, 90 * S);
        assert_eq!(map.offset_us(), Some(85 * S));
        assert_eq!(map.stamp(Stream::Gaze, 5 * S, 90 * S), 90 * S);
    }

    #[test]
    fn a_message_is_never_stamped_after_it_was_read() {
        let mut map = TimeMap::default();
        map.observe(Stream::Gaze, 5 * S, 90 * S + 8 * MS);
        // Presence is not observed, and may arrive faster than any gaze frame.
        assert_eq!(
            map.stamp(Stream::Presence, 5 * S + 10 * MS, 90 * S + 12 * MS),
            90 * S + 12 * MS
        );
    }

    /// Gaze alone (images off): a latency of 2 ms plus the jitter of
    /// [`late_us`], with a frame at the 2 ms floor every 7 s. From the first
    /// such frame on, every frame is stamped at its capture time plus the
    /// floor.
    #[test]
    fn stamps_settle_on_the_latency_floor_through_jitter() {
        let mut map = TimeMap::default();
        let (offset, floor) = (40 * S, 2 * MS);
        let frames = run(&mut map, Stream::Gaze, 3 * S, offset, 1_000, |i| {
            floor + late_us(i)
        });

        let first_floor = usize::try_from(FLOOR_EVERY - 1).expect("an index");
        for (i, &(device, rx, stamp)) in frames.iter().enumerate() {
            assert!(stamp <= rx, "frame {i} stamped {stamp}, read at {rx}");
            if i >= first_floor {
                assert_eq!(stamp, device + offset + floor, "frame {i}");
            }
        }
        // Before the floor was seen the stamps were late, but by no more
        // than the jitter.
        let (device, _, stamp) = frames[first_floor / 2];
        let truth = device + offset + floor;
        assert!(stamp > truth && stamp <= truth + MAX_LATE);
    }

    #[test]
    fn an_old_minimum_leaves_the_window_after_120_s() {
        let mut map = TimeMap::default();
        let offset = 40 * S;
        // One frame 5 ms faster than all the others, then a 5 ms floor.
        map.observe(Stream::Gaze, 0, offset);
        let frames = run(&mut map, Stream::Gaze, FRAME, offset, 4_500, |_| 5 * MS);

        let floor_at = |t: i64| {
            let (device, _, stamp) = frames[usize::try_from(t / FRAME).expect("index") - 1];
            stamp - device - offset
        };
        assert_eq!(
            floor_at(WINDOW_US - 2 * BUCKET_US),
            0,
            "still inside the window"
        );
        assert_eq!(
            floor_at(WINDOW_US + BUCKET_US),
            5 * MS,
            "the fast frame has left it"
        );
    }

    #[test]
    fn an_offset_from_before_the_window_maps_nothing() {
        let mut map = TimeMap::default();
        map.observe(Stream::Gaze, 10 * S, 50 * S + 4 * MS);
        // Presence on resume from a pause longer than the window, before any
        // gaze frame or image: the old offset is off by the drift since.
        let rx = 50 * S + WINDOW_US + BUCKET_US;
        assert_eq!(
            map.stamp(Stream::Presence, 10 * S + WINDOW_US + BUCKET_US, rx),
            rx
        );
        assert_eq!(map.offset_us(), None);
    }

    #[test]
    fn a_message_without_a_device_time_is_stamped_when_read_in_order() {
        let mut map = TimeMap::default();
        map.observe(Stream::Image, 10 * S, 50 * S + MS);
        assert_eq!(map.stamp(Stream::Image, 10 * S, 50 * S + MS), 50 * S + MS);
        assert_eq!(
            map.stamp_read(Stream::Image, 50 * S + 31 * MS),
            50 * S + 31 * MS
        );
        // Read in the same microsecond as the last: after it all the same.
        assert_eq!(
            map.stamp_read(Stream::Image, 50 * S + 31 * MS),
            50 * S + 31 * MS + 1
        );
        assert_eq!(map.offset_us(), Some(40 * S + MS), "nothing learnt");
    }

    /// An hour at 33 Hz with the device clock 15 ppm fast, then 15 ppm slow:
    /// once warmed up the stamps stay within 2 ms of the capture time plus the
    /// latency floor, where a fixed offset would be some 54 ms off by the end.
    #[test]
    fn the_map_follows_the_drift_between_the_clocks() {
        for ppm in [15, -15] {
            let mut map = TimeMap::default();
            let (offset, floor) = (40 * S, 2 * MS);
            let mut fixed = None;
            let mut worst: i64 = 0;
            let mut fixed_error = 0;
            for i in 0..120_000 {
                // Host time of the capture, and the device's reading of it.
                let host = i * FRAME;
                let device = 7 * S + host + host * ppm / 1_000_000;
                let rx = host + offset + floor + late_us(i);
                map.observe(Stream::Gaze, device, rx);
                let stamp = map.stamp(Stream::Gaze, device, rx);
                let truth = host + offset + floor;
                if host > 20 * S {
                    worst = worst.max((stamp - truth).abs());
                }
                let fixed = *fixed.get_or_insert(rx - device);
                fixed_error = (device + fixed - truth).abs();
            }
            assert!(worst <= 2 * MS, "{ppm} ppm: {worst} us off");
            assert!(
                fixed_error > 50 * MS,
                "{ppm} ppm: fixed only {fixed_error} us off"
            );
        }
    }

    #[test]
    fn a_device_clock_that_goes_back_starts_the_map_over() {
        let mut map = TimeMap::default();
        run(&mut map, Stream::Gaze, 600 * S, 40 * S, 100, |_| 3 * MS);
        run(&mut map, Stream::Image, 603 * S, 40 * S, 100, |_| MS);

        // The device clock restarts near 12 s, a minute later: the old offset
        // would stamp these frames 648 s before they were read.
        let rx = 700 * S;
        let device = 12 * S;
        map.observe(Stream::Gaze, device, rx);
        assert_eq!(
            map.offset_us(),
            Some(rx - device),
            "nothing of the old offset"
        );
        assert_eq!(map.stamp(Stream::Gaze, device, rx), rx);
        // The image stream's old device times went with it: its next frame,
        // also near 12 s, is no restart, and the gaze frame's smaller offset
        // stays (another restart would leave only the image's, 5 ms more).
        map.observe(Stream::Image, device + 5 * MS, rx + 10 * MS);
        assert_eq!(map.offset_us(), Some(rx - device));
        assert_eq!(
            map.stamp(Stream::Image, device + 5 * MS, rx + 10 * MS),
            rx + 5 * MS
        );
    }

    #[test]
    fn one_stream_arriving_after_another_is_no_restart() {
        let mut map = TimeMap::default();
        // An image read after a gaze frame that was stamped later.
        map.observe(Stream::Gaze, 10 * S, 50 * S);
        map.observe(Stream::Image, 10 * S - 5 * MS, 50 * S + 2 * MS);
        assert_eq!(
            map.offset_us(),
            Some(40 * S),
            "the gaze frame's offset kept"
        );
    }

    #[test]
    fn a_streams_host_times_keep_increasing_when_the_estimate_falls() {
        let mut map = TimeMap::default();
        // Presence before any arrival: stamped when read.
        assert_eq!(map.stamp(Stream::Presence, 10 * S, 50 * S), 50 * S);
        // A gaze frame then shows that presence was read 39 ms after it was
        // sent: the next one, 1 ms later on the device, maps 38 ms before the
        // first's host time.
        map.observe(Stream::Gaze, 10 * S + 2 * MS, 49 * S + 963 * MS);
        assert_eq!(
            map.stamp(Stream::Presence, 10 * S + MS, 50 * S + 5 * MS),
            50 * S + 1,
            "one microsecond after the last"
        );
        // A restart keeps the order too: with the device clock back at 1 s,
        // presence at 0 maps to 49.01 s.
        map.observe(Stream::Gaze, S, 50 * S + 10 * MS);
        assert_eq!(map.stamp(Stream::Presence, 0, 50 * S + 20 * MS), 50 * S + 2);
    }

    #[test]
    fn extreme_times_saturate_instead_of_overflowing() {
        let mut map = TimeMap::default();
        map.observe(Stream::Gaze, i64::MIN, i64::MAX);
        assert_eq!(map.offset_us(), Some(i64::MAX));
        assert_eq!(map.stamp(Stream::Gaze, i64::MAX, i64::MAX), i64::MAX);
        assert_eq!(map.stamp(Stream::Gaze, i64::MAX, i64::MAX), i64::MAX);
        map.observe(Stream::Image, i64::MAX, i64::MIN);
        assert_eq!(map.offset_us(), Some(i64::MIN));
        assert_eq!(map.stamp(Stream::Image, i64::MIN, i64::MIN), i64::MIN);
    }

    /// The gaze frames and images of session1's first 10 s, as
    /// `(stream, device, rx)`.
    fn session1_arrivals() -> Vec<(Stream, i64, i64)> {
        include_str!("../fixtures/session1-arrivals.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                let mut fields = line.split(' ');
                let stream = match fields.next() {
                    Some("g") => Stream::Gaze,
                    Some("i") => Stream::Image,
                    other => panic!("unknown stream {other:?}"),
                };
                let mut time =
                    || -> i64 { fields.next().and_then(|f| f.parse().ok()).expect("a time") };
                (stream, time(), time())
            })
            .collect()
    }

    /// For each arrival, observed by a map fed with the streams in `feed`,
    /// how far its device time plus the estimate lands after its read time.
    fn lateness(feed: &[Stream]) -> Vec<(Stream, i64)> {
        let mut map = TimeMap::default();
        session1_arrivals()
            .into_iter()
            .filter_map(|(stream, device, rx)| {
                if feed.contains(&stream) {
                    map.observe(stream, device, rx);
                }
                Some((stream, device + map.offset_us()? - rx))
            })
            .collect()
    }

    #[test]
    fn on_session1_no_gaze_frame_or_image_is_mapped_after_it_was_read() {
        let late = lateness(&[Stream::Gaze, Stream::Image]);
        assert!(late.iter().all(|&(_, late)| late <= 0));

        // Fed with gaze alone the map puts every image after it arrived
        // (by 6.2 to 10.6 ms here): that is what feeding it images fixes.
        let images: Vec<i64> = lateness(&[Stream::Gaze])
            .into_iter()
            .filter(|&(stream, _)| stream == Stream::Image)
            .map(|(_, late)| late)
            .collect();
        assert_eq!(images.len(), 251);
        assert!(images.iter().all(|&late| late > 0));
    }

    #[test]
    fn on_session1_gaze_is_spaced_as_the_device_spaced_it_once_images_come() {
        let mut map = TimeMap::default();
        let mut gaze = Vec::new();
        for (stream, device, rx) in session1_arrivals() {
            map.observe(stream, device, rx);
            let stamp = map.stamp(stream, device, rx);
            if stream == Stream::Gaze {
                gaze.push((device, rx, stamp));
            }
        }
        // The image stream starts 2.43 s in and lowers the estimate by
        // 10.6 ms over its first two frames; from 3 s on each gap between two
        // gaze host times is the device's gap to within 0.5 ms (0.03 ms here,
        // where a map fed with gaze alone steps by 2.8 ms at 3.2 s).
        for pair in gaze.windows(2) {
            let [(d0, _, s0), (d1, rx1, s1)] = pair else {
                unreachable!("windows of two")
            };
            if *rx1 >= 3 * S {
                let error = ((s1 - s0) - (d1 - d0)).abs();
                assert!(error <= MS / 2, "gaze read at {rx1} off by {error} us");
            }
        }
    }
}
