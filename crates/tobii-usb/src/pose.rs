//! The pose worker: head poses from the 0x50e images, on a thread of their
//! own so that a slow inference never blocks the USB reader (an unread IN
//! buffer stalls the firmware).
//!
//! The reader drops each image into a single-slot [`Mailbox`]; the worker
//! always takes the newest one, skipping (and counting) whatever it missed.
//! While a head pose is wanted, it runs each image it takes through a
//! [`HeadStep`] and sends one [`PoseSample`] of it, the Stream Engine's head
//! pose, valid or not. The engine's head flag is read once at each image,
//! with how many times it was set: the first image after a head pose was
//! wanted anew starts the pose over, even if the flag was cleared and set
//! again since the image before, as a new subscriber is not to inherit an
//! earlier one's filters.
//!
//! A tracker failure on an image makes the invalid pose of an image without
//! a face, like any other, and a warning at most every
//! [`FAILURE_WARNING_EVERY`]. `TOBII_IMAGE83_DEBUG=1` has the worker report
//! its statistics every [`STATS_EVERY`] ([`PoseStats`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tobii_pose::head::{FrameContext, HeadParams, HeadPose, HeadStep, StepOutput};
use tracing::{debug, error, info, warn};

use crate::device::{is_env_flag_set, to_i64_us};
use crate::engine::{HeadFlag, ImageSample, PoseSample, Sample, Shared};

/// How often the worker warns of the tracker's failures, at most: a tracker
/// that fails on every image would otherwise log 33 lines a second.
const FAILURE_WARNING_EVERY: Duration = Duration::from_secs(10);

/// How often `TOBII_IMAGE83_DEBUG` reports the worker's statistics.
const STATS_EVERY: Duration = Duration::from_secs(5);

/// The single-slot hand-off of the newest image from the USB reader to the
/// pose worker: the reader never waits for the worker, and an image the
/// worker has not taken yet when the next one comes is dropped.
#[derive(Debug, Default)]
pub(crate) struct Mailbox {
    slot: Mutex<Slot>,
    /// Rung for each image put, and for a stop.
    wake: Condvar,
}

/// What the [`Mailbox`] holds. Each field is written whole and nothing done
/// under the lock can panic, so a poisoned lock leaves nothing half-updated
/// and its guard is reused rather than propagating the panic to the reader.
#[derive(Debug, Default)]
struct Slot {
    /// The newest image, until the worker takes it.
    image: Option<ImageSample>,
    /// The images dropped since the worker last took one.
    overwritten: u64,
}

/// An image the worker took from the [`Mailbox`].
#[derive(Debug)]
pub(crate) struct Taken {
    /// The image.
    pub(crate) image: ImageSample,
    /// How many images the reader put after the one the worker took last
    /// and dropped for a newer one before the worker could take them.
    pub(crate) overwritten: u64,
}

impl Mailbox {
    /// Hand `image` to the worker, dropping the one it has not taken yet,
    /// if any, and wake it.
    pub(crate) fn put(&self, image: ImageSample) {
        {
            let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
            if slot.image.replace(image).is_some() {
                slot.overwritten = slot.overwritten.saturating_add(1);
            }
        }
        self.wake.notify_one();
    }

    /// Wait for an image and take it; `None` once `stop` is set, even with
    /// an image waiting.
    pub(crate) fn take(&self, stop: &AtomicBool) -> Option<Taken> {
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        // Relaxed: a pure signal; the image is handed over under the lock.
        while !stop.load(Ordering::Relaxed) {
            if let Some(image) = slot.image.take() {
                return Some(Taken {
                    image,
                    overwritten: std::mem::take(&mut slot.overwritten),
                });
            }
            slot = self
                .wake
                .wait_timeout(slot, Duration::from_millis(200))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        None
    }

    /// Wake the worker, so that it sees a stop at once.
    pub(crate) fn wake(&self) {
        self.wake.notify_all();
    }

    /// The image waiting for the worker, left where it is.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> Option<ImageSample> {
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .image
            .clone()
    }
}

/// The step from an image to its pose, which the worker drives:
/// [`HeadStep`] live; the tests give it a stand-in that says what it was
/// asked.
trait PoseStep {
    /// The pose of `image` (see [`HeadStep::step`]).
    fn step(&mut self, image: &ImageSample) -> Stepped;

    /// Start the pose over (see [`HeadStep::reset`]).
    fn reset(&mut self);
}

/// What the worker takes of a [`StepOutput`].
#[derive(Debug)]
struct Stepped {
    /// The Stream Engine's head pose, valid or not.
    head: HeadPose,
    /// Why the tracker failed on the image, if it did: the pose is then
    /// that of an image without a face.
    error: Option<anyhow::Error>,
    /// How many times the tracker has run its face detector so far.
    detector_runs: u64,
}

impl From<StepOutput> for Stepped {
    fn from(out: StepOutput) -> Self {
        Self {
            head: out.head,
            error: out.error,
            detector_runs: out.model_runs.detector,
        }
    }
}

impl PoseStep for HeadStep {
    fn step(&mut self, image: &ImageSample) -> Stepped {
        let frame = &image.frame;
        let context = frame_context(image);
        HeadStep::step(self, &frame.pixels, frame.width, frame.height, &context).into()
    }

    fn reset(&mut self) {
        HeadStep::reset(self);
    }
}

/// What [`HeadStep::step`] is to know of `image` besides its pixels: its
/// host time, which the filters step by; the display frame in effect when it
/// was read, and its generation; and the open that read it. A new generation
/// or open restarts the filters.
#[must_use]
fn frame_context(image: &ImageSample) -> FrameContext<'_> {
    FrameContext {
        t_us: image.host_us,
        display: image.display_frame.as_deref(),
        display_generation: image.display_generation.get(),
        open: image.open.get(),
    }
}

/// The worker's state from one image to the next, but for the step itself.
#[derive(Debug, Default)]
struct PoseWorker {
    /// The head flag as the last image read it; `None` before the first.
    last: Option<HeadFlag>,
    /// The rate limit on the warnings of the tracker's failures.
    failures: FailureWarnings,
    /// The statistics, when `TOBII_IMAGE83_DEBUG` asks for them.
    stats: Option<PoseStats>,
}

impl PoseWorker {
    /// A worker before its first image, keeping `stats` if given.
    fn new(stats: Option<PoseStats>) -> Self {
        Self {
            stats,
            ..Self::default()
        }
    }

    /// The pose of the image `taken`, made by `step` as the head flag `head`
    /// asks: one [`PoseSample`] while a head pose is wanted, valid or not,
    /// and `None` otherwise. The first image after a head pose was wanted
    /// anew, since the image before, starts the step over
    /// ([`PoseStep::reset`]). The statistics' period starts over at the
    /// first image stepped, the worker's very first included.
    fn image(
        &mut self,
        step: &mut impl PoseStep,
        taken: &Taken,
        head: HeadFlag,
    ) -> Option<PoseSample> {
        // At the worker's first image the step is new: nothing to start
        // over.
        let last = self.last.replace(head);
        let anew = last.is_some_and(|last| head.is_raised_since(last));
        if !head.on {
            return None;
        }
        if anew {
            debug!("head pose wanted anew: the pose starts over");
            step.reset();
        }
        if anew || last.is_none() {
            // Not at the worker's start: a tracker started cold sends its
            // first image only after an init of ~12 s, which would make the
            // first period's rates look like a stalled worker.
            if let Some(stats) = &mut self.stats {
                stats.restart(Instant::now());
            }
        }
        let image = &taken.image;
        let begun = Instant::now();
        let stepped = step.step(image);
        let took = begun.elapsed();
        if let Some(e) = &stepped.error
            && let Some(unlogged) = self.failures.failed(begun)
        {
            warn!(
                error = format_args!("{e:#}"),
                unlogged, "image83 head pose step failed; the image's pose is invalid"
            );
        }
        if let Some(stats) = &mut self.stats {
            stats.add(
                stepped.head.valid,
                taken.overwritten,
                took,
                stepped.detector_runs,
            );
            if let Some(report) = stats.report(Instant::now()) {
                report.log();
            }
        }
        Some(PoseSample::new(
            to_i64_us(image.frame.device_ts_us),
            image.host_us,
            stepped.head,
        ))
    }
}

/// The rate limit on the worker's warnings of the tracker's failures: the
/// first failure is logged, then at most one every
/// [`FAILURE_WARNING_EVERY`], saying how many were not.
#[derive(Debug, Default)]
struct FailureWarnings {
    /// When the last warning was logged.
    last: Option<Instant>,
    /// The failures since then that were not logged.
    unlogged: u64,
}

impl FailureWarnings {
    /// Count a failure at `now`: `Some` of how many failures since the last
    /// warning were not logged when this one is to be, else `None`.
    #[must_use]
    fn failed(&mut self, now: Instant) -> Option<u64> {
        if self
            .last
            .is_some_and(|last| now.saturating_duration_since(last) < FAILURE_WARNING_EVERY)
        {
            self.unlogged = self.unlogged.saturating_add(1);
            return None;
        }
        self.last = Some(now);
        Some(std::mem::take(&mut self.unlogged))
    }
}

/// `TOBII_IMAGE83_DEBUG`'s statistics of the images the worker stepped
/// since its last report, or since it began stepping them again: at its
/// first image, or the head pose wanted anew.
#[derive(Debug)]
struct PoseStats {
    /// When the period began.
    since: Instant,
    /// The time each image's step took, µs, in the order stepped (the
    /// buffer is reused from one period to the next).
    step_us: Vec<u64>,
    /// The images whose pose was valid.
    valid: u64,
    /// The images dropped in the mailbox before the worker could take them.
    overwritten: u64,
    /// How many times the tracker had run its face detector when the period
    /// began, and after the last image.
    detector_runs: [u64; 2],
}

/// One period's [`PoseStats`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct StatsReport {
    /// How long the period was, s.
    seconds: f64,
    /// The images stepped, each of which made a pose.
    frames: u64,
    /// The images whose pose was valid.
    valid: u64,
    /// The images dropped in the mailbox before the worker could take them.
    overwritten: u64,
    /// The time a step took, ms: the mean, the 99th percentile (by nearest
    /// rank) and the longest.
    step_ms: [f64; 3],
    /// The face detector's runs.
    detector_runs: u64,
}

impl PoseStats {
    /// Statistics of a period beginning at `now`, of a tracker that has not
    /// run its face detector yet.
    fn new(now: Instant) -> Self {
        Self {
            since: now,
            step_us: Vec::new(),
            valid: 0,
            overwritten: 0,
            detector_runs: [0; 2],
        }
    }

    /// Count an image that took `took` to step, after `overwritten` images
    /// were dropped before it, whose pose was `valid`, the tracker having
    /// run its face detector `detector_runs` times since it was made.
    fn add(&mut self, valid: bool, overwritten: u64, took: Duration, detector_runs: u64) {
        self.step_us
            .push(u64::try_from(took.as_micros()).unwrap_or(u64::MAX));
        self.valid = self.valid.saturating_add(u64::from(valid));
        self.overwritten = self.overwritten.saturating_add(overwritten);
        self.detector_runs[1] = detector_runs;
    }

    /// Drop what was counted and begin a new period at `now`.
    fn restart(&mut self, now: Instant) {
        self.since = now;
        self.step_us.clear();
        self.valid = 0;
        self.overwritten = 0;
        self.detector_runs[0] = self.detector_runs[1];
    }

    /// The period's report once [`STATS_EVERY`] has passed since it began,
    /// at `now`, which begins the next; else `None`.
    #[must_use]
    fn report(&mut self, now: Instant) -> Option<StatsReport> {
        let seconds = now.saturating_duration_since(self.since);
        if seconds < STATS_EVERY {
            return None;
        }
        self.step_us.sort_unstable();
        let frames = u64::try_from(self.step_us.len()).unwrap_or(u64::MAX);
        let ms = |us: u64| us as f64 / 1000.0;
        let total: u64 = self.step_us.iter().sum();
        let report = StatsReport {
            seconds: seconds.as_secs_f64(),
            frames,
            valid: self.valid,
            overwritten: self.overwritten,
            step_ms: [
                ms(total) / frames.max(1) as f64,
                ms(nearest_rank(&self.step_us, 99)),
                ms(self.step_us.last().copied().unwrap_or(0)),
            ],
            detector_runs: self.detector_runs[1].saturating_sub(self.detector_runs[0]),
        };
        self.restart(now);
        Some(report)
    }
}

impl StatsReport {
    /// Log the report, rates per second.
    fn log(&self) {
        let per_s = |n: u64| n as f64 / self.seconds;
        let [mean_ms, p99_ms, max_ms] = self.step_ms;
        info!(
            frames_per_s = format_args!("{:.1}", per_s(self.frames)),
            valid_per_s = format_args!("{:.1}", per_s(self.valid)),
            overwritten = self.overwritten,
            mean_ms = format_args!("{mean_ms:.1}"),
            p99_ms = format_args!("{p99_ms:.1}"),
            max_ms = format_args!("{max_ms:.1}"),
            detector_runs_per_s = format_args!("{:.1}", per_s(self.detector_runs)),
            "image83 pose worker"
        );
    }
}

/// The `percent`th percentile of `sorted` (ascending) by nearest rank: the
/// least of its values that at least `percent` % of them do not exceed; 0
/// for no values.
#[must_use]
fn nearest_rank(sorted: &[u64], percent: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percent).div_ceil(100);
    sorted.get(rank.saturating_sub(1)).copied().unwrap_or(0)
}

/// The pose worker's thread (see `run_gaze_engine`): a [`PoseWorker`] over
/// a [`HeadStep`] of [`HeadParams::FITTED`], driven by [`drive`]. Without
/// its models it logs why and ends, and the engine runs on without head
/// pose.
pub(crate) fn run_worker(mailbox: &Mailbox, shared: &Shared, tx: &Sender<Sample>) {
    let mut step = match HeadStep::new(HeadParams::FITTED) {
        Ok(step) => step,
        Err(e) => {
            error!(
                error = format_args!("{e:#}"),
                "image83 pose worker disabled"
            );
            return;
        }
    };
    let stats = is_env_flag_set("TOBII_IMAGE83_DEBUG").then(|| PoseStats::new(Instant::now()));
    drive(mailbox, shared, tx, &mut step, PoseWorker::new(stats));
}

/// The pose worker's loop: each image the reader puts in `mailbox` goes
/// through `worker` and `step` as the head flag of `shared` wants it when
/// the image is taken, and its pose, if any, to `tx`, until `shared` says
/// stop.
fn drive(
    mailbox: &Mailbox,
    shared: &Shared,
    tx: &Sender<Sample>,
    step: &mut impl PoseStep,
    mut worker: PoseWorker,
) {
    while let Some(taken) = mailbox.take(&shared.stop) {
        if let Some(pose) = worker.image(step, &taken, shared.head()) {
            // An engine dropped meanwhile reads no more samples; the stop
            // ends the loop.
            let _ = tx.send(Sample::Pose(pose));
        }
    }
}

#[cfg(test)]
// reason: unwrap on fixtures is the idiomatic test failure (test-* rules).
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use std::collections::VecDeque;
    use std::sync::{Arc, mpsc};
    use std::thread;
    use tobii_ipc::geometry::{DisplayArea, DisplayFrame};
    use tobii_proto::image83::ImageFrame;

    use crate::engine::{DisplayGeneration, OpenNumber, Wanted};

    /// How much later an image's device time is than its host time, here.
    const DEVICE_AHEAD_US: i64 = 7_000_000;

    /// A 2x1 image of host time `host_us`, its device time
    /// [`DEVICE_AHEAD_US`] later.
    fn image(host_us: i64) -> ImageSample {
        let frame = ImageFrame {
            device_ts_us: u64::try_from(host_us + DEVICE_AHEAD_US).unwrap(),
            width: 2,
            height: 1,
            pixels: vec![0, 0],
        };
        ImageSample::new(Arc::new(frame), host_us)
    }

    /// `image(host_us)` as the worker takes it, none dropped before it.
    fn taken(host_us: i64) -> Taken {
        Taken {
            image: image(host_us),
            overwritten: 0,
        }
    }

    /// The head flag once it is set to `on`, from none.
    fn flag(on: bool) -> HeadFlag {
        let mut flag = HeadFlag::default();
        flag.set(on);
        flag
    }

    /// What a [`Script`] was asked.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Call {
        /// To step the image of host time `.0`.
        Step(i64),
        Reset,
    }

    /// A [`PoseStep`] that notes what it is asked and makes the poses it is
    /// given, in turn; the invalid pose at zero once they run out.
    #[derive(Default)]
    struct Script {
        calls: Vec<Call>,
        poses: VecDeque<Stepped>,
    }

    impl PoseStep for Script {
        fn step(&mut self, image: &ImageSample) -> Stepped {
            self.calls.push(Call::Step(image.host_us));
            self.poses.pop_front().unwrap_or_else(|| stepped(None))
        }

        fn reset(&mut self) {
            self.calls.push(Call::Reset);
        }
    }

    /// A step's pose: `head`, or the invalid pose at zero.
    fn stepped(head: Option<HeadPose>) -> Stepped {
        Stepped {
            head: head.unwrap_or_default(),
            error: None,
            detector_runs: 0,
        }
    }

    fn valid_pose() -> HeadPose {
        HeadPose {
            valid: true,
            position_mm: [1.0, 2.0, 3.0],
            rotation_rad: [0.1, 0.2, 0.3],
        }
    }

    /// The step's poses go out, one for each image wanted, valid or not,
    /// with the image's device and host times; an image not wanted is not
    /// stepped and makes none.
    #[test]
    fn every_image_wanted_makes_one_pose_and_no_other_image_does() {
        let held = HeadPose {
            valid: false,
            ..valid_pose()
        };
        let mut step = Script::default();
        step.poses
            .extend([stepped(Some(valid_pose())), stepped(Some(held))]);
        let mut worker = PoseWorker::default();
        let wanted = flag(true);
        let mut dropped = wanted;
        dropped.set(false);
        let mut again = dropped;
        again.set(true);

        let poses = [
            worker.image(&mut step, &taken(1_000), wanted),
            worker.image(&mut step, &taken(2_000), wanted),
            worker.image(&mut step, &taken(3_000), dropped),
            worker.image(&mut step, &taken(4_000), again),
        ];

        let pose = |host_us, head| Some(PoseSample::new(host_us + DEVICE_AHEAD_US, host_us, head));
        assert_eq!(
            poses,
            [
                pose(1_000, valid_pose()),
                pose(2_000, held),
                None,
                pose(4_000, HeadPose::default()),
            ]
        );
        assert_eq!(
            step.calls,
            [
                Call::Step(1_000),
                Call::Step(2_000),
                Call::Reset,
                Call::Step(4_000),
            ]
        );
    }

    /// The step starts over at the first image that wants a head pose after
    /// one that did not, and only then: not at the first image, whose step
    /// is new, nor at an image that follows one that wanted a pose too.
    #[test]
    fn the_pose_starts_over_when_wanted_again_after_an_image_that_was_not() {
        let run = |wanted: &[bool]| {
            let shared = Shared::default();
            let mut step = Script::default();
            let mut worker = PoseWorker::default();
            for (t, &head) in (1..).zip(wanted) {
                shared.set_wanted(Wanted {
                    head,
                    ..Wanted::default()
                });
                let _ = worker.image(&mut step, &taken(t), shared.head());
            }
            step.calls
        };

        assert_eq!(
            run(&[true, true, false, false, true, true, false, true]),
            [
                Call::Step(1),
                Call::Step(2),
                Call::Reset,
                Call::Step(5),
                Call::Step(6),
                Call::Reset,
                Call::Step(8)
            ]
        );
        assert_eq!(run(&[false, true]), [Call::Reset, Call::Step(2)]);
    }

    /// A head pose wanted anew between two images, after it was not wanted
    /// for a while since the first (no image came meanwhile, as in a pause,
    /// or the subscriptions changed faster than the images come), is the
    /// subscribe it is: the pose starts over. Wanted throughout, wanted
    /// again while it was, or with the images asked for or dropped beside
    /// it, the pose goes on; the first image starts nothing over.
    #[test]
    fn a_pose_wanted_anew_between_two_images_starts_over_all_the_same() {
        const NONE: Wanted = Wanted {
            head: false,
            image: false,
        };
        const HEAD: Wanted = Wanted { head: true, ..NONE };
        const WITH_IMAGES: Wanted = Wanted {
            image: true,
            ..HEAD
        };
        let shared = Shared::default();
        let mut step = Script::default();
        let mut worker = PoseWorker::default();
        let mut image = |t, changes: &[Wanted]| {
            for &wanted in changes {
                shared.set_wanted(wanted);
            }
            let _ = worker.image(&mut step, &taken(t), shared.head());
        };

        image(1, &[NONE, HEAD, NONE, HEAD]);
        image(2, &[NONE, HEAD]);
        image(3, &[]);
        image(4, &[HEAD, HEAD]);
        image(5, &[WITH_IMAGES, HEAD]);
        image(6, &[NONE, WITH_IMAGES]);

        assert_eq!(
            step.calls,
            [
                Call::Step(1),
                Call::Reset,
                Call::Step(2),
                Call::Step(3),
                Call::Step(4),
                Call::Step(5),
                Call::Reset,
                Call::Step(6),
            ]
        );
    }

    /// A flag counts each time it is set when it was not, and is raised
    /// since an earlier reading when it is set and was set anew since.
    #[test]
    fn a_head_flag_counts_its_raises() {
        let mut flag = HeadFlag::default();
        let mut readings = vec![flag];
        for on in [true, true, false, false, true, false, true] {
            flag.set(on);
            readings.push(flag);
        }
        let raised: Vec<u64> = readings.iter().map(|flag| flag.raised).collect();
        assert_eq!(raised, [0, 1, 1, 1, 1, 2, 2, 3]);
        let (none, on, again, off) = (readings[0], readings[1], readings[2], readings[3]);
        assert!(on.is_raised_since(none));
        assert!(!again.is_raised_since(on), "set while set");
        assert!(!off.is_raised_since(none), "cleared");
        let last = readings[readings.len() - 1];
        assert!(last.is_raised_since(on), "cleared and set again twice");
        assert!(!last.is_raised_since(last));
    }

    /// An image the tracker failed on makes the pose the step gives for it:
    /// invalid, the last valid values held. The failure is counted for the
    /// warnings.
    #[test]
    fn a_tracker_failure_sends_the_invalid_pose_the_step_gave() {
        let held = HeadPose {
            valid: false,
            ..valid_pose()
        };
        let mut step = Script::default();
        step.poses.push_back(Stepped {
            error: Some(anyhow!("the landmark model failed")),
            ..stepped(Some(held))
        });
        let mut worker = PoseWorker::default();

        let pose = worker.image(&mut step, &taken(5), flag(true)).unwrap();

        assert_eq!(pose.head, held);
        assert!(worker.failures.last.is_some(), "a warning was logged");
    }

    /// The first failure is logged; the next ones only once ten seconds
    /// have passed since the last that was, each with how many were not.
    #[test]
    fn failures_are_logged_at_most_every_ten_seconds_with_a_count_of_the_rest() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let mut warnings = FailureWarnings::default();

        let logged: Vec<Option<u64>> = [0, 1, 9_999, 10_000, 10_001, 30_000, 30_000]
            .into_iter()
            .map(|ms| warnings.failed(at(ms)))
            .collect();

        assert_eq!(logged, [Some(0), None, None, Some(2), None, Some(1), None]);
    }

    /// A report comes once five seconds have passed: the images stepped,
    /// their valid poses, the images dropped before them, the time a step
    /// took (mean, 99th percentile, longest) and the face detector's runs.
    /// The next period starts afresh.
    #[test]
    fn the_statistics_report_each_period_and_start_the_next_afresh() {
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let mut stats = PoseStats::new(t0);
        // 1 to 100 ms, the longest first, every other pose valid; 3 images
        // dropped before one; the detector run 7 times.
        for i in (1..=100).rev() {
            stats.add(
                i % 2 == 0,
                if i == 50 { 3 } else { 0 },
                Duration::from_millis(i),
                (100 - i) / 14,
            );
        }

        assert_eq!(stats.report(at(4_999)), None);
        assert_eq!(
            stats.report(at(5_000)),
            Some(StatsReport {
                seconds: 5.0,
                frames: 100,
                valid: 50,
                overwritten: 3,
                step_ms: [50.5, 99.0, 100.0],
                detector_runs: 7,
            })
        );
        stats.add(false, 0, Duration::from_micros(2_500), 9);
        assert_eq!(stats.report(at(9_999)), None);
        assert_eq!(
            stats.report(at(10_000)),
            Some(StatsReport {
                seconds: 5.0,
                frames: 1,
                valid: 0,
                overwritten: 0,
                step_ms: [2.5; 3],
                detector_runs: 2,
            })
        );
    }

    /// The statistics count each image stepped, its pose valid or not and
    /// the images the mailbox dropped before it, and leave out what came
    /// before the head pose was last wanted again, so that the time it was
    /// not does not dilute the rates.
    #[test]
    fn the_statistics_count_the_images_stepped_since_the_pose_was_wanted_again() {
        let mut step = Script::default();
        step.poses.extend([
            stepped(Some(valid_pose())),
            stepped(None),
            stepped(Some(valid_pose())),
            stepped(None),
        ]);
        let mut worker = PoseWorker::new(Some(PoseStats::new(Instant::now())));
        let count = |worker: &PoseWorker| {
            let stats = worker.stats.as_ref().unwrap();
            (stats.step_us.len(), stats.valid, stats.overwritten)
        };
        let mut head = HeadFlag::default();
        let mut image = |t, on, overwritten| {
            head.set(on);
            let taken = Taken {
                overwritten,
                ..taken(t)
            };
            let _ = worker.image(&mut step, &taken, head);
            count(&worker)
        };

        assert_eq!(image(1, true, 1), (1, 1, 1));
        assert_eq!(image(2, true, 0), (2, 1, 1));
        assert_eq!(image(3, false, 5), (2, 1, 1), "not stepped");
        assert_eq!(image(4, true, 2), (1, 1, 2), "counted afresh");
        assert_eq!(image(5, true, 0), (2, 1, 2));
    }

    /// The statistics' period begins at the first image the worker steps,
    /// not when the worker started: a head-pose client that starts the
    /// engine gets its first image some 12 s later, after the tracker's
    /// init, and the first report would cover that one image.
    #[test]
    fn the_statistics_begin_at_the_first_image_stepped() {
        let started = Instant::now().checked_sub(Duration::from_secs(12)).unwrap();
        let mut worker = PoseWorker::new(Some(PoseStats::new(started)));
        let mut step = Script::default();
        let period = |worker: &PoseWorker| {
            let stats = worker.stats.as_ref().unwrap();
            (stats.since, stats.step_us.len())
        };

        let _ = worker.image(&mut step, &taken(1), flag(true));
        let (since, images) = period(&worker);
        assert_eq!(images, 1, "no report of the 12 s before the first image");
        assert!(since >= started + Duration::from_secs(12), "{since:?}");
        let _ = worker.image(&mut step, &taken(2), flag(true));
        assert_eq!(period(&worker), (since, 2), "the period goes on");
        assert_eq!(
            step.calls,
            [Call::Step(1), Call::Step(2)],
            "and the step, new, is not reset"
        );
    }

    /// What the engine is told is wanted is what the pose worker and the
    /// USB thread read: each flag from its own field, whatever the other
    /// says.
    #[test]
    fn the_flags_read_back_what_the_engine_was_told() {
        let shared = Shared::default();
        assert_eq!(shared.head(), HeadFlag::default());
        for bits in 0..4_u8 {
            let wanted = Wanted {
                head: bits & 1 != 0,
                image: bits & 2 != 0,
            };
            shared.set_wanted(wanted);
            assert_eq!(
                (
                    shared.head().on,
                    shared.image_wanted.load(Ordering::Relaxed)
                ),
                (wanted.head, wanted.image),
                "{wanted:?}"
            );
        }
    }

    #[test]
    fn the_99th_percentile_is_taken_by_nearest_rank() {
        let upto = |n: u64| (1..=n).collect::<Vec<u64>>();
        assert_eq!(nearest_rank(&[], 99), 0);
        assert_eq!(nearest_rank(&[7], 99), 7);
        assert_eq!(nearest_rank(&upto(100), 99), 99);
        assert_eq!(nearest_rank(&upto(100), 100), 100);
        assert_eq!(nearest_rank(&upto(100), 50), 50);
        // Five seconds of images: 163.35 rounds up to the 164th.
        assert_eq!(nearest_rank(&upto(165), 99), 164);
    }

    /// The worker takes the newest image, told how many it missed; a stop
    /// ends the wait at once, an image waiting or not.
    #[test]
    fn the_mailbox_hands_over_the_newest_image_and_counts_those_dropped() {
        let mailbox = Mailbox::default();
        let stop = AtomicBool::new(false);
        let take = |mailbox: &Mailbox| {
            mailbox
                .take(&stop)
                .map(|taken| (taken.image.host_us, taken.overwritten))
        };

        for t in 1..=3 {
            mailbox.put(image(t));
        }
        assert_eq!(take(&mailbox), Some((3, 2)));
        mailbox.put(image(4));
        assert_eq!(take(&mailbox), Some((4, 0)));

        mailbox.put(image(5));
        stop.store(true, Ordering::Relaxed);
        assert_eq!(take(&mailbox), None);
        assert_eq!(mailbox.waiting().map(|i| i.host_us), Some(5));
    }

    /// A worker waiting on an empty mailbox gets the next image put, and
    /// sees a stop it is woken for.
    #[test]
    fn a_worker_waiting_for_an_image_gets_the_next_one_or_the_stop() {
        let mailbox = Arc::new(Mailbox::default());
        let stop = Arc::new(AtomicBool::new(false));
        let waiting = || {
            let (mailbox, stop) = (Arc::clone(&mailbox), Arc::clone(&stop));
            thread::spawn(move || mailbox.take(&stop).map(|taken| taken.image.host_us))
        };

        let worker = waiting();
        mailbox.put(image(7));
        assert_eq!(worker.join().unwrap(), Some(7));

        let worker = waiting();
        stop.store(true, Ordering::Relaxed);
        mailbox.wake();
        assert_eq!(worker.join().unwrap(), None);
    }

    /// Stops the worker that [`drive`] runs when dropped, so that a failed
    /// assertion ends the test rather than leaving the worker waiting for
    /// an image.
    struct StopOnDrop<'a>(&'a Mailbox, &'a Shared);

    impl Drop for StopOnDrop<'_> {
        fn drop(&mut self) {
            self.1.stop.store(true, Ordering::Relaxed);
            self.0.wake();
        }
    }

    /// The worker's loop on its thread: each image put goes through the
    /// step as the engine's head flag wants it when the worker takes it,
    /// and its pose is sent; the pose starts over when wanted again, even
    /// if only between two images; an image not wanted is not stepped and
    /// sends none; a stop ends the loop.
    #[test]
    fn the_worker_sends_the_pose_of_each_image_as_the_flag_wants_it() {
        let mailbox = Mailbox::default();
        let shared = Shared::default();
        let (tx, rx) = mpsc::channel();
        let wait = Duration::from_secs(10);
        let wanted = |head| Wanted { head, image: false };
        let sent = |host_us| {
            let pose = PoseSample::new(host_us + DEVICE_AHEAD_US, host_us, HeadPose::default());
            Sample::Pose(pose)
        };

        let steps = thread::scope(|s| {
            let worker = s.spawn(|| {
                let mut step = Script::default();
                drive(&mailbox, &shared, &tx, &mut step, PoseWorker::default());
                step.calls
            });
            let stop = StopOnDrop(&mailbox, &shared);
            shared.set_wanted(wanted(true));
            mailbox.put(image(1));
            assert_eq!(rx.recv_timeout(wait).unwrap(), sent(1));
            shared.set_wanted(wanted(false));
            shared.set_wanted(wanted(true));
            mailbox.put(image(2));
            assert_eq!(rx.recv_timeout(wait).unwrap(), sent(2));
            // Not wanted: there is no pose to wait for, so the image's
            // leaving the mailbox says the worker has it. The flag stays
            // as it is, whenever the worker reads it.
            shared.set_wanted(wanted(false));
            mailbox.put(image(3));
            let deadline = Instant::now() + wait;
            while mailbox.waiting().is_some() {
                assert!(Instant::now() < deadline, "the worker takes the image");
                thread::sleep(Duration::from_millis(1));
            }
            drop(stop);
            worker.join().unwrap()
        });

        assert_eq!(steps, [Call::Step(1), Call::Reset, Call::Step(2)]);
        assert!(rx.try_recv().is_err(), "no pose of the image not wanted");
    }

    /// A display frame, of a 600 x 340 mm screen above the tracker.
    fn display_frame() -> DisplayFrame {
        DisplayFrame::new(&DisplayArea {
            top_left_mm: [-300.0, 360.0, 120.0],
            top_right_mm: [300.0, 360.0, 120.0],
            bottom_left_mm: [-300.0, 40.0, 4.0],
        })
        .unwrap()
    }

    /// The step is told the image's host time, display frame, display
    /// generation and open.
    #[test]
    fn a_step_is_told_the_images_host_time_display_frame_generation_and_open() {
        let frame = display_frame();
        let image = ImageSample {
            display_frame: Some(Arc::new(frame)),
            display_generation: DisplayGeneration(3),
            open: OpenNumber(2),
            ..image(1_234)
        };

        assert_eq!(
            frame_context(&image),
            FrameContext {
                t_us: 1_234,
                display: Some(&frame),
                display_generation: 3,
                open: 2,
            }
        );
        assert_eq!(
            frame_context(&ImageSample::new(Arc::clone(&image.frame), 5)),
            FrameContext {
                t_us: 5,
                display: None,
                display_generation: 0,
                open: 0,
            }
        );
    }

    /// With the real models: an image without a face makes one pose, the
    /// invalid one at zero; one the tracker cannot take makes an invalid
    /// pose too, and says why.
    #[test]
    fn the_head_step_makes_one_invalid_pose_of_an_image_without_a_face() {
        let mut step = HeadStep::new(HeadParams::FITTED).unwrap();
        let n = 280;
        let black = ImageSample {
            display_frame: Some(Arc::new(display_frame())),
            display_generation: DisplayGeneration(1),
            open: OpenNumber(1),
            ..ImageSample::new(
                Arc::new(ImageFrame {
                    device_ts_us: 9,
                    width: n,
                    height: n,
                    pixels: vec![0; n * n],
                }),
                1_000,
            )
        };
        let mut worker = PoseWorker::default();
        let black = Taken {
            image: black,
            overwritten: 0,
        };

        let pose = worker.image(&mut step, &black, flag(true));

        assert_eq!(pose, Some(PoseSample::new(9, 1_000, HeadPose::default())));
        assert_eq!(step.model_runs().detector, 1);
        let small = PoseStep::step(&mut step, &image(31_000));
        assert!(small.error.is_some(), "a 2x1 image is refused");
        assert_eq!(small.head, HeadPose::default());
        assert_eq!(small.detector_runs, 1);
    }

    /// The 0x50e fixture's frame (`TOBII_IMAGE83_FIXTURE`, a captured
    /// 78609-byte message; the user's face, so it is not committed), if set.
    fn image83_fixture() -> Option<ImageFrame> {
        let path = std::env::var("TOBII_IMAGE83_FIXTURE").ok()?;
        let msg = std::fs::read(path).unwrap();
        Some(tobii_proto::image83::decode_image_payload(&msg).unwrap())
    }

    /// With the real models and a face (the 0x50e fixture, if set): the
    /// worker's step is the head step's. The image's display frame and time
    /// go in and a valid pose comes out; an image without a face then makes
    /// an invalid pose that keeps the last valid values, and after a reset
    /// the invalid pose at zero, until the face is found again.
    #[test]
    fn the_worker_steps_and_resets_the_head_step() {
        let Some(face) = image83_fixture() else {
            return;
        };
        let face = Arc::new(face);
        let black = Arc::new(ImageFrame {
            pixels: vec![0; face.pixels.len()],
            ..(*face).clone()
        });
        let display = Arc::new(display_frame());
        let mut t_us = 0;
        let mut next = |frame: &Arc<ImageFrame>| {
            t_us += 30_000;
            ImageSample {
                display_frame: Some(Arc::clone(&display)),
                display_generation: DisplayGeneration(1),
                open: OpenNumber(1),
                ..ImageSample::new(Arc::clone(frame), t_us)
            }
        };
        let mut step = HeadStep::new(HeadParams::FITTED).unwrap();

        let valid = PoseStep::step(&mut step, &next(&face)).head;
        assert!(valid.valid, "the fixture has a face");
        let held = HeadPose {
            valid: false,
            ..valid
        };
        assert_eq!(PoseStep::step(&mut step, &next(&black)).head, held);
        PoseStep::reset(&mut step);
        let reset = PoseStep::step(&mut step, &next(&black)).head;
        assert_eq!(reset, HeadPose::default(), "zeros after a reset");
        let again = PoseStep::step(&mut step, &next(&face)).head;
        assert!(again.valid, "the face is found again");
    }
}
