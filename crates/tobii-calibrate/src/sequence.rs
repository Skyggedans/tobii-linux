//! The calibration procedure, run on a worker thread while the window shows
//! what it is doing.
//!
//! It mirrors what the Windows engine was captured doing: start, then per
//! round the 7-point pattern in batches of 1, 3 and 3 points with a compute
//! after each batch, then read the result back and stop. The tracker keeps
//! the last 14 points (two rounds), so two rounds replace every point of the
//! previous calibration.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use tobii_calib::STIMULUS_POINTS;
use tobii_ipc::request::{encode_point_2d, kind, status};

use crate::ipc::{Connection, Reply};

/// How the target moves and waits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Timing {
    /// Travel from one point to the next.
    pub(crate) travel: Duration,
    /// Time the user has to settle their gaze before collection starts.
    pub(crate) dwell: Duration,
}

/// What the window shows.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum UiEvent {
    /// A line of status text.
    Status(String),
    /// A target: where, what it is doing, and progress.
    Target {
        /// Normalised display position.
        at: [f32; 2],
        /// Where it travels from (the previous target).
        from: [f32; 2],
        /// What is happening at the target.
        phase: Phase,
        /// Index of this point, from 1.
        step: usize,
        /// Points in the whole session.
        total: usize,
    },
    /// The tracker is computing.
    Computing,
    /// The session finished.
    Finished(Summary),
    /// The latest gaze sample: normalised position and validity.
    Gaze([f32; 2], bool),
    /// The session failed.
    Failed(String),
}

/// What a target is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Moving into place.
    Travel,
    /// Waiting for the user's gaze to settle.
    Dwell,
    /// The tracker is collecting (the request blocks).
    Collecting,
}

/// The result of a session.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Summary {
    /// The new calibration id.
    pub(crate) id: u32,
    /// Points stored in the calibration.
    pub(crate) points: usize,
    /// Mean distance between each target and the eyes' mean measured point,
    /// as a fraction of the display.
    pub(crate) mean_error: Option<f32>,
    /// The calibration blob.
    pub(crate) blob: Vec<u8>,
}

/// Where requests go: the daemon, or a stand-in for `--dry-run`.
pub(crate) trait Backend: Send {
    /// Wait until the tracker streams.
    fn wait_ready(&mut self) -> Result<()>;
    /// Send a request and wait for the reply.
    fn request(&mut self, kind: u8, payload: &[u8], timeout: Duration) -> Result<Reply>;
}

impl Backend for Connection {
    fn wait_ready(&mut self) -> Result<()> {
        self.wait_for_gaze(Duration::from_secs(30))
    }

    fn request(&mut self, kind: u8, payload: &[u8], timeout: Duration) -> Result<Reply> {
        Connection::request(self, kind, payload, timeout)
    }
}

/// Pretends to be the tracker, with its timing, so the window can be tried
/// without one.
pub(crate) struct DryRun;

impl Backend for DryRun {
    fn wait_ready(&mut self) -> Result<()> {
        thread::sleep(Duration::from_millis(500));
        Ok(())
    }

    fn request(&mut self, kind: u8, _payload: &[u8], _timeout: Duration) -> Result<Reply> {
        let delay = match kind {
            kind::CALIBRATION_COLLECT_2D => 800,
            kind::CALIBRATION_COMPUTE => 2000,
            _ => 100,
        };
        thread::sleep(Duration::from_millis(delay));
        let payload = if kind == kind::CALIBRATION_COMPUTE {
            0x0d57_ce11u32.to_le_bytes().to_vec()
        } else {
            Vec::new()
        };
        Ok((status::OK, payload))
    }
}

/// The batches of each round: indices into [`STIMULUS_POINTS`].
const BATCHES: [&[usize]; 3] = [&[0], &[1, 2, 3], &[4, 5, 6]];

/// Every batch of the session, as points.
pub(crate) fn plan(rounds: usize) -> Vec<Vec<[f32; 2]>> {
    (0..rounds)
        .flat_map(|_| {
            BATCHES
                .iter()
                .map(|b| b.iter().map(|&i| STIMULUS_POINTS[i]).collect())
        })
        .collect()
}

fn describe(status_code: u8) -> String {
    match status_code {
        status::CALIBRATION_BUSY => "another client is calibrating the tracker".into(),
        status::CALIBRATION_ALREADY_STARTED => "a calibration is already running".into(),
        status::CALIBRATION_NOT_STARTED => "the calibration session ended".into(),
        status::TIMED_OUT => "the tracker did not answer in time".into(),
        status::CONNECTION_FAILED => "the tracker is not available".into(),
        status::NOT_SUPPORTED => "not supported by this tracker".into(),
        other => format!("the tracker refused (status {other})"),
    }
}

fn expect_ok((code, payload): Reply, what: &str) -> Result<Vec<u8>> {
    if code == status::OK {
        Ok(payload)
    } else {
        bail!("{what}: {}", describe(code))
    }
}

/// Mean target error of a calibration, as a fraction of the display.
fn mean_error(blob: &[u8]) -> Option<f32> {
    let points = tobii_calib::blob::points(blob).ok()?;
    let errors: Vec<f32> = points
        .iter()
        .map(|p| {
            let mx = (p.a[0] + p.b[0]) / 2.0 - p.target[0];
            let my = (p.a[1] + p.b[1]) / 2.0 - p.target[1];
            (mx * mx + my * my).sqrt()
        })
        .collect();
    #[allow(clippy::cast_precision_loss)] // reason: a handful of points
    let n = errors.len() as f32;
    (!errors.is_empty()).then(|| errors.iter().sum::<f32>() / n)
}

/// Run a session, reporting progress through `emit`. `abort` (Esc) stops it
/// between steps; the daemon restores the previous calibration.
///
/// # Errors
/// Fails when the tracker refuses a step, stops answering, or `abort` is set.
pub(crate) fn run(
    backend: &mut dyn Backend,
    rounds: usize,
    timing: Timing,
    abort: &AtomicBool,
    emit: &dyn Fn(UiEvent),
) -> Result<Summary> {
    emit(UiEvent::Status("waiting for the tracker...".into()));
    backend.wait_ready()?;
    let started = backend.request(kind::CALIBRATION_START, &[2], Duration::from_secs(25))?;
    expect_ok(started, "could not start")?;

    let result = collect_and_compute(backend, rounds, timing, abort, emit);
    // Stopping also restores the previous calibration when nothing was computed.
    let stopped = backend.request(kind::CALIBRATION_STOP, &[], Duration::from_secs(20));
    let summary = result?;
    if let Err(e) = stopped.and_then(|r| expect_ok(r, "could not stop")) {
        tracing::warn!(error = %e, "stopping the calibration");
    }
    Ok(summary)
}

fn collect_and_compute(
    backend: &mut dyn Backend,
    rounds: usize,
    timing: Timing,
    abort: &AtomicBool,
    emit: &dyn Fn(UiEvent),
) -> Result<Summary> {
    let batches = plan(rounds);
    let total = batches.iter().map(Vec::len).sum();
    let mut step = 0;
    let mut previous = [0.5, 0.5];
    let mut id = 0;
    for batch in &batches {
        for &at in batch {
            step += 1;
            for (phase, pause) in [(Phase::Travel, timing.travel), (Phase::Dwell, timing.dwell)] {
                emit(UiEvent::Target {
                    at,
                    from: previous,
                    phase,
                    step,
                    total,
                });
                thread::sleep(pause);
                // Relaxed: a pure signal from the UI thread.
                if abort.load(Ordering::Relaxed) {
                    bail!("aborted");
                }
            }
            emit(UiEvent::Target {
                at,
                from: previous,
                phase: Phase::Collecting,
                step,
                total,
            });
            let reply = backend.request(
                kind::CALIBRATION_COLLECT_2D,
                &encode_point_2d(at[0], at[1]),
                Duration::from_secs(8),
            )?;
            expect_ok(reply, "could not collect a point")?;
            previous = at;
        }
        emit(UiEvent::Computing);
        let reply = backend.request(kind::CALIBRATION_COMPUTE, &[], Duration::from_secs(20))?;
        let payload = expect_ok(reply, "could not compute")?;
        id = tobii_ipc::request::decode_u32(&payload).unwrap_or(id);
    }
    let reply = backend.request(kind::CALIBRATION_RETRIEVE, &[], Duration::from_secs(8))?;
    let blob = expect_ok(reply, "could not read the calibration back")?;
    Ok(Summary {
        id,
        points: tobii_calib::blob::points(&blob).map_or(0, |p| p.len()),
        mean_error: mean_error(&blob),
        blob,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    #[allow(clippy::float_cmp)] // reason: the pattern's exact constants
    fn two_rounds_mirror_the_captured_batches() {
        let p = plan(2);
        let sizes: Vec<usize> = p.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![1, 3, 3, 1, 3, 3]);
        assert_eq!(p[0][0], [0.5, 0.5]);
        assert_eq!(p[1], vec![[0.5, 0.1], [0.9, 0.9], [0.1, 0.9]]);
    }

    /// Records requests; answers every one with `status`.
    struct Recorder(Mutex<Vec<u8>>, u8);

    impl Backend for Recorder {
        fn wait_ready(&mut self) -> Result<()> {
            Ok(())
        }

        fn request(&mut self, kind: u8, _payload: &[u8], _timeout: Duration) -> Result<Reply> {
            self.0.lock().expect("log").push(kind);
            Ok((self.1, Vec::new()))
        }
    }

    const QUICK: Timing = Timing {
        travel: Duration::ZERO,
        dwell: Duration::ZERO,
    };

    #[test]
    fn a_session_starts_collects_computes_reads_and_stops() {
        let mut backend = Recorder(Mutex::new(Vec::new()), status::OK);
        let events = Mutex::new(Vec::new());
        let summary = run(&mut backend, 1, QUICK, &AtomicBool::new(false), &|e| {
            events.lock().expect("events").push(e);
        })
        .expect("session");

        let log = backend.0.into_inner().expect("log");
        let c = kind::CALIBRATION_COLLECT_2D;
        let k = kind::CALIBRATION_COMPUTE;
        assert_eq!(
            log,
            vec![
                kind::CALIBRATION_START,
                c,
                k,
                c,
                c,
                c,
                k,
                c,
                c,
                c,
                k,
                kind::CALIBRATION_RETRIEVE,
                kind::CALIBRATION_STOP
            ]
        );
        assert_eq!(summary.points, 0, "an empty blob has no points");
        assert!(events.lock().expect("events").contains(&UiEvent::Computing));
    }

    #[test]
    fn a_busy_tracker_is_reported_and_nothing_else_is_sent() {
        let mut backend = Recorder(Mutex::new(Vec::new()), status::CALIBRATION_BUSY);
        let err = run(&mut backend, 1, QUICK, &AtomicBool::new(false), &|_| {}).expect_err("busy");
        assert!(err.to_string().contains("another client"), "{err}");
        assert_eq!(
            backend.0.into_inner().expect("log"),
            vec![kind::CALIBRATION_START]
        );
    }

    #[test]
    fn abort_stops_the_session() {
        let mut backend = Recorder(Mutex::new(Vec::new()), status::OK);
        let err =
            run(&mut backend, 2, QUICK, &AtomicBool::new(true), &|_| {}).expect_err("aborted");
        assert_eq!(err.to_string(), "aborted");
        assert_eq!(
            backend.0.into_inner().expect("log"),
            vec![kind::CALIBRATION_START, kind::CALIBRATION_STOP]
        );
    }
}
