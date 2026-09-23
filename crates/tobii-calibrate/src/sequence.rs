//! The calibration procedure, run on a worker thread while the window shows
//! what it is doing.
//!
//! It mirrors what the Windows engine was captured doing: start, then per
//! round the 7-point pattern in batches of 1, 3 and 3 points with a compute
//! after each batch, then read the result back and stop. The tracker keeps
//! the last 14 points (two rounds), so two rounds replace every point of the
//! previous calibration.
//!
//! Before that, the display setup (unless skipped): the tracker's mounting
//! and current display area go to the window, the window answers with the
//! monitor's size and the tracker's offset, and the display area computed
//! from them is written, since a calibration holds only for the display area
//! it was made on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tobii_calib::STIMULUS_POINTS;
use tobii_ipc::geometry::{DisplayArea, GeometryMounting, display_area_basic};
use tobii_ipc::request::{
    decode_display_area, decode_geometry_mounting, encode_display_area, encode_point_2d, kind,
    status,
};

use crate::ipc::{Connection, Reply};
use crate::setup::Choice;

/// How the target moves and waits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Timing {
    /// Travel from one point to the next.
    pub(crate) travel: Duration,
    /// Time the user has to settle their gaze before collection starts.
    pub(crate) dwell: Duration,
}

/// What the window needs to run the display setup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SetupRequest {
    /// Distance between the tracker's two guide marks, mm.
    pub(crate) guide_mm: f64,
    /// The display area the tracker has now.
    pub(crate) current: Option<DisplayArea>,
}

/// What the window shows.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum UiEvent {
    /// A line of status text.
    Status(String),
    /// Line the ticks up with the tracker's marks; the answer comes back on
    /// the setup channel.
    DisplaySetup(SetupRequest),
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
    /// Wait until the tracker streams, or `abort` is set.
    fn wait_ready(&mut self, abort: &AtomicBool) -> Result<()>;
    /// Send a request and wait for the reply.
    fn request(&mut self, kind: u8, payload: &[u8], timeout: Duration) -> Result<Reply>;
}

impl Backend for Connection {
    fn wait_ready(&mut self, abort: &AtomicBool) -> Result<()> {
        self.wait_for_gaze(Duration::from_secs(30), abort)
    }

    fn request(&mut self, kind: u8, payload: &[u8], timeout: Duration) -> Result<Reply> {
        Connection::request(self, kind, payload, timeout)
    }
}

/// Pretends to be the tracker, with its timing, so the window can be tried
/// without one.
pub(crate) struct DryRun;

impl Backend for DryRun {
    fn wait_ready(&mut self, _abort: &AtomicBool) -> Result<()> {
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
        let payload = match kind {
            kind::CALIBRATION_COMPUTE => 0x0d57_ce11u32.to_le_bytes().to_vec(),
            kind::GEOMETRY_MOUNTING => {
                tobii_ipc::request::encode_geometry_mounting(&DRY_RUN_MOUNTING)
            }
            kind::DISPLAY_AREA_GET => {
                encode_display_area(&display_area_basic(597.0, 336.0, 0.0, &DRY_RUN_MOUNTING))
            }
            _ => Vec::new(),
        };
        Ok((status::OK, payload))
    }
}

/// The Eye Tracker 5's mounting, for `--dry-run`.
const DRY_RUN_MOUNTING: GeometryMounting = GeometryMounting {
    guides: 2,
    width_mm: 184.0,
    angle_deg: 20.0,
    external_offset_mm: [0.0, -0.16, 13.85],
    internal_offset_mm: [0.0, 5.38, 9.86],
};

/// How long device requests made by the display setup may take (the
/// daemon may still be starting the tracker).
const SETUP_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// How often the setup, waiting for the window's answer, looks at `abort`.
const SETUP_POLL: Duration = Duration::from_millis(100);

/// Have the window line its ticks up with the tracker's guide marks, then
/// write the display area they give. Runs inside the calibration session:
/// no other client can calibrate meanwhile, and should the session end
/// without a calibration the daemon puts the previous area back. Returns
/// whether the area changed. Skipped (with a warning) for a tracker that
/// does not say where its guide marks are.
fn display_setup(
    backend: &mut dyn Backend,
    answers: &Receiver<Choice>,
    abort: &AtomicBool,
    emit: &dyn Fn(UiEvent),
) -> Result<bool> {
    let mounting = match backend.request(kind::GEOMETRY_MOUNTING, &[], SETUP_REQUEST_TIMEOUT)? {
        (status::OK, payload) => decode_geometry_mounting(&payload),
        (code, _) => {
            tracing::warn!(status = code, "the tracker does not say how it is mounted");
            None
        }
    };
    let Some(mounting) =
        mounting.filter(|m| m.guides == 2 && m.width_mm.is_finite() && m.width_mm > 0.0)
    else {
        tracing::warn!("no pair of guide marks to line up with: skipping the display setup");
        return Ok(false);
    };
    let current = match backend.request(kind::DISPLAY_AREA_GET, &[], SETUP_REQUEST_TIMEOUT)? {
        (status::OK, payload) => decode_display_area(&payload),
        _ => None,
    };
    emit(UiEvent::DisplaySetup(SetupRequest {
        guide_mm: mounting.width_mm,
        current,
    }));
    let choice = loop {
        match answers.recv_timeout(SETUP_POLL) {
            Ok(choice) => break choice,
            // Relaxed: a pure signal from the UI thread.
            Err(RecvTimeoutError::Timeout) if !abort.load(Ordering::Relaxed) => {}
            Err(_) => bail!("aborted"),
        }
    };
    let area = display_area_basic(
        choice.width_mm,
        choice.height_mm,
        choice.offset_x_mm,
        &mounting,
    );
    emit(UiEvent::Status("setting the display area...".into()));
    let reply = backend.request(
        kind::DISPLAY_AREA_SET,
        &encode_display_area(&area),
        SETUP_REQUEST_TIMEOUT,
    )?;
    expect_ok(reply, "could not set the display area")?;
    tracing::info!(
        width_mm = choice.width_mm,
        height_mm = choice.height_mm,
        offset_x_mm = choice.offset_x_mm,
        "display area set"
    );
    Ok(current.is_none_or(|c| !same_area(&c, &area)))
}

/// Equal to within what the wire (f32) carries.
fn same_area(a: &DisplayArea, b: &DisplayArea) -> bool {
    [
        (a.top_left_mm, b.top_left_mm),
        (a.top_right_mm, b.top_right_mm),
        (a.bottom_left_mm, b.bottom_left_mm),
    ]
    .iter()
    .all(|(p, q)| (0..3).all(|i| (p[i] - q[i]).abs() < 0.01))
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

/// What the error of a session that computed nothing adds.
const NOT_CALIBRATED: &str = "not calibrated; the previous calibration stays";

/// Run a session, reporting progress through `emit`. With `setup`, the
/// display setup opens it and its answer arrives there: the calibration is
/// only valid for the display area it is made on. `abort` (Esc) stops the
/// session between steps, and the error then says which calibration is in
/// use: the previous one (the daemon also puts back the display area it was
/// made on) when nothing was computed, else the one computed so far.
///
/// # Errors
/// Fails when the tracker refuses a step, stops answering, or `abort` is set.
pub(crate) fn run(
    backend: &mut dyn Backend,
    rounds: usize,
    timing: Timing,
    setup: Option<&Receiver<Choice>>,
    abort: &AtomicBool,
    emit: &dyn Fn(UiEvent),
) -> Result<Summary> {
    emit(UiEvent::Status("waiting for the tracker...".into()));
    backend.wait_ready(abort).context(NOT_CALIBRATED)?;
    backend
        .request(kind::CALIBRATION_START, &[2], Duration::from_secs(25))
        .and_then(|r| expect_ok(r, "could not start"))
        .context(NOT_CALIBRATED)?;

    let mut computed = false;
    let result = (|| {
        let mut rounds = rounds;
        if let Some(answers) = setup
            && display_setup(backend, answers, abort, emit)?
            && rounds < 2
        {
            // The tracker keeps the last 14 points: with one round, 7 made on
            // the old display area would stay in the new calibration.
            tracing::info!("the display area changed: calibrating two rounds");
            rounds = 2;
        }
        collect_and_compute(backend, rounds, timing, abort, emit, &mut computed)
    })();
    // Stopping also restores the previous calibration, and display area,
    // when nothing was computed.
    let stopped = backend.request(kind::CALIBRATION_STOP, &[], Duration::from_secs(20));
    let summary = result.context(if computed {
        "stopped early; the calibration computed so far is in use"
    } else {
        NOT_CALIBRATED
    })?;
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
    computed: &mut bool,
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
            // Esc during the collect: stop before the batch is computed.
            // Relaxed: a pure signal from the UI thread.
            if abort.load(Ordering::Relaxed) {
                bail!("aborted");
            }
        }
        emit(UiEvent::Computing);
        let reply = backend.request(kind::CALIBRATION_COMPUTE, &[], Duration::from_secs(20))?;
        let payload = expect_ok(reply, "could not compute")?;
        *computed = true;
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
        fn wait_ready(&mut self, _abort: &AtomicBool) -> Result<()> {
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
        let summary = run(
            &mut backend,
            1,
            QUICK,
            None,
            &AtomicBool::new(false),
            &|e| {
                events.lock().expect("events").push(e);
            },
        )
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
        let err = run(
            &mut backend,
            1,
            QUICK,
            None,
            &AtomicBool::new(false),
            &|_| {},
        )
        .expect_err("busy");
        let err = format!("{err:#}");
        assert!(err.contains("another client"), "{err}");
        assert!(err.contains("previous calibration stays"), "{err}");
        assert_eq!(
            backend.0.into_inner().expect("log"),
            vec![kind::CALIBRATION_START]
        );
    }

    /// Answers like a tracker would; records every request with its payload.
    struct SetupDevice {
        log: Vec<(u8, Vec<u8>)>,
        /// What `CALIBRATION_START` answers.
        start: u8,
        /// What `GEOMETRY_MOUNTING` answers.
        mounting: u8,
        /// What `CALIBRATION_COLLECT_2D` answers.
        collect: u8,
        /// Set on this request (the user pressing Esc then).
        abort_on: Option<(u8, std::sync::Arc<AtomicBool>)>,
    }

    impl SetupDevice {
        fn new() -> Self {
            Self {
                log: Vec::new(),
                start: status::OK,
                mounting: status::OK,
                collect: status::OK,
                abort_on: None,
            }
        }

        fn kinds(&self) -> Vec<u8> {
            self.log.iter().map(|(k, _)| *k).collect()
        }

        fn count(&self, kind: u8) -> usize {
            self.log.iter().filter(|(k, _)| *k == kind).count()
        }
    }

    /// The display area the tracker has before the setup.
    fn before() -> DisplayArea {
        display_area_basic(633.6, 334.3, 1.0, &DRY_RUN_MOUNTING)
    }

    impl Backend for SetupDevice {
        fn wait_ready(&mut self, _abort: &AtomicBool) -> Result<()> {
            Ok(())
        }

        fn request(&mut self, kind: u8, payload: &[u8], _timeout: Duration) -> Result<Reply> {
            self.log.push((kind, payload.to_vec()));
            if let Some((on, abort)) = &self.abort_on
                && *on == kind
            {
                abort.store(true, Ordering::Relaxed);
            }
            Ok(match kind {
                kind::CALIBRATION_START => (self.start, Vec::new()),
                kind::GEOMETRY_MOUNTING => (
                    self.mounting,
                    tobii_ipc::request::encode_geometry_mounting(&DRY_RUN_MOUNTING),
                ),
                kind::DISPLAY_AREA_GET => (status::OK, encode_display_area(&before())),
                kind::CALIBRATION_COLLECT_2D => (self.collect, Vec::new()),
                _ => (status::OK, Vec::new()),
            })
        }
    }

    const CHOICE: Choice = Choice {
        width_mm: 597.0,
        height_mm: 336.0,
        offset_x_mm: -2.5,
    };

    fn answered(choice: Choice) -> Receiver<Choice> {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(choice).expect("send");
        rx
    }

    #[test]
    fn the_display_setup_runs_inside_the_session_before_the_points() {
        let rx = answered(CHOICE);
        let mut device = SetupDevice::new();
        let events = Mutex::new(Vec::new());

        run(
            &mut device,
            2,
            QUICK,
            Some(&rx),
            &AtomicBool::new(false),
            &|e| {
                events.lock().expect("events").push(e);
            },
        )
        .expect("session");

        assert_eq!(
            device.kinds()[..5],
            [
                kind::CALIBRATION_START,
                kind::GEOMETRY_MOUNTING,
                kind::DISPLAY_AREA_GET,
                kind::DISPLAY_AREA_SET,
                kind::CALIBRATION_COLLECT_2D
            ]
        );
        let written = decode_display_area(&device.log[3].1).expect("area");
        // Computed from the mounting as the wire carried it (f32).
        let mounting = decode_geometry_mounting(&tobii_ipc::request::encode_geometry_mounting(
            &DRY_RUN_MOUNTING,
        ))
        .expect("mounting");
        let expected = decode_display_area(&encode_display_area(&display_area_basic(
            597.0, 336.0, -2.5, &mounting,
        )))
        .expect("area");
        assert_eq!(written, expected);
        let asked = events
            .lock()
            .expect("events")
            .iter()
            .find_map(|e| match e {
                UiEvent::DisplaySetup(r) => Some(*r),
                _ => None,
            })
            .expect("setup shown");
        assert!((asked.guide_mm - 184.0).abs() < 1e-9 && asked.current.is_some());
        assert_eq!(device.kinds().last(), Some(&kind::CALIBRATION_STOP));
    }

    #[test]
    fn escaping_the_display_setup_stops_the_session_without_a_change() {
        let (_tx, rx) = std::sync::mpsc::channel();
        let mut device = SetupDevice::new();
        let err = run(
            &mut device,
            1,
            QUICK,
            Some(&rx),
            &AtomicBool::new(true),
            &|_| {},
        )
        .expect_err("aborted");
        assert_eq!(err.root_cause().to_string(), "aborted");
        assert!(
            format!("{err:#}").contains("previous calibration stays"),
            "{err:#}"
        );
        assert_eq!(
            device.kinds(),
            [
                kind::CALIBRATION_START,
                kind::GEOMETRY_MOUNTING,
                kind::DISPLAY_AREA_GET,
                kind::CALIBRATION_STOP
            ]
        );
    }

    #[test]
    fn a_busy_tracker_is_not_touched() {
        let rx = answered(CHOICE);
        let mut device = SetupDevice {
            start: status::CALIBRATION_BUSY,
            ..SetupDevice::new()
        };
        let err = run(
            &mut device,
            1,
            QUICK,
            Some(&rx),
            &AtomicBool::new(false),
            &|_| {},
        )
        .expect_err("busy");
        assert!(format!("{err:#}").contains("another client"), "{err:#}");
        assert_eq!(device.kinds(), [kind::CALIBRATION_START]);
    }

    #[test]
    fn a_session_failing_after_the_setup_is_stopped_for_the_daemon_to_restore() {
        let rx = answered(CHOICE);
        let mut device = SetupDevice {
            collect: status::OPERATION_FAILED,
            ..SetupDevice::new()
        };
        let err = run(
            &mut device,
            1,
            QUICK,
            Some(&rx),
            &AtomicBool::new(false),
            &|_| {},
        )
        .expect_err("failed");
        assert!(
            format!("{err:#}").contains("previous calibration stays"),
            "{err:#}"
        );
        assert_eq!(
            device.count(kind::DISPLAY_AREA_SET),
            1,
            "the daemon puts it back"
        );
        assert_eq!(device.kinds().last(), Some(&kind::CALIBRATION_STOP));
    }

    #[test]
    fn esc_during_the_last_collect_of_a_batch_computes_nothing() {
        let rx = answered(CHOICE);
        let abort = std::sync::Arc::new(AtomicBool::new(false));
        let mut device = SetupDevice {
            // The first batch is one point: its collect ends the batch.
            abort_on: Some((kind::CALIBRATION_COLLECT_2D, std::sync::Arc::clone(&abort))),
            ..SetupDevice::new()
        };
        let err = run(&mut device, 1, QUICK, Some(&rx), &abort, &|_| {}).expect_err("aborted");
        assert!(
            format!("{err:#}").contains("previous calibration stays"),
            "{err:#}"
        );
        assert_eq!(device.count(kind::CALIBRATION_COMPUTE), 0);
        assert_eq!(device.kinds().last(), Some(&kind::CALIBRATION_STOP));
    }

    #[test]
    fn a_session_stopped_after_a_compute_keeps_what_it_computed() {
        let rx = answered(CHOICE);
        let abort = std::sync::Arc::new(AtomicBool::new(false));
        let mut device = SetupDevice {
            abort_on: Some((kind::CALIBRATION_COMPUTE, std::sync::Arc::clone(&abort))),
            ..SetupDevice::new()
        };
        let err = run(&mut device, 1, QUICK, Some(&rx), &abort, &|_| {}).expect_err("aborted");
        assert!(
            format!("{err:#}").contains("computed so far is in use"),
            "{err:#}"
        );
        assert_eq!(device.count(kind::CALIBRATION_COMPUTE), 1);
        assert_eq!(device.kinds().last(), Some(&kind::CALIBRATION_STOP));
    }

    #[test]
    fn a_tracker_without_guide_marks_skips_the_setup() {
        let rx = answered(CHOICE);
        let mut device = SetupDevice {
            mounting: status::NOT_SUPPORTED,
            ..SetupDevice::new()
        };
        let events = Mutex::new(Vec::new());
        run(
            &mut device,
            1,
            QUICK,
            Some(&rx),
            &AtomicBool::new(false),
            &|e| {
                events.lock().expect("events").push(e);
            },
        )
        .expect("session");
        assert_eq!(device.count(kind::DISPLAY_AREA_SET), 0);
        assert!(
            !events
                .lock()
                .expect("events")
                .iter()
                .any(|e| matches!(e, UiEvent::DisplaySetup(_)))
        );
    }

    #[test]
    fn one_round_becomes_two_only_when_the_display_area_changed() {
        let collects = |choice| {
            let rx = answered(choice);
            let mut device = SetupDevice::new();
            run(
                &mut device,
                1,
                QUICK,
                Some(&rx),
                &AtomicBool::new(false),
                &|_| {},
            )
            .expect("session");
            device.count(kind::CALIBRATION_COLLECT_2D)
        };
        assert_eq!(collects(CHOICE), 14);
        let unchanged = Choice {
            width_mm: 633.6,
            height_mm: 334.3,
            offset_x_mm: 1.0,
        };
        assert_eq!(collects(unchanged), 7);
    }

    #[test]
    fn abort_stops_the_session() {
        let mut backend = Recorder(Mutex::new(Vec::new()), status::OK);
        let err = run(
            &mut backend,
            2,
            QUICK,
            None,
            &AtomicBool::new(true),
            &|_| {},
        )
        .expect_err("aborted");
        assert_eq!(err.root_cause().to_string(), "aborted");
        assert_eq!(
            backend.0.into_inner().expect("log"),
            vec![kind::CALIBRATION_START, kind::CALIBRATION_STOP]
        );
    }
}
