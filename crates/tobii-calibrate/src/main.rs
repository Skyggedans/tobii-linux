//! `tobii-calibrate`: calibrate the Tobii Eye Tracker 5 on Linux.
//!
//! First the display setup: two ticks at the bottom of the screen, lined up
//! with the two white marks on the tracker, tell the tracker the monitor's
//! size and where it sits under it (the display area; the daemon keeps it).
//! Then the 7-point pattern fullscreen on that monitor (two rounds by
//! default): `tobiid` collects each point and computes the calibration, and
//! the result is shown with a live gaze dot to check it. The daemon saves
//! the calibration and uploads it at every later start; `--reset` goes back
//! to the calibration built into the driver.
//!
//! Talks to the daemon directly over its socket, like the other clients.

mod draw;
mod edid;
mod ipc;
mod sequence;
mod setup;
mod ui;

use std::fmt;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use winit::event_loop::EventLoop;

use crate::sequence::{Backend, DryRun, SavedNotTaken, Summary, Timing, UiEvent};
use crate::ui::{MonitorChoice, Signals, Ui};

const USAGE: &str = "\
usage: tobii-calibrate [options]

  --monitor N|NAME   calibrate on this monitor (see --list-monitors);
                     default: the primary one
  --list-monitors    print the monitors and exit
  --no-display-setup keep the display area the tracker has (skip lining up
                     the ticks with the marks on the tracker)
  --rounds N         rounds of the 7-point pattern (default 2: the tracker
                     keeps the last 14 points)
  --dwell-ms N       time to settle on each point before collecting (1000)
  --verify-secs N    how long to show the result with the live gaze (15)
  --export PATH      also write the calibration to PATH
  --reset            go back to the driver's built-in calibration and exit
  --dry-run          run the screens without a tracker
  --windowed         a 1280x800 window instead of fullscreen

exit status: 0 calibrated; 2 the calibration was saved and the tracker takes
it at its next start, but may not have taken it at once; 1 anything else";

struct Options {
    monitor: MonitorChoice,
    list_monitors: bool,
    rounds: usize,
    dwell: Duration,
    verify: Duration,
    export: Option<String>,
    reset: bool,
    dry_run: bool,
    windowed: bool,
    display_setup: bool,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Options> {
    let mut o = Options {
        monitor: MonitorChoice::Default,
        list_monitors: false,
        rounds: 2,
        dwell: Duration::from_millis(1000),
        verify: Duration::from_secs(15),
        export: None,
        reset: false,
        dry_run: false,
        windowed: false,
        display_setup: true,
    };
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .with_context(|| format!("{name} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--monitor" => {
                let v = value("--monitor")?;
                o.monitor = v
                    .parse()
                    .map_or(MonitorChoice::Name(v), MonitorChoice::Index);
            }
            "--list-monitors" => o.list_monitors = true,
            "--rounds" => o.rounds = value("--rounds")?.parse().context("--rounds")?,
            "--dwell-ms" => {
                o.dwell =
                    Duration::from_millis(value("--dwell-ms")?.parse().context("--dwell-ms")?);
            }
            "--verify-secs" => {
                o.verify =
                    Duration::from_secs(value("--verify-secs")?.parse().context("--verify-secs")?);
            }
            "--export" => o.export = Some(value("--export")?),
            "--reset" => o.reset = true,
            "--dry-run" => o.dry_run = true,
            "--windowed" => o.windowed = true,
            "--no-display-setup" => o.display_setup = false,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown option {other}\n{USAGE}"),
        }
    }
    if o.rounds == 0 {
        bail!("--rounds must be at least 1");
    }
    if o.windowed && !o.dry_run {
        // Targets in a window are not where the tracker expects them.
        bail!("--windowed only goes with --dry-run: a calibration needs the whole monitor");
    }
    Ok(o)
}

/// `--reset`: have the daemon upload the built-in calibration and forget the
/// saved one.
fn reset() -> Result<()> {
    let mut conn = ipc::Connection::open(Box::new(|_, _, _| {}))?;
    // A cold tracker takes ~12 s to start, a stalled one longer.
    println!("going back to the built-in calibration: waiting for the tracker...");
    sequence::reset(&mut conn)?;
    println!("back to the built-in calibration");
    Ok(())
}

/// Exit status of any failure but [`EXIT_SAVED_NOT_TAKEN`].
const EXIT_FAILURE: u8 = 1;

/// Exit status when tobiid saved the calibration but the tracker may not
/// have taken it at the end of the session: it takes it at its next start.
const EXIT_SAVED_NOT_TAKEN: u8 = 2;

/// How the worker says a session ended badly.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Failure {
    /// What the terminal says.
    reason: String,
    /// tobiid saved the calibration all the same (a [`SavedNotTaken`]): the
    /// exit status is [`EXIT_SAVED_NOT_TAKEN`].
    saved: bool,
}

impl Failure {
    fn new(reason: String) -> Self {
        Self {
            reason,
            saved: false,
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for Failure {}

/// The exit status for what `run` failed with.
fn exit_status(e: &anyhow::Error) -> u8 {
    match e.downcast_ref::<Failure>() {
        Some(Failure { saved: true, .. }) => EXIT_SAVED_NOT_TAKEN,
        _ => EXIT_FAILURE,
    }
}

/// What the worker reports once the session has ended, having written its
/// calibration to `export` if it was kept, or saved by tobiid though the
/// tracker may not have taken it (it is the user's from the tracker's next
/// start).
fn report(outcome: &Result<Summary>, export: Option<&str>) -> Result<(), Failure> {
    let (summary, not_taken) = match outcome {
        Ok(summary) => (summary, None),
        Err(e) => match e.downcast_ref::<SavedNotTaken>() {
            Some(saved) => (&saved.summary, Some(format!("{e:#}"))),
            None => return Err(Failure::new(format!("{e:#}"))),
        },
    };
    let exported = export.map_or(Ok(()), |path| {
        std::fs::write(path, &summary.blob)
            .map_err(|e| format!("could not be exported to {path}: {e}"))
    });
    match (not_taken, exported) {
        (None, Ok(())) => Ok(()),
        (None, Err(why)) => Err(Failure::new(format!("the calibration is kept, but {why}"))),
        (Some(reason), exported) => Err(Failure {
            reason: match exported {
                Ok(()) => reason,
                Err(why) => format!("{reason}; the calibration {why}"),
            },
            saved: true,
        }),
    }
}

/// How long closing the window waits for the worker to wind the session
/// down. A pending step usually answers within seconds; one the tracker
/// holds up may take minutes, and tobiid then carries it out after this
/// exits (see [`outcome`]).
const WORKER_GRACE: Duration = Duration::from_secs(25);

/// How long closing the window waits for the worker before the terminal says
/// it waits: when the tracker answers at once, the worker notices the abort
/// and stops the session within about a second.
const WORKER_NOTICE: Duration = Duration::from_secs(1);

/// How the worker ended the session, waited for up to `grace` once the
/// window has closed. Past `notice`, `tell` says that the wait goes on, so
/// that a step the tracker holds up does not leave the terminal silent.
fn wait_for_worker(
    worker_done: &mpsc::Receiver<Result<(), Failure>>,
    notice: Duration,
    grace: Duration,
    tell: impl FnOnce(),
) -> Result<Result<(), Failure>, mpsc::RecvTimeoutError> {
    match worker_done.recv_timeout(notice.min(grace)) {
        Err(mpsc::RecvTimeoutError::Timeout) => {
            tell();
            worker_done.recv_timeout(grace.saturating_sub(notice))
        }
        done => done,
    }
}

/// What the terminal says once the window has closed: `worker` is how the
/// session ended, or that it had not within [`WORKER_GRACE`]; `escaped`
/// whether Esc or the window's close ended it; `saving` whether the stop
/// that keeps the session had been sent; `export` where `--export` would
/// have written the calibration. tobiid carries out the requests a client
/// sent even after it hangs up, and only then discards a session the client
/// still holds: a stop under way that keeps the session still keeps it.
/// That is still a failure here: this exits before the stop answers, so
/// the save is not confirmed (the stop may yet fail), and the calibration
/// is not exported. A [`Failure`] the worker reports is the error as it is,
/// for [`exit_status`].
fn outcome(
    worker: Result<Result<(), Failure>, mpsc::RecvTimeoutError>,
    escaped: bool,
    saving: bool,
    export: Option<&str>,
) -> Result<()> {
    match worker {
        Ok(Err(failure)) => Err(failure.into()),
        Ok(Ok(())) => {
            if escaped {
                println!("the calibration was kept: it was saved before the window closed");
            }
            Ok(())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => Ok(()),
        Err(mpsc::RecvTimeoutError::Timeout) if saving => {
            let exported = export.map_or_else(String::new, |path| {
                format!("; it was not exported to {path}")
            });
            bail!(
                "the calibration was still being saved when the window closed; tobiid finishes \
                 the save after this exits{exported}"
            )
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            bail!(
                "the calibration did not wind down in time; the daemon discards it when this exits"
            )
        }
    }
}

/// The window's own failure (no such monitor, no window), which the
/// terminal says before [`outcome`]: the worker then only saw the abort
/// that followed it. Not over a calibration tobiid saved, which the worker
/// tells itself: the window took the worker's failure for its own too.
fn window_failure(
    window_failed: Option<String>,
    worker: &Result<Result<(), Failure>, mpsc::RecvTimeoutError>,
) -> Option<String> {
    window_failed.filter(|_| !matches!(worker, Ok(Err(Failure { saved: true, .. }))))
}

fn run() -> Result<()> {
    let o = parse_args(std::env::args().skip(1))?;
    if o.reset {
        return reset();
    }
    let event_loop = EventLoop::<UiEvent>::with_user_event()
        .build()
        .context("no display to open a window on")?;
    let abort = Arc::new(AtomicBool::new(false));
    // Whether the window is fullscreen on its monitor: the points wait for
    // it (a window needs no placing).
    let placed = Arc::new(AtomicBool::new(o.windowed));
    let (answers, setup_answers) = mpsc::channel();
    // The setup measures the window, so it needs the whole monitor.
    let display_setup = o.display_setup && !o.windowed;
    if o.display_setup && o.windowed {
        tracing::info!("--windowed: skipping the display setup");
    }
    let timing = Timing {
        travel: Duration::from_millis(350),
        dwell: o.dwell,
    };

    // The worker reports how the session ended, so that closing the window
    // waits for it to stop the session, and the outcome reaches the terminal.
    let (done, worker_done) = mpsc::channel::<Result<(), Failure>>();
    // Whether the stop that keeps the session has been sent.
    let saving = Arc::new(AtomicBool::new(false));
    if !o.list_monitors {
        let proxy = event_loop.create_proxy();
        let gaze_proxy = event_loop.create_proxy();
        let abort = Arc::clone(&abort);
        let placed = Arc::clone(&placed);
        let saving = Arc::clone(&saving);
        let (rounds, dry_run, export) = (o.rounds, o.dry_run, o.export.clone());
        thread::spawn(move || {
            let emit = |e: UiEvent| {
                if matches!(e, UiEvent::Saving) {
                    // Relaxed: a pure signal, read once the window has closed.
                    saving.store(true, Ordering::Relaxed);
                }
                let _ = proxy.send_event(e);
            };
            let backend: Result<Box<dyn Backend>> = if dry_run {
                Ok(Box::new(DryRun))
            } else {
                ipc::Connection::open(Box::new(move |x, y, valid| {
                    let _ = gaze_proxy.send_event(UiEvent::Gaze([x, y], valid));
                }))
                .map(|c| Box::new(c) as Box<dyn Backend>)
            };
            let setup = display_setup.then_some(&setup_answers);
            let outcome = backend.and_then(|mut b| {
                sequence::run(b.as_mut(), rounds, timing, setup, &placed, &abort, &emit)
            });
            let reported = report(&outcome, export.as_deref());
            match outcome {
                Ok(summary) => emit(UiEvent::Finished(summary)),
                Err(e) => emit(UiEvent::Failed(format!("{e:#}"))),
            }
            let _ = done.send(reported);
        });
    } else {
        drop(done);
    }

    let mut app = Ui::new(
        Signals {
            abort: Arc::clone(&abort),
            placed,
        },
        o.monitor,
        o.list_monitors,
        o.windowed,
        timing,
        o.verify,
        answers,
    );
    event_loop.run_app(&mut app).context("window event loop")?;
    let window_failed = app.failed.take();
    let escaped = app.escaped;
    // Close the window now, not after the wait below.
    drop(app);
    // Esc or a closed window: have the worker stop the session (it checks
    // between steps) and wait for it, within reason.
    abort.store(true, Ordering::Relaxed);
    let worker = wait_for_worker(&worker_done, WORKER_NOTICE, WORKER_GRACE, || {
        println!(
            "waiting up to {} s for the calibration to wind down...",
            WORKER_GRACE.as_secs()
        );
    });
    if let Some(reason) = window_failure(window_failed, &worker) {
        bail!(reason);
    }
    // Relaxed: a pure signal from the worker.
    outcome(
        worker,
        escaped,
        saving.load(Ordering::Relaxed),
        o.export.as_deref(),
    )
}

fn main() -> ExitCode {
    tobii_log::init();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tobii-calibrate: {e:#}");
            ExitCode::from(exit_status(&e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options> {
        parse_args(args.iter().map(|s| (*s).to_string()))
    }

    #[test]
    fn parses_options() {
        let o = parse(&[
            "--monitor",
            "1",
            "--rounds",
            "3",
            "--dwell-ms",
            "600",
            "--dry-run",
            "--no-display-setup",
        ])
        .expect("options");
        assert_eq!(o.monitor, MonitorChoice::Index(1));
        assert_eq!(
            (o.rounds, o.dwell, o.dry_run, o.display_setup),
            (3, Duration::from_millis(600), true, false)
        );
        assert!(parse(&[]).expect("defaults").display_setup);
        assert_eq!(
            parse(&["--monitor", "DP-2"]).expect("name").monitor,
            MonitorChoice::Name("DP-2".into())
        );
        assert!(parse(&["--rounds", "0"]).is_err());
        assert!(parse(&["--bogus"]).is_err());
        assert!(parse(&["--export"]).is_err());
        assert!(parse(&["--windowed"]).is_err());
        assert!(parse(&["--windowed", "--dry-run"]).is_ok());
    }

    #[test]
    fn a_window_closed_on_a_save_under_way_says_the_save_goes_on() {
        let late = || Err(mpsc::RecvTimeoutError::Timeout);

        let saving = outcome(late(), true, true, None).expect_err("not confirmed");
        let discarding = outcome(late(), true, false, None).expect_err("not wound down");

        assert!(saving.to_string().contains("finishes the save"), "{saving}");
        assert!(!saving.to_string().contains("exported"), "{saving}");
        assert!(
            discarding.to_string().contains("discards it"),
            "{discarding}"
        );
        assert!(outcome(Ok(Ok(())), true, true, None).is_ok());
        let failed = outcome(Ok(Err(Failure::new("refused".into()))), false, true, None)
            .expect_err("failed");
        assert_eq!(failed.to_string(), "refused");
        assert_eq!(exit_status(&failed), EXIT_FAILURE);
        assert_eq!(
            exit_status(&saving),
            EXIT_FAILURE,
            "the save is not confirmed"
        );
    }

    /// A session run to its end whose calibration the device computed.
    fn summary() -> Summary {
        Summary {
            id: 0x1234_5678,
            points: 14,
            mean_error: None,
            blob: vec![1, 2, 3],
        }
    }

    /// What the worker makes of a session whose stop failed after tobiid
    /// saved its calibration.
    fn saved_not_taken(export: Option<&str>) -> Result<(), Failure> {
        let not_taken = SavedNotTaken {
            summary: summary(),
            status: tobii_ipc::request::status::OPERATION_FAILED,
        };
        report(&Err(not_taken.into()), export)
    }

    /// A calibration tobiid saved, though the tracker may not have taken it
    /// at the stop, is told as saved and exits with a status of its own; any
    /// other failure exits 1.
    #[test]
    fn a_calibration_saved_but_not_taken_has_an_exit_status_of_its_own() {
        let worker = Ok(saved_not_taken(None));

        let err = outcome(worker, false, true, None).expect_err("not taken");

        assert_eq!(exit_status(&err), EXIT_SAVED_NOT_TAKEN);
        assert_eq!(
            err.to_string(),
            "the calibration was saved and the tracker takes it at its next start; it may not \
             have taken it now: the tracker refused (status 13)"
        );
        let refused = report(&Err(anyhow::anyhow!("refused")), None);
        assert_eq!(refused, Err(Failure::new("refused".into())));
        let refused = outcome(Ok(refused), false, true, None).expect_err("failed");
        assert_eq!(exit_status(&refused), EXIT_FAILURE);
        assert_eq!(report(&Ok(summary()), None), Ok(()));
    }

    /// The window takes the worker's failure for its own; the terminal
    /// still says what the worker said of a saved calibration, with its exit
    /// status. The window's own failure comes first otherwise.
    #[test]
    fn a_saved_calibration_is_not_hidden_by_the_windows_copy_of_it() {
        let saved = Ok(saved_not_taken(None));
        let refused = Ok(Err(Failure::new("refused".into())));
        let failed = || Some("could not draw into the window".to_owned());

        assert_eq!(window_failure(failed(), &saved), None);
        assert_eq!(window_failure(failed(), &refused), failed());
        assert_eq!(window_failure(failed(), &Ok(Ok(()))), failed());
        assert_eq!(window_failure(None, &saved), None);
    }

    /// `--export` writes a calibration tobiid saved though the tracker may
    /// not have taken it: it is the user's from the tracker's next start. A
    /// write that fails is said too, and the exit status stays the save's.
    #[test]
    fn a_calibration_saved_but_not_taken_is_exported() {
        let dir =
            std::env::temp_dir().join(format!("tobii-calibrate-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("cal.bin");
        let path = path.to_str().expect("a UTF-8 path");
        let unwritable = dir.to_str().expect("a UTF-8 path");

        let exported = saved_not_taken(Some(path));
        let written = std::fs::read(path);
        let not_exported = saved_not_taken(Some(unwritable));
        let _ = std::fs::remove_dir_all(&dir);

        let Err(exported) = exported else {
            panic!("not taken");
        };
        assert!(exported.saved);
        assert!(!exported.reason.contains("export"), "{}", exported.reason);
        assert_eq!(written.expect("exported"), summary().blob);
        let Err(not_exported) = not_exported else {
            panic!("not taken");
        };
        assert!(not_exported.saved, "the exit status stays the save's");
        assert!(
            not_exported.reason.starts_with("the calibration was saved")
                && not_exported.reason.contains(&format!(
                    "; the calibration could not be exported to {unwritable}: "
                )),
            "{}",
            not_exported.reason
        );
    }

    #[test]
    fn a_save_cut_short_by_the_grace_says_the_export_was_not_written() {
        let late = Err(mpsc::RecvTimeoutError::Timeout);

        let err = outcome(late, true, true, Some("cal.bin")).expect_err("not confirmed");

        let err = err.to_string();
        assert!(err.contains("finishes the save"), "{err}");
        assert!(err.contains("not exported to cal.bin"), "{err}");
    }

    #[test]
    fn a_worker_that_already_reported_is_not_waited_on_aloud() {
        let (done, worker_done) = mpsc::channel();
        done.send(Ok(())).expect("send");
        let mut told = false;

        let worker = wait_for_worker(&worker_done, Duration::ZERO, Duration::ZERO, || {
            told = true;
        });

        assert_eq!(worker, Ok(Ok(())));
        assert!(!told);
    }

    #[test]
    fn a_worker_still_busy_past_the_notice_is_waited_on_aloud() {
        // The worker holds its end open without reporting.
        let (_done, worker_done) = mpsc::channel::<Result<(), Failure>>();
        let mut told = 0;

        let worker = wait_for_worker(&worker_done, Duration::ZERO, Duration::ZERO, || {
            told += 1;
        });

        assert_eq!(worker, Err(mpsc::RecvTimeoutError::Timeout));
        assert_eq!(told, 1);
    }

    #[test]
    fn a_worker_that_is_gone_is_not_waited_on_aloud() {
        // --list-monitors: no worker at all.
        let (done, worker_done) = mpsc::channel::<Result<(), Failure>>();
        drop(done);
        let mut told = false;

        let worker = wait_for_worker(&worker_done, Duration::ZERO, Duration::ZERO, || {
            told = true;
        });

        assert_eq!(worker, Err(mpsc::RecvTimeoutError::Disconnected));
        assert!(!told);
    }
}
