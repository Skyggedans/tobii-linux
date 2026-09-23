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

use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tobii_ipc::request::{kind, status};
use winit::event_loop::EventLoop;

use crate::sequence::{Backend, DryRun, Timing, UiEvent};
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
  --windowed         a 1280x800 window instead of fullscreen";

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
    let (code, _) = conn.request(kind::CALIBRATION_APPLY, &[], Duration::from_secs(20))?;
    if code != status::OK {
        bail!("the daemon refused (status {code})");
    }
    println!("back to the built-in calibration");
    Ok(())
}

/// How long closing the window waits for the worker to wind the session
/// down (a pending collect or compute answers within seconds).
const WORKER_GRACE: Duration = Duration::from_secs(25);

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
    let (done, worker_done) = mpsc::channel::<Result<(), String>>();
    if !o.list_monitors {
        let proxy = event_loop.create_proxy();
        let gaze_proxy = event_loop.create_proxy();
        let abort = Arc::clone(&abort);
        let placed = Arc::clone(&placed);
        let (rounds, dry_run, export) = (o.rounds, o.dry_run, o.export.clone());
        thread::spawn(move || {
            let emit = |e: UiEvent| {
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
            let reported = match &outcome {
                Ok(summary) => match &export {
                    Some(path) => std::fs::write(path, &summary.blob).map_err(|e| {
                        format!("the calibration is kept, but could not be exported to {path}: {e}")
                    }),
                    None => Ok(()),
                },
                Err(e) => Err(format!("{e:#}")),
            };
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
    // Close the window now, not after the wait below.
    drop(app);
    // Esc or a closed window: have the worker stop the session (it checks
    // between steps) and wait for it, within reason.
    abort.store(true, std::sync::atomic::Ordering::Relaxed);
    let worker = worker_done.recv_timeout(WORKER_GRACE);
    // The window's own failure (no such monitor, no window) comes first: the
    // worker then only saw the abort that followed it.
    if let Some(reason) = window_failed {
        bail!(reason);
    }
    match worker {
        Ok(Err(reason)) => bail!(reason),
        Ok(Ok(())) | Err(mpsc::RecvTimeoutError::Disconnected) => Ok(()),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            bail!(
                "the calibration did not wind down in time; the daemon discards it when this exits"
            )
        }
    }
}

fn main() -> ExitCode {
    tobii_log::init();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tobii-calibrate: {e:#}");
            ExitCode::FAILURE
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
}
