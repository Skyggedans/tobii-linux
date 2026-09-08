//! The Tobii daemon: claims the device and serves head/gaze/presence streams to
//! clients over a Unix socket. Run it directly, or let a client auto-spawn it.

fn main() {
    tobii::logging::init();
    if let Err(e) = tobii::daemon::run() {
        // Full cause chain on one line, written directly (not via `tracing`) so
        // a restrictive `RUST_LOG` can never hide the reason the daemon exited.
        eprintln!("tobiid: {e:#}");
        std::process::exit(1);
    }
}
