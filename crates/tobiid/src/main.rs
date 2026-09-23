//! The Tobii daemon: claims the device, serves its streams to clients over a
//! Unix socket and answers their requests (device info, geometry, display
//! area, calibration). Run it directly, or let a client auto-spawn it.

mod calibration;
mod daemon;
mod device;
mod display;
mod frames;
mod requests;

fn main() {
    tobii_log::init();
    if let Err(e) = daemon::run() {
        // Full cause chain on one line, written directly (not via `tracing`) so
        // a restrictive `RUST_LOG` can never hide the reason the daemon exited.
        eprintln!("tobiid: {e:#}");
        std::process::exit(1);
    }
}
