//! Process-wide diagnostics setup for the binaries.
//!
//! The library itself only emits through the [`tracing`] facade (it never
//! installs a subscriber, so `libtobii.so` stays silent inside a host
//! application unless that application sets one up). Each executable calls
//! [`init`] once at start.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

/// Install the default subscriber: human-readable lines on stderr, level
/// filtered by `RUST_LOG` (default `info`). Calling it twice is harmless; the
/// second call is ignored.
pub fn init() {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
