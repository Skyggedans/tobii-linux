//! Process-wide diagnostics setup for the binaries.
//!
//! The library itself only emits through the [`tracing`] facade (it never
//! installs a subscriber, so `libtobii.so` stays silent inside a host
//! application unless that application sets one up). Each executable calls
//! [`init`] once at start.

use std::io::IsTerminal;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

/// Default filter: our own events at `info`, ONNX Runtime's (it also emits
/// through `tracing`) only at `warn` — its session/arena chatter is not useful
/// in a service journal. `RUST_LOG` overrides both.
const DEFAULT_DIRECTIVES: &str = "ort=warn";

/// Install the default subscriber: human-readable lines on stderr, level
/// filtered by `RUST_LOG` (default `info`, see `DEFAULT_DIRECTIVES`), ANSI
/// colour only when stderr is a terminal. Calling it twice is harmless; the
/// second call is ignored.
pub fn init() {
    let mut filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    if std::env::var_os("RUST_LOG").is_none_or(|v| v.is_empty()) {
        for directive in DEFAULT_DIRECTIVES.split(',') {
            if let Ok(d) = directive.parse() {
                filter = filter.add_directive(d);
            }
        }
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(std::io::stderr().is_terminal())
        .with_writer(std::io::stderr)
        .try_init();
}
