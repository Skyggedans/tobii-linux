//! Process-wide diagnostics setup for the binaries.
//!
//! The libraries only emit through the `tracing` facade and never install a
//! subscriber; each executable calls [`init`] once at start. `libtobii.so`
//! is not reached by this: a cdylib with its own copy of `tracing`, it stays
//! silent inside a host application, C or Rust, whose only view of its
//! diagnostics is the `tobii_custom_log_t` it hands `tobii_api_create`.

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
