//! Command-line entry point: research/diagnostic subcommands (see `tobii5_init_replay::cli`).

fn main() {
    tobii5_init_replay::logging::init();
    if let Err(e) = tobii5_init_replay::run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
