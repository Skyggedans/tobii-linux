//! Command-line entry point: research/diagnostic subcommands (see `tobii::cli`).

fn main() {
    tobii::logging::init();
    if let Err(e) = tobii::run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
