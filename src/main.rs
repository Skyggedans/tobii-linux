fn main() {
    if let Err(e) = tobii::run() {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
