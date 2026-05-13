fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--version") | Some("-V") => {
            println!("entroly-rs {}", env!("CARGO_PKG_VERSION"));
        }
        Some("--help") | Some("-h") | None => {
            println!("entroly-rs {}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("Native launcher placeholder for future packaged Rust CLI.");
            println!("Use the Python `entroly` command for the current product CLI.");
        }
        Some(cmd) => {
            eprintln!(
                "entroly-rs does not implement `{}` yet; use the Python `entroly` command.",
                cmd
            );
            std::process::exit(64);
        }
    }
}
