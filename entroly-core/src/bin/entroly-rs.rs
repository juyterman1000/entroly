//! `entroly-rs` — the standalone, zero-dependency single-binary distribution.
//!
//! Phase 1: the core CLI (context compression) wired directly to the pure-Rust
//! `entroly_core` engine — no Python runtime required. The Python (`pip`) and
//! npm packages are unaffected; this binary is purely additive.
//!
//! Arg parsing is hand-rolled (no clap) to keep the binary dependency-free.

use std::io::{Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") | Some("-V") => {
            println!("entroly-rs {}", env!("CARGO_PKG_VERSION"));
        }
        Some("compress") => cmd_compress(&args[1..]),
        Some("proxy") => cmd_proxy(&args[1..]),
        Some("--help") | Some("-h") | None => print_help(),
        Some(other) => {
            eprintln!("entroly-rs: unknown command '{other}'. Try `entroly-rs --help`.");
            std::process::exit(64);
        }
    }
}

fn print_help() {
    println!("entroly-rs {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Single-binary context compressor — zero dependencies, no Python runtime.");
    println!();
    println!("USAGE:");
    println!("  entroly-rs compress [--budget N] [FILE]   compress FILE (or stdin) to ~N tokens");
    println!("  entroly-rs proxy [--port P] [--upstream URL] [--budget N] [--no-cache-align]");
    println!("                                            run the compressing proxy (needs `proxy` feature).");
    println!("                                            Cache-aligned by default (prefix-stable for provider");
    println!("                                            caches); --no-cache-align uses a global token budget.");
    println!("  entroly-rs --version");
    println!("  entroly-rs --help");
    println!();
    println!("With no FILE, reads stdin. Writes compressed text to stdout; the");
    println!("token before/after summary is written to stderr (so stdout stays clean");
    println!("for piping). Default budget: 2000 tokens.");
}

fn cmd_compress(args: &[String]) {
    let mut budget: usize = 2000;
    let mut file: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--budget" | "-b" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) if n > 0 => budget = n,
                    _ => {
                        eprintln!("entroly-rs compress: --budget needs a positive integer");
                        std::process::exit(64);
                    }
                }
            }
            other if !other.starts_with('-') => file = Some(other),
            other => {
                eprintln!("entroly-rs compress: unknown flag '{other}'");
                std::process::exit(64);
            }
        }
        i += 1;
    }

    let input = match file {
        Some(path) => std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("entroly-rs: cannot read {path}: {e}");
            std::process::exit(66);
        }),
        None => {
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("entroly-rs: cannot read stdin: {e}");
                std::process::exit(74);
            }
            s
        }
    };

    let est = |s: &str| (s.chars().count() / 4).max(1);
    let before = est(&input);
    let out = entroly_core::compress::compress_text(&input, budget);
    let after = est(&out);
    let saved = before.saturating_sub(after);
    let pct = if before > 0 {
        saved as f64 / before as f64 * 100.0
    } else {
        0.0
    };

    print!("{out}");
    let _ = std::io::stdout().flush();
    eprintln!("entroly-rs: ~{before} -> ~{after} tokens ({pct:.1}% saved, budget {budget})");
}

#[cfg(feature = "proxy")]
fn cmd_proxy(args: &[String]) {
    let mut port: u16 = 9377;
    let mut upstream = "https://api.anthropic.com".to_string();
    let mut budget: usize = 4000;
    let mut cache_aligned = true;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-cache-align" => cache_aligned = false,
            "--port" | "-p" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<u16>().ok()) {
                    Some(p) => port = p,
                    None => {
                        eprintln!("entroly-rs proxy: --port needs a valid port number");
                        std::process::exit(64);
                    }
                }
            }
            "--upstream" | "-u" => {
                i += 1;
                match args.get(i) {
                    Some(u) => upstream = u.clone(),
                    None => {
                        eprintln!("entroly-rs proxy: --upstream needs a URL");
                        std::process::exit(64);
                    }
                }
            }
            "--budget" | "-b" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    Some(b) if b > 0 => budget = b,
                    _ => {
                        eprintln!("entroly-rs proxy: --budget needs a positive integer");
                        std::process::exit(64);
                    }
                }
            }
            other => {
                eprintln!("entroly-rs proxy: unknown flag '{other}'");
                std::process::exit(64);
            }
        }
        i += 1;
    }
    if let Err(e) = entroly_core::proxy::run(port, &upstream, budget, cache_aligned) {
        eprintln!("entroly-rs proxy: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "proxy"))]
fn cmd_proxy(_args: &[String]) {
    eprintln!("entroly-rs: this binary was built without the proxy.");
    eprintln!("Rebuild with:  cargo build --release --bin entroly-rs --features proxy");
    std::process::exit(64);
}
