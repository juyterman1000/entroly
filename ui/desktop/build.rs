use std::fs;
use std::path::Path;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows" {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../assets/icon.ico");
        res.set("FileDescription", "Entroly Context Control Plane & Intelligence");
        res.set("ProductName", "Entroly Context Control Plane");
        res.set("OriginalFilename", "entroly.exe");
        res.set("LegalCopyright", "Copyright (c) 2026 Entroly Contributors");
        let _ = res.compile();
    }

    // Ensure target/release/entroly.exe exists so installer.rs include_bytes! never fails on fresh clone
    let target_release = Path::new("target/release");
    let target_exe = target_release.join("entroly.exe");
    if !target_exe.exists() {
        let _ = fs::create_dir_all(target_release);
        // If ui/dist/entroly.exe exists, copy it as the pre-warmed baseline
        let dist_exe = Path::new("../dist/entroly.exe");
        if dist_exe.exists() {
            let _ = fs::copy(dist_exe, &target_exe);
        } else {
            let _ = fs::write(&target_exe, b"ENTROLY_PREBUILD_PLACEHOLDER");
        }
    }
}
