fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows" {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../assets/icon.ico");
        res.set("FileDescription", "Entroly Control Plane Desktop Application");
        res.set("ProductName", "Entroly Control Plane");
        res.set("OriginalFilename", "EntrolyControlPlane.exe");
        res.set("LegalCopyright", "Copyright (c) 2026 Entroly Contributors");
        let _ = res.compile();
    }
}
