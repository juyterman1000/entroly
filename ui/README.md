# Entroly Desktop & CLI Application

The `ui/` directory contains the standalone, desktop-ready control plane interface and native Windows binaries for Entroly.

## Architecture: Just Like `docker.exe`

Entroly provides a dual-mode native Windows binary (`entroly.exe`) and a standalone 1-click installer (`EntrolySetup.exe`), matching the architecture of **Docker Desktop**:

1. **Dual CLI & GUI Binary (`entroly.exe`)**:
   - When run in a terminal: Functions as a fast CLI tool (`entroly --help`, `entroly status`, `entroly doctor`, `entroly start`).
   - When run with `entroly ui` or double-clicked from the desktop: Launches the native standalone Control Plane desktop application.
2. **Standalone 1-Click Installer (`EntrolySetup.exe`)**:
   - Single executable installer (like `Docker Desktop Installer.exe`).
   - Automatically installs binaries to `%LOCALAPPDATA%\Programs\Entroly`.
   - Registers `entroly` into the Windows user `PATH` environment variable so it works in any terminal.
   - Places `Entroly Desktop` shortcuts on the Desktop and Start Menu with the custom Entroly icon.
3. **Zero Framework Bloat**:
   - Pure native Rust binary (`411 KB`) embedding all web assets directly.
   - Uses native Windows Edge/Chrome App Mode: zero Electron overhead, zero memory bloat, 60 FPS hardware-accelerated rendering.

---

## Installation & Running

### Option 1: 1-Click Installer Executable
Double-click `ui/dist/EntrolySetup.exe` (or run it from PowerShell):
```powershell
& ui\dist\EntrolySetup.exe
```

### Option 2: CLI Usage (Available Globally After Install)
```powershell
entroly --help     # Show help and CLI commands
entroly --version  # View version and engine status
entroly status     # Check live daemon metrics and health score
entroly doctor     # Run full system diagnostics
entroly ui         # Launch the desktop Control Plane window
```

### Option 3: Compile From Source
```powershell
cd ui/desktop
cargo build --release --bin entroly
cargo build --release --bin EntrolySetup
```
