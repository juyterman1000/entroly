use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const ENTROLY_EXE: &[u8] = include_bytes!("../target/release/entroly.exe");
const ICON_ICO: &[u8] = include_bytes!("../../assets/icon.ico");

fn main() {
    println!("============================================================");
    println!("         Entroly Desktop & CLI Installer for Windows         ");
    println!("============================================================");
    println!();

    // 1. Target directory: %LOCALAPPDATA%\Programs\Entroly
    let local_app_data = match env::var("LOCALAPPDATA") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error: LOCALAPPDATA environment variable not found.");
            std::process::exit(1);
        }
    };

    let target_dir = PathBuf::from(&local_app_data).join("Programs").join("Entroly");
    println!("[1/4] Installing application binaries to {}...", target_dir.display());

    if let Err(e) = fs::create_dir_all(&target_dir) {
        eprintln!("Error creating target directory: {}", e);
        std::process::exit(1);
    }

    let exe_path = target_dir.join("entroly.exe");
    if let Err(e) = fs::write(&exe_path, ENTROLY_EXE) {
        eprintln!("Error writing entroly.exe: {}", e);
        std::process::exit(1);
    }

    let icon_path = target_dir.join("icon.ico");
    let _ = fs::write(&icon_path, ICON_ICO);
    println!("      Installed entroly.exe ({:.2} MB)", ENTROLY_EXE.len() as f64 / 1_048_576.0);

    // 2. Add to user PATH environment variable via registry/powershell
    println!("[2/4] Registering 'entroly' in User PATH environment...");
    register_user_path(&target_dir);

    // 3. Create Desktop and Start Menu shortcuts
    println!("[3/4] Creating Windows Desktop and Start Menu shortcuts...");
    create_shortcuts(&exe_path, &icon_path);

    // 4. Launch Entroly Desktop
    println!("[4/4] Launching Entroly Desktop Control Plane...");
    let _ = Command::new(&exe_path).arg("ui").spawn();

    println!();
    println!("============================================================");
    println!("                 Installation Successful!                   ");
    println!("============================================================");
    println!("Binary Location:  {}", exe_path.display());
    println!("Global CLI:       Type 'entroly' or 'entroly status' in any terminal!");
    println!("Desktop App:      Shortcut placed on your Windows Desktop.");
    println!("============================================================");
    println!();

    // Pause briefly if run by double clicking
    std::thread::sleep(std::time::Duration::from_millis(1500));
}

fn register_user_path(bin_dir: &Path) {
    let dir_str = bin_dir.to_str().unwrap_or("");
    let script = format!(
        r#"$p = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($p -notlike '*{0}*') {{ [Environment]::SetEnvironmentVariable('Path', "$p;{0}", 'User') }}"#,
        dir_str
    );

    let _ = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output();
}

fn create_shortcuts(exe_path: &Path, icon_path: &Path) {
    let script = format!(
        r#"
$ws = New-Object -ComObject WScript.Shell

# Desktop shortcut
$desktop = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$s1 = $ws.CreateShortcut((Join-Path $desktop 'Entroly Desktop.lnk'))
$s1.TargetPath = '{0}'
$s1.WorkingDirectory = '{1}'
$s1.IconLocation = '{2},0'
$s1.Description = 'Entroly Context Control Plane and Intelligence UI'
$s1.Save()

# Start menu shortcut
$sm = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Programs)
$s2 = $ws.CreateShortcut((Join-Path $sm 'Entroly Desktop.lnk'))
$s2.TargetPath = '{0}'
$s2.WorkingDirectory = '{1}'
$s2.IconLocation = '{2},0'
$s2.Description = 'Entroly Context Control Plane and Intelligence UI'
$s2.Save()
"#,
        exe_path.display(),
        exe_path.parent().unwrap_or(exe_path).display(),
        icon_path.display()
    );

    let _ = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output();
}
