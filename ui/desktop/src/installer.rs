use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const CREATE_NO_WINDOW: u32 = 0x08000000;

const ENTROLY_EXE: &[u8] = include_bytes!("../target/release/entroly.exe");
const ICON_ICO: &[u8] = include_bytes!("../../assets/icon.ico");

struct InstallConfig {
    silent: bool,
    no_launch: bool,
    uninstall: bool,
}

fn main() {
    let config = parse_args();

    if config.uninstall {
        run_uninstall(config.silent);
        return;
    }

    run_install(config);
}

fn parse_args() -> InstallConfig {
    let args: Vec<String> = env::args().collect();
    let mut silent = false;
    let mut no_launch = false;
    let mut uninstall = false;

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-s" | "--silent" | "/S" | "/s" => silent = true,
            "--no-launch" | "-n" => no_launch = true,
            "-u" | "--uninstall" | "/U" | "/u" => uninstall = true,
            "-h" | "--help" | "/?" => {
                print_installer_help();
                std::process::exit(0);
            }
            _ => {}
        }
    }

    InstallConfig {
        silent,
        no_launch,
        uninstall,
    }
}

fn print_installer_help() {
    println!(
        r#"Entroly Desktop & CLI Installer [Version 1.0.85]

Usage: EntrolySetup.exe [OPTIONS]

Options:
  -s, --silent       Run unattended installation (no interactive prompts)
  -u, --uninstall    Cleanly remove Entroly, shortcuts, and PATH entry
  -n, --no-launch    Install without launching the UI immediately
  -h, --help         Show this help message
"#
    );
}

fn log_info(msg: &str, silent: bool) {
    if !silent {
        println!("{}", msg);
    }
}

fn run_install(config: InstallConfig) {
    if !config.silent {
        println!("============================================================");
        println!("       Entroly Context Control Plane — Windows Setup        ");
        println!("============================================================");
        println!();
    }

    let local_app_data = match env::var("LOCALAPPDATA") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error: LOCALAPPDATA environment variable not found.");
            std::process::exit(1);
        }
    };

    let target_dir = PathBuf::from(&local_app_data).join("Programs").join("Entroly");
    log_info(&format!("[1/4] Preparing target directory: {}", target_dir.display()), config.silent);

    if let Err(e) = fs::create_dir_all(&target_dir) {
        eprintln!("Error creating target directory: {}", e);
        std::process::exit(1);
    }

    // Terminate any existing entroly processes to prevent "Access is denied" during file replacement
    terminate_existing_process();

    let exe_path = target_dir.join("entroly.exe");
    let icon_path = target_dir.join("icon.ico");

    // Resolve binary payload (embedded or fallback to adjacent entroly.exe)
    let binary_payload: Vec<u8> = if ENTROLY_EXE.len() >= 1024 {
        ENTROLY_EXE.to_vec()
    } else {
        let candidates = [
            PathBuf::from("entroly.exe"),
            PathBuf::from("ui/dist/entroly.exe"),
            PathBuf::from("../dist/entroly.exe"),
            PathBuf::from("target/release/entroly.exe"),
        ];
        let mut found = None;
        for c in &candidates {
            if let Ok(bytes) = fs::read(c) {
                if bytes.len() >= 1024 {
                    found = Some(bytes);
                    break;
                }
            }
        }
        found.unwrap_or_else(|| ENTROLY_EXE.to_vec())
    };

    log_info(&format!("[2/4] Deploying binary ({:.2} MB) and icons...", binary_payload.len() as f64 / 1_048_576.0), config.silent);
    
    // Attempt write with retry in case process termination is settling
    let mut written = false;
    for _ in 0..5 {
        if fs::write(&exe_path, &binary_payload).is_ok() {
            written = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    if !written {
        eprintln!("Error: Unable to write entroly.exe. Please ensure existing entroly instances are closed.");
        std::process::exit(1);
    }

    let _ = fs::write(&icon_path, ICON_ICO);

    // Register User PATH and broadcast environment change
    log_info("[3/4] Configuring User PATH and broadcasting environment update...", config.silent);
    register_user_path(&target_dir);

    // Create Start Menu and Desktop shortcuts
    log_info("[4/4] Creating Desktop and Start Menu application shortcuts...", config.silent);
    create_shortcuts(&exe_path, &icon_path);

    // Launch UI unless --no-launch was specified
    if !config.no_launch {
        log_info("      Starting Entroly daemon and launching Desktop Control Plane...", config.silent);
        let _ = Command::new(&exe_path).arg("ui").spawn();
    }

    if !config.silent {
        println!();
        println!("============================================================");
        println!("              [OK] Installation Successfully Completed!     ");
        println!("============================================================");
        println!("• Executable:      {}", exe_path.display());
        println!("• Terminal CLI:    Open any terminal and run 'entroly status'");
        println!("• Desktop App:     'Entroly Desktop' shortcut on your Desktop");
        println!("============================================================");
        println!();

        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
}

fn run_uninstall(silent: bool) {
    if !silent {
        println!("============================================================");
        println!("       Entroly Context Control Plane — Uninstaller          ");
        println!("============================================================");
        println!();
    }

    let local_app_data = match env::var("LOCALAPPDATA") {
        Ok(v) => v,
        Err(_) => return,
    };

    let target_dir = PathBuf::from(&local_app_data).join("Programs").join("Entroly");

    log_info("[1/3] Terminating active Entroly daemon instances...", silent);
    terminate_existing_process();

    log_info("[2/3] Removing shortcuts and unregistering User PATH...", silent);
    remove_shortcuts();
    unregister_user_path(&target_dir);

    log_info("[3/3] Removing installed files...", silent);
    if target_dir.exists() {
        let _ = fs::remove_dir_all(&target_dir);
    }

    if !silent {
        println!();
        println!("============================================================");
        println!("        [OK] Entroly Has Been Successfully Uninstalled      ");
        println!("============================================================");
        println!();
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}

fn run_powershell_script(script: &str) {
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let _ = cmd.output();
}

fn terminate_existing_process() {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/F", "/IM", "entroly.exe", "/T"]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let _ = cmd.output();
}

fn register_user_path(bin_dir: &Path) {
    let dir_str = bin_dir.to_str().unwrap_or("").replace('\'', "''");
    let script = format!(
        r#"
$dir = '{0}'
$p = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($p -notlike "*$dir*") {{
    $newP = if ([string]::IsNullOrWhiteSpace($p)) {{ $dir }} else {{ "$p;$dir" }}
    [Environment]::SetEnvironmentVariable('Path', $newP, 'User')
}}

# Broadcast setting change to notify active explorer and terminals
Add-Type -Namespace Win32 -Name Native -MemberDefinition '[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)] public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);' -ErrorAction SilentlyContinue
$result = [UIntPtr]::Zero
[Win32.Native]::SendMessageTimeout([IntPtr]0xffff, 0x001A, [UIntPtr]::Zero, "Environment", 2, 2000, [ref]$result) | Out-Null
"#,
        dir_str
    );

    run_powershell_script(&script);
}

fn unregister_user_path(bin_dir: &Path) {
    let dir_str = bin_dir.to_str().unwrap_or("").replace('\'', "''");
    let script = format!(
        r#"
$dir = '{0}'
$p = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($p -like "*$dir*") {{
    $items = ($p -split ';') | Where-Object {{ $_ -ne $dir -and $_.Trim() -ne '' }}
    $newP = $items -join ';'
    [Environment]::SetEnvironmentVariable('Path', $newP, 'User')
}}

# Broadcast setting change to notify active explorer and terminals
Add-Type -Namespace Win32 -Name Native -MemberDefinition '[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)] public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);' -ErrorAction SilentlyContinue
$result = [UIntPtr]::Zero
[Win32.Native]::SendMessageTimeout([IntPtr]0xffff, 0x001A, [UIntPtr]::Zero, "Environment", 2, 2000, [ref]$result) | Out-Null
"#,
        dir_str
    );

    run_powershell_script(&script);
}

fn create_shortcuts(exe_path: &Path, icon_path: &Path) {
    let exe_str = exe_path.display().to_string().replace('\'', "''");
    let work_dir = exe_path.parent().unwrap_or(exe_path).display().to_string().replace('\'', "''");
    let icon_str = icon_path.display().to_string().replace('\'', "''");

    let script = format!(
        r#"
$ws = New-Object -ComObject WScript.Shell

# Desktop shortcut
$desktop = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$s1 = $ws.CreateShortcut((Join-Path $desktop 'Entroly Desktop.lnk'))
$s1.TargetPath = '{0}'
$s1.Arguments = 'ui'
$s1.WorkingDirectory = '{1}'
$s1.IconLocation = '{2},0'
$s1.Description = 'Entroly Context Control Plane and Intelligence'
$s1.Save()

# Start menu shortcut
$sm = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Programs)
$s2 = $ws.CreateShortcut((Join-Path $sm 'Entroly Desktop.lnk'))
$s2.TargetPath = '{0}'
$s2.Arguments = 'ui'
$s2.WorkingDirectory = '{1}'
$s2.IconLocation = '{2},0'
$s2.Description = 'Entroly Context Control Plane and Intelligence'
$s2.Save()
"#,
        exe_str,
        work_dir,
        icon_str
    );

    run_powershell_script(&script);
}

fn remove_shortcuts() {
    let script = r#"
$desktop = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$dLink = Join-Path $desktop 'Entroly Desktop.lnk'
if (Test-Path $dLink) { Remove-Item $dLink -Force -ErrorAction SilentlyContinue }

$sm = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Programs)
$smLink = Join-Path $sm 'Entroly Desktop.lnk'
if (Test-Path $smLink) { Remove-Item $smLink -Force -ErrorAction SilentlyContinue }
"#;

    run_powershell_script(script);
}
