# Entroly Desktop Build Script
# Compiles entroly and EntrolySetup in release mode and copies artifacts to ui/dist/

$ErrorActionPreference = 'Stop'

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '       Entroly Desktop & Installer Release Builder          ' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$desktopDir = Join-Path $scriptDir 'desktop'
$distDir = Join-Path $scriptDir 'dist'

if (-not (Test-Path $distDir)) {
    New-Item -ItemType Directory -Path $distDir -Force | Out-Null
}

Write-Host '[1/3] Compiling entroly.exe (standalone dual CLI/GUI binary)...' -ForegroundColor Yellow
cargo build --release --bin entroly --manifest-path (Join-Path $desktopDir 'Cargo.toml')

Write-Host '[2/3] Compiling EntrolySetup.exe (standalone 1-click installer)...' -ForegroundColor Yellow
cargo build --release --bin EntrolySetup --manifest-path (Join-Path $desktopDir 'Cargo.toml')

Write-Host '[3/3] Synchronizing release binaries to ui/dist/...' -ForegroundColor Yellow
$entrolyBin = Join-Path $desktopDir 'target\release\entroly.exe'
$setupBin = Join-Path $desktopDir 'target\release\EntrolySetup.exe'

Copy-Item $entrolyBin (Join-Path $distDir 'entroly.exe') -Force
Copy-Item $setupBin (Join-Path $distDir 'EntrolySetup.exe') -Force

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '            [OK] Build Successfully Completed!              ' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Cyan
Get-ChildItem (Join-Path $distDir '*.exe') | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
