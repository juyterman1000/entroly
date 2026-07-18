[CmdletBinding()]
param(
    [string]$ModelDir = "$env:USERPROFILE\.entroly\models\qwen3-30b-a3b-thinking-2507",
    [string]$ConfigPath = "$env:USERPROFILE\.entroly\local_foundation.json",
    [int]$Port = 9378,
    [int]$ContextSize = 12288,
    [int]$GpuLayers = 0,
    [int]$Threads = 0,
    [switch]$SkipDownload,
    [switch]$Start
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ModelRepo = "bartowski/Qwen_Qwen3-30B-A3B-Thinking-2507-GGUF"
$ModelFile = "Qwen_Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf"
$ExpectedSha256 = "1359aa08e2f2dfe7ce4b5ff88c4c996e6494c9d916b1ebacd214bb74bbd5a9db"
$ModelPath = Join-Path $ModelDir $ModelFile

function Resolve-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        return @{
            Executable = $py.Source
            Prefix = @("-3")
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            Executable = $python.Source
            Prefix = @()
        }
    }

    throw "Python 3 is required. Install Python, then run this script again."
}

function Invoke-Python {
    param(
        [hashtable]$Python,
        [string[]]$Arguments
    )

    & $Python.Executable @($Python.Prefix) @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Resolve-LlamaServer {
    $commands = @("llama-server.exe", "llama-server")
    foreach ($name in $commands) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "llama.cpp is not installed and winget is unavailable."
    }

    Write-Host "Installing llama.cpp from WinGet..." -ForegroundColor Cyan
    & $winget.Source install llama.cpp `
        --accept-package-agreements `
        --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "WinGet could not install llama.cpp."
    }

    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    foreach ($name in $commands) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $wingetLink = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\llama-server.exe"
    if (Test-Path $wingetLink) {
        return (Resolve-Path $wingetLink).Path
    }

    throw "llama.cpp installed, but llama-server.exe was not found. Open a new terminal and rerun."
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $ConfigPath -Parent) | Out-Null

$python = Resolve-Python
$llamaServer = Resolve-LlamaServer

if (-not $SkipDownload) {
    Write-Host "Installing the one-time model downloader..." -ForegroundColor Cyan
    Invoke-Python -Python $python -Arguments @(
        "-m", "pip", "install", "--upgrade", "huggingface_hub"
    )

    $downloadScript = @'
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, destination = sys.argv[1:4]
Path(destination).mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=destination,
)
print(path)
'@

    $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) "entroly_download_qwen.py"
    Set-Content -Path $tempScript -Value $downloadScript -Encoding UTF8
    try {
        Write-Host "Downloading the Q4_K_M GGUF (~18.6 GB) once..." -ForegroundColor Cyan
        Invoke-Python -Python $python -Arguments @(
            $tempScript, $ModelRepo, $ModelFile, $ModelDir
        )
    }
    finally {
        Remove-Item $tempScript -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path $ModelPath)) {
    throw "Local model file was not found: $ModelPath"
}

Write-Host "Verifying the model SHA256..." -ForegroundColor Cyan
$actualSha256 = (Get-FileHash -Algorithm SHA256 -Path $ModelPath).Hash.ToLowerInvariant()
if ($actualSha256 -ne $ExpectedSha256) {
    throw "Model checksum mismatch. Expected $ExpectedSha256 but observed $actualSha256."
}

$moduleArgs = @(
    "-m", "entroly.local_foundation",
    "--config", $ConfigPath,
    "configure",
    "--model-path", $ModelPath,
    "--server-executable", $llamaServer,
    "--model-sha256", $ExpectedSha256,
    "--port", "$Port",
    "--context-size", "$ContextSize",
    "--gpu-layers", "$GpuLayers",
    "--threads", "$Threads",
    "--verify-hash"
)
Invoke-Python -Python $python -Arguments $moduleArgs

$launcherPath = Join-Path (Split-Path $ConfigPath -Parent) "start-qwen-strict-local.ps1"
$pythonPrefix = ($python.Prefix | ForEach-Object { '"' + $_.Replace('"', '""') + '"' }) -join " "
$launcher = @"
`$ErrorActionPreference = "Stop"
`$env:ENTROLY_LOCAL_ONLY = "1"
`$env:ENTROLY_DISABLE_UPDATE_CHECK = "1"
`$env:HF_HUB_OFFLINE = "1"
`$env:TRANSFORMERS_OFFLINE = "1"
`$env:NO_PROXY = "127.0.0.1,localhost,::1"
`$env:no_proxy = "127.0.0.1,localhost,::1"
& "$($python.Executable)" $pythonPrefix -m entroly.local_foundation --config "$ConfigPath" serve
"@
Set-Content -Path $launcherPath -Value $launcher -Encoding UTF8

Write-Host ""
Write-Host "Strict-local Qwen configuration completed." -ForegroundColor Green
Write-Host "Model:  $ModelPath"
Write-Host "Config: $ConfigPath"
Write-Host "Server: http://127.0.0.1:$Port/v1"
Write-Host "Start:  powershell -ExecutionPolicy Bypass -File `"$launcherPath`""
Write-Host ""
Write-Host "No cloud inference provider is configured. Runtime uses the local GGUF file only." -ForegroundColor Green

if ($Start) {
    & powershell.exe -ExecutionPolicy Bypass -File $launcherPath
}
