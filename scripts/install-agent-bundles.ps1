[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet("install", "status", "uninstall")]
    [string]$Action = "install",
    [ValidateSet("all", "codex", "claude", "gemini")]
    [string]$Agent = "all",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$userRoot = [Environment]::GetFolderPath("UserProfile")
$codexRoot = if ($env:CODEX_HOME) { [IO.Path]::GetFullPath($env:CODEX_HOME) } else { Join-Path $userRoot ".codex" }

if (-not (Get-Command entroly -ErrorAction SilentlyContinue)) {
    throw "The 'entroly' executable is not on PATH. Install Entroly before installing agent bundles."
}

$targets = @(
    [pscustomobject]@{
        Agent = "codex"
        Source = Join-Path $repoRoot "integrations/codex/entroly/skills/entroly-evidence-operations"
        Destination = Join-Path $codexRoot "skills/entroly-evidence-operations"
    },
    [pscustomobject]@{
        Agent = "claude"
        Source = Join-Path $repoRoot "skills/entroly-evidence-operations"
        Destination = Join-Path $userRoot ".claude/skills/entroly-evidence-operations"
    },
    [pscustomobject]@{
        Agent = "gemini"
        Source = Join-Path $repoRoot "integrations/gemini/entroly"
        Destination = Join-Path $userRoot ".gemini/extensions/entroly"
    }
) | Where-Object { $Agent -eq "all" -or $_.Agent -eq $Agent }

function Assert-SafeDestination {
    param([string]$Destination)
    $resolved = [IO.Path]::GetFullPath($Destination)
    $allowed = [IO.Path]::GetFullPath($userRoot).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing destination outside the user profile: $resolved"
    }
    return $resolved
}

function Test-EntrolyBundle {
    param([string]$Path)
    $marker = Join-Path $Path "entroly-bundle.json"
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { return $false }
    try { return (Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json).id -eq "entroly" }
    catch { return $false }
}

foreach ($target in $targets) {
    $destination = Assert-SafeDestination $target.Destination
    if ($Action -eq "status") {
        $state = if (Test-EntrolyBundle $destination) { "installed" } elseif (Test-Path -LiteralPath $destination) { "occupied-by-other-content" } else { "not-installed" }
        Write-Output "$($target.Agent): $state ($destination)"
        continue
    }

    if ($Action -eq "uninstall") {
        if (-not (Test-Path -LiteralPath $destination)) {
            Write-Output "$($target.Agent): not installed"
            continue
        }
        if (-not (Test-EntrolyBundle $destination)) {
            throw "Refusing to move unrecognized content at $destination"
        }
        $disabled = "$destination.entroly-disabled-$(Get-Date -Format 'yyyyMMddHHmmss')"
        if ($PSCmdlet.ShouldProcess($destination, "Move Entroly bundle to $disabled")) {
            Move-Item -LiteralPath $destination -Destination $disabled
            Write-Output "$($target.Agent): disabled; recoverable at $disabled"
        }
        continue
    }

    $source = (Resolve-Path -LiteralPath $target.Source).Path
    if (-not (Test-EntrolyBundle $source)) {
        throw "Invalid Entroly bundle source: $source"
    }
    if (Test-Path -LiteralPath $destination) {
        if (-not $Force) {
            throw "Destination exists: $destination. Re-run with -Force for a timestamped backup."
        }
        $backup = "$destination.entroly-backup-$(Get-Date -Format 'yyyyMMddHHmmss')"
        if ($PSCmdlet.ShouldProcess($destination, "Move existing directory to $backup")) {
            Move-Item -LiteralPath $destination -Destination $backup
            Write-Output "$($target.Agent): backed up existing directory to $backup"
        }
    }
    $parent = Split-Path -Parent $destination
    if ($PSCmdlet.ShouldProcess($destination, "Install Entroly bundle")) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Recurse
        Write-Output "$($target.Agent): installed at $destination"
    }
}

if ($Action -eq "install") {
    Write-Output "Restart the selected agent if it does not reload skills or extensions live."
}
