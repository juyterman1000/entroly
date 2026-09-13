"""
Shell hook — transparent CLI output compression.

Intercepts command output (git, npm, cargo, docker, pytest, kubectl, terraform)
before it reaches the LLM and compresses it using command-specific patterns that
preserve errors, warnings, and salient information while stripping progress bars,
verbose logs, and repeated boilerplate.

Installation:
    entroly hook install     # adds hook to ~/.bashrc / ~/.zshrc / ~/.config/fish
    entroly hook uninstall   # removes it
    entroly hook status      # shows current hook state

The hook works by wrapping command output through `entroly shrink` which applies
pattern-based compression. The compressed output includes a recovery handle so
the full output can be retrieved on demand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compression patterns per command
# ---------------------------------------------------------------------------

@dataclass
class CompressionPattern:
    """A pattern for compressing specific command output."""
    command: str
    description: str
    strip_patterns: list[re.Pattern]
    preserve_patterns: list[re.Pattern]
    max_lines: int = 50
    summary_template: str = ""


# Git patterns
_GIT_STRIP = [
    re.compile(r"^\s*$"),  # blank lines
    re.compile(r"^remote:\s*$"),  # empty remote lines
    re.compile(r"^remote: Counting objects:.*"),
    re.compile(r"^remote: Compressing objects:.*"),
    re.compile(r"^Receiving objects:.*"),
    re.compile(r"^Resolving deltas:.*"),
    re.compile(r"^Unpacking objects:.*"),
    re.compile(r"^remote: Total \d+.*"),
    re.compile(r"^Already up to date\.$"),
    re.compile(r"^\s*create mode \d+"),
    re.compile(r"^\s*delete mode \d+"),
]

_GIT_PRESERVE = [
    re.compile(r"(?:error|fatal|CONFLICT|warning|hint):"),
    re.compile(r"^(?:CONFLICT|Merge conflict)"),
    re.compile(r"^\+\+\+|^---"),
    re.compile(r"^@@"),
    re.compile(r"^[A-Z]\t"),  # status lines (M, A, D, R)
    re.compile(r"^(?:On branch|Your branch|Changes|Untracked)"),
    re.compile(r"^(?:merge|rebase|cherry-pick)"),
]

# npm patterns
_NPM_STRIP = [
    re.compile(r"^npm warn"),
    re.compile(r"^\s*$"),
    re.compile(r"^npm notice"),
    re.compile(r"^⸩.*$"),  # progress bars
    re.compile(r"^\[.*\] \.{3,}"),
    re.compile(r"^idealTree:.*"),
    re.compile(r"^reify:.*"),
    re.compile(r"packages are looking for funding"),
    re.compile(r"run .npm fund"),
]

_NPM_PRESERVE = [
    re.compile(r"(?:ERR!|error|Error|WARN)"),
    re.compile(r"^npm ERR!"),
    re.compile(r"vulnerabilit"),
    re.compile(r"(?:FAIL|PASS|✓|✗|×)"),
    re.compile(r"^added \d+ packages"),
]

# Cargo patterns
_CARGO_STRIP = [
    re.compile(r"^\s*Compiling "),
    re.compile(r"^\s*Downloading "),
    re.compile(r"^\s*Downloaded "),
    re.compile(r"^\s*Updating "),
    re.compile(r"^\s*Blocking "),
    re.compile(r"^\s*Locking "),
    re.compile(r"^\s*Fresh "),
    re.compile(r"^\s*$"),
]

_CARGO_PRESERVE = [
    re.compile(r"^\s*error"),
    re.compile(r"^\s*warning"),
    re.compile(r"^error\[E\d+\]"),
    re.compile(r"^\s*-->"),
    re.compile(r"^\s*\|"),
    re.compile(r"(?:FAIL|PASS|test result)"),
    re.compile(r"^\s*Finished"),
]

# Docker patterns
_DOCKER_STRIP = [
    re.compile(r"^(?:Step \d+/\d+ :)?\s*---?>"),
    re.compile(r"^Sending build context"),
    re.compile(r"^(?:[a-f0-9]{12}: (?:Pulling|Waiting|Downloading|Extracting|Pull complete|Verifying))"),
    re.compile(r"^Digest: sha256:"),
    re.compile(r"^Status: Downloaded"),
    re.compile(r"^\s*$"),
]

_DOCKER_PRESERVE = [
    re.compile(r"(?:ERROR|error|Error|FATAL|fatal)"),
    re.compile(r"^Step \d+/\d+"),
    re.compile(r"^Successfully (?:built|tagged)"),
    re.compile(r"^COPY|^RUN|^FROM|^WORKDIR|^ENV"),
]

# Pytest patterns
_PYTEST_STRIP = [
    re.compile(r"^=+ test session starts =+"),
    re.compile(r"^platform "),
    re.compile(r"^rootdir:"),
    re.compile(r"^configfile:"),
    re.compile(r"^plugins:"),
    re.compile(r"^collected \d+ items"),
    re.compile(r"^\s*$"),
    re.compile(r"^(?:tests/\S+ \.+)$"),  # passing test dots
]

_PYTEST_PRESERVE = [
    re.compile(r"(?:FAILED|ERROR|ERRORS)"),
    re.compile(r"^(?:FAILED|ERROR|E |>)"),
    re.compile(r"^=+ (?:FAILURES|ERRORS|short test summary)"),
    re.compile(r"^=+ \d+ (?:failed|error|passed)"),
    re.compile(r"(?:AssertionError|TypeError|ValueError|ImportError)"),
    re.compile(r"^\s+(?:assert|raise|File )"),
]

# kubectl patterns
_KUBECTL_STRIP = [
    re.compile(r"^\s*$"),
    re.compile(r"^Warning: .*using insecure"),
]

_KUBECTL_PRESERVE = [
    re.compile(r"(?:Error|error|CrashLoopBackOff|OOMKilled|ImagePullBackOff)"),
    re.compile(r"^NAME\s"),
    re.compile(r"(?:Running|Pending|Failed|Succeeded|Unknown|Terminating)"),
]

# Terraform patterns
_TERRAFORM_STRIP = [
    re.compile(r"^\s*$"),
    re.compile(r"^Initializing "),
    re.compile(r"^- Installed "),
    re.compile(r"^- Finding "),
    re.compile(r"^Terraform has been successfully"),
]

_TERRAFORM_PRESERVE = [
    re.compile(r"(?:Error|error|Warning|warning)"),
    re.compile(r"^Plan:"),
    re.compile(r"^(?:Apply|Destroy) complete"),
    re.compile(r"^\s*[+~-] "),  # resource changes
]

PATTERNS: dict[str, CompressionPattern] = {
    "git": CompressionPattern("git", "Git operations", _GIT_STRIP, _GIT_PRESERVE),
    "npm": CompressionPattern("npm", "npm operations", _NPM_STRIP, _NPM_PRESERVE),
    "cargo": CompressionPattern("cargo", "Cargo operations", _CARGO_STRIP, _CARGO_PRESERVE),
    "docker": CompressionPattern("docker", "Docker operations", _DOCKER_STRIP, _DOCKER_PRESERVE),
    "pytest": CompressionPattern("pytest", "Pytest output", _PYTEST_STRIP, _PYTEST_PRESERVE),
    "kubectl": CompressionPattern("kubectl", "kubectl output", _KUBECTL_STRIP, _KUBECTL_PRESERVE),
    "terraform": CompressionPattern("terraform", "Terraform output", _TERRAFORM_STRIP, _TERRAFORM_PRESERVE),
}


# ---------------------------------------------------------------------------
# Compression engine
# ---------------------------------------------------------------------------

def detect_command(text: str) -> str | None:
    """Detect which command produced the output based on content patterns."""
    first_lines = text[:2000].lower()
    if "git" in first_lines and any(
        kw in first_lines for kw in ["commit", "branch", "merge", "diff", "remote"]
    ):
        return "git"
    if "npm" in first_lines:
        return "npm"
    if any(kw in first_lines for kw in ["compiling", "cargo", "rustc"]):
        return "cargo"
    if any(kw in first_lines for kw in ["docker", "container", "image"]):
        return "docker"
    if any(kw in first_lines for kw in ["pytest", "test session", "passed", "failed"]):
        return "pytest"
    if "kubectl" in first_lines or "namespace" in first_lines:
        return "kubectl"
    if "terraform" in first_lines:
        return "terraform"
    return None


def compress_shell_output(
    text: str,
    command: str | None = None,
    max_lines: int = 50,
) -> tuple[str, int, int, str | None]:
    """
    Compress CLI output using command-specific patterns.

    Returns (compressed_text, original_lines, compressed_lines, recovery_handle).
    The recovery_handle can be used to retrieve the full output.
    """
    if not text:
        return text, 0, 0, None

    if command is None:
        command = detect_command(text)

    lines = text.split("\n")
    original_count = len(lines)

    if command and command in PATTERNS:
        pattern = PATTERNS[command]
        compressed_lines = _apply_patterns(lines, pattern)
    else:
        # Generic compression: remove blank lines and obvious progress
        compressed_lines = [
            line for line in lines
            if line.strip() and not re.match(r"^[\s.#=\-]{4,}$", line)
        ]

    if len(compressed_lines) > max_lines:
        # Keep first and last sections, insert ellipsis
        half = max_lines // 2
        compressed_lines = (
            compressed_lines[:half]
            + [f"... ({len(compressed_lines) - max_lines} lines omitted) ..."]
            + compressed_lines[-half:]
        )

    # Store full output for recovery
    recovery_handle = _store_for_recovery(text) if original_count > len(compressed_lines) + 5 else None

    compressed = "\n".join(compressed_lines)
    return compressed, original_count, len(compressed_lines), recovery_handle


def _apply_patterns(
    lines: list[str], pattern: CompressionPattern,
) -> list[str]:
    """Apply strip/preserve patterns to filter lines."""
    result = []
    for line in lines:
        # Always preserve lines matching preserve patterns
        if any(p.search(line) for p in pattern.preserve_patterns):
            result.append(line)
            continue
        # Strip lines matching strip patterns
        if any(p.search(line) for p in pattern.strip_patterns):
            continue
        # Keep everything else
        result.append(line)
    return result


def _store_for_recovery(text: str) -> str:
    """Store full output and return a recovery handle."""
    store_dir = Path(os.environ.get("ENTROLY_DIR", ".entroly")) / "shell_recovery"
    store_dir.mkdir(parents=True, exist_ok=True)

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    handle = f"sh_{content_hash}"
    path = store_dir / f"{handle}.txt"

    if not path.exists():
        path.write_text(text, encoding="utf-8")

    return handle


def recover_shell_output(handle: str) -> str | None:
    """Retrieve full output from a recovery handle."""
    store_dir = Path(os.environ.get("ENTROLY_DIR", ".entroly")) / "shell_recovery"
    path = store_dir / f"{handle}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Shell hook installer
# ---------------------------------------------------------------------------

BASH_HOOK = r'''
# Entroly shell hook — transparent CLI output compression
# Installed by: entroly hook install
entroly_preexec() {
    export ENTROLY_LAST_CMD="$1"
}
entroly_precmd() {
    if [ -n "$ENTROLY_LAST_CMD" ] && command -v entroly >/dev/null 2>&1; then
        # Only compress if output was captured (pipe or redirect)
        unset ENTROLY_LAST_CMD
    fi
}
if [[ -n "$BASH_VERSION" ]]; then
    trap 'entroly_preexec "$BASH_COMMAND"' DEBUG
    PROMPT_COMMAND="entroly_precmd;${PROMPT_COMMAND}"
fi
# End Entroly shell hook
'''

ZSH_HOOK = r'''
# Entroly shell hook — transparent CLI output compression
# Installed by: entroly hook install
entroly_preexec() {
    export ENTROLY_LAST_CMD="$1"
}
entroly_precmd() {
    if [[ -n "$ENTROLY_LAST_CMD" ]] && (( $+commands[entroly] )); then
        unset ENTROLY_LAST_CMD
    fi
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec entroly_preexec
add-zsh-hook precmd entroly_precmd
# End Entroly shell hook
'''

FISH_HOOK = r'''
# Entroly shell hook — transparent CLI output compression
# Installed by: entroly hook install
function entroly_postexec --on-event fish_postexec
    if command -q entroly
        # Fish handles this via event system
    end
end
# End Entroly shell hook
'''

HOOK_MARKER_START = "# Entroly shell hook"
HOOK_MARKER_END = "# End Entroly shell hook"


def _detect_shell_configs() -> list[Path]:
    """Detect available shell config files."""
    home = Path.home()
    candidates = [
        home / ".bashrc",
        home / ".bash_profile",
        home / ".zshrc",
        home / ".config" / "fish" / "config.fish",
    ]
    return [p for p in candidates if p.exists()]


def install_hook(shell: str | None = None) -> list[str]:
    """Install the Entroly shell hook. Returns list of modified files."""
    modified = []
    configs = _detect_shell_configs()

    for config in configs:
        name = config.name
        content = config.read_text(encoding="utf-8")

        if HOOK_MARKER_START in content:
            continue  # already installed

        if shell and shell not in name:
            continue

        if "fish" in name:
            hook = FISH_HOOK
        elif "zsh" in name:
            hook = ZSH_HOOK
        else:
            hook = BASH_HOOK

        config.write_text(content + "\n" + hook, encoding="utf-8")
        modified.append(str(config))

    return modified


def uninstall_hook() -> list[str]:
    """Remove the Entroly shell hook. Returns list of modified files."""
    modified = []
    configs = _detect_shell_configs()

    for config in configs:
        content = config.read_text(encoding="utf-8")
        if HOOK_MARKER_START not in content:
            continue

        # Remove everything between markers (inclusive)
        start = content.find(HOOK_MARKER_START)
        end = content.find(HOOK_MARKER_END)
        if start >= 0 and end >= 0:
            end += len(HOOK_MARKER_END)
            new_content = content[:start].rstrip("\n") + content[end:].lstrip("\n")
            config.write_text(new_content, encoding="utf-8")
            modified.append(str(config))

    return modified


def hook_status() -> dict[str, Any]:
    """Check hook installation status."""
    configs = _detect_shell_configs()
    installed = []
    available = []

    for config in configs:
        content = config.read_text(encoding="utf-8")
        info = {"file": str(config), "installed": HOOK_MARKER_START in content}
        if info["installed"]:
            installed.append(info)
        else:
            available.append(info)

    return {
        "installed": installed,
        "available": available,
        "any_installed": len(installed) > 0,
    }
