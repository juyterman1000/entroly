"""
Hook installer for Claude Code and other IDE integrations.

`entroly ravs hook install --claude-code` writes a PostToolUse hook
into Claude Code's settings that auto-feeds every Bash exit into RAVS.
Deterministic, no model compliance needed — the hook fires on every
tool use regardless of what the LLM decides to do.

Supported targets:
  - Claude Code: ~/.claude/settings.json (PostToolUse hook)
  - Generic: prints a shell snippet the user can paste into their workflow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ── Claude Code hook definition ──────────────────────────────────────

def _claude_code_hook_config() -> dict:
    """Generate the Claude Code PostToolUse hook configuration.

    The hook fires after every Bash tool use. It pipes the tool
    output to `entroly ravs capture --stdin` which classifies and
    records verifiable outcomes (tests, builds, lints, etc.).
    Non-verifiable commands are silently skipped.
    """
    # Resolve the entroly executable path
    entroly_bin = _find_entroly_bin()

    return {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{entroly_bin} ravs capture --stdin --quiet",
                        }
                    ],
                }
            ]
        }
    }


def _find_entroly_bin() -> str:
    """Find the entroly executable path."""
    import shutil

    # Try shutil.which first
    found = shutil.which("entroly")
    if found:
        return found

    # Try python -m entroly.cli as fallback
    return f"{sys.executable} -m entroly.cli"


def _claude_code_settings_path() -> Path:
    """Resolve Claude Code settings path (cross-platform)."""
    home = Path.home()
    return home / ".claude" / "settings.json"


def install_claude_code(dry_run: bool = False) -> dict:
    """Install the RAVS PostToolUse hook into Claude Code settings.

    Merges non-destructively: preserves existing settings and hooks,
    only adds the RAVS hook if it's not already present.

    Returns:
        dict with keys: path, action ("installed"|"already_present"|"dry_run"),
        and the full settings dict.
    """
    settings_path = _claude_code_settings_path()
    hook_config = _claude_code_hook_config()

    # Load existing settings
    existing = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            existing = {}

    # Check if our hook is already installed
    existing_hooks = existing.get("hooks", {})
    existing_post = existing_hooks.get("PostToolUse", [])

    already_present = any(
        "entroly ravs capture" in str(h.get("hooks", [{}]))
        for h in existing_post
    )

    if already_present:
        return {
            "path": str(settings_path),
            "action": "already_present",
            "settings": existing,
        }

    if dry_run:
        merged = _merge_settings(existing, hook_config)
        return {
            "path": str(settings_path),
            "action": "dry_run",
            "settings": merged,
        }

    # Merge and write
    merged = _merge_settings(existing, hook_config)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "path": str(settings_path),
        "action": "installed",
        "settings": merged,
    }


def _merge_settings(existing: dict, hook_config: dict) -> dict:
    """Non-destructively merge hook config into existing settings."""
    merged = dict(existing)

    if "hooks" not in merged:
        merged["hooks"] = {}

    if "PostToolUse" not in merged["hooks"]:
        merged["hooks"]["PostToolUse"] = []

    # Append our hook entries
    for entry in hook_config.get("hooks", {}).get("PostToolUse", []):
        merged["hooks"]["PostToolUse"].append(entry)

    return merged


def uninstall_claude_code() -> dict:
    """Remove the RAVS hook from Claude Code settings."""
    settings_path = _claude_code_settings_path()

    if not settings_path.exists():
        return {"path": str(settings_path), "action": "not_found"}

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return {"path": str(settings_path), "action": "parse_error"}

    post_hooks = settings.get("hooks", {}).get("PostToolUse", [])
    filtered = [
        h for h in post_hooks
        if "entroly ravs capture" not in str(h.get("hooks", [{}]))
    ]

    if len(filtered) == len(post_hooks):
        return {"path": str(settings_path), "action": "not_found"}

    settings["hooks"]["PostToolUse"] = filtered

    # Clean up empty structures
    if not settings["hooks"]["PostToolUse"]:
        del settings["hooks"]["PostToolUse"]
    if not settings["hooks"]:
        del settings["hooks"]

    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"path": str(settings_path), "action": "uninstalled"}


def print_generic_hook_snippet() -> None:
    """Print a generic shell snippet for non-Claude-Code workflows."""
    entroly_bin = _find_entroly_bin()
    print(f"""
  # Add this to your shell profile (.bashrc, .zshrc, etc.):

  entroly_capture() {{
    local cmd="$1"
    local exit_code=$?
    {entroly_bin} ravs capture \\
      --command "$cmd" \\
      --exit-code $exit_code \\
      --quiet 2>/dev/null &
  }}

  # Or for CI pipelines (GitHub Actions, etc.):
  # - run: pytest --json-report
  #   continue-on-error: true
  # - run: {entroly_bin} ravs capture --command "pytest" --exit-code ${{{{ job.status == 'success' && '0' || '1' }}}}
""")
