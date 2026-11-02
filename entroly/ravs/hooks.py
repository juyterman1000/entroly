"""
Hook installer for Claude Code and other IDE integrations.

`entroly ravs hook install --claude-code` writes PostToolUse and
PostToolUseFailure hooks into Claude Code's settings that auto-feed every
Bash exit into RAVS; `entroly ravs hook uninstall --claude-code` removes
them again. Deterministic, no model compliance needed — the hooks fire on
every tool use regardless of what the LLM decides to do.

Both events are needed: Claude Code fires PostToolUse only after exit 0 and
PostToolUseFailure after a non-zero exit (see `entroly.hook_outcome`).

Supported targets:
  - Claude Code: ~/.claude/settings.json (PostToolUse + PostToolUseFailure)
  - Generic: prints a shell snippet the user can paste into their workflow
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path


# ── Claude Code hook definition ──────────────────────────────────────

# The events the capture hook runs on, and the flag that lets capture read
# PostToolUse as exit 0. The flag is only ever written together with the
# failure registration: a hook that sees successes but no failures must not
# record either (see entroly.hook_outcome.exit_code).
_CAPTURE_EVENTS = ("PostToolUse", "PostToolUseFailure")
_HOST_FLAG = "--host claude-code"


def _capture_command() -> str:
    return f"{_find_entroly_bin()} ravs capture --stdin {_HOST_FLAG} --quiet"


def _claude_code_hook_config() -> dict:
    """Generate the Claude Code hook configuration.

    The hooks fire after every Bash tool use, successful or not. They pipe
    the payload to `entroly ravs capture --stdin` which classifies and
    records verifiable outcomes (tests, builds, lints, etc.).
    Non-verifiable commands are silently skipped.
    """
    command = _capture_command()
    return {
        "hooks": {
            event: [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": command}],
                }
            ]
            for event in _CAPTURE_EVENTS
        }
    }


def _hook_word(path: str, *, windows: bool | None = None) -> str:
    """Write ``path`` as one word for the shell that runs the hook.

    Claude Code runs hook commands with bash, Git Bash on Windows. Unquoted,
    a path with a space split into words ("/home/u/my: No such file or
    directory" on every Bash call), and bash removed every backslash, so
    ``C:\\Python312\\Scripts\\entroly.exe`` ran as a command that does not
    exist. Git Bash accepts forward slashes in Windows paths; quoting is added
    only when a word needs it, so existing POSIX registrations stay identical.
    """
    if windows is None:
        windows = os.name == "nt"
    if windows:
        path = path.replace("\\", "/")
    return shlex.quote(path)


def _find_entroly_bin() -> str:
    """Find the entroly executable path, as a word bash runs intact."""
    import shutil

    # Try shutil.which first
    found = shutil.which("entroly")
    if found:
        return _hook_word(found)

    # Try python -m entroly.cli as fallback
    return f"{_hook_word(sys.executable)} -m entroly.cli"


def _claude_code_settings_path() -> Path:
    """Resolve Claude Code settings path (cross-platform)."""
    home = Path.home()
    return home / ".claude" / "settings.json"


def _is_capture_registration(entry: object) -> bool:
    """True for a PostToolUse entry that runs ``ravs capture``.

    Must recognise every command this module writes. Matching the literal
    ``entroly ravs capture`` missed the ``<python> -m entroly.cli ravs capture``
    fallback that `_find_entroly_bin` returns when no ``entroly`` executable is
    on PATH, so each re-run appended another registration and the host
    recorded every Bash call once per copy. Same test as
    `_project_capture_scope` uses for a project registration.
    """
    return "ravs capture" in str(entry)


def _project_capture_scope() -> Path | None:
    """Return the project settings file already registering capture, if any.

    Only the current working directory is consulted. Walking upwards would let
    an unrelated ancestor checkout suppress a hook the user asked for, and the
    host resolves project settings from the workspace root it was started in.
    """
    for name in ("settings.json", "settings.local.json"):
        candidate = Path.cwd() / ".claude" / name
        try:
            if not candidate.is_file():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        hooks = payload.get("hooks")
        if not isinstance(hooks, dict):
            continue
        for entry in hooks.get("PostToolUse") or []:
            if "ravs capture" in str(entry):
                return candidate
    return None


def install_claude_code(dry_run: bool = False) -> dict:
    """Install the RAVS capture hooks into Claude Code settings.

    Merges non-destructively: preserves existing settings and hooks, adds
    the hooks if absent, and brings an older registration up to date in
    place (the failure event and the host flag) rather than adding a second.

    Returns:
        dict with keys: path, action ("installed"|"updated"|"already_present"|
        "already_present_in_project"|"unreadable"|"dry_run"), and the full
        settings dict.
    """
    settings_path = _claude_code_settings_path()
    hook_config = _claude_code_hook_config()

    # Load existing settings. A file that cannot be parsed used to be treated
    # as empty and then overwritten with only this hook, silently deleting
    # every permission, deny rule, hook and env entry it held -- with no
    # backup, while the CLI printed "Installed". A UTF-8 BOM (written by
    # Windows PowerShell 5.1 and Notepad), a trailing comma or a comment was
    # enough. Read the BOM as encoding, and refuse anything else unparseable.
    existing: dict = {}
    problem = None
    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problem = f"not valid JSON ({type(exc).__name__}: {exc})"
        else:
            if isinstance(loaded, dict):
                existing = loaded
            else:
                problem = "not a JSON object"

    # Check if our hook is already installed
    existing_hooks = existing.get("hooks", {})
    sections = {}
    if problem is None and not isinstance(existing_hooks, dict):
        problem = 'not mergeable ("hooks" is not an object)'
    if problem is None:
        for event in _CAPTURE_EVENTS:
            section = existing_hooks.get(event, [])
            if not isinstance(section, list):
                problem = f'not mergeable ("hooks.{event}" is not a list)'
                break
            sections[event] = section
    if problem is not None:
        return {
            "path": str(settings_path),
            "action": "unreadable",
            "detail": problem,
            "settings": None,
        }

    registered = {
        event: [entry for entry in sections[event] if _is_capture_registration(entry)]
        for event in _CAPTURE_EVENTS
    }
    if not any(registered.values()):
        # A project that ships its own capture hook already covers these
        # calls, and the host fans one tool call out to *every* matching
        # registration. Adding a second one here is not harmless duplication:
        # measured on a real session, a user-scope hook matching "Bash"
        # alongside a project hook matching "Bash|Read|Edit|..." recorded
        # every Bash call twice -- 12 byte-identical journal lines out of 84
        # -- and double-counted Bash in the tool-use counters that justify a
        # schema allowlist. Checking only the scope being written is what let
        # that happen.
        project_scope = _project_capture_scope()
        if project_scope is not None:
            return {
                "path": str(settings_path),
                "action": "already_present_in_project",
                "project_path": str(project_scope),
                "settings": existing,
            }
        merged = _merge_settings(existing, hook_config)
        outcome = {"path": str(settings_path), "action": "installed", "settings": merged}
    else:
        merged = _upgrade_registration(existing, registered)
        if merged is None:
            return {
                "path": str(settings_path),
                "action": "already_present",
                "settings": existing,
            }
        outcome = {"path": str(settings_path), "action": "updated", "settings": merged}

    if dry_run:
        return {
            "path": str(settings_path),
            "action": "dry_run",
            "settings": merged,
        }

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return outcome


def _capture_hooks(entry: object) -> list[dict]:
    """The individual hook objects in ``entry`` that run ``ravs capture``."""
    if not isinstance(entry, dict) or not isinstance(entry.get("hooks"), list):
        return []
    return [
        hook
        for hook in entry["hooks"]
        if isinstance(hook, dict)
        and isinstance(hook.get("command"), str)
        and "ravs capture" in hook["command"]
    ]


def _with_host_flag(command: str) -> str:
    """Add ``--host claude-code`` right after ``--stdin``, keeping any tail.

    A hand-written registration may end in ``2>/dev/null || true``; the flag
    has to stay an argument of capture, not land after the shell operators.
    """
    if _HOST_FLAG in command:
        return command
    if "--stdin" in command:
        return command.replace("--stdin", f"--stdin {_HOST_FLAG}", 1)
    return f"{command} {_HOST_FLAG}"


def _upgrade_registration(existing: dict, registered: dict) -> dict | None:
    """Bring an existing registration up to date, or None if it already is.

    Edits only the capture hook objects themselves: the user's matcher and
    any sibling hooks in the same entry are kept, and the failure event is
    added with the same matcher and command as the registration it pairs
    with, so both events cover the same tools.
    """
    merged = json.loads(json.dumps(existing))
    hooks = merged["hooks"]
    changed = False
    pairs_with: dict | None = None
    for event in _CAPTURE_EVENTS:
        for entry in hooks.get(event, []):
            if not _is_capture_registration(entry):
                continue
            for hook in _capture_hooks(entry):
                upgraded = _with_host_flag(hook["command"])
                if upgraded != hook["command"]:
                    hook["command"] = upgraded
                    changed = True
                if pairs_with is None:
                    pairs_with = {
                        "matcher": entry.get("matcher", "Bash"),
                        "command": hook["command"],
                    }
    for event in _CAPTURE_EVENTS:
        if registered[event]:
            continue
        template = pairs_with or {"matcher": "Bash", "command": _capture_command()}
        hooks.setdefault(event, []).append(
            {
                "matcher": template["matcher"],
                "hooks": [{"type": "command", "command": template["command"]}],
            }
        )
        changed = True
    return merged if changed else None


def _merge_settings(existing: dict, hook_config: dict) -> dict:
    """Non-destructively merge hook config into existing settings."""
    merged = dict(existing)

    if "hooks" not in merged:
        merged["hooks"] = {}

    # Append our hook entries
    for event, entries in hook_config.get("hooks", {}).items():
        merged["hooks"].setdefault(event, [])
        for entry in entries:
            merged["hooks"][event].append(entry)

    return merged


def capture_registration_path() -> Path | None:
    """The user settings file when it registers the capture hook, else None.

    Read-only, for callers that must say what they are about to remove.
    """
    settings_path = _claude_code_settings_path()
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    if not isinstance(hooks, dict):
        return None
    for event in _CAPTURE_EVENTS:
        section = hooks.get(event)
        if isinstance(section, list) and any(
            _is_capture_registration(entry) for entry in section
        ):
            return settings_path
    return None


def uninstall_claude_code() -> dict:
    """Remove the RAVS hooks from Claude Code settings."""
    settings_path = _claude_code_settings_path()

    if not settings_path.exists():
        return {"path": str(settings_path), "action": "not_found"}

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError, IOError):
        return {"path": str(settings_path), "action": "parse_error"}
    hooks = settings.get("hooks", {}) if isinstance(settings, dict) else None
    if not isinstance(hooks, dict) or not all(
        isinstance(hooks.get(event, []), list) for event in _CAPTURE_EVENTS
    ):
        return {"path": str(settings_path), "action": "parse_error"}

    removed = False
    for event in _CAPTURE_EVENTS:
        if event not in hooks:
            continue
        kept = [h for h in hooks[event] if not _is_capture_registration(h)]
        if len(kept) == len(hooks[event]):
            continue
        removed = True
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]

    if not removed:
        return {"path": str(settings_path), "action": "not_found"}

    # Clean up empty structures
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
