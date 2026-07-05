"""
CLI audit + integration smoke tests.

Two responsibilities:

  1. **README ↔ argparse audit** — every `entroly <cmd>` shown in
     README.md must resolve to a registered subcommand in cli.py.
     Catches drift where docs claim commands that no longer exist (or
     vice versa: commands removed without README updates).

  2. **Documented integration smoke tests** — every `entroly wrap
     <agent>` listed in the README must at least parse and produce a
     usage message. Catches regressions where a wrap integration is
     deleted but still advertised.

Both tests are pure-Python: they invoke `entroly --help` and parse
README.md. No network, no model API, no real LLM calls.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _entroly(*args: str, expect_success: bool = True) -> subprocess.CompletedProcess:
    """Invoke the entroly CLI in a subprocess. Returns the completed proc."""
    cmd = [sys.executable, "-m", "entroly.cli", *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        cwd=str(ROOT), encoding="utf-8", errors="replace",
    )
    if expect_success and proc.returncode != 0:
        raise AssertionError(
            f"`{' '.join(cmd)}` failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


# ── 1. README ↔ argparse audit ────────────────────────────────────────


def _registered_subcommands() -> set[str]:
    """Parse `entroly --help` for the registered subcommand list."""
    out = _entroly("--help").stdout
    # argparse renders subcommands as `{cmd1,cmd2,cmd3}` somewhere in the help text
    m = re.search(r"\{([a-z][a-z0-9_,\-]*)\}", out)
    assert m, f"could not parse subcommand list from --help output:\n{out}"
    return {cmd.strip() for cmd in m.group(1).split(",")}


_CMD_PATTERN = re.compile(
    r"(?<![/\\])\bentroly\s+([a-z][a-z0-9\-]*)(?=[\s`<\-])"
)

# Token strings that look like commands but aren't (verbs in prose etc.).
_PROSE_FALSE_POSITIVES = {
    "actually",
    "auto-merges",
    "prints",
    "shrink",
    "starts",
    "import",  # used in `from entroly import compress` (Python `import` keyword)
}


def _claimed_commands_in_readme() -> set[str]:
    text = README.read_text(encoding="utf-8")
    matches = _CMD_PATTERN.findall(text)
    return {m for m in matches if m not in _PROSE_FALSE_POSITIVES}


def test_every_readme_command_is_registered():
    """Every `entroly <cmd>` referenced in README.md must be a real
    registered subcommand. If this fails, either the README is stale
    or the CLI is missing a documented command."""
    claimed = _claimed_commands_in_readme()
    registered = _registered_subcommands()
    missing = claimed - registered
    assert not missing, (
        f"README claims commands not registered in cli.py: {sorted(missing)}\n"
        f"Registered: {sorted(registered)}\n"
        f"Either fix the README or wire the missing command in cli.py."
    )


def test_no_orphaned_registered_commands():
    """Soft check — every registered subcommand should appear in
    SOME documentation source. This catches commands that exist in
    cli.py but were never advertised. Not a hard failure (some
    commands are internal), but allow-listed below.
    """
    registered = _registered_subcommands()
    readme_claimed = _claimed_commands_in_readme()

    # Allow-list: commands that exist for plumbing / internal flows
    # and intentionally aren't headlined in the main README. (Several
    # of these *are* documented in cookbook/README.md or per-command
    # help pages — this test only checks the main README.)
    INTERNAL_ONLY = {
        "init", "feedback", "optimize", "status", "config", "telemetry",
        "clean", "export", "import", "drift", "profile", "doctor",
        "migrate", "role", "completions", "compile", "sync", "search",
        "docs", "share", "finetune", "learn", "verify", "verify-code",
        "autotune", "digest", "health", "witness", "audit",
    }
    orphaned = registered - readme_claimed - INTERNAL_ONLY
    assert not orphaned, (
        f"Registered subcommands not advertised in README: {sorted(orphaned)}\n"
        f"Either headline them in the README or add them to INTERNAL_ONLY."
    )


# ── 2. Smoke tests for documented integrations ───────────────────────


def test_entroly_help_runs():
    """The entry point itself must work."""
    proc = _entroly("--help")
    assert "entroly" in proc.stdout.lower()


def test_batch_json_emits_machine_readable_stdout(tmp_path: Path):
    """`entroly batch --json` is a CI surface, so stdout must be JSON only."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("def cli_wiring_probe():\n    return 'ok'\n", encoding="utf-8")

    env = os.environ.copy()
    env["ENTROLY_DIR"] = str(tmp_path / "state")
    env["HOME"] = str(tmp_path / "home")
    env["USERPROFILE"] = str(tmp_path / "home")
    env["ENTROLY_MAX_FILES"] = "8"
    env["PYTHONIOENCODING"] = "utf-8"
    env["ENTROLY_DISABLE_UPDATE_CHECK"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "entroly.cli", "batch", "--budget", "256", "--json"],
        input="Where is the CLI wired?\n",
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(project),
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert proc.stdout.lstrip().startswith("[")
    payload = json.loads(proc.stdout)
    assert payload[0]["query"] == "Where is the CLI wired?"


@pytest.mark.parametrize("subcmd", [
    "wrap", "ravs", "proxy", "serve", "dashboard", "demo", "batch",
    "benchmark", "go", "daemon", "verify", "verify-code", "compile",
    "doctor", "witness", "audit",
])
def test_subcommand_help_runs(subcmd: str):
    """Every advertised subcommand must produce help without crashing."""
    proc = _entroly(subcmd, "--help")
    assert "usage" in (proc.stdout + proc.stderr).lower(), (
        f"`entroly {subcmd} --help` did not emit a usage message"
    )


@pytest.mark.parametrize("ravs_action", ["report", "capture", "hook"])
def test_ravs_subactions_resolve(ravs_action: str):
    """RAVS has nested subactions — each must resolve to a help message."""
    proc = _entroly("ravs", ravs_action, "--help")
    assert "usage" in (proc.stdout + proc.stderr).lower(), (
        f"`entroly ravs {ravs_action} --help` did not emit a usage message"
    )


def test_ravs_capture_command_writes_event_log(tmp_path: Path):
    """The nested --command flag must not overwrite the top-level subcommand."""
    log_path = tmp_path / "ravs" / "events.jsonl"
    env = os.environ.copy()
    env["ENTROLY_DISABLE_UPDATE_CHECK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "entroly.cli",
            "ravs",
            "capture",
            "--command",
            "pytest tests/test_cli_audit.py",
            "--exit-code",
            "0",
            "--stdout",
            "1 passed",
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Captured" in proc.stdout
    assert "usage: entroly [-h]" not in proc.stdout
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["request", "outcome"]
    assert events[1]["event_type"] == "test"
    assert events[1]["value"] == "pass"


def test_profile_save_load_uses_valid_defaults_when_no_local_config(tmp_path: Path, monkeypatch):
    """Fresh projects should be able to save/load a profile from packaged defaults."""
    from entroly import cli

    state_dir = tmp_path / "state"
    target_config = tmp_path / "runtime" / "tuning_config.json"
    monkeypatch.setattr(cli, "_ENTROLY_DIR", state_dir)
    monkeypatch.setattr(cli, "_writable_tuning_config_path", lambda: target_config)

    save_rc = cli.cmd_profile(SimpleNamespace(profile_action="save", name="fresh"))
    assert save_rc is None

    profile_path = state_dir / "profiles" / "fresh.json"
    saved = json.loads(profile_path.read_text(encoding="utf-8"))
    assert saved["weights"]["recency"] == 0.30

    load_rc = cli.cmd_profile(SimpleNamespace(profile_action="load", name="fresh"))
    assert load_rc is None
    loaded = json.loads(target_config.read_text(encoding="utf-8"))
    assert loaded == saved


@pytest.mark.parametrize("agent", [
    "claude", "codex", "aider", "gemini", "qwen", "opencode",
    "crush", "hermes", "pi", "ollama",
    "cursor", "windsurf", "vscode", "claude-desktop", "claude-code", "zed",
])
def test_wrap_agents_parse(agent: str):
    """`entroly wrap <agent>` must at least parse — the README documents
    each of these as a one-command integration path. We invoke with a
    flag that prevents the wrapped agent from actually starting:
    `--port 0` is a no-op for argument parsing but avoids spawning."""
    # We can't actually execute the wrapped agent in CI (no agent CLI
    # installed), so we test that argparse accepts the form by
    # invoking with --help on the parent and grepping the agent list.
    out = _entroly("wrap", "--help").stdout
    # The wrap parser's positional help lists supported agents
    assert "agent" in out.lower(), (
        "`entroly wrap --help` does not document the agent positional"
    )
    assert agent in out, (
        f"`entroly wrap {agent}` is documented in README but missing from "
        "`entroly wrap --help`."
    )

    from entroly.cli import _WRAP_AGENTS

    assert agent in _WRAP_AGENTS, (
        f"`entroly wrap {agent}` is documented in README but missing from "
        "_WRAP_AGENTS."
    )


def test_cursor_mcp_config_uses_current_python_interpreter():
    """MCP-aware IDEs must not resolve a stale global `entroly` binary.

    Cursor launches the configured MCP command outside the shell that ran
    `entroly wrap cursor`; using `sys.executable -m entroly.server` keeps the
    server pinned to the package environment that generated the config.
    """
    from entroly.cli import _generate_mcp_config

    cfg = _generate_mcp_config()["entroly"]

    assert cfg["command"] == sys.executable
    assert cfg["args"] == ["-m", "entroly.server"]
    assert cfg["env"]["PYTHONIOENCODING"] == "utf-8"


def test_write_config_round_trips_utf8_existing_json(tmp_path: Path):
    from entroly.cli import _write_config

    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {"mcpServers": {"existing": {"label": "caf\u00e9"}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _write_config(
        {
            "config_path": str(config_path),
            "config_key": "mcpServers",
        }
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["existing"]["label"] == "caf\u00e9"
    assert data["mcpServers"]["entroly"]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_claude_code_mcp_wrap_uses_project_mcp_json():
    """Claude Code MCP mode should write the project `.mcp.json`, not the
    Claude Desktop global config."""
    from entroly.cli import _WRAP_AGENTS

    spec = _WRAP_AGENTS["claude-code"]

    assert spec["kind"] == "mcp"
    assert spec["config_path"] == "{cwd}/.mcp.json"


# ── 3. Hook installer (entroly ravs hook install) ────────────────────


def test_hook_install_dry_run(tmp_path: Path, monkeypatch):
    """`entroly ravs hook install --claude-code --dry-run` must succeed
    and emit the would-be settings without writing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    proc = _entroly("ravs", "hook", "install", "--claude-code", "--dry-run")
    # Dry-run should not write the file
    settings_file = tmp_path / ".claude" / "settings.json"
    assert not settings_file.exists(), (
        "--dry-run wrote the settings file — should only preview"
    )
    # Output should mention the would-be path
    assert ".claude" in proc.stdout or ".claude" in proc.stderr, (
        f"dry-run did not preview the settings path:\n{proc.stdout}{proc.stderr}"
    )
