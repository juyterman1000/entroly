"""RAVS capture must read the outcome Claude Code actually sends.

Claude Code does not put an exit code in a Bash hook payload. It states the
outcome with the event itself:

    PostToolUse         fires only when the command exited 0; `tool_response`
                        carries stdout, stderr and `interrupted`, no exit code
    PostToolUseFailure  fires when it exited non-zero; there is no
                        `tool_response`, and `error` opens with "Exit code N"
                        followed by the command's output

`capture_from_stdin` looked for a top-level `exit_code`, found none in either
payload, and returned None. Every Bash call made through Claude Code was
dropped, so the RAVS hook recorded nothing at all, and the session journal
wrote a failing test run down as "ok".

A pass is inferred from PostToolUse only when the hook declares
`--host claude-code`. The installer writes that flag on the same change that
registers PostToolUseFailure, so a registration that can only ever see
successes never records them: a sample of passes with every failure missing
would raise routing confidence on evidence it never saw.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from entroly.ravs.capture import capture_from_stdin

BASE = {
    "session_id": "7f3c9a2e-1b4d-4e8a-9c21-3d5e6f708192",
    "transcript_path": "/home/u/.claude/projects/p/7f3c.jsonl",
    "cwd": "/home/u/click",
    "permission_mode": "default",
    "tool_name": "Bash",
    "tool_input": {
        "command": "python -m pytest -q tests/test_options.py",
        "description": "Run the option tests",
    },
    "tool_use_id": "toolu_01Hk3sQy7",
}
SUCCEEDED = {
    **BASE,
    "hook_event_name": "PostToolUse",
    "tool_response": {
        "stdout": "....\n4 passed in 0.12s",
        "stderr": "",
        "interrupted": False,
        "isImage": False,
    },
}
FAILED = {
    **BASE,
    "hook_event_name": "PostToolUseFailure",
    "error": (
        "Exit code 1\nF...\nFAILED tests/test_options.py::test_envvar_empty"
        " - AssertionError\n1 failed, 3 passed in 0.12s"
    ),
    "is_interrupt": False,
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path / "home"))


def _capture(monkeypatch, payload, log: Path, host: str | None = None):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    return capture_from_stdin(log_path=str(log), host=host)


def _outcomes(log: Path) -> list[dict]:
    if not log.exists():
        return []
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [event for event in events if event.get("type") == "outcome"]


def test_a_succeeded_bash_call_from_claude_code_records_a_pass(tmp_path, monkeypatch) -> None:
    log = tmp_path / "events.jsonl"

    result = _capture(monkeypatch, SUCCEEDED, log, host="claude-code")

    assert result is not None
    assert [(o["value"], o["exit_code"]) for o in _outcomes(log)] == [("pass", 0)]


def test_without_the_host_flag_a_success_is_still_unknown(tmp_path, monkeypatch) -> None:
    """An old registration has no failure hook, so its passes would be one-sided."""
    log = tmp_path / "events.jsonl"

    assert _capture(monkeypatch, SUCCEEDED, log) is None
    assert _outcomes(log) == []


def test_a_failed_bash_call_records_its_exit_code(tmp_path, monkeypatch) -> None:
    log = tmp_path / "events.jsonl"

    _capture(monkeypatch, FAILED, log, host="claude-code")

    assert [(o["value"], o["exit_code"]) for o in _outcomes(log)] == [("fail", 1)]


def test_a_failure_needs_no_host_flag_because_it_states_its_code(tmp_path, monkeypatch) -> None:
    log = tmp_path / "events.jsonl"

    _capture(monkeypatch, FAILED, log)

    assert [o["exit_code"] for o in _outcomes(log)] == [1]


@pytest.mark.parametrize(
    "payload",
    [
        {**FAILED, "is_interrupt": True},
        {**FAILED, "error": "Exit code 0\n4 passed"},  # contradicts the event
        {**FAILED, "error": "Command timed out\nlast line says Exit code 2"},
        {**FAILED, "error": 17},
        {**SUCCEEDED, "tool_response": {**SUCCEEDED["tool_response"], "interrupted": True}},
        {**SUCCEEDED, "tool_response": "4 passed"},
        {**SUCCEEDED, "tool_name": "Read"},
    ],
    ids=[
        "interrupted",
        "failure-claiming-exit-0",
        "exit-code-not-on-the-first-line",
        "error-not-a-string",
        "success-interrupted",
        "success-without-a-response-object",
        "not-bash",
    ],
)
def test_anything_short_of_a_stated_outcome_records_nothing(
    tmp_path, monkeypatch, payload
) -> None:
    log = tmp_path / "events.jsonl"

    assert _capture(monkeypatch, payload, log, host="claude-code") is None
    assert _outcomes(log) == []


def test_an_explicit_exit_code_still_wins(tmp_path, monkeypatch) -> None:
    log = tmp_path / "events.jsonl"

    _capture(monkeypatch, {**SUCCEEDED, "exit_code": 3}, log, host="claude-code")

    assert [o["exit_code"] for o in _outcomes(log)] == [3]


def _journal_entries(entroly_dir: Path) -> list[dict]:
    from entroly.session_journal import journal_path

    path = journal_path(entroly_dir)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_journal_records_a_failure_as_a_failure(tmp_path) -> None:
    from entroly.session_journal import record_from_hook_payload

    assert record_from_hook_payload(FAILED, entroly_dir=tmp_path / "journal")

    [entry] = _journal_entries(tmp_path / "journal")
    assert entry["status"] == "exit:1"
    assert "1 failed, 3 passed" in entry["detail"]
    assert not entry["detail"].startswith("Exit code"), "the status already says it"


def test_the_journal_records_an_interrupted_failure_without_calling_it_ok(tmp_path) -> None:
    from entroly.session_journal import record_from_hook_payload

    record_from_hook_payload(
        {**FAILED, "error": "Interrupted by user", "is_interrupt": True},
        entroly_dir=tmp_path / "journal",
    )

    [entry] = _journal_entries(tmp_path / "journal")
    assert entry["status"] == "interrupted"


def test_the_journal_still_records_a_success_as_ok(tmp_path) -> None:
    from entroly.session_journal import record_from_hook_payload

    record_from_hook_payload(SUCCEEDED, entroly_dir=tmp_path / "journal")

    [entry] = _journal_entries(tmp_path / "journal")
    assert entry["status"] == "ok"


def test_the_installed_hook_command_records_both_outcomes(tmp_path) -> None:
    log = tmp_path / "events.jsonl"
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "ENTROLY_DIR": str(tmp_path / "entroly"),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    for payload in (SUCCEEDED, FAILED):
        completed = subprocess.run(
            [
                sys.executable, "-m", "entroly", "ravs", "capture",
                "--stdin", "--quiet", "--host", "claude-code", "--log", str(log),
            ],
            input=json.dumps(payload),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr

    assert [(o["value"], o["exit_code"]) for o in _outcomes(log)] == [
        ("pass", 0),
        ("fail", 1),
    ]
