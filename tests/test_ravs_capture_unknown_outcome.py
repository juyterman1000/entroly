"""A capture payload without an exit code is not success evidence.

`entroly ravs capture --stdin` defaulted a missing ``exit_code`` to 0, so a
PostToolUse payload that did not state one -- even for a failing test run --
was written to the RAVS log as a strong "pass". RAVS is fail-closed: unknown
must stay unknown, whatever the host sends.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

from entroly.ravs.capture import capture_from_stdin

FAILING_RUN_WITHOUT_EXIT_CODE = {
    "hook_event_name": "PostToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": "pytest -q tests/test_options.py"},
    "tool_response": {"stdout": "F...\n1 failed, 3 passed in 0.12s", "stderr": ""},
}


def _outcomes(log):
    if not log.exists():
        return []
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [event for event in events if event.get("type") == "outcome"]


def _capture(monkeypatch, payload, log):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    return capture_from_stdin(log_path=str(log))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path / "home"))


def test_a_payload_without_an_exit_code_records_no_outcome(tmp_path, monkeypatch) -> None:
    log = tmp_path / "events.jsonl"

    assert _capture(monkeypatch, FAILING_RUN_WITHOUT_EXIT_CODE, log) is None
    assert _outcomes(log) == []


@pytest.mark.parametrize(("exit_code", "verdict"), [(1, "fail"), (0, "pass"), ("1", "fail")])
def test_a_stated_exit_code_is_still_classified(tmp_path, monkeypatch, exit_code, verdict) -> None:
    log = tmp_path / "events.jsonl"
    payload = {**FAILING_RUN_WITHOUT_EXIT_CODE, "exit_code": exit_code}

    _capture(monkeypatch, payload, log)

    assert [event["value"] for event in _outcomes(log)] == [verdict]


def test_the_hook_command_does_not_log_a_pass(tmp_path) -> None:
    log = tmp_path / "events.jsonl"
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "ENTROLY_DIR": str(tmp_path / "entroly"),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "entroly", "ravs", "capture", "--stdin", "--quiet", "--log", str(log)],
        input=json.dumps(FAILING_RUN_WITHOUT_EXIT_CODE),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0
    assert [event["value"] for event in _outcomes(log)] == []
