"""One prompt, one Entroly activation, however many registrations fire.

`entroly activation install --host cursor` writes `.claude/settings.local.json`,
which Claude Code also loads. With the Entroly Claude Code plugin enabled, the
host ran both registrations for every prompt -- in parallel, with the same
payload -- so Entroly selected context twice, injected it twice and wrote two
receipts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from entroly.agent_activation import ActivationSelection, run_hook


def _payload(project, transcript):
    return {
        "session_id": "3f1c9a7e-1111-4c1e-9d55-aaaaaaaaaaaa",
        "transcript_path": str(transcript),
        "cwd": str(project),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Where is the retry loop?",
    }


def _setup(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "retry.py").write_text("def retry_loop():\n    return 1\n", encoding="utf-8")
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type": "user"}\n', encoding="utf-8")
    return project, transcript


def _receipts(state):
    return list(state.rglob("events/*.json"))


def test_a_second_registration_of_the_same_prompt_does_not_run(tmp_path) -> None:
    project, transcript = _setup(tmp_path)
    state = tmp_path / "state"
    calls = []

    def selector(*args):
        calls.append(args)
        return ActivationSelection("activated", ("retry.py",), "def retry_loop()", 5, True)

    first = run_hook(_payload(project, transcript), host="cursor", state_dir=state, selector=selector)
    second = run_hook(_payload(project, transcript), host="auto", state_dir=state, selector=selector)

    assert len(calls) == 1
    assert "retry.py" in first["hookSpecificOutput"]["additionalContext"]
    assert second["hookSpecificOutput"]["additionalContext"] == ""
    assert len(_receipts(state)) == 1


def test_the_same_prompt_in_a_later_turn_runs_again(tmp_path) -> None:
    project, transcript = _setup(tmp_path)
    state = tmp_path / "state"
    calls = []

    def selector(*args):
        calls.append(args)
        return ActivationSelection("no_match", (), "", 0, True)

    run_hook(_payload(project, transcript), state_dir=state, selector=selector)
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write('{"type": "assistant"}\n')
    run_hook(_payload(project, transcript), state_dir=state, selector=selector)

    assert len(calls) == 2


def test_manual_runs_without_a_session_are_never_deduplicated(tmp_path) -> None:
    project, _ = _setup(tmp_path)
    state = tmp_path / "state"
    calls = []

    def selector(*args):
        calls.append(args)
        return ActivationSelection("no_match", (), "", 0, True)

    for _ in range(2):
        run_hook({"prompt": "inspect", "cwd": str(project)}, state_dir=state, selector=selector)

    assert len(calls) == 2


def test_parallel_hook_processes_inject_context_once(tmp_path) -> None:
    """What the host does: two registrations, started together, same payload."""
    project, transcript = _setup(tmp_path)
    env = {
        **os.environ,
        "ENTROLY_DIR": str(tmp_path / "entroly"),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    payload = json.dumps(_payload(project, transcript))

    def run(host):
        completed = subprocess.run(
            [sys.executable, "-m", "entroly", "activation", "hook", "--host", host],
            input=payload,
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)["hookSpecificOutput"]["additionalContext"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        contexts = list(pool.map(run, ["cursor", "auto"]))

    assert sorted(bool(context) for context in contexts) == [False, True]
    assert len(_receipts(tmp_path / "entroly" / "activation")) == 1


def test_without_a_readable_transcript_a_repeated_prompt_still_runs(tmp_path) -> None:
    """No transcript means a turn cannot be told apart from a repeat: run."""
    project, _ = _setup(tmp_path)
    state = tmp_path / "state"
    calls = []

    def selector(*args):
        calls.append(args)
        return ActivationSelection("no_match", (), "", 0, True)

    payload = _payload(project, tmp_path / "missing-transcript.jsonl")
    for _ in range(2):
        run_hook(payload, state_dir=state, selector=selector)

    assert len(calls) == 2
