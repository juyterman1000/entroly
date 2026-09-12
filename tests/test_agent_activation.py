from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly.agent_activation import (
    ActivationSelection,
    activation_status,
    parse_hook_input,
    run_hook,
)


def test_host_hook_activates_without_any_model_mcp_call(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    observed: dict[str, object] = {}
    prompt = "Fix the retry race without weakening evidence checks"

    def selector(query: str, cwd: Path, budget: int, max_files: int):
        observed.update(
            query=query,
            cwd=cwd,
            budget=budget,
            max_files=max_files,
        )
        return ActivationSelection(
            status="activated",
            sources=("entroly/retry.py",),
            context="def retry(): ...",
            selected_tokens=17,
            native_engine=True,
        )

    output = run_hook(
        {
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
            "cwd": str(project),
            "session_id": "private-session-id",
        },
        host="claude-code",
        token_budget=900,
        max_files=80,
        state_dir=state,
        selector=selector,
    )

    assert observed == {
        "query": prompt,
        "cwd": project.resolve(),
        "budget": 900,
        "max_files": 80,
    }
    hook_output = output["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "UserPromptSubmit"
    assert "performed by the host hook" in hook_output["additionalContext"]
    assert "entroly/retry.py" in hook_output["additionalContext"]

    report = activation_status(project, state_dir=state)
    assert report["state"] == "active"
    assert report["activation_events"] == 1
    assert report["by_host"] == {"claude-code": 1}
    receipt = report["latest"]
    assert receipt["prompt_persisted"] is False
    assert receipt["prompt_sha256"]
    persisted = Path(receipt["receipt_path"]) if "receipt_path" in receipt else next(
        (state / report["project_fingerprint"] / "events").glob("*.json")
    )
    raw = persisted.read_text(encoding="utf-8")
    assert prompt not in raw
    assert "private-session-id" not in raw


def test_gemini_before_agent_injects_turn_context(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = run_hook(
        {"hook_event_name": "BeforeAgent", "prompt": "Review auth", "cwd": str(project)},
        state_dir=tmp_path / "state",
        selector=lambda *_: ActivationSelection(
            "no_match", (), "", 0, True, "no relevant local evidence"
        ),
    )
    assert output["hookSpecificOutput"]["hookEventName"] == "BeforeAgent"
    assert "status: no_match" in output["hookSpecificOutput"]["additionalContext"]


def test_activation_failure_fails_open_without_leaking_exception(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    def broken(*_):
        raise RuntimeError("secret provider token")

    output = run_hook(
        {"prompt": "continue my task", "cwd": str(project)},
        state_dir=tmp_path / "state",
        selector=broken,
    )
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "status: error" in context
    assert "secret provider token" not in context


def test_install_is_not_reported_as_activation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    report = activation_status(project, state_dir=tmp_path / "state")
    assert report["state"] == "installed_but_unobserved"
    assert report["activation_events"] == 0


def test_hook_input_requires_bounded_json_object() -> None:
    assert parse_hook_input('{"prompt":"hello"}') == {"prompt": "hello"}
    with pytest.raises(ValueError, match="JSON object"):
        parse_hook_input("[]")
    with pytest.raises(ValueError, match="1 MiB"):
        parse_hook_input(json.dumps({"prompt": "x" * 1_048_576}))
