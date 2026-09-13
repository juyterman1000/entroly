from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from entroly.agent_activation import (
    ACTIVE_FRESHNESS_SECONDS,
    ActivationSelection,
    activation_status,
    configure_cursor_hook,
    configure_kiro_hook,
    hook_context,
    parse_hook_input,
    run_hook,
)
import entroly.agent_activation as activation_module


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


def test_codex_environment_is_recorded_without_noisy_success_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    output = run_hook(
        {"hook_event_name": "UserPromptSubmit", "prompt": "Review auth", "cwd": ""},
        state_dir=tmp_path / "state",
        selector=lambda *_: ActivationSelection("no_match", (), "", 0, True),
    )
    assert "systemMessage" not in output
    assert activation_status(project_dir=Path.cwd(), state_dir=tmp_path / "state")[
        "by_host"
    ] == {"codex": 1}


def test_cursor_environment_supplies_project_and_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CURSOR_PROJECT_DIR", str(project))
    observed: dict[str, object] = {}

    def selector(query: str, cwd: Path, _budget: int, _max_files: int):
        observed.update(query=query, cwd=cwd)
        return ActivationSelection("no_match", (), "", 0, True)

    run_hook(
        {"hook_event_name": "UserPromptSubmit", "prompt": "Review auth"},
        state_dir=tmp_path / "state",
        selector=selector,
    )
    assert observed == {"query": "Review auth", "cwd": project.resolve()}
    report = activation_status(project, state_dir=tmp_path / "state")
    assert report["state"] == "active"
    assert report["by_host"] == {"cursor": 1}


def test_kiro_prompt_submit_uses_environment_and_emits_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("USER_PROMPT", "Find the budget accounting bug")
    observed: dict[str, object] = {}

    def selector(query: str, cwd: Path, budget: int, max_files: int):
        observed.update(query=query, cwd=cwd)
        return ActivationSelection(
            "activated", ("entroly/budget.py",), "budget evidence", 12, True
        )

    output = run_hook(
        {
            "hook_event_name": "promptSubmit",
            "cwd": str(project),
            "session_id": "kiro-session",
        },
        host="kiro",
        state_dir=tmp_path / "state",
        selector=selector,
    )

    assert observed == {
        "query": "Find the budget accounting bug",
        "cwd": project.resolve(),
    }
    context = hook_context(output)
    assert "performed by the host hook" in context
    assert "entroly/budget.py" in context
    assert activation_status(project, state_dir=tmp_path / "state")["by_host"] == {
        "kiro": 1
    }


def test_kiro_hook_install_is_idempotent_and_uninstall_is_recoverable(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    installed = configure_kiro_hook(project)
    assert installed["status"] == "installed"
    target = Path(installed["path"])
    assert target.is_file()
    assert configure_kiro_hook(project)["status"] == "already_installed"

    disabled = configure_kiro_hook(project, uninstall=True)
    assert disabled["status"] == "disabled"
    assert not target.exists()
    assert Path(disabled["recoverable_at"]).is_file()


def test_kiro_hook_install_refuses_unrecognized_content(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = project / ".kiro" / "hooks" / "entroly-activation.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"version":"v1","hooks":[]}', encoding="utf-8")

    report = configure_kiro_hook(project)
    assert report["status"] == "conflict"
    assert json.loads(target.read_text(encoding="utf-8"))["hooks"] == []


def test_kiro_force_install_preserves_unrecognized_content(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = project / ".kiro" / "hooks" / "entroly-activation.json"
    target.parent.mkdir(parents=True)
    original = '{"version":"v1","hooks":[{"name":"customer-hook"}]}'
    target.write_text(original, encoding="utf-8")

    report = configure_kiro_hook(project, force=True)

    assert report["status"] == "installed"
    assert Path(report["backup"]).read_text(encoding="utf-8") == original
    assert json.loads(target.read_text(encoding="utf-8"))["hooks"][0]["name"].startswith(
        "Entroly"
    )


def test_cursor_hook_merge_preserves_settings_and_is_reversible(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = project / ".claude" / "settings.local.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps({"permissions": {"allow": ["Read"]}, "hooks": {"Stop": []}}),
        encoding="utf-8",
    )

    installed = configure_cursor_hook(project)
    assert installed["status"] == "installed"
    assert Path(installed["backup"]).is_file()
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["permissions"] == {"allow": ["Read"]}
    assert document["hooks"]["Stop"] == []
    assert len(document["hooks"]["UserPromptSubmit"]) == 1
    assert configure_cursor_hook(project)["status"] == "already_installed"

    disabled = configure_cursor_hook(project, uninstall=True)
    assert disabled["status"] == "disabled"
    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["permissions"] == {"allow": ["Read"]}
    assert after["hooks"]["UserPromptSubmit"] == []
    assert Path(disabled["recoverable_at"]).is_file()


def test_cursor_hook_refuses_invalid_existing_settings(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = project / ".claude" / "settings.local.json"
    target.parent.mkdir(parents=True)
    target.write_text("not-json", encoding="utf-8")

    report = configure_cursor_hook(project)
    assert report["status"] == "conflict"
    assert target.read_text(encoding="utf-8") == "not-json"


def test_atomic_config_replace_restores_original_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    target = project / ".claude" / "settings.local.json"
    target.parent.mkdir(parents=True)
    original = '{"permissions":{"allow":["Read"]}}'
    target.write_text(original, encoding="utf-8")
    real_replace = os.replace

    def fail_new_config(source, destination):
        if Path(source).name.startswith(".entroly-config-"):
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(activation_module.os, "replace", fail_new_config)
    with pytest.raises(OSError, match="simulated replacement failure"):
        configure_cursor_hook(project)

    assert target.read_text(encoding="utf-8") == original


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
    assert activation_status(project, state_dir=tmp_path / "state")["state"] == (
        "observed_degraded"
    )


def test_latest_failed_hook_does_not_leave_status_active(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    payload = {"prompt": "inspect", "cwd": str(project)}
    run_hook(
        payload,
        state_dir=state,
        selector=lambda *_: ActivationSelection("activated", ("app.py",), "x", 1, True),
    )

    def broken(*_):
        raise RuntimeError("failed")

    run_hook(payload, state_dir=state, selector=broken)
    report = activation_status(project, state_dir=state)
    assert report["state"] == "observed_degraded"
    assert report["activation_events"] == 2
    assert report["effective_activation_events"] == 1
    assert report["latest"]["status"] == "error"
    assert report["latest_effective"]["status"] == "activated"


def test_install_is_not_reported_as_activation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    report = activation_status(project, state_dir=tmp_path / "state")
    assert report["state"] == "unobserved"
    assert report["activation_events"] == 0


def test_old_effective_receipt_is_reported_as_stale(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    run_hook(
        {"prompt": "inspect", "cwd": str(project)},
        state_dir=state,
        selector=lambda *_: ActivationSelection("activated", ("app.py",), "x", 1, True),
    )
    receipt_path = next(state.rglob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["recorded_at_unix"] = time.time() - ACTIVE_FRESHNESS_SECONDS - 60
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = activation_status(project, state_dir=state)
    assert report["state"] == "stale"
    assert report["latest_age_seconds"] > ACTIVE_FRESHNESS_SECONDS


def test_hook_clamps_resource_controls(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    observed: dict[str, int] = {}

    def selector(_query: str, _cwd: Path, budget: int, max_files: int):
        observed.update(budget=budget, max_files=max_files)
        return ActivationSelection("no_match", (), "", 0, True)

    run_hook(
        {"prompt": "inspect", "cwd": str(project)},
        token_budget=10**9,
        max_files=10**9,
        state_dir=tmp_path / "state",
        selector=selector,
    )
    assert observed == {"budget": 8_000, "max_files": 1_000}


def test_hook_input_requires_bounded_json_object() -> None:
    assert parse_hook_input('{"prompt":"hello"}') == {"prompt": "hello"}
    with pytest.raises(ValueError, match="JSON object"):
        parse_hook_input("[]")
    with pytest.raises(ValueError, match="1 MiB"):
        parse_hook_input(json.dumps({"prompt": "x" * 1_048_576}))
