"""Adapter contracts and real benign child-process bounds; no model calls."""

import json
import os
from pathlib import Path
import sys
import time

import pytest

from benchmarks import cross_agent_cli as cli


def codex_events(*, item="agent_message", text="patch", usage=None):
    return "\n".join(json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "test"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": item, "text": text}},
        {"type": "turn.completed", "usage": usage or {"input_tokens": 4, "cached_input_tokens": 3}},
    ])


def test_codex_raw_usage_and_final_answer_binding():
    text, usage, resolved = cli._parse_codex(codex_events(), "patch\n")
    assert text == "patch\n"
    assert usage == {"input_tokens": 4, "cached_input_tokens": 3}
    assert resolved == []
    with pytest.raises(RuntimeError):
        cli._parse_codex(codex_events(), "different")


@pytest.mark.parametrize("item", ["command_execution", "mcp_tool_call", "web_search", "file_change", "unknown"])
def test_codex_tool_or_unknown_events_invalidate_completion(item):
    with pytest.raises(RuntimeError, match="tool or unsupported"):
        cli._parse_codex(codex_events(item=item), "patch")


@pytest.mark.parametrize("transcript", ["", '{}', '{"type":"turn.failed"}', '[1]',
                                       '{"type":"future_tool_event"}'])
def test_incomplete_or_unknown_codex_transcripts_are_not_scores(transcript):
    with pytest.raises(RuntimeError):
        cli._parse_codex(transcript, "patch")


def test_claude_usage_remains_provider_specific():
    result = {"type": "result", "subtype": "success", "result": "patch", "num_turns": 1,
              "usage": {"input_tokens": 2, "cache_read_input_tokens": 18},
              "modelUsage": {"provider-reported-model": {"inputTokens": 2}}}
    text, usage, resolved = cli._parse_claude(json.dumps(result))
    assert text == "patch"
    assert usage["usage"]["cache_read_input_tokens"] == 18
    assert resolved == ["provider-reported-model"]
    for changed in ({"is_error": True}, {"num_turns": 2}, {"permission_denials": ["tool"]},
                    {"subtype": "error_max_turns"}, {"result": ""}):
        with pytest.raises(RuntimeError):
            cli._parse_claude(json.dumps({**result, **changed}))


def test_missing_and_unverified_providers_never_install(monkeypatch):
    with pytest.raises(ValueError, match="no verified"):
        cli.preflight(cli.AgentSpec("gemini", "gemini", "model", "gemini"))
    monkeypatch.setattr(cli.shutil, "which", lambda x: None)
    with pytest.raises(FileNotFoundError):
        cli.preflight(cli.AgentSpec("codex", "codex", "model", "missing"))


@pytest.mark.parametrize("extension", [".ps1", ".cmd", ".bat"])
def test_shell_shortcuts_rejected(monkeypatch, tmp_path, extension):
    executable = tmp_path / ("agent" + extension)
    executable.write_text("shortcut")
    monkeypatch.setattr(cli.shutil, "which", lambda x: str(executable))
    with pytest.raises(ValueError, match="shell shortcut"):
        cli.preflight(cli.AgentSpec("codex", "codex", "model", str(executable)))


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_generation_fresh_configuration_and_no_repo_access(monkeypatch, tmp_path, provider):
    spec = cli.AgentSpec(provider, provider, "explicit-model", "agent.exe")
    runtime = {"executable": "agent.exe", "version": "test", "requested_model": spec.model}
    monkeypatch.setattr(cli, "preflight", lambda s: runtime)
    old_home = tmp_path / "old-home"
    old_home.mkdir()
    (old_home / "auth.json").write_text('{"test":"fake-auth"}')
    (old_home / "config.toml").write_text('unwanted = "config"')
    monkeypatch.setenv("CODEX_HOME", str(old_home))
    observed = []

    def invoke(argv, root, *, env, prompt, timeout_s):
        observed.append((argv, root))
        assert not (root / ".git").exists()
        assert prompt == "task"
        assert timeout_s == 1
        assert "--model" in argv and argv[argv.index("--model") + 1] == "explicit-model"
        assert not any("bypass" in arg or "dangerously" in arg for arg in argv)
        if provider == "codex":
            isolated = Path(env["CODEX_HOME"])
            assert isolated != old_home
            assert json.loads((isolated / "auth.json").read_text()) == {"test": "fake-auth"}
            assert not (isolated / "config.toml").exists()
            assert "--ignore-user-config" in argv and "read-only" in argv
            assert {"shell_tool", "hooks", "multi_agent"} <= set(argv)
            (root / "answer").write_text("patch")
            return codex_events()
        assert "--safe-mode" in argv and "--restricted" in argv and "--strict-mcp-config" in argv
        assert "--bare" not in argv  # Bare ignores subscription OAuth/keychain auth.
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--setting-sources") + 1] == ""
        return json.dumps({"type": "result", "subtype": "success", "result": "patch", "usage": {}})

    monkeypatch.setattr(cli, "_invoke", invoke)
    result = cli.generate(spec, "task", timeout_s=1)
    assert result["runtime"] == runtime and result["text"] == "patch"
    assert len(observed) == 1
    assert not observed[0][1].exists()  # Private prompt and copied auth removed.
    assert (old_home / "config.toml").exists()


def test_real_benign_process_wall_timeout(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        cli._invoke([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path,
                    env=os.environ.copy(), prompt="", timeout_s=0.2)
    assert time.monotonic() - started < 8


def test_real_benign_process_capture_limit(tmp_path):
    with pytest.raises(RuntimeError, match="capture limit"):
        cli._invoke([sys.executable, "-c", "import sys; sys.stdout.write('x' * 5000000)"],
                    tmp_path, env=os.environ.copy(), prompt="", timeout_s=5)


def test_successful_parent_exit_does_not_leave_descendant(tmp_path):
    marker = tmp_path / "should-not-exist"
    child = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('survived')"
    parent = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}]); print('complete')"
    text = cli._invoke([sys.executable, "-c", parent], tmp_path, env=os.environ.copy(), prompt="", timeout_s=5)
    assert text.strip() == "complete"
    time.sleep(1.2)
    assert not marker.exists()


def test_child_environment_blocks_inherited_prompt_telemetry(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.invalid")
    monkeypatch.setenv("OTEL_LOG_USER_PROMPTS", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-never-use")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-never-use")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-subscription-token")
    result = cli._child_env()
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in result
    assert result["OTEL_LOG_USER_PROMPTS"] == "0"
    assert result["OTEL_LOG_ASSISTANT_RESPONSES"] == "0"
    assert result["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
    assert result["OTEL_SDK_DISABLED"] == "true"
    assert "ANTHROPIC_API_KEY" not in result and "OPENAI_API_KEY" not in result
    assert "CLAUDE_CODE_USE_BEDROCK" not in result
    assert result["CLAUDE_CODE_OAUTH_TOKEN"] == "fake-subscription-token"
    assert os.environ["OTEL_LOG_USER_PROMPTS"] == "1"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_generation_budget_does_not_call_agent(timeout, monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda s: pytest.fail("must not invoke agent"))
    with pytest.raises(ValueError):
        cli.generate(cli.AgentSpec("codex", "codex", "model", "agent.exe"), "task", timeout_s=timeout)
