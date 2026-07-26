from pathlib import Path
from types import SimpleNamespace

from entroly.cli import _WRAP_AGENTS, _resolved_wrap_env, _split_wrap_agent_args
from entroly.session_attach import (
    ATTACHMENT_CLIENTS,
    attachment_install_commands,
    attachment_remove_commands,
)


def test_copilot_attachment_uses_current_cli_contract(tmp_path):
    grant = SimpleNamespace(
        grant_id="att_0123456789abcdef",
        client="copilot",
        project_root=str(tmp_path),
    )
    install = attachment_install_commands(grant, tmp_path, tmp_path / "token")
    assert install[0][:4] == ("copilot", "mcp", "add", "entroly-att_0123456789abcdef")
    assert "--env" in install[0]
    assert attachment_remove_commands(grant) == (
        ("copilot", "mcp", "remove", "entroly-att_0123456789abcdef"),
    )
    assert "copilot" in ATTACHMENT_CLIENTS


def test_copilot_wrapper_uses_byok_provider_variables():
    spec = _WRAP_AGENTS["copilot"]
    assert spec["cmd"] == ["copilot"]
    assert "api_key_env" not in spec  # local providers may be unauthenticated
    assert spec["subscription_alt"].startswith("entroly attach create --client copilot")
    assert _resolved_wrap_env(spec, 9377) == {
        "COPILOT_PROVIDER_BASE_URL": "http://localhost:9377/v1",
        "COPILOT_PROVIDER_TYPE": "openai",
    }


def test_new_wrapper_statuses_are_explicit():
    assert _resolved_wrap_env(_WRAP_AGENTS["goose"], 9377)["OPENAI_BASE_PATH"] == "v1/chat/completions"
    assert _WRAP_AGENTS["openhands"]["cmd"] == ["openhands", "--override-with-envs"]
    for name in ("grok", "vibe", "omp", "zcode"):
        assert _WRAP_AGENTS[name]["kind"] == "print"


def test_public_matrix_keeps_subscription_and_validation_boundaries():
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/agent-compatibility.md").read_text(encoding="utf-8")
    assert "does not claim interception of GitHub-hosted subscription inference" in readme
    assert "Cortex Code" in readme and "Not validated as a wrap target" in readme
    assert "Provider-bound token and cost measurements exist only" in guide
    assert "MCP-only integrations" in guide


def test_wrap_splits_wrapper_options_from_client_arguments():
    wrapper_args, client_args, explicit = _split_wrap_agent_args(
        [
            "--port",
            "7777",
            "--force",
            "--",
            "--port",
            "8888",
            "--dry-run",
            "--prompt=Hello world",
        ]
    )
    assert explicit is True
    assert wrapper_args == ["--port", "7777", "--force"]
    assert client_args == [
        "--port",
        "8888",
        "--dry-run",
        "--prompt=Hello world",
    ]


def test_wrap_consumes_separator_when_it_is_first():
    wrapper_args, client_args, explicit = _split_wrap_agent_args(
        ["--", "-s", "--prompt=Hello world"]
    )
    assert explicit is True
    assert wrapper_args == []
    assert client_args == ["-s", "--prompt=Hello world"]


def test_wrap_without_separator_shares_recoverable_argument_list():
    wrapper_args, client_args, explicit = _split_wrap_agent_args(
        ["--port", "9379", "-p", "hello"]
    )
    assert explicit is False
    assert wrapper_args is client_args
    wrapper_args.pop(0)
    wrapper_args.pop(0)
    assert client_args == ["-p", "hello"]
