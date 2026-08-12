from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import entroly.cli as cli


def _tool(path: Path) -> dict[str, str]:
    return {"config_path": str(path), "config_key": "mcpServers"}


def test_write_config_is_atomic_and_preserves_unrelated_entries(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    original = {
        "mcpServers": {"existing": {"command": "existing-server"}},
        "theme": "dark",
    }
    original_bytes = (json.dumps(original, ensure_ascii=False) + "\n").encode()
    config.write_bytes(original_bytes)

    written = cli._write_config(_tool(config))

    assert written == str(config)
    assert Path(f"{config}.entroly-backup").read_bytes() == original_bytes
    result = json.loads(config.read_text(encoding="utf-8"))
    assert result["theme"] == "dark"
    assert result["mcpServers"]["existing"] == {"command": "existing-server"}
    assert result["mcpServers"]["entroly"]["command"] == cli.sys.executable
    assert not list(tmp_path.glob(".mcp.json.entroly-*.tmp"))


def test_repeated_identical_wrap_is_a_noop_without_redundant_backup(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    # write_bytes, not write_text: on Windows the text path translates "\n" to
    # "\r\n", so this fixture wrote CRLF and then asserted LF on the next line.
    # The backup was byte-faithful all along -- it copies the file with rb/xb --
    # so the test failed on every Windows machine while passing in Linux CI.
    config.write_bytes(b'{"mcpServers": {}}\n')

    cli._write_config(_tool(config))
    first_result = config.read_bytes()
    cli._write_config(_tool(config))

    assert Path(f"{config}.entroly-backup").read_bytes() == b'{"mcpServers": {}}\n'
    assert config.read_bytes() == first_result
    assert not Path(f"{config}.entroly-backup.1").exists()


def test_write_config_dry_run_never_discloses_unrelated_config(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({
            "mcpServers": {"private-server": {"token": "secret"}},
            "account_id": "private-account",
            "preferences": {"private_path": "C:/private"},
        }),
        encoding="utf-8",
    )

    preview = cli._write_config(_tool(config), dry_run=True)

    payload = json.loads(preview)
    assert payload["operation"] == "merge"
    assert payload["entry"]["entroly"]["command"] == cli.sys.executable
    assert payload["preserves_unrelated_configuration"] is True
    assert "private-server" not in preview
    assert "private-account" not in preview
    assert "C:/private" not in preview
    assert not list(tmp_path.glob("mcp.json.entroly-backup*"))


def test_write_config_rejects_malformed_json_without_mutation(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    original = b'{"mcpServers": '
    config.write_bytes(original)

    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        cli._write_config(_tool(config))

    assert config.read_bytes() == original
    assert not list(tmp_path.glob("mcp.json.entroly-backup*"))
    assert not list(tmp_path.glob(".mcp.json.entroly-*.tmp"))


def test_atomic_replace_failure_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "mcp.json"
    original = b'{"mcpServers": {"existing": {}}}\n'
    config.write_bytes(original)

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(cli.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        cli._atomic_write_config(str(config), {"mcpServers": {"entroly": {}}})

    assert config.read_bytes() == original
    assert not list(tmp_path.glob(".mcp.json.entroly-*.tmp"))


def test_remove_entroly_is_surgical_and_backed_up(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    original = {
        "mcpServers": {
            "existing": {"command": "keep-me"},
            "entroly": {"command": "remove-me"},
        },
        "other": {"preserve": True},
    }
    original_bytes = (json.dumps(original, ensure_ascii=False) + "\n").encode()
    config.write_bytes(original_bytes)

    status, backup = cli._remove_entroly_config(_tool(config))

    assert status == "removed"
    assert backup is not None
    assert Path(backup).read_bytes() == original_bytes
    result = json.loads(config.read_text(encoding="utf-8"))
    assert "entroly" not in result["mcpServers"]
    assert result["mcpServers"]["existing"] == {"command": "keep-me"}
    assert result["other"] == {"preserve": True}


def test_remove_entroly_dry_run_does_not_touch_disk(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    original = b'{"mcpServers":{"entroly":{},"other":{}}}\n'
    config.write_bytes(original)

    preview, backup = cli._remove_entroly_config(_tool(config), dry_run=True)

    assert backup is None
    payload = json.loads(preview)
    assert payload == {
        "operation": "remove",
        "config_key": "mcpServers",
        "entry": "entroly",
        "preserves_unrelated_configuration": True,
    }
    assert config.read_bytes() == original
    assert not list(tmp_path.glob("mcp.json.entroly-unwrapped-backup*"))


def test_cmd_unwrap_removes_only_entroly_for_mcp_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"entroly": {}, "other": {"command": "keep"}}}
        ),
        encoding="utf-8",
    )
    spec = dict(cli._WRAP_AGENTS["claude-code"])
    spec["config_path"] = str(config)
    monkeypatch.setitem(cli._WRAP_AGENTS, "test-mcp", spec)

    result = cli.cmd_unwrap(SimpleNamespace(agent="test-mcp", dry_run=False))

    assert result == 0
    stored = json.loads(config.read_text(encoding="utf-8"))
    assert stored["mcpServers"] == {"other": {"command": "keep"}}
    assert "Entroly MCP entry removed" in capsys.readouterr().out


def test_cmd_unwrap_cli_agent_is_explicit_noop(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli.cmd_unwrap(SimpleNamespace(agent="aider", dry_run=False))

    assert result == 0
    assert "process-scoped environment variables" in capsys.readouterr().out


def test_backup_copy_failure_removes_partial_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "mcp.json"
    config.write_bytes(b'{"mcpServers": {}}\n')

    def fail_copy(_source, destination):
        destination.write(b"partial")
        raise OSError("simulated backup failure")

    monkeypatch.setattr(cli.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="simulated backup failure"):
        cli._backup_config_file(config, ".entroly-backup")

    assert config.read_bytes() == b'{"mcpServers": {}}\n'
    assert not list(tmp_path.glob("mcp.json.entroly-backup*"))
