"""First-run setup must not invent an installed global MCP client."""

from __future__ import annotations

import json

from entroly import cli


def test_no_detected_client_does_not_target_global_claude_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")

    assert cli._detect_ai_tool()["tools"] == []
    assert not (tmp_path / "appdata" / "Claude").exists()


def test_existing_claude_config_is_detected_without_overriding_project_client(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    config = tmp_path / "appdata" / "Claude" / "claude_desktop_config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

    assert [tool["name"] for tool in cli._detect_ai_tool()["tools"]] == [
        "Claude Desktop"
    ]

    (tmp_path / ".cursor").mkdir()
    assert [tool["name"] for tool in cli._detect_ai_tool()["tools"]] == ["Cursor"]
