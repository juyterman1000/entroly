"""`entroly init` / `entroly wrap vscode` must write VS Code's mcp.json schema.

VS Code reads workspace MCP servers from `.vscode/mcp.json` under the
`servers` key (entries carry `"type": "stdio"`), the format this repository's
own `.vscode/mcp.json` uses. `entroly init` wrote the Claude/Cursor shape
(`mcpServers`) into that file, printed "Generated .vscode/mcp.json" and
"Restart your AI tool", and the server was never registered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import entroly.cli as cli

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path, monkeypatch, *editor_dirs: str) -> Path:
    project = tmp_path / "project"
    for name in editor_dirs:
        (project / name).mkdir(parents=True)
    (project / "app.py").write_text("print('hello')\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(project)
    return project


def _assert_vscode_entry(config: dict) -> None:
    assert "mcpServers" not in config, config
    entry = config["servers"]["entroly"]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["-m", "entroly.server"]


def test_init_writes_the_schema_vscode_reads(tmp_path, monkeypatch, capsys):
    project = _project(tmp_path, monkeypatch, ".vscode", ".cursor")

    cli.cmd_init(argparse.Namespace(dry_run=False, yes=False))
    capsys.readouterr()

    _assert_vscode_entry(
        json.loads((project / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    )
    # Cursor keeps its own (Claude-compatible) key.
    cursor = json.loads((project / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert "entroly" in cursor["mcpServers"]


def test_init_keeps_existing_vscode_servers(tmp_path, monkeypatch, capsys):
    project = _project(tmp_path, monkeypatch, ".vscode")
    existing = {"servers": {"other": {"type": "stdio", "command": "other-mcp"}}}
    (project / ".vscode" / "mcp.json").write_text(json.dumps(existing), encoding="utf-8")

    cli.cmd_init(argparse.Namespace(dry_run=False, yes=False))
    capsys.readouterr()

    config = json.loads((project / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    _assert_vscode_entry(config)
    assert config["servers"]["other"] == existing["servers"]["other"]


def test_wrap_vscode_writes_the_same_schema(tmp_path, monkeypatch, capsys):
    project = _project(tmp_path, monkeypatch, ".vscode")

    assert cli._wrap_via_mcp(cli._WRAP_AGENTS["vscode"], port=21050) is True
    capsys.readouterr()

    _assert_vscode_entry(
        json.loads((project / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    )


def test_repository_vscode_config_is_the_reference_format():
    reference = json.loads((ROOT / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    assert reference["servers"]["entroly"]["type"] == "stdio"
