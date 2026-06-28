from __future__ import annotations

import json

from entroly.memory_cli import main


def test_memory_cli_remember_recall_stats(tmp_path, capsys) -> None:
    path = tmp_path / "memory.json"

    rc = main([
        "remember",
        "Login timeout was fixed in auth/session.py",
        "--agent",
        "coder",
        "--importance",
        "0.9",
        "--file",
        str(path),
    ])
    assert rc == 0
    assert path.exists()

    rc = main([
        "recall",
        "login timeout",
        "--agent",
        "coder",
        "--budget",
        "100",
        "--file",
        str(path),
        "--json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n", 1)[-1] if out.startswith("stored memory") else out)
    assert payload["selected"]
    assert "auth/session.py" in payload["selected"][0]["content"]


def test_memory_cli_scan_blocks_secret(capsys) -> None:
    rc = main(["scan", "sk-abcdefghijklmnopqrstuvwxyz123456"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 2
    assert payload["allowed"] is False
    assert payload["findings"][0]["kind"] == "openai_key"
