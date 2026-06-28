from __future__ import annotations

import sys

import pytest

from entroly import _docker_launcher


def test_docker_launcher_routes_memory_without_docker(monkeypatch) -> None:
    calls: list[list[str] | None] = []

    def fake_memory_main(argv=None):
        calls.append(argv)
        return 0

    monkeypatch.setattr("entroly.memory_cli.main", fake_memory_main)
    monkeypatch.setattr(sys, "argv", ["entroly", "memory", "stats"])

    with pytest.raises(SystemExit) as exc:
        _docker_launcher.launch()

    assert exc.value.code == 0
    assert calls == [["stats"]]
