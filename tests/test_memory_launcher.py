from __future__ import annotations

import sys

import pytest

from entroly import _docker_launcher


class _FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_bare_launcher_shows_help_in_terminal(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("entroly.cli.main", lambda: calls.append("cli"))
    monkeypatch.setattr(_docker_launcher, "_run_native", lambda: calls.append("native"))
    monkeypatch.setattr(sys, "argv", ["entroly"])
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))

    _docker_launcher.launch()

    assert calls == ["cli"]


def test_bare_launcher_starts_mcp_under_stdio(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("entroly.cli.main", lambda: calls.append("cli"))
    monkeypatch.setattr(_docker_launcher, "_run_native", lambda: calls.append("native"))
    monkeypatch.setattr(sys, "argv", ["entroly"])
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))

    _docker_launcher.launch()

    assert calls == ["native"]


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


@pytest.mark.parametrize(
    "command",
    ["go", "optimize", "verify-claims", "witness", "daemon", "unknown-command"],
)
def test_docker_launcher_routes_every_non_serve_command_locally(
    monkeypatch, command: str
) -> None:
    calls: list[str] = []

    monkeypatch.setattr("entroly.cli.main", lambda: calls.append(command))
    monkeypatch.setattr(
        _docker_launcher,
        "_docker_available",
        lambda: pytest.fail("non-serve commands must not inspect Docker"),
    )
    monkeypatch.setattr(
        _docker_launcher,
        "_run_native",
        lambda: pytest.fail("non-serve commands must not start the MCP server"),
    )
    monkeypatch.setattr(sys, "argv", ["entroly", command])

    _docker_launcher.launch()

    assert calls == [command]


def test_docker_launcher_keeps_serve_as_the_only_docker_command(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_NO_DOCKER", raising=False)
    monkeypatch.setattr(_docker_launcher.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(_docker_launcher, "_docker_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["entroly", "serve"])

    with pytest.raises(SystemExit) as exc:
        _docker_launcher.launch()

    assert exc.value.code == 1
