"""`entroly proxy` / `entroly go` must not be blocked by a hard-coded :9378.

Reproduced: with anything listening on 9378 (a second project's proxy, or any
other program), `entroly proxy --port N` and `entroly go --port N` refused to
start ("Dashboard port 9378 is already in use. Stop the existing service"),
and there was no way to move the dashboard: cli.py checked and printed 9378
literally and create_proxy_app started the dashboard on port=9378.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.request

import pytest

import entroly.cli as cli


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("def handler():\n    return 'ok'\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.delenv("ENTROLY_DASHBOARD_PORT", raising=False)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def default_dashboard_port_taken():
    blocker = socket.socket()
    try:
        blocker.bind(("127.0.0.1", 9378))
        blocker.listen()
    except OSError:
        blocker.close()
        blocker = None  # already taken by something else: the same condition
    yield
    if blocker is not None:
        blocker.close()


def _no_serving(monkeypatch) -> dict:
    import uvicorn
    import webbrowser

    ran: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, log_level: ran.update(port=port))
    monkeypatch.setattr(webbrowser, "open", lambda *_a, **_k: True)
    return ran


def _proxy_args(port: int, dashboard_port: int | None) -> argparse.Namespace:
    return argparse.Namespace(
        port=port, host=None, quality=None, force=False, debug=False,
        autotune_daemon=False, bypass=False, witness=False, witness_profile=None,
        witness_embed=False, witness_nli=False, dashboard_port=dashboard_port,
    )


def _dashboard_health(port: int) -> dict:
    deadline = time.monotonic() + 10
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                return json.loads(resp.read())
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def test_proxy_moves_the_dashboard_when_9378_is_taken(
    project, default_dashboard_port_taken, monkeypatch, capsys
):
    ran = _no_serving(monkeypatch)
    proxy_port, dashboard_port = _free_port(), _free_port()

    cli.cmd_proxy(_proxy_args(proxy_port, dashboard_port))
    out = capsys.readouterr().out

    assert "already in use" not in out, out
    assert ran.get("port") == proxy_port
    assert f"http://localhost:{dashboard_port}" in out
    assert "localhost:9378" not in out
    assert _dashboard_health(dashboard_port).get("service") == "entroly-dashboard"


def test_go_moves_the_dashboard_when_9378_is_taken(
    project, default_dashboard_port_taken, monkeypatch, capsys
):
    ran = _no_serving(monkeypatch)
    proxy_port, dashboard_port = _free_port(), _free_port()
    monkeypatch.setenv("ENTROLY_DASHBOARD_PORT", str(dashboard_port))

    cli.cmd_go(argparse.Namespace(port=proxy_port, quality=None, force=True))
    out = capsys.readouterr().out

    assert "already in use" not in out, out
    assert ran.get("port") == proxy_port
    assert f"http://localhost:{dashboard_port}" in out
    assert _dashboard_health(dashboard_port).get("service") == "entroly-dashboard"


def test_busy_dashboard_port_failure_names_the_way_out(project, monkeypatch, capsys):
    _no_serving(monkeypatch)
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    busy_port = busy.getsockname()[1]
    try:
        cli.cmd_proxy(_proxy_args(_free_port(), busy_port))
    finally:
        busy.close()
    out = capsys.readouterr().out

    assert f"Dashboard port {busy_port} is already in use" in out, out
    assert "--dashboard-port" in out
