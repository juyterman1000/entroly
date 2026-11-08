"""A busy proxy port must fail the command, before any indexing.

Dogfooding 1.0.88: with another `entroly go` already on :9377 (started in a
different repository), a second `entroly go` indexed the project, printed
"Port 9377 is already in use." and exited 0. It did not say the port belonged
to an Entroly proxy that selects context from the other repository.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import entroly.cli as cli


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("def handler():\n    return 'ok'\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.delenv("ENTROLY_DASHBOARD_PORT", raising=False)
    monkeypatch.chdir(root)

    import entroly.auto_index as auto_index

    def refuse(*_a, **_k):
        raise AssertionError("indexed although the port was busy")

    monkeypatch.setattr(auto_index, "auto_index", refuse)
    return root


def _args(port: int) -> argparse.Namespace:
    return argparse.Namespace(
        port=port, host=None, quality=None, force=False, debug=False,
        autotune_daemon=False, bypass=False, witness=None, witness_profile=None,
        witness_embed=False, witness_nli=False, dashboard_port=None,
    )


@pytest.fixture
def plain_listener():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    yield sock.getsockname()[1]
    sock.close()


@pytest.fixture
def entroly_proxy_listener():
    class Health(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            body = json.dumps({"status": "ok", "service": "entroly-proxy"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("command", ["go", "proxy"])
def test_busy_port_returns_nonzero_without_indexing(
    project, plain_listener, monkeypatch, capsys, command
) -> None:
    monkeypatch.setattr(cli, "_check_codebase", lambda: True)
    handler = cli.cmd_go if command == "go" else cli.cmd_proxy

    assert handler(_args(plain_listener)) == 1
    out = capsys.readouterr().out
    assert f"Port {plain_listener} is already in use" in out
    assert f"entroly {command} --port" in out


def test_busy_port_held_by_entroly_says_so(
    project, entroly_proxy_listener, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(cli, "_check_codebase", lambda: True)

    assert cli.cmd_go(_args(entroly_proxy_listener)) == 1
    out = capsys.readouterr().out
    assert "An Entroly proxy is already running there" in out
    assert "project it was started in" in out
