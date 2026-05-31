"""Unit tests for entroly.cli."""
from __future__ import annotations

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly import cli


class _HealthHandler(BaseHTTPRequestHandler):
    payload = b'{"status":"ok"}'

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format, *args):
        pass


def test_free_port_does_not_terminate_existing_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        assert cli._free_port(port) is False

        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass


def test_proxy_identity_probe_requires_entroly_service():
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        assert cli._is_entroly_proxy_running(port) is False
        _HealthHandler.payload = b'{"status":"ok","service":"entroly-proxy"}'
        assert cli._is_entroly_proxy_running(port) is True
    finally:
        _HealthHandler.payload = b'{"status":"ok"}'
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_update_check_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENTROLY_DISABLE_UPDATE_CHECK", "1")

    def fail_if_started(*args, **kwargs):
        raise AssertionError("disabled update check must not start a network thread")

    monkeypatch.setattr(threading, "Thread", fail_if_started)
    cli._check_for_update()


def test_upstream_probe_is_opt_in(monkeypatch):
    monkeypatch.delenv("ENTROLY_CHECK_UPSTREAM", raising=False)
    assert cli._should_check_upstream() is False

    monkeypatch.setenv("ENTROLY_CHECK_UPSTREAM", "1")
    assert cli._should_check_upstream() is True


def test_telemetry_command_describes_local_preference(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ENTROLY_DIR", tmp_path)
    cli.cmd_telemetry(SimpleNamespace(action="on"))

    output = capsys.readouterr().out
    assert "Local telemetry preference enabled." in output
    assert "No outbound telemetry uploader is included in this release." in output


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_detect_project_type_unknown(chdir_tmp):
    result = cli._detect_project_type()
    assert result["primary"] == "unknown"
    assert result["languages"] == ["unknown"]
    assert result["frameworks"] == []
    assert result["name"] == chdir_tmp.name


def test_detect_project_type_python(chdir_tmp):
    (chdir_tmp / "pyproject.toml").write_text("[project]\nname='x'\n")
    result = cli._detect_project_type()
    assert "python" in result["languages"]
    assert result["primary"] == "python"


def test_detect_project_type_rust_and_go(chdir_tmp):
    (chdir_tmp / "Cargo.toml").write_text("[package]\n")
    (chdir_tmp / "go.mod").write_text("module x\n")
    result = cli._detect_project_type()
    assert "rust" in result["languages"]
    assert "go" in result["languages"]


def test_detect_project_type_react_via_package_json(chdir_tmp):
    (chdir_tmp / "package.json").write_text(json.dumps({
        "dependencies": {"react": "^18.0.0", "express": "^4.0.0"},
    }))
    result = cli._detect_project_type()
    assert "javascript" in result["languages"]
    assert "React" in result["frameworks"]
    assert "Express" in result["frameworks"]


def test_detect_project_type_nextjs_suppresses_react(chdir_tmp):
    (chdir_tmp / "next.config.js").write_text("module.exports={}")
    (chdir_tmp / "package.json").write_text(json.dumps({
        "dependencies": {"react": "^18.0.0"},
    }))
    result = cli._detect_project_type()
    assert "Next.js" in result["frameworks"]
    assert "React" not in result["frameworks"]


def test_detect_project_type_handles_malformed_package_json(chdir_tmp):
    (chdir_tmp / "package.json").write_text("{not valid json")
    result = cli._detect_project_type()
    assert "javascript" in result["languages"]
    assert result["frameworks"] == []
