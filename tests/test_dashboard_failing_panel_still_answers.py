"""A panel that cannot load must say why, not drop the connection.

Reported from a live dashboard: "1 subsystem error -- dashboard data may be
incomplete. fetch TypeError: Failed to fetch", with "Context session index
unavailable: Failed to fetch". Intermittent -- correct most of the time, blank
occasionally.

`do_GET` dispatched without an exception guard. A route whose data source was
momentarily unreadable raised, the exception propagated out of
`finish_request`, and the socket closed with no status line. A browser reports
exactly `TypeError: Failed to fetch` for that, which names no endpoint and no
cause, so an unavailable subsystem looked like a dead dashboard.

The panels already render "<panel> unavailable: <reason>". They were never
given a reason.
"""

from __future__ import annotations

import json
import socket
import tempfile

import pytest


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", tempfile.mkdtemp())
    from entroly import dashboard as module

    server = module.start_dashboard(port=0)
    yield module, server, server.server_address[1]
    server.shutdown()


def _get(port: int, path: str) -> tuple[str, bytes]:
    """A request the dashboard's own trust check accepts."""
    conn = socket.create_connection(("127.0.0.1", port), timeout=10)
    conn.settimeout(10)
    conn.sendall(
        (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: localhost:{port}\r\n"
            f"Origin: http://localhost:{port}\r\n"
            "Sec-Fetch-Site: same-origin\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
    )
    raw = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        raw += chunk
    conn.close()
    status = raw.split(b"\r\n", 1)[0].decode(errors="replace") if raw else ""
    body = raw.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in raw else b""
    return status, body


@pytest.mark.timeout(120)
def test_a_raising_panel_returns_a_reason_not_a_dropped_connection(dashboard):
    module, _server, port = dashboard

    def explode(self):
        raise RuntimeError("context store unreadable")

    module.DashboardHandler._handle_context_health = explode

    status, body = _get(port, "/api/context/health")

    assert status, (
        "the connection closed with no status line; a browser reports this as "
        "'TypeError: Failed to fetch' and the panel cannot say what broke"
    )
    assert "500" in status
    payload = json.loads(body)
    assert "context store unreadable" in payload["error"], (
        "the panel needs the actual cause, not a generic failure"
    )
    assert payload["path"] == "/api/context/health"


@pytest.mark.timeout(120)
def test_one_broken_panel_does_not_take_down_the_others(dashboard):
    module, _server, port = dashboard
    module.DashboardHandler._handle_context_health = lambda self: (_ for _ in ()).throw(
        RuntimeError("boom")
    )

    _get(port, "/api/context/health")
    status, _body = _get(port, "/health")

    assert "200" in status, (
        "a failure in one route must not leave the server unable to answer "
        "another -- the socket has to be closed either way"
    )


@pytest.mark.timeout(120)
def test_repeated_failures_do_not_exhaust_the_server(dashboard):
    """The old handler never called shutdown_request, so sockets leaked."""
    module, _server, port = dashboard
    module.DashboardHandler._handle_context_health = lambda self: (_ for _ in ()).throw(
        RuntimeError("boom")
    )

    for _ in range(25):
        _get(port, "/api/context/health")

    status, _body = _get(port, "/health")
    assert "200" in status, "the server stopped answering after repeated failures"
