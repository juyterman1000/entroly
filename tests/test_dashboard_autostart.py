"""The MCP server must show its work.

Every other entry point starts a dashboard -- the proxy, the daemon, and
`entroly dashboard` itself. The MCP server did not, which made the most
common install the one where indexing, compression, belief seeding and
hallucination blocking all ran and none of it was visible. Value that is
measured but never displayed is indistinguishable, to the person who
installed it, from value that was never produced.

Autostart on that path is only safe under three conditions, pinned here: it
must not bind a port something else is serving, it must not write to stdout,
and it must not raise.
"""

from __future__ import annotations

import socket
import threading

import pytest

from entroly import dashboard


@pytest.fixture
def taken_port():
    """A port with a real listener on it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield sock.getsockname()[1]
    sock.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestPortSafety:
    def test_an_occupied_port_is_detected(self, taken_port):
        assert dashboard._port_already_serving(taken_port) is True

    def test_a_free_port_is_detected(self):
        assert dashboard._port_already_serving(_free_port()) is False

    def test_it_refuses_to_start_on_an_occupied_port(self, taken_port):
        """SO_REUSEADDR makes bind-and-see an unreliable test on Windows.

        The dashboard server sets allow_reuse_address, so a second bind to a
        live port can succeed there and leave two servers splitting requests
        non-deterministically. Presence must be probed, not inferred.
        """
        assert dashboard.maybe_start_dashboard(engine=None, port=taken_port) is None


class TestFailsOpen:
    def test_a_start_failure_returns_none_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(dashboard, "_port_already_serving", lambda *_a, **_k: False)
        monkeypatch.setattr(
            dashboard, "start_dashboard",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("no socket")))

        assert dashboard.maybe_start_dashboard(port=_free_port()) is None

    def test_an_oserror_returns_none_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(dashboard, "_port_already_serving", lambda *_a, **_k: False)
        monkeypatch.setattr(
            dashboard, "start_dashboard",
            lambda **_k: (_ for _ in ()).throw(OSError("address in use")))

        assert dashboard.maybe_start_dashboard(port=_free_port()) is None

    def test_it_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_DASHBOARD_AUTOSTART", "0")
        assert dashboard.dashboard_autostart_enabled() is False
        assert dashboard.maybe_start_dashboard(port=_free_port()) is None

    def test_it_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("ENTROLY_DASHBOARD_AUTOSTART", raising=False)
        assert dashboard.dashboard_autostart_enabled() is True


class TestStdioIsNotCorrupted:
    def test_starting_writes_nothing_to_stdout(self, capsys):
        """stdout on the MCP path is the JSON-RPC channel.

        One stray line desynchronises the client and the session dies, so the
        dashboard must announce itself on stderr or not at all.
        """
        port = _free_port()
        server = dashboard.maybe_start_dashboard(engine=None, port=port)
        try:
            assert capsys.readouterr().out == "", (
                "a print() here would break every MCP client"
            )
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()

    def test_the_server_actually_serves(self):
        """Autostart that silently no-ops is the bug it was meant to fix."""
        port = _free_port()
        server = dashboard.maybe_start_dashboard(engine=None, port=port)
        assert server is not None
        try:
            assert dashboard._port_already_serving(port) is True
        finally:
            server.shutdown()
            server.server_close()


class TestMcpWiring:
    def test_the_mcp_startup_path_starts_a_dashboard(self):
        """The wiring, not the server: without this call nothing is shown."""
        import inspect

        from entroly import server as server_module

        source = inspect.getsource(server_module)
        assert "maybe_start_dashboard" in source, (
            "the MCP server would index, compress and verify invisibly again"
        )

    def test_dashboard_start_is_not_left_to_a_bare_thread(self):
        """It must run inside the guarded startup path, not beside it."""
        import inspect

        from entroly import server as server_module

        source = inspect.getsource(server_module)
        index = source.index("maybe_start_dashboard")
        assert "try:" in source[max(0, index - 400):index], (
            "an unguarded dashboard start can take down the stdio session"
        )


class TestEmptyBeliefPanelTellsTheTruth:
    """An empty panel must not ask for work the product already does.

    It read "No beliefs yet -- run compile_beliefs to seed the vault". Since
    indexing seeds them, that instruction is now false in the common case: an
    empty panel means seeding is in flight, not that the user must act. The
    two states have different fixes, so the panel needs to know which it is in.
    """

    def test_the_stale_instruction_is_gone(self):
        from entroly.dashboard import DASHBOARD_HTML

        assert "run <code>compile_beliefs</code> to seed the vault" not in DASHBOARD_HTML

    def test_both_empty_states_are_distinguished(self):
        from entroly.dashboard import DASHBOARD_HTML

        assert "seeding runs automatically after indexing" in DASHBOARD_HTML
        assert "ENTROLY_BELIEF_AUTOSEED=0" in DASHBOARD_HTML
        assert "c.autoseed" in DASHBOARD_HTML, "the panel must branch on real state"

    def test_the_snapshot_reports_seeding_state(self, monkeypatch):
        from entroly.dashboard import _autoseed_state

        monkeypatch.delenv("ENTROLY_BELIEF_AUTOSEED", raising=False)
        assert _autoseed_state() is True
        monkeypatch.setenv("ENTROLY_BELIEF_AUTOSEED", "0")
        assert _autoseed_state() is False

    def test_a_missing_native_module_does_not_read_as_seeding_off(self, monkeypatch):
        """Different causes, different fixes -- the panel must not conflate them."""
        monkeypatch.delenv("ENTROLY_BELIEF_AUTOSEED", raising=False)
        from entroly.dashboard import _cogops_unavailable_snapshot

        snapshot = _cogops_unavailable_snapshot("no entroly_core")
        assert snapshot["autoseed"] is True, (
            "a native-engine problem would be reported as a seeding switch"
        )
