from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

import entroly.daemon as daemon_module
import entroly.daemon_network_security as security
import entroly.server as server_module


def test_daemon_public_class_is_network_hardened() -> None:
    assert daemon_module.EntrolyDaemon is security.EntrolyDaemon


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "192.168.1.20",
        "10.0.0.4",
        "example.internal",
        "localhost.attacker.example",
        "http://127.0.0.1",
        "127.0.0.1/8",
        "127.0.0.1\n0.0.0.0",
        "",
        None,
    ],
)
def test_remote_or_malformed_daemon_hosts_fail_before_initialization(
    monkeypatch, host
) -> None:
    called = False

    def unexpected_init(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(security, "_ORIGINAL_INIT", unexpected_init)

    with pytest.raises(ValueError, match="host"):
        security.EntrolyDaemon(host=host)

    assert not called


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", "127.0.0.1"),
        ("LOCALHOST.", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("127.12.34.56", "127.12.34.56"),
        ("::1", "::1"),
    ],
)
def test_literal_loopback_hosts_are_normalized(host: str, expected: str) -> None:
    daemon = security.EntrolyDaemon(
        host=host,
        enable_proxy=False,
        enable_mcp=False,
    )

    assert daemon._host == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proxy_port", 0),
        ("proxy_port", 65536),
        ("proxy_port", True),
        ("proxy_port", 9377.5),
        ("proxy_port", float("nan")),
        ("dashboard_port", -1),
        ("dashboard_port", "9_378"),
        ("mcp_port", "९३७९"),
        ("mcp_port", None),
    ],
)
def test_invalid_daemon_ports_fail_before_initialization(
    monkeypatch, field: str, value
) -> None:
    called = False

    def unexpected_init(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(security, "_ORIGINAL_INIT", unexpected_init)
    kwargs = {field: value}

    with pytest.raises(ValueError, match="port"):
        security.EntrolyDaemon(**kwargs)

    assert not called


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_proxy", "false"),
        ("enable_proxy", 0),
        ("enable_proxy", None),
        ("enable_mcp", "true"),
        ("enable_mcp", 1),
        ("enable_mcp", []),
    ],
)
def test_service_enable_flags_require_real_booleans(
    monkeypatch, field: str, value
) -> None:
    called = False

    def unexpected_init(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(security, "_ORIGINAL_INIT", unexpected_init)

    with pytest.raises(ValueError, match="boolean"):
        security.EntrolyDaemon(**{field: value})

    assert not called


def test_enabled_services_cannot_share_a_port() -> None:
    with pytest.raises(ValueError, match="distinct ports"):
        security.EntrolyDaemon(proxy_port=9378, dashboard_port=9378)

    with pytest.raises(ValueError, match="distinct ports"):
        security.EntrolyDaemon(mcp_port=9378, dashboard_port=9378)

    with pytest.raises(ValueError, match="distinct ports"):
        security.EntrolyDaemon(proxy_port=9379, mcp_port=9379)


def test_disabled_services_do_not_create_false_port_collisions() -> None:
    daemon = security.EntrolyDaemon(
        proxy_port=9378,
        dashboard_port=9378,
        mcp_port=9378,
        enable_proxy=False,
        enable_mcp=False,
    )

    assert daemon.state.dashboard.port == 9378


def test_worker_launch_revalidates_host_after_runtime_mutation() -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._host = "0.0.0.0"

    with pytest.raises(ValueError, match="loopback"):
        daemon._start_proxy_worker()

    with pytest.raises(ValueError, match="loopback"):
        daemon._start_mcp_worker()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", 1.0),
        ("2.5", 2.5),
        ("300", 300.0),
        ("0", security._DEFAULT_START_TIMEOUT_S),
        ("301", security._DEFAULT_START_TIMEOUT_S),
        ("nan", security._DEFAULT_START_TIMEOUT_S),
        ("inf", security._DEFAULT_START_TIMEOUT_S),
        ("bad", security._DEFAULT_START_TIMEOUT_S),
    ],
)
def test_startup_timeout_environment_is_finite_and_bounded(
    monkeypatch, raw: str, expected: float
) -> None:
    monkeypatch.setenv("ENTROLY_DAEMON_START_TIMEOUT", raw)

    assert security._startup_timeout() == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.5", 0.5),
        ("4", 4.0),
        ("60", 60.0),
        ("0", security._DEFAULT_STOP_TIMEOUT_S),
        ("61", security._DEFAULT_STOP_TIMEOUT_S),
        ("nan", security._DEFAULT_STOP_TIMEOUT_S),
        ("inf", security._DEFAULT_STOP_TIMEOUT_S),
        ("bad", security._DEFAULT_STOP_TIMEOUT_S),
    ],
)
def test_stop_timeout_environment_is_finite_and_bounded(
    monkeypatch, raw: str, expected: float
) -> None:
    monkeypatch.setenv("ENTROLY_DAEMON_STOP_TIMEOUT", raw)

    assert security._stop_timeout() == expected


def test_occupied_dashboard_port_fails_before_engine_creation(
    monkeypatch,
) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    daemon = security.EntrolyDaemon(
        dashboard_port=port,
        enable_proxy=False,
        enable_mcp=False,
    )
    engine_created = False

    def unexpected_engine():
        nonlocal engine_created
        engine_created = True
        raise AssertionError("engine creation must not precede bind preflight")

    monkeypatch.setattr(server_module, "EntrolyEngine", unexpected_engine)
    try:
        with pytest.raises(RuntimeError, match="dashboard.*unavailable"):
            daemon.start()
    finally:
        blocker.close()

    assert not engine_created
    assert daemon.state.status == "stopped"
    assert daemon.state.started_at is None


def test_dashboard_start_failure_rolls_back_without_browser_success(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    dummy_dashboard = SimpleNamespace(shutdown_called=False)

    def shutdown() -> None:
        dummy_dashboard.shutdown_called = True

    dummy_dashboard.shutdown = shutdown
    browser_calls: list[str] = []

    monkeypatch.setattr(daemon, "_preflight_listeners", lambda: None)
    monkeypatch.setattr(server_module, "EntrolyEngine", lambda: object())
    monkeypatch.setattr(daemon, "_index_repos", lambda: None)
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: browser_calls.append(url),
    )

    def fail_dashboard() -> None:
        daemon._dashboard_server = dummy_dashboard
        daemon.state.dashboard.error = "dashboard bind failed"

    monkeypatch.setattr(daemon, "_start_dashboard_worker", fail_dashboard)

    with pytest.raises(RuntimeError, match="dashboard failed"):
        daemon.start()

    assert daemon.state.status == "stopped"
    assert daemon.state.started_at is None
    assert not daemon.state.dashboard.running
    assert dummy_dashboard.shutdown_called
    assert browser_calls == []


def test_dead_proxy_thread_fails_readiness_and_rolls_back_dashboard(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=True, enable_mcp=False)
    dummy_dashboard = SimpleNamespace(shutdown_called=False)
    dummy_dashboard.shutdown = lambda: setattr(
        dummy_dashboard,
        "shutdown_called",
        True,
    )

    monkeypatch.setattr(daemon, "_preflight_listeners", lambda: None)
    monkeypatch.setattr(server_module, "EntrolyEngine", lambda: object())
    monkeypatch.setattr(daemon, "_index_repos", lambda: None)

    def start_dashboard() -> None:
        daemon._dashboard_server = dummy_dashboard
        daemon.state.dashboard.running = True

    def start_proxy() -> None:
        worker = threading.Thread(target=lambda: None, name="dead-proxy")
        worker.start()
        worker.join()
        daemon._workers["proxy"] = worker

    def wait_listener(service: str, *, host: str, port: int) -> None:
        del host, port
        if service == "dashboard":
            return
        security.EntrolyDaemon._wait_listener(
            daemon,
            service,
            host="127.0.0.1",
            port=int(daemon.state.proxy.port),
        )

    monkeypatch.setattr(daemon, "_start_dashboard_worker", start_dashboard)
    monkeypatch.setattr(daemon, "_start_proxy_worker", start_proxy)
    monkeypatch.setattr(daemon, "_wait_listener", wait_listener)

    with pytest.raises(RuntimeError, match="proxy worker exited"):
        daemon.start()

    assert daemon.state.status == "stopped"
    assert not daemon.state.dashboard.running
    assert not daemon.state.proxy.running
    assert dummy_dashboard.shutdown_called


def test_readiness_timeout_is_visible_and_never_claims_running(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    dummy_dashboard = SimpleNamespace(shutdown_called=False)
    dummy_dashboard.shutdown = lambda: setattr(
        dummy_dashboard,
        "shutdown_called",
        True,
    )

    monkeypatch.setattr(daemon, "_preflight_listeners", lambda: None)
    monkeypatch.setattr(server_module, "EntrolyEngine", lambda: object())
    monkeypatch.setattr(daemon, "_index_repos", lambda: None)

    def start_dashboard() -> None:
        daemon._dashboard_server = dummy_dashboard
        daemon.state.dashboard.running = True

    monkeypatch.setattr(daemon, "_start_dashboard_worker", start_dashboard)
    monkeypatch.setattr(
        daemon,
        "_wait_listener",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("dashboard readiness timeout")
        ),
    )

    with pytest.raises(TimeoutError, match="readiness timeout"):
        daemon.start()

    assert daemon.state.status == "stopped"
    assert daemon.state.started_at is None
    assert dummy_dashboard.shutdown_called


def test_success_state_and_browser_open_only_after_all_readiness_gates(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=True, enable_mcp=True)
    events: list[str] = []
    browser_calls: list[str] = []

    monkeypatch.setattr(daemon, "_preflight_listeners", lambda: events.append("preflight"))
    monkeypatch.setattr(server_module, "EntrolyEngine", lambda: object())
    monkeypatch.setattr(daemon, "_index_repos", lambda: events.append("index"))

    def start_dashboard() -> None:
        events.append("start-dashboard")
        daemon.state.dashboard.running = True

    def start_proxy() -> None:
        events.append("start-proxy")
        daemon.state.proxy.running = True
        daemon._workers["proxy"] = threading.current_thread()

    def start_mcp() -> None:
        events.append("start-mcp")
        daemon.state.mcp.running = True
        daemon._workers["mcp"] = threading.current_thread()

    def wait_listener(service: str, **_kwargs) -> None:
        assert daemon.state.status == "starting"
        assert browser_calls == []
        events.append(f"ready-{service}")

    monkeypatch.setattr(daemon, "_start_dashboard_worker", start_dashboard)
    monkeypatch.setattr(daemon, "_start_proxy_worker", start_proxy)
    monkeypatch.setattr(daemon, "_start_mcp_worker", start_mcp)
    monkeypatch.setattr(daemon, "_wait_listener", wait_listener)
    monkeypatch.setattr(daemon, "_start_watcher", lambda: events.append("watcher"))
    monkeypatch.setattr(daemon, "_start_learning_loop", lambda: events.append("learning"))
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: browser_calls.append(url),
    )

    daemon.start()

    assert daemon.state.status == "running"
    assert browser_calls == [f"http://127.0.0.1:{daemon.state.dashboard.port}"]
    assert events == [
        "preflight",
        "index",
        "start-dashboard",
        "ready-dashboard",
        "start-proxy",
        "ready-proxy",
        "start-mcp",
        "ready-mcp",
        "watcher",
        "learning",
    ]


def test_start_and_stop_share_one_lifecycle_lock(monkeypatch) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon.state.status = "running"
    monkeypatch.setattr(
        daemon,
        "_preflight_listeners",
        lambda: pytest.fail("invalid state reached preflight"),
    )

    with pytest.raises(RuntimeError, match="cannot start"):
        daemon.start()

    locked = security.EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    assert locked._lifecycle_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="lifecycle operation"):
            locked.start()
        locked.state.status = "running"
        with pytest.raises(RuntimeError, match="lifecycle operation"):
            locked.stop()
    finally:
        locked._lifecycle_lock.release()


class _ExitServer:
    def __init__(self) -> None:
        self.should_exit = False
        self.force_exit = False


class _DashboardServer:
    def __init__(self, *, fail: bool = False) -> None:
        self.shutdown_called = False
        self.close_called = False
        self.fail = fail

    def shutdown(self) -> None:
        self.shutdown_called = True
        if self.fail:
            raise RuntimeError("dashboard refused shutdown")

    def server_close(self) -> None:
        self.close_called = True


class _StuckWorker:
    def __init__(self) -> None:
        self.join_calls: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return True


def _cooperative_worker(server: _ExitServer) -> None:
    while not server.should_exit:
        time.sleep(0.005)


def test_stop_closes_owned_servers_and_clears_truthful_state(monkeypatch) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=True, enable_mcp=True)
    daemon.state.status = "running"
    daemon.state.started_at = time.time()
    daemon.state.proxy.running = True
    daemon.state.dashboard.running = True
    daemon.state.mcp.running = True
    daemon._engine = object()
    dashboard_server = _DashboardServer()
    proxy_server = _ExitServer()
    mcp_server = _ExitServer()
    daemon._dashboard_server = dashboard_server
    daemon._proxy_server = proxy_server
    daemon._mcp_server = mcp_server
    daemon._workers["proxy"] = threading.Thread(
        target=_cooperative_worker,
        args=(proxy_server,),
        name="cooperative-proxy",
    )
    daemon._workers["mcp"] = threading.Thread(
        target=_cooperative_worker,
        args=(mcp_server,),
        name="cooperative-mcp",
    )
    for worker in daemon._workers.values():
        worker.start()
    monkeypatch.setattr(security, "_stop_timeout", lambda: 0.5)

    daemon.stop()

    assert proxy_server.should_exit
    assert mcp_server.should_exit
    assert dashboard_server.shutdown_called
    assert dashboard_server.close_called
    assert daemon._workers == {}
    assert daemon.state.status == "stopped"
    assert daemon.state.started_at is None
    assert daemon._engine is None
    assert not daemon.state.proxy.running
    assert not daemon.state.dashboard.running
    assert not daemon.state.mcp.running


def test_stuck_worker_forces_exit_and_preserves_running_failure_state(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=True)
    daemon.state.status = "running"
    daemon.state.mcp.running = True
    stuck = _StuckWorker()
    server = _ExitServer()
    daemon._workers["mcp"] = stuck
    daemon._mcp_server = server
    monkeypatch.setattr(security, "_stop_timeout", lambda: 0.5)

    with pytest.raises(RuntimeError, match="mcp"):
        daemon.stop()

    assert server.should_exit
    assert server.force_exit
    assert len(stuck.join_calls) == 2
    assert daemon.state.status == "stop_failed"
    assert daemon.state.mcp.running
    assert daemon.state.mcp.error == "worker did not stop within timeout"


def test_dashboard_close_failure_is_not_reported_as_stopped() -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon.state.status = "running"
    daemon.state.dashboard.running = True
    daemon._dashboard_server = _DashboardServer(fail=True)

    with pytest.raises(RuntimeError, match="dashboard"):
        daemon.stop()

    assert daemon.state.status == "stop_failed"
    assert daemon.state.dashboard.running
    assert "refused shutdown" in str(daemon.state.dashboard.error)


def test_startup_rollback_keeps_stop_failed_when_worker_is_stuck(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=True, enable_mcp=False)
    daemon.state.status = "starting"
    daemon.state.started_at = time.time()
    daemon.state.proxy.running = True
    daemon._workers["proxy"] = _StuckWorker()
    daemon._proxy_server = _ExitServer()
    monkeypatch.setattr(security, "_stop_timeout", lambda: 0.5)

    failures = daemon._rollback_startup()

    assert failures == ["proxy"]
    assert daemon.state.status == "stop_failed"
    assert daemon.state.started_at is not None
    assert daemon.state.proxy.running


def test_mcp_worker_uses_owned_sse_app_and_uvicorn_shutdown(
    monkeypatch,
) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=True)
    daemon._engine = object()
    daemon.state.status = "running"
    started = threading.Event()
    observed: dict[str, object] = {}

    class FakeMCP:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(host=None, port=None)
            self.app = object()

        def sse_app(self):
            observed["sse_app_called"] = True
            return self.app

    fake_mcp = FakeMCP()

    class FakeUvicornServer(_ExitServer):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            observed["server"] = self

        def run(self) -> None:
            started.set()
            while not self.should_exit:
                time.sleep(0.005)

    monkeypatch.setattr(
        server_module,
        "create_mcp_server",
        lambda engine: (fake_mcp, engine),
    )
    monkeypatch.setattr(
        uvicorn,
        "Config",
        lambda app, **kwargs: SimpleNamespace(app=app, kwargs=kwargs),
    )
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr(security, "_stop_timeout", lambda: 0.5)

    daemon._start_mcp_worker()
    assert started.wait(timeout=2)
    assert daemon.state.mcp.running
    assert daemon._mcp_server is observed["server"]
    assert observed["sse_app_called"] is True
    assert fake_mcp.settings.host == "127.0.0.1"
    assert fake_mcp.settings.port == daemon.state.mcp.port
    config = observed["server"].config
    assert config.app is fake_mcp.app
    assert config.kwargs["host"] == "127.0.0.1"
    assert config.kwargs["port"] == daemon.state.mcp.port

    daemon.stop()

    assert daemon.state.status == "stopped"
    assert daemon._workers == {}
    assert daemon._mcp_server is None


def test_missing_sse_app_fails_worker_auditably(monkeypatch) -> None:
    daemon = security.EntrolyDaemon(enable_proxy=False, enable_mcp=True)
    daemon._engine = object()
    fake_mcp = SimpleNamespace(settings=SimpleNamespace(host=None, port=None))
    monkeypatch.setattr(
        server_module,
        "create_mcp_server",
        lambda engine: (fake_mcp, engine),
    )

    daemon._start_mcp_worker()
    worker = daemon._workers["mcp"]
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert not daemon.state.mcp.running
    assert "sse_app" in str(daemon.state.mcp.error)
