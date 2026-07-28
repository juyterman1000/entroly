from __future__ import annotations

import socket
import threading
from types import SimpleNamespace

import pytest

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
    assert browser_calls == [f"http://localhost:{daemon.state.dashboard.port}"]
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


def test_start_is_single_entry_and_rejects_invalid_state(monkeypatch) -> None:
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
    assert locked._startup_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            locked.start()
    finally:
        locked._startup_lock.release()
