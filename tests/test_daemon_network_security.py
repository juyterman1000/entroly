from __future__ import annotations

import pytest

import entroly.daemon as daemon_module
import entroly.daemon_network_security as security


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
