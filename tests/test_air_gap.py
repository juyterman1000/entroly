"""Tests for air-gap mode (entroly/air_gap.py).

Proves the checkable guarantee: with the guard installed, a non-loopback
connection is refused while loopback is allowed — the enforceable core of the
"no context ever leaves the machine" claim.
"""

from __future__ import annotations

import socket

import pytest

from entroly import air_gap


@pytest.fixture()
def guard():
    air_gap.uninstall_air_gap_guard()
    air_gap.install_air_gap_guard(force=True)
    try:
        yield
    finally:
        air_gap.uninstall_air_gap_guard()


def test_blocks_non_loopback_connect(guard):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(air_gap.AirGapViolation, match="refused outbound"):
            s.connect(("8.8.8.8", 53))
        with pytest.raises(air_gap.AirGapViolation):
            s.connect(("example.com", 80))  # hostname → deny (could resolve anywhere)
    finally:
        s.close()


def test_blocks_connect_ex_too(guard):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(air_gap.AirGapViolation):
            s.connect_ex(("1.1.1.1", 443))
    finally:
        s.close()


def test_allows_loopback(guard):
    # A loopback connect must pass the guard (it fails with a normal connection
    # error because nothing is listening — NOT an AirGapViolation).
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        with pytest.raises(OSError) as exc:
            s.connect(("127.0.0.1", 1))
        assert not isinstance(exc.value, air_gap.AirGapViolation)
    finally:
        s.close()


def test_guard_is_reversible_and_idempotent():
    original = socket.socket.connect
    assert air_gap.install_air_gap_guard(force=True) is True
    assert socket.socket.connect is not original
    # Second install is a no-op (does not double-wrap).
    guarded = socket.socket.connect
    assert air_gap.install_air_gap_guard(force=True) is True
    assert socket.socket.connect is guarded
    air_gap.uninstall_air_gap_guard()
    assert socket.socket.connect is original


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    air_gap.uninstall_air_gap_guard()
    assert air_gap.air_gap_enabled() is False
    assert air_gap.install_air_gap_guard() is False  # not forced, env off → no-op
    assert air_gap.air_gap_status()["enforced"] is False


def test_status_reports_enforcement(guard):
    st = air_gap.air_gap_status()
    assert st["enforced"] is True
    assert st["allowed"] == "loopback-only"


def test_env_var_recognized(monkeypatch):
    for val in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ENTROLY_AIR_GAP", val)
        assert air_gap.air_gap_enabled() is True
    monkeypatch.setenv("ENTROLY_AIR_GAP", "0")
    assert air_gap.air_gap_enabled() is False
