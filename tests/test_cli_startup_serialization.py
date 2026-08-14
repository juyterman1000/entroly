from __future__ import annotations

import pytest

from entroly.cli_startup_serialization import (
    proxy_start_lock_path,
    serialized_proxy_start,
)


def test_waiting_wrapper_reuses_proxy_after_health_recheck(tmp_path):
    calls = []

    ok, outcome = serialized_proxy_start(
        port=9377,
        runtime_dir=tmp_path,
        is_running=lambda _port: True,
        start=lambda port: calls.append(port) or True,
    )

    assert ok is True
    assert outcome == "reused"
    assert calls == []


def test_first_wrapper_starts_exactly_once(tmp_path):
    calls = []

    ok, outcome = serialized_proxy_start(
        port=9377,
        runtime_dir=tmp_path,
        is_running=lambda _port: False,
        start=lambda port: calls.append(port) or True,
    )

    assert ok is True
    assert outcome == "started"
    assert calls == [9377]


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_startup_lock_rejects_invalid_ports(tmp_path, port):
    with pytest.raises(ValueError):
        proxy_start_lock_path(tmp_path, port)
