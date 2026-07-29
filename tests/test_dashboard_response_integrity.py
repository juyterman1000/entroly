from __future__ import annotations

from email.message import Message
from io import BytesIO
from types import SimpleNamespace

import entroly.dashboard_response_integrity as integrity
import entroly.dashboard_security as security


def _handler(*values: str, body: bytes = b""):
    headers = Message()
    for value in values:
        headers.add_header("Content-Length", value)
    return SimpleNamespace(
        headers=headers,
        rfile=BytesIO(body),
        close_connection=False,
    )


def test_response_integrity_patch_is_active_on_safe_handler() -> None:
    assert security.SafeDashboardHandler.do_POST is integrity._integrity_safe_do_post


def test_declared_length_accepts_one_canonical_bounded_value() -> None:
    assert integrity._bounded_declared_length(_handler()) == 0
    assert integrity._bounded_declared_length(_handler("0")) == 0
    assert integrity._bounded_declared_length(_handler("2")) == 2
    assert integrity._bounded_declared_length(_handler(" 2 ")) == 2


def test_declared_length_rejects_ambiguous_or_unbounded_values() -> None:
    assert integrity._bounded_declared_length(_handler("2", "2")) is None
    assert integrity._bounded_declared_length(_handler("+2")) is None
    assert integrity._bounded_declared_length(_handler("-1")) is None
    assert integrity._bounded_declared_length(_handler("2.0")) is None
    assert integrity._bounded_declared_length(_handler("x" * 21)) is None
    assert integrity._bounded_declared_length(_handler(str(64 * 1024 + 1))) is None


def test_rejected_body_is_consumed_exactly_without_parsing() -> None:
    handler = _handler("4", body=b"{}xxTAIL")

    integrity._drain_rejected_body(handler)

    assert handler.rfile.read() == b"TAIL"
    assert handler.close_connection is False


def test_incomplete_rejected_body_forces_connection_close() -> None:
    handler = _handler("10", body=b"{}")

    integrity._drain_rejected_body(handler)

    assert handler.close_connection is True


def test_ambiguous_length_forces_close_without_consuming_data() -> None:
    handler = _handler("2", "3", body=b"{}x")

    integrity._drain_rejected_body(handler)

    assert handler.close_connection is True
    assert handler.rfile.read() == b"{}x"


def test_chunked_rejected_body_is_never_interpreted_by_drain() -> None:
    headers = Message()
    headers.add_header("Transfer-Encoding", "chunked")
    handler = SimpleNamespace(
        headers=headers,
        rfile=BytesIO(b"2\r\n{}\r\n0\r\n\r\n"),
        close_connection=False,
    )

    integrity._drain_rejected_body(handler)

    assert handler.close_connection is True
    assert handler.rfile.read() == b"2\r\n{}\r\n0\r\n\r\n"
