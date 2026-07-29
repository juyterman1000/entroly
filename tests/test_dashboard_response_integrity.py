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


def _chunked_handler(
    body: bytes,
    *transfer_values: str,
):
    headers = Message()
    for value in transfer_values or ("chunked",):
        headers.add_header("Transfer-Encoding", value)
    return SimpleNamespace(
        headers=headers,
        rfile=BytesIO(body),
        close_connection=False,
    )


def test_response_integrity_patch_is_active_on_safe_handler() -> None:
    assert security.SafeDashboardHandler.do_POST is integrity._integrity_safe_do_post
    assert security.SafeDashboardHandler._respond is integrity._integrity_respond


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

    assert integrity._drain_rejected_body(handler)

    assert handler.rfile.read() == b"TAIL"
    assert handler.close_connection is False


def test_incomplete_rejected_body_forces_connection_close() -> None:
    handler = _handler("10", body=b"{}")

    assert not integrity._drain_rejected_body(handler)

    assert handler.close_connection is True


def test_ambiguous_length_forces_close_without_consuming_data() -> None:
    handler = _handler("2", "3", body=b"{}x")

    assert not integrity._drain_rejected_body(handler)

    assert handler.close_connection is True
    assert handler.rfile.read() == b"{}x"


def test_valid_bounded_chunked_body_is_drained_without_dispatch_or_reset() -> None:
    handler = _chunked_handler(
        b"2;ignored-extension=yes\r\n{}\r\n0\r\nX-Test: ok\r\n\r\nTAIL"
    )

    assert integrity._drain_rejected_body(handler)

    assert handler.close_connection is False
    assert handler.rfile.read() == b"TAIL"


def test_malformed_chunk_size_fails_closed() -> None:
    handler = _chunked_handler(b"not-hex\r\n{}\r\n0\r\n\r\n")

    assert not integrity._drain_rejected_body(handler)

    assert handler.close_connection is True


def test_oversized_chunk_fails_before_payload_read() -> None:
    body = b"10001\r\nSHOULD-NOT-BE-READ"
    handler = _chunked_handler(body)

    assert not integrity._drain_rejected_body(handler)

    assert handler.close_connection is True
    assert handler.rfile.read() == b"SHOULD-NOT-BE-READ"


def test_duplicate_or_unsupported_transfer_encoding_fails_closed() -> None:
    duplicate = _chunked_handler(b"0\r\n\r\n", "chunked", "chunked")
    unsupported = _chunked_handler(b"payload", "gzip")

    assert not integrity._drain_rejected_body(duplicate)
    assert not integrity._drain_rejected_body(unsupported)

    assert duplicate.close_connection is True
    assert unsupported.close_connection is True
    assert duplicate.rfile.read() == b"0\r\n\r\n"
    assert unsupported.rfile.read() == b"payload"


def test_incomplete_chunked_body_forces_connection_close() -> None:
    handler = _chunked_handler(b"2\r\n{")

    assert not integrity._drain_rejected_body(handler)

    assert handler.close_connection is True
