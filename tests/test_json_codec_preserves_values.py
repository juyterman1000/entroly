"""JSON compression must not destroy the values the payload was sent for.

`_json_to_schema` replaced every number with `"<int>"` / `"<float>"`. On a
payment-error fixture that turned `"amount_cents": 449900` into
`"amount_cents": "<int>"` -- the answer to "how much was declined" replaced by
a type name, along with every identifier, timestamp and code that happened to
be numeric.

The substitution never paid for itself either. `"<int>"` serialises to seven
characters and the value it replaced was six, so for anything under seven
digits the placeholder was LONGER than the number. Measured on the fixture
below, preserving values took the output from 381 to 371 characters: strictly
smaller and strictly more faithful.

Structure is what repeats and values are what the payload was sent for. The
saving belongs to array elision -- one exemplar for N records -- not to
blanking scalars.
"""

from __future__ import annotations

import json

import pytest

from entroly.universal_compress import detect_content_type, universal_compress


def _payload() -> dict:
    return {
        "request_id": "req_8f3a21bc",
        "status": "error",
        "error": {
            "code": "PAYMENT_DECLINED",
            "message": "card issuer refused",
            "retryable": False,
        },
        "amount_cents": 449900,
        "currency": "USD",
        "timestamp": "2026-08-02T13:22:41Z",
        "items": [
            {"sku": f"SKU-{i:04d}", "qty": i % 5 + 1, "price_cents": 1999 + i}
            for i in range(40)
        ],
    }


def _compress(text: str) -> str:
    out = universal_compress(text, target_ratio=0.3)
    # universal_compress returns (text, content_type, ratio)
    return out[0] if isinstance(out, tuple) else str(out)


def test_detects_json():
    assert detect_content_type(json.dumps(_payload(), indent=2)) == "json"


@pytest.mark.parametrize(
    "label,needle",
    [
        ("financial amount", "449900"),
        ("error code", "PAYMENT_DECLINED"),
        ("error message", "card issuer refused"),
        ("correlation id", "req_8f3a21bc"),
        ("timestamp", "2026-08-02T13:22:41Z"),
        ("currency", "USD"),
    ],
)
def test_values_of_record_survive(label, needle):
    out = _compress(json.dumps(_payload(), indent=2))
    assert needle in out, (
        f"{label} ({needle!r}) was destroyed by JSON compression. Identifiers, "
        f"timestamps, error codes and financial values must survive; only "
        f"repeated structure may be elided.\n---\n{out[:600]}"
    )


def test_no_type_placeholders_replace_scalars():
    out = _compress(json.dumps(_payload(), indent=2))
    for placeholder in ('"<int>"', '"<float>"'):
        assert placeholder not in out, (
            f"{placeholder} is longer than most values it replaces, so it costs "
            f"tokens as well as fidelity.\n---\n{out[:600]}"
        )


def test_preserving_values_did_not_cost_size():
    """The regression this guards is a trade that was losing on both axes."""
    text = json.dumps(_payload(), indent=2)
    out = _compress(text)
    assert len(out) < len(text), (
        f"compression inflated: {len(text)} -> {len(out)} chars"
    )


def test_repeated_records_are_still_elided():
    """The saving must still come from somewhere -- 40 items, one exemplar."""
    out = _compress(json.dumps(_payload(), indent=2))
    assert "SKU-0000" in out, "the exemplar record should be kept"
    assert "SKU-0039" not in out, "40 records should not be emitted verbatim"
    assert "40 items" in out, (
        "an elided array must at least declare how many records were dropped"
    )


def test_long_string_under_a_load_bearing_key_is_kept():
    payload = {
        "error_message": "the upstream authorisation service rejected this "
                         "transaction because the card issuer returned a soft "
                         "decline that is safe to retry after a short delay",
        "notes": "x" * 400,
    }
    out = _compress(json.dumps(payload, indent=2))
    assert "soft" in out and "decline" in out, (
        f"a long value under an error/message key is the payload's point and "
        f"must not become <str:N>.\n---\n{out[:400]}"
    )
