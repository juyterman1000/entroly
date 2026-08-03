"""`compress_with_receipt` — the codec contract reachable from the SDK.

Until this existed, every codec was a tested contract with no production
caller: nothing in the SDK, MCP, proxy or CLI could reach the registry, so a
user could not obtain provenance or recover what was dropped. `compress()`
returns a string, which cannot answer either question.

The properties below are what make compression safe to deploy, so each is
asserted against the public surface rather than against internals.
"""

from __future__ import annotations

import json

import pytest

import entroly
from entroly.codec import RecoveryStore


def _json_payload() -> str:
    return json.dumps(
        {
            "request_id": "req_8f3a21bc",
            "error": {"code": "PAYMENT_DECLINED"},
            "amount_cents": 449900,
            "items": [{"sku": f"SKU-{i:04d}", "price_cents": 1999 + i} for i in range(40)],
        },
        indent=2,
    )


def _log_payload() -> str:
    lines = ["2026-08-02T10:00:03Z ERROR db connect failed: FATAL bad password"]
    lines += [
        f"2026-08-02T10:00:{4 + i % 50:02d}Z ERROR pool exhausted (retry {i})"
        for i in range(120)
    ]
    return "\n".join(lines)


def test_exported_from_the_package_root():
    assert hasattr(entroly, "compress_with_receipt")


def test_returns_provenance_not_just_text():
    rep = entroly.compress_with_receipt(_json_payload(), source_id="resp.json")
    assert rep.text
    assert rep.source_sha256.startswith("sha256:")
    assert rep.codec and rep.codec_version
    assert 0.0 <= rep.distortion_risk <= 1.0


@pytest.mark.parametrize(
    "label,payload", [("json", _json_payload()), ("log", _log_payload())]
)
def test_dropped_bytes_are_recoverable_through_the_public_surface(label, payload):
    store = RecoveryStore()
    rep = entroly.compress_with_receipt(payload, source_id=label, store=store)
    if rep.recovery is None:
        assert rep.text == payload, "no recovery reference is only valid if lossless"
        return
    assert store.recover(rep.recovery) == payload, (
        f"{label}: recovery must return the original byte stream exactly"
    )


def test_protected_evidence_is_true_of_the_returned_text():
    rep = entroly.compress_with_receipt(_json_payload(), source_id="r")
    assert rep.verify_protected_evidence() == (), (
        "a codec that lists evidence it did not keep is worse than one that "
        "lists none"
    )


def test_unknown_content_is_returned_verbatim_not_mangled():
    prose = "A short note about nothing structured at all."
    rep = entroly.compress_with_receipt(prose)
    assert rep.text == prose
    assert rep.codec == "passthrough"
    assert rep.distortion_risk == 0.0


def test_never_returns_a_lossy_form_without_a_way_back():
    """The selection rule, stated as a test.

    A smaller representation that dropped content and offers no recovery
    reference must not be chosen over a larger one that does.
    """
    for payload in (_json_payload(), _log_payload()):
        rep = entroly.compress_with_receipt(payload, source_id="x")
        assert rep.recovery is not None or rep.text == payload


def test_caller_supplied_store_is_used():
    store = RecoveryStore()
    entroly.compress_with_receipt(_json_payload(), source_id="r", store=store)
    assert len(store) > 0, (
        "the caller's store must receive the omitted bytes, or the recovery "
        "reference it was handed cannot resolve"
    )


def test_declared_content_type_is_honoured():
    rep = entroly.compress_with_receipt(
        _json_payload(), source_id="r", content_type="json"
    )
    assert rep.content_type == "json"


def test_compress_still_returns_a_plain_string():
    """The existing surface is unchanged; this is additive."""
    out = entroly.compress(_json_payload(), budget=200)
    assert isinstance(out, str)
