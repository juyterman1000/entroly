"""The codec contract, and the exact-recovery gap it closes.

Both shipped codecs elided real content and reported only a count:

    "... (40 items)"                       39 records, unrecoverable
    "connection pool exhausted  [x200]"    199 lines, unrecoverable

A count is an admission, not a recovery reference. These tests hold the codecs
to the contract: what they drop goes to a content-addressed store, and the
reference verifies against the bytes that come back.

Two properties matter more than the compression:

* a codec must not claim sufficiency -- it reports what it did, and the
  decision about whether that is enough belongs to a caller that can see the
  whole selection;
* unknown or malformed content must decline rather than be rewritten by a
  codec that only assumed it could parse it.
"""

from __future__ import annotations

import json

import pytest

from entroly.codec import (
    CODEC_CONTRACT_VERSION,
    CodecRegistry,
    RecoveryStore,
    Representation,
    content_digest,
    verify_all,
)
from entroly.codecs_builtin import JsonCodec, LogCodec, default_registry


def _payload() -> dict:
    return {
        "request_id": "req_8f3a21bc",
        "error": {"code": "PAYMENT_DECLINED", "message": "card issuer refused"},
        "amount_cents": 449900,
        "items": [
            {"sku": f"SKU-{i:04d}", "qty": i % 5 + 1, "price_cents": 1999 + i}
            for i in range(40)
        ],
    }


def _log_text(repeats: int = 200) -> str:
    lines = [
        "2026-08-02T10:00:00Z INFO  worker starting pool_size=8",
        "2026-08-02T10:00:03Z ERROR db connect failed: FATAL password "
        "authentication failed for user 'svc_billing'",
    ]
    for i in range(repeats):
        lines.append(
            f"2026-08-02T10:00:{4 + i % 50:02d}Z ERROR request failed: "
            f"connection pool exhausted (retry {i})"
        )
    return "\n".join(lines)


# ── Recovery ────────────────────────────────────────────────────────────────


def test_elided_json_records_are_exactly_recoverable():
    store = RecoveryStore()
    reps = JsonCodec(store).representations(
        json.dumps(_payload(), indent=2), source_id="resp.json"
    )
    elided = next(r for r in reps if r.representation_id.endswith("elided"))
    assert elided.recovery is not None, (
        "the schema form drops 39 of 40 records; without a recovery reference "
        "they are simply gone"
    )
    recovered = store.recover(elided.recovery)
    assert elided.recovery.item_count == 39
    assert "SKU-0039" in recovered, "the dropped records must come back intact"
    assert elided.recovery.verify(recovered), "digest must match the bytes returned"


def test_collapsed_log_repeats_are_exactly_recoverable():
    store = RecoveryStore()
    reps = LogCodec(store).representations(_log_text(), source_id="worker.log")
    collapsed = next(r for r in reps if r.representation_id.endswith("collapsed"))
    assert collapsed.recovery is not None
    recovered = store.recover(collapsed.recovery)
    assert collapsed.recovery.item_count == 199, (
        f"200 occurrences, one kept, 199 collapsed; got "
        f"{collapsed.recovery.item_count}"
    )
    assert "(retry 199)" in recovered


def test_recovery_reference_detects_a_swapped_entry():
    """Content addressing must catch a store that returns the wrong thing."""
    store = RecoveryStore()
    ref = store.put("the original", item_count=1)
    store._mem[ref.digest] = "something else"
    with pytest.raises(ValueError, match="does not match"):
        store.recover(ref)


def test_recovery_survives_a_process_via_the_sidecar(tmp_path):
    path = tmp_path / "recovery.json"
    ref = RecoveryStore(path).put("dropped records", item_count=3)
    assert RecoveryStore(path).recover(ref) == "dropped records"


# ── The contract itself ─────────────────────────────────────────────────────


def test_codec_never_claims_sufficiency():
    """A codec sees one item; sufficiency is a property of the whole selection."""
    assert not hasattr(Representation, "sufficient")
    fields = set(Representation.__dataclass_fields__)
    for forbidden in ("sufficient", "sufficiency", "verdict", "is_enough"):
        assert forbidden not in fields, (
            f"Representation.{forbidden} would let a codec judge sufficiency "
            f"from a single item"
        )


def test_protected_evidence_is_checkable_not_just_asserted():
    rep = Representation(
        representation_id="x",
        source_id="s",
        content_type="text",
        text="kept alpha, dropped beta",
        token_cost=6,
        codec="test",
        codec_version="1",
        source_sha256=content_digest("whatever"),
        protected_evidence=("alpha", "gamma"),
    )
    assert rep.verify_protected_evidence() == ("gamma",)
    assert verify_all([rep]) == {"x": ("gamma",)}


def test_json_codec_actually_preserves_what_it_protects():
    reps = JsonCodec().representations(json.dumps(_payload(), indent=2), source_id="r")
    assert verify_all(reps) == {}, (
        "a codec that lists protected evidence it did not keep is worse than "
        "one that lists none"
    )
    elided = next(r for r in reps if r.representation_id.endswith("elided"))
    for value in ("req_8f3a21bc", "PAYMENT_DECLINED", "449900"):
        assert value in elided.text


def test_full_representation_is_offered_and_lossless():
    text = json.dumps(_payload(), indent=2)
    reps = JsonCodec().representations(text, source_id="r")
    full = next(r for r in reps if r.representation_id.endswith("full"))
    assert full.text == text
    assert full.distortion_risk == 0.0
    assert full.recovery is None, "nothing was dropped, so nothing to recover"


def test_elided_form_is_never_larger_than_the_original():
    """Anti-inflation: a bigger 'compressed' form is not an option."""
    tiny = '{"a": 1}'
    reps = JsonCodec().representations(tiny, source_id="tiny")
    assert all(len(r.text) <= len(tiny) for r in reps), (
        f"emitted a representation larger than the input: "
        f"{[(r.representation_id, len(r.text)) for r in reps]}"
    )


# ── Unknown and malformed input ─────────────────────────────────────────────


def test_malformed_json_is_declined_not_rewritten():
    broken = '{"a": 1, "b": [1, 2,,]}'
    assert not JsonCodec().supports(broken), (
        "content that opens like JSON but does not parse must be declined; "
        "rewriting it would corrupt something the codec never understood"
    )


def test_registry_selects_nothing_for_unknown_content():
    registry = default_registry()
    prose = "The quick brown fox jumps over the lazy dog, repeatedly and at length."
    assert registry.select(prose) is None
    assert registry.representations(prose, source_id="note.txt") == []


def test_registry_routes_each_type_to_its_codec():
    registry = default_registry()
    assert registry.select(json.dumps(_payload())).name == "json"
    assert registry.select(_log_text(3)).name == "log"


def test_declared_content_type_wins_over_sniffing():
    assert JsonCodec().supports("not json at all", content_type="json").confidence == 1.0


def test_representation_serialises_with_provenance():
    reps = JsonCodec().representations(json.dumps(_payload()), source_id="r.json")
    d = next(r for r in reps if r.representation_id.endswith("elided")).to_dict()
    assert d["contract_version"] == CODEC_CONTRACT_VERSION
    assert d["source_sha256"].startswith("sha256:")
    assert d["codec"] == "json" and d["codec_version"]
    assert d["recovery"]["item_count"] == 39
    assert 0.0 <= d["distortion_risk"] <= 1.0


def test_registry_is_empty_by_default_rather_than_guessing():
    assert CodecRegistry().select("anything") is None
