"""The codec contract and exact full-source recovery invariants."""

from __future__ import annotations

import json
from dataclasses import replace

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
        "2026-08-02T10:00:00Z INFO worker starting pool_size=8",
        "2026-08-02T10:00:03Z ERROR db connect failed: FATAL password "
        "authentication failed for user 'svc_billing'",
    ]
    lines.extend(
        f"2026-08-02T10:00:{4 + i % 50:02d}Z ERROR request failed: "
        f"connection pool exhausted (retry {i})"
        for i in range(repeats)
    )
    return "\n".join(lines)


def test_elided_json_records_are_exactly_recoverable():
    original = json.dumps(_payload(), indent=2)
    store = RecoveryStore()
    reps = JsonCodec(store).representations(original, source_id="resp.json")
    elided = next(rep for rep in reps if rep.representation_id.endswith("elided"))
    assert elided.recovery is not None
    recovered = store.recover(elided.recovery)
    assert recovered == original
    assert elided.recovery.item_count == 39
    assert "SKU-0039" in recovered
    assert elided.recovery.verify(recovered)


def test_collapsed_log_repeats_are_exactly_recoverable():
    original = _log_text()
    store = RecoveryStore()
    reps = LogCodec(store).representations(original, source_id="worker.log")
    collapsed = next(rep for rep in reps if rep.representation_id.endswith("collapsed"))
    assert collapsed.recovery is not None
    recovered = store.recover(collapsed.recovery)
    assert recovered == original
    assert collapsed.recovery.item_count == 199
    assert "(retry 199)" in recovered


def test_recovery_reference_detects_a_forged_digest():
    store = RecoveryStore()
    ref = store.put("the original", item_count=1)
    forged = replace(ref, digest=content_digest("something else"))
    with pytest.raises(ValueError, match="does not match"):
        store.recover(forged)


def test_recovery_survives_a_process_via_hardened_store(tmp_path):
    path = tmp_path / "recovery.json"
    first = RecoveryStore(path, scope_id="codec-test")
    ref = first.put("dropped records", item_count=3)
    restarted = RecoveryStore(path, scope_id="codec-test")
    assert restarted.recover(ref) == "dropped records"


def test_codec_never_claims_sufficiency():
    assert not hasattr(Representation, "sufficient")
    fields = set(Representation.__dataclass_fields__)
    for forbidden in ("sufficient", "sufficiency", "verdict", "is_enough"):
        assert forbidden not in fields


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
    assert verify_all(reps) == {}
    elided = next(rep for rep in reps if rep.representation_id.endswith("elided"))
    for value in ("req_8f3a21bc", "PAYMENT_DECLINED", "449900"):
        assert value in elided.text


def test_full_representation_is_offered_and_lossless():
    text = json.dumps(_payload(), indent=2)
    reps = JsonCodec().representations(text, source_id="r")
    full = next(rep for rep in reps if rep.representation_id.endswith("full"))
    assert full.text == text
    assert full.distortion_risk == 0.0
    assert full.recovery is None


def test_elided_form_is_never_larger_than_original():
    tiny = '{"a": 1}'
    reps = JsonCodec().representations(tiny, source_id="tiny")
    assert all(len(rep.text) <= len(tiny) for rep in reps)


def test_malformed_json_is_declined_not_rewritten():
    broken = '{"a": 1, "b": [1, 2,,]}'
    assert not JsonCodec().supports(broken)


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
    data = next(rep for rep in reps if rep.representation_id.endswith("elided")).to_dict()
    assert data["contract_version"] == CODEC_CONTRACT_VERSION
    assert data["source_sha256"].startswith("sha256:")
    assert data["codec"] == "json" and data["codec_version"]
    assert data["recovery"]["item_count"] == 39
    assert data["recovery"]["receipt_id"]
    assert data["recovery"]["span_id"]
    assert 0.0 <= data["distortion_risk"] <= 1.0


def test_registry_is_empty_by_default_rather_than_guessing():
    assert CodecRegistry().select("anything") is None


def _pytest_output(passing: int = 200) -> str:
    lines = ["$ pytest tests/", "tests/test_a.py::test_one PASSED"]
    lines.extend(f"tests/test_c.py::test_{i} PASSED" for i in range(passing))
    lines.extend(
        [
            "tests/test_z.py::test_bad FAILED",
            "E AssertionError: expected 3 got 4",
            f"1 failed, {passing + 1} passed",
            "exit code 1",
        ]
    )
    return "\n".join(lines)


@pytest.mark.parametrize(
    ("label", "needle"),
    [
        ("command", "pytest tests/"),
        ("failed target", "test_bad"),
        ("failure marker", "FAILED"),
        ("error text", "AssertionError"),
        ("summary counts", "1 failed"),
        ("exit status", "exit code 1"),
    ],
)
def test_shell_codec_keeps_failure_evidence(label, needle):
    from entroly.codecs_builtin import ShellCodec

    rep = ShellCodec().representations(_pytest_output(), source_id="run")[-1]
    assert needle in rep.text, f"{label} was dropped"


def test_shell_codec_complete_original_is_recoverable():
    from entroly.codecs_builtin import ShellCodec

    original = _pytest_output()
    store = RecoveryStore()
    rep = ShellCodec(store).representations(original, source_id="run")[-1]
    assert rep.recovery is not None and rep.recovery.item_count > 100
    assert store.recover(rep.recovery) == original


def test_shell_codec_protected_evidence_is_real():
    from entroly.codecs_builtin import ShellCodec

    reps = ShellCodec().representations(_pytest_output(), source_id="run")
    assert verify_all(reps) == {}
    compressed = reps[-1]
    assert any("exit code" in evidence for evidence in compressed.protected_evidence)
    assert any("failed" in evidence.lower() for evidence in compressed.protected_evidence)


def test_prose_is_not_claimed_by_shell_codec():
    from entroly.codecs_builtin import ShellCodec

    assert not ShellCodec().supports(
        "A quiet paragraph about nothing in particular, with no command in it."
    )
