from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from entroly.codec import RecoveryStore
from entroly.codecs_builtin import JsonCodec, LogCodec, ShellCodec
from entroly.qccr import _lexical_term_set
# Consolidated: the hotfix module was folded back into entroly.qccr, so
# there is one implementation rather than a legacy module plus an
# override module plus a wrapper that re-synchronised them per call.
from entroly.qccr import _attach_sufficiency as attach_sufficiency
from entroly.sufficiency import CalibrationPolicy, Candidate, certify
from entroly.universal_compress import _compress_log_universal, _json_to_schema


def _policy() -> CalibrationPolicy:
    return CalibrationPolicy(
        calibration_id="heldout-test-v1",
        residual_risk_limit=0.02,
    )


def test_json_recovery_is_complete_original_byte_stream(tmp_path):
    payload = {f"filler_{index}": index for index in range(25)}
    payload["requestId"] = "req-camel-case"
    payload["deep"] = {"a": {"b": {"c": {"d": {"errorMessage": "deep failure"}}}}}
    payload["items"] = [
        {"sku": f"SKU-{index}", "amountCents": index * 101}
        for index in range(30)
    ]
    original = json.dumps(payload, indent=2).replace("\n", "\r\n") + "\r\n"

    path = tmp_path / "secure-recovery.json"
    store = RecoveryStore(path, scope_id="test-json")
    representations = JsonCodec(store).representations(original, source_id="payload.json")
    # What matters is that a compressible form is still offered beside the
    # verbatim one, not how many exist. Pinning the exact count made this fail
    # when the codec gained a third representation -- a widening of choice, not
    # a regression. Every recoverable form is now checked rather than one.
    assert len(representations) >= 2, "safe late evidence should remain compressible"
    recoverable = [rep for rep in representations if rep.recovery is not None]
    assert recoverable, "no recoverable representation was offered"
    for candidate in recoverable:
        assert store.recover(candidate.recovery) == original

    representation = recoverable[-1]

    restarted = RecoveryStore(path, scope_id="test-json")
    assert restarted.recover(representation.recovery) == original


def test_camel_case_load_bearing_key_after_twentieth_field_survives():
    payload = {f"filler_{index}": "x" * 80 for index in range(25)}
    payload["requestId"] = "req-after-twenty"
    payload["items"] = [{"value": index} for index in range(50)]
    representation = JsonCodec().representations(
        json.dumps(payload, indent=2), source_id="late.json"
    )[-1]
    assert "req-after-twenty" in representation.text


def test_later_array_record_load_bearing_values_survive():
    payload = {
        "items": [
            {"sku": f"SKU-{index:04d}", "amountCents": 1000 + index}
            for index in range(40)
        ]
    }
    representation = JsonCodec().representations(
        json.dumps(payload, indent=2), source_id="records.json"
    )[-1]
    assert representation.representation_id.endswith("elided")
    assert "SKU-0039" in representation.text
    assert "1039" in representation.text


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("status=404", "status=500"),
        ("exit_code=1", "exit_code=137"),
        ("amount_cents=100", "amount_cents=100000"),
        ("version=1", "version=2"),
        ("port=443", "port=8443"),
    ],
)
def test_critical_numeric_log_values_never_collapse(left, right):
    text = "\n".join(
        [
            f"2026-08-02T10:00:00Z ERROR operation failed {left}",
            f"2026-08-02T10:00:01Z ERROR operation failed {right}",
        ]
    )
    output = _compress_log_universal(text)
    assert left in output and right in output
    assert "[×2]" not in output and "[x2]" not in output


def test_retry_counters_still_collapse():
    text = "\n".join(
        f"2026-08-02T10:00:{index:02d}Z ERROR connection pool exhausted retry {index}"
        for index in range(20)
    )
    output = _compress_log_universal(text)
    assert output.count("connection pool exhausted") == 1
    assert "[×20]" in output or "[x20]" in output


def test_untimestamped_level_and_event_text_are_not_stripped():
    text = "\n".join(
        [
            "ERROR billing failed retry 1",
            "WARN cache delayed retry 2",
        ]
    )
    output = _compress_log_universal(text)
    assert "ERROR billing failed" in output
    assert "WARN cache delayed" in output
    assert "[×2]" not in output


def test_event_identity_is_not_truncated_at_one_hundred_characters():
    prefix = "x" * 140
    text = "\n".join(
        [
            f"2026-08-02T10:00:00Z ERROR {prefix} alpha",
            f"2026-08-02T10:00:01Z ERROR {prefix} beta",
        ]
    )
    output = _compress_log_universal(text)
    assert "alpha" in output and "beta" in output


def test_log_codec_recovers_crlf_and_trailing_newline_exactly(tmp_path):
    original = (
        "2026-08-02T10:00:00Z ERROR retry 1\r\n"
        "2026-08-02T10:00:01Z ERROR retry 2\r\n"
    )
    store = RecoveryStore(tmp_path / "log-recovery.json", scope_id="test-log")
    representation = LogCodec(store).representations(
        original, source_id="run.log"
    )[-1]
    assert representation.recovery is not None
    assert store.recover(representation.recovery) == original


def test_shell_protection_is_derived_from_source(monkeypatch):
    original = "$ test\nFAILED important_case\nexit code 1\n"
    monkeypatch.setattr(
        "entroly.shell_codec.esc_compress",
        lambda text, budget: SimpleNamespace(compressed="$ test\n"),
    )
    representations = ShellCodec().representations(original, source_id="run")
    assert len(representations) == 1
    assert representations[0].text == original


def test_uncalibrated_certificate_never_claims_sufficient():
    candidates = [Candidate("answer", utility=1.0, cost=10, selected=True)]
    observational = certify(candidates, budget_exhausted=False)
    assert observational.verdict == "uncalibrated"
    assert not observational.sufficient
    assert not observational.calibrated

    boolean_only = certify(
        candidates,
        query_term_idf={"answer": 1.0},
        retained_terms={"answer"},
        budget_exhausted=False,
        calibrated=True,
    )
    assert boolean_only.verdict == "uncalibrated"
    assert not boolean_only.sufficient

    held_out_validated = certify(
        candidates,
        query_term_idf={"answer": 1.0},
        retained_terms={"answer"},
        budget_exhausted=False,
        calibration=_policy(),
    )
    assert held_out_validated.verdict == "sufficient"
    assert held_out_validated.sufficient


def test_any_observed_corpus_gap_is_fail_closed():
    certificate = certify(
        [Candidate("answer", utility=1.0, cost=10, selected=True, anchors=("a",))],
        query_term_idf={"a": 1.0, "critical": 4.0},
        retained_terms={"a"},
        unattainable_terms={"critical"},
        budget_exhausted=False,
        calibration=_policy(),
    )
    assert certificate.verdict == "expand_required"
    assert not certificate.sufficient


def test_stems_match_tokens_not_unrelated_substrings():
    terms = _lexical_term_set("discharge the capacitor")
    assert "charg" not in terms
    assert "discharg" in terms


def test_schema_helper_keeps_load_bearing_camel_case_key():
    payload = {f"filler_{index}": index for index in range(25)}
    payload["errorMessage"] = "the exact failure"
    schema = _json_to_schema(payload)
    assert schema["errorMessage"] == "the exact failure"


def test_qccr_certificate_marks_boundary_measurement_unavailable():
    selected = [{"source": "file:answer.py", "content": "return session token"}]
    attach_sufficiency(
        selected,
        candidate_utility={"file:answer.py": 2.0, "file:other.py": 0.1},
        chunk_utility={("file:answer.py", 0): 2.0},
        by_file={
            "file:answer.py": [{"content": "return session token"}],
            "file:other.py": [{"content": "unrelated cache"}],
        },
        query="session token",
        token_budget=100,
    )
    certificate = selected[0]["sufficiency"]
    assert certificate["verdict"] == "uncalibrated"
    assert certificate["boundary_exposure_measured"] is False
    assert certificate["calibration_id"] is None


def test_recovery_reference_rejects_forged_byte_length():
    from entroly.codec import RecoveryReference, content_digest

    recovered = "exact bytes"
    reference = RecoveryReference(
        digest=content_digest(recovered),
        byte_length=len(recovered.encode("utf-8")) + 1,
    )
    assert not reference.verify(recovered)
