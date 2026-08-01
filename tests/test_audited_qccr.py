from __future__ import annotations

import json

from entroly import audited_qccr


def test_identity_path_never_calls_native(monkeypatch) -> None:
    monkeypatch.setattr(
        audited_qccr,
        "_rust_select",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native called")),
    )
    fragments = [{"source": "a.txt", "content": "small", "token_count": 2}]
    result = audited_qccr.select_with_audit(fragments, 100, "small")
    assert result["selected"] == fragments
    assert result["metrics"]["verdict"] == "sufficient"
    assert result["selection_mode"] == "identity"


def test_native_envelope_is_validated_and_certificate_attached(monkeypatch) -> None:
    payload = {
        "selected": [{"source": "a.txt", "content": "answer", "token_count": 2}],
        "candidates": [{"unit_id": "u1", "selected": True}],
        "metrics": {
            "verdict": "sufficient",
            "scope": "candidate_units",
            "reasons": [],
            "source_span_integrity": True,
        },
        "requested_budget": 10,
        "raw_tokens": 100,
        "emitted_tokens": 2,
        "selection_mode": "atomic_audited",
    }
    monkeypatch.setattr(audited_qccr, "_rust_select", lambda *_args: json.dumps(payload))
    fragments = [
        {
            "fragment_id": "native-1",
            "source": "a.txt",
            "content": "x" * 400,
            "token_count": 100,
        }
    ]
    result = audited_qccr.select_with_audit(fragments, 10, "answer")
    selected = result["selected"][0]
    assert selected["_sufficiency"]["scope"] == "candidate_units"
    assert selected["source_fragment_ids"] == ["native-1"]
    assert result["candidates"]


def test_no_query_is_uncertain_identity() -> None:
    fragments = [{"source": "a.txt", "content": "x" * 400, "token_count": 100}]
    result = audited_qccr.select_with_audit(fragments, 10, "")
    assert result["selected"] == fragments
    assert result["metrics"]["verdict"] == "uncertain"
    assert result["metrics"]["scope"] == "unavailable"
