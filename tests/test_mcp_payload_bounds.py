from __future__ import annotations

import json

from entroly.provenance import build_provenance


def _fragment(index: int, *, content_size: int = 80) -> dict:
    content = f"fragment {index} " + ("x" * content_size)
    return {
        "id": f"frag-{index}",
        "fragment_id": f"frag-{index}",
        "source": f"file:src/module_{index}.py",
        "content": content,
        "token_count": max(1, len(content) // 4),
        "relevance": 0.9,
        "recency_score": 0.8,
        "frequency_score": 0.7,
        "semantic_score": 0.95,
        "entropy_score": 0.6,
        "feedback_multiplier": 1.2,
        "debug_vector": [0.1] * 16,
        "retrieval_handle": f"ccr:{index:064x}",
        "content_sha256": f"{index:064x}",
    }


def _result(count: int = 40, *, content_size: int = 80) -> dict:
    selected = [_fragment(i, content_size=content_size) for i in range(count)]
    return {
        "selected_fragments": selected,
        "selected": selected,
        "tokens_used": sum(f["token_count"] for f in selected),
        "token_budget": 8000,
        "online_prism": {"weights": {"semantic": 0.4}},
    }


def _wire(result: dict) -> dict:
    provenance = build_provenance(
        result,
        query="find root wiring bug",
        refined_query=None,
        turn=4,
        token_budget=8000,
    )
    result["provenance"] = provenance.to_dict()
    return result


def test_default_wire_result_removes_alias_and_internal_fragment_fields(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_MCP_FULL_DIAGNOSTICS", raising=False)
    payload = _wire(_result())

    assert "selected" not in payload
    assert payload["response"]["canonical_selection_key"] == "selected_fragments"
    assert payload["selected_fragments"][0]["content"].startswith("fragment 0")
    assert payload["selected_fragments"][0]["retrieval_handle"].startswith("ccr:")
    assert "semantic_score" not in payload["selected_fragments"][0]
    assert "debug_vector" not in payload["selected_fragments"][0]
    assert "fragments" not in payload["provenance"]
    assert "source_set" not in payload["provenance"]
    assert payload["provenance"]["details_omitted"]["fragment_records"] == 40


def test_diagnostics_env_preserves_rich_fields_without_duplicate_alias(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_MCP_FULL_DIAGNOSTICS", "1")
    payload = _wire(_result())

    assert "selected" not in payload
    assert payload["response"]["mode"] == "diagnostics"
    assert payload["selected_fragments"][0]["semantic_score"] == 0.95
    assert len(payload["provenance"]["fragments"]) == 40
    assert len(payload["provenance"]["source_set"]) == 40


def test_dogfood_shape_shrinks_metadata_dominated_response(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_MCP_FULL_DIAGNOSTICS", raising=False)
    raw = _result(count=395, content_size=50)
    before = len(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
    payload = _wire(raw)
    after = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    assert before > after * 3
    assert after < 200_000
    assert payload["tokens_used"] <= payload["token_budget"]
    assert len(payload["selected_fragments"]) == 395


def test_provenance_prefers_canonical_selected_fragments(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_MCP_FULL_DIAGNOSTICS", raising=False)
    result = _result(count=2)
    result["selected"] = [{"id": "wrong", "source": "synthetic"}]

    payload = _wire(result)

    assert payload["provenance"]["fragment_count"] == 2
    assert payload["provenance"]["verified_fraction"] == 1.0
