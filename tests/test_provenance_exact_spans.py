from __future__ import annotations

import copy

from entroly.provenance import build_provenance, compact_optimize_result_for_wire


def _result() -> dict:
    selected = [
        {
            "id": "frag-1",
            "source": "src/cache.py",
            "content": "def load():\n    return 1\n",
            "token_count": 8,
            "composite_score": 0.91,
            "start_line": 10,
            "end_line": 11,
            "byte_start": 120,
            "byte_end": 145,
            "content_sha256": "a" * 64,
            "commit": "deadbeef",
            "retrieval_handle": "ccr:123",
            "receipt_id": "receipt-123",
        }
    ]
    return {
        "selected": selected,
        "selected_fragments": selected,
        "tokens_used": 8,
    }


def test_build_provenance_carries_exact_span_and_does_not_mutate() -> None:
    result = _result()
    before = copy.deepcopy(result)

    provenance = build_provenance(
        result,
        query="find cache loader",
        refined_query=None,
        turn=1,
        token_budget=100,
    )

    assert result == before
    fragment = provenance.fragments[0]
    assert fragment.exact_span is True
    assert fragment.start_line == 10
    assert fragment.end_line == 11
    assert fragment.start_byte == 120
    assert fragment.end_byte == 145
    assert fragment.content_sha256 == "a" * 64
    assert fragment.source_version == "deadbeef"
    assert fragment.retrieval_handle == "ccr:123"
    assert fragment.transform_receipt_id == "receipt-123"
    assert provenance.exact_span_fraction == 1.0


def test_sdk_dict_includes_exact_span_metadata() -> None:
    provenance = build_provenance(
        _result(),
        query="find cache loader",
        refined_query=None,
        turn=1,
        token_budget=100,
    )

    payload = provenance.to_dict()
    fragment = payload["fragments"][0]
    assert payload["exact_span_fraction"] == 1.0
    assert fragment["exact_span"] is True
    assert fragment["start_line"] == 10
    assert fragment["start_byte"] == 120
    assert fragment["source_version"] == "deadbeef"
    assert fragment["transform_receipt_id"] == "receipt-123"


def test_mcp_wire_compaction_canonicalizes_coordinate_aliases(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_MCP_FULL_DIAGNOSTICS", raising=False)
    result = _result()

    compact_optimize_result_for_wire(result)

    assert "selected" not in result
    fragment = result["selected_fragments"][0]
    assert fragment["start_byte"] == 120
    assert fragment["end_byte"] == 145
    assert fragment["source_version"] == "deadbeef"
    assert fragment["transform_receipt_id"] == "receipt-123"
    assert "byte_start" not in fragment
    assert "commit" not in fragment
    assert "receipt_id" not in fragment


def test_wire_compaction_also_compacts_per_fragment_provenance():
    """`provenance.fragments` mirrors the selection and must not be sent raw.

    Dogfooding an 8,000-token `optimize_context` request returned a
    378,545-char payload that overflowed the MCP result cap, so the agent got
    an error instead of context -- while the selection itself was correct at
    7,004 tokens. Dropping the duplicate `selected` alias alone still left
    ~410,000 chars, because provenance kept 395 full fragments.

    Compact mode was already documented to strip per-fragment provenance (the
    diagnostics hint offers to restore it) and did not.
    """
    from entroly.provenance import compact_optimize_result_for_wire

    fragment = {
        "id": "f1",
        "source": "file:a.py",
        "content": "x" * 400,
        "token_count": 100,
        "relevance": 0.8,
        "content_sha256": "deadbeef",
        "retrieval_handle": "h1",
        "entropy_score": 0.5,
        "variant": "full",
    }
    selection = [dict(fragment) for _ in range(50)]
    result = {
        "selected_fragments": selection,
        "selected": selection,
        "provenance": {"fragments": [dict(fragment) for _ in range(50)], "query": "q"},
    }

    compact_optimize_result_for_wire(result)

    assert "selected" not in result, "duplicate alias must not reach the wire"
    provenance_fragments = result["provenance"]["fragments"]
    assert provenance_fragments, "provenance fragments must survive"
    compacted = provenance_fragments[0]
    assert "entropy_score" not in compacted, "internal scoring vectors must be stripped"
    for trust_field in ("source", "content", "token_count", "content_sha256",
                        "retrieval_handle"):
        assert trust_field in compacted, (
            f"{trust_field} is trust-critical and must survive compaction"
        )
