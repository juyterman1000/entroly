from __future__ import annotations

import json

from entroly.evidence_locked_compression import (
    compress_evidence_locked,
    compress_payload_messages,
    detect_heavy_content_type,
    estimate_tokens,
)


def test_elc_preserves_failure_anchor_and_nearby_context() -> None:
    lines = [f"Compiling crate_{i} ... ok" for i in range(300)]
    lines.extend(
        [
            "Running final check",
            "src/auth/session.py:184",
            "ERROR: auth refresh timeout after retry window",
            "hint: increase refresh slack before retry",
            "Build finished with exit code 1",
        ]
    )
    text = "\n".join(lines)

    result = compress_evidence_locked(
        text,
        query="why auth refresh timeout failed",
        budget_tokens=180,
    )

    assert result.changed
    assert "ERROR: auth refresh timeout" in result.compressed
    assert "src/auth/session.py:184" in result.compressed
    assert "increase refresh slack" in result.compressed
    assert "lines omitted" in result.compressed
    assert result.receipt.savings_ratio > 0.5
    assert result.receipt.anchors_preserved["anchor"] >= 1
    assert result.receipt.recoverable is True


def test_elc_json_keeps_schema_query_matches_and_outliers() -> None:
    records = []
    for i in range(120):
        records.append(
            {
                "id": i,
                "service": "billing" if i % 2 else "auth",
                "status": "ok",
                "latency_ms": i,
                "payload": "x" * 20,
            }
        )
    records[77] = {
        "id": 77,
        "service": "auth",
        "status": "failed",
        "latency_ms": 9912,
        "payload": "refresh timeout at retry boundary" + "z" * 500,
    }
    text = json.dumps(records)

    result = compress_evidence_locked(
        text,
        query="auth failed refresh timeout",
        budget_tokens=500,
    )

    assert result.changed
    assert "json evidence-locked compression" in result.compressed
    assert "query_matches" in result.compressed
    assert "refresh timeout" in result.compressed
    assert '"id": 77' in result.compressed
    assert '"latency_ms": 9912' in result.compressed
    assert '"status": "failed"' in result.compressed
    assert "outliers" in result.compressed
    assert result.receipt.content_type == "json"
    assert result.receipt.anchors_preserved["schema"] == 1
    assert result.receipt.savings_ratio > 0.5


def test_elc_payload_compresses_tool_messages_only() -> None:
    heavy = "\n".join(["download dependency ok" for _ in range(250)] + ["ERROR: final link failed"])
    messages = [
        {"role": "user", "content": "why did build fail?"},
        {"role": "tool", "content": heavy},
    ]

    compressed, receipts, saved = compress_payload_messages(
        messages,
        query="build fail",
        budget_tokens=120,
    )

    assert compressed[0] == messages[0]
    assert compressed[1]["content"] != heavy
    assert "entroly-elc" in compressed[1]["content"]
    assert "ERROR: final link failed" in compressed[1]["content"]
    assert receipts
    assert saved > 0


def test_elc_short_content_passes_through() -> None:
    result = compress_evidence_locked("small output", budget_tokens=1000)

    assert result.changed is False
    assert result.compressed == "small output"
    assert result.receipt.savings_ratio == 0.0


def test_elc_detects_json_and_log() -> None:
    assert detect_heavy_content_type('[{"a": 1}]') == "json"
    assert detect_heavy_content_type("2026-01-01 INFO starting\nERROR failed") == "log"


def test_elc_json_keeps_query_centered_long_scalar_within_budget() -> None:
    documents = [
        {
            "document_id": "noise",
            "text": "Routine unrelated material. " * 120,
        },
        {
            "document_id": "gold",
            "text": (
                "A long introduction discussed unrelated planning details. " * 30
                + "The Aurora mission launched on March 7, 2012 from the coastal pad. "
                + "Later reports focused on its scientific instruments. " * 20
            ),
        },
    ]
    text = json.dumps({"query_result": {"documents": documents, "returned": 2}})

    result = compress_evidence_locked(
        text,
        query="When did the Aurora mission launch?",
        budget_tokens=180,
    )

    assert result.changed
    assert "March 7, 2012" in result.compressed
    assert '"document_id": "gold"' in result.compressed
    assert estimate_tokens(result.compressed) <= 180
    assert result.receipt.anchors_preserved["query"] >= 1
