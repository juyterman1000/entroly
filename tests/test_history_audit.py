from __future__ import annotations

import json
from pathlib import Path

from entroly.history_audit import audit_histories


def test_codex_cumulative_usage_is_not_double_counted_and_content_is_blind(tmp_path: Path) -> None:
    secret = "private-history-marker-should-not-leak"
    history = tmp_path / "sessions"
    history.mkdir()
    records = [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"text": secret * 20}]}},
        {"type": "event_msg", "payload": {"type": "item_completed", "item": {"content": secret * 20}}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}}}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 140, "output_tokens": 20, "total_tokens": 160}}}},
    ]
    (history / "session.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    report = audit_histories({"codex": (history,)})

    known = report["provider_reported"]["known_semantics"]
    assert known["input_tokens"] == 140
    assert known["output_tokens"] == 20
    assert known["total_tokens"] == 160
    assert secret not in json.dumps(report)
    assert report["structural_estimate"]["tokens"] >= len(secret * 20) // 4
    assert report["privacy"].startswith("aggregate-only")


def test_unknown_usage_semantics_are_quarantined(tmp_path: Path) -> None:
    history = tmp_path / "sessions"
    history.mkdir()
    (history / "session.jsonl").write_text(
        json.dumps({"usage": {"input_tokens": 17, "output_tokens": 2}}), encoding="utf-8"
    )

    report = audit_histories({"custom": (history,)})

    assert report["provider_reported"]["known_semantics"]["input_tokens"] == 0
    assert report["provider_reported"]["unknown_semantics_observed_sum"]["input_tokens"] == 17


def test_history_audit_honors_per_file_and_total_caps(tmp_path: Path) -> None:
    history = tmp_path / "sessions"
    history.mkdir()
    (history / "large.json").write_text(json.dumps({"content": "x" * 1000}), encoding="utf-8")
    (history / "small.json").write_text(json.dumps({"content": "small"}), encoding="utf-8")

    report = audit_histories({"fixture": (history,)}, max_bytes=100, max_file_bytes=50)

    assert report["scope"]["files_read"] == 1
    assert report["scope"]["skipped_for_file_byte_cap"] == 1


def test_tool_pressure_produces_reversible_nonautomatic_recommendation(tmp_path: Path) -> None:
    history = tmp_path / "sessions"
    history.mkdir()
    (history / "one.jsonl").write_text(
        json.dumps({"type": "tool_result", "content": "log line " * 200}), encoding="utf-8"
    )
    report = audit_histories({"fixture": (history,)})
    recommendation = next(row for row in report["recommendations"] if row["id"] == "command-envelope")
    assert recommendation["automatic_apply"] is False
    assert recommendation["reversible"] is True
    assert "paired" in recommendation["evidence_gate"]
