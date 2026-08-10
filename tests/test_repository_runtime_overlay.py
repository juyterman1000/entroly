from __future__ import annotations

import hashlib
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.runtime_overlay import (
    build_verified_runtime_overlay,
    verify_runtime_overlay_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_runtime_overlay_aggregates_events_and_verifies_source(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "worker.py",
        "def run(value):\n"
        "    result = value + 1\n"
        "    return result\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_runtime_overlay(
        tmp_path,
        index,
        [
            {"path": "worker.py", "line": 2, "event": "line", "count": 2},
            {"path": "worker.py", "line": 2, "event": "line", "count": 3},
            {"path": "worker.py", "line": 3, "event": "return"},
        ],
        index_digest="test-index",
        producer="pytest-trace",
    )

    assert len(payload["observations"]) == 2
    assert payload["observations"][0]["count"] == 5
    assert payload["hot_symbols"][0]["count"] == 6
    assert payload["receipt"]["event_values_collected"] is False
    assert verify_runtime_overlay_commitment(payload)
    raw = (tmp_path / "worker.py").read_bytes()
    for observation in payload["observations"]:
        evidence = raw[observation["start_byte"]:observation["end_byte"]]
        assert hashlib.sha256(evidence).hexdigest() == observation["evidence_sha256"]

    payload["observations"][0]["count"] = 999
    assert not verify_runtime_overlay_commitment(payload)


def test_runtime_overlay_rejects_paths_ranges_values_and_stale_files(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "worker.py", "def run():\n    return 1\n")
    index = build_repository_index(tmp_path)
    _write(tmp_path, "worker.py", "def run():\n    return 2\n")
    payload = build_verified_runtime_overlay(
        tmp_path,
        index,
        [
            {"path": "worker.py", "line": 2, "event": "line", "value": "secret"},
            {"path": "../secret.py", "line": 1, "event": "line"},
            {"path": "missing.py", "line": 1, "event": "line"},
            {"path": "worker.py", "line": 999, "event": "line"},
        ],
        index_digest="test-index",
    )

    assert payload["observations"] == []
    omissions = payload["receipt"]["omissions_by_reason"]
    assert omissions["stale-index"] == 2
    assert omissions["invalid-event"] == 1
    assert omissions["unknown-path"] == 1
    assert "secret" not in str(payload)
    assert verify_runtime_overlay_commitment(payload)
