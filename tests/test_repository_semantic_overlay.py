from __future__ import annotations

import hashlib
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.semantic_overlay import (
    build_verified_semantic_overlay,
    verify_semantic_overlay_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_semantic_overlay_verifies_utf16_locations_and_receipt(tmp_path: Path) -> None:
    _write(tmp_path, "definition.py", "def target():\n    return 1\n")
    _write(tmp_path, "reference.py", "# 😀 target\ndef use():\n    return 1\n")
    index = build_repository_index(tmp_path)
    payload = build_verified_semantic_overlay(
        tmp_path,
        index,
        [{
            "kind": "definition",
            "source": {
                "path": "reference.py",
                "line": 0,
                "start_character": 5,
                "end_character": 11,
            },
            "target": {
                "path": "definition.py",
                "line": 0,
                "start_character": 4,
                "end_character": 10,
            },
        }],
        index_digest="test-index",
        provider="pyright",
    )

    assert len(payload["relationships"]) == 1
    edge = payload["relationships"][0]
    for location in (edge["source"], edge["target"]):
        raw = (tmp_path / location["path"]).read_bytes()
        evidence = raw[location["start_byte"]:location["end_byte"]]
        assert evidence == b"target"
        assert hashlib.sha256(evidence).hexdigest() == location["evidence_sha256"]
    assert payload["provider_trust"].startswith("untrusted")
    assert verify_semantic_overlay_commitment(payload)

    edge["kind"] = "invented"
    assert not verify_semantic_overlay_commitment(payload)


def test_semantic_overlay_fails_closed_on_stale_or_invalid_ranges(tmp_path: Path) -> None:
    _write(tmp_path, "main.py", "def target():\n    return 1\n")
    index = build_repository_index(tmp_path)
    _write(tmp_path, "main.py", "def target():\n    return 2\n")
    payload = build_verified_semantic_overlay(
        tmp_path,
        index,
        [
            {
                "kind": "definition",
                "source": {"path": "main.py", "line": 0, "start_character": 4, "end_character": 10},
                "target": {"path": "main.py", "line": 0, "start_character": 4, "end_character": 10},
            },
            {
                "kind": "not-semantic",
                "source": {},
                "target": {},
            },
        ],
        index_digest="test-index",
        provider="test",
    )
    assert payload["relationships"] == []
    assert payload["receipt"]["omissions_by_reason"] == {
        "invalid-kind": 1,
        "source-stale-index": 1,
    }
    assert verify_semantic_overlay_commitment(payload)
