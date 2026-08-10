from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.repository_intelligence.models import FileRecord, RepositoryIndex
from entroly.repository_intelligence.semantic_overlay import (
    build_verified_semantic_overlay,
    verify_semantic_overlay_commitment,
)


def _index(root: Path, path: str, source: str) -> RepositoryIndex:
    raw = source.encode("utf-8")
    return RepositoryIndex(
        root=str(root),
        files={
            path: FileRecord(
                path=path,
                language="test",
                sha256=hashlib.sha256(raw).hexdigest(),
                byte_length=len(raw),
                line_count=1,
                is_test=False,
            )
        },
    )


def _relationship(path: str, start: int, end: int) -> list[dict[str, object]]:
    location = {
        "path": path,
        "line": 0,
        "start_character": start,
        "end_character": end,
    }
    return [{"kind": "reference", "source": location, "target": location}]


def test_utf8_utf16_utf32_positions_bind_to_same_exact_bytes(tmp_path: Path) -> None:
    path = "sample.zig"
    source = "α😀target"
    (tmp_path / path).write_text(source, encoding="utf-8")
    index = _index(tmp_path, path, source)

    # Before "target": alpha is 2 UTF-8 bytes / 1 UTF-16 unit / 1 code point;
    # emoji is 4 UTF-8 bytes / 2 UTF-16 units / 1 code point.
    inputs = {
        "utf-8": (6, 12),
        "utf-16": (3, 9),
        "utf-32": (2, 8),
    }
    spans: list[tuple[int, int, str]] = []
    for encoding, (start, end) in inputs.items():
        payload = build_verified_semantic_overlay(
            tmp_path,
            index,
            _relationship(path, start, end),
            index_digest="sha256:index",
            provider=f"test-{encoding}",
            position_encoding=encoding,
        )
        assert verify_semantic_overlay_commitment(payload)
        assert payload["position_encoding"] == encoding
        assert payload["receipt"]["accepted_relationship_count"] == 1
        edge = payload["relationships"][0]
        location = edge["source"]
        spans.append((
            location["start_byte"],
            location["end_byte"],
            location["evidence_sha256"],
        ))
        assert location["position_encoding"] == encoding
        assert edge["epistemic_class"] == "external-semantic-source-verified"

    expected_hash = hashlib.sha256(b"target").hexdigest()
    assert spans == [(6, 12, expected_hash)] * 3


def test_default_position_encoding_remains_utf16(tmp_path: Path) -> None:
    path = "sample.py"
    source = "😀value"
    (tmp_path / path).write_text(source, encoding="utf-8")
    index = _index(tmp_path, path, source)
    payload = build_verified_semantic_overlay(
        tmp_path,
        index,
        _relationship(path, 2, 7),
        index_digest="sha256:index",
        provider="legacy-lsp",
    )
    assert payload["position_encoding"] == "utf-16"
    assert payload["receipt"]["accepted_relationship_count"] == 1
    assert payload["relationships"][0]["source"]["start_byte"] == 4


def test_utf8_offset_inside_multibyte_codepoint_is_rejected(tmp_path: Path) -> None:
    path = "sample.c"
    source = "αvalue"
    (tmp_path / path).write_text(source, encoding="utf-8")
    index = _index(tmp_path, path, source)
    payload = build_verified_semantic_overlay(
        tmp_path,
        index,
        _relationship(path, 1, 3),
        index_digest="sha256:index",
        provider="compiler",
        position_encoding="utf-8",
    )
    assert payload["relationships"] == []
    assert payload["receipt"]["omissions_by_reason"] == {
        "source-character-out-of-range": 1
    }


def test_invalid_position_encoding_fails_closed(tmp_path: Path) -> None:
    index = RepositoryIndex(root=str(tmp_path))
    with pytest.raises(ValueError, match="position_encoding"):
        build_verified_semantic_overlay(
            tmp_path,
            index,
            [],
            index_digest="sha256:index",
            provider="bad-provider",
            position_encoding="bytes-ish",
        )


def test_stale_source_is_never_accepted(tmp_path: Path) -> None:
    path = "sample.rs"
    original = "target"
    (tmp_path / path).write_text("changed", encoding="utf-8")
    index = _index(tmp_path, path, original)
    payload = build_verified_semantic_overlay(
        tmp_path,
        index,
        _relationship(path, 0, 6),
        index_digest="sha256:index",
        provider="rust-analyzer",
        position_encoding="utf-8",
    )
    assert payload["relationships"] == []
    assert payload["receipt"]["omissions_by_reason"] == {
        "source-stale-index": 1
    }
