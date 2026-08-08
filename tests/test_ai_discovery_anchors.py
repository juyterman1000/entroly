from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_compact_ai_anchor_is_mirrored_and_retrieval_sized() -> None:
    canonical = _text("ai.txt")
    assert canonical == _text("docs/ai.txt")
    assert len(canonical.split()) <= 230
    for anchor in (
        "# Entroly AI Agent Context Compression",
        "## Context Compression Mechanics",
        "## Streaming and Multi-Turn Context Compression",
        "## Exact Recovery and Context Receipts",
        "## Evidence and Limits",
    ):
        assert anchor in canonical


def test_primary_distribution_surfaces_use_exact_context_compression_phrase() -> None:
    assert "## Context compression mechanics" in _text("README.md")
    assert "## Context Compression Mechanics" in _text("llms.txt")
    assert _text("llms.txt") == _text("docs/llms.txt")
    assert "AI agent context compression" in _text("pyproject.toml")
    for package in (
        "entroly-wasm/package.json",
        "entroly/npm/package.json",
        "entroly/npm-alias/package.json",
    ):
        description = json.loads(_text(package))["description"]
        assert "context compression" in description.casefold()


def test_crawler_and_sitemap_surfaces_reference_compact_anchor() -> None:
    for path in ("robots.txt", "docs/robots.txt", "sitemap.xml", "docs/sitemap.xml"):
        assert "https://juyterman1000.github.io/entroly/ai.txt" in _text(path)
