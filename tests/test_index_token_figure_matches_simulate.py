"""Onboarding commands must print one token figure for one index.

Dogfooding 1.0.88 on encode/httpx: `entroly verify-claims` and `entroly go`
printed "92 files, 221,012 tokens"; `entroly simulate` a minute later printed
"92 of 97 indexable files (180,757 estimated tokens)". The first two printed
auto_index's selection-weighted total (configs x3, docs x2); simulate counts
content.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "ledger.py").write_text(
        "".join(f"def reconcile_{i}(rows):\n    return rows[{i}]\n\n" for i in range(40)),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "".join(f"## Shelf {i}\n\nShelf {i} holds ledger rows.\n\n" for i in range(80)),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"ledger\"\n" + "".join(f"opt_{i} = \"v{i}\"\n" for i in range(60)),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.chdir(root)
    return root


def test_printed_index_tokens_are_content_tokens(project) -> None:
    from entroly.auto_index import auto_index
    from entroly.cli import _index_display_tokens, _indexed_content_tokens
    from entroly.server import EntrolyEngine

    engine = EntrolyEngine()
    result = auto_index(engine, force=True)
    shown = _index_display_tokens(engine, result)

    assert shown == _indexed_content_tokens(engine, -1)
    assert shown < int(result["total_tokens"]), (
        "docs and configs are weighted up for selection; the printed figure must not be"
    )
