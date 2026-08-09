from __future__ import annotations

from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.tree_sitter_support import (
    LANGUAGE_BY_SUFFIX,
    _get_local_parser,
    language_for_path,
)


REGISTRY_ONLY_PATH = "src/core.clj"
REGISTRY_ONLY_LANGUAGE = "clojure"
REGISTRY_ONLY_SOURCE = "(ns demo.core)\n(defn answer [] 42)\n"


def test_registry_language_is_not_capped_by_entroly_fallback() -> None:
    """A registry-supported language outside Entroly's fallback must route dynamically."""
    pytest.importorskip("tree_sitter_language_pack")
    assert Path(REGISTRY_ONLY_PATH).suffix.lower() not in LANGUAGE_BY_SUFFIX
    assert language_for_path(REGISTRY_ONLY_PATH) == REGISTRY_ONLY_LANGUAGE


def test_registry_only_language_parser_loads() -> None:
    """The dynamic language must reach the generic parser path, not just detection."""
    pytest.importorskip("tree_sitter_language_pack")
    language = language_for_path(REGISTRY_ONLY_PATH)
    assert language == REGISTRY_ONLY_LANGUAGE
    parser = _get_local_parser(language)
    assert parser is not None
    tree = parser.parse(REGISTRY_ONLY_SOURCE.encode("utf-8"))
    assert tree.root_node is not None
    assert not bool(getattr(tree.root_node, "has_error", False))


def test_repository_index_accepts_registry_only_language(tmp_path: Path) -> None:
    """End-to-end indexing must preserve a language absent from the fallback table."""
    pytest.importorskip("tree_sitter_language_pack")
    source = tmp_path / "core.clj"
    source.write_text(REGISTRY_ONLY_SOURCE, encoding="utf-8")

    index = build_repository_index(tmp_path)

    assert "core.clj" in index.files
    assert index.files["core.clj"].language == REGISTRY_ONLY_LANGUAGE
