from __future__ import annotations

import json
from pathlib import Path

from entroly.repository_intelligence.incremental import (
    build_repository_index_incremental,
)
from entroly.repository_intelligence.service import RepositoryIntelligenceService


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _cache_diagnostic(index) -> str:
    return next(
        item for item in index.diagnostics if item.startswith("incremental-parse-cache")
    )


def test_incremental_cache_reuses_unchanged_exact_parse_results(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "a.py", "def source():\n    return 1\n")
    _write(
        root,
        "b.py",
        "from a import source\n"
        "def caller():\n    return source()\n",
    )

    cold = build_repository_index_incremental(root, cache_dir=cache)
    warm = build_repository_index_incremental(root, cache_dir=cache)

    assert "hits=0 misses=2 writes=2" in _cache_diagnostic(cold)
    assert "hits=2 misses=0 writes=0" in _cache_diagnostic(warm)
    cold_payload = cold.to_dict()
    warm_payload = warm.to_dict()
    cold_payload["diagnostics"] = []
    warm_payload["diagnostics"] = []
    assert cold_payload == warm_payload


def test_incremental_cache_invalidates_only_changed_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "a.py", "def source():\n    return 1\n")
    _write(root, "b.py", "def stable():\n    return 2\n")
    build_repository_index_incremental(root, cache_dir=cache)

    _write(root, "a.py", "def source():\n    return 3\n")
    changed = build_repository_index_incremental(root, cache_dir=cache)

    assert "hits=1 misses=1 writes=1" in _cache_diagnostic(changed)
    source = next(symbol for symbol in changed.symbols.values() if symbol.name == "source")
    assert source.evidence_sha256


def test_corrupt_cache_entry_fails_open_and_is_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "main.py", "def run():\n    return 1\n")
    build_repository_index_incremental(root, cache_dir=cache)
    entry = next(cache.rglob("*.json"))
    entry.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    rebuilt = build_repository_index_incremental(root, cache_dir=cache)

    assert "hits=0 misses=1" in _cache_diagnostic(rebuilt)
    assert "corruptions=1" in _cache_diagnostic(rebuilt)
    assert any(symbol.name == "run" for symbol in rebuilt.symbols.values())
    assert json.loads(entry.read_text(encoding="utf-8"))["schema_version"].endswith("v1")


def test_service_can_opt_into_incremental_cache(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "main.py", "def run():\n    return 1\n")

    first = RepositoryIntelligenceService(root, cache_dir=cache).summary()
    second = RepositoryIntelligenceService(root, cache_dir=cache).summary()

    assert any("hits=0 misses=1" in item for item in first["diagnostics"])
    assert any("hits=1 misses=0" in item for item in second["diagnostics"])
