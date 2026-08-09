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
    assert any(
        "persistent-index-snapshot hits=0 misses=1 writes=1" in item
        for item in cold.diagnostics
    )
    assert any(
        "persistent-index-snapshot hits=1 misses=0 writes=0" in item
        for item in warm.diagnostics
    )
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
    entry = next(
        path for path in cache.rglob("*.json")
        if "index-snapshots" not in path.parts
    )
    entry.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    rebuilt = build_repository_index_incremental(root, cache_dir=cache)

    assert "hits=0 misses=1" in _cache_diagnostic(rebuilt)
    assert "corruptions=1" in _cache_diagnostic(rebuilt)
    assert any(symbol.name == "run" for symbol in rebuilt.symbols.values())
    assert json.loads(entry.read_text(encoding="utf-8"))["schema_version"].endswith("v2")


def test_tampered_parse_payload_cannot_change_the_graph(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "main.py", "def run():\n    return 1\n")
    build_repository_index_incremental(root, cache_dir=cache)
    entry = next(
        path for path in cache.rglob("*.json")
        if "index-snapshots" not in path.parts
    )
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["symbols"][0]["name"] = "forged"
    entry.write_text(json.dumps(payload), encoding="utf-8")

    rebuilt = build_repository_index_incremental(root, cache_dir=cache)

    assert "corruptions=1" in _cache_diagnostic(rebuilt)
    assert {symbol.name for symbol in rebuilt.symbols.values()} == {"run"}
    repaired = json.loads(entry.read_text(encoding="utf-8"))
    assert repaired["symbols"][0]["name"] == "run"


def test_corrupt_global_snapshot_fails_open_and_is_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "main.py", "def run():\n    return 1\n")
    build_repository_index_incremental(root, cache_dir=cache)
    snapshot = next((cache / "index-snapshots").rglob("*.json"))
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["index"]["files"][0]["sha256"] = "0" * 64
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    rebuilt = build_repository_index_incremental(root, cache_dir=cache)

    assert any(
        "persistent-index-snapshot hits=0 misses=1 writes=1 corruptions=1" in item
        for item in rebuilt.diagnostics
    )
    assert any(symbol.name == "run" for symbol in rebuilt.symbols.values())
    repaired = json.loads(snapshot.read_text(encoding="utf-8"))
    assert repaired["index"]["files"][0]["sha256"] != "0" * 64


def test_content_identical_checkout_reuses_root_independent_graph_snapshot(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    cache = tmp_path / "cache"
    for root in (first_root, second_root):
        _write(root, "a.py", "def source():\n    return 1\n")
        _write(root, "b.py", "from a import source\ndef call():\n    return source()\n")

    first = build_repository_index_incremental(first_root, cache_dir=cache)
    second = build_repository_index_incremental(second_root, cache_dir=cache)

    assert first.root != second.root
    assert any(
        "persistent-index-snapshot hits=1 misses=0" in item
        for item in second.diagnostics
    )
    first_payload = first.to_dict()
    second_payload = second.to_dict()
    first_payload["root"] = second_payload["root"] = "."
    first_payload["diagnostics"] = second_payload["diagnostics"] = []
    assert first_payload == second_payload


def test_service_can_opt_into_incremental_cache(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "main.py", "def run():\n    return 1\n")

    first = RepositoryIntelligenceService(root, cache_dir=cache).summary()
    second = RepositoryIntelligenceService(root, cache_dir=cache).summary()

    assert any("hits=0 misses=1" in item for item in first["diagnostics"])
    assert any("hits=1 misses=0" in item for item in second["diagnostics"])
    assert first["index_digest"] == second["index_digest"]
