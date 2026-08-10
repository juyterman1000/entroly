from __future__ import annotations

import json
from pathlib import Path

import entroly.repository_intelligence.service as service_module
from entroly.repository_intelligence import (
    RepositoryIntelligenceService,
    verify_architecture_commitment,
    verify_code_health_commitment,
    verify_repository_map_commitment,
    verify_routes_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "source.py", "def source(value):\n    return value + 1\n")
    _write(
        root,
        "caller.py",
        "from source import source\ndef caller():\n    return source(1)\n",
    )


def test_unchanged_map_health_architecture_and_routes_reuse_verified_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _project(root)
    first_service = RepositoryIntelligenceService(root, cache_dir=cache)
    first_map = first_service.repository_map("source")
    first_health = first_service.code_health()
    first_architecture = first_service.architecture()
    first_routes = first_service.routes()
    assert verify_repository_map_commitment(first_map)
    assert verify_code_health_commitment(first_health)
    assert verify_architecture_commitment(first_architecture)
    assert verify_routes_commitment(first_routes)

    def unexpected(*args, **kwargs):
        raise AssertionError("unchanged verified analysis should load from cache")

    monkeypatch.setattr(service_module, "build_verified_repository_map", unexpected)
    monkeypatch.setattr(service_module, "build_verified_code_health", unexpected)
    monkeypatch.setattr(service_module, "build_verified_architecture", unexpected)
    monkeypatch.setattr(service_module, "build_verified_routes", unexpected)
    second_service = RepositoryIntelligenceService(root, cache_dir=cache)
    second_map = second_service.repository_map("source")
    second_health = second_service.code_health()
    second_architecture = second_service.architecture()
    second_routes = second_service.routes()

    assert second_map == first_map
    assert second_health == first_health
    assert second_architecture == first_architecture
    assert second_routes == first_routes


def test_corrupt_analysis_envelope_fails_open_and_is_replaced(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _project(root)
    first = RepositoryIntelligenceService(root, cache_dir=cache).repository_map("source")
    artifact = next((cache / "analysis" / "repository-map").rglob("*.json"))
    envelope = json.loads(artifact.read_text(encoding="utf-8"))
    envelope["payload"]["entries"][0]["qualified_name"] = "tampered"
    artifact.write_text(json.dumps(envelope), encoding="utf-8")

    rebuilt = RepositoryIntelligenceService(root, cache_dir=cache).repository_map("source")

    assert rebuilt == first
    repaired = json.loads(artifact.read_text(encoding="utf-8"))
    assert repaired["payload"]["entries"][0]["qualified_name"] != "tampered"
    assert verify_repository_map_commitment(repaired["payload"])


def test_root_independent_analysis_reuse_preserves_checkout_local_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    cache = tmp_path / "cache"
    _project(first_root)
    _project(second_root)
    first = RepositoryIntelligenceService(first_root, cache_dir=cache).repository_map("source")

    def unexpected(*args, **kwargs):
        raise AssertionError("identical content should share the verified analysis")

    monkeypatch.setattr(service_module, "build_verified_repository_map", unexpected)
    second = RepositoryIntelligenceService(second_root, cache_dir=cache).repository_map("source")
    assert second == first


def test_index_digest_ignores_cache_telemetry(tmp_path: Path) -> None:
    """An identical tree must hash identically whether the cache is cold or warm.

    The digest is the repository's identity, so it may only depend on content.
    Cache and snapshot diagnostics describe how a run was served, and the
    retention counters in particular differ between a cold and a warm cache
    (files=2 bytes=2195 against files=3 bytes=4804 on the same unchanged tree).

    They leaked in once already: the filter matched "incremental-parse-cache "
    with a trailing space, which excluded "incremental-parse-cache-retention"
    because the following character is a hyphen. One identical checkout then
    produced two digests depending only on cache warmth, which defeats the
    reuse this module exists to provide. Asserted here so a future prefix edit
    cannot quietly reintroduce it.
    """
    root = tmp_path / "repo"
    _project(root)

    cold_a = RepositoryIntelligenceService(root, cache_dir=tmp_path / "cache-a")
    cold_b = RepositoryIntelligenceService(root, cache_dir=tmp_path / "cache-b")
    assert cold_a._snapshot()[1] == cold_b._snapshot()[1], "two cold caches must agree"

    shared = tmp_path / "cache-shared"
    first = RepositoryIntelligenceService(root, cache_dir=shared)
    cold_digest = first._snapshot()[1]
    first.repository_map("source")  # warms the incremental parse cache

    warm = RepositoryIntelligenceService(root, cache_dir=shared)
    assert warm._snapshot()[1] == cold_digest, "a warm cache must not change identity"
