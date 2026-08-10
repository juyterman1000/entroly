from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.repository_map import (
    build_verified_repository_map,
    verify_repository_map_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "pkg/core.py", "def hub(value):\n    return value\n")
    for name in ("alpha", "beta", "gamma"):
        _write(
            root,
            f"pkg/{name}.py",
            "from pkg.core import hub\n"
            f"def {name}(value):\n"
            "    return hub(value)\n",
        )
    _write(
        root,
        "pkg/rare.py",
        "def calibrate_quantum_flux(sample):\n    return sample\n",
    )


def _build(root: Path, query: str = "", *, budget: int = 2_000):
    index = build_repository_index(root)
    return build_verified_repository_map(
        root,
        index,
        query,
        index_digest="sha256:test-index",
        token_budget=budget,
    )


def test_global_map_ranks_dependency_and_call_hub(tmp_path: Path) -> None:
    _project(tmp_path)
    payload = _build(tmp_path)

    assert payload["schema_version"] == "entroly.verified-repository-map.v1"
    assert payload["ranking"]["algorithm"] == "typed-personalized-pagerank"
    assert payload["entries"][0]["qualified_name"] == "hub"
    assert payload["receipt"]["remote_calls"] == 0
    assert verify_repository_map_commitment(payload)


def test_query_personalization_surfaces_rare_exact_symbol(tmp_path: Path) -> None:
    _project(tmp_path)
    payload = _build(tmp_path, "where is calibrate quantum flux implemented")

    first = payload["entries"][0]
    assert first["qualified_name"] == "calibrate_quantum_flux"
    assert first["query_relevance"] == 1.0
    assert payload["ranking"]["personalization"].startswith("query-")


def test_map_evidence_is_exact_budgeted_and_tamper_evident(tmp_path: Path) -> None:
    _project(tmp_path)
    payload = _build(tmp_path, budget=128)

    assert payload["budget"]["estimated_tokens"] <= 128
    assert payload["entries"]
    for entry in payload["entries"]:
        raw = (tmp_path / entry["path"]).read_bytes()
        evidence = raw[entry["start_byte"]:entry["end_byte"]]
        assert hashlib.sha256(evidence).hexdigest() == entry["evidence_sha256"]
        assert evidence.decode("utf-8") == entry["signature"]

    tampered = copy.deepcopy(payload)
    tampered["entries"][0]["signature"] = "def invented():"
    assert not verify_repository_map_commitment(tampered)


def test_map_refuses_stale_source_instead_of_emitting_old_signature(tmp_path: Path) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)
    _write(tmp_path, "pkg/core.py", "def replacement(value):\n    return value\n")

    payload = build_verified_repository_map(
        tmp_path,
        index,
        "hub",
        index_digest="sha256:stale-index",
    )

    assert all(entry["path"] != "pkg/core.py" for entry in payload["entries"])
    assert payload["receipt"]["omissions_by_reason"]["stale-index"] == 1


def test_component_identity_is_checkout_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _project(first)
    _project(second)

    first_payload = _build(first)
    second_payload = _build(second)

    assert [entry["component"] for entry in first_payload["entries"]] == [
        entry["component"] for entry in second_payload["entries"]
    ]
