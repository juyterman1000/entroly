from __future__ import annotations

import copy
from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_snapshot import (
    build_verified_graph_snapshot,
    check_verified_graph_snapshot,
    load_verified_graph_snapshot,
    verify_graph_snapshot_check_commitment,
    verify_graph_snapshot_commitment,
)


def _project(root: Path) -> None:
    (root / "api.py").write_text(
        "from service import execute\ndef invoke():\n    return execute()\n",
        encoding="utf-8",
    )
    (root / "service.py").write_text(
        "def execute():\n    return 1\n", encoding="utf-8"
    )


def _snapshot(root: Path) -> tuple[dict[str, object], str]:
    index = build_repository_index(root)
    digest = RepositoryIntelligenceService._digest_index(index)
    return build_verified_graph_snapshot(index, index_digest=digest), digest


def test_snapshot_is_root_independent_importable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _project(first)
    _project(second)
    snapshot, digest = _snapshot(first)
    other, other_digest = _snapshot(second)
    assert snapshot == other
    assert digest == other_digest
    assert verify_graph_snapshot_commitment(snapshot)

    imported = load_verified_graph_snapshot(second, snapshot)
    assert imported.root == str(second.resolve())
    assert set(imported.symbols) == set(build_repository_index(second).symbols)

    tampered = copy.deepcopy(snapshot)
    tampered["graph"]["symbols"][0]["name"] = "invented"
    assert not verify_graph_snapshot_commitment(tampered)


def test_import_rejects_source_drift_and_check_reports_it(tmp_path: Path) -> None:
    _project(tmp_path)
    snapshot, _digest = _snapshot(tmp_path)
    (tmp_path / "service.py").write_text(
        "def execute():\n    return 2\n", encoding="utf-8"
    )
    current = build_repository_index(tmp_path)
    current_digest = RepositoryIntelligenceService._digest_index(current)
    with pytest.raises(ValueError, match="source drift"):
        load_verified_graph_snapshot(tmp_path, snapshot)
    checked = check_verified_graph_snapshot(
        tmp_path,
        current,
        snapshot,
        index_digest=current_digest,
    )
    assert checked["snapshot_commitment_valid"] is True
    assert checked["snapshot_importable"] is False
    assert checked["in_sync"] is False
    assert "service.py" in checked["import_error"]
    assert verify_graph_snapshot_check_commitment(checked)


def test_check_proves_an_unchanged_graph_is_in_sync(tmp_path: Path) -> None:
    _project(tmp_path)
    snapshot, digest = _snapshot(tmp_path)
    current = build_repository_index(tmp_path)
    checked = check_verified_graph_snapshot(
        tmp_path, current, snapshot, index_digest=digest
    )
    assert checked["snapshot_importable"] is True
    assert checked["in_sync"] is True
    assert all(
        values["only_current_count"] == values["only_snapshot_count"] == 0
        for values in checked["drift"].values()
    )
