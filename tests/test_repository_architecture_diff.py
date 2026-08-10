from __future__ import annotations

import copy
from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.architecture_diff import (
    build_verified_architecture_diff,
    verify_architecture_diff_commitment,
)
from entroly.repository_intelligence.verified_architecture import (
    build_verified_architecture,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _architecture(root: Path, digest: str) -> dict[str, object]:
    return build_verified_architecture(
        root,
        build_repository_index(root),
        index_digest=digest,
    )


def test_architecture_diff_binds_inputs_and_explains_structural_change(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a.py", "import b\n")
    _write(tmp_path, "b.py", "VALUE = 1\n")
    before = _architecture(tmp_path, "sha256:before")

    _write(tmp_path, "b.py", "import a\nVALUE = 2\n")
    _write(tmp_path, "c.py", "import b\n")
    after = _architecture(tmp_path, "sha256:after")
    payload = build_verified_architecture_diff(before, after)

    assert payload["schema_version"] == "entroly.verified-architecture-diff.v1"
    assert payload["receipt"]["input_commitments_verified"] is True
    assert payload["counts"]["files_added"] == 1
    assert payload["counts"]["files_modified"] == 1
    assert payload["files"]["added"] == ["c.py"]
    assert payload["files"]["modified"] == ["b.py"]
    assert {"source": "b.py", "target": "a.py"} in (
        payload["dependency_edges"]["added"]
    )
    assert payload["cycles"]["introduced"][0]["members"] == ["a.py", "b.py"]
    assert any(
        move["path"] == "a.py" and move["delta"] == -1
        for move in payload["layer_moves"]
    )
    assert verify_architecture_diff_commitment(payload)


def test_architecture_diff_reports_resolved_cycles_in_reverse(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "import b\n")
    _write(tmp_path, "b.py", "import a\n")
    cyclic = _architecture(tmp_path, "sha256:cyclic")
    _write(tmp_path, "b.py", "VALUE = 1\n")
    acyclic = _architecture(tmp_path, "sha256:acyclic")
    payload = build_verified_architecture_diff(cyclic, acyclic)
    assert payload["counts"]["cycles_resolved"] == 1
    assert payload["cycles"]["resolved"][0]["members"] == ["a.py", "b.py"]


def test_architecture_diff_rejects_tampered_inputs_and_its_own_tampering(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a.py", "VALUE = 1\n")
    before = _architecture(tmp_path, "sha256:before")
    _write(tmp_path, "a.py", "VALUE = 2\n")
    after = _architecture(tmp_path, "sha256:after")
    tampered_input = copy.deepcopy(before)
    tampered_input["entrypoints"].append("invented.py")
    with pytest.raises(ValueError, match="before architecture commitment"):
        build_verified_architecture_diff(tampered_input, after)

    payload = build_verified_architecture_diff(before, after)
    tampered_diff = copy.deepcopy(payload)
    tampered_diff["counts"]["files_modified"] = 0
    assert not verify_architecture_diff_commitment(tampered_diff)


def test_architecture_diff_truncates_each_change_class_visibly(tmp_path: Path) -> None:
    _write(tmp_path, "root.py", "VALUE = 1\n")
    before = _architecture(tmp_path, "sha256:before")
    for position in range(4):
        _write(tmp_path, f"new_{position}.py", "import root\n")
    after = _architecture(tmp_path, "sha256:after")
    payload = build_verified_architecture_diff(before, after, limit=1)
    assert payload["counts"]["files_added"] == 4
    assert len(payload["files"]["added"]) == 1
    assert payload["truncation"]["files_added_omitted"] == 3
    assert verify_architecture_diff_commitment(payload)
