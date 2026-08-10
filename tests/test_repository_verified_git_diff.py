from __future__ import annotations

import copy
import subprocess
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.models import RepositoryLimits
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_git_diff import (
    build_verified_git_architecture_diff,
    verify_git_architecture_diff_commitment,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep the worktree bytes identical to the committed blob on Windows so
    # this test measures semantic source changes, not checkout newline filters.
    target.write_bytes(text.encode("utf-8"))


def _repository(root: Path) -> str:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Entroly Test")
    _write(root, "core.py", "def execute():\n    return 1\n")
    _write(root, "api.py", "from core import execute\nexecute()\n")
    _git(root, "add", "core.py", "api.py")
    _git(root, "commit", "-q", "-m", "baseline")
    return _git(root, "rev-parse", "HEAD")


def test_git_diff_materializes_commit_without_checkout_and_reports_semantics(
    tmp_path: Path,
) -> None:
    baseline = _repository(tmp_path)
    _write(tmp_path, "core.py", "def execute():\n    return 2\n")
    _write(tmp_path, "new.py", "from core import execute\n")
    service = RepositoryIntelligenceService(tmp_path)
    current, digest, _generation = service._snapshot()
    before_head = _git(tmp_path, "rev-parse", "HEAD")
    payload = build_verified_git_architecture_diff(
        tmp_path,
        current,
        current_index_digest=digest,
        ref=baseline,
        limits=RepositoryLimits(),
        build_index=build_repository_index,
    )
    assert _git(tmp_path, "rev-parse", "HEAD") == before_head
    assert payload["base_commit"] == baseline
    assert payload["baseline_materialization"]["checkout_mutated"] is False
    assert payload["architecture_diff"]["files"]["modified"] == ["core.py"]
    assert payload["architecture_diff"]["files"]["added"] == ["new.py"]
    assert payload["architecture_diff"]["counts"]["dependency_edges_added"] == 1
    assert verify_git_architecture_diff_commitment(payload)


def test_git_diff_receipt_detects_tampering(tmp_path: Path) -> None:
    baseline = _repository(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    current, digest, _generation = service._snapshot()
    payload = build_verified_git_architecture_diff(
        tmp_path,
        current,
        current_index_digest=digest,
        ref=baseline,
        limits=RepositoryLimits(),
        build_index=build_repository_index,
    )
    tampered = copy.deepcopy(payload)
    tampered["base_commit"] = "0" * 40
    assert not verify_git_architecture_diff_commitment(tampered)


def test_git_diff_rejects_option_like_ref(tmp_path: Path) -> None:
    _repository(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    current, digest, _generation = service._snapshot()
    try:
        build_verified_git_architecture_diff(
            tmp_path,
            current,
            current_index_digest=digest,
            ref="--upload-pack=evil",
            limits=RepositoryLimits(),
            build_index=build_repository_index,
        )
    except ValueError as exc:
        assert "non-option" in str(exc)
    else:
        raise AssertionError("option-like ref was accepted")
