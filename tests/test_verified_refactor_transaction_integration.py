from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.repository_intelligence import verified_move, verified_refactor
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex
from entroly.repository_intelligence.workspace_transaction import (
    WorkspaceTransactionError,
    WorkspaceTransactionReport,
)


def _index(root: Path, path: str, source: bytes, language: str = "python") -> RepositoryIndex:
    return RepositoryIndex(
        root=str(root),
        files={
            path: FileRecord(
                path=path,
                language=language,
                sha256=hashlib.sha256(source).hexdigest(),
                byte_length=len(source),
                line_count=max(1, source.count(b"\n") + 1),
                is_test=False,
            )
        },
    )


def test_rename_apply_delegates_exact_preimages_and_replacements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"def old():\n    return old()\n"
    path = "main.py"
    (tmp_path / path).write_bytes(source)
    index = _index(tmp_path, path, source)
    start = source.index(b"old")
    end = start + 3
    plan = {
        "operation": "rename",
        "resolution": "resolved",
        "index_digest": "sha256:index",
        "risk": {"requires_incomplete_acknowledgement": False},
        "changes": [{
            "path": path,
            "start_byte": start,
            "end_byte": end,
            "old_identifier": "old",
            "new_identifier": "new",
            "evidence_sha256": hashlib.sha256(source[start:end]).hexdigest(),
        }],
        "receipt": {"plan_sha256": "plan"},
    }
    monkeypatch.setattr(verified_refactor, "verify_refactor_plan_commitment", lambda value: True)
    monkeypatch.setattr(verified_refactor, "_syntax_status", lambda path, raw: "valid-python")
    captured: dict[str, object] = {}

    def fake_transaction(root, **kwargs):
        captured.update(kwargs)
        return WorkspaceTransactionReport(
            mutation_count=1,
            rollback_performed=False,
            rollback_complete=True,
            completed_mutations=("replace:main.py",),
            recovery_artifacts=(),
            rollback_errors=(),
        )

    monkeypatch.setattr(verified_refactor, "apply_workspace_transaction", fake_transaction)
    result = verified_refactor.apply_verified_rename_plan(
        tmp_path,
        index,
        plan,
        index_digest="sha256:index",
        expected_plan_sha256="plan",
    )
    assert captured["expected_originals"] == {path: source}
    assert captured["replacements"] == {path: source[:start] + b"new" + source[end:]}
    assert result["rollback_complete"] is True
    assert result["workspace_transaction"]["schema_version"] == (
        "entroly.workspace-transaction.v1"
    )


def test_public_refactor_preserves_recovery_error_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"old"
    path = "main.py"
    (tmp_path / path).write_bytes(source)
    index = _index(tmp_path, path, source)
    plan = {
        "operation": "rename",
        "resolution": "resolved",
        "index_digest": "sha256:index",
        "risk": {"requires_incomplete_acknowledgement": False},
        "changes": [{
            "path": path,
            "start_byte": 0,
            "end_byte": 3,
            "old_identifier": "old",
            "new_identifier": "new",
            "evidence_sha256": hashlib.sha256(b"old").hexdigest(),
        }],
        "receipt": {"plan_sha256": "plan"},
    }
    monkeypatch.setattr(verified_refactor, "verify_refactor_plan_commitment", lambda value: True)
    monkeypatch.setattr(verified_refactor, "_syntax_status", lambda path, raw: "unverified-syntax")
    report = WorkspaceTransactionReport(
        mutation_count=1,
        rollback_performed=True,
        rollback_complete=False,
        completed_mutations=("replace:main.py",),
        recovery_artifacts=(".main.py.entroly-backup.123",),
        rollback_errors=("replace:main.py:OSError",),
    )

    def fail_transaction(*args, **kwargs):
        raise WorkspaceTransactionError("rollback incomplete", report)

    monkeypatch.setattr(verified_refactor, "apply_workspace_transaction", fail_transaction)
    with pytest.raises(WorkspaceTransactionError) as captured:
        verified_refactor.apply_verified_rename_plan(
            tmp_path,
            index,
            plan,
            index_digest="sha256:index",
            expected_plan_sha256="plan",
        )
    assert captured.value.report.rollback_complete is False
    assert captured.value.report.recovery_artifacts == (
        ".main.py.entroly-backup.123",
    )


def test_file_move_apply_maps_move_to_create_and_verified_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"pub fn run() void {}\n"
    source_path = "src/old.zig"
    target_path = "src/new.zig"
    source_file = tmp_path / source_path
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(source)
    index = _index(tmp_path, source_path, source, "zig")
    plan = {
        "operation": "file-move",
        "safe_to_apply": True,
        "index_digest": "sha256:index",
        "source_path": source_path,
        "target_path": target_path,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "risk": {"requires_incomplete_acknowledgement": False},
        "changes": [],
        "receipt": {"plan_sha256": "plan"},
    }
    monkeypatch.setattr(verified_move, "verify_refactor_plan_commitment", lambda value: True)
    monkeypatch.setattr(verified_move, "_syntax_status", lambda path, raw: "valid-zig")
    captured: dict[str, object] = {}

    def fake_transaction(root, **kwargs):
        captured.update(kwargs)
        return WorkspaceTransactionReport(
            mutation_count=2,
            rollback_performed=False,
            rollback_complete=True,
            completed_mutations=(f"create:{target_path}", f"delete:{source_path}"),
            recovery_artifacts=(),
            rollback_errors=(),
        )

    monkeypatch.setattr(verified_move, "apply_workspace_transaction", fake_transaction)
    result = verified_move.apply_verified_file_move_plan(
        tmp_path,
        index,
        plan,
        index_digest="sha256:index",
        expected_plan_sha256="plan",
    )
    assert captured["creations"] == {target_path: source}
    assert captured["deletions"] == {source_path: source}
    assert captured["expected_originals"] == {source_path: source}
    assert captured["replacements"] == {}
    assert target_path in captured["creation_modes"]
    assert result["rollback_complete"] is True
