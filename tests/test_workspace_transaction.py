from __future__ import annotations

from pathlib import Path

import pytest

from entroly.repository_intelligence import workspace_transaction as tx


def _write(root: Path, path: str, data: bytes) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def _artifacts(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (".entroly-backup." in path.name or ".entroly-stage." in path.name)
    )


def test_successful_mixed_transaction_is_clean_and_deterministic(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", b"old-a")
    _write(tmp_path, "delete.py", b"old-delete")

    report = tx.apply_workspace_transaction(
        tmp_path,
        replacements={"a.py": b"new-a"},
        creations={"new.py": b"created"},
        deletions={"delete.py": b"old-delete"},
        expected_originals={
            "a.py": b"old-a",
            "delete.py": b"old-delete",
        },
    )

    assert (tmp_path / "a.py").read_bytes() == b"new-a"
    assert (tmp_path / "new.py").read_bytes() == b"created"
    assert not (tmp_path / "delete.py").exists()
    assert report.rollback_performed is False
    assert report.rollback_complete is True
    assert report.recovery_artifacts == ()
    assert _artifacts(tmp_path) == []


def test_mid_commit_failure_restores_every_completed_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "a.py", b"old-a")
    _write(tmp_path, "b.py", b"old-b")
    original_replace = tx._replace
    failed = False

    def fail_second_stage(source: Path, target: Path) -> None:
        nonlocal failed
        if (
            not failed
            and ".entroly-stage." in source.name
            and target.name == "b.py"
        ):
            failed = True
            raise OSError("injected commit failure")
        original_replace(source, target)

    monkeypatch.setattr(tx, "_replace", fail_second_stage)

    with pytest.raises(tx.WorkspaceTransactionError) as captured:
        tx.apply_workspace_transaction(
            tmp_path,
            replacements={"a.py": b"new-a", "b.py": b"new-b"},
            expected_originals={"a.py": b"old-a", "b.py": b"old-b"},
        )

    report = captured.value.report
    assert report.rollback_performed is True
    assert report.rollback_complete is True
    assert report.rollback_errors == ()
    assert report.recovery_artifacts == ()
    assert (tmp_path / "a.py").read_bytes() == b"old-a"
    assert (tmp_path / "b.py").read_bytes() == b"old-b"
    assert _artifacts(tmp_path) == []


def test_failed_rollback_preserves_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "a.py", b"old-a")
    _write(tmp_path, "b.py", b"old-b")
    original_replace = tx._replace
    commit_failed = False

    def fail_commit_and_a_rollback(source: Path, target: Path) -> None:
        nonlocal commit_failed
        if (
            not commit_failed
            and ".entroly-stage." in source.name
            and target.name == "b.py"
        ):
            commit_failed = True
            raise OSError("injected commit failure")
        if ".entroly-backup." in source.name and target.name == "a.py":
            raise OSError("injected rollback failure")
        original_replace(source, target)

    monkeypatch.setattr(tx, "_replace", fail_commit_and_a_rollback)

    with pytest.raises(
        tx.WorkspaceTransactionError,
        match="rollback incomplete; recovery artifacts preserved",
    ) as captured:
        tx.apply_workspace_transaction(
            tmp_path,
            replacements={"a.py": b"new-a", "b.py": b"new-b"},
            expected_originals={"a.py": b"old-a", "b.py": b"old-b"},
        )

    report = captured.value.report
    assert report.rollback_performed is True
    assert report.rollback_complete is False
    assert report.rollback_errors == ("replace:a.py:OSError",)
    assert report.recovery_artifacts
    assert (tmp_path / "a.py").read_bytes() == b"new-a"
    backup_paths = [tmp_path / path for path in report.recovery_artifacts]
    backups = [path for path in backup_paths if ".entroly-backup." in path.name]
    assert backups
    assert any(path.read_bytes() == b"old-a" for path in backups)


def test_creation_is_removed_when_later_delete_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "delete.py", b"keep-me")
    original_unlink = tx._unlink

    def fail_delete(path: Path) -> None:
        if path.name == "delete.py":
            raise OSError("injected delete failure")
        original_unlink(path)

    monkeypatch.setattr(tx, "_unlink", fail_delete)

    with pytest.raises(tx.WorkspaceTransactionError) as captured:
        tx.apply_workspace_transaction(
            tmp_path,
            creations={"new.py": b"created"},
            deletions={"delete.py": b"keep-me"},
            expected_originals={"delete.py": b"keep-me"},
        )
    assert captured.value.report.rollback_complete is True
    assert not (tmp_path / "new.py").exists()
    assert (tmp_path / "delete.py").read_bytes() == b"keep-me"


def test_preimage_mismatch_fails_before_any_temp_or_mutation(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", b"actual")
    with pytest.raises(ValueError, match="preimage changed"):
        tx.apply_workspace_transaction(
            tmp_path,
            replacements={"a.py": b"new"},
            expected_originals={"a.py": b"expected"},
        )
    assert (tmp_path / "a.py").read_bytes() == b"actual"
    assert _artifacts(tmp_path) == []


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        tx.apply_workspace_transaction(
            tmp_path,
            creations={"../escape.py": b"no"},
        )
    assert not (tmp_path.parent / "escape.py").exists()
