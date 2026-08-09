from __future__ import annotations

import os
from pathlib import Path

import pytest

from entroly.repository_intelligence.cache_retention import prune_cache_tree


def _entry(root: Path, name: str, size: int, mtime_ns: int) -> Path:
    target = root / name[:2] / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * size)
    os.utime(target, ns=(mtime_ns, mtime_ns))
    return target


def test_prunes_oldest_until_byte_and_file_limits_are_both_satisfied(
    tmp_path: Path,
) -> None:
    first = _entry(tmp_path, "aa-old", 40, 10)
    second = _entry(tmp_path, "bb-mid", 40, 20)
    third = _entry(tmp_path, "cc-new", 40, 30)

    report = prune_cache_tree(
        tmp_path,
        max_total_bytes=80,
        max_files=2,
    )
    assert report.bounded
    assert report.removed_files == 1
    assert report.removed_bytes == 40
    assert not first.exists()
    assert second.exists()
    assert third.exists()
    assert report.remaining_files == 2
    assert report.remaining_bytes == 80


def test_ties_are_deterministic_by_relative_path(tmp_path: Path) -> None:
    first = _entry(tmp_path, "aa-entry", 10, 100)
    second = _entry(tmp_path, "bb-entry", 10, 100)
    report = prune_cache_tree(tmp_path, max_total_bytes=10, max_files=1)
    assert report.bounded
    assert not first.exists()
    assert second.exists()


def test_protected_current_entry_is_never_deleted(tmp_path: Path) -> None:
    protected = _entry(tmp_path, "aa-current", 60, 1)
    removable = _entry(tmp_path, "bb-old", 60, 2)
    report = prune_cache_tree(
        tmp_path,
        max_total_bytes=60,
        max_files=1,
        protected=[protected],
    )
    assert protected.exists()
    assert not removable.exists()
    assert report.bounded


def test_excluded_snapshot_subtree_is_not_counted_or_deleted(tmp_path: Path) -> None:
    parse = _entry(tmp_path, "aa-parse", 20, 1)
    snapshot = tmp_path / "index-snapshots" / "ff" / "snapshot.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"y" * 200)

    report = prune_cache_tree(
        tmp_path,
        max_total_bytes=0,
        max_files=0,
        excluded_top_level=["index-snapshots"],
    )
    assert not parse.exists()
    assert snapshot.exists()
    assert report.scanned_bytes == 20
    assert report.remaining_bytes == 0


def test_symlink_cache_artifact_is_never_followed_or_deleted(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "aa" / "linked.json"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    report = prune_cache_tree(tmp_path, max_total_bytes=0, max_files=0)
    assert outside.read_text(encoding="utf-8") == "secret"
    assert link.is_symlink()
    assert report.scanned_files == 0


def test_missing_cache_directory_is_already_bounded(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    report = prune_cache_tree(missing, max_total_bytes=1, max_files=1)
    assert report.bounded
    assert report.scanned_files == 0
    assert report.errors == 0
