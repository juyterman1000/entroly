"""Staleness of the indexed corpus must be readable in returned data.

A loaded index is a cache of bytes that were on disk at ingest time, and
nothing on the retrieval path re-checks disk. Measured on a two-file tree
before this surface existed: index, edit one file, then query the same
long-lived engine. It served the pre-edit body and reported ``relevance 1.3``,
``query_coverage 1.0`` and ``residual_risk 0.0``, with no field anywhere named
freshness, age or mtime. The only mitigation was a background poll whose
findings reached ``logger.info`` / ``logger.warning`` -- invisible on a stdio
transport, where nothing consumes log records.

These tests pin the two halves of the repair:

* ``index_freshness()`` answers, from returned data, which indexed sources no
  longer match disk.
* the background reconciliation thread's counters, errors and its
  "incomplete" condition are published as a value a caller can read.

The status vocabulary is the point as much as the counts. "I could not check"
and "nothing is stale" are different answers, and a surface that merges them
reproduces the original defect under a new field name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from entroly import auto_index as auto_index_module
from entroly.auto_index import (
    _publish_incremental_scan,
    _summarize_incremental_scan,
    auto_index,
    index_freshness,
    last_incremental_scan,
    reconcile_index,
    start_incremental_watcher,
)
from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine

ORIGINAL = "def charge(amount):\n    # ORIGINAL_BODY\n    return amount * 100\n"
EDITED = "def charge(amount):\n    # EDITED_BODY_IS_LONGER\n    return amount * 1000\n"
OTHER = "def describe():\n    return 'an unrelated helper'\n"


@pytest.fixture(autouse=True)
def _isolated_watcher_state():
    """Keep the published watcher outcome from leaking between tests."""
    saved = last_incremental_scan()
    yield
    _publish_incremental_scan(saved)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "payments.py").write_text(ORIGINAL, encoding="utf-8")
    (project / "src" / "other.py").write_text(OTHER, encoding="utf-8")
    return project


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EntrolyEngine:
    state = tmp_path / "state"
    monkeypatch.setenv("ENTROLY_DIR", str(state))
    monkeypatch.setenv("ENTROLY_VAULT", str(state / "vault"))
    return EntrolyEngine(
        EntrolyConfig(
            use_persistent_index=False,
            checkpoint_dir=state / "checkpoints",
        )
    )


def _indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[EntrolyEngine, Path]:
    project = _project(tmp_path)
    engine = _engine(tmp_path, monkeypatch)
    result = auto_index(engine, str(project), seed_beliefs=False)
    assert result["files_indexed"] >= 2, result
    # Indexing must leave a record behind, or the freshness check has nothing
    # to compare against and can only answer "unavailable" forever.
    assert result["freshness_record"]["status"] == "persisted", result["freshness_record"]
    return engine, project


def test_unchanged_tree_reports_zero_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, project = _indexed(tmp_path, monkeypatch)

    report = index_freshness(engine, str(project))

    assert report["status"] == "fresh", report
    assert report["stale_count"] == 0
    assert report["stale_paths"] == []
    assert report["unknown_count"] == 0
    assert report["not_fresh_paths"] == []
    assert report["fresh_count"] == report["sources_tracked"] >= 2
    assert report["method"] == "stat"


def test_edited_source_is_reported_stale_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, project = _indexed(tmp_path, monkeypatch)
    assert index_freshness(engine, str(project))["status"] == "fresh"

    (project / "src" / "payments.py").write_text(EDITED, encoding="utf-8")

    report = index_freshness(engine, str(project))

    assert report["status"] == "stale", report
    assert report["stale_count"] == 1
    # Naming the file is the whole point: "something is stale" is not
    # actionable, and the caller cannot re-read a file it cannot identify.
    assert report["stale_paths"] == ["src/payments.py"]
    assert "src/other.py" not in report["not_fresh_paths"]
    entry = report["stale_sources"][0]
    assert entry["source"] == "file:src/payments.py"
    assert entry["reason"] in {"size_changed", "mtime_changed", "size_and_mtime_changed"}
    assert entry["indexed"]["size_bytes"] != entry["observed"]["size_bytes"]


def test_deleted_source_is_stale_not_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracked path that is gone proves the index is stale, not unverifiable.

    The index still holds that file's bytes and will still serve them, so the
    honest answer is "stale", and it must be distinguishable from a stat that
    was merely refused -- which the next test covers.
    """
    engine, project = _indexed(tmp_path, monkeypatch)

    (project / "src" / "other.py").unlink()

    report = index_freshness(engine, str(project))

    assert report["status"] == "stale", report
    assert "src/other.py" in report["stale_paths"]
    assert "src/other.py" in report["not_fresh_paths"]
    reasons = {item["path"]: item["reason"] for item in report["stale_sources"]}
    assert reasons["src/other.py"] == "missing"


def test_unreadable_source_is_could_not_check_not_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused stat must land in the "could not check" bucket.

    Counting it as fresh is the exact failure this surface exists to remove:
    the index would be certified current on the strength of a check that never
    happened.
    """
    engine, project = _indexed(tmp_path, monkeypatch)
    blocked = os.path.realpath(str(project / "src" / "payments.py"))
    real_stat = os.stat

    def guarded_stat(path, *args, **kwargs):
        # Only the one file is blocked; everything else reaches the real stat,
        # so the rest of the report stays genuine.
        try:
            if os.path.realpath(os.fspath(path)) == blocked:
                raise PermissionError(13, "Permission denied")
        except (TypeError, ValueError):
            pass
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(auto_index_module.os, "stat", guarded_stat)
    report = index_freshness(engine, str(project))

    assert report["status"] == "partial", report
    assert report["unknown_count"] == 1
    assert report["unknown_paths"] == ["src/payments.py"]
    assert report["unknown_sources"][0]["reason"].startswith("stat_failed:")
    # Not stale either -- we did not observe a change, we failed to look.
    assert "src/payments.py" not in report["stale_paths"]
    assert "src/payments.py" in report["not_fresh_paths"]
    # And it must not be quietly folded into the fresh tally.
    assert report["fresh_count"] == report["sources_tracked"] - 1


def test_no_record_reports_unavailable_rather_than_nothing_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine with no ledger cannot certify freshness.

    ``stale_count == 0`` is true here only because nothing was checked, so the
    status has to carry that. A caller branching on ``status == "fresh"`` must
    not be told the tree is current.
    """
    project = _project(tmp_path)
    engine = _engine(tmp_path, monkeypatch)

    report = index_freshness(engine, str(project))

    assert report["status"] == "unavailable", report
    assert report["status"] != "fresh"
    assert report["reason"] in {"never_recorded", "no_tracked_sources"}
    assert report["sources_tracked"] == 0
    assert report["ledger"]["status"] in {"missing", "disabled"}


def test_corrupt_record_reports_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, project = _indexed(tmp_path, monkeypatch)
    ledger = Path(str(engine.config.checkpoint_dir)) / "index_freshness.json"
    assert ledger.exists()
    ledger.write_text("{not json", encoding="utf-8")

    report = index_freshness(engine, str(project))

    assert report["status"] == "unavailable", report
    assert report["ledger"]["status"] == "unreadable"
    assert report["stale_count"] == 0  # nothing was checked, and the status says so


def test_reconcile_refreshes_the_record_it_acted_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, project = _indexed(tmp_path, monkeypatch)
    (project / "src" / "payments.py").write_text(EDITED, encoding="utf-8")
    assert index_freshness(engine, str(project))["status"] == "stale"

    result = reconcile_index(engine, str(project))

    assert result["freshness_record"]["status"] == "persisted", result["freshness_record"]
    assert index_freshness(engine, str(project))["status"] == "fresh"


def test_content_verification_catches_a_same_signature_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap check has a blind spot, and the report says which check ran.

    A filesystem with coarse mtime granularity can let an edit that preserves
    byte length pass a stat comparison. That is a real limit of the default
    mode, so the stored digest and ``verify_content`` exist to close it, and
    ``method`` tells the caller how strong the claim it just read was.
    """
    engine, project = _indexed(tmp_path, monkeypatch)
    target = project / "src" / "payments.py"
    recorded = os.stat(target)

    same_length = "def charge(amount):\n    # SWAPPED_BODY_\n    return amount * 100\n"
    replacement = same_length.ljust(len(ORIGINAL))[: len(ORIGINAL)]
    assert len(replacement.encode("utf-8")) == len(ORIGINAL.encode("utf-8"))
    assert replacement != ORIGINAL
    target.write_text(replacement, encoding="utf-8")
    os.utime(target, ns=(recorded.st_mtime_ns, recorded.st_mtime_ns))

    cheap = index_freshness(engine, str(project))
    assert cheap["method"] == "stat"
    assert "src/payments.py" not in cheap["stale_paths"]

    thorough = index_freshness(engine, str(project), verify_content=True)
    assert thorough["method"] == "content"
    assert thorough["status"] == "stale", thorough
    assert thorough["stale_paths"] == ["src/payments.py"]
    assert thorough["stale_sources"][0]["reason"] == "content_changed"


def test_watcher_findings_are_returned_not_only_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counters and errors that used to reach only ``logger`` are readable.

    ``files_replaced`` and the "Incremental reconciliation incomplete"
    condition were log lines, which a caller on a stdio transport never sees.
    """
    receipt = {
        "status": "partial",
        "project_dir": str(tmp_path),
        "files_added": 1,
        "files_replaced": 4,
        "files_removed": 2,
        "files_unchanged": 9,
        "pending_changes": 7,
        "rolled_back": False,
        "errors": ["native source removal is unavailable"],
        "stale_sources": ["file:src/payments.py"],
        "unavailable_sources": {"file:src/gone.py": "unreadable"},
        "persistence": {"status": "persisted"},
        "freshness_record": {"status": "persisted"},
        "duration_s": 0.5,
    }

    summary = _summarize_incremental_scan(receipt)

    assert summary["status"] == "partial"
    assert summary["incomplete"] is True
    assert summary["files_replaced"] == 4
    assert summary["pending_changes"] == 7
    assert summary["errors"] == ["native source removal is unavailable"]
    assert summary["stale_sources"] == ["file:src/payments.py"]
    assert summary["unavailable_sources"] == {"file:src/gone.py": "unreadable"}
    assert summary["persistence"] == "persisted"

    _publish_incremental_scan(summary)
    assert last_incremental_scan()["files_replaced"] == 4
    assert last_incremental_scan()["errors"] == summary["errors"]

    # And the freshness report carries it, so one call answers both "is the
    # index behind disk?" and "did the background pass manage to catch up?".
    engine, project = _indexed(tmp_path, monkeypatch)
    _publish_incremental_scan(summary)
    assert index_freshness(engine, str(project))["watcher"]["status"] == "partial"


def test_watcher_returns_a_handle_a_caller_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, project = _indexed(tmp_path, monkeypatch)

    # A long interval so the background pass never fires during the test: the
    # contract under test is the handle, not the poll.
    watcher = start_incremental_watcher(engine, str(project), interval_s=3600)

    assert watcher is not None
    assert watcher.is_alive()
    assert watcher.project_dir == os.path.abspath(str(project))
    assert watcher.interval_s == 3600
    assert isinstance(watcher.last_scan(), dict)
    assert "status" in watcher.last_scan()


def test_existing_return_fields_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freshness was added to the receipts, not carved out of them."""
    engine, project = _indexed(tmp_path, monkeypatch)

    result = reconcile_index(engine, str(project))

    for field in (
        "status",
        "project_dir",
        "discovery_method",
        "files_scanned",
        "files_unchanged",
        "files_added",
        "files_replaced",
        "files_removed",
        "removed_fragments",
        "added_sources",
        "replaced_sources",
        "removed_sources",
        "stale_sources",
        "unavailable_sources",
        "pending_changes",
        "dependency_refresh",
        "persistence",
        "duplicate_ledger",
        "rolled_back",
        "errors",
        "duration_s",
    ):
        assert field in result, field
    assert "freshness_record" in result
