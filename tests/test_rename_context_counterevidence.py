"""Deliver retained compiler witnesses as exact source, without safety claims."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.rename_context import build_rename_context, render_rename_context
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_refactor import apply_verified_rename_plan


TARGET = b"def execute():\n    return 1\n"
BLOCKERS = {
    "function": b"def perform():\n    return 2\n",
    "assignment": b"perform = 'sentinel'\n",
    "import-alias": b"from math import sqrt as perform\n",
    "unmodified-load": b"def observe():\n    return perform\n",
}
LEGACY_SELECTION = "top-level-Python-statements-covering-planned-edits"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _project(root: Path, raw: bytes):
    (root / "source.py").write_bytes(raw)
    service = RepositoryIntelligenceService(root)
    plan = service.rename_preview("execute", "perform")
    assert plan["resolution"] == "resolved"
    return service, plan, build_repository_index(root)


def _pages(service: RepositoryIntelligenceService) -> list[dict]:
    pages, cursor, seen = [], None, set()
    for _ in range(100):
        page = service.rename_context("execute", "perform", byte_budget=4096, cursor=cursor)
        wire = render_rename_context(page).encode("utf-8")
        assert len(wire) == page["receipt"]["wire_bytes"] <= 4096
        assert wire.endswith(b"\n")
        assert page["receipt"]["byte_budget"] == 4096
        assert page["receipt"]["remote_calls"] == page["receipt"]["writes_performed"] == 0
        assert page["scope"]["reference_completeness"] == "not-proven"
        assert page["scope"]["semantic_status"] == "not-checked"
        assert page["delivery"]["completeness_scope"] == "selected-plan-regions-only"
        pages.append(page)
        cursor = page["delivery"]["next_cursor"]
        if cursor is None:
            return pages
        assert cursor not in seen, "rename-context repeated a cursor without making progress"
        seen.add(cursor)
    pytest.fail("rename-context pagination did not terminate")


def _recover(pages: list[dict], raw: bytes) -> dict[tuple[int, int], bytes]:
    """Verify transport against original bytes, including contiguous UTF-8 chunks."""
    recovered = {}
    for page in pages:
        assert page["receipt"]["snapshot_sha256"] == pages[0]["receipt"]["snapshot_sha256"]
        assert page["receipt"]["source_bytes"] == sum(
            len(fragment["text"].encode("utf-8")) for fragment in page["fragments"]
        )
        for fragment in page["fragments"]:
            assert fragment["path"] == "source.py"
            start, end = fragment["start_byte"], fragment["end_byte"]
            a, b = fragment["chunk_start_byte"], fragment["chunk_end_byte"]
            assert 0 <= start <= a < b <= end <= len(raw)
            piece = fragment["text"].encode("utf-8")
            assert piece == raw[a:b]
            assert fragment["file_sha256"] == _sha(raw)
            assert fragment["region_sha256"] == _sha(raw[start:end])
            previous = recovered.get((start, end), b"")
            assert a == start + len(previous)
            recovered[start, end] = previous + piece
    for (start, end), content in recovered.items():
        assert content == raw[start:end]
    assert sum(map(len, recovered.values())) == pages[0]["delivery"]["selected_source_bytes"]
    assert len(recovered) == pages[0]["delivery"]["region_count"]
    return recovered


def _assert_witnesses_delivered(plan: dict, raw: bytes, recovered: dict) -> None:
    conflicts = 0
    for review in plan["binding_review"]["files"]:
        assert review["path"] == "source.py"
        assert review["source_sha256"] == _sha(raw)
        for conflict in review["conflicts"]:
            conflicts += 1
            a, b = conflict["start_byte"], conflict["end_byte"]
            assert raw[a:b] == conflict["identifier"].encode("utf-8")
            assert _sha(raw[a:b]) == conflict["evidence_sha256"]
            for start, end in [(a, b), *conflict.get("existing_binding_ranges", [])]:
                assert any(left <= start < end <= right for left, right in recovered), (
                    "retained counterevidence is absent from delivered source", start, end
                )
    assert conflicts > 0


@pytest.mark.parametrize("kind", BLOCKERS)
def test_unchanged_blocker_statement_is_delivered_as_original_source(tmp_path: Path, kind: str):
    blocker = BLOCKERS[kind]
    raw = TARGET + blocker
    service, plan, index = _project(tmp_path, raw)
    assert plan["binding_review"]["status"] == "conflict"
    blocker_start = len(TARGET)
    assert all(change["end_byte"] <= blocker_start for change in plan["changes"])
    pages = _pages(service)
    recovered = _recover(pages, raw)
    assert recovered.get((blocker_start, len(raw))) == blocker
    _assert_witnesses_delivered(plan, raw, recovered)
    assert "retained-binding-conflicts" in pages[0]["scope"]["selection"]
    assert build_rename_context(tmp_path, index, plan, byte_budget=4096) == pages[0]
    assert (tmp_path / "source.py").read_bytes() == raw


@pytest.mark.parametrize("kind", ["function", "unmodified-load"])
def test_paginated_counterevidence_preserves_unicode_crlf_and_exact_spans(tmp_path: Path, kind: str):
    target = TARGET.replace(b"\n", b"\r\n")
    header = b"def perform():\r\n" if kind == "function" else b"def observe():\r\n"
    body = b"    return 2\r\n" if kind == "function" else b"    return perform\r\n"
    blocker = header + ("    # α漢🙂 \\\"\t\r\n" * 260).encode("utf-8") + body
    prefix = "# π before byte coordinates\r\n".encode("utf-8")
    raw = prefix + target + blocker
    service, plan, _index = _project(tmp_path, raw)
    pages = _pages(service)
    assert len(pages) > 2
    recovered = _recover(pages, raw)
    assert recovered == {
        (len(prefix), len(prefix) + len(target)): target,
        (len(prefix) + len(target), len(raw)): blocker,
    }
    _assert_witnesses_delivered(plan, raw, recovered)
    assert (tmp_path / "source.py").read_bytes() == raw


@pytest.mark.parametrize("tail, frozen_wire_sha", [
    (b"", "fceae0048052feb26daf6d14736c06a3fd905165e0326c9c66d039ef631becac"),
    (b"def observe(perform):\n    return perform\n",
     "71021e5caf430e0b171af1266ca3df75b6c8b55fe7def8de3e7d531292665522"),
], ids=["direct", "unrelated-local-parameter"])
def test_conflict_free_payload_keeps_frozen_delivery(tmp_path: Path, tail: bytes, frozen_wire_sha: str):
    raw = TARGET + tail
    service, plan, index = _project(tmp_path, raw)
    assert plan["binding_review"]["conflict_count"] == 0
    pages = _pages(service)
    assert len(pages) == 1
    page = pages[0]
    assert page["scope"]["selection"] == LEGACY_SELECTION
    assert _recover(pages, raw) == {(0, len(TARGET)): TARGET}
    assert build_rename_context(tmp_path, index, plan, byte_budget=4096) == page
    # Frozen on dcf36c7 with the ordinary production pager, before the fix.
    # Only environment-dependent commitments are masked, preserving their
    # lengths. Every other field and the formatter's exact bytes stay frozen.
    normalized = copy.deepcopy(page)
    normalized["plan"]["sha256"] = "0" * 64
    normalized["plan"]["index_digest"] = "sha256:" + "0" * 64
    normalized["receipt"]["snapshot_sha256"] = "0" * 64
    assert _sha(render_rename_context(normalized).encode("utf-8")) == frozen_wire_sha
    assert (tmp_path / "source.py").read_bytes() == raw


def test_cursor_invalidated_when_only_unmodified_blocker_source_changes(tmp_path: Path):
    raw = TARGET + b"def perform():\n" + b"    # retained blocker\n" * 250 + b"    return 2\n"
    service, plan, _index = _project(tmp_path, raw)
    first = service.rename_context("execute", "perform", byte_budget=4096)
    cursor = first["delivery"]["next_cursor"]
    assert cursor is not None, "the unchanged blocker must participate in pagination"
    updated = raw.replace(b"return 2", b"return 3")
    (tmp_path / "source.py").write_bytes(updated)
    refreshed = service.rename_preview("execute", "perform")
    assert [(c["start_byte"], c["end_byte"]) for c in refreshed["changes"]] == [
        (c["start_byte"], c["end_byte"]) for c in plan["changes"]
    ]
    with pytest.raises(ValueError, match="source or task changed"):
        service.rename_context("execute", "perform", byte_budget=4096, cursor=cursor)
    _assert_witnesses_delivered(refreshed, updated, _recover(_pages(service), updated))


@pytest.mark.parametrize("corruption", [
    "source-sha", "primary-sha", "primary-negative", "primary-empty", "primary-past-eof",
    "primary-bool", "primary-identifier", "primary-unrelated-identifier",
    "existing-negative", "existing-empty", "existing-bool", "existing-wrong-identifier",
])
def test_invalid_retained_counterevidence_is_rejected(tmp_path: Path, corruption: str):
    raw = TARGET + BLOCKERS["function"]
    _service, original, index = _project(tmp_path, raw)
    plan = copy.deepcopy(original)
    review = plan["binding_review"]["files"][0]
    conflict = next(item for item in review["conflicts"] if item.get("existing_binding_ranges"))
    if corruption == "source-sha":
        review["source_sha256"] = "0" * 64
    elif corruption == "primary-sha":
        conflict["evidence_sha256"] = "0" * 64
    elif corruption == "primary-negative":
        conflict["start_byte"] = -1
    elif corruption == "primary-empty":
        conflict["end_byte"] = conflict["start_byte"]
    elif corruption == "primary-past-eof":
        conflict["end_byte"] = len(raw) + 1
    elif corruption == "primary-bool":
        conflict["start_byte"] = True
    elif corruption == "primary-identifier":
        conflict["identifier"] = "perform"  # Actual primary preimage is execute.
    elif corruption == "primary-unrelated-identifier":
        start = raw.index(b"return")
        conflict.update(start_byte=start, end_byte=start + 6, identifier="return",
                        evidence_sha256=_sha(b"return"))
    elif corruption == "existing-negative":
        conflict["existing_binding_ranges"] = [[-1, 2]]
    elif corruption == "existing-empty":
        conflict["existing_binding_ranges"] = [[len(TARGET), len(TARGET)]]
    elif corruption == "existing-bool":
        conflict["existing_binding_ranges"] = [[True, 2]]
    elif corruption == "existing-wrong-identifier":
        conflict["existing_binding_ranges"] = [[raw.index(b"execute"), raw.index(b"execute") + 7]]
    else:
        raise AssertionError(corruption)
    # Exercise the context boundary with a planner-shaped review, not apply's
    # independent plan-commitment gate. No edits or source bytes are changed.
    assert plan["changes"] == original["changes"]
    with pytest.raises(ValueError, match="counterevidence"):
        build_rename_context(tmp_path, index, plan, byte_budget=4096)
    assert (tmp_path / "source.py").read_bytes() == raw


@pytest.mark.parametrize("kind", BLOCKERS)
def test_incomplete_acknowledgement_cannot_authorize_a_binding_conflict(tmp_path: Path, kind: str):
    raw = TARGET + BLOCKERS[kind]
    _service, plan, index = _project(tmp_path, raw)
    assert plan["binding_review"]["conflict_count"] > 0
    assert plan["risk"]["requires_incomplete_acknowledgement"] is True
    with pytest.raises(ValueError, match="binding conflict"):
        apply_verified_rename_plan(
            tmp_path, index, plan, index_digest=plan["index_digest"],
            expected_plan_sha256=plan["receipt"]["plan_sha256"], acknowledge_incomplete=True,
        )
    assert (tmp_path / "source.py").read_bytes() == raw
