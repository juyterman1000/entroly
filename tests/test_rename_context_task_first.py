"""Task-first delivery must reorder exact source without changing its coverage.

Fixtures are untrusted Python text for the public planner/pager, never imported
or executed. Expected coverage comes from authored byte spans, not AST helpers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.repository_intelligence.rename_context import render_rename_context
from entroly.repository_intelligence.service import RepositoryIntelligenceService


BUDGET = 4096
LEGACY_SELECTION = "top-level-Python-statements-covering-planned-edits"
GUARD = b"raise AssertionError('repository fixtures must never execute')\n"


def _project(root: Path, raw: bytes) -> tuple[RepositoryIntelligenceService, dict]:
    (root / "source.py").write_bytes(raw)
    service = RepositoryIntelligenceService(root)
    plan = service.rename_preview("execute", "perform")
    assert plan["resolution"] == "resolved"
    assert plan["changes"], "fixture must exercise source-verified planned edits"
    return service, plan


def _pages(service: RepositoryIntelligenceService) -> list[dict]:
    pages, seen = [], set()
    cursor = None
    for _ in range(120):
        page = service.rename_context("execute", "perform", byte_budget=BUDGET, cursor=cursor)
        wire = render_rename_context(page).encode("utf-8")
        assert len(wire) == page["receipt"]["wire_bytes"] <= BUDGET
        assert wire.endswith(b"\n")
        assert page["receipt"]["byte_budget"] == BUDGET
        assert page["receipt"]["remote_calls"] == page["receipt"]["writes_performed"] == 0
        assert page["scope"]["reference_completeness"] == "not-proven"
        assert page["scope"]["semantic_status"] == "not-checked"
        assert page["delivery"]["completeness_scope"] == "selected-plan-regions-only"
        assert page["fragments"], "every page must make source-delivery progress"
        pages.append(page)
        cursor = page["delivery"]["next_cursor"]
        if cursor is None:
            return pages
        assert cursor not in seen, "pagination repeated a cursor"
        seen.add(cursor)
    pytest.fail("task-first pagination did not terminate")


def _assert_exact_coverage(
    pages: list[dict], raw: bytes, expected_spans: list[tuple[int, int]],
) -> list[dict]:
    """Check every original selected byte exactly once, regardless of partitions."""
    fragments = []
    regions: dict[tuple[int, int], bytes] = {}
    selected_bytes = sum(end - start for start, end in expected_spans)
    for page in pages:
        assert page["receipt"]["snapshot_sha256"] == pages[0]["receipt"]["snapshot_sha256"]
        assert page["delivery"]["selected_source_bytes"] == selected_bytes
        assert page["delivery"]["region_count"] == pages[0]["delivery"]["region_count"]
        assert page["receipt"]["source_bytes"] == sum(
            len(item["text"].encode("utf-8")) for item in page["fragments"]
        )
        for fragment in page["fragments"]:
            assert fragment["path"] == "source.py"
            start, end = fragment["start_byte"], fragment["end_byte"]
            left, right = fragment["chunk_start_byte"], fragment["chunk_end_byte"]
            assert 0 <= start <= left < right <= end <= len(raw)
            piece = fragment["text"].encode("utf-8")
            assert piece == raw[left:right]
            assert fragment["file_sha256"] == hashlib.sha256(raw).hexdigest()
            assert fragment["region_sha256"] == hashlib.sha256(raw[start:end]).hexdigest()
            previous = regions.get((start, end), b"")
            assert left == start + len(previous), "region chunks skipped or repeated bytes"
            regions[start, end] = previous + piece
            fragments.append(fragment)

    assert len(regions) == pages[0]["delivery"]["region_count"]
    for (start, end), content in regions.items():
        assert content == raw[start:end], "region was not completely delivered"

    # Independent of region boundaries: overlapping regions can each have valid
    # hashes and still duplicate source. Sorting physical chunks exposes that.
    ordered = iter(sorted(fragments, key=lambda item: item["chunk_start_byte"]))
    item = next(ordered, None)
    for start, end in expected_spans:
        position = start
        recovered = bytearray()
        while item is not None and item["chunk_start_byte"] < end:
            assert item["chunk_start_byte"] == position, "selected bytes lost or duplicated"
            position = item["chunk_end_byte"]
            assert position <= end, "delivered bytes outside the original selected statement"
            recovered.extend(item["text"].encode("utf-8"))
            item = next(ordered, None)
        assert position == end, "selected statement prelude or tail was lost"
        assert bytes(recovered) == raw[start:end]
    assert item is None, "unselected source was delivered"
    assert sum(len(item["text"].encode("utf-8")) for item in fragments) == selected_bytes
    return fragments


def _deep_factory() -> tuple[bytes, bytes, bytes]:
    target = b"        def execute(value):\n            return value + 1\n"
    prelude = (
        b"def factory():\n"
        + b"    # factory prelude: not disposable context\n" * 220
        + b"    class Container:\n"
        + b"        # enclosing class prelude: not the target\n" * 220
    )
    tail = (
        b"        marker = 'class tail'\n"
        b"    # factory tail must survive reordering\n"
        b"    return Container\n"
    )
    return prelude + target + tail, target, tail


def test_innermost_target_beyond_8192_source_bytes_arrives_in_first_4096_json_bytes(tmp_path: Path):
    factory, target, _tail = _deep_factory()
    raw = GUARD + factory + b"UNSELECTED = 'outside factory'\n"
    service, _plan = _project(tmp_path, raw)
    start = raw.index(target)
    assert start > 8192
    assert start - raw.index(b"    class Container:") > 8192

    first = service.rename_context("execute", "perform", byte_budget=BUDGET)
    wire = render_rename_context(first).encode("utf-8")
    assert len(wire) == first["receipt"]["wire_bytes"] <= BUDGET
    # A name in plan metadata is insufficient: the actual definition and body
    # must be present in the first source fragment, at their original offsets.
    fragment = first["fragments"][0]
    assert fragment["chunk_start_byte"] == start
    assert fragment["text"].encode("utf-8").startswith(target)
    assert first["scope"]["selection"] != LEGACY_SELECTION
    assert (tmp_path / "source.py").read_bytes() == raw


def test_full_pagination_preserves_factory_prelude_and_tail_exactly_once(tmp_path: Path):
    factory, _target, _tail = _deep_factory()
    raw = GUARD + factory + b"UNSELECTED = 'outside factory'\n"
    service, _plan = _project(tmp_path, raw)
    pages = _pages(service)
    assert len(pages) > 3
    _assert_exact_coverage(pages, raw, [(len(GUARD), len(GUARD) + len(factory))])
    assert (tmp_path / "source.py").read_bytes() == raw


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_decorated_async_focus_preserves_unicode_and_physical_lines(tmp_path: Path, newline: str):
    def encode(text: str) -> bytes:
        return text.replace("\n", newline).encode("utf-8")

    prefix = encode("# π before byte coordinates\n") + GUARD.replace(b"\n", newline.encode())
    prelude = encode("def factory():\n" + "    # α漢🙂 prelude\n" * 420)
    target = encode(
        "    @decorate(\n"
        "        'α漢🙂',\n"
        "    )\n"
        "    @another_decorator\n"
        "    async def execute(café):\n"
        "        return café, '漢🙂'\n"
    )
    tail = encode("    # factory tail α\n    return 'factory tail'\n")
    factory = prelude + target + tail
    raw = prefix + factory + encode("UNSELECTED = 'suffix'\n")
    service, _plan = _project(tmp_path, raw)
    assert raw.index(target) > 8192
    pages = _pages(service)
    fragments = _assert_exact_coverage(pages, raw, [(len(prefix), len(prefix) + len(factory))])
    assert fragments[0]["chunk_start_byte"] == len(prefix) + len(prelude)
    assert fragments[0]["text"].encode("utf-8").startswith(target)
    assert (tmp_path / "source.py").read_bytes() == raw


def test_overlapping_and_sibling_focuses_do_not_duplicate_enclosing_source(tmp_path: Path):
    prelude = b"def factory():\n" + b"    # factory prelude before the rename target\n" * 220
    target = (
        b"    def execute(value):\n"
        b"        # surrounding focus contains the recursive caller focus\n"
        b"        def recurse():\n"
        b"            return execute(value - 1)\n"
        b"        return recurse() if value else 0\n"
    )
    sibling = b"    def call_again():\n        return execute(2)\n"
    tail = b"    # factory tail\n    return call_again\n"
    raw = GUARD + prelude + target + sibling + tail
    service, plan = _project(tmp_path, raw)
    target_start, sibling_start = raw.index(target), raw.index(sibling)
    recursive_call = raw.index(b"execute(value - 1)")
    sibling_call = raw.index(b"execute(2)")
    changes = {change["start_byte"] for change in plan["changes"]}
    assert {recursive_call, sibling_call} <= changes, "fixture must create overlapping and sibling obligations"
    pages = _pages(service)
    fragments = _assert_exact_coverage(pages, raw, [(len(GUARD), len(raw))])
    assert fragments[0]["chunk_start_byte"] == target_start
    assert fragments[0]["text"].encode("utf-8").startswith(b"    def execute(value):\n")
    starts = [item["chunk_start_byte"] for item in fragments]
    assert starts.index(sibling_start) < starts.index(len(GUARD)), "sibling focus must precede factory prelude"
    assert (tmp_path / "source.py").read_bytes() == raw


def test_nested_retained_binding_conflict_precedes_factory_padding(tmp_path: Path):
    prelude = b"def factory():\n" + b"    # factory prelude before target\n" * 280
    target = b"    def execute():\n        return 1\n"
    middle = b"    # distant unchanged binding evidence\n" * 230
    blocker = b"    def perform():\n        return 2\n"
    tail = b"    # factory tail\n    return execute\n"
    raw = GUARD + prelude + target + middle + blocker + tail
    service, plan = _project(tmp_path, raw)
    assert plan["binding_review"]["status"] == "conflict"
    assert plan["binding_review"]["conflict_count"] > 0
    blocker_start = raw.index(blocker)
    binding_start = blocker_start + blocker.index(b"perform")
    retained = [
        tuple(span)
        for review in plan["binding_review"]["files"]
        for conflict in review["conflicts"]
        for span in conflict.get("existing_binding_ranges", [])
    ]
    assert (binding_start, binding_start + len(b"perform")) in retained
    assert all(change["start_byte"] < blocker_start for change in plan["changes"])
    pages = _pages(service)
    fragments = _assert_exact_coverage(pages, raw, [(len(GUARD), len(raw))])
    starts = [item["chunk_start_byte"] for item in fragments]
    assert starts[0] == raw.index(target)
    assert starts.index(blocker_start) < starts.index(len(GUARD)), "retained conflict must precede prelude"
    assert "retained-binding-conflicts" in pages[0]["scope"]["selection"]
    assert pages[0]["plan"]["binding_review"] == "conflict"
    assert (tmp_path / "source.py").read_bytes() == raw


def test_top_level_only_selection_and_source_order_are_unchanged(tmp_path: Path):
    caller = b"def observe():\n    return execute(1)\n"
    gap = b"UNSELECTED = 'between selected statements'\n"
    target = b"@decorate\ndef execute(value):\n    return value + 1\n"
    raw = GUARD + caller + gap + target + b"def unrelated():\n    return 'excluded'\n"
    service, plan = _project(tmp_path, raw)
    assert plan["binding_review"]["conflict_count"] == 0
    pages = _pages(service)
    assert len(pages) == 1
    target_start = len(GUARD) + len(caller) + len(gap)
    fragments = _assert_exact_coverage(pages, raw, [
        (len(GUARD), len(GUARD) + len(caller)),
        (target_start, target_start + len(target)),
    ])
    assert [item["text"].encode("utf-8") for item in fragments] == [caller, target]
    assert pages[0]["scope"]["selection"] == LEGACY_SELECTION
    assert (tmp_path / "source.py").read_bytes() == raw


def test_overlapping_top_level_physical_spans_keep_the_original_region_boundaries(tmp_path: Path):
    definition = b"def execute(value):\n    return value\n"
    # Both statements start on the same physical line. The retained name is
    # on that line, while the edited call extends the selected region.
    use = b"perform; (\n" + b"    # call prelude\n" * 350 + b"    execute(1)\n)\n"
    raw = GUARD + definition + use
    service, plan = _project(tmp_path, raw)
    assert plan["binding_review"]["conflict_count"] > 0
    pages = _pages(service)
    split = len(GUARD) + len(definition)
    fragments = _assert_exact_coverage(pages, raw, [(len(GUARD), split), (split, len(raw))])
    assert {(f["start_byte"], f["end_byte"]) for f in fragments} == {
        (len(GUARD), split), (split, len(raw)),
    }
    assert pages[0]["delivery"]["region_count"] == 2
    assert "task-regions-first" not in pages[0]["scope"]["selection"]


@pytest.mark.parametrize(
    "decorator",
    [b"    @decorate(\n" + b"        # decorator prelude\n" * 400 + b"        'tag'\n    )\n",
     b"    @(\n        decorate\n    )\n"],
    ids=["oversized", "grouped"],
)
def test_deferred_decorator_prefix_remains_recoverable(tmp_path: Path, decorator: bytes):
    prelude = b"def factory():\n" + b"    # factory prelude\n" * 350
    target = b"    def execute(value):\n        return value + 1\n"
    raw = GUARD + prelude + decorator + target + b"    return 'factory tail'\n"
    service, _plan = _project(tmp_path, raw)
    pages = _pages(service)
    fragments = _assert_exact_coverage(pages, raw, [(len(GUARD), len(raw))])
    assert fragments[0]["chunk_start_byte"] == raw.index(target)
    assert fragments[0]["text"].encode("utf-8").startswith(target)
    prefix_start = len(GUARD) + len(prelude)
    assert any(f["start_byte"] <= prefix_start < f["end_byte"] for f in fragments[1:])


@pytest.mark.parametrize("changed_part", ["prelude", "tail"])
def test_cursor_rejects_changed_surrounding_bytes_with_unchanged_focus(tmp_path: Path, changed_part: str):
    factory, target, _tail = _deep_factory()
    raw = GUARD + factory
    service, plan = _project(tmp_path, raw)
    first = service.rename_context("execute", "perform", byte_budget=BUDGET)
    cursor = first["delivery"]["next_cursor"]
    assert cursor is not None
    before = b"factory prelude:" if changed_part == "prelude" else b"factory tail must"
    after = b"factory PRELUDE:" if changed_part == "prelude" else b"factory TAIL must"
    updated = raw.replace(before, after, 1)
    assert updated != raw and len(updated) == len(raw)
    start = raw.index(target)
    assert updated[start:start + len(target)] == target
    (tmp_path / "source.py").write_bytes(updated)
    refreshed = service.rename_preview("execute", "perform")
    assert [(item["start_byte"], item["end_byte"]) for item in refreshed["changes"]] == [
        (item["start_byte"], item["end_byte"]) for item in plan["changes"]
    ]
    with pytest.raises(ValueError, match="source or task changed"):
        service.rename_context("execute", "perform", byte_budget=BUDGET, cursor=cursor)
    assert (tmp_path / "source.py").read_bytes() == updated
