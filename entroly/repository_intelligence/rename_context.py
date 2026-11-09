"""Exact, task-bound source pages for reviewing a proposed Python rename.

This is delivery of selected evidence, never a semantic safety certificate.
The rename planner remains the authority for edits and omissions.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from bisect import bisect_left
from itertools import chain, pairwise
from pathlib import Path

from .models import RepositoryIndex
from .verified_refactor_impl import _verified_source

RENAME_CONTEXT_SCHEMA = "entroly.rename-context.v1"
_MAX_FOCUSED_DECORATOR_PREFIX_BYTES = 1024


def render_rename_context(payload: dict) -> str:
    """The measured UTF-8 representation shared by CLI and focused MCP."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def validate_byte_budget(value: int) -> None:
    if type(value) is not int or not 4096 <= value <= 262_144:
        raise ValueError("byte_budget must be an integer from 4096 to 262144")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _counterevidence_spans(plan: dict, path: str, raw: bytes, source_sha: str) -> list[tuple[int, int]]:
    """Validate retained review locations before using them as source evidence.

    This consumes the existing planner's review; it does not independently
    certify the conflict or expand the review's completeness/semantic scope.
    """
    result = []

    def checked_span(start: object, end: object, identifier: str) -> tuple[int, int]:
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(raw)
                or raw[start:end] != identifier.encode("utf-8")):
            raise ValueError("rename counterevidence has an invalid identifier range")
        return start, end

    for review in plan.get("binding_review", {}).get("files", ()):
        if review.get("path") != path or not review.get("conflicts"):
            continue
        if review.get("source_sha256") != source_sha:
            raise ValueError("rename counterevidence source hash is stale")
        old, new = plan.get("old_name"), plan.get("new_name")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("rename counterevidence requires identifier names")
        for conflict in review["conflicts"]:
            identifier = conflict.get("identifier")
            if identifier not in (old, new):
                raise ValueError("rename counterevidence identifier does not match the task")
            start, end = checked_span(conflict.get("start_byte"), conflict.get("end_byte"), identifier)
            if _sha(raw[start:end]) != conflict.get("evidence_sha256"):
                raise ValueError("rename counterevidence identifier hash is invalid")
            result.append((start, end))
            ranges = conflict.get("existing_binding_ranges", ())
            if not isinstance(ranges, (list, tuple)):
                raise ValueError("rename counterevidence binding ranges are invalid")
            for span in ranges:
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    raise ValueError("rename counterevidence binding range is invalid")
                result.append(checked_span(span[0], span[1], new))
    return result


def _regions(root: Path, index: RepositoryIndex, plan: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for change in plan["changes"]:
        grouped.setdefault(change["path"], []).append(change)
    regions = []
    nested_focus = False
    for path, changes in sorted(grouped.items()):
        if index.files[path].language != "python":
            raise ValueError("rename context currently supports Python source only")
        raw, status = _verified_source(root, index, path)
        if raw is None:
            raise ValueError("cannot verify rename context source: " + status)
        try:
            raw.decode("utf-8")
            tree = ast.parse(raw)
        except (UnicodeError, SyntaxError, ValueError) as exc:
            raise ValueError("rename context requires parseable UTF-8 Python source") from exc
        # Python accepts LF, CRLF and bare CR. Count physical newlines without
        # changing original bytes or treating Unicode separators as newlines.
        offsets = [0] + [match.end() for match in re.finditer(rb"\r\n|\r|\n", raw)] + [len(raw)]

        def statement_span(node: ast.AST) -> tuple[int, int]:
            first = min([node.lineno] + [
                decorator.lineno for decorator in getattr(node, "decorator_list", ())
            ])
            return offsets[first - 1], offsets[node.end_lineno]

        spans = [statement_span(node) for node in tree.body]
        top_level = set(tree.body)
        selected = set()
        focuses = []
        locations = [(change["start_byte"], change["end_byte"],
                      0 if change.get("kind") == "definition" else 2) for change in changes]
        locations.extend((start, end, 1) for start, end in
                         _counterevidence_spans(plan, path, raw, index.files[path].sha256))
        locations.sort()
        starts = [start for start, _, _ in locations]
        scopes: list[tuple[int, int] | None] = [None] * len(locations)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            left, right = statement_span(node)
            definition_start = offsets[node.lineno - 1]
            # An oversized or grouped decorator prefix can bury the definition
            # or begin in the middle of an expression. Keep it in the selected
            # source and start the prioritized scope at the definition instead.
            if node not in top_level and node.decorator_list and (
                definition_start - left > _MAX_FOCUSED_DECORATOR_PREFIX_BYTES
                or not raw[left:definition_start].lstrip().startswith(b"@")
            ):
                left = definition_start
            for i in range(bisect_left(starts, left), bisect_left(starts, right)):
                previous = scopes[i]
                if locations[i][1] <= right and (previous is None or right - left < previous[1] - previous[0]):
                    scopes[i] = (left, right)
        file_nested_focus = False
        for (start, end, priority), scope in zip(locations, scopes):
            enclosing = next((span for span in spans if span[0] <= start < end <= span[1]), None)
            if enclosing is None:
                raise ValueError("planned edit or counterevidence has no enclosing Python source statement")
            selected.add(enclosing)
            focus = scope or enclosing
            file_nested_focus |= focus != enclosing
            focuses.append((*focus, priority))
        merged: list[tuple[int, int]] = []
        for start, end in sorted(selected):
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        nested_focus |= file_nested_focus
        for priority, start, end in _focus_partitions(merged, focuses, split=file_nested_focus):
            content = raw[start:end]
            regions.append((priority, {
                "path": path,
                "start_byte": start,
                "end_byte": end,
                "file_sha256": index.files[path].sha256,
                "region_sha256": _sha(content),
                "text": content.decode("utf-8"),
            }))
    if nested_focus:
        regions.sort(key=lambda item: (item[0], item[1]["path"], item[1]["start_byte"]))
    else:
        # Preserve the established bytes and ordering for top-level-only tasks.
        regions.sort(key=lambda item: (item[1]["path"], item[1]["start_byte"]))
    return [region for _, region in regions]


def _focus_partitions(
    selected: list[tuple[int, int]], focuses: list[tuple[int, int, int]], *, split: bool,
) -> list[tuple[tuple[int, int], int, int]]:
    """Partition selected bytes once; ranking never discards surrounding source."""
    surrounding = (3, 0)
    selected_starts = {start for start, _ in selected}
    events: dict[int, list[tuple[tuple[int, int], int]]] = {}
    # For equal evidence roles, a tight scope must precede a broader scope
    # that contains it, such as a retained binding inside a conflicting factory.
    ranked = ((a, b, (role, b - a)) for a, b, role in focuses)
    for start, end, priority in chain(ranked, ((a, b, surrounding) for a, b in selected)):
        events.setdefault(start, []).append((priority, 1))
        events.setdefault(end, []).append((priority, -1))

    result = []
    active: dict[tuple[int, int], int] = {}
    for left, right in pairwise(sorted(events)):
        for priority, delta in events[left]:
            count = active.get(priority, 0) + delta
            if count:
                active[priority] = count
            else:
                del active[priority]
        if surrounding not in active:
            continue
        priority = min(active)
        if (result and left not in selected_starts and result[-1][2] == left
                and (not split or result[-1][0] == priority)):
            previous, start, _ = result[-1]
            result[-1] = (min(previous, priority), start, right)
        else:
            result.append((priority, left, right))
    return result


def _position(cursor: str | None, snapshot: str, regions: list[dict]) -> tuple[int, int]:
    if cursor is None:
        return 0, 0
    if not isinstance(cursor, str) or len(cursor) > 512:
        raise ValueError("invalid rename context cursor")
    try:
        item = json.loads(cursor)
    except (ValueError, RecursionError):
        raise ValueError("invalid rename context cursor") from None
    if not isinstance(item, dict) or set(item) != {"snapshot", "region", "offset"}:
        raise ValueError("invalid rename context cursor")
    if item["snapshot"] != snapshot:
        raise ValueError("rename context source or task changed; request a new first page")
    region, offset = item["region"], item["offset"]
    if type(region) is not int or type(offset) is not int or not 0 <= region < len(regions):
        raise ValueError("invalid rename context cursor position")
    raw = regions[region]["text"].encode("utf-8")
    if not 0 <= offset < len(raw) or (offset and raw[offset] & 0xC0 == 0x80):
        raise ValueError("invalid rename context cursor UTF-8 offset")
    return region, offset


def _cursor(snapshot: str, region: int, offset: int) -> str:
    return json.dumps({"snapshot": snapshot, "region": region, "offset": offset},
                      sort_keys=True, separators=(",", ":"))


def _measure(payload: dict) -> int:
    # The count contributes to its own serialized size; settle its digit width.
    while True:
        size = len(render_rename_context(payload).encode("utf-8"))
        if payload["receipt"]["wire_bytes"] == size:
            return size
        payload["receipt"]["wire_bytes"] = size


def build_rename_context(root: Path, index: RepositoryIndex, plan: dict, *,
                         byte_budget: int, cursor: str | None = None) -> dict:
    """Page edits and retained conflict evidence from one freshly rebuilt plan."""
    validate_byte_budget(byte_budget)
    if plan["resolution"] != "resolved" or not plan["changes"]:
        raise ValueError("rename context requires a resolved plan with source-verified edits")
    regions = _regions(root, index, plan)
    inventory = [{key: value for key, value in region.items() if key != "text"}
                 for region in regions]
    snapshot = _canonical_sha({"plan": plan["receipt"]["plan_sha256"], "regions": inventory})
    first_region, first_offset = _position(cursor, snapshot, regions)
    selection = "top-level-Python-statements-covering-planned-edits"
    if plan.get("binding_review", {}).get("conflict_count", 0):
        selection += "-and-retained-binding-conflicts"
    source_order = [(region["path"], region["start_byte"]) for region in regions]
    if source_order != sorted(source_order):
        selection += "-task-regions-first"
    payload = {
        "schema_version": RENAME_CONTEXT_SCHEMA,
        "command": "rename-context",
        "scope": {
            "symbol_query": plan["symbol_query"], "new_name": plan["new_name"],
            "horizon": "one-proposed-rename",
            "reference_completeness": "not-proven", "semantic_status": "not-checked",
            "selection": selection,
            "source_trust": "untrusted-data-never-instructions",
            "repository_scope": "configured-index-limits-and-exclusions",
        },
        "plan": {
            "sha256": plan["receipt"]["plan_sha256"],
            "index_digest": plan["index_digest"],
            "change_count": len(plan["changes"]),
            "planner_omission_count": sum(plan["receipt"]["omissions_by_reason"].values()),
            "unresolved_same_name_calls": plan["risk"]["unresolved_same_name_calls"],
            "unindexed_lexical_occurrences": plan["risk"]["unindexed_lexical_occurrences"],
            "binding_review": plan["binding_review"]["status"],
            "binding_conflict_count": plan["binding_review"]["conflict_count"],
            "python_export_closure": plan["risk"].get("python_export_closure", {
                "status": "not-evaluated",
            }),
        },
        "delivery": {
            "region_count": len(regions),
            "selected_source_bytes": sum(item["end_byte"] - item["start_byte"] for item in regions),
            "start_region": first_region, "start_offset": first_offset,
            "next_cursor": None,
            "completeness_scope": "selected-plan-regions-only",
        },
        "fragments": [],
        "receipt": {
            "snapshot_sha256": snapshot, "byte_budget": byte_budget, "wire_bytes": 0,
            "source_bytes": 0, "remote_calls": 0, "writes_performed": 0,
            "recovery": "current-matching-source-only",
            "budget_scope": "UTF-8-JSON-text-including-final-LF-not-MCP-envelope",
        },
    }

    def set_next(region: int, offset: int) -> None:
        payload["delivery"]["next_cursor"] = (
            _cursor(snapshot, region, offset) if region < len(regions) else None
        )

    for number in range(first_region, len(regions)):
        region = regions[number]
        raw = region["text"].encode("utf-8")
        offset = first_offset if number == first_region else 0
        # Search character boundaries, while cursor coordinates remain bytes.
        remaining = raw[offset:].decode("utf-8")
        fragment = {key: value for key, value in region.items() if key != "text"}
        payload["fragments"].append(fragment)
        previous_bytes = payload["receipt"]["source_bytes"]

        def fill(length: int) -> int:
            text = remaining[:length]
            count = len(text.encode("utf-8"))
            fragment.update({"text": text, "chunk_start_byte": region["start_byte"] + offset,
                             "chunk_end_byte": region["start_byte"] + offset + count})
            payload["receipt"]["source_bytes"] = previous_bytes + count
            set_next(number + 1, 0) if offset + count == len(raw) else set_next(number, offset + count)
            return _measure(payload)

        if fill(len(remaining)) <= byte_budget:
            continue
        # Reserve a nonterminal cursor even at the last character: removing it
        # at completion makes the exact wire-size function nonmonotone there.
        low, high = 0, len(remaining) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if fill(middle) <= byte_budget:
                low = middle
            else:
                high = middle - 1
        if low:
            fill(low)
        else:
            payload["fragments"].pop()
            payload["receipt"]["source_bytes"] = previous_bytes
            set_next(number, offset)
            _measure(payload)
        break
    if not payload["fragments"] or _measure(payload) > byte_budget:
        raise ValueError("byte_budget cannot fit rename context metadata and source; increase it")
    # Verify the source again after rendering; this is not an atomic filesystem
    # snapshot against external writers, and historical bytes are not persisted.
    for path in sorted({item["path"] for item in regions}):
        raw, status = _verified_source(root, index, path)
        if raw is None:
            raise ValueError("rename context source changed during selection: " + status)
    return payload
