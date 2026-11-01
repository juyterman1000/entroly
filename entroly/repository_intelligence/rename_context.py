"""Exact, task-bound source pages for reviewing a proposed Python rename.

This is delivery of selected evidence, never a semantic safety certificate.
The rename planner remains the authority for edits and omissions.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .models import RepositoryIndex
from .verified_refactor_impl import _verified_source

RENAME_CONTEXT_SCHEMA = "entroly.rename-context.v1"


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


def _regions(root: Path, index: RepositoryIndex, plan: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for change in plan["changes"]:
        grouped.setdefault(change["path"], []).append(change)
    regions = []
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
        offsets = [0]
        # Only LF ends a Python physical line; bytes.splitlines also splits CR.
        for line in raw.split(b"\n")[:-1]:
            offsets.append(offsets[-1] + len(line) + 1)
        offsets.append(len(raw))
        spans = []
        for node in tree.body:
            first = min([node.lineno] + [
                decorator.lineno for decorator in getattr(node, "decorator_list", ())
            ])
            spans.append((offsets[first - 1], offsets[node.end_lineno]))
        selected = set()
        for change in changes:
            start, end = change["start_byte"], change["end_byte"]
            enclosing = next((span for span in spans if span[0] <= start < end <= span[1]), None)
            if enclosing is None:
                raise ValueError("planned edit has no enclosing Python source statement")
            selected.add(enclosing)
        merged: list[tuple[int, int]] = []
        for start, end in sorted(selected):
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            content = raw[start:end]
            regions.append({
                "path": path,
                "start_byte": start,
                "end_byte": end,
                "file_sha256": index.files[path].sha256,
                "region_sha256": _sha(content),
                "text": content.decode("utf-8"),
            })
    return regions


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
    """Page source regions covering the edits of one freshly rebuilt plan."""
    validate_byte_budget(byte_budget)
    if plan["resolution"] != "resolved" or not plan["changes"]:
        raise ValueError("rename context requires a resolved plan with source-verified edits")
    regions = _regions(root, index, plan)
    inventory = [{key: value for key, value in region.items() if key != "text"}
                 for region in regions]
    snapshot = _canonical_sha({"plan": plan["receipt"]["plan_sha256"], "regions": inventory})
    first_region, first_offset = _position(cursor, snapshot, regions)
    payload = {
        "schema_version": RENAME_CONTEXT_SCHEMA,
        "command": "rename-context",
        "scope": {
            "symbol_query": plan["symbol_query"], "new_name": plan["new_name"],
            "horizon": "one-proposed-rename",
            "reference_completeness": "not-proven", "semantic_status": "not-checked",
            "selection": "top-level-Python-statements-covering-planned-edits",
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
