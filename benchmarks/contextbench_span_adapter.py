"""Map an Entroly selection to exact source line intervals in a ContextBench checkout.

Entroly stores no per-fragment line offsets, and the qccr-compressed output is
NOT a contiguous slice of the source (measured: ~60% of its lines are duplicated
in-file, ~20% absent) — so line-attributing the compressed text with str.find is
unsound. Instead we attribute through Entroly's real linkage metadata,
`source_fragment_ids`, to the ORIGIN fragment, whose content IS a contiguous
block and can be located by a UNIQUE match. Anything not uniquely locatable fails
closed (contributes no attributed lines), never a guess.

Invariants:
  * stable path normalization (strip `file:`, `\\`->`/`, leading `./` and `/`)
  * deterministic ordering (input selection order = rank)
  * no fuzzy line attribution (unique contiguous match or fail closed)
  * explicit whole-file handling (block == file -> all lines)
  * exact mapping into the checkout (reads the pinned file bytes)
  * fail closed on missing/renamed/ambiguous/absent
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import PurePosixPath


def canonical_path(source: str) -> str:
    """Normalize an Entroly source id to a repo-relative POSIX path."""
    s = str(source or "")
    if s.startswith("file:"):
        s = s[5:]
    s = s.replace("\\", "/")
    if s.startswith("/"):
        raise ValueError(f"unsafe source path: {source!r}")
    while s.startswith("./"):
        s = s[2:]
    path = PurePosixPath(s)
    if (
        not s
        or ":" in path.parts[0]
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe source path: {source!r}")
    return path.as_posix()


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _count_lines(text: str) -> int:
    """Number of lines; a trailing newline terminates the last line, not a new one."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def locate_block(block: str, source_text: str) -> tuple[int, int] | str:
    """Return the (start_line, end_line) of a UNIQUE contiguous match, else a reason.

    Both inputs are newline-normalized. A block equal to the whole file (modulo
    trailing newlines) maps to the whole file. A block occurring zero or more than
    one time fails closed (returns a reason string), never a guess.
    """
    block_n = _normalize_newlines(block)
    source_n = _normalize_newlines(source_text)
    if not block_n:
        return "empty_block"
    total_lines = _count_lines(source_n)
    if block_n == source_n:
        return (1, total_lines)
    occurrences = source_n.count(block_n)
    if occurrences == 0:
        return "not_found"
    if occurrences > 1:
        return "ambiguous_duplicate"
    pos = source_n.index(block_n)
    start_line = source_n.count("\n", 0, pos) + 1
    end_line = start_line + block_n.count("\n")
    return (start_line, end_line)


@dataclass
class SelectedSpan:
    """One selected fragment mapped (or fail-closed) to exact source lines."""

    path: str
    score: float
    rank: int
    token_cost: int
    lines: set[int] = field(default_factory=set)
    mapped: bool = False
    reason: str = ""
    mapped_blocks: int = 0
    unmapped_blocks: int = 0
    unmapped_lines: int = 0

    def intervals(self) -> list[tuple[int, int]]:
        """Contiguous [start, end] runs of the covered lines (merged, sorted)."""
        if not self.lines:
            return []
        ordered = sorted(self.lines)
        runs: list[tuple[int, int]] = []
        start = prev = ordered[0]
        for n in ordered[1:]:
            if n == prev + 1:
                prev = n
                continue
            runs.append((start, prev))
            start = prev = n
        runs.append((start, prev))
        return runs


def _default_read(repo_dir: str, path: str) -> str | None:
    root = os.path.realpath(repo_dir)
    full = os.path.realpath(os.path.join(root, *path.split("/")))
    try:
        if os.path.commonpath([root, full]) != root:
            return None
    except ValueError:
        return None
    if not os.path.isfile(full):
        return None
    try:
        with open(full, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def map_selection(
    selected: list[dict],
    origin_by_id: dict[str, dict],
    repo_dir: str,
    *,
    read_file=_default_read,
) -> list[SelectedSpan]:
    """Map a qccr selection to exact line spans through origin-fragment linkage.

    `origin_by_id` maps `fragment_id -> {"source", "content"}` from the ingested
    index. Selection order defines rank. Attribution is exact-or-fail-closed.
    """
    spans: list[SelectedSpan] = []
    for rank, frag in enumerate(selected):
        score = float(frag.get("relevance", frag.get("relevance_score", 0.0)) or 0.0)
        if not math.isfinite(score):
            raise ValueError("selected fragment relevance must be finite")
        raw_token_cost = frag.get("token_count", 0) or 0
        if not isinstance(raw_token_cost, int) or isinstance(raw_token_cost, bool):
            raise ValueError("selected fragment token_count must be an integer")
        token_cost = raw_token_cost
        if token_cost < 0:
            raise ValueError("selected fragment token_count must be non-negative")
        raw_origin_ids = frag.get("source_fragment_ids") or []
        if (
            not isinstance(raw_origin_ids, (list, tuple))
            or any(not isinstance(item, str) or not item for item in raw_origin_ids)
            or len(set(raw_origin_ids)) != len(raw_origin_ids)
        ):
            fallback_lines = max(1, _count_lines(str(frag.get("content") or "")))
            spans.append(
                SelectedSpan(
                    path="",
                    score=score,
                    rank=rank,
                    token_cost=token_cost,
                    reason="invalid_origin_metadata",
                    unmapped_blocks=1,
                    unmapped_lines=fallback_lines,
                )
            )
            continue
        origin_ids = list(raw_origin_ids)
        try:
            path = canonical_path(frag.get("source"))
        except ValueError:
            fallback_lines = max(1, _count_lines(str(frag.get("content") or "")))
            spans.append(
                SelectedSpan(
                    path="",
                    score=score,
                    rank=rank,
                    token_cost=token_cost,
                    reason="unsafe_source_path",
                    unmapped_blocks=max(1, len(origin_ids)),
                    unmapped_lines=fallback_lines,
                )
            )
            continue
        span = SelectedSpan(path=path, score=score, rank=rank, token_cost=token_cost)

        blocks: list[str] = []
        reasons: list[str] = []
        for origin_id in origin_ids:
            origin = origin_by_id.get(origin_id)
            if not origin:
                reasons.append("missing_origin_metadata")
                span.unmapped_blocks += 1
                span.unmapped_lines += 1
                continue
            block = str(origin.get("content") or "")
            try:
                origin_path = canonical_path(origin.get("source"))
            except ValueError:
                reasons.append("unsafe_origin_path")
                span.unmapped_blocks += 1
                span.unmapped_lines += max(1, _count_lines(block))
                continue
            if origin_path != path:
                reasons.append("origin_source_mismatch")
                span.unmapped_blocks += 1
                span.unmapped_lines += max(1, _count_lines(block))
                continue
            blocks.append(block)
        if not blocks:
            if not reasons:
                reasons.append("no_origin_metadata")
                span.unmapped_blocks = 1
                span.unmapped_lines = max(1, _count_lines(str(frag.get("content") or "")))
            span.reason = ";".join(sorted(set(reasons)))
            spans.append(span)
            continue

        text = read_file(repo_dir, path)
        if text is None:
            reasons.append("missing_file")
            span.unmapped_blocks += len(blocks)
            span.unmapped_lines += sum(max(1, _count_lines(block)) for block in blocks)
            span.reason = ";".join(sorted(set(reasons)))
            spans.append(span)
            continue

        for block in blocks:
            res = locate_block(str(block or ""), text)
            if isinstance(res, tuple):
                span.lines.update(range(res[0], res[1] + 1))
                span.mapped_blocks += 1
            else:
                reasons.append(res)
                span.unmapped_blocks += 1
                span.unmapped_lines += max(1, _count_lines(block))
        span.mapped = bool(span.lines)
        if reasons:
            span.reason = ";".join(sorted(set(reasons)))
        elif not span.mapped:
            span.reason = "unmapped"
        spans.append(span)
    return spans


def to_spans(records: list[SelectedSpan]) -> dict[str, set[int]]:
    """Collapse mapped records into the metric-core Spans type ({path: {lines}})."""
    out: dict[str, set[int]] = {}
    for r in records:
        if r.mapped and r.lines:
            out.setdefault(r.path, set()).update(r.lines)
    return out


def unmapped_line_count(records: list[SelectedSpan]) -> int:
    """Return selected evidence mass that could not be attributed to source."""
    return sum(record.unmapped_lines for record in records)
