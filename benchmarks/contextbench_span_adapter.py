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
from dataclasses import dataclass, field


def canonical_path(source: str) -> str:
    """Normalize an Entroly source id to a repo-relative POSIX path."""
    s = str(source or "")
    if s.startswith("file:"):
        s = s[5:]
    s = s.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


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
    block_n = _normalize_newlines(block).strip("\n")
    source_n = _normalize_newlines(source_text)
    if not block_n.strip():
        return "empty_block"
    total_lines = _count_lines(source_n)
    if block_n == source_n.strip("\n"):
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
    full = os.path.join(repo_dir, *path.split("/"))
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
        path = canonical_path(frag.get("source"))
        score = float(frag.get("relevance", frag.get("relevance_score", 0.0)) or 0.0)
        token_cost = int(frag.get("token_count", 0) or 0)
        origin_ids = list(frag.get("source_fragment_ids") or [])
        span = SelectedSpan(path=path, score=score, rank=rank, token_cost=token_cost)

        text = read_file(repo_dir, path)
        if text is None:
            span.reason = "missing_file"
            spans.append(span)
            continue

        blocks = [origin_by_id[i]["content"] for i in origin_ids if i in origin_by_id]
        if not blocks:
            span.reason = "no_origin_metadata"
            spans.append(span)
            continue

        reasons: list[str] = []
        for block in blocks:
            res = locate_block(str(block or ""), text)
            if isinstance(res, tuple):
                span.lines.update(range(res[0], res[1] + 1))
            else:
                reasons.append(res)
        span.mapped = bool(span.lines)
        if not span.mapped:
            span.reason = ";".join(sorted(set(reasons))) or "unmapped"
        spans.append(span)
    return spans


def to_spans(records: list[SelectedSpan]) -> dict[str, set[int]]:
    """Collapse mapped records into the metric-core Spans type ({path: {lines}})."""
    out: dict[str, set[int]] = {}
    for r in records:
        if r.mapped and r.lines:
            out.setdefault(r.path, set()).update(r.lines)
    return out
