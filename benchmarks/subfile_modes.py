"""Three selection granularities producing verifiable SourceSpans, for the
minimum sub-file provenance experiment (prereg §"Minimum experiment").

Same BM25 ranker over the same candidate files at three granularities, so the
ONLY variable is selection granularity:
  * file-level      — whole files (current Entroly behaviour)
  * line-window     — fixed N-line windows (naive sub-file)
  * syntax-block    — AST function/method blocks (structure-aware sub-file)

Every produced span is byte-exact and independently verifiable; nothing here uses
fuzzy text matching.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter

from entroly.source_span import Representation, SourceSpan, derive_lines, digest

_WORD = re.compile(r"[A-Za-z0-9_]+")
Unit = tuple[str, int, int, str]  # path, byte_start, byte_end, representation


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _line_starts(src: bytes) -> list[int]:
    """Byte offset of the start of each 1-indexed line (offs[k] = start of line k+1)."""
    offs = [0]
    for i, byte in enumerate(src):
        if byte == 0x0A:
            offs.append(i + 1)
    return offs


def est_tokens(nbytes: int) -> int:
    return max(1, nbytes // 4)


def bm25_scores(docs: list[str], query: str, *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [_tok(d) for d in docs]
    n = len(toks) or 1
    df: Counter = Counter()
    for t in toks:
        df.update(set(t))
    avgdl = (sum(len(t) for t in toks) / n) or 1.0
    q = set(_tok(query))
    scores = []
    for t in toks:
        tf = Counter(t)
        dl = len(t)
        s = 0.0
        for term in q:
            if term in tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def file_units(files: list[tuple[str, bytes]]) -> list[Unit]:
    return [(p, 0, len(src), Representation.WHOLE_FILE.value) for p, src in files]


def window_units(files: list[tuple[str, bytes]], window: int = 40) -> list[Unit]:
    units: list[Unit] = []
    for p, src in files:
        starts = _line_starts(src)
        nlines = len(starts)
        for w in range(0, nlines, window):
            bs = starts[w]
            end_line = min(w + window, nlines)
            be = starts[end_line] if end_line < len(starts) else len(src)
            if be > bs:
                units.append((p, bs, be, Representation.LINE_WINDOW.value))
    return units


def block_units(files: list[tuple[str, bytes]]) -> list[Unit]:
    """AST function/method blocks for .py; whole-file fallback otherwise or on syntax error."""
    units: list[Unit] = []
    for p, src in files:
        if not p.endswith(".py"):
            units.append((p, 0, len(src), Representation.WHOLE_FILE.value))
            continue
        try:
            tree = ast.parse(src.decode("utf-8", "replace"))
        except SyntaxError:
            units.append((p, 0, len(src), Representation.REFERENCE_ONLY.value))
            continue
        starts = _line_starts(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                l0 = node.lineno
                l1 = getattr(node, "end_lineno", l0) or l0
                bs = starts[l0 - 1]
                be = starts[l1] if l1 < len(starts) else len(src)
                if be > bs:
                    units.append((p, bs, be, Representation.SYNTAX_BLOCK.value))
    return units


def _make_span(path: str, src: bytes, bs: int, be: int, rep: str, commit: str) -> SourceSpan:
    ls, le = derive_lines(src, bs, be)
    return SourceSpan(path, digest(src), bs, be, ls, le, digest(src[bs:be]), rep, commit)


def rank_and_select(
    files: list[tuple[str, bytes]], units: list[Unit], query: str, budget: int,
    *, source_commit: str = "",
) -> list[SourceSpan]:
    """Deterministic BM25 selection of units under a token budget → verifiable spans.

    Ordering is total (score DESC, path ASC, byte_start ASC) so the selection is
    reproducible regardless of unit enumeration order.
    """
    by_path = dict(files)
    texts = [by_path[p][bs:be].decode("utf-8", "replace") for (p, bs, be, _rep) in units]
    scores = bm25_scores(texts, query)
    order = sorted(range(len(units)), key=lambda i: (-scores[i], units[i][0], units[i][1]))
    spans: list[SourceSpan] = []
    cum = 0
    for i in order:
        if scores[i] <= 0:
            break
        p, bs, be, rep = units[i]
        tokens = est_tokens(be - bs)
        if cum + tokens > budget and spans:
            break
        cum += tokens
        spans.append(_make_span(p, by_path[p], bs, be, rep, source_commit))
        if cum >= budget:
            break
    return spans


def spans_to_lines(spans: list[SourceSpan]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for s in spans:
        out.setdefault(s.source_path, set()).update(range(s.line_start, s.line_end + 1))
    return out


def verify_rate(spans: list[SourceSpan], files: list[tuple[str, bytes]]) -> float:
    by_path = dict(files)
    if not spans:
        return 1.0
    ok = sum(1 for s in spans if s.source_path in by_path and s.verify(by_path[s.source_path]))
    return ok / len(spans)


MODES = {"file": file_units, "line_window": window_units, "syntax_block": block_units}
