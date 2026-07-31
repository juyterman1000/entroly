"""Evidence-Locked Compression for heavy agent inputs.

ELC is the safer alternative to blind aggressive compression. It compresses
around evidence, never through evidence:

- preserve error/warning/failure anchors and nearby context,
- preserve query-matching lines/records,
- preserve first/last boundaries,
- preserve high-information outliers when the budget allows,
- summarize omitted spans with receipts,
- keep JSON schema plus examples, matching records, and outliers.

The module is dependency-free and deterministic. It is designed to sit in front
of LLM API calls and tool-result proxying without requiring model inference.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

_CHARS_PER_TOKEN = 3.6
_DEFAULT_BUDGET = 1200
_MAX_RECEIPT_SPANS = 64

_ANCHOR_RE = re.compile(
    r"\b(error|failed|failure|fatal|panic|exception|traceback|warning|warn|"
    r"timeout|denied|refused|invalid|unexpected|unresolved|exit\s+code|"
    r"assertion|segfault|no\s+such|not\s+found)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:[\w./-]+\.(?:py|rs|ts|tsx|js|jsx|java|kt|go|c|cpp|h|hpp|json|ya?ml|toml))(?:[:#]\d+)?")
_ID_RE = re.compile(r"\b(?:[A-Fa-f0-9]{7,40}|[A-Z]{2,}-\d+|[\w.-]+@[\w.-]+)\b")
_WORD_RE = re.compile(r"\b[\w.-]{3,}\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_HARD_LOCK_REASONS = {"anchor", "query", "boundary", "path", "id"}
_RECEIPT_REASONS = _HARD_LOCK_REASONS | {"outlier"}
_QUERY_STOPWORDS = {
    "and",
    "are",
    "did",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "its",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(slots=True)
class OmittedSpan:
    start_line: int
    end_line: int
    line_count: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CompressionReceipt:
    original_tokens: int
    compressed_tokens: int
    savings_ratio: float
    compression_level: int
    content_type: str
    anchors_preserved: dict[str, int]
    omitted_spans: list[OmittedSpan] = field(default_factory=list)
    recoverable: bool = True

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["omitted_spans"] = [span.as_dict() for span in self.omitted_spans]
        return data


@dataclass(slots=True)
class CompressionResult:
    compressed: str
    receipt: CompressionReceipt
    changed: bool

    def with_receipt_header(self) -> str:
        pct = int(round(self.receipt.savings_ratio * 100))
        header = (
            f"[entroly-elc: {self.receipt.original_tokens}->{self.receipt.compressed_tokens} "
            f"tokens, {pct}% saved, anchors={self.receipt.anchors_preserved}]"
        )
        return f"{header}\n{self.compressed}" if self.compressed else header


@dataclass(slots=True)
class _LineScore:
    index: int
    text: str
    tokens: int
    score: float
    reasons: set[str]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN) + 1)


def detect_heavy_content_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if stripped[0] in "[{":
        try:
            json.loads(stripped)
            return "json"
        except Exception:
            pass
    head = "\n".join(text.splitlines()[:80])
    if "Traceback" in head or "Exception" in head or re.search(r"^\s+at\s+", head, re.MULTILINE):
        return "stacktrace"
    if _ANCHOR_RE.search(head) or re.search(r"^\d{4}[-/]\d{2}[-/]\d{2}", head, re.MULTILINE):
        return "log"
    if "|" in head and head.count("|") > 8:
        return "table"
    return "text"


def compress_evidence_locked(
    text: str,
    *,
    query: str = "",
    budget_tokens: int = _DEFAULT_BUDGET,
    content_type: str | None = None,
    context_radius: int = 2,
    min_savings: float = 0.08,
) -> CompressionResult:
    """Compress heavy text while preserving evidence anchors and receipts."""
    original_tokens = estimate_tokens(text)
    if not text or original_tokens <= max(128, budget_tokens):
        return _unchanged(text, original_tokens, content_type or "short")

    ctype = content_type or detect_heavy_content_type(text)
    if ctype == "json":
        result = _compress_json(text, query=query, budget_tokens=budget_tokens)
    else:
        result = _compress_lines(
            text,
            query=query,
            budget_tokens=budget_tokens,
            content_type=ctype,
            context_radius=context_radius,
        )

    if result.receipt.savings_ratio < min_savings or not result.compressed.strip():
        return _unchanged(text, original_tokens, ctype)
    return result


def compress_payload_messages(
    messages: list[dict[str, Any]],
    *,
    query: str = "",
    budget_tokens: int = _DEFAULT_BUDGET,
    include_receipt_header: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, object]], int]:
    """Compress heavy tool/function message content in an LLM payload.

    Returns (messages, receipts, approx_tokens_saved). Non-tool/user/assistant
    text is preserved to avoid changing natural conversation semantics.
    """
    new_messages: list[dict[str, Any]] = []
    receipts: list[dict[str, object]] = []
    total_saved = 0

    for msg in messages:
        role = str(msg.get("role", ""))
        content = msg.get("content")
        if role not in {"tool", "function"}:
            new_messages.append(msg)
            continue
        if isinstance(content, str):
            result = compress_evidence_locked(
                content,
                query=query,
                budget_tokens=budget_tokens,
            )
            if result.changed:
                new_msg = dict(msg)
                new_msg["content"] = result.with_receipt_header() if include_receipt_header else result.compressed
                receipts.append(result.receipt.as_dict())
                total_saved += max(0, result.receipt.original_tokens - result.receipt.compressed_tokens)
                new_messages.append(new_msg)
                continue
        new_messages.append(msg)

    return new_messages, receipts, total_saved


def _unchanged(text: str, original_tokens: int, content_type: str) -> CompressionResult:
    receipt = CompressionReceipt(
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        savings_ratio=0.0,
        compression_level=0,
        content_type=content_type,
        anchors_preserved={},
        omitted_spans=[],
        recoverable=True,
    )
    return CompressionResult(text, receipt, changed=False)


def _compress_lines(
    text: str,
    *,
    query: str,
    budget_tokens: int,
    content_type: str,
    context_radius: int,
) -> CompressionResult:
    raw_lines = text.splitlines()
    query_terms = _query_terms(query)
    scored = [_score_line(i, line, query_terms, len(raw_lines)) for i, line in enumerate(raw_lines)]

    keep: set[int] = set()
    anchor_counts: Counter[str] = Counter()
    for line in scored:
        # Hard locks are strict evidence anchors. High-entropy outliers are
        # useful candidates, but not hard locks: timestamped heartbeats and CI
        # progress lines are high entropy too, and hard-locking all of them
        # defeats compression on SRE/build logs.
        if line.reasons.intersection(_HARD_LOCK_REASONS):
            for idx in range(max(0, line.index - context_radius), min(len(scored), line.index + context_radius + 1)):
                keep.add(idx)
            for reason in line.reasons:
                if reason in _RECEIPT_REASONS:
                    anchor_counts[reason] += 1

    used = sum(scored[i].tokens for i in keep)
    remaining = max(0, budget_tokens - used)
    candidates = sorted(
        [line for line in scored if line.index not in keep],
        key=lambda item: (item.score / max(item.tokens, 1), item.score),
        reverse=True,
    )
    for line in candidates:
        if line.tokens <= remaining:
            keep.add(line.index)
            remaining -= line.tokens
            for reason in line.reasons:
                if reason == "outlier":
                    anchor_counts[reason] += 1

    if not keep and scored:
        keep.update(range(min(3, len(scored))))

    compressed, omitted = _render_kept_lines(raw_lines, keep)
    original_tokens = estimate_tokens(text)
    compressed_tokens = estimate_tokens(compressed)
    receipt = CompressionReceipt(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        savings_ratio=1.0 - compressed_tokens / max(original_tokens, 1),
        compression_level=3,
        content_type=content_type,
        anchors_preserved=dict(anchor_counts),
        omitted_spans=omitted[:_MAX_RECEIPT_SPANS],
        recoverable=True,
    )
    return CompressionResult(compressed, receipt, changed=True)


def _score_line(index: int, line: str, query_terms: set[str], total_lines: int) -> _LineScore:
    stripped = line.strip()
    tokens = estimate_tokens(stripped)
    reasons: set[str] = set()
    score = 0.0

    if index < 3 or index >= max(0, total_lines - 3):
        reasons.add("boundary")
        score += 2.0
    if _ANCHOR_RE.search(stripped):
        reasons.add("anchor")
        score += 12.0
    if _PATH_RE.search(stripped):
        reasons.add("path")
        score += 3.0
    if _ID_RE.search(stripped):
        reasons.add("id")
        score += 1.5
    matched = query_terms.intersection({w.lower() for w in _WORD_RE.findall(stripped)})
    if matched:
        reasons.add("query")
        score += 4.0 + len(matched)
    ent = _entropy(stripped)
    if ent >= 4.2 and len(stripped) > 40:
        reasons.add("outlier")
        score += 2.0
    score += ent / max(math.log2(max(tokens, 2)), 1.0)
    return _LineScore(index, stripped, tokens, score, reasons)


def _render_kept_lines(lines: list[str], keep: set[int]) -> tuple[str, list[OmittedSpan]]:
    output: list[str] = []
    omitted: list[OmittedSpan] = []
    sorted_keep = sorted(keep)
    prev = -1
    for idx in sorted_keep:
        if idx > prev + 1:
            start = prev + 1
            end = idx - 1
            count = end - start + 1
            output.append(f"  ... ({count} lines omitted; recoverable span {start + 1}-{end + 1})")
            omitted.append(OmittedSpan(start + 1, end + 1, count, "budget"))
        output.append(lines[idx].strip())
        prev = idx
    if lines and prev < len(lines) - 1:
        start = prev + 1
        end = len(lines) - 1
        count = end - start + 1
        output.append(f"  ... ({count} lines omitted; recoverable span {start + 1}-{end + 1})")
        omitted.append(OmittedSpan(start + 1, end + 1, count, "budget"))
    return "\n".join(output), omitted


def _compress_json(text: str, *, query: str, budget_tokens: int) -> CompressionResult:
    original_tokens = estimate_tokens(text)
    try:
        data = json.loads(text)
    except Exception:
        return _compress_lines(
            text,
            query=query,
            budget_tokens=budget_tokens,
            content_type="json_text",
            context_radius=1,
        )

    query_terms = _query_terms(query)
    schema = _json_schema(data, depth=0, max_depth=4)
    excerpt_chars = max(180, min(900, int(budget_tokens * 1.8)))
    matches = _json_query_matches(
        data,
        query_terms,
        limit=8,
        excerpt_chars=excerpt_chars,
    )
    outliers = _json_outliers(data, limit=5, excerpt_chars=excerpt_chars)
    rich_base = {
        "elc": "json evidence-locked compression",
        "schema": schema,
        "counts": _json_counts(data),
    }
    compact_base = {
        "elc": "json evidence-locked compression",
        "schema": _json_schema(data, depth=0, max_depth=2),
        "counts": _json_counts(data),
    }
    minimal_base = {
        "elc": "json evidence-locked compression",
        "schema": {"root": type(data).__name__},
        "counts": _json_counts(data),
    }
    required_matches = matches[:1]
    base = minimal_base
    for candidate_base in (rich_base, compact_base, minimal_base):
        candidate = _render_json_summary(candidate_base, required_matches, [])
        if estimate_tokens(candidate) <= budget_tokens:
            base = candidate_base
            break

    kept_matches: list[Any] = []
    for match in matches:
        candidate = _render_json_summary(base, [*kept_matches, match], [])
        if estimate_tokens(candidate) <= budget_tokens:
            kept_matches.append(match)

    kept_outliers: list[Any] = []
    match_hashes = {_stable_json(value) for value in kept_matches}
    for outlier in outliers:
        if _stable_json(outlier) in match_hashes:
            continue
        candidate = _render_json_summary(
            base,
            kept_matches,
            [*kept_outliers, outlier],
        )
        if estimate_tokens(candidate) <= budget_tokens:
            kept_outliers.append(outlier)

    compressed = _render_json_summary(base, kept_matches, kept_outliers)
    if estimate_tokens(compressed) > budget_tokens:
        # Tiny budgets may not fit schema plus one evidence record. Preserve a
        # valid, explicit receipt rather than silently returning an oversized
        # payload or malformed JSON.
        compressed = _render_json_summary(
            {"elc": "json evidence-locked compression"},
            [],
            [],
        )

    compressed_tokens = estimate_tokens(compressed)
    receipt = CompressionReceipt(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        savings_ratio=1.0 - compressed_tokens / max(original_tokens, 1),
        compression_level=3,
        content_type="json",
        anchors_preserved={
            "schema": int("schema" in base),
            "query": len(kept_matches),
            "outlier": len(kept_outliers),
        },
        omitted_spans=[OmittedSpan(1, max(1, text.count("\n") + 1), text.count("\n") + 1, "json_values")],
        recoverable=True,
    )
    return CompressionResult(compressed, receipt, changed=True)


def _json_schema(obj: Any, *, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        return "..."
    if isinstance(obj, dict):
        items = list(obj.items())[:30]
        return {str(k): _json_schema(v, depth=depth + 1, max_depth=max_depth) for k, v in items}
    if isinstance(obj, list):
        if not obj:
            return {"type": "array", "items": 0}
        return {
            "type": "array",
            "items": len(obj),
            "first": _json_schema(obj[0], depth=depth + 1, max_depth=max_depth),
            "last": _json_schema(obj[-1], depth=depth + 1, max_depth=max_depth),
        }
    if isinstance(obj, str):
        return obj if len(obj) <= 40 else f"<str:{len(obj)}>"
    if isinstance(obj, bool):
        return "<bool>"
    if isinstance(obj, int):
        return "<int>"
    if isinstance(obj, float):
        return "<float>"
    if obj is None:
        return "<null>"
    return f"<{type(obj).__name__}>"


def _render_json_summary(
    base: dict[str, Any], matches: Sequence[Any], outliers: Sequence[Any]
) -> str:
    summary = {
        **base,
        "query_matches": list(matches),
        "outliers": list(outliers),
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_query_matches(
    obj: Any,
    query_terms: set[str],
    *,
    limit: int,
    excerpt_chars: int = 900,
) -> list[Any]:
    if not query_terms:
        return []
    ranked: list[tuple[int, int, Any]] = []
    for order, item in enumerate(_walk_json_records(obj)):
        rendered = json.dumps(item, ensure_ascii=False, sort_keys=True).lower()
        words = {word.lower() for word in _WORD_RE.findall(rendered)}
        coverage = len(query_terms.intersection(words))
        if coverage:
            ranked.append((coverage, order, item))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [
        _compact_json_value(
            item,
            query_terms=query_terms,
            excerpt_chars=excerpt_chars,
        )
        for _, _, item in ranked[:limit]
    ]


def _json_outliers(
    obj: Any, *, limit: int, excerpt_chars: int = 900
) -> list[Any]:
    rows = list(_walk_json_records(obj))
    if not rows:
        return []
    scored = sorted(
        rows,
        key=lambda value: len(json.dumps(value, ensure_ascii=False, sort_keys=True)),
        reverse=True,
    )
    return [
        _compact_json_value(value, excerpt_chars=excerpt_chars)
        for value in scored[:limit]
    ]


def _json_counts(obj: Any) -> dict[str, int]:
    counts = Counter(type(value).__name__ for value in _walk_json(obj))
    return dict(sorted(counts.items()))


def _walk_json(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_json(value)


def _walk_json_records(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield record-like objects without treating container roots as evidence."""
    if isinstance(obj, dict):
        if not any(isinstance(value, (dict, list)) for value in obj.values()):
            yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _walk_json_records(value)
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                yield from _walk_json_records(value)


def _compact_json_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 3,
    query_terms: set[str] | None = None,
    excerpt_chars: int = 900,
) -> Any:
    """Bound evidence size while preserving answer-critical scalar values."""
    if depth > max_depth:
        return "<max-depth>"
    if isinstance(value, dict):
        items = list(value.items())
        compact = {
            str(key): _compact_json_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                query_terms=query_terms,
                excerpt_chars=excerpt_chars,
            )
            for key, item in items[:30]
        }
        if len(items) > 30:
            compact["<omitted_keys>"] = len(items) - 30
        return compact
    if isinstance(value, list):
        if len(value) <= 8:
            return [
                _compact_json_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    query_terms=query_terms,
                    excerpt_chars=excerpt_chars,
                )
                for item in value
            ]
        return {
            "items": len(value),
            "first": [
                _compact_json_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    query_terms=query_terms,
                    excerpt_chars=excerpt_chars,
                )
                for item in value[:3]
            ],
            "last": [
                _compact_json_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    query_terms=query_terms,
                    excerpt_chars=excerpt_chars,
                )
                for item in value[-3:]
            ],
        }
    if isinstance(value, str) and len(value) > excerpt_chars:
        return _relevant_text_excerpt(value, query_terms or set(), excerpt_chars)
    return value


def _relevant_text_excerpt(text: str, query_terms: set[str], max_chars: int) -> str:
    """Keep query-centered sentence windows from a long JSON scalar."""
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return text[:max_chars]
    scored: list[tuple[int, float, int]] = []
    for index, sentence in enumerate(sentences):
        words = {word.lower() for word in _WORD_RE.findall(sentence)}
        overlap = len(query_terms.intersection(words))
        density = overlap / max(1, len(words))
        scored.append((overlap, density, index))
    best_overlap, _, best_index = max(scored, key=lambda row: (row[0], row[1], -row[2]))
    if best_overlap == 0:
        return f"{text[:max_chars]}...<truncated:{max(0, len(text) - max_chars)}>"

    selected: list[int] = []
    for radius in (0, 1, 2):
        for index in range(max(0, best_index - radius), min(len(sentences), best_index + radius + 1)):
            if index not in selected:
                candidate = " ".join(sentences[value] for value in sorted([*selected, index]))
                if len(candidate) <= max_chars:
                    selected.append(index)
    excerpt = " ".join(sentences[index] for index in sorted(selected))
    omitted = max(0, len(text) - len(excerpt))
    return f"{excerpt} ...<query-centered; omitted:{omitted}>"


def _query_terms(query: str) -> set[str]:
    return {
        word.lower()
        for word in _WORD_RE.findall(query or "")
        if len(word) >= 3 and word.lower() not in _QUERY_STOPWORDS
    }


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


__all__ = [
    "CompressionReceipt",
    "CompressionResult",
    "OmittedSpan",
    "compress_evidence_locked",
    "compress_payload_messages",
    "detect_heavy_content_type",
    "estimate_tokens",
]
