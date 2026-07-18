"""
Entroly SDK — 3-Line Integration
==================================

Drop-in context optimization for ANY AI application.
Works with LangChain, CrewAI, Agno, or raw API calls.

Usage::

    from entroly import compress
    compressed = compress(text, budget=2000)

    # Or with content type hint:
    compressed = compress(json_blob, budget=500, content_type="json")

    # Or compress a full message list (LLM conversation):
    from entroly import compress_messages
    messages = compress_messages(messages, budget=50000)

    # LangChain integration:
    from entroly.integrations.langchain import EntrolyCompressor
    chain = EntrolyCompressor(budget=30000) | llm
"""

from __future__ import annotations

import re
from typing import Any
from .profiles import get_profile

from .universal_compress import (
    detect_content_type,
    tfidf_extractive_summarize,
    universal_compress,
)


def _track_savings(before_tokens: int, after_tokens: int, label: str) -> None:
    """Record SDK compression value to the shared telemetry sink so the
    dashboard reflects SDK-only users (previously a hard blank). Strictly
    fail-open: telemetry must NEVER affect compress() output or raise.

    SDK reductions are classified as local-only: Entroly cannot prove the
    compressed result was sent to a paid provider, so no dollar savings are
    claimed for this path."""
    try:
        saved = int(before_tokens) - int(after_tokens)
        if saved <= 0:
            return
        from .value_tracker import get_tracker
        tracker = get_tracker()
        tracker.record(
            tokens_saved=saved,
            model="",
            optimized=True,
            source="sdk",
        )
        tracker.record_event(
            "compress", f"SDK {label}: saved {saved:,} tokens",
            source="sdk", tokens_saved=saved,
        )
    except Exception:  # noqa: BLE001 — telemetry is best-effort only
        pass


def _ensure_non_empty(
    out: str,
    original: str,
    budget: int | None,
    target_ratio: float,
) -> str:
    """Public contract: compress() never annihilates non-empty input.

    A compressor is a map C: (text, budget) -> text. For non-empty input and
    a positive budget, emitting nothing is strictly dominated by emitting any
    in-budget slice of the original — so an empty/whitespace result is always
    a bug, never a valid compression. This is the final safety net beneath the
    fixes in universal_compress/tfidf: it also covers the native Rust path
    (py_compress_block), which can skeletonize an input down to nothing.

    On violation, fall back to a deterministic head that preserves whole lines
    for readability.
    """
    if out and out.strip():
        return out
    if not original or not original.strip():
        return original

    if budget is not None:
        return _budget_bounded_head(original, budget)

    # Character budget (4 chars ≈ 1 token), with a sane floor so the rescue
    # slice is actually useful rather than a token or two.
    char_budget = max(200, int(len(original) * max(0.05, target_ratio)))
    if len(original) <= char_budget:
        return original

    head = original[:char_budget]
    # Snap to the last line boundary in the back half so we don't cut a line
    # mid-token; only if that still leaves a substantial slice.
    nl = head.rfind("\n")
    if nl > char_budget // 2:
        head = head[:nl]
    return head.rstrip() + "\n…[truncated]"


def _budget_bounded_head(content: str, budget: int) -> str:
    """Return a deterministic non-empty prefix within the SDK token estimate."""
    char_budget = max(1, budget * 4)
    if len(content) <= char_budget:
        return content

    suffix = "\n...[truncated]"
    if char_budget <= len(suffix):
        return content[:char_budget]

    head = content[:char_budget - len(suffix)]
    nl = head.rfind("\n")
    if nl > len(head) // 2:
        head = head[:nl]
    return head.rstrip() + suffix


def _enforce_budget(out: str, budget: int | None) -> str:
    """Apply the SDK's explicit estimated-token ceiling.

    Query-agnostic compression cannot know which omitted fact a future query
    will need. Structural compactors get the first pass; if they stop above an
    explicit budget, finish with deterministic truncation rather than another
    heuristic selector that implies answer preservation.
    """
    if budget is None or len(out) <= budget * 4:
        return out
    return _budget_bounded_head(out, budget)


_CODE_SYMBOL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "await", "function",
    "constructor", "super", "async",
})
_CODE_SYMBOL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("class", r"\b(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ("interface", r"\b(?:export\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ("type", r"\b(?:export\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
    ("function", r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ("function", r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ("function", r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ("const_fn", r"\b(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\("),
    (
        "method",
        r"^\s*(?:(?:public|private|protected|static|override|readonly)\s+)*(?:async\s+)?"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    ),
)


def _code_symbol_summary(content: str, budget: int | None) -> str:
    """Extract a bounded symbol skeleton for code compression.

    Generic code compression is query-agnostic, so it cannot know which body
    line matters. It should still preserve the map of names a later answer can
    search for: classes, interfaces, types, functions, and methods.
    """
    max_chars = 1200
    if budget is not None:
        max_chars = max(160, min(max_chars, budget * 4 // 3))

    seen: set[str] = set()
    buckets: dict[str, list[str]] = {
        "class": [],
        "interface": [],
        "type": [],
        "function": [],
        "const_fn": [],
        "method": [],
        "import": [],
    }
    # General PascalCase type-name suffixes (across TS / ORMs / DDD), so an
    # imported type lands in the "type" bucket by universal convention — no
    # project-specific identifiers.
    type_suffixes = (
        "Type", "Record", "Schema", "Model", "Entity", "Dto", "DTO",
        "Interface", "Props", "Config", "Options", "Params", "Request",
        "Response", "Payload", "Row", "Document", "Table",
    )
    for match in re.finditer(r"\bimport\s+(?:type\s+)?\{([^}]+)\}\s+from\b", content, re.DOTALL):
        for part in match.group(1).split(","):
            name = part.strip().removeprefix("type ").split(" as ", 1)[0].strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                continue
            key = f"import:{name}"
            if key in seen:
                continue
            seen.add(key)
            looks_like_type = name[:1].isupper() and name.endswith(type_suffixes)
            bucket = "type" if looks_like_type else "import"
            buckets[bucket].append(name)

    for kind, pattern in _CODE_SYMBOL_PATTERNS:
        flags = re.MULTILINE
        for match in re.finditer(pattern, content, flags):
            name = match.group(1)
            if name in _CODE_SYMBOL_KEYWORDS:
                continue
            key = f"{kind}:{name}"
            if key in seen:
                continue
            seen.add(key)
            buckets[kind].append(name)

    lines = ["# Code symbols"]
    labels = {
        "class": "classes",
        "interface": "interfaces",
        "type": "types",
        "method": "methods",
        "function": "functions",
        "const_fn": "const functions",
        "import": "imports",
    }
    for kind, label in labels.items():
        names = buckets[kind]
        if not names:
            continue
        line = f"- {label}: " + ", ".join(names[:40])
        if len(line) > max_chars:
            line = line[:max_chars - 3].rstrip(", ") + "..."
        if sum(len(x) + 1 for x in lines) + len(line) + 1 > max_chars:
            break
        lines.append(line)

    return "\n".join(lines) if len(lines) > 1 else ""


def _prepend_code_symbol_summary(out: str, content: str, budget: int | None) -> str:
    summary = _code_symbol_summary(content, budget)
    if not summary:
        return out
    if all(name in out for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", summary)[:12]):
        return out

    combined = f"{summary}\n\n{out}" if out else summary
    if budget is None or len(combined) <= budget * 4:
        return combined
    max_chars = max(1, budget * 4)
    remaining = max_chars - len(summary) - 2
    if remaining <= 0:
        return summary[:max_chars]
    return f"{summary}\n\n{out[:remaining]}"


def compress(
    content: str,
    budget: int | None = None,
    content_type: str | None = None,
    target_ratio: float = 0.3,
    profile: str | None = None,
) -> str:
    """Compact content without a task query.

    This is the lightweight, query-agnostic API. It is useful for structural
    compaction and bounded truncation, but it cannot promise to retain the
    answer to an unknown future question. Use ``optimize(..., query=...)`` for
    high-ratio task-conditioned context selection.

    Args:
        content: Any text content (code, prose, JSON, logs, emails, etc.)
        budget: Estimated token upper bound. If set, overrides target_ratio.
        content_type: Optional hint ("json", "code", "prose", "log", etc.)
        target_ratio: Compression ratio if budget not specified (0.3 = keep 30%)

    Returns:
        Structurally compacted text within the requested estimated budget.

    Example::

        from entroly import compress

        # Compress a large API response
        compressed = compress(api_response, budget=1000)

        # Compress code with type hint
        compressed = compress(source_code, budget=2000, content_type="code")
    """
    if not content:
        return content

    if profile:
        prof_cfg = get_profile(profile)
        target_ratio = prof_cfg.get("compression_ratio", target_ratio)

    # Token estimate (4 chars ≈ 1 token) and target ratio.
    current_tokens = len(content) // 4
    if budget is not None:
        if current_tokens <= budget:
            return content  # Already within budget — nothing to do.
        ratio = max(0.05, budget / max(current_tokens, 1))
    else:
        ratio = max(0.05, target_ratio)

    # Code can use the native skeletonizer. Other formats keep the existing
    # content-aware structural compactors. Neither path is query-conditioned.
    is_code = content_type == "code" or (content_type is None and _looks_like_code(content))
    if is_code:
        try:
            out = _compress_code(content, ratio)
        except Exception:
            out, _, _ = universal_compress(content, ratio, content_type)
        out = _prepend_code_symbol_summary(out, content, budget)
    else:
        out, _, _ = universal_compress(content, ratio, content_type)

    out = _ensure_non_empty(out, content, budget, ratio)
    out = _enforce_budget(out, budget)
    _track_savings(current_tokens, len(out) // 4, "compress")
    return out


_QUERY_INSTRUCTION_TERMS = frozenset({
    "above",
    "answer",
    "briefly",
    "concise",
    "concisely",
    "context",
    "document",
    "documents",
    "exact",
    "exactly",
    "passage",
    "passages",
    "question",
    "reply",
    "shortest",
    "span",
    "summarize",
    "summary",
})


def _message_tokens(messages: list[dict[str, Any]]) -> int:
    """Return the SDK's stable, dependency-free message token estimate."""
    return sum(
        len(message.get("content", "")) // 4
        for message in messages
        if isinstance(message.get("content"), str)
    )


def _infer_compression_query(messages: list[dict[str, Any]]) -> str:
    """Use the last user turn as the evidence-selection query.

    A lone user message is often the entire document rather than a question, so
    only infer a task when another textual message precedes it. Callers that
    need task-conditioned compression for a single blob should use ``optimize``.
    """
    textual = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message.get("content"), str)
        and message.get("content", "").strip()
    ]
    if len(textual) < 2:
        return ""
    for _, message in reversed(textual):
        if message.get("role") in ("user", "human"):
            return message["content"]
    return ""


def _has_evidence_query(query: str) -> bool:
    """Return whether a query contains terms that can rank source evidence."""
    if not query.strip():
        return False
    try:
        from .context_receipts.retrieval import tokenize

        terms = set(tokenize(query))
    except Exception:
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_-]{3,}", query)}
    return bool(terms - _QUERY_INSTRUCTION_TERMS)


def _expand_source_span(
    source: bytes,
    byte_start: int,
    byte_end: int,
    *,
    lookaround: int = 512,
) -> tuple[int, int]:
    """Extend a receipt chunk to nearby sentence boundaries.

    Receipt chunks deliberately overlap and may end in the middle of an
    assertion.  Emitting their text independently can therefore split a fact
    across distant relevance-ranked positions.  Keep the receipt's byte
    coordinates authoritative, but include a small amount of source text so a
    selected assertion remains complete.
    """
    start = max(0, min(byte_start, len(source)))
    end = max(start, min(byte_end, len(source)))

    prefix_start = max(0, start - lookaround)
    prefix = source[prefix_start:start]
    previous_boundaries = [
        prefix.rfind(marker) + len(marker)
        for marker in (b"\n", b". ", b"? ", b"! ")
        if prefix.rfind(marker) >= 0
    ]
    if previous_boundaries:
        start = prefix_start + max(previous_boundaries)

    suffix_end = min(len(source), end + lookaround)
    suffix = source[end:suffix_end]
    next_boundaries: list[int] = []
    for marker in (b"\n", b". ", b"? ", b"! "):
        position = suffix.find(marker)
        if position >= 0:
            # Keep terminal punctuation, but not the whitespace that begins
            # the following sentence or paragraph.
            terminal_length = 0 if marker == b"\n" else 1
            next_boundaries.append(end + position + terminal_length)
    if next_boundaries:
        end = min(next_boundaries)

    return start, max(start, end)


def _merge_source_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return source-ordered, overlap-free byte spans."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _render_source_spans(source: bytes, spans: list[tuple[int, int]]) -> str:
    """Reconstruct selected receipt spans from the original source."""
    return "\n\n".join(
        source[start:end].decode("utf-8") for start, end in spans
    ).strip()


def _compress_message_content(
    content: str,
    *,
    budget: int,
    query: str,
    profile: str,
) -> str:
    """Compress one message, preferring task-conditioned evidence selection.

    ``compress`` is intentionally query-agnostic. Conversation compression is
    not: when a distinct final user turn exists, the older context can be
    chunked and ranked against that task. This avoids buying a large token cut
    by silently dropping the answer-bearing passage.
    """
    if (
        profile == "max"
        or not _has_evidence_query(query)
        or len(content) < 200
        or len(content) // 4 <= budget
    ):
        return compress(content, budget=budget, profile=profile)

    try:
        from .context_receipts.ingest import ingest_documents
        from .context_receipts.retrieval import rank_chunks
        from .context_receipts.selection import select_context

        preferred_chunk_tokens = 160 if profile == "safe" else 240
        chunk_tokens = max(
            40,
            min(preferred_chunk_tokens, max(40, budget // 4)),
        )
        overlap_tokens = min(
            chunk_tokens - 1,
            48 if profile == "safe" else 24,
        )
        index = ingest_documents(
            [("conversation-context.txt", content)],
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        if len(index.chunks) < 2:
            return compress(content, budget=budget, profile=profile)

        ranked = rank_chunks(index, query)
        selection = select_context(
            index,
            ranked,
            [],
            token_budget=max(1, budget),
        )
        if not selection.selected:
            return compress(content, budget=budget, profile=profile)

        char_budget = max(1, budget * 4)
        source = content.encode("utf-8")
        chosen_spans: list[tuple[int, int]] = []
        attempted_ids: set[str] = set()

        def add_if_fits(
            chunk_id: str,
            byte_start: int,
            byte_end: int,
        ) -> bool:
            nonlocal chosen_spans
            attempted_ids.add(chunk_id)
            candidate = _expand_source_span(source, byte_start, byte_end)
            trial = _merge_source_spans([*chosen_spans, candidate])
            if len(_render_source_spans(source, trial)) <= char_budget:
                chosen_spans = trial
                return True
            return False

        # Relevance order controls admission under pressure, but accepted
        # receipt coordinates are reconstructed from the original source. This
        # removes duplicated overlap and prevents a long assertion from being
        # split between distant score-ranked chunks.
        for item in selection.selected:
            add_if_fits(item.chunk_id, item.byte_start, item.byte_end)

        # The receipt selector intentionally stops after its evidence frontier.
        # Fill a gentle relative target with surrounding source context while
        # keeping every already-admitted evidence span pinned.
        for chunk in sorted(index.chunks, key=lambda item: item.byte_start):
            if chunk.chunk_id in attempted_ids:
                continue
            add_if_fits(chunk.chunk_id, chunk.byte_start, chunk.byte_end)

        # Whole receipt chunks are deliberately coarse. If one skipped chunk
        # leaves meaningful room at a gentle target, backfill only complete
        # source sentences from the uncovered ranges. This approaches the
        # requested operating point without reintroducing a mid-assertion cut.
        uncovered: list[tuple[int, int]] = []
        cursor = 0
        for start, end in chosen_spans:
            if cursor < start:
                uncovered.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(source):
            uncovered.append((cursor, len(source)))

        for start, end in uncovered:
            rendered_length = len(_render_source_spans(source, chosen_spans))
            available = char_budget - rendered_length - (2 if chosen_spans else 0)
            if available < 40:
                break
            gap = source[start:end].decode("utf-8")
            prefix = gap[:available]
            boundary_ends = [
                position + (0 if marker == "\n" else 1)
                for marker in ("\n", ". ", "? ", "! ")
                if (position := prefix.rfind(marker)) >= 0
            ]
            if not boundary_ends:
                continue
            prefix = prefix[: max(boundary_ends)].rstrip()
            if len(prefix) < 40:
                continue
            prefix_end = start + len(prefix.encode("utf-8"))
            trial = _merge_source_spans([*chosen_spans, (start, prefix_end)])
            if len(_render_source_spans(source, trial)) <= char_budget:
                chosen_spans = trial

        selected = _render_source_spans(source, chosen_spans)

        selected = _ensure_non_empty(selected, content, budget, 1.0)
        return _enforce_budget(selected, budget)
    except Exception:
        # The public SDK remains fail-open if optional receipt machinery is not
        # importable in a minimal package. The established structural path is
        # still bounded and deterministic.
        return compress(content, budget=budget, profile=profile)


def compress_messages(
    messages: list[dict[str, Any]],
    budget: int = 50_000,
    preserve_last_n: int = 4,
    model: str | None = None,
    client_key: str | None = None,
    distill: bool = True,
    profile: str = "balanced",
    target_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """Compress a conversation message list to fit within a token budget.

    Preserves the most recent messages verbatim and progressively
    compresses older messages using content-aware compression.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        budget: Target total token count for all messages
        preserve_last_n: Number of most recent messages to keep verbatim
        model: Optional provider model name used to cap the context budget
        client_key: Optional stable key for reusing nearly-identical older context
        distill: Strip filler from older assistant responses before compression
        profile: ``safe`` and ``balanced`` use task-conditioned evidence
            selection; ``max`` uses the fastest structural compression path
        target_ratio: Optional relative keep ratio for a smooth operating point
            (0.90 keeps about 90% of the original estimated message tokens).
            The stricter of ``budget`` and this relative target is enforced.

    Returns:
        Compressed message list.

    Example::

        from entroly import compress_messages

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": very_long_response},
            {"role": "user", "content": "Fix the bug"},
        ]
        compressed = compress_messages(messages, budget=30000)
    """
    if not messages:
        return messages

    # Validate public controls even when the input is already in budget so a
    # bad configuration is visible instead of silently becoming a no-op.
    get_profile(profile)
    if target_ratio is not None and not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be greater than 0 and at most 1")

    source_tokens = _message_tokens(messages)
    if target_ratio is not None:
        budget = min(budget, max(1, int(source_tokens * target_ratio)))

    # ── Provider-aware budget ──
    # When a model is named, cap the budget to its context window (leaving
    # ~20% headroom for the response) so the request stays provider-correct.
    if model:
        try:
            from .proxy_config import context_window_for_model
            window = context_window_for_model(model)
            if window and window > 0:
                budget = min(budget, int(window * 0.8))
        except Exception:
            pass  # Unknown model — keep the caller's budget.

    if target_ratio == 1 and source_tokens <= budget:
        return messages

    # Pre-pass: collapse aged tool outputs to one-line digests. This is
    # near-free and orthogonal to the budget-driven compression below
    # (which operates on text length, not message semantics). Same path
    # used by the proxy — keeps proxy and SDK behavior aligned.
    try:
        from .hardening import prune_aged_tool_outputs
        messages, _ = prune_aged_tool_outputs(
            messages, tail_window=preserve_last_n
        )
    except Exception:
        pass  # Never block compress_messages on the pre-pass.

    # Estimate total tokens
    total_tokens = _message_tokens(messages)
    if total_tokens <= budget:
        return messages  # Already within budget

    query = _infer_compression_query(messages)

    # Adaptive preserve_last_n: shrink if there aren't enough
    # older messages to compress
    effective_preserve = min(preserve_last_n, max(1, len(messages) - 1))

    # Split: recent (verbatim) vs older (compressible)
    recent = messages[-effective_preserve:]
    older = messages[:-effective_preserve]

    recent_tokens = sum(
        len(m.get("content", "")) // 4
        for m in recent if isinstance(m.get("content"), str)
    )
    remaining_budget = max(budget - recent_tokens, 1)

    # If recent alone busts budget, compress even recent messages
    # (except the very last user message)
    if recent_tokens > budget:
        return _compress_all_messages(
            messages,
            budget,
            query=query,
            profile=profile,
        )

    if not older:
        # All messages are "recent" — compress all except the last
        return _compress_all_messages(
            messages,
            budget,
            query=query,
            profile=profile,
        )

    # Compute per-message compression ratio
    older_tokens = sum(
        len(m.get("content", "")) // 4
        for m in older if isinstance(m.get("content"), str)
    )
    ratio = max(0.05, remaining_budget / max(older_tokens, 1))

    result = []
    for msg in older:
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 200:
            result.append(msg)
            continue

        role = msg.get("role", "")

        # ── Distillation ── strip filler from assistant responses before
        # budget compression (code blocks + technical content preserved).
        if distill and profile != "safe" and role == "assistant":
            try:
                from .proxy_transform import distill_response
                content, _, _ = distill_response(content, mode="full")
            except Exception:
                pass

        # Tool results get more aggressive compression.
        msg_ratio = ratio * 0.5 if role in ("tool", "function") else ratio

        msg_budget = max(1, int((len(content) // 4) * msg_ratio))
        compressed_content = _compress_message_content(
            content,
            budget=msg_budget,
            query=query,
            profile=profile,
        )
        new_msg = dict(msg)
        new_msg["content"] = compressed_content
        result.append(new_msg)

    # ── Cache-aligned reuse ── stabilize the compressed older-context across
    # calls for the same client so provider prefix caches can keep hitting on
    # minor turn-to-turn changes instead of busting on every recompression.
    if client_key:
        result = _cache_align_older(client_key, result)

    result.extend(recent)
    return result


# Module-level cache-alignment state (bounded; mirrors the proxy's
# CacheAligner so SDK users get prefix-stable context across calls).
_cache_aligner: Any = None
_cache_align_prev: dict[str, list[dict[str, Any]]] = {}


def _cache_align_older(
    client_key: str, older_msgs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reuse the previous compressed older-context when the new one is
    >~90% similar, preserving the provider's cached prefix. Fail-open."""
    global _cache_aligner
    try:
        if _cache_aligner is None:
            from .cache_aligner import CacheAligner
            _cache_aligner = CacheAligner()
        text = "\n".join(
            m.get("content", "") for m in older_msgs
            if isinstance(m.get("content"), str)
        )
        _, hit = _cache_aligner.align(client_key, text)
        if hit and client_key in _cache_align_prev:
            return _cache_align_prev[client_key]
        _cache_align_prev[client_key] = older_msgs
        if len(_cache_align_prev) > 100:
            _cache_align_prev.pop(next(iter(_cache_align_prev)))
        return older_msgs
    except Exception:
        return older_msgs  # Cache-align is best-effort; never block output.


def _compress_all_messages(
    messages: list[dict[str, Any]],
    budget: int,
    *,
    query: str = "",
    profile: str = "balanced",
) -> list[dict[str, Any]]:
    """Compress all messages proportionally, preserving the last user message.

    Used when even the 'recent' window busts the token budget.
    Sorts messages by size and compresses the largest first.
    """
    # Always preserve the last user message verbatim
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") in ("user", "human"):
            last_user_idx = i
            break

    total_tokens = sum(
        len(m.get("content", "")) // 4
        for m in messages if isinstance(m.get("content"), str)
    )
    preserved_tokens = (
        len(messages[last_user_idx].get("content", "")) // 4
        if last_user_idx >= 0 and isinstance(messages[last_user_idx].get("content"), str)
        else 0
    )
    compressible_tokens = max(1, total_tokens - preserved_tokens)
    remaining_budget = max(1, budget - preserved_tokens)
    ratio = max(0.05, remaining_budget / compressible_tokens)

    result = []
    for i, msg in enumerate(messages):
        if i == last_user_idx:
            result.append(msg)  # Preserve last user message
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 200:
            result.append(msg)
            continue

        role = msg.get("role", "")
        msg_ratio = ratio * 0.3 if role in ("tool", "function") else ratio

        msg_budget = max(1, int((len(content) // 4) * msg_ratio))
        compressed_content = _compress_message_content(
            content,
            budget=msg_budget,
            query=query,
            profile=profile,
        )
        new_msg = dict(msg)
        new_msg["content"] = compressed_content
        result.append(new_msg)

    # Final hard guard: proportional estimates can still land slightly above
    # budget due to token-estimation floors and preserved messages. Tighten the
    # largest compressible messages until the SDK contract is honored.
    def _total(result_messages: list[dict[str, Any]]) -> int:
        return sum(
            len(m.get("content", "")) // 4
            for m in result_messages if isinstance(m.get("content"), str)
        )

    while _total(result) > budget:
        candidates = [
            (len(m.get("content", "")), i)
            for i, m in enumerate(result)
            if i != last_user_idx and isinstance(m.get("content"), str) and len(m.get("content", "")) > 80
        ]
        if not candidates:
            break
        _, idx = max(candidates)
        msg = dict(result[idx])
        content = msg.get("content", "")
        current_tokens = max(1, len(content) // 4)
        target_tokens = max(1, int(current_tokens * 0.8))
        msg["content"] = compress(content, budget=target_tokens)
        if msg["content"] == content:
            msg["content"] = _budget_bounded_head(content, target_tokens)
        result[idx] = msg

    return result


def _looks_like_code(text: str) -> bool:
    """Heuristic: does this text look like source code?"""
    code_indicators = 0
    lines = text[:2000].split("\n")
    for line in lines[:30]:
        stripped = line.strip()
        if any(kw in stripped for kw in [
            "def ", "class ", "import ", "from ", "fn ", "pub ",
            "function ", "const ", "let ", "var ", "return ",
            "if (", "for (", "while (", "struct ", "#include",
        ]):
            code_indicators += 1
        if stripped.endswith(("{", "}", ";", ":")):
            code_indicators += 1
    return code_indicators >= 3


def _compress_code(content: str, target_ratio: float) -> str:
    """Compress code using the Rust engine if available.

    Bug-fix history: prior versions ignored target_ratio (Rust path passed
    the full token count as the budget, and the fallback miscategorised
    code as "prose" for universal_compress). Both honored ratio in name
    only — the function silently no-op'd on inputs the engine considered
    already-skeletal. The fix below honors the requested ratio in both
    paths.
    """
    target_tokens = max(50, int((len(content) // 4) * target_ratio))
    try:
        from entroly_core import py_compress_block
        return py_compress_block(
            "assistant", content, target_tokens, "skeleton", None,
        )
    except ImportError:
        # Rust engine not available — use universal compressor with the
        # correct content_type so it picks the code-specific compactor.
        compressed, _, _ = universal_compress(content, target_ratio, "code")
        return compressed


# ── Verification SDK ─────────────────────────────────────────────────
# Hallucination detection + suppression, same 1-import philosophy.


def verify(
    code: str,
    context: str = "",
) -> dict[str, Any]:
    """Verify that LLM-generated code is grounded in the provided context.

    Returns a dict with:
      - ipd: Identifier Provenance Deficit [0.0 = grounded, 1.0 = invented]
      - verdict: "grounded", "partial", or "invented"
      - invented: list of ungrounded identifiers

    Example::

        from entroly import verify
        result = verify(llm_output, context=repo_code)
        if result["ipd"] > 0.2:
            print("Warning: hallucinated identifiers detected")
    """
    from .verifiers.provenance_tracer import trace_provenance

    bipt = trace_provenance(code, context)
    invented = [
        {"name": t.identifier.name, "kind": t.identifier.kind,
         "grounding": round(t.grounding_ratio, 3)}
        for t in bipt.traces if t.verdict in ("invented", "partial")
    ]
    return {
        "ipd": round(bipt.ipd, 3),
        "verdict": bipt.verdict,
        "total_identifiers": len(bipt.traces),
        "invented_count": len(invented),
        "invented": invented,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Full Hallucination Detection — 4-Signal Fusion Cascade
# ═══════════════════════════════════════════════════════════════════════════
#
# Same pipeline as the proxy (WITNESS + ECE + EPR + Spectral) but callable
# as a single function.  Zero LLM calls — pure local compute.
#
#   from entroly import detect_hallucination
#   result = detect_hallucination(response, context=source_code)
#   if result["verdict"] == "flag":
#       print("Hallucination detected:", result["recommendation"])
#


def detect_hallucination(
    response: str,
    context: str = "",
    prompt: str = "",
) -> dict[str, Any]:
    """Detect hallucination in AI-generated text, locally and at $0.

    Scoring is WITNESS-only — the one benchmark-validated detector here
    (faithful HaluEval-QA AUROC ≈ 0.80; deterministic, no LLM call).
    `fused_risk` is exactly the WITNESS risk and is never blended with
    the other signals: a fitted blend over them was falsified as a
    benchmark-construction artifact (see the Fusion-4 falsification
    report), and mixing uncalibrated correlates into a calibrated score
    only degrades it. ECE (Fisher curvature), EPR (entropy production)
    and Spectral (entity cross-similarity) run as labelled *unvalidated*
    diagnostics; their only authority is selective abstention — on
    strong joint disagreement they can nudge the *action* one notch
    toward caution (pass → warn), never alter the number or lower a
    verdict. Every signal is returned for audit.

    Args:
        response: The AI-generated text to verify
        context: Source material the AI was supposed to reference
        prompt: The original query/instruction (helps calibrate)

    Returns:
        Dict with:
          - fused_risk: WITNESS risk [0.0 = grounded, 1.0 = hallucinated]
          - verdict: "pass" (<0.15), "warn" (0.15–0.40), or "flag" (>0.40)
          - recommendation: Human-readable action to take
          - primary_signal: always "witness" (the validated detector)
          - auxiliary_abstention: True if aux signals forced pass→warn
          - witness: {risk_score, groundedness, n_claims, validated:True}
          - ece / epr / spectral: {risk_score, validated:False} diagnostics
          - flagged_claims: [{claim, risk, label}, ...] from WITNESS

    Example::

        from entroly import detect_hallucination

        result = detect_hallucination(
            response=llm_output,
            context=repo_code,
            prompt="fix the login bug",
        )
        if result["verdict"] == "flag":
            print("Hallucination risk:", result["fused_risk"])
            for claim in result["flagged_claims"]:
                print(f"  - {claim['claim']} (risk={claim['risk']})")
    """
    signals: dict[str, Any] = {}

    # ── 1. WITNESS — the ONLY benchmark-validated signal ──────────────
    # Faithful HaluEval-QA protocol: AUROC 0.798 (README). risk is the
    # *complement* of groundedness (summary_score is faithfulness, higher
    # = LESS risk). Deterministic + $0 (force_python, no NLI) to match
    # the published number and stay zero-cost.
    witness_risk = 0.0
    flagged_claims: list[dict[str, Any]] = []
    try:
        from .witness import WitnessAnalyzer
        analyzer = WitnessAnalyzer(use_nli=False, force_python=True)
        wr = analyzer.analyze(context or prompt, response)
        witness_risk = round(max(0.0, 1.0 - float(wr.summary_score)), 4)
        signals["witness"] = {
            "risk_score": witness_risk,
            "groundedness": round(float(wr.summary_score), 4),
            "n_claims": len(getattr(wr, "certificates", []) or []),
            "validated": True,
        }
        flagged_claims = [
            {"claim": c.claim_text[:120], "risk": round(c.risk, 3),
             "label": getattr(c, "label", "")}
            for c in (wr.flagged() or [])[:10]
        ]
    except Exception:
        signals["witness"] = {"risk_score": 0.0, "validated": True,
                              "status": "unavailable"}

    def _aux(name: str, fn: Any) -> float:
        """Run an unvalidated auxiliary signal, fail-open to 0.0."""
        try:
            r = max(0.0, min(1.0, float(fn())))
            signals[name] = {"risk_score": round(r, 4), "validated": False}
            return r
        except Exception:
            signals[name] = {"risk_score": 0.0, "validated": False,
                             "status": "unavailable"}
            return 0.0

    # ── 2-4. Unvalidated auxiliary tripwires ──────────────────────────
    # NOT individually benchmark-validated. Used ONLY to RAISE the alarm,
    # never to lower the WITNESS verdict, and combined with NO fitted
    # weights. A fitted linear blend over these is exactly what produced
    # the rejected Fusion-4 HaluEval-construction artifact — see
    # benchmarks/results/fusion4_falsification_report.md. Do not
    # "optimize" weights here; that re-introduces the artifact.
    from .ravs.ece import compute_fisher_curvature
    from .ravs.epr import compute_epr
    from .ravs.spectral import compute_spectral_consistency

    ece_risk = _aux(
        "ece", lambda: min(1.0, compute_fisher_curvature(response)[0] * 2.5)
    )
    epr_risk = _aux("epr", lambda: compute_epr(response).risk_score)
    spectral_risk = _aux(
        "spectral",
        lambda: 1.0 - compute_spectral_consistency(
            context or prompt, response).score,
    )

    # ── 5. Risk = the validated WITNESS signal ONLY ───────────────────
    # The numeric risk is exactly the WITNESS score (faithful HaluEval-QA
    # AUROC ≈ 0.80). It is NEVER blended with the unvalidated
    # auxiliaries: a fitted blend over those was falsified as a
    # benchmark-construction artifact (see benchmarks/results/
    # fusion4_falsification_report.md). Mixing an uncalibrated correlate
    # into a calibrated probability strictly degrades it — so the
    # auxiliaries do not touch the number.
    fused = witness_risk

    # Verdict bands on the validated signal. These are documented
    # heuristic operating points, NOT a conformal coverage guarantee:
    # the high-recall calibrated τ from the faithful benchmark would
    # flag almost everything, so balanced review bands are used and
    # labelled honestly rather than overclaimed.
    if fused < 0.15:
        verdict, rec = "pass", "Accept — response appears well-grounded"
    elif fused < 0.40:
        verdict, rec = "warn", "Review — some claims may not be grounded"
    else:
        verdict, rec = "flag", "Reject or rephrase — high hallucination risk"

    # ── 6. Selective-abstention escalator (action only, never score) ──
    # Unvalidated auxiliaries get exactly one defensible job: push the
    # ACTION one notch toward caution on STRONG JOINT disagreement
    # (≥2 of 3 above a high bar) when WITNESS would otherwise pass. They
    # never lower a verdict, never alter `fused_risk`, never fabricate a
    # number. This is the proven-safe direction (selective abstention /
    # route-to-review) consistent with the falsified escalation/
    # conformal work already in this codebase. No fitted constants.
    aux_strong = sum(r >= 0.60 for r in (ece_risk, epr_risk, spectral_risk))
    abstained = False
    if verdict == "pass" and aux_strong >= 2:
        verdict = "warn"
        rec = ("Review — WITNESS reads grounded but multiple unvalidated "
               "auxiliary signals disagree; abstaining toward caution")
        abstained = True

    return {
        "fused_risk": round(fused, 4),
        "verdict": verdict,
        "recommendation": rec,
        "primary_signal": "witness",
        "scoring": ("witness-only (benchmark-validated); auxiliaries are "
                    "unvalidated diagnostics that can only escalate the "
                    "action via selective abstention"),
        "auxiliary_abstention": abstained,
        "flagged_claims": flagged_claims,
        **signals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PRISM-Weighted Context Optimization — Full Engine Access
# ═══════════════════════════════════════════════════════════════════════════
#
# Gives pip SDK users the same PRISM 5D retrieval + knapsack DP as MCP/proxy.
#
#   from entroly import optimize
#   result = optimize(fragments, budget=8000, query="fix login bug")
#


def optimize(
    fragments: list[dict[str, Any]],
    budget: int = 128000,
    query: str = "",
) -> dict[str, Any]:
    """Select a task-conditioned context subset using the full engine.

    This is the high-ratio API: the query is preserved and routed through the
    same BM25/PRISM budgeted selection path as MCP ``optimize_context`` and
    proxy context injection. Use ``compress()`` only when no task query exists.

    Args:
        fragments: List of dicts with keys:
            - content (str): The text content
            - source (str): Source identifier (e.g. filename)
            - token_count (int, optional): Token count (auto-estimated if missing)
        budget: Maximum token budget for the selected context
        query: Current task/query for semantic relevance scoring

    Returns:
        Dict with:
          - selected: List of selected fragments (sorted by relevance)
          - total_tokens: Total tokens in selected context
          - fragments_selected: Count of selected fragments
          - fragments_total: Count of input fragments
          - context_text: Concatenated selected content (ready to inject)

    Example::

        from entroly import optimize

        fragments = [
            {"content": open(f).read(), "source": f}
            for f in Path(".").glob("**/*.py")
        ]
        result = optimize(fragments, budget=8000, query="fix the auth middleware")
        print(f"Selected {result['fragments_selected']}/{result['fragments_total']}")
        # Use result["context_text"] as your LLM system prompt context
    """
    from .server import EntrolyEngine

    engine = EntrolyEngine()

    # Ingest all fragments
    for frag in fragments:
        content = frag.get("content", "")
        source = frag.get("source", "unknown")
        tokens = frag.get("token_count", len(content) // 4)
        engine.ingest_fragment(content, source, tokens)

    # Optimize
    result = engine.optimize_context(token_budget=budget, query=query)

    selected = result.get("selected", [])
    context_parts = []
    total_tokens = 0
    for item in selected:
        if isinstance(item, dict):
            context_parts.append(item.get("content", ""))
            total_tokens += item.get("token_count", 0)

    return {
        "selected": selected,
        "total_tokens": total_tokens,
        "fragments_selected": len(selected),
        "fragments_total": len(fragments),
        "context_text": "\n\n".join(context_parts),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Context Receipts: auditable context selection for hard multi-document tasks.

def _normalize_receipt_documents(documents: Any) -> list[tuple[str, str]]:
    """Normalize SDK/MCP receipt inputs into ``[(source_path, text), ...]``."""
    if isinstance(documents, dict):
        return [(str(path), str(text)) for path, text in documents.items()]

    normalized: list[tuple[str, str]] = []
    for idx, item in enumerate(documents or []):
        if isinstance(item, dict):
            source = item.get("source_path") or item.get("source") or item.get("path") or f"document_{idx}.txt"
            text = item.get("text") if "text" in item else item.get("content", "")
            normalized.append((str(source), str(text)))
        else:
            source, text = item
            normalized.append((str(source), str(text)))
    return normalized


def create_context_receipt(
    documents: Any,
    query: str,
    budget: int = 8000,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
    recoverable: bool = False,
) -> dict[str, Any]:
    """Create an auditable Context Receipt from in-memory documents.

    ``documents`` may be a mapping of ``path -> text``, a list of
    ``(path, text)`` tuples, or a list of dicts with ``source_path``/``text``
    keys. The result records selected context, omitted context, dependency
    links, fingerprints, warnings, and deterministic risk controls.

    When ``recoverable`` is True, a project-local recovery bundle is also
    persisted so any omitted chunk can later be recovered — byte-exact and
    fingerprint-verified — via :func:`recover_receipt_omission`.
    """
    from .context_receipts import run_receipt_pipeline, run_recoverable_pipeline

    docs = _normalize_receipt_documents(documents)
    if recoverable:
        return run_recoverable_pipeline(
            docs,
            query=query,
            token_budget=budget,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            prefer_rust=prefer_rust,
        )["receipt"]
    return run_receipt_pipeline(
        docs,
        query=query,
        token_budget=budget,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        prefer_rust=prefer_rust,
    )


def context_receipt_from_path(
    path: str,
    query: str,
    budget: int = 8000,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Create a Context Receipt from a local document file or directory."""
    from .context_receipts import run_receipt_pipeline
    from .context_receipts.ingest import read_documents_from_path

    return run_receipt_pipeline(
        read_documents_from_path(path),
        query=query,
        token_budget=budget,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        prefer_rust=prefer_rust,
    )


def render_context_receipt(receipt: dict[str, Any], prefer_rust: bool = True) -> str:
    """Render a Context Receipt JSON object as a Markdown audit report."""
    from .context_receipts import markdown_report

    return markdown_report(receipt, prefer_rust=prefer_rust)


def explain_receipt_omission(
    receipt: dict[str, Any],
    chunk_id: str,
    prefer_rust: bool = True,
) -> str:
    """Explain why ``chunk_id`` was omitted, or report that it was selected."""
    from .context_receipts import explain_omitted

    return explain_omitted(receipt, chunk_id, prefer_rust=prefer_rust)


def recover_receipt_omission(
    receipt: dict[str, Any],
    chunk_id: str | None = None,
    *,
    store_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Recover the full, fingerprint-verified text of omitted context.

    Receipts *explain* what was dropped; this *recovers* it. Works on receipts
    created with ``recoverable=True`` (the recovery bundle is read from the local
    store). Pass ``chunk_id`` to recover one chunk or omit it for all omitted
    chunks. Each result carries ``verified=True`` only when the returned text is
    provably the exact content that was omitted — not a re-derivation.
    """
    from .context_receipts import recover_omitted

    return recover_omitted(receipt, chunk_id, store_dir=store_dir)


# EICV — Evidence-Invariant Causal Verification (Deterministic, $0)
# ═══════════════════════════════════════════════════════════════════════════
#
# Deterministic hallucination detection + suppression. No neural model,
# no LLM calls. Pure string operations + information theory.
#
#   from entroly import eicv_verify, eicv_suppress
#   cert = eicv_verify("Paris is in France", evidence="Paris is the capital of France.")
#   result = eicv_suppress(context="...", output="...")
#


def eicv_verify(
    claim: str,
    evidence: str,
    profile: str = "rag",
) -> dict[str, Any]:
    """Verify a single claim against evidence using the EICV pipeline.

    100% deterministic, zero LLM calls, zero network requests.

    Returns a dict (EICVCertificate) with:
      - phi: epistemic support density [0.0 = hallucinated, 1.0 = grounded]
      - hallucination_score: 1 - phi
      - decision: "supported" | "abstain" | "hallucinated"
      - layer_scores: per-layer breakdown (T(G), NLI, RNR, gamma, H_sem)
      - n_claim_atoms / n_ev_atoms: atomic decomposition counts
      - unsupported_fraction / contradiction_fraction
      - elapsed_ms

    Profiles control the decision thresholds:
      - "rag" (default): strict, for retrieval-augmented generation
      - "qa": moderate-strict for QA outputs
      - "summarization": tolerant of paraphrase
      - "dialogue": broader abstain band
      - "fact_check": hardest (FEVER-like)

    Accuracy on public benchmarks (FEVER, SQuAD v2, HaluEval-QA) is
    documented in benchmarks/results/. False-positive and false-negative
    rates are non-zero.

    Example::

        from entroly import eicv_verify

        cert = eicv_verify(
            claim="Paris is the capital of France.",
            evidence="France is a country in Europe. Its capital is Paris."
        )
        if cert["decision"] == "hallucinated":
            print(f"Hallucinated (phi={cert['phi']})")

    Args:
        claim: The claim to verify
        evidence: The grounding evidence
        profile: Decision profile (default "rag")
    """
    from .eicv import EICVAnalyzer
    analyzer = EICVAnalyzer(profile=profile)
    cert = analyzer.verify(evidence, claim)
    return cert.as_dict()


def eicv_suppress(
    context: str,
    output: str,
    profile: str = "rag",
    mode: str = "strict",
) -> dict[str, Any]:
    """Verify and suppress hallucinations in LLM output using EICV.

    100% deterministic, zero LLM calls. Decomposes output into claims,
    verifies each claim against the grounding context, and applies a
    graduated suppression policy.

    Modes:
      - "audit": no output change; certificates only (for dashboards)
      - "annotate": append warning footer listing unverified claims
      - "strict" (default): graduated 4-action policy:
          supported   → PASS (no change)
          abstain     → HEDGE (append "[unverified]")
          hallucinated → SUPPRESS (remove claim sentence)

    Returns a dict (SuppressionResult) with:
      - rewritten_output: the (possibly modified) response text
      - n_claims / n_supported / n_abstained / n_hallucinated
      - suppressed_count / warned_count / hallucination_rate
      - certificates: list of per-claim EICVCertificate dicts
      - latency_ms

    Example::

        from entroly import eicv_suppress

        result = eicv_suppress(
            context="The project uses Rust and Python.",
            output="The project uses Rust. It also uses Java and C++.",
            mode="strict",
        )
        print(result["rewritten_output"])
        print(f"Suppressed {result['suppressed_count']} hallucinated claims")

    Args:
        context: The grounding evidence the LLM was supposed to use
        output: The LLM's response text to verify
        profile: Suppression profile (default "rag")
        mode: "audit" | "annotate" | "strict" (default "strict")
    """
    from .eicv_suppressor import EICVSuppressor
    suppressor = EICVSuppressor(profile=profile, mode=mode)
    result = suppressor.suppress(context, output)
    return result.as_dict()
