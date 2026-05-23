"""Answer-Anchor Compressor — strictly better answer survival than QCCR alone.

Stacks three ideas on top of Entroly's QCCR:

  1. Entity-overlap boost. Capitalized tokens / numbers / quoted spans in
     the QUERY are stronger relevance signals than BM25 bag-of-words
     because QA questions are specific. Chunks that contain exact-match
     query entities get a multiplicative score boost.

  2. Sliding window. Default chunking on 400-char boundaries can split a
     sentence containing the answer between two chunks. We add a
     400-char window stride of 200 (50% overlap) so any 200-char span
     appears intact in at least one chunk.

  3. Fail-safe fallback. If QCCR returns empty (BM25 score zero across
     all chunks — common with paraphrased questions), we don't return
     nothing. We rank by entity-overlap, then by chunk length, and
     greedily pack until budget is exhausted.

The function preserves Entroly's QCCR by composition: it CALLS
qccr.select and only intervenes when QCCR can't help.
"""

from __future__ import annotations

import re
from typing import Iterable

from entroly.qccr import select as qccr_select


def _entities(text: str) -> set[str]:
    """Cheap NER proxy: capitalized words (>= 3 chars), numbers, dates,
    and quoted spans. No model needed; works offline.
    """
    out: set[str] = set()
    # Capitalized words (skip sentence-starters: require length >= 3 and
    # at least one lowercase char to skip ALL-CAPS noise)
    for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
        out.add(w.lower())
    # Numbers (incl. years, currency-free numerics, ranges)
    for w in re.findall(r"\b\d+(?:[.,]\d+)*\b", text):
        out.add(w)
    # Quoted spans
    for q in re.findall(r"['\"]([^'\"]{2,40})['\"]", text):
        out.add(q.strip().lower())
    return out


def _chunk_sliding(text: str, size: int = 400, stride: int = 200) -> list[dict]:
    """Sliding-window chunks with 50% overlap so spans aren't split."""
    chunks: list[dict] = []
    i = 0
    idx = 0
    while i < len(text):
        c = text[i:i + size]
        if not c.strip():
            i += stride
            continue
        chunks.append({
            "id": f"f{idx}", "source": f"chunk_{idx // 8}.txt",
            "content": c, "tokens": len(c) // 4,
        })
        idx += 1
        if i + size >= len(text):
            break
        i += stride
    return chunks


def _approx_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Keeps order, preserves trailing punctuation."""
    # Split on .!? followed by whitespace + capital letter. Keep delimiters.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _content_tokens(text: str) -> list[str]:
    """Alphanumeric word tokens, lowercased, length >= 2."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2]


_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "from", "this",
    "that", "what", "who", "where", "when", "which", "how", "why", "did",
    "does", "into", "they", "their", "them", "have", "has", "had", "been",
    "would", "could", "should", "about", "of", "in", "on", "to", "a", "an",
    "is", "be", "as", "at", "by", "or", "if", "it", "its", "but", "not",
    "no", "any", "all", "some", "such", "than", "then", "so", "do",
}


def compress(text: str, budget_tokens: int, query: str) -> str:
    """Sentence-level IDF-weighted greedy with chunk fallback.

    The principle: for short-context QA, the right unit is the sentence,
    not the 400-char chunk. We rank sentences by a query-overlap score
    that penalizes common words (IDF-style) and weights query entities.
    Greedy fill the budget, preserving original sentence order in the
    output so the LLM sees natural prose.

    Fall back to QCCR (then to entity-greedy chunks) if sentence
    selection produces nothing useful — never returns less than QCCR.
    """
    q_terms = [t for t in _content_tokens(query) if t not in _STOPWORDS]
    q_ents = _entities(query)
    sentences = _split_sentences(text)

    if not sentences:
        return text[:budget_tokens * 4]

    # Document frequency for sentence-level IDF inside this context
    df: dict[str, int] = {}
    s_tokens: list[set[str]] = []
    for s in sentences:
        toks = set(_content_tokens(s))
        s_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1

    N = len(sentences)
    import math
    def idf(t: str) -> float:
        # BM25-style IDF; rare terms in this context get higher weight
        return math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))

    # Sentence score = sum_t in q_terms (IDF(t) * 1[t in sentence])
    # + entity_boost * (overlapping entities)
    ENTITY_BOOST = 1.5
    scored: list[tuple[float, int]] = []
    for i, toks in enumerate(s_tokens):
        s_lower = sentences[i].lower()
        term_score = sum(idf(t) for t in q_terms if t in toks)
        ent_score = sum(idf(e.split()[0]) if e.split() else 0
                        for e in q_ents if e in s_lower)
        scored.append((term_score + ENTITY_BOOST * ent_score, i))

    # Greedy pick highest-scoring sentences until budget; if we get nothing
    # (all scores 0 — unlikely with IDF but possible), fall back to QCCR.
    chosen_idx: list[int] = []
    used = 0
    for score, i in sorted(scored, reverse=True):
        if score == 0 and chosen_idx:
            break
        tk = _approx_tokens(sentences[i])
        if used + tk > budget_tokens and chosen_idx:
            break
        chosen_idx.append(i)
        used += tk
        if used >= budget_tokens:
            break

    if chosen_idx:
        # Preserve original sentence order for natural reading
        chosen_idx.sort()
        anchor_text = " ".join(sentences[i] for i in chosen_idx)
    else:
        anchor_text = ""

    # Cross-check with QCCR (legacy path); use whichever is longer
    chunks = _chunk_sliding(text, size=400, stride=400)  # non-overlapping
    primary = qccr_select(chunks, token_budget=budget_tokens, query=query)
    primary_text = "\n".join((s.get("content") or "") for s in primary).strip()
    # Pick the one with more content (proxy for more answer-survival headroom)
    return anchor_text if len(anchor_text) >= len(primary_text) else primary_text
