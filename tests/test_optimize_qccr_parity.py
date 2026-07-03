"""Regression gate: the SHIPPED query path == the VALIDATED compressor (qccr).

Background. Every committed accuracy benchmark (needle/longbench/squad/...) is
produced by ``entroly.qccr.select``. For a long time the programmatic surfaces
(SDK ``optimize()``, MCP ``optimize_context``, the proxy) instead ran the engine's
fragment-level selector, which those benchmarks never exercised — i.e. the
measured compressor and the shipped compressor diverged. ``optimize_context``
was wired to route its query path through qccr so the three surfaces ship what
is measured.

This test locks that property so it cannot silently regress:

  1. With a query, the engine selects via qccr (``selector == "qccr"``).
  2. The qccr path actually preserves the query-relevant block and drops dense,
     irrelevant distractors (deterministic — no LLM, no network).
  3. Without a query, it does NOT use qccr (the importance path).
"""
from __future__ import annotations

import pytest

from entroly.native_status import QCCR_SYMBOLS, native_status
from entroly.server import EntrolyEngine, EntrolyConfig


def _clean_engine() -> EntrolyEngine:
    if not native_status(QCCR_SYMBOLS).ok:
        pytest.skip("qccr routing requires current native QCCR symbols")
    cfg = EntrolyConfig()
    # Isolate from any on-disk checkpoint so the test store is exactly what we
    # ingest (otherwise a persisted repo index pollutes selection).
    cfg.use_persistent_index = False
    eng = EntrolyEngine(config=cfg)
    if not getattr(eng, "_use_rust", False):
        pytest.skip("qccr routing is gated on the native entroly-core engine")
    return eng


def _ingest_haystack_plus_answer(eng: EntrolyEngine) -> str:
    # 25 dense distractor fragments on unrelated topics + 1 relevant answer.
    # Density is controlled (distractors are information-rich) so that ONLY
    # query relevance — not entropy — can surface the answer.
    for i in range(25):
        content = (f"Region {i}: TLS certificate rotation, shard rebalancing with "
                   f"constant {i * 9}, connection-pool reaper interval, GC eden sizing.")
        eng.ingest_fragment(content, f"distractor_{i}.txt", len(content) // 4)
    answer = ("The rate limiter uses a token-bucket algorithm: bursts up to five "
              "thousand requests are absorbed by the bucket capacity, tokens refill "
              "at one hundred per second, and overflow returns HTTP 429.")
    eng.ingest_fragment(answer, "ratelimit.py", len(answer) // 4)
    return "token-bucket"


def test_query_path_uses_qccr_and_keeps_the_answer():
    eng = _clean_engine()
    marker = _ingest_haystack_plus_answer(eng)

    result = eng.optimize_context(
        token_budget=120, query="how does the rate limiter handle burst traffic"
    )

    assert result.get("selector") == "qccr", (
        "shipped query path must route through the validated qccr compressor; "
        f"got selector={result.get('selector')!r}"
    )
    selected = result.get("selected_fragments", [])
    kept = " ".join(f.get("content", "") for f in selected if isinstance(f, dict))
    assert marker in kept, "qccr must preserve the query-relevant answer block"
    # Discrimination: at this tight budget the dense distractors must be dropped.
    assert "shard rebalancing" not in kept, (
        "irrelevant distractors must be dropped at a tight budget"
    )
    tokens_used = sum(f.get("token_count", 0) for f in selected)
    total_available = sum(f.get("token_count", 0) for f in eng._rust.export_fragments())
    assert result["tokens_used"] == tokens_used
    assert result["total_tokens"] == tokens_used
    assert result["selected_count"] == len(selected)
    assert result["tokens_saved"] == total_available - tokens_used
    assert result["tokens_saved_this_call"] == result["tokens_saved"]
    assert result["optimization_stats"]["total_tokens"] == tokens_used


def test_no_query_does_not_use_qccr():
    eng = _clean_engine()
    _ingest_haystack_plus_answer(eng)
    result = eng.optimize_context(token_budget=120, query="")
    assert result.get("selector") != "qccr", (
        "the no-query path is query-agnostic importance selection, not qccr"
    )


def test_query_path_keeps_pinned_fragments_exact():
    eng = _clean_engine()
    pinned = eng.ingest_fragment(
        "Permanent project identity and safety constraints.",
        "IDENTITY.md",
        20,
        is_pinned=True,
    )
    _ingest_haystack_plus_answer(eng)

    result = eng.optimize_context(
        token_budget=120, query="how does the rate limiter handle burst traffic"
    )

    selected = result["selected_fragments"]
    identity = next(f for f in selected if f.get("source") == "IDENTITY.md")
    assert identity["id"] == pinned["fragment_id"]
    assert identity["content"] == "Permanent project identity and safety constraints."


def test_query_path_applies_native_fragment_feedback():
    eng = _clean_engine()
    eng.ingest_fragment(
        "The rate limiter uses a token bucket for burst traffic.",
        "ratelimit.py",
        20,
    )

    before = eng.optimize_context(256, "rate limiter burst traffic")[
        "selected_fragments"
    ][0]
    eng.record_failure([before["id"]])
    after = eng.optimize_context(256, "rate limiter burst traffic")[
        "selected_fragments"
    ][0]

    assert after["relevance"] < before["relevance"]
    assert before["source_fragment_ids"] == after["source_fragment_ids"]
