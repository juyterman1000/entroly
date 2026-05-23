"""
Query Refinement — Thin Python wrapper over Rust + optional LLM I/O
=====================================================================

All compute (TF-IDF key-term extraction, vagueness scoring, heuristic
refinement) runs in `entroly_core` (Rust). This module provides:

  1. `QueryRefiner` — dispatch class: tries Rust heuristic first,
     falls back to an optional LLM refine function if the query is vague.
  2. `make_openai_refine_fn` / `make_anthropic_refine_fn` — LLM I/O
     factories (must stay Python — external HTTP calls).

What moved to Rust (`entroly-core/src/query.rs`):
  - `analyze_query(text, fragment_summaries)` → (vagueness, terms, needs_refinement, reason)
  - `refine_heuristic(query, fragment_summaries)` → grounded refined query
  - TF-IDF scoring, stop-word filtering, specificity signals
"""

from __future__ import annotations

import logging
from collections.abc import Callable

try:
    from entroly_core import py_analyze_query, py_refine_heuristic
except ImportError:
    def py_analyze_query(query: str, summaries: list) -> tuple:  # type: ignore[misc]
        """Pure-Python fallback for query analysis."""
        terms = [w for w in query.lower().split() if len(w) > 2][:8]
        vagueness = max(0.0, min(1.0, 1.0 - len(terms) / 8.0))
        needs_refinement = vagueness > 0.45
        return (vagueness, terms, needs_refinement, "python_fallback")

    def py_refine_heuristic(query: str, summaries: list) -> str:  # type: ignore[misc]
        """Pure-Python fallback — returns query unchanged."""
        return query

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────

class QueryRefiner:
    """
    Refines vague developer queries into precise context-selection prompts.

    Rust handles all compute. Python only does LLM I/O (optional).

    Usage:
        refiner = QueryRefiner()
        result = refiner.refine("fix the bug", fragment_summaries=[...])
    """

    def __init__(self, llm_fn: Callable[[str], str] | None = None):
        """
        Args:
            llm_fn: Optional async/sync function that takes a query string and
                    returns a refined query string using an LLM. If None, only
                    the Rust heuristic path is used.
        """
        self._llm_fn = llm_fn

    def analyze(self, query: str, fragment_summaries: list[str] | None = None) -> dict:
        """
        Analyze a query for vagueness and key terms.

        Returns:
            dict with:
              - vagueness_score (float, 0–1)
              - key_terms (list[str])
              - needs_refinement (bool)
              - reason (str)
        """
        summaries = fragment_summaries or []
        vagueness, key_terms, needs_refinement, reason = py_analyze_query(query, summaries)
        return {
            "vagueness_score": vagueness,
            "key_terms": key_terms,
            "needs_refinement": needs_refinement,
            "reason": reason,
        }

    def refine(self, query: str, fragment_summaries: list[str] | None = None) -> str:
        """
        Refine a query using:
          1. Rust heuristic (always runs, zero latency)
          2. LLM refinement (only if query needs_refinement AND llm_fn is set)

        Returns the refined query string (original if no refinement applied).
        """
        summaries = fragment_summaries or []
        analysis = self.analyze(query, summaries)

        # Rust heuristic refinement
        heuristic = py_refine_heuristic(query, summaries)

        # LLM path: only if still vague and LLM function is configured
        if analysis["needs_refinement"] and self._llm_fn is not None:
            try:
                llm_result = self._llm_fn(heuristic)
                if llm_result and len(llm_result.strip()) > len(query):
                    logger.debug("QueryRefiner: LLM refinement applied (vagueness=%.2f)",
                                 analysis["vagueness_score"])
                    return llm_result.strip()
            except Exception as e:
                logger.warning("QueryRefiner: LLM call failed, using heuristic: %s", e)

        return heuristic


# ── LLM refine function factories (I/O only — must stay Python) ───────

def make_openai_refine_fn(api_key: str, model: str = "gpt-4o-mini") -> Callable[[str], str]:
    """
    Return a function that calls OpenAI to refine a query.

    Requires: openai package (`pip install openai`).
    The function is synchronous and makes one API call per invocation.

    Usage:
        refiner = QueryRefiner(llm_fn=make_openai_refine_fn(os.environ["OPENAI_API_KEY"]))
    """
    def _refine(query: str) -> str:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        system = (
            "You are a code context query refiner. Given a vague developer query, "
            "rewrite it to be more specific and precise. Include: specific symbol names, "
            "file patterns, error types, or CWE IDs if relevant. "
            "Return ONLY the refined query, nothing else."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    return _refine


def make_anthropic_refine_fn(
    api_key: str, model: str = "claude-haiku-20240307"
) -> Callable[[str], str]:
    """
    Return a function that calls Anthropic Claude to refine a query.

    Requires: anthropic package (`pip install anthropic`).

    Usage:
        refiner = QueryRefiner(llm_fn=make_anthropic_refine_fn(os.environ["ANTHROPIC_API_KEY"]))
    """
    def _refine(query: str) -> str:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=150,
            system=(
                "You are a code context query refiner. Rewrite vague developer queries "
                "to be specific: add symbol names, file paths, error codes, or CWE IDs. "
                "Return ONLY the refined query."
            ),
            messages=[{"role": "user", "content": query}],
        )
        return message.content[0].text.strip()

    return _refine
