"""
System 1 ↔ System 2 Coupling
============================

Bridges the two cognitions that have, until now, lived disjoint in entroly:

  System 1 (fast):  The Rust EntrolyEngine inside the proxy. Runs on every
                    request. Knapsack + entropy + dedup + guardrails +
                    channel + hierarchical + resonance + causal. No Python
                    verifiers. No vault.

  System 2 (slow):  The Python FlowOrchestrator + verifier stack + vault.
                    Runs on workspace events or explicit MCP `execute_flow`.
                    Compiles beliefs, runs BIPT/FORGE/TRIAD/PROVE/CAVE,
                    writes vault artifacts. No proxy.

These are two cognitions on the same substrate that, prior to this module,
did not exchange information. Verified beliefs produced by System 2 never
reached request-time context selection. Outcomes observed by System 1 never
triggered re-verification of the beliefs that informed them. This module
closes the loop.

Three coupling operations
-------------------------

1. inject_vault_beliefs(engine, vault, query) — S2 → S1
   Top-k verified, fresh, query-relevant beliefs are ingested as engine
   fragments. The engine's knapsack then weighs them against IDE-supplied
   fragments under a single objective. Verified beliefs naturally bubble up.

2. attribute_outcome(used_belief_ids, success, vault) — S1 → S2
   When the proxy observes outcome o for a context that contained belief b,
   a Bayesian update on c_b is applied. The update writes back to the
   belief's frontmatter (confidence, last_checked).

3. enqueue_reverification(failed_belief_ids, vault) — S1 → S2 escalation
   On bad outcomes, beliefs in the failing context are marked `stale` so
   FlowOrchestrator's verify-before-answer flow reverifies them next pass.

Mathematical foundation
-----------------------

Belief-conditioned selection. Let:
  B = vault belief set, each b ∈ B with (claim_id, entity, confidence c_b,
      sources S_b, status σ_b, last_checked t_b)
  F = raw candidate fragments at request time
  T = token budget
  q = query

Standard proxy:        C* = arg max optimize(F, T)
Coupled proxy:         C* = arg max optimize(F ∪ π(B; q), T)

The projection operator π filters B by σ_b ∈ {verified, inferred},
c_b ≥ τ_c, age(b) ≤ τ_a, and lifts surviving beliefs into the same
fragment space the IDE feeds. The engine's own BM25/entropy/dedup score
them — we do not duplicate scoring here.

Bayesian outcome update. For each belief b used in context C with observed
outcome o ∈ {success, failure}:

  Prior:           c_b
  Likelihoods:     p(o=success | b grounded)   = α
                   p(o=success | b not grounded) = β     (with α > β)
  Posterior:       c_b' = (c_b · L_g) / (c_b · L_g + (1 - c_b) · L_u)

  where L_g = α if success else (1-α)
        L_u = β if success else (1-β)

α and β are observation-model hyperparameters. Defaults (α=0.85, β=0.40)
encode "a grounded belief in context strongly predicts success; an
ungrounded belief weakly does." Posteriors are clamped to [0.01, 0.99] to
avoid degeneracy.

Feature flag
------------

This module is opt-in via ENTROLY_VAULT_COUPLING=1 (see proxy.py wiring).
Default off so existing users see no behavior change until the coupling is
field-tested.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MAX_AGE_DAYS = 30.0
DEFAULT_TOP_K = 20
DEFAULT_ALPHA_GROUNDED = 0.85
DEFAULT_BETA_UNGROUNDED = 0.40
PINNED_CONFIDENCE_THRESHOLD = 0.85

_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class ProjectedBelief:
    """A vault belief lifted into proxy fragment space."""
    claim_id: str
    entity: str
    body: str
    confidence: float
    recency: float
    relevance: float
    sources: tuple[str, ...]

    @property
    def score(self) -> float:
        """Combined projection score (used for top-k selection only —
        the engine re-scores once ingested)."""
        return self.confidence * self.recency * self.relevance

    @property
    def is_pinned(self) -> bool:
        return self.confidence >= PINNED_CONFIDENCE_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════
# Operation 1 — Project beliefs into the engine (System 2 → System 1)
# ═══════════════════════════════════════════════════════════════════════

def project_beliefs(
    vault: Any,
    query: str,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[ProjectedBelief]:
    """Select top-k verified vault beliefs relevant to the query.

    Filtering: c_b ≥ min_confidence ∧ σ_b ∉ {stale, retracted} ∧ age ≤ max_age_days
    Ranking:   confidence × recency × Jaccard(query_tokens, body_tokens)

    Returns at most top_k beliefs sorted by score descending. The relevance
    score here is intentionally cheap (Jaccard) — it's a pre-filter; the
    engine's BM25/entropy does the real scoring after ingestion.
    """
    try:
        index = vault.list_beliefs()
    except Exception as e:
        logger.debug("Vault list_beliefs failed: %s", e)
        return []

    if not index:
        return []

    now_ts = time.time()
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    candidates: list[ProjectedBelief] = []
    for entry in index:
        confidence = float(entry.get("confidence", 0.0))
        status = str(entry.get("status", "")).lower()

        if confidence < min_confidence:
            continue
        if status in ("stale", "retracted", "refuted"):
            continue

        recency = _compute_recency(
            entry.get("last_checked", ""), now_ts, max_age_days
        )
        if recency <= 0.0:
            continue

        # Lazy-load body only if it passes the cheap filters
        belief = _read_belief(vault, str(entry.get("entity", "")))
        if not belief:
            continue
        body = str(belief.get("body", "")).strip()
        if not body:
            continue

        relevance = _jaccard(query_tokens, _tokenize(body))
        if relevance == 0.0:
            continue

        fm = belief.get("frontmatter") or {}
        sources = fm.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        claim_id = str(fm.get("claim_id") or entry.get("entity", ""))

        candidates.append(ProjectedBelief(
            claim_id=claim_id,
            entity=str(entry.get("entity", "")),
            body=body,
            confidence=confidence,
            recency=recency,
            relevance=relevance,
            sources=tuple(str(s) for s in sources),
        ))

    candidates.sort(key=lambda p: p.score, reverse=True)
    return candidates[:top_k]


def inject_vault_beliefs(
    engine: Any,
    vault: Any,
    query: str,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    """Ingest top-k verified vault beliefs as engine fragments.

    Returns the list of claim_ids that were injected. The proxy should
    store this list per-request so it can later call attribute_outcome()
    with the same claim_ids when the outcome is observed.

    Fragments are ingested with:
        source = f"vault://beliefs/{entity}#{claim_id_short}"
        is_pinned = (confidence ≥ PINNED_CONFIDENCE_THRESHOLD)

    The pinned flag tells the engine these are critical-criticality
    fragments that should survive eviction under budget pressure.
    """
    beliefs = project_beliefs(
        vault, query,
        min_confidence=min_confidence,
        max_age_days=max_age_days,
        top_k=top_k,
    )
    if not beliefs:
        return []

    injected: list[str] = []
    for b in beliefs:
        # Approximate token count: 1 token ≈ 4 chars (engine recomputes precisely)
        token_count = max(1, len(b.body) // 4)
        source = f"vault://beliefs/{b.entity}#{b.claim_id[:8]}"
        try:
            # Engine API: remember_fragment(content, source, token_count, is_pinned)
            # — wrapper around Rust ingest().
            if hasattr(engine, "remember_fragment"):
                engine.remember_fragment(
                    content=b.body,
                    source=source,
                    token_count=token_count,
                    is_pinned=b.is_pinned,
                )
            elif hasattr(engine, "ingest"):
                engine.ingest(b.body, source, token_count, b.is_pinned)
            else:
                logger.warning("Engine has no fragment ingestion method; skipping coupling")
                return []
            injected.append(b.claim_id)
        except Exception as e:
            logger.debug("Vault belief ingestion failed for %s: %s", b.entity, e)
            continue

    if injected:
        logger.info(
            "Vault coupling: injected %d beliefs (top score=%.3f, threshold=%.2f)",
            len(injected),
            beliefs[0].score if beliefs else 0.0,
            min_confidence,
        )

    return injected


# ═══════════════════════════════════════════════════════════════════════
# Operation 2 — Attribute outcome to beliefs (System 1 → System 2)
# ═══════════════════════════════════════════════════════════════════════

def attribute_outcome(
    used_claim_ids: list[str],
    outcome_success: bool,
    vault: Any,
    *,
    alpha_grounded: float = DEFAULT_ALPHA_GROUNDED,
    beta_ungrounded: float = DEFAULT_BETA_UNGROUNDED,
) -> list[dict[str, Any]]:
    """Bayesian update of belief confidences given observed outcome.

    For each claim_id used in the proxy's context, locates the belief,
    applies the posterior update, and writes the new confidence +
    last_checked back to the belief's frontmatter.

    Math:
      L_g = α     if success else (1 - α)        # P(o | b grounded)
      L_u = β     if success else (1 - β)        # P(o | b not grounded)
      c'  = (c · L_g) / (c · L_g + (1 - c) · L_u)
      c'  ∈ [0.01, 0.99]   (clamped)

    Returns a list of update dicts (one per belief) for logging.
    """
    if not used_claim_ids:
        return []

    L_g = alpha_grounded if outcome_success else (1.0 - alpha_grounded)
    L_u = beta_ungrounded if outcome_success else (1.0 - beta_ungrounded)

    updates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cid in used_claim_ids:
        if cid in seen:
            continue
        seen.add(cid)
        try:
            entry, fm, path = _find_belief(vault, cid)
            if entry is None:
                continue
            c_prior = float(fm.get("confidence", entry.get("confidence", 0.5)))
            denom = c_prior * L_g + (1.0 - c_prior) * L_u
            if denom <= 0:
                continue
            c_post = (c_prior * L_g) / denom
            c_post = max(0.01, min(0.99, c_post))

            _rewrite_belief_field(
                path,
                confidence=c_post,
                last_checked=datetime.now(timezone.utc).isoformat(),
            )
            updates.append({
                "claim_id": cid,
                "entity": entry.get("entity"),
                "confidence_before": round(c_prior, 4),
                "confidence_after": round(c_post, 4),
                "outcome": "success" if outcome_success else "failure",
            })
        except Exception as e:
            logger.debug("Bayesian update failed for %s: %s", cid, e)
            continue

    return updates


# ═══════════════════════════════════════════════════════════════════════
# Operation 3 — Enqueue reverification (System 1 → System 2 escalation)
# ═══════════════════════════════════════════════════════════════════════

def enqueue_reverification(
    used_claim_ids: list[str],
    vault: Any,
) -> int:
    """Mark beliefs as needing re-verification.

    Sets status to "stale" for any belief currently in {verified, inferred}.
    FlowOrchestrator's verify-before-answer / change-driven flows pick up
    stale beliefs on their next pass.

    Returns count of beliefs marked.
    """
    if not used_claim_ids:
        return 0

    marked = 0
    seen: set[str] = set()
    for cid in used_claim_ids:
        if cid in seen:
            continue
        seen.add(cid)
        try:
            entry, _, path = _find_belief(vault, cid)
            if entry is None:
                continue
            if str(entry.get("status", "")).lower() in ("verified", "inferred"):
                _rewrite_belief_field(path, status="stale")
                marked += 1
        except Exception as e:
            logger.debug("Reverify enqueue failed for %s: %s", cid, e)
            continue

    if marked:
        logger.info("Vault coupling: marked %d beliefs stale for reverification", marked)
    return marked


# ── Helpers ──────────────────────────────────────────────────────────

def is_enabled() -> bool:
    """Feature gate: ENTROLY_VAULT_COUPLING=1 to enable in proxy."""
    return os.environ.get("ENTROLY_VAULT_COUPLING", "0") == "1"


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens of length ≥ 3. Cheap, ASCII-friendly."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _compute_recency(last_checked_iso: str, now_ts: float, max_age_days: float) -> float:
    """Exponential decay over age. Returns in [0, 1].

    No timestamp → 0.5 (neutral; can't tell freshness).
    Future timestamp → 1.0.
    Beyond max_age → 0.0.
    """
    if not last_checked_iso:
        return 0.5
    try:
        s = last_checked_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (now_ts - dt.timestamp()) / 86400.0
        if age_days < 0:
            return 1.0
        if age_days > max_age_days:
            return 0.0
        return math.exp(-age_days / max_age_days)
    except Exception:
        return 0.5


def _read_belief(vault: Any, entity: str) -> dict[str, Any] | None:
    """Call vault.read_belief defensively."""
    if not entity:
        return None
    try:
        return vault.read_belief(entity)
    except Exception:
        return None


def _find_belief(vault: Any, claim_id_or_entity: str):
    """Locate a belief by claim_id (preferred) or entity (fallback).

    Returns (index_entry, frontmatter, path) tuple. All None if not found.
    """
    if not claim_id_or_entity:
        return None, None, None

    try:
        index = vault.list_beliefs()
    except Exception:
        return None, None, None

    # First pass: try entity match (fast, no body load)
    target_entity = None
    for entry in index:
        if entry.get("entity") == claim_id_or_entity:
            target_entity = claim_id_or_entity
            break

    # Second pass: claim_id match (requires loading frontmatters)
    if target_entity is None:
        for entry in index:
            belief = _read_belief(vault, str(entry.get("entity", "")))
            if not belief:
                continue
            fm = belief.get("frontmatter") or {}
            if fm.get("claim_id") == claim_id_or_entity:
                target_entity = str(entry.get("entity"))
                break

    if target_entity is None:
        return None, None, None

    # Load the full belief once we know which one to fetch
    belief = _read_belief(vault, target_entity)
    if not belief:
        return None, None, None

    # Re-fetch the index entry to return alongside
    entry = next(
        (e for e in index if e.get("entity") == target_entity),
        None,
    )
    return entry, belief.get("frontmatter") or {}, belief.get("path", "")


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _rewrite_belief_field(path: str, **fields: Any) -> None:
    """Update specific frontmatter fields in a belief file, preserving body
    and other frontmatter fields. Idempotent."""
    if not path or not fields:
        return
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return

    text = p.read_text(encoding="utf-8", errors="replace")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        # No frontmatter — prepend one (rare; vault always writes it)
        fm_lines = [f"{k}: {_fmt_field(v)}" for k, v in fields.items()]
        p.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + text, encoding="utf-8")
        return

    fm_text = m.group(1)
    body = text[m.end():]

    lines = fm_text.split("\n")
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        if ":" in line and not line.lstrip().startswith("-"):
            key = line.split(":", 1)[0].strip()
            if key in fields:
                out_lines.append(f"{key}: {_fmt_field(fields[key])}")
                seen.add(key)
                continue
        out_lines.append(line)

    for key, value in fields.items():
        if key not in seen:
            out_lines.append(f"{key}: {_fmt_field(value)}")

    new_text = "---\n" + "\n".join(out_lines) + "\n---\n" + body
    p.write_text(new_text, encoding="utf-8")


def _fmt_field(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
