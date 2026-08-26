"""The Entroly context engine, separated from the MCP server that hosts it.

`server.py` held two unrelated things in one 6,748-line module: this class, and
`create_mcp_server`, a 3,509-line function containing 76 nested tool handlers.
Both defined a function named `optimize_context` -- a 419-line engine method and
a 540-line MCP handler -- which is how the no-match contract came to live on the
handler while every non-MCP caller (CLI, SDK, proxy) ran without it.

This is a relocation, not a redesign: no behaviour changed, and `server.py`
re-exports every moved name, so existing imports keep working -- including the
tests that import `_evidence_backed`, `_honest_tokens_saved` and
`apply_no_match_contract` from `entroly.server`.

The composition problem this exposes is not fixed here. `EntrolyEngine` still
leaves `set_fast_path_router`, `set_journal_callback` and
`set_crystallization_callback` for the caller to wire, and only
`create_mcp_server` does it, so a CLI or SDK engine is still assembled
differently from an MCP one.
"""

from __future__ import annotations
import copy
import gzip
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import threading
import uuid
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable
from .adaptive_pruner import EntrolyPruner, FragmentGuard
from .autotune import ComponentFeedbackBus, DreamingLoop, FeedbackJournal, TaskProfileOptimizer
from .online_learner import OnlinePrism, compute_implicit_reward, compute_contributions
from .path_safety import resolve_dir_within, resolve_file_within, resolve_output_within
from .cache_aligner import CacheAligner
from .belief_compiler import BeliefCompiler
from .change_listener import WorkspaceChangeListener
from .change_pipeline import ChangePipeline
from .checkpoint import (
    CheckpointManager,
    ContextFragment,
    _dict_to_fragment,
    _fragment_to_dict,
)
from .config import EntrolyConfig, load_active_tuning_config, resolve_tuning_kwargs
from .epistemic_router import (
    EpistemicRouter,
)
from .evolution_daemon import EvolutionDaemon
from .evolution_logger import EvolutionLogger
from .flow_orchestrator import FlowOrchestrator
from .multimodal import ingest_diagram as _mm_diagram
from .multimodal import ingest_diff as _mm_diff
from .multimodal import ingest_voice as _mm_voice
from .prefetch import PrefetchEngine
from .provenance import build_provenance, compact_optimize_result_for_wire
from .proxy_transform import calibrated_token_count as _calibrated_token_count
from .query_refiner import QueryRefiner
from .read_delivery_cache import ReadDeliveryCache
from .repo_map import build_repo_map, render_repo_map_markdown
from .skill_engine import SkillEngine, promoted_skill_execution_enabled
from .value_tracker import ValueTracker, get_tracker
from .vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
)
from .verification_engine import VerificationEngine


try:
    from .native_status import usable_core as _usable_core

    _core = _usable_core()
    if _core is None:
        raise ImportError("entroly_core unavailable or below minimum version")
    RustEngine = _core.EntrolyEngine
    py_analyze_query = _core.py_analyze_query
    py_refine_heuristic = _core.py_refine_heuristic
    _RUST_AVAILABLE = True
except (ImportError, AttributeError):
    _RUST_AVAILABLE = False
    RustEngine = None  # type: ignore[assignment,misc]

    # Provide Python stubs for the Rust query helpers
    def py_analyze_query(query: str) -> dict:  # type: ignore[misc]
        """Pure-Python stub when entroly_core is not available."""
        terms = [w for w in query.lower().split() if len(w) > 2][:8]
        return {
            "vagueness_score": 0.5,
            "key_terms": terms,
            "needs_refinement": len(terms) < 3,
            "reason": "python_fallback",
        }

    def py_refine_heuristic(query: str, context: str) -> str:  # type: ignore[misc]
        """Pure-Python stub — returns query unchanged."""
        return query


def _py_simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint from text.

    Uses trigrams (or unigrams for short text) hashed with MD5
    to build a locality-sensitive fingerprint.
    """
    import hashlib as _hl
    words = [w for w in text.lower().split() if w.isalnum() or any(c.isalnum() for c in w)]
    if not words:
        return 0

    # Build features: trigrams if >= 3 words, else unigrams
    features: list[str] = []
    if len(words) >= 3:
        for i in range(len(words) - 2):
            features.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    else:
        features = words

    bit_sums = [0] * 64
    for feat in features:
        h = int(_hl.md5(
            feat.encode("utf-8", errors="replace"),
            usedforsecurity=False,
        ).hexdigest(), 16)
        for i in range(64):
            if (h >> i) & 1:
                bit_sums[i] += 1
            else:
                bit_sums[i] -= 1

    fingerprint = 0
    for i in range(64):
        if bit_sums[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _py_hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit fingerprints."""
    return bin(a ^ b).count("1")


class _PyDedupIndex:
    """Pure-Python SimHash-based deduplication index (LSH banding)."""

    def __init__(self, hamming_threshold: int = 3):
        self._threshold = hamming_threshold
        self._fingerprints: dict[str, int] = {}
        self._bands: list[dict[int, list[str]]] = [{} for _ in range(4)]
        self._duplicates_detected = 0

    def insert(self, fragment_id: str, text: str) -> str | None:
        """Insert a fragment. Returns duplicate_id if near-duplicate found."""
        fp = _py_simhash(text)

        # Check bands for candidate matches
        for band_idx in range(4):
            band_hash = (fp >> (band_idx * 16)) & 0xFFFF
            candidates = self._bands[band_idx].get(band_hash, [])
            for cand_id in candidates:
                cand_fp = self._fingerprints.get(cand_id)
                if cand_fp is not None and _py_hamming_distance(fp, cand_fp) <= self._threshold:
                    self._duplicates_detected += 1
                    return cand_id

        # No duplicate — register this fragment
        self._fingerprints[fragment_id] = fp
        for band_idx in range(4):
            band_hash = (fp >> (band_idx * 16)) & 0xFFFF
            self._bands[band_idx].setdefault(band_hash, []).append(fragment_id)
        return None

    def remove(self, fragment_id: str) -> None:
        """Remove a fragment from the index."""
        fp = self._fingerprints.pop(fragment_id, None)
        if fp is None:
            return
        for band_idx in range(4):
            band_hash = (fp >> (band_idx * 16)) & 0xFFFF
            bucket = self._bands[band_idx].get(band_hash, [])
            if fragment_id in bucket:
                bucket.remove(fragment_id)

    def stats(self) -> dict:
        return {
            "total_fingerprints": len(self._fingerprints),
            "duplicates_detected": self._duplicates_detected,
        }


def _py_compute_information_score(
    text: str,
    global_token_counts: dict[str, int],
    total_tokens: int,
    other_fragments: list[str],
) -> float:
    """Compute information density score [0, 1] using Shannon entropy + redundancy."""
    import math as _m
    if not text:
        return 0.0

    # 1. Normalized Shannon entropy on bytes (40% weight)
    byte_counts: dict[int, int] = {}
    encoded = text.encode("utf-8", errors="replace")
    for b in encoded:
        byte_counts[b] = byte_counts.get(b, 0) + 1
    total_bytes = len(encoded)
    entropy = 0.0
    for count in byte_counts.values():
        p = count / total_bytes
        if p > 0:
            entropy -= p * _m.log2(p)
    normalized_entropy = min(entropy / 6.0, 1.0)

    # 2. Boilerplate penalty (30% weight)
    lines = text.strip().splitlines()
    boilerplate_lines = 0
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith(("import ", "from ")) or
                stripped in ("pass", "...", "}", ")", "]", '"""') or
                stripped.startswith("def __") or
                stripped in ("return None", "return self", "return True", "return False")):
            boilerplate_lines += 1
    bp_ratio = boilerplate_lines / max(len(lines), 1)
    bp_penalty = 1.0 - bp_ratio

    # 3. Cross-fragment redundancy (30% weight)
    uniqueness = 1.0
    if other_fragments:
        words = text.lower().split()
        if len(words) >= 3:
            text_trigrams = set()
            for i in range(len(words) - 2):
                text_trigrams.add(f"{words[i]} {words[i+1]} {words[i+2]}")
            if text_trigrams:
                other_trigrams: set = set()
                for other in other_fragments:
                    ow = other.lower().split()
                    for i in range(len(ow) - 2):
                        other_trigrams.add(f"{ow[i]} {ow[i+1]} {ow[i+2]}")
                overlap = len(text_trigrams & other_trigrams)
                redundancy = overlap / len(text_trigrams)
                uniqueness = 1.0 - min(redundancy, 1.0)

    score = 0.40 * normalized_entropy + 0.30 * bp_penalty + 0.30 * uniqueness
    return max(0.0, min(1.0, score))


def _py_compute_relevance(
    frag: ContextFragment,
    w_recency: float,
    w_frequency: float,
    w_semantic: float,
    w_entropy: float,
    feedback_multiplier: float = 1.0,
) -> float:
    """Compute weighted relevance score with softcap [0, 1]."""
    import math as _m
    total_weight = w_recency + w_frequency + w_semantic + w_entropy
    if total_weight == 0:
        return 0.0
    base = (
        w_recency * frag.recency_score
        + w_frequency * frag.frequency_score
        + w_semantic * frag.semantic_score
        + w_entropy * frag.entropy_score
    ) / total_weight
    raw = base * feedback_multiplier
    # Gemini-style logit softcap at 0.85
    cap = 0.85
    return cap * _m.tanh(raw / cap)


def _py_knapsack_optimize(
    fragments: list,
    token_budget: int,
    w_recency: float,
    w_frequency: float,
    w_semantic: float,
    w_entropy: float,
) -> tuple:
    """Pure-Python greedy knapsack optimizer.

    Returns (selected_fragments, stats_dict).
    """
    # Separate pinned fragments (always included)
    pinned = [f for f in fragments if f.is_pinned]
    candidates = [f for f in fragments if not f.is_pinned]

    pinned_tokens = sum(f.token_count for f in pinned)
    max(0, token_budget - pinned_tokens)

    # Score and sort candidates by relevance/token ratio (greedy)
    scored = []
    for frag in candidates:
        rel = _py_compute_relevance(frag, w_recency, w_frequency, w_semantic, w_entropy)
        if frag.token_count > 0:
            efficiency = rel / frag.token_count
        else:
            efficiency = rel
        scored.append((frag, rel, efficiency))
    scored.sort(key=lambda x: x[2], reverse=True)

    pinned_relevance = sum(
        _py_compute_relevance(f, w_recency, w_frequency, w_semantic, w_entropy) for f in pinned
    )

    # Density-greedy pass.
    selected = list(pinned)
    used_tokens = pinned_tokens
    total_relevance = pinned_relevance
    for frag, rel, _ in scored:
        if used_tokens + frag.token_count <= token_budget:
            selected.append(frag)
            used_tokens += frag.token_count
            total_relevance += rel

    # Khuller–Moss–Naor (1999) singleton champion. Pure density-greedy
    # has NO constant-factor guarantee: a low-value high-density item
    # can block a high-value budget-filling item (measured by
    # tests/test_compression_contract.py against brute-force OPT —
    # greedy hit 30% of optimal on the adversarial fixtures). Take the
    # better of {density-greedy, pinned ∪ best single feasible item}.
    #
    # What that provably buys, precisely: this objective is MODULAR
    # (_py_compute_relevance scores each fragment independently, so
    # value is a plain sum with no co-selection term). For a modular
    # objective under a knapsack, better-of-{density-greedy, best
    # singleton} is the classic Dantzig/LP-rounding ½. It is NOT
    # (1 − 1/e): that needs a monotone SUBMODULAR objective, and under a
    # knapsack rather than a cardinality constraint it further needs
    # Sviridenko (2004) partial enumeration. KMN's own better-of-two
    # result is ½(1 − 1/e) ≈ 0.32, for submodular coverage.
    # The Rust 0/1-DP hot path is exact for this modular objective; this
    # keeps the degraded fallback within ½ of that.
    cand_budget = token_budget - pinned_tokens
    best_single = None
    best_single_rel = 0.0
    for frag, rel, _ in scored:
        if frag.token_count <= cand_budget and rel > best_single_rel:
            best_single, best_single_rel = frag, rel
    method = "greedy_python"
    if best_single is not None and pinned_relevance + best_single_rel > total_relevance:
        selected = list(pinned) + [best_single]
        used_tokens = pinned_tokens + best_single.token_count
        total_relevance = pinned_relevance + best_single_rel
        method = "greedy_python+kmn_singleton"

    # If every candidate is larger than the budget, returning an empty
    # selection makes downstream CLI/proxy metrics claim "100% savings" while
    # giving the model no context. The Python fallback must degrade like a
    # compressor, not a filter: keep a budget-sized excerpt of the best
    # relevance candidate so first-run installs without the Rust wheel still
    # produce useful context.
    if not selected and candidates and token_budget > 0:
        from dataclasses import replace as _dc_replace

        best_frag, best_rel, _ = max(scored, key=lambda x: x[1])
        content = getattr(best_frag, "content", None)
        if isinstance(content, str):
            excerpt_tokens = max(1, min(int(token_budget), int(best_frag.token_count or token_budget)))
            excerpt_chars = max(256, excerpt_tokens * 4)
            selected = [
                _dc_replace(
                    best_frag,
                    fragment_id=f"{best_frag.fragment_id}:excerpt",
                    content=content[:excerpt_chars],
                    token_count=excerpt_tokens,
                )
            ]
            used_tokens = excerpt_tokens
            total_relevance = best_rel
            method = "greedy_python+oversize_excerpt"

    # Present by relevance, pack by density.
    #
    # The greedy loop sorts candidates by value-per-token, which is the correct
    # criterion for filling a budget -- but that ordering then leaked out as the
    # order the caller displays. Measured on a two-file project, for the query
    # "who verifies the password":
    #
    #     auth.py     24 tokens  relevance 0.58339  density 0.024308
    #     billing.py  23 tokens  relevance 0.57407  density 0.024960
    #
    # auth.py is the better answer and loses the top slot purely for being one
    # token longer, which is why `entroly simulate` reported "top: billing.py"
    # for an authentication question.
    #
    # Packing and presentation are different jobs. WHICH fragments are selected
    # is unchanged -- only the order they are handed back in. Pinned fragments
    # keep their position at the front.
    pinned_ids = {id(f) for f in pinned}
    head = [f for f in selected if id(f) in pinned_ids]
    tail = [f for f in selected if id(f) not in pinned_ids]
    tail.sort(
        key=lambda f: (
            -_py_compute_relevance(f, w_recency, w_frequency, w_semantic, w_entropy),
            # Deterministic tie-break so prompt prefixes stay byte-stable.
            # getattr, not f.source: `source` is not part of the minimal
            # fragment protocol this function accepts. Reading it directly
            # crashed the pure-Python fallback -- the path that runs on a
            # default pip install without the Rust wheel -- for any fragment
            # lacking it (caught by tests/test_compression_contract.py, whose
            # Frag declares __slots__ without `source`).
            getattr(f, "source", "") or "",
        )
    )
    selected = head + tail

    stats = {
        "total_tokens": used_tokens,
        "total_relevance": round(total_relevance, 4),
        "method": method,
        "pinned_count": len(pinned),
        "candidate_count": len(candidates),
    }
    return selected, stats


def _py_apply_ebbinghaus_decay(
    fragments: list,
    current_turn: int,
    half_life: int,
) -> None:
    """Apply Ebbinghaus forgetting curve to fragment recency scores (in-place)."""
    import math as _m
    if half_life <= 0:
        return
    decay_rate = _m.log(2) / half_life
    for frag in fragments:
        dt = max(0, current_turn - frag.turn_last_accessed)
        frag.recency_score = _m.exp(-decay_rate * dt)


logger = logging.getLogger("entroly")


class _WilsonFeedbackTracker:
    """
    Python-exact port of the Rust FeedbackTracker in guardrails.rs.

    Uses the Wilson score lower bound (the same formula Reddit uses for ranking)
    to produce a calibrated relevance multiplier for each fragment:

        p̂  = successes / (successes + failures)
        z  = 1.96   # 95% confidence interval
        lb = Wilson lower bound
        multiplier = 0.5 + lb × 1.5   → maps [0, 1] → [0.5, 2.0]

    > 1.0  fragment historically useful   (boosted)
    = 1.0  no data yet                    (neutral)
    < 1.0  fragment historically unhelpful (suppressed)

    Numerically identical to the Rust implementation — the Python fallback
    now has the same feedback quality as the Rust engine.
    """
    import math as _math

    _Z = 1.96  # 95% CI

    def __init__(self):
        self._success: dict[str, int] = {}
        self._failure: dict[str, int] = {}

    def record_success(self, fragment_ids: list[str]) -> None:
        for fid in fragment_ids:
            self._success[fid] = self._success.get(fid, 0) + 1

    def record_failure(self, fragment_ids: list[str]) -> None:
        for fid in fragment_ids:
            self._failure[fid] = self._failure.get(fid, 0) + 1

    def learned_value(self, fragment_id: str) -> float:
        import math
        s = self._success.get(fragment_id, 0)
        f = self._failure.get(fragment_id, 0)
        total = s + f
        if total == 0:
            return 1.0
        p = s / total
        z = self._Z
        denominator = 1.0 + z * z / total
        center = p + z * z / (2.0 * total)
        spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        lower_bound = (center - spread) / denominator
        raw = 0.5 + lower_bound * 1.5
        return max(0.5, min(2.0, raw))  # Clamp to documented [0.5, 2.0] range

    def remove_fragments(self, fragment_ids: set[str]) -> None:
        """Drop feedback for fragment identities that no longer exist."""
        for fragment_id in fragment_ids:
            self._success.pop(fragment_id, None)
            self._failure.pop(fragment_id, None)


def _build_rust_engine(config: EntrolyConfig):
    """Construct the Rust engine from an EntrolyConfig, threading every
    autotuned dimension the engine consumes (weights, decay, dedup Hamming,
    knapsack exploration, IOS resolution factors).

    Falls back to the core six parameters if the installed ``entroly_core``
    predates the extended constructor kwargs — so an older native wheel never
    breaks engine construction.
    """
    base = dict(
        w_recency=config.weight_recency,
        w_frequency=config.weight_frequency,
        w_semantic=config.weight_semantic_sim,
        w_entropy=config.weight_entropy,
        decay_half_life=config.decay_half_life_turns,
        min_relevance=config.min_relevance_threshold,
    )
    extra = dict(
        hamming_threshold=getattr(config, "dedup_hamming_threshold", 3),
        exploration_rate=getattr(config, "exploration_rate", 0.0),
        ios_skeleton_info_factor=getattr(config, "ios_skeleton_info_factor", 0.70),
        ios_reference_info_factor=getattr(config, "ios_reference_info_factor", 0.15),
        ios_diversity_floor=getattr(config, "ios_diversity_floor", 0.10),
    )
    try:
        return RustEngine(**base, **extra)
    except TypeError:
        # Older entroly-core without the extended kwargs — core six still work.
        return RustEngine(**base)


_EVIDENCE_WORD = re.compile(r"[A-Za-z0-9_]{3,}")


def _query_terms(query: str) -> set[str]:
    """Content-bearing query terms, lowercased. Stopwords carry no evidence."""
    stop = {
        "the", "and", "for", "are", "was", "were", "does", "did", "how", "why",
        "what", "where", "when", "which", "who", "into", "from", "with", "this",
        "that", "these", "those", "you", "your", "our", "not", "but", "can",
        "will", "would", "should", "could", "has", "have", "had", "been", "its",
    }
    return {t.lower() for t in _EVIDENCE_WORD.findall(query or "")} - stop


def _score_distribution_is_degenerate(selected: list) -> bool:
    """True when the ranker produced no discrimination between candidates.

    A ranker that found real evidence separates candidates; one that found none
    cannot. Measured on this repository (955 fragments):

        real query  -> 0.8534, 0.7220, 0.7095, 0.6971   (spread 0.156)
        no-match    -> 0.0800, 0.0800, 0.0800, 0.0800   (spread 0.000)

    So the discriminator is variance, not magnitude — which is why thresholding
    the score itself never worked: 0.08 and 0.62 are both "positive", but only
    one of them is a ranking. A flat distribution means every candidate is
    equally (un)related, i.e. the ordering carries no information.

    Requires >= 2 non-pinned candidates; a single result cannot be flat or
    spread, so it is judged by the lexical check instead.
    """
    scores = []
    for frag in selected or []:
        if not isinstance(frag, dict) or frag.get("is_pinned"):
            continue
        for key in ("relevance", "relevance_score", "score"):
            if key in frag:
                try:
                    scores.append(float(frag[key] or 0.0))
                except (TypeError, ValueError):
                    pass
                break
    if len(scores) < 2:
        return False
    spread = max(scores) - min(scores)
    # Exact-equality would be too brittle for float arithmetic; this tolerance
    # is far below the separation any real query produces.
    return spread < 1e-9


def _selection_matches_query(query: str, selected: list) -> bool:
    """Does the returned context actually contain any query term?

    Relevance scores cannot answer this: the native ranker floors and normalizes,
    so a query matching nothing still comes back at ~0.6. Verified on this
    repository — "zzqqxx blorptastic wubbleflux" scored 0.5969. Checking the
    delivered text is the one signal the scoring pipeline cannot fabricate, and
    it is what "evidence-backed" has to mean.
    """
    terms = _query_terms(query)
    if not terms:
        return True          # no lexical intent to satisfy; not a no-match

    # Token membership with stemming, not raw substring containment.
    #
    # Substring matching fails on ordinary English morphology, and the failure
    # is silent and one-directional: it reports "no term in common" for a
    # selection that plainly answers the question. Measured on a two-file
    # corpus, the query "How are credit cards charged?" against
    # `charge_card(customer.card, amount)` matched nothing -- "cards" is not a
    # substring of "card" and "charged" is not a substring of "charge" -- so a
    # correct selection was discarded as a no-match. Natural-language queries
    # are the normal case, and plurals and past tense are not exotic.
    #
    # `sufficiency._lexical_terms` is the tokenizer this repository already
    # standardised on for exactly this question; its own `attainable()` says
    # "use token membership, never unsafe raw-substring matching". One
    # definition of "a term appears here" is worth more than two.
    try:
        from .sufficiency import _lexical_terms, stem
    except Exception:  # pragma: no cover - sufficiency is always present
        _lexical_terms = None

    if _lexical_terms is None:
        for frag in selected or []:
            if not isinstance(frag, dict) or frag.get("is_pinned"):
                continue
            haystack = (
                str(frag.get("content") or "") + " " + str(frag.get("source") or "")
            ).lower()
            if any(t in haystack for t in terms):
                return True
        return False

    # Both sides through the *same* tokenizer, or the comparison is meaningless.
    # `_query_terms` keeps `parse_manifest` whole while `_lexical_terms` splits
    # the haystack into `parse`/`manifest`, so passing raw query words here made
    # every snake_case identifier -- the most precise thing a developer can
    # type -- fail to match the code that defines it. Substring matching used to
    # paper over this; token membership cannot, so the query gets tokenised too.
    wanted = {t for t in terms}
    wanted |= {s for s in (stem(t) for t in terms) if s}
    for term in terms:
        wanted |= _lexical_terms(term)
    for frag in selected or []:
        if not isinstance(frag, dict) or frag.get("is_pinned"):
            continue
        # Per fragment rather than over the whole selection: a match on the
        # first fragment costs one tokenisation, not all of them.
        present = _lexical_terms(
            str(frag.get("content") or "") + " " + str(frag.get("source") or "")
        )
        if wanted & present:
            return True
    return False


def apply_no_match_contract(
    result: dict,
    query: str,
    *,
    scores_are_ranked: bool = True,
) -> dict:
    """Refuse to pass off unrelated context as a match. Mutates and returns.

    A query matching nothing still yields a ranked list, so the tool used to
    hand back confident-looking unrelated files and bill the omitted corpus as
    "savings". Say so explicitly instead: keep pinned/required evidence
    (operator policy, not relevance), drop the rest, and credit zero savings.
    Withholding context because nothing matched is not a saving, and unrelated
    context is worse than none.

    This lives outside ``optimize_context`` because that is not the only place
    selection happens. ``cli.cmd_optimize`` deliberately bypasses
    ``optimize_context`` on its default path -- it re-selects over the full
    fragment store rather than through ``recall()``, which caps at ~25 items --
    so for a long time the guard ran only on the engine-less fallback branch,
    where ranking does not happen anyway. It was present exactly where it was
    useless and absent exactly where it mattered. One implementation, applied by
    every selection surface, is the fix; do not re-inline it.

    ``scores_are_ranked=False`` disables the variance discriminator for callers
    whose fragments carry a flat relevance by construction. QCCR is one:
    ``_rust_select`` returns synthetic per-file fragments scored 1.0 for a
    matching query and 0.0 for a non-matching one -- measured, not assumed --
    so every QCCR selection looks "degenerate" to a spread test while
    ``_evidence_backed`` and ``_selection_matches_query`` still separate the two
    cases cleanly. Leaving the variance test on for that caller would wipe every
    successful selection; leaving the whole contract off, which is what happened
    before, lets a nonsense query return twelve confident files.
    """
    if not query:
        return result

    selected = result.get("selected_fragments") or result.get("selected") or []
    if not selected:
        return result

    # Nothing was excluded -> there was no selection decision to be wrong about.
    #
    # "Unrelated context is worse than none" is a claim about *displacing*
    # relevant evidence under a budget. When the optimizer returned every
    # candidate it had, nothing was displaced, so withholding is pure loss.
    # Measured: a two-file, 47-token repository against an 8,000-token budget
    # asked "How does the authentication flow work?" returned both files and
    # was then wiped to zero, because `stem("authentication")` is None and never
    # reaches the token `auth` -- the whole corpus discarded over a vocabulary
    # gap, with no budget pressure that discarding could possibly relieve.
    #
    # The guard still applies wherever the optimizer actually chose: a selection
    # of 23 fragments drawn from 140 candidates is a decision, and a wrong one
    # costs the user the budget it consumed.
    considered = result.get("total_fragments")
    try:
        if considered and len(selected) >= int(considered):
            return result
    except (TypeError, ValueError):
        pass

    degenerate = scores_are_ranked and _score_distribution_is_degenerate(selected)
    evidence = _evidence_signal(selected)
    lexical = _selection_matches_query(query, selected)

    # Unmeasured relevance must not vote. For a measured selection this is
    # exactly the old condition -- `evidence is False` is `not _evidence_backed`
    # and `D or not (E and M)` expands to `D or not E or not M` -- so nothing
    # changes for any caller that ranks. When no score exists at all, the
    # lexical check decides alone rather than a missing field convicting a
    # selection that was never scored. Strictly fewer trips, never more.
    if not (degenerate or evidence is False or not lexical):
        return result

    pinned = [f for f in selected if isinstance(f, dict) and f.get("is_pinned")]
    result["selected_fragments"] = pinned
    result["selected"] = pinned
    result["selected_count"] = len(pinned)
    result["tokens_used"] = sum(int(f.get("token_count", 0) or 0) for f in pinned)
    result["total_tokens"] = result["tokens_used"]
    result["tokens_saved"] = 0
    result["tokens_saved_this_call"] = 0
    considered_count = result.get("total_fragments", 0) or 0

    # Two different situations reach this point and they need different answers.
    #
    # Normally the ranker saw candidates and none of them matched -- rephrasing
    # is genuine advice. But when no candidate was offered at all, ranking never
    # happened, and telling the user to rephrase blames their query for a
    # capability that is absent. QCCR's candidate set comes from
    # `_rust.export_fragments()`, which has no pure-Python equivalent, so a
    # default install with no usable native engine produces zero candidates for
    # *every* query and would otherwise be told, each time, to word it better.
    no_candidates = considered_count == 0
    engine_missing = no_candidates and _usable_core_absent()

    if engine_missing:
        reason = (
            "no candidates were generated, because query-conditioned retrieval "
            "requires the native engine and it is not available; returning "
            "pinned evidence only rather than unrelated files"
        )
        remediation = (
            'install the native engine with `pip install -U "entroly[native]"` '
            "(or run `entroly repair`); no rewording of the query can produce a "
            "match while candidate generation is unavailable"
        )
    elif no_candidates:
        reason = (
            "no candidates were offered to the ranker, so nothing could match; "
            "returning pinned evidence only rather than unrelated files"
        )
        remediation = (
            "run `entroly health` to confirm the repository is indexed; the "
            "query was not the limiting factor here"
        )
    else:
        reason = (
            "the ranker produced no discrimination for this query "
            "(flat score distribution) or the returned text shared "
            "no term with it; returning pinned evidence only rather "
            "than unrelated files"
        )
        remediation = (
            "rephrase with identifiers from the codebase, or run "
            "`entroly health` to confirm the repository is indexed"
        )

    result["status"] = "no_match"
    result["no_match"] = {
        "reason": reason,
        "degenerate_ranking": degenerate,
        # Says which signal convicted. `false` here means the selection carried
        # no relevance at all and the lexical check decided on its own -- report
        # what was measured rather than implying a score was consulted.
        "evidence_measured": evidence is not None,
        "lexical_match": lexical,
        "query": query,
        "candidates_considered": considered_count,
        "pinned_retained": len(pinned),
        # Separates "your query found nothing" from "this build cannot search".
        # A consumer that only reads `reason` still gets the right story, but a
        # machine consumer should not have to parse prose to tell them apart.
        "candidate_generation_available": not engine_missing,
        "remediation": remediation,
    }
    return result


def _usable_core_absent() -> bool:
    """True when this process has no native engine it is allowed to call.

    Routed through `native_status.usable_core`, which is the single source of
    truth for that question -- a bare `import entroly_core` here would re-create
    the split it exists to prevent, where one module accepts a core the shared
    gate has already refused.
    """
    try:
        from .native_status import usable_core

        return usable_core() is None
    except Exception:  # pragma: no cover - defensive; absence is the safe answer
        return True


def _evidence_signal(selected: list) -> bool | None:
    """Three-valued relevance verdict: True, False, or **unmeasured**.

    ``True``  at least one non-pinned fragment carries a positive score.
    ``False`` scores are present on non-pinned fragments and none is positive.
    ``None``  no non-pinned fragment carries a score at all -- the question was
              never asked, so it has no answer.

    The third state is not pedantry. Relevance is computed *during* selection
    and is not stored on a fragment, so a selection replayed from somewhere
    other than a ranker arrives without it -- the fast path replays a
    crystallized recipe straight out of the fragment store, and those fragments
    carry ``id``/``source``/``content``/``token_count`` and no score. Collapsing
    "never scored" into "scored zero" would read every one of those as
    unrelated and discard a selection that was promoted precisely because it was
    measured to work.

    Absence of evidence is not evidence of absence, and this codebase already
    settled that argument once: the sufficiency certificate reports
    ``boundary_exposure_measured=False`` rather than claiming an exposure of
    zero it could not compute. Same discipline here.

    Pinned fragments are excluded throughout: they are included by operator
    policy, not by evidence, so they must never make a no-match look matched.
    """
    measured = False
    for frag in selected or []:
        if not isinstance(frag, dict) or frag.get("is_pinned"):
            continue
        for key in ("relevance", "relevance_score", "score"):
            if key in frag:
                try:
                    value = float(frag[key] or 0.0)
                except (TypeError, ValueError):
                    continue
                measured = True
                if value > 0.0:
                    return True
                break
    return False if measured else None


def _evidence_backed(selected: list) -> bool:
    """True when at least one non-pinned selection carries positive relevance.

    A query whose terms match nothing still produces a ranked list: score
    normalization floors and uniform fallbacks give every fragment a middling
    score, so the engine returns confident-looking but unrelated files. Measured
    on this repo: the nonsense query "zzqqxx blorptastic wubbleflux" selected 8
    files at ~0.6 relevance, sharing its top hits with a real query.

    Kept as the two-valued view for callers that only need "is there evidence".
    Unmeasured reads as False here, which is the safe answer to that question;
    callers that must distinguish "no evidence" from "no measurement" -- the
    no-match contract among them -- use `_evidence_signal` instead.
    """
    return _evidence_signal(selected) is True


def _honest_tokens_saved(selected: list, tokens_saved: int) -> int:
    """Savings only count when the selection is actually evidence-backed.

    `corpus - selected` credits every fragment that was never going to be sent,
    so it stayed ~constant regardless of query and was largest exactly when the
    engine found nothing. Withholding context because nothing matched is not a
    saving.
    """
    return max(0, int(tokens_saved or 0)) if _evidence_backed(selected) else 0


class EntrolyEngine:
    """
    Orchestrates all subsystems. Delegates math to Rust when available.

    Rust handles: ingest, optimize, recall, stats, feedback, dep graph, ordering.
    Python handles: prefetch, checkpoint, MCP protocol.
    """

    def __init__(self, config: EntrolyConfig | None = None):
        self.config = config or EntrolyConfig()
        self._use_rust = _RUST_AVAILABLE
        # Python-routed selectors (currently QCCR) bypass the Rust optimizer,
        # so their session savings need engine-independent accounting.
        self._total_tokens_saved: int = 0

        if self._use_rust:
            self._rust = _build_rust_engine(self.config)
            logger.info("Using Rust engine (entroly_core)")
        else:
            # Python fallback state
            self._fragments: dict[str, Any] = {}
            self._current_turn: int = 0
            from collections import Counter
            self._global_token_counts: Counter = Counter()
            self._total_token_count: int = 0
            self._dedup = _PyDedupIndex(
                hamming_threshold=getattr(self.config, "dedup_hamming_threshold", 3)
            )
            self._total_optimizations: int = 0
            self._total_fragments_ingested: int = 0
            self._total_duplicates_caught: int = 0
            # Wilson-score feedback tracker — numerically identical to Rust FeedbackTracker
            self._wilson = _WilsonFeedbackTracker()
            logger.info("Using Python fallback engine (entroly_core not installed)")

        # Python-only subsystems
        self._prefetch = PrefetchEngine(co_access_window=5)
        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            auto_interval=self.config.auto_checkpoint_interval,
        )
        # Query refinement: vague queries are expanded using in-memory file
        # context before context selection, reducing hallucination from wrong files.
        self._refiner = QueryRefiner()

        # CodeQualityGuard: scans ingested fragments for secrets/TODO/unsafe
        self._guard = FragmentGuard()
        # AdaptivePruner: RL weight learning on feedback
        self._pruner = EntrolyPruner()
        # Turn counter for provenance
        self._turn_counter: int = 0

        # ── Online Bayesian PRISM — live weight adaptation ──
        # Closes the learning loop: weights evolve with every optimize_context()
        # call using Dirichlet posterior updates with REINFORCE gradient estimates.
        # Prior is anchored to the startup config weights.
        self._online_prism = OnlinePrism(
            prior_weights={
                "w_recency": self.config.weight_recency,
                "w_frequency": self.config.weight_frequency,
                "w_semantic": self.config.weight_semantic_sim,
                "w_entropy": self.config.weight_entropy,
            },
            prior_strength=20.0,
        )

        # ── Reward-driven skill crystallization ───────────────────
        # Closes the asymmetry: the failure path crystallizes skills
        # from misses (record_miss → EvolutionDaemon → SkillEngine);
        # this is the success-path mirror. Detection is on the engine
        # so it stays alive across MCP/proxy/SDK call surfaces; the
        # actual SkillEngine.crystallize_skill call is wired via a
        # callback so we don't couple the engine to vault-IO concerns.
        from .reward_crystallizer import RewardCrystallizer
        self._crystallizer = RewardCrystallizer()
        self._crystallization_callback: Any = None
        self._crystallized_count: int = 0

        # Fast-path router (set lazily by create_mcp_server). When unset,
        # _try_fast_path is a no-op.
        self._fast_path_router: Any = None
        # Cache for the Rust engine's fragment export, used by the
        # fast-path's by-id-or-source lookup. Cleared on ingest.
        self._fragment_cache: dict[str, dict[str, Any]] = {}
        self._fragment_cache_dirty: bool = True

        # ── Long-term memory (Hippocampus integration) ────────────
        # Initialized here (not in create_mcp_server) so that anyone
        # constructing an EntrolyEngine directly — proxy users, SDK
        # users, tests — gets a working engine. Falls back to None
        # when hippocampus-sharp-memory isn't installed; consumers
        # must guard with ``getattr(engine, '_ltm', None) is not None``
        # to handle both "absent" and "present-but-disabled" cases.
        try:
            from .long_term_memory import LongTermMemory, is_available
            if is_available():
                self._ltm = LongTermMemory()
            else:
                self._ltm = None
        except Exception as _ltm_err:
            logger.debug("LongTermMemory init skipped: %s", _ltm_err)
            self._ltm = None

        # ── Per-fragment selection counter ─────────────────────────
        # Drives the "consider pinning" memory nudge (P1.D1). A fragment
        # repeatedly selected across optimizations is a persistence
        # candidate — if it keeps appearing, it's worth pinning.
        self._fragment_selection_counts: dict[str, int] = {}
        # Tracks when D2 nudge last fired so we don't spam every call.
        self._crystallized_at_last_nudge: int = 0
        # External callback for FeedbackJournal logging from genuine
        # outcome signals (record_success/failure), not budget-shape
        # implicit rewards. Wired by create_mcp_server.
        self._journal_callback: Any = None

        # Fix #5: Validate that the checkpoint directory is writable at startup.
        # Fail fast with a clear error rather than a cryptic gzip/PermissionError
        # during the first auto-checkpoint (which could happen mid-session).
        self._validate_checkpoint_dir()

        # ── Persistent Repo-Level Indexing ──
        # On startup, try to load a previous session's index for instant warm retrieval.
        # Index is stored at <checkpoint_dir>/index.json.gz (gzip-compressed JSON).
        # Gated by config.use_persistent_index, which controls BOTH the load
        # path here and the save path in auto_checkpoint(), so test/probe
        # engines that opt out are fully ephemeral (no read, no write).
        self._index_path = str(Path(self.config.checkpoint_dir) / "index.json.gz")
        # ── Warm-start: load the persistent index LAZILY, on first use ──
        # Previously the index loaded synchronously here, blocking engine
        # construction (and, critically, an MCP server's startup handshake) for
        # seconds on a large warm-start cache — which reads as "broken" to
        # Claude Desktop / Cursor on first launch. Instead, defer the load to
        # the first optimize/recall call (_ensure_index_loaded). Construction —
        # and the MCP handshake — return instantly; the one-time load happens on
        # the FIRST request, on the request thread (no background thread, so no
        # concurrent Rust &mut borrow / "Already borrowed" race). A lock makes
        # the one-time load idempotent under concurrent first calls.
        self._index_load_lock = threading.Lock()
        # Reconciliation spans remove -> ingest -> dependency rebuild ->
        # persistence. Serialize that transaction across manual calls and the
        # background watcher; an RLock permits auto_index's warm-cache path to
        # invoke reconciliation without deadlocking itself.
        self._index_mutation_lock = threading.RLock()
        self._index_loaded = not self.config.use_persistent_index
        if not self.config.use_persistent_index:
            logger.debug("Persistent index disabled by config (isolated/ephemeral engine)")

        # Host GC policy is process-wide. Entroly deliberately leaves it
        # untouched: a library must not freeze, disable, enable, or collect
        # objects owned by the embedding application or its other threads.

    def _ensure_index_loaded(self) -> None:
        """Lazily load the persistent warm-start index on first use.

        Called at the top of the request-path entry points (optimize/recall).
        Runs on the calling thread — no background thread — so there is never a
        concurrent Rust ``&mut`` borrow. Idempotent and lock-guarded so two
        concurrent first calls don't both load. Cheap no-op after the first.
        """
        if self._index_loaded:
            return
        with self._index_load_lock:
            if self._index_loaded:
                return
            try:
                if not self._use_rust:
                    loaded = self._load_index_compat(self._index_path)
                    if loaded:
                        logger.info(
                            "Warm-start: restored %d Python index fragments from %s",
                            loaded,
                            self._index_path,
                        )
                    else:
                        logger.info("No persistent index found, starting fresh session")
                    return
                try:
                    loaded = self._rust.load_index(self._index_path)
                    if loaded:
                        n = self._rust.fragment_count()
                        logger.info(f"Warm-start: restored {n} index fragments from {self._index_path}")
                        self._ensure_gzip_index(self._index_path)
                    else:
                        logger.info("No persistent index found, starting fresh session")
                except AttributeError:
                    loaded = self._load_index_compat(self._index_path)
                    if loaded:
                        logger.info(f"Warm-start: restored {loaded} index fragments from {self._index_path}")
                    else:
                        logger.debug("Rust engine does not support load_index -- skipping persistent index")
                except OSError:
                    logger.info("No persistent index found, starting fresh session")
                except Exception as e:
                    try:
                        loaded = self._load_index_compat(self._index_path)
                        if loaded:
                            logger.info(f"Warm-start: restored {loaded} index fragments from {self._index_path}")
                        else:
                            logger.warning(f"Failed to load persistent index: {e}")
                    except Exception:
                        logger.warning(f"Failed to load persistent index: {e}")
            finally:
                self._index_loaded = True

    def wait_until_warm(self, timeout: float | None = None) -> bool:
        """Trigger the lazy warm-start load now and return True once loaded.

        The load is synchronous (no background thread); ``timeout`` is accepted
        for API symmetry but the call returns only after the load completes.
        Useful for tests and deterministic flows that want the index resident.
        """
        self._ensure_index_loaded()
        return True

    def advance_turn(self) -> None:
        """Advance the turn counter and apply Ebbinghaus decay."""
        if self._use_rust:
            self._rust.advance_turn()
        else:
            self._current_turn += 1
            fragments = list(self._fragments.values())
            _py_apply_ebbinghaus_decay(
                fragments,
                self._current_turn,
                self.config.decay_half_life_turns,
            )
            to_evict = [
                fid for fid, f in self._fragments.items()
                if f.recency_score < self.config.min_relevance_threshold
                and not f.is_pinned
            ]
            for fid in to_evict:
                self._dedup.remove(fid)
                del self._fragments[fid]

    def ingest_fragment(
        self,
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """Ingest a new context fragment without changing host GC policy."""
        # Lazy warm-start MUST run before the first mutation: load_index
        # replaces the fragment set, so it has to happen before any ingest or
        # it would wipe freshly-ingested fragments.
        self._ensure_index_loaded()

        # Invalidate the fast-path's fragment cache: the Rust engine has
        # gained a new fragment, so any cached export_fragments() snapshot
        # is now stale.
        self._fragment_cache_dirty = True
        if self._use_rust:
            # Enforce max_fragments cap on Rust engine (Rust doesn't enforce it)
            if self._rust.fragment_count() >= self.config.max_fragments:
                return {
                    "status": "rejected",
                    "reason": "max_fragments cap reached",
                    "max_fragments": self.config.max_fragments,
                }
            result = self._rust.ingest(content, source, token_count, is_pinned)
            if source:
                self._prefetch.record_access(source, self._rust.get_turn())
            if self._checkpoint_mgr.should_auto_checkpoint():
                self._auto_checkpoint()
            return dict(result)
        return self._ingest_python(content, source, token_count, is_pinned)

    def remove_sources(self, sources: list[str]) -> dict[str, Any]:
        """Remove all live fragments for exact source identifiers.

        Repository files are indexed with ``file:<relative-path>`` sources.
        A file update must remove the complete previous source atomically before
        ingesting its replacement; otherwise contradictory old and new versions
        can both be selected.  Unknown native wheels fail visibly instead of
        pretending the stale content was removed.
        """
        self._ensure_index_loaded()
        normalized = sorted({str(source) for source in sources if str(source)})
        if not normalized:
            return {
                "status": "unchanged",
                "removed_fragments": 0,
                "removed_sources": [],
                "missing_sources": [],
            }

        self._fragment_cache_dirty = True
        if self._use_rust:
            if not hasattr(self._rust, "remove_sources"):
                return {
                    "status": "unsupported",
                    "reason": "native_remove_sources_unavailable",
                    "removed_fragments": 0,
                    "removed_sources": [],
                    "missing_sources": normalized,
                }
            result = dict(self._rust.remove_sources(normalized))
            result["status"] = "removed"
        else:
            requested = set(normalized)
            removed_ids = [
                fragment_id
                for fragment_id, fragment in self._fragments.items()
                if fragment.source in requested
            ]
            removed_sources = sorted({
                self._fragments[fragment_id].source for fragment_id in removed_ids
            })
            for fragment_id in removed_ids:
                self._dedup.remove(fragment_id)
                del self._fragments[fragment_id]

            removed_id_set = set(removed_ids)
            self._wilson.remove_fragments(removed_id_set)

            # Token-frequency state contributes to Python fallback entropy scores.
            # Rebuild it from live content so removed bytes cannot bias later files.
            from collections import Counter
            self._global_token_counts = Counter()
            self._total_token_count = 0
            for fragment in self._fragments.values():
                tokens = fragment.content.lower().split()
                self._global_token_counts.update(tokens)
                self._total_token_count += len(tokens)

            result = {
                "status": "removed",
                "removed_fragments": len(removed_ids),
                "removed_fragment_ids": sorted(removed_ids),
                "removed_sources": removed_sources,
                "missing_sources": sorted(requested - set(removed_sources)),
                "remaining_fragments": len(self._fragments),
            }

        removed_id_set = set(result.get("removed_fragment_ids", []))
        for fragment_id in removed_id_set:
            self._fragment_selection_counts.pop(fragment_id, None)
        removed_paths: set[str] = set()
        for source in result.get("removed_sources", []):
            source = str(source)
            if source.startswith("file:"):
                # Prefetch callers historically used both the fragment source
                # and the bare workspace path; invalidate both aliases.
                removed_paths.update({source, source[5:]})
        self._prefetch.remove_paths(removed_paths)
        return result

    def rebuild_dependencies(self) -> dict[str, Any]:
        """Rebind the dependency graph to the current live fragment identities."""
        self._ensure_index_loaded()
        if not self._use_rust:
            return {"status": "disabled", "reason": "python_fallback_has_no_dependency_graph"}
        if not hasattr(self._rust, "rebuild_dependencies"):
            return {"status": "unsupported", "reason": "native_dependency_rebuild_unavailable"}
        result = dict(self._rust.rebuild_dependencies())
        result["status"] = "rebuilt"
        return result

    def snapshot_index_state(self) -> dict[str, Any]:
        """Capture the mutable index boundary for recoverable reconciliation."""
        self._ensure_index_loaded()
        snapshot: dict[str, Any] = {
            "engine": "rust" if self._use_rust else "python",
            "fragment_selection_counts": copy.deepcopy(self._fragment_selection_counts),
            "prefetch_co_access": copy.deepcopy(self._prefetch._co_access),
            "prefetch_recent_accesses": copy.deepcopy(self._prefetch._recent_accesses),
            "prefetch_pending_predictions": copy.deepcopy(self._prefetch._pending_predictions),
        }
        if self._use_rust:
            snapshot["rust_state"] = self._rust.export_state()
        else:
            snapshot.update({
                "fragments": copy.deepcopy(self._fragments),
                "dedup": copy.deepcopy(self._dedup),
                "global_token_counts": copy.deepcopy(self._global_token_counts),
                "total_token_count": self._total_token_count,
                "wilson": copy.deepcopy(self._wilson),
                "total_fragments_ingested": self._total_fragments_ingested,
                "total_duplicates_caught": self._total_duplicates_caught,
            })
        return snapshot

    def restore_index_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot after a failed multi-step index mutation."""
        expected_engine = "rust" if self._use_rust else "python"
        if snapshot.get("engine") != expected_engine:
            raise RuntimeError("index snapshot engine does not match the live engine")
        if self._use_rust:
            self._rust.import_state(snapshot["rust_state"])
        else:
            self._fragments = snapshot["fragments"]
            self._dedup = snapshot["dedup"]
            self._global_token_counts = snapshot["global_token_counts"]
            self._total_token_count = snapshot["total_token_count"]
            self._wilson = snapshot["wilson"]
            self._total_fragments_ingested = snapshot["total_fragments_ingested"]
            self._total_duplicates_caught = snapshot["total_duplicates_caught"]
        self._fragment_selection_counts = snapshot["fragment_selection_counts"]
        self._prefetch._co_access = snapshot["prefetch_co_access"]
        self._prefetch._recent_accesses = snapshot["prefetch_recent_accesses"]
        self._prefetch._pending_predictions = snapshot["prefetch_pending_predictions"]
        self._fragment_cache.clear()
        self._fragment_cache_dirty = True

    def optimize_context(
        self,
        token_budget: int = 0,
        query: str = "",
    ) -> dict[str, Any]:
        """Select a high-value subset of context fragments within the budget.

        Exact for the modular objective on the Rust 0/1-DP path; the Python
        fallback is density-greedy with a singleton champion, provably within
        ½ of optimal. See _py_knapsack_optimize for why ½ and not (1 - 1/e).
        """
        self._ensure_index_loaded()  # lazy warm-start on first request
        if token_budget <= 0:
            token_budget = self.config.default_token_budget

        # ── Fast-path: matched crystallized skill bypasses pipeline ─────
        # When a query matches a previously-crystallized skill (its
        # Hoeffding LCB beat the global baseline, so the recipe is
        # statistically validated), we can skip the full knapsack
        # pipeline and return the recorded recipe directly. The skill
        # tracks its own fitness; OnlinePrism observation is intentionally
        # skipped on the fast-path since the skill's exploration phase
        # is over.
        fp_result = self._try_fast_path(query, token_budget)
        if fp_result is not None:
            # The fast path returns before the whole pipeline, so it also
            # returns before the no-match contract at the far end of this
            # method. A promoted skill is fitness-gated and freshness-checked,
            # but its trigger is a regex over the query: a pattern that fires
            # too broadly replays a recipe that has nothing to do with what was
            # asked, and that is precisely the "confident unrelated files"
            # failure the contract exists to stop.
            #
            # `scores_are_ranked=False` because there is no ranking here to be
            # degenerate -- the selection is a memoized decision, not a scored
            # one. Its fragments come straight from the store and carry no
            # relevance, so `_evidence_signal` reports unmeasured and the
            # lexical check decides alone. Passing True instead would read every
            # fast-path hit as unranked and discard it.
            apply_no_match_contract(fp_result, query, scores_are_ranked=False)
            return fp_result

        # Query refinement: expand vague queries using in-memory file context.
        # This is the key fix for hallucination from incomplete context:
        # "fix the bug" → "bug fix in payments module (Python/Rust) involving
        # payment processing, error handling"
        refined_query = query
        refinement_info: dict[str, Any] = {}
        if query:
            fragment_summaries = []
            if self._use_rust:
                try:
                    recalled = list(self._rust.recall(query, 20))
                    fragment_summaries = [r.get("content", "") for r in recalled]
                except Exception:
                    pass
            # analyze() returns dict: vagueness_score, key_terms, needs_refinement, reason
            analysis_dict = self._refiner.analyze(query, fragment_summaries)
            # refine() returns the improved query string.
            #
            # Only adopt it when the analyzer says the query needs it. The
            # expansion terms come from `recall(query, 20)` -- pseudo-relevance
            # feedback -- so applying it unconditionally lets a mediocre first
            # pass rewrite the question and then rank against the rewrite. That
            # is query drift, and it was measured destroying retrieval on
            # specific queries that needed no help:
            #
            #   "what does entroly doctor check and how does it report failures"
            #     -> "... [context: tests, verifier, plugin, entry, point]"
            #
            # None of those terms are in the question. They pulled
            # test_verifier_plugins.py and plugins.py to the top while the file
            # that answers it, entroly/cli.py, fell from rank 4 to rank 49 --
            # far below the ~12 fragments a budget emits. Gating on the
            # analyzer's own signal took evidence retention on the frozen
            # eight-query set from 0.750 to 1.000 at 4k, 8k and 16k budgets.
            #
            # The refinement is still computed and still reported, so vague
            # queries keep the benefit and the receipt stays honest about what
            # was asked versus what was searched.
            candidate_refined = self._refiner.refine(query, fragment_summaries)
            if analysis_dict.get("needs_refinement"):
                refined_query = candidate_refined
                refinement_info = {
                    "original_query":    query,
                    "refined_query":     refined_query,
                    "vagueness_score":   analysis_dict["vagueness_score"],
                    "refinement_source": "rust_heuristic",
                    "key_terms":         analysis_dict.get("key_terms", []),
                }

        # Always capture query analysis (vagueness is needed by EGTC even
        # when the query doesn't trigger refinement)
        query_analysis = {
            "vagueness_score": analysis_dict["vagueness_score"],
            "key_terms": analysis_dict.get("key_terms", []),
        } if query and analysis_dict else {}

        # Keep optimization inside Entroly's own allocation discipline;
        # never mutate the embedding process's GC policy.
        if self._use_rust and refined_query.strip():
            # ── Query-conditioned selection via QCCR ──────────────────
            # qccr.select (sentence-level BM25 + MMR + entity boost +
            # anchor fallback) is the compressor validated by EVERY
            # committed accuracy benchmark (needle/longbench/squad/bfcl/
            # mmlu/gsm8k/truthfulqa). Routing the engine's query path
            # through it makes the SHIPPED behaviour identical to the
            # MEASURED behaviour across the SDK, MCP, and proxy — not just
            # the CLI. Any failure falls back to the native knapsack so
            # optimize_context never breaks.
            try:
                from .qccr import select as qccr_select
                candidates = [dict(f) for f in self._rust.export_fragments()]
                def _fragment_tokens(fragment: dict[str, Any]) -> int:
                    token_count = int(fragment.get("token_count") or 0)
                    if token_count > 0:
                        return token_count
                    content = str(fragment.get("content") or "")
                    return max(1, len(content) // 4) if content else 0

                pinned = [f for f in candidates if f.get("is_pinned")]
                pinned_cap = token_budget // 2
                pinned_tokens = sum(_fragment_tokens(f) for f in pinned)
                if pinned_tokens > pinned_cap:
                    def _pinned_priority(fragment: dict[str, Any]) -> float:
                        relevance = (
                            self.config.weight_recency
                            * float(fragment.get("recency_score") or 0.0)
                            + self.config.weight_frequency
                            * float(fragment.get("frequency_score") or 0.0)
                            + self.config.weight_semantic_sim
                            * float(fragment.get("semantic_score") or 0.0)
                            + self.config.weight_entropy
                            * float(fragment.get("entropy_score") or 0.0)
                        )
                        return relevance * float(
                            fragment.get("feedback_multiplier") or 1.0
                        )

                    pinned.sort(key=_pinned_priority, reverse=True)
                    exact_pinned = []
                    pinned_tokens = 0
                    for fragment in pinned:
                        fragment_tokens = _fragment_tokens(fragment)
                        if pinned_tokens + fragment_tokens <= pinned_cap:
                            exact_pinned.append(fragment)
                            pinned_tokens += fragment_tokens
                else:
                    exact_pinned = pinned

                exact_pinned_sources = {
                    str(f.get("source") or "") for f in exact_pinned
                }
                qccr_candidates = [
                    f for f in candidates
                    if str(f.get("source") or "") not in exact_pinned_sources
                ]
                qccr_budget = max(0, token_budget - pinned_tokens)
                selected = qccr_select(
                    qccr_candidates,
                    token_budget=qccr_budget,
                    query=refined_query,
                ) if qccr_budget else []
                selected = [
                    {
                        **f,
                        "id": f.get("id") or f.get("fragment_id"),
                    }
                    for f in exact_pinned
                ] + selected

                tokens_used = sum(
                    _fragment_tokens(f) for f in selected if isinstance(f, dict)
                )
                total_available_tokens = sum(
                    _fragment_tokens(f) for f in candidates if isinstance(f, dict)
                )
                tokens_saved = _honest_tokens_saved(
                    selected, total_available_tokens - tokens_used
                )
                self._total_tokens_saved += tokens_saved
                selected_count = len(selected)
                total_relevance = round(sum(
                    float(f.get("relevance", f.get("relevance_score", 0.0)) or 0.0)
                    for f in selected if isinstance(f, dict)
                ), 4)
                optimization_stats = {
                    "total_tokens": tokens_used,
                    "selected_count": selected_count,
                    "total_relevance": total_relevance,
                    "effective_budget": token_budget,
                    "budget_utilization": round(tokens_used / max(token_budget, 1), 4),
                }
                result = {
                    "selected_fragments": selected,
                    "selected": selected,
                    "tokens_used": tokens_used,
                    "total_tokens": tokens_used,
                    "selected_count": selected_count,
                    "total_relevance": total_relevance,
                    "tokens_saved": tokens_saved,
                    "tokens_saved_this_call": tokens_saved,
                    "total_tokens_saved_session": self._total_tokens_saved,
                    "optimization_stats": optimization_stats,
                    "method": "qccr",
                    "effective_budget": token_budget,
                    "user_budget": token_budget,
                    "budget_utilization": optimization_stats["budget_utilization"],
                    "total_fragments": len(candidates),
                    "selector": "qccr",
                }
            except Exception:
                result = dict(self._rust.optimize(token_budget, refined_query))
                if "selected" in result and "selected_fragments" not in result:
                    result["selected_fragments"] = result["selected"]
            if refinement_info:
                result["query_refinement"] = refinement_info
            if query_analysis:
                result["query_analysis"] = query_analysis
        elif self._use_rust:
            result = self._rust.optimize(token_budget, refined_query)
            result = dict(result)
            # Normalize key: Rust returns "selected", Python uses "selected_fragments"
            if "selected" in result and "selected_fragments" not in result:
                result["selected_fragments"] = result["selected"]
            if refinement_info:
                result["query_refinement"] = refinement_info
            if query_analysis:
                result["query_analysis"] = query_analysis
        else:
            result = self._optimize_python(token_budget, refined_query)
            if refinement_info:
                result["query_refinement"] = refinement_info
            if query_analysis:
                result["query_analysis"] = query_analysis

        # Normalize the public selection contract across QCCR, native,
        # pure-Python, fast-path, and oversize-excerpt selectors. Several
        # fallback paths returned real selected fragments while omitting
        # ``selected_count`` (or leaving it at zero), which made agents and
        # dashboards report an empty result despite receiving context.
        selected_payload = result.get(
            "selected_fragments", result.get("selected", [])
        )
        if isinstance(selected_payload, list):
            result["selected_count"] = len(selected_payload)
            optimization_stats = result.get("optimization_stats")
            if isinstance(optimization_stats, dict):
                optimization_stats["selected_count"] = len(selected_payload)

        # ── engine_s6 edit-target reordering (post-selection) ──
        # The knapsack picks WHICH fragments stay in budget; engine_s6
        # then re-orders those by file-edit-target priority (source >
        # test > non-source within a top-20 window, explicit-cue
        # freeze, doc/test intent guards, test→source mirror) so the
        # LLM sees the most plausible edit target first. Selection
        # is unchanged — only order — so token budget, savings, and
        # PRISM outcome math below remain correct. QCCR already applies
        # this rerank before selection, so repeating it here wastes a full
        # corpus scan. Recall-safe by construction (see file_localizer.py).
        if (
            refined_query
            and result.get("selected_fragments")
            and result.get("selector") != "qccr"
        ):
            try:
                from .file_localizer import localize_fragments
                result["selected_fragments"] = localize_fragments(
                    result["selected_fragments"], refined_query,
                )
                # Keep the "selected" alias (Rust path) in sync so
                # downstream consumers reading either key get the
                # same reordered list.
                if "selected" in result:
                    result["selected"] = result["selected_fragments"]
            except Exception:  # noqa: BLE001 — never fail optimize
                pass

        # ── Online PRISM: observe outcome and update weights live ──
        # This is the key integration: after every optimize_context() call,
        # compute an implicit reward from the result quality and update the
        # Dirichlet posterior. Weights shift toward configurations that
        # produce better budget utilization and selectivity.
        try:
            selected = result.get("selected_fragments", result.get("selected", []))
            tokens_used = result.get("tokens_used",
                sum(f.get("token_count", 0) for f in selected if isinstance(f, dict)))
            # Bug-fix: when the result dict lacks total_fragments AND
            # fragment_count (Rust engine path doesn't surface them),
            # fall back to the engine's live count via the public API.
            # Prior fallback `max(len(selected), 1)` set select_rate=1.0
            # always — far from the 0.4 optimum — collapsing reward to
            # near zero and making crystallization structurally
            # impossible from organic traffic.
            total_frags = result.get("total_fragments",
                result.get("fragment_count", None))
            if total_frags is None:
                try:
                    if self._use_rust:
                        total_frags = int(self._rust.fragment_count())
                    else:
                        total_frags = len(self._fragments)
                except Exception:
                    total_frags = max(len(selected), 1)
            total_frags = max(total_frags, len(selected), 1)

            # ── Record features for the RL pruner ──────────────────
            # For each selected fragment, capture scoring features so
            # that a later record_success/failure call can perform the
            # REINFORCE gradient step.  Without this, apply_feedback
            # has no feature record and silently drops every signal.
            try:
                for frag in (selected if isinstance(selected, list) else []):
                    if not isinstance(frag, dict):
                        continue
                    fid = str(frag.get("id") or frag.get("fragment_id") or "")
                    if not fid:
                        continue
                    self._pruner.record_fragment_features(
                        fragment_id=fid,
                        recency=float(frag.get("recency_score", frag.get("recency", 0.5))),
                        relevance=float(frag.get("relevance_score", frag.get("relevance", 0.5))),
                        complexity=float(frag.get("complexity", frag.get("token_count", 100)) / 500.0),
                        was_selected=True,
                    )
            except Exception:
                pass  # Best-effort; never break optimize_context

            reward = compute_implicit_reward(
                selected_count=len(selected),
                total_fragments=total_frags,
                tokens_used=tokens_used,
                query_present=bool(query),
                token_budget=token_budget,
            )
            contributions = compute_contributions(
                selected if isinstance(selected, list) else [],
                total_frags,
            )

            # Capture PRE-observe baseline + weights for crystallization.
            # Using post-observe values would let the cluster contaminate
            # its own baseline (false-positive bias) and would credit the
            # weights that *resulted from* the reward rather than the
            # weights that *earned* it.
            pre_baseline = self._online_prism._reward_ema  # noqa: SLF001
            pre_weights = self._online_prism.weights()

            new_weights = self._online_prism.observe(reward, contributions)

            # ── Reward-driven crystallization ─────────────────────
            # Hoeffding-LCB gated detection of sustained-high-reward
            # query clusters. When a cluster crosses the bound, fire
            # the registered callback (typically SkillEngine.crystallize_skill).
            # All bookkeeping is off the hot path: never blocks return.
            try:
                if query and isinstance(selected, list):
                    frag_ids = [
                        str(f.get("id") or f.get("fragment_id") or f.get("source", ""))
                        for f in selected if isinstance(f, dict)
                    ]
                    frag_ids = [fid for fid in frag_ids if fid]
                    event = self._crystallizer.observe(
                        query=query,
                        reward=reward,
                        weights=pre_weights,
                        selected_fragment_ids=frag_ids,
                        baseline_reward=pre_baseline,
                    )
                    if event is not None and self._crystallization_callback is not None:
                        try:
                            self._crystallization_callback(event)
                            self._crystallized_count += 1
                        except Exception as cb_err:
                            logger.debug(
                                "crystallization callback error: %s", cb_err
                            )
            except Exception as cryst_err:
                logger.debug("crystallizer error: %s", cryst_err)

            # ── P1: Count fragment selections for pin-candidate nudges ─
            # Simple frequency counter: fragments that keep getting
            # selected are worth pinning. No reward gating — selection
            # frequency is the right signal (the fragment keeps appearing
            # because the scoring pipeline considers it valuable).
            try:
                if frag_ids:
                    for fid in frag_ids:
                        self._fragment_selection_counts[fid] = (
                            self._fragment_selection_counts.get(fid, 0) + 1
                        )
            except Exception:
                pass

            # Apply updated weights to the live engine
            if self._online_prism._n >= 3:  # Wait for warmup
                w = self._online_prism.weights_tuple()
                if self._use_rust:
                    try:
                        self._rust.set_weights(w[0], w[1], w[2], w[3])
                    except Exception:
                        pass  # Rust engine may not support set_weights
                else:
                    self.config.weight_recency = w[0]
                    self.config.weight_frequency = w[1]
                    self.config.weight_semantic_sim = w[2]
                    self.config.weight_entropy = w[3]

            result["online_prism"] = {
                "reward": round(reward, 4),
                "implicit_advantage": round(reward - pre_baseline, 4),
                "contributions": {k: round(v, 4) for k, v in contributions.items()},
                "weights": {k: round(v, 4) for k, v in new_weights.items()},
                "n": self._online_prism._n,
                "phase": self._online_prism.stats()["phase"],
            }

            # Crystallizer surface: observability for the meta-loop.
            # Cheap to compute (locks are held briefly inside the
            # crystallizer); useful for the dashboard and for users
            # to understand when their patterns get frozen as skills.
            cr_stats = self._crystallizer.stats()
            result["crystallization"] = {
                "active_clusters": cr_stats["active_clusters"],
                "total_observations": cr_stats["total_observations"],
                "lifetime_crystallized": self._crystallized_count,
            }
        except Exception as e:
            logger.debug("OnlinePrism observation error: %s", e)

        # ── No-match contract ────────────────────────────────────────
        # This is the engine method every non-MCP caller reaches: the CLI,
        # the SDK, the proxy. The guard used to exist only on the MCP tool
        # handler of the same name nested inside `create_mcp_server`, so a
        # query matching nothing returned confident unrelated files
        # everywhere except MCP. Verified before this line existed: a
        # selection scoring `relevance: 0.0` with no query term in the
        # delivered text came back as an ordinary success.
        #
        # `selector` is set to "qccr" by the query path above. QCCR's
        # relevance is a match indicator (uniform 1.0 matched / 0.0 not),
        # not a rank, so the variance discriminator cannot read it and is
        # switched off for that path; `_evidence_backed` carries the signal
        # instead. The native knapsack path does produce a real ranking, so
        # it keeps the variance test.
        apply_no_match_contract(
            result,
            query,
            scores_are_ranked=result.get("selector") != "qccr",
        )
        return result
    def set_crystallization_callback(self, fn: Any) -> None:
        """Register a callback fired when a query cluster crystallizes.

        Signature: ``fn(event: CrystallizationEvent) -> Any``. Exceptions
        in the callback are caught and logged; they never affect
        ``optimize_context`` return value or latency. Pass ``None`` to
        unregister.

        Typical wiring (in ``create_mcp_server``):

            engine.set_crystallization_callback(skill_engine.crystallize_skill)
        """
        self._crystallization_callback = fn

    def set_fast_path_router(self, router: Any) -> None:
        """Register the fast-path router that bypasses the full pipeline
        when a query matches a promoted crystallized skill.

        Pass ``None`` to disable. The router is consulted at the top of
        ``optimize_context`` — a hit short-circuits the knapsack +
        OnlinePrism pipeline and returns the skill's recipe directly.
        """
        self._fast_path_router = router

    def _get_fragment(self, key: str) -> dict[str, Any] | None:
        """Look up a fragment by id or source.

        Used by the fast-path router to materialize a skill's recipe.
        Source-based lookup is preferred for stability (fragment IDs
        change across sessions; source paths persist).
        """
        # Python fallback: try id lookup first
        if not self._use_rust:
            f = self._fragments.get(key)
            if f is not None:
                return {
                    "id": getattr(f, "fragment_id", key),
                    "source": getattr(f, "source", ""),
                    "content": getattr(f, "content", ""),
                    "token_count": getattr(f, "token_count", 0),
                    "entropy_score": getattr(f, "entropy_score", 0.0),
                }
            # Source lookup
            for fid, frag in self._fragments.items():
                if getattr(frag, "source", "") == key:
                    return {
                        "id": fid, "source": key,
                        "content": getattr(frag, "content", ""),
                        "token_count": getattr(frag, "token_count", 0),
                        "entropy_score": getattr(frag, "entropy_score", 0.0),
                    }
            return None
        # Rust path: use export_fragments and search. Cache to amortize.
        if not hasattr(self, "_fragment_cache") or self._fragment_cache_dirty:
            try:
                items = list(self._rust.export_fragments())
            except Exception:
                items = []
            cache: dict[str, dict[str, Any]] = {}
            for it in items:
                d = dict(it)
                fid = d.get("id") or d.get("fragment_id") or ""
                src = d.get("source", "")
                if fid:
                    cache[fid] = d
                if src:
                    cache[src] = d
            self._fragment_cache = cache
            self._fragment_cache_dirty = False
        return self._fragment_cache.get(key)

    def _try_fast_path(self, query: str, token_budget: int) -> dict[str, Any] | None:
        """Consult the fast-path router; return a result dict on hit, None otherwise.

        Wraps every error path so a malformed router never breaks
        optimize_context — the worst case is a silent fall-through to
        the normal pipeline.
        """
        router = getattr(self, "_fast_path_router", None)
        if router is None:
            return None
        try:
            fp = router.try_match(query, token_budget=token_budget)
        except Exception as e:
            logger.debug("fast_path.try_match raised: %s — falling through", e)
            return None
        if fp is None:
            return None
        # Build a result dict that matches the optimize_context contract
        # closely enough that downstream consumers (MCP wrappers, sanitizer,
        # nudge computation) work without modification.
        return {
            "selected_fragments": fp.selected_fragments,
            "selected": fp.selected_fragments,  # alias
            "tokens_used": sum(
                int(f.get("token_count", 0) or 0) for f in fp.selected_fragments
            ),
            "total_fragments": (
                self._rust.fragment_count() if self._use_rust
                else len(self._fragments)
            ),
            "fast_path": {
                "hit": True,
                "skill_id": fp.skill_id,
                "cluster_id": fp.cluster_id,
                "fitness": fp.fitness,
                "recipe_size": fp.recipe_size,
                "matched_present": fp.matched_present,
                "matched_missing": fp.matched_missing,
                "elapsed_ms": fp.elapsed_ms,
            },
            # Provide the empty/zero analogues for fields the downstream
            # consumer expects, so it doesn't crash on missing keys.
            "online_prism": {
                "reward": 0.0, "weights": {},
                "n": self._online_prism._n, "phase": "fast_path",
            },
            "crystallization": self._crystallizer.stats() | {
                "lifetime_crystallized": self._crystallized_count,
            },
            "tokens_saved": 0,
            "method": "fast_path",
            "query": query,
            "token_budget": token_budget,
        }

    def set_journal_callback(self, fn: Any) -> None:
        """Register a callback for implicit-reward FeedbackJournal logging.

        Signature: ``fn(*, weights, reward, selected_count, query, token_budget)``

        Wired by ``create_mcp_server`` to ``_feedback_journal.log()``.
        This decouples the engine from journal IO concerns.
        """
        self._journal_callback = fn

    def _compute_memory_nudges(
        self,
        result: dict[str, Any],
        query: str,
    ) -> list[dict[str, Any]]:
        """Generate proactive persistence hints for the agent.

        Three deterministic detectors:

        D1 — Fragment selected ≥10 times across optimizations, not pinned.
             "This fragment keeps appearing — consider pinning it."

        D2 — Crystallizer just promoted a NEW skill (fires once per event).
             "A query pattern was just promoted to a skill."

        Returns a list of nudge dicts, each with:
            type: "pin_candidate" | "skill_crystallized"
            fragment_id: (D1 only) the fragment to pin
            message: human-readable suggestion
        """
        nudges: list[dict[str, Any]] = []

        # D1: Fragments selected ≥10 times across optimizations
        # Selection frequency is the right signal — fragments that keep
        # appearing are valued by the scoring pipeline and worth pinning.
        PIN_THRESHOLD = 10
        selected = result.get("selected_fragments", result.get("selected", []))
        selected_ids: set[str] = set()
        if isinstance(selected, list):
            for f in selected:
                if isinstance(f, dict):
                    fid = str(f.get("id") or f.get("fragment_id") or f.get("source", ""))
                    if fid:
                        selected_ids.add(fid)

        for fid, count in list(self._fragment_selection_counts.items()):
            if count >= PIN_THRESHOLD and fid in selected_ids:
                is_pinned = False
                if not self._use_rust:
                    frag = self._fragments.get(fid)
                    if frag and getattr(frag, "is_pinned", False):
                        is_pinned = True
                if not is_pinned:
                    nudges.append({
                        "type": "pin_candidate",
                        "fragment_id": fid,
                        "selection_count": count,
                        "message": (
                            f"Fragment '{fid}' has been selected {count} times "
                            f"but is not pinned. Consider calling "
                            f"remember_fragment with is_pinned=True to "
                            f"preserve it permanently."
                        ),
                    })
                    # Reset after surfacing (don't nag)
                    self._fragment_selection_counts[fid] = 0

        # D2: Crystallizer promoted a NEW skill since last nudge
        # Only fires when count increments — not every call.
        if self._crystallized_count > self._crystallized_at_last_nudge:
            new_skills = self._crystallized_count - self._crystallized_at_last_nudge
            self._crystallized_at_last_nudge = self._crystallized_count
            nudges.append({
                "type": "skill_crystallized",
                "new_skills": new_skills,
                "lifetime_skills": self._crystallized_count,
                "message": (
                    f"{new_skills} new query pattern(s) just promoted to "
                    f"reusable skills ({self._crystallized_count} total)."
                ),
            })

        return nudges

    def recall_relevant(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic recall of relevant fragments."""
        self._ensure_index_loaded()  # lazy warm-start on first request
        if self._use_rust:
            # Relevance retrieval ⇒ BM25 (recall_auto), not the SimHash
            # fingerprint primitive (which is for near-duplicate detection and
            # returns topically-unrelated fragments for a precise query).
            recall_fn = getattr(self._rust, "recall_auto", None) or self._rust.recall
            # Fetch a larger pool than requested so the path-type prior can
            # surface the implementation even when BM25 term-frequency floats
            # tests/docs to the top of the raw ranking; then trim to top_k.
            pool = max(top_k * 4, 40)
            result = recall_fn(query, pool)
            ranked = _apply_recall_path_prior([dict(r) for r in result], query)
            return ranked[:top_k]
        else:
            return self._recall_python(query, top_k)

    def _log_outcome_to_journal(
        self,
        fragment_ids: list[str],
        reward: float,
    ) -> None:
        """Feed FeedbackJournal from genuine outcome signals only.

        Called by record_success (+1.0) and record_failure (-1.0) —
        both come from real user validation (explicit record_outcome
        MCP calls or proxy rephrase/topic-change detection). Never
        from budget-shape implicit rewards.
        """
        if not self._journal_callback:
            return
        try:
            # Reconstruct current weights for the journal entry
            weights = {
                "w_r": self.config.weight_recency,
                "w_f": self.config.weight_frequency,
                "w_s": self.config.weight_semantic_sim,
                "w_e": self.config.weight_entropy,
            }
            self._journal_callback(
                weights=weights,
                reward=reward,
                selected_count=len(fragment_ids),
                query="",  # Not available at record time
                token_budget=0,
            )
        except Exception:
            pass  # Never fail outcome recording for journal IO

    def _resolve_native_feedback_ids(self, fragment_ids: list[str]) -> list[str]:
        """Map synthetic QCCR IDs back to the native fragments they represent."""
        if not self._use_rust or not fragment_ids:
            return list(fragment_ids)
        try:
            exported = [dict(f) for f in self._rust.export_fragments()]
        except Exception:
            return list(fragment_ids)

        native_ids = {
            str(f.get("fragment_id") or f.get("id") or "") for f in exported
        }
        by_source: dict[str, list[str]] = {}
        for fragment in exported:
            source = str(fragment.get("source") or "")
            fragment_id = str(
                fragment.get("fragment_id") or fragment.get("id") or ""
            )
            if source and fragment_id:
                by_source.setdefault(source, []).append(fragment_id)

        resolved: list[str] = []
        for fragment_id in fragment_ids:
            if fragment_id in native_ids:
                targets = [fragment_id]
            elif fragment_id.startswith("qccr::"):
                targets = by_source.get(fragment_id.removeprefix("qccr::"), [])
            else:
                targets = [fragment_id]
            for target in targets:
                if target and target not in resolved:
                    resolved.append(target)
        return resolved

    def record_success(self, fragment_ids: list[str]) -> None:
        """Record that selected fragments led to a successful output."""
        native_ids = self._resolve_native_feedback_ids(fragment_ids)
        if self._use_rust:
            self._rust.record_success(native_ids)
        else:
            self._wilson.record_success(native_ids)
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, 1.0)
        # Feed FeedbackJournal from genuine outcome signal.
        # This is real user-validated quality, not budget-shape proxy.
        self._log_outcome_to_journal(native_ids, reward=1.0)

    def record_failure(self, fragment_ids: list[str]) -> None:
        """Record that selected fragments led to a failed output."""
        native_ids = self._resolve_native_feedback_ids(fragment_ids)
        if self._use_rust:
            self._rust.record_failure(native_ids)
        else:
            self._wilson.record_failure(native_ids)
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, -1.0)
        # Feed FeedbackJournal from genuine outcome signal.
        self._log_outcome_to_journal(native_ids, reward=-1.0)

    def record_retrieval_miss(self, source_paths: list[str]) -> None:
        """Boost any fragments whose source matches a file the user actually
        edited but retrieval did NOT surface.

        This is the "should-have-retrieved" signal — the most valuable
        learning input, since it tells PRISM where its blind spots are.
        We translate the file-level miss into fragment-level positive
        feedback by looking up every indexed fragment whose ``source``
        matches one of the missed paths and giving it a moderate boost.
        Magnitude is capped (0.5, half a verified-hit) to acknowledge
        that "this file was relevant" is a weaker signal than "this
        fragment was the one that helped".
        """
        if not source_paths:
            return
        wanted = {p.replace("\\", "/").lstrip("./") for p in source_paths if p}
        if not wanted:
            return
        try:
            promoted: list[str] = []
            for frag in getattr(self, "_fragments", {}).values():
                src = (getattr(frag, "source", "") or "").replace("\\", "/").lstrip("./")
                if src in wanted:
                    promoted.append(frag.id if hasattr(frag, "id") else getattr(frag, "fragment_id", ""))
            promoted = [p for p in promoted if p]
            if not promoted:
                return
            # Half-strength positive: real but weaker than a verified hit.
            for fid in promoted:
                self._pruner.apply_feedback(fid, 0.5)
            # Wilson/Rust state also gets a half success — modeled as
            # one success out of two trials so it nudges without claiming
            # certainty.
            if self._use_rust and hasattr(self._rust, "record_success"):
                # Rust counter is integral; counts the fragments once.
                self._rust.record_success(promoted)
            else:
                self._wilson.record_success(promoted)
        except Exception:
            # Best-effort; never break record_outcome on miss-promotion.
            pass

    def record_reward(self, fragment_ids: list[str], reward: float) -> None:
        """Record a continuous reward signal for selected fragments.

        Unlike record_success/failure (binary), this allows graded feedback:
          reward > 0 → positive signal (boost fragment weight)
          reward < 0 → negative signal (suppress fragment weight)
          reward = 0 → neutral

        The Rust engine routes this to the EGSC cache's Thompson gate
        and hit predictor for continuous learning.
        """
        if self._use_rust:
            self._rust.record_reward(
                self._resolve_native_feedback_ids(fragment_ids), reward
            )
        # RL pruner also gets the continuous signal
        for fid in fragment_ids:
            self._pruner.apply_feedback(fid, reward)

    def record_resolution_outcome(
        self,
        resolutions: list[str],
        success: bool,
    ) -> None:
        """Teach IOS whether lower-resolution context was sufficient."""
        if (
            self._use_rust
            and hasattr(self._rust, "record_resolution_outcome")
            and resolutions
        ):
            self._rust.record_resolution_outcome(resolutions, success)

    def set_model(self, model_name: str) -> None:
        """Auto-configure cache cost model from model name.

        Covers 20+ models: OpenAI (gpt-4o, gpt-4, o1, o3), Anthropic
        (claude-3.5), Google (gemini), DeepSeek, Meta (llama), Mistral.
        Unknown models default to GPT-4o pricing ($0.000015/token).

        Example::

            engine.set_model("gpt-4o-mini")  # → $0.60/M tokens
            engine.set_model("claude-3-opus")  # → $75/M tokens
        """
        if self._use_rust:
            self._rust.set_model(model_name)

    def set_cache_cost_per_token(self, cost: float) -> None:
        """Set cost-per-token directly (power users only).

        Most developers should use set_model() instead.
        Default is already $0.000015 (GPT-4o output) — no config needed.
        """
        if self._use_rust:
            self._rust.set_cache_cost_per_token(cost)

    def cache_clear(self) -> None:
        """Clear all cached LLM responses.

        Useful when switching projects, after major refactors, or
        when cache correctness is suspect.
        """
        if self._use_rust:
            self._rust.cache_clear()

    def cache_len(self) -> int:
        """Return the number of entries in the response cache."""
        if self._use_rust:
            return self._rust.cache_len()
        return 0

    def cache_is_empty(self) -> bool:
        """Check if the response cache is empty."""
        if self._use_rust:
            return self._rust.cache_is_empty()
        return True

    def cache_hit_rate(self) -> float:
        """Return the cache hit rate (0.0 to 1.0).

        This is the primary observability metric for the EGSC cache.
        A healthy, warmed-up cache should show hit_rate > 0.3.
        """
        if self._use_rust:
            return self._rust.cache_hit_rate()
        return 0.0

    def prefetch_related(
        self,
        file_path: str,
        source_content: str = "",
        language: str = "python",
    ) -> list[dict[str, Any]]:
        """Predict and pre-load likely-needed context."""
        predictions = self._prefetch.predict(
            file_path, source_content, language
        )
        return [
            {
                "path": p.path,
                "reason": p.reason,
                "confidence": p.confidence,
            }
            for p in predictions
        ]

    def checkpoint(self, metadata: dict[str, Any] | None = None) -> str:
        """Manually create a checkpoint."""
        checkpoint_path = self._auto_checkpoint(metadata)
        if self._use_rust and self.config.use_persistent_index:
            self._ensure_gzip_index(self._index_path)
        return checkpoint_path

    def persist_index(self) -> dict[str, Any]:
        """Durably persist the live repository index without creating a checkpoint.

        Incremental workspace reconciliation can happen long before graceful
        shutdown. Persisting once per successful reconciliation prevents a crash
        from resurrecting deleted or superseded source content on the next run.
        """
        self._ensure_index_loaded()
        if not self.config.use_persistent_index:
            return {"status": "disabled", "reason": "persistent_index_disabled"}
        state = (
            self._rust.export_state()
            if self._use_rust
            else self._python_index_state()
        )
        self._write_gzip_index(self._index_path, state)
        return {
            "status": "persisted",
            "path": self._index_path,
            "fragment_count": (
                int(self._rust.fragment_count())
                if self._use_rust
                else len(self._fragments)
            ),
        }

    def resume(self, query: str = "", project: str = "") -> dict[str, Any]:
        """Resume the latest checkpoint or a query-relevant checkpoint."""
        match = None
        if query.strip():
            match = self._checkpoint_mgr.find_relevant(query, project=project)
            if match is None:
                return {
                    "status": "no_relevant_checkpoint_found",
                    "query": query,
                    "project": project,
                }
            ckpt = match.checkpoint
        else:
            ckpt = self._checkpoint_mgr.load_latest()
            if ckpt is None:
                return {"status": "no_checkpoint_found"}

        if self._use_rust:
            # Try to restore from full engine state (preferred)
            engine_state = ckpt.metadata.get("engine_state") if ckpt.metadata else None
            if engine_state:
                self._rust.import_state(engine_state)
            else:
                # Fallback: re-create engine and re-ingest fragments
                self._rust = _build_rust_engine(self.config)
                for frag_data in ckpt.fragments:
                    self._rust.ingest(
                        frag_data["content"],
                        frag_data.get("source", ""),
                        frag_data.get("token_count", 0),
                        frag_data.get("is_pinned", False),
                    )
        else:
            self._fragments.clear()
            for frag in self._checkpoint_mgr.restore_fragments(ckpt):
                self._fragments[frag.fragment_id] = frag
            self._dedup = _PyDedupIndex(
                hamming_threshold=getattr(self.config, "dedup_hamming_threshold", 3)
            )
            for fid, fp in ckpt.dedup_fingerprints.items():
                self._dedup._fingerprints[fid] = fp
            self._current_turn = ckpt.current_turn

        # Restore co-access patterns
        from collections import Counter
        for src, targets in ckpt.co_access_data.items():
            self._prefetch._co_access[src] = Counter(targets)

        decisions = ckpt.metadata.get("decisions")
        modified_files = ckpt.metadata.get("modified_files")
        public_metadata = {
            "instance_id": str(ckpt.metadata.get("instance_id", "")),
            "has_task": bool(ckpt.metadata.get("task")),
            "has_step": bool(ckpt.metadata.get("step")),
            "decision_count": len(decisions) if isinstance(decisions, list) else 0,
            "modified_file_count": (
                len(modified_files) if isinstance(modified_files, list) else 0
            ),
        }
        result: dict[str, Any] = {
            "status": "resumed",
            "checkpoint_id": ckpt.checkpoint_id,
            "restored_fragments": len(ckpt.fragments),
            "restored_turn": ckpt.current_turn,
            "metadata": public_metadata,
            "continuity_context": (
                self._checkpoint_mgr.render_recovery_context(match)
                if match is not None
                else self._checkpoint_mgr.render_checkpoint_context(ckpt)
            ),
        }
        if match is not None:
            result["relevance"] = {
                "score": match.score,
                "matched_terms": list(match.matched_terms),
                "lexical_score": match.lexical_score,
                "source_score": match.source_score,
                "project_score": match.project_score,
                "recency_score": match.recency_score,
            }
        return result

    @staticmethod
    def _honest_savings_block(savings: Any) -> dict[str, Any]:
        """Strip the fabricated dollar figure from the native `savings` block.

        The Rust core emits `estimated_cost_saved_usd` derived from
        `total_tokens_saved`, which is accumulated per call as
        (every candidate fragment) - (selected fragments). That counterfactual
        assumes the whole index would otherwise have been sent, so it grows
        without bound and routinely exceeds the corpus itself — an observed
        session reported 6.07M tokens "saved" against 2.58M tokens tracked, and
        a single 3,000-token request claimed 2.5M tokens saved.

        `optimize_context` already documents that MCP results "never fund
        evolution or support a dollar-savings claim"; the dashboard already
        strips this field for the same reason. This makes the MCP stats surface
        agree with both instead of publishing a number the code itself
        disclaims. Token telemetry is kept, renamed to say what it measures.
        """
        if not isinstance(savings, dict):
            return {} if savings is None else savings
        out = dict(savings)
        out.pop("estimated_cost_saved_usd", None)
        if "total_tokens_saved" in out:
            # Coerce: the raw value passed through verbatim, so a None or
            # non-numeric counter reached consumers as-is and any arithmetic on
            # it raised TypeError. A counter is always an int here.
            raw = out.pop("total_tokens_saved")
            try:
                out["dedup_tokens_avoided"] = int(raw or 0)
            except (TypeError, ValueError):
                out["dedup_tokens_avoided"] = 0
        out["baseline"] = (
            "dedup/selection telemetry only; counts candidates not selected. "
            "Not a provider bill delta and not a dollar-savings claim."
        )
        return out

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive session statistics."""
        if self._use_rust:
            rust_stats = dict(self._rust.stats())
            dep_stats = dict(self._rust.dep_graph_stats())
            rust_stats["dep_graph"] = dep_stats
            rust_stats["prefetch"] = self._prefetch.stats()
            rust_stats["checkpoint"] = self._checkpoint_mgr.stats()
            rust_stats["build"] = self._build_stamp()
            rust_stats["savings"] = self._honest_savings_block(
                rust_stats.get("savings")
            )
            return rust_stats
        stats = self._stats_python()
        stats["build"] = self._build_stamp()
        return stats

    def _build_stamp(self) -> dict[str, Any]:
        """Version/build identity so a caller can tell whether a shipped fix is
        actually live. Dogfooding otherwise had to cross-check the server
        process start time against the compiled core's mtime by hand; expose the
        engine version, whether the native core is active, and its build time.
        """
        import datetime
        import os
        import platform

        stamp: dict[str, Any] = {
            "native_engine": bool(self._use_rust),
            "python": platform.python_version(),
        }
        try:
            import entroly
            stamp["entroly_version"] = getattr(entroly, "__version__", "unknown")
        except Exception:
            pass
        try:
            import entroly_core
            core_file = getattr(entroly_core, "__file__", None)
            if core_file and os.path.exists(core_file):
                stamp["native_core_built"] = datetime.datetime.fromtimestamp(
                    os.path.getmtime(core_file)
                ).isoformat(timespec="seconds")
        except Exception:
            pass
        return stamp

    def explain_selection(self) -> dict[str, Any]:
        """Explain why each fragment was included or excluded."""
        if self._use_rust:
            result = self._rust.explain_selection()
            return dict(result)
        else:
            return {"error": "Explainability requires Rust engine"}

    def _auto_checkpoint(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create an auto-checkpoint."""
        if self._use_rust:
            co_access = {
                k: dict(v)
                for k, v in self._prefetch._co_access.items()
            }
            # Export full engine state (not empty fragments)
            engine_state = self._rust.export_state()
            # Auto-persist repo-level index alongside checkpoint, but only if
            # this engine is wired to the shared index. Ephemeral engines
            # (config.use_persistent_index=False) MUST NOT write — otherwise
            # they'd overwrite another caller's index with their tiny state.
            if self.config.use_persistent_index:
                try:
                    self._write_gzip_index(self._index_path, engine_state)
                except Exception as e:
                    logger.debug(f"Failed to persist index: {e}")
            checkpoint_path = self._checkpoint_mgr.save(
                fragments=[],
                dedup_fingerprints={},
                co_access_data=co_access,
                current_turn=self._rust.get_turn(),
                metadata={**(metadata or {}), "engine_state": engine_state},
                stats=self.get_stats(),
            )
            if self.config.use_persistent_index:
                self._ensure_gzip_index(self._index_path)
            return checkpoint_path
        else:
            co_access = {
                k: dict(v)
                for k, v in self._prefetch._co_access.items()
            }
            checkpoint_path = self._checkpoint_mgr.save(
                fragments=list(self._fragments.values()),
                dedup_fingerprints=dict(self._dedup._fingerprints),
                co_access_data=co_access,
                current_turn=self._current_turn,
                metadata=metadata,
                stats=self.get_stats(),
            )
            if self.config.use_persistent_index:
                try:
                    self._write_gzip_index(
                        self._index_path,
                        self._python_index_state(),
                    )
                except Exception as exc:
                    logger.debug("Failed to persist Python index: %s", exc)
            return checkpoint_path

    def _validate_checkpoint_dir(self) -> None:
        """Fix #5: Validate checkpoint directory is writable at startup."""
        ckpt_dir = self.config.checkpoint_dir
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            # Write a probe file to confirm the directory is actually writable
            probe = os.path.join(str(ckpt_dir), ".entroly_write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.unlink(probe)
        except OSError as e:
            raise RuntimeError(
                f"Entroly checkpoint directory '{ckpt_dir}' is not writable: {e}.\n"
                f"Set the ENTROLY_DIR env var or pass checkpoint_dir= to EntrolyConfig "
                f"to point to a writable location."
            ) from e

    @staticmethod
    def _ensure_gzip_index(path: str) -> None:
        """Normalize legacy native index writes to gzip.

        Some older ``entroly-core`` wheels wrote plain JSON to
        ``index.json.gz``. The Rust source now writes gzip, but Python users can
        still have an older native wheel installed. Normalizing here keeps the
        public Python contract true: files ending in ``.json.gz`` are readable
        by gzip tools and by the isolation tests.
        """
        p = Path(path)
        try:
            raw = p.read_bytes()
        except OSError:
            return
        if not raw or (len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B):
            return
        tmp = p.with_suffix(p.suffix + ".tmp")
        with gzip.open(tmp, "wb") as f:
            f.write(raw)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, p)

    @staticmethod
    def _write_gzip_index(path: str, state: str | dict[str, Any]) -> None:
        """Persist an exported engine state as real gzip JSON.

        The Python API documents ``index.json.gz`` as gzip-compressed JSON.
        Some published native wheels still write plain JSON from
        ``persist_index``; writing the exported state here makes the disk
        contract independent of the installed native wheel version.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(state, str):
            data = state
        else:
            data = json.dumps(state, separators=(",", ":"))
        tmp = p.with_suffix(p.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(data)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, p)

    # ── Python fallback implementations ──────────────────────────────

    def _python_index_state(self) -> dict[str, Any]:
        """Return a portable warm-start snapshot for the Python engine."""
        return {
            "schema_version": "entroly.python-index.v1",
            "current_turn": self._current_turn,
            "fragments": [
                _fragment_to_dict(fragment)
                for fragment in self._fragments.values()
            ],
            "co_access_data": {
                source: dict(targets)
                for source, targets in self._prefetch._co_access.items()
            },
            "feedback": {
                "success": dict(self._wilson._success),
                "failure": dict(self._wilson._failure),
            },
            "counters": {
                "fragments_ingested": self._total_fragments_ingested,
                "duplicates_caught": self._total_duplicates_caught,
                "optimizations": self._total_optimizations,
                "tokens_saved": self._total_tokens_saved,
            },
        }

    def _load_python_index(self, path: str) -> int:
        """Restore a Python-engine warm-start snapshot, failing closed."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            raw = p.read_bytes()
            if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            state = json.loads(raw.decode("utf-8"))
            if (
                not isinstance(state, dict)
                or state.get("schema_version") != "entroly.python-index.v1"
                or not isinstance(state.get("fragments"), list)
            ):
                return 0
            fragments = [_dict_to_fragment(item) for item in state["fragments"]]
            co_access = state.get("co_access_data", {})
            if not isinstance(co_access, dict):
                co_access = {}
            feedback = state.get("feedback", {})
            if not isinstance(feedback, dict):
                feedback = {}
            counters = state.get("counters", {})
            if not isinstance(counters, dict):
                counters = {}
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError, OSError, TypeError, ValueError, KeyError):
            return 0

        self._fragments = {fragment.fragment_id: fragment for fragment in fragments}
        self._dedup = _PyDedupIndex(
            hamming_threshold=getattr(self.config, "dedup_hamming_threshold", 3)
        )
        from collections import Counter, defaultdict

        self._global_token_counts = Counter()
        self._total_token_count = 0
        for fragment in fragments:
            self._dedup.insert(fragment.fragment_id, fragment.content)
            tokens = fragment.content.lower().split()
            self._global_token_counts.update(tokens)
            self._total_token_count += len(tokens)
        def _nonnegative_int(value: object, default: int = 0) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                return default

        self._total_fragments_ingested = _nonnegative_int(
            counters.get("fragments_ingested"), len(fragments)
        )
        self._total_duplicates_caught = _nonnegative_int(
            counters.get("duplicates_caught")
        )
        self._total_optimizations = _nonnegative_int(counters.get("optimizations"))
        self._total_tokens_saved = _nonnegative_int(counters.get("tokens_saved"))
        self._current_turn = _nonnegative_int(state.get("current_turn"))
        for field, target in (
            ("success", self._wilson._success),
            ("failure", self._wilson._failure),
        ):
            raw_counts = feedback.get(field, {})
            if isinstance(raw_counts, dict):
                target.update({
                    str(fragment_id): _nonnegative_int(count)
                    for fragment_id, count in raw_counts.items()
                })
        restored_co_access: defaultdict[str, Counter] = defaultdict(Counter)
        for source, targets in co_access.items():
            if isinstance(targets, dict):
                restored_co_access[str(source)] = Counter(targets)
        self._prefetch._co_access = restored_co_access
        return len(fragments)

    def _load_index_compat(self, path: str) -> int:
        """Load a persisted index without relying on native gzip support."""
        if not self._use_rust:
            return self._load_python_index(path)
        p = Path(path)
        if not p.exists():
            return 0
        raw = p.read_bytes()
        if not raw:
            return 0
        if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
            state = gzip.decompress(raw).decode("utf-8")
        else:
            state = raw.decode("utf-8")
        self._rust.import_state(state)
        self._write_gzip_index(path, state)
        return int(self._rust.fragment_count())

    def _ingest_python(self, content, source, token_count, is_pinned):
        """Python fallback for ingest (when Rust not available)."""
        # `hashlib` is imported at module scope here; the function-local import
        # this replaced was redundant once the engine moved out of server.py.
        self._total_fragments_ingested += 1

        if token_count <= 0:
            token_count = _calibrated_token_count(content, source)

        # Fix #4 (Python fallback): enforce max_fragments cap
        if len(self._fragments) >= self.config.max_fragments:
            return {
                "status": "rejected",
                "reason": "max_fragments cap reached",
                "max_fragments": self.config.max_fragments,
            }

        frag_id = hashlib.sha256(
            f"{source}:{content[:200]}:{self._total_fragments_ingested}".encode()
        ).hexdigest()[:16]

        dup_id = self._dedup.insert(frag_id, content)
        if dup_id is not None:
            self._total_duplicates_caught += 1
            existing = self._fragments.get(dup_id)
            if existing:
                existing.access_count += 1
                existing.turn_last_accessed = self._current_turn
                max_freq = max(
                    f.access_count for f in self._fragments.values()
                ) or 1
                existing.frequency_score = min(
                    existing.access_count / max_freq, 1.0
                )
            return {
                "status": "duplicate",
                "duplicate_of": dup_id,
                "fragment_id": frag_id,
                "tokens_saved": token_count,
            }

        # Sample up to 50 fragments for entropy comparison (O(n) instead of O(n log n))
        import random as _rng
        all_frags = list(self._fragments.values())
        sample = _rng.sample(all_frags, min(50, len(all_frags))) if len(all_frags) > 50 else all_frags
        other_contents = [f.content for f in sample]
        entropy_score = _py_compute_information_score(
            content,
            global_token_counts=dict(self._global_token_counts),
            total_tokens=self._total_token_count,
            other_fragments=other_contents,
        )

        tokens = content.lower().split()
        self._global_token_counts.update(tokens)
        self._total_token_count += len(tokens)

        frag = ContextFragment(
            fragment_id=frag_id,
            content=content,
            token_count=token_count,
            source=source,
            recency_score=1.0,
            frequency_score=0.0,
            semantic_score=0.0,
            entropy_score=entropy_score,
            turn_created=self._current_turn,
            turn_last_accessed=self._current_turn,
            access_count=1,
            is_pinned=is_pinned,
            simhash=_py_simhash(content),
        )

        self._fragments[frag_id] = frag

        if source:
            self._prefetch.record_access(source, self._current_turn)

        if self._checkpoint_mgr.should_auto_checkpoint():
            self._auto_checkpoint()

        # Fix #3: extract skeleton for code files (Python fallback)
        has_skeleton = False
        skeleton_tc = None
        try:
            if _RUST_AVAILABLE:
                # entroly_core exposes extract_skeleton directly
                from entroly_core import extract_skeleton
                skel = extract_skeleton(content, source)
            else:
                from .skeleton import extract_skeleton  # type: ignore[import]
                skel = extract_skeleton(content, source)
            if skel:
                has_skeleton = True
                skeleton_tc = _calibrated_token_count(skel, source)
        except Exception:
            pass  # skeleton is best-effort; never block ingest

        result: dict[str, Any] = {
            "status": "ingested",
            "fragment_id": frag_id,
            "token_count": token_count,
            "entropy_score": round(entropy_score, 4),
            "total_fragments": len(self._fragments),
            "has_skeleton": has_skeleton,
        }
        if skeleton_tc is not None:
            result["skeleton_token_count"] = skeleton_tc
        return result

    def _optimize_python(self, token_budget, query):
        """Python fallback for optimize."""
        self._total_optimizations += 1

        if query:
            query_hash = _py_simhash(query)
            for frag in self._fragments.values():
                dist = _py_hamming_distance(query_hash, frag.simhash)
                frag.semantic_score = max(0.0, 1.0 - (dist / 64.0))

        # Apply Wilson feedback via the frequency dimension. Wilson learned_value
        # returns [0.5, 2.0] (neutral=1.0). We map this to [0, 1] and use it as
        # the frequency_score, so fragments with positive feedback history get
        # boosted in the knapsack and fragments with negative history get suppressed.
        for frag in self._fragments.values():
            wilson = self._wilson.learned_value(frag.fragment_id)
            frag.frequency_score = max(0.0, min((wilson - 0.5) / 1.5, 1.0))

        fragments = list(self._fragments.values())
        selected, stats = _py_knapsack_optimize(
            fragments,
            token_budget,
            w_recency=self.config.weight_recency,
            w_frequency=self.config.weight_frequency,
            w_semantic=self.config.weight_semantic_sim,
            w_entropy=self.config.weight_entropy,
        )

        total_available_tokens = sum(f.token_count for f in fragments)
        tokens_saved = _honest_tokens_saved(
            [
                {"relevance": _py_compute_relevance(
                    f, self.config.weight_recency, self.config.weight_frequency,
                    self.config.weight_semantic_sim, self.config.weight_entropy),
                 "is_pinned": f.is_pinned}
                for f in selected
            ],
            total_available_tokens - stats["total_tokens"],
        )
        self._total_tokens_saved += max(0, tokens_saved)

        for frag in selected:
            frag.turn_last_accessed = self._current_turn
            frag.access_count += 1

        return {
            "selected_fragments": [
                {
                    "id": f.fragment_id,
                    "source": f.source,
                    "token_count": f.token_count,
                    "relevance": round(
                        _py_compute_relevance(
                            f,
                            self.config.weight_recency,
                            self.config.weight_frequency,
                            self.config.weight_semantic_sim,
                            self.config.weight_entropy,
                        ), 4
                    ),
                    "content_preview": f.content[:100] + "..." if len(f.content) > 100 else f.content,
                    "content": f.content,
                }
                for f in selected
            ],
            # The no-match contract must know whether this result represents a
            # real subset or the complete candidate universe.  Omitting the
            # count made the fallback look like it had generated zero
            # candidates, so the guard could erase a valid all-fragment result
            # as "no match" even though nothing had been displaced.
            "total_fragments": len(fragments),
            "optimization_stats": stats,
            "total_tokens": stats["total_tokens"],
            "tokens_saved_this_call": max(0, tokens_saved),
            "total_tokens_saved_session": self._total_tokens_saved,
        }

    def _recall_python(self, query, top_k):
        """Python fallback for recall."""
        if not self._fragments:
            return []

        query_hash = _py_simhash(query)

        scored = []
        for frag in self._fragments.values():
            dist = _py_hamming_distance(query_hash, frag.simhash)
            frag.semantic_score = max(0.0, 1.0 - (dist / 64.0))
            relevance = _py_compute_relevance(
                frag,
                self.config.weight_recency,
                self.config.weight_frequency,
                self.config.weight_semantic_sim,
                self.config.weight_entropy,
            )
            relevance *= self._wilson.learned_value(frag.fragment_id)
            scored.append((frag, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "fragment_id": f.fragment_id,
                "source": f.source,
                "relevance": round(rel, 4),
                "entropy": round(f.entropy_score, 4),
                "content": f.content,
            }
            for f, rel in scored[:top_k]
        ]

    def _stats_python(self):
        """Python fallback for stats."""
        fragments = list(self._fragments.values())
        total_tokens = sum(f.token_count for f in fragments)
        avg_entropy = (
            sum(f.entropy_score for f in fragments) / len(fragments)
            if fragments else 0.0
        )

        return {
            "session": {
                "current_turn": self._current_turn,
                "total_fragments": len(fragments),
                "total_tokens_tracked": total_tokens,
                "avg_entropy_score": round(avg_entropy, 4),
                "pinned_fragments": sum(1 for f in fragments if f.is_pinned),
            },
            # Engine-internal efficiency telemetry. NOT money saved.
            # `dedup_tokens_avoided` counts tokens the engine did not have to
            # re-ingest because of SimHash dedup. `cache_tokens_served` counts
            # tokens an EGSC cache hit returned without recomputation. Both go
            # up during CLI compile/optimize even when no LLM call ever
            # happened — so neither one can be priced as "cost saved." The
            # only honest cost-saved number lives in value_tracker (lifetime
            # view), which is incremented exclusively by proxy.py when a real
            # request is intercepted.
            "engine": {
                "dedup_tokens_avoided": self._total_tokens_saved,
                "duplicates_caught": self._total_duplicates_caught,
                "optimize_calls": self._total_optimizations,
                "fragments_ingested": self._total_fragments_ingested,
            },
            "savings": {
                "dedup_tokens_avoided": self._total_tokens_saved,
                "total_duplicates_caught": self._total_duplicates_caught,
                "total_optimizations": self._total_optimizations,
                "total_fragments_ingested": self._total_fragments_ingested,
                "baseline": (
                    "dedup/selection telemetry only; counts candidates not "
                    "selected. Not a provider bill delta and not a dollar-"
                    "savings claim."
                ),
            },
            "dedup": self._dedup.stats(),
            "prefetch": self._prefetch.stats(),
            "checkpoint": self._checkpoint_mgr.stats(),
        }


def _recall_path_prior(source: str, query_lower: str) -> float:
    """Mild path-type prior for locator recall.

    BM25 term-frequency lets test/doc/benchmark files (dense in a term like
    "proxy") outrank the implementation an agent asked to locate — dogfooding
    saw ``tests/_proxy_e2e.py`` rank #1 for "where the proxy injects context"
    while ``entroly/proxy.py`` fell off the top-8. Gently demote those file
    kinds unless the query is explicitly about them, so the impl surfaces
    without burying a genuinely best-matching test/doc.
    """
    s = source.lower().removeprefix("file:").replace("\\", "/")
    segments = set(s.split("/"))
    base = s.rsplit("/", 1)[-1]
    is_test = (
        bool(segments & {"tests", "test"})
        or base.startswith("test_") or "_test." in base
        or ".servertest" in s or "e2e" in base
    )
    if is_test:
        return 1.0 if "test" in query_lower else 0.75
    if segments & {"benchmarks", "bench"}:
        return 1.0 if "bench" in query_lower else 0.8
    if s.endswith((".md", ".rst", ".txt")) or "docs" in segments:
        wants_docs = any(t in query_lower for t in ("doc", "readme", "guide"))
        return 1.0 if wants_docs else 0.85
    return 1.0  # implementation / source


def _apply_recall_path_prior(
    results: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """Re-rank recall results by (relevance x path prior), stable and mild."""
    ql = (query or "").lower()

    def _key(raw: dict[str, Any]) -> float:
        base = raw.get("relevance", raw.get("score", raw.get("relevance_score", 0.0)))
        try:
            base = float(base)
        except (TypeError, ValueError):
            base = 0.0
        src = str(
            raw.get("source") or raw.get("source_path") or raw.get("path") or ""
        )
        return base * _recall_path_prior(src, ql)

    return sorted(
        (r for r in results if isinstance(r, dict)),
        key=_key,
        reverse=True,
    )


