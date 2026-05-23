"""
Entroly Verifiers — Hallucination Detection Layer
==================================================

Composable, fail-closed verifiers that catch model outputs entroly's
context layer alone cannot ground.

Six verification layers, each targeting a distinct hallucination class:

  Layer 1 — symbol_resolution (GRAPHS):
    Invented-API detector. Bayesian posterior over symbol manifest
    membership + codebase-conditioned n-gram surprisal.

  Layer 2 — scope_analyzer:
    3-state reachability model (grounded / unreachable / hallucinated).
    Catches real symbols that aren't imported.

  Layer 3 — ngram_model:
    Character 4-gram LM with Stupid Backoff (Brants 2007).
    Codebase-conditioned naming convention anomaly detection.

  Layer 4 — type_check (pyright):
    Semantic verification via deterministic type checking.
    Catches wrong kwargs, arity violations, type mismatches.

  Layer 5 — calibrator:
    Online SGD calibration of λ per task archetype.
    Closes the feedback loop on false positives/negatives.

  Layer 6 — PROVE / CAVE / TRIAD (new):
    - semantic_entropy (PROVE): prose hallucination via causal-weighted
      predicate alignment (Kuhn/Gal/Farquhar ICLR 2023, adapted)
    - reasoning_chain (CAVE): counterfactual decorative-premise ablation
      (Lightman 2023 PRM + Shi ICML 2023, adapted)
    - commit_alignment (TRIAD): diff-message-PR triangulation via
      three-signal Bayesian combination
"""

from .symbol_resolution import (
    SymbolManifest,
    SymbolVerifierResult,
    SymbolVerifier,
    verify_code,
)
from .ngram_model import CharNGramModel
from .scope_analyzer import (
    ReverseIndex,
    Scope,
    ReachabilityVerdict,
    build_reverse_index,
    compute_scope,
    judge_reachability,
)
from .calibrator import Calibrator, infer_archetype_from_query
from .service import VerifierService, ExtendedResult, ExtendedJudgment
from .semantic_entropy import prove_verify, ProveResult
from .reasoning_chain import cave_verify, CaveResult
from .commit_alignment import triad_verify, TriadResult
from .provenance_tracer import trace_provenance, BIPTResult
from .repair_loop import forge_loop, ForgeResult
from .lang_js import (
    JSSymbolReference,
    extract_js_definitions,
    extract_js_scope,
    extract_js_symbol_refs,
    collect_js_repo_symbols,
    collect_js_installed_packages,
)

__all__ = [
    # Core — Layer 1 (GRAPHS)
    "SymbolManifest",
    "SymbolVerifierResult",
    "SymbolVerifier",
    "CharNGramModel",
    "verify_code",
    # Scope / reachability — Layer 2
    "ReverseIndex",
    "Scope",
    "ReachabilityVerdict",
    "build_reverse_index",
    "compute_scope",
    "judge_reachability",
    # Calibration — Layer 5
    "Calibrator",
    "infer_archetype_from_query",
    # Daemon-resident service
    "VerifierService",
    "ExtendedResult",
    "ExtendedJudgment",
    # PROVE — Layer 6a (prose hallucination)
    "prove_verify",
    "ProveResult",
    # CAVE — Layer 6b (reasoning chain)
    "cave_verify",
    "CaveResult",
    # TRIAD — Layer 6c (commit alignment)
    "triad_verify",
    "TriadResult",
    # BIPT — Layer 7 (byte-level provenance)
    "trace_provenance",
    "BIPTResult",
    # FORGE — Layer 8 (hallucination suppression)
    "forge_loop",
    "ForgeResult",
    # JS / TS symbol extraction (companion to symbol_resolution.py)
    "JSSymbolReference",
    "extract_js_definitions",
    "extract_js_scope",
    "extract_js_symbol_refs",
    "collect_js_repo_symbols",
    "collect_js_installed_packages",
]

