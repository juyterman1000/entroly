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

from typing import Any

from .universal_compress import (
    detect_content_type,
    tfidf_extractive_summarize,
    universal_compress,
)


def _track_savings(before_tokens: int, after_tokens: int, label: str) -> None:
    """Record SDK compression value to the shared telemetry sink so the
    dashboard reflects SDK-only users (previously a hard blank). Strictly
    fail-open: telemetry must NEVER affect compress() output or raise.

    `model=""` on purpose: it books the default cost rate without
    emitting the unknown-model warning on every compress() call."""
    try:
        saved = int(before_tokens) - int(after_tokens)
        if saved <= 0:
            return
        from .value_tracker import get_tracker
        tracker = get_tracker()
        tracker.record(tokens_saved=saved, model="", optimized=True)
        tracker.record_event(
            "compress", f"SDK {label}: saved {saved:,} tokens",
            source="sdk", tokens_saved=saved,
        )
    except Exception:  # noqa: BLE001 — telemetry is best-effort only
        pass


def compress(
    content: str,
    budget: int | None = None,
    content_type: str | None = None,
    target_ratio: float = 0.3,
) -> str:
    """Compress any content to fit within a token budget.

    This is the simplest possible API — one function, one import.

    Args:
        content: Any text content (code, prose, JSON, logs, emails, etc.)
        budget: Target token count. If set, overrides target_ratio.
        content_type: Optional hint ("json", "code", "prose", "log", etc.)
        target_ratio: Compression ratio if budget not specified (0.3 = keep 30%)

    Returns:
        Compressed text that preserves the most important information.

    Example::

        from entroly import compress

        # Compress a large API response
        compressed = compress(api_response, budget=1000)

        # Compress code with type hint
        compressed = compress(source_code, budget=2000, content_type="code")
    """
    if not content:
        return content

    # Estimate current token count (4 chars ≈ 1 token)
    current_tokens = len(content) // 4

    # If budget specified, compute target ratio from it
    if budget is not None and current_tokens > budget:
        target_ratio = max(0.05, budget / max(current_tokens, 1))
    elif budget is not None and current_tokens <= budget:
        return content  # Already within budget

    # Handle code content with the Rust engine if available
    if content_type == "code" or (content_type is None and _looks_like_code(content)):
        try:
            out = _compress_code(content, target_ratio)
            _track_savings(current_tokens, len(out) // 4, "compress(code)")
            return out
        except Exception:
            pass  # Fall through to universal compressor

    compressed, _, _ = universal_compress(content, target_ratio, content_type)
    _track_savings(current_tokens, len(compressed) // 4, "compress")
    return compressed


def compress_messages(
    messages: list[dict[str, Any]],
    budget: int = 50_000,
    preserve_last_n: int = 4,
) -> list[dict[str, Any]]:
    """Compress a conversation message list to fit within a token budget.

    Preserves the most recent messages verbatim and progressively
    compresses older messages using content-aware compression.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        budget: Target total token count for all messages
        preserve_last_n: Number of most recent messages to keep verbatim

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
    total_tokens = sum(
        len(m.get("content", "")) // 4
        for m in messages if isinstance(m.get("content"), str)
    )
    if total_tokens <= budget:
        return messages  # Already within budget

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
    remaining_budget = max(budget - recent_tokens, 500)

    # If recent alone busts budget, compress even recent messages
    # (except the very last user message)
    if recent_tokens > budget:
        return _compress_all_messages(messages, budget)

    if not older:
        # All messages are "recent" — compress all except the last
        return _compress_all_messages(messages, budget)

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
        # Tool results get more aggressive compression
        msg_ratio = ratio * 0.5 if role in ("tool", "function") else ratio

        compressed_content = compress(content, target_ratio=msg_ratio)
        new_msg = dict(msg)
        new_msg["content"] = compressed_content
        result.append(new_msg)

    result.extend(recent)
    return result


def _compress_all_messages(
    messages: list[dict[str, Any]], budget: int
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
    ratio = max(0.05, budget / max(total_tokens, 1))

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

        compressed_content = compress(content, target_ratio=msg_ratio)
        new_msg = dict(msg)
        new_msg["content"] = compressed_content
        result.append(new_msg)

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
    """Select the optimal context subset using PRISM 5D + knapsack DP.

    Same algorithm as MCP optimize_context and proxy context injection,
    exposed as a simple function call for pip users.

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
