"""
FORGE -- Feedback-Oriented Regeneration with Grounding Enforcement
===================================================================

The repair loop that moves Entroly from hallucination DETECTOR
to hallucination SUPPRESSOR.

Prior Art & What's Different
-----------------------------
- Self-Refine (Madaan 2023): LLM critiques own output -> LLM is its
  own verifier -> circular (hallucinates about hallucinations)
- Reflexion (Shinn 2023): stores verbal reflections -> still relies
  on LLM judgment for what went wrong
- Self-RAG (Asai 2024): retrieves on demand -> but retrieval is
  triggered by the LLM, not by verification failure

FORGE is different:
  1. VERIFICATION is external (BIPT/GRAPHS/PROVE) -- not LLM self-critique
  2. RETRIEVAL is guided by rejection reasons -- not LLM-chosen queries
  3. CONVERGENCE is mathematically guaranteed -- IPD monotonically decreases
     or the loop terminates

The Loop
--------
    O_0 = LLM(prompt, context_0)

    for t in 1..max_iters:
        rejections = verify(O_{t-1})
        if rejections.ipd < threshold:
            return O_{t-1}  # GROUNDED

        # Novel step: rejection-guided retrieval
        grounding_patches = retrieve(rejections, codebase)
        context_t = context_{t-1} + grounding_patches + rejection_feedback

        O_t = LLM(prompt, context_t)

        # Convergence check
        if IPD(O_t) >= IPD(O_{t-1}):
            return O_t  # NOT CONVERGING -- stop

    return O_{max_iters}  # BUDGET EXHAUSTED

Convergence Guarantee
---------------------
Define Grounding Convergence Rate:

    GCR(t) = (IPD(t-1) - IPD(t)) / IPD(t-1)

If GCR(t) < epsilon for any iteration, we halt. This prevents
infinite loops and wasted tokens. Expected iterations:

    E[T] = log(IPD_0 / IPD_target) / log(1 / (1 - GCR_avg))

For typical values (IPD_0=0.7, IPD_target=0.1, GCR_avg=0.5):
    E[T] = log(7) / log(2) ~ 2.8 iterations

References
----------
- Madaan et al. (2023): Self-Refine
- Shinn et al. (2023): Reflexion
- Asai et al. (2024): Self-RAG
- Kolmogorov (1965): K(O|C) as grounding metric (BIPT foundation)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .provenance_tracer import trace_provenance, BIPTResult, ProvenanceTrace
from .symbol_resolution import SymbolManifest


# =====================================================================
# Core Data Types
# =====================================================================


@dataclass
class Rejection:
    """A single verified rejection with retrieval-actionable metadata."""
    identifier: str          # the rejected identifier name
    kind: str                # "ungrounded", "not_in_manifest", "type_error"
    source_layer: str        # "BIPT", "GRAPHS", "PROVE", "PYRIGHT"
    grounding_ratio: float   # 0.0 = fully invented, 1.0 = fully grounded
    retrieval_query: str     # auto-generated query to find grounding
    suggested_fix: str       # human-readable fix suggestion


@dataclass
class GroundingPatch:
    """Context retrieved to ground a specific rejection."""
    rejection: Rejection
    retrieved_context: str   # the actual code/docs retrieved
    source_file: str         # where it came from
    confidence: float        # retrieval confidence [0, 1]


@dataclass
class ForgeIteration:
    """One iteration of the repair loop."""
    iteration: int
    output: str
    ipd: float
    n_rejections: int
    n_patches: int
    gcr: float               # grounding convergence rate
    bipt_result: BIPTResult | None


@dataclass
class ForgeResult:
    """Complete FORGE repair loop result."""
    original_output: str
    final_output: str
    original_ipd: float
    final_ipd: float
    iterations: list[ForgeIteration]
    total_iterations: int
    converged: bool           # did IPD fall below threshold?
    rejections_resolved: int
    rejections_remaining: int
    total_context_added: int  # bytes of grounding patches injected

    def explain(self) -> str:
        lines = [
            "=== FORGE -- Hallucination Suppression Loop ===",
            f"iterations: {self.total_iterations}  "
            f"converged: {self.converged}",
            f"IPD: {self.original_ipd:.3f} -> {self.final_ipd:.3f}  "
            f"reduction: {(1 - self.final_ipd / max(self.original_ipd, 1e-9)):.0%}",
            f"rejections: {self.rejections_resolved} resolved, "
            f"{self.rejections_remaining} remaining",
            f"context added: {self.total_context_added:,} bytes",
            "",
        ]
        for it in self.iterations:
            gcr_str = f"{it.gcr:+.1%}" if it.iteration > 0 else "---"
            lines.append(
                f"  [{it.iteration}] IPD={it.ipd:.3f}  "
                f"GCR={gcr_str}  "
                f"rejections={it.n_rejections}  "
                f"patches={it.n_patches}"
            )
        return "\n".join(lines)


# =====================================================================
# Rejection Extraction -- Translates verifier output to retrieval queries
# =====================================================================


def _extract_rejections(
    bipt_result: BIPTResult,
    manifest: SymbolManifest | None = None,
) -> list[Rejection]:
    """Convert BIPT and GRAPHS results into actionable rejections.

    The key innovation: each rejection carries a RETRIEVAL QUERY
    that can be used to find the missing grounding in the codebase.
    """
    rejections: list[Rejection] = []

    # BIPT rejections: identifiers with low grounding
    for trace in bipt_result.traces:
        if trace.verdict in ("invented", "partial"):
            name = trace.identifier.name

            # Generate retrieval query from the identifier
            # Split compound names: "process_split_payment" -> "process split payment"
            query_tokens = re.sub(r"[_.]", " ", name).lower().split()
            retrieval_query = " ".join(query_tokens)

            rejections.append(Rejection(
                identifier=name,
                kind="ungrounded",
                source_layer="BIPT",
                grounding_ratio=trace.grounding_ratio,
                retrieval_query=retrieval_query,
                suggested_fix=f"Use an identifier from the codebase instead of '{name}'",
            ))

    # GRAPHS rejections: symbols not in manifest
    if manifest is not None:
        for trace in bipt_result.traces:
            name = trace.identifier.name
            if name not in manifest and trace.identifier.kind in ("name", "attr", "import"):
                # Check if already in BIPT rejections
                if not any(r.identifier == name for r in rejections):
                    rejections.append(Rejection(
                        identifier=name,
                        kind="not_in_manifest",
                        source_layer="GRAPHS",
                        grounding_ratio=0.0,
                        retrieval_query=name,
                        suggested_fix=f"'{name}' is not defined in the codebase",
                    ))

    return rejections


# =====================================================================
# Grounding Retrieval -- Fetches context to fix rejections
# =====================================================================


class ContextStore(Protocol):
    """Interface for retrieving grounding context from the codebase."""
    def search(self, query: str, max_results: int = 3) -> list[tuple[str, str, float]]:
        """Search codebase for relevant context.

        Returns list of (content, source_file, confidence) tuples.
        """
        ...


class SimpleContextStore:
    """Basic context store using substring matching.

    In production, this would be backed by Entroly's retrieval engine
    (embedding search, BM25, or the existing SKS index).
    """

    def __init__(self, files: dict[str, str]):
        """files: {filename: content}"""
        self._files = files

    def search(self, query: str, max_results: int = 3) -> list[tuple[str, str, float]]:
        results: list[tuple[str, str, float]] = []
        query_tokens = set(query.lower().split())

        for filename, content in self._files.items():
            content_lower = content.lower()
            # Score: fraction of query tokens found in content
            found = sum(1 for t in query_tokens if t in content_lower)
            if found > 0:
                score = found / len(query_tokens) if query_tokens else 0
                results.append((content, filename, score))

        results.sort(key=lambda x: -x[2])
        return results[:max_results]


def _retrieve_grounding(
    rejections: list[Rejection],
    store: ContextStore,
    max_patches: int = 10,
) -> list[GroundingPatch]:
    """For each rejection, retrieve context that could ground it.

    This is the novel step: verifier rejection reasons become
    retrieval queries. The BIPT tells us WHAT is missing, and
    we use that to find WHERE it should come from.
    """
    patches: list[GroundingPatch] = []
    seen_queries: set[str] = set()

    for rej in rejections:
        if len(patches) >= max_patches:
            break

        query = rej.retrieval_query
        if query in seen_queries or len(query) < 2:
            continue
        seen_queries.add(query)

        results = store.search(query, max_results=1)
        if results:
            content, source, confidence = results[0]
            patches.append(GroundingPatch(
                rejection=rej,
                retrieved_context=content,
                source_file=source,
                confidence=confidence,
            ))

    return patches


def _extract_intent_keywords(prompt: str) -> list[str]:
    """Extract intent-bearing keywords from a prompt.

    Uses the same keyword patterns as the existing EpistemicRouter
    and PROVE verb clusters to identify what the user WANTS.

    Examples:
        "Send email in the background" -> ["send", "background", "async"]
        "Run requests concurrently"    -> ["run", "concurrent"]
    """
    # Intent signal keywords (from epistemic_router._INTENT_PATTERNS
    # and PROVE verb clusters, composed here without circular imports)
    _ASYNC_SIGNALS = {"background", "async", "concurrent", "parallel",
                      "non-blocking", "deferred", "queue", "worker"}
    _SYNC_SIGNALS = {"synchronous", "blocking", "sequential", "wait"}
    _TX_SIGNALS = {"transaction", "atomic", "commit", "rollback", "begin"}

    words = set(re.sub(r"[^a-zA-Z]", " ", prompt.lower()).split())

    keywords = []
    if words & _ASYNC_SIGNALS:
        keywords.append("async")
    if words & _SYNC_SIGNALS:
        keywords.append("sync")
    if words & _TX_SIGNALS:
        keywords.append("transaction")

    # Also extract action verbs from the prompt
    _ACTION_VERBS = {"send", "fetch", "load", "save", "cache", "read",
                     "write", "create", "delete", "update", "process",
                     "dispatch", "merge", "split", "run", "build"}
    keywords.extend(sorted(words & _ACTION_VERBS))

    return keywords


def _rank_apis_by_intent(
    keywords: list[str],
    context: str,
) -> list[tuple[str, float]]:
    """Rank available APIs from context by semantic fit to intent keywords.

    Scores each function/class name by how many intent keywords appear
    in its name (substring match). This leverages the naming convention
    insight: well-named APIs encode their semantics (send_email_async
    contains both "send" and "async").
    """
    # Extract function/class names from context
    api_names: list[str] = re.findall(
        r'(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', context
    )
    if not api_names:
        return []

    scored: list[tuple[str, float]] = []
    for name in api_names:
        name_lower = name.lower()
        # Score = fraction of intent keywords present in name
        matches = sum(1 for kw in keywords if kw in name_lower)
        score = matches / len(keywords) if keywords else 0
        scored.append((name, score))

    scored.sort(key=lambda x: -x[1])
    return scored


def compose_repair_prompt(
    original_prompt: str,
    previous_output: str,
    rejections: list[Rejection],
    patches: list[GroundingPatch],
) -> tuple[str, str]:
    """Compose the system + user prompts for the repair iteration.

    Returns (system_prompt, user_prompt).

    The system prompt injects grounding context + intent-ranked APIs.
    The user prompt explains what was wrong and asks for a fix.
    """
    # System: inject grounding patches as additional context
    grounding_lines = []
    for patch in patches:
        grounding_lines.append(
            f"# Available code from {patch.source_file}:\n"
            f"{patch.retrieved_context}\n"
        )
    grounding_context = "\n".join(grounding_lines)

    # Intent-aware API ranking: extract what the user WANTS,
    # then rank available APIs by semantic fit
    intent_keywords = _extract_intent_keywords(original_prompt)
    if intent_keywords and grounding_context:
        ranked = _rank_apis_by_intent(intent_keywords, grounding_context)
        if ranked:
            best = [f"  {name} (fit={score:.0%})" for name, score in ranked[:5] if score > 0]
            if best:
                grounding_context += (
                    "\n# APIs ranked by semantic fit to your request "
                    f"(intent: {', '.join(intent_keywords)}):\n"
                    + "\n".join(best)
                    + "\n# Prefer higher-ranked APIs.\n"
                )

    system = (
        "You are a Python developer. "
        "Use ONLY the APIs, classes, and functions shown in the codebase below. "
        "Do NOT invent functions, methods, or fields that are not defined.\n\n"
        f"{grounding_context}"
    )

    # User: explain rejections + ask for fix
    rejection_list = []
    for rej in rejections[:8]:  # cap to avoid prompt bloat
        rejection_list.append(
            f"- '{rej.identifier}': {rej.suggested_fix}"
        )
    rejection_text = "\n".join(rejection_list)

    user = (
        f"Your previous code had the following issues:\n"
        f"{rejection_text}\n\n"
        f"Original request: {original_prompt}\n\n"
        f"Please rewrite the code using ONLY the APIs shown in the context above. "
        f"Write only Python code."
    )

    return system, user


# =====================================================================
# FORGE Loop -- The Core Innovation
# =====================================================================


# Configuration
DEFAULT_MAX_ITERS = 3        # at most 3 repair attempts
DEFAULT_IPD_THRESHOLD = 0.20 # IPD below this = grounded
DEFAULT_GCR_EPSILON = 0.05   # if IPD improves by less than 5%, stop


def forge_loop(
    prompt: str,
    initial_context: str,
    generate_fn: Callable[[str, str], str],
    context_store: ContextStore | None = None,
    manifest: SymbolManifest | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
    ipd_threshold: float = DEFAULT_IPD_THRESHOLD,
    gcr_epsilon: float = DEFAULT_GCR_EPSILON,
) -> ForgeResult:
    """Run the FORGE repair loop.

    Args:
        prompt: The user's original request.
        initial_context: The initial context provided to the LLM.
        generate_fn: Callable(system_prompt, user_prompt) -> code.
                     This is the LLM generation function.
        context_store: Codebase search interface for grounding retrieval.
        manifest: Symbol manifest for GRAPHS verification.
        max_iters: Maximum repair iterations.
        ipd_threshold: IPD below this = considered grounded.
        gcr_epsilon: Minimum improvement rate to continue.

    Returns:
        ForgeResult with full iteration history and final output.
    """
    iterations: list[ForgeIteration] = []
    total_context_added = 0

    # Iteration 0: initial generation
    system_0 = "You are a Python developer. Here is the codebase:\n" + initial_context
    output = generate_fn(system_0, prompt)
    code = _extract_code_blocks(output)

    bipt = trace_provenance(code, initial_context)
    prev_ipd = bipt.ipd

    iterations.append(ForgeIteration(
        iteration=0,
        output=code,
        ipd=bipt.ipd,
        n_rejections=sum(1 for t in bipt.traces if t.verdict in ("invented", "partial")),
        n_patches=0,
        gcr=0.0,
        bipt_result=bipt,
    ))

    original_output = code
    original_ipd = bipt.ipd
    initial_rejections = iterations[0].n_rejections

    # Check if already grounded
    if bipt.ipd <= ipd_threshold:
        return ForgeResult(
            original_output=original_output,
            final_output=code,
            original_ipd=original_ipd,
            final_ipd=bipt.ipd,
            iterations=iterations,
            total_iterations=1,
            converged=True,
            rejections_resolved=0,
            rejections_remaining=iterations[0].n_rejections,
            total_context_added=0,
        )

    # Repair iterations
    accumulated_context = initial_context

    for t in range(1, max_iters + 1):
        # Step 1: Extract rejections
        rejections = _extract_rejections(bipt, manifest)

        if not rejections:
            break  # nothing to fix

        # Step 2: Retrieve grounding patches
        if context_store is not None:
            patches = _retrieve_grounding(rejections, context_store)
        else:
            patches = []

        # Step 3: Add grounding patches to accumulated context
        for patch in patches:
            patch_text = f"\n# From {patch.source_file}:\n{patch.retrieved_context}\n"
            accumulated_context += patch_text
            total_context_added += len(patch_text.encode("utf-8"))

        # Step 4: Compose repair prompt
        system, user = compose_repair_prompt(
            prompt, code, rejections, patches,
        )

        # Step 5: Regenerate
        output = generate_fn(system, user)
        code = _extract_code_blocks(output)

        # Step 6: Verify again
        bipt = trace_provenance(code, accumulated_context)

        # Step 7: Compute convergence rate
        if prev_ipd > 0:
            gcr = (prev_ipd - bipt.ipd) / prev_ipd
        else:
            gcr = 0.0

        n_rej = sum(1 for t2 in bipt.traces if t2.verdict in ("invented", "partial"))

        iterations.append(ForgeIteration(
            iteration=t,
            output=code,
            ipd=bipt.ipd,
            n_rejections=n_rej,
            n_patches=len(patches),
            gcr=gcr,
            bipt_result=bipt,
        ))

        # Step 8: Check convergence
        if bipt.ipd <= ipd_threshold:
            break  # GROUNDED

        if gcr < gcr_epsilon:
            break  # NOT CONVERGING -- stop wasting tokens

        prev_ipd = bipt.ipd

    # Final stats
    final = iterations[-1]
    return ForgeResult(
        original_output=original_output,
        final_output=final.output,
        original_ipd=original_ipd,
        final_ipd=final.ipd,
        iterations=iterations,
        total_iterations=len(iterations),
        converged=final.ipd <= ipd_threshold,
        rejections_resolved=max(0, initial_rejections - final.n_rejections),
        rejections_remaining=final.n_rejections,
        total_context_added=total_context_added,
    )


def _extract_code_blocks(text: str) -> str:
    """Extract Python code blocks from markdown-formatted LLM output."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks)
    return text
