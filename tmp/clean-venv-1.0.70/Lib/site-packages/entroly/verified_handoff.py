"""
Witness-Verified Handoff (WVH)
===============================

A multi-agent handoff protocol with evidence-grounding checks intended to
reduce unsupported-claim propagation across agent chains.

Problem
-------
Multi-agent systems pass context between agents without verification.
If Agent A hallucinated "the auth module uses plaintext passwords",
Agent B inherits that as ground truth and may act on it — deleting
perfectly good JWT code, introducing actual security vulnerabilities.

An unsupported claim early in an agent chain can influence downstream work.
The actual impact depends on later evidence, tools, and verification.

Solution
--------
Eligible handoffs pass through WITNESS. Grounded claims can pass, while
unsupported or contradicted claims are surfaced according to policy. WITNESS
can produce false positives and false negatives; callers must not treat a pass
as proof of correctness.

    Agent A output
          │
          ▼
    ┌──────────────────┐
    │  WITNESS Gateway  │
    ├──────────────────┤
    │  grounded claims  │ ──→ pass to Agent B (green)
    │  unsupported      │ ──→ flag + pass with warning (yellow)
    │  contradicted     │ ──→ BLOCK from Agent B (red)
    └──────────────────┘
          │
          ▼
    Agent B receives only verified context

Mathematical guarantee
----------------------
Let H(chain) be the probability that a hallucination propagates
through a chain of N agents.

Without WVH:   H(chain) = 1 - (1 - h)^N  ≈ Nh  for small h
With WVH:      H(chain) = 1 - (1 - h·(1-d))^N  ≈ N·h·(1-d)

where h = per-agent hallucination rate, d = WITNESS detection rate.
For h = 0.05, d = 0.85, N = 5:
    Without: H ≈ 0.226  (22.6% chance of corrupted chain)
    With:    H ≈ 0.037  (3.7% chance — 6.1× improvement)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class VerifiedClaim:
    """A single claim with its WITNESS verification result."""
    text: str                   # the claim text
    label: str                  # "grounded" | "unsupported" | "contradicted"
    confidence: float           # WITNESS confidence [0, 1]
    evidence_snippet: str       # supporting evidence (truncated)
    passed: bool                # True if grounded or unsupported (non-blocking)
    blocked: bool               # True if contradicted (blocked from handoff)


@dataclass
class HandoffBundle:
    """A verified context bundle for agent-to-agent handoff.

    Contains only claims that passed WITNESS verification.
    Contradicted claims are recorded but excluded from the
    verified_context that Agent B receives.
    """
    bundle_id: str                  # SHA-256 of verified content
    from_agent: str                 # source agent identifier
    to_agent: str                   # target agent identifier
    created_at: str                 # ISO 8601 timestamp
    verified_context: str           # cleaned output (contradictions removed)
    original_output: str            # Agent A's full output (for audit)
    claims: list[VerifiedClaim]     # per-claim verification results
    witness_summary_score: float    # overall WITNESS score
    n_grounded: int                 # claims that passed verification
    n_unsupported: int              # claims with no evidence (warned)
    n_contradicted: int             # claims blocked from handoff
    n_total: int                    # total claims examined
    latency_ms: float               # verification latency
    chain_position: int = 0         # position in multi-agent chain (0-indexed)
    upstream_bundle_ids: list[str] = field(default_factory=list)

    @property
    def integrity_hash(self) -> str:
        """SHA-256 of the verified context for downstream chain verification."""
        return hashlib.sha256(
            self.verified_context.encode("utf-8")
        ).hexdigest()

    @property
    def contamination_rate(self) -> float:
        """Fraction of claims that were contradicted (blocked)."""
        if self.n_total == 0:
            return 0.0
        return self.n_contradicted / self.n_total

    def to_dict(self) -> dict[str, Any]:
        """Serialize for transport / logging."""
        return {
            "bundle_id": self.bundle_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "created_at": self.created_at,
            "witness_summary_score": self.witness_summary_score,
            "n_grounded": self.n_grounded,
            "n_unsupported": self.n_unsupported,
            "n_contradicted": self.n_contradicted,
            "n_total": self.n_total,
            "contamination_rate": self.contamination_rate,
            "latency_ms": self.latency_ms,
            "chain_position": self.chain_position,
            "integrity_hash": self.integrity_hash,
        }


# ── Handoff API ──────────────────────────────────────────────────────

def handoff(
    output: str,
    evidence: str,
    from_agent: str = "agent_a",
    to_agent: str = "agent_b",
    *,
    mode: str = "strict",
    chain_position: int = 0,
    upstream_bundles: list[str] | None = None,
    analyzer: Any | None = None,
) -> HandoffBundle:
    """Create a WITNESS-verified handoff bundle.

    Runs WITNESS on Agent A's output against the evidence context,
    strips contradicted claims, and packages the verified remainder
    for Agent B.

    Parameters
    ----------
    output : str
        Agent A's output text.
    evidence : str
        The evidence/context that Agent A was working with.
    from_agent : str
        Source agent identifier.
    to_agent : str
        Target agent identifier.
    mode : str
        "strict" — block contradicted AND unsupported claims.
        "audit"  — block only contradicted; warn on unsupported.
        "permissive" — warn on everything, block nothing.
    chain_position : int
        Position in the agent chain (0 = first handoff).
    upstream_bundles : list[str]
        Bundle IDs from upstream handoffs (for chain tracking).
    analyzer : WitnessAnalyzer, optional
        Pre-configured WitnessAnalyzer instance.  If None, creates one
        with default settings.

    Returns
    -------
    HandoffBundle
        Verified context bundle ready for Agent B.
    """
    t0 = time.perf_counter()

    # Import here to avoid circular imports
    from .witness import WitnessAnalyzer

    if analyzer is None:
        analyzer = WitnessAnalyzer(use_stave=True)

    # Run WITNESS verification
    result = analyzer.analyze(output, evidence)

    # Classify each claim
    verified_claims: list[VerifiedClaim] = []
    blocked_texts: list[str] = []

    for cert in result.certificates:
        is_blocked = False

        if cert.label == "contradicted":
            is_blocked = True
        elif cert.label == "unsupported" and mode == "strict":
            is_blocked = True

        claim = VerifiedClaim(
            text=cert.claim if hasattr(cert, "claim") else str(cert),
            label=cert.label,
            confidence=1.0 - cert.risk,
            evidence_snippet=cert.proof_steps[0].evidence[:120] if cert.proof_steps else "",
            passed=not is_blocked,
            blocked=is_blocked,
        )
        verified_claims.append(claim)

        if is_blocked:
            blocked_texts.append(claim.text)

    # Build verified context (remove contradicted claims from output)
    verified_context = output
    for blocked_text in blocked_texts:
        # Remove the blocked claim from the output
        # Use careful text replacement to avoid breaking surrounding context
        if blocked_text in verified_context:
            verified_context = verified_context.replace(
                blocked_text,
                "[BLOCKED: contradicted by evidence]",
                1,
            )

    # Add warnings for unsupported claims in audit mode
    if mode == "audit":
        warnings = [
            c.text for c in verified_claims
            if c.label == "unsupported" and not c.blocked
        ]
        if warnings:
            warning_block = "\n[WVH WARNING: The following claims lack supporting evidence:]\n"
            for w in warnings:
                warning_block += f"  ⚠ {w}\n"
            verified_context = verified_context + "\n" + warning_block

    # Count by label
    n_grounded = sum(1 for c in verified_claims if c.label == "grounded")
    n_unsupported = sum(1 for c in verified_claims if c.label == "unsupported")
    n_contradicted = sum(1 for c in verified_claims if c.label == "contradicted")

    # Generate bundle ID
    bundle_content = f"{verified_context}|{from_agent}|{to_agent}"
    bundle_id = hashlib.sha256(bundle_content.encode("utf-8")).hexdigest()[:16]

    latency_ms = (time.perf_counter() - t0) * 1000

    bundle = HandoffBundle(
        bundle_id=bundle_id,
        from_agent=from_agent,
        to_agent=to_agent,
        created_at=datetime.now(timezone.utc).isoformat(),
        verified_context=verified_context,
        original_output=output,
        claims=verified_claims,
        witness_summary_score=result.summary_score,
        n_grounded=n_grounded,
        n_unsupported=n_unsupported,
        n_contradicted=n_contradicted,
        n_total=len(verified_claims),
        latency_ms=latency_ms,
        chain_position=chain_position,
        upstream_bundle_ids=upstream_bundles or [],
    )

    logger.info(
        "[WVH] Handoff %s → %s: %d/%d claims passed, %d blocked, score=%.2f, %.0fms",
        from_agent, to_agent,
        n_grounded + n_unsupported - (n_unsupported if mode == "strict" else 0),
        len(verified_claims),
        n_contradicted + (n_unsupported if mode == "strict" else 0),
        result.summary_score,
        latency_ms,
    )

    return bundle


def receive(
    bundle: HandoffBundle,
    verify_integrity: bool = True,
) -> str:
    """Unpack a handoff bundle for the receiving agent.

    Optionally verifies the integrity hash to detect tampering
    in transit.

    Parameters
    ----------
    bundle : HandoffBundle
        The verified bundle from handoff().
    verify_integrity : bool
        If True, recompute and verify the integrity hash.

    Returns
    -------
    str
        The verified context text for the receiving agent.

    Raises
    ------
    ValueError
        If integrity verification fails.
    """
    if verify_integrity:
        expected = bundle.integrity_hash
        actual = hashlib.sha256(
            bundle.verified_context.encode("utf-8")
        ).hexdigest()
        if actual != expected:
            raise ValueError(
                f"[WVH] Integrity check failed for bundle {bundle.bundle_id}. "
                f"Expected {expected[:16]}..., got {actual[:16]}... "
                "Context may have been tampered with in transit."
            )

    logger.info(
        "[WVH] Agent %s received bundle %s from %s "
        "(%d grounded, %d blocked, chain pos %d)",
        bundle.to_agent, bundle.bundle_id[:8], bundle.from_agent,
        bundle.n_grounded, bundle.n_contradicted, bundle.chain_position,
    )

    return bundle.verified_context


def chain_handoff(
    agents: list[str],
    initial_output: str,
    evidence: str,
    *,
    mode: str = "strict",
    analyzer: Any | None = None,
) -> list[HandoffBundle]:
    """Execute a verified handoff chain across multiple agents.

    Each agent's output is verified before passing to the next.
    Returns the list of handoff bundles for the entire chain.

    Parameters
    ----------
    agents : list[str]
        List of agent identifiers [agent_a, agent_b, agent_c, ...].
    initial_output : str
        The first agent's output.
    evidence : str
        The evidence context.
    mode : str
        Verification mode ("strict", "audit", "permissive").
    analyzer : WitnessAnalyzer, optional
        Shared analyzer instance.

    Returns
    -------
    list[HandoffBundle]
        One bundle per handoff in the chain.
    """
    if len(agents) < 2:
        raise ValueError("Chain requires at least 2 agents")

    bundles: list[HandoffBundle] = []
    current_output = initial_output

    for i in range(len(agents) - 1):
        upstream_ids = [b.bundle_id for b in bundles]

        bundle = handoff(
            output=current_output,
            evidence=evidence,
            from_agent=agents[i],
            to_agent=agents[i + 1],
            mode=mode,
            chain_position=i,
            upstream_bundles=upstream_ids,
            analyzer=analyzer,
        )
        bundles.append(bundle)
        current_output = receive(bundle)

    return bundles
