"""
Cost Cortex — the control-plane policy layer for context.
=========================================================

Entroly's thesis is that every context unit should pass a decision layer
*before* it reaches the model: should this byte be sent exact, skeletonized,
digested, replaced with a recoverable handle, or blocked for compliance — and
is the resulting spend within budget for *this* model's price?

This module is the first brick of that brain. It is deliberately small,
pure, and side-effect-free so it can be trusted as the single place that:

  1. resolves a model's price from ONE source (``value_tracker``), and
  2. clamps the size of Entroly's *injected* context to a token / dollar
     budget — so a cheap long-context model can never silently permit a
     600K-token injection.

Compliance invariant (matches docs/provider-compliance.md and our standing
rule): this layer only ever *reduces or annotates Entroly's own injected
context*. It never reads or mutates the user's request, the model, generation
parameters (temperature / max_tokens / top_p / thinking), or tool definitions.
Clamping can only lower spend, never raise it. Provider terms and data policies
still govern whatever the user themselves sends.

Honesty invariant: prices come from ``value_tracker`` (bundled defaults +
optional local override); unknown models fall back to the documented default
rate (``value_tracker`` logs a warning) — we never invent a price, and we never
claim a saving we did not compute from a real rate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from . import value_tracker

# Safety net: Entroly will not inject more than this many context tokens by
# default, regardless of the model's context window. Prevents accidental
# runaway injections on million-token-context models. Override with
# ENTROLY_MAX_CONTEXT_TOKENS (set 0 to disable the hard cap).
try:
    DEFAULT_HARD_TOKEN_CAP = int(os.getenv("ENTROLY_MAX_CONTEXT_TOKENS", "256000"))
except ValueError:
    DEFAULT_HARD_TOKEN_CAP = 256_000


def _env_dollar_ceiling() -> float | None:
    """Optional per-request dollar ceiling for Entroly's injected context."""
    raw = os.getenv("ENTROLY_MAX_CONTEXT_USD")
    if not raw:
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except ValueError:
        return None


@dataclass(frozen=True)
class ProviderPrice:
    """A model's price, sourced solely from ``value_tracker`` (one source)."""

    model: str
    input_per_1k: float
    output_per_1k: float

    @classmethod
    def for_model(cls, model: str) -> "ProviderPrice":
        # estimate_cost(1000, model, kind) == USD for 1k tokens of that kind.
        inp = value_tracker.estimate_cost(1000, model or "", "input")
        out = value_tracker.estimate_cost(1000, model or "", "output")
        return cls(model=model or "", input_per_1k=inp, output_per_1k=out)

    def input_usd(self, tokens: int) -> float:
        return max(0, tokens) / 1000.0 * self.input_per_1k


@dataclass
class CostBudget:
    """Budget for the size of Entroly's *injected* context for one request.

    ``clamp`` only ever returns a value ``<= requested_tokens`` — the cortex
    can shrink the injected context to fit a token / dollar budget, never grow
    it. It does not touch the user's request in any way.
    """

    model: str
    dollar_ceiling: float | None = None
    hard_token_cap: int = DEFAULT_HARD_TOKEN_CAP

    def clamp(self, requested_tokens: int) -> tuple[int, str]:
        """Return ``(clamped_tokens, reason)``. Never exceeds requested."""
        if requested_tokens <= 0:
            return max(0, requested_tokens), "no budget requested"
        clamped = requested_tokens
        reasons: list[str] = []

        if self.hard_token_cap and clamped > self.hard_token_cap:
            clamped = self.hard_token_cap
            reasons.append(f"hard cap {self.hard_token_cap:,}t")

        ceiling = self.dollar_ceiling if self.dollar_ceiling is not None else _env_dollar_ceiling()
        if ceiling and ceiling > 0:
            per_1k = ProviderPrice.for_model(self.model).input_per_1k
            if per_1k > 0:
                max_by_dollar = int(ceiling / per_1k * 1000)
                if clamped > max_by_dollar:
                    clamped = max_by_dollar
                    reasons.append(f"${ceiling:.2f} ceiling @ ${per_1k:.4f}/1k")

        clamped = max(0, min(clamped, requested_tokens))
        return clamped, ("; ".join(reasons) if reasons else "within budget")


def clamp_injected_budget(
    model: str, requested_tokens: int, dollar_ceiling: float | None = None
) -> tuple[int, str]:
    """Convenience: clamp an injected-context token budget for ``model``.

    Drop-in for the proxy's budget selection — call it on the budget BEFORE it
    is handed to ``optimize_context`` so the engine never selects more context
    than the token/dollar budget allows. Returns ``(tokens, reason)``.
    """
    return CostBudget(model=model, dollar_ceiling=dollar_ceiling).clamp(requested_tokens)


# ── Per-unit context decisions + ledger ──────────────────────────────────
# The cortex's output for a request: what it did to each context unit, and
# whether the exact source is still recoverable. This is what lets us push
# "max compression without losing quality" honestly: a unit is only dropped
# losslessly if it carries a recovery handle; otherwise it is marked lossy.


class Decision(str, Enum):
    EXACT = "exact"                    # sent verbatim
    SKELETON = "skeleton"              # signatures / docstrings only
    DIGEST = "digest"                  # summarized (e.g., aged tool output)
    OMITTED_WITH_HANDLE = "omitted_with_handle"  # dropped but recoverable on demand
    BLOCKED = "blocked"               # withheld for compliance (e.g., a secret)


@dataclass
class ContextDecision:
    unit_id: str
    decision: Decision
    tokens_before: int
    tokens_after: int
    reason: str = ""
    handle: str | None = None  # recovery handle when omitted/compressed

    @property
    def recoverable(self) -> bool:
        """Exact source still retrievable (lossless) iff a handle exists, or
        the unit was sent exact. BLOCKED is intentionally not recoverable."""
        if self.decision == Decision.EXACT:
            return True
        if self.decision == Decision.BLOCKED:
            return False
        return self.handle is not None

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class ContextLedger:
    """Auditable record of every cut the cortex made for a request."""

    decisions: list[ContextDecision] = field(default_factory=list)

    def record(self, d: ContextDecision) -> None:
        self.decisions.append(d)

    @property
    def tokens_before(self) -> int:
        return sum(d.tokens_before for d in self.decisions)

    @property
    def tokens_after(self) -> int:
        return sum(d.tokens_after for d in self.decisions)

    @property
    def tokens_saved(self) -> int:
        return sum(d.tokens_saved for d in self.decisions)

    @property
    def lossy_units(self) -> list[ContextDecision]:
        """Units that lost information without a recovery handle — the only
        places quality could degrade. 'Max compression without losing quality'
        means keeping this list empty (everything recoverable or sent exact)."""
        return [d for d in self.decisions if not d.recoverable and d.decision != Decision.BLOCKED]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for d in self.decisions:
            counts[d.decision.value] = counts.get(d.decision.value, 0) + 1
        return {
            "units": len(self.decisions),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "by_decision": counts,
            "lossy_units": len(self.lossy_units),
            "fully_recoverable": len(self.lossy_units) == 0,
        }
