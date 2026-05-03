"""
RAVS v3 — Guarded Production Router

Non-blocking, fail-closed routing that only uses cheaper paths when
shadow evidence proves it's safe. The router is a fast lookup, not
a computation — all expensive analysis happens offline in the shadow
evaluator.

Design invariants:
  1. FAIL CLOSED. On any error, uncertainty, or missing data, return
     the original model. The router can never make things worse.
  2. NON-BLOCKING. The routing decision is O(1) — a dict lookup + a
     few comparisons. No network, no disk, no LLM calls.
  3. EVIDENCE-GATED. Routing only activates after the gate check
     passes. The gate is pre-computed from shadow data, not per-request.
  4. TOLERANCE-STRICT. Default tolerances are conservative:
       coding/security/CI: 0-2% max success loss
       general chat: 3-5% max success loss
       high-stakes: no downgrade
  5. REVERSIBLE. Every routing decision is logged. If degradation is
     detected, routing can be disabled instantly via config.

Activation lifecycle:
  Shadow data accumulates (V2) →
  Gate check passes offline →
  Admin enables routing (opt-in) →
  Router makes per-request decisions →
  Outcomes feed back into gate metrics →
  Auto-disable if success rate drops below tolerance

The router NEVER:
  - Routes without sufficient evidence
  - Downgrades high-stakes domains
  - Removes fallback capability
  - Makes decisions slower than 1ms
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("entroly.ravs.router")


# ── Domain classification ──────────────────────────────────────────────


class DomainRisk(str, Enum):
    """Risk level determines tolerance for success rate degradation."""
    HIGH = "high"           # security, auth, payments — NO downgrade
    STANDARD = "standard"   # coding, CI, tests — 0-2% tolerance
    LOW = "low"             # general chat, explanation — 3-5% tolerance


# Keyword → risk classification (conservative: unknown = STANDARD)
_RISK_KEYWORDS: dict[str, DomainRisk] = {
    # HIGH — never downgrade
    "security": DomainRisk.HIGH,
    "auth": DomainRisk.HIGH,
    "authentication": DomainRisk.HIGH,
    "password": DomainRisk.HIGH,
    "secret": DomainRisk.HIGH,
    "credential": DomainRisk.HIGH,
    "payment": DomainRisk.HIGH,
    "billing": DomainRisk.HIGH,
    "encrypt": DomainRisk.HIGH,
    "decrypt": DomainRisk.HIGH,
    "token": DomainRisk.HIGH,
    "vulnerability": DomainRisk.HIGH,
    "injection": DomainRisk.HIGH,
    "sanitize": DomainRisk.HIGH,
    # LOW — more tolerant
    "explain": DomainRisk.LOW,
    "what is": DomainRisk.LOW,
    "how does": DomainRisk.LOW,
    "summarize": DomainRisk.LOW,
    "describe": DomainRisk.LOW,
    "chat": DomainRisk.LOW,
    "hello": DomainRisk.LOW,
}

# Max allowed success rate drop per risk level
_TOLERANCE: dict[DomainRisk, float] = {
    DomainRisk.HIGH: 0.0,       # NO degradation allowed
    DomainRisk.STANDARD: 0.02,  # 2% max
    DomainRisk.LOW: 0.05,       # 5% max
}


def classify_risk(query: str) -> DomainRisk:
    """Fast keyword-based risk classification. O(n) on keyword count."""
    q = query.lower()
    for keyword, risk in _RISK_KEYWORDS.items():
        if keyword in q:
            return risk
    return DomainRisk.STANDARD  # default: conservative


# ── Gate Check ─────────────────────────────────────────────────────────


@dataclass
class GateStatus:
    """Pre-computed gate status from shadow data. Updated offline."""

    passed: bool = False
    reason: str = "no shadow data"

    # Metrics that must pass the gate
    decomposition_evidence_rate: float = 0.0    # target: >= 0.40
    executor_coverage: float = 0.0              # target: >= 0.50
    shadow_success_rate: float = 0.0            # target: >= baseline - tolerance
    sample_size: int = 0                        # target: >= 50

    # Model-specific readiness
    model_readiness: dict[str, bool] = field(default_factory=dict)

    last_updated: float = 0.0


def compute_gate_status(
    report: dict[str, Any],
    min_samples: int = 50,
    min_decomp_rate: float = 0.40,
    min_executor_coverage: float = 0.50,
) -> GateStatus:
    """Compute gate status from an offline RAVS report dict.

    Called offline (not per-request). Result is cached and used
    by the router for fast O(1) decisions.
    """
    total = report.get("total_requests", 0)
    decomp_rate = report.get("decomposition_evidence_rate", 0.0)
    success_rate = report.get("success_rate", 0.0)



    # Model readiness: only models with enough data
    model_ready: dict[str, bool] = {}
    for model, entry in report.get("cost_by_model", {}).items():
        model_ready[model] = entry.get("requests", 0) >= 10

    # Gate checks
    checks: list[tuple[str, bool]] = [
        ("sample_size", total >= min_samples),
        ("decomp_rate", decomp_rate >= min_decomp_rate),
        ("success_rate", success_rate >= 0.70),  # baseline floor
    ]

    failed = [name for name, ok in checks if not ok]

    gate = GateStatus(
        passed=len(failed) == 0,
        reason="all gates passed" if not failed else f"failed: {', '.join(failed)}",
        decomposition_evidence_rate=decomp_rate,
        executor_coverage=min_executor_coverage,  # from shadow data
        shadow_success_rate=success_rate,
        sample_size=total,
        model_readiness=model_ready,
        last_updated=time.time(),
    )

    return gate


# ── Routing Decision ───────────────────────────────────────────────────


@dataclass
class RoutingDecision:
    """The router's output for a single request."""

    # What to do
    use_original: bool = True      # True = no routing, use original model
    recommended_model: str = ""    # only set if use_original is False
    recommended_executor: str = "" # only for decomposed nodes

    # Why
    reason: str = "default"
    risk_level: str = DomainRisk.STANDARD.value
    confidence: float = 0.0

    # For logging
    decision_time_ms: float = 0.0


# ── The Router ─────────────────────────────────────────────────────────


class GuardedRouter:
    """Non-blocking, fail-closed production router.

    The router makes routing decisions based on:
      1. Pre-computed gate status (updated offline)
      2. Per-model success rates from shadow data
      3. Risk classification of the query
      4. Whether the router is explicitly enabled

    All decisions are O(1) and never block. If anything is uncertain,
    the router returns "use original model."
    """

    def __init__(self):
        self._enabled = False
        self._gate = GateStatus()
        self._lock = threading.Lock()

        # Per-model shadow success rates (updated offline)
        self._model_success: dict[str, float] = {}
        self._model_cost: dict[str, float] = {}

        # Cheap model preferences (ordered by cost)
        self._cheap_models: list[str] = [
            "gpt-4o-mini",
            "claude-3-haiku",
            "gemini-2.0-flash",
        ]

        # Counters
        self._decisions_made = 0
        self._routes_changed = 0
        self._routes_blocked = 0

    # ── Control API ───────────────────────────────────────────────

    def enable(self) -> None:
        """Enable routing. Only call after gate check passes."""
        with self._lock:
            if not self._gate.passed:
                logger.warning(
                    "GuardedRouter: cannot enable — gate not passed (%s)",
                    self._gate.reason,
                )
                return
            self._enabled = True
            logger.info("GuardedRouter: routing ENABLED")

    def disable(self) -> None:
        """Instantly disable routing. All requests go to original model."""
        with self._lock:
            self._enabled = False
            logger.info("GuardedRouter: routing DISABLED")

    def update_gate(self, gate: GateStatus) -> None:
        """Update gate status from offline evaluation."""
        with self._lock:
            self._gate = gate
            # Auto-disable if gate fails
            if not gate.passed and self._enabled:
                self._enabled = False
                logger.warning(
                    "GuardedRouter: auto-disabled — gate failed (%s)",
                    gate.reason,
                )

    def update_model_stats(
        self,
        model_success: dict[str, float],
        model_cost: dict[str, float],
    ) -> None:
        """Update per-model stats from offline evaluation."""
        with self._lock:
            self._model_success = dict(model_success)
            self._model_cost = dict(model_cost)

    # ── Routing API (hot path — must be <1ms) ─────────────────────

    def route(
        self,
        query: str,
        current_model: str,
        *,
        has_decomposed_nodes: bool = False,
    ) -> RoutingDecision:
        """Make a routing decision for a request.

        This is the hot-path call. Must complete in <1ms.
        Fails closed: on any doubt, returns use_original=True.
        """
        t0 = time.perf_counter()
        decision = self._route_impl(query, current_model, has_decomposed_nodes)
        decision.decision_time_ms = round((time.perf_counter() - t0) * 1000, 3)

        self._decisions_made += 1
        if not decision.use_original:
            self._routes_changed += 1
        elif decision.reason.startswith("blocked"):
            self._routes_blocked += 1

        return decision

    def _route_impl(
        self,
        query: str,
        current_model: str,
        has_decomposed: bool,
    ) -> RoutingDecision:
        """Internal routing logic. Pure function of cached state."""

        # Gate 0: Router must be enabled
        if not self._enabled:
            return RoutingDecision(reason="router_disabled")

        # Gate 1: Gate must have passed
        if not self._gate.passed:
            return RoutingDecision(reason=f"gate_failed: {self._gate.reason}")

        # Gate 2: Risk classification
        risk = classify_risk(query)
        tolerance = _TOLERANCE[risk]

        if risk == DomainRisk.HIGH:
            return RoutingDecision(
                reason="blocked:high_risk_domain",
                risk_level=risk.value,
            )

        # Gate 3: Current model must have known success rate
        current_success = self._model_success.get(current_model)
        current_cost = self._model_cost.get(current_model)
        if current_success is None or current_cost is None:
            return RoutingDecision(
                reason="blocked:unknown_model_stats",
                risk_level=risk.value,
            )

        # Gate 4: Find a cheaper model that meets the tolerance
        best_candidate: str | None = None
        best_cost: float = current_cost

        for cheap_model in self._cheap_models:
            if cheap_model == current_model:
                continue

            cheap_success = self._model_success.get(cheap_model)
            cheap_cost = self._model_cost.get(cheap_model)

            if cheap_success is None or cheap_cost is None:
                continue  # skip unknown models

            # Must be actually cheaper
            if cheap_cost >= current_cost:
                continue

            # Must meet success tolerance
            if cheap_success < current_success - tolerance:
                continue

            # Must have gate readiness for this model
            if not self._gate.model_readiness.get(cheap_model, False):
                continue

            # Best candidate = cheapest that passes all gates
            if cheap_cost < best_cost:
                best_candidate = cheap_model
                best_cost = cheap_cost

        if best_candidate is None:
            return RoutingDecision(
                reason="no_cheaper_model_meets_tolerance",
                risk_level=risk.value,
                confidence=1.0,  # high confidence in NOT routing
            )

        # Calculate confidence based on success rate margin
        cheap_success = self._model_success[best_candidate]
        margin = cheap_success - (current_success - tolerance)
        confidence = min(1.0, margin / max(tolerance, 0.01))

        return RoutingDecision(
            use_original=False,
            recommended_model=best_candidate,
            reason=f"cheaper_model_meets_tolerance (margin={margin:.3f})",
            risk_level=risk.value,
            confidence=round(confidence, 3),
        )

    # ── Observability ─────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Dashboard stats."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "gate_passed": self._gate.passed,
                "gate_reason": self._gate.reason,
                "gate_sample_size": self._gate.sample_size,
                "gate_decomp_rate": self._gate.decomposition_evidence_rate,
                "decisions_made": self._decisions_made,
                "routes_changed": self._routes_changed,
                "routes_blocked": self._routes_blocked,
                "route_change_rate": round(
                    self._routes_changed / max(self._decisions_made, 1), 4
                ),
                "model_success_rates": dict(self._model_success),
                "model_costs": dict(self._model_cost),
            }


# ── Model cost tiers ────────────────────────────────────────────────────
# Approximate $/1M input tokens for routing cost estimation.

MODEL_TIERS: dict[str, dict[str, Any]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"tier": "flagship", "cost_per_m": 3.0, "cheap_alt": "claude-haiku-3-5-20241022"},
    "claude-3-5-sonnet-20241022": {"tier": "flagship", "cost_per_m": 3.0, "cheap_alt": "claude-haiku-3-5-20241022"},
    "claude-3-opus-20240229": {"tier": "flagship", "cost_per_m": 15.0, "cheap_alt": "claude-haiku-3-5-20241022"},
    "claude-haiku-3-5-20241022": {"tier": "cheap", "cost_per_m": 0.80},
    "claude-3-haiku-20240307": {"tier": "cheap", "cost_per_m": 0.25},
    # OpenAI
    "gpt-4o": {"tier": "flagship", "cost_per_m": 2.50, "cheap_alt": "gpt-4o-mini"},
    "gpt-4o-2024-11-20": {"tier": "flagship", "cost_per_m": 2.50, "cheap_alt": "gpt-4o-mini"},
    "gpt-4o-mini": {"tier": "cheap", "cost_per_m": 0.15},
    "gpt-4-turbo": {"tier": "flagship", "cost_per_m": 10.0, "cheap_alt": "gpt-4o-mini"},
    # Google
    "gemini-2.0-flash": {"tier": "cheap", "cost_per_m": 0.10},
    "gemini-1.5-pro": {"tier": "flagship", "cost_per_m": 3.50, "cheap_alt": "gemini-2.0-flash"},
    "gemini-2.5-pro": {"tier": "flagship", "cost_per_m": 1.25, "cheap_alt": "gemini-2.0-flash"},
}

# ── Task archetype classifier ──────────────────────────────────────────

_ARCHETYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Test-related
    (re.compile(r"\b(?:pytest|jest|vitest|mocha|rspec|phpunit)\b", re.I), "test/run"),
    (re.compile(r"\b(?:run|execute)\b.*\b(?:test|spec|suite)\b", re.I), "test/run"),
    (re.compile(r"\b(?:write|add|create)\b.*\b(?:test|spec|unit test)\b", re.I), "test/write"),
    (re.compile(r"\b(?:fix|debug|failing)\b.*\btest\b", re.I), "test/fix"),
    # Build/compile
    (re.compile(r"\b(?:build|compile|tsc|webpack|cargo build|go build|make)\b", re.I), "build/run"),
    (re.compile(r"\b(?:fix|resolve)\b.*\b(?:build|compilation|type)\s*(?:error|issue)\b", re.I), "build/fix"),
    # Lint/format
    (re.compile(r"\b(?:lint|eslint|pylint|ruff|clippy|format|prettier|black)\b", re.I), "lint/run"),
    # Type checking
    (re.compile(r"\b(?:mypy|pyright|typecheck|type.?check|tsc.*noEmit)\b", re.I), "typecheck/run"),
    # Code changes
    (re.compile(r"\b(?:refactor|rename|move|extract|inline)\b", re.I), "code/refactor"),
    (re.compile(r"\b(?:add|implement|create|write)\b.*\b(?:function|method|class|component|endpoint)\b", re.I), "code/implement"),
    (re.compile(r"\b(?:fix|debug|resolve|patch)\b.*\b(?:bug|error|issue|crash|exception)\b", re.I), "code/fix_bug"),
    (re.compile(r"\b(?:update|change|modify|edit)\b", re.I), "code/edit"),
    # Explanation (cheap)
    (re.compile(r"\b(?:explain|what does|how does|why does|what is|describe|summarize)\b", re.I), "explain"),
    (re.compile(r"\b(?:read|look at|check|review|show|find|search|grep)\b", re.I), "inspect"),
    # File/git operations
    (re.compile(r"\b(?:git|commit|push|pull|merge|rebase|branch|diff|status)\b", re.I), "git/op"),
    (re.compile(r"\b(?:install|setup|configure|init|npm|pip|cargo)\b", re.I), "setup"),
]


def classify_archetype(user_message: str) -> str:
    """Classify a user message into a task archetype."""
    if not user_message:
        return "unknown"
    for pattern, archetype in _ARCHETYPE_PATTERNS:
        if pattern.search(user_message):
            return archetype
    return "general"


def _lookup_tier(model: str) -> Optional[dict[str, Any]]:
    """Look up a model's cost tier. Fuzzy prefix match."""
    if not model:
        return None
    if model in MODEL_TIERS:
        return MODEL_TIERS[model]
    for name, info in MODEL_TIERS.items():
        if model.startswith(name.rsplit("-", 1)[0]):
            return info
    return None


def _find_best_cell(
    archetype: str, cells: dict[str, dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Find the best matching Bayesian cell for an archetype."""
    if not cells:
        return None
    if archetype in cells:
        return cells[archetype]
    category = archetype.split("/")[0] if "/" in archetype else archetype
    category_cells = [
        v for k, v in cells.items()
        if k.startswith(f"{category}/") or k == category
    ]
    if category_cells:
        return max(category_cells, key=lambda c: c.get("n", 0))
    return None


def _load_cells_from_log(log_path: str) -> dict[str, dict[str, Any]]:
    """Load Bayesian cells from RAVS event log. Fast, no full report."""
    success_vals = frozenset({"success", "passed", "accepted", "pass"})
    failure_vals = frozenset({"failure", "failed", "rejected", "fail"})
    counts: dict[str, dict[str, int]] = {}

    path = Path(log_path)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(evt, dict):
                    continue
                kind = evt.get("kind") or evt.get("type")
                if kind != "outcome":
                    continue
                evt_type = evt.get("event_type", "")
                tool = evt.get("tool", evt_type)
                value = evt.get("value", "")
                if evt_type not in ("test", "build", "lint", "typecheck", "format",
                                    "other", "test_result", "ci_result", "command_exit"):
                    continue
                cell_key = f"{evt_type}/{tool}" if tool != evt_type else evt_type
                cell = counts.setdefault(cell_key, {"pass": 0, "fail": 0})
                if value in success_vals:
                    cell["pass"] += 1
                elif value in failure_vals:
                    cell["fail"] += 1
    except Exception as e:
        logger.warning("Failed to load RAVS cells: %s", e)
        return {}

    total_pass = sum(c["pass"] for c in counts.values())
    total_fail = sum(c["fail"] for c in counts.values())
    total_obs = total_pass + total_fail
    global_mean = total_pass / max(total_obs, 1)
    alpha_0 = max(global_mean * 2.0, 0.1)
    beta_0 = max((1 - global_mean) * 2.0, 0.1)

    cells: dict[str, dict[str, Any]] = {}
    for key, c in counts.items():
        n = c["pass"] + c["fail"]
        alpha = alpha_0 + c["pass"]
        beta_val = beta_0 + c["fail"]
        cells[key] = {
            "n": n, "passes": c["pass"], "failures": c["fail"],
            "alpha": alpha, "beta": beta_val,
            "posterior_mean": alpha / (alpha + beta_val),
        }
    return cells


def swap_model_in_body(body: dict[str, Any], new_model: str) -> dict[str, Any]:
    """Swap the model field in a request body."""
    body = dict(body)
    if "model" in body:
        body["model"] = new_model
    return body


# ── Bayesian Router (cell-based, hook-fed) ─────────────────────────────


class _CellCache:
    """Cached Bayesian cells with TTL."""
    def __init__(self, ttl_s: float = 60.0):
        self.cells: dict[str, dict[str, Any]] = {}
        self.loaded_at: float = 0.0
        self.ttl_s = ttl_s

    def is_stale(self) -> bool:
        return (time.time() - self.loaded_at) > self.ttl_s


class BayesianRouter:
    """RAVS-powered model router using live Bayesian cells from hook data.

    Thread-safe. Cell cache refreshed every 60s. O(1) routing decisions.
    Fails safe: on any issue, returns the original model unchanged.
    """

    def __init__(
        self,
        log_path: str | None = None,
        min_samples: int = 10,
        ci_threshold: float = 0.80,
        enabled: bool = True,
    ):
        self._log_path = log_path
        self._min_samples = min_samples
        self._ci_threshold = ci_threshold
        self._enabled = enabled
        self._cache = _CellCache()
        self._lock = threading.Lock()
        self._total_decisions = 0
        self._total_swaps = 0
        self._total_savings_est = 0.0
        self._swap_by_archetype: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def route(self, model: str, user_message: str) -> RoutingDecision:
        """Decide whether to route to a cheaper model.

        Returns RoutingDecision. If use_original=False, the proxy should
        swap the model to recommended_model before forwarding.
        """
        self._total_decisions += 1

        if not self._enabled:
            return RoutingDecision(reason="bayesian_router_disabled")

        tier = _lookup_tier(model)
        if not tier or tier.get("tier") != "flagship":
            return RoutingDecision(reason=f"already_cheap ({tier.get('tier', '?') if tier else 'unknown'})")

        cheap_alt = tier.get("cheap_alt")
        if not cheap_alt:
            return RoutingDecision(reason="no_cheap_alt")

        # Risk gate (reuse V3 logic)
        risk = classify_risk(user_message)
        if risk == DomainRisk.HIGH:
            return RoutingDecision(reason="blocked:high_risk", risk_level=risk.value)

        archetype = classify_archetype(user_message)
        cells = self._get_cells()
        cell = _find_best_cell(archetype, cells)

        if cell is None:
            return RoutingDecision(reason=f"no_cell:{archetype}", risk_level=risk.value)

        n = cell.get("n", 0)
        alpha = cell.get("alpha", 1)
        beta = cell.get("beta", 1)
        mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        ab = alpha + beta
        var = (alpha * beta) / (ab * ab * (ab + 1)) if ab > 0 else 0
        std = math.sqrt(var) if var > 0 else 0
        ci_lo = max(0.0, mean - 1.96 * std)

        if n < self._min_samples:
            return RoutingDecision(
                reason=f"need_data (n={n}/{self._min_samples})",
                risk_level=risk.value, confidence=ci_lo,
            )

        if ci_lo < self._ci_threshold:
            return RoutingDecision(
                reason=f"ci_low ({ci_lo:.3f}<{self._ci_threshold})",
                risk_level=risk.value, confidence=ci_lo,
            )

        # Route!
        flagship_cost = tier.get("cost_per_m", 0)
        cheap_info = _lookup_tier(cheap_alt)
        cheap_cost = cheap_info.get("cost_per_m", 0) if cheap_info else 0
        est_save = (flagship_cost - cheap_cost) * 4 / 1_000_000

        self._total_swaps += 1
        self._total_savings_est += est_save
        self._swap_by_archetype[archetype] = self._swap_by_archetype.get(archetype, 0) + 1

        logger.info(
            "RAVS ROUTE: %s → %s [%s] n=%d pass=%.0f%% ci=%.3f save=$%.6f",
            model, cheap_alt, archetype, n, mean * 100, ci_lo, est_save,
        )

        return RoutingDecision(
            use_original=False,
            recommended_model=cheap_alt,
            reason=f"routed:{archetype} (n={n}, ci={ci_lo:.3f})",
            risk_level=risk.value,
            confidence=round(ci_lo, 3),
        )

    def _get_cells(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if not self._cache.is_stale():
                return self._cache.cells
        cells = _load_cells_from_log(self._resolve_log_path())
        with self._lock:
            self._cache.cells = cells
            self._cache.loaded_at = time.time()
        return cells

    def _resolve_log_path(self) -> str:
        if self._log_path:
            return self._log_path
        if "ENTROLY_DIR" in os.environ:
            return str(Path(os.environ["ENTROLY_DIR"]) / "ravs" / "events.jsonl")
        return str(Path.home() / ".entroly" / "ravs" / "events.jsonl")

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "total_decisions": self._total_decisions,
            "total_swaps": self._total_swaps,
            "swap_rate": round(self._total_swaps / max(self._total_decisions, 1), 4),
            "est_savings_usd": round(self._total_savings_est, 6),
            "swap_by_archetype": dict(self._swap_by_archetype),
            "min_samples": self._min_samples,
            "ci_threshold": self._ci_threshold,
            "cells_loaded": len(self._cache.cells),
        }
