"""
Epistemic Ingress Controller
=============================

The control-plane policy engine at query ingress.  Lives in the Action
layer but decides whether to invoke Belief, Verification, Truth, or
Evolution.

This is NOT request forwarding — it is epistemic routing.

5 Canonical Flows:
  ① Fast Answer:         Query → Belief → Action
  ② Verify Before Answer: Query → Belief → Verification → Action
  ③ Compile On Demand:   Query → Truth → Belief → Verification → Action
  ④ Change-Driven:       Event → Truth → Belief → Verification → Action
  ⑤ Self-Improvement:    Misses → Verification → Evolution → Belief

Router inspects 4 signals: intent, belief_coverage, freshness, risk.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class EpistemicIntent(str, Enum):
    """What kind of answer is needed."""
    ARCHITECTURE = "architecture"       # "How does auth work?"
    PR_BRIEF = "pr_brief"               # "Write a PR summary"
    CODE_GENERATION = "code_generation"  # "Generate a migration script"
    REPORT = "report"                    # "Draw the dependency graph"
    RESEARCH = "research"               # "Compare vector DBs"
    INCIDENT = "incident"               # "Why is latency spiking?"
    AUDIT = "audit"                     # "Check for PII leakage"
    REUSE = "reuse"                     # "Do we already have a retry helper?"
    ONBOARDING = "onboarding"           # "Explain checkout to a new engineer"
    TEST_GAP = "test_gap"               # "What tests are missing?"
    RELEASE = "release"                 # "Is this service ready to release?"
    REPAIR = "repair"                   # "This test started failing"
    GENERAL = "general"                 # Catch-all


class EpistemicFlow(str, Enum):
    """Which canonical flow to execute."""
    FAST_ANSWER = "fast_answer"                 # ① Belief → Action
    VERIFY_BEFORE_ANSWER = "verify_before"      # ② Belief → Verification → Action
    COMPILE_ON_DEMAND = "compile_on_demand"      # ③ Truth → Belief → Verification → Action
    CHANGE_DRIVEN = "change_driven"              # ④ Event → Truth → Belief → ...
    SELF_IMPROVEMENT = "self_improvement"         # ⑤ Misses → Evolution → Belief


class RiskLevel(str, Enum):
    """How dangerous is the answer domain."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BeliefCoverage:
    """Result of checking belief coverage in the vault."""
    exists: bool = False
    fresh: bool = False
    verified: bool = False
    confidence: float = 0.0
    matching_claims: list[str] = field(default_factory=list)
    stalest_check: str | None = None  # ISO-8601 of oldest last_checked


@dataclass
class RoutingDecision:
    """The router's output: which flow to execute and why."""
    flow: EpistemicFlow
    intent: EpistemicIntent
    coverage: BeliefCoverage
    risk: RiskLevel
    reasoning: str
    routing_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_id": self.routing_id,
            "flow": self.flow.value,
            "intent": self.intent.value,
            "risk": self.risk.value,
            "reasoning": self.reasoning,
            "coverage": {
                "exists": self.coverage.exists,
                "fresh": self.coverage.fresh,
                "verified": self.coverage.verified,
                "confidence": self.coverage.confidence,
                "matching_claims": self.coverage.matching_claims,
            },
            "timestamp": self.timestamp,
        }


# ══════════════════════════════════════════════════════════════════════
# Intent Classifier
# ══════════════════════════════════════════════════════════════════════

# Keyword patterns for intent classification (order matters — first match wins)
_INTENT_PATTERNS: list[tuple[EpistemicIntent, re.Pattern]] = [
    (EpistemicIntent.INCIDENT, re.compile(
        r"\b(incident|outage|spike|latency|down|crash|alert|page|5\d\d|timeout|"
        r"spiking|degraded|error.rate|p99|on.?call)\b", re.I)),
    (EpistemicIntent.AUDIT, re.compile(
        r"\b(audit|compliance|pii|gdpr|hipaa|security.review|vulnerability|"
        r"cve|leak|injection|xss|csrf|secrets?)\b", re.I)),
    (EpistemicIntent.REPAIR, re.compile(
        r"\b(fail|broke|broken|regression|bisect|root.?cause|fix|debug|"
        r"stack.?trace|exception|error|traceback|started.failing)\b", re.I)),
    (EpistemicIntent.TEST_GAP, re.compile(
        r"\b(test.gap|missing.test|uncovered|coverage|untested|edge.case|"
        r"what.tests)\b", re.I)),
    (EpistemicIntent.RELEASE, re.compile(
        r"\b(release.?ready|ship|deploy|rollout|canary|staging|production.ready|"
        r"go.?live|launch)\b", re.I)),
    (EpistemicIntent.PR_BRIEF, re.compile(
        r"\b(pr|pull.request|diff|code.review|review.this|patch|changeset|"
        r"commit|merge)\b", re.I)),
    (EpistemicIntent.CODE_GENERATION, re.compile(
        r"\b(generate|implement|create|write|scaffold|migration|boilerplate|"
        r"add.?feature|build.?a|code.?for)\b", re.I)),
    (EpistemicIntent.REPORT, re.compile(
        r"\b(report|diagram|slide|presentation|chart|graph|visuali|mermaid|"
        r"marp|draw|render)\b", re.I)),
    (EpistemicIntent.RESEARCH, re.compile(
        r"\b(research|benchmark|compare|evaluate|survey|trade.?off|"
        r"pros.?cons|analysis|investigate)\b", re.I)),
    (EpistemicIntent.REUSE, re.compile(
        r"\b(already.have|existing|reuse|duplicate|reinvent|shared.?util|"
        r"helper|library|common)\b", re.I)),
    (EpistemicIntent.ARCHITECTURE, re.compile(
        r"\b(architect|design|structure|module|service|component|layer|"
        r"talk.?to|dependency|flow|pipeline|system|how.does.+work|"
        r"connect|interact|communicat)\b", re.I)),
    (EpistemicIntent.ONBOARDING, re.compile(
        r"\b(onboard|overview|walkthrough|tutorial|new.?engineer|"
        r"explain.+to|introduction|getting.started|for.?beginners)\b", re.I)),
]

# Risk escalation keywords
_HIGH_RISK_PATTERNS = re.compile(
    r"\b(security|compliance|pii|gdpr|hipaa|credential|secret|key|token|"
    r"password|auth|encrypt|permission|rbac|acl|cve|vulnerability|"
    r"injection|deletion|drop.table|rm.\-rf|production|customer.data)\b", re.I)

_MEDIUM_RISK_PATTERNS = re.compile(
    r"\b(migration|schema.change|breaking|backward|deprecat|refactor|"
    r"database|payment|billing|transaction|rollback|deploy)\b", re.I)


def classify_intent(query: str) -> EpistemicIntent:
    """Classify a query into an epistemic intent using keyword patterns."""
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return EpistemicIntent.GENERAL


def assess_risk(query: str, intent: EpistemicIntent) -> RiskLevel:
    """Assess the risk level of a query domain."""
    # Certain intents are always high-risk
    if intent in (EpistemicIntent.AUDIT,):
        return RiskLevel.HIGH

    if _HIGH_RISK_PATTERNS.search(query):
        return RiskLevel.HIGH
    if _MEDIUM_RISK_PATTERNS.search(query):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ══════════════════════════════════════════════════════════════════════
# The Router
# ══════════════════════════════════════════════════════════════════════

class EpistemicRouter:
    """
    Epistemic Ingress Controller.

    Inspects 4 signals (intent, coverage, freshness, risk) and selects
    one of 5 canonical flows.

    Primary home: Action Layer (server.py + query_refiner.py)
    Inputs from:  Belief coverage + Verification confidence
    Feedback to:  Evolution (repeated miss logging)
    """

    def __init__(
        self,
        vault_path: str | None = None,
        miss_threshold: int = 3,
        freshness_hours: float = 24.0,
        min_confidence: float = 0.6,
    ):
        """
        Args:
            vault_path: Path to the Obsidian vault root. If None, vault
                        checks are skipped (all queries go compile path).
            miss_threshold: Number of consecutive misses on the same entity
                            before triggering the Evolution feedback path.
            freshness_hours: How many hours before a belief is considered stale.
            min_confidence: Minimum confidence score to consider a belief trusted.
        """
        self._vault_path = Path(vault_path) if vault_path else None
        self._miss_threshold = miss_threshold
        self._freshness_hours = freshness_hours
        self._min_confidence = min_confidence

        # Miss counter: entity -> consecutive miss count
        self._miss_counts: dict[str, int] = {}

        # Routing history for observability
        self._history: list[RoutingDecision] = []

        if self._vault_path:
            logger.info(f"EpistemicRouter: vault at {self._vault_path}")
        else:
            logger.info("EpistemicRouter: no vault configured (compile-only mode)")

    def route(
        self,
        query: str,
        is_event: bool = False,
        event_type: str | None = None,
    ) -> RoutingDecision:
        """
        Route a query or event through the epistemic decision matrix.

        Args:
            query: The user query or event description.
            is_event: True if this is a change-driven event (PR, commit, etc.)
            event_type: Type of event (pr, commit, release, incident, scheduled)

        Returns:
            RoutingDecision with the selected flow and reasoning.
        """
        # ── Signal 1: Intent ──
        if is_event:
            intent = self._classify_event_intent(event_type or "")
        else:
            intent = classify_intent(query)

        # ── Signal 2: Belief Coverage ──
        coverage = self._check_coverage(query, intent)

        # ── Signal 3: Risk ──
        risk = assess_risk(query, intent)

        # ── Signal 4: Miss history ──
        entity_key = self._extract_entity_key(query)
        miss_count = self._miss_counts.get(entity_key, 0)

        # ── Flow Selection (the routing matrix) ──
        flow, reasoning = self._select_flow(
            intent=intent,
            coverage=coverage,
            risk=risk,
            is_event=is_event,
            miss_count=miss_count,
        )

        # Update miss tracking
        if not coverage.exists:
            self._miss_counts[entity_key] = miss_count + 1
        else:
            self._miss_counts[entity_key] = 0

        decision = RoutingDecision(
            flow=flow,
            intent=intent,
            coverage=coverage,
            risk=risk,
            reasoning=reasoning,
        )

        self._history.append(decision)

        logger.info(
            f"EpistemicRouter: {decision.flow.value} | "
            f"intent={intent.value} risk={risk.value} "
            f"coverage={'✓' if coverage.exists else '✗'} "
            f"fresh={'✓' if coverage.fresh else '✗'} "
            f"verified={'✓' if coverage.verified else '✗'} "
            f"conf={coverage.confidence:.2f}"
        )

        return decision

    def record_miss(self, query: str) -> None:
        """Explicitly record a miss (system couldn't answer satisfactorily)."""
        key = self._extract_entity_key(query)
        self._miss_counts[key] = self._miss_counts.get(key, 0) + 1
        logger.info(f"EpistemicRouter: miss recorded for '{key}' "
                     f"(count={self._miss_counts[key]})")

    def record_outcome(
        self,
        flow: str,
        success: bool,
        confidence: float = 0.0,
        component_bus: Any = None,
    ) -> None:
        """Record the outcome of a routed flow for self-improvement.

        After each flow execution, this logs the result so the router
        can adaptively tune its thresholds. Every 10 episodes, the
        router self-tunes toward flows with higher success rates.

        Zero token cost — pure local O(1) computation.

        Args:
            flow: Flow name (e.g., 'fast_answer', 'compile_on_demand')
            success: Whether the flow produced a satisfactory result
            confidence: Confidence of the result (0.0-1.0)
            component_bus: Optional ComponentFeedbackBus for persistent logging
        """
        if not hasattr(self, "_flow_outcomes"):
            self._flow_outcomes: dict[str, list[bool]] = {}
            self._total_outcomes = 0

        self._flow_outcomes.setdefault(flow, []).append(success)
        self._total_outcomes = getattr(self, "_total_outcomes", 0) + 1

        # Log to component bus for cross-session persistence
        if component_bus:
            component_bus.log(
                component="epistemic_router",
                metric="flow_success",
                value=1.0 if success else 0.0,
                params={"flow": flow, "confidence": confidence},
            )

        # Self-tune every 10 episodes
        if self._total_outcomes % 10 == 0:
            self._self_tune()

    def _self_tune(self) -> None:
        """Adaptive threshold tuning from flow outcome history.

        Adjusts routing parameters based on which flows succeed most:
        - If fast_answer has high success → lower min_confidence (trust more)
        - If compile_on_demand is frequently needed → lower freshness_hours
        - If self_improvement triggers too often → raise miss_threshold
        - If self_improvement triggers too rarely → lower miss_threshold

        Mathematical basis: Online gradient-free optimization (SPSA).
        θ_{t+1} = θ_t + α · sign(success_rate - target_rate) · step
        """
        if not hasattr(self, "_flow_outcomes"):
            return

        for flow, outcomes in self._flow_outcomes.items():
            if len(outcomes) < 3:
                continue
            recent = outcomes[-10:]  # Last 10 outcomes
            success_rate = sum(recent) / len(recent)

            if flow == "fast_answer":
                # High success → trust beliefs more (lower confidence threshold)
                # Low success → require higher confidence before fast-answering
                if success_rate > 0.8:
                    self._min_confidence = max(0.3, self._min_confidence - 0.02)
                elif success_rate < 0.5:
                    self._min_confidence = min(0.9, self._min_confidence + 0.02)

            elif flow == "compile_on_demand":
                # Frequently triggered → beliefs are going stale faster
                # Reduce freshness window to proactively recompile
                if success_rate < 0.6:
                    self._freshness_hours = max(4.0, self._freshness_hours - 1.0)
                elif success_rate > 0.9:
                    self._freshness_hours = min(72.0, self._freshness_hours + 1.0)

            elif flow == "self_improvement":
                # High success → miss threshold is well calibrated
                # Low success → too many false positives, raise threshold
                if success_rate < 0.4:
                    self._miss_threshold = min(10, self._miss_threshold + 1)
                elif success_rate > 0.8 and self._miss_threshold > 2:
                    self._miss_threshold = max(2, self._miss_threshold - 1)

        logger.debug(
            f"EpistemicRouter self-tune: miss_threshold={self._miss_threshold}, "
            f"freshness_hours={self._freshness_hours:.1f}, "
            f"min_confidence={self._min_confidence:.2f}"
        )

    def stats(self) -> dict[str, Any]:
        """Return routing statistics for observability."""
        flow_counts: dict[str, int] = {}
        intent_counts: dict[str, int] = {}
        for d in self._history:
            flow_counts[d.flow.value] = flow_counts.get(d.flow.value, 0) + 1
            intent_counts[d.intent.value] = intent_counts.get(d.intent.value, 0) + 1

        return {
            "total_routed": len(self._history),
            "flow_distribution": flow_counts,
            "intent_distribution": intent_counts,
            "active_miss_entities": {
                k: v for k, v in self._miss_counts.items() if v > 0
            },
            "evolution_triggers": sum(
                1 for d in self._history
                if d.flow == EpistemicFlow.SELF_IMPROVEMENT
            ),
        }

    # ── Private Methods ──────────────────────────────────────────────

    def _select_flow(
        self,
        intent: EpistemicIntent,
        coverage: BeliefCoverage,
        risk: RiskLevel,
        is_event: bool,
        miss_count: int,
    ) -> tuple[EpistemicFlow, str]:
        """Apply the routing matrix to select a canonical flow."""

        # ④ Change-Driven Pipeline: event triggers bypass the query path
        if is_event:
            return (
                EpistemicFlow.CHANGE_DRIVEN,
                "Event-triggered: route through Truth → Belief → Verification → Action"
            )

        # ⑤ Self-Improvement: repeated misses trigger evolution
        if miss_count >= self._miss_threshold:
            return (
                EpistemicFlow.SELF_IMPROVEMENT,
                f"Repeated miss (count={miss_count}) on this entity: "
                f"logging for Evolution skill synthesis"
            )

        # No belief exists → must compile from Truth
        if not coverage.exists:
            return (
                EpistemicFlow.COMPILE_ON_DEMAND,
                "No relevant beliefs found: compile from Truth first"
            )

        # Beliefs exist but are stale or low-confidence → verify first
        if not coverage.fresh or coverage.confidence < self._min_confidence:
            return (
                EpistemicFlow.VERIFY_BEFORE_ANSWER,
                f"Beliefs exist but stale/low-confidence "
                f"(fresh={coverage.fresh}, conf={coverage.confidence:.2f}): "
                f"verify before answering"
            )

        # Beliefs exist but are not verified → verify first
        if not coverage.verified:
            return (
                EpistemicFlow.VERIFY_BEFORE_ANSWER,
                "Beliefs exist and are fresh but unverified: verify first"
            )

        # High-risk domain → always verify, even if beliefs look good
        if risk == RiskLevel.HIGH:
            return (
                EpistemicFlow.VERIFY_BEFORE_ANSWER,
                "High-risk domain: re-verify before answering"
            )

        # Code generation → always verify (you never ship unverified code)
        if intent == EpistemicIntent.CODE_GENERATION:
            return (
                EpistemicFlow.VERIFY_BEFORE_ANSWER,
                "Code generation: always verify before shipping"
            )

        # ① Fast Answer: everything checks out
        return (
            EpistemicFlow.FAST_ANSWER,
            f"Beliefs are fresh, verified, and confident "
            f"(conf={coverage.confidence:.2f}): fast answer"
        )

    def _check_coverage(
        self,
        query: str,
        intent: EpistemicIntent,
    ) -> BeliefCoverage:
        """Check belief coverage in the vault.

        Scans beliefs/ directory for artifacts matching query entities.
        Reads frontmatter to assess freshness, confidence, verification.
        """
        if not self._vault_path:
            return BeliefCoverage()

        beliefs_dir = self._vault_path / "beliefs"
        if not beliefs_dir.exists():
            return BeliefCoverage()

        # Extract key terms from query for matching
        terms = set(
            w.lower() for w in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', query)
            if len(w) > 2
        )
        if not terms:
            return BeliefCoverage()

        matching_claims: list[str] = []
        min_confidence = 1.0
        all_fresh = True
        all_verified = True
        stalest: str | None = None

        for md_file in beliefs_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                frontmatter = self._parse_frontmatter(content)
                if not frontmatter:
                    continue

                # Check if this belief is relevant to the query
                entity = frontmatter.get("entity", "").lower()
                claim_id = frontmatter.get("claim_id", "")
                file_stem = md_file.stem.lower()

                # Match on entity name, file name, or content terms
                matched = False
                for term in terms:
                    if term in entity or term in file_stem:
                        matched = True
                        break

                if not matched:
                    continue

                matching_claims.append(claim_id or file_stem)

                # Check confidence
                conf = float(frontmatter.get("confidence", 0.0))
                min_confidence = min(min_confidence, conf)

                # Check freshness
                status = frontmatter.get("status", "")
                if status == "stale":
                    all_fresh = False

                last_checked = frontmatter.get("last_checked", "")
                if last_checked:
                    if stalest is None or last_checked < stalest:
                        stalest = last_checked
                    # Simple freshness: check if last_checked is within threshold
                    try:
                        from datetime import datetime, timezone
                        checked_dt = datetime.fromisoformat(
                            last_checked.replace("Z", "+00:00")
                        )
                        now = datetime.now(timezone.utc)
                        hours_ago = (now - checked_dt).total_seconds() / 3600
                        if hours_ago > self._freshness_hours:
                            all_fresh = False
                    except (ValueError, TypeError):
                        pass

                # Check verification
                if status not in ("verified",):
                    all_verified = False

            except Exception as e:
                logger.debug(f"EpistemicRouter: failed to read {md_file}: {e}")
                continue

        if not matching_claims:
            return BeliefCoverage()

        return BeliefCoverage(
            exists=True,
            fresh=all_fresh,
            verified=all_verified,
            confidence=min_confidence if min_confidence <= 1.0 else 0.0,
            matching_claims=matching_claims,
            stalest_check=stalest,
        )

    def _parse_frontmatter(self, content: str) -> dict[str, str] | None:
        """Parse YAML frontmatter from a markdown file."""
        if not content.startswith("---"):
            return None
        end = content.find("---", 3)
        if end < 0:
            return None

        fm_text = content[3:end].strip()
        result: dict[str, str] = {}
        for line in fm_text.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key and value:
                    result[key] = value
        return result if result else None

    def _extract_entity_key(self, query: str) -> str:
        """Extract a stable entity key from a query for miss tracking.

        Priority: compound identifiers (snake_case / dot.path) first,
        then fall back to the longest non-stopword token.  This ensures
        rephrased queries about the same entity converge to the same key.
        """
        _STOP = {
            "how", "does", "the", "what", "is", "explain", "show", "can",
            "you", "please", "work", "works", "about", "for", "and", "this",
            "that", "with", "from", "have", "are", "module", "architecture",
            "design", "pipeline", "system", "component", "service",
        }
        # 1. Compound identifiers (snake_case, dot.path, ::path)
        compounds = re.findall(r'[a-zA-Z][a-zA-Z0-9]*(?:[_.:][a-zA-Z][a-zA-Z0-9]*)+', query)
        if compounds:
            # Pick the longest compound — most specific entity
            best = max(compounds, key=len)
            return best.lower().replace("::", "_").replace(".", "_")

        # 2. Longest non-stopword token
        tokens = [
            w.lower() for w in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', query)
            if len(w) > 3 and w.lower() not in _STOP
        ]
        if tokens:
            return max(tokens, key=len)

        return "unknown"

    def _classify_event_intent(self, event_type: str) -> EpistemicIntent:
        """Classify a change-driven event into an intent."""
        et = event_type.lower()
        if "pr" in et or "pull" in et or "merge" in et:
            return EpistemicIntent.PR_BRIEF
        if "incident" in et or "alert" in et:
            return EpistemicIntent.INCIDENT
        if "release" in et or "deploy" in et:
            return EpistemicIntent.RELEASE
        if "schedule" in et or "nightly" in et or "cron" in et:
            return EpistemicIntent.AUDIT
        return EpistemicIntent.ARCHITECTURE
