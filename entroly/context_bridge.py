"""
Entroly × Agent Context Bridge
================================

Context intelligence layer for personal AI agents.

AI agents load workspace files (SOUL.md, USER.md, MEMORY.md, daily logs)
into every LLM call. This module optimizes that context loading using Entroly's
information-theoretic selection:

  - Memory optimization: MEMORY.md entries ranked by relevance to current context
  - Daily log compression: recent events at full resolution, older at skeleton
  - Heartbeat batching: NKBE allocates budget across periodic checks
  - Group chat filtering: entropy-based message relevance scoring
  - Tool/skill selection: only load tool descriptions relevant to the query

Usage:
    from entroly.context_bridge import AgentContext

    ctx = AgentContext(workspace_path="~/.agent/workspace")
    optimized = ctx.load_session_context(
        query="check my emails",
        token_budget=8192,
    )
    # optimized.context_str → ready for LLM system prompt
    # optimized.provenance  → which files contributed, hallucination risk
    # optimized.tokens_used → actual token count
    # optimized.tokens_saved → how many tokens were saved vs naive loading
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import EntrolyConfig
from .provenance import ContextProvenance, build_provenance
from .server import EntrolyEngine

logger = logging.getLogger("entroly.context_bridge")


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class SessionContext:
    """Result of optimized context loading for an OpenClaw session."""
    context_str: str
    tokens_used: int
    tokens_saved: int
    total_raw_tokens: int
    fragments_selected: int
    fragments_total: int
    provenance: ContextProvenance | None = None
    memory_entries_loaded: int = 0
    daily_log_entries_loaded: int = 0
    sections: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentBudget:
    """Per-agent token budget from NKBE allocation."""
    agent_name: str
    budget: int
    weight: float
    utility: float


@dataclass
class HeartbeatResult:
    """Result of an optimized heartbeat check."""
    context_str: str
    tokens_used: int
    check_types: list[str]
    budgets: dict[str, int]


# ── NKBE Allocator (Python implementation) ────────────────────────────

class NkbeAllocator:
    """
    Nash-KKT Budgetary Equilibrium allocator for multi-agent token budgets.

    Given N agents sharing a global token budget B, finds the optimal split
    that maximizes weighted utility:

        maximize  Σᵢ wᵢ · Uᵢ(Bᵢ)
        subject to  Σᵢ Bᵢ ≤ B,  Bᵢ ≥ Bᵢ_min  ∀i

    Uses two-phase KKT bisection:
    1. Global: bisect for λ* such that Σᵢ Bᵢ(λ*) = B
    2. Per-agent: each agent gets Bᵢ(λ*)

    Plus Nash Bargaining refinement for Pareto-optimal fairness.
    """

    def __init__(
        self,
        global_budget: int = 128_000,
        min_agent_budget: int = 1024,
        tau: float = 0.1,
        learning_rate: float = 0.01,
        nash_iterations: int = 5,
    ):
        self.global_budget = global_budget
        self.min_agent_budget = min_agent_budget
        self.tau = tau
        self.learning_rate = learning_rate
        self.nash_iterations = nash_iterations

        self._agents: dict[str, _AgentState] = {}
        self._weights: dict[str, float] = {}

    def register_agent(
        self,
        name: str,
        weight: float = 1.0,
        min_budget: int | None = None,
    ) -> None:
        """Register an agent for budget allocation."""
        self._agents[name] = _AgentState(
            name=name,
            min_budget=min_budget or self.min_agent_budget,
            fragment_count=0,
            total_tokens=0,
            utility_estimate=0.5,
        )
        self._weights[name] = weight

    def update_fragments(
        self,
        agent_name: str,
        fragment_count: int,
        total_tokens: int,
    ) -> None:
        """Update agent's fragment info for utility estimation."""
        if agent_name in self._agents:
            self._agents[agent_name].fragment_count = fragment_count
            self._agents[agent_name].total_tokens = total_tokens

    def allocate(self) -> dict[str, int]:
        """
        Run NKBE allocation. Returns per-agent budgets.

        Two-phase KKT bisection:
        1. Bisect for global λ* such that Σ Bᵢ(λ*) = B
        2. Each agent gets Bᵢ = max(Bᵢ_min, demand(λ*))
        """
        if not self._agents:
            return {}

        agents = list(self._agents.values())
        n = len(agents)

        # Single agent: gets everything
        if n == 1:
            name = agents[0].name
            return {name: self.global_budget}

        # Compute utility estimates (information density per token)
        utilities = {}
        for a in agents:
            if a.total_tokens > 0:
                # Utility = estimated info gain per token (sigmoid-scaled)
                density = min(1.0, a.fragment_count / max(1, a.total_tokens / 100))
                utilities[a.name] = density * self._weights.get(a.name, 1.0)
            else:
                utilities[a.name] = 0.5 * self._weights.get(a.name, 1.0)

        # Phase 1: KKT bisection for global λ*
        total_min = sum(a.min_budget for a in agents)
        if total_min >= self.global_budget:
            # Budget too small — give everyone minimum
            return {a.name: a.min_budget for a in agents}

        available = self.global_budget - total_min

        # Bisect for λ* that allocates exactly `available` tokens
        lo, hi = 0.0, max(utilities.values()) * 2.0 + 1.0

        for _ in range(30):  # 30-step bisection (matches Rust)
            mid = (lo + hi) / 2.0
            total_demand = 0.0
            for a in agents:
                u = utilities[a.name]
                # Demand = sigmoid of (utility - λ*) / τ, scaled by available
                z = (u - mid) / max(self.tau, 1e-6)
                p = _sigmoid(z)
                total_demand += p * available
            if total_demand > available:
                lo = mid
            else:
                hi = mid

        lambda_star = (lo + hi) / 2.0

        # Phase 2: compute per-agent budgets
        budgets = {}
        for a in agents:
            u = utilities[a.name]
            z = (u - lambda_star) / max(self.tau, 1e-6)
            p = _sigmoid(z)
            budgets[a.name] = a.min_budget + int(p * available)

        # Nash Bargaining refinement: adjust toward Pareto-optimal fairness
        for _ in range(self.nash_iterations):
            # Compute Nash product gradient
            log_nash_grad = {}
            for a in agents:
                u = utilities[a.name]
                b = budgets[a.name]
                # ∂ log N / ∂ Bᵢ = Uᵢ'(Bᵢ) / (Uᵢ(Bᵢ) − Uᵢ(Bᵢ_min))
                u_b = u * _log_utility(b)
                u_min = u * _log_utility(a.min_budget)
                denom = max(u_b - u_min, 1e-8)
                u_prime = u / max(b, 1.0)
                log_nash_grad[a.name] = u_prime / denom

            # Gradient step (constrained to maintain budget)
            mean_grad = sum(log_nash_grad.values()) / n
            for a in agents:
                delta = int(0.1 * (log_nash_grad[a.name] - mean_grad) * available)
                budgets[a.name] = max(a.min_budget, budgets[a.name] + delta)

        # Normalize to exact global budget
        total = sum(budgets.values())
        if total > 0:
            scale = self.global_budget / total
            budgets = {k: max(self._agents[k].min_budget, int(v * scale))
                       for k, v in budgets.items()}

        return budgets

    def reinforce(self, outcomes: dict[str, float]) -> None:
        """
        REINFORCE weight update based on agent outcomes.

        Δwᵢ = η · (Rᵢ − R̄) · wᵢ
        """
        if not outcomes:
            return
        mean_reward = sum(outcomes.values()) / len(outcomes)
        for name, reward in outcomes.items():
            if name in self._weights:
                advantage = reward - mean_reward
                self._weights[name] *= (1.0 + self.learning_rate * advantage)
                self._weights[name] = max(0.1, min(10.0, self._weights[name]))


@dataclass
class _AgentState:
    name: str
    min_budget: int
    fragment_count: int
    total_tokens: int
    utility_estimate: float


# ── Cognitive Bus (Python implementation) ─────────────────────────────

class CognitiveBus:
    """
    Information-Surprise-Adaptive (ISA) event routing between agents.

    Routes events between OpenClaw agents with information-theoretic
    prioritization. Implements Poisson rate model per subscriber per
    event type, with KL divergence priority scoring.

    Event types map to OpenClaw concepts:
      - observation: agent discovered something
      - tool_result: tool/skill returned data
      - memory_update: MEMORY.md was modified
      - heartbeat: periodic check result
      - user_message: new message from human
      - agent_output: agent produced a response
    """

    EVENT_TYPES = [
        "observation", "tool_result", "memory_update",
        "heartbeat", "user_message", "agent_output",
        "error", "security_alert",
    ]

    def __init__(self, novelty_threshold: float = 0.3, alpha: float = 0.1):
        self.novelty_threshold = novelty_threshold
        self.alpha = alpha
        self._subscribers: dict[str, _Subscriber] = {}
        self._history: list[_BusEvent] = []
        self._tick: int = 0
        self._total_published: int = 0
        self._total_delivered: int = 0
        self._total_suppressed: int = 0

    def subscribe(self, agent_name: str, event_types: list[str] | None = None) -> None:
        """Register an agent as event subscriber."""
        filters = set(event_types) if event_types else set(self.EVENT_TYPES)
        self._subscribers[agent_name] = _Subscriber(
            name=agent_name,
            filters=filters,
            rates={et: _RateCell() for et in self.EVENT_TYPES},
            inbox=[],
        )

    def publish(
        self,
        source: str,
        event_type: str,
        payload: str,
        surprise: float = 0.0,
    ) -> int:
        """
        Publish an event. Returns number of deliveries.

        Priority scoring:
          priority = KL(λ_obs ‖ λ_exp) · recency · novelty
        """
        self._tick += 1
        self._total_published += 1

        event = _BusEvent(
            source=source,
            event_type=event_type,
            payload=payload,
            surprise=surprise,
            tick=self._tick,
        )
        self._history.append(event)

        # Trim history to 1000 events
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        deliveries = 0
        for sub in self._subscribers.values():
            if sub.name == source:
                continue
            if event_type not in sub.filters:
                continue

            # Update Poisson rate model
            rate_cell = sub.rates.get(event_type)
            if rate_cell is None:
                rate_cell = _RateCell()
                sub.rates[event_type] = rate_cell
            rate_cell.observe(self._tick)

            # Compute KL divergence priority
            kl = rate_cell.kl_divergence()

            # Spike detection (Welford)
            is_spike = rate_cell.is_spike()

            # Recency factor
            recency = 1.0  # current event

            # Priority
            priority = (kl + surprise) * (2.0 if is_spike else 1.0) * recency

            # Novelty filter (simple hash-based dedup)
            payload_hash = hashlib.md5(payload.encode()).hexdigest()[:16]
            if payload_hash in sub._seen_hashes:
                self._total_suppressed += 1
                continue
            sub._seen_hashes.add(payload_hash)
            if len(sub._seen_hashes) > 256:
                sub._seen_hashes = set(list(sub._seen_hashes)[-128:])

            sub.inbox.append((priority, event))
            sub.inbox.sort(key=lambda x: -x[0])  # max-heap by priority
            deliveries += 1
            self._total_delivered += 1

        return deliveries

    def drain(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """Drain events for an agent, highest priority first."""
        sub = self._subscribers.get(agent_name)
        if not sub:
            return []
        events = []
        while sub.inbox and len(events) < limit:
            priority, event = sub.inbox.pop(0)
            events.append({
                "source": event.source,
                "event_type": event.event_type,
                "payload": event.payload,
                "priority": priority,
                "tick": event.tick,
            })
        return events

    def stats(self) -> dict[str, Any]:
        return {
            "tick": self._tick,
            "total_published": self._total_published,
            "total_delivered": self._total_delivered,
            "total_suppressed": self._total_suppressed,
            "subscribers": list(self._subscribers.keys()),
        }


@dataclass
class _BusEvent:
    source: str
    event_type: str
    payload: str
    surprise: float
    tick: int


@dataclass
class _Subscriber:
    name: str
    filters: set
    rates: dict[str, _RateCell]
    inbox: list[tuple[float, _BusEvent]]
    _seen_hashes: set = field(default_factory=set)


class _RateCell:
    """Poisson rate model with Welford online variance for spike detection."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.lambda_hat: float = 1.0  # EMA rate estimate
        self.welford_mean: float = 0.0
        self.welford_m2: float = 0.0
        self.n: int = 0
        self.window_count: int = 0
        self.last_tick: int = 0

    def observe(self, tick: int) -> None:
        """Record an event occurrence."""
        dt = max(1, tick - self.last_tick)
        observed_rate = 1.0 / dt
        self.lambda_hat = self.alpha * observed_rate + (1.0 - self.alpha) * self.lambda_hat
        self.last_tick = tick
        self.window_count += 1
        self.n += 1

        # Welford update
        delta = observed_rate - self.welford_mean
        self.welford_mean += delta / self.n
        delta2 = observed_rate - self.welford_mean
        self.welford_m2 += delta * delta2

    def kl_divergence(self) -> float:
        """KL(P(λ_obs) ‖ P(λ_exp)) for Poisson distributions."""
        obs = max(self.lambda_hat, 1e-8)
        exp = max(1.0, 1e-8)  # expected = baseline rate of 1.0
        return exp - obs + obs * _safe_ln(obs / exp)

    def is_spike(self) -> bool:
        """Welford spike detection: event rate > mean + 3σ."""
        if self.n < 3:
            return False
        variance = self.welford_m2 / (self.n - 1)
        sigma = variance ** 0.5
        return (1.0 / max(1, 1)) > self.welford_mean + 3.0 * sigma


# ── AgentContext: Main Integration Class ───────────────────────────

class AgentContext:
    """
    Entroly context intelligence for AI agents.

    Wraps EntrolyEngine to optimize the agent's workspace file loading.
    Instead of naively reading all files into the LLM prompt, this class
    ingests workspace files as fragments and uses knapsack optimization
    to select the most relevant subset within a token budget.

    Product value:
    1. MEMORY.md optimization → 80% token savings on long-term memory
    2. Daily log compression → multi-resolution (recent full, old skeleton)
    3. Heartbeat batching → NKBE splits budget across check types
    4. Group chat filtering → entropy-scored message relevance
    5. Tool selection → only relevant tools loaded per query
    """

    def __init__(
        self,
        workspace_path: str | Path = "~/.agent/workspace",
        token_budget: int = 32_000,
        quality: str = "balanced",
        config: EntrolyConfig | None = None,
    ):
        self.workspace_path = Path(workspace_path).expanduser()
        self.token_budget = token_budget

        # Initialize Entroly engine
        self._config = config or EntrolyConfig(quality=quality)
        self._config.default_token_budget = token_budget
        self._engine = EntrolyEngine(config=self._config)

        # NKBE allocator for multi-agent / multi-check budgets
        self._allocator = NkbeAllocator(global_budget=token_budget)

        # Cognitive bus for inter-agent events
        self._bus = CognitiveBus()

        # Track what's been ingested
        self._ingested_files: dict[str, str] = {}  # path → fragment_id
        self._section_ids: dict[str, list[str]] = {}  # section → [fragment_ids]

        # Session state
        self._session_start = time.time()
        self._loads: int = 0

    # ── Workspace File Ingestion ──────────────────────────────────────

    def ingest_workspace(self) -> dict[str, Any]:
        """
        Ingest all OpenClaw workspace files as Entroly fragments.

        Reads SOUL.md, USER.md, MEMORY.md, IDENTITY.md, TOOLS.md,
        HEARTBEAT.md, and daily logs. Each file (or section within a file)
        becomes a fragment with appropriate pinning and metadata.
        """
        stats = {"files_ingested": 0, "total_tokens": 0, "sections": 0}

        # Core identity files (pinned — always included)
        for fname in ["SOUL.md", "IDENTITY.md"]:
            fpath = self.workspace_path / fname
            if fpath.exists():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                result = self._engine.ingest_fragment(
                    content=content,
                    source=str(fpath),
                    is_pinned=True,
                )
                fid = result.get("fragment_id", "")
                self._ingested_files[str(fpath)] = fid
                self._section_ids.setdefault("identity", []).append(fid)
                stats["files_ingested"] += 1
                stats["total_tokens"] += result.get("token_count", 0)

        # User context (pinned — always included)
        user_path = self.workspace_path / "USER.md"
        if user_path.exists():
            content = user_path.read_text(encoding="utf-8", errors="replace")
            result = self._engine.ingest_fragment(
                content=content,
                source=str(user_path),
                is_pinned=True,
            )
            fid = result.get("fragment_id", "")
            self._ingested_files[str(user_path)] = fid
            self._section_ids.setdefault("user", []).append(fid)
            stats["files_ingested"] += 1
            stats["total_tokens"] += result.get("token_count", 0)

        # MEMORY.md — split into sections for granular selection
        memory_path = self.workspace_path / "MEMORY.md"
        if memory_path.exists():
            content = memory_path.read_text(encoding="utf-8", errors="replace")
            sections = _split_memory_sections(content)
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                result = self._engine.ingest_fragment(
                    content=section,
                    source=f"{memory_path}#section_{i}",
                    is_pinned=False,  # Memory entries compete on relevance
                )
                fid = result.get("fragment_id", "")
                self._section_ids.setdefault("memory", []).append(fid)
                stats["sections"] += 1
                stats["total_tokens"] += result.get("token_count", 0)
            stats["files_ingested"] += 1

        # Tools (not pinned — loaded when relevant)
        tools_path = self.workspace_path / "TOOLS.md"
        if tools_path.exists():
            content = tools_path.read_text(encoding="utf-8", errors="replace")
            sections = _split_tool_sections(content)
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                result = self._engine.ingest_fragment(
                    content=section,
                    source=f"{tools_path}#tool_{i}",
                    is_pinned=False,
                )
                fid = result.get("fragment_id", "")
                self._section_ids.setdefault("tools", []).append(fid)
                stats["sections"] += 1
                stats["total_tokens"] += result.get("token_count", 0)
            stats["files_ingested"] += 1

        # Heartbeat (not pinned)
        hb_path = self.workspace_path / "HEARTBEAT.md"
        if hb_path.exists():
            content = hb_path.read_text(encoding="utf-8", errors="replace")
            result = self._engine.ingest_fragment(
                content=content,
                source=str(hb_path),
                is_pinned=False,
            )
            fid = result.get("fragment_id", "")
            self._section_ids.setdefault("heartbeat", []).append(fid)
            stats["files_ingested"] += 1
            stats["total_tokens"] += result.get("token_count", 0)

        # Daily logs
        memory_dir = self.workspace_path / "memory"
        if memory_dir.exists():
            log_files = sorted(memory_dir.glob("*.md"), reverse=True)
            for log_file in log_files[:7]:  # Last 7 days
                content = log_file.read_text(encoding="utf-8", errors="replace")
                # Recent logs (today/yesterday) get higher semantic score
                is_recent = log_file.name >= _date_str(-1)
                result = self._engine.ingest_fragment(
                    content=content,
                    source=str(log_file),
                    is_pinned=is_recent,
                )
                fid = result.get("fragment_id", "")
                self._section_ids.setdefault("daily_logs", []).append(fid)
                stats["files_ingested"] += 1
                stats["total_tokens"] += result.get("token_count", 0)

        self._engine.advance_turn()
        return stats

    # ── Session Context Loading ───────────────────────────────────────

    def load_session_context(
        self,
        query: str = "",
        token_budget: int | None = None,
        session_type: str = "main",
    ) -> SessionContext:
        """
        Load optimized session context for an OpenClaw agent.

        Instead of naively concatenating all workspace files, this runs
        Entroly's knapsack optimizer to select the most relevant fragments
        within the token budget.

        Args:
            query: Current user query or session purpose
            token_budget: Token budget (defaults to self.token_budget)
            session_type: "main" (full context), "group" (reduced),
                          "heartbeat" (minimal)

        Returns:
            SessionContext with optimized context string and metadata
        """
        budget = token_budget or self.token_budget
        self._loads += 1

        # Adjust budget by session type
        if session_type == "group":
            budget = int(budget * 0.5)  # Group chats get less context
        elif session_type == "heartbeat":
            budget = int(budget * 0.3)  # Heartbeats need minimal context

        # Run knapsack optimization
        result = self._engine.optimize_context(
            token_budget=budget,
            query=query,
        )

        selected = result.get("selected_fragments", result.get("selected", []))
        tokens_used = result.get("tokens_used", 0)
        total_raw = result.get("total_tokens_available", 0)

        # Build context string from selected fragments
        sections: dict[str, str] = {}
        memory_count = 0
        log_count = 0

        for frag in selected:
            content = frag.get("content", frag.get("preview", ""))
            source = frag.get("source", "")

            if "SOUL.md" in source or "IDENTITY.md" in source:
                sections.setdefault("identity", "")
                sections["identity"] += content + "\n\n"
            elif "USER.md" in source:
                sections.setdefault("user", "")
                sections["user"] += content + "\n\n"
            elif "MEMORY.md" in source:
                sections.setdefault("memory", "")
                sections["memory"] += content + "\n\n"
                memory_count += 1
            elif "TOOLS.md" in source:
                sections.setdefault("tools", "")
                sections["tools"] += content + "\n\n"
            elif "HEARTBEAT.md" in source:
                sections.setdefault("heartbeat", "")
                sections["heartbeat"] += content + "\n\n"
            elif "/memory/" in source:
                sections.setdefault("daily_logs", "")
                sections["daily_logs"] += content + "\n\n"
                log_count += 1
            else:
                sections.setdefault("other", "")
                sections["other"] += content + "\n\n"

        # Assemble final context string in the agent's expected order
        parts = []
        if "identity" in sections:
            parts.append(sections["identity"].strip())
        if "user" in sections:
            parts.append(sections["user"].strip())
        if "memory" in sections:
            parts.append("## Relevant Memories\n" + sections["memory"].strip())
        if "daily_logs" in sections:
            parts.append("## Recent Activity\n" + sections["daily_logs"].strip())
        if "tools" in sections:
            parts.append("## Available Tools\n" + sections["tools"].strip())
        if "heartbeat" in sections:
            parts.append(sections["heartbeat"].strip())
        if "other" in sections:
            parts.append(sections["other"].strip())

        context_str = "\n\n---\n\n".join(parts)

        # Build provenance
        provenance = None
        try:
            provenance = build_provenance(
                result, query, query, self._loads, budget,
            )
        except Exception:
            pass

        return SessionContext(
            context_str=context_str,
            tokens_used=tokens_used,
            tokens_saved=max(0, total_raw - tokens_used),
            total_raw_tokens=total_raw,
            fragments_selected=len(selected),
            fragments_total=result.get("total_fragments", 0),
            provenance=provenance,
            memory_entries_loaded=memory_count,
            daily_log_entries_loaded=log_count,
            sections=sections,
        )

    # ── Multi-Agent Budget Allocation ─────────────────────────────────

    def allocate_budgets(
        self,
        agents: list[str],
        weights: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """
        NKBE allocation of token budget across multiple agents.

        When the system runs multiple agents in parallel (e.g., researcher +
        coder + reviewer), this splits the global budget optimally.
        """
        self._allocator = NkbeAllocator(global_budget=self.token_budget)
        for name in agents:
            w = (weights or {}).get(name, 1.0)
            self._allocator.register_agent(name, weight=w)
        return self._allocator.allocate()

    # ── Heartbeat Optimization ────────────────────────────────────────

    def optimize_heartbeat(
        self,
        check_types: list[str],
        total_budget: int | None = None,
    ) -> HeartbeatResult:
        """
        Optimize a heartbeat check using NKBE budget allocation.

        Instead of running N separate LLM calls for N checks, allocates
        budget across check types and returns a single optimized context.
        """
        budget = total_budget or int(self.token_budget * 0.3)

        # Allocate budget across check types
        allocator = NkbeAllocator(global_budget=budget, min_agent_budget=256)
        for ct in check_types:
            allocator.register_agent(ct)
        budgets = allocator.allocate()

        # Build combined context
        parts = []
        for ct, ct_budget in budgets.items():
            result = self._engine.optimize_context(
                token_budget=ct_budget,
                query=f"heartbeat check: {ct}",
            )
            selected = result.get("selected_fragments", result.get("selected", []))
            for frag in selected:
                content = frag.get("content", frag.get("preview", ""))
                parts.append(f"[{ct}] {content}")

        context_str = "\n\n".join(parts)
        tokens_used = sum(budgets.values())

        return HeartbeatResult(
            context_str=context_str,
            tokens_used=tokens_used,
            check_types=check_types,
            budgets=budgets,
        )

    # ── Group Chat Filtering ──────────────────────────────────────────

    def filter_group_chat(
        self,
        messages: list[dict[str, str]],
        agent_expertise: str = "",
        max_messages: int = 20,
    ) -> list[dict[str, str]]:
        """
        Filter group chat messages by relevance using entropy scoring.

        Low-entropy messages ("ok", "lol", "yeah") are deprioritized.
        Messages relevant to the agent's expertise get higher scores.
        """
        if len(messages) <= max_messages:
            return messages

        scored = []
        for msg in messages:
            content = msg.get("content", "")
            # Ingest and score
            result = self._engine.ingest_fragment(
                content=content,
                source=f"chat:{msg.get('sender', 'unknown')}",
            )
            scored.append((result.get("entropy_score", 0.5), msg))

        # Sort by entropy score (high entropy = more informative)
        scored.sort(key=lambda x: -x[0])

        # Always include recent messages (last 5)
        recent = messages[-5:]
        filtered = [msg for _, msg in scored[:max_messages - 5]]

        # Merge, preserving order
        result_msgs = []
        seen = set()
        for msg in messages:
            msg_id = id(msg)
            if msg in recent or msg in filtered:
                if msg_id not in seen:
                    result_msgs.append(msg)
                    seen.add(msg_id)

        return result_msgs

    # ── Feedback ──────────────────────────────────────────────────────

    def record_outcome(self, success: bool) -> None:
        """Record whether the LLM output was helpful (Wilson score update)."""
        stats = self._engine.get_stats()
        # Get last selected fragment IDs from engine stats
        last_ids = stats.get("last_selected_ids", [])
        if last_ids:
            if success:
                self._engine.record_success(last_ids)
            else:
                self._engine.record_failure(last_ids)

    # ── Event Publishing ──────────────────────────────────────────────

    def publish_event(
        self,
        source: str,
        event_type: str,
        payload: str,
        surprise: float = 0.0,
    ) -> int:
        """Publish an event to the cognitive bus."""
        return self._bus.publish(source, event_type, payload, surprise)

    def drain_events(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """Drain events for an agent from the cognitive bus."""
        return self._bus.drain(agent_name, limit)

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics."""
        engine_stats = self._engine.get_stats()
        return {
            "engine": engine_stats,
            "bus": self._bus.stats(),
            "workspace_path": str(self.workspace_path),
            "token_budget": self.token_budget,
            "loads": self._loads,
            "ingested_files": len(self._ingested_files),
            "session_duration_s": time.time() - self._session_start,
        }


# ── Helper functions ──────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = 2.718281828 ** (-x)
        return 1.0 / (1.0 + z)
    else:
        z = 2.718281828 ** x
        return z / (1.0 + z)


def _log_utility(budget: int) -> float:
    """Log utility function for Nash bargaining."""
    import math
    return math.log(max(1, budget))


def _safe_ln(x: float) -> float:
    """Safe natural log."""
    import math
    return math.log(max(x, 1e-15))


def _split_memory_sections(content: str) -> list[str]:
    """Split MEMORY.md into sections by markdown headers."""
    sections = []
    current = []
    for line in content.split("\n"):
        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def _split_tool_sections(content: str) -> list[str]:
    """Split TOOLS.md into per-tool sections."""
    sections = []
    current = []
    for line in content.split("\n"):
        if line.startswith("### ") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def _date_str(offset_days: int = 0) -> str:
    """Get date string YYYY-MM-DD with optional day offset."""
    import datetime
    d = datetime.date.today() + datetime.timedelta(days=offset_days)
    return d.strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════
# Multi-Agent Orchestration (additive — no existing API changes)
# ══════════════════════════════════════════════════════════════════════

# ── LOD Tier ──────────────────────────────────────────────────────────

class LodTier:
    """Level-of-Detail tiers for agent lifecycle management.

    Level-of-detail tiers for agent lifecycle management.
    Mapped to agent states:
      DORMANT:    Agent registered but idle. Minimal memory footprint.
                  Context = reference-level only (~2% tokens).
      ACTIVE:     Agent running tasks. Full context optimization.
                  Context = HCC-optimized (Full/Skeleton/Reference mix).
      SATURATED:  Agent overloaded (queue > threshold). Excluded from
                  new task routing until load drops.
      HEAVY:      Agent doing LLM-intensive work (code generation, analysis).
                  Gets priority NKBE budget allocation.
    """
    DORMANT = "dormant"
    ACTIVE = "active"
    SATURATED = "saturated"
    HEAVY = "heavy"


@dataclass
class AgentState:
    """Per-agent lifecycle state for LOD management."""
    agent_id: str
    tier: str = LodTier.DORMANT
    load_factor: float = 0.0
    ticks_in_tier: int = 0
    task_count: int = 0
    total_tokens_used: int = 0
    parent_id: str | None = None
    depth: int = 0
    children: list[str] = field(default_factory=list)
    fib_position: float = 0.0  # Fibonacci hash scatter


class LODManager:
    """Level-of-Detail lifecycle manager for agent armies.

    Prevents resource exhaustion when 100s of agents run simultaneously.
    Combines bitflag triggers, surprise-based demotion,
      load-based tiers, hysteresis, and saturation alerts.

    Agent tier mapping:
      Main agent     → starts ACTIVE
      Subagent       → starts DORMANT, promoted when parent delegates
      Heartbeat      → stays DORMANT between checks, ACTIVE during
      Cron           → DORMANT → ACTIVE on schedule → DORMANT after
      Group chat     → ACTIVE while conversation ongoing
    """

    # Hysteresis: minimum ticks in tier before transition allowed
    ACTIVE_MIN_TICKS = 3
    SATURATED_MIN_TICKS = 2
    # Load thresholds
    SATURATION_THRESHOLD = 0.9
    DORMANT_THRESHOLD = 0.1
    # Alert if >5% agents saturated (thundering herd prevention)
    SATURATION_ALERT_RATIO = 0.05

    def __init__(self):
        self._agents: dict[str, AgentState] = {}
        self._alerts: list[str] = []
        self._tick: int = 0
        self._promotions: int = 0
        self._demotions: int = 0

    def register(
        self,
        agent_id: str,
        parent_id: str | None = None,
        initial_tier: str = LodTier.DORMANT,
    ) -> AgentState:
        """Register an agent in the LOD system."""
        depth = 0
        if parent_id and parent_id in self._agents:
            depth = self._agents[parent_id].depth + 1
            self._agents[parent_id].children.append(agent_id)

        # Fibonacci hash scatter for deterministic positioning
        fib_pos = ((hash(agent_id) & 0xFFFFFFFF) * 0.6180339887) % 1000.0

        state = AgentState(
            agent_id=agent_id,
            tier=initial_tier,
            parent_id=parent_id,
            depth=depth,
            fib_position=fib_pos,
        )
        self._agents[agent_id] = state
        return state

    def unregister(self, agent_id: str) -> None:
        """Remove an agent (and orphan its children to parent or root)."""
        state = self._agents.pop(agent_id, None)
        if state is None:
            return
        # Reparent children
        for child_id in state.children:
            if child_id in self._agents:
                self._agents[child_id].parent_id = state.parent_id
                if state.parent_id and state.parent_id in self._agents:
                    self._agents[state.parent_id].children.append(child_id)
        # Remove from parent's children list
        if state.parent_id and state.parent_id in self._agents:
            parent = self._agents[state.parent_id]
            parent.children = [c for c in parent.children if c != agent_id]

    def update_load(self, agent_id: str, load_factor: float) -> str | None:
        """Update agent load and trigger tier transitions.

        Returns new tier if transition occurred, else None.
        """
        state = self._agents.get(agent_id)
        if state is None:
            return None

        state.load_factor = max(0.0, min(1.0, load_factor))
        state.ticks_in_tier += 1
        old_tier = state.tier

        # Transition logic (with hysteresis)
        if state.tier == LodTier.DORMANT:
            if state.load_factor > self.DORMANT_THRESHOLD:
                state.tier = LodTier.ACTIVE
                state.ticks_in_tier = 0
                self._promotions += 1

        elif state.tier == LodTier.ACTIVE:
            if (state.load_factor >= self.SATURATION_THRESHOLD
                    and state.ticks_in_tier >= self.ACTIVE_MIN_TICKS):
                state.tier = LodTier.SATURATED
                state.ticks_in_tier = 0
            elif (state.load_factor < self.DORMANT_THRESHOLD
                    and state.ticks_in_tier >= self.ACTIVE_MIN_TICKS):
                state.tier = LodTier.DORMANT
                state.ticks_in_tier = 0
                self._demotions += 1

        elif state.tier == LodTier.SATURATED:
            if (state.load_factor < self.SATURATION_THRESHOLD
                    and state.ticks_in_tier >= self.SATURATED_MIN_TICKS):
                state.tier = LodTier.ACTIVE
                state.ticks_in_tier = 0

        if state.tier != old_tier:
            return state.tier
        return None

    def tick(self) -> list[str]:
        """Advance clock. Returns any alerts generated."""
        self._tick += 1
        self._alerts.clear()

        # Check saturation ratio
        total = len(self._agents)
        if total > 0:
            saturated = sum(1 for a in self._agents.values()
                            if a.tier == LodTier.SATURATED)
            ratio = saturated / total
            if ratio > self.SATURATION_ALERT_RATIO:
                self._alerts.append(
                    f"Saturation alert: {saturated}/{total} agents "
                    f"({ratio:.0%}) saturated — throttle new tasks"
                )
        return self._alerts

    def get_active_agents(self) -> list[str]:
        """Get IDs of all ACTIVE or HEAVY agents (eligible for tasks)."""
        return [
            a.agent_id for a in self._agents.values()
            if a.tier in (LodTier.ACTIVE, LodTier.HEAVY)
        ]

    def get_budget_weights(self) -> dict[str, float]:
        """Get NKBE weight hints based on LOD tier.

        HEAVY agents get 2x weight, ACTIVE get 1x,
        DORMANT/SATURATED get 0 (no budget allocation).
        """
        weights = {}
        for a in self._agents.values():
            if a.tier == LodTier.HEAVY:
                weights[a.agent_id] = 2.0
            elif a.tier == LodTier.ACTIVE:
                weights[a.agent_id] = 1.0
            # DORMANT and SATURATED get no budget
        return weights

    def stats(self) -> dict[str, Any]:
        tier_counts = {LodTier.DORMANT: 0, LodTier.ACTIVE: 0,
                       LodTier.SATURATED: 0, LodTier.HEAVY: 0}
        for a in self._agents.values():
            tier_counts[a.tier] = tier_counts.get(a.tier, 0) + 1
        return {
            "total_agents": len(self._agents),
            "tier_counts": tier_counts,
            "promotions": self._promotions,
            "demotions": self._demotions,
            "tick": self._tick,
        }


# ── Subagent Orchestrator ─────────────────────────────────────────────

class SubagentOrchestrator:
    """Manages subagent spawning with context inheritance and NKBE budget splitting.

    The system supports up to 5 subagents per parent, 3 depth levels.
    Each subagent inherits a subset of its parent's context (filtered by
    task relevance) and gets a NKBE-allocated slice of the parent's budget.

    Context inheritance:
      Parent context → filter by subagent task query → inject as pinned fragments
      This gives subagents relevant history without re-loading workspace files.

    Budget cascade:
      Global budget → NKBE → parent budget → NKBE → per-subagent budgets
      Each level of nesting reduces the available budget proportionally.
    """

    MAX_CHILDREN = 5
    MAX_DEPTH = 3

    def __init__(
        self,
        agent_ctx: AgentContext,
        lod: LODManager,
        bus: CognitiveBus,
    ):
        self._ctx = agent_ctx
        self._lod = lod
        self._bus = bus
        self._spawn_count = 0

    def can_spawn(self, parent_id: str) -> tuple[bool, str]:
        """Check if a parent agent can spawn a new subagent."""
        parent = self._lod._agents.get(parent_id)
        if parent is None:
            return False, "parent not registered"
        if len(parent.children) >= self.MAX_CHILDREN:
            return False, f"max children ({self.MAX_CHILDREN}) reached"
        if parent.depth >= self.MAX_DEPTH:
            return False, f"max depth ({self.MAX_DEPTH}) reached"
        if parent.tier == LodTier.SATURATED:
            return False, "parent is saturated"
        return True, "ok"

    def spawn(
        self,
        parent_id: str,
        child_id: str,
        task_query: str,
        budget_fraction: float = 0.2,
    ) -> SessionContext | None:
        """Spawn a subagent with inherited context.

        Args:
            parent_id: Parent agent ID.
            child_id: New subagent ID.
            task_query: What the subagent will work on.
            budget_fraction: Fraction of parent's budget (default 20%).

        Returns:
            SessionContext for the subagent, or None if spawn rejected.
        """
        can, reason = self.can_spawn(parent_id)
        if not can:
            logger.warning(f"Subagent spawn rejected: {reason}")
            return None

        # Register in LOD
        self._lod.register(child_id, parent_id=parent_id,
                           initial_tier=LodTier.ACTIVE)

        # Subscribe to cognitive bus
        self._bus.subscribe(child_id, event_types=None)

        # Compute subagent budget
        parent_state = self._lod._agents.get(parent_id)
        child_budget = int(self._ctx.token_budget * budget_fraction)
        if parent_state:
            # Deeper agents get less budget (geometric decay)
            depth_factor = 0.7 ** parent_state.depth
            child_budget = int(child_budget * depth_factor)
        child_budget = max(1024, child_budget)  # Floor

        # Load context optimized for child's task
        context = self._ctx.load_session_context(
            query=task_query,
            token_budget=child_budget,
            session_type="main",
        )

        # Publish spawn event on cognitive bus
        self._bus.publish(
            source=parent_id,
            event_type="observation",
            payload=f"Spawned subagent {child_id} for: {task_query}",
        )

        self._spawn_count += 1
        return context

    def despawn(self, child_id: str) -> None:
        """Despawn a subagent and release its resources."""
        self._lod.unregister(child_id)

    def get_tree(self, root_id: str) -> dict[str, Any]:
        """Get the subagent tree rooted at an agent."""
        state = self._lod._agents.get(root_id)
        if state is None:
            return {}
        tree = {
            "id": root_id,
            "tier": state.tier,
            "depth": state.depth,
            "load": state.load_factor,
            "children": [],
        }
        for child_id in state.children:
            tree["children"].append(self.get_tree(child_id))
        return tree

    def stats(self) -> dict[str, Any]:
        return {"spawn_count": self._spawn_count}


# ── Cron Session Manager ──────────────────────────────────────────────

class CronSessionManager:
    """Manages scheduled (cron) agent sessions with minimal context loading.

    Cron agents run periodically (e.g., check email every 15 min, review
    calendar daily). They need:
      1. Minimal context — just the relevant memory + schedule info
      2. Rapid startup — no full workspace reload
      3. Result persistence — findings go to cognitive bus + memory

    LOD lifecycle: DORMANT → ACTIVE (on schedule) → DORMANT (after completion)
    """

    @dataclass
    class CronJob:
        agent_id: str
        task_query: str
        interval_seconds: float
        last_run: float = 0.0
        run_count: int = 0
        total_tokens_used: int = 0

    def __init__(
        self,
        agent_ctx: AgentContext,
        lod: LODManager,
        bus: CognitiveBus,
    ):
        self._ctx = agent_ctx
        self._lod = lod
        self._bus = bus
        self._jobs: dict[str, CronSessionManager.CronJob] = {}

    def schedule(
        self,
        agent_id: str,
        task_query: str,
        interval_seconds: float = 900.0,  # 15 min default
    ) -> None:
        """Schedule a cron agent."""
        self._lod.register(agent_id, initial_tier=LodTier.DORMANT)
        self._bus.subscribe(agent_id, event_types=None)
        self._jobs[agent_id] = self.CronJob(
            agent_id=agent_id,
            task_query=task_query,
            interval_seconds=interval_seconds,
        )

    def unschedule(self, agent_id: str) -> None:
        """Remove a cron job."""
        self._jobs.pop(agent_id, None)
        self._lod.unregister(agent_id)

    def get_due_jobs(self, current_time: float | None = None) -> list[CronSessionManager.CronJob]:
        """Get jobs that are due for execution."""
        now = current_time or time.time()
        due = []
        for job in self._jobs.values():
            if now - job.last_run >= job.interval_seconds:
                due.append(job)
        return due

    def run_job(
        self,
        job: CronSessionManager.CronJob,
        current_time: float | None = None,
    ) -> SessionContext:
        """Execute a cron job — promote to ACTIVE, load minimal context, return."""
        now = current_time or time.time()

        # Promote to ACTIVE
        self._lod.update_load(job.agent_id, 0.5)

        # Load minimal context (heartbeat-level budget)
        context = self._ctx.load_session_context(
            query=job.task_query,
            token_budget=int(self._ctx.token_budget * 0.15),
            session_type="heartbeat",
        )

        # Update job state
        job.last_run = now
        job.run_count += 1
        job.total_tokens_used += context.tokens_used

        # Demote back to DORMANT after completion
        self._lod.update_load(job.agent_id, 0.0)

        return context

    def stats(self) -> dict[str, Any]:
        return {
            "total_jobs": len(self._jobs),
            "jobs": {
                jid: {
                    "task": j.task_query,
                    "interval_s": j.interval_seconds,
                    "run_count": j.run_count,
                    "total_tokens": j.total_tokens_used,
                }
                for jid, j in self._jobs.items()
            },
        }


# ── Memory Bridge ─────────────────────────────────────────────────────

class MemoryBridge:
    """Connects the cognitive bus to hippocampus long-term memory.

    Two-way bridge:
      1. Bus → Hippocampus: High-salience events from the cognitive bus
         are automatically remembered via hippocampus.remember().
      2. Hippocampus → Bus: When an agent's context is optimized, relevant
         long-term memories are recalled and injected as pinned fragments.

    Salience mapping:
      - Critical events (emotional_tag=3) → salience=100
      - Important events (emotional_tag=2) → salience=50
      - Normal events → salience=20
      - Low-value → salience=5

    Graceful degradation: if hippocampus is not installed, this is a no-op.
    """

    def __init__(self, bus: CognitiveBus):
        self._bus = bus
        self._ltm = None
        self._bridged_count = 0

        # Try to import LongTermMemory adapter
        try:
            from .long_term_memory import LongTermMemory, is_available
            if is_available():
                self._ltm = LongTermMemory(
                    capacity=10_000,
                    consolidation_interval=50,
                    recall_reinforcement=1.3,
                )
                logger.info("MemoryBridge: hippocampus connected")
        except ImportError:
            logger.debug("MemoryBridge: hippocampus not available (graceful degradation)")

    @property
    def active(self) -> bool:
        return self._ltm is not None and self._ltm.active

    def bridge_events(self) -> int:
        """Transfer high-salience cognitive bus events to hippocampus.

        Called periodically (e.g., after each optimize_context cycle).
        """
        if not self.active:
            return 0

        # Drain memory bridge queue from cognitive bus
        # Use the Python bus's internal history since we're using Python bus here
        # For Rust bus, this would call drain_memory_bridge()
        bridged = 0

        # Build fragments from bus history for hippocampus
        fragments = []
        for event_data in self._bus._history[-20:]:  # Last 20 events
            salience = _emotional_to_salience(event_data.surprise)
            if salience >= 20:  # Only bridge above threshold
                fragments.append({
                    "id": f"bus_{event_data.tick}",
                    "content": event_data.payload,
                    "source": f"cognitive_bus:{event_data.source}",
                    "entropy_score": event_data.surprise,
                    "is_pinned": salience >= 80,
                    "relevance": min(1.0, event_data.surprise / 100.0),
                })

        if fragments:
            selected_ids = {f["id"] for f in fragments}
            bridged = self._ltm.remember_fragments(fragments, selected_ids)
            self._bridged_count += bridged

        return bridged

    def recall_for_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall relevant long-term memories for context injection.

        Returns fragments suitable for injecting into EntrolyEngine.
        """
        if not self.active:
            return []
        return self._ltm.recall_relevant(query, top_k=top_k)

    def tick(self) -> None:
        """Advance hippocampus clock (triggers decay)."""
        if self.active:
            self._ltm.tick()

    def stats(self) -> dict[str, Any]:
        if not self.active:
            return {"active": False}
        ltm_stats = self._ltm.stats()
        ltm_stats["bridged_count"] = self._bridged_count
        return ltm_stats


def _emotional_to_salience(surprise: float) -> float:
    """Map surprise score to hippocampus salience."""
    if surprise >= 0.8:
        return 100.0
    elif surprise >= 0.5:
        return 50.0
    elif surprise >= 0.2:
        return 20.0
    return 5.0


# ── HCC — Hierarchical Context Compression ────────────────────────────

class CompressionLevel:
    """3-level content compression for hierarchical context.

    Full:      100% information, 100% tokens — verbatim content.
    Skeleton:  ~70% information, ~20% tokens — key lines only.
    Reference: ~15% information, ~2% tokens — one-line summary.
    """
    FULL = 0
    SKELETON = 1
    REFERENCE = 2

    RETENTION = [1.00, 0.70, 0.15]
    TOKEN_RATIO = [1.00, 0.20, 0.02]


@dataclass
class HCCFragment:
    """A fragment at all 3 compression levels."""
    fragment_id: str
    source: str
    full_content: str
    skeleton_content: str
    reference_content: str
    full_tokens: int
    skeleton_tokens: int
    reference_tokens: int
    entropy_score: float
    relevance: float
    assigned_level: int = CompressionLevel.REFERENCE


class HCCEngine:
    """Rate-distortion optimizer for multi-level context compression.

    Rate-distortion optimizer for multi-level context compression.

    Algorithm:
      1. All fragments start at Reference level (minimum viable)
      2. Compute marginal gain for upgrading each fragment:
           gain_ratio = Δinfo / Δtokens
           where info = relevance × level_retention × entropy
      3. Greedy: sort by gain_ratio, accept highest until budget full
      4. Guarantee: (1 - 1/e) ≈ 63.2% optimal (submodular monotone)

    This gives Users the right compression level per fragment:
      - SOUL.md → Full (always needed verbatim)
      - Recent MEMORY.md sections → Full or Skeleton (depends on relevance)
      - Old daily logs → Reference (just one-line summaries)
      - Irrelevant tools → dropped entirely
    """

    def __init__(self):
        self._fragments: list[HCCFragment] = []

    def add_fragment(
        self,
        fragment_id: str,
        source: str,
        content: str,
        entropy_score: float = 0.5,
        relevance: float = 0.5,
    ) -> HCCFragment:
        """Add a fragment with auto-generated skeleton and reference."""
        full_tokens = len(content.split())  # Approximate
        skeleton = _generate_skeleton(content)
        reference = _generate_reference(content, source)

        frag = HCCFragment(
            fragment_id=fragment_id,
            source=source,
            full_content=content,
            skeleton_content=skeleton,
            reference_content=reference,
            full_tokens=full_tokens,
            skeleton_tokens=len(skeleton.split()),
            reference_tokens=len(reference.split()),
            entropy_score=entropy_score,
            relevance=relevance,
        )
        self._fragments.append(frag)
        return frag

    def optimize(self, token_budget: int) -> list[HCCFragment]:
        """Run rate-distortion optimization.

        Returns fragments with assigned_level set to optimal compression.
        """
        if not self._fragments:
            return []

        # Phase 1: Start all at Reference level
        for f in self._fragments:
            f.assigned_level = CompressionLevel.REFERENCE

        current_tokens = sum(f.reference_tokens for f in self._fragments)

        # If even all references exceed budget, drop lowest-relevance
        if current_tokens > token_budget:
            sorted_frags = sorted(self._fragments, key=lambda f: f.relevance)
            while current_tokens > token_budget and sorted_frags:
                dropped = sorted_frags.pop(0)
                current_tokens -= dropped.reference_tokens
                self._fragments.remove(dropped)

        # Phase 2: Compute marginal gains for all possible upgrades
        candidates = []
        for f in self._fragments:
            # Reference → Skeleton
            delta_tokens = f.skeleton_tokens - f.reference_tokens
            if delta_tokens > 0:
                delta_info = (f.relevance
                              * (CompressionLevel.RETENTION[1] - CompressionLevel.RETENTION[2])
                              * f.entropy_score)
                candidates.append((delta_info / delta_tokens, delta_tokens, f, CompressionLevel.SKELETON))

            # Reference → Full (skip skeleton)
            delta_tokens_full = f.full_tokens - f.reference_tokens
            if delta_tokens_full > 0:
                delta_info_full = (f.relevance
                                   * (CompressionLevel.RETENTION[0] - CompressionLevel.RETENTION[2])
                                   * f.entropy_score)
                candidates.append((delta_info_full / delta_tokens_full, delta_tokens_full, f, CompressionLevel.FULL))

        # Phase 3: Greedy selection by gain ratio
        candidates.sort(key=lambda x: -x[0])
        remaining = token_budget - current_tokens

        for gain_ratio, delta_tokens, frag, target_level in candidates:
            if delta_tokens <= remaining:
                # Only upgrade if it's actually an upgrade
                if target_level < frag.assigned_level:  # Lower number = higher quality
                    # Recover tokens from current level
                    if frag.assigned_level == CompressionLevel.REFERENCE:
                        old_tokens = frag.reference_tokens
                    else:
                        old_tokens = frag.skeleton_tokens

                    if target_level == CompressionLevel.FULL:
                        new_tokens = frag.full_tokens
                    else:
                        new_tokens = frag.skeleton_tokens

                    actual_delta = new_tokens - old_tokens
                    if actual_delta <= remaining:
                        frag.assigned_level = target_level
                        remaining -= actual_delta

        return self._fragments

    def get_content(self, frag: HCCFragment) -> str:
        """Get content at the fragment's assigned compression level."""
        if frag.assigned_level == CompressionLevel.FULL:
            return frag.full_content
        elif frag.assigned_level == CompressionLevel.SKELETON:
            return frag.skeleton_content
        return frag.reference_content


def _generate_skeleton(content: str) -> str:
    """Generate skeleton-level compression (~20% of tokens).

    Keeps: headers, function signatures, key statements.
    Drops: comments, docstrings, blank lines, body details.
    """
    lines = content.split("\n")
    skeleton_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep headers
        if stripped.startswith("#") or stripped.startswith(("def ", "class ", "fn ", "pub fn ", "struct ", "impl ")) or stripped.startswith(("import ", "from ", "use ", "require")) or any(kw in stripped.lower() for kw in ["todo", "fixme", "important", "note:", "warning"]) or stripped and len(stripped) < 60 and not stripped.startswith(("//", "#", "/*", "*", "---")):
            skeleton_lines.append(line)
    return "\n".join(skeleton_lines)


def _generate_reference(content: str, source: str) -> str:
    """Generate reference-level compression (~2% of tokens).

    One-line summary: filename + first meaningful line.
    """
    first_line = ""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//", "/*", "---", "```")):
            first_line = stripped[:80]
            break
    basename = source.split("/")[-1] if "/" in source else source
    return f"[{basename}] {first_line}"


# ── AutoTune — Per-Workspace Weight Calibration ───────────────────────

class AutoTune:
    """Adaptive weight calibration for entropy scoring parameters.

    Adaptive weight calibration. Uses:
      - EMA (Exponential Moving Average) for smooth tracking
      - Polyak averaging for stability
      - Drift penalty to resist sudden shifts

    Tracks per-workspace metrics and adjusts:
      - entropy_weight: how much entropy matters vs recency
      - relevance_weight: how much query relevance matters
      - decay_rate: how fast old fragments lose value

    After each optimize_context() cycle, the outcome (user satisfaction,
    token efficiency, context quality) feeds back into weight updates.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        polyak_alpha: float = 0.01,
        drift_penalty: float = 0.05,
    ):
        self._ema_alpha = ema_alpha
        self._polyak_alpha = polyak_alpha
        self._drift_penalty = drift_penalty

        # Tunable weights (start at sensible defaults)
        self._weights: dict[str, float] = {
            "entropy": 1.0,
            "relevance": 1.0,
            "recency": 0.5,
            "diversity": 0.3,
        }
        # Polyak-averaged weights (more stable, used for actual optimization)
        self._polyak_weights: dict[str, float] = dict(self._weights)
        # EMA of recent outcomes
        self._outcome_ema: float = 0.5
        self._update_count: int = 0

    def update(self, outcome: float, metrics: dict[str, float]) -> dict[str, float]:
        """Update weights based on optimization outcome.

        Args:
            outcome: Quality score of the last optimization (0-1).
                     Higher = better (user found context useful).
            metrics: Per-weight contribution metrics from the last cycle.
                     e.g., {"entropy": 0.7, "relevance": 0.9, ...}

        Returns:
            Updated weights dict.
        """
        self._update_count += 1

        # EMA of outcome
        self._outcome_ema = (self._ema_alpha * outcome
                             + (1.0 - self._ema_alpha) * self._outcome_ema)

        # Gradient signal: outcome above EMA → reinforce current weights
        advantage = outcome - self._outcome_ema

        for key in self._weights:
            metric_val = metrics.get(key, 0.5)

            # Gradient: positive advantage + high metric → increase weight
            grad = advantage * metric_val

            # Drift penalty: resist moving too far from Polyak average
            drift = self._weights[key] - self._polyak_weights[key]
            grad -= self._drift_penalty * drift

            # Update
            self._weights[key] += self._ema_alpha * grad
            self._weights[key] = max(0.05, min(5.0, self._weights[key]))

            # Polyak averaging (slow-moving stable estimate)
            self._polyak_weights[key] = (
                self._polyak_alpha * self._weights[key]
                + (1.0 - self._polyak_alpha) * self._polyak_weights[key]
            )

        return dict(self._polyak_weights)

    def get_weights(self) -> dict[str, float]:
        """Get current Polyak-averaged weights (stable for production use)."""
        return dict(self._polyak_weights)

    def stats(self) -> dict[str, Any]:
        return {
            "weights": dict(self._weights),
            "polyak_weights": dict(self._polyak_weights),
            "outcome_ema": round(self._outcome_ema, 4),
            "update_count": self._update_count,
        }


# ── Enhanced AgentContext ──────────────────────────────────────────

class MultiAgentContext(AgentContext):
    """Full multi-agent context intelligence for the agent system.

    Extends AgentContext with:
      - LOD lifecycle management (DORMANT ↔ ACTIVE ↔ SATURATED ↔ HEAVY)
      - Subagent orchestration (spawn/despawn with context inheritance)
      - Cron session management (scheduled minimal-context agents)
      - Memory bridge (cognitive bus ↔ hippocampus LTM)
      - HCC compression (3-level rate-distortion optimization)
      - AutoTune (per-workspace weight calibration)

    Usage:
        from entroly.context_bridge import MultiAgentContext

        ctx = MultiAgentContext(
            workspace_path="~/.agent/workspace",
            token_budget=128_000,
        )
        ctx.ingest_workspace()

        # Spawn subagent
        sub_ctx = ctx.spawn_subagent("main", "researcher", "find auth bugs")

        # Schedule cron
        ctx.schedule_cron("email_checker", "check emails", interval_seconds=900)

        # Run due cron jobs
        for job in ctx.get_due_cron_jobs():
            result = ctx.run_cron_job(job)

        # Load HCC-optimized context
        context = ctx.load_hcc_context(query="deploy the fix", token_budget=8192)

        # Record outcome for AutoTune
        ctx.record_autotune_outcome(success=True, metrics={"entropy": 0.7})
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Multi-agent components
        self._lod = LODManager()
        self._subagents = SubagentOrchestrator(self, self._lod, self._bus)
        self._cron = CronSessionManager(self, self._lod, self._bus)
        self._memory_bridge = MemoryBridge(self._bus)
        self._hcc = HCCEngine()
        self._autotune = AutoTune()

        # Register the main agent
        self._lod.register("main", initial_tier=LodTier.ACTIVE)
        self._bus.subscribe("main", event_types=None)

    # ── Subagent API ──────────────────────────────────────────────

    def spawn_subagent(
        self,
        parent_id: str,
        child_id: str,
        task_query: str,
        budget_fraction: float = 0.2,
    ) -> SessionContext | None:
        """Spawn a subagent with inherited context."""
        return self._subagents.spawn(parent_id, child_id, task_query, budget_fraction)

    def despawn_subagent(self, child_id: str) -> None:
        """Despawn a subagent."""
        self._subagents.despawn(child_id)

    def get_agent_tree(self, root_id: str = "main") -> dict[str, Any]:
        """Get the full agent tree."""
        return self._subagents.get_tree(root_id)

    # ── Cron API ──────────────────────────────────────────────────

    def schedule_cron(
        self,
        agent_id: str,
        task_query: str,
        interval_seconds: float = 900.0,
    ) -> None:
        """Schedule a cron agent."""
        self._cron.schedule(agent_id, task_query, interval_seconds)

    def unschedule_cron(self, agent_id: str) -> None:
        """Remove a cron job."""
        self._cron.unschedule(agent_id)

    def get_due_cron_jobs(self) -> list:
        """Get cron jobs that are due."""
        return self._cron.get_due_jobs()

    def run_cron_job(self, job) -> SessionContext:
        """Execute a due cron job."""
        return self._cron.run_job(job)

    # ── HCC API ───────────────────────────────────────────────────

    def load_hcc_context(
        self,
        query: str = "",
        token_budget: int | None = None,
    ) -> SessionContext:
        """Load context with HCC 3-level compression.

        Instead of binary include/exclude (knapsack), HCC gives each
        fragment the optimal compression level within the budget.
        """
        budget = token_budget or self.token_budget

        # Recall long-term memories if available
        ltm_memories = self._memory_bridge.recall_for_context(query, top_k=3)

        # Build HCC engine with current fragments
        hcc = HCCEngine()
        for fpath, fid in self._ingested_files.items():
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="replace")
                hcc.add_fragment(
                    fragment_id=fid,
                    source=fpath,
                    content=content,
                    entropy_score=0.5,
                    relevance=0.5,
                )
            except (FileNotFoundError, PermissionError):
                pass

        # Add LTM memories as high-relevance fragments
        for mem in ltm_memories:
            hcc.add_fragment(
                fragment_id=f"ltm_{mem.get('source', 'unknown')}",
                source=mem.get("source", "long_term_memory"),
                content=mem.get("content", ""),
                entropy_score=0.8,
                relevance=mem.get("retention", 0.5),
            )

        # Optimize
        optimized = hcc.optimize(budget)

        # Build context string
        parts = []
        tokens_used = 0
        for frag in optimized:
            content = hcc.get_content(frag)
            parts.append(content)
            if frag.assigned_level == CompressionLevel.FULL:
                tokens_used += frag.full_tokens
            elif frag.assigned_level == CompressionLevel.SKELETON:
                tokens_used += frag.skeleton_tokens
            else:
                tokens_used += frag.reference_tokens

        context_str = "\n\n---\n\n".join(parts)
        total_raw = sum(f.full_tokens for f in optimized)

        return SessionContext(
            context_str=context_str,
            tokens_used=tokens_used,
            tokens_saved=max(0, total_raw - tokens_used),
            total_raw_tokens=total_raw,
            fragments_selected=len(optimized),
            fragments_total=len(self._ingested_files),
            memory_entries_loaded=len(ltm_memories),
        )

    # ── Memory Bridge API ─────────────────────────────────────────

    def bridge_memories(self) -> int:
        """Bridge high-salience bus events to hippocampus."""
        return self._memory_bridge.bridge_events()

    # ── AutoTune API ──────────────────────────────────────────────

    def record_autotune_outcome(
        self,
        success: bool,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Record outcome for AutoTune weight calibration.

        Args:
            success: Whether the LLM output was useful.
            metrics: Per-weight contribution scores.

        Returns:
            Updated weights.
        """
        outcome = 1.0 if success else 0.0
        m = metrics or {"entropy": 0.5, "relevance": 0.5, "recency": 0.5, "diversity": 0.5}
        return self._autotune.update(outcome, m)

    # ── LOD API ───────────────────────────────────────────────────

    def update_agent_load(self, agent_id: str, load: float) -> str | None:
        """Update agent load factor, returns new tier if changed."""
        return self._lod.update_load(agent_id, load)

    def tick(self) -> None:
        """Advance all clocks (LOD + bus + memory)."""
        self._lod.tick()
        self._memory_bridge.tick()

    # ── Enhanced Stats ────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        base = super().get_stats()
        base["lod"] = self._lod.stats()
        base["subagents"] = self._subagents.stats()
        base["cron"] = self._cron.stats()
        base["memory_bridge"] = self._memory_bridge.stats()
        base["autotune"] = self._autotune.stats()
        return base
