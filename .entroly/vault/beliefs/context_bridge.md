---
claim_id: ec191468-974c-4cfc-b37c-166efe85f7bd
entity: context_bridge
status: inferred
confidence: 0.75
sources:
  - entroly\context_bridge.py:50
  - entroly\context_bridge.py:65
  - entroly\context_bridge.py:74
  - entroly\context_bridge.py:84
  - entroly\context_bridge.py:255
  - entroly\context_bridge.py:265
  - entroly\context_bridge.py:407
  - entroly\context_bridge.py:416
  - entroly\context_bridge.py:424
  - entroly\context_bridge.py:468
last_checked: 2026-04-23T03:07:07.801413+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: context_bridge

**Language:** python
**Lines of code:** 2018

## Types
- `class SessionContext()` — Result of optimized context loading for an OpenClaw session.
- `class AgentBudget()` — Per-agent token budget from NKBE allocation.
- `class HeartbeatResult()` — Result of an optimized heartbeat check.
- `class NkbeAllocator()` — Nash-KKT Budgetary Equilibrium allocator for multi-agent token budgets. Given N agents sharing a global token budget B, finds the optimal split that maximizes weighted utility: maximize  Σᵢ wᵢ · Uᵢ(Bᵢ
- `class _AgentState()`
- `class CognitiveBus()` — Information-Surprise-Adaptive (ISA) event routing between agents. Routes events between OpenClaw agents with information-theoretic prioritization. Implements Poisson rate model per subscriber per even
- `class _BusEvent()`
- `class _Subscriber()`
- `class _RateCell()` — Poisson rate model with Welford online variance for spike detection.
- `class AgentContext()` — Entroly context intelligence for AI agents. Wraps EntrolyEngine to optimize the agent's workspace file loading. Instead of naively reading all files into the LLM prompt, this class ingests workspace f
- `class LodTier()` — Level-of-Detail tiers for agent lifecycle management. Adapted from both ebbiforge (4-tier swarm) and agentOS (3-tier routing). Mapped to agent states: DORMANT:    Agent registered but idle. Minimal me
- `class AgentState()` — Per-agent lifecycle state for LOD management.
- `class LODManager()` — Level-of-Detail lifecycle manager for agent armies. Prevents resource exhaustion when 100s of agents run simultaneously. Adapted from: - ebbiforge ProductionTensorSwarm: bitflag triggers, surprise-bas
- `class SubagentOrchestrator()` — Manages subagent spawning with context inheritance and NKBE budget splitting. The system supports up to 5 subagents per parent, 3 depth levels. Each subagent inherits a subset of its parent's context 
- `class CronSessionManager()` — Manages scheduled (cron) agent sessions with minimal context loading. Cron agents run periodically (e.g., check email every 15 min, review calendar daily). They need: 1. Minimal context — just the rel
- `class MemoryBridge()` — Connects the cognitive bus to hippocampus long-term memory. Two-way bridge: 1. Bus → Hippocampus: High-salience events from the cognitive bus are automatically remembered via hippocampus.remember(). 2
- `class CompressionLevel()` — 3-level content compression from agentOS HCC. Full:      100% information, 100% tokens — verbatim content. Skeleton:  ~70% information, ~20% tokens — key lines only. Reference: ~15% information, ~2% t
- `class HCCFragment()` — A fragment at all 3 compression levels.
- `class HCCEngine()` — Rate-distortion optimizer for multi-level context compression. Ported from agentOS context.rs HccEngine. Algorithm: 1. All fragments start at Reference level (minimum viable) 2. Compute marginal gain 
- `class AutoTune()` — Adaptive weight calibration for entropy scoring parameters. Ported from agentOS autotune.rs. Uses: - EMA (Exponential Moving Average) for smooth tracking - Polyak averaging for stability - Drift penal
- `class MultiAgentContext(AgentContext)` — Full multi-agent context intelligence for the agent system. Extends AgentContext with: - LOD lifecycle management (DORMANT ↔ ACTIVE ↔ SATURATED ↔ HEAVY) - Subagent orchestration (spawn/despawn with co

## Functions
- `def __init__(
        self,
        global_budget: int = 128_000,
        min_agent_budget: int = 1024,
        tau: float = 0.1,
        learning_rate: float = 0.01,
        nash_iterations: int = 5,
    )`
- `def register_agent(
        self,
        name: str,
        weight: float = 1.0,
        min_budget: int | None = None,
    ) -> None`
- `def update_fragments(
        self,
        agent_name: str,
        fragment_count: int,
        total_tokens: int,
    ) -> None`
- `def allocate(self) -> dict[str, int]` — Run NKBE allocation. Returns per-agent budgets. Two-phase KKT bisection: 1. Bisect for global λ* such that Σ Bᵢ(λ*) = B 2. Each agent gets Bᵢ = max(Bᵢ_min, demand(λ*))
- `def reinforce(self, outcomes: dict[str, float]) -> None` — REINFORCE weight update based on agent outcomes. Δwᵢ = η · (Rᵢ − R̄) · wᵢ
- `def __init__(self, novelty_threshold: float = 0.3, alpha: float = 0.1)`
- `def subscribe(self, agent_name: str, event_types: list[str] | None = None) -> None` — Register an agent as event subscriber.
- `def publish(
        self,
        source: str,
        event_type: str,
        payload: str,
        surprise: float = 0.0,
    ) -> int`
- `def drain(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]` — Drain events for an agent, highest priority first.
- `def stats(self) -> dict[str, Any]`
- `def __init__(self, alpha: float = 0.1)`
- `def observe(self, tick: int) -> None` — Record an event occurrence.
- `def kl_divergence(self) -> float` — KL(P(λ_obs) ‖ P(λ_exp)) for Poisson distributions.
- `def is_spike(self) -> bool` — Welford spike detection: event rate > mean + 3σ.
- `def __init__(
        self,
        workspace_path: str | Path = "~/.agent/workspace",
        token_budget: int = 32_000,
        quality: str = "balanced",
        config: EntrolyConfig | None = None,
    )`
- `def ingest_workspace(self) -> dict[str, Any]` — Ingest all OpenClaw workspace files as Entroly fragments. Reads SOUL.md, USER.md, MEMORY.md, IDENTITY.md, TOOLS.md, HEARTBEAT.md, and daily logs. Each file (or section within a file) becomes a fragmen
- `def load_session_context(
        self,
        query: str = "",
        token_budget: int | None = None,
        session_type: str = "main",
    ) -> SessionContext`
- `def allocate_budgets(
        self,
        agents: list[str],
        weights: dict[str, float] | None = None,
    ) -> dict[str, int]`
- `def optimize_heartbeat(
        self,
        check_types: list[str],
        total_budget: int | None = None,
    ) -> HeartbeatResult`
- `def filter_group_chat(
        self,
        messages: list[dict[str, str]],
        agent_expertise: str = "",
        max_messages: int = 20,
    ) -> list[dict[str, str]]`
- `def record_outcome(self, success: bool) -> None` — Record whether the LLM output was helpful (Wilson score update).
- `def publish_event(
        self,
        source: str,
        event_type: str,
        payload: str,
        surprise: float = 0.0,
    ) -> int`
- `def drain_events(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]` — Drain events for an agent from the cognitive bus.
- `def get_stats(self) -> dict[str, Any]` — Get combined statistics.
- `def __init__(self)`
- `def register(
        self,
        agent_id: str,
        parent_id: str | None = None,
        initial_tier: str = LodTier.DORMANT,
    ) -> AgentState`
- `def unregister(self, agent_id: str) -> None` — Remove an agent (and orphan its children to parent or root).
- `def update_load(self, agent_id: str, load_factor: float) -> str | None` — Update agent load and trigger tier transitions. Returns new tier if transition occurred, else None.
- `def tick(self) -> list[str]` — Advance clock. Returns any alerts generated.
- `def get_active_agents(self) -> list[str]` — Get IDs of all ACTIVE or HEAVY agents (eligible for tasks).
- `def get_budget_weights(self) -> dict[str, float]` — Get NKBE weight hints based on LOD tier. HEAVY agents get 2x weight, ACTIVE get 1x, DORMANT/SATURATED get 0 (no budget allocation).
- `def stats(self) -> dict[str, Any]`
- `def __init__(
        self,
        agent_ctx: AgentContext,
        lod: LODManager,
        bus: CognitiveBus,
    )`
- `def can_spawn(self, parent_id: str) -> tuple[bool, str]` — Check if a parent agent can spawn a new subagent.
- `def spawn(
        self,
        parent_id: str,
        child_id: str,
        task_query: str,
        budget_fraction: float = 0.2,
    ) -> SessionContext | None`
- `def despawn(self, child_id: str) -> None` — Despawn a subagent and release its resources.
- `def get_tree(self, root_id: str) -> dict[str, Any]` — Get the subagent tree rooted at an agent.
- `def stats(self) -> dict[str, Any]`
- `def __init__(
        self,
        agent_ctx: AgentContext,
        lod: LODManager,
        bus: CognitiveBus,
    )`
- `def schedule(
        self,
        agent_id: str,
        task_query: str,
        interval_seconds: float = 900.0,  # 15 min default
    ) -> None`
- `def unschedule(self, agent_id: str) -> None` — Remove a cron job.
- `def get_due_jobs(self, current_time: float | None = None) -> list[CronSessionManager.CronJob]` — Get jobs that are due for execution.
- `def run_job(
        self,
        job: CronSessionManager.CronJob,
        current_time: float | None = None,
    ) -> SessionContext`
- `def stats(self) -> dict[str, Any]`
- `def __init__(self, bus: CognitiveBus)`
- `def active(self) -> bool`
- `def bridge_events(self) -> int` — Transfer high-salience cognitive bus events to hippocampus. Called periodically (e.g., after each optimize_context cycle).
- `def recall_for_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]`
- `def tick(self) -> None` — Advance hippocampus clock (triggers decay).
- `def stats(self) -> dict[str, Any]`
- `def __init__(self)`
- `def add_fragment(
        self,
        fragment_id: str,
        source: str,
        content: str,
        entropy_score: float = 0.5,
        relevance: float = 0.5,
    ) -> HCCFragment`
- `def optimize(self, token_budget: int) -> list[HCCFragment]` — Run rate-distortion optimization. Returns fragments with assigned_level set to optimal compression.
- `def get_content(self, frag: HCCFragment) -> str` — Get content at the fragment's assigned compression level.
- `def __init__(
        self,
        ema_alpha: float = 0.1,
        polyak_alpha: float = 0.01,
        drift_penalty: float = 0.05,
    )`
- `def update(self, outcome: float, metrics: dict[str, float]) -> dict[str, float]` — Update weights based on optimization outcome. Args: outcome: Quality score of the last optimization (0-1). Higher = better (user found context useful). metrics: Per-weight contribution metrics from th
- `def get_weights(self) -> dict[str, float]` — Get current Polyak-averaged weights (stable for production use).
- `def stats(self) -> dict[str, Any]`
- `def __init__(self, **kwargs)`
- `def spawn_subagent(
        self,
        parent_id: str,
        child_id: str,
        task_query: str,
        budget_fraction: float = 0.2,
    ) -> SessionContext | None`
- `def despawn_subagent(self, child_id: str) -> None` — Despawn a subagent.
- `def get_agent_tree(self, root_id: str = "main") -> dict[str, Any]` — Get the full agent tree.
- `def schedule_cron(
        self,
        agent_id: str,
        task_query: str,
        interval_seconds: float = 900.0,
    ) -> None`
- `def unschedule_cron(self, agent_id: str) -> None` — Remove a cron job.
- `def get_due_cron_jobs(self) -> list` — Get cron jobs that are due.
- `def run_cron_job(self, job) -> SessionContext` — Execute a due cron job.
- `def load_hcc_context(
        self,
        query: str = "",
        token_budget: int | None = None,
    ) -> SessionContext`
- `def bridge_memories(self) -> int` — Bridge high-salience bus events to hippocampus.
- `def record_autotune_outcome(
        self,
        success: bool,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, float]`
- `def update_agent_load(self, agent_id: str, load: float) -> str | None` — Update agent load factor, returns new tier if changed.
- `def tick(self) -> None` — Advance all clocks (LOD + bus + memory).
- `def get_stats(self) -> dict[str, Any]`

## Dependencies
- `.config`
- `.provenance`
- `.server`
- `AgentContext`
- `__future__`
- `dataclasses`
- `hashlib`
- `logging`
- `pathlib`
- `time`
- `typing`

## Linked Beliefs
- [[config]]
- [[AgentContext]]
