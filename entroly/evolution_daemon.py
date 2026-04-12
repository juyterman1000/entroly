"""
Evolution Daemon — Zero-Token Autonomous Self-Improvement
=========================================================

The orchestrator for the 3 Pillars of Zero-Token Autonomy:

  Pillar 1 — Token Economy (Self-Funded Evolution):
    Before any LLM-based synthesis, checks ValueTracker.get_evolution_budget().
    The system can only spend τ% (5%) of its lifetime savings on evolution.
    Invariant: C_spent(t) ≤ τ · S(t)  →  strictly token-negative.

  Pillar 2 — Local Structural Induction:
    ALWAYS tried first. Uses StructuralSynthesizer to generate tools
    from the entropy gradient of the code graph. Zero tokens, deterministic.
    Only falls back to LLM synthesis if structural synthesis fails AND
    the budget allows it.

  Pillar 3 — Dreaming Loop (Self-Play):
    During idle cycles (>60s), runs counterfactual weight optimization
    via DreamingLoop.run_dream_cycle(). Generates synthetic queries from
    FeedbackJournal history, tests weight perturbations, keeps improvements.

Architecture:
  run_once() → EvolutionLogger.get_pending_gaps()
             → for each gap:
                  1. Try StructuralSynthesizer ($0)
                  2. If fail & budget allows: LLM synthesis (budget-gated)
                  3. Benchmark synthesized skill
                  4. Promote or prune
             → if idle: DreamingLoop.run_dream_cycle()

  start_daemon() → background thread calling run_once() every interval

Usage (integrated into server.py):
    from entroly.evolution_daemon import EvolutionDaemon
    daemon = EvolutionDaemon(vault, evolution_logger, value_tracker, ...)
    daemon.start()   # non-blocking background thread
    daemon.stop()    # graceful shutdown
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("entroly.evolution_daemon")


class EvolutionDaemon:
    """Background daemon for autonomous self-improvement.

    Orchestrates the 3 pillars:
      1. ROI-gated evolution (ValueTracker budget guardrail)
      2. Structural synthesis first ($0), LLM fallback (budget-gated)
      3. Dreaming loop during idle cycles
    """

    # How often the daemon checks for pending gaps (seconds)
    POLL_INTERVAL_S = 30.0
    # Minimum time between evolution attempts on the same entity
    COOLDOWN_S = 300.0

    def __init__(
        self,
        vault: Any,
        evolution_logger: Any,
        value_tracker: Any,
        feedback_journal: Any = None,
        rust_engine: Any = None,
    ):
        """
        Args:
            vault: VaultManager instance
            evolution_logger: EvolutionLogger instance
            value_tracker: ValueTracker instance (has get_evolution_budget)
            feedback_journal: FeedbackJournal instance (for dreaming)
            rust_engine: Optional Rust EntrolyEngine (for structural synthesis)
        """
        from .skill_engine import SkillEngine, StructuralSynthesizer

        self._vault = vault
        self._evo_logger = evolution_logger
        self._value_tracker = value_tracker
        self._skill_engine = SkillEngine(vault)
        self._structural = StructuralSynthesizer(rust_engine)
        self._dreaming = None

        if feedback_journal:
            from .autotune import DreamingLoop
            self._dreaming = DreamingLoop(feedback_journal)

        # State
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cooldowns: dict[str, float] = {}  # entity_key -> last_attempt_time
        self._stats = {
            "structural_successes": 0,
            "structural_failures": 0,
            "llm_successes": 0,
            "llm_rejections": 0,
            "skills_promoted": 0,
            "skills_pruned": 0,
            "dream_cycles": 0,
        }

    def start(self) -> None:
        """Start the daemon in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="entroly-evolution-daemon",
            daemon=True,
        )
        self._thread.start()
        logger.info("EvolutionDaemon: started")

    def stop(self) -> None:
        """Gracefully stop the daemon."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("EvolutionDaemon: stopped")

    def record_activity(self) -> None:
        """Reset the dreaming idle timer (called on user queries)."""
        if self._dreaming:
            self._dreaming.record_activity()

    def _loop(self) -> None:
        """Main daemon loop: poll for gaps + dream during idle."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.debug("EvolutionDaemon: cycle error: %s", e)

            self._stop_event.wait(timeout=self.POLL_INTERVAL_S)

    def run_once(self) -> dict[str, Any]:
        """Execute one daemon cycle.

        1. Process pending skill gaps (structural first, LLM fallback)
        2. Dream if idle
        """
        results: dict[str, Any] = {"gaps_processed": 0, "dream": None}

        # ── Phase 1: Process pending skill gaps ──────────────────────
        gaps = self._evo_logger.get_pending_gaps()
        now = time.time()

        for gap in gaps:
            entity_key = gap["entity_key"]

            # Cooldown check
            last_attempt = self._cooldowns.get(entity_key, 0)
            if now - last_attempt < self.COOLDOWN_S:
                continue

            self._cooldowns[entity_key] = now
            result = self._process_gap(gap)
            results["gaps_processed"] += 1

            if result.get("status") == "promoted":
                self._stats["skills_promoted"] += 1
            elif result.get("status") == "pruned":
                self._stats["skills_pruned"] += 1

        # ── Phase 2: Dreaming loop (idle-time self-play) ───────────
        if self._dreaming and self._dreaming.should_dream():
            dream_result = self._dreaming.run_dream_cycle()
            results["dream"] = dream_result
            if dream_result.get("status") == "completed":
                self._stats["dream_cycles"] += 1

        return results

    def _process_gap(self, gap: dict[str, Any]) -> dict[str, Any]:
        """Process a single skill gap.

        Priority order:
          1. Structural synthesis ($0) — ALWAYS try first
          2. LLM synthesis (budget-gated) — only if structural fails
        """
        entity_key = gap["entity_key"]
        source_files = gap.get("source_files", [])
        queries = gap.get("queries", [])

        # ── Attempt 1: Structural synthesis (Pillar 2, $0) ──────
        spec = self._structural.synthesize_structural(
            entity_key, source_files, queries
        )

        if spec:
            self._stats["structural_successes"] += 1
            logger.info(
                "EvolutionDaemon: structural synthesis succeeded for '%s'",
                entity_key,
            )
            return self._deploy_skill(spec)

        self._stats["structural_failures"] += 1

        # ── Attempt 2: LLM synthesis (Pillar 1, budget-gated) ───
        budget = self._value_tracker.get_evolution_budget()
        if not budget["can_evolve"]:
            self._stats["llm_rejections"] += 1
            logger.info(
                "EvolutionDaemon: LLM synthesis skipped for '%s' "
                "(budget exhausted: $%.4f available)",
                entity_key, budget["available_usd"],
            )
            return {
                "status": "budget_exhausted",
                "entity_key": entity_key,
                "available_usd": budget["available_usd"],
            }

        # For now, use the existing SkillSynthesizer (non-LLM template).
        # In future, this is where we'd call an LLM API to generate the
        # tool code, debiting from the evolution budget.
        # The key point: even this path is GATED by the budget.
        result = self._skill_engine.create_skill(
            entity_key, queries, gap.get("intents", [""])[0] if gap.get("intents") else ""
        )

        if result.get("status") == "created":
            self._stats["llm_successes"] += 1
            # Record the spend (template synthesis is effectively $0,
            # but we track it for completeness)
            self._value_tracker.record_evolution_spend(0.0, success=True)
            return self._benchmark_and_promote(result["skill_id"])

        return {"status": "failed", "entity_key": entity_key}

    def _deploy_skill(self, spec: Any) -> dict[str, Any]:
        """Deploy a structurally synthesized skill: write → benchmark → promote."""
        from .skill_engine import SkillSpec
        result = self._skill_engine.create_skill(
            spec.entity,
            [tc["input"] for tc in spec.test_cases],
            "",
        )

        # Overwrite the tool.py with our structural tool code
        if result.get("status") == "created":
            from pathlib import Path
            skill_dir = Path(result["path"])
            tool_file = skill_dir / "tool.py"
            tool_file.write_text(spec.tool_code, encoding="utf-8")

            return self._benchmark_and_promote(result["skill_id"])

        return {"status": "deploy_failed", "entity": spec.entity}

    def _benchmark_and_promote(self, skill_id: str) -> dict[str, Any]:
        """Benchmark a skill, then promote or prune based on fitness."""
        bench = self._skill_engine.benchmark_skill(skill_id)
        promote = self._skill_engine.promote_or_prune(skill_id)
        return {
            "status": promote.get("status", "unknown"),
            "skill_id": skill_id,
            "fitness": bench.get("fitness", 0.0),
        }

    def stats(self) -> dict[str, Any]:
        """Return daemon statistics."""
        result = dict(self._stats)
        result["budget"] = self._value_tracker.get_evolution_budget()
        if self._dreaming:
            result["dreaming"] = self._dreaming.stats()
        result["running"] = bool(self._thread and self._thread.is_alive())
        return result

