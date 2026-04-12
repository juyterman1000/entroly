"""
Flow Orchestrator
==================

Chains the 5 canonical flows end-to-end. Each flow is a deterministic
sequence of steps that the orchestrator executes.

  â‘  Fast Answer:         Belief â†’ Action
  â‘¡ Verify Before Answer: Belief â†’ Verification â†’ Action
  â‘¢ Compile On Demand:   Truth â†’ Belief â†’ Verification â†’ Action
  â‘£ Change-Driven:       Event â†’ Truth â†’ Belief â†’ Verification â†’ Action
  â‘¤ Self-Improvement:    Misses â†’ Verification â†’ Evolution â†’ Belief
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .belief_compiler import BeliefCompiler
from .change_pipeline import ChangePipeline, parse_diff
from .epistemic_router import EpistemicFlow, EpistemicRouter, RoutingDecision
from .evolution_logger import EvolutionLogger
from .vault import VaultManager
from .verification_engine import VerificationEngine

logger = logging.getLogger(__name__)


@dataclass
class FlowResult:
    """Result of executing a canonical flow."""
    flow: str
    status: str  # completed, partial, failed
    steps_completed: list[str] = field(default_factory=list)
    beliefs_used: list[str] = field(default_factory=list)
    artifacts_created: list[str] = field(default_factory=list)
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow,
            "status": self.status,
            "steps_completed": self.steps_completed,
            "beliefs_used": self.beliefs_used,
            "artifacts_created": self.artifacts_created,
            "answer_preview": self.answer[:200] if self.answer else "",
            "metadata": self.metadata,
            "duration_ms": round(self.duration_ms, 1),
        }


class FlowOrchestrator:
    """
    Executes the 5 canonical epistemic flows.

    Takes a RoutingDecision from the EpistemicRouter and chains
    the appropriate pipeline steps together.
    """

    def __init__(
        self,
        vault: VaultManager,
        router: EpistemicRouter,
        compiler: BeliefCompiler,
        verifier: VerificationEngine,
        change_pipe: ChangePipeline,
        evolution: EvolutionLogger,
        source_dir: str | None = None,
    ):
        self._vault = vault
        self._router = router
        self._compiler = compiler
        self._verifier = verifier
        self._change_pipe = change_pipe
        self._evolution = evolution
        self._source_dir = source_dir or "."
        self._component_bus: Any = None

    def execute(
        self,
        query: str,
        decision: RoutingDecision | None = None,
        diff_text: str = "",
        is_event: bool = False,
        event_type: str = "",
    ) -> FlowResult:
        """Execute the appropriate canonical flow for a query.

        Args:
            query: The user query or event description.
            decision: Pre-computed routing decision. If None, routes automatically.
            diff_text: Raw diff text for change-driven flows.
            is_event: Whether this is an event trigger.
            event_type: Type of event (pr, commit, release, etc.)
        """
        t0 = time.time()

        if decision is None:
            decision = self._router.route(query, is_event=is_event, event_type=event_type or None)

        flow_map = {
            EpistemicFlow.FAST_ANSWER: self._fast_answer,
            EpistemicFlow.VERIFY_BEFORE_ANSWER: self._verify_then_answer,
            EpistemicFlow.COMPILE_ON_DEMAND: self._compile_on_demand,
            EpistemicFlow.CHANGE_DRIVEN: self._change_driven,
            EpistemicFlow.SELF_IMPROVEMENT: self._self_improvement,
        }

        executor = flow_map.get(decision.flow, self._fast_answer)
        result = executor(query, decision, diff_text)
        result.duration_ms = (time.time() - t0) * 1000
        result.metadata["routing"] = decision.to_dict()

        # ── Self-improvement: feed outcome back to router ──────────
        # This closes the feedback loop: every flow execution tells the
        # router whether it succeeded, enabling adaptive threshold tuning.
        flow_success = result.status == "completed" and len(result.beliefs_used) > 0
        try:
            self._router.record_outcome(
                flow=result.flow,
                success=flow_success,
                confidence=decision.coverage.confidence,
                component_bus=self._component_bus,
            )
        except Exception:
            pass  # Never fail the flow for self-improvement

        logger.info(
            f"FlowOrchestrator: {result.flow} completed in {result.duration_ms:.0f}ms | "
            f"steps={len(result.steps_completed)} beliefs={len(result.beliefs_used)}"
        )
        return result

    # â”€â”€ Flow Implementations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _fast_answer(self, query: str, decision: RoutingDecision, diff: str) -> FlowResult:
        """â‘  Fast Answer: Belief â†’ Action"""
        result = FlowResult(flow="fast_answer", status="completed")

        # Step 1: Look up relevant beliefs
        result.steps_completed.append("belief_lookup")
        beliefs = self._find_relevant_beliefs(query)

        if not beliefs:
            result.status = "partial"
            result.answer = "No relevant beliefs found. Consider running compile_beliefs first."
            return result

        result.beliefs_used = [b.get("entity", "") for b in beliefs]

        # Step 2: Assemble answer from beliefs
        result.steps_completed.append("answer_assembly")
        answer_parts = [f"## Answer (from {len(beliefs)} belief(s))\n"]
        for b in beliefs:
            entity = b.get("entity", "unknown")
            conf = b.get("confidence", 0)
            body = b.get("body", "")[:500]
            answer_parts.append(
                f"### {entity} (confidence: {conf})\n{body}\n"
            )
        result.answer = "\n".join(answer_parts)

        # Step 3: Write action artifact
        result.steps_completed.append("action_write")
        action = self._vault.write_action(
            title=f"Answer: {query[:60]}",
            content=result.answer,
            action_type="answer",
        )
        result.artifacts_created.append(action.get("path", ""))

        return result

    def _verify_then_answer(self, query: str, decision: RoutingDecision, diff: str) -> FlowResult:
        """â‘¡ Verify Before Answer: Belief â†’ Verification â†’ Action"""
        result = FlowResult(flow="verify_before_answer", status="completed")

        # Step 1: Look up beliefs
        result.steps_completed.append("belief_lookup")
        beliefs = self._find_relevant_beliefs(query)
        result.beliefs_used = [b.get("entity", "") for b in beliefs]

        # Step 2: Verify the relevant beliefs
        result.steps_completed.append("verification")
        verification_results = []
        for b in beliefs:
            claim_id = b.get("claim_id", "")
            if claim_id:
                vr = self._verifier.check_belief(claim_id)
                verification_results.append(vr)

        # Step 3: Assemble answer with verification status
        result.steps_completed.append("answer_assembly")
        answer_parts = [f"## Verified Answer (from {len(beliefs)} belief(s))\n"]
        for b, vr in zip(beliefs, verification_results, strict=False):
            entity = b.get("entity", "unknown")
            conf = b.get("confidence", 0)
            body = b.get("body", "")[:500]
            v_status = vr.get("status", "unknown") if vr else "unverified"
            issues = vr.get("issues", []) if vr else []
            answer_parts.append(
                f"### [{v_status}] {entity} (confidence: {conf}, verification: {v_status})\n{body}\n"
            )
            if issues:
                answer_parts.append("**Issues:**")
                for iss in issues:
                    answer_parts.append(f"- {iss}")
                answer_parts.append("")

        result.answer = "\n".join(answer_parts)

        # Step 4: Write verified action
        result.steps_completed.append("action_write")
        action = self._vault.write_action(
            title=f"Verified Answer: {query[:50]}",
            content=result.answer,
            action_type="answer",
        )
        result.artifacts_created.append(action.get("path", ""))

        result.metadata["verification_results"] = verification_results
        return result

    def _compile_on_demand(self, query: str, decision: RoutingDecision, diff: str) -> FlowResult:
        """â‘¢ Compile On Demand: Truth â†’ Belief â†’ Verification â†’ Action"""
        result = FlowResult(flow="compile_on_demand", status="completed")

        # Step 1: Compile beliefs from source
        result.steps_completed.append("truth_compilation")
        compilation = self._compiler.compile_directory(self._source_dir)
        result.metadata["compilation"] = {
            "files_processed": compilation.files_processed,
            "beliefs_written": compilation.beliefs_written,
            "entities_extracted": compilation.entities_extracted,
            "diagrams_generated": compilation.diagrams_generated,
        }

        # Step 2: Look up newly compiled beliefs
        result.steps_completed.append("belief_lookup")
        beliefs = self._find_relevant_beliefs(query)
        result.beliefs_used = [b.get("entity", "") for b in beliefs]

        # Step 3: Verify
        result.steps_completed.append("verification")
        report = self._verifier.full_verification_pass()
        result.metadata["verification"] = report.to_dict()

        # Step 4: Assemble answer
        result.steps_completed.append("answer_assembly")
        if beliefs:
            answer_parts = [
                f"## Compiled Answer (compiled {compilation.beliefs_written} beliefs "
                f"from {compilation.files_processed} files)\n"
            ]
            for b in beliefs:
                entity = b.get("entity", "unknown")
                conf = b.get("confidence", 0)
                body = b.get("body", "")[:500]
                answer_parts.append(f"### {entity} (confidence: {conf})\n{body}\n")
            result.answer = "\n".join(answer_parts)
        else:
            result.answer = (
                f"Compiled {compilation.beliefs_written} beliefs from "
                f"{compilation.files_processed} files, but no beliefs matched "
                f"the query '{query}'. The vault now has broader coverage."
            )
            result.status = "partial"

        # Step 5: Write action
        result.steps_completed.append("action_write")
        action = self._vault.write_action(
            title=f"Compiled Answer: {query[:50]}",
            content=result.answer,
            action_type="answer",
        )
        result.artifacts_created.append(action.get("path", ""))

        return result

    def _change_driven(self, query: str, decision: RoutingDecision, diff: str) -> FlowResult:
        """â‘£ Change-Driven: Event â†’ Truth â†’ Belief â†’ Verification â†’ Action"""
        result = FlowResult(flow="change_driven", status="completed")

        if not diff:
            result.status = "partial"
            result.answer = "No diff provided for change-driven pipeline."
            return result

        # Step 1: Parse the diff and mark impacted beliefs stale.
        result.steps_completed.append("diff_analysis")
        changeset = parse_diff(diff, query)
        changed_files = changeset.files_added + changeset.files_modified
        deleted_files = changeset.files_deleted

        result.steps_completed.append("belief_refresh")
        refresh_result = self._change_pipe.refresh_docs(changed_files + deleted_files)

        # Step 2: Recompile changed files that still exist in the workspace.
        result.steps_completed.append("truth_compilation")
        compilation = self._compiler.compile_paths(self._source_dir, changed_files)

        # Step 3: Re-run verification and blast-radius analysis after the refresh.
        result.steps_completed.append("verification_pass")
        verification = self._verifier.full_verification_pass()
        blast_radius = self._verifier.blast_radius(changed_files + deleted_files)

        # Step 4: Generate the action-layer brief from the updated state.
        result.steps_completed.append("pr_brief_generation")
        brief = self._change_pipe.process_diff(diff, query)
        result.answer = brief.to_markdown()

        # Step 5: Persist artifact references and structured metadata.
        result.steps_completed.append("vault_write")
        if brief.action_path:
            result.artifacts_created.append(brief.action_path)
        if brief.verification_path:
            result.artifacts_created.append(brief.verification_path)
        result.metadata["refresh"] = refresh_result
        result.metadata["compilation"] = {
            "files_processed": compilation.files_processed,
            "beliefs_written": compilation.beliefs_written,
            "entities_extracted": compilation.entities_extracted,
            "diagrams_generated": compilation.diagrams_generated,
            "errors": compilation.errors,
        }
        result.metadata["verification"] = verification.to_dict()
        result.metadata["blast_radius"] = {
            "changed_entity": blast_radius.changed_entity,
            "affected_beliefs": blast_radius.affected_beliefs,
            "affected_entities": blast_radius.affected_entities,
            "risk_level": blast_radius.risk_level,
            "description": blast_radius.description,
        }
        result.metadata["changeset"] = {
            "intent": brief.changeset.intent,
            "files_added": len(brief.changeset.files_added),
            "files_modified": len(brief.changeset.files_modified),
            "files_deleted": len(brief.changeset.files_deleted),
            "lines_added": brief.changeset.lines_added,
            "lines_removed": brief.changeset.lines_removed,
        }
        result.metadata["belief_diff"] = {
            "stale": brief.belief_diff.stale_beliefs,
            "invalidated": brief.belief_diff.invalidated_beliefs,
            "new_coverage": brief.belief_diff.new_coverage_needed,
            "blast_radius": brief.belief_diff.blast_radius,
        }
        result.metadata["findings"] = len(brief.findings)
        result.metadata["risk_level"] = brief.risk_level

        return result

    def _self_improvement(self, query: str, decision: RoutingDecision, diff: str) -> FlowResult:
        """â‘¤ Self-Improvement: Misses â†’ Verification â†’ Evolution â†’ Belief"""
        result = FlowResult(flow="self_improvement", status="completed")

        # Step 1: Record the miss (with source files for structural synthesis)
        result.steps_completed.append("miss_recording")
        entity_key = self._router._extract_entity_key(query)

        # Extract source files from compiled beliefs for Pillar 2 structural synthesis
        source_files: list[str] = []
        try:
            beliefs = self._find_relevant_beliefs(query)
            for b in beliefs:
                src = b.get("source_file", b.get("source", ""))
                if src and src not in source_files:
                    source_files.append(src)
        except Exception:
            pass

        miss_result = self._evolution.record_miss(
            query=query,
            entity_key=entity_key,
            intent=decision.intent.value,
            flow_attempted=decision.flow.value,
            reason=decision.reasoning,
            source_files=source_files,
        )

        # Step 2: Run verification to understand what's broken
        result.steps_completed.append("verification_pass")
        report = self._verifier.full_verification_pass()

        # Step 3: Check if skill gap was triggered
        result.steps_completed.append("evolution_check")
        is_gap = miss_result.get("is_skill_gap", False)

        result.answer = (
            f"## Self-Improvement Report\n\n"
            f"**Entity:** `{entity_key}`\n"
            f"**Miss count:** {miss_result.get('miss_count', 0)}\n"
            f"**Skill gap detected:** {'YES' if is_gap else 'Not yet'}\n\n"
            f"### Verification Status\n"
            f"- Beliefs checked: {report.total_beliefs_checked}\n"
            f"- Contradictions: {len(report.contradictions)}\n"
            f"- Stale: {report.stale_count}\n"
            f"- Mean confidence: {report.mean_confidence:.3f}\n"
        )

        if is_gap:
            result.steps_completed.append("skill_gap_report")
            gap_report = miss_result.get("skill_gap_report", {})
            result.answer += (
                f"\n### Skill Gap Report\n"
                f"A new skill should be created at:\n"
                f"`evolution/skills/{entity_key}/`\n\n"
                f"Report written to: `{gap_report.get('path', 'N/A')}`\n"
            )
            result.artifacts_created.append(gap_report.get("path", ""))

        # Step 4: Still try to compile and answer
        result.steps_completed.append("compile_fallback")
        self._compiler.compile_directory(self._source_dir)
        beliefs = self._find_relevant_beliefs(query)
        if beliefs:
            result.answer += (
                f"\n### Compiled Fallback\n"
                f"After compilation, found {len(beliefs)} relevant belief(s).\n"
            )
            result.beliefs_used = [b.get("entity", "") for b in beliefs]

        result.steps_completed.append("action_write")
        action = self._vault.write_action(
            title=f"Self-Improvement Report: {query[:40]}",
            content=result.answer,
            action_type="report",
        )
        result.artifacts_created.append(action.get("path", ""))

        result.metadata["miss_result"] = miss_result
        result.metadata["verification"] = report.to_dict()
        return result

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _find_relevant_beliefs(self, query: str) -> list[dict[str, Any]]:
        """Find beliefs relevant to a query by keyword matching."""
        import re as _re
        terms = set(
            w.lower() for w in _re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', query)
            if len(w) > 2
        )

        beliefs = self._vault.list_beliefs()
        results = []

        for b in beliefs:
            entity = b.get("entity", "").lower()
            for term in terms:
                if term in entity:
                    # Load full belief
                    full = self._vault.read_belief(b.get("entity", ""))
                    if full:
                        results.append({
                            **b,
                            "body": full.get("body", ""),
                            "claim_id": full.get("frontmatter", {}).get("claim_id", ""),
                        })
                    break

        return results


