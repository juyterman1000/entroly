"""
Skill Engine — Evolution Layer
===============================

Handles the full skill lifecycle:
  1. Skill Synthesis:    Generate skill specs from gap reports
  2. Sandboxed Runner:   Execute skills in isolation
  3. Benchmark Harness:  Evaluate skill fitness
  4. Promotion Engine:   Promote, merge, or prune skills
  5. Registry Manager:   Maintain the skill index
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import re as _re
import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, TYPE_CHECKING

from .context_receipts.models import stable_json
from .path_safety import resolve_dir_within, resolve_file_within, resolve_output_within
from .vault import VaultManager

if TYPE_CHECKING:
    from .reward_crystallizer import CrystallizationEvent

logger = logging.getLogger(__name__)
_SKILL_ID_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_BENCHMARK_SCHEMA = "entroly.skill-benchmark-evidence.v1"
_EVIDENCE_ID_RE = _re.compile(r"^[0-9a-f]{64}$")


def _valid_benchmark_cases(cases: Any) -> bool:
    def meaningful(expected: Any) -> bool:
        if expected is None:
            return False
        if isinstance(expected, str):
            return bool(expected.strip()) and not expected.strip().lower().startswith("should_")
        if isinstance(expected, dict):
            if not expected:
                return False
            return any(
                (key == "$min_items" and type(value) is int and value > 0)
                or (not key.startswith("$") and meaningful(value))
                for key, value in expected.items()
                if isinstance(key, str)
            )
        return True

    return (
        isinstance(cases, list)
        and bool(cases)
        and all(
            isinstance(case, dict)
            and isinstance(case.get("input"), str)
            and bool(case["input"].strip())
            and "expected" in case
            and meaningful(case["expected"])
            for case in cases
        )
    )


def _wilson_bounds(passed: int, total: int, confidence: float) -> tuple[float, float]:
    """Finite-sample interval for a Bernoulli pass rate, without a prior."""
    if total <= 0 or passed < 0 or passed > total:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = passed / total
    z2 = z * z
    scale = 1.0 + z2 / total
    centre = (rate + z2 / (2.0 * total)) / scale
    radius = z / scale * math.sqrt(
        rate * (1.0 - rate) / total + z2 / (4.0 * total * total)
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def promoted_skill_execution_enabled() -> bool:
    """Return whether writable vault tools may run automatically."""
    return os.environ.get("ENTROLY_EXECUTE_PROMOTED_SKILLS", "0") == "1"


# ══════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SkillSpec:
    """A skill specification."""
    skill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    entity: str = ""
    trigger: str = ""  # pattern that triggers this skill
    procedure: str = ""  # step-by-step SOP
    tool_code: str = ""  # Python tool implementation
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    status: str = "draft"  # draft, testing, promoted, pruned
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a skill benchmark."""
    skill_id: str
    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    fitness_score: float = 0.0  # 0.0-1.0
    duration_ms: float = 0
    observations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.passed + self.failed
        return self.passed / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# Skill Synthesizer
# ══════════════════════════════════════════════════════════════════════

class SkillSynthesizer:
    """Generates skill specs from gap reports and failure patterns."""

    def synthesize_from_gap(
        self,
        entity_key: str,
        failing_queries: list[str],
        intent: str = "",
    ) -> SkillSpec:
        """Generate a skill spec from a gap report."""
        # Derive skill name and description from entity
        name = entity_key.replace(":", "_").replace("/", "_")

        # Generate trigger pattern from failing queries
        common_words = self._extract_common_terms(failing_queries)
        trigger = "|".join(common_words[:5]) if common_words else entity_key

        # Generate procedure
        procedure = self._generate_procedure(entity_key, intent, failing_queries)

        # Generate tool code template
        tool_code = self._generate_tool_template(name, entity_key, trigger)

        # Generate test cases
        # A generated template is deliberately *not* considered implemented.
        # Its contract can only pass after the placeholder is replaced by a
        # separately verified implementation.
        tests = [
            {
                "input": q,
                "expected": {"status": "implemented", "entity": entity_key},
            }
            for q in failing_queries[:5]
        ]

        return SkillSpec(
            name=name,
            description=f"Skill for handling {entity_key} queries",
            entity=entity_key,
            trigger=trigger,
            procedure=procedure,
            tool_code=tool_code,
            test_cases=tests,
            status="draft",
        )

    def _extract_common_terms(self, queries: list[str]) -> list[str]:
        """Find common terms across failing queries."""
        import re
        word_counts: dict[str, int] = {}
        for q in queries:
            words = set(
                w.lower() for w in re.findall(r'[a-zA-Z_]\w+', q) if len(w) > 3
            )
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Return words that appear in >50% of queries
        threshold = max(1, len(queries) // 2)
        return sorted(
            [w for w, c in word_counts.items() if c >= threshold],
            key=lambda w: -word_counts[w],
        )

    def _generate_procedure(self, entity: str, intent: str, queries: list[str]) -> str:
        return (
            f"# Procedure for {entity}\n\n"
            f"## Trigger\n"
            f"This skill activates when a query relates to `{entity}`.\n\n"
            f"## Steps\n"
            f"1. Check if relevant source files exist for `{entity}`\n"
            f"2. Extract structural information (AST, dependencies)\n"
            f"3. Build a belief artifact with proper frontmatter\n"
            f"4. Cross-reference with existing beliefs for consistency\n"
            f"5. Generate an answer using the compiled understanding\n\n"
            f"## Evidence Required\n"
            f"- Source file references with line numbers\n"
            f"- Dependency graph edges\n"
            f"- Test coverage status\n"
        )

    # ── Success-driven crystallization ────────────────────────────
    #
    # Mirror of synthesize_from_gap, but driven by RewardCrystallizer
    # events: a query family that consistently beats the global reward
    # baseline (Hoeffding LCB > baseline + ε) gets frozen as a skill.
    #
    # Differences from the failure path:
    #   - status starts as "testing" (not "promoted"). The Hoeffding LCB
    #     proves the selection recipe was useful, not that generated code
    #     satisfies an executable output contract.
    #   - Test cases are real positive samples from the cluster (the
    #     queries that earned the reward), not synthetic placeholders.
    #   - The procedure encodes the actual recipe: weight profile +
    #     fragment IDs that delivered the wins, so a future structural
    #     planner can use them directly.

    def synthesize_from_success(self, event: "CrystallizationEvent") -> SkillSpec:  # type: ignore[name-defined]
        """Generate a SkillSpec from a RewardCrystallizer event.

        The event carries statistical evidence for the source recipe. We
        materialize it as a candidate and require independent validation.
        """
        from .reward_crystallizer import CrystallizationEvent  # noqa: F401, runtime check

        terms = event.common_terms or [event.cluster_id]
        name = "crystallized_" + "_".join(terms[:3])[:48]
        entity = "|".join(terms[:5]) or event.cluster_id
        trigger = "|".join(terms[:5]) if terms else event.cluster_id

        procedure = (
            f"# Crystallized skill — {name}\n\n"
            f"## Provenance\n"
            f"- cluster_id: `{event.cluster_id}`\n"
            f"- samples (n): {event.n_samples}\n"
            f"- mean reward: {event.mean_reward:.3f}\n"
            f"- Hoeffding lower bound: {event.lcb_reward:.3f}\n"
            f"- baseline reward: {event.baseline_reward:.3f}\n"
            f"- effect size (LCB − baseline): {event.effect_size:.3f}\n\n"
            f"## Trigger\n"
            f"This skill activates when a query matches the pattern: "
            f"`{trigger}`.\n\n"
            f"## Recipe\n"
            f"Sample queries that earned high reward:\n"
            + "\n".join(f"- {q!r}" for q in event.sample_queries[:5])
            + "\n\n"
            "## Selection strategy (snapshot at crystallization)\n"
            "PRISM weight profile that delivered the wins:\n"
            + "\n".join(
                f"- `{k}` = {v:.4f}"
                for k, v in sorted(event.weight_profile.items())
            )
            + "\n\n"
            "## Fragment recipe (most-frequently selected)\n"
            + "\n".join(f"- `{fid}`" for fid in event.fragment_recipe[:8])
            + "\n"
        )

        tool_code = self._generate_crystallized_tool(
            name=name,
            trigger=trigger,
            fragment_recipe=event.fragment_recipe,
            weight_profile=event.weight_profile,
        )

        # Real positive samples as test cases — the queries that won.
        tests = [
            {
                "input": q,
                "expected": {
                    "status": "success",
                    "fragment_recipe": list(event.fragment_recipe),
                },
            }
            for q in event.sample_queries[:5]
        ]

        spec = SkillSpec(
            name=name,
            description=(
                f"Crystallized from sustained high reward "
                f"(LCB={event.lcb_reward:.3f} > baseline={event.baseline_reward:.3f})"
            ),
            entity=entity,
            trigger=trigger,
            procedure=procedure,
            tool_code=tool_code,
            test_cases=tests,
            status="testing",
            metrics={
                "fitness_score": round(event.lcb_reward, 4),
                "samples": float(event.n_samples),
                "effect_size": round(event.effect_size, 4),
                "source": "crystallization",
            },
        )
        return spec

    def _generate_crystallized_tool(
        self,
        name: str,
        trigger: str,
        fragment_recipe: list[str],
        weight_profile: dict[str, float],
    ) -> str:
        """Emit a tool whose execute() returns the cluster's winning recipe.

        The runtime can use this output to short-circuit fragment selection
        for matched queries (see future fast-path work). For now the tool
        is a faithful, deterministic record of the strategy that worked.
        """
        pattern_lit = repr(self._literal_trigger_pattern(trigger))
        recipe_lit = json.dumps(list(fragment_recipe))
        weights_lit = json.dumps(weight_profile)
        return (
            f'"""Crystallized skill tool.\n'
            f'\n'
            f'Auto-generated by RewardCrystallizer. Encodes the fragment\n'
            f'recipe and PRISM weight profile that delivered sustained\n'
            f'high reward for the trigger pattern.\n'
            f'"""\n\n'
            f'import re\n\n'
            f'TRIGGER_PATTERN = re.compile({pattern_lit}, re.I)\n'
            f'FRAGMENT_RECIPE = {recipe_lit}\n'
            f'WEIGHT_PROFILE = {weights_lit}\n\n\n'
            f'def matches(query: str) -> bool:\n'
            f'    return bool(TRIGGER_PATTERN.search(query or ""))\n\n\n'
            f'def execute(query: str, context: dict) -> dict:\n'
            f'    """Return the recipe that previously won for this query family."""\n'
            f'    return {{\n'
            f'        "status": "success",\n'
            f'        "skill": {name!r},\n'
            f'        "fragment_recipe": list(FRAGMENT_RECIPE),\n'
            f'        "weight_profile": dict(WEIGHT_PROFILE),\n'
            f'        "results": [\n'
            f'            {{"file": fid, "snippet": ""}}\n'
            f'            for fid in FRAGMENT_RECIPE\n'
            f'        ],\n'
            f'    }}\n'
        )

    def _generate_tool_template(self, name: str, entity: str, trigger: str) -> str:
        pattern_lit = repr(self._literal_trigger_pattern(trigger))
        return (
            f'"""\n'
            f'Auto-generated skill tool.\n'
            f'"""\n\n'
            f'import re\n\n'
            f'TRIGGER_PATTERN = re.compile({pattern_lit}, re.I)\n\n\n'
            f'def matches(query: str) -> bool:\n'
            f'    """Check if this skill should handle the query."""\n'
            f'    return bool(TRIGGER_PATTERN.search(query))\n\n\n'
            f'def execute(query: str, context: dict) -> dict:\n'
            f'    """Execute the skill logic."""\n'
            f'    return {{\n'
            f'        "status": "executed",\n'
            f'        "skill": {name!r},\n'
            f'        "entity": {entity!r},\n'
            f'        "result": "Skill implementation needed",\n'
            f'    }}\n'
        )

    @staticmethod
    def _literal_trigger_pattern(trigger: str) -> str:
        """Build a regex from literal alternation terms."""
        terms = [term for term in trigger.split("|") if term]
        escaped = "|".join(_re.escape(term) for term in terms)
        return rf"\b({escaped})\b"


# ══════════════════════════════════════════════════════════════════════
# Structural Synthesizer — Entropy-Gradient Program Synthesis (Pillar 2)
# ══════════════════════════════════════════════════════════════════════
#
# Instead of asking an LLM to write a tool, this synthesizer derives a
# candidate tool from repository structure and subjects its output to a
# benchmark contract before promotion.
#
# Given a skill gap at entity E, it computes:
#   1. Dependency Closure: all files/functions reachable from E via
#      import graph + call graph (transitive, bounded by depth K).
#   2. Entropy Ranking: for each node in the closure, the Shannon
#      entropy score from the Rust engine — high entropy = high
#      information density = the code that matters.
#   3. Structural Invariants: function signatures, type annotations,
#      return types extracted from source files — the "contract" of E.
#
# The output tool's execute() function performs a local code search
# along the entropy gradient, returning the most informative context
# about the entity. This path is deterministic and makes no provider API call,
# but it still consumes local compute and filesystem I/O. Returning source
# excerpts does not prove that a candidate is correct or safe.
#
# Mathematical grounding:
#   - The entropy gradient ∇H(E) points toward the direction of
#     maximum information gain. Following it yields the minimal set
#     of code fragments that maximally reduces uncertainty about E.
#   - This is equivalent to solving the rate-distortion problem
#     R(D) = min_{p(ê|e)} I(E; Ê) s.t. E[d(e, ê)] ≤ D
#     where the "distortion" is miss rate and "rate" is token cost.

class StructuralSynthesizer:
    """Provider-call-free skill synthesis via structural analysis.

    Uses the Rust SAST/entropy engine to analyze code structure and
    generate tools that navigate the information topology of a codebase.
    Synthesis is deterministic and runs on the local CPU. Local compute and
    storage still have operational cost.
    """

    # Maximum depth of dependency traversal
    MAX_CLOSURE_DEPTH = 3
    # Minimum entropy score to include a fragment in the closure
    ENTROPY_FLOOR = 0.15

    def __init__(self, rust_engine: Any = None):
        """
        Args:
            rust_engine: Optional entroly_core.EntrolyEngine instance.
                         If None, falls back to pure-Python heuristics.
        """
        self._engine = rust_engine

    def synthesize_structural(
        self,
        entity_key: str,
        source_files: list[str],
        failing_queries: list[str],
    ) -> SkillSpec | None:
        """Synthesize a skill from structural analysis of source files.

        Returns None if structural synthesis cannot produce a useful tool
        (e.g., no source files, no parseable signatures). The daemon
        falls back to LLM synthesis (budget-gated) in that case.
        """
        if not source_files:
            return None

        # Step 1: Extract structural invariants from source files
        invariants = self._extract_invariants(source_files)
        if not invariants["signatures"] and not invariants["imports"]:
            return None  # Nothing useful to synthesize from

        # Step 2: Compute entropy-ranked closure
        closure = self._compute_entropy_closure(
            entity_key, source_files, invariants
        )

        # Step 3: Generate the tool code
        name = entity_key.replace(":", "_").replace("/", "_").replace(".", "_")
        tool_code = self._emit_structural_tool(name, entity_key, invariants, closure)

        # Step 4: Generate trigger pattern from failing queries
        common_terms = self._extract_key_terms(failing_queries)
        trigger = "|".join(common_terms[:5]) if common_terms else entity_key

        # Step 5: Build test cases from failing queries
        tests = [
            {
                "input": q,
                "expected": {
                    "status": "executed",
                    "entity": entity_key,
                    "synthesis_method": "structural_induction",
                    "results": {"$min_items": 1},
                },
            }
            for q in failing_queries[:5]
        ]

        return SkillSpec(
            name=name,
            description=(
                f"Structural candidate skill for {entity_key} "
                "(no provider call; local compute applies)"
            ),
            entity=entity_key,
            trigger=trigger,
            procedure=self._generate_structural_procedure(entity_key, invariants),
            tool_code=tool_code,
            test_cases=tests,
            status="draft",
            metrics={"synthesis_method": 0.0},  # 0.0 = structural, 1.0 = LLM
        )

    def _extract_invariants(self, source_files: list[str]) -> dict[str, Any]:
        """Extract structural invariants from source files.

        Parses function signatures, class definitions, import statements,
        and type annotations using regex-based AST approximation.
        This is O(N·L) where N=files, L=avg lines — microseconds.
        """
        signatures: list[dict[str, str]] = []
        imports: list[str] = []
        classes: list[str] = []
        type_hints: list[str] = []
        file_summaries: list[dict[str, Any]] = []

        for fpath in source_files:
            try:
                from pathlib import Path
                p = Path(fpath)
                if not p.exists() or p.stat().st_size > 500_000:
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                file_sigs: list[str] = []
                file_imports: list[str] = []

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Function/method signatures
                    m = _re.match(
                        r'^(\s*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(\s*->\s*(.+?))?\s*:',
                        line,
                    )
                    if m:
                        indent, async_kw, fname, params, _, ret_type = m.groups()
                        sig = {
                            "name": fname,
                            "params": params.strip(),
                            "return_type": (ret_type or "").strip(),
                            "file": str(p),
                            "line": i + 1,
                            "is_async": bool(async_kw),
                            "indent": len(indent or ""),
                        }
                        signatures.append(sig)
                        file_sigs.append(fname)

                    # Class definitions
                    cm = _re.match(r'^\s*class\s+(\w+)\s*(\([^)]*\))?\s*:', line)
                    if cm:
                        classes.append(cm.group(1))

                    # Import statements
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        imports.append(stripped)
                        file_imports.append(stripped)

                    # Type annotations on assignments
                    tm = _re.match(r'^\s*(\w+)\s*:\s*(\w[\w\[\], |]*)\s*=', line)
                    if tm:
                        type_hints.append(f"{tm.group(1)}: {tm.group(2)}")

                file_summaries.append({
                    "path": str(p),
                    "lines": len(lines),
                    "functions": file_sigs,
                    "imports": file_imports,
                })

            except Exception:
                continue  # Skip unreadable files silently

        return {
            "signatures": signatures,
            "imports": list(set(imports)),
            "classes": classes,
            "type_hints": type_hints,
            "file_summaries": file_summaries,
        }

    def _compute_entropy_closure(
        self,
        entity_key: str,
        source_files: list[str],
        invariants: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Compute the entropy-ranked dependency closure around entity E.

        For each function in the invariants, computes its information
        value (entropy score from the Rust engine if available, else
        heuristic based on cyclomatic complexity proxy).

        Returns nodes sorted by entropy (highest first) — following
        the gradient ∇H(E) toward maximum information gain.
        """
        closure: list[dict[str, Any]] = []

        for sig in invariants["signatures"]:
            # Heuristic entropy: functions with more parameters, return types,
            # and async markers carry more structural information
            param_count = len([p for p in sig["params"].split(",") if p.strip()])
            has_return = 1.0 if sig["return_type"] else 0.0
            is_async = 0.3 if sig["is_async"] else 0.0
            # Nesting depth (indent level) inversely correlates with
            # architectural importance
            depth_penalty = max(0.0, 1.0 - sig["indent"] / 16.0)

            # Composite entropy proxy: H(f) ≈ log2(params+1)·depth·return_bonus
            import math
            entropy_proxy = (
                math.log2(param_count + 1) * depth_penalty
                + has_return * 0.5
                + is_async
            )

            closure.append({
                "name": sig["name"],
                "file": sig["file"],
                "line": sig["line"],
                "entropy": round(entropy_proxy, 4),
                "signature": f"def {sig['name']}({sig['params']})"
                             + (f" -> {sig['return_type']}" if sig["return_type"] else ""),
            })

        # Sort by entropy descending — the gradient direction
        closure.sort(key=lambda n: n["entropy"], reverse=True)

        # Filter below entropy floor
        closure = [n for n in closure if n["entropy"] >= self.ENTROPY_FLOOR]

        return closure

    def _emit_structural_tool(
        self,
        name: str,
        entity: str,
        invariants: dict[str, Any],
        closure: list[dict[str, Any]],
    ) -> str:
        """Emit a Python tool that navigates the structural closure.

        The generated execute() function:
          1. Reads the source files associated with the entity
          2. Extracts the top-K most informative functions (by entropy)
          3. Returns their signatures + surrounding context
        This is deterministic and returns bounded current-source excerpts when
        files remain available. The benchmark contract must still reject empty,
        stale, malformed, or unsafe output before promotion.
        """
        pattern_lit = repr(SkillSynthesizer._literal_trigger_pattern(entity))
        closure_lit = repr(closure[:15])
        imports_lit = repr([imp[:80] for imp in invariants["imports"][:10]])
        classes_lit = repr(invariants["classes"][:10])

        return f'''"""
Structural skill tool.
Synthesis: entropy-gradient structural induction (no provider call; CPU-only)

This tool was generated WITHOUT any LLM call. It navigates the
information topology of the codebase, returning
the most informative code fragments ranked by Shannon entropy.
"""

import re
import os

TRIGGER_PATTERN = re.compile({pattern_lit}, re.I)

# Static knowledge table — entropy-ranked structural closure
# Computed by StructuralSynthesizer from AST analysis
_CLOSURE = {closure_lit}

_IMPORTS = {imports_lit}
_CLASSES = {classes_lit}


def matches(query: str) -> bool:
    """Check if this skill should handle the query."""
    return bool(TRIGGER_PATTERN.search(query))


def execute(query: str, context: dict) -> dict:
    """Navigate the entropy gradient.

    Returns the most informative code fragments, ranked by
    information density. All data comes from local file I/O —
    no provider API calls. Local compute and I/O still apply.
    """
    results = []
    for node in _CLOSURE:
        try:
            if os.path.exists(node["file"]):
                with open(node["file"], "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    start = max(0, node["line"] - 1)
                    end = min(len(lines), start + 20)
                    snippet = "".join(lines[start:end])
                    results.append({{
                        "function": node["name"],
                        "signature": node["signature"],
                        "entropy": node["entropy"],
                        "file": node["file"],
                        "line": node["line"],
                        "snippet": snippet,
                    }})
        except Exception:
            continue

    return {{
        "status": "executed",
        "skill": {name!r},
        "entity": {entity!r},
        "synthesis_method": "structural_induction",
        "token_cost": 0,
        "token_cost_scope": "provider_tokens_only",
        "local_compute": True,
        "closure_size": len(_CLOSURE),
        "imports": _IMPORTS,
        "classes": _CLASSES,
        "results": results[:10],
    }}
'''

    def _generate_structural_procedure(
        self, entity: str, invariants: dict[str, Any]
    ) -> str:
        """Generate a procedure doc for the structural skill."""
        sig_count = len(invariants["signatures"])
        class_count = len(invariants["classes"])
        import_count = len(invariants["imports"])

        return (
            f"# Structural Procedure for {entity}\n\n"
            f"## Synthesis Method\n"
            f"Entropy-gradient structural induction (no provider call; CPU-only).\n"
            f"Generated from AST analysis of {sig_count} functions, "
            f"{class_count} classes, {import_count} imports.\n\n"
            f"## Steps\n"
            f"1. Read source files associated with `{entity}`\n"
            f"2. Rank functions by Shannon entropy (information density)\n"
            f"3. Return top-K most informative code fragments\n"
            f"4. Include dependency context (imports, classes)\n\n"
            f"## Operational properties\n"
            f"- No provider API call; local compute and I/O still apply\n"
            f"- Deterministic output (same input → same result)\n"
            f"- Returns bounded source excerpts; benchmark and security gates still apply\n"
        )

    @staticmethod
    def _extract_key_terms(queries: list[str]) -> list[str]:
        """Extract common terms from queries for trigger patterns."""
        word_counts: dict[str, int] = {}
        for q in queries:
            words = set(
                w.lower() for w in _re.findall(r'[a-zA-Z_]\w+', q) if len(w) > 3
            )
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        threshold = max(1, len(queries) // 2)
        return sorted(
            [w for w, c in word_counts.items() if c >= threshold],
            key=lambda w: -word_counts[w],
        )


# ══════════════════════════════════════════════════════════════════════
# Sandboxed Runner
# ══════════════════════════════════════════════════════════════════════

class SandboxedRunner:
    """Runs skill tools in a separate subprocess with a timeout.

    This is an availability boundary, not a security sandbox. Callers must
    only run writable vault tools automatically after explicit user opt-in.
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self._timeout = timeout_seconds

    def run_tool(self, tool_code: str, query: str) -> dict[str, Any]:
        """Execute a skill tool in a subprocess sandbox."""
        # Write tool to temp file and run in subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            # Wrap the tool with execution harness
            harness = (
                f"{tool_code}\n\n"
                f'if __name__ == "__main__":\n'
                f'    import json, sys\n'
                f'    query = sys.argv[1] if len(sys.argv) > 1 else ""\n'
                f'    result = execute(query, {{}})\n'
                f'    print(json.dumps(result))\n'
            )
            f.write(harness)
            temp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, temp_path, query],
                capture_output=True, text=True,
                timeout=self._timeout,
            )
            if proc.returncode == 0:
                try:
                    result = json.loads(proc.stdout.strip())
                    return {"status": "success", "result": result}
                except json.JSONDecodeError:
                    return {"status": "success", "result": proc.stdout.strip()}
            else:
                return {
                    "status": "error",
                    "error": proc.stderr.strip(),
                    "returncode": proc.returncode,
                }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "timeout": self._timeout}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════
# Benchmark Harness
# ══════════════════════════════════════════════════════════════════════

class SkillBenchmark:
    """Evaluates skill fitness by running test cases."""

    def __init__(self, runner: SandboxedRunner | None = None):
        self._runner = runner or SandboxedRunner()

    def benchmark(self, skill: SkillSpec) -> BenchmarkResult:
        """Run all test cases for a skill and compute fitness."""
        t0 = time.time()
        result = BenchmarkResult(skill_id=skill.skill_id)

        for index, tc in enumerate(skill.test_cases):
            query = tc.get("input", "")
            expected = tc.get("expected")
            observation = {
                "case_index": index,
                "input_sha256": hashlib.sha256(
                    str(query).encode("utf-8")
                ).hexdigest(),
                "expected_sha256": hashlib.sha256(
                    stable_json(expected).encode("utf-8")
                ).hexdigest(),
            }
            try:
                run = self._runner.run_tool(skill.tool_code, query)
                observation["execution_status"] = run.get("status", "unknown")
                if run.get("status") != "success":
                    result.failed += 1
                    observation["matched"] = False
                    result.errors.append(
                        f"Query '{query}': {run.get('error', run.get('status', 'unknown'))}"
                    )
                    continue

                matched, reason = self._matches_expected(run.get("result"), expected)
                observation["output_sha256"] = hashlib.sha256(
                    json.dumps(
                        run.get("result"), sort_keys=True, default=str
                    ).encode("utf-8")
                ).hexdigest()
                observation["matched"] = matched
                if matched:
                    result.passed += 1
                else:
                    result.failed += 1
                    result.errors.append(
                        f"Query '{query}': output contract failed: {reason}"
                    )
            except Exception as e:
                result.failed += 1
                observation["execution_status"] = "error"
                observation["matched"] = False
                result.errors.append(f"Query '{query}': {e}")
            finally:
                result.observations.append(observation)

        result.duration_ms = (time.time() - t0) * 1000
        total = result.passed + result.failed
        result.fitness_score = result.passed / total if total > 0 else 0.0

        return result

    @classmethod
    def _matches_expected(
        cls,
        actual: Any,
        expected: Any,
        path: str = "result",
    ) -> tuple[bool, str]:
        """Verify an explicit deterministic output contract.

        Dictionaries are recursively matched as subsets. ``$min_items`` is
        available for collections. Empty expectations and historical
        ``should_*`` placeholders fail closed instead of silently passing.
        """
        if expected is None or expected == "":
            return False, "missing expected output"

        if isinstance(expected, str):
            if expected.strip().lower().startswith("should_"):
                return False, f"unverifiable legacy expectation {expected!r}"
            if isinstance(actual, str):
                matched = actual.strip() == expected.strip()
            else:
                matched = expected.strip() in json.dumps(
                    actual, sort_keys=True, default=str
                )
            return matched, "matched" if matched else f"expected {expected!r}"

        if isinstance(expected, dict):
            if "$min_items" in expected:
                minimum = expected["$min_items"]
                if not isinstance(minimum, int) or minimum < 0:
                    return False, f"{path} has invalid $min_items contract"
                if not isinstance(actual, (list, tuple, dict, str)):
                    return False, f"{path} is not a collection"
                if len(actual) < minimum:
                    return (
                        False,
                        f"{path} has {len(actual)} items; expected at least {minimum}",
                    )

            remaining = {k: v for k, v in expected.items() if not k.startswith("$")}
            if remaining and not isinstance(actual, dict):
                return False, f"{path} is not an object"
            for key, value in remaining.items():
                if key not in actual:
                    return False, f"{path}.{key} is missing"
                matched, reason = cls._matches_expected(
                    actual[key], value, f"{path}.{key}"
                )
                if not matched:
                    return False, reason
            return True, "matched"

        if isinstance(expected, list):
            if actual != expected:
                return False, f"{path} did not equal the expected list"
            return True, "matched"

        matched = actual == expected
        return (
            matched,
            "matched" if matched else f"{path} expected {expected!r}, got {actual!r}",
        )


# ══════════════════════════════════════════════════════════════════════
# Skill Engine (Promotion / Pruning / Registry)
# ══════════════════════════════════════════════════════════════════════

class SkillEngine:
    """
    Full skill lifecycle manager.

    Creates skills from gap reports, benchmarks them, promotes or prunes,
    and maintains the registry.
    """

    PROMOTION_THRESHOLD = 0.7  # fitness score to promote
    PRUNE_THRESHOLD = 0.3      # fitness score to prune
    DECISION_CONFIDENCE = 0.95

    def __init__(self, vault: VaultManager):
        self._vault = vault
        self._synthesizer = SkillSynthesizer()
        self._runner = SandboxedRunner()
        self._benchmark = SkillBenchmark(self._runner)

    def _benchmark_material(self, skill_id: str) -> dict[str, str]:
        """Fingerprint the files actually supplied to the benchmark runner."""
        skill_dir = self._skill_dir(skill_id)
        if skill_dir is None:
            raise ValueError("skill directory is unavailable")
        tool = resolve_file_within(skill_dir, "tool.py")
        cases = resolve_file_within(skill_dir, "tests/test_cases.json")
        if tool is None or cases is None:
            raise ValueError("skill code or test contract is missing or uncontained")
        code_bytes, case_bytes = tool.read_bytes(), cases.read_bytes()
        try:
            loaded = json.loads(case_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("test contract is not valid JSON") from exc
        if not _valid_benchmark_cases(loaded):
            raise ValueError("test contract requires nonempty cases with expectations")
        return {
            "tool_sha256": hashlib.sha256(code_bytes).hexdigest(),
            "tests_sha256": hashlib.sha256(case_bytes).hexdigest(),
            "evaluator_sha256": hashlib.sha256("\n".join((
                inspect.getsource(SandboxedRunner.run_tool),
                inspect.getsource(SkillBenchmark.benchmark),
                inspect.getsource(SkillBenchmark._matches_expected),
                sys.version,
                sys.executable,
            )).encode("utf-8")).hexdigest(),
        }

    def _benchmark_key(self, *, create: bool) -> bytes:
        evolution = self._vault.config.path / "evolution"
        if create:
            self._vault.ensure_structure()
        key_path = resolve_output_within(evolution, "benchmark.key")
        if key_path is None:
            raise ValueError("benchmark key path escapes the vault")
        if create and not key_path.exists():
            try:
                with key_path.open("x", encoding="ascii") as handle:
                    handle.write(secrets.token_hex(32) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.chmod(key_path, 0o600)
                except OSError:
                    pass
            except FileExistsError:
                pass
        try:
            key = bytes.fromhex(key_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError) as exc:
            raise ValueError("benchmark key is unavailable or invalid") from exc
        if len(key) != 32:
            raise ValueError("benchmark key must contain 32 bytes")
        return key

    def _persist_benchmark_evidence(
        self,
        skill_id: str,
        material: dict[str, str],
        result: BenchmarkResult,
        *,
        evaluation_scope: str,
        evaluation_sha256: str,
    ) -> str:
        if result.passed + result.failed < 1:
            raise ValueError("an empty benchmark cannot justify a skill decision")
        payload = {
            "schema_version": _BENCHMARK_SCHEMA,
            "skill_id": skill_id,
            "material": material,
            "evaluation_scope": evaluation_scope,
            "evaluation_sha256": evaluation_sha256,
            "passed": result.passed,
            "failed": result.failed,
            "fitness_score": result.fitness_score,
            "error_sha256": [
                hashlib.sha256(error.encode("utf-8")).hexdigest()
                for error in result.errors
            ],
            "observations": result.observations,
            "duration_ms": result.duration_ms,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        serialized = stable_json(payload).encode("utf-8")
        evidence_id = hashlib.sha256(serialized).hexdigest()
        signature = hmac.new(
            self._benchmark_key(create=True), serialized, hashlib.sha256
        ).hexdigest()
        evolution = self._vault.config.path / "evolution"
        (evolution / "benchmark-evidence").mkdir(exist_ok=True)
        evidence_root = resolve_dir_within(evolution, "benchmark-evidence")
        if evidence_root is None:
            raise ValueError("benchmark evidence root escapes the vault")
        directory = resolve_dir_within(evidence_root, skill_id)
        if directory is None:
            candidate = resolve_output_within(evidence_root, skill_id)
            if candidate is None or candidate.exists():
                raise ValueError("benchmark evidence directory is unsafe")
            candidate.mkdir()
            directory = resolve_dir_within(evidence_root, skill_id)
        if directory is None:
            raise ValueError("benchmark evidence directory is unavailable")
        path = resolve_output_within(directory, f"{evidence_id}.json")
        if path is None:
            raise ValueError("benchmark evidence path is unsafe")
        record = stable_json({"payload": payload, "signature": signature}) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(record)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if path.read_text(encoding="utf-8") != record:
                raise ValueError("benchmark evidence ID collision or tampering")
        return evidence_id

    def _verified_benchmark_evidence(
        self, skill_id: str, metrics: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        evidence_id = metrics.get("benchmark_evidence_id")
        if not isinstance(evidence_id, str) or not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            return None, "benchmark evidence is missing"
        evolution = self._vault.config.path / "evolution"
        root = resolve_dir_within(evolution, "benchmark-evidence")
        directory = resolve_dir_within(root, skill_id) if root is not None else None
        path = (
            resolve_file_within(directory, f"{evidence_id}.json")
            if directory is not None else None
        )
        if path is None:
            return None, "benchmark evidence file is missing or uncontained"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            signature = record["signature"]
            serialized = stable_json(payload).encode("utf-8")
            material = self._benchmark_material(skill_id)
            key = self._benchmark_key(create=False)
            score = payload["fitness_score"]
            passed, failed = payload["passed"], payload["failed"]
            valid = (
                isinstance(payload, dict)
                and payload.get("schema_version") == _BENCHMARK_SCHEMA
                and payload.get("skill_id") == skill_id
                and payload.get("material") == material
                and payload.get("evaluation_scope") in {"development", "caller_holdout"}
                and isinstance(payload.get("evaluation_sha256"), str)
                and bool(_EVIDENCE_ID_RE.fullmatch(payload["evaluation_sha256"]))
                and hashlib.sha256(serialized).hexdigest() == evidence_id
                and isinstance(signature, str)
                and hmac.compare_digest(
                    signature, hmac.new(key, serialized, hashlib.sha256).hexdigest()
                )
                and type(passed) is int and type(failed) is int
                and passed >= 0 and failed >= 0 and passed + failed > 0
                and type(score) in (int, float)
                and score == passed / (passed + failed)
                and isinstance(payload.get("observations"), list)
                and len(payload["observations"]) == passed + failed
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None, "benchmark evidence is unreadable or invalid"
        if not valid:
            return None, "benchmark evidence is stale or altered"

        # A mutable metrics file must not be able to point back to an older
        # successful run after a newer regression. Every retained record is
        # authenticated before the latest evaluation is chosen.
        latest: tuple[str, str] | None = None
        try:
            for candidate in directory.glob("*.json"):
                safe_candidate = resolve_file_within(directory, candidate)
                if safe_candidate is None or not _EVIDENCE_ID_RE.fullmatch(candidate.stem):
                    return None, "benchmark evidence directory contains an unsafe record"
                entry = json.loads(safe_candidate.read_text(encoding="utf-8"))
                past = entry["payload"]
                past_bytes = stable_json(past).encode("utf-8")
                if (
                    past.get("schema_version") != _BENCHMARK_SCHEMA
                    or past.get("skill_id") != skill_id
                    or hashlib.sha256(past_bytes).hexdigest() != candidate.stem
                    or not hmac.compare_digest(
                        entry["signature"],
                        hmac.new(key, past_bytes, hashlib.sha256).hexdigest(),
                    )
                    or not isinstance(past.get("evaluated_at"), str)
                ):
                    return None, "benchmark evidence directory contains an altered record"
                position = (past["evaluated_at"], candidate.stem)
                if latest is None or position > latest:
                    latest = position
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None, "benchmark evidence history is unreadable"
        if latest is None or latest[1] != evidence_id:
            return None, "a newer benchmark supersedes this result"
        return payload, "verified"

    def create_skill(
        self,
        entity_key: str,
        failing_queries: list[str],
        intent: str = "",
    ) -> dict[str, Any]:
        """Create a new skill from a gap report."""
        self._vault.ensure_structure()
        spec = self._synthesizer.synthesize_from_gap(entity_key, failing_queries, intent)

        # Write skill package
        skill_dir = self._vault.config.path / "evolution" / "skills" / spec.skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        # SKILL.md
        (skill_dir / "SKILL.md").write_text(
            f"---\n"
            f"skill_id: {spec.skill_id}\n"
            f"name: {spec.name}\n"
            f"entity: {spec.entity}\n"
            f"status: {spec.status}\n"
            f"created_at: {spec.created_at}\n"
            f"---\n\n"
            f"# {spec.name}\n\n"
            f"{spec.description}\n\n"
            f"{spec.procedure}\n",
            encoding="utf-8",
        )

        # tool.py
        (skill_dir / "tool.py").write_text(spec.tool_code, encoding="utf-8")

        # metrics.json
        (skill_dir / "metrics.json").write_text(
            json.dumps({
                "created_at": spec.created_at,
                "fitness_score": 0.0,
                "runs": 0,
                "successes": 0,
                "failures": 0,
            }, indent=2),
            encoding="utf-8",
        )

        # tests/
        tests_dir = skill_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test_cases.json").write_text(
            json.dumps(spec.test_cases, indent=2),
            encoding="utf-8",
        )

        # Update registry
        self._update_registry(spec, "created")

        logger.info(f"SkillEngine: created skill {spec.skill_id} for {entity_key}")
        return {
            "status": "created",
            "skill_id": spec.skill_id,
            "name": spec.name,
            "path": str(skill_dir),
        }

    def crystallize_skill(self, event: "CrystallizationEvent") -> dict[str, Any]:  # type: ignore[name-defined]
        """Materialize a RewardCrystallizer event as a benchmark candidate.

        The event's Hoeffding LCB is evidence for the context-selection
        recipe. It is not evidence that newly generated executable code is
        correct, so the skill remains in testing until its output contract
        passes an independent benchmark.

        Returns the same shape as ``create_skill``.
        """
        from .reward_crystallizer import CrystallizationEvent  # noqa: F401, runtime check

        self._vault.ensure_structure()
        spec = self._synthesizer.synthesize_from_success(event)

        skill_dir = self._vault.config.path / "evolution" / "skills" / spec.skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        (skill_dir / "SKILL.md").write_text(
            f"---\n"
            f"skill_id: {spec.skill_id}\n"
            f"name: {spec.name}\n"
            f"entity: {spec.entity}\n"
            f"status: {spec.status}\n"
            f"created_at: {spec.created_at}\n"
            f"source: crystallization\n"
            f"cluster_id: {event.cluster_id}\n"
            f"---\n\n"
            f"# {spec.name}\n\n"
            f"{spec.description}\n\n"
            f"{spec.procedure}\n",
            encoding="utf-8",
        )
        (skill_dir / "tool.py").write_text(spec.tool_code, encoding="utf-8")
        (skill_dir / "metrics.json").write_text(
            json.dumps({
                "created_at": spec.created_at,
                "fitness_score": spec.metrics.get("fitness_score", 0.0),
                "samples": spec.metrics.get("samples", 0),
                "effect_size": spec.metrics.get("effect_size", 0.0),
                "source": "crystallization",
                "cluster_id": event.cluster_id,
                "runs": 0, "successes": 0, "failures": 0,
            }, indent=2),
            encoding="utf-8",
        )
        tests_dir = skill_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test_cases.json").write_text(
            json.dumps(spec.test_cases, indent=2),
            encoding="utf-8",
        )

        self._update_registry(spec, "crystallized")

        logger.info(
            "SkillEngine: crystallized skill %s (cluster=%s, lcb=%.3f, n=%d)",
            spec.skill_id, event.cluster_id, event.lcb_reward, event.n_samples,
        )
        return {
            "status": "crystallized",
            "skill_status": spec.status,
            "skill_id": spec.skill_id,
            "name": spec.name,
            "path": str(skill_dir),
            "fitness_score": spec.metrics.get("fitness_score", 0.0),
            "cluster_id": event.cluster_id,
        }

    def benchmark_skill(
        self, skill_id: str, *, validation_cases: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Benchmark a frozen candidate with development or caller-held-out cases.

        A caller-held-out set is never read from the writable skill directory.
        The caller is responsible for the independence and quality of those
        cases; this method proves only their exact use and observed outcomes.
        """
        if not self._valid_skill_id(skill_id):
            return {"status": "invalid_skill_id", "skill_id": skill_id}
        spec = self._load_skill(skill_id)
        if not spec:
            return {"status": "not_found", "skill_id": skill_id}
        if validation_cases is not None:
            if not _valid_benchmark_cases(validation_cases):
                return {
                    "status": "invalid_contract", "skill_id": skill_id,
                    "reason": "validation_cases require nonempty inputs and expectations",
                }
            development_inputs = {case["input"] for case in spec.test_cases}
            if any(case["input"] in development_inputs for case in validation_cases):
                return {
                    "status": "invalid_contract", "skill_id": skill_id,
                    "reason": "validation cases must use unseen inputs",
                }
            validation_inputs = [case["input"] for case in validation_cases]
            if len(validation_inputs) != len(set(validation_inputs)):
                return {
                    "status": "invalid_contract", "skill_id": skill_id,
                    "reason": "validation inputs must be distinct",
                }
            evaluation_scope = "caller_holdout"
            evaluation_cases = json.loads(stable_json(validation_cases))
        else:
            evaluation_scope = "development"
            evaluation_cases = spec.test_cases
        try:
            evaluation_sha256 = hashlib.sha256(
                stable_json(evaluation_cases).encode("utf-8")
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            return {"status": "invalid_contract", "skill_id": skill_id, "reason": str(exc)}
        try:
            before = self._benchmark_material(skill_id)
            if (hashlib.sha256(spec.tool_code.encode("utf-8")).hexdigest()
                    != before["tool_sha256"]):
                raise ValueError("skill code changed while loading the benchmark")
            skill_dir = self._skill_dir(skill_id)
            tests_file = (
                resolve_file_within(skill_dir, "tests/test_cases.json")
                if skill_dir is not None else None
            )
            if tests_file is None or spec.test_cases != json.loads(tests_file.read_bytes()):
                raise ValueError("test contract changed while loading the benchmark")
        except (OSError, ValueError) as exc:
            return {"status": "invalid_contract", "skill_id": skill_id, "reason": str(exc)}
        result = self._benchmark.benchmark(replace(spec, test_cases=evaluation_cases))
        try:
            after = self._benchmark_material(skill_id)
            after_evaluation_sha256 = hashlib.sha256(
                stable_json(evaluation_cases).encode("utf-8")
            ).hexdigest()
            if before != after or evaluation_sha256 != after_evaluation_sha256:
                return {
                    "status": "stale_inputs", "skill_id": skill_id,
                    "reason": "skill code, tests, or evaluator changed during the run",
                }
            evidence_id = self._persist_benchmark_evidence(
                skill_id, before, result,
                evaluation_scope=evaluation_scope,
                evaluation_sha256=evaluation_sha256,
            )
            self._update_metrics(skill_id, result, evidence_id)
        except (OSError, ValueError) as exc:
            return {
                "status": "evidence_unavailable", "skill_id": skill_id,
                "reason": str(exc),
            }

        return {
            "status": "benchmarked",
            "skill_id": skill_id,
            "evidence_id": evidence_id,
            "evaluation_scope": evaluation_scope,
            "fitness": result.fitness_score,
            "passed": result.passed,
            "failed": result.failed,
            "duration_ms": result.duration_ms,
            "errors": result.errors[:5],
        }

    def promote_or_prune(self, skill_id: str) -> dict[str, Any]:
        """Evaluate a skill for promotion or pruning."""
        if not self._valid_skill_id(skill_id):
            return {"status": "invalid_skill_id", "skill_id": skill_id}
        spec = self._load_skill(skill_id)
        if not spec:
            return {"status": "not_found"}

        evidence, reason = self._verified_benchmark_evidence(skill_id, spec.metrics)
        fitness = evidence["fitness_score"] if evidence is not None else 0.0
        lower, upper = (
            _wilson_bounds(
                evidence["passed"], evidence["passed"] + evidence["failed"],
                self.DECISION_CONFIDENCE,
            ) if evidence is not None else (0.0, 1.0)
        )

        if evidence is None or evidence.get("evaluation_scope") != "caller_holdout":
            action = "kept"
            spec.status = "testing"
            if evidence is not None:
                reason = "development-only benchmark cannot authorize promotion"
            decision_reason = reason
        elif lower >= self.PROMOTION_THRESHOLD:
            action = "promoted"
            spec.status = "promoted"
            decision_reason = "held-out pass-rate lower bound clears promotion threshold"
        elif upper <= self.PRUNE_THRESHOLD:
            action = "pruned"
            spec.status = "pruned"
            decision_reason = "held-out pass-rate upper bound is below prune threshold"
        else:
            action = "kept"
            spec.status = "testing"
            decision_reason = "held-out evidence is inconclusive at the configured confidence"

        # Update skill status
        skill_dir = self._skill_dir(skill_id)
        if skill_dir is None:
            return {"status": "not_found", "skill_id": skill_id}
        skill_md = resolve_file_within(skill_dir, "SKILL.md")
        if skill_md is not None:
            content = skill_md.read_text(encoding="utf-8")
            import re
            content = re.sub(r"status: \w+", f"status: {spec.status}", content)
            skill_md.write_text(content, encoding="utf-8")

        self._update_registry(spec, action)

        logger.info(f"SkillEngine: {action} skill {skill_id} (fitness={fitness:.2f})")
        return {
            "status": action,
            "skill_id": skill_id,
            "fitness": fitness,
            "fitness_lower_bound": lower,
            "fitness_upper_bound": upper,
            "new_status": spec.status,
            "evidence_status": reason,
            "decision_reason": decision_reason,
        }

    def load_promoted_skill(self, skill_id: str) -> SkillSpec | None:
        """Load only code whose current bytes match promotion evidence."""
        spec = self._load_skill(skill_id)
        if spec is None or spec.status != "promoted":
            return None
        evidence, _ = self._verified_benchmark_evidence(skill_id, spec.metrics)
        if (
            evidence is None
            or evidence.get("evaluation_scope") != "caller_holdout"
            or _wilson_bounds(
                evidence["passed"], evidence["passed"] + evidence["failed"],
                self.DECISION_CONFIDENCE,
            )[0] < self.PROMOTION_THRESHOLD
        ):
            return None
        skill_dir = self._skill_dir(skill_id)
        if skill_dir is None:
            return None
        tool = resolve_file_within(skill_dir, "tool.py")
        tests = resolve_file_within(skill_dir, "tests/test_cases.json")
        if tool is None or tests is None:
            return None
        try:
            if (spec.tool_code != tool.read_bytes().decode("utf-8")
                    or spec.test_cases != json.loads(tests.read_bytes())):
                return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return spec

    def validated_promoted_skill(self, skill_id: str) -> SkillSpec | None:
        """Compatibility name for the evidence-checked promoted snapshot."""
        return self.load_promoted_skill(skill_id)

    def list_skills(self) -> list[dict[str, Any]]:
        """List all skills in the registry."""
        self._vault.ensure_structure()
        skills_dir = self._vault.config.path / "evolution" / "skills"
        results = []

        for skill_dir in sorted(skills_dir.iterdir()) if skills_dir.exists() else []:
            safe_dir = self._skill_dir(skill_dir.name)
            if safe_dir is None:
                continue
            metrics_file = resolve_file_within(safe_dir, "metrics.json")
            skill_md = resolve_file_within(safe_dir, "SKILL.md")

            info = {"skill_id": safe_dir.name, "path": str(safe_dir)}
            if metrics_file is not None:
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                    info["metrics"] = metrics if isinstance(metrics, dict) else {}
                except Exception:
                    pass
            if skill_md is not None:
                try:
                    from .vault import _parse_frontmatter
                    content = skill_md.read_text(encoding="utf-8")
                    fm = _parse_frontmatter(content)
                    if fm:
                        info.update(fm)
                except Exception:
                    pass
            if info.get("status") == "promoted":
                metrics = info.get("metrics")
                evidence, reason = self._verified_benchmark_evidence(
                    safe_dir.name, metrics if isinstance(metrics, dict) else {}
                )
                if (
                    evidence is None
                    or evidence.get("evaluation_scope") != "caller_holdout"
                    or _wilson_bounds(
                        evidence["passed"], evidence["passed"] + evidence["failed"],
                        self.DECISION_CONFIDENCE,
                    )[0] < self.PROMOTION_THRESHOLD
                ):
                    info["status"] = "testing"
                    info["promotion_integrity"] = (
                        reason if evidence is None else "latest benchmark is not promotable"
                    )
            results.append(info)

        return results

    # ── Private ──────────────────────────────────

    def _load_skill(self, skill_id: str) -> SkillSpec | None:
        """Load a skill spec from the vault."""
        skill_dir = self._skill_dir(skill_id)
        if skill_dir is None:
            return None

        spec = SkillSpec(skill_id=skill_id)

        tool_file = skill_dir / "tool.py"
        tool_file = resolve_file_within(skill_dir, tool_file)
        if tool_file is not None:
            spec.tool_code = tool_file.read_bytes().decode("utf-8")

        tests_file = resolve_file_within(skill_dir, "tests/test_cases.json")
        if tests_file is not None:
            try:
                spec.test_cases = json.loads(tests_file.read_bytes())
            except Exception:
                pass

        metrics_file = resolve_file_within(skill_dir, "metrics.json")
        if metrics_file is not None:
            try:
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                spec.metrics = metrics if isinstance(metrics, dict) else {}
            except Exception:
                pass

        skill_md = resolve_file_within(skill_dir, "SKILL.md")
        if skill_md is not None:
            try:
                from .vault import _parse_frontmatter
                content = skill_md.read_text(encoding="utf-8")
                fm = _parse_frontmatter(content)
                if fm:
                    spec.name = fm.get("name", "")
                    spec.entity = fm.get("entity", "")
                    spec.status = fm.get("status", "draft")
            except Exception:
                pass

        return spec

    def _update_metrics(
        self, skill_id: str, result: BenchmarkResult, evidence_id: str
    ) -> None:
        skill_dir = self._skill_dir(skill_id)
        if skill_dir is None:
            return
        metrics_file = resolve_output_within(skill_dir, "metrics.json")
        if metrics_file is None:
            raise ValueError("skill metrics path is unsafe")
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}
        else:
            data = {}

        data["fitness_score"] = result.fitness_score
        data["benchmark_contract_version"] = 1
        data["benchmark_evidence_id"] = evidence_id
        data["benchmark_runs"] = data.get("benchmark_runs", 0) + 1
        data["runs"] = data.get("runs", 0) + 1
        data["successes"] = data.get("successes", 0) + result.passed
        data["failures"] = data.get("failures", 0) + result.failed
        data["last_benchmark"] = datetime.now(timezone.utc).isoformat()
        data["last_duration_ms"] = result.duration_ms

        metrics_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _valid_skill_id(skill_id: str) -> bool:
        return bool(_SKILL_ID_RE.fullmatch(skill_id))

    def _skill_dir(self, skill_id: str) -> Path | None:
        if not self._valid_skill_id(skill_id):
            return None
        skills_dir = self._vault.config.path / "evolution" / "skills"
        return resolve_dir_within(skills_dir, skill_id)

    def _update_registry(self, spec: SkillSpec, action: str) -> None:
        """Update the registry.md index."""
        registry = self._vault.config.path / "evolution" / "registry.md"
        if not registry.exists():
            self._vault.ensure_structure()

        content = registry.read_text(encoding="utf-8")
        entry = f"| {spec.skill_id} | {action} | {spec.created_at[:10]} | {spec.description[:50]} |"

        # Check if already in registry
        if spec.skill_id in content:
            # Update existing line
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if spec.skill_id in line:
                    lines[i] = entry
                    break
            content = "\n".join(lines)
        else:
            content = content.rstrip() + "\n" + entry + "\n"

        registry.write_text(content, encoding="utf-8")
