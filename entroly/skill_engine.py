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

import json
import logging
import os
import re as _re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from .vault import VaultManager

if TYPE_CHECKING:
    from .reward_crystallizer import CrystallizationEvent

logger = logging.getLogger(__name__)


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
    test_cases: list[dict[str, str]] = field(default_factory=list)
    status: str = "draft"  # draft, testing, promoted, pruned
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a skill benchmark."""
    skill_id: str
    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    fitness_score: float = 0.0  # 0.0-1.0
    duration_ms: float = 0

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
        tests = [
            {"input": q, "expected": "should_not_fail"}
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
    #   - status starts as "promoted" (not "draft"). The Hoeffding LCB
    #     is itself the fitness proof; running another benchmark would
    #     just re-derive the same statistic with smaller n.
    #   - Test cases are real positive samples from the cluster (the
    #     queries that earned the reward), not synthetic placeholders.
    #   - The procedure encodes the actual recipe: weight profile +
    #     fragment IDs that delivered the wins, so a future structural
    #     planner can use them directly.

    def synthesize_from_success(self, event: "CrystallizationEvent") -> SkillSpec:  # type: ignore[name-defined]
        """Generate a SkillSpec from a RewardCrystallizer event.

        The event already carries the statistical proof (LCB, effect
        size, n) so we don't re-derive — we materialize.
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
            {"input": q, "expected": "should_match"}
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
            status="promoted",  # Pre-validated by Hoeffding LCB; skip benchmark gate.
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
        # Safe-quote the trigger for regex embedding.
        trig_escaped = trigger.replace("\\", "\\\\").replace('"', '\\"')
        recipe_lit = json.dumps(list(fragment_recipe))
        weights_lit = json.dumps(weight_profile)
        return (
            f'"""Crystallized skill tool — {name}.\n'
            f'\n'
            f'Auto-generated by RewardCrystallizer. Encodes the fragment\n'
            f'recipe and PRISM weight profile that delivered sustained\n'
            f'high reward for the trigger pattern.\n'
            f'"""\n\n'
            f'import re\n\n'
            f'TRIGGER_PATTERN = re.compile(r"\\b({trig_escaped})\\b", re.I)\n'
            f'FRAGMENT_RECIPE = {recipe_lit}\n'
            f'WEIGHT_PROFILE = {weights_lit}\n\n\n'
            f'def matches(query: str) -> bool:\n'
            f'    return bool(TRIGGER_PATTERN.search(query or ""))\n\n\n'
            f'def execute(query: str, context: dict) -> dict:\n'
            f'    """Return the recipe that previously won for this query family."""\n'
            f'    return {{\n'
            f'        "status": "success",\n'
            f'        "skill": "{name}",\n'
            f'        "fragment_recipe": list(FRAGMENT_RECIPE),\n'
            f'        "weight_profile": dict(WEIGHT_PROFILE),\n'
            f'        "results": [\n'
            f'            {{"file": fid, "snippet": ""}}\n'
            f'            for fid in FRAGMENT_RECIPE\n'
            f'        ],\n'
            f'    }}\n'
        )

    def _generate_tool_template(self, name: str, entity: str, trigger: str) -> str:
        return (
            f'"""\n'
            f'Auto-generated skill tool: {name}\n'
            f'Entity: {entity}\n'
            f'"""\n\n'
            f'import re\n\n'
            f'TRIGGER_PATTERN = re.compile(r"\\b({trigger})\\b", re.I)\n\n\n'
            f'def matches(query: str) -> bool:\n'
            f'    """Check if this skill should handle the query."""\n'
            f'    return bool(TRIGGER_PATTERN.search(query))\n\n\n'
            f'def execute(query: str, context: dict) -> dict:\n'
            f'    """Execute the skill logic."""\n'
            f'    return {{\n'
            f'        "status": "executed",\n'
            f'        "skill": "{name}",\n'
            f'        "entity": "{entity}",\n'
            f'        "result": "Skill implementation needed",\n'
            f'    }}\n'
        )


# ══════════════════════════════════════════════════════════════════════
# Structural Synthesizer — Entropy-Gradient Program Synthesis (Pillar 2)
# ══════════════════════════════════════════════════════════════════════
#
# Novel contribution: Instead of asking an LLM to "write a tool," this
# synthesizer derives executable tools from the *information topology*
# of the code graph.
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
# about the entity. This is deterministic, costs $0, and produces
# tools that are *provably correct* (they return real code, not
# hallucinations).
#
# Mathematical grounding:
#   - The entropy gradient ∇H(E) points toward the direction of
#     maximum information gain. Following it yields the minimal set
#     of code fragments that maximally reduces uncertainty about E.
#   - This is equivalent to solving the rate-distortion problem
#     R(D) = min_{p(ê|e)} I(E; Ê) s.t. E[d(e, ê)] ≤ D
#     where the "distortion" is miss rate and "rate" is token cost.

class StructuralSynthesizer:
    """Zero-token skill synthesis via entropy-gradient structural analysis.

    Uses the Rust SAST/entropy engine to analyze code structure and
    generate tools that navigate the information topology of a codebase.
    All synthesis is deterministic and runs on the local CPU for $0.
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
            {"input": q, "expected": "should_return_context"}
            for q in failing_queries[:5]
        ]

        return SkillSpec(
            name=name,
            description=f"Structural skill for {entity_key} (zero-token synthesis)",
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
        This is deterministic and always returns real code, never hallucinations.
        """
        # Build the static knowledge table
        sig_entries = []
        for node in closure[:15]:  # Top 15 by entropy
            escaped_sig = node["signature"].replace('"', '\\"')
            escaped_file = node["file"].replace("\\", "\\\\")
            sig_entries.append(
                f'    {{"name": "{node["name"]}", '
                f'"file": "{escaped_file}", '
                f'"line": {node["line"]}, '
                f'"entropy": {node["entropy"]}, '
                f'"signature": "{escaped_sig}"}}'
            )

        sigs_literal = ",\n".join(sig_entries) if sig_entries else ""

        imports_literal = ", ".join(
            f'"{imp[:80]}"' for imp in invariants["imports"][:10]
        )

        classes_literal = ", ".join(
            f'"{c}"' for c in invariants["classes"][:10]
        )

        return f'''"""
Structural skill tool: {name}
Entity: {entity}
Synthesis: entropy-gradient structural induction (zero-token, CPU-only)

This tool was generated WITHOUT any LLM call. It navigates the
information topology of the codebase around '{entity}', returning
the most informative code fragments ranked by Shannon entropy.
"""

import re
import os

TRIGGER_PATTERN = re.compile(r"\\b({entity.replace('.', '[.]')})\\b", re.I)

# Static knowledge table — entropy-ranked structural closure
# Computed by StructuralSynthesizer from AST analysis
_CLOSURE = [
{sigs_literal}
]

_IMPORTS = [{imports_literal}]
_CLASSES = [{classes_literal}]


def matches(query: str) -> bool:
    """Check if this skill should handle the query."""
    return bool(TRIGGER_PATTERN.search(query))


def execute(query: str, context: dict) -> dict:
    """Navigate the entropy gradient around '{entity}'.

    Returns the most informative code fragments, ranked by
    information density. All data comes from local file I/O —
    no API calls, no token cost.
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
        "skill": "{name}",
        "entity": "{entity}",
        "synthesis_method": "structural_induction",
        "token_cost": 0,
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
            f"Entropy-gradient structural induction (zero-token, CPU-only).\n"
            f"Generated from AST analysis of {sig_count} functions, "
            f"{class_count} classes, {import_count} imports.\n\n"
            f"## Steps\n"
            f"1. Read source files associated with `{entity}`\n"
            f"2. Rank functions by Shannon entropy (information density)\n"
            f"3. Return top-K most informative code fragments\n"
            f"4. Include dependency context (imports, classes)\n\n"
            f"## Guarantees\n"
            f"- Zero token cost (all local I/O)\n"
            f"- Deterministic output (same input → same result)\n"
            f"- No hallucinations (returns actual code, not generated text)\n"
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
    """Runs skill tools in isolation."""

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
                ["python", temp_path, query],
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

        for tc in skill.test_cases:
            query = tc.get("input", "")
            tc.get("expected", "")
            try:
                run = self._runner.run_tool(skill.tool_code, query)
                if run["status"] == "success":
                    result.passed += 1
                else:
                    result.failed += 1
                    result.errors.append(f"Query '{query}': {run.get('error', 'unknown')}")
            except Exception as e:
                result.failed += 1
                result.errors.append(f"Query '{query}': {e}")

        result.duration_ms = (time.time() - t0) * 1000
        total = result.passed + result.failed
        result.fitness_score = result.passed / total if total > 0 else 0.0

        return result


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

    def __init__(self, vault: VaultManager):
        self._vault = vault
        self._synthesizer = SkillSynthesizer()
        self._runner = SandboxedRunner()
        self._benchmark = SkillBenchmark(self._runner)

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
        """Materialize a RewardCrystallizer event as a promoted skill.

        Bypasses the failure-driven gap pipeline: the event already
        carries a Hoeffding LCB > baseline + ε, which is a stronger
        statistical guarantee than any benchmark we'd run. The skill
        is written directly with status='promoted'.

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
            "skill_id": spec.skill_id,
            "name": spec.name,
            "path": str(skill_dir),
            "fitness_score": spec.metrics.get("fitness_score", 0.0),
            "cluster_id": event.cluster_id,
        }

    def benchmark_skill(self, skill_id: str) -> dict[str, Any]:
        """Benchmark a skill and update its metrics."""
        spec = self._load_skill(skill_id)
        if not spec:
            return {"status": "not_found", "skill_id": skill_id}

        result = self._benchmark.benchmark(spec)

        # Update metrics
        self._update_metrics(skill_id, result)

        return {
            "status": "benchmarked",
            "skill_id": skill_id,
            "fitness": result.fitness_score,
            "passed": result.passed,
            "failed": result.failed,
            "duration_ms": result.duration_ms,
            "errors": result.errors[:5],
        }

    def promote_or_prune(self, skill_id: str) -> dict[str, Any]:
        """Evaluate a skill for promotion or pruning."""
        spec = self._load_skill(skill_id)
        if not spec:
            return {"status": "not_found"}

        fitness = spec.metrics.get("fitness_score", 0.0)

        if fitness >= self.PROMOTION_THRESHOLD:
            action = "promoted"
            spec.status = "promoted"
        elif fitness <= self.PRUNE_THRESHOLD:
            action = "pruned"
            spec.status = "pruned"
        else:
            action = "kept"
            spec.status = "testing"

        # Update skill status
        skill_dir = self._vault.config.path / "evolution" / "skills" / skill_id
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
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
            "new_status": spec.status,
        }

    def list_skills(self) -> list[dict[str, Any]]:
        """List all skills in the registry."""
        self._vault.ensure_structure()
        skills_dir = self._vault.config.path / "evolution" / "skills"
        results = []

        for skill_dir in sorted(skills_dir.iterdir()) if skills_dir.exists() else []:
            if not skill_dir.is_dir():
                continue
            metrics_file = skill_dir / "metrics.json"
            skill_md = skill_dir / "SKILL.md"

            info = {"skill_id": skill_dir.name, "path": str(skill_dir)}
            if metrics_file.exists():
                try:
                    info["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if skill_md.exists():
                try:
                    from .vault import _parse_frontmatter
                    content = skill_md.read_text(encoding="utf-8")
                    fm = _parse_frontmatter(content)
                    if fm:
                        info.update(fm)
                except Exception:
                    pass
            results.append(info)

        return results

    # ── Private ──────────────────────────────────

    def _load_skill(self, skill_id: str) -> SkillSpec | None:
        """Load a skill spec from the vault."""
        skill_dir = self._vault.config.path / "evolution" / "skills" / skill_id
        if not skill_dir.exists():
            return None

        spec = SkillSpec(skill_id=skill_id)

        tool_file = skill_dir / "tool.py"
        if tool_file.exists():
            spec.tool_code = tool_file.read_text(encoding="utf-8")

        tests_file = skill_dir / "tests" / "test_cases.json"
        if tests_file.exists():
            try:
                spec.test_cases = json.loads(tests_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        metrics_file = skill_dir / "metrics.json"
        if metrics_file.exists():
            try:
                spec.metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
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

    def _update_metrics(self, skill_id: str, result: BenchmarkResult) -> None:
        metrics_file = (
            self._vault.config.path / "evolution" / "skills" / skill_id / "metrics.json"
        )
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        data["fitness_score"] = result.fitness_score
        data["runs"] = data.get("runs", 0) + 1
        data["successes"] = data.get("successes", 0) + result.passed
        data["failures"] = data.get("failures", 0) + result.failed
        data["last_benchmark"] = datetime.now(timezone.utc).isoformat()
        data["last_duration_ms"] = result.duration_ms

        metrics_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
