---
claim_id: a7bdbc71-1450-4339-9077-617366120d81
entity: skill_engine
status: inferred
confidence: 0.75
sources:
  - entroly\skill_engine.py:36
  - entroly\skill_engine.py:52
  - entroly\skill_engine.py:71
  - entroly\skill_engine.py:199
  - entroly\skill_engine.py:558
  - entroly\skill_engine.py:616
  - entroly\skill_engine.py:652
  - entroly\skill_engine.py:62
  - entroly\skill_engine.py:74
  - entroly\skill_engine.py:212
last_checked: 2026-04-23T03:07:07.821642+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: skill_engine

**Language:** python
**Lines of code:** 907

## Types
- `class SkillSpec()` — A skill specification.
- `class BenchmarkResult()` — Result of running a skill benchmark.
- `class SkillSynthesizer()` — Generates skill specs from gap reports and failure patterns.
- `class StructuralSynthesizer()` — Zero-token skill synthesis via entropy-gradient structural analysis. Uses the Rust SAST/entropy engine to analyze code structure and generate tools that navigate the information topology of a codebase
- `class SandboxedRunner()` — Runs skill tools in isolation.
- `class SkillBenchmark()` — Evaluates skill fitness by running test cases.
- `class SkillEngine()` — Full skill lifecycle manager. Creates skills from gap reports, benchmarks them, promotes or prunes, and maintains the registry.

## Functions
- `def success_rate(self) -> float`
- `def synthesize_from_gap(
        self,
        entity_key: str,
        failing_queries: list[str],
        intent: str = "",
    ) -> SkillSpec`
- `def __init__(self, rust_engine: Any = None)` — Args: rust_engine: Optional entroly_core.EntrolyEngine instance. If None, falls back to pure-Python heuristics.
- `def synthesize_structural(
        self,
        entity_key: str,
        source_files: list[str],
        failing_queries: list[str],
    ) -> SkillSpec | None`
- `def matches(query: str) -> bool` — Check if this skill should handle the query.
- `def execute(query: str, context: dict) -> dict` — Navigate the entropy gradient around '{entity}'. Returns the most informative code fragments, ranked by information density. All data comes from local file I/O — no API calls, no token cost.
- `def __init__(self, timeout_seconds: float = 10.0)`
- `def run_tool(self, tool_code: str, query: str) -> dict[str, Any]` — Execute a skill tool in a subprocess sandbox.
- `def __init__(self, runner: SandboxedRunner | None = None)`
- `def benchmark(self, skill: SkillSpec) -> BenchmarkResult` — Run all test cases for a skill and compute fitness.
- `def __init__(self, vault: VaultManager)`
- `def create_skill(
        self,
        entity_key: str,
        failing_queries: list[str],
        intent: str = "",
    ) -> dict[str, Any]`
- `def benchmark_skill(self, skill_id: str) -> dict[str, Any]` — Benchmark a skill and update its metrics.
- `def promote_or_prune(self, skill_id: str) -> dict[str, Any]` — Evaluate a skill for promotion or pruning.
- `def list_skills(self) -> list[dict[str, Any]]` — List all skills in the registry.

## Dependencies
- `.vault`
- `__future__`
- `dataclasses`
- `datetime`
- `json`
- `logging`
- `os`
- `re`
- `subprocess`
- `time`
- `typing`
- `uuid`
