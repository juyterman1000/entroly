"""
Canonical repo file map for Entroly.

Builds a grouped inventory of the three product repos plus top-level support files,
assigning each file a practical ownership role so product and engineering stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "target", "node_modules", ".tmp"}


@dataclass
class FileMapEntry:
    repo: str
    path: str
    role: str
    category: str


_ROOT_ROLE_MAP = {
    "README.md": ("product positioning and install surface", "root-doc"),
    "pyproject.toml": ("top-level Python packaging and workspace metadata", "root-meta"),
    "Dockerfile.entroly": ("container runtime packaging", "root-ops"),
    "docker-compose.yml": ("local multi-service runtime", "root-ops"),
    "CONTRIBUTING.md": ("contribution workflow", "root-doc"),
    "SECURITY.md": ("security policy", "root-doc"),
    "LICENSE": ("license", "root-doc"),
    "extractor.py": ("one-off code extraction utility", "root-tool"),
    "extractor_cogops.py": ("one-off CogOps extraction utility", "root-tool"),
    "super_extractor.py": ("one-off repo summarization utility", "root-tool"),
    "all_files_list.txt": ("generated repo inventory artifact", "root-artifact"),
    "arch_summary.txt": ("generated architecture summary artifact", "root-artifact"),
    "super_dump.txt": ("generated repository dump artifact", "root-artifact"),
    "build_errors.txt": ("generated build log artifact", "root-artifact"),
    "build_output.txt": ("generated build log artifact", "root-artifact"),
    "ruff_errors.txt": ("generated lint artifact", "root-artifact"),
    "job_logs.txt": ("generated job log artifact", "root-artifact"),
    "test_output.txt": ("generated test artifact", "root-artifact"),
    "test_output2.txt": ("generated test artifact", "root-artifact"),
    "tuning_config.json": ("shared tuning defaults", "root-config"),
    "test_cogops_smoke.py": ("Python CogOps integration smoke test", "root-test"),
    "test_rust_cogops.py": ("Rust CogOps integration smoke test", "root-test"),
    "test_auth.py": ("auth-focused regression test", "root-test"),
}

_PY_ROLE_MAP = {
    "server.py": ("primary Python MCP server and product shell", "python-runtime"),
    "cli.py": ("primary CLI and operator surface", "python-runtime"),
    "proxy.py": ("HTTP proxy prompt compiler runtime", "python-runtime"),
    "context_bridge.py": ("multi-agent context and orchestration surface", "python-runtime"),
    "belief_compiler.py": ("Truth to Belief compiler", "python-cogops"),
    "verification_engine.py": ("belief verification and confidence engine", "python-cogops"),
    "change_pipeline.py": ("change-driven PR and review pipeline", "python-cogops"),
    "flow_orchestrator.py": ("canonical flow executor", "python-cogops"),
    "epistemic_router.py": ("ingress routing policy engine", "python-cogops"),
    "vault.py": ("vault persistence and artifact schema", "python-cogops"),
    "skill_engine.py": ("dynamic skill synthesis and lifecycle", "python-cogops"),
    "evolution_logger.py": ("miss tracking and capability-gap logging", "python-cogops"),
    "change_listener.py": ("workspace change sync and listener glue", "python-cogops"),
    "repo_map.py": ("canonical repo inventory and ownership map", "python-cogops"),
    "auto_index.py": ("workspace discovery and raw ingest indexing", "python-support"),
    "prefetch.py": ("predictive dependency and file prefetch", "python-support"),
    "query_refiner.py": ("query shaping and refinement", "python-support"),
    "adaptive_pruner.py": ("conversation/context pruning policy", "python-support"),
    "checkpoint.py": ("checkpoint persistence and recovery", "python-support"),
    "proxy_transform.py": ("provider-specific context injection", "python-support"),
    "proxy_config.py": ("proxy quality and model budget config", "python-support"),
    "provenance.py": ("provenance graph and context trace builder", "python-support"),
    "autotune.py": ("parameter tuning and feedback journal", "python-support"),
    "value_tracker.py": ("cost and value accounting", "python-support"),
    "long_term_memory.py": ("cross-session memory adapter", "python-support"),
    "multimodal.py": ("image, diff, diagram, and voice ingestion", "python-support"),
    "dashboard.py": ("developer-facing runtime dashboard", "python-runtime"),
    "config.py": ("project configuration and paths", "python-support"),
    "benchmark_harness.py": ("benchmark execution harness", "python-support"),
    "entroly_mcp_client.py": ("example MCP client", "python-example"),
    "integrate_entroly_mcp.py": ("example MCP integration", "python-example"),
    "_docker_launcher.py": ("Docker launcher shim", "python-runtime"),
    "README.md": ("package-level README", "python-doc"),
    "pyproject.toml": ("package metadata", "python-meta"),
    "tuning_config.json": ("package tuning defaults", "python-config"),
    "__init__.py": ("package entry metadata", "python-meta"),
}

_RUST_ROLE_MAP = {
    "lib.rs": ("core Rust engine and PyO3 export surface", "rust-runtime"),
    "cogops.rs": ("Rust epistemic engine and CogOps data plane", "rust-cogops"),
    "fragment.rs": ("context fragment model and scoring helpers", "rust-core"),
    "knapsack.rs": ("budgeted context selection optimizer", "rust-core"),
    "knapsack_sds.rs": ("streaming/diverse selection and IOS logic", "rust-core"),
    "entropy.rs": ("information density scoring", "rust-core"),
    "dedup.rs": ("SimHash deduplication", "rust-core"),
    "semantic_dedup.rs": ("semantic deduplication refinement", "rust-core"),
    "depgraph.rs": ("dependency graph extraction", "rust-core"),
    "guardrails.rs": ("criticality, safety, and ordering policy", "rust-verification"),
    "lsh.rs": ("approximate recall index", "rust-core"),
    "prism.rs": ("reinforcement and spectral optimizer", "rust-learning"),
    "skeleton.rs": ("multi-language structure extraction", "rust-truth"),
    "sast.rs": ("static security analysis engine", "rust-verification"),
    "health.rs": ("codebase health analysis", "rust-verification"),
    "query.rs": ("query analysis and refinement heuristics", "rust-action"),
    "query_persona.rs": ("query manifold and archetype modeling", "rust-learning"),
    "hierarchical.rs": ("hierarchical compression", "rust-belief"),
    "anomaly.rs": ("anomaly detection", "rust-verification"),
    "utilization.rs": ("fragment utilization scoring", "rust-verification"),
    "conversation_pruner.rs": ("conversation compression runtime", "rust-action"),
    "channel.rs": ("channel-coding reward and contradiction logic", "rust-learning"),
    "nkbe.rs": ("multi-agent token budget allocator", "rust-action"),
    "cognitive_bus.rs": ("inter-agent event bus", "rust-action"),
    "cache.rs": ("EGSC cache and retrieval economics", "rust-core"),
    "resonance.rs": ("pairwise fragment resonance modeling", "rust-learning"),
    "causal.rs": ("causal context graph and intervention logic", "rust-belief"),
}

_WASM_JS_ROLE_MAP = {
    "src/lib.rs": ("WASM export surface for the Rust engine", "wasm-runtime"),
    "js/server.js": ("Node MCP server over WASM engine", "wasm-runtime"),
    "js/cli.js": ("Node CLI over WASM engine", "wasm-runtime"),
    "js/config.js": ("Node configuration wrapper", "wasm-support"),
    "js/auto_index.js": ("Node workspace indexing wrapper", "wasm-support"),
    "js/checkpoint.js": ("Node checkpoint wrapper", "wasm-support"),
    "js/autotune.js": ("Node autotune wrapper", "wasm-support"),
    "index.js": ("package export surface", "wasm-meta"),
    "package.json": ("npm package metadata", "wasm-meta"),
    "test_wasm_e2e.js": ("WASM end-to-end test suite", "wasm-test"),
    "test_autotune.js": ("WASM autotune test suite", "wasm-test"),
}


def build_repo_map(root: str | Path) -> dict[str, list[FileMapEntry]]:
    root_path = Path(root).resolve()
    grouped: dict[str, list[FileMapEntry]] = {
        "root": [],
        "python": [],
        "rust-core": [],
        "wasm": [],
        "tests": [],
        "other": [],
    }

    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_path).as_posix()
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        entry = _entry_for(rel)
        grouped[entry.repo].append(entry)

    return grouped


def render_repo_map_markdown(grouped: dict[str, list[FileMapEntry]]) -> str:
    lines = [
        "# Entroly Repo File Map",
        "",
        "Canonical ownership map across the Python product shell, Rust core, and WASM/JS surface.",
        "",
    ]
    for section in ["root", "python", "rust-core", "wasm", "tests", "other"]:
        entries = grouped.get(section, [])
        if not entries:
            continue
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Path | Role | Category |")
        lines.append("|---|---|---|")
        for entry in entries:
            lines.append(f"| `{entry.path}` | {entry.role} | `{entry.category}` |")
        lines.append("")
    return "\n".join(lines) + "\n"

def _entry_for(rel: str) -> FileMapEntry:
    path = Path(rel)
    parts = path.parts
    name = path.name

    if len(parts) == 1:
        role, category = _ROOT_ROLE_MAP.get(name, ("top-level support artifact", "root-support"))
        return FileMapEntry("root", rel, role, category)

    if parts[0] == "entroly":
        role, category = _PY_ROLE_MAP.get(name, ("Python package support file", "python-support"))
        return FileMapEntry("python", rel, role, category)

    if parts[0] == "entroly-core":
        if parts[1:2] == ("src",):
            role, category = _RUST_ROLE_MAP.get(name, ("Rust core support module", "rust-core"))
        elif parts[1:2] == ("tests",):
            role, category = ("Rust core integration test driver", "rust-test")
        else:
            role, category = ("Rust core package metadata", "rust-meta")
        return FileMapEntry("rust-core", rel, role, category)

    if parts[0] == "entroly-wasm":
        key = rel if rel in _WASM_JS_ROLE_MAP else f"{parts[1]}/{name}" if len(parts) > 1 else name
        role, category = _WASM_JS_ROLE_MAP.get(key, _RUST_ROLE_MAP.get(name, ("WASM package support file", "wasm-support")))
        return FileMapEntry("wasm", rel, role, category)

    if parts[0] == "tests":
        return FileMapEntry("tests", rel, "Python integration or functional test", "python-test")

    if parts[0] in {"docs", "examples", "bench", "benchmarks", "tuning_strategies"}:
        return FileMapEntry("other", rel, "documentation, example, or benchmark asset", "support")

    return FileMapEntry("other", rel, "support file", "support")
