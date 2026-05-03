"""
Context Scaffolding Engine (CSE) — Structural Dependency Preamble Generator.

Generates a compact structural map (~150-300 tokens) that describes the
dependency relationships between selected code fragments. This preamble
is injected BEFORE the code fragments in the LLM context, giving the model
a "cognitive scaffold" that pre-connects the dots between files.

Research foundation:
  - GRACG (NeurIPS 2025): heterogeneous code graph → retrieval
  - Scaffold Reasoning (arxiv 2025): structured reasoning streams
  - OCD (arxiv 2026): minimal sufficient context via delta debugging
  - S2LPP (arxiv 2025): prompt preference transfers across model sizes
  - Structure-Grounded Knowledge Retrieval (arxiv 2025): dependency-aware context
  - SAC (arxiv 2025): anchor-token selection for semantic compression

Key insight: small models (Haiku) fail not because they lack intelligence,
but because they can't infer cross-file relationships from raw code alone.
The scaffold provides those relationships explicitly at ~200 tokens — less
than a single support file that would otherwise be included "just in case."

Token economics:
  +150-300 tokens for the scaffold
  −2000-6000 tokens from dropping redundant "safety" files
  = net savings of 1700-5700 tokens per request

Performance: pure Python dict traversal, <1ms, zero I/O.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger("entroly.scaffold")

# ═══════════════════════════════════════════════════════════════════════
# Import Pattern Extraction — Language-Polymorphic
# ═══════════════════════════════════════════════════════════════════════
#
# Each language has a different import syntax. We extract the imported
# symbols AND the module path so we can resolve cross-file edges.
# This is intentionally fast (~100μs per file) using string ops, not regex.

# Compiled patterns for speed (compiled once at module load)
_PY_FROM_IMPORT = re.compile(r"^from\s+([\w.]+)\s+import\s+(.+)", re.MULTILINE)
_PY_IMPORT = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
_JS_IMPORT_FROM = re.compile(r"import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]")
_RUST_USE = re.compile(r"^use\s+([\w:]+(?:::\{[^}]+\})?);", re.MULTILINE)
_GO_IMPORT = re.compile(r'"([\w/.]+)"')


def _normalize_source(source: str) -> str:
    """Normalize a fragment source path for matching.

    Strips 'file:' prefix and normalizes slashes.
    """
    s = source.removeprefix("file:").replace("\\", "/")
    return s


def _basename_stem(source: str) -> str:
    """Extract the filename stem from a source path.

    e.g. 'file:entroly/proxy.py' → 'proxy'
         'file:src/lib.rs' → 'lib'
    """
    norm = _normalize_source(source)
    base = norm.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def _extract_imports_from_content(content: str, source: str) -> list[str]:
    """Extract imported module/symbol names from source code.

    Returns a list of import target names (module basenames, symbols).
    Language is auto-detected from the source file extension.

    This is the Python-side equivalent of depgraph.rs:extract_import_targets(),
    operating on the already-selected fragments without needing Rust FFI.
    """
    ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
    targets: list[str] = []

    if ext in ("py", "pyi", "pyw"):
        # Python: from module import X, Y  /  import module
        for m in _PY_FROM_IMPORT.finditer(content):
            module_path = m.group(1)
            # The module itself is a dependency
            targets.append(module_path.rsplit(".", 1)[-1])
            # Individual imported names
            for name in m.group(2).split(","):
                clean = name.strip().split(" as ")[0].strip().strip("()")
                if clean and clean != "*":
                    targets.append(clean)
        for m in _PY_IMPORT.finditer(content):
            targets.append(m.group(1).rsplit(".", 1)[-1])

    elif ext in ("js", "jsx", "ts", "tsx", "mjs", "mts", "cjs", "cts"):
        # JS/TS: import { X, Y } from 'module'  /  import X from 'module'
        for m in _JS_IMPORT_FROM.finditer(content):
            named = m.group(1)
            default = m.group(2)
            module_path = m.group(3)
            # Module basename as dependency
            mod_name = module_path.rsplit("/", 1)[-1].split(".")[0]
            if mod_name and mod_name not in (".", ".."):
                targets.append(mod_name)
            if named:
                for name in named.split(","):
                    clean = name.strip().split(" as ")[0].strip()
                    if clean:
                        targets.append(clean)
            if default:
                targets.append(default)

    elif ext == "rs":
        # Rust: use crate::module::Symbol;
        for m in _RUST_USE.finditer(content):
            path = m.group(1)
            if "::{" in path:
                base, brace = path.split("::{", 1)
                for name in brace.rstrip("}").split(","):
                    clean = name.strip().split(" as ")[0].strip()
                    if clean:
                        targets.append(clean)
            else:
                targets.append(path.rsplit("::", 1)[-1])

    elif ext == "go":
        # Go: import "pkg/path"
        for m in _GO_IMPORT.finditer(content):
            targets.append(m.group(1).rsplit("/", 1)[-1])

    elif ext in ("java", "kt", "scala"):
        # Java/Kotlin: import com.example.ClassName;
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") and "." in stripped:
                path = stripped.removeprefix("import ").removesuffix(";").strip()
                path = path.removeprefix("static ").strip()
                targets.append(path.rsplit(".", 1)[-1])

    return targets


def _extract_definitions_from_content(content: str, source: str) -> list[str]:
    """Extract defined symbol names (functions, classes, structs) from source code.

    Returns names that this fragment PROVIDES to other fragments.
    """
    ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
    defs: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()

        if ext in ("py", "pyi", "pyw"):
            if stripped.startswith("def ") or stripped.startswith("async def "):
                name = stripped.split("(", 1)[0].split()[-1]
                if name:
                    defs.append(name)
            elif stripped.startswith("class ") and (":" in stripped or "(" in stripped):
                name = stripped.split("(", 1)[0].split(":", 1)[0].split()[-1]
                if name:
                    defs.append(name)

        elif ext in ("js", "jsx", "ts", "tsx", "mjs", "mts"):
            if stripped.startswith(("function ", "async function ", "export function ",
                                    "export async function ", "export default function ")):
                parts = stripped.split("(", 1)[0].split()
                name = parts[-1] if parts else ""
                if name and name not in ("function", "async", "export", "default"):
                    defs.append(name)
            elif stripped.startswith(("class ", "export class ", "export default class ")):
                parts = stripped.split("{", 1)[0].split("extends", 1)[0].split()
                name = [p for p in parts if p not in ("class", "export", "default")]
                if name:
                    defs.append(name[0])

        elif ext == "rs":
            for prefix in ("pub fn ", "fn ", "pub async fn ", "async fn ",
                           "pub struct ", "struct ", "pub enum ", "enum ",
                           "pub trait ", "trait "):
                if stripped.startswith(prefix):
                    rest = stripped[len(prefix):]
                    name = rest.split("(", 1)[0].split("<", 1)[0].split("{", 1)[0].split(":")[0].strip()
                    if name:
                        defs.append(name)
                    break

        elif ext == "go":
            if stripped.startswith("func "):
                # func (r *Receiver) Name(  or  func Name(
                rest = stripped[5:]
                if rest.startswith("("):
                    # Method: skip receiver
                    close = rest.find(")")
                    if close >= 0:
                        rest = rest[close + 1:].strip()
                name = rest.split("(", 1)[0].strip()
                if name:
                    defs.append(name)
            elif stripped.startswith("type "):
                name = stripped.split()[1] if len(stripped.split()) > 1 else ""
                if name:
                    defs.append(name)

    return defs


# ═══════════════════════════════════════════════════════════════════════
# Task-Aware Scaffold Strategy
# ═══════════════════════════════════════════════════════════════════════
#
# Different tasks benefit from different scaffold emphasis.
# Based on S2LPP (2025): task classification is transferable across
# model sizes — if a strategy helps Opus, it helps Haiku more.

_TASK_SCAFFOLD_STRATEGY = {
    "BugFix": {
        "focus": "error propagation",
        "show_reverse_deps": True,     # Show what DEPENDS ON the buggy file
        "show_test_mapping": True,     # Show which tests cover it
        "max_depth": 2,
    },
    "Feature": {
        "focus": "interface hierarchy",
        "show_reverse_deps": False,
        "show_test_mapping": False,
        "max_depth": 2,
    },
    "Refactor": {
        "focus": "full dependency cluster",
        "show_reverse_deps": True,
        "show_test_mapping": True,
        "max_depth": 3,
    },
    "Question": {
        "focus": "definitions and types",
        "show_reverse_deps": False,
        "show_test_mapping": False,
        "max_depth": 1,
    },
    "Test": {
        "focus": "test → implementation mapping",
        "show_reverse_deps": False,
        "show_test_mapping": True,
        "max_depth": 1,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Edge Types — Semantic Classification
# ═══════════════════════════════════════════════════════════════════════

def _classify_edge(
    importer_source: str,
    target_source: str,
    shared_symbols: list[str],
) -> str:
    """Classify the type of dependency between two fragments.

    Returns a human-readable edge label that helps the LLM understand
    the relationship: 'imports', 'calls', 'extends', 'tests', 'configures'.
    """
    imp_norm = _normalize_source(importer_source).lower()
    tgt_norm = _normalize_source(target_source).lower()

    # Test → implementation
    if any(p in imp_norm for p in ("/test", "test_", "_test.", ".test.", ".spec.")):
        return "tests"

    # Config → module
    tgt_ext = tgt_norm.rsplit(".", 1)[-1] if "." in tgt_norm else ""
    if tgt_ext in ("toml", "yaml", "yml", "json"):
        return "configures"

    # Check if symbols suggest inheritance/extension
    class_names = [s for s in shared_symbols if s[0:1].isupper() and not s.isupper()]
    if class_names:
        return f"uses {', '.join(class_names[:3])}"

    # Default: imports with specific symbols
    if shared_symbols:
        return f"imports ({', '.join(shared_symbols[:4])})"

    return "imports"


# ═══════════════════════════════════════════════════════════════════════
# Core Scaffold Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_scaffold(
    selected_fragments: list[dict[str, Any]],
    task_type: str = "Unknown",
    *,
    max_tokens: int = 300,
    min_fragments: int = 3,
) -> str:
    """Generate a structural dependency scaffold for selected fragments.

    This is the main entry point. Given the fragments selected by the
    knapsack optimizer, it:

    1. Extracts import/definition symbols from each fragment
    2. Resolves cross-fragment dependency edges
    3. Identifies the entry point (most depended-upon file)
    4. Renders a compact text preamble

    The scaffold is designed to be:
    - Compact: hard-capped at max_tokens (~300 tokens)
    - Informative: shows ONLY relationships between selected files
    - Additive: never removes or modifies fragments
    - Task-aware: emphasis changes based on task type

    Args:
        selected_fragments: List of fragment dicts from optimize_context().
            Each must have 'source' and 'content' keys.
        task_type: Classified task type from Rust classify_task().
        max_tokens: Hard cap on scaffold token count.
        min_fragments: Don't generate scaffold for fewer than this many files.

    Returns:
        A multi-line string scaffold preamble, or "" if scaffold is not beneficial.
    """
    if len(selected_fragments) < min_fragments:
        return ""

    # ── Phase 1: Build the fragment-local symbol table ──────────────
    # For each fragment, extract what it IMPORTS and what it DEFINES.
    # This mirrors depgraph.rs but operates on the already-selected subset.

    frag_imports: dict[str, list[str]] = {}   # source → [imported_names]
    frag_defines: dict[str, list[str]] = {}   # source → [defined_names]
    frag_content: dict[str, str] = {}         # source → content
    sources: list[str] = []

    for frag in selected_fragments:
        source = frag.get("source", "")
        content = frag.get("content", frag.get("preview", ""))
        if not source or not content:
            continue

        sources.append(source)
        frag_content[source] = content
        frag_imports[source] = _extract_imports_from_content(content, source)
        frag_defines[source] = _extract_definitions_from_content(content, source)

    if len(sources) < min_fragments:
        return ""

    # ── Phase 2: Resolve cross-fragment edges ──────────────────────
    # An edge (A → B) exists when A imports a symbol that B defines,
    # OR when A imports the module name that matches B's filename.
    #
    # This is O(N² × S) where N = selected files (~10-20) and
    # S = avg symbols per file (~20). Total: ~4000 ops → <1ms.

    # Build reverse lookup: symbol → source that defines it
    symbol_to_source: dict[str, str] = {}
    stem_to_source: dict[str, str] = {}
    for src in sources:
        stem = _basename_stem(src)
        stem_to_source[stem.lower()] = src
        for defn in frag_defines.get(src, []):
            symbol_to_source[defn] = src

    # Resolve edges
    edges: list[tuple[str, str, list[str]]] = []  # (from, to, shared_symbols)
    seen_edges: set[tuple[str, str]] = set()

    for src in sources:
        imports = frag_imports.get(src, [])
        resolved_targets: dict[str, list[str]] = defaultdict(list)  # target → symbols

        for imp in imports:
            # Direct symbol match
            if imp in symbol_to_source:
                target = symbol_to_source[imp]
                if target != src:
                    resolved_targets[target].append(imp)

            # Module name match (e.g., importing 'proxy' matches 'proxy.py')
            imp_lower = imp.lower()
            if imp_lower in stem_to_source:
                target = stem_to_source[imp_lower]
                if target != src:
                    resolved_targets[target].append(imp)

        for target, symbols in resolved_targets.items():
            edge_key = (src, target)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append((src, target, list(set(symbols))))

    # ── Phase 3: Compute structural metrics ────────────────────────

    # In-degree: how many files import this file (centrality proxy)
    in_degree: dict[str, int] = defaultdict(int)
    out_degree: dict[str, int] = defaultdict(int)
    for src, tgt, _ in edges:
        in_degree[tgt] += 1
        out_degree[src] += 1

    # Entry point: file with highest in-degree (most depended upon)
    entry_point = max(sources, key=lambda s: in_degree.get(s, 0)) if sources else ""

    # Detect test files
    test_files = [
        s for s in sources
        if any(p in _normalize_source(s).lower()
               for p in ("/test", "test_", "_test.", ".test.", ".spec."))
    ]

    # ── Phase 4: Task-aware rendering ──────────────────────────────

    strategy = _TASK_SCAFFOLD_STRATEGY.get(task_type, _TASK_SCAFFOLD_STRATEGY["Feature"])

    lines: list[str] = []
    lines.append("## Context Map (auto-generated by entroly)")
    lines.append("")

    # Task hint
    if task_type != "Unknown":
        lines.append(f"**Task type**: {task_type} — focus on {strategy['focus']}")
        lines.append("")

    # Entry point
    if entry_point and in_degree.get(entry_point, 0) > 0:
        ep_short = _normalize_source(entry_point)
        n_deps = in_degree[entry_point]
        defs = frag_defines.get(entry_point, [])
        key_defs = ", ".join(defs[:5]) if defs else "—"
        lines.append(f"**Entry point**: `{ep_short}` ({n_deps} dependents) — defines: {key_defs}")
        lines.append("")

    # Dependency edges — the core of the scaffold
    if edges:
        lines.append("**Dependencies**:")
        # Sort: edges FROM the entry point first, then by in-degree of target
        edges.sort(key=lambda e: (
            e[0] != entry_point,
            -in_degree.get(e[1], 0),
        ))
        for src, tgt, symbols in edges:
            src_short = _normalize_source(src)
            tgt_short = _normalize_source(tgt)
            label = _classify_edge(src, tgt, symbols)
            lines.append(f"- `{src_short}` → `{tgt_short}` ({label})")

            # Enforce token budget: ~4 chars per token, check periodically
            current_chars = sum(len(line) for line in lines)
            if current_chars > max_tokens * 4:
                remaining = len(edges) - edges.index((src, tgt, symbols)) - 1
                if remaining > 0:
                    lines.append(f"- ... and {remaining} more edges")
                break
        lines.append("")

    # Test mapping (task-dependent)
    if strategy.get("show_test_mapping") and test_files:
        lines.append("**Test coverage**:")
        for tf in test_files[:3]:
            tf_short = _normalize_source(tf)
            # Find what this test imports
            test_targets = []
            for src, tgt, syms in edges:
                if src == tf:
                    test_targets.append(_normalize_source(tgt))
            if test_targets:
                lines.append(f"- `{tf_short}` tests: {', '.join(f'`{t}`' for t in test_targets[:3])}")
            else:
                lines.append(f"- `{tf_short}`")
        lines.append("")

    # Isolated files warning (no edges = model can't infer purpose)
    isolated = [s for s in sources if s not in in_degree and s not in out_degree]
    if isolated and len(isolated) < len(sources):
        iso_short = [_normalize_source(s) for s in isolated[:3]]
        lines.append(f"**Standalone**: {', '.join(f'`{s}`' for s in iso_short)}")
        lines.append("")

    scaffold = "\n".join(lines).rstrip()

    # Final token budget enforcement
    estimated_tokens = len(scaffold) // 4
    if estimated_tokens > max_tokens:
        # Truncate to budget — cut from the bottom (less important edges)
        while estimated_tokens > max_tokens and len(lines) > 3:
            lines.pop(-2)  # Remove second-to-last (keep trailing "")
            scaffold = "\n".join(lines).rstrip()
            estimated_tokens = len(scaffold) // 4

    if not edges and not test_files:
        # No structural information to add — scaffold is not beneficial
        return ""

    logger.debug(
        "Scaffold: %d edges, entry=%s, ~%d tokens",
        len(edges),
        _basename_stem(entry_point) if entry_point else "none",
        estimated_tokens,
    )

    return scaffold
