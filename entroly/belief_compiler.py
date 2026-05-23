"""
Belief Compiler — Truth → Belief Pipeline
==========================================

Automatically compiles raw code/repository truth into machine-auditable
belief artifacts. This is the engine that makes the vault self-populating.

Pipeline:
  1. Code Claim Extraction:  AST parse → extract entities, invariants, deps
  2. Entity Resolution:      Link entities across files, resolve aliases
  3. Architecture Synthesis:  Build module maps, service topologies
  4. Diagram Generation:     Mermaid diagrams from dependency graphs
  5. Belief Writing:         Generate vault belief artifacts with frontmatter
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .vault import BeliefArtifact, VaultManager

logger = logging.getLogger(__name__)

# ── Try Rust-powered AST extraction ──────────────────────────────────
try:
    from entroly_core import extract_skeleton
    _HAS_SKELETON = True
except ImportError:
    _HAS_SKELETON = False
    def extract_skeleton(content: str, source: str) -> str:  # type: ignore
        return ""


# ══════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CodeEntity:
    """An entity extracted from source code."""
    name: str
    kind: str  # function, class, module, struct, trait, const, import
    file_path: str
    line: int = 0
    docstring: str = ""
    signature: str = ""
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        module = Path(self.file_path).stem
        return f"{module}::{self.name}"


@dataclass
class ModuleMap:
    """A module-level view of the codebase."""
    name: str
    file_path: str
    entities: list[CodeEntity] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    description: str = ""
    loc: int = 0
    language: str = ""


@dataclass
class CompilationResult:
    """Result of a belief compilation run."""
    beliefs_written: int = 0
    entities_extracted: int = 0
    modules_mapped: int = 0
    diagrams_generated: int = 0
    files_processed: int = 0
    errors: list[str] = field(default_factory=list)
    belief_ids: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Code Claim Extractor
# ══════════════════════════════════════════════════════════════════════

# ── Python extraction patterns ──
_PY_CLASS = re.compile(r'^class\s+(\w+)(?:\(([^)]*)\))?:', re.M)
_PY_FUNC = re.compile(r'^(?:    )?(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^\n:]+))?:', re.M)
_PY_IMPORT = re.compile(r'^(?:from\s+([\w.]+)\s+)?import\s+(.+)', re.M)
_PY_DOCSTRING = re.compile(r'"""(.*?)"""', re.S)
_PY_CONST = re.compile(r'^([A-Z][A-Z_0-9]+)\s*[:=]', re.M)

# ── Rust extraction patterns ──
_RS_STRUCT = re.compile(r'^pub\s+struct\s+(\w+)', re.M)
_RS_ENUM = re.compile(r'^pub\s+enum\s+(\w+)', re.M)
_RS_FN = re.compile(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^\n{]+))?', re.M)
_RS_TRAIT = re.compile(r'^pub\s+trait\s+(\w+)', re.M)
_RS_IMPL = re.compile(r'^impl(?:<[^>]*>)?\s+(\w+)', re.M)
_RS_USE = re.compile(r'^use\s+([\w:]+)', re.M)
_RS_MOD = re.compile(r'^(?:pub\s+)?mod\s+(\w+)', re.M)
_RS_CONST = re.compile(r'^(?:pub\s+)?(?:const|static)\s+(\w+)', re.M)
_RS_DOC = re.compile(r'///\s*(.*)')


def extract_entities(content: str, file_path: str) -> list[CodeEntity]:
    """Extract code entities from a source file."""
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return _extract_python_entities(content, file_path)
    elif ext == ".rs":
        return _extract_rust_entities(content, file_path)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        return _extract_js_entities(content, file_path)
    return []


def _extract_python_entities(content: str, file_path: str) -> list[CodeEntity]:
    entities: list[CodeEntity] = []
    lines = content.splitlines()

    # Classes
    for m in _PY_CLASS.finditer(content):
        line = content[:m.start()].count('\n') + 1
        doc = _get_next_docstring(lines, line)
        entities.append(CodeEntity(
            name=m.group(1), kind="class", file_path=file_path,
            line=line, docstring=doc,
            signature=f"class {m.group(1)}({m.group(2) or ''})",
            dependencies=[b.strip() for b in (m.group(2) or "").split(",") if b.strip()],
        ))

    # Functions (top-level and methods)
    for m in _PY_FUNC.finditer(content):
        line = content[:m.start()].count('\n') + 1
        doc = _get_next_docstring(lines, line)
        name = m.group(1)
        if name.startswith("_") and name != "__init__":
            continue  # skip private, keep __init__
        ret = (m.group(3) or "").strip()
        entities.append(CodeEntity(
            name=name, kind="function", file_path=file_path,
            line=line, docstring=doc,
            signature=f"def {name}({m.group(2)}) -> {ret}" if ret else f"def {name}({m.group(2)})",
        ))

    # Imports → dependencies
    imports = []
    for m in _PY_IMPORT.finditer(content):
        module = m.group(1) or ""
        names = m.group(2).strip()
        if module:
            imports.append(module)
        else:
            imports.append(names.split(",")[0].strip().split(" ")[0])

    # Constants
    for m in _PY_CONST.finditer(content):
        entities.append(CodeEntity(
            name=m.group(1), kind="const", file_path=file_path,
            line=content[:m.start()].count('\n') + 1,
        ))

    # Tag all entities with import dependencies
    for e in entities:
        e.dependencies.extend(imports)

    return entities


def _extract_rust_entities(content: str, file_path: str) -> list[CodeEntity]:
    entities: list[CodeEntity] = []
    lines = content.splitlines()

    for m in _RS_STRUCT.finditer(content):
        line = content[:m.start()].count('\n') + 1
        doc = _get_rust_doc(lines, line - 1)
        entities.append(CodeEntity(
            name=m.group(1), kind="struct", file_path=file_path,
            line=line, docstring=doc, signature=f"pub struct {m.group(1)}",
        ))

    for m in _RS_ENUM.finditer(content):
        line = content[:m.start()].count('\n') + 1
        doc = _get_rust_doc(lines, line - 1)
        entities.append(CodeEntity(
            name=m.group(1), kind="enum", file_path=file_path,
            line=line, docstring=doc, signature=f"pub enum {m.group(1)}",
        ))

    for m in _RS_TRAIT.finditer(content):
        line = content[:m.start()].count('\n') + 1
        doc = _get_rust_doc(lines, line - 1)
        entities.append(CodeEntity(
            name=m.group(1), kind="trait", file_path=file_path,
            line=line, docstring=doc, signature=f"pub trait {m.group(1)}",
        ))

    for m in _RS_FN.finditer(content):
        line = content[:m.start()].count('\n') + 1
        name = m.group(1)
        if name.startswith("_"):
            continue
        doc = _get_rust_doc(lines, line - 1)
        ret = (m.group(3) or "").strip()
        sig = f"fn {name}({m.group(2)})"
        if ret:
            sig += f" -> {ret}"
        entities.append(CodeEntity(
            name=name, kind="function", file_path=file_path,
            line=line, docstring=doc, signature=sig,
        ))

    # Use statements → dependencies
    uses = [m.group(1) for m in _RS_USE.finditer(content)]

    for e in entities:
        e.dependencies.extend(uses)

    return entities


def _extract_js_entities(content: str, file_path: str) -> list[CodeEntity]:
    entities: list[CodeEntity] = []
    # Class
    for m in re.finditer(r'(?:export\s+)?class\s+(\w+)', content):
        line = content[:m.start()].count('\n') + 1
        entities.append(CodeEntity(name=m.group(1), kind="class", file_path=file_path, line=line))
    # Functions
    for m in re.finditer(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', content):
        line = content[:m.start()].count('\n') + 1
        entities.append(CodeEntity(name=m.group(1), kind="function", file_path=file_path, line=line))
    # Arrow exports
    for m in re.finditer(r'export\s+(?:const|let)\s+(\w+)\s*=', content):
        line = content[:m.start()].count('\n') + 1
        entities.append(CodeEntity(name=m.group(1), kind="const", file_path=file_path, line=line))
    return entities


def _get_next_docstring(lines: list[str], after_line: int) -> str:
    """Get Python docstring after a def/class line."""
    for i in range(after_line, min(after_line + 3, len(lines))):
        stripped = lines[i].strip()
        if '"""' in stripped or "'''" in stripped:
            # Single-line docstring
            m = re.search(r'"""(.*?)"""', stripped) or re.search(r"'''(.*?)'''", stripped)
            if m:
                return m.group(1).strip()
            # Multi-line: collect until closing
            doc_lines = [stripped.replace('"""', '').replace("'''", '')]
            for j in range(i + 1, min(i + 20, len(lines))):
                if '"""' in lines[j] or "'''" in lines[j]:
                    doc_lines.append(lines[j].strip().replace('"""', '').replace("'''", ''))
                    break
                doc_lines.append(lines[j].strip())
            return " ".join(line for line in doc_lines if line)[:200]
    return ""


def _get_rust_doc(lines: list[str], before_line: int) -> str:
    """Get Rust /// doc comments above a line."""
    docs = []
    for i in range(before_line - 1, max(before_line - 10, -1), -1):
        if i < 0 or i >= len(lines):
            break
        stripped = lines[i].strip()
        if stripped.startswith("///") or stripped.startswith("//!"):
            docs.append(stripped[3:].strip())
        elif not stripped or stripped.startswith("#["):
            continue
        else:
            break
    docs.reverse()
    return " ".join(docs)[:200]


# ══════════════════════════════════════════════════════════════════════
# Entity Resolver
# ══════════════════════════════════════════════════════════════════════

class EntityResolver:
    """Resolves and links entities across files."""

    def __init__(self):
        self._entities: dict[str, CodeEntity] = {}  # qualified_name -> entity
        self._by_name: dict[str, list[CodeEntity]] = {}  # name -> entities
        self._dep_graph: dict[str, set[str]] = {}  # entity -> dependencies

    def add_entities(self, entities: list[CodeEntity]) -> None:
        for e in entities:
            qn = e.qualified_name
            self._entities[qn] = e
            self._by_name.setdefault(e.name, []).append(e)
            self._dep_graph.setdefault(qn, set())

    def resolve_dependencies(self) -> None:
        """Cross-link entities by matching import names to entity names."""
        entity_names = {e.name: e.qualified_name for e in self._entities.values()}
        for qn, entity in self._entities.items():
            for dep in entity.dependencies:
                dep_base = dep.split(".")[-1].split("::")[-1]
                if dep_base in entity_names:
                    target = entity_names[dep_base]
                    self._dep_graph[qn].add(target)
                    # Set reverse link
                    if target in self._entities:
                        self._entities[target].dependents.append(qn)

    def dependency_graph(self) -> dict[str, list[str]]:
        return {k: sorted(v) for k, v in self._dep_graph.items() if v}

    def get_modules(self) -> dict[str, list[CodeEntity]]:
        """Group entities by file path (module)."""
        modules: dict[str, list[CodeEntity]] = {}
        for e in self._entities.values():
            modules.setdefault(e.file_path, []).append(e)
        return modules


# ══════════════════════════════════════════════════════════════════════
# Architecture Synthesizer & Diagram Generator
# ══════════════════════════════════════════════════════════════════════

def synthesize_module_map(
    file_path: str,
    entities: list[CodeEntity],
    content: str,
) -> ModuleMap:
    """Create a ModuleMap for a single file."""
    ext = Path(file_path).suffix.lower()
    lang = {"py": "python", "rs": "rust", "ts": "typescript", "js": "javascript"}.get(
        ext.lstrip("."), "unknown"
    )
    imports = sorted(set(d for e in entities for d in e.dependencies))
    exports = [e.name for e in entities if e.kind in ("class", "struct", "trait", "function")]

    return ModuleMap(
        name=Path(file_path).stem,
        file_path=file_path,
        entities=entities,
        imports=imports,
        exports=exports,
        loc=content.count("\n") + 1,
        language=lang,
    )


def generate_dependency_diagram(
    dep_graph: dict[str, list[str]],
    title: str = "Dependency Graph",
) -> str:
    """Generate a Mermaid dependency diagram from the entity graph."""
    lines = ["---", f"title: {title}", "---", "flowchart LR"]
    # Sanitize node IDs
    seen = set()
    for src, deps in dep_graph.items():
        src_id = _mermaid_id(src)
        src_label = src.split("::")[-1]
        if src_id not in seen:
            lines.append(f'    {src_id}["{src_label}"]')
            seen.add(src_id)
        for dep in deps:
            dep_id = _mermaid_id(dep)
            dep_label = dep.split("::")[-1]
            if dep_id not in seen:
                lines.append(f'    {dep_id}["{dep_label}"]')
                seen.add(dep_id)
            lines.append(f"    {src_id} --> {dep_id}")
    return "\n".join(lines)


def generate_module_diagram(modules: dict[str, list[CodeEntity]]) -> str:
    """Generate a Mermaid module-level architecture diagram."""
    lines = ["flowchart TB"]
    for fpath, entities in sorted(modules.items()):
        mod_name = Path(fpath).stem
        mod_id = _mermaid_id(mod_name)
        classes = [e.name for e in entities if e.kind in ("class", "struct")]
        funcs = [e.name for e in entities if e.kind == "function"]
        label_parts = [mod_name]
        if classes:
            label_parts.append(f"{len(classes)} types")
        if funcs:
            label_parts.append(f"{len(funcs)} fns")
        label = "\\n".join(label_parts)
        lines.append(f'    {mod_id}["{label}"]')
    # Add edges based on imports
    for fpath, entities in modules.items():
        src_id = _mermaid_id(Path(fpath).stem)
        all_deps = set()
        for e in entities:
            for d in e.dependencies:
                dep_mod = d.split(".")[0].split("::")[0]
                if dep_mod != Path(fpath).stem:
                    all_deps.add(dep_mod)
        for dep_mod in all_deps:
            dep_id = _mermaid_id(dep_mod)
            if any(Path(fp).stem == dep_mod for fp in modules):
                lines.append(f"    {src_id} --> {dep_id}")
    return "\n".join(lines)


def _mermaid_id(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)


# ══════════════════════════════════════════════════════════════════════
# The Belief Compiler
# ══════════════════════════════════════════════════════════════════════

class BeliefCompiler:
    """
    Compiles source code into machine-auditable belief artifacts.

    This is the automated Truth → Belief pipeline. It:
      1. Scans source files for code entities
      2. Resolves cross-file dependencies
      3. Generates belief artifacts with proper frontmatter
      4. Creates architecture diagrams
      5. Writes everything to the vault
    """

    # File extensions to parse
    SUPPORTED_EXTENSIONS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx"}

    # Directories to skip
    SKIP_DIRS = {
        "__pycache__", "node_modules", ".git", ".hg", "target",
        "dist", "build", ".tox", ".pytest_cache", ".mypy_cache",
        "venv", ".venv", "env", ".env",
    }

    def __init__(self, vault: VaultManager):
        self._vault = vault
        self._resolver = EntityResolver()

    def compile_directory(
        self,
        directory: str,
        max_files: int = 500,
    ) -> CompilationResult:
        """Compile all source files in a directory into beliefs."""
        result = CompilationResult()
        root = Path(directory)

        if not root.is_dir():
            result.errors.append(f"Not a directory: {directory}")
            return result

        target_files = self._walk_source_files(root, max_files)
        return self._compile_paths(root, target_files)

    def compile_paths(
        self,
        root_dir: str,
        relative_paths: list[str],
    ) -> CompilationResult:
        """Compile a targeted set of relative paths inside a source tree."""
        root = Path(root_dir)
        result = CompilationResult()
        if not root.is_dir():
            result.errors.append(f"Not a directory: {root_dir}")
            return result

        selected: list[Path] = []
        seen: set[Path] = set()
        for rel in relative_paths:
            candidate = (root / rel).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.exists() or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            selected.append(candidate)

        if not selected:
            return result

        return self._compile_paths(root, selected)

    def _compile_paths(
        self,
        root: Path,
        file_paths: list[Path],
    ) -> CompilationResult:
        """Compile a set of absolute file paths under a root into beliefs."""
        result = CompilationResult()
        resolver = EntityResolver()

        # Phase 1: Extract entities from all source files
        all_modules: dict[str, ModuleMap] = {}
        for fpath in file_paths:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                rel = str(fpath.relative_to(root))

                entities = extract_entities(content, rel)
                if entities:
                    resolver.add_entities(entities)
                    module = synthesize_module_map(rel, entities, content)
                    all_modules[rel] = module
                    result.entities_extracted += len(entities)

                result.files_processed += 1
            except Exception as e:
                result.errors.append(f"{fpath}: {e}")
        result.modules_mapped = len(all_modules)

        # Phase 2: Resolve cross-file dependencies
        resolver.resolve_dependencies()
        self._resolver = resolver

        # Phase 3: Generate beliefs for each module
        for rel_path, module in all_modules.items():
            try:
                belief = self._module_to_belief(module, rel_path, resolver)
                self._vault.write_belief(belief)
                result.beliefs_written += 1
                result.belief_ids.append(belief.claim_id)
            except Exception as e:
                result.errors.append(f"Belief write failed for {rel_path}: {e}")

        # Phase 4: Generate architecture overview belief
        if all_modules:
            try:
                arch_belief = self._create_architecture_belief(all_modules, str(root))
                self._vault.write_belief(arch_belief)
                result.beliefs_written += 1
            except Exception as e:
                result.errors.append(f"Architecture belief failed: {e}")

        # Phase 5: Generate dependency diagram
        dep_graph = self._resolver.dependency_graph()
        if dep_graph:
            try:
                diagram = generate_dependency_diagram(dep_graph, f"Dependency Graph: {root.name}")
                media_dir = self._vault.config.path / "media"
                media_dir.mkdir(parents=True, exist_ok=True)
                diagram_path = media_dir / f"depgraph_{root.name}.md"
                diagram_path.write_text(diagram, encoding="utf-8")
                result.diagrams_generated += 1
            except Exception as e:
                result.errors.append(f"Diagram failed: {e}")

        # Module diagram
        if all_modules:
            try:
                mod_diagram = generate_module_diagram(
                    resolver.get_modules()
                )
                media_dir = self._vault.config.path / "media"
                media_dir.mkdir(parents=True, exist_ok=True)
                mod_path = media_dir / f"modules_{root.name}.md"
                mod_path.write_text(mod_diagram, encoding="utf-8")
                result.diagrams_generated += 1
            except Exception as e:
                result.errors.append(f"Module diagram failed: {e}")

        logger.info(
            f"BeliefCompiler: compiled {result.files_processed} files → "
            f"{result.beliefs_written} beliefs, {result.entities_extracted} entities, "
            f"{result.diagrams_generated} diagrams"
        )
        return result

    def compile_file(self, file_path: str, content: str) -> BeliefArtifact | None:
        """Compile a single file into a belief artifact."""
        entities = extract_entities(content, file_path)
        if not entities:
            return None
        module = synthesize_module_map(file_path, entities, content)
        resolver = EntityResolver()
        resolver.add_entities(entities)
        resolver.resolve_dependencies()
        return self._module_to_belief(module, file_path, resolver)

    def _module_to_belief(
        self,
        module: ModuleMap,
        file_path: str,
        resolver: EntityResolver | None = None,
    ) -> BeliefArtifact:
        """Convert a ModuleMap to a BeliefArtifact."""
        # Build body
        body_parts = []
        if module.description:
            body_parts.append(module.description)

        # Summary
        classes = [e for e in module.entities if e.kind in ("class", "struct", "enum", "trait")]
        funcs = [e for e in module.entities if e.kind == "function"]
        [e for e in module.entities if e.kind == "const"]

        body_parts.append(f"**Language:** {module.language}")
        body_parts.append(f"**Lines of code:** {module.loc}")
        body_parts.append("")

        if classes:
            body_parts.append("## Types")
            for c in classes:
                doc = f" — {c.docstring}" if c.docstring else ""
                body_parts.append(f"- `{c.signature}`{doc}")

        if funcs:
            body_parts.append("\n## Functions")
            for f in funcs:
                doc = f" — {f.docstring}" if f.docstring else ""
                body_parts.append(f"- `{f.signature}`{doc}")

        if module.imports:
            body_parts.append("\n## Dependencies")
            for imp in module.imports[:20]:
                body_parts.append(f"- `{imp}`")

        dep_graph = resolver.dependency_graph() if resolver else {}
        module_entities = {e.qualified_name for e in module.entities}
        linked_deps: list[str] = []
        for entity_name in module_entities:
            for dep in dep_graph.get(entity_name, []):
                dep_name = dep.split("::")[-1]
                if dep_name not in linked_deps:
                    linked_deps.append(dep_name)
        if linked_deps:
            body_parts.append("\n## Linked Beliefs")
            for dep_name in linked_deps[:20]:
                body_parts.append(f"- [[{dep_name}]]")

        # Invariants from docstrings
        invariants = []
        for e in module.entities:
            if "invariant" in e.docstring.lower() or "must" in e.docstring.lower():
                invariants.append(f"- {e.name}: {e.docstring[:100]}")
        if invariants:
            body_parts.append("\n## Key Invariants")
            body_parts.extend(invariants)

        sources = [f"{file_path}:{e.line}" for e in module.entities[:10]]

        return BeliefArtifact(
            entity=f"{module.name}",
            title=f"Module: {module.name}",
            body="\n".join(body_parts),
            confidence=0.75,  # auto-compiled → moderate confidence
            status="inferred",
            sources=sources,
            derived_from=["belief_compiler", "sast"],
        )

    def _create_architecture_belief(
        self,
        modules: dict[str, ModuleMap],
        root: str,
    ) -> BeliefArtifact:
        """Create a high-level architecture overview belief."""
        body_parts = [
            f"Architecture overview for `{Path(root).name}`.\n",
            f"**Total modules:** {len(modules)}",
            f"**Total LOC:** {sum(m.loc for m in modules.values()):,}",
            "",
            "## Module Index",
        ]

        for rel, mod in sorted(modules.items()):
            n_types = len([e for e in mod.entities if e.kind in ("class", "struct")])
            n_fns = len([e for e in mod.entities if e.kind == "function"])
            body_parts.append(f"- **{mod.name}** ({mod.language}) — {n_types} types, {n_fns} fns, {mod.loc} LOC")

        # Language breakdown
        langs: dict[str, int] = {}
        for mod in modules.values():
            langs[mod.language] = langs.get(mod.language, 0) + mod.loc
        body_parts.append("\n## Language Distribution")
        for lang, loc in sorted(langs.items(), key=lambda x: -x[1]):
            body_parts.append(f"- {lang}: {loc:,} LOC")

        return BeliefArtifact(
            entity=f"architecture::{Path(root).name}",
            title=f"Architecture: {Path(root).name}",
            body="\n".join(body_parts),
            confidence=0.70,
            status="inferred",
            sources=[f"{root}/"],
            derived_from=["belief_compiler", "architecture_synthesizer"],
        )

    def _walk_source_files(self, root: Path, max_files: int) -> list[Path]:
        """Walk and yield source files. max_files <= 0 means unlimited."""
        unlimited = max_files is None or max_files <= 0
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
            for fn in filenames:
                if Path(fn).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(Path(dirpath) / fn)
                    if not unlimited and len(files) >= max_files:
                        return files
        return files
