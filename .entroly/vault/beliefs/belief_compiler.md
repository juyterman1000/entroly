---
claim_id: a31fe83b-2b13-475d-bd4c-c2ead17ea8cf
entity: belief_compiler
status: inferred
confidence: 0.75
sources:
  - entroly\belief_compiler.py:43
  - entroly\belief_compiler.py:61
  - entroly\belief_compiler.py:74
  - entroly\belief_compiler.py:283
  - entroly\belief_compiler.py:413
  - entroly\belief_compiler.py:34
  - entroly\belief_compiler.py:55
  - entroly\belief_compiler.py:108
  - entroly\belief_compiler.py:286
  - entroly\belief_compiler.py:291
last_checked: 2026-04-23T03:07:07.791120+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: belief_compiler

**Language:** python
**Lines of code:** 702

## Types
- `class CodeEntity()` — An entity extracted from source code.
- `class ModuleMap()` — A module-level view of the codebase.
- `class CompilationResult()` — Result of a belief compilation run.
- `class EntityResolver()` — Resolves and links entities across files.
- `class BeliefCompiler()` — Compiles source code into machine-auditable belief artifacts. This is the automated Truth → Belief pipeline. It: 1. Scans source files for code entities 2. Resolves cross-file dependencies 3. Generate

## Functions
- `def extract_skeleton(content: str, source: str) -> str`
- `def qualified_name(self) -> str`
- `def extract_entities(content: str, file_path: str) -> list[CodeEntity]` — Extract code entities from a source file.
- `def __init__(self)`
- `def add_entities(self, entities: list[CodeEntity]) -> None`
- `def resolve_dependencies(self) -> None` — Cross-link entities by matching import names to entity names.
- `def dependency_graph(self) -> dict[str, list[str]]`
- `def get_modules(self) -> dict[str, list[CodeEntity]]` — Group entities by file path (module).
- `def synthesize_module_map(
    file_path: str,
    entities: list[CodeEntity],
    content: str,
) -> ModuleMap`
- `def generate_dependency_diagram(
    dep_graph: dict[str, list[str]],
    title: str = "Dependency Graph",
) -> str`
- `def generate_module_diagram(modules: dict[str, list[CodeEntity]]) -> str` — Generate a Mermaid module-level architecture diagram.
- `def __init__(self, vault: VaultManager)`
- `def compile_directory(
        self,
        directory: str,
        max_files: int = 500,
    ) -> CompilationResult`
- `def compile_paths(
        self,
        root_dir: str,
        relative_paths: list[str],
    ) -> CompilationResult`
- `def compile_file(self, file_path: str, content: str) -> BeliefArtifact | None` — Compile a single file into a belief artifact.

## Dependencies
- `.vault`
- `__future__`
- `dataclasses`
- `logging`
- `os`
- `pathlib`
- `re`
