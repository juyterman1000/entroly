---
claim_id: afd22ed5-fd65-4756-b948-94f40c5221ee
entity: archetype_optimizer
status: inferred
confidence: 0.75
sources:
  - entroly\archetype_optimizer.py:97
  - entroly\archetype_optimizer.py:118
  - entroly\archetype_optimizer.py:246
  - entroly\archetype_optimizer.py:131
  - entroly\archetype_optimizer.py:256
  - entroly\archetype_optimizer.py:277
  - entroly\archetype_optimizer.py:328
  - entroly\archetype_optimizer.py:332
  - entroly\archetype_optimizer.py:336
  - entroly\archetype_optimizer.py:363
last_checked: 2026-04-23T03:07:07.787073+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: archetype_optimizer

**Language:** python
**Lines of code:** 617

## Types
- `class CodebaseStats()` — Raw statistics collected from scanning the codebase.
- `class ArchetypeInfo()` — Information about the detected archetype.
- `class ArchetypeOptimizer()` — Manages archetype detection and per-archetype weight optimization. Lifecycle: 1. On startup: scan codebase → fingerprint → classify 2. Load weight profile for detected archetype 3. DreamingLoop optimi

## Functions
- `def scan_codebase(root: Path, max_files: int = 5000) -> CodebaseStats` — Scan a codebase directory and collect structural statistics. Fast scan: reads file extensions and first 100 lines of each file. Skips hidden dirs, node_modules, .git, __pycache__, etc. O(N) in number 
- `def __init__(self, data_dir: str | Path, project_root: str | Path | None = None)`
- `def detect_and_load(self) -> ArchetypeInfo` — Scan the codebase, detect archetype, and load its weights. This is the main entry point, called once per session. Returns ArchetypeInfo with the detected archetype and weights.
- `def current_weights(self) -> dict[str, float]` — Return the current weight profile for the active archetype.
- `def current_archetype(self) -> str | None` — Return the label of the current archetype.
- `def update_weights(self, new_weights: dict[str, float]) -> None` — Update the weights for the current archetype. Called by the DreamingLoop when it finds an improvement. Persists the update to disk.
- `def get_export_weights(self) -> dict[str, float]` — Export the 5 PRISM scoring weights (4D + resonance).
- `def stats(self) -> dict[str, Any]` — Return optimizer statistics for dashboard/monitoring.

## Dependencies
- `__future__`
- `dataclasses`
- `json`
- `logging`
- `math`
- `os`
- `pathlib`
- `re`
- `time`
- `typing`
