---
claim_id: b76dd981-f265-452d-acee-14a0f0621975
entity: prefetch
status: inferred
confidence: 0.75
sources:
  - entroly\prefetch.py:45
  - entroly\prefetch.py:181
  - entroly\prefetch.py:91
  - entroly\prefetch.py:103
  - entroly\prefetch.py:133
  - entroly\prefetch.py:161
  - entroly\prefetch.py:200
  - entroly\prefetch.py:215
  - entroly\prefetch.py:234
  - entroly\prefetch.py:303
last_checked: 2026-04-23T03:07:07.812608+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: prefetch

**Language:** python
**Lines of code:** 367

## Types
- `class PrefetchResult()` — A predicted context fragment that might be needed next.
- `class PrefetchEngine()` — Predictive pre-fetcher that learns co-access patterns across sessions and combines them with static analysis for predictions. Two prediction strategies: 1. **Static**: Parse imports, calls, and naming

## Functions
- `def extract_callees(source: str, language: str = "python") -> list[str]` — Extract function/method names called from a source code fragment. Returns a list of callee names (not fully qualified — just the function name as it appears in source).
- `def extract_imports(source: str, language: str = "python") -> list[str]` — Extract import targets from a source code fragment. Returns module/path strings that could be resolved to files.
- `def infer_test_files(file_path: str) -> list[str]` — Infer likely test file paths from a source file path. Heuristics: foo.py       → test_foo.py, foo_test.py, tests/test_foo.py utils/bar.py → tests/test_bar.py, utils/test_bar.py src/baz.rs   → tests/ba
- `def module_to_file_candidates(
    module_path: str,
    base_dir: str = "",
    language: str = "python",
) -> list[str]`
- `def __init__(self, co_access_window: int = 5)`
- `def record_access(self, file_path: str, turn: int) -> None` — Record that a file was accessed at a given turn. Updates co-access counts with all files accessed within the co-access window.
- `def predict(
        self,
        file_path: str,
        source_content: str,
        language: str = "python",
        max_results: int = 10,
    ) -> list[PrefetchResult]`
- `def record_actual_access(self, file_path: str) -> None` — Record that a file was actually accessed by the agent. Checks if this file was predicted by any pending prediction set. Computes hit rate and auto-adjusts co_access_window.
- `def set_component_bus(self, bus: Any) -> None` — Attach a ComponentFeedbackBus for persistent metric logging.
- `def stats(self) -> dict`

## Dependencies
- `__future__`
- `collections`
- `dataclasses`
- `os`
- `pathlib`
- `re`
- `typing`
