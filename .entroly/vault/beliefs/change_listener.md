---
claim_id: 67ddfa78-ba2d-494d-8423-81bc23cb0c9a
entity: change_listener
status: inferred
confidence: 0.75
sources:
  - entroly\change_listener.py:41
  - entroly\change_listener.py:68
  - entroly\change_listener.py:53
  - entroly\change_listener.py:71
  - entroly\change_listener.py:91
  - entroly\change_listener.py:151
  - entroly\change_listener.py:183
last_checked: 2026-04-23T03:07:07.795401+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: change_listener

**Language:** python
**Lines of code:** 239

## Types
- `class WorkspaceSyncResult()`
- `class WorkspaceChangeListener()` — Polls a workspace and feeds file changes into the belief pipeline.

## Functions
- `def to_dict(self) -> dict[str, Any]`
- `def __init__(
        self,
        vault: VaultManager,
        compiler: BeliefCompiler,
        verifier: VerificationEngine,
        change_pipe: ChangePipeline,
        project_dir: str,
        state_path: str | None = None,
    )`
- `def scan_once(self, force: bool = False, max_files: int = 100) -> WorkspaceSyncResult`
- `def start(self, interval_s: int = 120, max_files: int = 100, force_initial: bool = False) -> dict[str, Any]`
- `def stop(self) -> dict[str, Any]`

## Dependencies
- `.belief_compiler`
- `.change_pipeline`
- `.vault`
- `.verification_engine`
- `__future__`
- `dataclasses`
- `json`
- `logging`
- `pathlib`
- `threading`
- `typing`
