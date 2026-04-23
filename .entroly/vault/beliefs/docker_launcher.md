---
claim_id: bf11a823-a3be-4acf-b21a-9f5f529f5c2a
entity: _docker_launcher
status: inferred
confidence: 0.75
sources:
  - entroly\_docker_launcher.py:90
  - entroly\_docker_launcher.py:22
last_checked: 2026-04-23T03:07:07.826649+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: _docker_launcher

**Language:** python
**Lines of code:** 194


## Functions
- `def launch() -> None` — Main entry point — docker launch or native fallback. Routes CLI subcommands (init, dashboard, health, autotune, benchmark, status, proxy) to the local CLI handler. Only `serve` and bare `entroly` go t

## Dependencies
- `__future__`
- `os`
- `pathlib`
- `subprocess`
- `sys`
- `time`
