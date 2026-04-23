---
claim_id: 1cd2a057-0685-4e31-8003-667b4dc0ab1f
entity: proxy_config
status: inferred
confidence: 0.75
sources:
  - entroly\proxy_config.py:180
  - entroly\proxy_config.py:52
  - entroly\proxy_config.py:145
  - entroly\proxy_config.py:243
  - entroly\proxy_config.py:18
  - entroly\proxy_config.py:136
last_checked: 2026-04-23T03:07:07.815248+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: proxy_config

**Language:** python
**Lines of code:** 383

## Types
- `class ProxyConfig()` — Configuration for the entroly prompt compiler proxy. Supports two configuration modes: 1. Explicit: set each parameter individually via env vars 2. Single-dial: set ENTROLY_QUALITY=[0,1] to auto-deriv

## Functions
- `def context_window_for_model(model: str) -> int` — Look up context window size for a model name, with fuzzy prefix matching.
- `def resolve_quality(value: str) -> float` — Accept either a named preset or a float 0.0-1.0.
- `def from_env(cls) -> ProxyConfig` — Create config from environment variables, with tuning_config.json overlay. Supports single-dial mode: set ENTROLY_QUALITY=0.0–1.0 to auto-derive all numeric params from Pareto-interpolated profiles.

## Dependencies
- `__future__`
- `dataclasses`
- `json`
- `logging`
- `os`
- `pathlib`
