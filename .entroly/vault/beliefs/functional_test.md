---
claim_id: 5ae4427b-ce2e-40d3-a2b2-4f3facd888c5
entity: functional_test
status: inferred
confidence: 0.75
sources:
  - tests\functional_test.py:24
last_checked: 2026-04-23T03:07:07.920090+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: functional_test

**Language:** python
**Lines of code:** 137


## Functions
- `def run_functional_test()`

## Dependencies
- `entroly.config`
- `entroly.server`
- `entroly_core`
- `json`
- `logging`
- `os`
- `pathlib`
- `shutil`
- `sys`
- `tempfile`
- `uuid`

## Linked Beliefs
- [[entroly_core]]
- [[config]]
