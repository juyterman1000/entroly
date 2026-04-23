---
claim_id: caa69a7d-508f-4421-b9ea-3233acb5efae
entity: bump_version
status: inferred
confidence: 0.75
sources:
  - scripts\bump_version.py:30
  - scripts\bump_version.py:11
  - scripts\bump_version.py:13
  - scripts\bump_version.py:27
last_checked: 2026-04-23T03:07:07.915104+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: bump_version

**Language:** python
**Lines of code:** 50


## Functions
- `def main(argv: list[str]) -> int`

## Dependencies
- `__future__`
- `pathlib`
- `re`
- `sys`
