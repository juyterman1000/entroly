---
claim_id: f4094afc-5690-410c-b704-89932e05a5fd
entity: super_extractor
status: inferred
confidence: 0.75
sources:
  - scripts\super_extractor.py:4
  - scripts\super_extractor.py:25
  - scripts\super_extractor.py:45
  - scripts\super_extractor.py:53
last_checked: 2026-04-23T03:07:07.918229+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: super_extractor

**Language:** python
**Lines of code:** 87


## Functions
- `def extract_python(filepath)`
- `def extract_rust(filepath)`
- `def extract_generic(filepath)`
- `def main()`

## Dependencies
- `ast`
- `os`
