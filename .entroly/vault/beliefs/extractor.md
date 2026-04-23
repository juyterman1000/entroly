---
claim_id: 1c838789-6090-4de5-845c-6b5b52eb9753
entity: extractor
status: inferred
confidence: 0.75
sources:
  - scripts\extractor.py:5
  - scripts\extractor.py:22
  - scripts\extractor.py:52
last_checked: 2026-04-23T03:07:07.916127+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: extractor

**Language:** python
**Lines of code:** 72


## Functions
- `def extract_python(filepath)`
- `def extract_rust(filepath)`
- `def main()`

## Dependencies
- `ast`
- `json`
- `os`
