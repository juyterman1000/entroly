---
claim_id: b3601eed-ebda-4556-aced-4c8db0aa30d2
entity: agentskills
status: inferred
confidence: 0.75
sources:
  - entroly\integrations\agentskills.py:87
  - entroly\integrations\agentskills.py:19
  - entroly\integrations\agentskills.py:32
last_checked: 2026-04-23T03:07:07.827438+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: agentskills

**Language:** python
**Lines of code:** 178


## Functions
- `def export_promoted(
    vault_path: str | Path = ".entroly/vault",
    out_dir: str | Path = "./dist/agentskills",
) -> dict[str, Any]`

## Dependencies
- `__future__`
- `json`
- `pathlib`
- `shutil`
- `sys`
- `typing`
