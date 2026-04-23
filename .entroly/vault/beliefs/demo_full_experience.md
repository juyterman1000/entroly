---
claim_id: 9ca6b517-2687-4369-bdeb-e10a76fc797b
entity: demo_full_experience
status: inferred
confidence: 0.75
sources:
  - examples\demo_full_experience.py:23
  - examples\demo_full_experience.py:304
  - examples\demo_full_experience.py:397
  - examples\demo_full_experience.py:428
  - examples\demo_full_experience.py:546
  - examples\demo_full_experience.py:583
  - examples\demo_full_experience.py:648
  - examples\demo_full_experience.py:761
  - examples\demo_full_experience.py:55
  - examples\demo_full_experience.py:297
last_checked: 2026-04-23T03:07:07.912842+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: demo_full_experience

**Language:** python
**Lines of code:** 788

## Types
- `class S()`

## Functions
- `def act1_the_pain()`
- `def act2_installation()` — print(f {S.WH}Two ways to install:{S.R} {S.CY}Option A — Docker (zero dependencies):{S.R} {S.GY}  $ {S.WH}docker run -it --rm entroly:latest{S.R} {S.CY}Option B — pip (for MCP integration with Cursor/
- `def act3_real_engine()`
- `def act4_dashboard(engine)`
- `def act5_autotuner()` — print(f {S.WH}The autotuner runs as a background daemon thread (nice+10).{S.R} {S.GY}It NEVER interrupts your coding. It only tunes when your CPU is idle.{S.R} {S.GY}Each iteration takes ~12ms. It mut
- `def act6_business_value(naive_recall, naive_precision, naive_noise, naive_tokens,
                         entroly_recall, entroly_precision, entroly_f1, opt_ms,
                         dupes, tokens_saved, tokens_used, baseline_score, final_score)`
- `def main()`

## Dependencies
- `json`
- `os`
- `pathlib`
- `sys`
- `textwrap`
- `threading`
- `time`
