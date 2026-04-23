---
claim_id: ba4f32f5-b9c3-4a6c-b065-5aba881ea934
entity: demo_value
status: inferred
confidence: 0.75
sources:
  - examples\demo_value.py:27
  - examples\demo_value.py:46
  - examples\demo_value.py:52
  - examples\demo_value.py:61
  - examples\demo_value.py:68
  - examples\demo_value.py:71
  - examples\demo_value.py:74
  - examples\demo_value.py:151
  - examples\demo_value.py:81
  - examples\demo_value.py:94
last_checked: 2026-04-23T03:07:07.913978+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: demo_value

**Language:** python
**Lines of code:** 403

## Types
- `class C()`

## Functions
- `def bar(value, max_val, width=30, color=C.GREEN)` — Render a Unicode progress bar.
- `def sparkline(values, color=C.CYAN)` — Render a sparkline chart from a list of values.
- `def header(text, width=72)` — Render a styled header.
- `def subheader(text)`
- `def metric(label, value, color=C.WHITE, indent=4)`
- `def divider(char="─", width=72)`
- `def run_demo()`

## Dependencies
- `json`
- `sys`
- `time`
