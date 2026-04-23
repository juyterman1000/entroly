---
claim_id: 45e95e6b-36ad-4348-9f79-dd4e7325c610
entity: needle_heatmap
status: inferred
confidence: 0.75
sources:
  - bench\needle_heatmap.py:29
  - bench\needle_heatmap.py:47
  - bench\needle_heatmap.py:107
  - bench\needle_heatmap.py:163
last_checked: 2026-04-23T03:07:07.782882+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: needle_heatmap

**Language:** python
**Lines of code:** 194


## Functions
- `def generate_heatmap(
    results: dict,
    output_path: str = "needle_heatmap.png",
    title: str = "NeedleInAHaystack: Entroly vs Baseline",
)`
- `def build_matrix(data)`
- `def run_needle_sweep(
    model: str = "gpt-4o-mini",
    sizes: list[int] | None = None,
    depths: list[float] | None = None,
    budget: int = 50_000,
) -> dict`
- `def main()`

## Dependencies
- `__future__`
- `json`
- `os`
- `pathlib`
- `sys`
