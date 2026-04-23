---
claim_id: a8dd34a7-b187-4d7d-8ed0-cac0bc185fc4
entity: test_deep_functional
status: inferred
confidence: 0.75
sources:
  - tests\test_deep_functional.py:64
  - tests\test_deep_functional.py:77
  - tests\test_deep_functional.py:83
  - tests\test_deep_functional.py:91
  - tests\test_deep_functional.py:96
  - tests\test_deep_functional.py:102
  - tests\test_deep_functional.py:108
  - tests\test_deep_functional.py:114
  - tests\test_deep_functional.py:122
  - tests\test_deep_functional.py:132
last_checked: 2026-04-23T03:07:07.927202+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_deep_functional

**Language:** python
**Lines of code:** 851


## Functions
- `def check(label: str, condition: bool, detail: str = "") -> bool`
- `def skip_check(label: str, reason: str = "")`
- `def section(name: str)`
- `def get_selected(opt: dict) -> list[dict]` — Extract selected fragments from optimize response.
- `def get_total_tokens(opt: dict) -> int` — Extract total_tokens from optimize response.
- `def get_effective_budget(opt: dict, fallback: int = 0) -> int` — Extract effective budget from optimize response.
- `def get_total_fragments(stats: dict) -> int` — Extract total_fragments from stats response.
- `def get_current_turn(stats: dict) -> int` — Extract current_turn from stats response.
- `def real_sources() -> list[tuple[str, Path]]`
- `def fresh_engine(tmp_dir: str | None = None, **cfg_kwargs)`
- `def test_unicode_binary()`
- `def test_massive_fragment()`
- `def test_rapid_fire_ingest()`
- `def test_dedup_near_miss()`
- `def test_dedup_reorder()`
- `def test_feedback_saturation()`
- `def test_feedback_nonexistent()`
- `def test_optimize_after_eviction()`
- `def test_interleaved()`
- `def test_special_char_queries()`
- `def test_pin_everything()`
- `def test_concurrent_like()`
- `def test_checkpoint_fidelity()`
- `def test_checkpoint_then_ingest()`
- `def test_corrupted_checkpoint()`
- `def test_zero_token_fragments()`
- `def test_same_source_diff_content()`
- `def test_recall_topk_boundary()`
- `def test_budget_equals_corpus()`
- `def test_extreme_aging()`
- `def test_multi_engine_isolation()`
- `def test_empty_whitespace()`
- `def test_self_similar_corpus()`
- `def test_query_refinement()`
- `def test_prefetch_coacccess()`
- `def test_prefetch_imports()` — source =  from mypackage.utils import helper from mypackage.models import User import json def process(data): return helper(User.from_dict(data))
- `def process(data)` — preds = engine.prefetch_related("/project/mypackage/main.py", source_content=source) pred_paths = [p.get("path", "") for p in preds] check("import analysis returns predictions", len(preds) > 0, f"coun
- `def test_stats_consistency()`
- `def test_wilson_score_math()`
- `def test_ebbinghaus_decay_math()`
- `def test_knapsack_dp()`
- `def test_entropy_scoring()` — high =  def compute_gradient_descent(weights, learning_rate, loss_fn): gradients = loss_fn.backward(weights) updated = weights - learning_rate * gradients momentum = 0.9 * previous_velocity + gradient
- `def compute_gradient_descent(weights, learning_rate, loss_fn)`
- `def test_full_lifecycle_stress()`
- `def run()`

## Dependencies
- `json`
- `math`
- `mypackage.models`
- `mypackage.utils`
- `os`
- `pathlib`
- `sys`
- `tempfile`
- `time`
