---
claim_id: ffe93b21-c0c6-447b-adc9-811f121f6af5
entity: test_archetype
status: inferred
confidence: 0.75
sources:
  - tests\test_archetype.py:104
  - tests\test_archetype.py:149
  - tests\test_archetype.py:182
  - tests\test_archetype.py:262
  - tests\test_archetype.py:285
  - tests\test_archetype.py:33
  - tests\test_archetype.py:54
  - tests\test_archetype.py:76
  - tests\test_archetype.py:94
  - tests\test_archetype.py:105
last_checked: 2026-04-23T03:07:07.922200+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_archetype

**Language:** python
**Lines of code:** 299

## Types
- `class TestCodebaseScanner()`
- `class TestArchetypeClassification()`
- `class TestWeightManagement()`
- `class TestStats()`
- `class TestUtilities()`

## Functions
- `def tmp_project(tmp_path)` — Create a mock Python backend project.
- `def tmp_rust_project(tmp_path)` — Create a mock Rust systems library.
- `def tmp_js_project(tmp_path)` — Create a mock JS frontend project.
- `def optimizer(tmp_path, tmp_project)` — Create an optimizer for the Python project.
- `def test_scan_counts_python_files(self, tmp_project)`
- `def test_scan_counts_functions(self, tmp_project)`
- `def test_scan_counts_imports(self, tmp_project)`
- `def test_scan_detects_rust(self, tmp_rust_project)`
- `def test_scan_detects_js_ts(self, tmp_js_project)`
- `def test_scan_skips_hidden_dirs(self, tmp_project)`
- `def test_scan_respects_max_files(self, tmp_project)`
- `def test_scan_produces_entropy_values(self, tmp_project)`
- `def test_classifies_python_backend(self, tmp_path, tmp_project)`
- `def test_classifies_rust_project(self, tmp_path, tmp_rust_project)`
- `def test_classifies_js_project(self, tmp_path, tmp_js_project)`
- `def test_classification_returns_weights(self, optimizer)`
- `def test_classification_returns_confidence(self, optimizer)`
- `def test_current_weights_match_default(self, optimizer)`
- `def test_update_weights_persists(self, optimizer)`
- `def test_export_weights_prism_5d(self, optimizer)`
- `def test_strategy_table_persists_to_disk(self, tmp_path, tmp_project)`
- `def test_different_projects_get_different_weights(self, tmp_path)`
- `def test_stats_after_detection(self, optimizer)`
- `def test_stats_fingerprint_populated(self, optimizer)`
- `def test_log_norm_zero(self)`
- `def test_log_norm_max(self)`
- `def test_log_norm_monotonic(self)`
- `def test_log_norm_compresses_tails(self)`

## Dependencies
- `entroly.archetype_optimizer`
- `json`
- `os`
- `pathlib`
- `pytest`
- `tempfile`
