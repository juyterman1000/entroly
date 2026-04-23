---
claim_id: 13cb1768-dbab-42a4-b640-59268962a451
entity: test_ecc
status: inferred
confidence: 0.75
sources:
  - tests\test_ecc.py:21
  - tests\test_ecc.py:158
  - tests\test_ecc.py:175
  - tests\test_ecc.py:55
  - tests\test_ecc.py:67
  - tests\test_ecc.py:73
  - tests\test_ecc.py:82
  - tests\test_ecc.py:90
  - tests\test_ecc.py:102
  - tests\test_ecc.py:117
last_checked: 2026-04-23T03:07:07.930243+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_ecc

**Language:** python
**Lines of code:** 223

## Types
- `class TestFormatHierarchicalContext()` — Tests for format_hierarchical_context().
- `class TestHCCConfig()` — Tests for the HCC config flag.
- `class TestHCCVsFlat()` — Verify HCC and flat formatters produce compatible output.

## Functions
- `def test_basic_formatting(self)` — Should produce non-empty output with all 3 levels.
- `def test_empty_result_returns_empty(self)` — Should return empty string for empty HCC result.
- `def test_includes_security_warnings(self)` — Security issues should appear in the output.
- `def test_includes_ltm_memories(self)` — LTM memories should appear in the output.
- `def test_includes_refinement_info(self)` — Query refinement should appear in the output.
- `def test_includes_preamble_when_warranted(self)` — Preamble should appear when signals warrant it.
- `def test_l1_shows_file_count(self)` — L1 header should show the file count.
- `def test_l2_shows_cluster_count(self)` — L2 header should show the cluster size.
- `def test_l3_shows_fragment_count(self)` — L3 header should show the fragment count.
- `def test_l3_infers_language(self)` — L3 code fences should have the correct language.
- `def test_no_l2_if_empty(self)` — If L2 cluster is empty, that section should be omitted.
- `def test_no_l3_if_empty(self)` — If L3 fragments are empty, that section should be omitted.
- `def test_flag_exists_and_defaults_true(self)`
- `def test_flag_can_be_disabled(self)`
- `def test_both_have_start_end_markers(self)` — Both formatters should have the same start/end markers.
- `def test_hcc_has_more_structure(self)` — HCC output should have explicit level headers.

## Dependencies
- `entroly.proxy_config`
- `entroly.proxy_transform`
- `pytest`
