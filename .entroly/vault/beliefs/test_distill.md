---
claim_id: 3cb8c813-2b54-4bcf-bcfd-e095835b1c92
entity: test_distill
status: inferred
confidence: 0.75
sources:
  - tests\test_distill.py:18
  - tests\test_distill.py:155
  - tests\test_distill.py:21
  - tests\test_distill.py:42
  - tests\test_distill.py:57
  - tests\test_distill.py:70
  - tests\test_distill.py:79
  - tests\test_distill.py:92
  - tests\test_distill.py:99
  - tests\test_distill.py:106
last_checked: 2026-04-23T03:07:07.928102+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_distill

**Language:** python
**Lines of code:** 174

## Types
- `class TestDistillResponse()` — Output-side token compression — strip filler, preserve code.
- `class TestDistillSSEChunk()` — Streaming chunk distillation — lightweight pattern matching.

## Functions
- `def test_preserves_code_blocks(self)` — Code blocks must NEVER be modified.
- `def test_strips_pleasantries(self)` — Pleasantries carry zero technical content.
- `def test_strips_hedging(self)` — Hedging adds noise without information.
- `def test_verbose_connectors_simplified(self)` — Verbose connectors are replaced with terse equivalents.
- `def test_lite_mode_less_aggressive(self)` — Lite mode only removes pleasantries, not verbose connectors.
- `def test_ultra_mode_strips_articles(self)` — Ultra mode also removes articles and filler words.
- `def test_empty_input(self)` — Empty input should pass through.
- `def test_short_input_passthrough(self)` — Very short input should pass through unchanged.
- `def test_multiblock_preservation(self)` — Multiple code blocks should all be preserved.
- `def test_pure_filler_lines_removed(self)` — Entire lines that are pure filler should be dropped.
- `def test_returns_token_counts(self)` — Should return accurate original and compressed counts.
- `def test_strips_filler_in_chunk(self)` — Should strip filler from individual chunks.
- `def test_preserves_code_chunks(self)` — Code block chunks must pass through.
- `def test_short_chunks_passthrough(self)` — Very short chunks should pass through.

## Dependencies
- `entroly.proxy_transform`
- `pytest`

## Key Invariants
- test_preserves_code_blocks: Code blocks must NEVER be modified.
- test_preserves_code_chunks: Code block chunks must pass through.
