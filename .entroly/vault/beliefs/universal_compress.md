---
claim_id: 7a579d11-4200-4769-935f-93cf830a52c9
entity: universal_compress
status: inferred
confidence: 0.75
sources:
  - entroly\universal_compress.py:50
  - entroly\universal_compress.py:127
  - entroly\universal_compress.py:248
  - entroly\universal_compress.py:277
  - entroly\universal_compress.py:308
  - entroly\universal_compress.py:341
  - entroly\universal_compress.py:357
  - entroly\universal_compress.py:382
last_checked: 2026-04-23T03:07:07.822541+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: universal_compress

**Language:** python
**Lines of code:** 517


## Functions
- `def detect_content_type(text: str) -> str` — Detect the content type of arbitrary text. Returns one of: json, xml, markdown, log, csv, table, email, legal, stacktrace, or 'prose' as fallback.
- `def tfidf_extractive_summarize(
    text: str,
    target_ratio: float = 0.3,
    min_sentences: int = 3,
    max_sentences: int = 100,
) -> str`
- `def compress_markdown(text: str, target_ratio: float = 0.3) -> str` — Compress markdown: keep headers + first paragraph of each section.
- `def compress_email_thread(text: str) -> str` — Compress email thread: keep headers + first 3 lines of each email.
- `def compress_stacktrace(text: str) -> str` — Compress stack traces: deduplicate repeated frames, keep boundaries.
- `def compress_csv(text: str, max_sample_rows: int = 5) -> str` — Compress CSV: keep header + N sample rows + schema summary.
- `def compress_xml(text: str, max_depth: int = 3) -> str` — Compress XML: keep structure up to max_depth, strip deep content.
- `def universal_compress(
    content: str,
    target_ratio: float = 0.3,
    content_type: str | None = None,
) -> tuple[str, str, float]`

## Dependencies
- `__future__`
- `collections`
- `math`
- `re`
- `typing`
