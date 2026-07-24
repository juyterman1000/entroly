# Sub-file provenance — minimum experiment findings

Runner: `benchmarks/subfile_experiment.py` · Modes: `benchmarks/subfile_modes.py` ·
Coordinate system: `entroly/source_span.py` · Raw results: `subfile_experiment.json`

5 ContextBench tasks (astropy/django, Python), budget 8000 tokens, BM25 top-40
candidate files shared across modes so **granularity is the only variable**.

## Result (mean over tasks)

| mode | file_recall | file_F1 | line_recall | line_precision | line_F1 | tokens | verify | repro |
|---|---|---|---|---|---|---|---|---|
| file (current)   | 0.10 | 0.133 | 0.029 | 0.003 | 0.006 | 15979 | 1.0 | ✅ |
| line_window      | 0.80 | 0.129 | 0.524 | 0.042 | 0.074 | 7702 | 1.0 | ✅ |
| syntax_block     | 0.60 | 0.190 | 0.351 | 0.044 | 0.071 | 7161 | 1.0 | ✅ |

## Success criteria (all met)

- **100% of selected spans independently verifiable** (byte re-derivation) ✅
- **100% deterministic span ordering** (identical across reversed unit input) ✅
- **material line-precision gain**: block − file = **+0.041 (~14× relative)** ✅
- **no file-recall regression**: sub-file *improved* file recall (0.10 → 0.60/0.80) ✅

## Reading

- Whole-file selection blows the token budget (mean **15,979 tokens** on 1–2 giant
  files) → it finds few gold files and covers gold lines imprecisely.
- Sub-file selection fits many targeted units → **finds the gold files** (recall
  6–8×), covers gold lines with ~14× better precision, and uses **fewer tokens**.
  Precision and compression together, not a trade.
- `line_window` favours recall; `syntax_block` favours precision/file-F1. A
  block-with-surrounding-context hybrid is the natural next step.

## Honest limits

- 5 tasks, all Python; **1/5 missed entirely** (gold outside the top-40 candidates).
  Directional evidence, not a general claim. Absolute line-F1 is still ~0.07 —
  real headroom remains.
- Prototype BM25 ranker, not Entroly's qccr — deliberately, to isolate granularity.

## Verdict

The provenance-precision gap is **resolvable and cheap**. This clears the
preregistered gate to move from the minimum experiment to the full
implementation: wire `SourceSpan` byte offsets + sub-file selection into ingest
with a v1→v2 schema migration (rebuild or mark `precision="file"`; never fabricate
offsets for legacy data).
