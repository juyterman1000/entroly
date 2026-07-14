# PRISM-R Query-Shift and Rehydration Pilot

**PILOT GATE PASSED — NOT YET A BREAKTHROUGH CLAIM.**

Exact answer-string retention before and after source-span rehydration under same-document future-query shift.

| Method | Current q1 evidence | Unseen q2 evidence | q2 after recovery |
|---|---:|---:|---:|
| Lexical compression | 24.0% | 5.5% | n/a |
| PRISM-R active context | 81.0% | 3.5% | n/a |
| PRISM-R + exact rehydration | 81.0% | 3.5% | 87.0% |

- Active context ratio: 18.2%
- Additional recovery ratio: 25.3%
- Active + recovery ratio: 43.5%
- Current-q1 exact McNemar p: 9.630e-35
- Rehydration exact McNemar p: 1.069e-50

## Caveats

- This measures exact evidence retention, not generated answer correctness.
- Approximate token counts use four characters per token.
- SQuAD paragraphs are short; long-agent repeated-compaction evaluation remains required.
- Future q2 is hidden from initial selection but comes from the same source paragraph.
