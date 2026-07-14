# PRISM-R Query-Shift and Rehydration Pilot

**PILOT GATE PASSED — NOT YET A BREAKTHROUGH CLAIM.**

Exact answer-string retention before and after source-span rehydration under same-document future-query shift.

| Method | Current q1 evidence | Unseen q2 evidence | q2 after recovery |
|---|---:|---:|---:|
| Lexical compression | 60.5% | 18.5% | n/a |
| PRISM-R active context | 87.0% | 9.0% | n/a |
| PRISM-R + exact rehydration | 87.0% | 9.0% | 90.5% |

- Active context ratio: 24.1%
- Additional recovery ratio: 26.5%
- Active + recovery ratio: 50.6%
- Current-q1 exact McNemar p: 3.109e-15
- Rehydration exact McNemar p: 1.711e-49

## Caveats

- This measures exact evidence retention, not generated answer correctness.
- Approximate token counts use four characters per token.
- SQuAD paragraphs are short; long-agent repeated-compaction evaluation remains required.
- Future q2 is hidden from initial selection but comes from the same source paragraph.
