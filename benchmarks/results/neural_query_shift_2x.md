# PRISM-R Query-Shift and Rehydration Pilot

**RESEARCH PILOT — NO BREAKTHROUGH CLAIM.**

Exact answer-string retention before and after source-span rehydration under same-document future-query shift.

| Method | Current q1 evidence | Unseen q2 evidence | q2 after recovery |
|---|---:|---:|---:|
| Lexical compression | 92.0% | 36.0% | n/a |
| PRISM-R active context | 95.0% | 34.0% | n/a |
| PRISM-R + exact rehydration | 95.0% | 34.0% | 95.0% |

- Active context ratio: 44.2%
- Additional recovery ratio: 28.6%
- Active + recovery ratio: 72.7%
- Current-q1 exact McNemar p: 1.094e-01
- Rehydration exact McNemar p: 3.762e-37

## Caveats

- This measures exact evidence retention, not generated answer correctness.
- Approximate token counts use four characters per token.
- SQuAD paragraphs are short; long-agent repeated-compaction evaluation remains required.
- Future q2 is hidden from initial selection but comes from the same source paragraph.
