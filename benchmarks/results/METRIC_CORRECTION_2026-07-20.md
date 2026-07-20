# AUROC Metric Correction — 2026-07-20

Several benchmark scripts and three production evaluation helpers computed
Mann-Whitney AUROC **without midrank tie correction**. The worst variant
(`sorted(zip(scores, labels))`) breaks score-ties by label, deterministically
ranking positive items above negative items at every tie. With heavily tied
score distributions (ESG/EICV scores saturate at exactly 0.0/1.0) this
inflates AUROC — demonstrated in `diagnose_ablation_artifact.json`, where two
semantically identical inputs "separated" at AUROC 0.9431.

Fix: `entroly/metrics.py::tie_corrected_auroc` is now the single source of
truth; 12 call sites delegate to it; `tests/test_auroc_tie_correction.py`
pins it against a brute-force pairwise reference and bans the biased idiom
from `entroly/` and `benchmarks/`.

## Corrected artifacts (regenerated with current source + corrected metric)

Regenerated values also absorb source drift since the artifacts were first
committed; the AUROC deltas below are dominated by the tie fix wherever score
distributions were heavily tied.

| Artifact | Metric | Committed | Corrected | Conclusion |
|---|---|---:|---:|---|
| `eicv_endtoend.json` | SQuAD min C1–C4 | 0.9976 | 0.9915 | still survives falsification |
| `eicv_endtoend.json` | HaluEval C1 / C4 | 0.9444 / ~0.94 | 0.9273 / 0.8998 | still survives falsification |
| `esg_falsification.json` | min C1–C4 (SQuAD) | 0.9910 | 0.9831 | still survives |
| `esg_falsification.json` | HaluEval C1 | 0.9487 | 0.9262 | still survives |
| `fever_esg.json` | AUROC | 0.7504 | **0.5927** | **materially inflated**; threshold accuracy 0.8100 unaffected — the FEVER *ranking* signal is weak even though the thresholded classifier meets its target |
| `fever_baseline.json` | AUROC | 0.7040 | 0.6095 | inflated; P/R/F1 unaffected |
| `phi_weight_search.json` | worst-case AUROC | 0.7703 | 0.6716 | winners still ESG-dominant; shipped `PHI_WEIGHTS` (`esg: 1.00`, group-DRO) was **not** derived from the biased search and is unchanged |
| `epr_benchmark.json` | — | — | — | no AUROC fields affected |

## Not yet regenerated (require OpenAI API calls)

`truthfulqa_benchmark.json`, `ragas_faithfulness_benchmark.json`,
`fusion_cascade_breakthrough.json` were produced by scripts that call the
OpenAI API. Their committed AUROC values should be treated as potentially
inflated until regenerated with the corrected metric. Their tie exposure is
lower (fusion scores are continuous), but this is an expectation, not a
measurement.

## Practical impact

- No falsification verdict flipped.
- No product configuration changed (`PHI_WEIGHTS` predates and is independent
  of the biased search).
- The headline EICV numbers move from "0.998 / 0.944" to "0.991 / 0.927".
  Quote the corrected values.
- FEVER-based AUROC claims must not be made from the old artifacts.
