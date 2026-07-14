# Neural Evidence Retrieval Frontier

**NO BREAKTHROUGH CLAIM: statistical gate not passed.**

Answer-bearing paragraph retrieval under fixed one-of-N selection on a frozen SQuAD v2 validation subset.

| Selector | Top-1 answer-passage recall | Top-2 recall | MRR |
|---|---:|---:|---:|
| Deterministic BM25 | 99.0% | 99.7% | 0.9937 |
| Local transformer | 97.7% | 99.7% | 0.9870 |
| Risk-gated selector | 99.0% | n/a | n/a |
| Dual-channel disagreement guard | 99.3% | n/a | n/a |

## Paired test

- Transformer-only correct: 1
- Lexical-only correct: 5
- Exact McNemar p-value: 0.21875000
- Held-out trials: 300
- Distractors per trial: 15
- Dual-channel guard average selected passages: 1.020 / 16
- Dual-channel guard passage compression: 15.69x

## Calibration

```json
{
  "eligible": false,
  "max_override_error_upper_95": 0.1,
  "minimum_overrides": 40,
  "reason": "no threshold passed the finite-sample risk and non-inferiority gates",
  "threshold": null
}
```

## Caveats

- This benchmark measures answer-bearing paragraph retrieval, not downstream LLM answers.
- The local transformer is a pretrained semantic encoder, not an Entroly-trained compressor.
- The calibration policy uses a Wilson error bound and is not a conformal guarantee.
- A headline requires a positive transformer delta and paired McNemar p < 0.05 on the held-out partition.
