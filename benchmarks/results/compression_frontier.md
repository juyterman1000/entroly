# Matched token-cap quality frontier

**Entroly wins every matched token-cap quality gate against Headroom.**

60 frozen SQuAD v2 long-context trials; Entroly 1.0.59 source candidate vs released Headroom 0.31.0; achieved ratios use `o200k_base`.
Active-context scope: Headroom's CCR pointers remain in its output, but retrieval recovery is not invoked; this is not an end-to-end Headroom CCR comparison.

| Requested compression | System | Answer retained | Actual tokens kept | p50 latency |
|---:|---|---:|---:|---:|
| 2x | Entroly | 95.0% | 39.3% | 3.2 ms |
| 2x | Headroom | 1.7% | 18.6% | 9.7 ms |
| 4x | Entroly | 93.3% | 19.2% | 2.5 ms |
| 4x | Headroom | 1.7% | 18.6% | 9.5 ms |
| 8x | Entroly | 88.3% | 10.4% | 2.4 ms |
| 8x | Headroom | 1.7% | 18.6% | 28.8 ms |

## Paired retained-answer statistics

| Target | Entroly only | Headroom only | Exact McNemar p |
|---:|---:|---:|---:|
| 2x | 56 | 0 | 2.8e-17 |
| 4x | 55 | 0 | 5.6e-17 |
| 8x | 52 | 0 | 4.44e-16 |

## Local short-answer guard

Model: `qwen2.5:1.5b` at the 4x target. This is a local-model guard, not a hosted frontier-model claim.

| Context | Exact match | Token F1 | Trials |
|---|---:|---:|---:|
| Raw | 62.5% | 80.2% | 8 |
| Entroly | 87.5% | 93.3% | 8 |
| Headroom | 12.5% | 12.5% | 8 |

## Reproduce and verify

```bash
python -m benchmarks.compression_frontier verify benchmarks/results/compression_frontier.json
```

## Scope limits

- SQuAD v2 contexts test extractive answer retention, not full agent behavior.
- Requested target ratios are controls, not assumed outcomes; achieved ratios are primary.
- The local Ollama answer guard is not evidence about hosted frontier models.
- No retrieval or post-compression recovery is invoked in this active-context benchmark.
- Headroom is measured through its released public compress() API and declared config.
- Entroly is a 1.0.59 source candidate; publication must follow before claiming released-package parity.

Payload SHA-256: `3659c60b64c4554dd581318a2ae8dae38aa3d5a9953065dfa4a9d3808b689fd8`
