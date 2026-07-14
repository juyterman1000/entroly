# Compression latency holdout

This benchmark measures the public local compressor calls already used by the
same-input Compression Gauntlet. It does not measure provider latency, model
answers, proxy networking, or general product performance.

The machine-readable protocol was frozen before the development and holdout
runs in
[`benchmarks/compression_latency_protocol.json`](../../benchmarks/compression_latency_protocol.json).
The complete holdout, including every latency observation, output, error field,
participant identity, and bootstrap result, is
[`benchmarks/results/compression_latency_holdout.json`](../../benchmarks/results/compression_latency_holdout.json).

## Result

On Windows with Python 3.10, Entroly 1.0.59 merged source was **2.94x faster**
than the published Headroom 0.31.0 wheel for warm public compression calls
(95% stratified bootstrap CI **2.74x to 3.13x**). For product import plus the
first compression call in a fresh process, Entroly was **2.39x faster**
(**1.89x to 2.70x**).

Both systems completed all four fixtures, returned deterministic output,
retained every preregistered evidence needle, and never inflated tokens. The
claim gate passes separately for warm and cold latency. It does not authorize
an aggregate product score or a universal-superiority claim.

| Fixture | Entroly warm p50 | Headroom warm p50 | Speedup | Entroly cold p50 | Headroom cold p50 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Cargo build failure | 32.063 ms | 69.118 ms | 2.16x | 267.367 ms | 629.098 ms | 2.35x |
| Code-search JSON | 22.332 ms | 42.541 ms | 1.91x | 283.829 ms | 680.485 ms | 2.40x |
| JSON incident | 22.225 ms | 147.298 ms | 6.63x | 337.281 ms | 843.526 ms | 2.50x |
| SRE incident tail | 35.234 ms | 96.773 ms | 2.75x | 295.609 ms | 685.957 ms | 2.32x |

The overall statistic is the geometric mean of the four per-fixture ratios of
Headroom median latency to Entroly median latency. The confidence interval
resamples raw observations within each fixture and participant, recomputes the
medians and geometric mean, and takes the 2.5th and 97.5th percentiles over
10,000 deterministic bootstrap iterations. This prevents the largest fixture
from dominating a pooled timing result.

## Frozen conditions

- Entroly entry point: `entroly.compression_proxy.compress_proxy_payload`,
  1,200-token budget, receipt header disabled.
- Headroom entry point: `headroom.compress`, documented `agent-90` profile,
  `protect_recent=0`, model metadata `gpt-4o`.
- Fixtures: generated build log, JSON incident, SRE log, and code-search JSON
  from `benchmarks.compression_gauntlet.build_scenarios`.
- Warm mode: five warm-ups followed by 30 measured calls per fixture and
  participant (120 observations per participant).
- Cold mode: ten separate fresh processes per fixture and participant (40
  observations per participant). The clock includes product import and the
  first public compressor call, but excludes Python interpreter startup.
- Quality tokenizer: controller `tiktoken==0.9.0` with `o200k_base` for both
  outputs. Participant dependency versions remain recorded separately.
- Headroom environment: exact `headroom-ai==0.31.0` wheel; `pip check` reported
  no broken requirements before publication.

## Reproduce and verify

```powershell
python -m benchmarks.compression_latency run `
  --phase holdout `
  --headroom-python C:\path\to\headroom-0.31.0\Scripts\python.exe `
  --output benchmarks/results/compression_latency_holdout.json

python -m benchmarks.compression_latency verify `
  benchmarks/results/compression_latency_holdout.json
```

The development artifact is retained at
[`benchmarks/results/compression_latency_development.json`](../../benchmarks/results/compression_latency_development.json).
Development results are never eligible for a public claim.

## Limits

This is a deterministic, synthetic, no-model local-compression result. It does
not establish downstream answer quality, provider cost, throughput under
concurrent load, proxy latency, Linux/macOS performance, neural superiority, or
overall product superiority. Those remain separate evidence dimensions.
