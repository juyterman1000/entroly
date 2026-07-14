# Agent Context Compression Gauntlet

The Compression Gauntlet compares released compressor entry points on
byte-identical agent-tool fixtures. It exists to prevent a smaller prompt from
being marketed as a win when answer-critical evidence was removed.

## Claim gate

A system is eligible on a fixture only when it:

1. returns deterministic output across the measured runs;
2. retains every preregistered answer needle;
3. does not inflate tokens under the shared `o200k_base` tokenizer; and
4. completes the same fixture matrix as every comparator.

Pass-through is valid but earns zero savings. The JSON output records input and output hashes, package versions, exact
configuration, tokenizer version, evidence misses, per-run latency, and native
transform metadata. Failures and missing fixtures stop the competitive claim.
The pinned Headroom adapter uses its documented `agent-90` high-savings profile
with `protect_recent=0` so every single-turn tool fixture is eligible.

## Reproduce against a pinned Headroom release

Install the competitor in an isolated environment so its dependencies cannot
change Entroly's runtime:

```bash
python -m venv .venv-headroom
.venv-headroom/bin/python -m pip install "headroom-ai[proxy]==0.31.0"

python -m benchmarks.compression_gauntlet run \
  --headroom-python .venv-headroom/bin/python \
  --require-comparator \
  --runs 5 --warmups 2 \
  --output benchmarks/results/compression_gauntlet.json \
  --markdown benchmarks/results/compression_gauntlet.md \
  --svg docs/assets/compression_gauntlet.svg

python -m benchmarks.compression_gauntlet verify \
  benchmarks/results/compression_gauntlet.json
```

On Windows, use `.venv-headroom\Scripts\python.exe` for
`--headroom-python`.

## What this can and cannot prove

The gauntlet proves deterministic compression and exact retention of the
declared evidence on its generated build-log, JSON, SRE-log, and code-search
fixtures. It does not prove downstream model-answer quality, production cost,
or general superiority. Those require the real-provider Context Efficiency
Frontier with public datasets, provider-observed usage, exact responses, and
paired confidence intervals.

The Entroly adapter currently exercises query-conditioned evidence locks,
entropy scoring, schema preservation, and outlier selection. Do not describe
this adapter as a neural compressor or use it as evidence of ML superiority.
