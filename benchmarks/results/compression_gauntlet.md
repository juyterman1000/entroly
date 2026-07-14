# Agent Context Compression Gauntlet

**SUITE WIN: entroly** by 63.8 percentage points of weighted token savings after the evidence and determinism gates.

Deterministic compression and preregistered string-evidence retention on 4 generated agent-tool fixtures; no LLM answer-quality claim.

## Overall

| System | Version | Evidence recall | Weighted savings | Median latency | Valid scenarios | Compressed | Result |
|---|---|---:|---:|---:|---:|---:|---|
| entroly | 1.0.58 | 100.0% | 95.2% | 26.9 ms | 4/4 | 4/4 | PASS |
| headroom | 0.31.0 | 100.0% | 31.4% | 53.0 ms | 4/4 | 2/4 | PASS |

## Per-fixture evidence gate

| Fixture | System | Input tokens | Output tokens | Savings | Evidence | Deterministic | Valid |
|---|---|---:|---:|---:|---:|---|---|
| cargo-build-failure | entroly | 30,850 | 2,678 | 91.3% | 100% | yes | yes |
| json-incident-middle | entroly | 51,809 | 994 | 98.1% | 100% | yes | yes |
| sre-incident-tail | entroly | 36,087 | 2,864 | 92.1% | 100% | yes | yes |
| code-search-middle | entroly | 34,530 | 771 | 97.8% | 100% | yes | yes |
| cargo-build-failure | headroom | 30,850 | 30,850 | 0.0% | 100% | yes | yes |
| json-incident-middle | headroom | 51,809 | 19,844 | 61.7% | 100% | yes | yes |
| sre-incident-tail | headroom | 36,087 | 19,869 | 44.9% | 100% | yes | yes |
| code-search-middle | headroom | 34,530 | 34,530 | 0.0% | 100% | yes | yes |

## Protocol

- Fixtures: `agent-tool-evidence-v1` (`ba27689ecfeb0ab97ac6ff5a745dbf46b56c7eda58ec4ea06754c11a95dac490`)
- Tokenizer: `tiktoken==0.9.0` / `o200k_base`
- Runs: 5 measured after 2 warm-up(s)
- Budget: 1,200 Entroly tokens per tool block
- Gate: complete matrix + deterministic output + 100% preregistered evidence retention + no token inflation

## Caveats

- This suite measures compressed prompt evidence, not downstream model answers.
- Synthetic fixtures are useful regression controls but do not replace public real-task datasets.
- A suite win applies only to the pinned versions, entry points, configuration, and tokenizer.
- Latency includes each public compression call but excludes process startup and model-provider latency.
- Entroly's gauntlet adapter is deterministic statistical selection; it is not evidence of neural-model superiority.

## Reproduce

```bash
python -m benchmarks.compression_gauntlet run \
  --headroom-python /path/to/headroom-0.31.0-venv/bin/python \
  --require-comparator \
  --runs 5 --warmups 2 \
  --output benchmarks/results/compression_gauntlet.json \
  --markdown benchmarks/results/compression_gauntlet.md
```

The JSON artifact is the source of truth. The Markdown file is generated from it.
