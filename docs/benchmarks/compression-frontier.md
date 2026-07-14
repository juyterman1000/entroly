# Matched token-cap quality frontier

This benchmark asks a constrained question: when two compressors receive the
same long tool result and the same maximum token cap, how often does the
compressed context retain an accepted answer?

It compares the Entroly 1.0.59 source candidate with the published Headroom
0.31.0 `compress()` API. It is intentionally narrower than a general model or
agent benchmark. Headroom's compressed output can contain CCR pointers; this
benchmark records those pointers but does not invoke retrieval recovery. The
result therefore measures immediately visible active context, not Headroom's
end-to-end CCR workflow.

## Frozen workload

- Dataset: answerable examples from the cached SQuAD v2 validation split.
- Holdout seed: `20260715`; development used the separate seed `20260714`.
- Matrix: 60 questions, each mixed with 15 distinct distractor documents.
- Agent shape: one structured JSON RAG/tool-search result per question.
- Caps: 50%, 25%, and 12.5% of original `o200k_base` tokens (2x, 4x, 8x).
- Repetitions: one warmup and two measured deterministic runs per cell.
- Quality: any SQuAD-accepted answer string remains in active context.
- Local answer guard: eight randomly ordered, label-free prompts to
  `qwen2.5:1.5b` at the 4x cap, temperature 0 and seed 0.

Distractors containing an accepted answer string are excluded. The artifact
stores the complete inputs, outputs, hashes, package identities, exact
configuration, achieved token ratios, target misses, latency samples, model
digest, prompts, predictions, and errors.

## Fair token-cap control

Requested ratios are never reported as achieved compression. Both public APIs
receive up to three feedback attempts using their own target control and the
same measured `o200k_base` output count. A system that still exceeds the cap is
recorded as missing it; the row is not dropped.

The Headroom adapter declares `savings_profile="agent-90"`,
`protect_recent=0`, `min_tokens_to_compress=0`, and the matrix target ratio.
The Entroly adapter declares the exact target budget and disables receipt
headers. Both run in isolated Python processes with user secrets removed and
network-dependent model downloads disabled.

## Fail-closed claim gate

The public win label appears only when all of these conditions pass:

1. both systems complete the full deterministic, non-inflating matrix;
2. Entroly meets every measured token cap;
3. Entroly has strictly higher retained-answer quality at every cap;
4. every paired retained-answer win has two-sided exact McNemar `p <= 0.05`;
5. the local judge scores at least 50% exact match on raw context;
6. Entroly is not below Headroom on local exact match or token F1; and
7. the downstream sample contains no errors.

Changing an output, token count, aggregate, decision, or payload hash makes the
committed verifier fail.

## Reproduce

Use an isolated Headroom environment. On Windows, Headroom 0.31.0 currently
needs an older compatible `ast-grep-cli` wheel because the latest wheel can
fail while creating `sg.exe`:

```powershell
py -3.10 -m venv .venv-headroom
.venv-headroom\Scripts\python.exe -m pip install ast-grep-cli==0.30.0
.venv-headroom\Scripts\python.exe -m pip install "headroom-ai[proxy]==0.31.0"

ollama pull qwen2.5:1.5b

python -m benchmarks.compression_frontier run `
  --headroom-python .venv-headroom\Scripts\python.exe `
  --trials 60 --distractors 15 --runs 2 --warmups 1 `
  --ollama-model qwen2.5:1.5b --answer-trials 8 `
  --output benchmarks/results/compression_frontier.json `
  --markdown benchmarks/results/compression_frontier.md `
  --svg docs/assets/compression_frontier.svg

python -m benchmarks.compression_frontier verify `
  benchmarks/results/compression_frontier.json
```

SQuAD v2 must already be cached; the runner refuses an implicit dataset
download. Ollama is used only for the optional local answer guard.

## What this does not prove

This benchmark does not establish superiority across every task, compressor,
provider, model, or production workload. SQuAD is extractive, the tool output
is structured JSON, the local judge sample is small, and no retrieval or
post-compression recovery is invoked. The Entroly participant is a source
candidate until 1.0.59 is published; the artifact must not be described as a
released-package result before that release exists.
