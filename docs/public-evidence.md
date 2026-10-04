# Entroly: results you can inspect and reproduce

Entroly helps developers work within context budgets while keeping selection
visible and omitted evidence recoverable. Start with the workflow you need,
inspect the linked results, and evaluate it on your own repository.

## Start with your use case

| You want to… | Start here |
| --- | --- |
| Select useful context within an explicit budget | [Context Receipts example](examples/context_receipt.md): inspect selected evidence, omissions, and ranking reasons. |
| Recover evidence after compression | [Context Commits](context-commits.md): replay captured context and verify recovery integrity. |
| Add context tools to an AI coding client | [MCP setup](mcp-server-guide.html): register the installed `entroly` stdio command. |
| Apply context controls to provider requests | [Proxy guide](compression-proxy.md): configure forwarding, recovery, and optional verification. |
| Inspect what your installation supports | Run `entroly doctor --json`; use [the architecture guide](architecture.md) to follow the actual execution path. |

For a local starting point:

```bash
pip install entroly
entroly doctor --json
entroly simulate
```

`simulate` provides a local context estimate. Compare provider-observed usage
and completed-task quality when evaluating an integration with your model.

## Measured results

Each result below links to its artifact and names the workload it establishes.
Recorded results describe the tested revision and configuration; they give you
a reproducible starting point for evaluating a newer release.

### Context Commit integrity

The synthetic conformance run achieved **128/128** deterministic replays,
**576/576** exact omission recoveries, and **768/768** detected tamper mutations.
This demonstrates replay, recovery, and tamper detection on the protocol's cases.
The measurement is artifact integrity, **not answer quality** or a promise of
identical selection across engines.

- [Inspect the artifact](../benchmarks/results/context_commit_conformance.json)
- Reproduce: `python -m benchmarks.context_commit_conformance`

### Recovery across processes and restarts

The frozen holdout recorded **66/66** exact entries for Entroly 1.0.66 source
and **66/66** for the External Baseline A 0.31.0 comparison. Every recorded entry
was recovered exactly under that concurrent-write and restart workload.

This is historical, release-scoped evidence. Later implementations
require a new frozen revalidation. The comparison establishes parity, not leadership;
it does not establish universal recovery superiority or production reliability.

- [Inspect the recovery holdout](../benchmarks/results/recovery_resilience_holdout_revalidation_v5.json)
- [Read the protocol implementation](../benchmarks/recovery_resilience.py)

### WITNESS: identifying unsupported answers

The HaluEval-QA protocol recorded **0.7976** full-dataset AUROC and **84.92%**
accuracy on the **16,000**-decision held-out split. On the shared **1,200**-decision
GPT sample, accuracy was **86.58%** for WITNESS and **86.25%** for gpt-4o-mini.

These results provide a concrete reference for evaluating the verifier on QA
workloads. The uncertainty overlaps, so Entroly does not claim superiority from
this comparison. WITNESS is configured by integration and enforcement mode;
see [architecture](architecture.md) before enabling it on a response path.

- [Inspect HaluEval-QA results](../benchmarks/results/halueval_qa_faithful.json)
- Reproduce: `python benchmarks/halueval_qa_faithful.py`

### Model-triggered recovery

A frozen 24-case local Qwen2.5-1.5B holdout recorded **24/24** exact final answers
for Entroly and **18/24** for External Baseline A 0.31.0. It demonstrates a
synthetic workflow in which the model retrieves omitted evidence before
answering. The result is scoped to that model, workload, and recorded revision.

- [Inspect the holdout](../benchmarks/results/model_recovery_v7_holdout.json)
- [Protocol and reproduction scope](benchmarks/model-triggered-recovery.md)

### Token reduction and task quality

Token reduction varies by corpus, query, budget, tokenizer, integration,
provider, cache behavior, and recovery path. The useful evaluation is how much
context you can reduce while preserving the evidence and outcomes your task
needs. Include recovered text, latency, and cache effects in that comparison.

The compression gauntlet compares named fixtures under a versioned synthetic
protocol; it is not production-outcome evidence.

- [Same-input compression gauntlet](../benchmarks/results/compression_gauntlet.json)
- [Context Efficiency Frontier protocol](benchmarks/context-efficiency-frontier.md)

### PRISM-R: query-shift research

**PRISM-R is an opt-in research prototype, not the default compressor.**

On a frozen 200-pair same-document query-shift pilot at a nominal 25% active
budget, PRISM-R retained **87.0%** of current-query exact evidence versus
**60.5%** for lexical selection. When a different future question was revealed
after compression, exact local span recovery raised future evidence retention
from **9.0%** to **90.5%**. Active plus recovered text was approximately **50.6%**
of the original.

The experiment explores a useful direction: preserve evidence for today's
question and recover additional spans when the question changes. These figures
measure exact answer-string retention on short SQuAD paragraphs. They
do not measure generated answers, production latency, or billing savings.

- [Research design](research/prism-r-neural-compression.md)
- [Evidence-selection results](benchmarks/neural-evidence-frontier.md)
- [Retrieval artifact](../benchmarks/results/neural_evidence_frontier.json)
- [Query-shift artifact](../benchmarks/results/neural_query_shift.json)
- Verify the recorded artifact: `python -m benchmarks.neural_query_shift verify benchmarks/results/neural_query_shift.json`

## Choose an integration

- [PyPI `entroly`](https://pypi.org/project/entroly/): Python SDK, CLI, and MCP entry point.
- [npm `entroly`](https://www.npmjs.com/package/entroly): Node/WASM runtime.
- [npm `entroly-mcp`](https://www.npmjs.com/package/entroly-mcp): MCP launch bridge.
- [npm `entroly-wasm`](https://www.npmjs.com/package/entroly-wasm): WASM package.
- [Apache-2.0 license](../LICENSE).

The standard Python install declares `entroly-core` as a required dependency
and uses the native engine when a compatible wheel is available. Supported
Python fallback capabilities are reported separately by runtime diagnostics.
The npm runtime is a separate integration surface.

For installed-Python MCP setup, register the argument-free `entroly` stdio
command. `uvx` and `entroly-mcp` registrations also use no `serve` argument.
`entroly serve` selects the explicit Docker-first deployment path;
`ENTROLY_NO_DOCKER=1 entroly serve` selects installed Python.

## Reading the evidence

Source and tests establish implementation; a committed benchmark establishes a
result on its recorded workload. Provider-observed usage and a matched task
baseline establish outcomes for your deployment. Use each kind of evidence to
answer the corresponding question.

[Independent review](independent-review-program.md) and reproducible reports
help extend this evidence to new workloads. For external indexing,
[visit the LobeHub listing](https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score).
Only the live LobeHub page can establish its current external result.

A first-party Entroly page records the project's own evidence; independent
coverage names its external source. A private transcript can guide a review,
while answer-engine probes are nondeterministic observations recorded with a
query and date. Repository metadata alone does not establish a Google ranking.

Maintainers can find publishing checks and historical page decisions in
[Maintaining public evidence](evidence-maintenance.md). For contributors,
[start with the development guide](../CONTRIBUTING.md).
