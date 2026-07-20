# Entroly public evidence policy

Entroly separates availability, implementation, measured results, and external
marketplace status. These are different kinds of evidence and must not be
presented as interchangeable trust signals.

## Evidence tiers

| Tier | What it establishes | Required public support |
|---|---|---|
| Distribution | A package or license is publicly available | Direct registry or license link and the exact package name |
| Implementation | A runtime or capability exists in the repository | Source, tests, and accurate optional/default-runtime wording |
| Reproducible measurement | A result occurred under a specific protocol | Committed result, sample size, configuration, reproduction command, and caveats |
| Production outcome | A user workload achieved an outcome | Provider-observed usage, workload definition, baseline, and uncertainty |
| Marketplace status | A third party indexed or validated a release | The live third-party page, checked after publication |

A lower tier cannot be used to imply a higher one. For example, a package badge
does not prove benchmark quality, a local protocol test does not prove an
external marketplace validated the server, and a local token estimate does not
prove a billing reduction.

## Canonical distribution links

- Python package: [PyPI `entroly`](https://pypi.org/project/entroly/)
- Node/WASM runtime: [npm `entroly`](https://www.npmjs.com/package/entroly)
- npm bridge to an installed Python runtime: [npm `entroly-mcp`](https://www.npmjs.com/package/entroly-mcp)
- Standalone WASM package: [npm `entroly-wasm`](https://www.npmjs.com/package/entroly-wasm)
- License: [Apache-2.0](../LICENSE)

The base PyPI installation includes the Python runtime. Rust acceleration is an
optional extra (`pip install 'entroly[native]'`). The npm `entroly` package is a
separate Node/WASM runtime. Documentation must not imply that the base Python
installation is Rust-backed unless runtime verification reports that mode.

## MCP launch contract

The canonical installed-Python MCP registration is the `entroly` stdio command
with no positional arguments. Under an MCP client's stdin pipe, it launches the
installed Python server. The `uvx` and `entroly-mcp` registry entries likewise
use no `serve` argument.

`entroly serve` is the explicit Docker-first deployment command. It uses the
published container by default; `ENTROLY_NO_DOCKER=1` selects the installed
Python runtime. Public setup instructions must state that distinction.

## Prominent reproducible results

### Context Commit integrity

The committed synthetic conformance run reports 128/128 deterministic replays,
576/576 exact omission recoveries, and 768/768 detected tamper mutations across
requested Python and Rust modes. It measures artifact integrity and recovery,
not answer quality or identical cross-engine selection.

- Result: [`context_commit_conformance.json`](../benchmarks/results/context_commit_conformance.json)
- Reproduce: `python -m benchmarks.context_commit_conformance`

### WITNESS HaluEval-QA run

The committed faithful run scores both answers in the balanced HaluEval-QA
protocol. WITNESS reports 0.7976 full-dataset AUROC and 84.92% accuracy on the
16,000-decision held-out split. On the shared 1,200-decision GPT sample, the
committed accuracies are 86.58% for WITNESS and 86.25% for gpt-4o-mini. The
reported uncertainty overlaps, so Entroly does not claim superiority or general
hallucination prevention from this run.

- Result: [`halueval_qa_faithful.json`](../benchmarks/results/halueval_qa_faithful.json)
- Reproduce: `python benchmarks/halueval_qa_faithful.py`
- Exploratory STAVE result: [`stave_benchmark.json`](../benchmarks/results/stave_benchmark.json), kept separate because it uses a different evaluation setup

### Token reduction and task quality

Token reduction depends on the corpus, query, budget, tokenizer, integration,
and recovery behavior. The README's accuracy-retention table links each row to
its exact committed artifact. Those rows are examples from small benchmark
runs, not a universal savings range.

Use `entroly simulate` for a no-network local estimate. Use provider-observed
request usage and the [Context Efficiency Frontier protocol](benchmarks/context-efficiency-frontier.md)
before publishing a production or billing claim.

### Same-input compression gauntlet

The committed no-model gauntlet sends four byte-identical generated agent-tool
fixtures through current Entroly source (package version `1.0.59`) and the
released Headroom `0.31.0[proxy]` public `compress()` entry point with its
documented `agent-90` high-savings profile. Both systems retain 100% of the
preregistered answer strings. Under the shared
`tiktoken==0.9.0` `o200k_base` counter, Entroly records 95.2% weighted token
reduction and Headroom records 31.4%. Pass-through is valid and earns zero
savings; Headroom passes two fixtures through.

This is reproducible-measurement evidence for the named synthetic fixtures. It
is not production-outcome evidence, downstream model-answer evidence, or proof
of neural/ML superiority.

- Generated report: [`compression_gauntlet.md`](../benchmarks/results/compression_gauntlet.md)
- Raw inputs, outputs, hashes, versions, and runtime metadata: [`compression_gauntlet.json`](../benchmarks/results/compression_gauntlet.json)
- Protocol: [`compression-gauntlet.md`](benchmarks/compression-gauntlet.md)
- Verify: `python -m benchmarks.compression_gauntlet verify benchmarks/results/compression_gauntlet.json`

### Model-triggered recovery holdout

The frozen 24-case synthetic query-shift holdout uses a local
`qwen2.5:1.5b` Q4_K_M guard at temperature zero. Raw context and current Entroly
source at package version `1.0.59` scored 24/24 final exact answers; published
Headroom 0.31.0 scored 18/24. All six discordant pairs favored Entroly
(two-sided exact McNemar `p = 0.03125`). Entroly's mean effective-context ratio
was 28.88%, including recovery evidence on every triggered retry, versus 42.97%
for Headroom.

This supports only a scoped result for the named synthetic workflow. It does
not establish hosted-frontier-model quality, universal agent quality,
provider-observed savings, MCP transport performance, or overall product
superiority. The rejected development variants and the stale-metadata artifact
remain linked in the protocol report rather than being hidden.

- Result: [`model_recovery_v7_holdout.json`](../benchmarks/results/model_recovery_v7_holdout.json)
- Protocol and limits: [`model-triggered-recovery.md`](benchmarks/model-triggered-recovery.md)
- Verify: `python -m benchmarks.model_recovery verify --input benchmarks/results/model_recovery_v7_holdout.json`

### PRISM-R neural research pilot

PRISM-R is an opt-in research prototype, not the default compressor. On the
held-out half of a frozen 600-trial SQuAD v2 paragraph-retrieval experiment,
`sentence-transformers/all-MiniLM-L6-v2` revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` underperformed deterministic BM25
as the primary selector (97.7% versus 99.0%). The repository therefore rejects a
neural-primary claim. Retaining both champions only when the systems disagreed
reached 99.3% evidence recall while selecting 1.02 of 16 passages on average.

In a separate 200-pair same-document query-shift pilot at a nominal 25% active
budget, PRISM-R retained 87.0% of current-query exact evidence versus 60.5% for
lexical selection. A different future question was revealed only after
compression; exact locally stored span recovery raised future evidence
retention from 9.0% to 90.5%. Active plus recovered text was approximately
50.6% of the original.

These results measure exact answer-string retention on short SQuAD paragraphs.
They do not measure generated answers, general neural superiority, long-agent
memory, production latency, or billing savings.

- Research design and prior art: [`prism-r-neural-compression.md`](research/prism-r-neural-compression.md)
- Evidence story and full rerun identity: [`neural-evidence-frontier.md`](benchmarks/neural-evidence-frontier.md)
- Held-out retrieval artifact: [`neural_evidence_frontier.json`](../benchmarks/results/neural_evidence_frontier.json)
- Query-shift artifact: [`neural_query_shift.json`](../benchmarks/results/neural_query_shift.json)
- Verify: `python -m benchmarks.neural_evidence_frontier verify benchmarks/results/neural_evidence_frontier.json`
- Verify: `python -m benchmarks.neural_query_shift verify benchmarks/results/neural_query_shift.json`

## Marketplace status

The LobeHub listing is an external discovery surface. The dated
[LobeHub score audit](lobehub-score-audit.md) recorded stale version, license,
capability, ownership, and validation state. Repository readiness and local MCP
tests do not change that public state. Only the
[live LobeHub page](https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score)
can establish the current external result.

The LobeHub ownership badge remains in the README's marketplace section so the
external claim workflow can detect it. It is deliberately excluded from the
top distribution and evidence rows until the listing reflects the current
release and passes external validation.

## Quarantined public surfaces

Trust-sensitive pages are removed from navigation and search indexing when
their claims cannot be supported. The legacy savings, cost-reduction,
hallucination, prompt-compression, and projected-dashboard pages currently
redirect to this policy while their protocols and copy are rebuilt.

The external Hugging Face demo is also excluded from the README's first fold.
Its public metadata must be updated to remove the unsupported universal savings
and zero-accuracy-loss language before it can be presented as an Entroly trust
or demo surface again. An HTTP-successful page is not sufficient evidence that
its copy or runtime is current.

Legacy translated READMEs are marked as archived and removed from the primary
language selector because they predate this evidence policy. They must be
regenerated from the canonical README, including links and caveats, before they
are promoted again.

## Maintainer rules

Before adding or strengthening a public claim:

1. Link the exact package, source, or result—not a directory or screenshot.
2. State the workload, model, budget, sample size, baseline, and caveats needed
   to interpret a measured result.
3. Keep different benchmark protocols in separate rows.
4. Describe estimated usage as estimated and provider-reported usage as
   observed.
5. Do not claim an external marketplace score from repository-local evidence.
6. Remove or soften a claim when its evidence cannot be reproduced.

Run `python scripts/verify_public_trust.py` before release. Use `--online` to
also check canonical public destinations. After publication, add
`--require-published-version` to require PyPI and npm latest versions to match
`server.json`. Online checks are retried but remain subject to third-party
availability.
