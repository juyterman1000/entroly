# Entroly public evidence policy

Entroly separates **distribution**, **implementation**, **reproducible measurements**, **production outcomes**, and **external marketplace status**. A lower evidence tier must never be presented as proof of a higher one.

| Evidence tier | What it establishes | Required support |
|---|---|---|
| Distribution | A package or license is publicly available | Direct registry or license link and exact package name |
| Implementation | A capability exists in the repository | Source, tests, and accurate default/optional wording |
| Reproducible measurement | A result occurred under a defined protocol | Committed artifact, configuration, sample size, command, and caveats |
| Production outcome | A user workload achieved an outcome | Provider-observed usage, workload, baseline, and uncertainty |
| Marketplace status | A third party indexed or validated a release | The current third-party page after publication |

A package badge does not prove benchmark quality. A local token estimate does not prove a billing reduction. Repository-local tests do not prove that an external marketplace validated a release.

## Canonical distribution links

- [PyPI `entroly`](https://pypi.org/project/entroly/)
- [npm `entroly`](https://www.npmjs.com/package/entroly)
- [npm `entroly-mcp`](https://www.npmjs.com/package/entroly-mcp)
- [npm `entroly-wasm`](https://www.npmjs.com/package/entroly-wasm)
- [Apache-2.0 license](../LICENSE)

The base Python package declares `entroly-core` as a required dependency for the standard supported install and uses the native engine when a compatible wheel is available. An internal pure-Python fallback remains for unsupported platforms, but the benchmarked query-conditioned selection path must not be described as an optional native extra. The npm `entroly` package is a separate Node/WASM runtime.

## MCP launch contract

The canonical installed-Python MCP registration is the argument-free `entroly` stdio command. Under an MCP client's stdin pipe, it launches the installed server. The `uvx` and `entroly-mcp` registrations likewise use no `serve` argument.

`entroly serve` is the explicit Docker-first deployment path. `ENTROLY_NO_DOCKER=1 entroly serve` selects the installed Python runtime.

## Reproducible evidence

### Context Commit integrity

The committed synthetic conformance artifact reports:

- **128/128** deterministic replays;
- **576/576** exact omission recoveries;
- **768/768** detected tamper mutations.

These results measure artifact integrity and recovery, **not answer quality** or identical cross-engine selection.

- [Artifact](../benchmarks/results/context_commit_conformance.json)
- Reproduce: `python -m benchmarks.context_commit_conformance`

### Recovery-resilience holdout

The committed v5 recovery-resilience revalidation records **66/66** exact entries for Entroly 1.0.66 source and **66/66** for the External Baseline A 0.31.0 comparison under the frozen holdout protocol. This result establishes **parity, not leadership** for that recovery-integrity workload. It is historical, release-scoped evidence; later Entroly implementations require a new frozen revalidation before this result can be applied to them.

The clean revalidation reproduced no External Baseline A worker errors. It therefore does not permit a public leadership claim and **does not establish universal recovery superiority**, production reliability, task-quality improvement, or provider-cost reduction.

- [Artifact](../benchmarks/results/recovery_resilience_holdout_revalidation_v5.json)
- [Protocol implementation](../benchmarks/recovery_resilience.py)

### WITNESS HaluEval-QA

The committed faithful protocol reports **0.7976** full-dataset AUROC and **84.92%** accuracy on the **16,000**-decision held-out split. On the shared **1,200**-decision GPT sample, committed accuracy is **86.58%** for WITNESS and **86.25%** for gpt-4o-mini.

The uncertainty overlaps, so Entroly **does not claim superiority**, universal truth, or general hallucination prevention from this run.

- [Artifact](../benchmarks/results/halueval_qa_faithful.json)
- Reproduce: `python benchmarks/halueval_qa_faithful.py`

### Token reduction and task quality

Token reduction varies by corpus, query, budget, tokenizer, integration, provider, cache behavior, and recovery path. Use `entroly simulate` for a local estimate and provider-observed request usage before making a production billing claim.

The same-input compression gauntlet is **not production-outcome evidence**. It is a versioned synthetic protocol for named fixtures.

- [Compression gauntlet](../benchmarks/results/compression_gauntlet.json)
- [Context Efficiency Frontier protocol](benchmarks/context-efficiency-frontier.md)

### Model-triggered recovery

A frozen 24-case local Qwen2.5-1.5B holdout recorded 24/24 exact final answers for Entroly and 18/24 for the published External Baseline A 0.31.0 baseline. This is a synthetic, versioned workflow—not a universal product, provider-savings, or model-quality claim.

- [Artifact](../benchmarks/results/model_recovery_v7_holdout.json)
- [Protocol and limitations](benchmarks/model-triggered-recovery.md)

### PRISM-R neural research pilot

**PRISM-R is an opt-in research prototype, not the default compressor.**

On a frozen 200-pair same-document query-shift pilot at a nominal 25% active budget, PRISM-R retained **87.0%** of current-query exact evidence versus **60.5%** for lexical selection. A different future question was revealed only after compression; exact local span recovery raised future evidence retention from **9.0%** to **90.5%**. Active plus recovered text was approximately **50.6%** of the original.

These results measure **exact answer-string retention** on short SQuAD paragraphs. They **do not measure generated answers**, general neural superiority, long-agent memory, production latency, or billing savings.

- [Research design](research/prism-r-neural-compression.md)
- [Evidence story](benchmarks/neural-evidence-frontier.md)
- [Retrieval artifact](../benchmarks/results/neural_evidence_frontier.json)
- [Query-shift artifact](../benchmarks/results/neural_query_shift.json)
- Reproduce: `python -m benchmarks.neural_query_shift verify benchmarks/results/neural_query_shift.json`

## Marketplace status

The LobeHub listing is an external discovery surface. **Only the live LobeHub page can establish the current external result.** Repository readiness, package publication, or local MCP tests cannot establish third-party validation.

- [Live LobeHub listing](https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score)
- [Dated score audit](lobehub-score-audit.md)

Marketplace badges are deliberately excluded from the simplified README until external indexing reflects the current release and validation state.

## Discoverability evidence boundary

The [discoverability registry](discoverability-registry.json) maps each priority
AI-token intent to one canonical public answer, the evidence that answer may use,
and the boundary it must preserve. It also records the measurement channels that
must be connected before search visibility can be reported as observed rather
than assumed.

Search metadata, structured data, internal links, crawler access, and a sitemap
can make Entroly easier to identify and index. They do **not** establish a Google
ranking, an answer-engine citation, product superiority, or independent authority.
A private transcript is useful as a requirements source but is not a public
discovery signal. A first-party Entroly page is evidence of Entroly's published
position only; it becomes third-party coverage only when an external source
independently publishes or cites it.

Current visibility must therefore be measured separately by query intent, engine,
locale, and observation time. Google Search Console and Bing Webmaster Tools
require owner-controlled connections. ChatGPT, Claude, and other answer-engine
probes are nondeterministic observations, not durable rank guarantees. Until those
channels contain dated observations, the registry keeps their status as pending.

## Quarantined public surfaces

Legacy savings, prompt-compression, hallucination, projected-dashboard, and stale setup pages remain `noindex` redirects until their claims and setup instructions are rebuilt. An HTTP-successful page is not sufficient evidence that its copy or runtime is current.

Archived translated READMEs must be regenerated from the canonical README, including trust links and caveats, before they return to primary navigation.

## Retired and republished public pages

A set of topic pages was reduced to `noindex` tombstones redirecting here,
because their claims could not be sourced: a universal 70–95% range, a 0.844
AUROC that a later tie-correction retired, equivalence conclusions drawn against
an API judge, and verifier coverage described as "every response".

Retirement removed those claims by removing the pages. It also removed every
entry point to the topics, and left the repository with no indexable answer to
questions Entroly can answer honestly.

Those pages were republished on 2026-08-15 under a stricter condition than the
one they failed: **every figure on a republished page must resolve to a
committed artifact under `benchmarks/results/`, and the page must state the
workload that produced it.** Where a benchmark set contains a loss case, the
page states it next to the wins rather than omitting it — the SQuAD 2.0 row
(43.8% savings, 90% retention on 233-token inputs) appears alongside the
long-context results it is worse than.

Retirement is no longer what keeps these pages honest. `STALE_PUBLIC_CLAIMS` in
`scripts/verify_context_assurance_public.py` is, and every republished page is
listed in `CLAIM_SENSITIVE_PUBLIC_FILES` so that scan applies to it. A page must
not be republished by deleting its retirement entry without adding it there.

`docs/dashboard.html` remains retired: it is an application view, not content.

## Maintainer rules

Before adding or strengthening a public claim:

1. Link the exact package, source, or result.
2. State the workload, model, budget, sample size, baseline, and caveats.
3. Keep different benchmark protocols separate.
4. Label estimates as estimates and provider-observed usage as observed.
5. Do not infer a marketplace score from repository-local evidence.
6. Remove or soften claims that cannot be reproduced.

Run:

```bash
python scripts/verify_public_trust.py
python scripts/verify_readme.py
```

Use `--online` only for bounded destination checks. After publication, `--require-published-version` can require PyPI and npm latest versions to match `server.json`.
