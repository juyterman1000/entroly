# Choosing a context-assurance approach

There is no universally best context tool. The correct choice depends on what
content enters the model, what may be omitted, whether exact recovery matters,
how the agent is integrated, and which costs or failures can actually be
observed.

This guide helps users evaluate Entroly, raw context, and alternative context
systems without relying on star counts, headline compression ratios, or generic
rankings.

## Start with the failure you need to prevent

| Primary risk | Evaluation priority |
|---|---|
| Model misses an answer-bearing file or passage | Task-conditioned retrieval and evidence recall |
| Important source is compressed away | Exact recovery and recovery latency |
| Operator cannot explain a context decision | Receipts, provenance, and omission records |
| Long sessions exceed the model window | Conversation compaction and preserved state |
| Structured tool output dominates input | JSON, logs, shell, and API-result handling |
| Provider input cost is high | Provider-observed usage, cache behavior, and total-session cost |
| Local or regulated data must be controlled | Local processing boundary, proxy behavior, storage, and permissions |
| Agent setup is too difficult | Installation, host support, rollback, and status visibility |
| Unsupported answers reach users | Evidence verification and fail-open/fail-closed policy |
| Repository navigation is poor | Symbol index, change impact, dependency graph, and test localization |

A tool optimized for one row may be weak on another. Shell-output filters,
structured-data compressors, repository retrievers, conversation compactors,
proxies, and recovery systems solve overlapping but different problems.

## Required comparison contract

Before running a comparison, freeze:

- repository, document set, or conversation revision;
- task prompts and expected outcome;
- model, provider, temperature, tools, and permissions;
- effective context and output limits;
- cache settings and cache-warming policy;
- baseline and treatment commands;
- task-level scoring rules;
- timeout, retry, and failure classification;
- whether recovery calls are allowed;
- token source: provider-observed, tokenizer estimate, or byte proxy.

Changing the workload or scoring after viewing results invalidates a held-out
comparison.

## Dimensions to measure

### Task outcome

Measure whether the agent solved the task, not only whether fewer tokens were
sent. For coding work, include tests, static checks, functional behavior, and
whether the patch addressed the requested scope.

### Evidence retention

Record whether answer-bearing source remained active, was recoverable, or was
lost. File-level recall should not be presented as line- or symbol-level
precision.

### Recovery

Test whether omitted source can be recovered exactly after restart, concurrency,
store corruption, stale revisions, and invalid handles. Semantic retrieval of a
similar passage is not exact recovery.

### Total provider usage

Separate:

- active-context size;
- provider-observed input tokens;
- provider-observed output tokens;
- cache writes and cache reads;
- retrieval turns and retries;
- total session usage.

A smaller active context can still increase total usage if it causes extra model
turns or retrieval calls.

### Latency

Measure end-to-end task latency as well as local compression or retrieval time.
Report warm and cold conditions separately when caches or indexes matter.

### Auditability

Check whether the system records source identity, selected and omitted content,
recovery handles, budgets, estimates, provider observations, and residual risk.

### Integration and operations

Test clean install, configuration, normal use, status inspection, upgrade,
uninstall, rollback, and behavior when optional native components are missing.

### Privacy and security

Document what remains local, what is persisted, what is sent to a model provider,
which ports or files are exposed, how credentials are handled, and whether the
system fails open or closed.

## When raw context is the right answer

Use raw context when:

- the input already fits comfortably;
- every byte must remain unchanged;
- the prompt is short or used once;
- additional selection or compression latency is not justified;
- no reliable task signal exists;
- the cost of dropping one detail exceeds the benefit of reduction.

A trustworthy context system should pass through rather than force compression
in these cases.

## When Entroly is a strong fit

Entroly is designed for repeated agent work over medium or large repositories,
structured tool output, conversations, and RAG-like evidence sets where users
value explicit budgets, recoverable omissions, Context Receipts, local
verification, and multiple integration paths.

The fit must still be tested on the user's workload. `entroly simulate` provides
a no-model estimate, while provider-bound accounting and task evaluation require
a supported live integration.

## Comparison worksheet

| Dimension | Raw baseline | Entroly | Alternative | Evidence source |
|---|---:|---:|---:|---|
| Task success |  |  |  |  |
| Active input |  |  |  |  |
| Provider input |  |  |  |  |
| Provider output |  |  |  |  |
| Cache reads/writes |  |  |  |  |
| Retrieval calls |  |  |  |  |
| End-to-end latency |  |  |  |  |
| Answer-bearing evidence retained |  |  |  |  |
| Exact restart recovery |  |  |  |  |
| Receipt/provenance quality |  |  |  |  |
| Installation and rollback |  |  |  |  |
| Local/provider data boundary |  |  |  |  |
| Failures and exclusions |  |  |  |  |

## Interpretation guardrails

- A ceiling where every arm succeeds is not proof of non-inferiority.
- A context-free task cannot measure context quality.
- More reduction with lower task quality is not a win.
- One content type cannot establish universal compression quality.
- Maintainer-run results are not independent evidence.
- Downloads and stars measure attention, not task performance.
- A feature checklist does not establish maturity or reliability.

Publish the complete worksheet, commands, raw artifacts, and negative results so
others can reproduce the decision.
