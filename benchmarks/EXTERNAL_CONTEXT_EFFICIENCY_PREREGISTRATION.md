# External Context Efficiency Benchmark Track

Status: **preregistered protocol; no model-in-the-loop results collected**

This track evaluates Entroly as a context-management system, not as a language
model. Every reported result must compare the **same model and agent** with and
without Entroly on identical tasks, environments, prompts, tools, seeds, limits,
and provider settings.

A raw benchmark score is never attributed to Entroly. The publishable question
is:

> At equal or non-inferior task success, does Entroly reduce actual provider
> tokens, cost, context failures, or recovery-adjusted work?

## Benchmark tiers

### Tier 1: primary causal benchmarks

| Benchmark | Why it is suitable | Primary Entroly mechanism |
|---|---|---|
| SWE-bench Verified | Repository issue resolution with executable tests | repository selection, symbol graph, code/tool-output compression, recovery |
| Terminal-Bench 2.0 | Long-horizon terminal work in reproducible containers | shell/log compression, context overflow prevention, recoverable history |
| DeepSearchQA | Multi-step research with exhaustive answer sets and stopping decisions | retrieval deduplication, evidence selection, stopping, citation preservation |
| tau2-bench Telecom | Multi-turn policy/tool interaction | conversation compression, policy preservation, tool-result management |

These four benchmarks may support a public context-efficiency claim once the
complete paired protocol passes.

### Tier 2: diagnostic benchmarks

| Benchmark | Allowed use | Restriction |
|---|---|---|
| LiveCodeBench Pro | Prompt/context no-regression and token-efficiency diagnostic | do not imply repository intelligence; most tasks are self-contained |
| HealthBench Hard | Multi-turn context-retention and safety no-regression diagnostic | do not claim improved medical competence; use the official rubric grader |
| MMMU-Pro, CharXiv, ZeroBench, ScreenSpot-Pro | Multimodal no-regression after a production multimodal codec exists | do not use as a headline Entroly test while images are unchanged/no-op |

### Tier 3: base-model capability controls

GPQA Diamond, Humanity's Last Exam, ARC-AGI-2, MedXpertQA Text, and similar
single-prompt knowledge/reasoning suites mainly measure the base model. They may
be used only as **no-regression controls** or with a separately preregistered
long-context/distractor augmentation. Standard scores must not be marketed as an
Entroly win.

## Frozen experimental arms

For each benchmark/model/agent/task, run:

- `A_full`: unmodified full-context baseline.
- `B_native`: agent/model-native compaction, when the agent normally enables it.
- `C_entroly_conservative`: Entroly with fail-closed codecs and recovery.
- `D_entroly_balanced`: Entroly's intended production profile.
- `E_entroly_no_recovery`: ablation only; never a recommended deployment.

A benchmark may omit `B_native` only when no native compaction exists. Arms C-E
must use the same Entroly commit, release identity, and configuration manifest.

## `A_full` baseline proof contract

The baseline is not merely a label. The committed baseline wrapper must:

1. Refuse commands whose arguments reference Entroly.
2. Refuse execution when an Entroly Python module is already loaded.
3. Remove every `ENTROLY_*` environment variable from the child process.
4. Remove known local proxy/base-URL overrides unless explicitly preserved and
   documented for a non-Entroly provider.
5. Record credential **variable names only**, never values.
6. Write an atomic manifest before execution and a completed manifest after it.
7. Record command, task-set, environment, stdout and stderr digests.
8. Require `context_tokens_after == context_tokens_before`, zero recovery work,
   no sufficiency verdict, and null Entroly identity on every `A_full` task row.
9. Default to manifest-only dry-run; paid or network execution requires an
   explicit `--execute` switch.

This proves the treatment is absent from the measured arm. It does not prove
that a provider or external harness is correct; their versions and artifacts
remain separately pinned.

## Identity controls

The following fields must be identical across paired arms unless the field is
the treatment itself:

- benchmark version, dataset hash, task IDs and order;
- model provider, exact model ID and dated revision where available;
- agent/harness commit and container image digest;
- task-input digest, system prompt, task prompt, tool schemas and permissions;
- temperature, top-p, seed policy and retry policy;
- maximum turns, wall-clock timeout, input/output token limits and tool budget;
- network policy and search snapshot where applicable;
- hardware class and concurrency;
- cache mode and cache warm/cold state.

Any unmatched pair is excluded before scores are viewed.

## Required per-task record

Every JSONL row must explicitly include all nullable fields in the committed
schema. Unknown usage values remain `null`; they are never estimated silently.
The record includes at least:

```text
protocol_version
benchmark
benchmark_version
benchmark_task_id
arm
run_id
pair_id
model_id
agent_id
provider_id
harness_commit
environment_digest
task_input_digest
treatment_manifest_digest
entroly_commit
entroly_config_digest
seed
success
benchmark_score
provider_input_tokens
provider_output_tokens
cached_input_tokens
context_tokens_before
context_tokens_after
recovery_tokens
recovery_calls
wall_time_ms
compression_time_ms
peak_rss_bytes
context_overflow
sufficiency_verdict
calibration_policy_id
required_evidence_present
false_sufficient
error_class
excluded
exclusion_reason
artifact_digests
```

## Primary metrics

Report per benchmark and pool only after per-benchmark reporting:

1. Official benchmark task success/score.
2. Actual provider input and output tokens.
3. Recovery-adjusted total tokens.
4. Provider cost using a dated, committed price manifest.
5. Wall time and local compression overhead.
6. Context overflow and forced-compaction rate.
7. Recovery-call rate and recovery success.
8. False-sufficient and false-insufficient rates when evidence labels exist.
9. Cache-prefix stability and cache-read/write tokens when supported.
10. Error and unsupported-input rates.

## Statistical plan

- Use paired task-level comparisons.
- Binary success: paired bootstrap confidence interval and McNemar test.
- Continuous tokens/cost/latency: paired bootstrap and paired permutation test.
- Report median, mean, p50, p95 and 95% confidence intervals.
- Correct families of claims with Holm's method.
- Publish all task-level rows and excluded-pair reasons.
- Do not replace failed runs unless the preregistered retry policy permits it.

## Claim gates

Entroly may be called **more context-efficient** on one benchmark only when all
of these hold on the frozen evaluation set:

1. The lower 95% confidence bound for task-success delta is at least `-2.0`
   percentage points.
2. The lower 95% confidence bound for recovery-adjusted input-token reduction
   is at least `15%`, **or** task success improves significantly without higher
   total provider cost.
3. There is no statistically credible increase in critical safety failures,
   context overflows, corrupted tool calls, or unrecoverable omissions.
4. p95 local overhead is reported and remains within the benchmark's declared
   operational budget.
5. The result is reproduced from a clean environment using committed commands.

A **Pareto-dominance** claim additionally requires non-inferior success and a
strict improvement in at least one of tokens, cost, latency, overflow, or
recoverability without regression on the others.

A cross-benchmark leadership claim requires passing the gate on at least three
Tier-1 benchmarks, at least two model families, and at least two independent
agent/harness implementations. No aggregate may hide a failed benchmark.

## Publication rules

Every public table must show:

- exact commits and release versions;
- task count and excluded pairs;
- model, agent, provider and configuration;
- actual tokens and recovery-adjusted tokens;
- confidence intervals and test method;
- latency and cost provenance;
- failures and negative results;
- reproduction commands;
- a statement that Entroly changes context management, not base-model weights.

Do not publish the result in the README until the immutable task-level artifact,
its checksum, the renderer, and the reproduction command are committed.

## Execution order

1. Validate the common schema, pair identity and raw-baseline manifest.
2. Add SWE-bench Verified adapter and run official gold/oracle smoke tests.
3. Add Terminal-Bench 2.0/Harbor adapter and run oracle plus one agent smoke.
4. Add DeepSearchQA adapter with frozen search and citation accounting.
5. Add tau2-bench Telecom adapter.
6. Run pilot sets only to debug the harness; discard pilot results.
7. Freeze evaluation task IDs and analysis code.
8. Run the full paired evaluation.
9. Publish regardless of whether Entroly wins.

## Current evidence status

No model-in-the-loop external benchmark result has been collected under this
protocol. This document authorizes no quality, superiority, or rank claim.
