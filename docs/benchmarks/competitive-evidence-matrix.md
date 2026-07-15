# Competitive evidence matrix

This protocol measures Entroly against the current released Headroom package
without collapsing unlike properties into a promotional score. Quality,
recovery, latency, provider behavior, reliability, security, packaging, cost,
and operator experience are reported separately. A win in one dimension cannot
hide a loss in another.

The current additive index is
[`benchmarks/competitive_evidence_protocol_v2.json`](../../benchmarks/competitive_evidence_protocol_v2.json).
The original
[`competitive_evidence_protocol.json`](../../benchmarks/competitive_evidence_protocol.json)
remains immutable because recovery artifacts embed it. Later dimensions use
their own immutable protocol files rather than rewriting prior evidence.
Thresholds and holdout parameters are frozen before results are inspected.
The current 1.0.60 durability revalidation is frozen separately in
[`recovery_resilience_protocol_v2.json`](../../benchmarks/recovery_resilience_protocol_v2.json);
it does not rewrite the original 1.0.59 protocol or artifact.

## Claim rules

- Preserve every failed, timed-out, missing, and incorrect row.
- Publish development losses as engineering evidence, not as product wins.
- Never replace achieved measurements with requested targets.
- Never claim universal superiority; claims name the system versions, workload,
  boundary, metric, and caveat.
- Correctness and data integrity cannot be traded for compression or latency.
- Provider cost claims require provider-observed usage and immutable pricing or
  invoice provenance.

## Evaluation dimensions

| Dimension | Required evidence | Current state |
|---|---|---|
| Active-context quality | matched caps, exact outputs, paired statistics | implemented |
| Recovery resilience | restart replay, concurrent writers, exact bytes | implemented |
| End-to-end model recovery | model-triggered retrieval and final answer | [implemented](model-triggered-recovery.md) |
| Compression latency | warm/cold p50 and p95 by content type | [implemented](compression-latency.md) |
| Provider conformance | Anthropic, OpenAI Chat/Responses, Gemini shapes | planned |
| Interruption recovery | crash, retry, replay, idempotency | planned |
| Security | secret logs, marker injection, path and tenant isolation | planned |
| Packaging and first run | clean install, doctor, wrap, rollback | planned |
| Operator UX | actionable errors, dashboard truth, recovery steps | planned |
| Provider-observed cost | paired usage and pricing provenance | planned |

## Recovery-resilience suite

The first suite stresses the local source-of-truth store behind reversible
compression. Independent processes write unique omitted evidence concurrently,
exit, and a fresh process attempts to recover every payload by its emitted
reference.

The development matrix uses four writers with eight entries each. It is allowed
to reveal defects and guide implementation. The frozen holdout uses six writers
with eleven entries each and is run only after the repair.

A participant passes only when:

1. every worker exits successfully;
2. every intended write returns a reference;
3. a new process recovers every reference after the writers exit;
4. every recovered payload matches its expected SHA-256; and
5. no wrong payload is ever returned.

Store and retrieval latency are descriptive. A faster system that loses or
misroutes one payload fails.

```powershell
python -m benchmarks.recovery_resilience run `
  --phase development `
  --headroom-python C:\path\to\headroom-0.31.0\Scripts\python.exe `
  --output benchmarks/results/recovery_resilience_development.json

python -m benchmarks.recovery_resilience verify `
  benchmarks/results/recovery_resilience_holdout.json
```

The JSON artifact includes the frozen protocol hash, package and implementation
identities, complete entry matrix, worker errors and exit codes, exact recovered
hashes, latency samples, state-file sizes, aggregate gates, and a canonical
payload hash.

## Recorded recovery results

The development run found a real Entroly data-loss bug. With four pre-opened
writers, only 8 of 32 intended entries survived restart byte-exactly; Headroom
recovered 32 of 32. Three Entroly workers failed. This unfavorable result is
retained in
[`recovery_resilience_development_before.json`](../../benchmarks/results/recovery_resilience_development_before.json).

Entroly then added cross-process read/merge/write serialization, unique durable
temporary files, file and directory synchronization, stale-reader refresh, and
process-level regression tests. The frozen holdout was not changed.

On the fresh-seed six-writer, 66-entry Windows/Python 3.10 revalidation, Entroly 1.0.60 source
and the published Headroom 0.31.0 wheel both wrote and recovered 66 of 66
payloads byte-exactly after restart, with zero incorrect payloads and no worker
or retrieval errors. Therefore the verifier explicitly disallows a
recovery-integrity leadership claim.

| Holdout metric | Entroly 1.0.60 source | Headroom 0.31.0 |
|---|---:|---:|
| Successful writes | 66 / 66 | 66 / 66 |
| Byte-exact restart recovery | 66 / 66 | 66 / 66 |
| Incorrect payloads | 0 | 0 |
| Store-call p50 / p95 | 36.571 / 639.941 ms | **1.945 / 40.486 ms** |
| Retrieval p50 / p95 | **0.077 / 0.099 ms** | 0.851 / 1.837 ms |
| Live state files | **95,438 bytes** | 1,536,096 bytes |

These latency and size observations describe this workload and platform; they
are not standalone product-superiority claims. Headroom was substantially
faster for writes. Entroly was faster for retrieval and used less live state.
Headroom used SQLite WAL with `synchronous=NORMAL`; Entroly fsynced its state
file on every commit and its parent directory where supported. The suite did
not simulate machine power loss, so the store-call latency is not a matched
power-loss-durability comparison.
The complete samples, environment identities, hashes, and errors are in the
[`current-implementation revalidation`](../../benchmarks/results/recovery_resilience_holdout_revalidation.json).
The original post-repair
[`holdout artifact`](../../benchmarks/results/recovery_resilience_holdout.json)
remains unchanged.
The post-repair development iterations remain available as
[`first repair`](../../benchmarks/results/recovery_resilience_development_after.json)
and
[`Windows lock optimization`](../../benchmarks/results/recovery_resilience_development_optimized.json)
evidence.

## Recorded compression-latency result

The quality-gated Windows/Python 3.10 holdout measured Entroly 1.0.59 source as
2.94x faster than Headroom 0.31.0 for warm public compressor calls (95%
stratified bootstrap CI 2.74x to 3.13x) and 2.39x faster for import plus the
first call in a fresh process (1.89x to 2.70x). Both participants completed all
fixtures, retained every preregistered evidence needle, remained deterministic,
and never inflated tokens. See the
[protocol, samples, and limits](compression-latency.md).

## Recorded model-triggered recovery result

On the frozen 24-case synthetic query-shift holdout, raw context and Entroly
both scored 24/24 final exact answers; Headroom scored 18/24. All six discordant
pairs favored Entroly (two-sided exact McNemar `p = 0.03125`). Entroly's mean
effective-context ratio was 28.88%, including source-exact recovery evidence on
all 24 triggered retries, versus 42.97% for Headroom.

This is a scoped result for a local `qwen2.5:1.5b` workflow guard, not a
frontier-model or universal-agent claim. The full protocol evolution—including
timeouts, an over-budget variant, an incomplete-JSON failure, a stale metadata
label, and a rejected low-token projection—is retained in the
[model-triggered recovery report](model-triggered-recovery.md).
