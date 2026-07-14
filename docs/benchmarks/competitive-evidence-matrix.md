# Competitive evidence matrix

This protocol measures Entroly against the current released Headroom package
without collapsing unlike properties into a promotional score. Quality,
recovery, latency, provider behavior, reliability, security, packaging, cost,
and operator experience are reported separately. A win in one dimension cannot
hide a loss in another.

The first machine-readable source of truth is
[`benchmarks/competitive_evidence_protocol.json`](../../benchmarks/competitive_evidence_protocol.json).
It is immutable because it is embedded in the recovery artifacts. Later
dimensions use their own immutable protocol files rather than rewriting prior
evidence. Thresholds and holdout parameters are frozen before results are
inspected.

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
| End-to-end model recovery | model-triggered retrieval and final answer | planned |
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

On the six-writer, 66-entry Windows/Python 3.10 holdout, Entroly 1.0.59 source
and the published Headroom 0.31.0 wheel both wrote and recovered 66 of 66
payloads byte-exactly after restart, with zero incorrect payloads and no worker
or retrieval errors. Therefore the verifier explicitly disallows a
recovery-integrity leadership claim.

| Holdout metric | Entroly 1.0.59 source | Headroom 0.31.0 |
|---|---:|---:|
| Successful writes | 66 / 66 | 66 / 66 |
| Byte-exact restart recovery | 66 / 66 | 66 / 66 |
| Incorrect payloads | 0 | 0 |
| Store-call p50 / p95 | 22.606 / 435.966 ms | **0.797 / 24.503 ms** |
| Retrieval p50 / p95 | **0.042 / 0.050 ms** | 0.279 / 0.514 ms |
| Live state files | **95,438 bytes** | 1,581,416 bytes |

These latency and size observations describe this workload and platform; they
are not standalone product-superiority claims. Headroom was substantially
faster for writes. Entroly was faster for retrieval and used less live state.
Headroom used SQLite WAL with `synchronous=NORMAL`; Entroly fsynced its state
file on every commit and its parent directory where supported. The suite did
not simulate machine power loss, so the store-call latency is not a matched
power-loss-durability comparison.
The complete samples, environment identities, hashes, and errors are in the
[`holdout artifact`](../../benchmarks/results/recovery_resilience_holdout.json).
The post-repair development iterations remain available as
[`first repair`](../../benchmarks/results/recovery_resilience_development_after.json)
and
[`Windows lock optimization`](../../benchmarks/results/recovery_resilience_development_optimized.json)
evidence.
