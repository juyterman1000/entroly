# Entroly architecture and system boundaries

Entroly is an open-source context-control layer for AI agents. It combines repository indexing, budgeted selection, compression, provenance, recoverable handles, memory, output-risk signals, and guarded adaptation. Those capabilities are available through different runtime surfaces; installing one package does not imply that every optional path is active.

This document describes implementation boundaries. Numeric public claims belong in the [public evidence ledger](public-evidence.md), where each result is tied to a protocol and artifact.

## Request lifecycle

```text
eligible sources
    ↓
index + fingerprint + dependency metadata
    ↓
query-conditioned scoring and deduplication
    ↓
explicit token-budget selection
    ↓
full / skeleton / reference representations
    ↓
context receipt + omitted-evidence record + recovery handles
    ↓
configured model or agent runtime
    ↓
optional output-risk signals and request-bound outcome recording
```

Entroly does not silently turn an omitted fragment into proof that the fragment was irrelevant. Receipts expose the decision so a caller can inspect, rehydrate, retry, or change the budget.

## Runtime surfaces

| Surface | Role | Important boundary |
|---|---|---|
| Python package | Reference SDK, CLI, MCP server, proxy, memory, and verification paths | `pip install entroly` can use the Python fallback; verify native capability separately |
| Optional Rust core | Accelerates supported indexing, scoring, selection, and receipt operations through PyO3 | Importability alone is not a readiness check; use capability probes |
| Rust binary | Standalone supported native paths without a Python runtime | Validate target architecture and operating-system dependencies |
| npm MCP package | Node-distributed MCP entry point | Package identity and supported commands are version-specific |
| WASM package | Browser/Node-compatible supported algorithms | It is a separate runtime, not proof of full Python parity |
| OpenClaw plugin | Context-engine integration after OpenClaw normalizes provider messages | OpenClaw retains provider selection, authentication, failover, and wire ownership |

See [product-surface.md](product-surface.md) for the current command and package map.

## Indexing and freshness

The index stores fragments, token estimates, fingerprints, source metadata, and supported dependency information. Startup reconciliation and change listeners update stale entries rather than assuming a previous snapshot is current.

Important boundaries:

- ignored, unsupported, unreadable, oversized, or policy-rejected sources may not enter the eligible corpus;
- token counts are local estimates unless provider-observed usage is recorded;
- source hashes and normalized paths are part of receipt identity;
- index mutation must be atomic or recoverable; a partial update must surface an actionable error;
- prompt-injection scanning occurs before untrusted MCP memory content is stored.

## Scoring and selection

Entroly can combine lexical relevance, entropy-derived signals, dependencies, recency, provenance, risk, and diversity. A knapsack solver selects against Entroly’s scoring objective and explicit budget.

“Optimal” applies only to the stated objective and eligible candidates for an exact solver path. It does not mean the selected context is universally best for answer quality. Learned, heuristic, fallback, and submodular paths must identify themselves as such.

Token reduction, answer quality, latency, cache behavior, and provider cost are separate measurements.

## Resolution and recovery

Selected content can be represented at multiple resolutions:

- **full** — captured source text;
- **skeleton** — signatures or structural detail;
- **reference** — compact identity and retrieval metadata.

Content-Compressed Retrieval and Context Commit paths can attach handles and commitments to captured content. Recovery depends on retaining the corresponding state and authorization. A valid hash proves integrity of bytes, not truth, safety, or entitlement to disclose them.

Recovery stores should follow the same access, retention, deletion, and backup policy as the original source.

## Receipts and provenance

Receipts can record selected and omitted fragment identities, source hashes, budgets, model metadata, costs, and decision traces. Coverage depends on the integration path; Entroly does not claim that every response automatically receives every receipt type.

Byte-level, symbol, relational, and semantic verifiers emit risk signals. They can produce false positives and false negatives. A “pass” is not a correctness proof and does not replace compilation, tests, execution, review, or domain validation.

See [limitations.md](limitations.md) and [context-commits.md](context-commits.md).

## Memory and task dreaming

Memory OS separates short-lived working state, episodic records, consolidated knowledge, and retrieval policy. Task dreaming can recall related memories and bounded current-source excerpts to prepare a task overlay.

Durable task memory is fail-closed:

- the outcome must be externally verifiable and request-bound;
- accepted classes are explicit, such as a passing test, successful command exit, accepted edit, or passing CI result;
- agent self-reports do not become durable success evidence;
- stored metadata is bounded and provenance is retained;
- unsafe recalled or source content is quarantined rather than silently injected.

An overlay is task context, not an authority to modify repository policy or impersonate expertise. Stable project instructions remain in the repository’s human-reviewed guidance files.

See [memory-ecosystem.md](memory-ecosystem.md) and [verified-dreaming.md](verified-dreaming.md).

## Skill synthesis and promotion

Structural synthesis can create a candidate tool from repository structure without a provider call. Local compute, filesystem access, and operational cost still apply.

The lifecycle is:

```text
gap → candidate → output-contract benchmark → testing → promote or prune
```

Candidate execution is disabled by default. Promoted-skill execution requires explicit enablement. Promotion requires at least one eligible benchmark run under the current contract; a tool that merely returns a successful process exit cannot pass when its structured output is wrong, empty, unsafe, or stale.

Generated tools return bounded source excerpts and remain subject to the same security and authorization rules as handwritten tooling.

## Verified dreaming and adaptation

The dreaming loop uses scenarios to diversify bounded policy proposals. A world model may rank which proposals deserve a real evaluation. Synthetic transitions do not promote a policy.

Promotion requires real benchmark evidence and Pareto gates. The stronger verified-dream controller additionally requires disjoint committed candidate and incumbent holdout transitions bound to one policy version. Entroly must not manufacture duplicate holdout evidence merely to satisfy that gate.

All adaptive paths need a retained incumbent and a rollback path. Insufficient evidence means “keep testing,” not “promote.”

## Security model

Core protections include:

- prompt-injection and secret scanning for untrusted context;
- scoped, revocable, expiring attach grants;
- token redaction and bounded event payloads;
- idempotency and durable replay for supported gateways;
- atomic state writes and restart reconciliation;
- explicit opt-in for federation, remote providers, and promoted-skill execution.

“Local-first” means core analysis paths can run locally. When a cloud model is configured, selected prompt content still goes to that provider. Review [SECURITY.md](../SECURITY.md) and [first-run-trust.md](first-run-trust.md) before production use.

## Observability and cost

The dashboard and receipts can show source tokens, selected tokens, cache signals, model identity, and modeled cost. A modeled dollar value depends on the configured price table and is not a provider invoice.

No-provider-call paths still use local CPU, memory, storage, and operator time. Public copy should say “no additional provider call” rather than “free.”

## Failure behavior

Critical workflows should either complete atomically or fail visibly:

- corrupted or incompatible state is rejected with recovery guidance;
- unavailable optional native symbols select a declared fallback or return an actionable error;
- unsafe memory input is rejected before storage;
- marketplace, package, and release publication checks verify exact identity and version;
- downloaded release tooling is checksum-verified before execution;
- adaptive candidates stay inactive until their gates pass.

Silent failure, partial publication, and unlabelled fallback are trust defects.

## Evidence and reproduction

Start with:

```bash
pip install entroly
cd /path/to/repository
entroly verify-claims
```

That command is a bounded local install smoke, not an answer-quality or billing guarantee. For benchmark commands, raw artifacts, scope, and known limitations, use:

- [public-evidence.md](public-evidence.md)
- [benchmarks/README.md](../benchmarks/README.md)
- [neural evidence frontier](benchmarks/neural-evidence-frontier.md)
- [competitive evidence matrix](benchmarks/competitive-evidence-matrix.md)

Keep failures in benchmark denominators. Publish repository revision, package version, environment, budget, cache state, and uncertainty. Never generalize a single workload into universal superiority.

## Code map

| Area | Representative implementation |
|---|---|
| Index and reconciliation | `entroly/auto_index.py`, `entroly/change_listener.py` |
| Selection and compression | `entroly/sdk.py`, `entroly-core/src/knapsack.rs`, `entroly/proxy_transform.py` |
| Receipts and recovery | `entroly/context_commit.py`, `entroly/ccr.py`, `entroly/provenance.py` |
| Memory and task overlays | `entroly/memory_os.py`, `entroly/task_dream.py` |
| Skill lifecycle | `entroly/skill_engine.py`, `entroly/evolution_daemon.py` |
| Dreaming and world model | `entroly/autotune.py`, `entroly/verified_dreaming.py` |
| Security and attach | `entroly/context_firewall.py`, `entroly/session_attach.py` |
| MCP and proxy surfaces | `entroly/server.py`, `entroly/proxy.py` |
| Public trust gate | `scripts/verify_public_trust.py`, `tests/test_public_trust.py` |

The source and tests are authoritative when documentation drifts. If a public statement cannot be reproduced from a committed artifact or inspected code path, treat it as unverified and correct it.
