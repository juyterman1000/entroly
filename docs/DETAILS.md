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

## Engine & install options

Python is the reference runtime. The optional Rust core accelerates supported
compute-heavy paths through PyO3, and a separate Node runtime ships through
WASM. The base Python install does not imply that the Rust extension is active;
`entroly verify-claims` reports the engine mode it actually exercised.

```bash
pip install entroly            # core: MCP server + Python engine
pip install entroly[proxy]     # + HTTP proxy
pip install entroly[native]    # + Rust engine
pip install entroly[full]      # everything

npm install -g entroly         # WASM runtime, no Python needed
docker pull ghcr.io/juyterman1000/entroly:latest
```

**Single binary, no Python** — a standalone Rust proxy that auto-detects Anthropic/OpenAI/Gemini and stays cache-aligned:

```bash
cd entroly/entroly-core && cargo build --release --bin entroly-rs --features proxy
./target/release/entroly-rs proxy --upstream https://api.anthropic.com
```

## Command reference

| Command | What it does |
|---|---|
| `entroly go` | One shot: detect IDE, wrap your agent, open the dashboard |
| `entroly wrap <agent>` | Wrap a specific coding agent (38 supported) |
| `entroly unwrap <agent>` | Remove Entroly's persistent MCP registration without changing other tools |
| `entroly capabilities --json` | Report installed runtime surfaces offline without claiming provider connectivity |
| `entroly attach create/list/revoke` | Grant, inspect, or revoke scoped and expiring MCP access for Claude Code, Codex, or OpenClaw |
| `entroly proxy` | Start the HTTP proxy on `localhost:9377` |
| `entroly` as an MCP stdio command | Start the installed Python MCP server when launched by an MCP client |
| `entroly serve` | Start through the Docker image by default; set `ENTROLY_NO_DOCKER=1` for the installed Python runtime |
| `entroly daemon` | Supervise proxy + dashboard + MCP + file watcher |
| `entroly dashboard` | Open the live metrics dashboard |
| `entroly demo` | Before/after token + cost estimate on your repo |
| `entroly ingest` | Ingest documents into a local Context Receipt index |
| `entroly select` | Select context under budget and write a Context Receipt |
| `entroly context-commit` | Create or verify a replayable, recoverable context artifact |
| `entroly proof prepare/advance/inspect/run` | Run the durable, bounded proof-guided exact-recovery protocol |
| `entroly receipt` | Render a Context Receipt as a Markdown report |
| `entroly explain` | Explain why a chunk was selected or omitted |
| `entroly compress` / `entroly recover` | Compress one file with a receipt; recover the exact original from a digest |
| `entroly simulate` | Local no-LLM savings estimate with an explicit baseline |
| `entroly perf` | Local no-LLM savings and optimizer latency |
| `entroly value` | Evidence-classified provider value, local token reduction, and legacy history |
| `entroly benchmark` | Local comparison: Entroly vs raw context vs top-K |
| `entroly health` | Codebase health grade (A–F) |
| `entroly cache stats` | Persistent cross-session cache stats |
| `entroly ravs report` | Model-routing cost-savings report |
| `entroly witness` | Check an answer against supplied evidence |
| `entroly verify-claims` | Run the packaged self-test → JSON report |

## Context Receipts

Receipt-producing selection workflows record what was used, what was omitted,
why, and what risks remain. This is useful for hard multi-document work such as
contracts, policies, addenda, code reviews, and audit evidence where a bare
top-k result is not enough.

```bash
entroly ingest ./docs
entroly select --query "Does this contract have a change-of-control clause?" --budget 8000
entroly receipt .entroly/receipts/cr_example.json
entroly audit .entroly/session_chain.json
entroly explain --why-omitted chk_example --receipt .entroly/receipts/cr_example.json
```

The receipt JSON includes selected chunks, omitted relevant chunks, ranking
reasons, dependency links, source fingerprints, token ratio, warnings, and a
reproducibility hash. It also includes a selection certificate: bounded
frontiers record exact optimality for Entroly's internal retrieval-score
objective; larger frontiers record a conservative regret ceiling and a ranked
recovery frontier instead of pretending to be optimal. The Markdown report is
designed for human review before a compressed context is trusted.

An independent exhaustive oracle found pure rank-order packing suboptimal in
378 of 47,862 declared small-graph/budget cases. The certified selector improved
all 378, regressed in zero, and matched the oracle in all 47,862—with zero
partial dependency closures, budget violations, or invalid certificates.
[Inspect the machine-readable result.](../benchmarks/results/closed_set_selection_frontier.json)
This is a synthetic internal-objective result, not an answer-quality or
competitor claim.

Implementation notes:

- Rust core (`entroly-core/src/context_receipts.rs`) handles deterministic ingestion, BM25-style ranking, dependency scans, selection, and hashes when the native wheel is available.
- Python control plane (`entroly/context_receipts/`) provides CLI wiring and a pure-Python fallback for source checkouts.
- The semantic/vector scorer and reranker are explicit extension points; the local MVP ships with lexical scoring and dependency heuristics, not a legal-accuracy guarantee.

Examples:

- [Example receipt JSON](examples/context_receipt.json)
- [Example Markdown report](examples/context_receipt.md)
- [Limitations](limitations.md#context-receipts)

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
