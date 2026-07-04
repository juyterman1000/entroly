# Entroly Product Surface

This page maps what Entroly exposes today from the codebase. It is meant to help developers choose the right integration path without guessing from marketing copy.

Entroly is best understood as a local context OS for AI agents:

1. **Context engine** — rank, deduplicate, compress, and select evidence under budget.
2. **Recovery ledger** — keep omitted originals reachable by handle.
3. **Receipt system** — explain selected context, omitted context, risks, hashes, and dependencies.
4. **Verification layer** — check model outputs against retained evidence.
5. **Gateway control plane** — normalize provider requests, control cost, cache prefixes, and account for usage.
6. **Learning/memory layer** — adapt from outcomes, cache useful selections, and preserve agent memory locally.
7. **Security layer** — scan code, detect prompt injection, apply redaction policy, and keep file reads contained.
8. **Multi-runtime packaging** — Python, Rust native, Node/WASM, Docker, CLI, SDK, MCP, and proxy.
9. **Self-improvement layer** — convert real outcomes into safer policies, skills, and weight profiles.
10. **Memory/session/value layer** — retain useful working memory, session continuity, checkpoint relevance, and lifetime savings evidence.
11. **Multimodal intake layer** — convert diagrams, images, voice, diffs, and structured artifacts into budgetable context.
12. **CogOps/vault layer** — compile beliefs, route epistemically, verify knowledge, and sync workspace changes.
13. **Integration/event layer** — connect framework helpers and team event gateways into the same local feedback loop.

## Installable packages

| Package | Registry | Purpose |
|---|---|---|
| `entroly` | PyPI | Python CLI, SDK, MCP server, pure-Python engine, optional proxy/native extras |
| `entroly-core` | PyPI | Rust native engine for Python via PyO3 |
| `entroly` | npm | Node compatibility alias that delegates to the WASM runtime |
| `entroly-mcp` | npm | NPX bridge for MCP users |
| `entroly-wasm` | npm | WASM runtime for Node users without a Python-first setup |
| `ghcr.io/juyterman1000/entroly` | GHCR | Docker image |

Recommended first impression:

```bash
pip install -U entroly
entroly verify-claims
entroly simulate
```

Then choose one integration path. Python is the fullest CLI/SDK path. npm/WASM
is the no-Python Node path. The MCP bridge exists for MCP clients that prefer
`npx`, and proxy mode is for users who control provider API keys.

## CLI surface

Core first-run commands:

```bash
entroly verify-claims
entroly simulate
entroly perf
entroly doctor
entroly status
```

Agent and provider integration:

```bash
entroly wrap claude
entroly wrap cursor
entroly wrap codex
entroly wrap aider
entroly proxy
entroly serve
entroly daemon
```

Context and receipt workflow:

```bash
entroly optimize
entroly ingest
entroly select
entroly receipt
entroly explain
entroly batch
```

Quality, safety, and learning:

```bash
entroly witness
entroly verify
entroly verify-code
entroly feedback
entroly learn
entroly benchmark
entroly health
entroly ravs report
entroly cache stats
```

Operational commands:

```bash
entroly config
entroly telemetry
entroly clean
entroly export
entroly import
entroly migrate
entroly completions
```

## SDK surface

Common imports:

```python
from entroly import compress, compress_messages, optimize, verify
from entroly import create_context_receipt, render_context_receipt
from entroly import explain_receipt_omission, context_receipt_from_path
```

Advanced local control:

```python
from entroly import localize_files, localize_fragments
from entroly import CacheAligner
from entroly import ContextLedger, ProviderPrice, clamp_injected_budget
from entroly import MemoryOS
```

Verification and trust:

```python
from entroly import WitnessAnalyzer
from entroly import EICVAnalyzer, EICVSuppressor
from entroly import stave_verify, stave_risk
from entroly import acf_scan, acf_sanitize
```

Compression and recovery:

```python
from entroly import compress_evidence_locked
from entroly import compress_proxy_payload
from entroly import CompressionRetrievalStore
from entroly import answer_with_retrieval_verification
```

## Memory, session, and value intelligence

Entroly includes memory surfaces that make long-running agent work less wasteful and less forgetful.

| Area | Code surface | What it supports |
|---|---|---|
| Memory OS | `memory.py`, `memory_cli.py` | Budget-aware working, episodic, and semantic recall with safety scanning and receipts |
| Memory Fabric | `memory_fabric.py`, `long_term_memory.py`, `memory_kernels.py` | Longer-horizon memory, consolidation, and retrieval patterns |
| Session intelligence | `session_intelligence.py`, `checkpoint.py` | Decision digests, checkpoint relevance, cache-retention forecasting, behavioral-waste detection |
| Value evidence | `value_tracker.py`, `cost_cortex.py`, `usage_ledger.py` | Lifetime savings, observed provider usage, cache economics, and spend summaries |
| Agent bridge | `context_bridge.py`, `verified_handoff.py` | Share context between tools while preserving verification and handoff boundaries |

The product claim this enables is stronger than "compress this prompt": Entroly can help a developer keep useful state across a coding session, spend tokens where they are likely to matter, and explain what value was actually measured.

## Multimodal and image context

Not all valuable context starts as plain text. Entroly has intake surfaces for turning developer artifacts into structured, budgetable evidence.

| Area | Code surface | Notes |
|---|---|---|
| Multimodal ingestion | `multimodal.py`, MCP `ingest_diff`, `ingest_diagram`, `ingest_voice` | Converts non-text or semi-structured material into fragments the context engine can rank |
| Image optimization | `image_optimizer.py` | Estimates provider image token cost and applies compliance-gated optimization where enabled |
| Smart reads | `semantic_resolution.py`, `context_scaffold.py`, `file_localizer.py`, `query_refiner.py`, `prefetch.py` | Helps agents ask for the right files, symbols, and context slices instead of bulk-reading the repo |

This should be described as context intake and budgeting, not as magic vision understanding. The model still receives selected evidence through the configured provider path.

## Gateway, provider policy, and accounting

The proxy path is backed by policy and accounting modules, not just URL forwarding.

| Area | Code surface | What it supports |
|---|---|---|
| Provider adapters | `provider_adapters.py`, `proxy.py`, `compression_proxy.py`, `compression_proxy_live.py` | Anthropic/OpenAI-compatible request handling and provider-shaped responses |
| Provider policy | `provider_policy.py`, `proxy_config.py` | Capability planning, explicit routing constraints, and gateway redaction policy |
| Cache routing | `cache_routing.py`, `cache_aligner.py`, `stable_prefix.py` | Prefix stability, cache leases, and cache-aware request planning |
| Usage accounting | `usage_ledger.py`, `gateway_control_plane.py`, `cost_cortex.py` | Provider-reported token categories, pricing catalogs, spend summaries, and team/project filters |
| Budget harness | `harness_budget.py`, `adaptive_budget.py` | Subagent budget allocation and cost-aware execution planning |

The honest customer story is: Entroly does local context optimization first; when users opt into proxy mode, it also accounts for provider usage and preserves provider protocol boundaries.

## CogOps knowledge vault

Several modules support a higher-level knowledge layer for codebases and agent workflows.

| Area | Code surface | What it supports |
|---|---|---|
| Belief compilation | `belief_compiler.py`, `vault.py` | Build and query local beliefs from workspace evidence |
| Epistemic routing | `epistemic_router.py`, `flow_orchestrator.py` | Choose flows based on evidence coverage, freshness, and risk |
| Verification engine | `verification_engine.py`, `verifiers/` | Check claims, code facts, symbols, provenance, type/scope signals, and repair paths |
| Change awareness | `change_pipeline.py`, `change_listener.py` | Keep local knowledge aligned as files change |
| Native CogOps | `entroly-core/src/cogops.rs`, `entroly-wasm/js/cogops.js` | Rust/WASM runtime support for the same family of primitives |

This is a real differentiator for developer trust: the local runtime can reason over what it knows, what changed, what is stale, and what needs verification.

## Framework and event integrations

Entroly also ships integration points for teams and agent ecosystems.

| Integration | Code surface | Use |
|---|---|---|
| LangChain | `integrations/langchain.py` | Programmatic context optimization in LangChain-style apps |
| AgentSkills | `integrations/agentskills.py`, `entroly-wasm/js/agentskills_export.js` | Export reusable skills produced or curated by the runtime |
| Hermes/events | `integrations/hermes.py` | Event bridge for operational workflows |
| Team gateways | `integrations/slack_gateway.py`, `discord_gateway.py`, `telegram_gateway.py` | Optional feedback/status channels for teams |
| Dashboard/daemon | `dashboard.py`, `compression_dashboard.py`, `controls_html.py`, `daemon.py`, `control_plane.py` | Local observability, control-plane UX, and supervised runtime processes |

These surfaces should be framed as optional integrations. A first-time user should not need them to see value from `verify-claims`, `simulate`, MCP, SDK, or proxy mode.

## MCP tool families

The MCP server exposes tools across these groups:

| Family | Examples |
|---|---|
| Context memory | `remember_fragment`, `optimize_context`, `recall_relevant`, `entroly_retrieve` |
| Context Receipts | `create_context_receipt`, `render_context_receipt`, `explain_receipt_omission`, `recover_receipt_omission` |
| Outcome learning | `record_outcome`, `record_test_result`, `record_command_exit`, `record_ci_result`, `record_edit_outcome` |
| Explainability | `explain_context`, `entroly_dashboard`, `get_stats` |
| Security and health | `scan_for_vulnerabilities`, `security_report`, `analyze_codebase_health`, `security_scan` |
| Multimodal/context intake | `ingest_diff`, `ingest_diagram`, `ingest_voice`, `smart_read` |
| Knowledge vault | `compile_beliefs`, `verify_beliefs`, `vault_query`, `vault_search`, `sync_workspace_changes` |
| Response verification | `verify_provenance`, `verify_and_repair`, `verify_response`, `eicv_verify_claim`, `eicv_suppress_hallucinations` |

## Trust stack

Entroly's trust layer is intentionally layered:

- **Context Receipts** show selected context, omitted context, ranking reasons, fingerprints, dependency links, warnings, and reproducibility hashes.
- **Exact recovery** gives compressed fragments handles so originals can be fetched later.
- **WITNESS** checks generated answers against supplied evidence.
- **EICV/STAVE/verifier modules** provide additional deterministic and statistical verification signals.
- **Adversarial Context Firewall** scans for prompt injection, encoded payloads, repetition flooding, and integrity issues.
- **Receipt proof modules** provide attestation, disclosure, Merkle, and witness layers.

## Self-improvement stack

Entroly's self-improvement surfaces are deliberately guarded. They adapt from evidence, tests, CI, command exits, verification outcomes, and user acceptance.

| Component | Role |
|---|---|
| `FeedbackJournal` / `autotune.py` | Stores results and evaluates configuration mutations against benchmark cases |
| `OnlinePrism` | Bayesian online weight adaptation from reward and contribution signals |
| `RAVS` | Collects honest outcomes from tests, commands, CI, retries, edits, and verification |
| `OutcomeBridge` | Corrects PRISM posterior weights with delayed honest outcomes |
| `RewardCrystallizer` | Turns statistically repeated wins into candidate reusable skills |
| `SkillEngine` | Synthesizes, benchmarks, promotes, merges, or prunes skills |
| `EvolutionDaemon` | Orchestrates structural synthesis, idle dreaming, archetype-aware evolution, and optional federation |
| `PromotionGate` | Promotes shadow policies only when non-inferior and supports rollback |
| `ValueTracker` | Tracks lifetime savings so optional evolution can be budget-gated |
| `ArchetypeOptimizer` | Detects project shape and loads project-appropriate weight priors |

The intended loop:

```text
selection -> model/tool work -> tests/CI/user outcome -> RAVS event
         -> PRISM correction -> safer weights/routes -> promotion gate
         -> repeated wins crystallize into skills
```

Safety boundaries:

- Structural synthesis is attempted before any LLM fallback.
- LLM-backed synthesis is budget-gated.
- Promotions require non-inferiority checks.
- Rollback is available when post-promotion repair/retry/success metrics regress.
- Federation is opt-in.

## Native Rust/WASM engine

The native engine is not a thin wrapper. It contains the high-throughput context machinery used by Python and Node/WASM paths:

| Engine area | Code surface |
|---|---|
| Retrieval/ranking | `bm25`, `query`, `qccr`, `query_persona`, `localization` |
| Selection/packing | `knapsack`, `knapsack_sds`, `hierarchical`, `skeleton`, `utilization` |
| Information scoring | `entropy`, `anomaly`, `resonance`, `rnr`, `semantic_dedup` |
| Dedup/dependencies | `dedup`, `lsh`, `depgraph`, `causal` |
| Learning/cache | `learning`, `prism`, `cache`, `trajectory`, `conversation_pruner` |
| Safety/health | `sast`, `guardrails`, `health`, `compliance` |
| Verification | `witness`, `eicv`, `eicv_suppressor` |
| Agent memory | `memory`, `ipc`, `pollination`, `cognitive_bus`, `cogops` |
| Proxy/runtime | `proxy`, `compress`, `context_receipts`, `entroly-rs` binary |

The WASM package mirrors the native engine shape for JavaScript/TypeScript users and adds app-level SDK helpers for OpenAI, Anthropic, Gemini, and Vercel AI SDK-style middleware.

## Proof and benchmark assets

The repository includes committed benchmark/proof artifacts rather than only screenshots:

| Area | Examples |
|---|---|
| Accuracy retention | `benchmarks/results/needle_accuracy.json`, `longbench_accuracy.json`, `squad_accuracy.json`, `gsm8k_accuracy.json`, `mmlu_accuracy.json`, `truthfulqa_accuracy.json`, `bfcl_accuracy.json` |
| Verification | `stave_benchmark.json`, `witness_benchmarks.json`, `halueval_qa_faithful.json`, `fusion4_spectral_benchmark.json`, `epr_benchmark.json` |
| Compression/recovery | `recovery_policy_benchmark.json`, `compression_proxy_scoreboard.py`, `anchor_compress.py` |
| Research/proofs | `proofs/knapsack/README.md`, `proofs/bipt/README.md`, `benchmarks/EICV_PREREGISTRATION.md` |
| Real code workloads | `bench/swebench_real.py`, `bench/repobench_retrieval.py`, `bench/swebench_real_result.json` |

## When to use each path

| Need | Path |
|---|---|
| Quick confidence check | `entroly verify-claims && entroly simulate` |
| Claude Code subscription workflow | `claude mcp add entroly -- entroly` |
| API-key proxy workflow | `entroly proxy` |
| Programmatic compression | Python SDK |
| Node/WASM workflow | npm `entroly` / `entroly-wasm` |
| CI budget enforcement | `entroly batch --budget 8000 --fail-over-budget` |
| Audit-heavy document/code review | Context Receipts + WITNESS |

## Honest boundary

Entroly should not claim savings when there is nothing to compress. Small prompts and tiny repos should pass through. `simulate` estimates token reduction locally and does not judge final answer quality. Quality claims should be tied to committed benchmark JSON, local verification output, or provider-backed evaluation reports.
