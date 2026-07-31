# Competitive reality check — 2026-07-30

This note prevents Entroly's public story from drifting into claims that current
projects have already falsified. It uses project repositories, documentation,
and papers as primary sources. Popularity is a discovery signal, not technical
evidence.

## One-sentence wedge

**Entroly reduces the context an existing agent receives under a hard budget,
while every selected or omitted source fragment carries an independently
recomputable source-span digest and a local exact-recovery path.**

That sentence is deliberately narrower than “reversible context compression.”
Headroom and LeanCTX already make public recovery and cache-stability claims.
Entroly's defensible distinction is the public source-authentication contract:
source digest, byte range, fragment digest, omission record, and fail-closed
recovery verification are emitted together.

## Representative systems

| Project | Primary public focus | Capability that falsifies a broad Entroly claim | Evidence to learn from |
|---|---|---|---|
| [Headroom](https://github.com/headroomlabs-ai/headroom) | Local context optimization across proxy, library, MCP, and agent integrations | Publicly documents reversible CCR, cached originals, cache-aware handling, and broad integrations. Entroly cannot claim recovery or cache safety is unique | Extensive integration surface, active releases, and benchmark-oriented README |
| [LeanCTX](https://github.com/yvgude/lean-ctx) | Deterministic context compression and recovery for coding agents | Publicly documents content-addressed handles, several recovery paths, prompt-cache-safe behavior, and an audited savings ledger | Tight install story, byte-stability language, and concrete agent compatibility |
| [LLMLingua](https://github.com/microsoft/LLMLingua) | Learned prompt compression | Demonstrates that learned token-level and coarse-to-fine compression are established research directions | Peer-reviewed papers, released code, and standard datasets |
| [RECOMP](https://github.com/carriex/recomp) | Selective and abstractive compression for retrieval-augmented language models | Demonstrates query-conditioned compression is established prior art | Clear task-level evaluation against retrieval baselines |
| [AutoCompressors](https://github.com/princeton-nlp/AutoCompressors) | Long-context compression into summary vectors | Demonstrates learned soft compression and recursive compression are prior art | Reproducible research code tied to a paper |
| [Selective Context](https://github.com/liyucheng09/Selective_Context) | Self-information-based prompt pruning | Demonstrates information-theoretic context selection is established prior art | Simple, inspectable selection rule and published evaluation |
| [Letta](https://github.com/letta-ai/letta) | Stateful agents and memory management | Falsifies any claim that persistent agent memory alone creates a new category | Coherent agent-memory product model and public SDK surface |
| [Mem0](https://github.com/mem0ai/mem0) | Cross-agent long-term memory | Falsifies any generic “universal memory layer” positioning | Fast onboarding, open evaluation framework, SDKs, CLI, and integrations |
| [Graphiti](https://github.com/getzep/graphiti) | Temporal context graphs for agents | Falsifies a generic provenance or relationship-graph novelty claim | Temporal validity, episode provenance, hybrid retrieval, and MCP |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/context-compression-and-caching.md) | Agent runtime with pluggable context engines | Shows compression, prompt caching, and replaceable context engines can live inside the agent harness | A clean extension boundary; Entroly should remain complementary |

## What Entroly should and should not say

| Safe when linked to evidence | Not established |
|---|---|
| A public receipt contains exact UTF-8 source and fragment SHA-256 values plus byte ranges | “Best compression” or “better than Headroom” |
| Recovery fails visibly when a receipt digest, bundle digest, source identity, or byte range is changed | Generated-answer accuracy improvement |
| A hard token budget and omission list are part of the same deterministic artifact | Provider cost savings for SDK or MCP operations that were not observed reaching a provider |
| Python and native backends are tested against the same source-span contract | Novelty of reversible compression, memory, knowledge graphs, or learned selection |
| Runaway-session rescue is available for traffic routed through the Entroly proxy | Rescue of sessions that never pass through Entroly |

## Baselines for future evaluations

Use baselines by the job they perform; do not construct one winner-take-all
scoreboard.

- Retrieval and evidence selection: raw/full context, BM25, Selective Context,
  and a pinned learned retriever.
- Prompt compression: raw/full context, LLMLingua, and a deterministic
  truncation or summarization baseline.
- Coding-agent context delivery: raw repository tools, the agent's native
  compaction, and any integration being claimed.
- Long-term memory: full transcript, Mem0, and Graphiti where their documented
  deployment assumptions fit the task.
- Recovery and integrity: Headroom and LeanCTX only after pinning exact versions
  and defining a shared, black-box source-span oracle.

Entroly should publish a comparison only after the competing versions,
configuration, hardware, warm-up, task set, exclusions, and failure handling
are preregistered. Missing integrations and setup failures remain outcomes; they
are not silently removed.

## Product consequence

The priority is not another compression algorithm. It is making the
source-authenticated receipt contract unavoidable across SDK, MCP, proxy, Rust,
and packaged installs, then proving whether that contract improves real task
outcomes. Until a model-in-the-loop study exists, the public claim stays at
evidence delivery and exact recovery.
