# Entroly Memory OS

Entroly Memory OS is the product surface for Entroly's memory ecosystem.

It should be explained as more than persistent storage. The differentiator is that Entroly decides **what should be remembered, recalled, suppressed, shared, verified, and sent under a token budget**.

## One-line positioning

**Entroly Memory OS gives AI agents a budget-aware working memory, long-term memory, verifier, and safe multi-agent nervous system.**

## Why this exists

Most agent-memory products focus on storing and retrieving facts. That is necessary, but incomplete for coding agents and multi-agent systems.

A production agent does not only need to remember. It needs to:

1. avoid sending redundant context,
2. remember only high-value evidence,
3. recall memories under a context budget,
4. separate working, episodic, and semantic memory,
5. consolidate important memories instead of retaining everything,
6. suppress stale or low-retention memories,
7. prevent memory traffic from leaking PII or prompt-injection payloads,
8. share lessons across agents only when sharing is useful,
9. preserve privacy when learning across installations,
10. verify whether generated answers are supported by evidence.

Entroly is built around that runtime problem.

## Product promise

```text
Not just "remember and recall".
Entroly controls memory the way an operating system controls CPU, cache, IO, and permissions.
```

## Architecture map

| Layer | What it does | Primary files |
|---|---|---|
| Public facade | Stable Python API for remember/recall/decay/consolidation | `entroly/memory.py` |
| Working / episodic / semantic memory | Three-tier memory with token budgets | `entroly-core/src/memory/mod.rs`, `episode.rs` |
| Salience and forgetting | Ebbinghaus retention, emotional tags, spaced recall | `entroly-core/src/memory/episode.rs` |
| Neocortex | Kanerva Sparse Distributed Memory for consolidated patterns | `entroly-core/src/memory/kanerva.rs` |
| Fast recall | Multi-probe LSH over 1024-bit SimHash addresses | `entroly-core/src/memory/lsh.rs` |
| Sleep replay | Consolidates important memories and evicts weak ones | `entroly-core/src/memory/consolidation.rs` |
| Context memory bridge | Recalls long-term memories into context selection | `entroly/long_term_memory.py` |
| Inter-agent message memory | Suppresses redundant agent messages before token explosion | `entroly-core/src/ipc.rs` |
| Safety gate | Blocks PII, prompt injection, and rate-limit abuse | `entroly-core/src/compliance.rs` |
| Pollination | Shares learned lessons across agents with TD(0) feedback | `entroly-core/src/pollination.rs` |
| Federation | Shares learned archetype weights with differential privacy | `entroly/federation.py` |
| Verification | Checks whether answers are grounded in selected evidence | `entroly/witness.py`, `entroly-core/src/witness.rs` |
| Context control | Selects what memory/code enters the prompt | `entroly-core/src/lib.rs`, `knapsack.rs`, `knapsack_sds.rs`, `channel.rs` |

## The four memory jobs

### 1. Remember

Entroly stores memories with an explicit tier, salience, token cost, emotional tag, and binary address.

The memory manager supports:

- `working`: current task memory,
- `episodic`: session/history memory,
- `semantic`: persistent pattern memory.

Semantic memory is intentionally protected from normal forgetting.

### 2. Recall

Recall is not a blind nearest-neighbor search. Entroly scores candidates and selects memories under a token budget.

This matters because the model does not need every memory. It needs the few memories that fit the task and budget.

### 3. Consolidate

Entroly uses sleep-replay style consolidation:

- weak memories decay,
- frequently recalled high-retention memories promote,
- working memory can become episodic,
- episodic memory can become semantic,
- semantic memory is protected.

### 4. Share

For multi-agent systems, Entroly treats memory sharing as a decision, not a broadcast.

The stack includes:

- SCHIPC novelty filter for redundant messages,
- compliance gate for PII, injection, and rate limits,
- pollination engine for sharing lessons,
- TD(0) feedback so agents learn when sharing helps.

## How this is different from graph memory

Graph/vector memory tools are strongest when the problem is:

```text
Build a company brain from documents and relationships.
```

Entroly is strongest when the problem is:

```text
Run an AI agent without drowning it in noisy, unsafe, stale, or unaudited context.
```

| Capability | Graph-memory platforms | Entroly Memory OS |
|---|---|---|
| Store long-term facts | Strong | Present through memory + vault surfaces |
| Knowledge graph ontology | Strong | Not Entroly's main product surface today |
| Token-budget-aware recall | Usually secondary | Core design goal |
| Working / episodic / semantic tiers | Usually abstract | Explicit runtime model |
| Sleep-replay consolidation | Usually not central | Built into the memory design |
| Forgetting and salience | Often manual metadata | Native retention model |
| Inter-agent message filtering | Usually not central | SCHIPC novelty filter |
| PII / injection gate on memory traffic | Product-dependent | Built as kernel compliance gate |
| Agent lesson sharing | Usually high-level | TD(0) pollination engine |
| Privacy-preserving shared learning | Product-dependent | DP-noised archetype federation |
| Answer verification | Usually external | WITNESS gateway in Entroly stack |
| Context receipts | Usually external | Native Entroly differentiator |

## Shipped, internal, and experimental surfaces

Entroly should be honest about maturity.

| Surface | Status | Notes |
|---|---|---|
| MemoryOS Python facade | Shipped | Dependency-free public API: remember, recall, decay, consolidate, snapshot |
| Context selection and optimization | Shipped | Public CLI/proxy/library path |
| Context Receipts | Shipped | Public Python + Rust-backed receipt pipeline |
| WITNESS verification | Shipped | Python gateway with Rust verifier support |
| RAVS guarded routing | Shipped / gated | Fail-closed router behavior covered by tests |
| Long-term memory Python bridge | Optional | Activates when `hippocampus-sharp-memory` is installed |
| Rust memory manager | Internal core surface | Present in `entroly-core/src/memory`, but should be exposed more deeply through PyO3 |
| SCHIPC IPC bus | Internal core surface | Present in Rust; needs public examples |
| Compliance gate | Internal core surface | Present in Rust; needs public examples |
| Pollination engine | Internal core surface | Present in Rust; needs integration guide |
| Federation | Experimental / opt-in | Off by default; shares no code, paths, or fingerprints |

## Product gap to close

The strongest technology is already in the repository, but it was not presented as a coherent product.

The first gap is now closed with:

1. one name: **Entroly Memory OS**,
2. one public Python facade: `MemoryOS`,
3. one maturity matrix,
4. one demo narrative,
5. one benchmark story,
6. one comparison against graph-memory tools.

The remaining product gaps are:

1. expose native Rust `MemoryManager`, `IpcBus`, `ComplianceGate`, and `PollinationEngine` as public PyO3 classes,
2. add CLI commands for `entroly memory remember`, `entroly memory recall`, and `entroly memory stats`,
3. add a memory-specific benchmark,
4. add a public README section linking to this guide.

## Public API

A simple public API is available through `entroly.memory.MemoryOS` and exported from `entroly`:

```python
from entroly import MemoryOS

mem = MemoryOS()
mem.remember(agent_id="coder", content="Auth timeout bug was fixed in auth/session.py", importance=0.9)

ctx = mem.recall(
    agent_id="coder",
    query="why is login timing out again?",
    budget=1200,
)

print(ctx.as_text())
print(ctx.receipt())
```

Current facade behavior:

- dependency-free,
- local-only,
- no embeddings API,
- exact deduplication,
- working/episodic/semantic tiers,
- Ebbinghaus-style retention,
- recall reinforcement,
- budget-aware recall,
- selected/omitted memory receipt,
- snapshot/restore.

The facade should later delegate to existing native primitives:

- Rust `MemoryManager` for tiered recall,
- Entroly context optimizer for token-aware selection,
- WITNESS for answer verification,
- SCHIPC for multi-agent filtering,
- ComplianceGate for safe memory traffic,
- FederationClient for opt-in shared learning.

## Demo narrative

Use this story in videos and README sections:

```text
Cognee-like systems give agents a company brain.
Entroly gives agents a working memory, long-term memory, verifier, and safe multi-agent nervous system.
```

Then show this flow:

```text
User asks task
   ↓
Entroly recalls working + episodic + semantic memories under budget
   ↓
SCHIPC suppresses redundant agent chatter
   ↓
ComplianceGate blocks unsafe memory traffic
   ↓
Context optimizer selects code + memory + receipts
   ↓
WITNESS checks answer against evidence
   ↓
Pollination learns whether sharing helped
```

## Demo script

### Setup

```bash
pip install entroly[full]
cd your-large-repo
entroly doctor
entroly simulate
```

### Show the public MemoryOS facade

```python
from entroly import MemoryOS

mem = MemoryOS(default_budget=1200)
mem.remember(
    "Login timeout was fixed in auth/session.py by increasing refresh slack.",
    agent_id="coder",
    importance=0.9,
    source="incident/auth-timeout",
    tags=["critical"],
)

ctx = mem.recall("why is login timing out again?", agent_id="coder", budget=1200)
print(ctx.as_text())
print(ctx.receipt())
```

### Show context memory

```bash
entroly proxy --quality balanced
```

Ask through an agent:

```text
We fixed login timeout last week. Find the relevant code paths and explain what context matters now.
```

Show:

- selected code fragments,
- remembered high-value historical fragments,
- omitted low-value fragments,
- receipt/risk summary,
- WITNESS result.

### Show memory OS differentiator

On screen, use this split:

| Generic memory | Entroly Memory OS |
|---|---|
| Stores facts | Selects memories under token budget |
| Recalls by similarity | Recalls by relevance + salience + retention + cost |
| Shares context | Suppresses redundant agent chatter |
| Adds more memory | Forgets, consolidates, and verifies |
| Trusts retrieval | Emits receipts and WITNESS evidence |

## Benchmark story to add next

Entroly needs a memory-specific benchmark, separate from token benchmarks.

Proposed benchmark: **Agent Memory Stress Test**

Scenarios:

1. long project with repeated bug recurrence,
2. stale memory that should be forgotten,
3. high-salience safety memory that must survive,
4. two agents sharing repeated tool results,
5. injected secret/prompt injection in memory traffic,
6. limited token budget recall,
7. answer verification from recalled memory.

Metrics:

- recall precision,
- stale-memory suppression,
- token budget compliance,
- redundant message suppression rate,
- unsafe memory blocked,
- WITNESS grounded/unsupported ratio,
- cost saved from suppressed/reused context.

## Current strongest claim

Use this claim publicly:

```text
Entroly is not just a memory store. It is a local memory-control runtime for agents: budget-aware recall, decay, consolidation, safe sharing, receipts, and verification.
```

Avoid this claim until the public API and benchmark are added:

```text
Entroly is universally better than all AI memory platforms.
```

The better claim is sharper and more defensible:

```text
Graph-memory tools build a company brain. Entroly controls the agent's working memory and context budget at runtime.
```
