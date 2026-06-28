# Entroly Memory OS

Entroly Memory OS is the product surface for Entroly's memory ecosystem.

It should be explained as more than persistent storage. The differentiator is that Entroly decides **what should be remembered, recalled, suppressed, shared, verified, persisted, and sent under a token budget**.

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
7. prevent memory traffic from leaking secrets, PII, or prompt-injection payloads,
8. bound memory growth,
9. persist memory locally and atomically,
10. share lessons across agents only when sharing is useful,
11. preserve privacy when learning across installations,
12. verify whether generated answers are supported by evidence.

Entroly is built around that runtime problem.

## Product promise

```text
Not just "remember and recall".
Entroly controls memory the way an operating system controls CPU, cache, IO, and permissions.
```

## Architecture map

| Layer | What it does | Primary files |
|---|---|---|
| Public facade | Stable Python API for remember/recall/decay/consolidation/save/load/safety | `entroly/memory.py` |
| Standalone CLI | Local memory CLI entrypoint for demos and scripts | `entroly/memory_cli.py` |
| Working / episodic / semantic memory | Three-tier memory with token budgets | `entroly-core/src/memory/mod.rs`, `episode.rs` |
| Salience and forgetting | Ebbinghaus retention, emotional tags, spaced recall | `entroly-core/src/memory/episode.rs` |
| Neocortex | Kanerva Sparse Distributed Memory for consolidated patterns | `entroly-core/src/memory/kanerva.rs` |
| Fast recall | Multi-probe LSH over 1024-bit SimHash addresses | `entroly-core/src/memory/lsh.rs` |
| Sleep replay | Consolidates important memories and evicts weak ones | `entroly-core/src/memory/consolidation.rs` |
| Context memory bridge | Recalls long-term memories into context selection | `entroly/long_term_memory.py` |
| Inter-agent message memory | Suppresses redundant agent messages before token explosion | `entroly-core/src/ipc.rs` |
| Safety gate | Blocks PII, prompt injection, and rate-limit abuse | `entroly-core/src/compliance.rs`, `entroly/memory.py` |
| Pollination | Shares learned lessons across agents with TD(0) feedback | `entroly-core/src/pollination.rs` |
| Federation | Shares learned archetype weights with differential privacy | `entroly/federation.py` |
| Verification | Checks whether answers are grounded in selected evidence | `entroly/witness.py`, `entroly-core/src/witness.rs` |
| Context control | Selects what memory/code enters the prompt | `entroly-core/src/lib.rs`, `knapsack.rs`, `knapsack_sds.rs`, `channel.rs` |

## The four memory jobs

### 1. Remember

Entroly stores memories with an explicit tier, salience, token cost, source, tags, and safety policy.

The public facade supports:

- `working`: current task memory,
- `episodic`: session/history memory,
- `semantic`: persistent pattern memory.

Semantic memory is intentionally protected from normal forgetting, but the runtime still has global capacity limits.

### 2. Recall

Recall is not a blind nearest-neighbor dump. Entroly scores candidates and selects memories under a token budget.

The public facade combines:

- query overlap across content, source path, and tags,
- Ebbinghaus retention,
- recall frequency,
- tier bonus,
- importance,
- score-per-token ordering,
- explicit over-budget omission reasons.

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

The deeper stack includes:

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
| Local safety guard before storage | Product-dependent | Public facade blocks/redacts secrets, PII, injection patterns |
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
| MemoryOS Python facade | Shipped | Dependency-free public API: remember, recall, decay, consolidate, save, load, snapshot, safety scan |
| Standalone memory CLI module | Shipped | `python -m entroly.memory_cli`; main `entroly memory ...` dispatcher still needs wiring |
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

## Product gap now closed

The first production gap is now closed with:

1. one name: **Entroly Memory OS**,
2. one public Python facade: `MemoryOS`,
3. bounded memory growth,
4. local safety scanning before storage,
5. durable atomic save/load,
6. selected/omitted receipts,
7. production tests for safety, persistence, invalid snapshots, capacity, and budget omissions,
8. one maturity matrix,
9. one demo narrative,
10. one benchmark story,
11. one comparison against graph-memory tools.

Remaining gaps:

1. wire `python -m entroly.memory_cli` into the main `entroly memory ...` command,
2. expose native Rust `MemoryManager`, `IpcBus`, `ComplianceGate`, and `PollinationEngine` as public PyO3 classes,
3. add a memory-specific benchmark,
4. add a public README section linking to this guide.

## Public API

A simple public API is available through `entroly.memory.MemoryOS` and exported from `entroly`:

```python
from entroly import MemoryOS

mem = MemoryOS(max_entries=50_000, max_tokens=500_000, safety_policy="block")
mem.remember(
    agent_id="coder",
    content="Auth timeout bug was fixed in auth/session.py",
    importance=0.9,
    source="incident/auth-timeout",
    tags=["critical"],
)

ctx = mem.recall(
    agent_id="coder",
    query="why is login timing out again?",
    budget=1200,
)

print(ctx.as_text())
print(ctx.receipt())
mem.save(".entroly/memory.json")
```

Load it later:

```python
from entroly import MemoryOS

mem = MemoryOS.load(".entroly/memory.json")
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
- max entries and max token capacity,
- secret/PII/prompt-injection scan with block/redact/allow policies,
- atomic save/load,
- snapshot/restore.

The facade should later delegate to existing native primitives:

- Rust `MemoryManager` for tiered recall,
- Entroly context optimizer for token-aware selection,
- WITNESS for answer verification,
- SCHIPC for multi-agent filtering,
- ComplianceGate for safe memory traffic,
- FederationClient for opt-in shared learning.

## Standalone CLI

Until the main monolithic CLI is wired, use:

```bash
python -m entroly.memory_cli remember "Login timeout was fixed in auth/session.py" \
  --agent coder \
  --importance 0.9 \
  --source incident/auth-timeout \
  --tag critical

python -m entroly.memory_cli recall "why is login timing out again?" \
  --agent coder \
  --budget 1200

python -m entroly.memory_cli stats
```

Use `ENTROLY_MEMORY=/path/to/memory.json` or `--file /path/to/memory.json` to choose the memory file.

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
Entroly is not just a memory store. It is a local memory-control runtime for agents: budget-aware recall, decay, consolidation, safety scanning, durable local persistence, receipts, and verification.
```

Avoid this claim until the native API and benchmark are added:

```text
Entroly is universally better than all AI memory platforms.
```

The better claim is sharper and more defensible:

```text
Graph-memory tools build a company brain. Entroly controls the agent's working memory and context budget at runtime.
```
