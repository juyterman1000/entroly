<p align="center">
  <img src="docs/assets/logo.png" width="200" alt="Entroly Logo">
</p>

<h1 align="center">Entroly</h1>

<p align="center">
  <b>Information-theoretic context compression for AI coding agents.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-Engine-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Tests-259%20Passing-success" alt="Tests">
</p>

---

Every AI coding tool — Cursor, Copilot, Claude Code, Cody — manages context with dumb heuristics: stuff tokens until the window fills, then cut. Entroly uses mathematics to compress an **entire codebase** into the optimal context window.

## 🛑 The Problem
Current AI tools use **Cosine Similarity** (Vector Search). It's great for finding "things that look like my query," but terrible for coding because:
1. **Context Blindness**: It finds the "top 5 files" but missed the 6th file that contains the critical interface definition.
2. **Boilerplate Waste**: 40% of retrieved code is often imports or repetitive boilerplate, wasting expensive tokens.
3. **Correlation vs Causation**: Vector search finds *related* code, not *causally necessary* code.

## ✅ The Solution: Entroly
Entroly replaces "dumb search" with **Information-Theoretic Compression**. It treats your context window as a finite resource and uses **Knapsack Optimization** to pack the most "informative" (highest entropy) and "causally relevant" (dependency-linked) fragments.

---


```
pip install entroly
```

## How It's Different

**Sourcegraph Cody** does *search*: "Find 5–10 files that look relevant." **Entroly** does *compression*: "Show the LLM the **entire codebase** at variable resolution."

| | Cody / Copilot | Entroly |
|--|----------------|---------|
| **Approach** | Embedding similarity search | Information-theoretic compression |
| **Coverage** | 5–10 files (the rest is invisible) | 100% codebase visible via 3-level hierarchy |
| **Selection** | Top-K by cosine distance | Knapsack-optimal with submodular diversity |
| **Dedup** | None | SimHash + LSH in O(1) |
| **Learning** | Static | Online Wilson-score feedback + autotune |
| **Security** | None | Built-in SAST (55 rules, taint-aware) |
| **Temperature** | User-set or model default | Auto-calibrated via Fisher information |

## Architecture

Hybrid Rust + Python. All math runs in Rust via PyO3 (50–100× faster). MCP protocol and orchestration run in Python. Pure Python fallbacks activate automatically if the Rust extension isn't available.

```
┌─────────────────────────────────────────────────────────┐
│  IDE (Cursor / Claude Code / Cline / Copilot)           │
│                                                         │
│  ┌──── MCP mode ────┐    ┌──── Proxy mode ────┐        │
│  │ entroly MCP server│    │ localhost:9377     │        │
│  │ (JSON-RPC stdio)  │    │ (HTTP reverse proxy)│       │
│  └────────┬──────────┘    └────────┬───────────┘        │
│           │                        │                    │
│  ┌────────▼────────────────────────▼───────────┐        │
│  │          Entroly Engine (Python)             │        │
│  │  ┌─────────────────────────────────────┐     │       │
│  │  │  entroly-core (Rust via PyO3)       │     │       │
│  │  │  14 modules · 330 KB · 93 tests     │     │       │
│  │  └─────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

Two deployment modes:
- **MCP Server** — IDE calls `remember_fragment`, `optimize_context`, etc. via MCP protocol
- **Prompt Compiler Proxy** — invisible HTTP proxy at `localhost:9377`, intercepts every LLM request and auto-optimizes (zero IDE changes beyond API base URL)

## Engines

### Rust Core (14 modules)

| Module | What | How |
|--------|------|-----|
| **hierarchical.rs** | 3-level codebase compression (ECC) | L1: skeleton map of ALL files · L2: dep-graph cluster expansion · L3: knapsack-optimal full fragments with submodular diversity |
| **knapsack.rs** | Optimal context subset selection | 0/1 Knapsack DP (N ≤ 2000) · greedy with Dantzig 0.5-guarantee (N > 2000) |
| **entropy.rs** | Information density scoring | Shannon entropy (40%) + boilerplate detection (30%) + cross-fragment n-gram redundancy (30%) |
| **depgraph.rs** | Dependency graph + symbol table | Auto-linking: imports (1.0) · type refs (0.9) · function calls (0.7) · same-module (0.5) |
| **skeleton.rs** | AST-lite code skeleton extraction | Preserves signatures, class/struct/trait layouts, strips bodies → 60–80% token reduction |
| **dedup.rs** | Near-duplicate detection | 64-bit SimHash fingerprints · Hamming threshold ≤ 3 · 4-band LSH buckets |
| **lsh.rs** | Semantic recall index | 12-table multi-probe LSH · 10-bit sampling · ~3 μs over 100K fragments |
| **sast.rs** | Static Application Security Testing | 55 rules across 8 CWE categories · taint-flow analysis · severity scoring |
| **health.rs** | Codebase health analysis | Clone detection (Type-1/2/3) · dead symbol finder · god file detector · arch violation checker |
| **guardrails.rs** | Safety-critical file pinning | Criticality levels (Safety/Critical/Important/Normal) · task-aware budget multipliers |
| **prism.rs** | Spectral weight optimizer | Jacobi eigendecomposition on 4×4 covariance matrix · anisotropic gain adaptation |
| **query.rs** | Query analysis + refinement | Vagueness scoring · keyword extraction · intent classification |
| **fragment.rs** | Core data structure | Content, metadata, scoring dimensions, skeleton, SimHash fingerprint |
| **lib.rs** | PyO3 bridge + orchestrator | All modules exposed to Python · 93 tests |

### Python Layer

| Module | What |
|--------|------|
| **proxy.py** | Invisible HTTP reverse proxy (prompt compiler mode) |
| **proxy_transform.py** | Request parsing · context formatting (flat + hierarchical) · EGTC · APA |
| **proxy_config.py** | Model context windows · all feature flags · autotune overlay |
| **server.py** | MCP server with 10+ tools · pure Python fallbacks |
| **long_term_memory.py** | Cross-session memory via hippocampus-sharp-memory integration |
| **multimodal.py** | Image OCR · diagram parsing (Mermaid/PlantUML/DOT) · voice transcript extraction |
| **autotune.py** | Autonomous hyperparameter optimization (mutate → evaluate → keep/discard) |
| **auto_index.py** | File-system crawler for automatic codebase indexing |
| **adaptive_pruner.py** | Online RL-based fragment pruning |
| **checkpoint.py** | Gzipped JSON state serialization (~100 KB per checkpoint) |
| **prefetch.py** | Predictive context pre-loading via import analysis + co-access patterns |
| **provenance.py** | Hallucination risk detection via source verification + confidence scoring |

## Novel Algorithms

### Entropic Context Compression (ECC)

Three-level hierarchical codebase compression. The LLM sees **everything** at variable resolution:

```mermaid
graph TD
    Query["User Query"] --> L1["L1: Skeleton Map (5%)"]
    Query --> L2["L2: Dependency Cluster (25%)"]
    Query --> L3["L3: Full Fragments (70%)"]
    
    L1 --> C1["Signatures of ALL 1000+ files"]
    L2 --> C2["Expanded skeletons of related modules"]
    L3 --> C3["Submodular diversified full code"]
    
    C1 --> Context["Optimal Context Window"]
    C2 --> Context
    C3 --> Context
```

Novel techniques:
1. **Symbol-reachability slicing** — BFS through dep graph from query-relevant symbols (cf. NeurIPS 2025)
2. **Submodular diversity selection** — diminishing returns per module (Nemhauser 1978, 1-1/e guarantee)
3. **PageRank centrality** — hub files get priority in L2 expansion
4. **Entropy-gated budget allocation** — complex codebases get more L3 budget

### EGTC v2 (Entropy-Gap Temperature Calibration)

Automatically derives the optimal LLM sampling temperature from information-theoretic properties of the selected context. Uses Fisher information scaling with 4 signals:

```
τ = clip(τ_base + Σ signal_weights × [vagueness, entropy_gap, sufficiency, task_type])
```

### APA (Adaptive Prompt Augmentation)

1. **Calibrated token estimation** — per-language chars/token ratios (Python: 3.0, Rust: 3.5, ...)
2. **Task-aware preamble** — conditional hints from security findings, vagueness, and task type
3. **Content deduplication** — MD5 hash-based dedup saves 10–20% in multi-turn sessions

## Setup

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "entroly": {
      "command": "entroly"
    }
  }
}
```

### Claude Code

```bash
claude mcp add entroly -- entroly
```

### Prompt Compiler Proxy (any IDE)

Change your IDE's API base URL to `http://localhost:9377`:

```bash
entroly --proxy
# or
ENTROLY_PROXY_PORT=9377 python -m entroly.proxy
```

Every LLM request is intercepted, optimized with the full pipeline (ECC + EGTC + APA + SAST), and forwarded transparently. < 10ms overhead.

### Docker (cross-platform)

```bash
docker pull ghcr.io/juyterman1000/entroly:latest
docker run --rm -p 9377:9377 ghcr.io/juyterman1000/entroly:latest
```

Multi-arch image: `linux/amd64` and `linux/arm64` (Apple Silicon, AWS Graviton).

## MCP Tools

| Tool | Purpose |
|------|---------|
| `remember_fragment` | Store context with auto-dedup, entropy scoring, dep linking, criticality detection |
| `optimize_context` | Select optimal context subset for a token budget (knapsack + ECC) |
| `recall_relevant` | Sub-linear semantic recall via multi-probe LSH |
| `record_outcome` | Feed the Wilson-score feedback loop |
| `explain_context` | Per-fragment scoring breakdown with sufficiency analysis |
| `checkpoint_state` | Save full session state (gzipped JSON) |
| `resume_state` | Restore from checkpoint |
| `prefetch_related` | Predict and pre-load likely-needed context |
| `get_stats` | Session statistics and cost savings |
| `health_check` | Clone detection, dead symbols, god files, arch violations |

## The Math

### Multi-Dimensional Relevance Scoring

```
r(f) = (w_rec · recency + w_freq · frequency + w_sem · semantic + w_ent · entropy)
       / (w_rec + w_freq + w_sem + w_ent)
       × feedback_multiplier
```

- **Recency**: Ebbinghaus forgetting curve — `exp(-ln(2) × Δt / half_life)`
- **Frequency**: Normalized access count with spaced repetition boost
- **Semantic similarity**: SimHash Hamming distance to query, normalized to [0, 1]
- **Information density**: Shannon entropy + boilerplate + redundancy

### Knapsack Context Selection

```
Maximize:   Σ r(fᵢ) · x(fᵢ)     for selected fragments
Subject to: Σ c(fᵢ) · x(fᵢ) ≤ B  (token budget)
```

- **N ≤ 2000**: Exact DP with budget quantization — O(N × 1000)
- **N > 2000**: Greedy density sort — O(N log N), Dantzig 0.5-optimality guarantee

### SAST Security Categories

| Category | CWE | Rules |
|----------|-----|-------|
| Hardcoded Secrets | CWE-798 | API keys, passwords, tokens, private keys |
| SQL Injection | CWE-89 | f-strings, concatenation, raw queries (taint-aware) |
| Command Injection | CWE-78 | os.system, subprocess with shell=True |
| Path Traversal | CWE-22 | open() with user input, os.path.join |
| XSS | CWE-79 | innerHTML, template injection |
| SSRF | CWE-918 | requests with user-controlled URLs |
| Insecure Crypto | CWE-327 | MD5/SHA1 for auth, weak key sizes |
| Auth Flaws | CWE-287 | Hardcoded roles, missing auth checks |

## Configuration

```python
EntrolyConfig(
    default_token_budget=128_000,     # GPT-4 Turbo equivalent
    max_fragments=10_000,             # session fragment cap
    weight_recency=0.30,              # scoring weights (sum to 1.0)
    weight_frequency=0.25,
    weight_semantic_sim=0.25,
    weight_entropy=0.20,
    decay_half_life_turns=15,         # Ebbinghaus half-life
    enable_hierarchical_compression=True,  # 3-level ECC
    enable_temperature_calibration=True,   # EGTC v2
    enable_prompt_directives=True,         # APA preamble
    enable_security_scan=True,             # SAST
)
```

## References

- Shannon (1948) — Information Theory
- Charikar (2002) — SimHash
- Ebbinghaus (1885) — Forgetting Curve
- Weiser (1981) — Program Slicing
- Nemhauser, Wolsey & Fisher (1978) — Submodular Maximization
- Dantzig (1957) — Greedy Knapsack Approximation
- Wilson (1927) — Score Confidence Intervals
- LLMLingua (EMNLP 2023) — Perplexity-based Token Compression
- LongLLMLingua (ACL 2024) — Query-aware Context Compression
- RepoFormer (ICML 2024 Oral) — Selective Retrieval for Repo-Level Code
- FILM-7B (NeurIPS 2024) — Structure-First Layout
- CodeSage (ICLR 2024) — Code Embedding Representation Learning
- SWE-bench (ICLR 2024) / SWE-agent (NeurIPS 2024) — Evaluation

## Part of the Ebbiforge Ecosystem

Entroly integrates with [hippocampus-sharp-memory](https://pypi.org/project/hippocampus-sharp-memory/) for persistent cross-session memory and [Ebbiforge](https://pypi.org/project/ebbiforge/) for TF embeddings and RL weight learning. Both are optional.

## License

MIT
