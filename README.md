# Entroly

[![PyPI](https://img.shields.io/pypi/v/entroly)](https://pypi.org/project/entroly/)
[![CI](https://github.com/juyterman1000/entroly/actions/workflows/ci.yml/badge.svg)](https://github.com/juyterman1000/entroly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/engine-100%25%20Rust-orange)](entroly-core/)

### Your AI coding tool wastes 40–60% of its context window on irrelevant files. Entroly fixes that.

When you ask Cursor to "fix the SQL injection bug," it stuffs your context window with README.md, CSS, changelogs, and duplicate boilerplate — then drops the actual database code because it ran out of room.

**Entroly is an MCP server that selects the mathematically optimal subset of context for every query.** Your AI sees the right code, not all the code.

<div align="center">
  <br/>

  https://github.com/juyterman1000/entroly/raw/main/docs/assets/entroly_demo.mp4

  <p><i>Live engine metrics from <code>entroly demo</code> — real Rust engine, zero mocks.</i></p>
</div>

---

## What You Get

| Metric | Without Entroly | With Entroly |
|--------|:-:|:-:|
| **Context relevance** | ~50% (FIFO truncation) | **91%** (knapsack-optimized) |
| **Cost per API call** | $0.0115 | **$0.0044** (62% savings) |
| **Duplicate detection** | None | **Automatic** (SimHash) |
| **Selection speed** | N/A | **320µs** (sub-millisecond) |
| **Crash recovery** | Lost | **Checkpoint & resume** |

At scale (100K+ files, 128K token budget), these savings compound: more noise to filter = bigger improvement.

---

## Quick Start (30 seconds)

```bash
pip install entroly
cd your-project
entroly init        # auto-detects your AI tool, writes mcp.json
# Restart your AI tool — done.
```

> **🐳 Mac / Windows users:** Entroly auto-detects [Docker Desktop](https://docker.com/products/docker-desktop) and runs the Rust engine inside a container — **zero compilation, zero Rust toolchain, just works.** Make sure Docker Desktop is running when you start your AI tool.

`entroly init` auto-detects **Cursor, VS Code, Windsurf, and Claude Code** and writes the correct MCP config for you. For other MCP clients:

<details>
<summary>Manual MCP configuration</summary>

**Claude Code:**
```bash
claude mcp add entroly -- entroly serve
```

**Any MCP client (mcp.json):**
```json
{
  "mcpServers": {
    "entroly": {
      "command": "entroly",
      "args": ["serve"]
    }
  }
}
```

**npm (for tools that prefer npx):**
```bash
npx -y entroly-mcp
```
</details>

> **Tip:** Run `entroly demo` to see a side-by-side before/after comparison using the real Rust engine.

---

## How It Works

Entroly sits between your AI tool and the LLM as an MCP server. When your agent asks for context, Entroly:

1. **Scores** every code fragment on 4 dimensions (recency, frequency, semantic similarity, information density)
2. **Deduplicates** via 64-bit SimHash fingerprints — catches near-identical code in O(1)
3. **Solves the 0/1 Knapsack Problem** to select the optimal subset within your token budget
4. **Learns** from feedback — fragments that lead to good outputs get boosted next time

All computation runs in **100% Rust** via PyO3. The Python layer only handles MCP protocol and I/O.

```
Your AI Tool → MCP (JSON-RPC) → Python (FastMCP) → Rust Engine → Optimal Context
                                                      ↑
                                              Selection in 320µs
```

---

## MCP Tools

Once `entroly serve` is running, your AI agent has access to these tools:

| Tool | What it does |
|------|-------------|
| `optimize_context` | Select the optimal context subset for a token budget. **The core tool.** |
| `remember_fragment` | Store context with auto-dedup, entropy scoring, and security scanning |
| `recall_relevant` | Semantic search over stored fragments via multi-probe LSH |
| `record_outcome` | Feed the RL loop: mark fragments as helpful or unhelpful |
| `explain_context` | See exactly why each fragment was included or excluded |
| `prefetch_related` | Predict what files will be needed next (import analysis + co-access) |
| `checkpoint_state` | Save full session state for crash recovery |
| `resume_state` | Restore from the latest checkpoint |
| `entroly_dashboard` | Live ROI metrics — cost saved, latency, compression ratio |
| `get_stats` | Comprehensive session statistics |
| `scan_fragment` | Security scan (SQL injection, hardcoded secrets, unsafe patterns) |
| `analyze_health` | Codebase health report (clone detection, dead code, god files) |

### Example: optimize_context

```
optimize_context(token_budget=128000, query="fix payment bug")
→ {
    "selected_fragments": [...],  // The good stuff
    "tokens_saved_this_call": 42000,
    "sufficiency": 0.91,         // 91% of referenced symbols included
    "hallucination_risk": "low",
    "optimization_stats": {"method": "exact_dp", "budget_utilization": 0.73}
  }
```

### Example: entroly_dashboard

```
entroly_dashboard()
→ {
    "money": {
      "cost_per_call_without_entroly": "$0.0115",
      "cost_per_call_with_entroly": "$0.0044",
      "savings_pct": "62%"
    },
    "performance": {"avg_optimize_latency": "320µs"},
    "bloat_prevention": {"context_compression": "39%", "duplicates_caught": 12}
  }
```

---

## Try It (See the Value in 5 Seconds)

```bash
pip install entroly
entroly demo
```

This runs a simulated coding session (fixing a SQL injection bug) and shows you side-by-side what happens with and without Entroly. Uses the real Rust engine — zero mocks.

---

## Architecture

**Hybrid Rust + Python.** CPU-intensive math runs in Rust via PyO3 for 50-100x speedup. MCP protocol and orchestration run in Python via FastMCP.

| Component | What it does | How |
|-----------|-------------|-----|
| **Knapsack Optimizer** | Selects optimal context subset | Exact DP (N ≤ 2000) or greedy (N > 2000) |
| **Entropy Scorer** | Measures information density | Shannon entropy + boilerplate + cross-fragment redundancy |
| **SimHash Dedup** | Catches near-duplicates in O(1) | 64-bit fingerprints, Hamming threshold ≤ 3 |
| **Multi-Probe LSH** | Sub-linear semantic recall | 12-table LSH with multi-probe queries |
| **Dependency Graph** | Pulls related code together | Symbol table + import/type/call linking |
| **PRISM Optimizer** | Adapts weights to your codebase | Anisotropic spectral optimization (4×4 covariance) |
| **Feedback Loop** | Learns from outcomes | Wilson score confidence intervals |
| **Predictive Pre-fetch** | Pre-loads likely context | Import analysis + co-access patterns |
| **Long-Term Memory** | Cross-session recall | Ebbinghaus decay + salience boosting |
| **Security Scanner** | Finds vulnerabilities | Pattern-based SAST (SQL injection, secrets, unsafe) |
| **Health Analyzer** | Codebase quality metrics | Clone detection, dead symbols, god files |

---

## The Math (For the Curious)

<details>
<summary>Click to expand the mathematical foundations</summary>

### Multi-Dimensional Relevance Scoring

Each fragment is scored across four dimensions:

```
r(f) = (w_rec · recency + w_freq · frequency + w_sem · semantic + w_ent · entropy)
       / (w_rec + w_freq + w_sem + w_ent)
       × feedback_multiplier
```

Default weights: recency 0.30, frequency 0.25, semantic 0.25, entropy 0.20.

- **Recency**: Ebbinghaus forgetting curve — `exp(-ln(2) × Δt / half_life)`, half_life = 15 turns
- **Frequency**: Normalized access count (spaced repetition boost)
- **Semantic similarity**: SimHash Hamming distance to query, normalized to [0, 1]
- **Information density**: Shannon entropy + boilerplate + redundancy

### Knapsack Context Selection

```
Maximize:   Σ r(fᵢ) · x(fᵢ)     for selected fragments
Subject to: Σ c(fᵢ) · x(fᵢ) ≤ B  (token budget)
```

- **N ≤ 2000**: Exact DP with budget quantization into 1000 bins — O(N × 1000)
- **N > 2000**: Greedy density sort — O(N log N), Dantzig 0.5-optimality guarantee

### Task-Aware Budget Multipliers

```
Bug tracing / debugging     → 1.5× budget
Exploration / understanding → 1.3× budget
Refactoring / code review   → 1.0× budget
Code generation             → 0.7× budget
```

### PRISM Spectral Optimizer

Tracks a 4×4 covariance matrix with EMA updates (β=0.95). Jacobi eigendecomposition finds principal axes. Anisotropic spectral gain dampens noisy dimensions — automatic learning rate adaptation without hyperparameter tuning.

</details>

---

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
    min_relevance_threshold=0.05,     # auto-evict below this
    dedup_similarity_threshold=0.92,
    prefetch_depth=2,
    auto_checkpoint_interval=5,       # checkpoint every N tool calls
)
```

---

## Build from Source

If you want the Rust engine locally (instead of Docker):

```bash
git clone https://github.com/juyterman1000/entroly
cd entroly
pip install maturin
cd entroly-core && maturin develop --release && cd ..
pip install -e ".[native]"
entroly init
```

---

## References

Shannon (1948) • Charikar (2002) SimHash • Ebbinghaus (1885) Forgetting Curve • Dantzig (1957) Greedy Knapsack • Wilson (1927) Score Intervals • ICPC (2025) Prompt Compression • Proximity (2025) LSH Caching • RCC (ICLR 2025) Context Compression

## Part of the Ebbiforge Ecosystem

Entroly integrates with [hippocampus-sharp-memory](https://pypi.org/project/hippocampus-sharp-memory/) for persistent memory and [Ebbiforge](https://pypi.org/project/ebbiforge/) for TF embeddings. Both are optional — Entroly works standalone.

## License

MIT