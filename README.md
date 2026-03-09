# Entroly

[![PyPI](https://img.shields.io/pypi/v/entroly)](https://pypi.org/project/entroly/)
[![CI](https://github.com/juyterman1000/entroly/actions/workflows/ci.yml/badge.svg)](https://github.com/juyterman1000/entroly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/core-Rust%20%2B%20PyO3-orange)](entroly-core/)

**Information-theoretic context optimization for AI coding agents.**

> **The Problem:** Every AI coding tool manages context with dumb FIFO truncation. When you ask it to fix a SQL injection, it stuffs the context window with your CSS files, READMEs, and Docker configs until it's full, then cuts off the actual database connection code you needed. You get a broken answer, waste API credits, and have to re-prompt.
> 
> **The Solution:** Entroly applies mathematics to select the **optimal** context subset. It sees the *right* code, not just *all* the code.

<div align="center">
  <br/>
  <h2>Watch Entroly in Action (Real Engine Metrics)</h2>
  
  https://github.com/juyterman1000/entroly/raw/main/entroly_demo.mp4

  <p><i>The video above shows the real 100% Rust <code>entroly_core</code> engine executing mathematically optimal context selection in under a millisecond.</i></p>
</div>

---

## 🚀 The Value Proposition

When using Entroly as your agent's MCP context server:

- **100% Signal, 0% Noise:** Filters out irrelevant markdown, CSS, and test files automatically.
- **Zero Hallucinated Dependencies:** Auto-links related files via static import analysis. If it includes `auth/db.py`, it knows it *must* include `config/database.py`.
- **Deduplication:** 64-bit SimHash catches near-duplicates instantly, saving you thousands of wasted tokens on redundant code.
- **Lightning Fast:** 0/1 Knapsack optimization runs in **< 0.5ms** in Rust. Faster than a network ping to the LLM.

```bash
pip install entroly[native]
```

## ⚡ Zero-Friction Setup

```bash
pip install entroly[native]
cd your-project
entroly init        # auto-detects Cursor / VS Code / Windsurf / Claude Desktop
                    # writes the correct MCP config in one command
# Restart your AI tool — done.
```

`entroly init` detects your project type and AI tool, generates the right `mcp.json`, and confirms how many files it will auto-index on first run. The MCP server then automatically indexes your codebase (via `git ls-files`) when it starts.

```bash
entroly serve       # start MCP server with auto-indexing
entroly dashboard   # show live ROI metrics (cost saved, latency, compression)
```

## 🏗 Architecture

Hybrid Rust + Python: CPU-intensive math (knapsack DP, entropy, SimHash, dependency graph) runs in Rust via PyO3 for 50-100x speedup. MCP protocol and orchestration run in Python via FastMCP.

## 🧠 10 Real Subsystems Under the Hood

An MCP server that sits between your AI coding tool and the LLM, providing:

| Engine | What it does | How it works |
|--------|-------------|--------------|
| **Knapsack Optimizer** | Selects mathematically optimal context subset | 0/1 Knapsack DP with budget quantization (N ≤ 2000), greedy fallback (N > 2000) |
| **Entropy Scorer** | Measures information density per fragment | Shannon entropy (40%) + boilerplate detection (30%) + cross-fragment multi-scale n-gram redundancy (30%) |
| **SimHash Dedup** | Catches near-duplicate content in O(1) | 64-bit SimHash fingerprints with 4-band LSH bucketing, Hamming threshold = 3 |
| **Multi-Probe LSH Index** | Sub-linear semantic recall over 100K+ fragments | 12-table LSH with 10-bit sampling + 3-neighbor multi-probe queries |
| **Dependency Graph** | Pulls in related code fragments together | Symbol table + auto-linking (imports, type refs, function calls) + two-pass knapsack refinement |
| **Predictive Pre-fetch** | Pre-loads context before the agent asks | Static import analysis + test file inference + learned co-access patterns |
| **Checkpoint & Resume** | Crash recovery for multi-step tasks | Gzipped JSON state serialization (~100 KB per checkpoint) |
| **Feedback Loop** | Learns which context leads to good outputs | Wilson score lower-bound confidence intervals |
| **Context Ordering** | Orders fragments for optimal LLM attention | Pinned → criticality level → dependency count → relevance score |
| **PRISM Optimizer** | Adapts scoring weights to the codebase | Anisotropic spectral optimization via Jacobi eigendecomposition |

## 🎮 Try it Live

Want to run the exact demo shown in the video yourself? All demo assets are included in the repository:

```bash
git clone https://github.com/juyterman1000/entroly.git
cd entroly
pip install -e .[native]

# Run the terminal-based interactive value demonstrator
python demo_value.py

# Run the full live-engine dashboard experience
python demo_full_experience.py

# View the presentation slide-deck
open demo_presentation.html
```

---

## 🛠 Client Setup

### Cursor

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

### Claude Code

```bash
claude mcp add entroly -- entroly serve
```

### Cline / Any MCP Client

```json
{
  "entroly": {
    "command": "entroly",
    "args": ["serve"]
  }
}
```

## 📊 Live ROI Dashboard

The `entroly_dashboard()` MCP tool gives you a live look at your savings:

```json
{
  "money": {
    "cost_per_call_without_entroly": "$0.0115",
    "cost_per_call_with_entroly":    "$0.0044",
    "savings_pct": "62%",
    "insight": "Each optimize call costs $0.0044 instead of $0.0115. Over 38 calls, that's $0.27 saved."
  },
  "performance": {
    "avg_optimize_latency": "320µs (0.32ms)",
    "vs_api_roundtrip":     "6250x faster than a typical API call"
  },
  "bloat_prevention": {
    "context_compression": "39.00%",
    "memory_footprint":    "612 KB",
    "duplicates_caught":   "12"
  }
}
```

## 📐 The Math

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

Context selection is exactly the 0/1 Knapsack Problem:

```
Maximize:   Σ r(fᵢ) · x(fᵢ)     for selected fragments
Subject to: Σ c(fᵢ) · x(fᵢ) ≤ B  (token budget)
```

**Two strategies** based on fragment count:
- **N ≤ 2000**: Exact DP with budget quantization into 1000 bins — O(N × 1000)
- **N > 2000**: Greedy density sort — O(N log N), Dantzig 0.5-optimality guarantee

### Task-Aware Budget Multipliers

```
Bug tracing / debugging     → 1.5× budget
Exploration / understanding → 1.3× budget
Refactoring / code review   → 1.0× budget
Testing                     → 0.8× budget
Code generation             → 0.7× budget
Documentation               → 0.6× budget
```

## 📚 References

- Shannon (1948) — Information Theory
- Charikar (2002) — SimHash
- Ebbinghaus (1885) — Forgetting Curve
- Dantzig (1957) — Greedy Knapsack Approximation
- Wilson (1927) — Score Confidence Intervals
- ICPC (arXiv 2025) — In-context Prompt Compression

## 🤝 Part of the Ebbiforge Ecosystem

Entroly integrates with [hippocampus-sharp-memory](https://pypi.org/project/hippocampus-sharp-memory/) for persistent memory and [Ebbiforge](https://pypi.org/project/ebbiforge/) for TF embeddings and RL weight learning. Both are optional — Entroly works standalone.

## 📄 License

MIT
