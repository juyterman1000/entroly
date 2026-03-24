<p align="center">
  <img src="docs/assets/logo.png" width="200" alt="Entroly Logo">
</p>

<h1 align="center">Entroly</h1>

<p align="center">
  <b>Information-theoretic context compression for AI coding agents.</b>
  <br/>
  <i>40% fewer tokens. 100% codebase coverage. < 10ms overhead. Zero config changes.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-Engine-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="PyPI">
  <img src="https://img.shields.io/badge/Tests-436%20Passing-success" alt="Tests">
  <img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker">
</p>

---

## See It In Action

<p align="center">
  <img src="docs/assets/demo.svg" alt="Entroly Demo — 78% token savings in 8ms" width="800">
</p>

> **Run it yourself:** `pip install entroly && entroly demo`
>
> Open the [interactive HTML demo](docs/assets/demo.html) for the full animated experience, or generate your own with `python docs/generate_demo.py`.

---

## The Value

<p align="center">
  <img src="docs/assets/value.svg" alt="The Entroly Difference — before and after comparison" width="840">
</p>

---

## How It Works

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="Entroly Pipeline — 5-stage context optimization" width="880">
</p>

---

## What You Get

- **40% token reduction** on average — your AI sees the same codebase, but pays for less
- **100% codebase coverage** — every file is visible at variable resolution, not just the top 5 matches
- **< 10ms overhead** per request — invisible to the developer
- **Zero code changes** — drop-in MCP server or transparent HTTP proxy
- **Self-improving** — RL learns better context selection from every session outcome
- **Cross-platform** — native on Linux, macOS, and Windows (no WSL required)

---

## Quick Start

```bash
pip install entroly
entroly init        # auto-detects Cursor, VS Code, Claude Desktop, Windsurf
entroly status      # verify it's running
```

That's it. Entroly runs as an MCP server and auto-indexes your codebase on the first request.

For proxy mode (works with any AI tool, no IDE plugin needed):

```bash
pip install entroly[proxy]
entroly proxy --quality balanced
# Point your AI tool's API base URL to http://localhost:9377/v1
```

<details>
<summary><b>Windows-specific notes</b></summary>

- If `pip` is not on your PATH, use `python -m pip install entroly`
- Entroly runs natively — no WSL required
- Docker Desktop is optional
- `entroly init` auto-detects Claude Desktop config at `%APPDATA%\Claude\`
</details>

<details>
<summary><b>Docker (any platform)</b></summary>

```bash
docker pull ghcr.io/juyterman1000/entroly:latest
docker run --rm -p 9377:9377 -p 9378:9378 -v .:/workspace:ro ghcr.io/juyterman1000/entroly:latest
```

Multi-arch: `linux/amd64` and `linux/arm64` (Apple Silicon, AWS Graviton).
</details>

**Quality presets:** `--quality speed` | `fast` | `balanced` | `quality` | `max` (or any float 0.0-1.0)

**Something not working?** Run `entroly doctor` — it checks everything automatically.

---

Every AI coding tool — Cursor, Copilot, Claude Code, Cody — manages context with dumb heuristics: stuff tokens until the window fills, then cut. Entroly uses mathematics to compress an **entire codebase** into the optimal context window, and then **learns** which types of context produce better outcomes.

## The Problem

Current AI tools use **Cosine Similarity** (Vector Search). It's great for finding "things that look like my query," but terrible for coding because:
1. **Context Blindness**: Finds "the top 5 files" but misses the 6th file with the critical interface definition.
2. **Boilerplate Waste**: 40% of retrieved code is often imports or repetitive boilerplate.
3. **Static Heuristics**: Selection weights never improve — every session starts from zero.

## The Solution: Entroly

Entroly replaces "dumb search" with **Information-Theoretic Compression + Online RL**. It treats the context window as a finite resource, uses principled combinatorial optimization to fill it, and learns better selection strategies from every session outcome.

---

```
pip install entroly
```

## How It's Different

**Sourcegraph Cody** does *search*: "Find 5–10 files that look relevant." **Entroly** does *compression*: "Show the LLM the **entire codebase** at variable resolution — and learn what works."

| | Cody / Copilot | Entroly |
|--|----------------|---------|
| **Approach** | Embedding similarity search | Information-theoretic compression + online RL |
| **Coverage** | 5–10 files (the rest is invisible) | 100% codebase at variable resolution |
| **Selection** | Top-K by cosine distance | KKT-optimal bisection with submodular diversity |
| **Dedup** | None | SimHash + LSH in O(1) |
| **Learning** | Static | REINFORCE with KKT-consistent baseline (first-order) |
| **Security** | None | Built-in SAST (55 rules, taint-aware) |
| **Temperature** | User-set | ADGT × PCNT self-calibrating (no tuning needed) |

## Architecture

Hybrid Rust + Python. All math runs in Rust via PyO3 (50–100× faster). MCP protocol and orchestration run in Python.

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
│  │  │  15 modules · 340 KB · 113 tests    │     │       │
│  │  └─────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

Two deployment modes:
- **MCP Server** — IDE calls `remember_fragment`, `optimize_context`, etc. via MCP protocol
- **Prompt Compiler Proxy** — invisible HTTP proxy at `localhost:9377`, intercepts every LLM request and auto-optimizes (zero IDE changes beyond API base URL)

## Engines

### Rust Core (15 modules)

| Module | What | How |
|--------|------|-----|
| **hierarchical.rs** | 3-level codebase compression (ECC) | L1: skeleton map of ALL files · L2: dep-graph cluster expansion · L3: knapsack-optimal full fragments with submodular diversity |
| **knapsack.rs** | Optimal context subset selection | τ ≥ 0.05: exact KKT dual bisection O(30·N) · τ < 0.05: exact 0/1 DP O(N×1000) |
| **knapsack_sds.rs** | IOS: Information-Optimal Selection | Submodular Diversity Selection (SDS) + Multi-Resolution Knapsack (MRK) in one greedy pass · (1−1/e) optimality guarantee |
| **prism.rs** | Spectral weight optimizer (PRISM) | Exact Jacobi eigendecomposition of 4×4 gradient covariance · Q Λ⁻¹/² Qᵀ natural gradient · exposes condition_number() for PCNT |
| **entropy.rs** | Information density scoring | Shannon entropy (40%) + boilerplate detection (30%) + cross-fragment n-gram redundancy (30%) |
| **depgraph.rs** | Dependency graph + symbol table | Auto-linking: imports (1.0) · type refs (0.9) · function calls (0.7) · same-module (0.5) |
| **skeleton.rs** | AST-lite code skeleton extraction | Preserves signatures, class/struct/trait layouts, strips bodies → 60–80% token reduction |
| **dedup.rs** | Near-duplicate detection | 64-bit SimHash fingerprints · Hamming threshold ≤ 3 · 4-band LSH buckets |
| **lsh.rs** | Semantic recall index | 12-table multi-probe LSH · 10-bit sampling · ~3 μs over 100K fragments |
| **sast.rs** | Static Application Security Testing | 55 rules across 8 CWE categories · taint-flow analysis · severity scoring |
| **health.rs** | Codebase health analysis | Clone detection (Type-1/2/3) · dead symbol finder · god file detector · arch violation checker |
| **guardrails.rs** | Safety-critical file pinning | Criticality levels (Safety/Critical/Important/Normal) · task-aware budget multipliers |
| **query.rs** | Query analysis + refinement | Vagueness scoring · keyword extraction · intent classification |
| **fragment.rs** | Core data structure | Content, metadata, scoring dimensions, skeleton, SimHash fingerprint |
| **lib.rs** | PyO3 bridge + orchestrator | All modules exposed to Python · KKT-REINFORCE online learning · 113 tests |

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
    Query["User Query"] --> L1["L1: Skeleton Map · 5%"]
    Query --> L2["L2: Dependency Cluster · 25%"]
    Query --> L3["L3: Full Fragments · 70%"]

    L1 --> C1["Signatures of all files"]
    L2 --> C2["Expanded skeletons of related modules"]
    L3 --> C3["Submodular diversified full code"]

    C1 --> Context["Optimal Context Window"]
    C2 --> Context
    C3 --> Context
```

Techniques:
1. **Symbol-reachability slicing** — BFS through dep graph from query-relevant symbols
2. **Submodular diversity selection** — diminishing returns per module (1−1/e guarantee)
3. **PageRank centrality** — hub files get priority in L2 expansion
4. **Entropy-gated budget allocation** — complex codebases get more L3 budget

### IOS — Information-Optimal Selection

Combines two algorithms in one greedy pass (`knapsack_sds.rs`):

**Submodular Diversity Selection (SDS):**
Standard knapsack assumes value(A ∪ B) = value(A) + value(B). Information has diminishing returns. SDS applies a SimHash-based diversity penalty to each candidate relative to what's already selected:

```
marginal_value(i, S) = base_value(i) × diversity_factor(simhash_i, {simhash_j : j ∈ S})
```
Greedy on `marginal_value / token_cost`. Guarantee: (1 − 1/e) ≈ 63% of optimal (Nemhauser-Wolsey-Fisher 1978).

**Multi-Resolution Knapsack (MRK):**
Each fragment has up to 3 representations with different value/cost tradeoffs:

| Resolution | Information | Tokens |
|---|---|---|
| Full | 100% | 100% |
| Skeleton (signatures only) | ~70% | ~20% |
| Reference (path + name) | ~15% | ~2% |

This is the Multiple Choice Knapsack Problem (MCKP). When budget is tight, IOS selects the lowest-cost representation that still fits.

### KKT-REINFORCE: Online Weight Learning

The scoring function is:
```
sᵢ = w_r·recency_i + w_f·frequency_i + w_s·semantic_i + w_e·entropy_i
```

Weights `w = [w_r, w_f, w_s, w_e]` are learned from session outcomes via a theoretically novel coupling between the forward selection and the backward policy gradient.

#### Forward Pass — Exact KKT Dual Bisection

The continuous relaxation of the 0/1 knapsack has the KKT condition:
```
p*ᵢ = σ((sᵢ − λ*·tokensᵢ) / τ)
```
where `λ*` is the Lagrange multiplier for the budget constraint, found via bisection:
```
g(λ) = Σᵢ σ((sᵢ − λ·tokensᵢ)/τ)·tokensᵢ − B = 0
dg/dλ = −1/τ · Σᵢ pᵢ(1−pᵢ)·tokensᵢ² < 0   (strictly monotone → bisection converges)
```
Fragments are then sorted by `pᵢ` (= by reduced cost `sᵢ − λ*·tokensᵢ`) and greedily filled to hard budget. Complexity: **O(30·N)**.

#### Backward Pass — REINFORCE with KKT Baseline

After observing the outcome (success/failure → reward R):
```
∂E[R]/∂w_k = Σᵢ (actionᵢ − p*ᵢ) · R · p*ᵢ(1−p*ᵢ)/τ · featureᵢ_k
```

where `p*ᵢ = σ((sᵢ − λ*·tokensᵢ)/τ)` uses the **same λ* from the forward pass**. This baseline is:
- **Per-item** (not a constant scalar) — calibrated to each fragment's budget cost
- **Derived from the KKT condition** — not a learned value function
- **Self-consistent** — forward and backward use the exact same probability

The σ′(·/τ) = p(1-p)/τ term focuses gradient mass on fragments near the selection boundary (those with p ≈ 0.5), automatically handling credit assignment.

#### PRISM: Natural Gradient Preconditioning

The gradient is preconditioned by the inverse square root of the gradient covariance:
```
Δw = α · Q Λ⁻¹/² Qᵀ · g
```
where `C = Q Λ Qᵀ` is the Jacobi eigendecomposition of the 4×4 running gradient covariance. High-variance weight dimensions are damped; stable ones are amplified. This is exact (not approximate) because d=4 is small enough for direct Jacobi iteration.

#### ADGT: Adaptive Dual Gap Temperature

Replaces the ad-hoc `τ *= 0.995` annealing with a principled information-theoretic signal.

The log-sum-exp dual of the budget constraint is:
```
D(λ*) = τ · Σᵢ log(1 + exp((sᵢ − λ*·cᵢ)/τ)) + λ*·B
```
The **duality gap** `G = D(λ*) − primal ∈ [0, τ·N·log(2)]`:
- G → 0: weights converged, hard selection → lower τ
- G → τ·N·log(2): all pᵢ = 0.5, maximum uncertainty → keep τ high

Natural temperature: `τ_nat = G / (N·log(2))`. Self-regulating — no decay constant needed.

#### PCNT: PRISM Condition-Number Temperature

The spectral condition number `κ = √(λ_max/λ_min)` of the PRISM gradient covariance encodes weight-space uncertainty:
- κ ≈ 1: well-conditioned weights, commit sharply (low τ)
- κ >> 1: ill-conditioned, some dimensions noisy, stay soft (high τ)

Final temperature: `τ = EMA(G/N · κ_norm, 0.90)`. No hyperparameters beyond τ_min/τ_max.

ADGT and PCNT are orthogonal: ADGT asks "how far is the current selection from the continuous optimum?", PCNT asks "how uncertain are the learned weights themselves?"

### EGTC v2 (Entropy-Gap Temperature Calibration)

Automatically derives the optimal LLM sampling temperature from context information-theoretic properties:
```
τ = clip(τ_base + Σ signal_weights × [vagueness, entropy_gap, sufficiency, task_type])
```

### APA (Adaptive Prompt Augmentation)

1. **Calibrated token estimation** — per-language chars/token ratios (Python: 3.0, Rust: 3.5, ...)
2. **Task-aware preamble** — conditional hints from security findings, vagueness, and task type
3. **Content deduplication** — MD5 hash-based dedup saves 10–20% in multi-turn sessions

---

## Installation

### Requirements

| | Linux | macOS | Windows |
|--|-------|-------|---------|
| **Python** | 3.10+ | 3.10+ | 3.10+ |
| **Rust engine** | Pre-built wheel | Pre-built wheel | Pre-built wheel |
| **Docker** | Optional (native works) | Optional (Docker Desktop) | Optional (Docker Desktop) |
| **WSL** | N/A | N/A | **Not required** |

### Install from PyPI

```bash
# Core (MCP server, Python engine fallback)
pip install entroly

# With proxy mode (invisible prompt compiler)
pip install entroly[proxy]

# With native Rust engine (50-100× faster)
pip install entroly[native]

# Everything
pip install entroly[full]
```

> **Windows users:** If `pip` is not on your PATH, use `python -m pip install entroly`.
> **macOS users:** On Apple Silicon (M1/M2/M3/M4), the pre-built `arm64` wheel installs automatically.

---

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

### VS Code / Windsurf

Run `entroly init` — it auto-detects your IDE and generates the correct MCP config.

### Prompt Compiler Proxy (works with ANY AI tool)

The proxy is an invisible HTTP layer that intercepts every LLM request, optimizes context with the full pipeline (ECC + IOS + EGTC + APA + SAST), and forwards it transparently. < 10ms overhead.

```bash
pip install entroly[proxy]
entroly proxy --quality balanced
```

> **Windows:** The proxy binds to `127.0.0.1:9377`. No admin rights required.
> Firewall may prompt for network access — allow it.

Optional flags:
```bash
entroly proxy --quality 0.8       # Quality dial 0.0 (speed) → 1.0 (quality)
entroly proxy --port 9377         # Custom port
entroly proxy --force             # Force re-index codebase
```

Then point your AI tool's API base URL to `http://localhost:9377/v1` — done. Auto-indexes your codebase on startup.

### Docker (cross-platform, build-once-run-everywhere)

```bash
# All platforms
docker pull ghcr.io/juyterman1000/entroly:latest
docker run --rm -p 9377:9377 -p 9378:9378 -v .:/workspace:ro ghcr.io/juyterman1000/entroly:latest
```

Multi-arch: `linux/amd64` and `linux/arm64` (Apple Silicon, AWS Graviton).

> **Docker Desktop (macOS / Windows):** Uses port mapping (`-p 9377:9377`) which works on all platforms.
> **Linux:** Optionally use `--network=host` for lower latency.

### Docker Compose

```bash
docker compose up -d
```

Includes healthcheck, persistent volume for learned state, and auto-restart.

### Live Intelligence Dashboard

The proxy auto-launches a live dashboard at `http://localhost:9378` showing real-time metrics:

<p align="center">
  <img src="docs/assets/dashboard.png" width="800" alt="Entroly Intelligence Dashboard">
</p>

- **Savings counter** — total cost saved, tokens saved, dedup hits
- **PRISM Intelligence radar** — RL-learned weights evolving in real-time
- **Code Health ring** — A-F grade with clone/dead-symbol/god-file detection
- **SAST Security shield** — 55-rule scanner with CWE category breakdown
- **Knapsack decisions** — per-fragment scores with inclusion/exclusion reasons
- **Request flow sparkline** — live token savings per proxy request

---

## MCP Tools

| Tool | Purpose |
|------|---------|
| `remember_fragment` | Store context with auto-dedup, entropy scoring, dep linking, criticality detection |
| `optimize_context` | Select optimal context subset for a token budget (IOS + ECC + KKT bisection) |
| `recall_relevant` | Sub-linear semantic recall via multi-probe LSH |
| `record_outcome` | Feed the KKT-REINFORCE weight learning loop |
| `explain_context` | Per-fragment scoring breakdown with sufficiency analysis |
| `checkpoint_state` | Save full session state (gzipped JSON, includes λ*, dual_gap, PRISM covariance) |
| `resume_state` | Restore from checkpoint |
| `prefetch_related` | Predict and pre-load likely-needed context |
| `get_stats` | Session statistics and cost savings |
| `health_check` | Clone detection, dead symbols, god files, arch violations |

---

## The Math

### Multi-Dimensional Fragment Scoring

```
sᵢ = w_r·recency + w_f·frequency + w_s·semantic + w_e·entropy
```

Applied via soft sigmoid: `pᵢ = σ((sᵢ − λ*·tokensᵢ) / τ)` — items are weighted both by their intrinsic score and by their token cost relative to the budget dual variable.

**Score components:**
- **Recency**: Ebbinghaus forgetting curve — `exp(−ln(2) × Δt / half_life)`
- **Frequency**: Normalized access count with spaced repetition boost
- **Semantic similarity**: SimHash Hamming distance to query, normalized to [0, 1]
- **Information density**: Shannon entropy + boilerplate ratio + cross-fragment redundancy

### Context Selection — Two Paths

**Soft path (τ ≥ 0.05):** KKT dual bisection — O(30·N):
```
Find λ* ≥ 0 s.t. Σᵢ σ((sᵢ − λ*·tokensᵢ)/τ)·tokensᵢ = B
Sort by pᵢ = σ((sᵢ − λ*·tokensᵢ)/τ), greedy fill to hard budget
```

**Hard path (τ < 0.05):** Exact 0/1 DP — O(N × 1000):
```
Maximize:   Σ sᵢ·xᵢ
Subject to: Σ tokensᵢ·xᵢ ≤ B,   xᵢ ∈ {0, 1}
```

The soft path is the default during learning. As τ → 0 (via ADGT), probabilities converge to hard 0/1 selections and the DP fallback activates automatically.

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

---

## Configuration

```python
EntrolyConfig(
    default_token_budget=128_000,     # GPT-4 Turbo equivalent
    max_fragments=10_000,             # session fragment cap
    weight_recency=0.30,              # initial scoring weights (learned online)
    weight_frequency=0.25,
    weight_semantic_sim=0.25,
    weight_entropy=0.20,
    decay_half_life_turns=15,         # Ebbinghaus half-life
    enable_hierarchical_compression=True,  # 3-level ECC
    enable_ios=True,                       # IOS (SDS + MRK)
    enable_ios_diversity=True,             # submodular diversity penalty
    enable_ios_multi_resolution=True,      # skeleton/reference fallback
    enable_temperature_calibration=True,   # EGTC v2
    enable_prompt_directives=True,         # APA preamble
    enable_security_scan=True,             # SAST
)
```

---

## References

**Foundations used:**
- Shannon (1948) — Information Theory
- Charikar (2002) — SimHash
- Ebbinghaus (1885) — Forgetting Curve
- Nemhauser, Wolsey & Fisher (1978) — Submodular Maximization (1−1/e guarantee)
- Sviridenko (2004) — Submodular + Knapsack constraint
- Minoux (1978) — Lazy greedy acceleration
- Kellerer, Pferschy, Pisinger (2004) — Multiple Choice Knapsack (MCKP)
- Boyd & Vandenberghe — Convex Optimization §5.2 (KKT duality, dual bisection)
- Williams (1992) — REINFORCE policy gradient
- Dantzig (1957) — Greedy Knapsack Approximation
- Wilson (1927) — Score Confidence Intervals

**Related work (RAG / context compression):**
- LLMLingua (EMNLP 2023) — Perplexity-based Token Compression
- LongLLMLingua (ACL 2024) — Query-aware Context Compression
- RepoFormer (ICML 2024 Oral) — Selective Retrieval for Repo-Level Code
- FILM-7B (NeurIPS 2024) — Structure-First Layout
- CodeSage (ICLR 2024) — Code Embedding Representation Learning
- SWE-bench (ICLR 2024) / SWE-agent (NeurIPS 2024) — Evaluation

**Novel compositions (not in prior literature):**
- KKT-REINFORCE: the dual variable λ* of the forward budget constraint as the per-item REINFORCE baseline (forward + backward use the same probability)
- ADGT: duality gap G = D(λ*) − primal as a self-regulating temperature signal
- PCNT: PRISM spectral condition number as a weight-uncertainty-aware temperature modulator

---

## CLI Commands

| Command | Purpose |
|---------|--------|
| `entroly init` | Auto-detect project type + AI tool, generate MCP config |
| `entroly serve` | Start MCP server (default — used by Cursor, Claude Code) |
| `entroly proxy` | Start invisible prompt compiler proxy on :9377 |
| `entroly dashboard` | Launch value dashboard on :9378 |
| `entroly demo` | Run before/after comparison showing token savings |
| `entroly doctor` | 7-point diagnostic check (Python, Rust engine, config, proxy, index, weights, Docker) |
| `entroly digest` | Weekly summary of optimization value delivered |
| `entroly role` | Apply role-based weight presets (`frontend` / `backend` / `sre` / `data` / `fullstack`) |
| `entroly migrate` | Auto-migrate config/index/checkpoints to current version |
| `entroly health` | Run code health analysis — A-F grade (clones, dead symbols, god files, arch violations) |
| `entroly autotune` | Run autonomous hyperparameter optimization (mutate → evaluate → keep) |
| `entroly benchmark` | Run competitive benchmark (Entroly vs Raw vs Top-K) |
| `entroly status` | Check if proxy/dashboard is running |
| `entroly config` | Show current configuration with all feature flags |
| `entroly clean` | Clear cached state (checkpoints, index, pull cache) |
| `entroly completions` | Generate shell completion scripts (bash/zsh/fish) |

## Works with OpenClaw

[OpenClaw](https://openclaw.ai) is a personal AI assistant that manages email, calendar, code, and 50+ integrations — all from WhatsApp, Telegram, or any chat app. Its workspace holds your identity (SOUL.md), persistent memory (MEMORY.md), daily logs, skill definitions, and tool schemas.

The problem: when your OpenClaw agent loads context for a task, it reads files sequentially until the token budget is full — then stops. Your agent can't see files it never loaded.

Entroly fixes this. Measured on a real 24-file OpenClaw workspace:

<p align="center">
  <img src="docs/assets/openclaw_benchmark.png" alt="Entroly + OpenClaw — 100% coverage with 22.7% fewer tokens" width="840">
</p>

<p align="center">
  <b>Benchmark: 2,048 Token Budget (typical heartbeat agent)</b>
</p>

| | Without Entroly | With Entroly |
|--|-----------------|--------------|
| **Files visible to AI** | 15 of 24 | **24 of 24** |
| **Codebase coverage** | 62.5% | **100.0%** |
| **Tokens used** | 2,031 (99.2% of budget) | 1,812 (88.5%) |
| **Token savings** | — | **22.7%** |
| **Optimization time** | — | 0.05ms |

**9 files invisible** without Entroly — including email and system skill definitions, and all tool schemas your agent needs to function:

```
INVISIBLE without Entroly:
  x skills/system.md          — agent can't manage Docker or disk cleanup
  x tools/search_emails.json  — agent can't search your inbox
  x tools/send_email.json     — agent can't send emails
  x tools/disk_usage.json     — agent can't check disk space
  x tools/docker_ps.json      — agent can't list containers
  x tools/read_file.json      — agent can't read code files
  x tools/run_tests.json      — agent can't run your tests
  x tools/git_status.json     — agent can't check repo status
  x tools/web_search.json     — agent can't search the web
```

Your user asks via WhatsApp: _"Check my emails and prep me for standup."_ Without Entroly, your agent literally cannot see the email skill definition. With Entroly, every file is loaded at the right compression level.

> Reproduce this yourself: `python benchmarks/openclaw_benchmark.py`

**Integration — 3 lines of code:**

```python
from entroly.context_bridge import MultiAgentContext

ctx = MultiAgentContext(workspace_path="~/.openclaw/workspace", token_budget=128_000)
ctx.ingest_workspace()

# Main agent — full HCC-optimized context
context = ctx.load_hcc_context(query="check emails and prep for standup", token_budget=8192)

# Cron heartbeat — wakes every 15 min, auto LOD lifecycle
ctx.schedule_cron("email_checker", "check for urgent emails", interval_seconds=900)

# Subagent — inherits parent context, NKBE budget allocation
sub = ctx.spawn_subagent("main", "code_reviewer", "review PR #847 for security issues")
```

**What each component does for OpenClaw:**

| Component | What It Does For Your Agent |
|-----------|---------------------------|
| **HCC Compression** | SOUL.md verbatim, recent logs as skeletons, old logs as one-liners — 100% visibility, 22.7% fewer tokens |
| **NKBE Budget Allocator** | 5 subagents running? Each gets the mathematically optimal token slice via KKT bisection |
| **Cognitive Bus** | Email agent finds something urgent — code agent is notified instantly via ISA-prioritized routing |
| **LOD Manager** | Cron agents sleep at 0 cost between runs, wake to 15% budget on schedule |
| **AutoTune** | Learns that your SOUL.md and recent MEMORY.md entries matter most — weights self-calibrate |

---

## Part of the Ebbiforge Ecosystem

Entroly integrates with [hippocampus-sharp-memory](https://pypi.org/project/hippocampus-sharp-memory/) for persistent cross-session memory and [Ebbiforge](https://pypi.org/project/ebbiforge/) for TF embeddings and RL weight learning. Both are optional.

## Production Features

### Reliability & Self-Healing

| Feature | What it does |
|---------|--------------|
| **Connection Recovery** | Auto-reconnects dropped `httpx` connections — no manual restart |
| **Massive File Shield** | Hard ceiling (500 KB) prevents OOM on giant logs, vendor files, minified JS |
| **Binary File Detection** | 40+ binary extensions auto-skipped (images, audio, video, archives, fonts, databases) + null-byte header check |
| **Schema Migration** | `entroly migrate` upgrades config, checkpoints, and index format between versions |
| **Checkpoint & Resume** | Auto-saves state every N tool calls · gzipped JSON · crash recovery in <100ms |
| **Distributed Locking** | Cross-platform file locking (`fcntl` on POSIX, `msvcrt` on Windows) for multi-instance safety |

### Developer Experience

| Feature | What it does |
|---------|--------------|
| **One-Click Demo** | `entroly demo` — indexes your project and shows before/after token savings with exact percentages |
| **Self-Diagnostics** | `entroly doctor` — 7 checks (Python version, Rust engine, config validity, proxy reachability, index freshness, weight drift, Docker) |
| **Role Presets** | `entroly role apply backend` — pre-tuned weights for frontend, backend, SRE, data, and fullstack roles |
| **Weekly Digest** | `entroly digest` — summary of optimization value delivered this week |
| **Fragment Feedback** | `POST /feedback` — thumbs-up/down on individual fragments, builds Wilson Score confidence |
| **Explain Decisions** | `GET /explain` — per-fragment selection reasoning: entropy level, query relevance, variant type, file indicators |

---

## Troubleshooting

> **First step:** Run `entroly doctor` — it checks everything automatically.

<details>
<summary><b>Linux</b></summary>

**Docker not needed:**
```bash
entroly proxy  # Runs natively — no Docker required
pip install entroly[native]  # Install Rust engine for 50-100× speedup
```

**Port conflict:**
```bash
entroly proxy --port 9378
# or: export ENTROLY_PROXY_PORT=9378
```
</details>

<details>
<summary><b>macOS</b></summary>

**Apple Silicon (M1/M2/M3/M4):**
```bash
pip install entroly  # arm64 wheel installs automatically
```

**Docker Desktop networking:**
Docker Desktop on macOS doesn't support `--network=host`. Use port mapping (the default):
```bash
docker run --rm -p 9377:9377 -p 9378:9378 ghcr.io/juyterman1000/entroly:latest
```

**Homebrew Python:**
If `pip install` fails with "externally-managed-environment":
```bash
python3 -m venv ~/.venvs/entroly
source ~/.venvs/entroly/bin/activate
pip install entroly[full]
```
</details>

<details>
<summary><b>Windows</b></summary>

**pip not found:**
```powershell
python -m pip install entroly
```

**Rust wheel not available:**
Entroly falls back to a pure Python engine automatically. For the Rust speedup:
```powershell
pip install entroly[native]
# If no pre-built wheel exists, install Rust toolchain first:
# https://rustup.rs/ → then: pip install entroly[native]
```

**Permission errors:**
- No admin rights required — Entroly writes only to `%APPDATA%\entroly\` and `~\.entroly\`
- Windows Firewall may prompt when the proxy binds to `127.0.0.1:9377` — allow it

**Claude Desktop config location:**
```
%APPDATA%\Claude\claude_desktop_config.json
```
`entroly init` detects this automatically.

**Docker Desktop networking:**
Same as macOS — use port mapping (the default):
```powershell
docker run --rm -p 9377:9377 -p 9378:9378 ghcr.io/juyterman1000/entroly:latest
```
</details>

<details>
<summary><b>General Troubleshooting</b></summary>

**Proxy seems slow on first request:**
This is normal — the first request warms up HTTP connections and the Rust pipeline. Subsequent requests add <10ms.

**Rust engine not loading:**
```bash
ENTROLY_NO_DOCKER=1 entroly serve  # Falls back to Python engine (slower but functional)
```

**Shell completions:**
```bash
# Bash
eval "$(entroly completions bash)"
# Zsh
eval "$(entroly completions zsh)"
# Fish
entroly completions fish > ~/.config/fish/completions/entroly.fish
```

```powershell
# PowerShell — no built-in completion yet, but CLI --help works for all commands
entroly --help
entroly proxy --help
```
</details>

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ENTROLY_QUALITY` | `0.5` | Quality dial (0.0-1.0 or preset name) |
| `ENTROLY_PROXY_PORT` | `9377` | Proxy port |
| `ENTROLY_PULL_TTL` | `3600` | Docker pull cache TTL (seconds, 0=always) |
| `ENTROLY_MAX_FILES` | `5000` | Max files to auto-index |
| `ENTROLY_RATE_LIMIT` | `0` | Max requests/min (0=unlimited) |
| `ENTROLY_NO_DOCKER` | - | Skip Docker, run natively |
| `ENTROLY_MCP_TRANSPORT` | `stdio` | MCP transport (`stdio` or `sse`) |
| `ENTROLY_TEMPERATURE_CALIBRATION` | `1` | Set to `0` to disable EGTC |

## License

MIT
