# Entroly

**Information-theoretic context optimization for AI coding agents.**

Every AI coding tool manages context with dumb FIFO truncation — stuffing tokens until the window is full, then cutting from the top. Entroly applies mathematics to select the **optimal** context subset.

```
pip install entroly
```

## Architecture

Hybrid Rust + Python: CPU-intensive math (knapsack DP, entropy, SimHash, LSH, dependency graph) runs in Rust via PyO3 for 50-100x speedup. MCP protocol and orchestration run in Python via FastMCP. Pure Python fallbacks activate automatically if the Rust extension isn't available.

## What It Does

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
| **Feedback Loop** | Learns which context leads to good outputs | Wilson score lower-bound confidence intervals (same formula as Reddit ranking) |
| **Context Ordering** | Orders fragments for optimal LLM attention | Pinned → criticality level → dependency count → relevance score |
| **Guardrails** | Auto-pins safety-critical files, classifies tasks | Criticality levels (Safety/Critical/Important/Normal) + task-aware budget multipliers |
| **PRISM Optimizer** | Adapts scoring weights to the codebase | Anisotropic spectral optimization via Jacobi eigendecomposition on 4×4 covariance matrix |
| **Provenance Chain** | Detects hallucination risk in selected context | Tracks source verification + confidence scoring per fragment |

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

### Cline / Any MCP Client

```json
{
  "entroly": {
    "command": "entroly",
    "args": []
  }
}
```

## MCP Tools

### `remember_fragment`
Store context with auto-dedup, entropy scoring, dependency linking, and criticality detection.

```
remember_fragment(content="def process_payment(...)...", source="payments.py", token_count=45)
→ {"status": "ingested", "entropy_score": 0.82}

remember_fragment(content="def process_payment(...)...")  # same content
→ {"status": "duplicate", "duplicate_of": "a1b2c3", "tokens_saved": 45}
```

### `optimize_context`
Select the optimal context subset for a token budget. Includes dependency boosting, ε-greedy exploration, context sufficiency scoring, and provenance metadata.

```
optimize_context(token_budget=128000, query="fix payment bug")
→ {
    "selected_fragments": [...],
    "optimization_stats": {"method": "exact_dp", "budget_utilization": 0.73},
    "tokens_saved_this_call": 42000,
    "sufficiency": 0.91,
    "hallucination_risk": "low"
  }
```

### `recall_relevant`
Sub-linear semantic recall via multi-probe LSH. Falls back to brute-force scan on cold start.

```
recall_relevant(query="database connection pooling", top_k=5)
→ [{"fragment_id": "...", "relevance": 0.87, "content": "..."}]
```

### `record_outcome`
Feed the Wilson score feedback loop. Adjusts fragment scoring multipliers in the range [0.5, 2.0].

```
record_outcome(fragment_ids=["a1b2c3", "d4e5f6"], success=true)
→ {"status": "recorded", "fragments_updated": 2}
```

### `explain_context`
Per-fragment scoring breakdown with sufficiency analysis and exploration swap log.

```
explain_context()
→ {
    "fragments": [{"id": "...", "recency": 0.9, "frequency": 0.3, "semantic": 0.7, "entropy": 0.8}],
    "sufficiency": 0.91,
    "exploration_swaps": 1
  }
```

### `checkpoint_state` / `resume_state`
Save and restore full session state — fragments, dedup index, co-access patterns, feedback scores.

```
checkpoint_state(task_description="Refactoring auth module", current_step="Step 5/8")
→ {"status": "checkpoint_saved", "fragments_saved": 47}

resume_state()
→ {"status": "resumed", "restored_fragments": 47, "metadata": {"step": "Step 5/8"}}
```

### `prefetch_related`
Predict and pre-load likely-needed context using import analysis, test file inference, and co-access history.

```
prefetch_related(file_path="src/payments.py", source_content="from utils import...")
→ [{"path": "src/utils.py", "reason": "import", "confidence": 0.70}]
```

### `get_stats`
Session statistics and cost savings.

```
get_stats()
→ {
    "fragments": 142,
    "total_tokens": 384000,
    "savings": {
      "total_tokens_saved": 284000,
      "total_duplicates_caught": 12,
      "estimated_cost_saved_usd": 0.85
    }
  }
```

## The Math

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
- **Information density**: Shannon entropy + boilerplate + redundancy (see below)

### Knapsack Context Selection

Context selection is the 0/1 Knapsack Problem:

```
Maximize:   Σ r(fᵢ) · x(fᵢ)     for selected fragments
Subject to: Σ c(fᵢ) · x(fᵢ) ≤ B  (token budget)
```

**Two strategies** based on fragment count:
- **N ≤ 2000**: Exact DP with budget quantization into 1000 bins — O(N × 1000)
- **N > 2000**: Greedy density sort — O(N log N), Dantzig 0.5-optimality guarantee

Pinned fragments (safety-critical files, config files) are always included; remaining budget is allocated via DP/greedy.

### Shannon Entropy Scoring

Three components combined:

```
score = 0.40 × normalized_entropy + 0.30 × (1 - boilerplate_ratio) + 0.30 × (1 - redundancy)
```

- **Shannon entropy** (40%): `H = -Σ p(char) · log₂(p(char))`, normalized by 6.0 bits/char. Stack-allocated 256-byte histogram, single O(n) pass.
- **Boilerplate detection** (30%): Pattern matching for imports, pass, dunder methods, closing delimiters.
- **Cross-fragment redundancy** (30%): Multi-scale n-gram overlap with adaptive weights by fragment length — bigram-heavy for short fragments (<20 words), 4-gram-heavy for long fragments (>100 words). Parallelized with rayon.

### SimHash Deduplication

64-bit fingerprints from word trigrams hashed via MD5:
- Hamming distance ≤ 3 → near-duplicate
- 4-band LSH bucketing for O(1) candidate lookup
- Separate 12-table multi-probe LSH index for semantic recall (~3 μs over 100K fragments)

### Dependency Graph

Auto-linking via source analysis:
- **Imports** (strength 1.0): Python `from X import Y`, Rust `use`, JS `import`
- **Type references** (0.9): Type annotations, isinstance checks
- **Function calls** (0.7): General identifier usage matching against symbol table
- **Same module** (0.5): Co-located definitions

Two-pass knapsack refinement: initial selection → boost dependencies of selected fragments → re-optimize.

### Task-Aware Budget Multipliers

```
Bug tracing / debugging     → 1.5× budget
Exploration / understanding → 1.3× budget
Refactoring / code review   → 1.0× budget
Testing                     → 0.8× budget
Code generation             → 0.7× budget
Documentation               → 0.6× budget
```

### PRISM Spectral Optimizer

Tracks a 4×4 covariance matrix over scoring dimensions [recency, frequency, semantic, entropy] with EMA updates (β=0.95). Jacobi eigendecomposition finds principal axes. Anisotropic spectral gain dampens noisy dimensions and amplifies clean signals — automatic learning rate adaptation without hyperparameter tuning.

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
    max_prefetch_fragments=10,
    auto_checkpoint_interval=5,       # checkpoint every N tool calls
)
```

## References

- Shannon (1948) — Information Theory
- Charikar (2002) — SimHash
- Ebbinghaus (1885) — Forgetting Curve
- Dantzig (1957) — Greedy Knapsack Approximation
- Wilson (1927) — Score Confidence Intervals
- ICPC (arXiv 2025) — In-context Prompt Compression
- Proximity (arXiv 2026) — LSH-bucketed Semantic Caching
- RCC (ICLR 2025) — Recurrent Context Compression
- ILRe (ICLR 2026) — Intermediate Layer Retrieval
- Agentic Plan Caching (arXiv 2025)

## Part of the Ebbiforge Ecosystem

Entroly integrates with [hippocampus-sharp-memory](https://pypi.org/project/hippocampus-sharp-memory/) for persistent memory and [Ebbiforge](https://pypi.org/project/ebbiforge/) for TF embeddings and RL weight learning. Both are optional — Entroly works standalone with pure Python fallbacks.

## License

MIT
