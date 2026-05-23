# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Install Python package with all extras (includes Rust engine)
pip install -e ".[full]"

# Compile Rust core → Python bindings (required after Rust changes)
maturin develop --release

# Rust only
cd entroly-core && cargo build --release
```

## Test

```bash
# Full Python test suite
pytest tests/ -v --tb=short --timeout=60

# Single test file
pytest tests/test_cli.py -v --tb=short

# Rust unit tests
cd entroly-core && cargo test --lib

# Functional smoke test
python tests/functional_test.py
```

## Lint

```bash
# Python
ruff check entroly/

# Rust
cd entroly-core && cargo clippy --all-targets -- -D warnings
```

## Run

```bash
entroly              # Start MCP server (STDIO)
entroly proxy        # HTTP reverse proxy on localhost:9377
entroly go           # Full onboarding (detect IDE, generate config)
entroly dashboard    # Interactive dashboard
entroly health       # Codebase health grade (A–F)
```

## Architecture

The system has two layers: a **Python orchestration layer** (`entroly/`) and a **Rust computation engine** (`entroly-core/`), bound together via PyO3/maturin. Python handles MCP protocol, HTTP proxy, CLI, and flow orchestration. Rust handles all compute-heavy work at 50–100× Python speed.

### Entry Points

| Entry | File | Purpose |
|-------|------|---------|
| MCP server | `entroly/server.py` | Thin wrapper — delegates computation to Rust |
| HTTP proxy | `entroly/proxy.py` | Intercepts API calls, injects compressed context |
| CLI | `entroly/cli.py` | 20+ commands via Click |
| Public SDK | `entroly/sdk.py` | `compress()` / `compress_messages()` |

### Epistemic Router (5 Flows)

`epistemic_router.py` selects which pipeline runs for each query:

1. **Fast Answer** — beliefs are fresh, act immediately
2. **Verify Before Answer** — beliefs are stale, recompile + verify first
3. **Compile On Demand** — no beliefs exist, index + extract + verify
4. **Change-Driven** — triggered by PR/commit, analyzes blast radius, updates vault
5. **Self-Improvement** — repeated failures trigger skill synthesis → promote/prune

`flow_orchestrator.py` executes the selected pipeline. `query_refiner.py` expands vague queries before routing.

### Rust Core Modules (`entroly-core/src/`)

| Module | Role |
|--------|------|
| `knapsack.rs` / `knapsack_sds.rs` | 0/1 DP token budget solver with (1-1/e) guarantee |
| `entropy.rs` | Shannon entropy = information density per token |
| `semantic_dedup.rs` | SimHash O(1) duplicate detection |
| `bm25.rs` | TF-IDF + BM25 relevance ranking |
| `depgraph.rs` | Cross-file import/dependency resolution |
| `prism.rs` | Reinforcement loop — learns fragment→outcome mappings |
| `cogops.rs` | Unified engine combining all of the above |
| `sast.rs` | Static security scanning (55 rules) |
| `archetype.rs` | Role-based context presets |

### Knowledge Vault (`vault.py`)

Persistent learning store under `vault/`:
- `vault/beliefs/` — durable code-entity understanding (confidence, staleness, sources)
- `vault/verification/` — challenges and staleness tracking
- `vault/actions/` — task outputs, PR briefs
- `vault/evolution/skills/` — skill specs with test cases and fitness metrics

Every artifact carries `claim_id`, `entity`, `status`, `confidence`, `sources`.

### RAVS (`entroly/ravs/`)

Request Aware Verifier System — routes tasks to the cheapest capable model:

- `router.py`: Bayesian confidence tracking; routes to Haiku by default, escalates to Opus if confidence < 80%
- `verifiers.py`: Deterministic executors (run tests, lint, file reads) — zero LLM cost
- `capture.py`: Observes outcomes for the confidence update loop
- `controller.py`: Manages the Bayesian state
- `report.py`: Session/weekly cost savings reports

Fail-closed: unknown or low-confidence → Opus.

### Context Compression Pipeline

```
Query → Query Refiner → Epistemic Router → Rust CogOps Engine
         (expand)         (5-flow select)    (knapsack + entropy + BM25 + SimHash + depgraph)
                                                      ↓
                                              Vault Manager (read/write beliefs)
                                                      ↓
                                              RAVS (route to cheapest model)
                                                      ↓
                                              LLM API → PRISM feedback → Evolution Daemon
```

### Evolution Daemon (`evolution_daemon.py`)

Monitors failed queries → clusters by entity → synthesizes skill SOPs (`skill_engine.py`) → benchmarks (`benchmark_harness.py`) → promotes (fitness ≥ 0.7) or prunes (fitness ≤ 0.3). Spend-gated: learning cost must be covered by projected savings.

Federation (`federation.py`) shares anonymized learned patterns across all instances via GitHub — no servers, no cloud cost.

## Key Constraints

- Rust changes require `maturin develop --release` before Python tests will pick them up.
- RAVS is fail-closed — always routes to Opus when uncertain; never sacrifice correctness for cost.
- Vault beliefs are machine-auditable: every write must include `claim_id`, `entity`, `confidence`, and `sources`.
- Token-negative learning contract: evolution daemon cannot spend more on skill synthesis than the projected savings budget.
