# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Gstack Operating Protocol

Use a gstack-style workflow for non-trivial Entroly work: **Think -> Plan -> Build -> Review -> Test -> Ship -> Reflect**.

Entroly is an auditable context-control plane for AI agents, not a normal utility library. Every change must preserve trust: receipts must remain explainable, compression must remain reversible, verification must fail closed, and release automation must stay boring.

### Default workflow

1. Clarify the smallest useful user outcome before writing code.
2. Challenge the product claim: what is the real wedge, what can be cut, and what evidence would prove it?
3. Lock architecture before implementation: data flow, state transitions, failure modes, and test matrix.
4. Implement the smallest safe change.
5. Review for production bugs, trust regressions, packaging breakage, and overclaims.
6. Run targeted tests first, then expand to release tests if packaging/native surfaces changed.
7. Ship with a rollback path and exact verification commands.
8. Record any fragile release or test behavior so the next agent does not repeat it.

If gstack skills are installed, prefer this sequence:

```text
/office-hours -> /plan-ceo-review -> /plan-eng-review -> implement -> /review -> /qa or targeted tests -> /ship -> /retro
```

If gstack is not installed, follow the same workflow manually.

### Entroly trust invariants

Do not merge a change that weakens these invariants:

- **Receipt honesty:** selected context, omitted evidence, risks, hashes, and token ratios must be inspectable.
- **Reversibility:** compressed or summarized context must remain traceable back to source spans.
- **Fail-closed verification:** WITNESS, RAVS, and native-status checks must degrade safely, not silently claim confidence.
- **Local-first operation:** no surprise remote calls for ranking, receipts, verification, or diagnostics.
- **Cache stability:** prompt prefixes should remain byte-stable unless intentionally changed.
- **Release consistency:** Python, Rust, WASM, npm, Homebrew, docs, and native minimum versions must agree.
- **Benchmark honesty:** claims must include baseline, token budget, workload, and caveats.

### Review checklist

Before approving a PR, answer:

- Does this change affect context selection, receipts, WITNESS, RAVS, routing, cache alignment, or packaging?
- Are old and new behaviors covered by tests or a reproducible command?
- Could this silently disable Rust/native acceleration or verification?
- Could it break PyPI, npm, Homebrew, Docker, or binary release paths?
- Does the README or docs overclaim compared with measured results?
- Is the failure mode visible to the user?

## Build & Install

```bash
# Install Python package with all extras (includes Rust engine)
pip install -e ".[full]"

# Compile Rust core -> Python bindings (required after Rust changes)
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

### Release test matrix

For packaging/release changes, add these checks before publishing:

```bash
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
entroly doctor
```

For Homebrew changes, verify the PyPI sdist URL and SHA-256 from the PyPI JSON API before updating the formula.

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
entroly health       # Codebase health grade (A-F)
```

## Architecture

The system has two layers: a **Python orchestration layer** (`entroly/`) and a **Rust computation engine** (`entroly-core/`), bound together via PyO3/maturin. Python handles MCP protocol, HTTP proxy, CLI, and flow orchestration. Rust handles all compute-heavy work at 50-100x Python speed.

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
5. **Self-Improvement** — repeated failures trigger skill synthesis -> promote/prune

`flow_orchestrator.py` executes the selected pipeline. `query_refiner.py` expands vague queries before routing.

### Rust Core Modules (`entroly-core/src/`)

| Module | Role |
|--------|------|
| `knapsack.rs` / `knapsack_sds.rs` | 0/1 DP token budget solver with (1-1/e) guarantee |
| `entropy.rs` | Shannon entropy = information density per token |
| `semantic_dedup.rs` | SimHash O(1) duplicate detection |
| `bm25.rs` | TF-IDF + BM25 relevance ranking |
| `depgraph.rs` | Cross-file import/dependency resolution |
| `prism.rs` | Reinforcement loop — learns fragment->outcome mappings |
| `cogops.rs` | Unified engine combining all of the above |
| `sast.rs` | Static security scanning (151 rules) |
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

Fail-closed: unknown or low-confidence -> Opus.

### Context Compression Pipeline

```text
Query -> Query Refiner -> Epistemic Router -> Rust CogOps Engine
         (expand)         (5-flow select)    (knapsack + entropy + BM25 + SimHash + depgraph)
                                                      |
                                              Vault Manager (read/write beliefs)
                                                      |
                                              RAVS (route to cheapest model)
                                                      |
                                              LLM API -> PRISM feedback -> Evolution Daemon
```

### Evolution Daemon (`evolution_daemon.py`)

Monitors failed queries -> clusters by entity -> synthesizes skill SOPs (`skill_engine.py`) -> benchmarks (`benchmark_harness.py`) -> promotes (fitness >= 0.7) or prunes (fitness <= 0.3). Spend-gated: learning cost must be covered by projected savings.

Federation (`federation.py`) shares anonymized learned patterns across all instances via GitHub — no servers, no cloud cost.

## Key Constraints

- Rust changes require `maturin develop --release` before Python tests will pick them up.
- RAVS is fail-closed — always routes to Opus when uncertain; never sacrifice correctness for cost.
- Vault beliefs are machine-auditable: every write must include `claim_id`, `entity`, `confidence`, and `sources`.
- Token-negative learning contract: evolution daemon cannot spend more on skill synthesis than the projected savings budget.

## Release Discipline

When bumping versions, update every release surface together:

- `pyproject.toml`
- `entroly/pyproject.toml`
- `entroly-core/pyproject.toml`
- `entroly-core/Cargo.toml`
- `entroly-qccr/Cargo.toml`
- `entroly-wasm/Cargo.toml`
- `entroly-wasm/package.json`
- `entroly/npm/package.json`
- `entroly/npm-alias/package.json`
- `entroly/__init__.py`
- `entroly/native_status.py`
- Homebrew formula URL and SHA-256
- README/docs install pins

After a release, verify the published package first, then update downstream formulas/checksums.
