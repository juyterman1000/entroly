# Entroly steering

Steering doc for [Kiro](https://kiro.dev) and any other agentic IDE that
reads project-level guidance files. This is the single-source brief on
how to work in this codebase.

## What this project is

Entroly is a token-saving proxy + context compression engine for AI
coding agents. Two main layers:

- **Python orchestration** (`entroly/`): MCP server, HTTP proxy, CLI,
  flow orchestration, vault, RAVS, federation.
- **Rust computation core** (`entroly-core/`): submodular knapsack,
  Shannon entropy, BM25, SimHash dedup, dependency graph, SAST.

PyO3/maturin binds them. Everything that's compute-heavy lives in
Rust; everything coordination-heavy lives in Python.

## When you write code here

- **Rust changes need a rebuild before Python tests pick them up:**
  `maturin develop --release`.
- **RAVS is fail-closed.** Never trade correctness for cost. Unknown or
  low-confidence task types route to the flagship model.
- **Vault writes are auditable:** every belief artifact must include
  `claim_id`, `entity`, `confidence`, and `sources`. Don't skip these
  fields to land a faster path.
- **Token-negative learning is a hard contract.** The evolution daemon
  must not spend more on skill synthesis than the projected savings
  budget. If you change the synthesis path, update the spend gate.

## When you change behavior visible in the README

The README has a Benchmarks section with reproducible commands
(`python bench/trust_bench.py`, `python -m tests.verify_claims`,
`python -m bench.swebench_retrieval`, etc.). If you change a path that
those benchmarks measure, **re-run them and update the numbers in the
README in the same commit.** Drift between the README and what the
benchmarks actually print is the worst kind of trust break.

## Build & test commands

```bash
pip install -e ".[full]"              # Python install with extras
maturin develop --release             # Rust → Python bindings
pytest tests/ -v --tb=short --timeout=60
cd entroly-core && cargo test --lib   # Rust unit tests
ruff check entroly/                    # Python lint
cd entroly-core && cargo clippy --all-targets -- -D warnings
```

## Run

```bash
entroly                # Start MCP server (STDIO)
entroly proxy          # HTTP reverse proxy on localhost:9377
entroly go             # Full onboarding (detect IDE, generate config)
entroly dashboard      # Interactive dashboard
entroly health         # Codebase health grade (A–F)
```

## What to avoid

- **Don't add fictional benchmarks or claims to the README.** Every
  number must be reproducible by the script next to it.
- **Don't hard-code paths in `bench/tuning_config.json`** — autotune
  rewrites it; manual edits get clobbered.
- **Don't break the proxy in flight.** Existing users have
  `ANTHROPIC_BASE_URL=http://localhost:9377` pointed at this code.
  Backwards-compatible changes only on the proxy contract.
- **Don't mock the Rust engine in tests** that the autotuner reads
  from. The benchmark harness is read-only by contract — autotune
  reads its results but never modifies it.

## Pointers

- [`README.md`](../../README.md) — the public face
- [`docs/DETAILS.md`](../../docs/DETAILS.md) — architecture deep dive
- [`cookbook/README.md`](../../cookbook/README.md) — usage recipes
- [`CLAUDE.md`](../../CLAUDE.md) — build/test reference for AI agents
- [`bench/trust_bench.py`](../../bench/trust_bench.py) — 5-claim trust bench
- [`tests/verify_claims.py`](../../tests/verify_claims.py) — third-party-runnable claim suite
