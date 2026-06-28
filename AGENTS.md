# AGENTS.md

This repository is configured for OpenAI Codex CLI and other agentic coding tools.

## Use gstack with Codex

When gstack is installed for Codex, use the gstack workflow for non-trivial Entroly changes:

```text
/office-hours -> /plan-ceo-review -> /plan-eng-review -> implement -> /review -> /qa or targeted tests -> /ship -> /retro
```

Install gstack for Codex with:

```bash
git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git ~/gstack
cd ~/gstack
./setup --host codex
```

This installs gstack skills under `~/.codex/skills/gstack-*/`.

## Entroly operating rule

Entroly is an auditable context-control plane. Optimize for trust before speed:

- preserve receipt honesty
- preserve reversible context compression
- keep WITNESS / RAVS fail-closed
- avoid surprise remote calls
- keep prompt/cache prefixes stable unless intentionally changed
- keep Python, Rust, WASM, npm, Homebrew, docs, and native-status versions aligned
- never make savings or quality claims without baseline, token budget, workload, and caveats

## Default Codex workflow

For any meaningful change:

1. Define the smallest useful user-facing improvement.
2. Write a short plan before editing.
3. Identify affected surfaces: selection, receipts, WITNESS, RAVS, proxy, CLI, packaging, docs, or release.
4. Make the smallest safe patch.
5. Run targeted tests first.
6. Expand to release tests only if packaging/native/release surfaces changed.
7. Summarize risk, test evidence, and rollback path.

## Commands

```bash
pip install -e ".[full]"
pytest tests/ -v --tb=short --timeout=60
ruff check entroly/
python tests/functional_test.py
```

Rust/native changes:

```bash
maturin develop --release
cd entroly-core && cargo test --lib
cd entroly-core && cargo clippy --all-targets -- -D warnings
```

Release/package changes:

```bash
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
entroly doctor
```

For Homebrew updates, derive the sdist URL and SHA-256 from the PyPI JSON API before editing the formula.
