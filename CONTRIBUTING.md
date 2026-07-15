# Contributing to Entroly

Thank you for helping make AI-agent context more reliable and auditable. Small,
well-proven changes are preferred to broad changes that are difficult to review.

## Before you start

- Use [GitHub Discussions](https://github.com/juyterman1000/entroly/discussions)
  for setup questions and design exploration.
- Search [existing issues](https://github.com/juyterman1000/entroly/issues)
  before opening a new one.
- Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
- For a large or compatibility-sensitive change, open a proposal before writing
  the implementation. This avoids spending time on a direction that cannot be
  maintained safely.

Good first contributions include documentation corrections, focused regression
tests, reproducible compatibility fixes, and benchmark fixtures with clear
provenance. Issues labelled
[`good first issue`](https://github.com/juyterman1000/entroly/labels/good%20first%20issue)
are intended to be independently completable.

## Development setup

Entroly supports Python 3.10 or newer. The Python-only path is enough for most
documentation, CLI, and orchestration work. Rust is required only when changing
the native engine or validating native parity.

```bash
git clone https://github.com/juyterman1000/entroly.git
cd entroly
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the surface you are changing:

```bash
# Python control plane and test dependency
python -m pip install -e ".[test]"

# Proxy work
python -m pip install -e ".[proxy,test]"

# All optional runtime features, including the published native wheel
python -m pip install -e ".[full,test]"
```

For local Rust changes, install a Rust toolchain and maturin, then rebuild the
extension after every native change:

```bash
python -m pip install "maturin>=1,<2"
cd entroly-core
maturin develop --release
cd ..
```

## Make a focused change

1. Branch from current `main`.
2. Keep unrelated refactors out of the pull request.
3. Add a regression test for every bug fix.
4. Preserve Entroly's trust invariants:
   - receipts remain inspectable;
   - omitted material remains recoverable where recovery is promised;
   - verification does not silently claim confidence;
   - local paths do not introduce surprise network calls;
   - provider request and cache semantics remain stable;
   - public claims cite a reproducible artifact and limitations.
5. Update documentation when behavior, configuration, or compatibility changes.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system boundaries and
[STYLE_GUIDE.md](STYLE_GUIDE.md) for code, error-message, and documentation
conventions.

## Run the relevant checks

Start with the smallest relevant test, then expand in proportion to risk.

```bash
# Python formatting/lint contract
python -m ruff check entroly tests

# Focused Python test
python -m pytest tests/test_<area>.py -q

# Full Python suite
python -m pytest tests -v --tb=short --timeout=60

# Public documentation and evidence contracts
python scripts/verify_readme.py
python scripts/verify_public_trust.py
```

Native changes also require:

```bash
cd entroly-core
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
```

OpenClaw plugin changes also require:

```bash
cd integrations/openclaw
npm test
npm run check
npm pack --dry-run
```

Packaging or release-surface changes require the version-sync tests plus a clean
package build. Do not update the Homebrew formula until the matching PyPI sdist
exists and its SHA-256 has been verified.

```bash
python -m pytest tests/test_release_surface.py tests/test_bump_version.py -q
python -m build
python -m twine check dist/*
```

## Pull requests

A reviewable pull request explains:

- the user-visible problem and outcome;
- the root cause for a bug fix;
- trust, compatibility, and release impact;
- exact commands and results used for validation;
- rollback behavior or why the change is naturally reversible.

Draft pull requests are welcome for early design feedback. A pull request should
be marked ready only when its focused tests pass and any known incomplete check
is stated plainly. Maintainers may ask to split changes that combine unrelated
runtime, release, benchmark, and documentation work.

## Benchmarks and scientific claims

Benchmark changes must include the workload, baseline version, token budget,
model and sampling settings where applicable, raw result artifact, and known
limitations. Do not generalize a result beyond the tested workload. See
[BENCHMARKS.md](BENCHMARKS.md) and [docs/public-evidence.md](docs/public-evidence.md).

## Community standards

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Project
roles and decision-making are described in [GOVERNANCE.md](GOVERNANCE.md).
