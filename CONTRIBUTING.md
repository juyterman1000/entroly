# Contributing to Entroly

Thanks for your interest in contributing! PRs are welcome.

## Development Setup

### Prerequisites

- Python 3.10+
- Rust toolchain (for `entroly-core`)
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)

### Getting Started

```bash
# Clone the repo
git clone https://github.com/juyterman1000/entroly.git
cd entroly

# Create a virtualenv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Build the Rust core and install in dev mode
cd entroly-core
maturin develop --release
cd ..

# Install the Python package with all extras
pip install -e ".[full]" psutil

# Install dev tools
pip install pytest pytest-cov ruff
```

## Running Tests

```bash
# Rust tests
cd entroly-core && cargo test --lib && cd ..

# Python tests
pytest tests/ -v --tb=short

# entroly-core integration tests
pytest entroly-core/tests/ -v --tb=short

# Functional test
python tests/functional_test.py
```

## Linting

```bash
# Python
ruff check entroly/

# Rust
cd entroly-core && cargo clippy --all-targets -- -D warnings
```

## Submitting a PR

1. Fork the repo and create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass and linting is clean
4. Open a pull request with a clear description of the change

## Code of Conduct

Be respectful and constructive. We're all here to build something great.
