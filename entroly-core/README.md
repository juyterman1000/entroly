# entroly-core

Rust core for [Entroly](https://github.com/juyterman1000/entroly) — information-theoretic context optimization for AI coding agents.

Provides high-performance PyO3 bindings for:

- **Knapsack optimizer** — 0/1 DP context selection within token budget
- **Shannon entropy scorer** — boilerplate detection, information density
- **SimHash deduplication** — near-duplicate fragment detection
- **Query analysis** — TF-IDF vagueness scoring, heuristic refinement
- **SAST scanner** — 30+ security rules (XSS, SQL injection, secrets, unsafe memory)
- **LSH index** — approximate nearest-neighbor semantic recall
- **PRISM RL optimizer** — online feedback-driven fragment weight learning

## Install

```bash
pip install entroly-core
```

Prebuilt wheels for Linux, macOS, Windows (Python 3.10–3.13).

## Usage

Usually used via the higher-level `entroly` package:

```bash
pip install entroly
entroly  # starts the MCP server
```

Or directly:
```python
from entroly_core import ContextFragment, py_knapsack_optimize, py_shannon_entropy
```

## License

MIT
