# entroly-core

Rust core for [Entroly](https://github.com/juyterman1000/entroly), the
information-theoretic context optimization engine for AI coding agents.

Provides high-performance PyO3 bindings for:

- **Knapsack optimizer** - 0/1 DP context selection within token budget
- **Shannon entropy scorer** - boilerplate detection, information density
- **SimHash deduplication** - near-duplicate fragment detection
- **Query analysis** - TF-IDF vagueness scoring, heuristic refinement
- **SAST scanner** - security rules for common vulnerability classes
- **LSH index** - approximate nearest-neighbor semantic recall
- **PRISM RL optimizer** - online feedback-driven fragment weight learning

## Install

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -U "entroly-core>=1.0.37"
```

Prebuilt abi3 wheels are published for Linux, macOS, and Windows. One
wheel per platform covers Python 3.10+ including Python 3.14. If pip
tries to compile an old source distribution such as `0.2.0`, upgrade pip
and retry with `--no-cache-dir`; the intended path is the prebuilt wheel,
not a local PyO3 build.

## Usage

Usually used via the higher-level `entroly` package:

```bash
python -m pip install entroly
entroly  # starts the MCP server
```

Or directly:

```python
from entroly_core import ContextFragment, py_knapsack_optimize, py_shannon_entropy
```

## License

Apache-2.0