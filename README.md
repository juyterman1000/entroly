<p align="center">
  <img src="docs/assets/logo.png" width="200" alt="Entroly Logo">
</p>

<h1 align="center">Entroly</h1>

<p align="center">
  <b>Context optimization for AI coding agents.</b>
  <br/>
  <i>Your AI sees your entire codebase. You pay for 40% fewer tokens. Zero config changes.</i>
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

## What Entroly Does

Every AI coding tool — Cursor, Copilot, Claude Code, Cody — stuffs tokens into the context window until it's full, then cuts. This means your AI tool sees 5-10 files and the rest of your codebase is invisible.

Entroly fixes this. It compresses your **entire codebase** into the context window at variable resolution, removes duplicate and boilerplate content, and learns which context produces better AI responses over time.

You install it once. It runs invisibly. Your AI gives better answers and you spend less on tokens.

---

## What You Get

| Benefit | Details |
|---------|---------|
| **40% fewer tokens per request** | Duplicate code, boilerplate, and low-information content are stripped automatically |
| **100% codebase visibility** | Every file is represented — critical files in full, supporting files as signatures, peripheral files as references |
| **AI responses improve over time** | Reinforcement learning adjusts context selection weights from session outcomes — no manual tuning |
| **Built-in security scanning** | 55 SAST rules catch hardcoded secrets, SQL injection, command injection, and 5 more CWE categories in selected context |
| **Codebase health grades** | Clone detection, dead symbol finder, god file detection — get an A-F health grade for your project |
| **< 10ms overhead** | The Rust engine adds under 10ms per request. You won't notice it |
| **Works with any AI tool** | MCP server for Cursor/Claude Code, or transparent HTTP proxy for anything else |
| **Runs on Linux, macOS, and Windows** | Native support. No WSL required on Windows. Docker optional on all platforms |

---

## Install

```bash
pip install entroly
```

That's it. One command. Works on Linux, macOS, and Windows.

**Windows users:** If `pip` is not on your PATH, use `python -m pip install entroly`.

### Connect to your AI tool

**Cursor** — run `entroly init` in your project. It generates `.cursor/mcp.json` automatically.

**Claude Code** — run `claude mcp add entroly -- entroly`.

**VS Code / Windsurf** — run `entroly init`. Auto-detected.

**Any other AI tool** — use proxy mode:
```bash
pip install entroly[proxy]
entroly proxy --quality balanced
```
Then point your AI tool's API base URL to `http://localhost:9377/v1`. Done.

### Verify it's working

```bash
entroly status     # check if the server/proxy is running
entroly demo       # see a before/after comparison of token savings on your project
entroly dashboard  # open the live metrics dashboard at localhost:9378
```

### Install options

```bash
pip install entroly           # Core — MCP server + Python fallback engine
pip install entroly[proxy]    # Add proxy mode (transparent HTTP interception)
pip install entroly[native]   # Add native Rust engine (50-100x faster)
pip install entroly[full]     # Everything
```

### Docker

```bash
docker pull ghcr.io/juyterman1000/entroly:latest
docker run --rm -p 9377:9377 -p 9378:9378 -v .:/workspace:ro ghcr.io/juyterman1000/entroly:latest
```

Multi-arch: `linux/amd64` and `linux/arm64` (Apple Silicon, AWS Graviton).

Or with Docker Compose: `docker compose up -d`

---

## Platform Support

| | Linux | macOS | Windows |
|--|-------|-------|---------|
| **Python 3.10+** | Yes | Yes | Yes |
| **Pre-built Rust wheel** | Yes | Yes (Intel + Apple Silicon) | Yes |
| **Docker** | Optional | Optional (Docker Desktop) | Optional (Docker Desktop) |
| **WSL required** | N/A | N/A | No |
| **Admin rights required** | No | No | No |

---

## CLI Commands

| Command | What it does |
|---------|-------------|
| `entroly init` | Detects your project and AI tool, generates config — one command setup |
| `entroly proxy` | Starts the invisible proxy. Point your AI tool to localhost:9377 |
| `entroly demo` | Shows before/after token savings on your actual project |
| `entroly doctor` | Runs 7 diagnostic checks — finds problems before you do |
| `entroly dashboard` | Live metrics: tokens saved, cost reduction, health grade, security findings |
| `entroly health` | Codebase health grade (A-F): clones, dead code, god files, architecture violations |
| `entroly role` | Weight presets for your workflow: `frontend`, `backend`, `sre`, `data`, `fullstack` |
| `entroly autotune` | Auto-optimizes engine parameters using mutation-based search |
| `entroly digest` | Weekly summary of value delivered — tokens saved, cost reduction, improvements |
| `entroly status` | Check if server/proxy/dashboard are running |
| `entroly migrate` | Upgrades config and checkpoints when you update Entroly |
| `entroly clean` | Clear cached state and start fresh |
| `entroly benchmark` | Run competitive benchmark: Entroly vs raw context vs top-K retrieval |
| `entroly completions` | Generate shell completions for bash, zsh, or fish |

---

## How It Works (the short version)

1. **Entroly indexes your codebase** — auto-detects languages, builds dependency graphs, extracts code signatures
2. **When your AI tool makes a request**, Entroly selects the optimal context — not just "top 5 similar files" but a mathematically optimal subset that maximizes information within the token budget
3. **Duplicate and boilerplate content is removed** — SimHash fingerprinting detects near-duplicate code in O(1)
4. **Every file is represented at the right resolution** — critical files get full content, related files get signatures, peripheral files get references
5. **The system learns from outcomes** — reinforcement learning adjusts selection weights after each session, so context quality improves the more you use it
6. **Security scanning runs automatically** — 55 SAST rules flag vulnerabilities in selected context before your AI sees them

---

## Production Ready

Entroly is built for real-world reliability, not demos.

- **Connection recovery** — auto-reconnects dropped connections without restarting
- **Large file protection** — 500 KB ceiling prevents out-of-memory on giant logs or vendor files
- **Binary file detection** — 40+ file types (images, audio, video, archives, databases) are auto-skipped
- **Crash recovery** — gzipped checkpoints restore state in under 100ms
- **Cross-platform file locking** — safe to run multiple instances
- **Schema migration** — `entroly migrate` handles config upgrades between versions
- **Fragment feedback** — `POST /feedback` lets your AI tool rate context quality, improving future selections
- **Explainable decisions** — `GET /explain` shows exactly why each code fragment was included or excluded

---

## Need Help?

**Self-service:**
```bash
entroly doctor    # runs 7 diagnostic checks automatically
entroly --help    # see all available commands
```

**Get support:**

If you run into any issue, email **autobotbugfix@gmail.com** with:
1. The output of `entroly doctor`
2. A screenshot of the error
3. Your OS (Windows/macOS/Linux) and Python version

We respond within 24 hours.

**Common issues:**

<details>
<summary><b>macOS: "externally-managed-environment" error</b></summary>

Homebrew Python requires a virtual environment:
```bash
python3 -m venv ~/.venvs/entroly
source ~/.venvs/entroly/bin/activate
pip install entroly[full]
```
</details>

<details>
<summary><b>Windows: pip not found</b></summary>

```powershell
python -m pip install entroly
```
</details>

<details>
<summary><b>Port 9377 already in use</b></summary>

```bash
entroly proxy --port 9378
```
</details>

<details>
<summary><b>Rust engine not loading</b></summary>

Entroly falls back to the Python engine automatically. For the Rust speedup:
```bash
pip install entroly[native]
```
If no pre-built wheel exists for your platform, install the [Rust toolchain](https://rustup.rs/) first.
</details>

---

## Part of the Ebbiforge Ecosystem

Entroly integrates with [hippocampus-sharp-memory](https://pypi.org/project/hippocampus-sharp-memory/) for persistent cross-session memory and [Ebbiforge](https://pypi.org/project/ebbiforge/) for TF embeddings and RL weight learning. Both are optional.

---

## Quality Presets

Control the speed vs. quality tradeoff:

```bash
entroly proxy --quality speed       # minimal optimization, lowest latency
entroly proxy --quality fast        # light optimization
entroly proxy --quality balanced    # recommended for most projects
entroly proxy --quality quality     # deeper analysis, more context diversity
entroly proxy --quality max         # full pipeline, best results
entroly proxy --quality 0.7         # or any float from 0.0 to 1.0
```

## Environment Variables

| Variable | Default | What it does |
|----------|---------|-------------|
| `ENTROLY_QUALITY` | `0.5` | Quality dial (0.0-1.0 or preset name) |
| `ENTROLY_PROXY_PORT` | `9377` | Proxy port |
| `ENTROLY_MAX_FILES` | `5000` | Max files to auto-index |
| `ENTROLY_RATE_LIMIT` | `0` | Max requests/min (0 = unlimited) |
| `ENTROLY_NO_DOCKER` | - | Skip Docker, run natively |
| `ENTROLY_MCP_TRANSPORT` | `stdio` | MCP transport (stdio or sse) |

---

<details>
<summary><b>Technical Deep Dive</b></summary>

## How Entroly Compares

| | Cody / Copilot | Entroly |
|--|----------------|---------|
| **Approach** | Embedding similarity search | Information-theoretic compression + online RL |
| **Coverage** | 5-10 files (the rest is invisible) | 100% codebase at variable resolution |
| **Selection** | Top-K by cosine distance | KKT-optimal bisection with submodular diversity |
| **Dedup** | None | SimHash + LSH in O(1) |
| **Learning** | Static | REINFORCE with KKT-consistent baseline |
| **Security** | None | Built-in SAST (55 rules, taint-aware) |
| **Temperature** | User-set | Self-calibrating (no tuning needed) |

## Architecture

Hybrid Rust + Python. All math runs in Rust via PyO3 (50-100x faster). MCP protocol and orchestration run in Python.

```
+-----------------------------------------------------------+
|  IDE (Cursor / Claude Code / Cline / Copilot)             |
|                                                           |
|  +---- MCP mode ----+    +---- Proxy mode ----+          |
|  | entroly MCP server|    | localhost:9377     |          |
|  | (JSON-RPC stdio)  |    | (HTTP reverse proxy)|         |
|  +--------+----------+    +--------+-----------+          |
|           |                        |                      |
|  +--------v------------------------v-----------+          |
|  |          Entroly Engine (Python)             |          |
|  |  +-------------------------------------+    |          |
|  |  |  entroly-core (Rust via PyO3)       |    |          |
|  |  |  15 modules . 340 KB . 126 tests    |    |          |
|  |  +-------------------------------------+    |          |
|  +---------------------------------------------+          |
+-----------------------------------------------------------+
```

## Rust Core (15 modules)

| Module | What | How |
|--------|------|-----|
| **hierarchical.rs** | 3-level codebase compression | Skeleton map + dep-graph expansion + knapsack-optimal fragments |
| **knapsack.rs** | Context subset selection | KKT dual bisection O(30N) or exact 0/1 DP |
| **knapsack_sds.rs** | Information-Optimal Selection | Submodular diversity + multi-resolution knapsack |
| **prism.rs** | Weight optimizer | Spectral natural gradient on 4x4 gradient covariance |
| **entropy.rs** | Information density scoring | Shannon entropy + boilerplate detection + redundancy |
| **depgraph.rs** | Dependency graph | Auto-linking imports, type refs, function calls |
| **skeleton.rs** | Code skeleton extraction | Preserves signatures, strips bodies (60-80% reduction) |
| **dedup.rs** | Duplicate detection | 64-bit SimHash, Hamming threshold 3, LSH buckets |
| **lsh.rs** | Semantic recall index | 12-table multi-probe LSH, ~3 us over 100K fragments |
| **sast.rs** | Security scanning | 55 rules, 8 CWE categories, taint-flow analysis |
| **health.rs** | Codebase health | Clone detection, dead symbols, god files, arch violations |
| **guardrails.rs** | Safety-critical pinning | Criticality levels with task-aware budget multipliers |
| **query.rs** | Query analysis | Vagueness scoring, keyword extraction, intent classification |
| **fragment.rs** | Core data structure | Content, metadata, scoring dimensions, SimHash fingerprint |
| **lib.rs** | PyO3 bridge | All modules exposed to Python, 126 tests |

## Python Layer

| Module | What |
|--------|------|
| **proxy.py** | Invisible HTTP reverse proxy |
| **proxy_transform.py** | Request parsing, context formatting, temperature calibration |
| **server.py** | MCP server with 10+ tools and Python fallbacks |
| **auto_index.py** | File-system crawler for automatic codebase indexing |
| **checkpoint.py** | Gzipped JSON state serialization |
| **prefetch.py** | Predictive context pre-loading |
| **provenance.py** | Hallucination risk detection |
| **multimodal.py** | Image OCR, diagram parsing, voice transcript extraction |

## MCP Tools

| Tool | Purpose |
|------|---------|
| `remember_fragment` | Store context with auto-dedup, entropy scoring, dep linking |
| `optimize_context` | Select optimal context subset for a token budget |
| `recall_relevant` | Sub-linear semantic recall via multi-probe LSH |
| `record_outcome` | Feed the reinforcement learning loop |
| `explain_context` | Per-fragment scoring breakdown |
| `checkpoint_state` | Save full session state |
| `resume_state` | Restore from checkpoint |
| `prefetch_related` | Predict and pre-load likely-needed context |
| `get_stats` | Session statistics and cost savings |
| `health_check` | Clone detection, dead symbols, god files |

## Novel Algorithms

**Entropic Context Compression (ECC)** — 3-level hierarchical codebase representation. L1: skeleton map of all files (5% budget). L2: dependency cluster expansion (25%). L3: submodular diversified full fragments (70%).

**IOS (Information-Optimal Selection)** — Combines Submodular Diversity Selection with Multi-Resolution Knapsack in one greedy pass. (1-1/e) optimality guarantee.

**KKT-REINFORCE** — The dual variable from the forward budget constraint serves as a per-item REINFORCE baseline. Forward and backward use the same probability.

**PRISM** — Natural gradient preconditioning via exact Jacobi eigendecomposition of the 4x4 gradient covariance.

**ADGT** — Duality gap as a self-regulating temperature signal. No decay constant needed.

**PCNT** — PRISM spectral condition number as a weight-uncertainty-aware temperature modulator.

## References

Shannon (1948), Charikar (2002), Ebbinghaus (1885), Nemhauser-Wolsey-Fisher (1978), Sviridenko (2004), Boyd & Vandenberghe (Convex Optimization), Williams (1992), LLMLingua (EMNLP 2023), RepoFormer (ICML 2024), FILM-7B (NeurIPS 2024), CodeSage (ICLR 2024).

</details>

---

## License

MIT
