<p align="center">
  <img src="docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Drop-In Context Assurance to Lower AI Operational Cost</h1>

<p align="center"><b>Reduce unnecessary context without losing control of critical evidence.</b><br>
Select the highest-value evidence first, compress it, keep originals recoverable, and emit a receipt — without rewriting your codebase or agent architecture.</p>

<p align="center">
  <sub>Entroly is a local-first Context OS: content-addressed evidence, recoverable compression, and auditable receipts. Works through proxy, MCP, plugin, wrapper, and SDK paths with Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider, and OpenAI/Anthropic-compatible apps.</sub>
</p>

<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="https://pypistats.org/packages/entroly"><img src="https://img.shields.io/pypi/dm/entroly?color=blueviolet&label=PyPI%20downloads" alt="Entroly PyPI downloads per month"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/dm/entroly?color=orange&label=npm%20downloads" alt="Entroly npm downloads per month"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="benchmarks/results/receipt_fragment_fidelity_default.json"><img src="https://img.shields.io/badge/Source_spans-5%2C117%2F5%2C117_verified-0A7B83" alt="5,117 of 5,117 native source fragments independently verified"></a>
  <a href="benchmarks/results/receipt_public_integrity.json"><img src="https://img.shields.io/badge/SDK_recovery-13%2F13_exact-blueviolet" alt="13 of 13 public SDK recovery probes exactly matched their source spans"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><sub>~72,000 combined PyPI + npm downloads.</sub></p>

---

## Install

Pick your platform. Every path installs the same engine — Python is the
reference runtime; npm and Rust are alternate ways to run it.

| Platform | Install | What you get |
|---|---|---|
| **Python** (pip) | `pip install -U entroly` | Full CLI, SDK, MCP server, proxy, optional Rust acceleration |
| **Node / npm** | `npm install -g entroly` | Same engine via WASM, no Python required |
| **Rust** (source build) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | A standalone native proxy binary — no Python runtime at all |
| **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI, macOS/Linux |
| **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Containerized MCP server / proxy |

**Verify it worked, no API key needed:**

```bash
cd /your/repo
entroly verify-claims   # proves the installed trust/recovery path
entroly simulate        # estimates token reduction on this repo, no model call
```

Extras (`entroly[proxy]`, `entroly[native]`, `entroly[full]`), the standalone
Rust binary, and uninstall steps: [Engine & install options](docs/DETAILS.md#engine--install-options).

---

## Quickstart — by how you work

| I am a... | Do this | Why |
|---|---|---|
| **pip / Python user** | `pip install -U entroly && entroly go` | One command auto-detects your IDE, wraps your agent, opens a dashboard showing before/after tokens |
| **npm / Node user** | `npm install -g entroly && entroly init` | Same engine over WASM — no Python install needed |
| **Rust user** | `cargo build --release --bin entroly-rs --features proxy` (from `entroly-core/`) | A single native binary proxy — no runtime dependency at all |
| **MCP client user** (Claude Code, Cursor, Windsurf, VS Code) | `entroly attach create --client claude --project . --ttl 4h --install` (or `entroly init` for Cursor/VS Code) | Your agent gets scoped, expiring tool access — compression, receipts, recovery — with zero code changes |
| **SDK / library user** | `from entroly import compress, compress_messages, optimize` | Call it directly from Python; drop it into any pipeline that assembles a prompt |
| **API-key / custom app user** | `entroly proxy` → point `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL` at `localhost:9377` | Every request through the proxy is optimized transparently — no code changes on your side |

**Why bother:** less unnecessary context reaches the model (lower bill, less
distraction for the model), nothing is silently lost (every drop is
recoverable and receipted), and you can prove it — `entroly verify-claims`
and `entroly simulate` show real numbers on your own repo before you connect
a paid key.

```python
from entroly import compress, compress_messages, optimize

compressed = compress(api_response, budget=2000)          # query-agnostic
messages   = compress_messages(messages, budget=30000)    # whole conversation
context    = optimize(fragments, budget=8000, query="fix the login bug")  # task-conditioned
```

```bash
entroly compress response.json --out small.json    # compress one file, get a receipt
entroly recover sha256:0b957c79... --out restored.json   # get the exact original back
```

Full setup paths for every agent, IDE, and CI use case: [Get started in depth](docs/first-run-trust.md) · [Command reference](docs/DETAILS.md#command-reference).

---

## See it work in 30 seconds

Not mocked recordings — each video is rendered from a checked-in command that
verifies its source artifact before printing a number.

<p align="center">
  <a href="docs/assets/proof_local.mp4"><img src="docs/assets/proof_local.gif" width="700" alt="Entroly local verification: twelve checks pass without an API key"></a>
</p>

<p align="center"><code>entroly verify-claims</code> — import, compression, receipts, WITNESS checks, recovery, proxy routing, replay. No API key.</p>

<p align="center">
  <a href="docs/assets/proof_model_recovery.mp4"><img src="docs/assets/proof_model_recovery.gif" width="700" alt="Frozen model-recovery holdout: Entroly 24/24, published baseline 18/24"></a>
</p>

<p align="center">On a frozen 24-case holdout, Entroly answered <b>24/24</b>; a published baseline answered <b>18/24</b> at roughly 1.5x the effective context. <code>python scripts/readme_proof.py model-recovery</code></p>

<p align="center">
  <a href="docs/assets/proof_restart_recovery.mp4"><img src="docs/assets/proof_restart_recovery.gif" width="700" alt="Fresh-seed restart recovery: 66 of 66 payloads recovered byte-exactly"></a>
</p>

<p align="center">Omitted evidence recovered <b>byte-exact</b> after a process restart, 66/66 payloads. <code>python scripts/readme_proof.py restart-recovery</code></p>

Full protocols, sample sizes, and every caveat: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

---

## Benchmarks

Does compression hurt answers? Measured with `gpt-4o-mini`; intervals are Wilson 95% CIs.

| Benchmark | Baseline | With Entroly | Retention | Token savings |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |
| GSM8K | 85% | 85% | **100%** | pass-through* |

<sub>*pass-through: context already fit the budget, left unchanged. n=20–50 per row. Reproduce: `python benchmarks/run_readme_benchmarks.py` (needs `OPENAI_API_KEY`).</sub>

Hallucination detection (WITNESS, local, no API): **84.92%** accuracy / **0.7976 AUROC** on 20,000 [HaluEval-QA](https://github.com/RUCAIBox/HaluEval) decisions — within the reported uncertainty of `gpt-4o-mini` as an API judge on the same shared sample.

Recovery, latency, and head-to-head frontier results — [context-cap retention](docs/BENCHMARKS.md#matched-token-cap-active-context-quality-frontier-1059-source-candidate), [compression gauntlet](docs/BENCHMARKS.md#same-input-compression-gauntlet), [cross-process recovery](docs/BENCHMARKS.md#cross-process-recovery-holdout), [model-triggered recovery](docs/benchmarks/model-triggered-recovery.md) — are in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** with every raw artifact linked, including the [PRISM-R neural evidence frontier](docs/benchmarks/neural-evidence-frontier.md). None of these numbers are a universal or production-savings guarantee for your workload — reproduce them on your own repo with `entroly simulate` and `entroly value`.

---

## Features

- **Select, then compress** — ranks your repo or document set (BM25 + entropy + dependency graph), then sends only the answer-relevant context under a token budget. Most tools compress whatever text was already chosen; Entroly chooses first.
- **Exact recovery** — every compressed representation carries a digest- and length-verified handle back to the original bytes. `entroly compress` / `entroly recover` prove it across a process boundary.
- **Receipts** — every selection explains what was kept, what was omitted and why, dependency links, source-span digests, and residual risk.
- **Verify (WITNESS)** — checks an answer against supplied evidence locally, no extra model call.
- **Cache-safe live zone** — stable system/history bytes stay ahead of changing context, so provider prompt caching isn't broken by every turn.
- **Runaway-session rescue** — the proxy compacts recoverable tool output before a provider rejects an overlong request, instead of failing mid-session.
- **Route + learn** — optionally sends easy tasks to a cheaper model (fail-closed), and adapts local ranking signals from recorded outcomes — no embeddings API required.

Runs as a **CLI**, **Python/TypeScript SDK**, **MCP server**, **HTTP proxy**, or **library import**. Full surface map — every command, every SDK function, every MCP tool: **[docs/product-surface.md](docs/product-surface.md)**. Architecture and Rust internals: **[docs/DETAILS.md](docs/DETAILS.md)**.

---

## Works with your stack

| Agent / platform | Path | Status |
|---|---|---|
| Claude Code | Scoped MCP attachment; API-key proxy | Native |
| Codex CLI | Scoped MCP attachment; API-key proxy | Native |
| OpenClaw | Context-engine plugin + scoped MCP | Native |
| Cursor / Windsurf / VS Code | Automatic MCP config | Automatic |
| GitHub Copilot CLI | MCP (subscription) / proxy (BYOK) | Supported |
| Cortex Code | SDK/library boundary only | Not validated as a wrap target |
| Aider, OpenCode, and 30+ more | Session-scoped OpenAI-compatible proxy | One command |

Status describes integration depth, not a savings guarantee — provider-observed savings require requests to actually traverse an Entroly proxy route. Entroly does not claim interception of GitHub-hosted subscription inference on Copilot's native path. Full compatibility matrix with every client and its authentication boundary: **[docs/agent-compatibility.md](docs/agent-compatibility.md)**.

---

## When to use it · when to skip it

**Great fit:** large repos where the agent only sees a few files at a time · chatty multi-turn agents (cache alignment compounds savings) · anywhere you want answers checked against evidence · cutting a real, growing AI bill.

**Skip it:** tiny repos or short prompts that already fit the budget (Entroly passes them through unchanged) · judgment-heavy tasks where you always want the full flagship model.

---

## More commands

Also available: `entroly wrap`, `entroly unwrap`, `entroly serve`,
`entroly daemon`, `entroly dashboard`, `entroly demo`, `entroly capabilities`,
`entroly ingest`, `entroly select`, `entroly receipt`, `entroly explain`,
`entroly context-commit`, `entroly proof`, `entroly benchmark`,
`entroly cache`, `entroly ravs`, `entroly perf`, `entroly batch`. Full
description of every command: [command reference](docs/DETAILS.md#command-reference).

---

## Docs & community

- **[Full benchmark evidence](docs/BENCHMARKS.md)** — every number, protocol, artifact, and caveat, including [Context Commit conformance](benchmarks/results/context_commit_conformance.json) (128/128 deterministic replay, 576/576 exact recovery, 768/768 tamper detection).
- **[Product surface map](docs/product-surface.md)** — CLI, SDK, MCP, proxy, npm/WASM, verification, memory, security.
- **[Architecture & full spec](docs/DETAILS.md)** — Rust modules, 3-resolution compression, provenance, command reference.
- **[Agent compatibility](docs/agent-compatibility.md)** — every supported client and its exact authentication boundary.
- **[First-run trust guide](docs/first-run-trust.md)** — exactly what to run before wiring a paid model key.
- **[For teams](docs/for-teams.md)** — ROI, security, deployment one-pager.
- **[Limitations](docs/limitations.md)** — where Entroly helps, where it passes through, what it doesn't guarantee.
- **[Public evidence policy](docs/public-evidence.md)** — claim tiers and package links.
- **[Context Commits](docs/context-commits.md)** · **[Context Receipts](docs/DETAILS.md#context-receipts)** · **[Proof-guided recovery](docs/proof-guided-context-fixed-point.md)** — portable, recoverable, verifiable context artifacts.
- **[Cookbook](cookbook/README.md)** — copy-paste recipes.
- **[Discord](https://juyterman1000.github.io/entroly/docs/discord.html)** · **[Discussions](https://github.com/juyterman1000/entroly/discussions)** · **[Issues](https://github.com/juyterman1000/entroly/issues)**

> Compressing a *bad* selection is still a bad selection. Entroly ranks first, then compresses — so the model gets structure, not just fewer tokens.

<p align="center"><sub>Apache-2.0 · local-first · no outbound analytics by default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>

<!-- mcp-name: io.github.juyterman1000/entroly -->
