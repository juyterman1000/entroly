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
  <a href="https://pypistats.org/packages/entroly"><img src="https://img.shields.io/pypi/dm/entroly?color=blueviolet&label=PyPI%20downloads" alt="Entroly on PyPI downloads"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/dm/entroly?color=orange&label=npm%20downloads" alt="Entroly on npm downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="benchmarks/results/receipt_fragment_fidelity_default.json"><img src="https://img.shields.io/badge/Source_spans-5%2C117%2F5%2C117_verified-0A7B83" alt="5,117 of 5,117 native source fragments independently verified"></a>
  <a href="benchmarks/results/receipt_public_integrity.json"><img src="https://img.shields.io/badge/SDK_recovery-13%2F13_exact-blueviolet" alt="13 of 13 public SDK recovery probes exactly matched their source spans"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100K+ downloads</b> across verified package distribution channels.</p>

<p align="center"><b>⭐ If Entroly is useful to you, please star the repository on GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Star Entroly on GitHub</a> — it helps the project grow and reach more developers.</p>

## ⚡ Live Tokenomics

<p align="center"><b>Tokens saved</b> · <b>Estimated cost avoided</b> · <b>Compression savings</b> · <b>Tool-schema deferral savings</b></p>

| Live metric | Meaning | Source of truth |
|---|---|---|
| **Tokens saved** | Cumulative tokens reduced by the active Entroly workload | Local value ledger / proxy metrics |
| **Estimated cost avoided** | Modeled USD value of provider-bound input reduction using configured pricing | Local value ledger; provider invoice remains billing truth |
| **Compression tokens saved** | Savings produced by the compression pipeline | OTEL / Prometheus metrics |
| **Tool-schema tokens deferred** | Savings from deferring tool schemas until they are needed | OTEL / Prometheus metrics |

> **Live means measured by Entroly, not a fabricated global number.** Exact token and dollar totals stay local; privacy-safe product-health telemetry uses coarse buckets rather than exact global savings. Run `entroly value` or open `entroly dashboard` for live cumulative totals on your installation. For proxy observability, scrape `/metrics` and see [Metrics & Monitoring](docs/grafana/README.md).

<p align="center"><a href="docs/ai-efficiency.html">AI efficiency hub</a> · <a href="docs/ai-cost-optimization.html">Cost methodology</a> · <a href="docs/grafana/README.md">Metrics & monitoring</a> · <a href="docs/telemetry-privacy.md">Privacy-safe telemetry</a></p>

<p align="center">
  <b><a href="#what-is-entroly-in-plain-english">What is it?</a> · <a href="#live-tokenomics">Tokenomics</a> · <a href="#install">Install</a> · <a href="#quickstart--by-how-you-work">Quickstart</a> · <a href="#see-it-work-in-30-seconds">See it work</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#common-questions">Questions</a></b>
</p>

---
## What is Entroly? (in plain English)

AI coding assistants have a memory limit. Hand one your whole codebase and it
gets slow, expensive, and distracted — like giving someone a 500-page manual
when they only needed page 47.

**Entroly finds page 47.**

It sits between your code and the AI, reads everything, and passes along only
the parts that matter for the question actually being asked. Three things make
that safe to do:
|  |  |
|---|---|
| 💰 **Your bill goes down** | Fewer words sent to the AI means a smaller invoice. How much depends on the job — see the [real numbers](#benchmarks) below. |
| 🔍 **Nothing is lost** | Whatever Entroly sets aside is kept and can be pulled back *exactly* as it was, character for character. |
| 🧾 **You can check its work** | Every decision comes with a receipt: what was kept, what was left out, and why. |
**Do I have to change my code?** No. Entroly works with the tools you already
use — Claude Code, Cursor, Copilot and 30+ others — and runs in the background.

**Do I need to pay for anything to try it?** No. The two commands in the
[Install](#install) section below run entirely on your own machine, with no API
key, and show you real numbers on your own project before you connect anything
paid.

---
## Install

> **Not sure which one?** Pick **Python**. It's the complete version and what
> most people use. The others are alternate ways to run the same engine.
| Platform | Install | What you get |
|---|---|---|
| 🐍 **Python** (pip) — *recommended* | `pip install -U entroly` | Everything: the command-line tool, the server your AI editor talks to, and the code library |
| 📦 **Node / npm** | `npm install -g entroly` | The same engine, if you'd rather not install Python |
| 🦀 **Rust** (source build) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | One self-contained program, no Python or Node needed |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | The command-line tool on macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Runs in a container, nothing installed on your machine |
**Now check that it worked — free, offline, no API key:**

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

<sub>Both run locally. Neither one calls an AI or costs anything.</sub>

Extras (`entroly[proxy]`, `entroly[native]`, `entroly[full]`), the standalone
Rust binary, and uninstall steps: [Engine & install options](docs/DETAILS.md#engine--install-options).

---
## Quickstart — by how you work

> **Just want it working?** `pip install -U entroly && entroly go` — that's the
> whole thing. It finds your editor, sets itself up, and shows you a
> before/after dashboard. The rest of this table is for specific setups.
| Your situation | Do this | What it gets you |
|---|---|---|
| 🟢 **"I just want it on."** *(pip / Python user)* | `pip install -U entroly && entroly go` | Auto-detects your editor, wraps your agent, opens a dashboard showing tokens before and after |
| **"I use Node, not Python."** *(npm user)* | `npm install -g entroly && entroly init` | Same engine, nothing Python required |
| **"I want one binary, no runtime."** *(Rust user)* | `cargo build --release --bin entroly-rs --features proxy` (from `entroly-core/`) | A single native program with no dependencies |
| **"I use Claude Code / Cursor / Windsurf / VS Code."** *(MCP user)* | `entroly attach create --client claude --project . --ttl 4h --install` (or `entroly init` for Cursor/VS Code) | Your editor gets compression, receipts, and recovery as built-in tools — access expires on its own, and you change zero code |
| **"I'm building my own app in Python."** *(SDK user)* | `from entroly import compress, compress_messages, optimize` | Call it straight from your code, anywhere you assemble a prompt |
| **"I have an API key and my own app."** *(proxy user)* | `entroly proxy` → point `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL` at `localhost:9377` | Every request gets optimized on the way past — no code changes on your side |
**Why bother:** less unnecessary context reaches the model (lower bill, less
distraction for the model), nothing is silently lost (every drop is
recoverable and receipted), and you can prove it — `entroly verify-claims`
and `entroly simulate` show real numbers on your own repo before you connect
a paid key.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="fix the login bug")
```

```bash
entroly compress response.json --out small.json
entroly recover sha256:0b957c79... --out restored.json
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

The question that matters: **if you send less, does the AI start getting things
wrong?** These are standard public tests, run with and without Entroly.

*How to read this:* **Retention** is how well the AI still answered — 100% means
it did just as well on far less text. **Token savings** is how much less was
sent (and therefore paid for). Measured with `gpt-4o-mini`; intervals are Wilson 95% CIs.
| Benchmark | Baseline | With Entroly | Retention | Token savings |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |
| GSM8K | 85% | 85% | **100%** | pass-through* |
<sub>*pass-through: context already fit the budget, left unchanged. n=20–50 per row. Reproduce: `python benchmarks/run_readme_benchmarks.py` (needs `OPENAI_API_KEY`).</sub>

**Being straight with you:** look at the SQuAD 2.0 row — accuracy went *down*
(80% → 72%). Compression is a trade, not magic, and it doesn't win everywhere.
That's why `entroly simulate` exists: run it on your own project and see your
own numbers before you commit to anything.

Hallucination detection (WITNESS, local, no API): **84.92%** accuracy / **0.7976 AUROC** on 20,000 [HaluEval-QA](https://github.com/RUCAIBox/HaluEval) decisions — within the reported uncertainty of `gpt-4o-mini` as an API judge on the same shared sample.

Frozen evidence-selection benchmark (opt-in PRISM-R research prototype, not the default compressor): a disagreement guard kept the answer-bearing passage in 298 of 300 cases while selecting an average of 1.02 of 16 passages (paired exact McNemar p=0.21875 vs. BM25 alone) — this experiment measures retrieval of the known-answer passage, not generated-answer quality. Full protocol: [PRISM-R neural evidence frontier](docs/benchmarks/neural-evidence-frontier.md).

Recovery, latency, and head-to-head frontier results are in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** with raw artifacts linked. None of these numbers are a universal or production-savings guarantee for your workload — reproduce them on your own repo with `entroly simulate` and `entroly value`.

---
## Features

- **Picks first, shrinks second** — it works out which files actually answer your question, *then* compresses them.
- **Gives you the original back, exactly** — anything left out can be restored character-for-character and checked against a fingerprint.
- **Shows its work** — a receipt for every decision: what was kept, what was left out and why, and what risk remains.
- **Fact-checks answers** — compares what the AI said against the evidence it was given, on your machine, without paying for a second AI call.
- **Doesn't wreck your caching** — keeps the unchanging parts of your prompt stable so your provider's discount for repeated text still applies.
- **Rescues sessions before they crash** — when a conversation grows too big, it trims recoverable output instead of letting the provider reject the request mid-task.
- **Can route cheap work to cheap models** — optional and fail-closed when uncertain.

Runs as a **CLI**, **Python/TypeScript SDK**, **MCP server**, **HTTP proxy**, or **library import**. Full surface map: **[docs/product-surface.md](docs/product-surface.md)**. Architecture and Rust internals: **[docs/DETAILS.md](docs/DETAILS.md)**.

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

Status describes integration depth, not a savings guarantee — provider-observed savings require requests to actually traverse an Entroly proxy route. Entroly does not claim interception of GitHub-hosted subscription inference on Copilot's native path. Full compatibility matrix: **[docs/agent-compatibility.md](docs/agent-compatibility.md)**.

### Current model support

Entroly is tested against GPT-5.6, Gemini 3.6 Flash, Gemini 3.5 Flash-Lite, and local Ollama models. GPT-5.6 access is currently restricted to selected governments and trusted partners. See **[docs/model-support.html](https://juyterman1000.github.io/entroly/docs/model-support.html)** for the full verified model support matrix, transport boundaries, and fail-closed handling of gated announcements.

### NVIDIA Nemotron 3.5 Lightning with Ollama

Entroly supports NVIDIA Nemotron 3.5 Lightning through its loopback-only Ollama model discovery and OpenAI-compatible proxy path. Use the Ollama model ID `nemotron-3.5-lightning`. Entroly discovers the installed tag's context metadata rather than assuming the standard 1M and Apple-silicon MLX 256K variants are interchangeable. See **[docs/nemotron-3-5-lightning-ollama.html](https://juyterman1000.github.io/entroly/docs/nemotron-3-5-lightning-ollama.html)** for setup, tag-specific context limits, and privacy boundaries.



---
## When to use it · when to skip it

**Great fit:** large repos where the agent only sees a few files at a time · chatty multi-turn agents · anywhere you want answers checked against evidence · cutting a real, growing AI bill.

**Skip it:** tiny repos or short prompts that already fit the budget · judgment-heavy tasks where you always want the full flagship model.

---
## More commands

Also available: `entroly wrap`, `entroly unwrap`, `entroly serve`, `entroly daemon`, `entroly dashboard`, `entroly demo`, `entroly capabilities`, `entroly ingest`, `entroly select`, `entroly receipt`, `entroly explain`, `entroly context-commit`, `entroly proof`, `entroly benchmark`, `entroly cache`, `entroly ravs`, `entroly perf`, `entroly batch`. Full description: [command reference](docs/DETAILS.md#command-reference).

---
## Common questions

<details>
<summary><b>Will this change my code or my files?</b></summary>
<br>
No. Entroly reads your files and decides what to send to the AI. It never edits, moves, or deletes anything in your project.
</details>

<details>
<summary><b>Does my code get uploaded anywhere?</b></summary>
<br>
No. All selecting, compressing, and checking happens on your own machine. Entroly makes no outbound calls of its own — the only thing that leaves your computer is the request you were already sending to your AI provider, just smaller. There are no analytics on by default.
</details>

<details>
<summary><b>What if it leaves out something important?</b></summary>
<br>
Nothing is thrown away. Anything left out is stored and can be restored exactly as it was — `entroly recover` gives you back the original, character for character, and it's verified against a fingerprint.
</details>

<details>
<summary><b>How much money will this actually save me?</b></summary>
<br>
Honestly: it depends on your project. Run `entroly simulate` in your project — it's free, needs no API key, and estimates the reduction on your own files. If your prompts are already small, Entroly passes them through untouched.
</details>

<details>
<summary><b>I'm not a developer. Can I use this?</b></summary>
<br>
If you use an AI coding tool like Claude Code or Cursor, yes. Install it (`pip install -U entroly`), then run `entroly go` — it finds your editor, configures itself, and opens a dashboard.
</details>

<details>
<summary><b>Something broke / I'm stuck.</b></summary>
<br>
Run `entroly doctor`. If that doesn't sort it, [open an issue](https://github.com/juyterman1000/entroly/issues) or ask in [Discussions](https://github.com/juyterman1000/entroly/discussions).
</details>

---
## Docs & community

- **[AI efficiency hub](docs/ai-efficiency.html)** — token economics, AI cost optimization, memory, hallucination reduction, model routing, adaptive context, and verified code intelligence.
- **[AI cost optimization](docs/ai-cost-optimization.html)** — provider-bound input savings, billing boundaries, and workload-specific measurement.
- **[Token economics](docs/token-economics.html)** — token saving, context compression, cache-aware context control, and more room in the context window.
- **[Memory OS](docs/memory-os.html)** — budget-aware working, episodic, and semantic memory.
- **[Hallucination reduction](docs/hallucination-reduction.html)** — WITNESS evidence-support verification.
- **[Guarded model routing](docs/model-routing.html)** — RAVS routing, uncertainty control, and fail-closed escalation.
- **[Adaptive context](docs/adaptive-context.html)** — bounded self-improving context.
- **[Verified Code Context](docs/verified-code-context.md)** — parser-backed repository intelligence, typed graphs, architecture, value flow, LSP enrichment, source verification, and refactoring contracts.
- **[Full benchmark evidence](docs/BENCHMARKS.md)** — every number, protocol, artifact, and caveat, including [Model-triggered recovery](docs/benchmarks/model-triggered-recovery.md) and [Context Commit conformance](benchmarks/results/context_commit_conformance.json).
- **[Product surface map](docs/product-surface.md)** — CLI, SDK, MCP, proxy, verification, memory, security.
- **[Architecture & full spec](docs/DETAILS.md)** — Rust modules, compression, provenance, command reference.
- **[Agent compatibility](docs/agent-compatibility.md)** — every supported client and its exact authentication boundary.
- **[First-run trust guide](docs/first-run-trust.md)** — exactly what to run before wiring a paid model key.
- **[For teams](docs/for-teams.md)** — ROI, security, deployment one-pager.
- **[Limitations](docs/limitations.md)** — where Entroly helps, where it passes through, what it doesn't guarantee.
- **[Public evidence policy](docs/public-evidence.md)** — claim tiers and package links.
- **[Context Commits](docs/context-commits.md)** · **[Context Receipts](docs/DETAILS.md#context-receipts)** · **[Proof-guided recovery](docs/proof-guided-context-fixed-point.md)**
- **[Cookbook](cookbook/README.md)** — copy-paste recipes.
- **[Discord](https://juyterman1000.github.io/entroly/docs/discord.html)** · **[Discussions](https://github.com/juyterman1000/entroly/discussions)** · **[Issues](https://github.com/juyterman1000/entroly/issues)**

> Compressing a *bad* selection is still a bad selection. Entroly ranks first, then compresses — so the model gets structure, not just fewer tokens.

<p align="center"><sub>Apache-2.0 · local-first · no outbound analytics by default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>

<!-- mcp-name: io.github.juyterman1000/entroly -->