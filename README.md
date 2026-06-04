<p align="center">
  <a href="docs/i18n/README.zh-CN.md">中文</a> •
  <a href="docs/i18n/README.ja.md">日本語</a> •
  <a href="docs/i18n/README.ko.md">한국어</a> •
  <a href="docs/i18n/README.pt-BR.md">Português</a> •
  <a href="docs/i18n/README.es.md">Español</a> •
  <a href="docs/i18n/README.de.md">Deutsch</a> •
  <a href="docs/i18n/README.fr.md">Français</a> •
  <a href="docs/i18n/README.ru.md">Русский</a> •
  <a href="docs/i18n/README.hi.md">हिन्दी</a> •
  <a href="docs/i18n/README.tr.md">Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="160" alt="Entroly">
</p>

<h1 align="center">Entroly</h1>

<p align="center"><b>Cut your AI coding bill — without losing answer quality.</b></p>

<p align="center">
  <sub>70–95% fewer input tokens · $0 hallucination guard · local-first · reversible · works with Claude, Cursor, Codex, Aider + 34 more</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="PyPI">
  <img src="https://img.shields.io/npm/v/entroly-wasm?color=red&label=npm" alt="npm">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License">
  <img src="https://img.shields.io/badge/Token_Savings-tested_70--95%25-brightgreen" alt="Token savings">
  <img src="https://img.shields.io/badge/Hallucination-HaluEval--QA_0.844_AUROC_·_%240-blueviolet" alt="Hallucination guard">
  <img src="https://img.shields.io/badge/Engine-Rust_+_WASM-orange?logo=rust" alt="Rust + WASM">
</p>

<p align="center">
  <code>pip install entroly && cd /your/repo && entroly go</code>
</p>

<p align="center">
  <a href="#get-started-60-seconds"><b>Get started</b></a> ·
  <a href="#proof"><b>Proof</b></a> ·
  <a href="#works-with-your-stack"><b>Integrations</b></a> ·
  <a href="#whats-inside"><b>What's inside</b></a> ·
  <a href="docs/DETAILS.md"><b>Architecture</b></a> ·
  <a href="docs/for-teams.md"><b>For teams</b></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/entroly/entroly-context-compression"><img src="https://img.shields.io/badge/▶_Try_it_live-no_install-FF4B4B?logo=huggingface&logoColor=white" alt="Live demo"></a>
  &nbsp;
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><img src="https://img.shields.io/badge/See_the_dashboard-live-2EA44F" alt="Dashboard"></a>
</p>

---

## What it does

Your AI coding agent dumps huge, repetitive context into every request. You pay for all of it, and the model still misses files it can't see. Entroly fixes both — **locally, on your machine.**

- **Compress** — ranks your whole repo, then sends only the answer-relevant files under a token budget. 70–95% fewer input tokens on large repos.
- **Verify** — WITNESS checks the model's answer against the evidence it was given and flags unsupported claims. $0, ~3 ms, no extra API call.
- **Route** — sends easy, repeated tasks to a cheaper model and keeps the flagship for hard ones (opt-in, fail-closed).
- **Cache-align** — keeps the injected prefix byte-stable so the provider's cache keeps hitting (Anthropic gives up to 90% off cached tokens).
- **Learn** — improves which files it picks for *your* workflow from local feedback. No embeddings API, no training job.

Use it however you work: **wrap** your agent, run it as a **proxy**, plug it in as an **MCP server**, or import the **library**.

---

## How it works (30 seconds)

```
your agent  ──►  Entroly (local)  ──►  LLM provider
                 │
                 ├─ rank the repo        (BM25 + entropy + dep-graph)
                 ├─ select under budget  (knapsack, reversible)
                 ├─ cache-align prefix    (keep provider cache hot)
                 └─ verify the reply      (WITNESS hallucination guard)
```

Critical files go in full. Supporting files become signatures. Everything else becomes a reference you can expand on demand — so the model gets a **broader** view of your codebase in a **smaller** prompt. Nothing is lost: every compressed fragment is fully retrievable.

---

## Get started (60 seconds)

```bash
pip install entroly        # or: npm i -g entroly  ·  brew install juyterman1000/entroly/entroly
```

**1. One command — auto-detects your IDE, wraps your agent, opens the dashboard:**

```bash
cd /your/repo && entroly go
```

**2. Or wrap a specific agent:**

```bash
entroly wrap claude     # Claude Code
entroly wrap cursor     # Cursor
entroly wrap codex      # Codex CLI
entroly wrap aider      # Aider
```

**3. Or run the proxy — zero code changes, any language:**

```bash
entroly proxy                                   # http://localhost:9377
ANTHROPIC_BASE_URL=http://localhost:9377     your-app
OPENAI_BASE_URL=http://localhost:9377/v1     your-app
```

**4. Or measure it on your own repo first:**

```bash
entroly demo            # before/after token + cost estimate
entroly verify-claims   # runs the packaged self-test, writes a JSON report
```

> Local-first: your code is indexed and selected on-device, never sent anywhere for analysis. Apache-2.0. No outbound analytics by default.

---

## Proof

Every number below is reproducible and backed by a committed JSON artifact you can audit — not a screenshot.

**Token savings** (this repo, `entroly verify-claims`, local, no API):

| Budget | Token reduction |
|---|---|
| 8K  | **99.1%** |
| 32K | **96.7%** |
| average across workloads | **87.0%** |

**Accuracy retention** — does compression hurt answers? Measured with `gpt-4o-mini`; intervals are Wilson 95% CIs. Each row links its raw result file.

| Benchmark | n | Budget | Baseline | With Entroly | Retention | Token savings |
|---|---|---|---|---|---|---|
| [NeedleInAHaystack](benchmarks/results/needle_accuracy.json) | 20 | 2K | 100% | 100% | **100%** | **99.5%** |
| [LongBench (HotpotQA)](benchmarks/results/longbench_accuracy.json) | 50 | 2K | 64% | 66% | **103%** | **85.3%** |
| [Berkeley Function Calling](benchmarks/results/bfcl_accuracy.json) | 50 | 500 | 100% | 100% | **100%** | **79.3%** |
| [SQuAD 2.0](benchmarks/results/squad_accuracy.json) | 50 | 100 | 80% | 72% | **90%** | **43.8%** |
| [GSM8K](benchmarks/results/gsm8k_accuracy.json) | 20 | 50K | 85% | 85% | **100%** | pass-through* |

<sub>*pass-through: context already fit the budget, so Entroly left it unchanged. Reproduce: `python benchmarks/run_readme_benchmarks.py` (needs `OPENAI_API_KEY`). Full table + MMLU/TruthfulQA in [DETAILS](docs/DETAILS.md).</sub>

**Hallucination guard** — [HaluEval-QA](https://github.com/RUCAIBox/HaluEval), standard protocol, GPT-judge baseline on identical data:

| System | Accuracy | AUROC | Cost / latency |
|---|---|---|---|
| **WITNESS + STAVE** (default) | **85.8%** | **0.844** | **$0, ~3 ms/decision** |
| gpt-4o-mini (grounded judge) | 86.3% | — | LLM call |
| gpt-3.5-turbo (HaluEval paper) | 62.6% | — | LLM call |

<sub>$0, zero-network verifier that statistically ties a strong LLM judge. Reproduce: `python benchmarks/halueval_qa_faithful.py`. [Proof JSON](benchmarks/results/stave_benchmark.json).</sub>

---

## Works with your stack

`entroly wrap <agent>` picks the best integration for each tool — proxy env-wrap for CLIs, auto-merged `mcp.json` for MCP-aware IDEs, or a copy-paste endpoint hint.

**Wrap in one command:** `claude` · `cursor` · `codex` · `aider` · `gemini` · `windsurf` · `vscode` · `zed` · `cline` · `continue` and **28 more**.

<details>
<summary><b>Full agent list (38 targets)</b></summary>

| Type | Agents |
|---|---|
| **CLI (env-wrap + exec)** | Claude Code, Codex CLI, Aider, Gemini CLI, Qwen Code, OpenCode, Charm CRUSH, Hermes, Pi, Ollama |
| **MCP IDEs (auto-merge `mcp.json`)** | Cursor, Windsurf, VS Code, Claude Desktop, Claude Code (MCP), Zed |
| **Copy-paste endpoint** | Cline, Roo Code, Continue, Cody, Amp, Kiro, Qoder, Trae, Antigravity, Amazon Q, Verdent, JetBrains AI, Helix, Tabby, Twinny, Sublime, Emacs, Neovim, Fitten, Tabnine, Supermaven |

Any tool that supports a custom `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` works via the proxy. Run `entroly wrap` (no agent) for the full grouped list. Use wrappers only with tools whose terms permit local proxies / custom endpoints.
</details>

**As a library** (LangChain, LlamaIndex, your own code):

```python
from entroly import compress, compress_messages, optimize

compressed = compress(api_response, budget=2000)          # query-agnostic
messages   = compress_messages(messages, budget=30000)    # whole conversation
context    = optimize(fragments, budget=8000, query="fix the login bug")  # task-conditioned
```

**In CI** — fail the build if a prompt blows the token budget:

```yaml
- run: pip install entroly && entroly batch --budget 8000 --fail-over-budget
```

---

## When to use it · when to skip

**Great fit**
- Large repos where the agent only sees a few files at a time
- Chatty, multi-turn agents (cache alignment compounds the savings)
- Anywhere you want answers checked against evidence before you trust them
- Teams trying to cut a real, growing AI bill

**Skip it (it'll just pass through)**
- Tiny repos or short prompts that already fit the budget
- Judgment-heavy tasks where you want the full flagship model every time

---

## What's inside

Most people install Entroly for input-token compression. It actually ships **19 local cost-saving mechanisms** across input, inference, output, verification, and learning — each one readable in the source with a committed benchmark where applicable.

<details>
<summary><b>The 19 levers (and the file that implements each)</b></summary>

| # | Lever | Win | Source |
|---|---|---|---|
| 1 | Context compression (knapsack + 9 compressors + dep-graph) | 39–99% input tokens | `proxy_transform.py`, `qccr.py` |
| 2 | WITNESS + STAVE hallucination gateway | AUROC 0.844, $0 | `witness.py`, `verifiers/stave.py` |
| 3 | Cache Aligner | up to 90% off cached calls | `cache_aligner.py` |
| 4 | Escalation cascade (conformally calibrated) | avoids most flagship calls | `escalation.py` |
| 5 | Conformal cascade | proven cost/coverage tradeoff | `conformal_cascade.py` |
| 6 | RAVS Bayesian router | routes easy tasks to cheaper models | `ravs/router.py` |
| 7 | Fast-path crystallized skills | 100% LLM cost saved on cache hits | `fast_path.py` |
| 8 | Adaptive compression budget | right-sizes budget per query | `adaptive_budget.py` |
| 9 | Entropic conversation pruning | flattens history-growth cost | `proxy_transform.py` |
| 10 | Shell-output compression | 60–95% on tool output | `proxy_transform.py`, `shell_codec.py` |
| 11 | Response distillation | fewer output tokens billed | `proxy_transform.py` |
| 12 | Local DeBERTa NLI (opt-in) | $0 offline NLI | `witness.py` |
| 13 | EICV suppressor | stops bad info propagating | `eicv_suppressor.py` |
| 14 | PRISM 5D adaptive weights | quality improves with use | `online_learner.py`, `prism.rs` |
| 15 | Federation (opt-in) | amortized cold-start | `federation.py` |
| 16 | Entropic Shell Codec | universal tool-output fallback | `shell_codec.py` |
| 17 | Semantic Resolution Protocol | 40–70% fewer tokens on file reads | `semantic_resolution.py` |
| 18 | Adversarial Context Firewall | blocks prompt-injection / poisoning | `context_firewall.py` |
| 19 | Witness-Verified Handoff | filters hallucinations between agents | `verified_handoff.py` |

Most levers are **multiplicative**: input compression × cache alignment × cheaper-model routing × output distillation can leave well under 1% of the original input-token spend on the bill. Per-lever contribution shows up in the dashboard's Cost Intelligence panel. Full math and proofs in [docs/DETAILS.md](docs/DETAILS.md).
</details>

<details>
<summary><b>Engine & install options</b></summary>

Python is the reference runtime; the Rust core (via PyO3) does the heavy compute at 50–100× Python speed, and the same engine ships to Node via WASM.

```bash
pip install entroly            # core: MCP server + Python engine
pip install entroly[proxy]     # + HTTP proxy
pip install entroly[native]    # + Rust engine
pip install entroly[full]      # everything

npm install -g entroly         # WASM runtime, no Python needed
docker pull ghcr.io/juyterman1000/entroly:latest
```

**Single binary, no Python** — a standalone Rust proxy that auto-detects Anthropic/OpenAI/Gemini and stays cache-aligned:

```bash
cd entroly/entroly-core && cargo build --release --bin entroly-rs --features proxy
./target/release/entroly-rs proxy --upstream https://api.anthropic.com
```
</details>

---

## WITNESS — check answers before you trust them

```bash
entroly witness --context-file evidence.txt --output-file answer.txt --mode strict
entroly proxy --witness strict --witness-profile rag    # suppress unsupported claims inline
```

Profiles tune false-positive behavior per workload (`rag`, `qa`, `code` fail closed; `chat`, `summary` warn). Every non-streaming response gets a proof certificate; the dashboard shows flagged claims, evidence snippets, and suppression counts. Optional offline DeBERTa NLI (`ENTROLY_LOCAL_NLI=1`) raises accuracy further at $0.

---

## Compared to

| | **Entroly** | Compression tools | Top-K / RAG | Raw truncation |
|---|---|---|---|---|
| Approach | Rank → select → compress | Compress whatever's given | Embedding retrieval | Cut off |
| Token savings | **70–95%** (large repos) | 50–70% | 30–50% | 0% |
| Quality loss | **None measured** | 2–5% | Variable | High |
| Needs embeddings API | **No** | Varies | Yes | No |
| Reversible | **Yes** | Varies | Yes | No |
| Learns over time | **Yes (PRISM)** | No | No | No |
| Verifies the answer | **Yes (WITNESS)** | No | No | No |

> Compressing a *bad* selection is still a bad selection. Entroly ranks first, then compresses — so the model gets structure, not just fewer tokens.

---

## Docs & community

- **[Architecture & full spec](docs/DETAILS.md)** — Rust modules, 3-resolution compression, provenance, RAG comparison, SDK, LangChain.
- **[For teams](docs/for-teams.md)** — ROI, security, deployment one-pager.
- **[Cookbook](cookbook/README.md)** — copy-paste recipes for common workflows.
- **[Discussions](https://github.com/juyterman1000/entroly/discussions)** · **[Issues](https://github.com/juyterman1000/entroly/issues)**

<p align="center"><sub>Apache-2.0 · local-first · no outbound analytics by default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
