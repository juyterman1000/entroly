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
  <img src="docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<p align="center"><b>The local context OS for AI coding agents.</b><br>
Give Claude, OpenAI, Gemini, Cursor, Codex, and Aider the right evidence with fewer tokens, recoverable context, receipts, memory, gateway controls, and a <b>$0 hallucination guard</b>.</p>

<p align="center">
  <sub>Drop-in for <b>Cursor, Claude Code, Codex, Aider + 34 more</b> and custom providers — 60s, no code changes.</sub>
</p>

<p align="center">
  <sub>Auditable context control plane · every answer gets a receipt: what was used, what was omitted, why, and the risks that remain · local-first · Rust + WASM · reversible · savings measured on real workloads</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="PyPI">
  <img src="https://img.shields.io/npm/v/entroly-wasm?color=red&label=npm" alt="npm">
  <a href="https://github.com/juyterman1000/entroly/stargazers"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="GitHub stars"></a>
  <a href="https://juyterman1000.github.io/entroly/docs/discord.html"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License">
  <img src="https://img.shields.io/badge/Token_Savings-workload_dependent-brightgreen" alt="Token savings">
  <img src="https://img.shields.io/badge/Hallucination-HaluEval--QA_0.844_AUROC_·_%240-blueviolet" alt="Hallucination guard">
  <img src="https://img.shields.io/badge/Engine-Rust_+_WASM-orange?logo=rust" alt="Rust + WASM">
</p>

<p align="center">
  <code>pip install -U entroly && cd /your/repo && entroly verify-claims && entroly simulate</code>
</p>

<p align="center">
  <a href="#get-started-60-seconds"><b>Get started</b></a> ·
  <a href="#proof"><b>Proof</b></a> ·
  <a href="#works-with-your-stack"><b>Integrations</b></a> ·
  <a href="#whats-inside"><b>What's inside</b></a> ·
  <a href="docs/DETAILS.md"><b>Architecture</b></a> ·
  <a href="docs/for-teams.md"><b>For teams</b></a> ·
  <a href="docs/limitations.md"><b>Limitations</b></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/entroly/entroly-context-compression"><img src="https://img.shields.io/badge/▶_Try_it_live-no_install-FF4B4B?logo=huggingface&logoColor=white" alt="Live demo"></a>
  &nbsp;
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><img src="https://img.shields.io/badge/See_the_dashboard-live-2EA44F" alt="Dashboard"></a>
</p>

<p align="center">
  <b>Deciding whether to star?</b> Run the no-key proof first: <code>entroly verify-claims && entroly simulate</code>.<br>
  <sub>If it finds meaningful savings or gives you auditable receipts on your repo, star it so other agent builders can find it. If it does not, open an issue with the verification JSON.</sub>
</p>

---

## What it does

Entroly is an auditable context control plane for AI agents. It decides what context to send, records what it left out, and produces a receipt you can inspect before trusting a hard multi-file answer.

Most compression tools shrink whatever text the agent already chose. Entroly starts one step earlier: it chooses the highest-value evidence first, compresses only after selection, keeps originals recoverable, then verifies the answer against the evidence.

- **Receipts** - every selection run can explain selected chunks, omitted nearby evidence, dependency links, fingerprints, token ratio, and residual risks.
- **Select** - ranks your repo or document set, then sends the answer-relevant context under a token budget.
- **Verify** - WITNESS checks the model's answer against the evidence it was given and flags unsupported claims. $0, ~3 ms, no extra API call.
- **Route** - sends easy, repeated tasks to a cheaper model and keeps the flagship for hard ones (opt-in, fail-closed).
- **Cache-align** - keeps the injected prefix byte-stable so provider prefix caches can keep hitting where terms and API shape allow it.
- **Learn** - improves which files it picks for *your* workflow from local feedback. No embeddings API, no training job.

Use it however you work: **wrap** your agent, run it as a **proxy**, plug it in as an **MCP server**, or import the **library**.

### Why teams care

| What usually breaks AI coding at scale | What Entroly adds |
|---|---|
| Context windows fill with logs, duplicate files, and irrelevant chunks | Budgeted selection that favors answer-critical files, dependency links, failures, and anomalies |
| Token savings look good but quality silently drops | Accuracy-retention benchmarks, receipts, and WITNESS verification |
| Agents lose the exact line, stack trace, or omitted file they later need | Reversible compressed fragments and retrieval handles |
| First-time setup depends on one IDE or one provider | CLI, SDK, MCP, proxy, npm, PyPI, Docker, and local simulation paths |
| Enterprise teams need proof, not screenshots | Committed JSON artifacts, local self-tests, and reproducible commands |

### Product surface

Entroly ships as a full local runtime, not one proxy command:

| Surface | What users get |
|---|---|
| **CLI** | `verify-claims`, `simulate`, `perf`, `wrap`, `proxy`, `serve`, `daemon`, `benchmark`, `witness`, `receipt`, `audit`, `doctor`, `health`, `batch`, `learn`, `ravs`, `cache`, and more |
| **SDK** | `compress`, `compress_messages`, `optimize`, `verify`, hallucination detection, Context Receipts, localizers, cache alignment, cost cortex, Memory OS |
| **MCP server** | Context optimization, exact retrieval, receipts, recovery, feedback, security scans, codebase health, smart reads, belief verification, response verification |
| **Proxy** | Anthropic/OpenAI-compatible local optimization path for API-key users and custom apps |
| **Node/WASM** | `entroly`, `entroly-mcp`, and `entroly-wasm` packages for npm users |
| **Trust layer** | WITNESS, EICV, STAVE, receipt proofs, provenance checks, prompt-injection scanning, and local verification reports |
| **Memory/session intelligence** | Memory OS, Memory Fabric, long-term memory, session digests, checkpoint relevance, cache-retention forecasting, and lifetime value tracking |
| **Multimodal intake** | Diff, diagram, voice, image, and structured-context ingestion with provider-aware image token estimates and compliance-gated optimization |
| **Gateway/accounting** | Provider capability planning, failover policy, redaction receipts, usage ledger, cache routing, spend math, and budget harnesses |
| **Knowledge vault/CogOps** | Belief compilation, vault search, workspace change sync, epistemic routing, verification engines, and flow orchestration |
| **Framework/event gateways** | LangChain helpers, Ebbiforge provenance auditing, AgentSkills export, Hermes, Slack, Discord, and Telegram gateway hooks for teams that want operational feedback loops |
| **Self-improvement** | PRISM/RAVS feedback, autotune, skill crystallization, promotion gates, evolution logging, and budget-gated skill synthesis |
| **Observability** | Dashboard, daemon supervisor, control plane, health reports, value tracker, release-surface checks, and local JSON proof reports |

Under the hood, the Python control plane is backed by a Rust/WASM engine with BM25, entropy scoring, SimHash dedup, dependency graphs, knapsack/IOS selection, EGSC caching, PRISM learning, SAST, QCCR, EICV, witness checks, CogOps, cache economics, and memory primitives.

See the full code-derived map in [docs/product-surface.md](docs/product-surface.md).

---

## How it works (30 seconds)

```
your agent  ──►  Entroly (local)  ──►  LLM provider
                 │
                 ├─ rank the repo        (BM25 + entropy + dep-graph)
                 ├─ select under budget  (knapsack, reversible)
                 ├─ emit receipt         (included, omitted, risks)
                 ├─ cache-align prefix    (keep provider cache hot)
                 └─ verify the reply      (WITNESS hallucination guard)
```

Critical files go in full. Supporting files become signatures. Everything else becomes a reference you can expand on demand — so the model gets a **broader** view of your codebase in a **smaller** prompt. Nothing is lost: every compressed fragment is fully retrievable.

---

## Get started (60 seconds)

The best first run is local and proof-driven. It should work before you connect
an API key, proxy, paid model, or enterprise setup.

```bash
pip install -U entroly     # or: npm i -g entroly  ·  brew install juyterman1000/entroly/entroly
```

**1. Prove the package works on your machine:**

```bash
entroly verify-claims      # SDK import, indexing, optimization, exact recovery, engine mode
entroly simulate           # local no-LLM savings estimate on your current repo
```

**2. Pick one integration path:**

| You are using | Run this | Why |
|---|---|---|
| Claude Code subscription | `claude mcp add entroly -- entroly` | Adds Entroly tools without proxy/API-key assumptions |
| Cursor, VS Code, Windsurf, or another MCP client | `entroly init` or `entroly serve` | Local MCP tools for context, receipts, recovery, and feedback |
| Pay-as-you-go API keys or a custom app | `entroly proxy` | Transparent Anthropic/OpenAI-compatible optimization path |
| Python app | `from entroly import compress, compress_messages, optimize` | Direct SDK control |
| Node/npm workflow | `npm install -g entroly` | WASM runtime without a Python-first setup |
| CI or release gate | `entroly batch --budget 8000 --fail-over-budget` | Enforce prompt budgets before merge |

**3. Best setup for Claude Code subscription users:**

```bash
claude mcp add entroly -- entroly
```

Claude Code stays your client. Entroly adds local tools for compression, retrieval, receipts, and savings reports.

**4. One command — auto-detects your IDE, wraps your agent, opens the dashboard:**

```bash
cd /your/repo && entroly go
```

**5. Or wrap a specific agent:**

```bash
entroly wrap claude     # Claude Code
entroly wrap cursor     # Cursor
entroly wrap codex      # Codex CLI
entroly wrap aider      # Aider
```

**6. Or run the proxy — best for pay-as-you-go API keys and custom apps:**

```bash
entroly proxy                                   # http://localhost:9377
ANTHROPIC_BASE_URL=http://localhost:9377     your-app
OPENAI_BASE_URL=http://localhost:9377/v1     your-app
GOOGLE_GEMINI_BASE_URL=http://localhost:9377/v1beta     your-app
```

**7. Or measure it on your own repo first:**

```bash
entroly demo            # before/after token + cost estimate
entroly simulate        # local no-LLM savings estimate
entroly perf            # local no-LLM savings + optimizer latency
entroly verify-claims   # runs the packaged self-test, writes a JSON report
```

> Local-first: your code is indexed and selected on-device, never sent anywhere for analysis. Apache-2.0. No outbound analytics by default.

### First-run success contract

Entroly should feel useful before you connect a paid model key:

- `entroly verify-claims` proves SDK import, local indexing, optimization, exact recovery, and native/pure-Python engine mode.
- `entroly simulate` shows the likely token reduction on your repo without making an LLM call.
- MCP setup works for Claude Code subscription users who do not want proxy/API-key mode.
- Proxy mode is available when you control the provider key and want transparent request optimization.
- npm/WASM is available for Node-first users, but Python remains the fullest CLI/SDK path.

If your repo is tiny or already under budget, Entroly should say so and pass through rather than invent fake savings.

---

## Context Receipts

Entroly gives every AI answer a context receipt: what was used, what was omitted, why, and what risks remain. This is built for hard multi-document work such as contracts, policies, addenda, code reviews, and audit evidence where "top-k chunks" is not enough.

```bash
entroly ingest ./docs
entroly select --query "Does this contract have a change-of-control clause?" --budget 8000
entroly receipt .entroly/receipts/cr_example.json
entroly audit .entroly/session_chain.json
entroly explain --why-omitted chk_example --receipt .entroly/receipts/cr_example.json
```

The receipt JSON includes selected chunks, omitted relevant chunks, ranking reasons, dependency links, source fingerprints, token ratio, warnings, and a reproducibility hash. The Markdown report is designed for human review before a compressed context is trusted.

Implementation notes:

- Rust core (`entroly-core/src/context_receipts.rs`) handles deterministic ingestion, BM25-style ranking, dependency scans, selection, and hashes when the native wheel is available.
- Python control plane (`entroly/context_receipts/`) provides CLI wiring and a pure-Python fallback for source checkouts.
- The semantic/vector scorer and reranker are explicit extension points; the local MVP ships with lexical scoring and dependency heuristics, not a legal-accuracy guarantee.

Examples:

- [Example receipt JSON](docs/examples/context_receipt.json)
- [Example Markdown report](docs/examples/context_receipt.md)
- [Limitations](docs/limitations.md#context-receipts)

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

`entroly wrap <agent>` picks the best integration for each tool — proxy env-wrap for CLIs, auto-merged `mcp.json` for MCP-aware IDEs, or a best-effort endpoint/config hint.

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

## Why Entroly is different

The winning product is not the one that makes the prompt smallest. It is the one that helps the model do the best work for the fewest tokens.

Entroly is built around that trust contract: select the right evidence, compress supporting material, keep originals recoverable, emit a receipt, and verify the answer against the retained evidence.

| Layer | Entroly answer |
|---|---|
| **Context engine** | BM25 + entropy + dependency graph + knapsack/IOS selection under budget |
| **Compression/recovery** | Evidence-Locked Compression, exact CCR handles, omitted-span retrieval store |
| **Trust** | Context Receipts, WITNESS, EICV, STAVE, provenance, receipt proofs |
| **Gateway** | Provider adapters, cache-aware routing, usage ledger, cost cortex, harness budgets |
| **Memory/session** | Memory OS, Memory Fabric, long-term memory, checkpoint relevance, session digests, value tracking |
| **Multimodal** | Diff, diagram, voice, image, and structured-context ingestion with provider-aware token estimates |
| **CogOps/vault** | Belief compiler, vault search, epistemic router, flow orchestrator, verification engine, workspace change sync |
| **Learning** | Feedback, PRISM/RAVS, archetype adaptation, cache and routing signals |
| **Self-improvement** | Autotune, dreaming loops, reward crystallization, skill synthesis, promotion gates, rollback, optional federation |
| **Security** | SAST, prompt-injection scanning, redaction policy, path containment |
| **Observability** | Dashboard, daemon, control plane, health reports, usage accounting, local proof JSON |
| **Runtime** | Python SDK/CLI/MCP plus Rust native engine and Node/WASM runtime |

The goal is **same-quality or better model work at materially lower token cost**.

---

## Self-improving local runtime

Entroly has a guarded self-improvement loop. It is designed to learn from real outcomes without letting adaptation run wild.

| Loop | What it does |
|---|---|
| **Feedback** | `record_test_result`, `record_command_exit`, `record_ci_result`, and `record_edit_outcome` turn real outcomes into learning signals |
| **PRISM/RAVS** | Online Bayesian weights and honest-outcome correction move selection toward what actually passes tests, CI, and user acceptance |
| **Autotune/dreaming** | Idle/offline loops test weight perturbations against benchmark cases before promotion |
| **Reward crystallization** | Repeated high-reward query families become reusable skills with statistical lower-bound checks |
| **Skill synthesis** | Structural synthesis tries local, deterministic skill generation before any LLM fallback |
| **Promotion gate** | Shadow policies must be non-inferior before promotion; rollback triggers on repair/retry/success regression |
| **Budget guardrail** | Evolution is intended to stay token-negative by spending only a bounded fraction of measured lifetime savings |
| **Optional federation** | Weight contributions can be shared only when explicitly enabled |

This is the important distinction: Entroly does not just remember context. It can learn which context-selection strategies, routes, and skills actually produce successful work.

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

<details>
<summary><b>Command reference</b></summary>

| Command | What it does |
|---|---|
| `entroly go` | One shot: detect IDE, wrap your agent, open the dashboard |
| `entroly wrap <agent>` | Wrap a specific coding agent (38 supported) |
| `entroly proxy` | Start the HTTP proxy on `localhost:9377` |
| `entroly serve` | Start the MCP server |
| `entroly daemon` | Supervise proxy + dashboard + MCP + file watcher |
| `entroly dashboard` | Open the live metrics dashboard |
| `entroly demo` | Before/after token + cost estimate on your repo |
| `entroly ingest` | Ingest documents into a local Context Receipt index |
| `entroly select` | Select context under budget and write a Context Receipt |
| `entroly receipt` | Render a Context Receipt as a Markdown report |
| `entroly explain` | Explain why a chunk was selected or omitted |
| `entroly simulate` | Local no-LLM savings estimate with an explicit baseline |
| `entroly perf` | Local no-LLM savings and optimizer latency |
| `entroly benchmark` | Local comparison: Entroly vs raw context vs top-K |
| `entroly health` | Codebase health grade (A–F) |
| `entroly cache stats` | Persistent cross-session cache stats |
| `entroly ravs report` | Model-routing cost-savings report |
| `entroly witness` | Check an answer against supplied evidence |
| `entroly verify-claims` | Run the packaged self-test → JSON report |

</details>

- **[Architecture & full spec](docs/DETAILS.md)** — Rust modules, 3-resolution compression, provenance, RAG comparison, SDK, LangChain.
- **[Product surface map](docs/product-surface.md)** — CLI, SDK, MCP, proxy, npm/WASM, verification, memory, and security surfaces.
- **[First-run trust guide](docs/first-run-trust.md)** — exactly what a new user should run before wiring a paid model.
- **[For teams](docs/for-teams.md)** — ROI, security, deployment one-pager.
- **[Limitations](docs/limitations.md)** — where Entroly helps, where it passes through, and what it does not guarantee.
- **[Cookbook](cookbook/README.md)** — copy-paste recipes for common workflows.
- **[Discord Community](https://juyterman1000.github.io/entroly/docs/discord.html)** · **[Discussions](https://github.com/juyterman1000/entroly/discussions)** · **[Issues](https://github.com/juyterman1000/entroly/issues)**

<p align="center"><sub>Apache-2.0 · local-first · no outbound analytics by default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>

<!-- mcp-name: io.github.juyterman1000/entroly -->
