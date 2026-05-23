<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly — Context Engineering Engine">
</p>

<h1 align="center">Entroly</h1>

<h3 align="center">Evidence-aware context engineering with local learning loops.<br/>Context selection, output verification, and optional federated learning.</h3>

<p align="center">
  <b>Local Context Engine &nbsp;•&nbsp; Structural Skill Synthesis &nbsp;•&nbsp; MCP-Native</b><br/>
  <i>Entroly is a context engine with a local self-improvement loop: it detects coverage gaps, synthesizes new skills from your codebase's structure, benchmarks them, and promotes the winners — using structural analysis rather than model calls when possible. Budget-gated. Deterministic. Local-first. On large-repo workloads, release checks observed <b>70–95%</b> input token reduction.</i>
</p>

<p align="center">
  <code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>npm install -g entroly && entroly</code>
</p>

<p align="center">
  <a href="#the-problem">Problem</a> &bull;
  <a href="#the-fix">Solution</a> &bull;
  <a href="#30-second-install">Install</a> &bull;
  <a href="#see-it-in-action">Demo</a> &bull;
  <a href="#works-with-compatible-tools">Integrations</a> &bull;
  <a href="#how-it-works">Architecture</a> &bull;
  <a href="#self-improving-ai-runtime--gets-smarter-every-session">Self-Improving</a> &bull;
  <a href="#pillar-4--federated-swarm-learning">Federation</a> &bull;
  <a href="#pillar-5--response-distillation">Distillation</a> &bull;
  <a href="https://github.com/juyterman1000/entroly/discussions">Community</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/entroly"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="npm"></a>
  <img src="https://img.shields.io/badge/Rust_Engine-50--100x_faster-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License">
  <img src="https://img.shields.io/badge/Tests-484_Passing-success" alt="Tests">
  <img src="https://img.shields.io/badge/Latency-<10ms-purple" alt="Latency">
</p>

---

## Example Evolution Trace

Example trace from this repo's local development vault:

```
[detect]    gap observed → entity="auth", miss_count=3
[synthesize] StructuralSynthesizer ($0, deterministic, no LLM)
[benchmark]  skill=ddb2e2969bb0 → fitness 1.0 (1 pass / 0 fail, 338 ms)
[promote]    status: draft → promoted
[registry]   .entroly/vault/evolution/registry.md updated
[spend]      $0.0000 — invariant C_spent ≤ τ·S(t) holds
```

The structural synthesizer reads your code graph rather than calling an LLM. When structural synthesis can't solve a gap, the LLM fallback is **budget-gated by cumulative token savings** — intended to keep learning cost below lifetime savings.

→ See [The 3 Pillars of Zero-Token Autonomy](#the-3-pillars-of-zero-token-autonomy) for how.

---

## The Problem

AI coding tools that send raw file dumps often face the same limitation:

> **The model may only receive a handful of files at a time. The rest of your codebase is not represented.**

This causes:
- **Hallucinated function calls** — the AI invents APIs that don't exist
- **Broken imports** — it references modules it can't see
- **Missed dependencies** — it changes `auth.py` without knowing about `auth_config.py`
- **Wasted tokens** — raw-dumping files burns your budget on boilerplate and duplicates
- **Incomplete answers** — without broader context, models may give incomplete solutions

Entroly addresses this by selecting compact, variable-resolution context from the full repository.

---

## The Fix

**Entroly selects context from your entire codebase at variable resolution.**

| What changes | Without Entroly | With Entroly |
|---|---|---|
| **Files visible to AI** | 5-10 files | **Supported files selected at variable resolution** |
| **Tokens per request** | ~186,000 raw example | **9,300 – 55,000 in listed release examples** |
| **Cost per 1K requests** | depends on provider/model | **lower when input tokens drop** |
| **AI answer grounding** | depends on supplied context | **auditable against selected evidence** |
| **Setup time** | manual prompt engineering | **30 seconds** |
| **Overhead** | N/A | **< 10ms local core paths** |

Critical files appear in full. Supporting files appear as signatures. Everything else appears as references. The AI receives broader structural context within a smaller token budget.

### How is this different from RAG?

| | RAG (vector search) | Entroly (context engineering) |
|--|---|---|
| **What it sends** | Top-K similar chunks | **Selected codebase context** at variable resolution |
| **Handles duplicates** | No — sends same code 3x | **SimHash dedup** in O(1) |
| **Dependency-aware** | No | **Yes** — auto-includes related files |
| **Learns from usage** | No | **Yes** — RL optimizes from AI response quality |
| **Needs embeddings API** | Yes (extra cost + latency) | **No** — runs locally |
| **Budgeted selection** | Approximate | **Knapsack optimizer over Entroly's scoring objective** |

---

## See It In Action

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/demo.svg" alt="Entroly Demo — AI context optimization, 70-95% token savings" width="800">
</p>

```bash
pip install entroly && entroly demo    # see savings on YOUR codebase
```

> Open the [interactive demo](docs/assets/demo.html) for the animated experience.

---

## 30-Second Install

**Python:**
```bash
pip install entroly[full]
entroly go
```

**Node.js / TypeScript:**
```bash
npm install entroly-wasm
npx entroly-wasm serve     # MCP server
npx entroly-wasm optimize  # CLI optimizer
npx entroly-wasm demo      # see savings on YOUR codebase
```

Or use the short compatibility package:

```bash
npm install -g entroly
entroly serve
entroly optimize 8000 "fix the auth bug"
entroly demo
```

Both npm packages run the full Rust engine natively in Node.js — **no Python required**.

**That's it.** `entroly go` (Python) or `entroly serve` / `npx entroly-wasm serve` (Node.js) auto-detects your IDE, starts the engine, and begins optimizing. Point your AI tool to `http://localhost:9377/v1`.

### Or step by step

```bash
# Python
pip install entroly                # core engine
entroly init                       # detect IDE + generate config
entroly proxy --quality balanced   # start proxy

# Node.js
npm install -g entroly             # short alias for the WASM runtime
entroly serve                      # start MCP server

# Or install the WASM package directly
npm install entroly-wasm           # WASM engine, zero dependencies
npx entroly-wasm serve             # start MCP server
```

### npm packages

| Package | What you get |
|---------|---|
| `npm install -g entroly` | Short CLI alias that installs and delegates to `entroly-wasm` |
| `npm install entroly-wasm` | Full Rust engine via WebAssembly — MCP server, CLI, autotune, health |

### pip packages

| Package | What you get |
|---------|---|
| `pip install entroly` | Core — MCP server + Python engine |
| `pip install entroly[proxy]` | + HTTP proxy mode |
| `pip install entroly[native]` | + Rust engine (50-100x faster) |
| `pip install entroly[full]` | Everything |

### Docker

```bash
docker pull ghcr.io/juyterman1000/entroly:latest
docker run --rm -p 9377:9377 -p 9378:9378 -v .:/workspace:ro ghcr.io/juyterman1000/entroly:latest
```

---

## Works With Compatible Tools

| AI Tool | Setup | Method |
|---------|-------|--------|
| **Cursor** | `entroly init` | MCP server |
| **Claude Code** | `claude mcp add entroly -- entroly` | MCP server |
| **VS Code MCP clients** | `entroly init` | MCP server |
| **Windsurf** | `entroly init` | MCP server |
| **Cline** | `entroly init` | MCP server |
| **Compatible LLM APIs** | `entroly proxy` | HTTP proxy |

---

## Why Developers Choose Entroly

> **"Entroly handled the context selection so I stopped manually pasting code."**

- **Low config** — `entroly go` handles the common local setup path. No embeddings API is required.
- **Instant results** — See the difference on your first request. No training period.
- **Privacy-first** — Local indexing and selection. Your code is not sent to Entroly's servers.
- **Tested** — 436 tests, crash recovery, connection auto-reconnect, cross-platform file locking.
- **Built-in security** — 55 SAST rules catch hardcoded secrets, SQL injection, command injection across 8 CWE categories.
- **Codebase health grades** — Clone detection, dead code finder, god file detection. Get an A-F grade.

---

## Beyond Basic Token Saving Proxies

When developers search for **"token saving proxy"** or **"context compression"**, Entroly offers distinct advantages over standard alternatives:

| Feature | Entroly | Basic Proxies |
|---|---|---|
| **Setup** | Zero-config (`entroly go`) | Requires YAML/embedding setup |
| **Codebase Intelligence** | Deep (dead code, god files) | Proxy transport only |
| **Security** | 55 SAST rules (catches hardcoded secrets) | None builtin |
| **Savings Strategy** | Information-theoretic Knapsack (retains 100% visibility) | Standard reduction techniques |
| **Primary Use Case** | Context compression for AI agents | Basic token reduction |

---

## OpenClaw Integration

[OpenClaw](https://github.com/openclaw/openclaw) users get the deepest integration — Entroly plugs in as a Context Engine:

| Agent Type | What Entroly Does | Token Savings |
|------------|---|---|
| **Main agent** | Full codebase at variable resolution | ~95% |
| **Heartbeat** | Only loads changes since last check | ~90% |
| **Subagents** | Inherited context + Nash bargaining budget split | ~92% |
| **Cron jobs** | Minimal context — relevant memories + schedule | ~93% |
| **Group chat** | Entropy-filtered messages — only high-signal kept | ~90% |

```python
from entroly.context_bridge import MultiAgentContext

ctx = MultiAgentContext(workspace_path="~/.openclaw/workspace")
ctx.ingest_workspace()
sub = ctx.spawn_subagent("main", "researcher", "find auth bugs")
```

---

## Accuracy Benchmarks

> *Does compression hurt accuracy? In these release checks, compressed context stayed statistically close to baseline.*

Entroly selects context at variable resolution. We measure **accuracy retention** across industry-standard benchmarks:

| Benchmark | What it tests | Baseline | Entroly | Retention |
|---|---|---|---|---|
| **NeedleInAHaystack** | Info retrieval from long context | 100% | 100% | **100%** |
| **HumanEval** | Code generation | 13.3% | 13.3% | **100%** |
| **GSM8K** | Math reasoning | 86.7% | 80.0% | **92%** |
| **SQuAD 2.0** | Reading comprehension | 93.3% | 86.7% | **92%** |

> *Results from release checks via `bench/accuracy.py`. Performance depends on model, dataset, prompt shape, and token budget.*

### Evaluation Status

| Benchmark | Status inside `bench/accuracy.py` | Validated Results (`gpt-4o-mini`) |
|---|---|---|
| **NeedleInAHaystack** | Implemented | 100% retention |
| **HumanEval** | Implemented | 100% retention |
| **GSM8K** | Implemented | 92% retention |
| **SQuAD 2.0** | Implemented | 92% retention |

### Reproduce These Results

```bash
pip install entroly[full] matplotlib

# Export your API key
export OPENAI_API_KEY="sk-..."

# Run the full validation suite
python -m bench.accuracy --benchmark all --model gpt-4o-mini --samples 15

# Generate the NeedleInAHaystack Heatmap
python -m bench.needle_heatmap --model gpt-4o-mini
```

---

## How It Works

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="Entroly Pipeline — context engineering for AI coding" width="880">
</p>

| Stage | What | Result |
|---|---|---|
| **1. Ingest** | Index codebase, build dependency graph, fingerprint fragments | Complete map in <2s |
| **2. Score** | Rank by information density — high-value code up, boilerplate down | Every fragment scored |
| **3. Select** | Mathematically optimal subset fitting your token budget | Proven optimal (knapsack) |
| **4. Deliver** | 3 resolution levels: full → signatures → references | 100% coverage |
| **5. Learn** | Track which context produced good AI responses | Gets smarter over time |

---

## The 3 Pillars of Zero-Token Autonomy

> **Most agent frameworks that learn do so by calling LLMs. Entroly's structural path tries to learn from code-graph analysis first.**

Many self-improving agent frameworks spend API tokens to synthesize skills, reflect on failures, and update policies. The bill grows with experience.

Entroly's self-evolution loop is designed around three principles intended to keep the runtime **budget-negative** — learning cost should stay below savings.

### Pillar 1 — Token Economy (Self-Funded Evolution)

A `ValueTracker` measures cumulative token savings `S(t)` across every optimized request. The evolution budget is a strict fraction of savings:

```
C_spent(t) ≤ τ · S(t)        (τ = 5%)
```

Any LLM-based synthesis is **gated by this invariant**. The intent is that the system spends less on learning than it saves you.

### Pillar 2 — Local Structural Induction ($0, Deterministic)

Before the budget is ever touched, the `StructuralSynthesizer` tries first. It reads the entropy gradient of your code graph — AST patterns, dependency edges, type signatures — and can emit candidate Python tools from structural analysis. No LLM. No embeddings API. No cloud call. **Zero tokens.**

The `auth` skill in the trace above was synthesized this way. Fitness 1.0, cost $0.0000.

### Pillar 3 — Dreaming Loop (Idle-Time Self-Play)

When no user activity is detected for >60 s, the `DreamingLoop` generates synthetic queries from `FeedbackJournal` history, perturbs the PRISM scoring weights, and runs counterfactual experiments against itself. Improvements are kept when they beat the local acceptance gate; regressions are discarded. This is designed to improve local ranking while idle — with no API calls.

### The Closed Loop

```
User query → miss → EvolutionLogger registers gap
                             ↓
            [Pillar 2] StructuralSynthesizer ($0)
                    ↓ (if fails)
            [Pillar 1] LLM fallback — only if C_spent ≤ τ·S(t)
                    ↓
            Benchmark → Promote (fitness ≥ threshold) or Prune
                    ↓
            Skill registry live in .entroly/vault/evolution/
                    ↓
        [Pillar 3] Idle? Dream: perturb weights, self-play, keep wins
                    ↓
            Next session starts strictly smarter
```

No manual tuning. No config files. No tokens spent on learning. The daemon ships with the runtime and starts the moment you run `entroly go`.

---

### Pillar 4 — Federated Learning (Experimental, Opt-In)

> **Optional: share anonymous optimization weights across installations.**

When federation is enabled, Entroly installations can exchange anonymized optimization weights. The intent is that participants benefit from each other's local learning — without sharing code.

```
Your daemon learns locally → shares anonymous weights → absorbs others' improvements
```

**Design principles:**

| | Without federation | **With federation** |
|---|---|---|
| Who improves your AI? | Your local data only | **Your data + anonymous weights from other installations** |
| Network effect | None | **More participants = broader weight diversity** |
| Infrastructure cost | None | **$0 — uses GitHub for transport** |
| Privacy | Local only | **Differential privacy + anonymous IDs; code never shared** |

Federation is experimental. Shared payloads are optimization statistics/weights, not code.

**Privacy safeguards:**
- Your code should not leave your machine. Only anonymized optimization weights are shared
- Each contribution is noise-protected
- Your identity is a random ID stored locally
- Poisoning attacks are filtered by trimmed-mean aggregation

```bash
# Opt-in (default: off — your choice, always)
export ENTROLY_FEDERATION=1
```

Python and Node.js use the same protocol shape. Feature parity can vary by package version; privacy controls remain opt-in.

---

### Pillar 5 — Response Distillation

> **LLM responses often include filler — greetings, hedging, meta-commentary. Entroly can strip common filler while leaving code blocks untouched.**

LLM responses often include tokens that don't carry information: "Sure, I'd be happy to help!", "Let me think about that...", "Hope this helps!". Response Distillation strips the prose filler. Code blocks are never touched.

```
Before: "Sure! I'd be happy to help you with that. Let me take a careful look
          at your code. The issue is in the auth module — specifically the
          token validation logic. Hope this helps! Let me know if you need
          anything else."

After:  "The issue is in the auth module — specifically the token validation logic."

         → 75% fewer output tokens. Same information. Zero filler.
```

**Three levels — you choose:**

| Mode | What goes | What stays | Typical savings |
|---|---|---|---|
| `lite` | Greetings, sign-offs | Everything else | 15–25% |
| `full` | + hedging, meta-commentary, transitions | Code + technical content | 30–50% |
| `ultra` | + articles, function words | Pure signal | 50–70% |

**Safety design:** Code blocks, JSON, YAML, XML are protected from prose distillation. The distiller is designed to touch prose only.

```bash
export ENTROLY_DISTILL=1           # Turn it on
export ENTROLY_DISTILL_MODE=full   # lite | full | ultra
```

Works in real-time on streaming responses. <1ms overhead per chunk.


### Make The Autonomy Visible

The daemon is useful silently — but silent autonomy doesn't build trust. Two first-class integrations let you see and share every evolution event:

**Chat gateways** — live-stream gap detections, structural syntheses, promotions, and dream-cycle wins to **Telegram**, **Discord**, or **Slack**. Zero extra dependencies — stdlib only.

```bash
# Telegram (interactive: /status /skills /gaps /dream)
export ENTROLY_TG_TOKEN=...        # from @BotFather
export ENTROLY_TG_CHAT_ID=...
python -m entroly.integrations.telegram_gateway

# Discord (incoming webhook)
export ENTROLY_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
python -m entroly.integrations.discord_gateway

# Slack (incoming webhook)
export ENTROLY_SLACK_WEBHOOK=https://hooks.slack.com/services/...
python -m entroly.integrations.slack_gateway
```

**agentskills.io export** — promoted skills aren't vault-locked. Export to the portable agentskills.io v0.1 spec so any compatible runtime can consume them:

```bash
python -m entroly.integrations.agentskills ./dist/agentskills
# → dist/agentskills/<skill_id>/{skill.json,procedure.md,tool.py,tests.json}
```

Every exported `skill.json` carries `origin.synthesis: "structural"` and `origin.token_cost: 0.0` — the zero-token provenance is portable too.

---

### Why This Matters

| | Typical self-improving agent | **Entroly** |
|---|---|---|
| **Skill synthesis** | LLM generates code (pays tokens) | **Structural induction first — $0** |
| **Learning budget** | Unbounded (you pay the bill) | **Gated: C_spent ≤ 5% of savings** |
| **Gap detection** | Implicit (re-encounters failure) | **Explicit: `EvolutionLogger` miss counter** |
| **Idle time** | Process sleeps | **DreamingLoop runs self-play** |
| **Persistence** | Session memory + FTS | **Epistemic vault + belief graph + registry** |
| **Net cost of learning** | Positive (always) | **Designed to be ≤ 0** |

### What Makes It Self-Improving?

| Capability | What It Does | Cost |
|---|---|---|
| **PRISM Reinforcement Learning** | Learns which context produces good AI responses. Updates 4D scoring weights (recency, frequency, semantic, entropy) via policy gradients with counterfactual credit assignment. | Zero — runs on CPU |
| **Dreaming Loop** | During idle time (>60s inactivity), generates synthetic queries and runs self-play experiments to find better weight configurations. Monotonic improvement guarantee. | Zero — no API calls |
| **Task-Conditioned Profiles** | Automatically detects task type (debugging, feature, refactor, performance, testing, docs) and loads task-specific learned weights. Debugging prioritizes recency; documentation prioritizes semantic similarity. | Zero |
| **Skill Synthesis** | Identifies gaps in coverage, synthesizes new tools from AST analysis, benchmarks them, promotes winners, prunes losers. Full lifecycle — no human intervention. | Zero — structural analysis only |
| **Adaptive Exploration (RAVEN-UCB)** | Thompson sampling + Upper Confidence Bound automatically balances exploring new strategies vs exploiting known-good ones. Exploration rate anneals as confidence grows. | Zero |

### How The Learning Loop Works

```
User Query → Optimize Context → AI Response → Feedback Signal
                                                    ↓
                                        PRISM RL Weight Update
                                        Task Profile Update
                                        Feedback Journal Entry
                                                    ↓
                                        [Idle > 60s detected]
                                                    ↓
                                        Dreaming Loop activates:
                                        → Synthetic query generation
                                        → Self-play weight experiments
                                        → Skill gap detection
                                        → Structural tool synthesis
                                                    ↓
                                        Better weights saved to disk
                                        → Next session starts smarter
```

### Local Self-Improvement

The default self-improvement loop runs **locally on your CPU**. No embeddings API or fine-tuning job is required. The dreaming loop, RL updates, and structural skill synthesis operate on local signals; optional federation or LLM fallback must be enabled separately.

**Day 1:** Entroly selects context with default weights.
**Day 30:** PRISM weights have shifted based on local feedback signals. Savings and ranking quality may improve as the engine learns your codebase patterns.

```bash
entroly dashboard    # Watch the PRISM weights evolve in real-time
entroly autotune     # Manually trigger optimization (usually not needed)
```

---

## Trust & Transparency

> *"If you compress my codebase by 80%, how do I know you didn't strip the code my AI actually needs?"*

Fair question. Here's the honest answer:

### The 3-Resolution System

Entroly never "strips" code from files the LLM needs. It uses **three resolution levels**:

| Resolution | What the LLM sees | When used |
|---|---|---|
| **Full (100%)** | Complete source code — every line, every comment | Files that directly match your query |
| **Signatures** | Function/class signatures with types + docstrings | Tangential imports your query doesn't target |
| **Reference** | File path + 1-line summary | Files the LLM should know exist, but doesn't need to read |

**Selection policy:** If a file directly matches the query, Entroly tries to include it at full resolution before compressing lower-priority files to signatures or references. Use `/explain` to inspect the actual selection for a request.

### Inline Context Report

By default, optimized requests include a visible report inside the LLM context:

```
[Entroly: worker.ts (Full), schema.prisma (Full), types.ts (Full),
 8 files (Signatures only), 12 files (Reference only). 8,777 tokens. GET /explain for details.]
```

Your AI sees this. You can see this. No hidden truncation.

### The `/explain` Endpoint

After any request, call `GET localhost:9377/explain` to see:
- **Included** — Every included file with its resolution level and why it was included
- **Excluded** — Every excluded file and why it was dropped
- **Summary** — Resolution exact breakdown (e.g., 5 Full, 8 Skeleton, 12 Reference)

### Honest Savings Claims

| Claim | What it actually means |
|---|---|
| **70–95% token savings** | Observed in release checks on large-repo workloads. Varies by query specificity, repo size, and token budget. |
| **Variable-resolution visibility** | Every supported file in your codebase is represented at some resolution. |
| **< 10ms latency** | Some Rust core paths are sub-10ms. End-to-end optimization depends on repo size, engine mode, filesystem, and cache warmth. Network to the LLM API is unchanged. |

The range reflects real variability: a narrow bug-fix query against a 1000-file repo may hit 95%. A broad "explain the architecture" query against a 50-file repo lands closer to 70%. We publish the range, not the peak.

### Disable the Report

If the ~40 token overhead bothers you:
```bash
export ENTROLY_CONTEXT_REPORT=0
```

---

## Context Engineering, Automated

> *"The LLM is the CPU, the context window is RAM."*

| Layer | What it solves |
|---|---|
| **Documentation tools** | Give your agent up-to-date API docs |
| **Memory systems** | Remember things across conversations |
| **RAG / retrieval** | Find relevant code chunks |
| **Entroly (optimization)** | **Makes selected context fit** — compresses codebase + docs + memory under the configured token budget |

These layers are **complementary.** Entroly is the optimization layer that helps fit high-value context under a budget.

---

## Not Just For Code: Universal Text Compression

While Entroly was built for codebases, its core relies on **Shannon Entropy and Knapsack Mathematics**, meaning it is completely agnostic to the text it compresses. Entroly is widely used as a universal context compressor for:

| Text Type | The Problem | How Entroly Compresses It |
|---|---|---|
| **Massive Server Logs** | 100K lines of identical `INFO` logs bury the one `ERROR` stack trace. | Drops repetitive logs (low entropy), strictly retains exceptions and novel timestamps. |
| **Agent Memory** | Multi-agent swarms fill up the context window with conversational fluff. | Extracts only the high-signal, decision-making paragraphs to pass to the next agent. |
| **Legal/Financial Docs** | RAG systems retrieve 50 pages of PDFs, blowing the token budget. | Scans the retrieved paragraphs, isolates the exact clauses answering the query, drops the boilerplate. |

*In our `NeedleInAHaystack` benchmark, Entroly perfectly compressed 128,000 tokens of **Paul Graham essays** (pure English text) to 2,000 tokens while maintaining a 100% retrieval success rate.*

---

## CLI Commands

| Command | What it does |
|---------|---|
| `entroly go` | **One command** — auto-detect, init, proxy, dashboard |
| `entroly wrap claude` | Start proxy + launch Claude Code in one command |
| `entroly wrap codex` | Start proxy + launch Codex CLI when its provider settings permit a custom endpoint |
| `entroly wrap aider` | Start proxy + launch Aider |
| `entroly wrap cursor` | Start proxy + print Cursor config |
| `entroly demo` | Before/after comparison with dollar savings on YOUR project |
| `entroly dashboard` | Live metrics: savings trends, health grade, PRISM weights |
| `entroly doctor` | 7 diagnostic checks — finds problems before you do |
| `entroly health` | Codebase health grade (A-F): clones, dead code, god files |
| `entroly benchmark` | Competitive benchmark: Entroly vs raw context vs top-K |
| `entroly role` | Weight presets: `frontend`, `backend`, `sre`, `data`, `fullstack` |
| `entroly autotune` | Auto-optimize engine parameters |
| `entroly learn` | Analyze session for failure patterns, write to CLAUDE.md |
| `entroly digest` | Weekly summary: tokens saved, cost reduction |
| `entroly status` | Check running services |

---

## Coding Agents — One Command

```bash
entroly wrap claude              # Starts proxy + launches Claude Code
entroly wrap codex               # Starts proxy + launches Codex CLI when custom endpoints are supported
entroly wrap aider               # Starts proxy + launches Aider
entroly wrap cursor              # Starts proxy + prints Cursor config
```

Entroly starts the proxy, sets the documented base URL environment variable where the tool supports one, and launches your tool. If a vendor CLI requires provider configuration instead, use that tool's documented settings and review its terms before proxying.

---

## Python SDK — One Function

```python
from entroly import compress

result = compress(messages, budget=50_000)
response = client.messages.create(model="claude-sonnet-4-5-20250929", messages=result)
```

Or compress any content directly:

```python
from entroly.universal_compress import universal_compress

compressed = universal_compress(huge_json_blob)    # auto-detects JSON
compressed = universal_compress(log_output)        # auto-detects logs
compressed = universal_compress(csv_data)          # auto-detects CSV
```

Content-type auto-detection routes each input to the best compressor — JSON, logs, code, CSV, XML, stacktraces, tables.

---

## Drop Into Your Existing Stack

| Your setup | Add Entroly | One-liner |
|---|---|---|
| Any Python app | `compress()` | `result = compress(messages, budget=50_000)` |
| Any app (proxy) | `entroly proxy` | Point base URL at `localhost:9377` |
| LangChain | `EntrolyCompressor` | `chain = compressor \| llm` |
| Multi-agent | `MultiAgentContext` | `ctx = MultiAgentContext(...)` |
| Claude Code | `entroly wrap claude` | One command |
| Codex / Aider | `entroly wrap codex` / `entroly wrap aider` | Custom endpoint where supported |
| MCP tools | `entroly init` | Auto-config |

### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from entroly.integrations.langchain import EntrolyCompressor

llm = ChatOpenAI(model="gpt-4o")
compressor = EntrolyCompressor(budget=30000)
chain = compressor | llm
result = chain.invoke("Explain the auth module")
```

### Multi-Agent Context (SharedContext)

```python
from entroly.context_bridge import MultiAgentContext

ctx = MultiAgentContext(workspace_path="~/.agent/workspace", token_budget=128_000)
ctx.ingest_workspace()

# NKBE allocates budget optimally across agents
budgets = ctx.allocate_budgets(["researcher", "coder", "reviewer"])

# Spawn subagent with inherited context
sub = ctx.spawn_subagent("main", "researcher", "find auth bugs")

# Schedule cron jobs with minimal context
ctx.schedule_cron("monitor", "check error rates", interval_seconds=900)
```

---

## Lossless Compression (CCR)

Entroly never permanently discards data. When a fragment is compressed to a skeleton, the original is stored in the **Compressed Context Store**. The LLM can retrieve the full original on demand:

```bash
# List all retrievable fragments
curl localhost:9377/retrieve

# Get full original of a compressed file
curl localhost:9377/retrieve?source=file:src/auth.py
```

This is the architectural answer to "silent truncation": nothing is permanently lost. If the LLM needs the full body of a skeletonized function, it asks for it.

---

## Cache Optimization

Entroly stabilizes context prefixes across turns to improve provider KV-cache reuse where the configured provider supports prompt caching. Cache discounts and behavior are provider-specific and can change.

---

## Failure Learning

```bash
entroly learn                    # Analyze session for failure patterns
entroly learn --apply            # Write learnings to CLAUDE.md / AGENTS.md
```

Reads the proxy's passive feedback data, identifies patterns where the LLM was confused or gave low-quality responses, and writes actionable corrections to your agent config files.

---

## Quality Presets

```bash
entroly proxy --quality speed       # minimal optimization, lowest latency
entroly proxy --quality balanced    # recommended (default)
entroly proxy --quality max         # full pipeline, best results
entroly proxy --quality 0.7         # any float 0.0-1.0
```

---

## Platform Support

| | Linux | macOS | Windows |
|--|---|---|---|
| **Python 3.10+** | Yes | Yes | Yes |
| **Rust wheel** | Yes | Yes (Intel + Apple Silicon) | Yes |
| **Docker** | Optional | Optional | Optional |
| **Admin/WSL required** | No | No | No |

---

## Operational Features

- **Persistent savings tracking** — lifetime savings in `~/.entroly/value_tracker.json`, trend charts in dashboard
- **IDE status bar** — `/confidence` endpoint for real-time VS Code widgets
- **Rich headers** — `X-Entroly-Confidence`, `X-Entroly-Coverage-Pct`, `X-Entroly-Cost-Saved-Today`
- **Crash recovery** — gzipped checkpoints restore in <100ms
- **Large file protection** — 500 KB ceiling prevents OOM
- **Binary detection** — 40+ file types auto-skipped
- **Fragment feedback** — `POST /feedback` lets your AI rate context quality
- **Explainable** — `GET /explain` shows why each fragment was included/excluded, with resolution labels and drop reasons

---

## Need Help?

```bash
entroly doctor    # runs 7 diagnostic checks
entroly --help    # all commands
```

**Email:** autobotbugfix@gmail.com — we aim to respond within 24 hours.

<details>
<summary><b>Common Issues</b></summary>

**macOS "externally-managed-environment":**
```bash
python3 -m venv ~/.venvs/entroly && source ~/.venvs/entroly/bin/activate && pip install entroly[full]
```

**Windows pip not found:**
```powershell
python -m pip install entroly
```

**Port 9377 in use:**
```bash
entroly proxy --port 9378
```

**Rust engine not loading:** Entroly auto-falls back to Python. For Rust speed: `pip install entroly[native]`

</details>

---

## Environment Variables

| Variable | Default | What it does |
|---|---|---|
| `ENTROLY_QUALITY` | `0.5` | Quality dial (0.0-1.0 or preset) |
| `ENTROLY_PROXY_PORT` | `9377` | Proxy port |
| `ENTROLY_MAX_FILES` | `5000` | Max files to index |
| `ENTROLY_RATE_LIMIT` | `0` | Requests/min (0 = unlimited) |
| `ENTROLY_MCP_TRANSPORT` | `stdio` | MCP transport (stdio/sse) |
| `ENTROLY_CONTEXT_REPORT` | `1` | Inline context report in LLM prompts (0 to disable) |
| `ENTROLY_CACHE_ALIGN` | `1` | Provider KV cache prefix stabilization (0 to disable) |
| `ENTROLY_FEDERATION` | `0` | Enable federated swarm learning (1 to enable) |
| `ENTROLY_FEDERATION_BOT` | *(none)* | Shared GitHub bot token for anonymous federation writes |
| `ENTROLY_DISTILL` | `0` | Enable response distillation / output compression (1 to enable) |
| `ENTROLY_DISTILL_MODE` | `full` | Distillation intensity: `lite`, `full`, or `ultra` |

---

<details>
<summary><b>Technical Deep Dive — Architecture & Algorithms</b></summary>

### Architecture

Hybrid Rust + Python. Math-heavy core paths use Rust via PyO3 where available; MCP and orchestration stay in Python.

```
+-----------------------------------------------------------+
|  IDE (Cursor / Claude Code / Cline / VS Code)             |
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
|  |  |  21 modules · 380 KB · 249 tests    |    |          |
|  |  +-------------------------------------+    |          |
|  +---------------------------------------------+          |
+-----------------------------------------------------------+
```

### Rust Core (21 modules)

| Module | What | How |
|---|---|---|
| **hierarchical.rs** | 3-level codebase compression | Skeleton map + dep-graph + knapsack fragments |
| **knapsack.rs** | Context selection | KKT dual bisection O(30N) or exact DP |
| **knapsack_sds.rs** | Information-Optimal Selection | Submodular diversity + multi-resolution |
| **prism.rs** | Weight optimizer | Spectral natural gradient on 4x4 covariance |
| **entropy.rs** | Information density | Shannon entropy + boilerplate detection |
| **depgraph.rs** | Dependency graph | Auto-link imports, type refs, function calls |
| **skeleton.rs** | Code skeletons | Preserves signatures, strips bodies (60-80% reduction) |
| **dedup.rs** | Duplicate detection | 64-bit SimHash, Hamming threshold 3 |
| **lsh.rs** | Semantic recall | 12-table multi-probe LSH, ~3μs over 100K fragments |
| **sast.rs** | Security scanning | 55 rules, 8 CWE categories, taint analysis |
| **health.rs** | Codebase health | Clones, dead symbols, god files, arch violations |
| **guardrails.rs** | Safety-critical pinning | Criticality levels + task-aware budget multipliers |
| **query.rs** | Query analysis | Vagueness scoring, keyword extraction, intent |
| **query_persona.rs** | Query archetypes | RBF kernel + Pitman-Yor + per-archetype weights |
| **anomaly.rs** | Entropy anomaly detection | MAD-based robust Z-scores |
| **semantic_dedup.rs** | Semantic dedup | Greedy marginal information gain, (1-1/e) optimal |
| **utilization.rs** | Response utilization | Trigram + identifier overlap feedback |
| **nkbe.rs** | Multi-agent budgets | Arrow-Debreu KKT + Nash bargaining + REINFORCE |
| **cognitive_bus.rs** | Agent event routing | Poisson rate models, Welford spike detection |
| **fragment.rs** | Core data structure | Content, metadata, scoring, SimHash fingerprint |
| **lib.rs** | PyO3 bridge | All modules exposed to Python |




</details>

---

---

## License

Apache-2.0

---

<p align="center">
  <b>Measure and reduce wasted context tokens with local, evidence-aware tooling.</b><br/>
  <code>pip install entroly[full] && entroly go</code>
</p>
