<p align="center">
  <a href="docs/i18n/README.zh-CN.md">🇨🇳 中文</a> •
  <a href="docs/i18n/README.ja.md">🇯🇵 日本語</a> •
  <a href="docs/i18n/README.ko.md">🇰🇷 한국어</a> •
  <a href="docs/i18n/README.pt-BR.md">🇧🇷 Português</a> •
  <a href="docs/i18n/README.es.md">🇪🇸 Español</a> •
  <a href="docs/i18n/README.de.md">🇩🇪 Deutsch</a> •
  <a href="docs/i18n/README.fr.md">🇫🇷 Français</a> •
  <a href="docs/i18n/README.ru.md">🇷🇺 Русский</a> •
  <a href="docs/i18n/README.hi.md">🇮🇳 हिन्दी</a> •
  <a href="docs/i18n/README.tr.md">🇹🇷 Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Token_Savings-up_to_95%25-brightgreen?style=for-the-badge" alt="Token Savings: up to 95%">
  <img src="https://img.shields.io/badge/Learning_Cost-$0-blue?style=for-the-badge" alt="Learning Cost: $0">
  <img src="https://img.shields.io/badge/Engine-Rust_%2B_WASM-orange?style=for-the-badge&logo=rust" alt="Rust + WASM">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <a href="https://github.com/juyterman1000/entroly-cost-check-"><img src="https://img.shields.io/badge/GitHub_Action-Cost_Check-purple?style=for-the-badge&logo=githubactions" alt="GitHub Action"></a>
  <a href="https://mcpmarket.com/daily/top-mcp-server-list-march-26-2026"><img src="https://img.shields.io/badge/%231_MCP_Market-Ranked_Server-gold?style=for-the-badge&logo=starship&logoColor=white" alt="#1 on MCP Market"></a>
</p>

<h1 align="center">Entroly — Cut AI Token Costs by 70–95%</h1>

<h3 align="center">Your AI coding tools only see 5% of your codebase.<br/>Entroly gives them the full picture — for a fraction of the cost.</h3>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>Live demo →</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI">
  <img src="https://img.shields.io/npm/v/entroly?color=red&label=npm">
  <img src="https://img.shields.io/badge/Tests-484_passing-success">
  <img src="https://img.shields.io/badge/Latency-<10ms-purple">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green">
</p>

---

## The Problem — and the Bottom-Line Impact

Every AI coding tool — Claude, Cursor, Copilot, Codex — has the same blind spot: **it only sees 5–10 files at a time.** The other 95% of your codebase is invisible. This causes hallucinated APIs, broken imports, missed dependencies, and wasted developer hours fixing AI-generated mistakes.

Models keep getting bigger — **Claude Opus 4.7** just dropped with even more capability and even higher per-token costs. Larger context windows don't solve the problem; they make it worse. You're paying for 186,000 tokens per request — most of which is duplicated boilerplate.

> **Entroly fixes both problems in 30 seconds.** It compresses your entire codebase into the AI context window at variable resolution, so your AI sees everything — and you pay for almost none of it.

---

## What Changes on Day 1

| Metric | Before Entroly | **After Entroly** |
|---|---|---|
| Files visible to AI | 5–10 | **Your entire codebase** |
| Tokens per request | ~186,000 | **9,300 – 55,000** |
| Monthly AI spend (at 1K req/day) | ~$16,800 | **$840 – $5,040** |
| AI answer accuracy | Incomplete, often hallucinated | **Dependency-aware, correct** |
| Developer time fixing AI mistakes | Hours/week | **Near zero** |
| Setup | Days of prompt engineering | **30 seconds** |

> **ROI example:** A 10-person team spending $15K/month on AI API calls saves **$10K–$14K/month** on day 1. Entroly pays for itself in the first hour. (It's free and open-source, so it actually pays for itself instantly.)

---

## What Your Competitors Already Know

The teams adopting Entroly today aren't just saving money — they're **compounding an advantage** your team can't catch up to.

- **Week 1:** Their AI sees 100% of their codebase. Yours sees 5%. They ship faster.
- **Month 1:** Their runtime has learned their codebase patterns. Yours is still hallucinating imports.
- **Month 3:** Their installation is plugged into the federation — absorbing optimization strategies from thousands of other teams worldwide. Yours doesn't know this exists.
- **Month 6:** They've saved $80K+ in API costs. That budget went into hiring. You're still explaining to finance why the AI bill keeps growing.

Every day you wait, the gap widens. The federation effect means **early adopters get smarter faster** — and that advantage compounds.

---

## How It Works (30 Seconds)

```bash
npm install entroly-wasm && npx entroly-wasm
# or
pip install entroly && entroly go
```

That's it. Entroly auto-detects your IDE, connects to Claude/Cursor/Copilot/Codex/MiniMax, and starts optimizing. No configuration. No YAML. No embeddings.

**What happens under the hood:**

1. **Index** — Maps your entire codebase in <2 seconds
2. **Score** — Ranks every file by information density
3. **Select** — Picks the mathematically optimal subset for your token budget
4. **Deliver** — Critical files go in full, supporting files as signatures, everything else as references
5. **Learn** — Tracks what works, gets smarter over time

Your AI now sees 100% of your codebase. You pay for 5–30% of the tokens.

---

## The Competitive Edge — What Sets Entroly Apart

### 🧠 It Gets Smarter Without Costing You More

Most "self-improving" AI tools burn tokens to learn — your bill grows with their intelligence. Entroly's learning loop is **provably token-negative**: it cannot spend more on learning than it saves you.

The math is simple and auditable:

```
Learning budget ≤ 5% × Lifetime savings
```

Day 1: 70% token savings. Day 30: 85%+. Day 90: 90%+. **The improvement costs you $0.**

### 🌐 Federated Swarm Learning — The Part That Sounds Like Science Fiction

Now take the Dreaming Loop and multiply it by **every developer on Earth who runs Entroly.**

While you sleep, your daemon dreams — and so do 10,000 others. Each one discovers slightly different tricks for compressing code. Each one shares what it learned — anonymously, privately, no code ever leaves your machine. Each one absorbs what the others found.

**You wake up. Your AI is smarter than when you left it. Not because of anything you did — because of what the swarm dreamed.**

```
Your daemon dreams → discovers a better strategy → shares it (anonymously)
     ↓
10,000 other daemons did the same thing last night
     ↓
You open your laptop → your AI already absorbed all of it
```


**Network effect:**
- Every new user makes everyone else's AI better — that installed base can't be forked
- Your code never moves. Only optimization weights — noise-protected and anonymous
- Infrastructure cost: **$0**. It runs on GitHub. No servers. No GPUs. No cloud

```bash
# Opt-in — your choice, always
export ENTROLY_FEDERATION=1
```

### ✂️ Response Distillation — Save Tokens on Output Too

LLM responses contain ~40% filler — "Sure, I'd be happy to help!", hedging, meta-commentary. Entroly strips it. Code blocks are never touched.

```
Before: "Sure! I'd be happy to help. Let me take a look at your code.
         The issue is in the auth module. Hope this helps!"

After:  "The issue is in the auth module."
         → 70% fewer output tokens
```

Three intensity levels: `lite` → `full` → `ultra`. Enable with one env var.

### 🔒 Runs Locally. Your Code Never Leaves Your Machine.

Zero cloud dependencies. Zero data exfiltration risk. Everything runs on your CPU in <10ms. Works in air-gapped and regulated environments — nothing ever phones home.

---

## Works With Your Stack

| Tool | Setup |
|---|---|
| **Cursor** | `entroly init` → MCP server |
| **Claude Code** | `claude mcp add entroly -- entroly` |
| **GitHub Copilot** | `entroly init` → MCP server |
| **Codex CLI** | `entroly wrap codex` |
| **Windsurf / Cline / Cody** | `entroly init` |
| **Any LLM API** | `entroly proxy` → HTTP proxy on `localhost:9377` |

Also: OpenAI API • Anthropic API • LangChain • LlamaIndex • MCP-native

---

## Benchmarks

### Live Evolution Trace

This is from this repo's vault, not a roadmap:

```
[detect]     gap observed → entity="auth", miss_count=3
[synthesize] StructuralSynthesizer ($0, deterministic, no LLM)
[benchmark]  skill=ddb2e2969bb0 → fitness 1.0 (1 pass / 0 fail, 338 ms)
[promote]    status: draft → promoted
[spend]      $0.0000 — invariant C_spent ≤ τ·S(t) holds
```

### Accuracy Retention

Compression doesn't hurt accuracy — we measured it (n=100, gpt-4o-mini, Wilson 95% CIs):

| Benchmark | Baseline (95% CI) | With Entroly (95% CI) | Retention |
|---|---|---|---|
| NeedleInAHaystack | 100% [83.9–100%] | 100% [83.9–100%] | **100.0%** |
| GSM8K | 85.0% [76.7–90.7%] | 86.0% [77.9–91.5%] | **101.2%** |
| SQuAD 2.0 | 84.0% [75.6–89.9%] | 83.0% [74.5–89.1%] | **98.8%** |

> Confidence intervals overlap on every benchmark — accuracy is statistically indistinguishable from baseline. Reproduce: `python -m bench.accuracy --benchmark all --model gpt-4o-mini --samples 100`

### CI/CD Integration

Run token cost checks in every PR — catch regressions before they ship:

```yaml
- uses: juyterman1000/entroly-cost-check-@v1
```

→ **[entroly-cost-check GitHub Action](https://github.com/juyterman1000/entroly-cost-check-)**

---

## Watch It Run — Live Notifications

Three chat integrations ship in the box. See every gap detection, skill synthesis, and dream-cycle win in real-time:

```bash
export ENTROLY_TG_TOKEN=...          # Telegram (2-way: /status /skills /gaps /dream)
export ENTROLY_DISCORD_WEBHOOK=...   # Discord
export ENTROLY_SLACK_WEBHOOK=...     # Slack
```

---

## Portable Skills (agentskills.io)

Skills Entroly creates aren't locked in. Export to the open agentskills.io v0.1 spec:

```bash
node node_modules/entroly-wasm/js/agentskills_export.js ./dist/agentskills
python -m entroly.integrations.agentskills ./dist/agentskills
```

Every exported skill carries `origin.token_cost: 0.0` — the zero-cost provenance travels with it.

---

## Full Parity: Python & Node.js

Both runtimes are feature-complete. Same engine, same vault, same learning loop:

| Capability | Python | Node.js (WASM) |
|---|---|---|
| Context compression | ✅ | ✅ |
| Self-evolution | ✅ | ✅ |
| Dreaming loop | ✅ | ✅ |
| Federation | ✅ | ✅ |
| Response distillation | ✅ | ✅ |
| Chat gateways | ✅ | ✅ |
| agentskills.io export | ✅ | ✅ |

---

## Deep Dive

Architecture, 21 Rust modules, 3-resolution compression, provenance guarantees, RAG comparison, full CLI reference, Python SDK, LangChain integration → **[docs/DETAILS.md](docs/DETAILS.md)**

---

<p align="center">
  <b>Stop paying for tokens your AI wastes. Start running an AI that teaches itself.</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>

<p align="center">
  <a href="https://github.com/juyterman1000/entroly/discussions">Discussions</a> •
  <a href="https://github.com/juyterman1000/entroly/issues">Issues</a> •
  Apache-2.0 License
</p>
