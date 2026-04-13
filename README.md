<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<h1 align="center">Entroly Daemon</h1>

<h3 align="center">Your AI is blind. Fix it in 30 seconds — then watch it teach itself.</h3>

<p align="center">
  <i>Claude, Cursor, Copilot, Codex, and MiniMax only see 5% of your codebase. Entroly gives them a <b>2M-token brain for 90% less</b> — then its daemon <b>dreams up new skills while you sleep, at $0 token cost, compresses tokens, and improves answer quality</b>. The first AI runtime whose learning is provably token-negative.</i>
</p>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>Live demo →</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI">
  <img src="https://img.shields.io/npm/v/entroly?color=red&label=npm">
  <img src="https://img.shields.io/badge/Tests-484_passing-success">
  <img src="https://img.shields.io/badge/Latency-<10ms-purple">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## What you actually get

| | Without Entroly | **With Entroly** |
|---|---|---|
| Files the AI sees | 5–10 | **Your entire repo** |
| Tokens per request | ~186,000 | **9,300 – 55,000** |
| Cost per 1K requests | ~$560 | **$28 – $168** |
| Effective context window | 200K | **~2M (via variable-resolution compression)** |
| Learning cost over time | Grows (tokens) | **$0 — provably token-negative** |
| Setup | Hours of prompt hacks | **30 seconds** |

Critical files go in full. Supporting files as signatures. The rest as references. Your AI gets the whole picture. You pay for almost none of it.

---

## Live proof: the self-evolution loop just ran

Not a roadmap. This trace is from this repo's vault, right now:

```
[detect]     gap observed → entity="auth", miss_count=3
[synthesize] StructuralSynthesizer ($0, deterministic, no LLM)
[benchmark]  skill=ddb2e2969bb0 → fitness 1.0 (1 pass / 0 fail, 338 ms)
[promote]    status: draft → promoted
[registry]   .entroly/vault/evolution/registry.md updated
[spend]      $0.0000 — invariant C_spent ≤ τ·S(t) holds
```

Every other self-improving agent burns tokens to learn. Entroly's evolution ledger stays at **$0** because the synthesizer reads your code graph, not an LLM.

---

## Why it costs $0 to get smarter — the 3 Pillars

**1. Token Economy** — A `ValueTracker` measures lifetime savings `S(t)`. The evolution budget is strictly capped:

```
C_spent(t)  ≤  τ · S(t)       (τ = 5%)
```

The runtime is **mathematically incapable** of costing more to improve than it saves.

**2. Structural Induction ($0)** — Before any token is touched, a deterministic synthesizer reads the AST, dependency edges, and entropy gradient of your code and emits a working tool. No LLM. No embeddings. No cloud.

**3. Dreaming Loop** — When idle for >60 s, the system generates synthetic queries, perturbs its scoring weights, and self-plays against benchmarks. Strict improvements are kept; regressions are discarded. You open your laptop in the morning to a smarter runtime.

---

## Install

```bash
npm install entroly-wasm && npx entroly-wasm
# or
pip install entroly && entroly go
```

That's it. It detects your IDE, wires itself into Claude/Cursor/Copilot/Codex/MiniMax, and starts compressing. **Both runtimes have full parity** — budget invariant, agentskills.io export, the three chat gateways, and a shared on-disk vault so skills promoted by one runtime are visible to the other.

**Node:**
```js
const { VaultObserver, TelegramGateway, ValueTracker, exportAgentSkills } = require('entroly-wasm');

const obs = new VaultObserver('.entroly/vault');
new TelegramGateway({ token, chatId }).attach(obs).start();
```

**Python:**
```python
from entroly.evolution_daemon import EvolutionDaemon
from entroly.integrations.telegram_gateway import TelegramGateway

daemon.start()
TelegramGateway(token, chat_id).attach(daemon).start()
```

---

## Watch the autonomy live

The daemon is useful silently — but seeing it move is what makes it real. Three chat gateways ship in the box — Telegram, Discord, Slack. Zero extra dependencies on either runtime.

```bash
# 1. Set one (or all three) of these
export ENTROLY_TG_TOKEN=...          # from @BotFather
export ENTROLY_TG_CHAT_ID=...
export ENTROLY_DISCORD_WEBHOOK=...   # Discord channel → Integrations → Webhooks
export ENTROLY_SLACK_WEBHOOK=...     # Slack app → Incoming Webhooks
```

**Node** (native `fetch`, no deps):
```bash
node node_modules/entroly-wasm/js/gateways.js
```

**Python** (stdlib `urllib`, no deps):
```bash
python -m entroly.integrations.telegram_gateway
python -m entroly.integrations.discord_gateway
python -m entroly.integrations.slack_gateway
```

Every gap detection, synthesis, promotion, and dream-cycle win streams to your chat. Telegram is 2-way — `/status`, `/skills`, `/gaps`, `/dream`.

---

## Portable skills (agentskills.io)

Promoted skills aren't locked in Entroly. Export to the open agentskills.io v0.1 spec and any compatible runtime can consume them:

```bash
# Node
node node_modules/entroly-wasm/js/agentskills_export.js ./dist/agentskills

# Python
python -m entroly.integrations.agentskills ./dist/agentskills
```

Every exported skill carries `origin.token_cost: 0.0` — the zero-token provenance is portable too.

---

## Works with your stack

Claude Code • Cursor • Copilot • Codex CLI • MiniMax • Windsurf • Cody • OpenAI API • Anthropic API • LangChain • LlamaIndex • MCP-native

---

## Deep dive

Architecture, benchmarks, PRISM RL internals, 3-resolution compression, provenance guarantees, RAG comparison, full API → **[docs/DETAILS.md](docs/DETAILS.md)**

---

<p align="center">
  <b>Stop paying for tokens your AI wastes. Start running an AI that teaches itself.</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>
</p>

<p align="center">
  <a href="https://github.com/juyterman1000/entroly/discussions">Discussions</a> •
  <a href="https://github.com/juyterman1000/entroly/issues">Issues</a> •
  MIT License
</p>
