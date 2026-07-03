# How to Slash Your Claude Code & Cursor API Bills by 90% with a Local Context Proxy

If you use autonomous AI coding agents (like **Claude Code**, **Aider**, or **Roo-Cline**) or advanced IDEs like **Cursor**, you have likely experienced that sinking feeling when checking your API billing console. 

Large repositories easily average **150,000+ tokens per request**. At Claude 3.5 Sonnet pricing, that translates to roughly **$0.45 per single message**. Run a feedback loop of 10-15 steps, and you've spent $7 on a single feature.

Here is the exact technical breakdown of why this happens, and how to configure a local context proxy to cut your API costs by 70% to 95% without sacrificing code quality or switching to weaker models.

---

## The Hidden Culprit: Prefix Cache Busting

Most developers focus on **input compression** (slashing whitespace, removing comments, or dropping files). While helpful, this ignores the single largest discount leverage: **Prompt Caching**.

Anthropic provides a **90% discount** for cached prompt prefixes ($0.30 per million tokens instead of $3.00). OpenAI offers a **50% discount**. 

Prompt caching works by reusing the key-value (KV) cache of the prefix bytes *if* the prefix remains identical byte-for-byte between requests.

### The Re-Ranking Trap
Every time your agentic coding loop runs:
1. It updates the terminal history.
2. It fetches git status or runs lint tools.
3. It re-ranks and re-orders context files to find the most relevant snippets.

Because the re-ranking changes the exact order of the files in the prompt, **the byte-by-byte prefix changes slightly on every single message**. Your cache hits drop to 0%, and you pay full price for the entire 150K codebase context over and over.

---

## The Solution: Entroly Local Gateway

[Entroly](https://github.com/juyterman1000/entroly) is an open-source, fully offline local context engine that acts as a middleware between your coding tool and your LLM provider.

```
┌─────────────────┐       ┌───────────────────────────┐       ┌──────────────┐
│  AI Coding Tool │ ────> │   Entroly Proxy (Local)   │ ────> │ LLM Provider │
│  (Cursor/Claude)│       │                           │       │ (Anthropic)  │
└─────────────────┘       └───────────────────────────┘       └──────────────┘
                                │
                                ├─ Cache Aligner (Prefix stability)
                                ├─ Knapsack Optimizer (Evidence-locked selection)
                                └─ WITNESS Verification ($0 hallucination guard)
```

Instead of simply pruning text, Entroly manages context through three distinct layers:
1. **Cache Aligner:** Anchors your context files at the front of the prompt and mathematically stabilizes their prefix bytes. Even if history changes, the large codebase remains cached, capturing the 90% discount.
2. **Evidence-Locked Selection:** Emits content-compressed retrieval (CCR) handles. If the model needs details from a compressed snippet, it queries a local lookup rather than requesting the entire raw file.
3. **WITNESS Verification:** A $0 local NLI verifier that checks if the model's response is grounded in local evidence *before* streaming, blocking hallucinations without running a second API check.

---

## 🛠️ Step-by-Step Setup Guide

### 1. Install Entroly
Entroly runs entirely locally with no outbound telemetry or cloud dependencies:
```bash
pip install entroly
```

### 2. Run a Local Smoke Test
Before routing live API traffic, run a bounded simulation on your current repository:
```bash
cd /path/to/your/project
entroly verify-claims
```
This generates a `.entroly_verification.json` report showing you exactly how much context can be optimized based on your directory structures.

### 3. Spin Up the Local Proxy
Start the proxy to intercept API traffic:
```bash
entroly proxy --port 9377
```
This automatically stands up a live dashboard at `http://localhost:9378` to track your real-time cache hits, original-vs-compressed token sizes, and cumulative cost savings.

### 4. Configure Your Tools

#### For Claude Code / Aider
Redirect the base URL of your API requests to your local Entroly gateway:
```bash
# For Anthropic Sonnet / Opus
export ANTHROPIC_BASE_URL="http://localhost:9377/v1"

# For OpenAI models
export OPENAI_BASE_URL="http://localhost:9377/v1"
```

#### For Cursor (MCP Mode)
Entroly can also run as a Model Context Protocol (MCP) server:
1. Run `entroly serve` in your terminal.
2. Open Cursor Settings -> **Features** -> **MCP**.
3. Click **+ Add New MCP Server**:
   - **Name:** `entroly`
   - **Type:** `command`
   - **Command:** `entroly serve`

---

## 📊 Expected Cost Savings

| Repo Size | Average Input Tokens | Cache Hit Rate | Bill Reduction |
|---|---|---|---|
| Medium (100–300 files) | 120,000 | ~85% | **~75% lower** |
| Large (500+ files) | 280,000 | ~92% | **~90% lower** |

*Note: For very small repositories (under 30 files) or simple single-file lookups, context compression is less effective. Entroly is designed for complex, multi-file codebases and autonomous loop tasks.*

---

## 🔗 Links & Resources

- **GitHub Repository:** [juyterman1000/entroly](https://github.com/juyterman1000/entroly)
- **Detailed Guides:** [juyterman1000.github.io/entroly](https://juyterman1000.github.io/entroly)
- **License:** Apache-2.0
