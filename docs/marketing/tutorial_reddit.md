# Community Reddit Marketing Templates

Use these templates to share Entroly on relevant subreddits. Focus on providing raw value, explaining the mechanics, and letting users verify their savings locally before making any external network requests.

---

## 1. Subreddit: r/ClaudeAI / r/ClaudeCode

**Title:** Guide: Why your Claude Code / Cline loops are costing you a fortune (and how prompt cache alignment fixes it)

**Body:**
```markdown
If you’ve been running autonomous coding agents like Claude Code, Cline, or Roo-Cline heavily, you’ve probably seen some staggering numbers on your Anthropic API bill. 

For a medium-sized codebase, prompts easily hit **120k–180k tokens** on every message loop. At Sonnet 3.5 rates, that's ~$0.45 per run. A single debugging loop of 10–15 steps can run you $6.00.

### The Problem: Re-Ranking busts Anthropic's 90% Cache Discount
Anthropic offers a massive 90% discount on cached prompt prefixes. So why isn't it saving you money? 

Every time your coding agent loops:
1. It updates the chat history with tool results or shell outputs.
2. It re-ranks and re-orders context files to try and fit them under the token limit.

This re-ranking changes the exact sequence of bytes in the prompt prefix. Because caching requires a **100% exact byte-by-byte match**, the prefix cache is invalidated (busted) on every single turn. You end up paying full price for the entire context over and over.

### The Solution: Prompt Cache Alignment
I built **Entroly**—a local context control proxy—to address this exact issue. It sits between your coding tool and the API.

Instead of re-ordering files blindly, Entroly’s **Cache Aligner** keeps the context files anchored at the beginning of the prompt and stabilizes their prefix structure. By isolating dynamic chat history to the end, the large codebase remains permanently cached at the provider level, securing the 90% discount.

It also features:
- **Knapsack Selection:** Solves optimal file packing under a strict token budget.
- **WITNESS:** A local $0 Natural Language Inference (NLI) verifier that checks model output against local files before showing it to you (preventing hallucinated imports/APIs).

### Setup in 30 Seconds:
It runs entirely locally with no phone-home telemetry:

1. Install it:
   ```bash
   pip install entroly
   ```
2. Run a local simulation on your repo with no API keys:
   ```bash
   entroly verify-claims
   ```
3. Start the proxy:
   ```bash
   entroly proxy --port 9377
   ```
4. Point Claude Code / Aider to it:
   ```bash
   export ANTHROPIC_BASE_URL="http://localhost:9377/v1"
   ```

It opens a dashboard at `http://localhost:9378` showing your real-time cache hits and billing savings.

It’s open source (Apache-2.0). If you are running long agent loops on large codebases, check it out and see if it helps keep your API bills reasonable!

👉 **GitHub:** https://github.com/juyterman1000/entroly
👉 **Documentation & Guides:** https://juyterman1000.github.io/entroly/
```

---

## 2. Subreddit: r/LocalLLaMA

**Title:** Entroly - Open source local context-engineering gateway (Rust/Python, Apache-2.0)

**Body:**
```markdown
Hi LocalLLaMA community,

If you are using local models (like Ollama, Llama.cpp, or vLLM) for coding agents or complex RAG tasks, you know that keeping context windows optimized is crucial for both speed (prompt processing latency) and reasoning accuracy.

I developed **Entroly**, a local context-control plane designed specifically to solve this.

Unlike generic summarizers that just delete lines or drop structural elements of the code, Entroly handles context allocation as a resource-constrained knapsack optimization problem:

1. **Entropy Density Ranking:** Evaluates code blocks based on information density (using Shannon entropy + BM25 + SimHash deduplication).
2. **Knapsack Pack:** Fits the most relevant code fragments exactly within the local model's token limits.
3. **Reversibility (CCR):** Emits Content-Compressed Retrieval handles. If the local LLM needs raw details from a compressed file snippet, it queries a local backend to retrieve full-resolution detail rather than flooding the context window up-front.
4. **WITNESS Guard:** Includes a local, deterministic $0 Natural Language Inference (NLI) verifier that runs locally to check if model outputs are grounded in your source files.

It compiles down to a PyO3-based Rust core (`entroly-core`) for sub-10ms performance, wrapped in a developer-friendly Python/CLI toolkit.

It runs 100% locally with no telemetry by default.

### Test it locally:
```bash
pip install entroly
cd /your/repo
entroly verify-claims
```
This runs a local smoke test simulation on your project directory and outputs `.entroly_verification.json` containing token-packing and index efficiency statistics.

Would love to get feedback on how it behaves on large repositories and if it speeds up your local code generation pipelines.

👉 **GitHub:** https://github.com/juyterman1000/entroly
👉 **Specs:** https://juyterman1000.github.io/entroly/docs/context-engineering.html
```

---

## 3. Subreddit: r/selfhosted

**Title:** Show selfhosted: Entroly — Self-hosted context compression & verification proxy for AI developer workflows (Apache-2.0)

**Body:**
```markdown
Hey r/selfhosted,

I wanted to share **Entroly**, a project I've been working on to self-host context control and verification for AI coding tools. 

If you self-host LLM APIs or use third-party endpoints with IDE tools like Cursor or Continue, prompt volume and hallucination rates are the biggest pain points. Entroly sits as a local gateway on your machine to optimize and audit these connections.

### What it does:
- **Reduces token usage by 70–95%** on large directories using local knapsack optimization.
- **Preserves KV prompt caches** by stabilizing the prompt prefix byte-structure (ensures you actually capture Anthropic's 90% or OpenAI's 50% cache discounts).
- **WITNESS verification:** Audits AI responses locally against your files to block hallucinated methods/configs before they render.
- **Offline first:** Zero outbound telemetry or cloud dependencies.

### Quick Start:
```bash
pip install entroly
entroly proxy --port 9377
```
Then configure your self-hosted tools (like Open WebUI, Cline, or Continue) to use the proxy base URL: `http://localhost:9377/v1`.

You can view real-time optimization logs, exact cache hit ratios, and estimated dollar savings via a local dashboard at `http://localhost:9378`.

Check out the repo if you want to optimize your self-hosted AI developer setup!

👉 **GitHub:** https://github.com/juyterman1000/entroly
👉 **Documentation:** https://juyterman1000.github.io/entroly/docs/reduce-llm-api-costs.html
```
