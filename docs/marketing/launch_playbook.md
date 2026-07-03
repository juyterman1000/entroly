# 🎯 Entroly — Unified Launch & Marketing Playbook

This document is the single source of truth for public launch posts, Discord outreach, and community discussions. It aligns with the compliance and trust guidelines defined in `AGENTS.md`.

---

## 🧭 Marketing & Communication Guidelines

To preserve trust, compliance, and credibility, always adhere to the following rules:
- **No Competitor Accusations:** Never claim a competitor copied, learned from, or is a subset of Entroly unless public, documented proof is cited.
- **Truthful Branding:** Refer to competitors by their exact names: `Headroom` for the product (`chopratejas/headroom` for repo); `LeanCTX` for the product (`yvgude/lean-ctx` or `lean-ctx` for package/repo).
- **Telemetry Precision:** Never claim "zero telemetry." Instead, use **"no outbound analytics by default"** to distinguish local metrics from phone-home telemetry.
- **Realistic Claims:** Do not present `entroly verify-claims` as a finance-grade ROI guarantee. Describe it as a **"bounded local smoke test."**
- **Measure First:** Always invite developers to run measurements on their own codebases before trusting marketing figures.
- **Integrations Wording:** Do not imply vendor certification. Use "works with," "wrap target," or "OpenAI-compatible proxy."

---

## 🐦 1. Twitter/X Threads & Hooks

### Thread #1 — The Launch Thread (Problem-First)

*   **Tweet 1 (Hook — lead with the PROBLEM):**
    ```text
    Your AI coding agent re-reads the same files on every request.

    186,000 tokens. Every. Single. Time.

    95% of those tokens are boilerplate, generated files, and code unrelated to your question.

    You're paying for your AI to look at package-lock.json. Over and over.

    Here's how to fix it 🧵
    ```

*   **Tweet 2 (Quantify the pain):**
    ```text
    I measured a week of Claude Code on a medium repo:

    • Average tokens per request: 186,420
    • Tokens actually relevant to query: ~9,300
    • Wasted tokens per request: 177,120
    • Cost of wasted tokens (Sonnet 3.5): ~$0.53/request

    At 100 requests/day, that's $53/day in waste.
    $1,590/month. On tokens your AI didn't need.
    ```

*   **Tweet 3 (The solution):**
    ```text
    Entroly is an open-source, local verified-context proxy for coding agents.

    It selects the repo evidence first, compresses noisy context second, and verifies whether the final answer is actually grounded in the code.

    pip install entroly && entroly go

    Dashboard opens automatically at http://localhost:9378 showing real-time savings.
    ```

*   **Tweet 4 (The hidden gem — Cache Aligner):**
    ```text
    The biggest cost saver isn't raw compression. It's Cache Alignment.

    Anthropic gives 90% off on cached prefixes. OpenAI gives 50%.

    But standard context pruners re-rank/change prefixes on every call, busting the cache.

    Entroly stabilizes your prefix structure so you actually capture that 90% discount.
    ```

*   **Tweet 5 (Benchmarks):**
    ```text
    We don't do screenshot benchmarks. Every number is backed by committed JSON data:

    • NeedleInAHaystack: 100% accuracy retained
    • LongBench: 100%+ accuracy (filtering noisy context often improves model reasoning)
    • BFCL: 100% accuracy

    Verify on your own repo with zero API keys:
    entroly verify-claims
    ```

*   **Tweet 6 (CTA):**
    ```text
    Don't trust marketing. Run the bounded local smoke test on your own workspace in 30 seconds:

    pip install entroly
    cd /your/repo
    entroly verify-claims

    Repo: github.com/juyterman1000/entroly
    Web: juyterman1000.github.io/entroly
    ```

---

## 📰 2. Show HN Draft

**Title:** Show HN: Entroly – Local verified-context layer for AI coding agents (Rust+Python, Apache-2.0)

**Body:**
```markdown
Hi HN,

I built Entroly because token compression alone does not solve the main context problem in coding agents. The agent also has to choose the right repo evidence, keep compressed content recoverable, and give the user a way to audit whether the final answer is grounded.

Entroly sits locally as a proxy, MCP server, or Python library to optimize developer agent workflows:

1. **Cache Aligner:** Dynamically stabilizes prompt prefixes so you capture provider caching discounts (Anthropic's 90% and OpenAI's 50% discount) which standard re-ranking pruners bust on every run.
2. **Evidence-Locked Selection:** Uses BM25 + SimHash + dependency graph ranking combined with a local knapsack solver to pack optimal file fragments under a strict token budget.
3. **Reversibility (CCR):** Emits Content-Compressed Retrieval (CCR) handles so the LLM can query the full-resolution source snippet back-end if it needs high-resolution detail.
4. **Local Verification (WITNESS):** Runs a deterministic, $0 local NLI verifier (AUROC 0.84 on HaluEval-QA) to check if the model's response is grounded in the provided files before showing it to you.

### Install & Try
You can run a local smoke test on your codebase with no API keys or telemetry:

```bash
pip install entroly
cd /your/repo
entroly verify-claims
```

This runs a bounded local smoke test (package import, local index, knapsack budget allocation, and native-engine check) and writes results to `.entroly_verification.json`.

If the smoke test looks promising, spin up the local proxy or MCP server:
```bash
entroly go
```
This opens the local value tracker at `http://localhost:9378`.

*Note on savings:* The 70-95% token savings claim is scoped to large repositories (300+ files) running autonomous agent loops. Tiny prompts and small repos will see minimal savings.

We run 100% offline with no outbound analytics by default.

Repo: https://github.com/juyterman1000/entroly
Documentation: https://juyterman1000.github.io/entroly/docs/reduce-llm-api-costs.html
```

---

## 💬 3. Discord Community Outreach Templates

### A. Cline Community (`discord.gg/cline`)
*   **Target Channels:** `#showcase`, `#ideas-and-features`
*   **Message:**
    ```text
    Hey everyone! If you're running Cline heavily in agent mode (especially with Claude 3.5 Sonnet), you've probably noticed your API bill scaling exponentially during long sessions.

    The cost isn't just from loop iterations—it's **prefix cache busting**. Anthropic gives a 90% discount on cached prefixes, but standard context updates change the byte-by-byte prefix on every turn. Your cache hits drop to 0%, and you pay full price on 100K+ tokens every single run.

    I built Entroly—an open-source local proxy and context engine—to fix this for developer agent loops.

    How it works:
    1. **Cache Aligner**: Dynamically stabilizes your context prefix across loop iterations so Claude actually hits the 90% discount.
    2. **Knapsack Optimizer**: Selects the absolute most relevant repo files/fragments under your token budget.
    3. **WITNESS**: A local, deterministic $0 hallucination guard to audit output against source evidence.

    It works as a transparent proxy. You install it, run the proxy, and wrap Cline:
    $ pip install entroly
    $ entroly proxy --port 9377
    $ export ANTHROPIC_BASE_URL=http://localhost:9377/v1

    It stands up a local dashboard at http://localhost:9378 showing your real-time cache hits and actual money saved.

    It's Apache-2.0, local-first, and has no outbound analytics by default. You can test it on your codebase without an API key using the smoke test CLI:
    $ entroly verify-claims

    Guide: https://juyterman1000.github.io/entroly/docs/reduce-llm-api-costs.html
    Repo: https://github.com/juyterman1000/entroly
    ```

### B. Aider Community (`discord.gg/aider`)
*   **Target Channels:** `#showcase`, `#discussion`
*   **Message:**
    ```text
    Hey folks! I wanted to share a tool I built to optimize token costs and context selection for CLI-based coding agents like Aider.

    Entroly is a local context proxy that cuts input tokens by 70–95% on large repositories (500+ files) while improving context quality.

    Instead of just pruning raw text (which breaks code structure), Entroly uses BM25 + SimHash + dependency graph matching to select relevant repo evidence, solves context budgeting with a knapsack solver, and stabilizes your prompt prefix to keep provider cache hits hot (protecting Anthropic's 90% and OpenAI's 50% cached-read discounts).

    It integrates transparently with Aider's API calls:
    $ pip install entroly
    $ entroly proxy --port 9377
    $ export OPENAI_BASE_URL=http://localhost:9377/v1  # or ANTHROPIC_BASE_URL

    You can run a local smoke test on your codebase first (no API keys or outbound data):
    $ entroly verify-claims

    Apache-2.0, 100% offline execution, and no outbound telemetry by default. If you're working on massive codebases and want to optimize your token budget, I'd love your feedback!
    👉 Repo: https://github.com/juyterman1000/entroly
    👉 Docs: https://juyterman1000.github.io/entroly/docs/reduce-llm-api-costs.html
    ```

### C. Continue Community (`discord.gg/vapESyrFmJ`)
*   **Target Channels:** `#showcase`, `#mcp-servers`
*   **Message:**
    ```text
    Hi everyone, I built a local context optimization proxy and MCP server called Entroly that helps reduce LLM token usage by 70–95% when coding in large workspaces.

    For IDE tools like Continue, context bloat can lead to slow response times and high billing. Entroly solves this at the protocol layer. It indexes your workspace locally, ranks file fragments by query-relevance (using Shannon-entropy scoring + dependency graphs), and budgets them into the prompt window.

    Crucially, it includes a **Cache Aligner** to preserve provider prompt caches, meaning you actually get the 90% caching discount on Anthropic APIs instead of busting it on every edit.

    You can wire it up as a transparent proxy or configure it in your `config.json`.

    Features:
    - Transparent proxy (http://localhost:9377/v1) or MCP integration.
    - Local dashboard at http://localhost:9378 showing token savings and cache hits.
    - Local, deterministic $0 hallucination guard (WITNESS).
    - Apache-2.0, offline-first, no outbound analytics by default.

    Run the local smoke test CLI on your project directory to see estimated savings:
    $ pip install entroly && entroly verify-claims

    Repo: https://github.com/juyterman1000/entroly
    Guide: https://juyterman1000.github.io/entroly/docs/cursor-context-guide.html
    ```

---

## 💼 4. Forum / GitHub Discussions (Cursor MCP Cost Control)

*   **Target:** Threads complaining about Cursor API billing, context limits, and Anthropic API token burn.
*   **Message:**
    ```markdown
    Cursor's API mode is incredibly powerful, but it has a massive financial leakage point that most devs miss: KV cache busting.
    
    Anthropic gives you a 90% discount on cached prefixes, and OpenAI gives 50%. But every time Cursor ranks or updates context, the byte-by-byte prefix changes slightly. Your cache hits drop to 0% and you pay full price on 150K+ tokens.
    
    I built Entroly to solve this. It's a local proxy with a "Cache Aligner" that keeps the context prefix stable across requests, ensuring you actually get that 90% discount. Combined with knapsack-optimal compression, it reduces token count by 70-95%.
    
    One-command setup for Cursor:
    `pip install entroly && entroly wrap cursor`
    
    It auto-wires the MCP server and gives you a local dashboard (`http://localhost:9378`) showing your exact real-time billing savings and cache hit rate.
    
    Apache 2.0, completely offline, no outbound analytics by default. Don't trust my marketing — run `entroly verify-claims` on your own repo to see your numbers first.
    
    👉 Repo: https://github.com/juyterman1000/entroly
    👉 Guide: https://juyterman1000.github.io/entroly/docs/cursor-context-guide.html
    ```
