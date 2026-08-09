<p align="center">
  <img src="docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Verified Code Intelligence + Context Assurance for AI Agents</h1>

<p align="center"><b>Advanced repository intelligence that finds, verifies, budgets, and preserves the code evidence an AI agent actually needs.</b><br>
AST and Tree-sitter structure, dependency and call graphs, interprocedural flow, architecture analysis, semantic change intelligence, recoverable compression, and tamper-evident receipts — in one local-first context system.</p>

<p align="center">
  <sub>Entroly is a local-first Context OS and verified code-intelligence layer for AI coding agents: content-addressed evidence, repository understanding, recoverable compression, and auditable receipts. Works through proxy, MCP, plugin, wrapper, and SDK paths with Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider, and OpenAI/Anthropic-compatible apps.</sub>
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

<p align="center">
  <b><a href="#what-is-entroly-in-plain-english">What is it?</a> · <a href="#advanced-code-intelligence-for-ai-agents">Code intelligence</a> · <a href="#install">Install</a> · <a href="#quickstart--by-how-you-work">Quickstart</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#common-questions">Questions</a></b>
</p>

---

## What is Entroly? (in plain English)

AI coding assistants have a memory limit. Hand one your whole codebase and it
gets slow, expensive, and distracted — like giving someone a 500-page manual
when they only needed page 47.

**Entroly finds page 47 — and can prove why it is page 47.**

It sits between your code and the AI, builds a bounded understanding of the
repository, selects the evidence that matters for the current task, and passes
along only the useful context. Three things make that safe to do:

|  |  |
|---|---|
| 🧠 **It understands repository structure** | AST/parser-backed declarations, calls, dependencies, architecture, change relationships, and verified source spans help agents reason over code instead of treating a repo as a bag of text. |
| 🔍 **Nothing is silently lost** | Whatever Entroly sets aside is kept and can be pulled back *exactly* as it was, character for character. |
| 🧾 **You can check its work** | Every decision can carry evidence: what was selected, what was omitted, source hashes, ambiguity, and what remains unproven. |

**Do I have to change my code?** No. Entroly works with the tools you already
use — Claude Code, Cursor, Copilot and 30+ others — and runs in the background.

**Do I need to pay for anything to try it?** No. The two commands in the
[Install](#install) section below run entirely on your own machine, with no API
key, and show you real numbers on your own project before you connect anything
paid.

---

## Advanced Code Intelligence for AI Agents

**Entroly is a verified code-intelligence and repository-understanding engine for AI coding agents.** It combines parser-backed AST structure with graph reasoning, source verification, architecture intelligence, change analysis, and budget-aware context selection so an agent gets the *right code evidence*, not merely more code.

Entroly is built to be **one of the world's most advanced open-source code-intelligence systems for AI agents**, with an unusual emphasis on evidence, freshness, bounded reasoning, and fail-closed behavior. Instead of turning a heuristic guess into a fact, Entroly keeps ambiguity and omissions visible and attaches exact source identities to the relationships it claims.

### What makes Entroly code intelligence advanced?

| Intelligence layer | What Entroly does |
|---|---|
| **AST + parser intelligence** | Native Python AST analysis plus an open Tree-sitter language registry for parser-backed declarations, recognized call sites, exact byte spans, signatures, and structural profiles. Missing grammars fall back conservatively instead of pretending parsing succeeded. |
| **Repository graph intelligence** | Builds symbol, import, call, containment, dependency, reverse-impact, neighborhood, shortest-path, and relatedness views with explicit unresolved and ambiguous evidence. |
| **Typed call resolution** | Distinguishes same-name methods using receiver/type evidence where it is actually known and refuses ambiguous dispatch instead of guessing. |
| **Interprocedural reasoning** | Source-verified Python argument→parameter and explicit return→result value-flow summaries cross function boundaries while preserving the exact source spans that justify every relationship. |
| **Architecture intelligence** | Computes SCCs/cycles, condensation layers, communities, entry-to-foundation routes, hotspots, architecture diffs, and deterministic witnesses rather than producing an unexplained architecture score. |
| **Semantic change intelligence** | Uses Git-object semantic diffs, source hashes, architecture diffs, portable graph snapshots, freshness checks, and incremental invalidation so stale analysis is not silently reused. |
| **Build/test topology** | Maps repository components from verified manifests across Cargo, Python, Node, Go, Zig, CMake, Maven/Gradle, Bazel, Swift, Elixir, Ruby, PHP, Dart, Clojure, Haskell, Meson, Make, and more, while explicitly distinguishing manifest ancestry from proven build inclusion. |
| **LSP-enriched intelligence** | An operator-configured Language Server can add bounded definitions, references, overrides, and workspace relationships; Entroly verifies returned source ranges before using them as evidence. |
| **Verified refactoring intelligence** | Two-phase rename, safe-delete, and Python module-move planning bind edits to exact preimages, reject stale/tampered plans, preserve ambiguity, validate staged syntax, and make rollback failures visible. |
| **Evidence-aware code context** | Code slices combine structural graph evidence with token budgets, protected dependency signatures, exact source-span digests, recoverability, and tamper-evident receipts. |
| **Local-first trust model** | Repository analysis is local by default; parser acquisition is explicit, air-gap mode wins, and external-process network behavior is labeled rather than silently claimed safe. |

### Why this matters for AI coding agents

Most retrieval systems answer **“which text looks similar?”** Entroly can also answer higher-order repository questions such as:

- *Where is this symbol defined, called, imported, or contained?*
- *Which implementation can this typed call resolve to — and where is it ambiguous?*
- *What code can be affected if this function or module changes?*
- *What is the shortest verified relationship path between these symbols?*
- *Which components form cycles, architectural layers, communities, routes, or hotspots?*
- *Which source evidence is stale, missing, unresolved, truncated, or outside the current token budget?*
- *Can this rename/delete/move be planned without silently editing ambiguous or stale references?*
- *Which exact evidence should an AI agent receive under a fixed context budget?*

That combination is why Entroly is more than a context compressor. It is a **code-intelligence control plane for AI agents**: understand the repository, select evidence, verify freshness, preserve provenance, then compress only after the right context has been chosen.

**Evidence, not slogans:** the repository includes a [39-dimension code-intelligence conformance protocol](benchmarks/CODE_INTELLIGENCE_CONFORMANCE.md) covering structural correctness, typed dispatch, call/dependency graphs, value flow, LSP ranges, cache invalidation, architecture reasoning, refactoring safety, stale-source rejection, and tamper evidence. The design and limitations are documented in [Verified Code Context](docs/verified-code-context.md) and [Universal Code Intelligence](docs/research/universal-code-intelligence.md).

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
entroly verify-claims   # checks the install really does what this page claims
entroly simulate        # shows how much smaller YOUR project would get
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
| **"I use Claude Code / Cursor / Windsurf / VS Code."** *(MCP user)* | `entroly attach create --client claude --project . --ttl 4h --install` (or `entroly init` for Cursor/VS Code) | Your editor gets compression, receipts, recovery, and repository intelligence as built-in tools — access expires on its own, and you change zero code |
| **"I'm building my own app in Python."** *(SDK user)* | `from entroly import compress, compress_messages, optimize` | Call it straight from your code, anywhere you assemble a prompt |
| **"I have an API key and my own app."** *(proxy user)* | `entroly proxy` → point `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL` at `localhost:9377` | Every request gets optimized on the way past — no code changes on your side |

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

Recovery, latency, and head-to-head frontier results — [context-cap retention](docs/BENCHMARKS.md#matched-token-cap-active-context-quality-frontier-1059-source-candidate), [compression gauntlet](docs/BENCHMARKS.md#same-input-compression-gauntlet), [cross-process recovery](docs/BENCHMARKS.md#cross-process-recovery-holdout), [model-triggered recovery](docs/benchmarks/model-triggered-recovery.md) — are in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** with every raw artifact linked, including the [PRISM-R neural evidence frontier](docs/benchmarks/neural-evidence-frontier.md). None of these numbers are a universal or production-savings guarantee for your workload — reproduce them on your own repo with `entroly simulate` and `entroly value`.

---

## Features

- **Picks first, shrinks second** — it works out which files actually answer your question, *then* compresses them. Most tools just squeeze whatever text was already handed to them. <sub>Ranks your repo or document set with BM25 + entropy + dependency graph, then fills a token budget.</sub>
- **Gives you the original back, exactly** — anything left out can be restored character-for-character, and it's checked against a fingerprint so you know it wasn't altered. <sub>Digest- and length-verified handle to the original bytes; `entroly compress` / `entroly recover` prove it across a process boundary.</sub>
- **Shows its work** — a receipt for every decision: what was kept, what was left out and why, and what risk remains. <sub>Includes dependency links and source-span digests.</sub>
- **Fact-checks answers** — compares what the AI said against the evidence it was given, on your machine, without paying for a second AI call. <sub>WITNESS verifier, fully local.</sub>
- **Doesn't wreck your caching** — keeps the unchanging parts of your prompt stable so your provider's discount for repeated text still applies. <sub>Stable system/history bytes stay ahead of changing context.</sub>
- **Rescues sessions before they crash** — when a conversation grows too big, it trims recoverable output instead of letting the provider reject the request mid-task. <sub>Proxy-side compaction of tool output.</sub>
- **Lets agents choose the evidence fidelity** — `smart_read` supports exact full text, inclusive line ranges, whole-file diffs, budget-aware automatic reads, and lightweight multi-language structure outlines. Exact same-session re-reads collapse to an opaque content handle; `fresh=true` restores the rendered text. <sub>Source-, contract-, and output-digest guarded; structure fails open to full source when no useful outline is available.</sub>
- **Builds verified code context** — an optional local-first parser pack uses an open language registry and emits exact declaration and recognized call-site spans. Proof-carrying partial slices combine budgeted source, typed graph queries, architecture layers/communities/routes, Git-object semantic diffs, build/test topology, portable graph snapshots, and intra- and interprocedural Python value-flow summaries; structural health and two-phase rename, safe-delete, or Python module-move plans use the same source hashes, ambiguity, omissions, and tamper-evident receipts. <sub>Optional learned scores may rank existing symbols but cannot create facts or raise confidence. The cross-function layer covers verified argument/parameter and explicit return/result bindings, not whole-program heap or side-effect flow. An operator-configured LSP can add bounded workspace references; Entroly verifies returned ranges but cannot enforce the external server's network behavior. Community and hotspot scores are disclosed heuristics, and refactors do not claim compiler-complete references; [design and limitations](docs/verified-code-context.md).</sub>
- **Understands command outcomes** — shell profiles preserve invocations, failures, and terminal summaries under tight budgets. <sub>Unknown commands retain deterministic fallbacks and exact recovery.</sub>
- **Fits existing Python gateways** — metadata-safe LangChain messages/documents, a LiteLLM pre-call hook, and bounded ASGI middleware reuse the same budgeted message compressor without changing tool contracts. <sub>Framework dependencies remain optional.</sub>
- **Can route cheap work to cheap models** — optional, and when it isn't confident it always picks the stronger model rather than gambling. <sub>Fail-closed routing; local ranking adapts from recorded outcomes, no embeddings API.</sub>

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

**Great fit:** large repositories where an AI agent needs repository understanding, dependency-aware code context, architecture reasoning, or verified source evidence · chatty multi-turn agents (cache alignment compounds savings) · anywhere you want answers checked against evidence · cutting a real, growing AI bill.

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

## Common questions

<details>
<summary><b>Is Entroly only a context compressor?</b></summary>
<br>

No. Entroly includes a local-first **code-intelligence and repository-intelligence layer** for AI agents: AST/parser-backed structure, symbol and call relationships, dependency graphs, typed dispatch, architecture analysis, semantic Git/change intelligence, source-verified interprocedural flow, LSP-enriched relationships, build/test topology, verified refactoring plans, evidence receipts, and budget-aware code-context selection. Compression happens after evidence selection; it is one layer of the system, not the whole product.
</details>

<details>
<summary><b>What makes Entroly different from ordinary code search or RAG?</b></summary>
<br>

Text search and RAG are primarily similarity systems. Entroly can combine lexical relevance with **structural, relational, and source-verified evidence**: where a symbol is defined, which calls are resolvable, what depends on what, which paths are ambiguous, what changed, whether the indexed source is still fresh, and which evidence can fit safely inside the model's context budget. Where it cannot prove a relationship, it keeps that uncertainty visible instead of silently manufacturing certainty.
</details>

<details>
<summary><b>Will this change my code or my files?</b></summary>
<br>

No. Entroly's context-selection and analysis paths do not silently edit your project. Explicit refactoring operations use preview/plan/apply contracts with source verification and tamper checks before mutation.
</details>

<details>
<summary><b>Does my code get uploaded anywhere?</b></summary>
<br>

No. All the selecting, compressing, and checking happens on your own machine.
Entroly makes no outbound calls of its own — the only thing that leaves your
computer is the request you were already sending to your AI provider, just
smaller. There are no analytics on by default. Optional external tools such as an operator-configured language server keep their own network boundary, which Entroly reports rather than hiding.
</details>

<details>
<summary><b>What if it leaves out something important?</b></summary>
<br>

Nothing is thrown away. Anything left out is stored and can be restored exactly
as it was — `entroly recover` gives you back the original, character for
character, and it's verified against a fingerprint so you know it wasn't
altered. The receipt also tells you what was left out and why, so you're never
guessing.
</details>

<details>
<summary><b>How much money will this actually save me?</b></summary>
<br>

It depends on your project, and anyone who gives you a single number without seeing your work is guessing. Run `entroly simulate` in your project — it's free, needs no API key, and estimates the reduction on *your* files. If your prompts are already small, Entroly passes them through untouched and saves you nothing, which it will tell you.
</details>

<details>
<summary><b>I'm not a developer. Can I use this?</b></summary>
<br>

If you use an AI coding tool like Claude Code or Cursor, yes. Install it
(`pip install -U entroly`), then run `entroly go` — it finds your editor,
configures itself, and opens a dashboard. You don't need to write any code or
understand what's under the hood. If you don't use an AI coding assistant,
Entroly isn't for you.
</details>

<details>
<summary><b>Something broke / I'm stuck.</b></summary>
<br>

Run `entroly doctor` — it checks your setup and reports what's wrong. If that
doesn't sort it, [open an issue](https://github.com/juyterman1000/entroly/issues)
or ask in [Discussions](https://github.com/juyterman1000/entroly/discussions).
</details>

<details>
<summary><b>What do all these words mean? (jargon decoder)</b></summary>
<br>

| Word | What it actually means |
|---|---|
| **AST** | Abstract Syntax Tree — parser-produced code structure used to identify declarations, calls, scopes, and other syntax relationships more reliably than plain text matching. |
| **Code intelligence** | Structured understanding of symbols, calls, dependencies, architecture, changes, and source evidence that helps an AI agent reason about a repository. |
| **Token** | Roughly ¾ of a word. AI providers charge per token, so "fewer tokens" = smaller bill. |
| **Context** | Everything you send the AI along with your question — your code, past messages, documents. |
| **Context window** | The AI's memory limit. Go over it and things get dropped or rejected. |
| **CLI** | Command-line tool — the thing you type `entroly ...` into. |
| **SDK / library** | Code you can call from your own program, instead of typing commands. |
| **MCP** | A standard way for AI editors (Claude Code, Cursor) to plug into outside tools. Entroly speaks it, so those editors can use it directly. |
| **Proxy** | A middleman that sits on the path between your app and the AI, so it can shrink requests without you changing any code. |
| **Receipt** | Entroly's written record of what it kept, what it left out, and why. |
| **Compression** | Sending less text while keeping the meaning. |
| **Recovery** | Getting the left-out original back, exactly as it was. |

</details>

---

## Docs & community

- **[Verified Code Context](docs/verified-code-context.md)** — parser-backed repository intelligence, typed graphs, architecture, value flow, LSP enrichment, source verification, and refactoring contracts.
- **[Code-intelligence conformance](benchmarks/CODE_INTELLIGENCE_CONFORMANCE.md)** — 39 explicit structural, semantic, repository-understanding, freshness, safety, and evidence dimensions.
- **[Universal Code Intelligence](docs/research/universal-code-intelligence.md)** — design direction for language-open, evidence-backed repository understanding.
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

> Compressing a *bad* selection is still a bad selection. Entroly understands and ranks repository evidence first, then compresses — so the model gets verified structure, not just fewer tokens.

<p align="center"><sub>Apache-2.0 · local-first · verified code intelligence · recoverable context · no outbound analytics by default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>

<!-- mcp-name: io.github.juyterman1000/entroly -->