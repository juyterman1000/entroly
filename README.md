<p align="center">
  <img src="docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Context Assurance That Helps Lower AI Costs</h1>

<p align="center"><b>Use less unnecessary AI context without rebuilding how you work.</b><br>
Entroly sits between your AI application and the model. It selects the useful information, removes avoidable repetition, keeps important originals recoverable, and records what the AI received.</p>

<p align="center">
  <sub>Works through supported proxy, wrapper, plugin, SDK, and MCP integrations with Claude Code, Codex, OpenClaw, Hermes Agent, OpenCode, GitHub Copilot, Cursor, Aider, local models, and OpenAI/Anthropic-compatible applications. A small one-time setup is required; no agent-architecture rewrite.</sub>
</p>

<p align="center">
  <b>Budgeted context selection · recoverable compression · content-addressed evidence · Context Receipts · local-first · model neutral</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center">
  <a href="#start-here"><b>Start here</b></a> ·
  <a href="#what-entroly-does-in-plain-language"><b>Plain-language guide</b></a> ·
  <a href="#works-with-your-ai-tools"><b>Integrations</b></a> ·
  <a href="#technical-section"><b>Technical section</b></a> ·
  <a href="#evidence-and-limitations"><b>Evidence</b></a> ·
  <a href="docs/limitations.md"><b>Limitations</b></a>
</p>

---

## What Entroly does in plain language

AI applications often send large amounts of text, files, chat history, logs, and repeated information to a model. More input can mean more tokens, more API spending, slower requests, and more chances for important evidence to get buried.

Entroly acts as a **Context Assurance layer**:

1. It examines the available information locally.
2. It selects the evidence most useful for the current request.
3. It removes duplicates and compresses supporting material when appropriate.
4. It keeps omitted originals recoverable through exact content handles.
5. It creates a receipt showing what was included, omitted, and considered risky.
6. It can verify whether an answer is supported by the supplied evidence.

Entroly does **not** replace Claude, ChatGPT models, Codex, OpenClaw, Hermes, OpenCode, your local model, or your existing agent. It controls the context around them.

```text
Your application
      ↓
Entroly Context Assurance
  • choose useful evidence
  • remove avoidable repetition
  • preserve exact originals
  • create a receipt
      ↓
Your chosen AI model
```

### Why people use it

- **Lower avoidable inference expenditure:** send less unnecessary input through supported provider-bound routes.
- **Fewer context-induced failures:** reduce distraction from duplicate, stale, or low-value material.
- **Recoverable evidence:** retrieve an omitted original by its exact content-addressed handle.
- **Clear records:** inspect what the model saw and what it did not see.
- **No model lock-in:** use the same Context Assurance layer across cloud and local models.
- **No architecture rewrite:** connect through a supported proxy, wrapper, plugin, MCP, or SDK path.

Savings and answer quality depend on the workload, model, budget, provider, and integration. Entroly reports what it can observe and does not promise a universal compression percentage or guaranteed bill reduction.

---

## Who is Entroly for?

### Everyday AI users

Entroly's long-term goal is a simple experience for librarians, teachers, healthcare staff, cleaners and field-service workers, students, writers, office workers, and small-business owners:

```text
Install Entroly → choose an AI app → turn on Context Assurance → use the app normally
```

That no-terminal desktop experience is specified in [Entroly Simple Mode](docs/product/entroly-simple-mode.md), but it is **not shipped yet**. Today's release is easiest for users of supported developer tools, AI agents, local models, and API-compatible applications. A technical person or administrator may need to perform the small one-time connection step.

### Developers and AI teams

Entroly is useful when:

- an agent repeatedly reads a medium or large repository;
- API input costs are meaningful;
- long sessions accumulate duplicate context;
- multiple agents need the same recoverable evidence;
- teams need receipts, verification, replay, privacy boundaries, or budget gates;
- model providers or agent frameworks may change over time.

Entroly is usually a weak fit for tiny prompts, tiny repositories, or workflows that require every input byte to remain unchanged.

---

## Start here

### 1. Install

```bash
pip install -U entroly
```

Node users can install the separate npm/WASM runtime:

```bash
npm install -g entroly
```

### 2. Check it locally before connecting a paid model

```bash
entroly verify-claims   # installation, receipts, verification, and recovery smoke test
entroly simulate        # local estimate on the current repository; no model call
entroly value           # evidence-classified usage and value report
```

`simulate` is an estimate, not provider billing or a quality guarantee. `value` keeps provider-observed usage separate from local-only reductions.

### 3. Connect one supported tool

The simplest automatic path is:

```bash
cd /your/repo
entroly go
```

Or choose a specific integration:

| You use | Supported setup | What stays unchanged |
|---|---|---|
| Claude Code | `entroly attach create --client claude --project . --ttl 4h --install` | Claude Code remains the client and model route |
| Codex | `entroly attach create --client codex --project . --ttl 4h --install` | Codex remains the coding agent |
| OpenClaw | `entroly attach create --client openclaw --project . --ttl 4h --install` or the first-class context-engine plugin | OpenClaw remains the agent runtime |
| Hermes Agent | Entroly `ContextEngine` adapter | Hermes remains responsible for its transcript, tools, and provider |
| OpenCode | Local MCP configuration and compaction plugin | OpenCode remains the coding interface |
| Cursor, VS Code, Windsurf, Cline, Continue, Zed | Register `entroly` as a local stdio MCP command | Your editor and model stay the same |
| Pay-as-you-go API or custom application | `entroly proxy` | Your existing OpenAI-, Anthropic-, or Gemini-compatible application stays in place |
| Python application | `from entroly import compress, compress_messages, optimize` | Your application controls when optimization runs |

Detailed integration pages:

- [Agent integration hub](docs/agent-integrations.html)
- [Entroly for OpenClaw](docs/openclaw-context-engine.html)
- [Entroly for Hermes Agent](docs/hermes-context-engine.html)
- [Entroly for OpenCode](docs/opencode-context-assurance.html)
- [MCP server guide](docs/mcp-server-guide.html)

### Proxy mode for API applications

```bash
entroly proxy   # starts locally at http://localhost:9377

ANTHROPIC_BASE_URL=http://localhost:9377 your-app
OPENAI_BASE_URL=http://localhost:9377/v1 your-app
GOOGLE_GEMINI_BASE_URL=http://localhost:9377/v1beta your-app
```

Indexing, ranking, receipts, and recovery storage run locally. The selected prompt still goes to the model provider configured by your application. Entroly sends no outbound analytics by default.

---

## Works with your AI tools

Entroly is designed to be model- and runtime-neutral. Supported surfaces include:

- Claude Code and Anthropic-compatible applications
- Codex and OpenAI-compatible applications
- OpenClaw
- Hermes Agent
- OpenCode
- GitHub Copilot and VS Code MCP
- Cursor, Windsurf, Cline, Continue, Zed, and other MCP clients
- Ollama, LM Studio, and other local-model workflows
- Python, Node/WASM, CLI, CI, proxy, and MCP integrations

A listed tool may use a different connection method. “Works with” does not mean every consumer web application can be intercepted automatically. For example, a fixed-price ChatGPT or Claude web subscription does not necessarily become cheaper; provider-bound cost reduction is most directly measurable for API and agent routes where Entroly can observe the input sent.

---

## What a Context Receipt tells you

A receipt can answer:

- Which evidence was selected?
- Which nearby or relevant evidence was omitted?
- Why was an item selected or omitted?
- How much context was reduced?
- Which originals remain recoverable?
- Were there unresolved evidence risks?
- Was the reduction observed on a provider-bound request or only locally?

Example:

```text
84 items were available.
12 relevant items were selected.
7 duplicate or low-value items were omitted.
3 omitted originals remain recoverable.
1 evidence risk needs review.
Provider-bound input reduction: measured / not observed.
```

Technical examples:

- [Context Receipt JSON](docs/examples/context_receipt.json)
- [Human-readable receipt](docs/examples/context_receipt.md)
- [Context Commit contract](docs/context-commits.md)

Exact recovery remains available only while the corresponding receipt and recovery store are retained. Deleting that state deletes Entroly's recovery path.

---

## How Entroly reports savings honestly

Entroly separates different kinds of evidence instead of turning every local reduction into a dollar claim.

| Evidence class | What Entroly can report | Cost statement |
|---|---|---|
| Provider-bound proxy requests | Observed pre/post input tokens and model-rate provenance | Modeled input-cost avoidance; not a provider invoice |
| SDK, MCP, plugin, and npm operations | Local context and tokens reduced | `$0` claimed when provider delivery is not observable |
| Fixed-price subscriptions | Context efficiency and local reductions where supported | The subscription price may not change |
| Unknown or legacy history | Preserved operational history | Excluded from provider-savings claims |

If a repository is already small or a request is already under budget, Entroly should pass it through rather than invent savings.

---

# Technical section

The sections below are for engineers, platform teams, researchers, and security reviewers.

## Technical architecture

Entroly is a local control plane around model context:

```text
Sources and tool output
        ↓
Ingestion and fingerprints
        ↓
Ranking: BM25 + entropy + dependency signals
        ↓
Budgeted evidence selection
        ↓
Recoverable compression and exact CCR handles
        ↓
Context Receipt / Context Commit
        ↓
Existing model or agent runtime
        ↓
Optional WITNESS verification and outcome feedback
```

### Core guarantees and boundaries

- **Budgeted selection:** context is selected under an explicit token budget.
- **Content addressing:** exact recovery handles refer to one stored original, not a semantic search query.
- **Fail-open integration:** when a safe optimized replacement cannot be produced, adapters preserve the host's original context.
- **Provider neutrality:** Entroly does not choose or receive provider credentials for host-managed routes.
- **Replayability:** Context Commits bind selected text, omitted evidence, recovery data, engine identity, and optional lineage.
- **Local-first processing:** indexing and selection run locally; provider-bound prompts follow the route configured by the user.
- **Evidence before claims:** benchmark and cost statements are scoped to committed artifacts and observable paths.

See [architecture](docs/DETAILS.md), [complete product surface](docs/product-surface.md), [team and security guide](docs/for-teams.md), and [limitations](docs/limitations.md).

## Exact recovery contract

Known recovery handles use a strict content-addressed form:

```text
ccr:<24-hex>
```

`entroly_retrieve(hash)` performs a hash-only full-content lookup. It does not accept a natural-language query, source path, or partial hash, and it does not silently substitute a newer source revision when historical content is missing.

Discovery and recovery are separate operations:

```text
retrieval/ranking → discover which evidence matters
exact CCR lookup  → return this exact stored original
```

## Context Commits

```bash
entroly context-commit ./repo \
  --query "Where is token rotation enforced?" \
  --budget 8000 \
  --out context-commit.json

entroly context-commit --verify context-commit.json
```

A Context Commit may contain source text in its recovery bundle. Protect it under the same access and retention rules as the source. Content addressing detects mutation; it does not by itself prove signer identity.

## Context Receipts and verification

```bash
entroly ingest ./docs
entroly select --query "What evidence supports this answer?" --budget 8000
entroly receipt .entroly/receipts/cr_example.json
entroly audit .entroly/session_chain.json
```

WITNESS can evaluate whether generated claims are supported by supplied evidence. It does not establish universal truth. Strict suppression is opt-in; default proxy behavior is audit-oriented.

## CI and budget gates

```bash
entroly batch --budget 8000 --fail-over-budget
entroly verify-claims
entroly doctor
```

Entroly includes a pure-Python runtime, optional Rust acceleration, and a separate Node/WASM runtime. Installed capabilities depend on the selected distribution and extras.

---

## Evidence and limitations

### Reproducible local verification

```bash
entroly verify-claims
```

This checks the installed path for imports, receipts, recovery, verification, routing, and replay. It is a smoke test—not a model-quality or savings benchmark.

### Frozen evidence-selection benchmark

In a frozen 300-question retrieval experiment, Entroly selected **1.02 of 16 passages on average** while retaining the answer-bearing passage in **298 of 300** questions. BM25 retained it in 297 and a local transformer in 293. The difference was not statistically conclusive (`p=0.21875`), and the experiment measures retrieval, not generated-answer quality or production cost.

- [Protocol](docs/benchmarks/neural-evidence-frontier.md)
- [Raw artifact](benchmarks/results/neural_evidence_frontier.json)

### Model-triggered recovery benchmark

On one frozen 24-case local Qwen2.5-1.5B holdout, Entroly answered 24/24 cases and the published Headroom 0.31.0 baseline answered 18/24 at different effective-context levels. This is a synthetic versioned comparison, not a universal product claim.

- [Protocol and limitations](docs/benchmarks/model-triggered-recovery.md)
- [Raw artifact](benchmarks/results/model_recovery_v7_holdout.json)

### Restart recovery

A fresh-seed Windows revalidation recorded byte-exact restart recovery for 66/66 Entroly payloads and 66/66 payloads for the published Headroom 0.31.0 baseline. This demonstrates parity on that run, not universal durability leadership.

- [Competitive evidence matrix](docs/benchmarks/competitive-evidence-matrix.md)
- [Raw artifact](benchmarks/results/recovery_resilience_holdout_revalidation_v4.json)

### Context Commit conformance

Committed synthetic fixtures report:

- 128/128 deterministic replays
- 576/576 exact omitted-chunk recoveries
- 768/768 tamper mutations detected

These measure artifact integrity and recovery on committed fixtures, not end-to-end model quality.

- [Raw artifact](benchmarks/results/context_commit_conformance.json)

---

## Frequently asked questions

### What is Context Assurance?

Context Assurance is the process of controlling what information an AI receives, preserving important originals, recording omissions, and checking whether the resulting answer is supported by evidence.

### Is Entroly just prompt compression?

No. Compression is one optional step. Entroly first selects useful evidence under a budget, keeps exact recovery handles, produces receipts, and can verify the answer.

### Can Entroly reduce AI API bills?

It can reduce avoidable input sent through supported provider-bound routes. Actual cost impact depends on pricing, cache behavior, model, workload, and the amount of reducible context. Entroly does not guarantee a fixed percentage.

### Will Entroly lower my ChatGPT Plus or Claude subscription price?

Usually not directly—the subscription price is fixed. Entroly is most directly useful for API-based applications, coding agents, local models, and supported tools where context usage can be controlled. Efficient context may help with usage limits or long sessions, but that is different from lowering the subscription fee.

### Does Entroly work with local models?

Yes. Entroly can be used with Ollama, LM Studio, and custom local-model applications through supported SDK, proxy, or agent integrations.

### Does Entroly send my files to its own cloud?

Local indexing, selection, receipts, and recovery storage do not require an Entroly cloud service. The selected prompt still goes to the model provider chosen by the user. No outbound analytics are enabled by default.

### Can Entroly guarantee that an answer is correct?

No. WITNESS checks support against supplied evidence; it does not establish universal truth. Tests, domain review, and appropriate professional oversight still matter.

### Is there a one-click version for nontechnical users?

Not yet. [Entroly Simple Mode](docs/product/entroly-simple-mode.md) defines the desktop experience and release gates required before “one-click” can be advertised honestly.

---

## Project links

- [Documentation](https://juyterman1000.github.io/entroly/docs/index.html)
- [Agent integrations](https://juyterman1000.github.io/entroly/docs/agent-integrations.html)
- [PyPI](https://pypi.org/project/entroly/)
- [npm runtime](https://www.npmjs.com/package/entroly)
- [npm MCP bridge](https://www.npmjs.com/package/entroly-mcp)
- [Public evidence policy](docs/public-evidence.md)
- [Security and team deployment](docs/for-teams.md)
- [Known limitations](docs/limitations.md)
- [Discord community](https://juyterman1000.github.io/entroly/docs/discord.html)

## License

Apache-2.0. See [LICENSE](LICENSE).

---

<p align="center"><b>Keep your AI tool. Add Context Assurance.</b></p>
