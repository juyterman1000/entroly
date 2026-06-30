# Entroly Real-Gain Demo Video

This video should show Entroly as it actually exists in the codebase: a local context-control and trust layer for AI agents, not only a token-savings gadget.

## Core story

**One-liner:** Entroly makes AI coding agents cheaper, safer, and more auditable by controlling what context they see, proving what was included or omitted, and checking whether answers are supported by evidence.

**Narrative arc:**

1. AI agents fail because they see noisy context and provide no audit trail.
2. Entroly indexes the repo locally and decides what context is worth sending.
3. The proxy/MCP layer inserts optimized context into normal agent workflows.
4. Context Receipts show selected chunks, omitted chunks, dependency links, risks, fingerprints, token ratio, and reproducibility hash.
5. WITNESS checks factual claims against evidence and can warn or suppress unsupported claims.
6. RAVS routes to cheaper models only when shadow evidence says it is safe; otherwise it fails closed.
7. The dashboard/value tracker turns the benefit into lifetime tokens saved, cost saved, optimized requests, and confidence.

## What to prove on screen

Do not say "people save 95%" without a live artifact. Show:

- A real repo being indexed.
- A real task query.
- The selected files/chunks.
- The omitted but relevant chunks.
- The receipt/risk summary.
- A before/after token count.
- A WITNESS verification result.
- The value dashboard or tracker output.

Use actual terminal output from the current release. If a command prints different numbers, use the numbers from the recording.

## Demo setup

Use a medium-sized real codebase, preferably Entroly itself or another open-source repo.

```bash
pip install --upgrade entroly
cd /path/to/repo
entroly doctor
entroly simulate
entroly perf
```

For the receipt segment:

```bash
entroly ingest ./docs
entroly select --query "Which code paths affect release reliability?" --budget 8000
entroly receipt .entroly/receipts/<receipt>.json
entroly explain --why-omitted <chunk_id> --receipt .entroly/receipts/<receipt>.json
```

For the proxy segment:

```bash
entroly proxy
```

Then set the AI tool base URL to the local Entroly proxy and ask the same task through the agent.

For the verification segment:

```bash
entroly witness --help
entroly verify-claims
```

Use `verify-claims` for packaged proof. Use `witness` only if the CLI supports the concrete mode you want to show in the installed version.

## 2-minute script

### 0:00-0:08 — Hook: the real pain

**Visual:** Screen recording of a large repo. Scroll file tree quickly. Then show an AI prompt: "Fix the release reliability issue."

**Voiceover:**

> The problem is not that your AI model is too weak. The problem is that it is reading the wrong context, spending money on noise, and giving you no receipt for what it used.

**On-screen text:**

```text
AI coding problem:
Too much context. No audit trail. Hard to trust.
```

### 0:08-0:22 — What Entroly actually is

**Visual:** Simple diagram.

```text
Repo -> Entroly local index -> selected context + receipt -> AI agent
                         \-> WITNESS verifier
                         \-> value tracker / dashboard
```

**Voiceover:**

> Entroly is a local context-control plane for AI agents. It indexes your repo, selects the smallest useful context, records what it omitted, and gives you a receipt before you trust the answer.

**On-screen text:**

```text
Not just compression.
Context control + receipt + verification.
```

### 0:22-0:36 — One-command install and first proof

**Visual:** Terminal.

```bash
pip install entroly
cd /your/repo
entroly simulate
entroly perf
```

**Voiceover:**

> First, you can measure the opportunity without making an LLM call. Entroly simulates token savings locally, then measures optimizer latency.

**On-screen text:**

```text
Local estimate first.
Spend money only after you know the gain.
```

### 0:36-0:58 — Before/after context selection

**Visual:** Show `entroly simulate` or `entroly perf` output. Highlight total context vs selected context.

**Voiceover:**

> Here is the gain people can feel: instead of dumping a giant repo into the prompt, Entroly ranks the repo and selects only the fragments that matter for the task.

**On-screen text:**

```text
Before: noisy repo dump
After: task-conditioned context
```

### 0:58-1:20 — The receipt: the differentiator

**Visual:** Show rendered receipt Markdown or JSON. Zoom on selected_context, omitted_context, dependency_links, risk_summary, compression_ratio, reproducibility_hash.

**Voiceover:**

> This is the part most tools do not give you: a receipt. Entroly shows the chunks it selected, the relevant chunks it omitted, why they were omitted, dependency links, fingerprints, risk summary, token ratio, and a reproducibility hash.

**On-screen text:**

```text
Receipt = what was used + what was omitted + why + remaining risk
```

### 1:20-1:40 — WITNESS: trust the answer, not the vibes

**Visual:** Show WITNESS / `verify-claims` result. Highlight grounded, unsupported, contradicted, unknown, latency.

**Voiceover:**

> Entroly does not just compress the prompt and hope. WITNESS checks whether factual claims are supported by the evidence. Unsupported or unknown claims can be warned or suppressed depending on the profile.

**On-screen text:**

```text
Grounded? Unsupported? Contradicted? Unknown?
```

### 1:40-1:52 — RAVS: cheaper models only when safe

**Visual:** Show a simple decision card.

```text
High-risk task -> original model
Low-risk task + evidence gate passed -> cheaper model
Unknown -> fail closed
```

**Voiceover:**

> If you use model routing, RAVS is conservative. It only routes to cheaper models when shadow evidence says it is safe. For security, auth, payments, or uncertainty, it fails closed and keeps the original model.

**On-screen text:**

```text
Cost saving without blind downgrades.
```

### 1:52-2:05 — Proxy / normal workflow

**Visual:** Start proxy and show local base URL.

```bash
entroly proxy
```

**Voiceover:**

> You do not need to change how you work. Entroly can run as a proxy, MCP server, wrapper, or library. Your coding agent keeps working, but now the context is controlled and auditable.

**On-screen text:**

```text
Proxy. MCP. Wrapper. Library.
```

### 2:05-2:15 — Close: why this matters

**Visual:** Dashboard/value tracker. Show tokens saved, requests optimized, confidence, and cost estimate.

**Voiceover:**

> The real gain is not only fewer tokens. It is fewer wasted tokens, better evidence, safer routing, and a record you can inspect when the answer matters.

**On-screen text:**

```text
Entroly: context control for AI agents.
```

## 30-second short version

**Voiceover:**

> Your AI coding agent is probably reading too much. Entroly fixes that locally. It indexes your repo, selects the context that matters, records what it left out, and gives you a receipt. Then WITNESS checks whether the answer is supported by the evidence. If routing is enabled, RAVS only moves work to cheaper models when shadow data says it is safe; otherwise it fails closed. The gain is not just lower cost. It is cheaper, cleaner, more auditable AI coding.

**On-screen flow:**

```text
Repo -> local index -> selected context -> receipt -> WITNESS -> value dashboard
```

## YouTube title ideas

- Your AI Agent Is Reading Too Much Code
- Entroly: The Missing Receipt Layer for AI Coding Agents
- Stop Dumping Your Whole Repo Into Claude/Codex
- How to Make AI Coding Cheaper and More Auditable

## Thumbnail ideas

1. Split screen: "100 files" crossed out vs "7 files + receipt".
2. Big text: "AI READS TOO MUCH" with Entroly receipt screenshot.
3. Big text: "SHOW ME THE RECEIPT" with selected/omitted chunks.

## Recording checklist

- Use a real repo, not a toy two-file demo.
- Capture terminal output with readable font size.
- Highlight selected vs omitted context.
- Show one receipt artifact in detail.
- Show one WITNESS/verification artifact.
- Show dashboard/value tracker only after the technical proof.
- Avoid claiming universal savings. Say "on this repo/run" unless showing committed benchmark artifacts.
- Be explicit that Entroly's analysis and receipts are local; when proxying to an LLM provider, the selected prompt is still sent to that provider.

## Call to action

```text
pip install entroly
cd your-repo
entroly simulate
```

Then:

```text
If the simulation shows real savings, run Entroly through your agent and inspect the receipt.
```
