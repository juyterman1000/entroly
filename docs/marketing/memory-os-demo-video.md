# Entroly Memory OS Demo Video

Purpose: make people understand why Entroly's memory ecosystem is valuable in under two minutes.

Do not frame this as a generic memory database. Frame it as a runtime memory-control system for AI agents.

## Core hook

```text
Most AI memory tools remember more.
Entroly decides what is worth remembering, recalling, sharing, suppressing, verifying, and sending to the model.
```

## 90-second script

### 0:00-0:07 — Pain

**Visual:** Agent chat with repeated context, copied files, and noisy memory snippets.

**Voiceover:**

> AI memory is not useful if it just gives your agent more stuff to read. More memory can mean more noise, more tokens, and more hallucinations.

**On-screen text:**

```text
Memory problem:
More context ≠ better context
```

### 0:07-0:18 — Entroly positioning

**Visual:** Diagram.

```text
Working memory → Episodic memory → Semantic memory
       ↓              ↓                 ↓
   budgeted recall + receipts + WITNESS verification
```

**Voiceover:**

> Entroly Memory OS gives agents a working memory, long-term memory, verifier, and safe multi-agent nervous system. It controls memory under a token budget instead of blindly retrieving everything.

**On-screen text:**

```text
Entroly Memory OS
Budget-aware memory control for AI agents
```

### 0:18-0:34 — Remembering is selective

**Visual:** Show selected code fragments and memory entries.

**Voiceover:**

> Entroly remembers high-value evidence: selected code, critical files, high-entropy fragments, and repeated successful context. Low-value noise decays.

**On-screen text:**

```text
Remember high-value evidence
Let weak memory fade
```

### 0:34-0:50 — Recall is budget-aware

**Visual:** Show a small context budget and a selected set of memories.

**Voiceover:**

> Recall is not a nearest-neighbor dump. Entroly ranks memories by relevance, salience, retention, and token cost, then fits them into the model's budget.

**On-screen text:**

```text
Recall = relevance + salience + retention + cost
```

### 0:50-1:04 — Sleep replay and consolidation

**Visual:** Animation: working memory promotes to episodic, episodic promotes to semantic, weak memories disappear.

**Voiceover:**

> Important memories consolidate. Weak memories disappear. Semantic memories survive. This is closer to how an agent should learn over time.

**On-screen text:**

```text
Decay → consolidate → promote → retain
```

### 1:04-1:18 — Multi-agent safety

**Visual:** Multi-agent diagram with repeated messages suppressed and unsafe message blocked.

**Voiceover:**

> For multi-agent systems, Entroly filters redundant messages before they explode the token budget. The compliance gate blocks PII, injection payloads, and rate-limit abuse before memory is shared.

**On-screen text:**

```text
Suppress redundancy
Block unsafe memory traffic
```

### 1:18-1:32 — Verification

**Visual:** Show Context Receipt + WITNESS result.

**Voiceover:**

> Then Entroly verifies. Context Receipts show what was selected and omitted. WITNESS checks whether the answer is grounded in the evidence.

**On-screen text:**

```text
Receipt + WITNESS = auditable memory
```

### 1:32-1:40 — Comparison

**Visual:** Split card.

```text
Graph memory: company brain
Entroly: agent memory runtime
```

**Voiceover:**

> Graph-memory tools build a company brain. Entroly controls the agent's working memory and context budget at runtime.

### 1:40-1:55 — Close

**Visual:** Terminal.

```bash
pip install entroly[full]
entroly simulate
entroly proxy --quality balanced
```

**Voiceover:**

> If your AI agent reads too much, forgets what matters, repeats itself, or gives answers without evidence, Entroly is the memory-control layer it is missing.

**On-screen text:**

```text
Entroly Memory OS
Remember less noise. Recall better evidence.
```

## 30-second short

**Voiceover:**

> Most AI memory tools remember more. Entroly decides what is actually worth remembering, recalling, sharing, suppressing, verifying, and sending to the model. It gives agents working, episodic, and semantic memory under a token budget. Weak memories decay. Important memories consolidate. Redundant agent messages get suppressed. Unsafe memory traffic is blocked. Context Receipts and WITNESS make the final answer auditable. Entroly is not just memory storage. It is memory control for AI agents.

**On-screen flow:**

```text
Memory → budgeted recall → receipt → WITNESS → safer agent answer
```

## Demo structure

### Demo A — Why generic memory fails

Show a prompt like:

```text
We fixed a login timeout last week. Why is it happening again?
```

Then show the bad path:

```text
Generic memory returns too many old notes.
Agent reads stale context.
Answer is plausible but unsupported.
```

### Demo B — Entroly path

Show the Entroly path:

```text
1. recall relevant high-salience memories
2. select current repo code under budget
3. omit low-retention/stale noise
4. produce receipt
5. verify answer with WITNESS
```

### Demo C — Multi-agent path

Show three agents:

```text
Planner → Coder → Reviewer
```

Then demonstrate:

```text
Repeated tool result: suppressed
PII-containing memory: blocked
Novel high-value lesson: delivered
Successful shared lesson: pollination reward increases future share probability
```

## Recording checklist

- Use a real repo, not a toy example.
- Show selected and omitted evidence, not just a final answer.
- Show at least one stale/low-value memory being ignored.
- Show at least one receipt.
- Show one WITNESS result.
- Show one multi-agent suppression or safety gate example if available.
- Do not claim universal superiority over graph memory tools.
- Say the exact positioning: graph memory builds a company brain; Entroly controls agent memory at runtime.

## Thumbnail ideas

1. Big text: `AI MEMORY IS TOO NOISY`
2. Split: `Remember Everything` crossed out vs `Recall What Matters`
3. Big text: `MEMORY NEEDS A RECEIPT`
4. Visual: brain + firewall + receipt

## Title ideas

- Your AI Agent Remembers Too Much
- Entroly Memory OS: Memory Control for AI Agents
- Why AI Memory Needs Receipts
- Stop Giving Agents Noisy Memory
- Working Memory, Long-Term Memory, and Verification for AI Agents

## Product page section

Use this as a short README/product-page block:

```markdown
### Entroly Memory OS

Most AI memory systems focus on storing more. Entroly focuses on controlling what memory reaches the model.

Entroly Memory OS gives agents:

- working, episodic, and semantic memory,
- budget-aware recall,
- salience and forgetting,
- sleep-replay consolidation,
- redundant-message suppression,
- PII and prompt-injection gates,
- context receipts,
- WITNESS verification,
- opt-in privacy-preserving federation.

Graph-memory tools build a company brain. Entroly controls the agent's memory at runtime.
```

## Claims to avoid

Avoid:

```text
Entroly is better than every memory platform.
```

Use:

```text
Entroly solves a different layer: runtime memory control for agents.
```

Avoid:

```text
Entroly sends no data anywhere.
```

Use:

```text
Entroly indexing and selection are local. If you use proxy mode with an LLM provider, the selected prompt is sent to that provider.
```

Avoid:

```text
Memory is always active.
```

Use:

```text
Core context memory is part of Entroly. Long-term memory surfaces depend on the installed runtime and should be verified with `entroly doctor` or native-status checks.
```
