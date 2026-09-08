# New Target PR Strategy — Evidence-First, Rules-Read

## Real Evidence from verify-claims (2026-08-09)

```
$ pip install entroly && entroly verify-claims
→ 12/12 checks passed
  [PASS] SDK import — compress() returns text, compress_messages() returns messages
  [PASS] Local indexing — 160 files, 717,707 tokens, 13.7s
  [PASS] Context optimization — 24 fragments, 7,590/8,000 tokens, 6.1ms
  [PASS] Exact recovery — ccr:811e14e88963b07f71a564a1 restores byte-exact content
  [PASS] Engine mode — Python fallback, no API key required

$ entroly simulate
→ Average reduction: 87.9% (local estimate, not billing guarantee)
  Indexed 140 files, 620,162 estimated tokens
  "Auth flow?" → 3,877 tokens (87.9% fewer vs 32k baseline)
  Caveat: No LLM call made; quality not judged; provider cache excluded
```

Honest limitations committed in code:
- SQuAD: 80% → 72% at 43.8% savings (benchmarks/squad/ artifact)
- Savings are workload-dependent
- Small repos pass through unchanged

---

## Target 1: awesome-mcp-servers (40k⭐)

**RULE READ:** CONTRIBUTING.md says add `🤖` to PR title for agent fast-track merge.
**Format:** `- [owner/repo](url) [glama badge] [emoji flags] - Description.`
**Section:** Alphabetical under Server Implementations
**Position:** Between `ejwhite7/brandkit-mcp` and `erajasekar/ai-diagram-maker-mcp`

**PR Title:** `Add juyterman1000/entroly MCP server 🤖`

**Entry line:**
```markdown
- [juyterman1000/entroly](https://github.com/juyterman1000/entroly) 🏠 🐍 🦀 ☁️ 🍎 - Local-first context-control plane and MCP server for AI coding agents. Selects evidence under an explicit token budget, keeps omitted content byte-exactly recoverable via content-addressed handles (CCR), and emits auditable Context Receipts. Works as stdio MCP server, HTTP proxy, CLI, or Python/Rust SDK. Verify locally: `entroly verify-claims` (12/12 checks pass, no API key). Apache-2.0.
```

**PR Body:**
```markdown
## Adding juyterman1000/entroly MCP server 🤖

**Category:** Server Implementations (alphabetical: between ejwhite7/brandkit-mcp and erajasekar/ai-diagram-maker-mcp)

### What it does
Entroly is a local-first context-control plane that wraps any MCP session to:
- Select evidence under an explicit token budget (BM25 + dependency graph scoring)
- Keep all omitted content byte-exactly recoverable via content-addressed CCR handles
- Emit auditable Context Receipts showing what was included/excluded and why
- Verify model claims against supplied evidence (WITNESS, zero LLM cost)

### Verification (reproducible, no API key needed)
```bash
pip install entroly
entroly verify-claims
```
Result: **12/12 checks passed** (SDK import, local indexing of 160 files, context optimization at 6.1ms, exact byte recovery, Python fallback engine).

### Honest limitations
- Savings are workload-dependent — `entroly simulate` measures your repo locally before connecting a paid model
- Compression can reduce answer quality (SQuAD: 80% → 72% at 43.8% savings, committed artifact)
- Small codebases that fit the context window pass through unchanged
- Full limitations: [docs/limitations.md](https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md)

### Checklist
- [x] Entry follows existing format (name, glama badge, emoji flags, description)
- [x] Entry is in alphabetical order
- [x] One server per line
- [x] Description is concise and accurate
- [x] Links verified

Disclosure: I maintain Entroly.
```

---

## Target 2: anthropic-cookbook (Official Anthropic, 11k⭐)

**RULE READ:** Requires runnable Jupyter notebooks tested via nbconvert + ruff. Needs real Claude API key in CI. High bar — needs a WORKING notebook, not just a doc.

**Action:** Create `misc/context_optimization_with_entroly/context_optimization.ipynb`

**PR Title:** `Add context optimization with Entroly notebook`

**Content:**
- Working notebook demonstrating Entroly + Claude API
- Shows Context Receipts, verify-claims output, honest cost comparison
- Uses `pip install anthropic entroly` setup cell
- Includes real benchmark caveat in markdown cell
- Tested locally

---

## Target 3: simonw/llm (Plugin ecosystem, 6k⭐)

**RULE READ:** Plugin directory uses `**[plugin-name](url)**` bold format with description of what it does in one sentence.

**Entroly would fit as:** An LLM plugin that pre-compresses prompts before sending to any LLM.

**But:** Current Entroly is not packaged as an `llm-*` plugin. Would need:
```bash
llm install llm-entroly
llm -m claude-3-5-sonnet "question about big codebase" --entroly
```

**Action:** Not ready yet. Needs an `llm-entroly` plugin package first.

---

## Target 4: mckaywrigley/takeoff-ai / open-interpreter / Aider

**aider:** 35k⭐, AI pair programming CLI that manages context explicitly.
**Fit:** Entroly = transparent proxy that reduces Aider's context costs.
**Rules to check:** https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md

**Entry type:** Integration guide in their docs, not a listing PR.

---

## Target 5: awesome-claude-code (Claude Code specific, growing fast)

**URL:** https://github.com/anthropics/anthropic-cookbook (in Claude Code section)
**Also:** https://github.com/hesreallyhim/awesome-claude-code

**Format:** Needs checking.
**Fit:** Entroly MCP server integrates directly with Claude Code via `.mcp.json`.

---

## Immediate Next Action

**Submit awesome-mcp-servers PR** — Fast-track with 🤖 in title, follows exact CONTRIBUTING.md rules.
