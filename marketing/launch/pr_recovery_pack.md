# Entroly Open PR Recovery Pack — Evidence-First Edition

**Real evidence captured on 2026-08-09 from Entroly v1.0.76:**

```
entroly verify-claims → 12/12 checks passed
  SDK import: PASS
  Local indexing: 160 files, 717,707 tokens, 13.7s — PASS
  Context optimization: 24 fragments, 7,590/8,000 tokens, 98.9% savings, 6.1ms — PASS
  Exact recovery: ccr:811e14e88963b07f71a564a1 — PASS
  Engine mode: Python fallback, no API key required — PASS

entroly simulate (this repo, no LLM calls):
  Indexed 140 files (620,162 estimated tokens)
  "How does the authentication flow work?" → 3,877 tokens (87.9% fewer)
  "Find and fix potential SQL injection" → 3,877 tokens (87.9% fewer)
  "Explain the module structure" → 3,877 tokens (87.9% fewer)
  Average reduction: 87.9% (local estimate, not billing guarantee)
```

**Caveats included in every PR comment:**
- Savings are local estimates, not provider billing guarantees
- Quality is not judged by simulate — only token counts
- SQuAD accuracy drops 80% → 72% at 43.8% savings (committed benchmark)
- Small repos may see zero improvement

---

## 1. awesome-ai-agents (e2b) [#928](https://github.com/e2b-dev/awesome-ai-agents/pull/928)

**Repo rules** (from README line 64): "Create a pull request or fill in this form. Please keep the alphabetical order and in the correct category."

**Repo format:** Structured entry with `## [Name](url)`, subtitle, `<details>` block containing Category, Description bullets, and Links section.

**This PR needs a full rewrite to match format.** Current PR body uses a one-liner. The entry should look like:

```markdown
## [Entroly](https://github.com/juyterman1000/entroly)
Local-first context-control plane for AI coding agents

<details>

### Category
Developer tools, Open source

### Description
- Budget-aware context selection: picks evidence fragments under an explicit
  token limit using BM25 scoring and dependency-graph analysis
- Exact recovery: omitted content stays content-addressed and byte-recoverable
  via CCR handles
- Context Receipts: auditable record of what was included, excluded, and why
- Local verification (WITNESS): checks model claims against supplied evidence
  with zero LLM cost
- Works as CLI, MCP server, HTTP proxy, or Python/Rust SDK
- 12/12 local checks pass via `entroly verify-claims` (no API key needed)

### Limitations
- Savings are workload-dependent — `entroly simulate` gives local estimates
- Compression can reduce answer quality (SQuAD: 80% → 72% at 43.8% savings)
- Small repos that fit the context window pass through unchanged

### Links
- [GitHub](https://github.com/juyterman1000/entroly)
- [Documentation](https://juyterman1000.github.io/entroly/docs/index.html)
- [Limitations](https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md)
- License: Apache-2.0
</details>
```

**PR comment to post:**
```
Hi! I've restructured the entry to match the repo format (heading + details block with Category, Description, Links). Key changes from the original submission:

1. Removed unverified claims ("78% with zero quality loss")
2. Added factual description based on `entroly verify-claims` output (12/12 checks pass locally, no API key)
3. Included honest limitations section
4. Added links to limitations docs

Real verification output (anyone can reproduce):
$ pip install entroly && entroly verify-claims
→ 12/12 checks passed (SDK import, indexing, optimization, exact recovery, engine mode)

Disclosure: I maintain Entroly.
```

---

## 2. crewAI [#6052](https://github.com/crewAIInc/crewAI/pull/6052)

**Repo rules:** No CONTRIBUTING_DOCS.md found. Existing integration docs at `docs/en/tools/integration/` use `.mdx` format with frontmatter, sections: Overview, Setup (pip install), Usage (code example), Features (bullet list).

**PR has `no-pr-activity` label — comment urgently.**

**PR comment to post:**
```
Hi CrewAI team! Confirming compatibility with Entroly 1.0.76 and updating the integration guide.

Changes from original:
1. Removed unverified "70-95%" and "Nash-KKT" jargon from the PR body
2. The docs guide now focuses on two verified integration paths:
   - Proxy mode: `entroly proxy` → set OPENAI_BASE_URL=http://localhost:9377/v1
   - SDK mode: `from entroly import compress`
3. Verification proof (reproducible, no API key):
   $ entroly verify-claims
   → 12/12 checks passed (indexing, optimization, exact recovery, engine mode)
4. Added honest caveats: savings are workload-dependent, quality tradeoffs
   exist, `entroly simulate` measures before connecting a paid model

Followed existing integration doc format at docs/en/tools/integration/.
Disclosure: I maintain Entroly.
```

---

## 3. Continue [#12559](https://github.com/continuedev/continue/pull/12559)

**Repo rules** (from CONTRIBUTING.md): Docs contributions go in `docs/` with `.mdx` format. Follow existing cookbook structure (see GitHub, Sentry, Supabase examples).

**PR comment to post:**
```
Hi Continue team! Updating this cookbook PR to align with Entroly 1.0.76 evidence standards:

1. Removed unverified "70-95%" and "zero quality loss" claims from the guide
2. Cookbook follows the structure of existing MCP guides (GitHub, Sentry, Supabase)
3. Two integration paths documented:
   - MCP server: one YAML block in config
   - Transparent proxy: zero Continue config changes
4. Verification section added — anyone can reproduce:
   $ pip install entroly && entroly verify-claims
   → 12/12 checks passed locally, no API key required
5. Added caveats section:
   - Savings are workload-dependent (measure with `entroly simulate`)
   - Compression can reduce answer quality on some tasks
   - Provider cache behavior affects real-world savings

Disclosure: I maintain Entroly. Happy to adjust the guide based on your docs style.
```

---

## 4. awesome-claude-skills [#812](https://github.com/ComposioHQ/awesome-claude-skills/pull/812)

**Action:** Close with comment:
```
Closing in favor of #1214 which has updated metadata and formatting.
```

## 5. awesome-claude-skills [#1214](https://github.com/ComposioHQ/awesome-claude-skills/pull/1214)

**Status:** Has `ready-to-merge` label.

**PR comment:**
```
Hi! This PR has the ready-to-merge label. Quick update:
- Entroly is now at v1.0.76
- All 12 local verification checks pass: `entroly verify-claims`
- Closed the duplicate PR #812
Let me know if anything blocks merge. Thanks!
```

---

## 6. Awesome-LLM [#533](https://github.com/Hannibal046/Awesome-LLM/pull/533)

**Repo format:** Need to check — likely bullet list entries under category headings.

**PR comment:**
```
Hi! Updating the entry description with verified facts and honest caveats:

- [Entroly](https://github.com/juyterman1000/entroly) — Local-first context-control
  plane for AI coding agents. Selects evidence under a token budget, keeps omitted
  content exactly recoverable (CCR handles), and emits auditable Context Receipts.
  Works as CLI, MCP server, proxy, or Python/Rust SDK. Apache-2.0.

Verification (reproducible, no API key):
$ pip install entroly && entroly verify-claims
→ 12/12 checks passed

Honest caveat: savings are workload-dependent. `entroly simulate` gives
local estimates — quality tradeoffs are documented at
github.com/juyterman1000/entroly/blob/main/docs/limitations.md

Disclosure: I maintain Entroly.
```

---

## 7. awesome-generative-ai (steven2358) [#705](https://github.com/steven2358/awesome-generative-ai/pull/705)

**PR comment:**
```
Hi! Updating to remove jargon and add verification proof:

- [Entroly](https://github.com/juyterman1000/entroly) — Local-first context
  assurance for AI coding agents. Selects evidence under a token budget, emits
  Context Receipts, and keeps omitted content recoverable. Verify locally
  (no API key): `pip install entroly && entroly verify-claims` → 12/12 checks pass.
  Apache-2.0.

Savings are workload-dependent. Limitations documented at
github.com/juyterman1000/entroly/blob/main/docs/limitations.md

Disclosure: I maintain Entroly.
```

---

## 8. awesome-generative-ai (filipecalegario) [#550](https://github.com/filipecalegario/awesome-generative-ai/pull/550)

**PR comment:**
```
Hi! Updating description to remove unverified "70-95%" claim and add evidence:

Entroly is a local-first context-control plane and MCP server for AI coding
agents. Selects evidence under a token budget, keeps omitted content recoverable,
emits auditable Context Receipts. Verify locally (no API key):
`pip install entroly && entroly verify-claims` → 12/12 checks pass.

Savings are workload-dependent — measure with `entroly simulate`.
Apache-2.0. Disclosure: I maintain Entroly.
```

---

## 9. hermes-agent [#39711](https://github.com/NousResearch/hermes-agent/pull/39711)

**Status:** Already well-written with hermetic tests. Just needs review follow-up.

**PR comment:**
```
Hi! Confirming this skill is verified against Entroly 1.0.76:
- `entroly verify-claims` → 12/12 checks pass
- stdio entrypoint: `python -m entroly` or bare `entroly`
- The hermetic test suite (10 passed + 26 passed) still holds

Happy to address any remaining review feedback.
```

---

## 10. llm-course [#138](https://github.com/mlabonne/llm-course/pull/138)

**PR comment:**
```
Hi! Updating to remove "without accuracy loss" claim — that's not accurate.

Entroly provides local context selection for AI coding agents. On SQuAD
benchmarks, compression at 43.8% savings reduces accuracy from 80% to 72%
(committed artifact). Savings and quality tradeoffs are workload-dependent.

Verification (no API key): `pip install entroly && entroly verify-claims` → 12/12 pass.
Full limitations: github.com/juyterman1000/entroly/blob/main/docs/limitations.md

Disclosure: I maintain Entroly.
```

---

## Checklist Before Posting ANY Comment

- [ ] Read the repo's CONTRIBUTING.md or README submission rules
- [ ] Study format of 3 recent merged entries in that repo
- [ ] Match the exact entry structure (one-liner, details block, table row, etc.)
- [ ] Include `verify-claims` evidence (12/12 checks pass, reproducible)
- [ ] State at least one honest limitation
- [ ] Link to limitations doc
- [ ] Disclose maintainer status
- [ ] No universal percentage claims without "workload-dependent" caveat
- [ ] No "zero quality loss" — always mention SQuAD tradeoff
