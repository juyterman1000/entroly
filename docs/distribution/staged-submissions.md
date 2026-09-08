# Staged submissions

Status: **prepared, not submitted.** Nothing here has been sent.

Every entry below is paste-ready against the target's observed format. Formats
were read from each list's live `README.md` on 2026-09-07 — re-check before
submitting, because list maintainers reformat without notice.

## Rules that bind every entry

From `marketing/POSITIONING.md` and `marketing/README.md`:

- **No savings percentage, in any language.** The figure is bounded by the
  configured token budget before selection runs.
- No "zero accuracy loss" or equivalent. Verification fails closed by design.
- Disclose maintainer affiliation in the PR or issue body.
- Do not quote stars, downloads, or rankings unless sourced and dated.
- Record a public URL in `targets.json` **only after** the submission is live.

One PR per list. Do not batch, and do not open all eight in one session — a
visible wave of self-submissions from one account reads as spam to maintainers
who talk to each other.

---

## 1. awesome-claude-code

Repo: https://github.com/jqueryscript/awesome-claude-code
Mechanism: pull request

**Observed format:** `- [**name**](url) - (stars ⭐) - Description.`
The star count is maintainer-maintained; omit it and let them add it.

```markdown
- [**entroly**](https://github.com/juyterman1000/entroly) - Context selection that emits a receipt naming what it omitted, with content-addressed handles that recover the exact original bytes. Installs as a Claude Code plugin.
```

---

## 2. awesome-claude-plugins

Repo: https://github.com/quemsah/awesome-claude-plugins
Mechanism: pull request

**Observed format:** a ranked table —
`| rank | [name](url) | Description | <metric> | <metric> | <metric> |`
The rank and the three numeric columns are computed by the maintainer. Supply
name, URL, and description only, and say so in the PR body.

```markdown
| [entroly](https://github.com/juyterman1000/entroly) | Context selection with auditable receipts: every omitted fragment keeps a content-addressed handle that recovers its exact original bytes. |
```

**This is the strongest of the eight**, because the plugin is genuinely
installable as of 2026-09-07 — `/plugin marketplace add juyterman1000/entroly`
then `/plugin install entroly@entroly`. Put that in the PR body as the
verification surface.

---

## 3. awesome-claude (MCP catalog)

Repo: https://github.com/JSONbored/awesome-claude
Mechanism: pull request — a dedicated MCP page, per `submission-kit.md`

Verification surface: the server is listed on the official MCP registry as
`io.github.juyterman1000/entroly`; canonical metadata lives in `server.json`.

```markdown
Entroly exposes MCP tools for context selection, recoverable compression,
content-addressed retrieval, and Context Receipts. Every omission keeps a handle
that returns the exact original bytes, so a retrieval cannot silently substitute
a newer or merely similar source. Local-first: selection makes no outbound
calls. Apache-2.0.
```

---

## 4. Awesome-MCP-ZH (Chinese MCP catalog)

Repo: https://github.com/yzfly/Awesome-MCP-ZH
Mechanism: pull request

Derive from canonical facts. **Do not translate a savings guarantee into
Chinese** — that is how a banned claim re-enters through a side door.

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) - 面向 AI 智能体的上下文选择工具。每次选择都会生成回执，记录保留与省略的内容，并为每个被省略的片段提供内容寻址句柄，可精确还原原始字节。本地优先，Apache-2.0。
```

Back-translation for review: "Context selection for AI agents. Every selection
emits a receipt recording what was kept and omitted, and provides a
content-addressed handle for each omitted fragment that recovers the exact
original bytes. Local-first, Apache-2.0." No percentage, no accuracy claim.

---

## 5. awesome-cli-apps

Repo: https://github.com/agarrharr/awesome-cli-apps
Mechanism: pull request

⚠️ **Format unverified.** The README did not resolve on `main` or `master`
during staging. Read the file and match the surrounding section's format before
submitting; this list is strict about placement and alphabetical ordering.

```markdown
- [entroly](https://github.com/juyterman1000/entroly) - Local-first context selection for AI agents, with receipts and exact recovery of omitted evidence.
```

---

## 6. awesome-local-llms

Repo: https://github.com/vince-lam/awesome-local-llms
Mechanism: **issue, not pull request.**

⚠️ `targets.json` records `submission_url` as `/pulls`. That is wrong — the list
is generated automatically from a database, and its README says to suggest a
repo by opening an issue. A PR here would be closed unmerged.

Issue title: `Suggest repo: juyterman1000/entroly`

```markdown
Local-first context selection for AI agents. Indexing, selection, compression,
recovery, and default verification all run locally with no outbound calls.

Stating the boundary explicitly, since it matters for this list: proxy mode
sends the selected prompt to whichever provider the user configures, so Entroly
is not a privacy layer for the model call itself — only for everything before
it.

Disclosure: I maintain the project.
```

---

## 7. agentic-ai-landscape

Repo: https://github.com/antgroup/agentic-ai-landscape
Mechanism: **issue**

Category: context engineering / agent infrastructure. Omit adoption metrics —
none are independently sourced.

```markdown
**Entroly** — https://github.com/juyterman1000/entroly

Context engineering / agent infrastructure. Budgeted evidence selection with
auditable receipts and content-addressed recovery of omitted evidence.
Apache-2.0, local-first. Available as a Claude Code plugin, MCP server, proxy,
CLI, and SDK.

Disclosure: I maintain the project.
```

---

## 8. claude-code-ultimate-guide

Repo: https://github.com/FlorianBruniaux/claude-code-ultimate-guide
Mechanism: pull request

A guide rather than a list, so the entry should teach something rather than
advertise. Lead with the command and what the reader sees.

```markdown
### Entroly — see what your context step discarded

`/plugin marketplace add juyterman1000/entroly` then
`/plugin install entroly@entroly`

Runs a selection against your repository and prints what it kept, what it
omitted, and a content-addressed handle for each omission. Recover any omitted
fragment and diff it against the source — the bytes match, or the receipt is
wrong. Useful when you want to know whether a context step dropped the one
function you needed.
```

---

## Suggested order

1. **The Awesome-LLMOps correction first** (`corrections/awesome-llmops-claim-correction.md`).
   A live false claim is worse than an absent listing, and it is the entry a
   skeptical maintainer of another list will find if they look you up.
2. `awesome-claude-plugins` and `awesome-claude-code` — the plugin is newly real,
   which makes these the two with a genuine reason to exist today.
3. `awesome-claude`, `agentic-ai-landscape`.
4. The rest, spaced out.

Space them over days, not hours.
