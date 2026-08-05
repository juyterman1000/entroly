# Community, directory, and newsletter drafts

Status: prepared, not submitted.

Use only in communities where project announcements are allowed. Disclose that
the poster maintains Entroly. Adapt the technical depth to the audience instead
of cross-posting identical text everywhere.

## Developer community post

**Title:** Open-source context assurance for coding agents: recoverable selection,
receipts, and local verification

I maintain [Entroly](https://github.com/juyterman1000/entroly), an Apache-2.0
context-control layer for AI agents.

It selects evidence under a token budget, keeps omitted originals recoverable by
content-addressed handle, emits Context Receipts, and checks model claims against
the evidence supplied. It can run through CLI, SDK, MCP, proxy, coding-agent
wrapper, Rust, npm/WASM, Docker, or Homebrew paths.

The no-key local test is:

```bash
pip install -U entroly
cd /path/to/repository
entroly verify-claims
entroly simulate
```

The project explicitly documents workloads where compression can be neutral or
harmful and does not claim a universal reduction percentage. Technical feedback,
failed reproductions, and integration reports are welcome when they include the
version, workload, budget, commands, and raw artifacts.

## MCP community post

**Title:** Entroly MCP server for recoverable context selection and evidence checks

Disclosure: I maintain Entroly.

Entroly exposes MCP tools for context selection, recoverable compression,
content-addressed retrieval, Context Receipts, repository context, and local
evidence-grounding checks. The canonical MCP package metadata is in
`server.json`, and the repository documents local processing and the proxy
provider boundary.

Repository: https://github.com/juyterman1000/entroly
MCP guide: https://juyterman1000.github.io/entroly/docs/mcp-server-guide.html
Privacy: https://github.com/juyterman1000/entroly/blob/main/PRIVACY.md

I am looking for host-specific installation feedback, permission-boundary review,
and reproducible reports from Claude Code, Cursor, Copilot, Codex, OpenCode,
OpenClaw, and other MCP clients.

## Newsletter pitch

**Subject:** Open-source context assurance with exact recovery and auditable receipts

Entroly is an Apache-2.0 context-assurance layer for AI agents. Unlike a
compression-only demo, it combines budget-aware evidence selection,
content-addressed recovery of omitted source, Context Receipts, and local
claim-to-evidence verification.

A useful technical angle for your readers is the project's falsification-first
benchmark policy: context evaluations include control checks for tasks solvable
without context, report task quality alongside token effects, preserve raw
artifacts, and avoid universal savings claims.

Readers can reproduce the local installation and receipt/recovery checks without
an API key:

```bash
pip install -U entroly
entroly verify-claims
```

Repository: https://github.com/juyterman1000/entroly
Methodology: https://github.com/juyterman1000/entroly/blob/main/docs/BENCHMARKS.md
Limitations: https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md

Disclosure: this pitch comes from the Entroly maintainer. Independent testing and
critical coverage are preferred over repeating project claims.

## Podcast or technical interview pitch

Possible discussion topics:

- Why token reduction is not enough to establish context quality.
- Designing exact recovery so an agent cannot retrieve a similar but different
  source revision.
- What a Context Receipt should record.
- How to measure provider-bound input without confusing it with local estimates.
- Benchmark null controls and the failure of context-free tasks.
- Privacy boundaries for local selection versus cloud-model proxying.
- Keeping Python, Rust, WASM, npm, MCP, plugin, Docker, and Homebrew surfaces in
  release sync.

Do not promise exclusive benchmark results or leadership claims before the host
has access to the raw evidence.

## Directory description — 160 characters

Open-source Context Assurance for AI agents: budgeted evidence selection,
recoverable compression, Context Receipts, MCP, proxy, SDK, and verification.

## Directory description — 300 characters

Entroly is a local-first Context Assurance layer for AI agents. It selects
evidence under a token budget, keeps omitted source recoverable, emits auditable
Context Receipts, and verifies claims against supplied evidence through CLI,
SDK, MCP, proxy, wrappers, Rust, npm, and Docker.

## Alternative-product directory description

Entroly is an open-source context-control layer for coding agents and LLM
applications. It combines evidence selection, recoverable compression,
content-addressed retrieval, Context Receipts, local verification, and
provider-aware accounting. It is best evaluated on the user's own workload;
short prompts and tasks already within budget may pass through unchanged.

## Submission log requirements

For every external post or pitch, record:

- channel and audience;
- submission date;
- exact public URL when available;
- Entroly version referenced;
- claims or metrics included;
- outcome: published, rejected, removed, or unanswered;
- corrections requested by the external maintainer;
- referral or reproduction evidence, when legitimately available.

Do not treat an email sent, draft saved, or private message delivered as a
published mention.
