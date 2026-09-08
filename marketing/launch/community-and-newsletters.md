# Community, directory, and newsletter drafts

Status: prepared, not submitted.

Positioning is frozen — see `marketing/POSITIONING.md`. **No savings percentage
in any of this copy**, in any language. The figure is decided by the configured
token budget before selection runs, so it measures the budget rather than the
tool.

Use only in communities where project announcements are allowed. Disclose that
the poster maintains Entroly. Adapt the technical depth to the audience instead
of cross-posting identical text everywhere.

## Developer community post

**Title:** Compression that hands back a receipt: what it dropped, and how to
get it back exactly

Disclosure: I maintain [Entroly](https://github.com/juyterman1000/entroly)
(Apache-2.0).

Every context tool shows you a smaller prompt. None of them tell you what they
threw away.

Entroly emits a receipt for each selection: what was kept, what was omitted, and
a content-addressed handle that recovers the exact original bytes of anything it
dropped. Take any omitted fragment, recover it, diff it against the source. The
bytes match or the receipt is wrong.

No API key needed to check that:

```bash
pip install -U entroly
cd /path/to/repository
entroly simulate
```

It runs through CLI, SDK, MCP, proxy, coding-agent wrappers, Rust, npm/WASM,
Docker, and Homebrew, and installs into Claude Code as a plugin.

The project documents workloads where compression is neutral or harmful, and
deliberately claims no universal reduction percentage. Technical feedback,
failed reproductions, and integration reports are welcome when they include the
version, workload, budget, commands, and raw artifacts.

## Claude Code community post

**Title:** Entroly is now installable as a Claude Code plugin

Disclosure: I maintain Entroly.

```
/plugin marketplace add juyterman1000/entroly
/plugin install entroly@entroly
```

It adds an MCP server for context selection and recovery, plus a first-run
command that runs a selection against your own repository and shows what it
omitted along with the handles that recover each omission exactly.

The launcher resolves `uvx`, then `npx`, then an `entroly` already on PATH, so
it works without a preinstalled `uv`. Everything in the selection path runs
locally.

Repository: https://github.com/juyterman1000/entroly

I would value feedback on first-run ergonomics and on any host where the plugin
fails to start.

## MCP community post

**Title:** Entroly MCP server: recoverable context selection with auditable
receipts

Disclosure: I maintain Entroly.

Entroly exposes MCP tools for context selection, recoverable compression,
content-addressed retrieval, Context Receipts, repository context, and local
evidence-grounding checks. What distinguishes it from a compression server is
that every omission keeps a handle that returns the exact original bytes, so a
retrieval cannot silently substitute a newer or merely similar source.

Canonical package metadata is in `server.json`; it is listed on the official MCP
registry.

Repository: https://github.com/juyterman1000/entroly
MCP guide: https://juyterman1000.github.io/entroly/docs/mcp-server-guide.html
Privacy: https://github.com/juyterman1000/entroly/blob/main/PRIVACY.md

I am looking for host-specific installation feedback, permission-boundary
review, and reproducible reports from Claude Code, Cursor, Copilot, Codex,
OpenCode, OpenClaw, and other MCP clients.

## Newsletter pitch

**Subject:** Context compression that can prove what it discarded

Entroly is an Apache-2.0, local-first context layer for AI agents. The angle
that distinguishes it from a compression demo: every selection emits a receipt
naming what was omitted, and every omission keeps a content-addressed handle
that recovers the exact original bytes. Readers can falsify that claim on their
own repository in one command, without an API key.

A second angle your readers may find more interesting than the product: the
project treats its own savings percentage as unpublishable. The figure is
computed against a baseline capped at the configured token budget, so it is
bounded before selection runs and describes the budget rather than the selector.
The repository bans the number in CI and documents why.

```bash
pip install -U entroly
entroly verify-claims
```

Repository: https://github.com/juyterman1000/entroly
Methodology: https://github.com/juyterman1000/entroly/blob/main/BENCHMARKS.md
Limitations: https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md

Disclosure: this pitch comes from the Entroly maintainer. Independent testing
and critical coverage are preferred over repeating project claims.

## Podcast or technical interview pitch

Possible discussion topics:

- Why a compression ratio cannot establish context quality, and what it hides.
- Designing exact recovery so an agent cannot retrieve a similar but different
  source revision.
- What a Context Receipt should record, and the three different things usually
  reported as one: what was selected, what stayed recoverable, and what was
  actually sent to a provider.
- Why we ban our own savings percentage in CI, and how a metric can be pinned by
  its baseline before any work happens.
- Benchmark null controls, and why a task solvable with no context measures
  nothing.
- Privacy boundaries for local selection versus cloud-model proxying.
- Keeping Python, Rust, WASM, npm, MCP, plugin, Docker, and Homebrew surfaces in
  release sync — and treating channel presence as a release surface that fails
  loudly.

Do not promise exclusive benchmark results or leadership claims before the host
has access to the raw evidence.

## Directory description — 160 characters

Open-source context selection for AI agents. Every selection emits a receipt
naming what was omitted, with handles that recover the exact original bytes.

## Directory description — 300 characters

Entroly is a local-first context layer for AI agents. It selects evidence under
a token budget and emits a receipt naming what was kept, what was omitted, and a
content-addressed handle recovering each omission byte-for-byte. Runs as a
Claude Code plugin, MCP server, proxy, CLI, or SDK.

## Alternative-product directory description

Entroly is an open-source context layer for coding agents and LLM applications.
It combines budgeted evidence selection, recoverable compression,
content-addressed retrieval, Context Receipts, local verification, and
provider-aware accounting. What it adds over compression-only tools is
falsifiability: any omitted fragment can be recovered byte-for-byte through its
handle. Best evaluated on your own workload — short prompts and tasks already
within budget pass through unchanged.

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

**Do not treat an email sent, draft saved, or private message delivered as a
published mention.** `docs/distribution/targets.json` records a target as
published only once a public URL exists.
