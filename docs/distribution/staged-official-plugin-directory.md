# Staged submission: Anthropic official plugin directory

Status: **prepared, not submitted.**

This is the highest-value distribution target currently open to us, and the one
the marketplace work in PR #429 was actually for.

## Why this one matters more than the awesome lists

Our own marketplace is live — `/plugin marketplace add juyterman1000/entroly`
works today. But it is only reachable by someone who already knows the
repository exists. The official directory carries **291 plugins** and is what
users browse from inside Claude Code without having heard of us.

That is the difference between installable and discoverable, and it is the
entire premise of the marketplace wedge.

## How to submit

**A web form, not a pull request:** https://clau.de/plugin-directory-submission

Submissions are subject to Anthropic quality and security review. There is no
PR to open and no issue to file, so this cannot be automated — someone has to
fill the form.

## Stated requirements, and where we stand

| Requirement | Status | Evidence |
|---|---|---|
| Meets quality and security standards | Believed met | Apache-2.0; full CI matrix (wheels on five Python versions, Rust + Clippy, WASM drift, journey diagnostics on three operating systems); static security and dependency audit in CI |
| Users can verify what MCP servers, files, or software are included | Met | `mcpServers` is declared in plain sight in `.claude-plugin/plugin.json`; the launcher is a readable ~40-line script at `scripts/entroly-plugin-launch.mjs`; whole repository is public |
| Plugin homepage provides more information | Met | https://github.com/juyterman1000/entroly and https://juyterman1000.github.io/entroly/docs/index.html |

**One disclosure to make proactively in the submission**, because a security
reviewer will find it and it is better volunteered than discovered: when the
native engine is absent, `entroly/self_heal.py` installs `entroly-core` from
PyPI before measuring. It is a package install — no code, prompts, or telemetry
leave the machine — it is skipped when the engine is present, and
`ENTROLY_NO_SELF_HEAL=1` disables it. It is documented in `PRIVACY.md`,
`CLAUDE.md`, and the README. The reason it exists: without the native engine,
selection never reads the query at all, so any figure reported would be
arithmetic on the token budget rather than a result.

## Form content

**Plugin name:** `entroly`

⚠️ **This slug is immutable once published.** Anthropic's documentation is
explicit that a published `name` must never change, because users have it
installed under that slug and renaming breaks their install with
`plugin-not-found`. `entroly` matches `.claude-plugin/plugin.json` and
`.claude-plugin/marketplace.json`. Do not submit a different slug, and do not
change ours afterwards.

**Marketplace / source repository:** https://github.com/juyterman1000/entroly

**Homepage:** https://juyterman1000.github.io/entroly/docs/index.html

**Short description:**

> Context selection that emits a receipt naming what it omitted, with
> content-addressed handles that recover the exact original bytes.

**Longer description:**

> Entroly selects the evidence an agent actually needs under a token budget,
> then hands back a receipt: what was kept, what was omitted, and a
> content-addressed handle recovering each omission byte-for-byte. Take
> anything it dropped, recover it, and diff against the source — the bytes
> match, or the receipt is wrong.
>
> The plugin adds an MCP server for context selection and recovery, plus a
> first-run command that runs a selection against the user's own repository and
> shows the omissions alongside the handles that recover them.
>
> Indexing, selection, compression, recovery, and default verification all run
> locally. Proxy mode sends the selected prompt to whichever provider the user
> configures, so it is not a privacy layer for the model call itself.
> Apache-2.0.

**Category:** developer tools / context engineering

## Rules this copy is bound by

From `marketing/POSITIONING.md`: **no savings percentage**, in the form or
anywhere else. The figure is bounded by the configured token budget before
selection runs. Also no "zero accuracy loss" — verification fails closed by
design, and an unqualified accuracy claim contradicts the architecture.

## Before submitting

- [ ] Confirm the plugin installs from a clean machine that has neither `uv`
      nor `entroly` on PATH — this is the fallback path the launcher exists for,
      and the one most likely to be exercised by a reviewer.
- [ ] Confirm `/plugin marketplace add juyterman1000/entroly` then
      `/plugin install entroly@entroly` succeeds end to end.
- [ ] Confirm the first-run command produces a readable receipt on a repository
      the reviewer is likely to try — a small one, not just this repo.
- [ ] Record the submission date and outcome in `targets.json` **after**
      submitting, never before.
