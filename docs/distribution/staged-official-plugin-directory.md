# Staged submission: Anthropic community plugin marketplace

Status: **prepared, not submitted.**

## Correction to an earlier version of this file

An earlier draft targeted `claude-plugins-official` and pointed at
`clau.de/plugin-directory-submission`. Both were wrong, and the docs are
explicit about why:

> The official marketplace, `claude-plugins-official`, is curated separately.
> Anthropic decides which plugins to include at its discretion. **There is no
> application process, and the submission form does not add plugins to the
> official marketplace.**

`clau.de/plugin-directory-submission` is a redirect into the plugins
documentation, not a form. There is nothing to submit to the official
marketplace, and no amount of preparation changes that.

**The real target is the community marketplace**, `claude-community` —
`anthropics/claude-plugins-community` — which is where third-party submissions
land after review. Users add it with:

```
/plugin marketplace add anthropics/claude-plugins-community
```

and install from it as `@claude-community`.

Being listed there is still the discovery win. Our own marketplace is only
reachable by someone who already knows the repository exists.

## How to submit

Two in-app forms, and which one applies depends on the account:

| Form | Requires |
|---|---|
| [claude.ai](https://claude.ai/admin-settings/directory/submissions/plugins/new) | A **Team or Enterprise** organization with directory-management access (Owners have it by default) |
| [Console](https://platform.claude.com/plugins/submit) | Nothing extra — **this is the path for individual authors** not in a Team or Enterprise org |

Use the Console form unless the account is on a Team or Enterprise plan.

## Pre-submission validation — done

The docs state the review pipeline runs `claude plugin validate` on every
submission, alongside automated safety screening. Run against a **fresh clone of
`main`**, not the working tree, so it reflects what a reviewer actually fetches:

```
$ claude plugin validate /tmp/entroly-clean --strict
✔ Validation passed
```

`--strict` treats warnings as errors. It passed with none.

Also verified on that fresh clone:

- Every plugin file is tracked — `.claude-plugin/{marketplace,plugin,manifest}.json`,
  `commands/entroly-first-run.md`, `scripts/entroly-plugin-launch.mjs`.
- The launcher on a PATH carrying neither `uvx`, `npx`, nor `entroly` prints one
  actionable line rather than failing silently. That is the path a reviewer on a
  clean machine is most likely to hit.

## What happens after approval

Approved plugins are pinned to a **specific commit SHA** in
`anthropics/claude-plugins-community`, and CI bumps the pin automatically as new
commits land. The public catalog syncs nightly, so there is a delay between
approval and the plugin becoming installable.

To check whether it has landed, search the
[community catalog](https://github.com/anthropics/claude-plugins-community/blob/main/.claude-plugin/marketplace.json)
for `entroly`. **Do not mark this target `published` in `targets.json` until it
appears there** — that is exactly the "prepared status recorded as published"
failure this registry exists to prevent.

## Form content

**Plugin name:** `entroly`

⚠️ **This slug is immutable once published.** Anthropic's documentation is
explicit that a published `name` must never change, because users have it
installed under that slug and renaming breaks their install with
`plugin-not-found`. It matches `.claude-plugin/plugin.json` and
`.claude-plugin/marketplace.json`.

**Repository:** https://github.com/juyterman1000/entroly

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

**Disclosure to volunteer**, because automated safety screening will surface it
and volunteered reads very differently from discovered:

> When the native engine is absent, `entroly/self_heal.py` installs
> `entroly-core` from PyPI before measuring. It is a package install — no code,
> prompts, or telemetry leave the machine — it is skipped when the engine is
> present, and `ENTROLY_NO_SELF_HEAL=1` disables it. It is documented in
> `PRIVACY.md`, `CLAUDE.md`, and the README. Without the native engine,
> selection never reads the query, so any figure reported would be arithmetic on
> the configured token budget rather than a result.

## Rules this copy is bound by

From `marketing/POSITIONING.md`: **no savings percentage**, in the form or
anywhere else — the figure is bounded by the configured token budget before
selection runs. And no "zero accuracy loss": verification fails closed by
design, so an unqualified accuracy claim contradicts the architecture.
