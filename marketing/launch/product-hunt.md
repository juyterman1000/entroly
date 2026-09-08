# Product Hunt launch draft

Status: prepared, not submitted.

Positioning is frozen — see `marketing/POSITIONING.md`. Do not add a savings
percentage to this draft. The figure is decided by the configured token budget
before selection runs, so it measures the budget rather than the tool.

## Product name

Entroly

## Tagline

Cut AI context cost and prove nothing was lost

## Short description

Entroly selects the evidence your AI agent actually needs, then hands back a
receipt: what was kept, what was omitted, and a content-addressed handle that
recovers the exact original bytes of anything it dropped. Open-source,
local-first, and checkable on your own repository in one command. Works as a
Claude Code plugin, MCP server, proxy, CLI, or SDK.

## Maker comment

Every context tool optimises the same visible number: fewer tokens. That number
is easy to show and it answers the wrong question. It cannot tell you whether
the one function you needed was the thing that got dropped.

So Entroly's contract is narrower and checkable. Each selection emits a receipt
naming what it kept, what it omitted, and the handle that recovers each omitted
fragment byte-for-byte. Pick anything it dropped, recover it, compare. Either
the bytes match or the receipt is wrong and you have caught me.

You can test that claim before connecting a paid model, and without an API key:

```bash
pip install -U entroly
cd /path/to/your/repository
entroly simulate
```

Or, inside Claude Code:

```
/plugin marketplace add juyterman1000/entroly
/plugin install entroly@entroly
```

Current measurements, both linking to raw artifacts in the repo: 5,117/5,117
native source fragments verified byte-exact, and 13/13 public SDK recovery
probes matching their source spans.

**The honest limitations.** Compression is workload-dependent — it can reduce
useful context or add overhead on the wrong task, and it passes through
untouched when context already fits the budget. There is no universal savings
number here, deliberately: any percentage is bounded by the token budget you
configure before selection runs, so it would describe your budget rather than
Entroly. And proxy mode still sends the selected prompt to whichever provider
you configured, so it is not a privacy layer.

Apache-2.0. Indexing, selection, compression, recovery, and default verification
all run locally.

Repository: https://github.com/juyterman1000/entroly
Documentation: https://juyterman1000.github.io/entroly/docs/index.html
Limitations: https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md

## First comment

The fastest way to judge this is to make it fail on your own code.

```bash
pip install -U entroly
cd /path/to/your/repository
entroly simulate
```

Take any fragment it reports as omitted, recover it through its handle, and diff
against the original. That is the whole claim, and it is the part I would most
like people to attack.

Critical reproductions are welcome. Please include the Entroly version,
workload, model and provider when used, token budget, commands, and raw
artifacts so results can be compared fairly.

## Gallery plan

1. The receipt itself: selected, omitted, and the recovery handle for each
   omission.
2. A live recovery — omitted fragment in, byte-identical original out.
3. `entroly simulate` running on a real repository.
4. The Claude Code plugin installing and running its first-run command.
5. Integration map: Claude Code, Codex, Cursor, Copilot, OpenClaw, MCP, proxy,
   SDK, Rust, npm, Docker.
6. Limitations slide: where Entroly passes through, and where it may trade
   quality.

Use repository-owned images only. Do not fabricate dashboards, customer logos,
quotes, or benchmark rankings.

## Launch checklist

- [ ] Latest PyPI and npm packages are publicly available.
- [ ] README and documentation install commands match the release.
- [ ] `entroly verify-claims` passes from a clean install.
- [ ] The Claude Code plugin installs from a clean machine.
- [ ] Product Hunt links use canonical repository and docs URLs.
- [ ] Screenshots are current and reproducible.
- [ ] No savings percentage appears anywhere in the listing.
- [ ] No dynamic adoption metric is included without a dated source.
- [ ] Maintainer affiliation is disclosed.
- [ ] Public launch URL is recorded in the distribution registry **after**
      posting, never before.
