# Marketplace Wedge: Presence as a Release Surface

Date: 2026-09-06
Status: design, approved for planning

## Problem

Entroly has capability parity with its competitors and loses on adoption. The
gap is not depth. Between 2026-08-17 and 2026-09-06 the project took **364
commits on `main`** and moved from **437 to 443 GitHub stars** — six stars in
twenty days, against a target of 120 per week. Exactly **one** of those commits
touched `marketing/`.

Building a further capability is the intervention that has already been tried
hardest and returned least. This design instead targets the channels where a
developer discovers a tool from inside the editor they already have open.

## Evidence

The distribution surface was audited on 2026-09-06. Every row below was verified
against the live service, not against the repository's intent.

| Channel | Present in repo | Live |
|---|---|---|
| Official MCP registry | `server.json`, `.github/workflows/publish-mcp-registry.yml` | **Yes** — 20 versions, all `active`; publish workflow succeeded 2026-09-06 |
| Smithery (one-click for Cursor, Claude Desktop, Windsurf) | `smithery.yaml` | **No** — server page does not exist |
| Claude Code plugin | `.claude-plugin/manifest.json`, `.claude-plugin/plugin.json` | **No** — no `marketplace.json`, so there is no install path |

One channel is alive. It is also the only channel with a workflow that publishes
it without a human deciding to. The two dead channels each require someone to
remember, and neither has been remembered. The same pattern governs
`marketing/launch/` — three drafts, all still marked *"Status: prepared, not
submitted."*

The conclusion this design is built on: **a distribution channel that depends on
human memory decays to zero, and nothing reports that it has.** Smithery's
configuration is correct and has been for months. Nothing anywhere failed.

## Goals

1. Make Entroly installable from the Claude Code plugin marketplace, which today
   is impossible rather than merely unadvertised.
2. Make the first run inside the host tool demonstrate the one thing parity does
   not cover.
3. Make channel presence a release surface that fails loudly when it regresses,
   so no channel can be silently dead again.

## Non-goals

- Positioning, README headline, and the three unsent launch drafts. These were
  assessed as the larger constraint and deliberately deferred; they are not in
  this spec.
- Any new engine, selection, or receipt capability. This design ships what
  exists through channels that do not currently carry it.
- Marketplaces beyond the three named above.

## Design

### Part 1 — Make it installable

`.claude-plugin/manifest.json` is a hand-rolled descriptor. Its `install.command`
field (`pip install entroly`) is not read by anything; Claude Code discovers
plugins through a marketplace manifest, which this repository does not have.

- Add `.claude-plugin/marketplace.json`, the file `/plugin marketplace add
  juyterman1000/entroly` reads.
- Rewrite `.claude-plugin/plugin.json` to the Claude Code plugin schema.
- The plugin carries an `.mcp.json` and the existing
  `skills/entroly-evidence-operations/SKILL.md`. No new engine work.

The MCP server command is `entroly-plugin-launch`, a shim that tries `uvx`, then
`npx`, then an `entroly` already on `PATH`, and on total failure prints one
actionable line. It ships as a script inside the plugin directory rather than as
a new `[project.scripts]` console entry: a console entry would not exist until
the package is installed, which is the exact condition the shim has to survive.
This is deliberate: MCP configuration has no fallback chain, so
naming `uvx` directly means that a user without `uv` installed sees a plugin that
is silently inert. A dead plugin that reports nothing is the worst outcome
available to this design, and it is worth twenty lines to avoid.

### Part 2 — Make the first run prove something

A plugin that installs and shows nothing gets uninstalled. First activation runs
a bounded selection over the user's own repository and shows the omitted
evidence *together with the handles that recover it* — the property competitors
do not have. It reuses `simulate` and `verify-claims` as they exist.

**The savings percentage must not appear on that screen.** `saved = max(0,
baseline - selected_tokens)` with `baseline = min(total_tokens, 32_000)` in
`entroly/cli.py` pins the figure at or above 75% for a budget of 8,000 before
selection has run. It is budget arithmetic wearing the costume of a measurement.
Shipping it into a first-run screen would distribute a number that flatters
Entroly for a reason unrelated to Entroly, and it would be found.

### Part 3 — Make presence self-enforcing

`publish-mcp-registry.yml` already contains the mechanism this design needs. Its
"Verify exact registry ownership and listing" step polls the registry up to 60
times and asserts the server name, the released version, the canonical
repository URL, and the exact package set. It is careful and it works — it is
also hard-coded to one channel and duplicated as inline heredocs across two
steps.

The work is extraction, not invention.

- `scripts/marketplace_presence.py` — given a channel and an expected version,
  answers whether it is listed. It does not publish and has no side effects.
- **Channel adapters**, one per channel, each returning three states:
  `Present`, `Absent`, `Unknown`. Three rather than two is load-bearing;
  see failure mode 2.
- `publish-marketplaces.yml` — orchestration only, modeled on the workflow that
  already works.
- `publish-mcp-registry.yml`'s inline Python is replaced by a call into the
  shared script, deleting the duplicated heredoc.

Version synchronisation hooks into `scripts/bump_version.py` `TARGETS`, which
already carries `.claude-plugin/manifest.json` and `.claude-plugin/plugin.json`
at lines 88 and 91, with its repository-wide sweep as the backstop that names
anything missed. It does **not** hook into a list in a document: the 1.0.82 bump
left seven manifests behind precisely because a document list cannot know what
was added after it was written.

## Failure modes

1. **`uvx` absent on the user's machine.** The plugin never starts and reports
   nothing. Mitigated by the `entroly-plugin-launch` shim in Part 1.
2. **Registry API drift.** `publish-mcp-registry.yml` queries `/v0.1/servers`;
   the audit probe used `/v0/servers`. Both answered on 2026-09-06, so two paths
   are live and the schema is moving. A gate that hard-fails on a shape change
   turns a green release red for a reason that has nothing to do with Entroly.
   `Unknown` therefore warns and is recorded; it never silently passes and never
   blocks.
3. **The gate blocks releases indefinitely.** Smithery has no publish API — it
   indexes from GitHub on its own schedule, so `Absent` there can be nobody's
   fault. Each channel carries a policy: `blocking` for channels Entroly pushes
   to, `advisory` for channels that pull.
4. **Version skew.** The registry currently lists 1.0.81 while the repository is
   at 1.0.83. Probes compare against the released version, never `HEAD`.

## Test matrix

| Layer | Coverage | Network |
|---|---|---|
| Unit | Each adapter against recorded fixtures: present, absent, malformed, timeout | No |
| Regression | The MCP adapter reproduces the current workflow's assertions exactly — name, version, repository URL, package set — proving the extraction is behavior-preserving | No |
| Schema | `marketplace.json` and `plugin.json` validated in CI so a malformed manifest cannot ship | No |
| Contract | One live probe per channel, in the release workflow only | Yes |

## Rollout

1. Land Parts 1 and 2 behind the schema test.
2. Extract the presence gate; confirm the MCP regression test reproduces current
   behavior before switching the workflow over to it.
3. Submit to Smithery by hand once. From then on the advisory probe reports it.
4. Add the marketplace probes to the release workflow as `blocking` only after
   one full release cycle has run them green in advisory mode.

Release remains tag-driven: merge to `main`, then tag the post-merge commit on
`main`. Tagging a branch commit that is later squash-merged orphans the tag and
every publish then fails.

## Expected result, stated honestly

The MCP registry is live at 20 published versions and did not move the star
line. Registry presence is table stakes rather than a growth engine. The Claude
Code marketplace is a better channel because it has a browse surface and an
audience already inside the tool, but this work is a slower burn than the
positioning and launch work deferred under Non-goals. It is recorded here so
that the outcome is measured against the claim rather than against hope.

What this design does guarantee is narrower and worth having on its own: after
it lands, a channel cannot be dead for months without the release telling
someone.
