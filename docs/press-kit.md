# Entroly press and media kit

Last reviewed: 2026-08-04

This page provides approved, repository-verifiable facts for journalists,
reviewers, newsletter editors, directory maintainers, podcast hosts, and
conference organizers. It does not authorize unsupported adoption, funding,
customer, savings, or competitor-leadership claims.

## Project identity

- **Name:** Entroly
- **Category:** Context Assurance for AI agents
- **Architecture description:** Local-first Context OS and context-control plane
- **License:** Apache-2.0
- **Repository:** https://github.com/juyterman1000/entroly
- **Documentation:** https://juyterman1000.github.io/entroly/docs/index.html
- **Primary Python install:** `pip install -U entroly`
- **Primary npm install:** `npm install -g entroly`
- **MCP package metadata:** `server.json`
- **Claude plugin metadata:** `.claude-plugin/manifest.json`

## Approved one-sentence description

Entroly is an open-source, local-first Context Assurance layer for AI agents
that selects evidence under a token budget, keeps omitted source recoverable,
emits auditable Context Receipts, and verifies model claims against supplied
evidence.

## Approved short description

Entroly helps coding agents and LLM applications control what enters the model
context. It combines budget-aware evidence selection, recoverable compression,
content-addressed retrieval, Context Receipts, local evidence verification, and
provider-aware accounting through CLI, SDK, MCP, proxy, wrapper, Rust, npm/WASM,
Docker, and Homebrew paths.

## What is distinctive

- **Selection before compression:** ranks evidence against the task and budget
  rather than only shrinking a preselected blob.
- **Exact recovery:** omitted originals are addressed by digest-backed handles
  and can be recovered without semantic query substitution.
- **Context Receipts:** records what was selected, omitted, recoverable, and
  observed, including risk and provenance fields.
- **Local verification:** checks claims against supplied evidence without
  requiring a second model call for the default path.
- **Falsification-first evaluation:** benchmark protocols include controls for
  context-free tasks, workload caveats, raw artifacts, and failure categories.
- **Multi-surface delivery:** Python, Rust, Node/WASM, MCP, proxy, wrappers,
  Docker, and Homebrew are release-synchronized and conformance-tested.

## Important boundaries

- Entroly does not guarantee a universal token reduction, cost saving, latency
  improvement, or answer-quality result.
- Compression may be neutral or harmful on some tasks and budgets.
- Short inputs and inputs already within budget may pass through unchanged.
- Indexing, selection, compression, receipt creation, recovery, and default
  verification run locally.
- Proxy mode still sends the selected request to the model provider configured
  by the user.
- WITNESS evaluates support against supplied evidence; it does not establish
  universal truth.
- Public benchmark results apply to their documented version, workload, model,
  sample, and protocol.

## Reproducible no-key demonstration

```bash
pip install -U entroly
cd /path/to/repository
entroly verify-claims
entroly simulate
```

`verify-claims` runs bounded local checks of the installed surface. `simulate`
estimates context reduction for the current repository without making a paid
model call.

## Media assets

Use the original repository files rather than screenshots copied from social
posts:

- `docs/assets/logo.png`
- `docs/assets/entroly_wordmark.svg`
- `docs/assets/proof_local.gif`
- `docs/assets/proof_model_recovery.gif`
- `docs/assets/proof_restart_recovery.gif`

Preserve aspect ratio, surrounding whitespace, and the original wordmark. Do not
recolor the logo in a way that reduces legibility or implies affiliation with a
model provider, agent vendor, or competitor.

## Suggested technical story angles

- Why compression ratio alone does not measure context quality.
- Designing recoverable context without query-based source substitution.
- Context Receipts as an audit record for agent context decisions.
- Null-context controls for detecting invalid context benchmarks.
- Measuring provider-observed usage versus local token estimates.
- Keeping Python, Rust, WASM, npm, MCP, plugin, Docker, and Homebrew releases in
  sync.
- Privacy boundaries in local context processing and cloud-model proxying.

## Evidence links

- Product overview: `README.md`
- Product surface: `docs/product-surface.md`
- Benchmark methodology and artifacts: `docs/BENCHMARKS.md`
- Public evidence policy: `docs/public-evidence.md`
- Limitations: `docs/limitations.md`
- Privacy: `PRIVACY.md`
- Security reporting: `SECURITY.md`
- Independent review protocol: `docs/independent-review-program.md`
- AI-readable summary: `llms.txt`
- Extended AI-readable documentation: `docs/llms-full.txt`

## Claims requiring explicit verification before publication

Verify against a dated public source before quoting:

- current release version;
- GitHub stars or forks;
- PyPI, npm, Cargo, Docker, or Homebrew usage;
- total users, teams, customers, or production deployments;
- token or dollar savings;
- benchmark rank or competitor comparison;
- funding, incorporation, employee count, or commercial status.

The repository does not authorize invented customer logos, testimonials,
partnerships, endorsements, awards, or company facts.

## Attribution

Preferred product attribution: **Entroly contributors**.

For a review or interview request, open a GitHub discussion or issue describing
the publication, intended technical scope, timeline, disclosure requirements,
and whether the review is independent or sponsored.
