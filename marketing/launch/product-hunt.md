# Product Hunt launch draft

Status: prepared, not submitted.

## Product name

Entroly

## Tagline

Context Assurance for AI agents — select evidence, recover originals, verify claims.

## Short description

Entroly is an open-source, local-first context-control layer for AI agents. It
selects useful evidence under a token budget, compresses recoverably, emits
Context Receipts, and checks model claims against supplied evidence. Use it as a
CLI, Python or TypeScript SDK, MCP server, proxy, coding-agent wrapper, native
Rust binary, or container.

## Maker comment

I built Entroly because most context tools optimize one visible number: fewer
tokens. That is useful, but incomplete. The harder questions are what evidence
was removed, whether the original can be recovered exactly, whether the model's
answer is supported, and whether the claimed saving was measured at the actual
provider boundary.

Entroly treats context as a controlled system:

1. select evidence under an explicit budget;
2. compress content while retaining content-addressed originals;
3. issue a receipt describing what was kept, omitted, and recoverable;
4. verify claims against the supplied evidence;
5. expose the same contract through CLI, SDK, MCP, proxy, wrappers, Rust, WASM,
   npm, Docker, and Homebrew paths.

The project is Apache-2.0 and runs indexing, selection, compression, recovery,
and default verification locally. Proxy mode still sends the selected request
to the model provider configured by the user.

The honest limitation: compression is workload-dependent. It can reduce useful
context or add overhead on the wrong task. Entroly includes `verify-claims` and
`simulate` so users can inspect behavior before connecting a paid model.

Repository: https://github.com/juyterman1000/entroly
Documentation: https://juyterman1000.github.io/entroly/docs/index.html

## First comment

A useful first test requires no API key:

```bash
pip install -U entroly
cd /path/to/your/repository
entroly verify-claims
entroly simulate
```

`verify-claims` runs bounded local checks for installation, compression,
receipts, recovery, and routing. `simulate` estimates context reduction on the
current repository without making a paid model call.

Questions and critical reproductions are welcome. Please include the Entroly
version, workload, model/provider when used, token budget, commands, and raw
artifacts so results can be compared fairly.

## Gallery plan

1. Product overview: evidence selection -> receipt -> exact recovery.
2. `entroly verify-claims` local proof.
3. `entroly simulate` on a real repository.
4. Context Receipt example with omitted and recoverable evidence.
5. Integration map: Claude Code, Codex, Cursor, Copilot, OpenClaw, Hermes,
   OpenCode, MCP, proxy, SDK, Rust, npm, Docker.
6. Limitations slide: where Entroly passes through or may trade quality.

Use repository-owned images only. Do not fabricate dashboards, customer logos,
quotes, or benchmark rankings.

## Launch checklist

- [ ] Latest PyPI and npm packages are publicly available.
- [ ] README and documentation install commands match the release.
- [ ] `entroly verify-claims` passes from a clean install.
- [ ] Product Hunt links use canonical repository and docs URLs.
- [ ] Screenshots are current and reproducible.
- [ ] No dynamic adoption metric is included without a dated source.
- [ ] Maintainer affiliation is disclosed.
- [ ] Public launch URL is recorded in the distribution registry.
