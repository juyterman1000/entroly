# Marketplace and directory submission copy

Use these templates only after checking the published package and the target
listing. A marketplace is a discovery surface, not evidence that Entroly was
installed, validated, or benchmarked successfully.

## Canonical facts to verify before submission

- Repository: `https://github.com/juyterman1000/entroly`
- License: Apache-2.0
- Python package: `https://pypi.org/project/entroly/`
- Node/WASM package: `https://www.npmjs.com/package/entroly`
- npm bridge to an installed Python runtime: `https://www.npmjs.com/package/entroly-mcp`
- MCP identity: `io.github.juyterman1000/entroly`
- Python MCP command: `entroly` with no arguments
- Package-runner command: `uvx --from entroly entroly` with no `serve` argument
- npm bridge command: `npx -y entroly-mcp` with no `serve` argument

Confirm that the versions on PyPI, npm, `server.json`, and the release tag agree
before copying any version into an external form.

## Short description

> Local context control for AI agents with budgeted selection, recoverable
> omissions, Context Receipts, and optional answer verification.

## Long description

> Entroly is an Apache-2.0 local context-control plane for AI agents. It can
> select context under a token budget, record selected and omitted evidence,
> retain recovery handles, and expose those capabilities through Python, MCP,
> proxy, and Node/WASM integrations. The base Python package includes a
> pure-Python path; Rust acceleration is optional. Token reduction and answer
> quality depend on the workload, model, budget, and integration. Reproducible
> results and caveats are linked from the repository's public evidence policy.

## Submission rules

1. Do not publish a universal token-savings or bill-reduction percentage.
2. Do not describe WITNESS as preventing or eliminating hallucinations.
3. Do not claim that cache alignment guarantees a provider cache hit or discount.
4. Do not call the base PyPI install Rust-backed without runtime verification.
5. Do not claim a marketplace score, ownership state, or validation result from
   repository-local tests.
6. Link measured claims to the exact result file, not the results directory.
7. Recheck every third-party listing after publication and record its visible
   version, capability counts, ownership state, and validation state.

## LobeHub-specific status

The ownership badge must remain in the primary README for LobeHub's external
claim workflow, but it belongs in the contextualized marketplace section rather
than the top evidence row. The dated
[LobeHub score audit](../lobehub-score-audit.md) is repository evidence only.
Use the
[live score page](https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score)
for current external status, and do not report an improvement until that page
shows the published version and successful external validation.

## Other directory targets

For Glama, awesome lists, AlternativeTo, LibHunt, and similar directories, use
the canonical facts and descriptions above. Treat each form as untrusted until
the resulting public page is reviewed. If a directory rewrites the package
name, command, license, version, or benchmark language, request correction or
remove the listing from release messaging.
