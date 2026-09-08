# Style Guide

This guide covers project conventions not enforced automatically. Existing
subsystem patterns and tests take precedence over aesthetic rewrites.

## General principles

- Optimize for predictable behavior, explicit failure, and recoverability.
- Keep changes small enough to review and roll back.
- Name concepts consistently with the public product boundary: Entroly is an
  auditable context-control plane, not a model provider or chat client.
- Prefer stable policy functions and adapters over provider conditionals spread
  across entry points.
- Do not hide errors that affect data, requests, receipts, release state, or
  user expectations.

## Python

- Support Python 3.10+ and use modern type hints compatible with that baseline.
- Follow Ruff and existing `ruff.toml` configuration.
- Keep imports side-effect free; optional dependencies fail with an actionable
  capability message, not a partially initialized module.
- Catch broad exceptions only at explicit fail-open boundaries and include the
  reason in diagnostics. Never use a broad catch to claim success.
- Public functions include types, behavior, failure modes, and a minimal example
  in their docstring.
- Tests use descriptive `test_<condition>_<expected_result>` names and synthetic
  fixtures that expose the regression.

## Rust

- Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and the
  focused unit tests.
- Avoid panics on request or user-data paths. Return structured errors with
  enough context for the Python or CLI boundary to produce an actionable
  message.
- Keep deterministic algorithms deterministic across platforms. Normalize paths
  and serialization where identifiers or hashes depend on them.
- Native capability checks probe required symbols and versions, not only crate
  importability.

## JavaScript and Node

- Preserve the Node baseline declared by each package.
- Validate untrusted bridge output again at the JavaScript boundary.
- Avoid shell interpolation for user-controlled values; pass argument arrays.
- Keep plugin configuration schema, defaults, README, and integration tests in
  sync.

## Errors and logs

An actionable error answers: what failed, which operation is affected, whether
the original data/request is safe, what recovery occurred, and what the user can
do next. Do not log provider keys, bearer tokens, raw private prompts, source
text, or recovery payloads. Use stable error identifiers for recurring public
failures.

## Public APIs and schemas

- Prefer additive fields and keyword-only options within a major version.
- Validate at every process or language boundary.
- Keep receipt fields explicit about authority and measurement tier.
- Deprecations include a replacement, warning, test, migration note, and removal
  release.

## Documentation

- Lead with the user outcome and the shortest verified path.
- Use sentence case for headings and descriptive alt text for images.
- Every command must identify its prerequisites and destructive or network
  effects.
- Quantitative claims name the baseline version, workload, token budget, model
  settings, raw artifact, and limitation. Avoid “best,” “same answers,” or
  universal percentages unless the linked evidence supports that exact scope.
- Distinguish supported, tested, compatible by protocol, experimental, and
  planned integrations.
- Use relative links for repository files and canonical HTTPS links for public
  destinations.

## Commits and pull requests

Use a concise imperative subject, optionally with a conventional scope, such as
`fix(proxy): preserve upstream status on retry exhaustion`. Keep generated
artifacts with the source change that produced them. Pull-request expectations
are in [CONTRIBUTING.md](CONTRIBUTING.md).
