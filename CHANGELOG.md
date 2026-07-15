# Changelog

Entroly follows [Semantic Versioning](https://semver.org/). GitHub Releases are
the authoritative published changelog because they are tied to the automated
release workflow and versioned tags. This file provides a human-maintained index and
an `Unreleased` section for changes that affect users.

## Unreleased

### Added

- Added a five-minute quickstart, public API reference, architecture guide,
  examples index, migration guide, FAQ, and troubleshooting guide.
- Added current support, governance, maintainer, security, and contributor
  policies plus structured issue and pull-request templates.
- Added an editable social-preview source and a release-ready rendered image.
- Exposed `recover_receipt_omission` from the top-level Python package alongside
  the other documented Context Receipt helpers.
- Added CodeQL, pull-request dependency review, and a 70% branch-coverage gate
  for trust-critical Python modules using commit-pinned Actions.

### Changed

- Reworked the README navigation and landing-page metadata around Entroly's
  verifiable context-control contract.
- Expanded dependency update coverage across the Rust, WASM, npm, and OpenClaw
  workspaces.

### Fixed

- Added offline local-link and spelling gates for public documentation.
- Removed an unsupported fixed token-savings claim from the Python package
  description.
- Made README proof-source verification stable across LF and CRLF checkouts
  while retaining byte-exact hashes for rendered media.

Pull requests that change behavior, compatibility, data formats, configuration,
or installation must add an entry here or explain why no release note is needed.

## Recent releases

### [1.0.60](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.60) — 2026-07-15

- Published a quality-gated compression-latency holdout.
- Added evidence-gated model rehydration for recoverable context.
- Compare: [`1.0.59...1.0.60`](https://github.com/juyterman1000/entroly/compare/entroly-v1.0.59...entroly-v1.0.60)

### [1.0.59](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.59) — 2026-07-14

- Made post-release publication race-free and reviewable.
- Added evidence-gated neural-compression benchmarks.
- Fixed cross-process recovery durability and added a verified holdout.
- Compare: [`1.0.58...1.0.59`](https://github.com/juyterman1000/entroly/compare/entroly-v1.0.58...entroly-v1.0.59)

### [1.0.58](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.58) — 2026-07-14

- Restored verifiable public-trust contracts for release metadata and claims.
- Synchronized the Homebrew formula after the matching artifact was published.
- Compare: [`1.0.57...1.0.58`](https://github.com/juyterman1000/entroly/compare/entroly-v1.0.57...entroly-v1.0.58)

See [all releases](https://github.com/juyterman1000/entroly/releases) and the
versioned notes under [`docs/releases/`](docs/releases/).

## Entry format

Add changes under `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or
`Security`. State the user impact, not only the internal implementation. Link a
migration guide for breaking changes and a public evidence artifact for new
quantitative claims.
