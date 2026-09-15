# Entroly changelog

This file is the stable entry point for user-facing release history. Published
version-specific notes live in [GitHub Releases](https://github.com/juyterman1000/entroly/releases).

## Unreleased

- Runtime native-engine repair now requires explicit consent. Install
  `entroly-core` yourself, call `entroly.repair()`, or opt in to startup repair
  with `ENTROLY_ENABLE_SELF_HEAL=1`. `ENTROLY_NO_SELF_HEAL=1` and
  `ENTROLY_AIR_GAP=1` override consent. Missing-native diagnostics remain visible.
- Corrected the PEP 668 marker path so repair respects externally managed Python.
- Context Receipt and Context Commit file writes use atomic replacement and
  private POSIX file modes. Windows inherits directory ACLs; existing directory
  permissions and unencrypted storage remain operator responsibilities.
- Corrected privacy/telemetry disclosures and removed unsupported accuracy and
  unnamed-product comparison claims. Privacy diagnostics no longer certify the
  absence of network traffic. Historical loss cases remain documented.

These changes are not part of the published releases below until a patch release
is completed. See [recovery-data security](docs/recovery-data-security.md).

## Current release

- [`v1.0.84`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.84)

## Recent releases

- [`v1.0.83`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.83)
- [`v1.0.82`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.82)

## Changelog contract

A release note must describe user-visible behavior rather than only internal
implementation work. Include, where applicable:

- capabilities added, changed, or removed;
- installation and upgrade impact;
- compatibility and migration requirements;
- security, privacy, persistence, or network-boundary changes;
- fixes for user-visible failures;
- verification commands and linked evidence;
- known limitations and regressions;
- rollback or uninstall guidance.

Quantitative statements must identify the exact version, workload, model or
provider when applicable, token or context budget, sample, protocol, and raw
artifact. A result from an earlier version is not automatically evidence for a
new release.

## Publication rule

Do not add a release as current until its tag and coordinated artifacts are
public and resolve to the intended commit. The canonical release procedure is
[`docs/RELEASE.md`](docs/RELEASE.md), and the coordinated publication workflow is
[`.github/workflows/entroly-publish.yml`](.github/workflows/entroly-publish.yml).
