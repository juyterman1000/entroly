# Entroly changelog

This file is the stable entry point for user-facing release history. Detailed,
version-specific notes live under [`docs/releases/`](docs/releases/).

## Current release

- [`v1.0.75`](docs/releases/v1.0.75.md)

## Recent releases

- [`v1.0.73`](docs/releases/v1.0.73.md)
- [`v1.0.68`](docs/releases/v1.0.68.md)
- [`v1.0.55`](docs/releases/v1.0.55.md)
- [`v1.0.54`](docs/releases/v1.0.54.md)

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
