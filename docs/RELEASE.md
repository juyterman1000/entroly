# Entroly release runbook

This is the canonical fail-closed procedure for publishing an Entroly release.
A version is not considered released merely because a version string changed or
a tag was created.

## Source of truth

The release commit must be on `main`, and every version-bearing surface must be
synchronized by the repository's version tooling and release tests. The current
user-facing history is indexed in [`../CHANGELOG.md`](../CHANGELOG.md), with
version-specific notes under [`releases/`](releases/).

Primary automation and verification surfaces:

- `scripts/bump_version.py`
- `scripts/sync_release_version.py`
- `tests/test_release_surface.py`
- `tests/test_bump_version.py`
- `.github/workflows/entroly-publish.yml`
- `.github/workflows/publish-core-wheels.yml`
- `packaging/homebrew/entroly.rb`
- `packaging/homebrew/README.md`

## 1. Freeze the release scope

Before changing the version:

1. Identify the exact source commit and intended version.
2. Freeze user-visible capabilities, fixes, migrations, and known limitations.
3. Confirm no required pull request or security fix remains outside the release
   head.
4. Prepare version-specific release notes under `docs/releases/`.
5. Remove claims that are not supported by the final release artifacts.

Do not tune benchmarks, documentation, or acceptance thresholds after viewing a
held-out outcome merely to improve the announcement.

## 2. Synchronize versions

Use the canonical version tooling rather than editing package files by hand.
The version contract includes Python, Rust, npm/WASM, MCP, plugin, package,
citation, and release metadata surfaces.

After synchronization, run the release-focused tests and inspect the complete
diff. No version-bearing file may be silently excluded.

## 3. Validate the unchanged release head

The exact commit intended for publication must pass all required repository
checks, including:

- Python 3.10–3.14 test and wheel matrices;
- pure-Python fallback;
- Rust tests and lint;
- WASM build and export checks;
- MCP, proxy, SDK, plugin, and integration contracts;
- release-surface and version-alignment tests;
- security, dependency, user-journey, and dogfood gates;
- distribution and visibility integrity checks.

A green result from an earlier commit does not validate a later release head.
Do not merge, tag, or publish while a required check is failing or pending.

## 4. Create tag and GitHub release

The release tag and GitHub release must resolve to the validated `main` commit.
Release notes must include:

- concrete user-visible changes;
- installation or upgrade commands;
- migration and rollback guidance;
- compatibility boundaries;
- security, privacy, persistence, and network changes;
- evidence and verification commands;
- known limitations and unresolved regressions.

Never invent customer, usage, savings, ranking, or production-deployment claims.

## 5. Publish coordinated artifacts

Use `.github/workflows/entroly-publish.yml` for coordinated publication. Verify
each advertised channel independently before announcing it:

- PyPI source distribution and wheels;
- npm and Node/WASM packages;
- Rust/native artifacts where published;
- MCP and plugin metadata;
- container image;
- Homebrew formula and checksum;
- any integration-specific package or registry entry included in the release.

A successful upload step is not sufficient. Perform a clean install, import, or
startup check against the public artifact.

## 6. Verify Homebrew

The Homebrew formula must reference the released source artifact and its actual
checksum. Validate installation on a clean supported environment and confirm:

- `entroly --version` matches the release;
- `entroly doctor` reports the expected runtime;
- uninstall leaves no unexpected executable or configuration changes;
- formula URLs and checksums resolve publicly.

Do not update the formula to an artifact that is not yet public.

## 7. Publish communications

Only after package availability is verified:

1. Mark the version current in `CHANGELOG.md`.
2. Publish the reviewed GitHub release notes.
3. Use `marketing/release-announcement-template.md` for channel-specific drafts.
4. Include only channels that are publicly available and verified.
5. Record relevant public URLs in the distribution registry.

Announcements must preserve workload caveats, limitations, and the distinction
between local estimates and provider-observed usage.

## 8. Post-release verification

After publication:

- install from each public channel on a clean environment;
- run `entroly doctor` and `entroly verify-claims`;
- verify package metadata, documentation links, citation version, and manifests;
- confirm release notes and install commands match the shipped behavior;
- check that rollback or uninstall guidance still works;
- record and triage any publication drift or unavailable downstream listing.

If a critical artifact is wrong, stop promotion, document the problem, and use a
new corrective release. Do not silently replace immutable artifacts.

## Definition of done

A release is complete only when the validated commit, tag, GitHub release,
advertised packages, documentation, release notes, citation metadata, and
installation checks all identify the same version and behavior.
