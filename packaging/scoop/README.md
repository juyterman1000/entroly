# Entroly — Scoop (Windows)

Scoop manifest for installing entroly on Windows.

## Status

**Manifest written, not yet submitted.** [`entroly.json`](entroly.json) pins
release `entroly-v1.0.81` and carries a SHA-256 verified against the published
sidecar — see
[windows-artifact-verification.md](../../docs/distribution/windows-artifact-verification.md).

This file previously said the manifest was "on the v0.19.x roadmap" and should
point at the PyPI wheel. Both are now wrong: the product is at 1.0.82, and the
manifest points at the standalone `entroly-rs` release binary, which is what
Scoop can install without a Python toolchain.

Windows users can still install via `pip install entroly` (works under
PowerShell) or `npm i -g entroly` for the WASM build.

## Submission checklist

- [x] Write `entroly.json` manifest — points at the GitHub release binary, not
      the PyPI wheel, because Scoop installs an executable rather than a Python
      package
- [ ] Cut a release carrying the corrected `.sha256` sidecar. Sidecars up to and
      including `entroly-v1.0.81` name the file as `dist/<archive>`, which
      breaks both `sha256sum -c` for anyone verifying a download and Scoop's
      autoupdate hash extraction. Fixed at source in
      [`release-binary.yml`](../../.github/workflows/release-binary.yml); the
      already-published assets are only corrected by the next release.
- [ ] Test under `scoop install entroly` from a local manifest on a real
      Windows host
- [ ] Submit to a Scoop bucket (`extras` or maintain `scoop-entroly`)
- [ ] Document the install path in [../../README.md](../../README.md)

Tracked as `scoop-main` in
[targets.json](../../docs/distribution/targets.json).

## References

- [Scoop App Manifests](https://github.com/ScoopInstaller/Scoop/wiki/App-Manifests)
- [PyPI source distribution](https://pypi.org/project/entroly/)
