# Entroly — Scoop (Windows)

Scoop manifest for installing entroly on Windows.

## Status

**Manifest written, not yet submitted.** [`entroly.json`](entroly.json) pins
release `entroly-v1.0.88`. Its SHA-256 was checked against the downloaded archive
and [published sidecar](https://github.com/juyterman1000/entroly/releases/download/entroly-v1.0.88/entroly-rs-x86_64-pc-windows-msvc.zip.sha256).
Earlier verification notes remain in
[windows-artifact-verification.md](../../docs/distribution/windows-artifact-verification.md).

This file previously said the manifest was "on the v0.19.x roadmap" and should
point at the PyPI wheel. Both were wrong: that version series is long gone, and
the manifest points at the standalone `entroly-rs` release binary, which is what
Scoop can install without a Python toolchain.

Windows users can still install via `pip install entroly` (works under
PowerShell) or `npm i -g entroly` for the WASM build.

## Submission checklist

- [x] Write `entroly.json` manifest — points at the GitHub release binary, not
      the PyPI wheel, because Scoop installs an executable rather than a Python
      package
- [x] Cut a release carrying the corrected `.sha256` sidecar. The 1.0.85
      sidecar uses the archive basename and its digest matches a fresh download.
- [ ] Test under `scoop install entroly` from a local manifest on a real
      Windows host
- [ ] Submit to a Scoop bucket (`extras` or maintain `scoop-entroly`)
- [ ] Document the install path in [../../README.md](../../README.md)

Tracked as `scoop-main` in
[targets.json](../../docs/distribution/targets.json).

## References

- [Scoop App Manifests](https://github.com/ScoopInstaller/Scoop/wiki/App-Manifests)
- [PyPI source distribution](https://pypi.org/project/entroly/)
