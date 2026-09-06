# Entroly — Nix flake

`flake.nix` for installing entroly via Nix / NixOS.

## Status

**Not implemented.** There is no `flake.nix` in this repository, and no work
is scheduled. This directory holds the checklist only.

This file previously said the flake was "on the v0.19.x roadmap" -- a version
series the product left three majors ago, which read as a commitment rather than
an unstarted item. Tracked as a blocked target in
[`docs/distribution/targets.json`](../../docs/distribution/targets.json).

Until a flake exists, install via `pip install entroly` inside a Python
virtualenv, or use the Docker image at `ghcr.io/juyterman1000/entroly:latest`.

## Submission checklist

- [ ] Write `flake.nix` (with `nixpkgs` and `pyproject-nix` inputs)
- [ ] Build the Rust PyO3 extension via `maturin` overlay
- [ ] Add `nix run github:juyterman1000/entroly` command path
- [ ] Test under `nix flake check`

## References

- [Nix flake reference](https://nixos.wiki/wiki/Flakes)
- [pyproject-nix](https://github.com/nix-community/pyproject.nix)
