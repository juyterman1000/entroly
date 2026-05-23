# Entroly — Nix flake

`flake.nix` for installing entroly via Nix / NixOS.

## Status

Placeholder. Nix flake is on the v0.19.x roadmap. Until then, Nix users
should install via `pip install entroly` inside a Python virtualenv,
or use the Docker image at `ghcr.io/juyterman1000/entroly:latest`.

## Submission checklist

- [ ] Write `flake.nix` (with `nixpkgs` and `pyproject-nix` inputs)
- [ ] Build the Rust PyO3 extension via `maturin` overlay
- [ ] Add `nix run github:juyterman1000/entroly` command path
- [ ] Test under `nix flake check`

## References

- [Nix flake reference](https://nixos.wiki/wiki/Flakes)
- [pyproject-nix](https://github.com/nix-community/pyproject.nix)
