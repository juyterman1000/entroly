# Migration Guide

Entroly uses semantic versioning across Python, Rust, npm/WASM, MCP, OpenClaw,
Docker, binary, and Homebrew surfaces. Read the versioned
[GitHub Release](https://github.com/juyterman1000/entroly/releases) before
upgrading production environments.

## Safe upgrade procedure

1. Export or back up configuration, receipts, recovery stores, learned state,
   and any session artifacts required for replay.
2. Record `entroly --version`, `entroly doctor`, and a known-good
   `entroly verify-claims` report.
3. Pin the target version in a test environment.
4. Run `entroly migrate` only after reviewing its planned changes and retained
   backup.
5. Rerun `doctor`, `verify-claims`, one receipt/recovery smoke, and the real
   integration's streaming/tool-call smoke.
6. Compare a fixed held-out workload before increasing compression or routing
   scope.
7. Keep the previous package and exact-passthrough configuration available for
   rollback.

## From an older 1.0.x patch

Patch releases are intended to remain backward compatible. Upgrade every
installed surface together when they interact in one deployment:

```bash
python -m pip install --upgrade "entroly[full]"
npm install --global entroly@latest
entroly doctor
entroly verify-claims
```

Marketplace and package registries may converge at different times. Verify
PyPI, npm, GitHub Release, Docker/GHCR, ClawHub, and Homebrew independently.
Do not interpret a tag as proof that every artifact is live.

## From 0.x to 1.x

The 1.x line treats receipts, recovery, local-first behavior, and release
consistency as public contracts. Use a clean environment rather than installing
over an unknown 0.x dependency graph.

1. Preserve the old environment and state directory.
2. Install the current 1.x release in a new environment.
3. Run `entroly doctor` and `entroly verify-claims` before importing state.
4. Use supported `entroly export` / `entroly import` or `entroly migrate` paths;
   do not hand-edit receipts or recovery fingerprints.
5. Recreate wrapper, MCP, proxy, or attachment configuration explicitly and
   remove stale base-URL variables from the old session.
6. Validate exact passthrough and one known receipt before production use.

Open a bug report if migration cannot preserve data. Do not delete the original
state while the report is investigated.

## Public API and schema changes

- Additive receipt fields are allowed within 1.x; readers should ignore unknown
  fields and validate required fields.
- Deprecated CLI options and Python APIs should emit a replacement and remain
  available for at least one minor release when practical.
- A breaking removal, schema rewrite, or default network-boundary change
  requires a major release and a dedicated section here.

## Rollback

Rollback the package and configuration together. A newer process may have
written state that an older release cannot understand. Restore the matching
backup rather than forcing an older binary to modify newer state. Verify the
rollback with `doctor`, `verify-claims`, and an exact-passthrough request.

## Version-specific notes

- [1.0.62 release](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.62)
- [1.0.61 release](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.61)
- [1.0.60 release](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.60)
- [All versioned notes](docs/releases/)
