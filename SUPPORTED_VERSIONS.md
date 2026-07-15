# Supported Versions

This document separates package compatibility from security-support policy.

## Security support

Only the newest published `1.0.x` patch receives security fixes. Upgrade before
reporting a problem against an older patch unless the older version is necessary
to reproduce a regression.

## Runtime compatibility

| Surface | Supported baseline | Verification source |
| --- | --- | --- |
| Python package | Python 3.10+ | CI builds and tests Python 3.10 through 3.14 |
| Pure-Python fallback | Python 3.10+ | Dedicated CI job without `entroly_core` |
| Native Python extension | CPython 3.10+ ABI | `entroly-core` PyO3 `abi3-py310` build |
| Node/WASM package | Node.js 16+ | `entroly-wasm/package.json` engine contract |
| OpenClaw plugin | Node.js 22.19+ | Plugin engine contract and minimum-host smoke test |
| Operating systems | Linux, macOS, Windows | Wheel, binary, and CI release matrices |

Support means the project accepts reproducible compatibility bugs for the
current release. A platform listed here may still require optional tooling for
source builds, such as Rust, maturin, a C/C++ linker, or FFmpeg for proof-video
regeneration.

## Installation tiers

- `pip install entroly` installs the Python control plane and MCP dependency.
- `pip install "entroly[proxy]"` adds the HTTP proxy runtime.
- `pip install "entroly[native]"` adds the published Rust extension.
- `pip install "entroly[full]"` installs proxy, native, and receipt-proof
  dependencies.
- `npm install -g entroly` installs the Node/WASM distribution.

Run `entroly doctor` and `entroly verify-claims` when reporting an environment
problem. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for a redaction-safe report
template.

## Deprecation policy

The project uses semantic versioning. A deprecated public CLI option, Python
API, receipt field, or configuration key should remain available for at least
one minor release when practical and emit an actionable migration message.
Security fixes may remove unsafe behavior without a full deprecation window.
Breaking changes require a major release and an entry in
[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).
