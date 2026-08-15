# Entroly distribution evidence

The README's **100K+ observed distribution events** milestone is a bounded
event count, not a user, installation, activation, retention, or value claim.
The machine-readable snapshot is
[`adoption-evidence.json`](adoption-evidence.json).

## Snapshot on August 14, 2026

| Source | Count | Boundary |
|---|---:|---|
| PyPI `entroly` | 18,359 | Download events excluding known mirrors |
| PyPI `entroly-core` | 22,743 | Download events excluding known mirrors |
| npm `entroly` | 9,864 | Since package publication |
| npm `entroly-mcp` | 12,877 | Since package publication |
| npm `entroly-wasm` | 11,469 | Since package publication |
| npm `entroly-openclaw` | 3,813 | Since package publication |
| GitHub full clones | 20,971 | Latest available 14-day window; 1,061 unique cloners |
| GitHub release binaries | 342 | Binary archives only |
| **Observed distribution events** | **100,438** | Events may repeat or overlap |

The arithmetic is `41,102 PyPI + 38,023 npm + 20,971 clones + 342 release
binaries = 100,438`.

## What is deliberately not added

- **PyPI mirrors.** The two PyPI projects recorded 126,619 downloads including
  mirrors, but known mirror synchronization is excluded from the headline.
  PyPI's `without_mirrors` series is a subset of `with_mirrors`; adding both
  series would double count.
- **Checksum files.** The 462 release checksum downloads are not product-binary
  retrievals.
- **GitHub source ZIP/tar archives.** GitHub exposes release-asset download
  counts, but not a separate counter for automatic repository source archives.
- **MCP Registry discovery.** The official entry points to PyPI `entroly` and
  npm `entroly-mcp`. The registry does not provide a separate install counter,
  so adding one would duplicate package events.
- **SDKs.** The Python SDK, native Python core, JavaScript SDK and MCP bridge
  ship inside the packages already listed. They are product surfaces, not new
  distribution events.
- **Python/Node CLI and proxy.** These commands ship inside `entroly`,
  `entroly-core` and `entroly-mcp`. A package retrieval is counted once even if
  the installation later runs both the CLI and proxy.
- **Rust proxy/CLI.** Prebuilt `entroly-rs` archives are included in the 342
  release-binary downloads. A source build starts from a repository clone,
  already represented by GitHub clone traffic.
- **Homebrew.** The formula downloads the PyPI `entroly` source distribution,
  which is already counted. The tap repository recorded 18 recent clones, but
  they are excluded because one Homebrew install can produce both a tap clone
  and the counted PyPI retrieval.
- **GHCR.** The container is published, but no auditable official pull total
  was available to this snapshot, so the headline adds zero container pulls.
- **Rust crates.** The repository contains four Cargo packages, but none was a
  published crates.io distribution at the observation time.

## Source APIs

- PyPI package history: `https://pypi.org/pypi/<package>/json`
- PyPI download series: `https://pypistats.org/api/packages/<package>/overall`
- npm package history: `https://registry.npmjs.org/<package>`
- npm download totals: `https://api.npmjs.org/downloads/point/<start>:<end>/<package>`
- GitHub clone traffic: `GET /repos/juyterman1000/entroly/traffic/clones`
- GitHub release assets: `GET /repos/juyterman1000/entroly/releases`
- MCP registry entry:
  `https://registry.modelcontextprotocol.io/v0.1/servers/io.github.juyterman1000%2Fentroly/versions/latest`

Registry downloads and clones include repeat requests, CI, automation and
possibly overlapping people. They are useful distribution-volume evidence,
but they must never be presented as unique users, installations, active users,
successful activations, retained users, or token/cost savings.
