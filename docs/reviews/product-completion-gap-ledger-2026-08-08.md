# Product-completion gap ledger — 2026-08-08

This ledger records a source-level audit, not a marketing comparison. The
baseline is fetched `origin/main` at
`90f7e50c1acd08cf7e1c2392407d944a1c2739db`; the local gateway hardening base is
`8660cf45dcebf4f09078be667aedfaad7fe83b0b`. No external product name or
unexecuted performance claim is needed to justify these decisions.

## Present before this work

- Five priced read resolutions, including caller-reachable full and diff modes.
- Line-range and density-aware reads, structural outlines, and warm rereads.
- Content-addressed exact recovery, Context Receipts, and WITNESS evidence.
- Python, Node/WASM, local proxy, MCP, wrapper, daemon, and team documentation
  surfaces.
- Image token estimation plus an optional library-level optimizer.
- Cache-aware accounting and signed receipt primitives.

These capabilities were not duplicated.

## Added because the production path was missing

| Capability | Missing production boundary | Added boundary |
|---|---|---|
| Persistent install | No reversible user-service lifecycle | User-scoped systemd, launchd, and Task Scheduler definitions with dry-run, status, restart, ownership digest, and safe removal |
| Framework SDK | No receipt-first gateway client for common SDK traffic | Python and TypeScript local-gateway clients, provider wrappers, exact recovery, and a LiteLLM callback |
| Failure learning | Existing learning path could write without transcript-level causal evidence | Redacted failure-then-success proposals with source hashes, separate review/apply, source revalidation, and backup |
| Proxy images | Optimizer was not wired to provider request shapes or recovery | Opt-in provider-shape transformation, pre-mutation content-addressed originals, receipt headers, and authenticated exact recovery |
| Platform evidence | Cross-platform support was prose-only | Three-OS contract workflow and machine-readable readiness statuses that distinguish CI verification from contract tests |
| Supply-chain evidence | No release SBOM/provenance workflow | SPDX and CycloneDX artifacts plus release-only signed attestation |
| Team evaluation | No bounded pilot/conversion contract | Matched-arm schema for quality, provider usage, cache, latency, recovery, and retained failures |
| AI discovery | Deep docs lacked compact exact-phrase retrieval anchors | Root `ai.txt`, mirrored docs surface, exact context-compression headers, sitemap/robots links, and metadata contracts |

All new capabilities are labeled beta where their real operating boundary is
bounded. The image path makes no universal savings claim; the install matrix
does not claim hosted CI registered a real desktop service; the pilot treats
missing data as unknown, never zero.

## Verified absences deliberately not added

- **Generation-control rewriting.** Entroly deliberately passes through
  `temperature`, `thinking`, `reasoning_effort`, and equivalent provider-owned
  controls. Automatically shaping answers would violate the documented
  compliance boundary.
- **A new multi-language parser dependency.** No tree-sitter dependency is
  present. Entroly already exposes native structural outlines and a documented
  lightweight fallback. Parser weight is not justified until a shared workload
  measures better task outcomes, not merely more parsed languages.
- **A learned compression model.** No model-in-loop evidence demonstrates a
  better quality/cost frontier than the deterministic engine. Shipping a model
  download and inference path without that evidence would add operational cost
  and an unsupported claim.
- **A second plugin marketplace format.** The existing Claude manifest was
  strengthened. Other plugin ecosystems use different schemas; no cross-format
  manifest was fabricated.

## Release gate

The branch must pass focused runtime, recovery, platform, discovery, packaging,
and distribution checks before publication. Live-provider quality and billing
claims remain outside this local audit unless provider usage evidence is
captured on a declared workload.
