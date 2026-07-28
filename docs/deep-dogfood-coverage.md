# Deep dogfood coverage

This document describes the adversarial test boundary added in draft PR #217.
It is a coverage map, not a release-readiness claim. The PR remains draft until
all required checks on the final product commit complete successfully and every
failure is either repaired or explicitly classified with evidence.

## Principles

- Existing documentation, benchmarks, badges, and green tests are treated as
  hypotheses until a documented user journey reproduces them.
- Startup crashes, malformed protocol output, missing public symbols, corrupted
  persistence, and contradictory accounting fail closed.
- Each product family runs in an independent job with its own log and enforced
  exit status. A passing compressor cannot hide a broken recovery, security,
  memory, integration, or release surface.
- Published artifacts are tested separately from repository source. Passing a
  repository test does not prove the public package contains an unreleased fix.
- Tests that require optional runtimes must install or build those runtimes;
  optional-dependency skips are not counted as evidence for the skipped feature.

## Executable product-family map

| Advertised layer | Executable gate |
|---|---|
| Intake and repository indexing | `core-selection-indexing`, installed MCP entrypoint, multimodal gate |
| Task understanding and query refinement | `core-selection-indexing`, SDK/content-aware tests |
| Evidence selection and compression | `sdk-receipts-compression`, native CI, pure-Python CI |
| Memory and session intelligence | `persistence-recovery-memory-value`, MemoryOS production gate |
| Trust, verification, and security | `verification-security`, Bandit, dependency audit |
| Exact recovery and receipts | installed MCP entrypoint, `sdk-receipts-compression`, persistence gate |
| Provider and gateway controls | `proxy-provider-gateway`, single-binary proxy CI |
| Learning and self-improvement | `learning-world-model`, standalone CogOps/federation contracts |
| Multimodal intake | `multimodal-image-planning` |
| Runtime and packaging | Python 3.10/3.12/3.14 entrypoints, public PyPI, Node/WASM/OpenClaw |
| Integrations and discovery | `integrations-discovery`, agent integration contracts |
| Observability and value accounting | `cli-observability`, persistence/value concurrency tests |
| Release and public claims | `public-trust-release-surfaces`, exact public artifact black-box |

## Adversarial conditions covered

The gate includes real stdio JSON-RPC, malformed requests, unknown methods,
pipelined calls, Unicode/emoji/RTL/combining characters, BOM and NUL recovery,
large MCP responses, unusable home directories, process restarts, corrupt state,
concurrent writers, lock contention, hostile numeric telemetry, invalid image
dimensions, multi-file diffs, Windows command-shim injection risk, Docker cache
corruption, offline image fallback, exact-version container selection, package
export disappearance, and strict JSON serialization.

## Merge boundary

Do not mark the PR ready or merge it based on this document. The merge boundary
is the final branch head with:

1. the deep-dogfood matrix complete,
2. the normal Python/Rust/native/fallback CI matrix complete,
3. package, integration, benchmark, footprint, and identity checks complete,
4. no unresolved P0/P1 finding,
5. no temporary repair workflow or script in the diff,
6. no claim that unreleased repository fixes already exist in public packages.
