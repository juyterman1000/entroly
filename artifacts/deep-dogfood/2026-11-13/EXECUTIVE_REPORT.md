# Entroly deep-dogfood evidence — 2026-11-13

## Scope and attribution

Baseline: `main` at `1de499a58835f13eba923a89a8020a9e43eed92e`, after PR #587 and PR #588 merged. This is an **evidence inventory and partial investigation**, **not** a claim that all ten deep-dogfood phases were fully executed.

| Phase | Status | Evidence / limitation |
| --- | --- | --- |
| 0 — establish truth | VERIFIED REMOTELY | Baseline SHA, connected repository metadata, `AGENTS.md`, `CLAUDE.md`, code paths and workflow definitions inspected |
| 1 — clean install | CI-VERIFIED ONLY | Installed MCP entrypoint tests passed for Python 3.10/3.12/3.14; no independent consumer-machine installation performed |
| 2 — real agent workloads | NOT EXECUTED | No controlled no-Entroly vs default vs tuned trials with live agents |
| 3 — proxy fidelity | SYNTHETIC CI-VERIFIED | PR #587 file-read, latency and cache regression tests; no independent live provider replay in this audit |
| 4 — pricing/ROI | PARTIAL | Bundled provider tables and cost-model code inspected; no invoice-backed A/B cost comparison |
| 5 — adversarial trust | PARTIAL + FIX PROPOSED | Nonfinite/malformed pricing catalog rate failure identified and bounded regression tests added |
| 6 — complete code audit | NOT COMPLETED | Focused ledger/catalog review only, not a repository-wide formal audit |
| 7 — fixes | IN PR | Reject nonfinite catalog rates and maintain per-provider fail-closed isolation |
| 8 — readiness gates | BASELINE CI GREEN | Post-merge `main` CI passed; branch CI must be evaluated separately before merge |
| 9 — artifacts | RECORDED | This evidence directory, workload matrix, trust audit, cost/quality, defect register and reproduction steps |
| 10 — delivery | IN PR | Focused fix and audit evidence; no automatic merge requested |

## Verified GitHub results

- Post-merge CI for `main`: 12/12 jobs successful, [workflow run](https://github.com/juyterman1000/entroly/actions/runs/43425789632).
- PR #588 deep-dogfood runtime: 16/16 jobs successful on `e39802f94a71eec005cf050bc0c1c281ff543c48`, [workflow run](https://github.com/juyterman1000/entroly/actions/runs/43419434393).
- PR #588 pre-merge checks: 41 success, 1 skipped, no failure.
- Production billing, task success, and independent source-fidelity measurements are *not established* by those checks.

## Architecture read

Follow the supported entrypoints before relying on module-level claims: `entroly` console entrypoint, `entroly proxy`, SDK, MCP, OpenClaw bridge, Rust core, receipt ledger and Traffic Value projection. `AGENTS.md` and `CLAUDE.md` require truthful receipts, exact recovery, fail-closed verification and no surprise network calls.

## Environment limits

The analysis environment had connected read/write GitHub access but could not resolve `github.com` for a local `git clone`. No live provider API credentials, provider invoices, independent end-user installs, or paired live agent traces were available. Do not reinterpret missing evidence as PASS.

## Decision

Current verdict: **CONDITIONAL / UNVERIFIED FOR REAL-WORLD ROI**. Baseline CI passed. The pricing parser defect is reproducible by Python's decimal behavior and addressed by regression tests in this PR. A fresh branch CI pass is required; independent live A/B quality, economic and end-to-end recovery trials are still outstanding.

See [WORKLOAD_MATRIX.md](WORKLOAD_MATRIX.md), [COST_AND_QUALITY.md](COST_AND_QUALITY.md), [TRUST_AUDIT.md](TRUST_AUDIT.md) and [REPRODUCE.md](REPRODUCE.md).
