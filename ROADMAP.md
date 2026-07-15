# Entroly Roadmap

The roadmap is organized by outcomes rather than promised dates. Reliability,
data integrity, and verifiable evidence take priority over feature count.
Proposals belong in [GitHub Discussions](https://github.com/juyterman1000/entroly/discussions);
accepted work should link an issue with an owner and measurable completion
criteria.

## Current priorities

### 1. Make every supported install boring

- Keep Python, Rust, npm/WASM, MCP, OpenClaw, Docker, binary, and Homebrew
  versions synchronized through one release manifest.
- Exercise fresh-install, update, rollback, and interrupted-update paths on
  Linux, macOS, and Windows.
- Keep marketplace listings verifiable without making external listing health a
  prerequisite for package installation.
- Report partial publication visibly and make reruns idempotent.

**Done when:** a new user can install, verify, integrate, and remove Entroly in
under five minutes on each supported platform, and a failed release cannot leave
ambiguous public versions.

### 2. Prove context quality, not only token reduction

- Expand preregistered, versioned holdouts across coding, structured data,
  retrieval, and long-session workloads.
- Publish raw per-case outputs, budgets, latency, model settings, costs, and
  confidence intervals.
- Separate lossless compression, task-conditioned selection, recoverability,
  and answer-quality results.
- Add regression thresholds that block claims when evidence becomes stale.

**Done when:** every prominent quantitative claim links to a reproducible
artifact and remains within its stated scope.

### 3. Make receipts operationally useful

- Improve omitted-evidence exploration, recovery diagnostics, and receipt
  portability.
- Make trust authority, model metadata, network use, and fail-open decisions
  visible in one place.
- Add stable schemas and migration tests for long-lived Context Commits and
  session receipts.

**Done when:** a power user can answer what the model saw, what it missed, why,
and how to recover it without reading implementation code.

### 4. Reduce maintenance risk

- Decompose the largest CLI, proxy, and MCP modules behind stable policy and
  adapter interfaces.
- Add coverage reporting for critical trust paths without turning percentage
  targets into a substitute for behavior tests.
- Pin or attest release dependencies and harden protected-branch policy.
- Expand dependency automation to every maintained Rust and npm workspace.

**Done when:** common provider, CLI, and release changes touch bounded modules,
and critical workflows have named owners and regression gates.

## Next outcomes

### Contributor scale

- Maintain a weekly triage queue with `good first issue`, `help wanted`, area,
  risk, and reproduction-status labels.
- Publish small extension examples and a reviewer checklist for each major
  subsystem.
- Recognize documentation, reproduction, review, and community support—not only
  code volume.

### Integration depth

- Keep first-party smoke tests for Claude Code, Codex, OpenClaw, MCP, and proxy
  paths.
- Add compatibility fixtures for popular agent frameworks where Entroly has a
  clear context-control boundary.
- Prefer provider-independent contracts over model-name conditionals.

### Documentation system

- Turn the current static site into a searchable documentation hub with version
  selectors, task-based navigation, API generation, and a public evidence index.
- Keep the README focused on decision, proof, install, and integration; move
  exhaustive reference material to maintained docs.

## Later exploration

- Signed, federated receipt verification across organizations.
- Reproducible policy packs for regulated or air-gapped deployments.
- Cross-language SDKs where usage evidence justifies their maintenance cost.
- Research prototypes that graduate only after falsification tests and stable
  fallbacks exist.

## Explicit non-goals

- Becoming a general-purpose chat client or agent host.
- Claiming universal quality improvements from one benchmark.
- Adding remote services that silently receive source code or prompts.
- Trading data integrity for a larger compression ratio.
- Supporting integrations without a maintainer, smoke test, and removal path.

## How work is selected

Maintainers score proposals on user impact, trust risk, reproducibility,
maintenance cost, interoperability, and whether a simpler solution exists.
Security, data-loss, install, and silent-failure regressions outrank roadmap
features. Governance is described in [GOVERNANCE.md](GOVERNANCE.md).
