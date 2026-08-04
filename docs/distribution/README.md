# Entroly distribution control plane

This directory is the source of truth for Entroly's external discovery, listing,
review, benchmark, and package-manager work.

Distribution claims are product claims. They must be as auditable as Context
Receipts: every published listing needs a public proof URL, every benchmark
needs a version-pinned adapter and unchanged protocol, and every package-manager
entry needs a reproducible install path.

## Files

- [`targets.json`](targets.json) — machine-readable status registry.
- [`submission-kit.md`](submission-kit.md) — canonical, evidence-bounded copy for
  directories, newsletters, guides, reviewers, and benchmark maintainers.
- [`../../scripts/check_distribution_surface.py`](../../scripts/check_distribution_surface.py)
  — local validation for version alignment, required discovery assets, target
  states, and proof URLs.

## Status rules

- **prepared** means Entroly has the required repository-local assets. It does
  not mean an external submission exists.
- **submitted** requires the external pull request or issue URL to be recorded
  as `proof_url`.
- **published** requires a currently visible third-party entry recorded as
  `proof_url`.
- **blocked** requires a concrete missing prerequisite and next action.
- **rejected** requires the external decision URL.

Never mark a target submitted or published from a search snippet, copied list,
maintainer intention, or unpublished draft.

## Operating sequence

1. Run `python scripts/check_distribution_surface.py`.
2. Confirm the target's repository rules and category fit.
3. Use the shortest tailored copy from `submission-kit.md`.
4. Remove claims the target cannot independently verify.
5. Open one focused external contribution.
6. Record the public URL and update the target status.
7. Re-check the listing after upstream edits or major Entroly releases.

## Priority model

Priority 1 closes direct product-discovery gaps for Claude Code and MCP users.
Priority 2 expands neutral evaluation and agent-infrastructure discovery.
Priority 3 adds operating-system and package-manager distribution only after the
required artifacts are reproducible on that platform.

Package-manager reach and benchmark inclusion are not marketing checkboxes.
They are maintained product surfaces with support and compatibility obligations.

## Claim policy

Allowed without qualification:

- Entroly is open source under Apache-2.0.
- Entroly provides CLI, Python SDK, MCP, proxy, Node/WASM, Rust, Docker, and
  Homebrew paths when the linked release surface is current.
- Entroly provides budget-aware evidence selection, recoverable compression,
  Context Receipts, and verification.
- Indexing, selection, compression, receipt generation, and default verification
  run locally; proxy mode forwards the selected request to the user's configured
  model provider.

Claims requiring linked evidence and workload caveats:

- token reduction percentages;
- answer-quality retention or improvement;
- latency comparisons;
- competitor comparisons;
- download, star, install, or user counts;
- production-readiness or universal compatibility statements.

Do not use "best", "leading", "zero loss", "guaranteed savings", or universal
superiority language in external submissions.

## Definition of done

The distribution program is healthy when:

- every high-priority target has an honest status and next action;
- Claude, MCP, CLI, local-first, benchmark, and package-manager discovery each
  have maintained canonical metadata;
- external listings resolve to current install commands;
- neutral benchmark adapters stay version-pinned and runnable;
- no release can silently drift the plugin, MCP, package, or submission copy.
