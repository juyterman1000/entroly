---
name: entroly-lobehub-audit
description: Audit and remediate Entroly's MCP marketplace quality with evidence, adversarial validation, and no score gaming.
status: active
---

# Entroly LobeHub MCP Audit

## Mission

Audit Entroly's complete LobeHub MCP score at:

`https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score`

Act as a senior open-source product architect, MCP engineer, security engineer,
Rust/Python/TypeScript developer, and release engineer.

## Non-negotiable execution contract

1. Read every score category, deduction, warning, and missing requirement.
2. Map every deduction to the exact repository file, runtime behavior, external
   index state, or genuinely missing capability.
3. Classify every finding as one of:
   - product or security defect;
   - packaging or discovery defect;
   - missing documentation;
   - stale external indexing;
   - criterion irrelevant to Entroly's context-control category.
4. Improve only legitimate weaknesses. Never add empty keywords, fabricated
   evidence, fake benchmarks, or unnecessary features merely to game a score.
5. Preserve Entroly's positioning as the auditable context, memory, and
   verification control plane for AI agents.
6. Implement production-quality fixes with tests, schemas, validation,
   security controls, documentation, and reproducible evidence.
7. Add CI regression gates so corrected findings cannot return.
8. Run the official MCP, npm, PyPI, OpenClaw/ClawHub, and LobeHub-relevant
   validation paths.
9. Open a focused PR with a deduction-by-deduction remediation table.
10. Merge only after all checks pass, publish the necessary patch release, and
    verify the public LobeHub page after its index refresh.
11. Never claim the score improved until the public page visibly confirms it.

## Priority and release isolation

1. Finish and publicly verify the ClawHub v1.0.54 metadata correction.
2. Keep LobeHub remediation in a separate branch, PR, and release.
3. Do not mix unrelated product work into marketplace remediation.

## LobeHub score registry

The current public LobeHub implementation assigns 100 total points:

| Criterion | Weight | Required |
| --- | ---: | :---: |
| Claimed listing | 4 | No |
| Non-manual deployment | 12 | No |
| Any deployment | 15 | Yes |
| Detected license | 8 | No |
| MCP prompts | 8 | No |
| README | 10 | Yes |
| MCP resources | 8 | No |
| MCP tools | 15 | Yes |
| Runtime validation | 20 | Yes |

All required criteria must pass. With all required criteria present, 80% or
higher is grade A, 60-79% is grade B, and lower is grade F.

## Evidence registry

Maintain a table for each run:

| Criterion | Public observation | Repository evidence | Classification | Action | Test | External verification |
| --- | --- | --- | --- | --- | --- | --- |

Never infer an external success from local code alone. Mark external-only
results as `pending`, `blocked`, or `confirmed` with a direct artifact,
registry response, public page, or screenshot.

## Search and implementation loop

1. Collect independent evidence from the public listing, repository, package
   registries, official specifications, and executable protocol probes.
2. Keep distinct hypotheses separate until each has concrete evidence.
3. Mark routes blocked when they depend on an unavailable external refresh or
   an unproved assumption; do not disguise a blocked route as progress.
4. Implement the smallest real fix that improves users' ability to install,
   understand, validate, or safely use Entroly.
5. Add adversarial tests for malformed inputs, bounded output, path safety,
   secret leakage, prompt injection, protocol compatibility, clean installs,
   and stale metadata.
6. Re-run local validators, package dry-runs, protocol smoke tests, and the full
   CI matrix.
7. Publish only after all gates are green.
8. Re-check the exact public listing and record the observed score and flags.

## Adversarial review checklist

Reject any candidate remediation that:

- changes only wording without creating the claimed capability;
- exposes secrets, unrestricted files, unbounded receipts, or unsafe paths;
- adds MCP prompts or resources that are duplicates or decorative;
- relies on a local editable install but fails from the published artifact;
- reports validation without starting the server and listing its capabilities;
- confuses MCP Registry, LobeHub, ClawHub, npm, and PyPI indexing states;
- claims a marketplace score before the marketplace confirms it;
- broadens Entroly into an unrelated framework merely to satisfy a directory.

## Completion condition

The task is complete only when:

- every legitimate deduction has a tested remediation or a documented external
  blocker;
- the focused PR is green and merged;
- required artifacts are published and installable;
- the public LobeHub score page visibly reflects the new state; and
- the final report distinguishes confirmed improvements from unresolved
  external indexing.
