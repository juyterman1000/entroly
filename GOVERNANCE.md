# Project Governance

Entroly currently uses a maintainer-led, evidence-driven governance model. The
goal is to make decisions predictable while the contributor base grows.

## Roles

- **Users** run Entroly and provide reproducible feedback.
- **Contributors** improve code, tests, documentation, benchmarks, or community
  support through reviewed contributions.
- **Reviewers** have demonstrated sound judgment in one or more project areas
  and may review changes without having merge access.
- **Maintainers** can merge changes, manage releases, handle security reports,
  and enforce community standards.

Current maintainers and ownership areas are listed in
[MAINTAINERS.md](MAINTAINERS.md) and [`.github/CODEOWNERS`](.github/CODEOWNERS).

## Decision principles

1. Reliability and data integrity outrank feature breadth.
2. Public claims require reproducible evidence and stated limitations.
3. Backward compatibility is preserved when practical.
4. Security, receipt honesty, recoverability, local-first behavior, provider
   compliance, and release consistency are non-negotiable invariants.
5. Decisions should be recorded in an issue, discussion, pull request, or RFC
   rather than private chat.

Routine changes are decided through pull-request review. Major architecture,
governance, data-format, network-boundary, or compatibility changes begin with a
GitHub Discussion or RFC and remain open long enough for affected users to
respond. Maintainers seek consensus; when consensus is not possible, the lead
maintainer decides and records the rationale.

## Becoming a reviewer or maintainer

There is no contribution-count threshold. Candidates should demonstrate:

- sustained, constructive participation;
- careful handling of trust and compatibility boundaries;
- high-quality reviews or support, not only code volume;
- respect for the Code of Conduct;
- willingness to maintain existing behavior after launch.

A maintainer nominates the candidate in a public Discussion. Existing
maintainers approve the role by consensus and update `MAINTAINERS.md` and
`CODEOWNERS`. Inactive maintainers may move to emeritus status after six months
without project activity, following private outreach and a public record of the
role change.

## Releases and security

Maintainers follow the release gates in [CLAUDE.md](CLAUDE.md), keep all package
surfaces version-aligned, and publish from protected automation. Security
reports follow [SECURITY.md](SECURITY.md) and may be handled privately until a
coordinated fix is available.

## Changes to governance

Governance changes require a pull request, a linked public Discussion, and
approval from all active maintainers. The change must explain its effect on
contributors and include a rollback or transition plan.
