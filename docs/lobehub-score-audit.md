# Entroly LobeHub MCP Score Audit

This document records evidence for the LobeHub MCP score without treating an
external marketplace score as a substitute for product quality.

Target listing:

`https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score`

## Scoring contract

LobeHub's current public implementation assigns 100 points:

| Criterion | Weight | Required | Entroly evidence | Classification | Current state |
| --- | ---: | :---: | --- | --- | --- |
| Claimed listing | 4 | No | LobeHub's detail score currently hardcodes `isClaimed: false` | External implementation limitation | Blocked externally |
| Non-manual deployment | 12 | No | `server.json` publishes PyPI/`uvx` and npm/`npx` install paths | Packaging/discovery | Repository-ready; public ingestion pending |
| Any deployment | 15 | Yes | Two canonical package deployment options in `server.json` | Packaging/discovery | Repository-ready; public ingestion pending |
| Detected license | 8 | No | Apache-2.0 repository and package metadata | Metadata | Repository-ready; public detection pending |
| MCP prompts | 8 | No | Context optimization and verification workflows | Product capability | Implemented on this branch; protocol verification required |
| README | 10 | Yes | Repository and package READMEs | Documentation | Repository-ready; public ingestion pending |
| MCP resources | 8 | No | Bounded health and aggregate-stat resources | Product capability/security | Implemented on this branch; protocol verification required |
| MCP tools | 15 | Yes | Production FastMCP tool surface including optimization, recovery, receipts, and verification | Product capability | Existing; protocol tests present |
| Runtime validation | 20 | Yes | Clean-install startup plus MCP initialize/list/call protocol tests | External validation | Local readiness testable; LobeHub result pending |

A required criterion missing causes grade F regardless of percentage. If all
required criteria pass, 80% or higher is A, 60-79% is B, and lower is F.

## Release isolation

The ClawHub v1.0.54 install-metadata correction was completed in PR #124. Its
merge commit has a successful `ClawHub Public Listing` status for
`entroly-openclaw` v1.0.54. This LobeHub work remains isolated in PR #125 and
must not rewrite the completed v1.0.54 release.

## Legitimate remediation

The added MCP prompts are reusable workflows backed by real Entroly tools. The
added resources expose only bounded, read-only operational summaries. They do
not expose source content, receipt bodies, filesystem paths, credentials, or
unbounded logs.

The protocol tests must prove that a fresh stdio server can:

1. initialize successfully;
2. list the two prompts;
3. render an optimization prompt;
4. list `entroly://health` and `entroly://stats`;
5. read and parse the bounded health resource; and
6. continue exposing the existing tool surface.

## External blockers

The following cannot be honestly inferred from repository code:

- the exact score currently rendered by LobeHub;
- whether LobeHub's crawler has refreshed the latest package metadata;
- whether LobeHub has completed runtime validation;
- whether the listing's deployment options were normalized correctly; and
- the claimed-listing point while the detail page hardcodes the claim flag off.

These remain `pending` until the public score page or marketplace API visibly
confirms them. No release note or PR may claim an improved LobeHub score before
that confirmation.
