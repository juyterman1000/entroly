# Entroly LobeHub MCP Score Audit

This document records evidence for the LobeHub MCP score without treating an
external marketplace score as a substitute for product quality.

Target listing:

`https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score`

## Verified public baseline

A GitHub-hosted audit fetched and decoded the public page on 2026-07-13. The
listing currently reports:

- **45/100, grade F**;
- **2/4 required items**;
- version **0.4.0**, last updated **2026-05-14**;
- `isValidated=false` and `isClaimed=false`;
- `toolsCount=0`, `promptsCount=0`, and `resourcesCount=0`;
- two deployment methods (`python` and `manual`);
- a non-empty README; and
- stale MIT metadata even though the current repository is Apache-2.0.

The 45 points are exactly deployment (15), non-manual installation (12),
license detection (8), and README (10). Tools and validation are required and
are the two missing required items.

## Scoring contract and deduction map

LobeHub's current public implementation assigns 100 points:

| Criterion | Weight | Required | Public observation | Repository evidence | Classification/action |
| --- | ---: | :---: | --- | --- | --- |
| Claimed listing | 4 | No | False; page instructs owners to add the exact LobeHub GitHub badge and check claim status | Exact ownership badge added to the primary README on this branch | Legitimate ownership proof; public claim check still required |
| Non-manual deployment | 12 | No | Passing | `server.json` publishes PyPI/`uvx` and npm/`npx` paths | No product change required |
| Any deployment | 15 | Yes | Passing; two methods | Two canonical package deployment options | No product change required |
| Detected license | 8 | No | Passing but stale as MIT | Current repository and packages are Apache-2.0 | Metadata refresh required; current score already receives the points |
| MCP prompts | 8 | No | Zero | Context optimization and verification workflows | Genuine capability added and protocol-tested |
| README | 10 | Yes | Passing but stale | Current primary and package READMEs | Metadata refresh required; current score already receives the points |
| MCP resources | 8 | No | Zero | Bounded `entroly://health` and `entroly://stats` resources | Genuine read-only capability added and protocol-tested |
| MCP tools | 15 | Yes | Zero despite the existing production tool surface | FastMCP tools for optimization, exact recovery, receipts, memory, and verification | Stale/failed external validation; clean published-artifact validation required |
| Runtime validation | 20 | Yes | False | Stdio initialize/list/get/read/call protocol tests | External validation must be re-run after publishing |

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

Protocol tests prove that a fresh stdio server can:

1. initialize successfully;
2. list the two prompts;
3. render an optimization prompt;
4. list `entroly://health` and `entroly://stats`;
5. read and parse the bounded health resource; and
6. continue exposing the existing tool surface.

The repository audit script separately computes local readiness and labels it
as local evidence rather than the public score. With protocol validation in the
same CI run, the repository is capable of 96/100 before the external claim
point. That number is **not** a claim about the current LobeHub page.

## Remaining external gates

The work is not complete until all of these are visibly confirmed:

- publish a patch release containing the prompts, resources, tests, and badge;
- refresh LobeHub's stale 0.4.0 metadata to the new release;
- make LobeHub install and start the published package successfully;
- verify non-zero tool, prompt, and resource counts;
- confirm `isValidated=true`;
- invoke LobeHub's badge-based claim check and confirm `isClaimed=true`; and
- capture the resulting public score page.

No release note or PR may claim an improved LobeHub score before that public
confirmation.
