# Entroly LobeHub MCP Score Audit

This document records evidence for the LobeHub MCP score without treating an
external marketplace score as a substitute for product quality.

Target listing:

`https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score`

## Verified public baseline

A GitHub-hosted audit fetched and decoded the public page on 2026-07-13. The
listing reported:

- **45/100, grade F**;
- **2/4 required items**;
- version **0.4.0**, last updated **2026-05-14**;
- `isValidated=false` and `isClaimed=false`;
- `toolsCount=0`, `promptsCount=0`, and `resourcesCount=0`;
- two deployment methods (`python` and `manual`);
- a non-empty README; and
- stale MIT metadata even though the current repository is Apache-2.0.

The 45 points were deployment (15), non-manual installation (12), license
detection (8), and README (10). Tools and validation are required and were the
two missing required items.

## Scoring contract and deduction map

LobeHub's public implementation assigns 100 points:

| Criterion | Weight | Required | Public observation | Repository evidence | Classification/action |
| --- | ---: | :---: | --- | --- | --- |
| Claimed listing | 4 | No | False; the page requests its exact GitHub badge before claim verification | Exact LobeHub ownership badge is committed in the primary README | Legitimate ownership proof; public claim check remains external |
| Non-manual deployment | 12 | No | Passing | `server.json` declares PyPI/`uvx` and npm/`npx` install paths | No product change required |
| Any deployment | 15 | Yes | Passing; two methods | Two canonical package deployment options | No product change required |
| Detected license | 8 | No | Passing but stale as MIT | Current repository and package metadata are Apache-2.0 | External metadata refresh required; points already awarded |
| MCP prompts | 8 | No | Zero | Context-optimization and evidence-verification workflows | Genuine capability added and protocol-tested |
| README | 10 | Yes | Passing but stale | Current primary and package READMEs | External metadata refresh required; points already awarded |
| MCP resources | 8 | No | Zero | Bounded `entroly://health` and `entroly://stats` resources | Genuine read-only capability added and protocol-tested |
| MCP tools | 15 | Yes | Zero despite the existing production tool surface | FastMCP tools for optimization, exact recovery, receipts, memory, and verification | Stale or failed external validation; published-artifact validation required |
| Runtime validation | 20 | Yes | False | Stdio initialize/list/get/read/call protocol tests | LobeHub must re-run validation after publication |

A required criterion missing causes grade F regardless of percentage. If all
required criteria pass, 80% or higher is A, 60–79% is B, and lower is F.

## Release isolation

The ClawHub v1.0.54 install-metadata correction was completed in PR #124. Its
merge commit has a successful `ClawHub Public Listing` status for
`entroly-openclaw` v1.0.54. This LobeHub work remains isolated in PR #125 and
does not rewrite the completed v1.0.54 release.

## Legitimate remediation

The added MCP prompts are reusable workflows backed by real Entroly tools. The
added resources expose only bounded, read-only operational summaries. They do
not expose source content, receipt bodies, filesystem paths, credentials, or
unbounded logs.

A clean Python 3.12 validation job installed Entroly from the branch and proved
that a fresh stdio server can:

1. initialize successfully;
2. list and render both prompts;
3. list `entroly://health` and `entroly://stats`;
4. read and parse the bounded health resource;
5. continue exposing the existing tool surface; and
6. pass the local score-evidence regression tests.

The repository audit script separately computes local readiness and labels it
as local evidence rather than the public score. With protocol validation in the
same run, the repository has evidence for 96/100 before the external claim
point. This is **not** the current public LobeHub score.

## Remaining external gates

The work is not complete until all of these are visibly confirmed:

- merge the focused, fully green PR;
- publish a patch release containing the prompts, resources, tests, and badge;
- refresh LobeHub's stale 0.4.0 metadata to the new release;
- make LobeHub install and start the published package successfully;
- verify non-zero tool, prompt, and resource counts;
- confirm `isValidated=true`;
- invoke LobeHub's badge-based claim check and confirm `isClaimed=true`; and
- capture the resulting public score page.

No release note or PR may claim an improved LobeHub score before that public
confirmation.
