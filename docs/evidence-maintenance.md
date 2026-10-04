# Maintaining public evidence

This guide is for maintainers publishing benchmark claims and updating public
pages. Reader-facing results and reproduction commands are in
[Public evidence](public-evidence.md).

## Quarantined public surfaces

Legacy savings, prompt-compression, hallucination, projected-dashboard, and stale setup pages remain `noindex` redirects until their claims and setup instructions are rebuilt. An HTTP-successful page is not sufficient evidence that its copy or runtime is current.

Archived translated READMEs must be regenerated from the canonical README, including trust links and caveats, before they return to primary navigation.

## Retired and republished public pages

A set of topic pages was reduced to `noindex` tombstones redirecting here,
because their claims could not be sourced: a universal 70–95% range, a 0.844
AUROC that a later tie-correction retired, equivalence conclusions drawn against
an API judge, and verifier coverage described as "every response".

Retirement removed those claims by removing the pages. It also removed every
entry point to the topics, and left the repository with no indexable answer to
questions Entroly can answer honestly.

Those pages were republished on 2026-08-15 under a stricter condition than the
one they failed: **every figure on a republished page must resolve to a
committed artifact under `benchmarks/results/`, and the page must state the
workload that produced it.** Where a benchmark set contains a loss case, the
page states it next to the wins rather than omitting it — the SQuAD 2.0 row
(43.8% savings, 90% retention on 233-token inputs) appears alongside the
long-context results it is worse than.

Retirement is no longer what keeps these pages honest. `STALE_PUBLIC_CLAIMS` in
`scripts/verify_context_assurance_public.py` is, and every republished page is
listed in `CLAIM_SENSITIVE_PUBLIC_FILES` so that scan applies to it. A page must
not be republished by deleting its retirement entry without adding it there.

`docs/dashboard.html` remains retired: it is an application view, not content.

## Maintainer rules

Before adding or strengthening a public claim:

1. Link the exact package, source, or result.
2. State the workload, model, budget, sample size, baseline, and caveats.
3. Keep different benchmark protocols separate.
4. Label estimates as estimates and provider-observed usage as observed.
5. Do not infer a marketplace score from repository-local evidence.
6. Remove or soften claims that cannot be reproduced.

Run:

```bash
python scripts/verify_public_trust.py
python scripts/verify_readme.py
```

Use `--online` only for bounded destination checks. After publication, `--require-published-version` can require PyPI and npm latest versions to match `server.json`.
