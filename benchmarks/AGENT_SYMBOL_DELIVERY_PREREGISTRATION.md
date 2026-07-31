# Preregistration — Intact symbol delivery to a coding agent

Written **before** the experiment was run, per the research contract. The claim
language permitted for each outcome is fixed here so it cannot be chosen after
seeing the numbers.

---

> ## Amendment 1 — the original primary metric was mis-specified
>
> **This amendment was made after seeing a first result, which is exactly the
> pattern that produces p-hacking. It is recorded in full so a reviewer can
> judge it rather than take it on trust.**
>
> The first run reported an "intact delivery rate" of 54.1% (742 intact, 629
> altered, n=1638). The original classifier called a task `altered` when the
> symbol's *name* reached the agent but its complete source did not, and treated
> that as corruption.
>
> That classification is wrong on its face, and the diagnostic is mechanical
> rather than a judgement call: for every sampled `altered` case, **every
> delivered fragment was verbatim present in the source file**. Nothing was
> corrupted. One of the sampled symbols is 839 estimated tokens against a
> 400-token budget, so delivering it whole is arithmetically impossible.
>
> The original metric therefore measured *symbol coverage under a budget*, a
> property of chunk granularity, while claiming to measure *corruption*, a
> property of fidelity. It would have reported a fidelity failure on a run where
> fidelity was provably perfect.
>
> **What changed:** `altered` is split into `partial` (fragments all real, but
> not covering the whole symbol — not a failure) and `corrupted` (a fragment
> that appears nowhere in the file — the actual defect). The primary metric
> becomes `uncorrupted_delivery_rate = (complete + partial) / delivered`.
>
> **What did not change:** the corpus, the task set, the queries, the budget, the
> backend, and the rule that the oracle is the original file bytes. The
> amendment reclassifies outcomes; it does not re-select data, and no task was
> added or dropped.
>
> **Why this is not metric-shopping:** the correction was forced by a
> demonstrable logical error, not by the direction of the number, and it makes
> the metric *harder* to move — under the corrected definition the repaired
> implementation can only score 100% if literally nothing corrupt is delivered.
> A reviewer who disagrees can recompute both metrics from the committed
> per-task outcomes, which are recorded in full in the artifact.
>
> Schema moves from `agent-symbol-delivery.v1` to `.v2`. The v1 result is
> superseded and must not be cited.

---

## Why this experiment exists

Every existing benchmark in this repository measures mechanism: retrieval
recall, token ratios, passage counts, fragment fidelity. None answers the
question a developer actually has, which is whether the code their agent
receives is usable.

This experiment does not measure task success either — that needs a model in the
loop and is preregistered separately. It measures the step immediately before
task success, and the one where the receipt defect did its damage: **when the
system decides to hand the agent a function, does the agent receive that
function, or a corrupted copy of it?**

A corrupted copy is worse than a miss. A miss is visible — the agent says it
lacks context. A corrupted function is invisible: it looks like code, so the
agent reasons about it and edits against source that does not exist in the
repository.

## Hypothesis

**H1.** Addressing fragments by byte range rather than rebuilding their text
increases the proportion of delivered symbols that arrive unaltered, at a
matched token budget and with retrieval behaviour otherwise unchanged.

**H0 (null).** The intact-delivery rate is unchanged.

The comparison is paired: the same corpus, the same tasks, the same budget, the
same backend, differing only in the ingest implementation.

## Conditions

| Condition | Implementation |
|---|---|
| `before` | `entroly/context_receipts/` as of `1ecf1e0` (entroly 1.0.69, pre-repair) |
| `after` | the repaired implementation at the measured commit |

Both run with `prefer_rust=False`. The native backend is compared separately in
`tests/test_receipt_backend_parity.py`; holding the backend fixed here isolates
the algorithm change rather than confounding it with a wheel rebuild.

## Corpus

Deterministic, no sampling, no manual curation:

- Python files tracked at `1ecf1e0` (`git ls-tree -r`), sorted lexicographically
- at most 400,000 bytes, decoding as strict UTF-8
- parseable by `ast.parse` (a file that does not parse cannot yield a symbol)
- containing at least one qualifying symbol

Reading from a pinned ref keeps the corpus immutable and platform-independent:
git stores LF, so a CRLF checkout cannot change the result.

## Task generation

For each file, every top-level `def`, `async def` and `class` whose exact source
segment (`ast.get_source_segment`) is at least 20 estimated tokens, in source
order, capped at 3 per file to stop large modules dominating the denominator.

The **query** is derived only from the symbol's public identity, never from its
body: the symbol name split on underscores, plus the first line of its docstring
when present. Using body text would leak the answer into the query.

**Budget:** 400 tokens, fixed. Small enough that selection must choose, large
enough to admit a typical function.

## Outcome classification

Each task lands in exactly one bucket, decided against the **original file
bytes**, never against the system's own output:

| Bucket | Meaning |
|---|---|
| `intact` | the symbol's exact source appears verbatim in the delivered context |
| `altered` | the symbol's name appears in the delivered context but its exact source does not — the agent received a corrupted version |
| `absent` | neither appears; retrieval did not select this symbol |

`absent` is **not** counted as a failure. A retrieval backend may honestly
decide a symbol is not relevant, and penalising that would push the metric
toward indiscriminate inclusion. Only `altered` is a fidelity failure.

## Metrics

**Primary:** `intact_delivery_rate = intact / (intact + altered)` — of the
symbols the system chose to deliver, the proportion that arrived unaltered.

**Secondary:** `altered_rate` over all tasks; `delivery_rate`
(`(intact + altered) / tasks`), reported to show that the primary metric's
denominator did not shift between conditions.

## Statistics

- Wilson score 95% confidence interval on the primary rate in each condition.
- Exact McNemar test on the paired per-task outcome, with the p-value reported
  whatever it shows.
- Fixed corpus, no interim analysis, no stopping rule, no exclusions beyond
  those listed above. Every excluded file is recorded with its reason.

## Claim language fixed in advance

- **If `after` > `before` with McNemar p < 0.05:** "At a matched 400-token
  budget across N symbol tasks, the repaired implementation delivered the
  requested function unaltered in X% of deliveries against Y% before, p = Z."
- **If the interval includes no difference, or p ≥ 0.05:** report a statistical
  tie. Do not describe the change as an improvement.
- **If `after` < `before`:** report the regression prominently and do not ship.

In every case the result is reported as **evidence delivery**, never as task
success, agent accuracy, cost saving, or developer productivity. Those require a
model in the loop and are not measured here.

## Known limitations, stated before results

- Symbol retrieval is a proxy for real work. A developer's task is rarely "fetch
  this exact function".
- Queries are synthetic, derived from the symbol's own name and docstring. A
  real developer question is messier and less well matched to the target.
- One repository, one language. Results may not transfer to other codebases.
- No model is invoked, so nothing here shows the agent produced a better answer.
