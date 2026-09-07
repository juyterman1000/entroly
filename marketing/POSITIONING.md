# Positioning — frozen 2026-09-07

## The headline

> **Entroly — Cut AI context cost and prove nothing was lost.**

Subhead:

> Every selection emits a receipt: what was kept, what was omitted, and the
> handle that recovers the exact original bytes.

**Do not change this without reading the rest of this file.** The headline is
frozen deliberately, and the freeze is the point.

## Why it is frozen

The README headline changed roughly ten times between April and September 2026.
Star velocity tracked the drift, not the product:

| Period | Headline shape | Stars/week |
|---|---|---|
| April 2026 | Benefit-led, concrete, named the providers | ~148 |
| June–September 2026 | Abstract category framing — "Context OS", "Context Assurance", then a keyword list | 2–3 |

Between 2026-08-17 and 2026-09-06 the project took 364 commits and gained six
stars. The constraint was never product depth. Every rewrite reset whatever
recognition the previous wording had accumulated, and a reader who saw the
project twice under two different descriptions had no reason to connect them.

A headline compounds only if it stays still.

## Why this wording

It leads with the benefit — cost — because the version that ran at ~148
stars/week did, and the versions that drifted into category nouns did not.

It closes with the claim no competitor can copy. Compression is commodity;
every tool in this space shrinks a prompt. Entroly is the one that hands back a
receipt naming what it dropped and a handle that recovers the exact bytes. That
is checkable by a stranger in one command, which is what makes it survive a
skeptical reader instead of merely impressing a friendly one.

## The rule that constrains it

**No savings percentage. Not in the headline, not in the subhead, not in any
launch copy, not in a directory listing.**

`saved = max(0, baseline - selected_tokens)` with
`baseline = min(total_tokens, 32_000)` in `entroly/cli.py`. At a budget of
8,000 the figure is pinned at or above 75% *before selection has run*. It
measures the configured budget, not the selection. Any percentage derived from
it describes arithmetic, not a result.

`tests/test_first_run_copy.py` enforces this for the plugin's first-run screen.
Nothing enforces it for prose written by a human, which is why this file exists.

The same rule covers "zero accuracy loss" and equivalents. Verification in this
project fails closed by design; an unqualified accuracy claim contradicts the
architecture it is describing.

## What may change

Claims that are measured, sourced, and linked. `5,117/5,117 source spans
verified` and `13/13 exact SDK recoveries` are of this kind — they name a
denominator and point at the artifact that produced them. Prefer these.

## When to revisit

Not before 2027-01-01, and then only against measured evidence that this
wording underperforms — not against a feeling that it could be sharper. Record
the reasoning here rather than replacing the file, so the next revisit can see
the last one.
