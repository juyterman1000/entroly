# AI Traffic Value

Entroly exposes an executive value view at:

```text
http://127.0.0.1:9377/traffic-value
```

The machine-readable surface is:

```text
GET /traffic-value.json
```

The page is designed to answer three questions immediately:

1. **Is Entroly helping me in the work session I am in right now?**
2. **What did Entroly do to my AI traffic in this recent period?**
3. **How much value has Entroly accumulated for me overall?**

## This session

`This session` is a process-local, in-memory rollup built natively in the same Traffic Value accounting path as the durable dashboard. It consumes the same verified Traffic Receipt stream; there is no session monkey patch, second savings ledger, or second database.

After a handful of requests, a user can immediately see the conservative net saving and how many requests it covers, requests optimized, tokens received/sent/avoided, context reduction, the list-price model when model pricing is available, the provider's cache discount when provider evidence is available, warm-cache protection, and verification/recovery evidence.

The session view is deliberately non-durable and resets when the proxy process restarts. It is **not** added again to All Time, so showing the immediate number cannot double-count durable value.

During the first day of Traffic Value collection, once the current process has observed traffic, `This session` becomes the default view. After that, the durable age-adaptive default below takes over while the session tab remains available.

The dashboard reports **Session duration**, not "working time": it measures wall-clock time since the current proxy process session began and does not pretend to infer human activity.

## Rolling executive view

The selected durable period can be Today, 7 days, 30 days, 60 days, or 90 days. The period card can show:

- requests optimized;
- tokens received by Entroly before its context optimization;
- tokens sent on the final observed provider-bound request;
- tokens avoided and context-reduction percentage;
- estimated input value avoided;
- measured cache benefit when provider usage and auditable pricing exist;
- provider input spend observed when provider usage and auditable pricing exist;
- requests with explicit verification evidence and verification pass rate;
- recovery evidence and final PASS rates;
- warm-cache tokens protected;
- observed cache-hit request rate;
- total AI value protected when the required dollar evidence is available.

The rolling windows do not reset at calendar week/month boundaries.

## Adaptive default

After the first day, Entroly selects a useful recent window based on how long Traffic Value has actually been collecting receipts:

```text
< 30 days collecting     -> 7D
30-59 days collecting    -> 30D
60-99 days collecting    -> 60D
100+ days collecting     -> 90D
```

So an install with 3 days of history opens on 7D, one with 45 days opens on 30D, and one with 100+ days opens on 90D. `This session` remains available in every case.

## All Time never disappears

The selected session/recent period is always accompanied by a separate **ALL TIME** card. It highlights:

```text
Net saving (conservative)                       <- the headline
List-price model (ignores caching)
Provider cache discount (measured)
Modeled total (list price + cache discount)
Tokens avoided
Requests optimized
Warm-cache tokens protected
```

All Time persists in the existing `ValueTracker` file across proxy restarts and keeps accumulating until the user resets or deletes that telemetry.

## Evidence contract

The dashboard intentionally keeps evidence classes visible instead of turning every number into a generic "savings" claim.

**Net saving (conservative)** is the headline, and the only figure presented as what Entroly saved. It is the signed sum of `net_saving_conservative_micro_usd` over the Traffic Receipts that carry one: removed content priced at the cheapest rate the provider could have billed it (a cache read if the content had been sent before), minus what Entroly added and any cached prefix it broke at the dearest rate. It is shown with its coverage, "priced N of M requests". A request without provider usage or without a verified price for its model is counted as unpriced and adds nothing; a window with no priced request shows no dollar figure at all, and the list-price model is never substituted. It can be negative. It is an estimate calibrated to the provider's token counts, input side only, not an invoice. API: `headline` and `net_saving_conservative_usd` on every window, `headline_metric` on the snapshot. See the [Traffic Receipts truth contract](traffic-receipts.md#truth-contract) for the pricing rules and the bundled price tables.

**List-price model (ignores caching)** (`estimated_value_avoided_usd`, formerly labelled "Estimated value avoided") uses the locally observed token difference on a provider-bound request multiplied by the configured input price for the executed model. It is modeled economic value, not an invoice counterfactual: in a cached conversation most removed tokens would have been billed as cache reads, at a tenth to a half of this rate, so the model overstates the bill difference. It is shown for comparison and is not the headline. If the executed model has no auditable configured price, the UI shows the dollar value as unavailable rather than pretending that an unknown value is `$0.00`.

**Provider cache discount (measured)** (`measured_cache_benefit_usd`, formerly "Measured cache benefit") is the provider's own discount, not Entroly's saving. It appears only when the Traffic Receipt has provider-reported cache usage plus auditable pricing. Missing provider/pricing evidence is displayed as unavailable, not as a measured zero.

**Provider input spend observed** prices the provider-reported input/cache token categories currently carried by Traffic Receipt v1. It is intentionally not labeled full provider invoice spend because output cost is not yet part of that receipt field.

**Modeled total (list price + cache discount)** (`total_ai_value_protected_usd`, formerly "Total AI value protected") is the sum of the list-price model and the provider's cache discount when the required evidence for those components is available. It is a modeled total and is not the headline. The two components remain visible separately so an operator can distinguish modeled optimization value from provider-observed cache economics.

Verification percentages count only explicit PASS/FAIL evidence. Traffic Receipt v1 currently exposes recovery evidence rather than a dedicated recovery-invocation event, so the dashboard labels that evidence conservatively; final PASS is additionally required for the corresponding successful-recovery evidence metric.

Traffic Receipt coverage that is only available through shared mutable proxy state is withheld from per-request receipts rather than risk cross-request attribution. The receipt UI labels observed recovery state as **Recovery evidence**, and its SHA-256 check as **Receipt integrity**, not as a cryptographic signature.

The dashboard persists only aggregate counters and bounded receipt identifiers in the existing value-tracker file. It does not add prompt, tool-output, code, API-key, or raw user-agent content to the executive rollups.
