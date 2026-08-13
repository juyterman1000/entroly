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

`This session` is an in-process rollup built from the same verified Traffic Receipt stream as the durable dashboard. It is intended for the first-use wow moment: after a handful of requests, a user can immediately see requests optimized, tokens received/sent/avoided, context reduction, estimated value avoided, measured cache benefit when available, warm-cache protection, verification, recovery, and Total AI value protected.

The session view is deliberately non-durable and resets when the proxy process restarts. It is **not** added again to All Time, so showing the immediate number cannot double-count durable value.

During the first day of Traffic Value collection, once the current process has observed traffic, `This session` becomes the default view. After that, the durable age-adaptive default below takes over while the session tab remains available.

## Rolling executive view

The selected durable period can be Today, 7 days, 30 days, 60 days, or 90 days. The period card can show:

- requests optimized;
- tokens received by Entroly before its context optimization;
- tokens sent on the final observed provider-bound request;
- tokens avoided and context-reduction percentage;
- estimated input value avoided;
- measured cache benefit when provider usage and auditable pricing exist;
- provider input spend when provider usage and auditable pricing exist;
- requests with explicit verification evidence and verification pass rate;
- recovery invocation and success rates;
- warm-cache tokens protected;
- observed cache-hit request rate;
- total AI value protected.

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
Estimated value avoided
Measured cache benefit
Total AI value protected
Tokens avoided
Requests optimized
Warm-cache tokens protected
```

All Time persists in the existing `ValueTracker` file across proxy restarts and keeps accumulating until the user resets or deletes that telemetry.

## Evidence contract

The dashboard intentionally keeps evidence classes visible instead of turning every number into a generic "savings" claim.

**Estimated value avoided** uses the locally observed token difference on a provider-bound request multiplied by the configured input price for the executed model. It is modeled economic value, not an invoice counterfactual.

**Measured cache benefit** appears only when the Traffic Receipt has provider-reported cache usage plus auditable pricing.

**Provider input spend** prices the provider-reported input/cache token categories currently carried by Traffic Receipt v1. It is intentionally not labeled full provider invoice spend because output cost is not yet part of that receipt field.

**Total AI value protected** is the sum of avoided-input value and measured cache benefit. The two components remain visible separately so an operator can distinguish modeled optimization value from provider-observed cache economics.

Verification percentages count only explicit PASS/FAIL evidence. Recovery success requires recovery evidence plus a final explicit PASS signal; an attempted recovery is never counted as successful merely because it ran.

Traffic Receipt coverage that is only available through shared mutable proxy state is withheld from per-request receipts rather than risk cross-request attribution. The receipt UI labels observed recovery state as **Recovery evidence**, and its SHA-256 check as **Receipt integrity**, not as a cryptographic signature.

The dashboard persists only aggregate counters and bounded receipt identifiers in the existing value-tracker file. It does not add prompt, tool-output, code, API-key, or raw user-agent content to the executive rollups.
