# AI Traffic Value

Entroly exposes an executive value view at:

```text
http://127.0.0.1:9377/traffic-value
```

The machine-readable surface is:

```text
GET /traffic-value.json
```

The page is designed to answer two questions immediately:

1. **What did Entroly do to my AI traffic in this period?**
2. **How much value has Entroly accumulated for me overall?**

## Rolling executive view

The selected period can be Today, 7 days, 30 days, 60 days, or 90 days. The period card can show:

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

Entroly selects a useful recent window based on how long Traffic Value has actually been collecting receipts:

```text
< 30 days collecting     -> 7D
30-59 days collecting    -> 30D
60-99 days collecting    -> 60D
100+ days collecting     -> 90D
```

So an install with 3 days of history opens on 7D, one with 45 days opens on 30D, and one with 100+ days opens on 90D.

## All Time never disappears

The selected rolling period is always accompanied by a separate **ALL TIME** card. It highlights:

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

The dashboard persists only aggregate counters and bounded receipt identifiers in the existing value-tracker file. It does not add prompt, tool-output, code, API-key, or raw user-agent content to the executive rollups.
