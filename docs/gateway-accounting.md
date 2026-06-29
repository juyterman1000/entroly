# Live gateway accounting and cache-aware routing

Entroly can use provider-reported token usage to maintain an idempotent local
cost ledger and to avoid model switches that would discard a valuable warm
prompt cache.

Both persistent accounting and catalog-backed routing are operator-controlled.
Entroly never downloads or invents provider prices.

## Configuration

Set a durable SQLite path to enable the ledger:

```bash
export ENTROLY_USAGE_LEDGER=.entroly/usage.sqlite3
```

A trusted ingress can attach bounded cost-allocation dimensions:

```bash
export ENTROLY_TRUST_USAGE_HEADERS=1
# The ingress sets x-entroly-team, x-entroly-project, and x-entroly-tool.
```

Leave this disabled for direct or untrusted clients. When enabled, the reverse
proxy must remove client-supplied `x-entroly-*` attribution headers and inject
authenticated values. Invalid or oversized dimension values are ignored.

Set a versioned pricing catalog to calculate cost and enable the cache-economics
gate for RAVS recommendations:

```bash
export ENTROLY_PRICING_CATALOG=/etc/entroly/pricing.json
export ENTROLY_RAVS_ROUTER=1
```

Example catalog structure:

```json
{
  "source": "internal-finops-catalog-2026-06-28",
  "models": {
    "openai:model-name": {
      "input_per_million": "0.00",
      "output_per_million": "0.00",
      "cache_read_per_million": "0.00",
      "cache_write_per_million": "0.00"
    },
    "anthropic:*": {
      "input_per_million": "0.00",
      "output_per_million": "0.00",
      "cache_read_per_million": "0.00"
    }
  }
}
```

The values above are schema examples, not current provider prices. Replace them
with the rates from the organization’s approved billing catalog. Exact
`provider:model` entries take precedence over a `provider:*` fallback. An
invalid catalog prevents proxy startup; a configured catalog that lacks either
model in a proposed RAVS switch keeps the current model.

Normal SSE forwarding retains only the final bounded transcript tail for usage
parsing. The default is 256 KiB and can be adjusted from 16 KiB through 1 MiB:

```bash
export ENTROLY_USAGE_SSE_TAIL_BYTES=262144
```

## Accounting model

The ledger normalizes these components:

- uncached input tokens
- cache-read input tokens
- cache-write input tokens
- output tokens

Cost is stored as integer micro-USD:

```text
cost =
  uncached_input * input_rate
  + cache_read * cache_read_rate
  + cache_write * cache_write_rate
  + output * output_rate
```

Rates are USD per one million tokens, so multiplying tokens by a rate produces
micro-USD directly. Decimal arithmetic and half-up rounding avoid binary
floating-point drift.

If usage is present but no price is available, Entroly stores the token event
with zero computed cost and an `unpriced:provider:model` provenance marker.
This preserves usage for later reconciliation without presenting a fabricated
invoice amount.

Request IDs are unique ledger keys. An identical replay is idempotent; a replay
with different provider, model, token, cost, or pricing identity raises a
conflict. Verification recovery calls use deterministic `:recovery:N` attempt
suffixes because each successful provider call is independently billable.

## Cache-aware routing

RAVS remains the quality and risk gate. Cache economics run only after RAVS has
recommended a different model.

The cache router projects cost over a bounded turn horizon and compares:

- the current model with provider-observed cached prefix tokens
- the recommended model with a cold prefix on the first turn
- provider-specific cache TTL
- cache read and write rates
- expected new input and output tokens
- a switch hysteresis threshold

Cache state is learned only from provider usage fields. Entroly does not infer a
hit from conversation shape. A lease is reusable only for the same conversation
anchor, provider, model, stable prefix hash, and unexpired TTL.

## Operations

The guarded `/stats` endpoint exposes:

- `provider_cache.active_leases`
- `provider_cache.observed_hits`
- `provider_cache.observed_misses`
- `provider_cache.routing_stays`
- `provider_cache.routing_switches`
- `provider_cache.last_decision`
- `usage_accounting.recorded`
- `usage_accounting.unpriced`
- `usage_accounting.failures`
- `usage_accounting.ledger`

Back up, retain, and delete the SQLite ledger under the same policy as other
billing telemetry. The ledger stores identifiers and normalized usage metadata,
not prompt or response content.
