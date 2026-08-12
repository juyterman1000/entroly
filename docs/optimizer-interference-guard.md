# Optimizer interference and provider prompt caches

Reducing the number of tokens in one request is not the same as reducing total
provider cost. An optimizer can rewrite old conversation history, shorten the
current request, and still lose money when the rewrite invalidates a discounted
provider prefix cache.

Entroly separates four quantities:

1. **Gross reduction**: tokens removed from the provider-bound request.
2. **Recovery tax**: omitted tokens later re-expanded or retrieved.
3. **Provider cache usage**: cache-read, cache-write, and uncached input tokens
   reported by the provider.
4. **Prefix continuity**: a local estimate of how much append-only prefix the
   upstream agent offered versus how much the optimized request preserved.

The Context Health dashboard calls the first two quantities
**retrieval-adjusted net tokens**. It does not call them economic net savings.
Economic attribution requires provider usage, pricing provenance, and a paired
baseline with the same workload.

## Hash-only continuity measurement

`PrefixContinuityGuard` renders only provider prompt-bearing fields into a
transient canonical surface. It divides that surface into fixed-size blocks and
retains SHA-256 digests plus byte counts. Prompt text, code, paths, raw request
or conversation identifiers, and model names are not retained.

For each conversation transition it compares:

- the common prefix available in consecutive raw agent requests; and
- the common prefix present in consecutive Entroly outbound requests.

If the outbound prefix is materially shorter, Entroly reports an estimated
optimizer-interference risk. This is a local estimate, not proof of a provider
cache miss. Provider usage metadata remains authoritative.

## Automatic guard

The proxy captures the original provider request before optional image or
history optimization. If emergency session rescue is required, its recoverable
output becomes the new baseline. Optional image changes, context injection, or
history pruning are compared with that baseline.
When the provider has reported a warm cache and the optional candidate would
materially shorten the reusable prefix, Entroly forwards the safer baseline.

The guard cannot undo:

- outbound redaction;
- fail-closed safety controls;
- recovery persistence; or
- emergency session rescue required to stay inside the context window.

Once session rescue compresses historical evidence, its deterministic frozen
representation remains byte-stable on later turns.

## Evidence levels

| Dashboard field | Evidence basis |
|---|---|
| Retrieval-adjusted net | Optimization ledger events minus measured recovery adjustments |
| Live request cache-hit rate | Bounded in-memory provider observations |
| Cached input-token ratio | Content-blind process-local provider totals by default |
| Priced and durable usage | Opt-in local usage ledger and pricing catalog |
| Prefix continuity | Local hash-only estimate |
| Economic net | Unavailable without a paired baseline |

The dashboard does not infer causal savings from a cache correlation and does
not convert an unavailable price into `$0.00`.
