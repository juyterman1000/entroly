# Session intelligence validation checklist

Use this checklist before merging the P0/P1/P2 session-intelligence branch.

## P0 retrieval-adjusted savings

- Store a compression receipt with omitted spans.
- Retrieve one span through `get_span`.
- Verify `retrieved_tokens` increases.
- Retrieve the same span again.
- Verify `repeated_expansion_tokens` increases.
- Verify `net_realized_saved_tokens < gross_saved_tokens`.
- Reopen the store from disk and verify counters persist.

## P0 confidence tiers

- Record measured, estimated, and opportunity savings separately.
- Verify summaries do not merge tiers.

## P1 checkpoint continuity

- Extract a digest from a transcript containing decisions, failures, remaining work, and modified paths.
- Put that digest under checkpoint metadata key `continuity`.
- Rank checkpoints for a query.
- Verify the relevant trusted checkpoint outranks stale or untrusted checkpoints.

## P2 cache retention

- Feed short pause samples to the forecaster.
- Verify short or long cache retention is selected only when expected savings clears the threshold.
- Verify `none` is selected when savings is not material.

## P2 behavior loops

- Record duplicate errors, duplicate tool calls, retries, and model switches.
- Verify the waste report counts each class and computes a positive waste score.
- Advance time past the configured window and verify old events are evicted.
