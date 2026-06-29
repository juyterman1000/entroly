# Production economics and session continuity

Entroly separates provider billing facts from optimizer claims.

## Accounting tiers

`UsageLedger` remains the source of truth for provider-reported token usage and
invoice-grade cost. `OptimizationLedger` stores optimizer outcomes in three
non-interchangeable tiers:

- `measured`: an observed reduction, adjusted by later retrieval or re-expansion;
- `estimated`: a forecast produced from observed history and explicit prices;
- `opportunity`: detected avoidable work that has not yet been prevented.

Dashboards must report the tiers separately. Only measured net savings are
realized savings. Estimated and opportunity values must not be added to them.

Configure the durable optimization ledger for the live proxy with:

```bash
export ENTROLY_OPTIMIZATION_LEDGER=.entroly/optimization-ledger.sqlite3
```

Compression retrieval MCP uses the same environment variable. A plain
`get_span()` is inspection-only; `retrieve_span()` and MCP responses debit the
returned tokens from the compression event. Supplying a stable `retrieval_id`
makes that debit idempotent.

## Checkpoint continuity

`resume_state(query=..., project=...)` ranks checkpoints using task metadata,
explicit decisions, fragment source paths, project scope, and time decay. It
returns `no_relevant_checkpoint_found` below the relevance threshold instead of
silently restoring an unrelated session.

Checkpoint decisions are explicit:

```python
checkpoint_state(
    task_description="finish cache-aware routing",
    current_step="provider response accounting",
    decisions=["Keep invoice usage separate from optimizer estimates"],
    modified_files=["entroly/usage_ledger.py"],
)
```

Only decisions carry forward automatically. Engine snapshots, current steps,
and arbitrary metadata never leak into a later checkpoint. Recovery context is
rendered from a small metadata allowlist and fenced as recovered data, not as
instructions.

## Cache retention forecasting

`CacheRetentionForecaster` observes pauses already seen by `CacheAwareRouter`.
It uses a Beta prior to avoid confident recommendations from tiny samples and
compares only caller-supplied, provider-supported retention plans.

The forecaster is advisory. It performs no provider I/O and cannot create
keep-warm traffic. The built-in Anthropic plans encode documented five-minute
and one-hour write/read multipliers; callers still supply their current input
price and enable only plans supported by the selected model.

## Behavioral waste telemetry

The live proxy observes bounded conversation history for:

- identical tool retries;
- repeated normalized errors;
- alternating tool loops;
- model-switch churn.

Findings are exposed as `opportunity` tier estimates. They never block a tool,
retry a request, reroute a model, or claim realized savings.
