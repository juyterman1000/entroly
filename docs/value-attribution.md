# Evidence-bounded value attribution

Entroly explains where AI Traffic Value came from without creating a second savings total.

## Request-local contract

Each admitted proxy request owns one bounded, content-blind attribution state. A contribution carries a bounded `source`, evidence `tier` (`measured`, `estimated`, `opportunity`), accounting `role` (`additive`, `adjustment`, `explanatory`, `protected`), signed token/USD deltas, an evidence source, and an explicit `headline_included` flag. Details are scalar-only; prompt text, tool output, credentials, raw request IDs, and user-agent strings are excluded.

Only rows marked `headline_included` participate in canonical token arithmetic. The final receipt installs one measured additive `context_optimization` row equal to `tokens_avoided`; explanatory/protected rows are never added again. `attribution_reconciled` is true only when included rows exactly reconcile to the receipt headline. Attribution metadata is inside the Traffic Receipt SHA-256 integrity envelope.

The public `record_value_contribution(...)` seam cannot self-promote a caller to measured or headline value. External observations remain estimated or opportunity explanations.

## Automatic observations

When existing runtime components already expose exact evidence, Entroly records request-local explanations for tool-output compression, conversation compression, session rescue, OptimizationLedger events, retrieval/re-expansion adjustments, provider-cache benefit, warm-prefix protection, and recovery evidence.

## Additional provider work

Provider usage recorded while an outer request is active with a different request ID is conservatively treated as additional Entroly-caused work. Provider-reported token overhead is counted. When the UsageLedger has auditable pricing, the cost becomes a measured negative `extra_provider_call` adjustment. Unpriced work remains visible without inventing a dollar value.

This debit never changes `tokens_avoided` and never turns estimated value into measured counterfactual savings.

## Lifecycle

Attribution observes already-hardened proxy seams; it is not a router, retry loop, compressor, or transport boundary. Buffered requests settle as completed/error responses. Streaming requests keep the same state until iteration completes or fails, then finalize exactly once. Recursive exact recovery remains owned by the outer request lifecycle.

## Product projections

The canonical rows are projected into Traffic Receipts, `/traffic-receipts`, Traffic Value JSON/window rollups, the Traffic Value dashboard, `/stats`, ValueTracker receipts used by `entroly value --json`, the existing Prometheus stream, and optional OTEL metrics. No product surface independently recalculates attribution.

Prometheus/OTEL source labels use a finite allowlist; custom sources collapse to `other` to bound cardinality.

## Money truth

`net_value_after_observed_extra_provider_cost_usd` is an executive mixed-evidence arithmetic view: existing Traffic Value minus observed priced additional-provider work. It is not called measured counterfactual savings. `net_measured_saving_micro_usd` remains unavailable until a linked measured counterfactual exists.

Observability failures fail open for user traffic; accounting claims fail closed when evidence is missing.
