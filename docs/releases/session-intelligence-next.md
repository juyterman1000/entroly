# Session intelligence hardening

This draft release note covers the P0/P1/P2 session-intelligence hardening work.

## Added

- Retrieval-adjusted savings in `CompressionRetrievalStore`.
- Per-span retrieval counts, retrieved token estimates, repeated expansion penalties, and net realized savings.
- Confidence-tiered savings records via `SavingsTierLedger`.
- Decision, failure, remaining-task, and modified-path extraction for compaction continuity.
- Query-conditioned checkpoint scoring with a trust fence.
- Forecast-only cache retention economics.
- Behavioral loop telemetry for repeated errors, duplicate tool calls, retry loops, and model-switch churn.

## Invariants

- Gross compression savings are never treated as realized savings after omitted spans are retrieved.
- Measured, estimated, and opportunity savings remain separate.
- Checkpoint restoration is scored and fenced; untrusted checkpoints are discounted.
- Cache retention is forecast-only and does not issue keep-warm network calls.
- Behavioral telemetry is bounded by a sliding window.

## Validation

- `tests/test_session_intelligence.py` covers P0, P1, and P2 primitives.
