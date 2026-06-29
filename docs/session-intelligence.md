# Session intelligence rollout

This release adds license-clean session intelligence primitives inspired by external optimizer workflows, implemented independently for Entroly.

## P0: retrieval-adjusted savings

`CompressionRetrievalStore` now records when omitted spans are retrieved. Each receipt reports:

- `gross_saved_tokens`
- `retrieved_tokens`
- `repeated_expansion_tokens`
- `net_realized_saved_tokens`
- `confidence`

This prevents inflated compression claims. A context block that is later re-expanded is netted against savings.

```python
from entroly.compression_retrieval_store import CompressionRetrievalStore

store = CompressionRetrievalStore(".entroly/compression-store.json")
summary = store.realized_savings_summary()
```

## P0: confidence-tiered savings

`SavingsTierLedger` separates savings into three buckets:

- `measured`: provider or retrieval-store backed numbers
- `estimated`: locally estimated but plausible values
- `opportunity`: what could be saved if a policy were enabled

These tiers must not be mixed in invoice-grade reporting.

```python
from entroly.session_intelligence import SavingsTierLedger, RealizedSavingsRecord
```

## P1: checkpoint relevance and decision preservation

`extract_decision_digest()` extracts compact continuity facts before compaction:

- decisions
- modified paths
- failures
- remaining tasks

`CheckpointRelevanceScorer` ranks checkpoints by query relevance, continuity hits, freshness, coverage, and a trust fence. Untrusted checkpoints are still searchable but heavily discounted.

```python
from entroly.session_intelligence import extract_decision_digest, CheckpointRelevanceScorer
```

## P2: forecast-only cache retention

`CacheRetentionForecaster` estimates whether no retention, short retention, or long retention is economically best given observed pause distributions. It deliberately does not issue keep-warm calls. The caller must decide whether a provider-specific retention mechanism is allowed.

```python
from entroly.session_intelligence import CacheRetentionForecaster
```

## P2: behavior-loop telemetry

`BehavioralWasteDetector` tracks repeated errors, repeated tool calls, retry loops, and model-switch churn inside a bounded window. The result is a normalized `waste_score` for dashboards and routing diagnostics.

```python
from entroly.session_intelligence import BehavioralWasteDetector
```

## Production invariant

Savings are reported in this order of trust:

```text
invoice-grade provider usage > retrieval-adjusted measured savings > estimated local savings > opportunity savings
```

External optimizer source code was not copied. These are independent Entroly primitives designed to plug into the existing ELC retrieval store, checkpoint system, cache router, and gateway dashboards.
