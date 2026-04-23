---
claim_id: 6267c5ff-f68f-4cb6-a916-ebcacf0196aa
entity: test_brutal
status: inferred
confidence: 0.75
sources:
  - entroly-core\tests\test_brutal.py:53
  - entroly-core\tests\test_brutal.py:106
  - entroly-core\tests\test_brutal.py:113
  - entroly-core\tests\test_brutal.py:171
  - entroly-core\tests\test_brutal.py:23
  - entroly-core\tests\test_brutal.py:54
  - entroly-core\tests\test_brutal.py:59
  - entroly-core\tests\test_brutal.py:68
  - entroly-core\tests\test_brutal.py:85
  - entroly-core\tests\test_brutal.py:123
last_checked: 2026-04-23T03:07:07.862170+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_brutal

**Language:** python
**Lines of code:** 733

## Types
- `class PaymentProcessor()`
- `class Currency()`
- `class Transaction()`
- `class DatabaseConnection()`

## Functions
- `def test(name, fn)`
- `def __init__(self, db_conn, fee_pct: float = 0.029)`
- `def process(self, amount: decimal.Decimal, currency: Currency,
                 user_id: int) -> Transaction`
- `def refund(self, txn_id: str) -> bool`
- `def get_exchange_rate(from_currency: str, to_currency: str) -> float`
- `def total_charged(self) -> float` — , "auth.py": 
- `def generate_token(user_id: int, expiry_seconds: int = 3600) -> str`
- `def verify_token(token: str) -> Optional[int]`
- `def __init__(self, dsn: str)`
- `def connect(self)`
- `def save(self, obj: Any) -> None`
- `def get(self, txn_id: str) -> Optional[Any]`
- `def db()`
- `def processor(db)`
- `def test_process_payment_usd(processor, db)`
- `def test_refund_success(processor, db)`
- `def test_refund_already_refunded(processor, db)`
- `def test_real_session_full()` — Simulate a real developer debug session over 5 turns.
- `def test_selection_respects_dependency_ordering()` — When A calls B, both selected: B should come before A in output order.
- `def test_budget_forces_tradeoffs()` — With tiny budget, highest-value fragments win.
- `def test_budget_with_pinned_overflow()` — If pinned items alone exceed budget, still include them all.
- `def test_budget_one_token()` — Degenerate case: budget of 1 token.
- `def test_duplicate_updates_existing()` — When duplicate ingested, existing fragment gets access_count boosted.
- `def test_many_duplicates_accumulate()` — 10 duplicate ingests → same fragment; stats reflect all 10.
- `def test_whitespace_variants_not_deduped()` — Extra whitespace changes content significantly → different hash.
- `def test_feedback_convergence()` — After 20 success signals, boosted fragment dominates selection.
- `def test_feedback_negative_suppresses()` — Fragment with 0 successes, 10 failures → Wilson-score pushes it below 0.5× multiplier.
- `def test_explain_scores_are_bounded()` — All score dimensions must be in valid ranges.
- `def test_explain_critical_marked()` — Critical files appear in explain with criticality=Critical.
- `def test_explain_consistency_with_optimize()` — IDs in explain must match exactly what optimize returned.
- `def test_export_import_produces_same_optimize()` — After export→import, optimize on same query gives same result.
- `def test_export_import_preserves_feedback()` — Feedback learned in e1 should carry over to e2.
- `def test_sufficiency_full_deps_present()` — If all referenced symbols have definitions in context → sufficiency = 1.0
- `def test_sufficiency_missing_definition_warns()` — Fragment referencing a missing definition → sufficiency < 1.0, warning issued.
- `def test_dep_boost_changes_selection()` — Dep boost should cause dependency to be selected over a higher-raw-score fragment.
- `def test_exploration_fires_over_many_calls()` — Over 100 optimize calls with ε=0.1, ≥5 should involve exploration.
- `def test_no_exploration_when_rate_zero()` — With ε=0, optimize should be fully deterministic.
- `def test_ingest_1000_fragments()` — Engine must handle 1000 fragments without crash or slowdown.
- `def test_optimize_1000_fragments()` — Optimize over 1000 fragments must stay within budget and be fast.
- `def test_rapid_advance_turns()` — Rapidly advance 1000 turns with periodic optimizations.
- `def test_recall_semantic_ranking()` — Recall query for 'payment bug' should return payment-related files higher.
- `def test_recall_top_k_respected()` — Recall never returns more than k results.
- `def test_auth_content_safety()` — auth.py contains 'api_key' keyword → should be auto-pinned as safety signal.

## Dependencies
- `dataclasses`
- `datetime`
- `decimal`
- `entroly_core`
- `hashlib`
- `hmac`
- `logging`
- `models`
- `payments`
- `psycopg2`
- `pytest`
- `rates`
- `requests`
- `secrets`
- `sys`
- `time`
- `traceback`
- `typing`
- `unittest.mock`
- `uuid`

## Linked Beliefs
- [[entroly_core]]

## Key Invariants
- test_explain_scores_are_bounded: All score dimensions must be in valid ranges.
- test_explain_consistency_with_optimize: IDs in explain must match exactly what optimize returned.
- test_ingest_1000_fragments: Engine must handle 1000 fragments without crash or slowdown.
- test_optimize_1000_fragments: Optimize over 1000 fragments must stay within budget and be fast.
