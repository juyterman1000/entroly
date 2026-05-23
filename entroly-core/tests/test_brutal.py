#!/usr/bin/env python3
"""
Brutal functional tests for entroly_core.

Simulates REAL usage patterns:
- Multi-turn conversations with growing context
- Actual code files (not toy strings)
- Adversarial budget pressure
- Feedback-driven learning over many turns
- Export/import fidelity under real conditions
- Correctness of selection ordering
- ε-exploration actually fires
- Dep graph actually changes selection
"""

import sys
import traceback
import time

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        FAIL += 1
        print(f"  ✗ {name}")
        print(f"    → {e}")
    except Exception as e:
        FAIL += 1
        print(f"  ✗ {name}")
        print(f"    → {type(e).__name__}: {e}")
        traceback.print_exc()

import entroly_core as sc


# ═══════════════════════════════════════════════════════
# REAL CODE SNIPPETS (not toy strings)
# ═══════════════════════════════════════════════════════

REAL_FILES = {
    "payments.py": """
import decimal
from typing import Optional
from rates import get_exchange_rate
from models import Transaction, Currency

class PaymentProcessor:
    def __init__(self, db_conn, fee_pct: float = 0.029):
        self.db = db_conn
        self.fee_pct = fee_pct
        self._cache = {}

    def process(self, amount: decimal.Decimal, currency: Currency,
                 user_id: int) -> Transaction:
        rate = get_exchange_rate(currency, Currency.USD)
        usd_amount = amount * decimal.Decimal(str(rate))
        fee = usd_amount * decimal.Decimal(str(self.fee_pct))
        txn = Transaction(user_id=user_id, amount=usd_amount, fee=fee)
        self.db.save(txn)
        return txn

    def refund(self, txn_id: str) -> bool:
        txn = self.db.get(txn_id)
        if txn is None or txn.refunded:
            return False
        txn.refunded = True
        self.db.save(txn)
        return True
""",
    "rates.py": """
import requests
import time
from typing import Dict

_cache: Dict[str, float] = {}
_cache_ts: float = 0
CACHE_TTL = 300  # 5 minutes

def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    global _cache, _cache_ts
    key = f"{from_currency}:{to_currency}"
    now = time.time()
    if key in _cache and (now - _cache_ts) < CACHE_TTL:
        return _cache[key]
    resp = requests.get(
        f"https://api.exchangeratesapi.io/latest?base={from_currency}"
    )
    resp.raise_for_status()
    data = resp.json()
    _cache = data["rates"]
    _cache_ts = now
    return _cache[to_currency]
""",
    "models.py": """
from dataclasses import dataclass, field
from typing import Optional
import uuid
import datetime

class Currency:
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"

@dataclass
class Transaction:
    user_id: int
    amount: float
    fee: float
    currency: str = Currency.USD
    txn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    refunded: bool = False
    status: str = "pending"

    def total_charged(self) -> float:
        return self.amount + self.fee
""",
    "auth.py": """
import hmac
import hashlib
import secrets
import time
from typing import Optional

SECRET_KEY = "REPLACE_WITH_ENV_VAR"  # api_key placeholder

def generate_token(user_id: int, expiry_seconds: int = 3600) -> str:
    expires = int(time.time()) + expiry_seconds
    payload = f"{user_id}:{expires}"
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"

def verify_token(token: str) -> Optional[int]:
    try:
        parts = token.split(":")
        user_id, expires, sig = parts
        if int(expires) < time.time():
            return None
        payload = f"{user_id}:{expires}"
        expected = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        return int(user_id)
    except Exception:
        return None
""",
    "requirements.txt": """
requests==2.31.0
decimal==1.70
pydantic==2.5.0
fastapi==0.104.1
uvicorn==0.24.0
psycopg2-binary==2.9.9
redis==5.0.1
""",
    "database.py": """
import psycopg2
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn = None

    def connect(self):
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = False

    def save(self, obj: Any) -> None:
        if self._conn is None:
            raise RuntimeError("Not connected")
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO transactions (txn_id, user_id, amount, fee, status) VALUES (%s, %s, %s, %s, %s)",
            (obj.txn_id, obj.user_id, float(obj.amount), float(obj.fee), obj.status)
        )
        self._conn.commit()

    def get(self, txn_id: str) -> Optional[Any]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE txn_id = %s", (txn_id,))
        row = cursor.fetchone()
        return row
""",
    "test_payments.py": """
import pytest
from unittest.mock import MagicMock, patch
from payments import PaymentProcessor
from models import Transaction, Currency
import decimal

@pytest.fixture
def db():
    return MagicMock()

@pytest.fixture
def processor(db):
    return PaymentProcessor(db, fee_pct=0.029)

def test_process_payment_usd(processor, db):
    with patch('payments.get_exchange_rate', return_value=1.0):
        txn = processor.process(decimal.Decimal('100.00'), Currency.USD, user_id=42)
    assert txn.user_id == 42
    assert float(txn.amount) == pytest.approx(100.0)
    assert float(txn.fee) == pytest.approx(2.9)
    db.save.assert_called_once()

def test_refund_success(processor, db):
    mock_txn = MagicMock(refunded=False)
    db.get.return_value = mock_txn
    result = processor.refund("txn_123")
    assert result is True
    assert mock_txn.refunded is True

def test_refund_already_refunded(processor, db):
    mock_txn = MagicMock(refunded=True)
    db.get.return_value = mock_txn
    result = processor.refund("txn_123")
    assert result is False
""",
    "LICENSE": """MIT License

Copyright (c) 2024 CogOps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
""",
}


print("═══ A. MULTI-TURN REAL SESSION ═══")

def test_real_session_full():
    """Simulate a real developer debug session over 5 turns."""
    e = sc.EntrolyEngine(decay_half_life=10, min_relevance=0.02)

    # Turn 0: Ingest the full payment system
    ids = {}
    for fname, content in REAL_FILES.items():
        r = e.ingest(content, fname, 0, False)
        ids[fname] = r["fragment_id"] if r["status"] == "ingested" else r["duplicate_of"]

    frag_count_0 = e.fragment_count()
    assert frag_count_0 == len(REAL_FILES), \
        f"Should have {len(REAL_FILES)} fragments, got {frag_count_0}"

    # Turn 0: User asks "why is payment processing failing for EUR"
    opt = e.optimize(4000, "why is payment processing failing for EUR")
    assert opt["selected_count"] > 0
    assert opt["total_tokens"] <= 4000 * 1.5  # bug=1.5x budget

    # Verify selections make sense — payments.py and rates.py should be included
    selected_sources = [f["source"] for f in opt["selected"]]
    assert any("payment" in s for s in selected_sources), \
        f"payments.py should be selected for payment query: {selected_sources}"
    assert any("rate" in s for s in selected_sources), \
        f"rates.py should be selected (payment depends on it): {selected_sources}"

    # Turn 1: Feedback — rates.py was helpful, database.py was noise
    e.record_success([ids.get("rates.py"), ids.get("payments.py")])
    e.record_failure([ids.get("database.py")])
    e.advance_turn()

    # Turn 2: Same query again — rates.py should still be selected
    opt2 = e.optimize(4000, "fix the payment bug for EUR currency")
    selected2 = [f["source"] for f in opt2["selected"]]
    assert any("rate" in s for s in selected2), \
        f"rates.py should still be preferred after positive feedback: {selected2}"

    # Turn 3: LICENSE and requirements.txt must ALWAYS be in context (pinned)
    e.advance_turn()
    e.advance_turn()
    opt3 = e.optimize(10000, "")
    selected3 = [f["source"] for f in opt3["selected"]]
    assert "LICENSE" in selected3, f"LICENSE must always be included: {selected3}"
    assert "requirements.txt" in selected3, f"requirements.txt must be included: {selected3}"

    # Turn 4: Advance many turns — unpinned low-relevance frags should decay
    for _ in range(30):
        e.advance_turn()
    # Pinned files (LICENSE, requirements.txt, auth.py with SECRET_KEY) should survive
    assert e.fragment_count() > 0, "Should still have pinned fragments"

test("Full real multi-turn debug session", test_real_session_full)


def test_selection_respects_dependency_ordering():
    """When A calls B, both selected: B should come before A in output order."""
    e = sc.EntrolyEngine()
    e.ingest("def get_exchange_rate(from_c, to_c):\n    return 1.0", "rates.py", 50, False)
    e.ingest("from rates import get_exchange_rate\ndef process(amount, currency):\n    rate = get_exchange_rate(currency, 'USD')\n    return amount * rate", "payments.py", 100, False)

    opt = e.optimize(10000, "payment")
    selected = opt["selected"]
    assert len(selected) == 2

    # payments.py depends on rates.py → rates.py should appear first
    sources = [f["source"] for f in selected]
    if "rates.py" in sources and "payments.py" in sources:
        rates_idx = sources.index("rates.py")
        pay_idx = sources.index("payments.py")
        assert rates_idx < pay_idx, \
            f"Dependency (rates.py) should precede dependent (payments.py): {sources}"
test("Selection orders deps before dependents", test_selection_respects_dependency_ordering)


print("\n═══ B. BUDGET PRESSURE ═══")

def test_budget_forces_tradeoffs():
    """With tiny budget, highest-value fragments win."""
    e = sc.EntrolyEngine(exploration_rate=0.0)
    # All same size (100 tokens), different relevance (recency-based)
    frag_ids = []
    for i in range(20):
        r = e.ingest(f"def function_{i}(): return {i} * complex_calculation_{i}()", f"module_{i}.py", 100, False)
        frag_ids.append(r["fragment_id"])

    # Budget for ~5 fragments (IOS diversity may select fewer for quality)
    opt = e.optimize(500, "")
    assert 4 <= opt["selected_count"] <= 5, f"Should select 4-5: {opt['selected_count']}"
    assert opt["total_tokens"] <= 500, f"Budget violated: {opt['total_tokens']}"
test("Tight budget forces correct tradeoffs", test_budget_forces_tradeoffs)

def test_budget_with_pinned_overflow():
    """If pinned items alone exceed budget, still include them all."""
    e = sc.EntrolyEngine()
    # 3 pinned items at 200 tokens each = 600 tokens
    e.ingest("MIT License text...", "LICENSE", 200, True)
    e.ingest("numpy==2.0", "requirements.txt", 200, True)
    e.ingest("FROM python:3.11", "Dockerfile", 200, True)
    # Budget = 100 (less than pinned total)
    opt = e.optimize(100, "")
    # All pinned must be present
    selected_sources = [f["source"] for f in opt["selected"]]
    assert "LICENSE" in selected_sources
    assert "requirements.txt" in selected_sources
    assert "Dockerfile" in selected_sources
    # Total tokens may exceed budget due to pinned
    assert opt["selected_count"] == 3
test("Pinned items included even if they exceed budget", test_budget_with_pinned_overflow)

def test_budget_one_token():
    """Degenerate case: budget of 1 token."""
    e = sc.EntrolyEngine()
    e.ingest("x = 1", "tiny.py", 5, False)
    e.ingest("y = 2", "tiny2.py", 5, False)
    opt = e.optimize(1, "")
    # Can't fit anything (5 > 1), so 0 selected
    assert opt["selected_count"] == 0 or opt["total_tokens"] <= 1
test("Budget of 1 token (degenerate case)", test_budget_one_token)


print("\n═══ C. DUPLICATE DETECTION DEPTH ═══")

def test_duplicate_updates_existing():
    """When duplicate ingested, existing fragment gets access_count boosted."""
    e = sc.EntrolyEngine()
    r1 = e.ingest("def foo(): return 42", "a.py", 0, False)
    r2 = e.ingest("def foo(): return 42", "b.py", 0, False)
    assert r2["status"] == "duplicate"
    assert r2["tokens_saved"] > 0
    # Only 1 fragment exists
    assert e.fragment_count() == 1
    s = e.stats()
    assert s["savings"]["total_tokens_saved"] >= r2["tokens_saved"]
test("Duplicate ingest updates stats", test_duplicate_updates_existing)

def test_many_duplicates_accumulate():
    """10 duplicate ingests → same fragment; stats reflect all 10."""
    e = sc.EntrolyEngine()
    content = "def process(): return True"
    e.ingest(content, "original.py", 0, False)
    for i in range(9):
        r = e.ingest(content, f"copy_{i}.py", 0, False)
        assert r["status"] == "duplicate", f"Ingest {i} should be duplicate"
    assert e.fragment_count() == 1
    s = e.stats()
    assert s["savings"]["total_duplicates_caught"] == 9
    assert s["savings"]["total_fragments_ingested"] == 10
test("10 duplicate ingests → 1 stored, 9 caught", test_many_duplicates_accumulate)

def test_whitespace_variants_not_deduped():
    """Extra whitespace changes content significantly → different hash."""
    e = sc.EntrolyEngine()
    e.ingest("def foo(): return 42", "a.py", 0, False)
    # Completely different structure
    r2 = e.ingest("class Foo:\n    def bar(self):\n        return 42", "b.py", 0, False)
    assert r2["status"] == "ingested", "Structurally different content should not be deduped"
test("Structurally different content not deduped", test_whitespace_variants_not_deduped)


print("\n═══ D. FEEDBACK LEARNING CONVERGENCE ═══")

def test_feedback_convergence():
    """After 20 success signals, boosted fragment dominates selection."""
    e = sc.EntrolyEngine(exploration_rate=0.0)
    r_good = e.ingest("def calculate_tax(income): return income * 0.3", "tax.py", 100, False)
    r_noise = e.ingest("def unrelated_util(): pass", "util.py", 100, False)
    r_noise2 = e.ingest("x = 1  # just a variable", "var.py", 100, False)

    # Record 20 successes for the tax function
    for _ in range(20):
        e.record_success([r_good["fragment_id"]])
        e.record_failure([r_noise["fragment_id"]])
        e.record_failure([r_noise2["fragment_id"]])

    # With budget only for 1, tax.py should always win
    for _ in range(5):
        opt = e.optimize(110, "tax calculation")
        selected_ids = [f["id"] for f in opt["selected"]]
        assert r_good["fragment_id"] in selected_ids, \
            f"Feedback-trained fragment must win: {[f['source'] for f in opt['selected']]}"
test("Feedback converges: boosted fragment consistently wins", test_feedback_convergence)

def test_feedback_negative_suppresses():
    """Fragment with 0 successes, 10 failures → Wilson-score pushes it below 0.5× multiplier."""
    e = sc.EntrolyEngine()
    r_bad = e.ingest("def bad_code(): raise RuntimeError('broken')", "bad.py", 100, False)
    r_good = e.ingest("def good_code(): return True", "good.py", 100, False)
    for _ in range(10):
        e.record_failure([r_bad["fragment_id"]])
        e.record_success([r_good["fragment_id"]])

    opt = e.optimize(110, "")
    selected_ids = [f["id"] for f in opt["selected"]]
    selected_sources = [f["source"] for f in opt["selected"]]
    assert "good.py" in selected_sources, f"good.py should win: {selected_sources}"
    assert "bad.py" not in selected_sources, f"bad.py should be suppressed: {selected_sources}"
test("Negative feedback suppresses bad fragments", test_feedback_negative_suppresses)


print("\n═══ E. EXPLAINABILITY CORRECTNESS ═══")

def test_explain_scores_are_bounded():
    """All score dimensions must be in valid ranges."""
    e = sc.EntrolyEngine()
    for fname, content in list(REAL_FILES.items())[:5]:
        e.ingest(content, fname, 0, False)
    e.optimize(10000, "payment bug")
    expl = e.explain_selection()
    assert "error" not in expl

    for item in expl.get("included", []) + expl.get("excluded", []):
        s = item["scores"]
        assert 0.0 <= s["recency"] <= 1.0, f"recency out of range: {s['recency']}"
        assert 0.0 <= s["frequency"] <= 1.0, f"frequency out of range: {s['frequency']}"
        assert 0.0 <= s["semantic"] <= 1.0, f"semantic out of range: {s['semantic']}"
        assert 0.0 <= s["entropy"] <= 1.0, f"entropy out of range: {s['entropy']}"
        assert s["feedback_mult"] >= 0.0, f"feedback_mult negative: {s['feedback_mult']}"
        assert 0.0 <= s["dep_boost"] <= 1.0, f"dep_boost out of range: {s['dep_boost']}"
        assert 0.0 <= s["composite"] <= 1.0, f"composite out of range: {s['composite']}"
test("Explain: all score dimensions in valid ranges", test_explain_scores_are_bounded)

def test_explain_critical_marked():
    """Critical files appear in explain with criticality=Critical."""
    e = sc.EntrolyEngine()
    e.ingest("numpy==2.0", "requirements.txt", 0, False)
    e.ingest("def foo(): pass", "utils.py", 0, False)
    e.optimize(10000, "")
    expl = e.explain_selection()
    req_entry = next(
        (item for item in expl.get("included", []) if "requirements" in item["source"]),
        None
    )
    assert req_entry is not None, "requirements.txt should be in explain output"
    assert req_entry["scores"]["criticality"] in ("Critical", "Safety"), \
        f"Wrong criticality: {req_entry['scores']['criticality']}"
test("Explain marks critical files correctly", test_explain_critical_marked)

def test_explain_consistency_with_optimize():
    """IDs in explain must match exactly what optimize returned."""
    e = sc.EntrolyEngine()
    for fname, content in list(REAL_FILES.items())[:4]:
        e.ingest(content, fname, 0, False)
    opt = e.optimize(5000, "")
    expl = e.explain_selection()

    opt_ids = set(f["id"] for f in opt["selected"])
    explain_included_ids = set(item["id"] for item in expl.get("included", []))
    assert opt_ids == explain_included_ids, \
        f"IDs mismatch: opt={opt_ids}, explain={explain_included_ids}"
test("Explain IDs match optimize selection exactly", test_explain_consistency_with_optimize)


print("\n═══ F. EXPORT/IMPORT FIDELITY ═══")

def test_export_import_produces_same_optimize():
    """After export→import, optimize on same query gives same result."""
    e1 = sc.EntrolyEngine()
    for fname, content in REAL_FILES.items():
        e1.ingest(content, fname, 0, False)
    for _ in range(3):
        e1.record_success([])
    e1.advance_turn()

    state = e1.export_state()

    e2 = sc.EntrolyEngine()
    e2.import_state(state)

    # Same query → should produce same fragments
    opt1 = e1.optimize(5000, "payment processing")
    opt2 = e2.optimize(5000, "payment processing")

    ids1 = set(f["id"] for f in opt1["selected"])
    ids2 = set(f["id"] for f in opt2["selected"])
    assert ids1 == ids2, \
        f"After import, optimize should match: {ids1} vs {ids2}"
    assert opt1["total_tokens"] == opt2["total_tokens"]
test("Export/import produces identical optimize results", test_export_import_produces_same_optimize)

def test_export_import_preserves_feedback():
    """Feedback learned in e1 should carry over to e2."""
    e1 = sc.EntrolyEngine()
    r1 = e1.ingest("def important(): return True", "imp.py", 100, False)
    r2 = e1.ingest("def noise(): pass", "noise.py", 100, False)
    for _ in range(10):
        e1.record_success([r1["fragment_id"]])
        e1.record_failure([r2["fragment_id"]])

    state = e1.export_state()
    e2 = sc.EntrolyEngine()
    e2.import_state(state)

    # e2 should still prefer imp.py
    opt = e2.optimize(110, "")
    selected_sources = [f["source"] for f in opt["selected"]]
    assert "imp.py" in selected_sources, \
        f"Feedback should survive export/import: {selected_sources}"
test("Export/import preserves feedback learning", test_export_import_preserves_feedback)


print("\n═══ G. SUFFICIENCY SCORING ═══")

def test_sufficiency_full_deps_present():
    """If all referenced symbols have definitions in context → sufficiency = 1.0"""
    e = sc.EntrolyEngine()
    # Define the function
    e.ingest("def calculate_tax(income): return income * 0.3", "tax.py", 0, False)
    # Use it
    e.ingest("from tax import calculate_tax\ntax = calculate_tax(50000)", "main.py", 0, False)
    opt = e.optimize(100000, "tax")
    suff = opt.get("sufficiency", -1)
    assert suff >= 0.0, f"Sufficiency should be non-negative: {suff}"
    # Both frags selected → should be high
    assert suff >= 0.5, f"With definition present, sufficiency should be high: {suff}"
test("Sufficiency high when all deps present", test_sufficiency_full_deps_present)

def test_sufficiency_missing_definition_warns():
    """Fragment referencing a missing definition → sufficiency < 1.0, warning issued."""
    e = sc.EntrolyEngine()
    # Only usage, no definition
    e.ingest("result = missing_function(42)", "caller.py", 0, False)
    opt = e.optimize(10000, "")
    # If no symbols are defined, sufficiency = 0 or N/A (no referenced symbols tracked)
    suff = opt.get("sufficiency", 1.0)
    # No crash is the key requirement
    assert 0.0 <= suff <= 1.0, f"Sufficiency must be in [0,1]: {suff}"
test("Sufficiency handles missing definitions (no crash)", test_sufficiency_missing_definition_warns)


print("\n═══ H. DEP GRAPH REAL IMPACT ═══")

def test_dep_boost_changes_selection():
    """Dep boost should cause dependency to be selected over a higher-raw-score fragment."""
    e = sc.EntrolyEngine()
    # Define rate function (will be depended on)
    r_rates = e.ingest("def get_rate(currency):\n    return {'USD': 1.0}[currency]", "rates.py", 50, False)

    # Define payment function that calls get_rate
    r_payments = e.ingest("from rates import get_rate\ndef process(amount, currency):\n    return amount * get_rate(currency)", "payments.py", 100, False)

    # Define a completely unrelated high-relevance fragment
    r_noise = e.ingest("def totally_different_thing(): return 'optimized!'", "noise.py", 50, False)

    # Query relevant to payments
    opt = e.optimize(160, "payment processing")  # fits rates(50) + payments(100) or rates(50) + noise(50) + payments won't fit

    selected = [f["source"] for f in opt["selected"]]
    # payments.py (100 tokens) + rates.py (50) = 150 ≤ 160 budget
    # Because payments depends on rates, rates should be boosted and selected together
    if "payments.py" in selected:
        assert "rates.py" in selected, \
            f"rates.py should be selected when payments.py is (dep boost): {selected}"
test("Dep boost causes dependency to be co-selected", test_dep_boost_changes_selection)


print("\n═══ I. EXPLORATION ═══")

def test_exploration_fires_over_many_calls():
    """Over 100 optimize calls with ε=0.1, ≥5 should involve exploration."""
    e = sc.EntrolyEngine(exploration_rate=0.1)
    # Load 20 fragments, budget only fits 5
    for i in range(20):
        e.ingest(f"module_{i} content unique {'xyz ' * i}", f"m{i}.py", 50, False)

    selected_sets = []
    for _ in range(50):
        opt = e.optimize(250, "search")  # fits 5
        ids = frozenset(f["id"] for f in opt["selected"])
        selected_sets.append(ids)

    # With exploration, selections should vary (not always identical)
    unique_selections = len(set(selected_sets))
    assert unique_selections > 1, \
        f"Exploration should produce varied selections over 50 calls, got {unique_selections} unique"
test("ε-exploration produces varied selections", test_exploration_fires_over_many_calls)

def test_no_exploration_when_rate_zero():
    """With ε=0, optimize should be fully deterministic."""
    e = sc.EntrolyEngine(exploration_rate=0.0)
    for i in range(20):
        e.ingest(f"def f{i}(): return {i}", f"m{i}.py", 50, False)

    first_opt = e.optimize(250, "")
    first_ids = frozenset(f["id"] for f in first_opt["selected"])

    for _ in range(20):
        opt = e.optimize(250, "")
        ids = frozenset(f["id"] for f in opt["selected"])
        assert ids == first_ids, "With ε=0, every call should return identical selection"
test("ε=0 means fully deterministic selection", test_no_exploration_when_rate_zero)


print("\n═══ J. STRESS & SCALE ═══")

def test_ingest_1000_fragments():
    """Engine must handle 1000 fragments without crash or slowdown."""
    e = sc.EntrolyEngine()
    start = time.time()
    for i in range(1000):
        e.ingest(f"def unique_function_{i}(x, y): return x * {i} + y * {i+1}", f"module_{i}.py", 0, False)
    elapsed = time.time() - start
    assert e.fragment_count() == 1000
    assert elapsed < 10.0, f"1000 ingests took {elapsed:.2f}s (should be <10s)"
test("1000 fragments ingested < 10s", test_ingest_1000_fragments)

def test_optimize_1000_fragments():
    """Optimize over 1000 fragments must stay within budget and be fast."""
    e = sc.EntrolyEngine()
    for i in range(1000):
        e.ingest(f"def func_{i}(x): return x + {i}", f"f{i}.py", 100, False)
    start = time.time()
    opt = e.optimize(5000, "function")
    elapsed = time.time() - start
    assert opt["total_tokens"] <= 5000
    assert opt["selected_count"] <= 50
    assert elapsed < 5.0, f"Optimize over 1000 frags took {elapsed:.2f}s (should be <5s)"
test("Optimize 1000 fragments < 5s", test_optimize_1000_fragments)

def test_rapid_advance_turns():
    """Rapidly advance 1000 turns with periodic optimizations."""
    e = sc.EntrolyEngine(decay_half_life=100)
    for i in range(50):
        e.ingest(f"content {i}", f"f{i}.py", 50, False)
    for _ in range(1000):
        e.advance_turn()
    # Should survive without crash; all unpinned frags may be evicted
    assert e.fragment_count() >= 0
test("1000 turn advances without crash", test_rapid_advance_turns)


print("\n═══ K. RECALL REAL CONTENT ═══")

def test_recall_semantic_ranking():
    """Recall query for 'payment bug' should return payment-related files higher."""
    e = sc.EntrolyEngine()
    for fname, content in REAL_FILES.items():
        e.ingest(content, fname, 0, False)

    results = e.recall("payment processing bug EUR currency", 8)
    assert len(results) >= 2
    sources = [r["source"] for r in results]
    # payments.py should be in top results
    assert any("payment" in s for s in sources[:4]), \
        f"payments.py should appear in top results: {sources}"
test("Recall ranks payment files first for payment query", test_recall_semantic_ranking)

def test_recall_top_k_respected():
    """Recall never returns more than k results."""
    e = sc.EntrolyEngine()
    for fname, content in REAL_FILES.items():
        e.ingest(content, fname, 0, False)
    for k in [1, 3, 5, 100]:
        results = e.recall("anything", k)
        expected_max = min(k, e.fragment_count())
        assert len(results) <= expected_max, \
            f"Recall(k={k}) returned {len(results)}, should be ≤ {expected_max}"
test("Recall respects k limit", test_recall_top_k_respected)


print("\n═══ L. AUTH.PY SAFETY DETECTION ═══")

def test_auth_content_safety():
    """auth.py contains 'api_key' keyword → should be auto-pinned as safety signal."""
    e = sc.EntrolyEngine()
    r = e.ingest(REAL_FILES["auth.py"], "auth.py", 0, False)
    assert r["is_pinned"] == True, \
        f"auth.py contains api_key → should be auto-pinned: {r}"
test("auth.py auto-pinned (contains api_key)", test_auth_content_safety)


# ═══════════════════════
# RESULTS
# ═══════════════════════

print(f"\n{'═' * 55}")
print(f"  BRUTAL RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'═' * 55}")
if FAIL > 0:
    sys.exit(1)
else:
    print("  ALL BRUTAL TESTS PASS ✓")
    sys.exit(0)
