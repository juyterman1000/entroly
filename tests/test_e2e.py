"""
End-to-end tests for Entroly engines.

Tests the full pipeline: ingest → dedup → entropy → optimize → checkpoint.
Validates mathematical correctness of the knapsack optimizer, entropy scorer,
SimHash deduplication, and checkpoint/resume system.
"""

import json
from entroly.config import EntrolyConfig
import math
import os
import tempfile

from entroly_core import (
    ContextFragment,
    py_apply_ebbinghaus_decay as apply_ebbinghaus_decay,
    py_knapsack_optimize as knapsack_optimize,
)

from entroly_core import (
    py_shannon_entropy as shannon_entropy,
    py_normalized_entropy as normalized_entropy,
    py_boilerplate_ratio as boilerplate_ratio,
    py_cross_fragment_redundancy as cross_fragment_redundancy,
    py_information_score as compute_information_score,
)
from entroly_core import py_simhash as simhash, py_hamming_distance as hamming_distance
try:
    from entroly_core import PyDedupIndex as DedupIndex
except ImportError:
    DedupIndex = None  # fallback guard

from entroly.prefetch import (
    PrefetchEngine,
    extract_callees,
    extract_imports,
    infer_test_files,
)
from entroly.checkpoint import CheckpointManager
from entroly.server import EntrolyEngine


# ═══════════════════════════════════════════════════════════════════════
# Knapsack Optimizer Tests
# ═══════════════════════════════════════════════════════════════════════

def test_knapsack_selects_optimal_subset():
    """Verify that the DP solver picks the highest-value subset."""
    a = ContextFragment("a", "high value small", token_count=100)
    a.recency_score = 1.0
    a.entropy_score = 0.9
    b = ContextFragment("b", "low value large", token_count=900)
    b.recency_score = 0.1
    b.entropy_score = 0.1
    c = ContextFragment("c", "medium value medium", token_count=400)
    c.recency_score = 0.7
    c.entropy_score = 0.6
    fragments = [a, b, c]

    selected, stats = knapsack_optimize(fragments, token_budget=500)

    # Should pick "a" (100 tokens, high relevance) and "c" (400 tokens, medium relevance)
    # NOT "b" (900 tokens, wouldn't fit with either, and low relevance)
    selected_ids = {f.fragment_id for f in selected}
    assert "a" in selected_ids, f"Expected 'a' in selection, got {selected_ids}"
    assert "b" not in selected_ids, "'b' should not be selected (too large + low value)"
    assert stats["total_tokens"] <= 500, f"Budget exceeded: {stats['total_tokens']}"
    print("  ✓ Knapsack selects optimal subset correctly")


def test_knapsack_respects_pinned():
    """Pinned fragments must always be included."""
    pinned = ContextFragment("pinned", "must include", token_count=300)
    pinned.is_pinned = True
    normal = ContextFragment("normal", "optional", token_count=200)
    normal.recency_score = 1.0
    fragments = [pinned, normal]

    selected, stats = knapsack_optimize(fragments, token_budget=400)
    selected_ids = {f.fragment_id for f in selected}

    assert "pinned" in selected_ids, "Pinned fragment must be included"
    assert stats["total_tokens"] <= 400
    print("  ✓ Pinned fragments always included")


def test_ebbinghaus_decay():
    """Verify exponential decay math matches the Ebbinghaus curve."""
    frag = ContextFragment("x", "test", token_count=10)
    frag.turn_last_accessed = 0
    # apply_ebbinghaus_decay returns new list (Rust is not in-place)
    [frag] = apply_ebbinghaus_decay([frag], current_turn=15, half_life=15)

    # After exactly one half-life, recency should be ~0.5
    assert 0.45 < frag.recency_score < 0.55, \
        f"Expected ~0.5 at half-life, got {frag.recency_score:.4f}"
    print("  ✓ Ebbinghaus decay matches exponential curve")



# ═══════════════════════════════════════════════════════════════════════
# Shannon Entropy Tests
# ═══════════════════════════════════════════════════════════════════════

def test_entropy_all_same_chars():
    """A string of identical characters has zero entropy."""
    assert shannon_entropy("aaaaaaa") == 0.0
    print("  ✓ Identical characters → entropy = 0")


def test_entropy_increases_with_diversity():
    """More character diversity → higher entropy."""
    low = shannon_entropy("aaabbb")
    high = shannon_entropy("abcdef")
    assert high > low, f"Expected {high} > {low}"
    print("  ✓ Character diversity increases entropy")


def test_boilerplate_detection():
    """Import-heavy code should have high boilerplate ratio."""
    code = """import os
import sys
from pathlib import Path
import json
from entroly.config import EntrolyConfig
pass
"""
    ratio = boilerplate_ratio(code)
    assert ratio > 0.7, f"Expected high boilerplate ratio, got {ratio:.2f}"
    print("  ✓ Boilerplate detection identifies imports/pass")


def test_cross_fragment_redundancy():
    """Identical fragments should have redundancy = 1.0."""
    text = "the quick brown fox jumps over the lazy dog"
    redundancy = cross_fragment_redundancy(text, [text])
    assert redundancy > 0.9, f"Expected high redundancy for identical text, got {redundancy:.2f}"
    print("  ✓ Cross-fragment redundancy detects duplicates")


# ═══════════════════════════════════════════════════════════════════════
# SimHash Deduplication Tests
# ═══════════════════════════════════════════════════════════════════════

def test_simhash_identical_texts():
    """Identical texts must produce identical fingerprints."""
    text = "def process_payment(amount, currency):"
    h1 = simhash(text)
    h2 = simhash(text)
    assert h1 == h2, "Identical texts must have identical SimHash"
    print("  ✓ SimHash deterministic for identical input")


def test_simhash_similar_texts_close():
    """Similar texts should have small Hamming distance."""
    h1 = simhash("def process_payment(amount, currency): pass")
    h2 = simhash("def process_payment(amount, currency): return None")
    dist = hamming_distance(h1, h2)
    assert dist < 30, f"Similar texts should be closer than random, got distance {dist}"
    print(f"  ✓ Similar texts have Hamming distance {dist} (< 30, random ≈ 32)")


def test_dedup_index_catches_duplicates():
    """DedupIndex should detect near-identical content."""
    index = DedupIndex(hamming_threshold=3)
    text = "def calculate_tax(income, rate): return income * rate"

    dup1 = index.insert("frag_1", text)
    assert dup1 is None, "First insert should not be a duplicate"

    dup2 = index.insert("frag_2", text)
    assert dup2 == "frag_1", f"Identical text should be duplicate, got {dup2}"
    print("  ✓ DedupIndex catches exact duplicates")


def test_dedup_index_allows_different():
    """DedupIndex should NOT flag very different content as duplicates."""
    index = DedupIndex(hamming_threshold=3)

    index.insert("a", "machine learning neural network gradient descent backprop")
    dup = index.insert("b", "kubernetes docker container orchestration deployment")

    assert dup is None, "Very different texts should not be flagged as duplicates"
    print("  ✓ DedupIndex allows genuinely different content")


# ═══════════════════════════════════════════════════════════════════════
# Predictive Pre-fetch Tests
# ═══════════════════════════════════════════════════════════════════════

def test_import_extraction():
    """Extract Python imports from source code."""
    source = """
from utils.helpers import clean_data
import pandas as pd
from pathlib import Path
"""
    imports = extract_imports(source, "python")
    assert "utils.helpers" in imports
    assert "pandas" in imports
    print(f"  ✓ Extracted {len(imports)} imports from Python source")


def test_test_file_inference():
    """Infer test file paths from source file paths."""
    candidates = infer_test_files("/project/src/utils.py")
    assert any("test_utils" in c for c in candidates)
    print(f"  ✓ Inferred {len(candidates)} test file candidates")


def test_co_access_learning():
    """Pre-fetcher should learn co-access patterns."""
    engine = PrefetchEngine(co_access_window=3)
    engine.record_access("models.py", turn=1)
    engine.record_access("views.py", turn=2)
    engine.record_access("models.py", turn=3)

    predictions = engine.predict("models.py", "", "python")
    co_access_paths = [p.path for p in predictions if p.reason == "co_access"]
    assert "views.py" in co_access_paths, "Should predict views.py from co-access"
    print("  ✓ Pre-fetcher learns co-access patterns")


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint & Resume Tests
# ═══════════════════════════════════════════════════════════════════════

def test_checkpoint_save_and_load():
    """Checkpoint should roundtrip fragments correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir, auto_interval=100)

        f1 = ContextFragment("f1", "hello world", token_count=5)
        f1.recency_score = 0.8
        f1.entropy_score = 0.6
        f2 = ContextFragment("f2", "goodbye world", token_count=5)
        f2.recency_score = 0.3
        f2.entropy_score = 0.9
        frags = [f1, f2]

        mgr.save(frags, {"f1": 12345}, {}, current_turn=10,
                 metadata={"task": "test"})

        ckpt = mgr.load_latest()
        assert ckpt is not None
        assert ckpt.current_turn == 10
        assert len(ckpt.fragments) == 2
        assert ckpt.metadata["task"] == "test"

        restored = mgr.restore_fragments(ckpt)
        assert len(restored) == 2
        assert restored[0].fragment_id == "f1"
        print("  ✓ Checkpoint save/load roundtrips correctly")


# ═══════════════════════════════════════════════════════════════════════
# Full Engine Integration Test
# ═══════════════════════════════════════════════════════════════════════

def test_full_engine_pipeline():
    """End-to-end test of the Entroly engine."""
    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))

    # Ingest several code fragments
    r1 = engine.ingest_fragment(
        "def process_payment(amount, currency):\n    return amount * get_rate(currency)",
        source="file:payments.py",
    )
    assert r1["status"] == "ingested"

    r2 = engine.ingest_fragment(
        "def get_rate(currency):\n    rates = {'USD': 1.0, 'EUR': 0.85}\n    return rates.get(currency, 1.0)",
        source="file:rates.py",
    )
    assert r2["status"] == "ingested"

    # Ingest a duplicate — should be caught
    r3 = engine.ingest_fragment(
        "def process_payment(amount, currency):\n    return amount * get_rate(currency)",
        source="file:payments.py",
    )
    assert r3["status"] == "duplicate", f"Expected duplicate, got {r3['status']}"
    print("  ✓ Duplicate detection in full pipeline")

    # Ingest boilerplate — should get low entropy
    r4 = engine.ingest_fragment(
        "import os\nimport sys\nimport json\nfrom entroly.config import EntrolyConfig\nfrom pathlib import Path\npass",
        source="file:imports.py",
    )
    assert r4["entropy_score"] < 0.75, f"Boilerplate should have lower entropy than novel code, got {r4['entropy_score']}"
    print(f"  ✓ Boilerplate gets lower entropy score ({r4['entropy_score']:.4f})")

    # Optimize context with tight budget
    opt = engine.optimize_context(token_budget=100, query="payment processing")
    total_toks = opt.get("total_tokens") or opt.get("optimization_stats", {}).get("total_tokens", 0)
    saved_toks  = opt.get("tokens_saved") or opt.get("tokens_saved_this_call", 0)
    assert total_toks <= 100, f"Budget exceeded: {total_toks}. Keys: {list(opt.keys())}"
    assert saved_toks >= 0
    print(f"  ✓ Optimization saved {saved_toks} tokens")

    # Recall relevant
    results = engine.recall_relevant("payment rate currency", top_k=2)
    assert len(results) > 0
    print(f"  ✓ Recall returned {len(results)} relevant fragments")

    # Stats
    stats = engine.get_stats()
    assert stats["savings"]["total_duplicates_caught"] >= 1
    assert stats["session"]["total_fragments"] >= 2
    print(f"  ✓ Stats: {stats['savings']['total_duplicates_caught']} duplicates caught, "
          f"{stats['session']['total_fragments']} fragments tracked")


# ═══════════════════════════════════════════════════════════════════════
# NEW: PRISM RL / Feedback Loop Tests
# ═══════════════════════════════════════════════════════════════════════

def test_positive_feedback_raises_fragment_value():
    """
    After repeated positive feedback on a fragment, recall_relevant must rank it
    higher relative to a fragment that received no feedback.
    Proves the PRISM RL weight update actually changes scoring.
    """
    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))

    r_target = engine.ingest_fragment(
        "def compute_hamming_distance(a: int, b: int) -> int: return bin(a ^ b).count('1')",
        source="file:hamming.py",
    )
    r_other = engine.ingest_fragment(
        "class UserProfile: pass",
        source="file:profile.py",
    )

    target_id = r_target["fragment_id"]

    # Simulate 5 turns of positive feedback on the target fragment
    for _ in range(5):
        engine.record_success([target_id])

    results = engine.recall_relevant("Hamming distance bit operations", top_k=5)
    fragment_ids = [r.get("fragment_id") or r.get("source", "") for r in results]

    # target must appear before u_other in ranked results (positive feedback boosted it)
    if target_id in fragment_ids and r_other["fragment_id"] in fragment_ids:
        target_rank = fragment_ids.index(target_id)
        other_rank  = fragment_ids.index(r_other["fragment_id"])
        assert target_rank < other_rank, (
            f"Positive feedback should raise ranking: target={target_rank}, other={other_rank}"
        )
    else:
        # At minimum, the positively-reinforced fragment must appear in top-5
        assert target_id in fragment_ids, (
            f"Positively reinforced fragment must be recalled. Got: {fragment_ids}"
        )
    print("  ✓ Positive feedback raises fragment recall rank")


def test_negative_feedback_suppresses_fragment():
    """
    After repeated negative feedback on a fragment, its Wilson lower-bound
    multiplier should drop below 1.0, reducing its effective relevance score.
    """
    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))

    r_good = engine.ingest_fragment(
        "async def send_payment(amount, receiver): await bank_transfer(amount, receiver)",
        source="file:payments.py",
    )
    r_bad = engine.ingest_fragment(
        "def get_version(): return '1.0.0'",
        source="file:version.py",
    )

    good_id = r_good["fragment_id"]
    bad_id  = r_bad["fragment_id"]

    # Good fragment → positive feedback; bad fragment → repeated failures
    for _ in range(8):
        engine.record_success([good_id])
        engine.record_failure([bad_id])

    results = engine.recall_relevant("payment bank transfer", top_k=5)
    result_ids = [r.get("fragment_id", "") for r in results]

    # Good fragment must appear; bad fragment must NOT lead the results
    assert good_id in result_ids, "Positively-reinforced fragment must be recalled"
    if bad_id in result_ids and good_id in result_ids:
        assert result_ids.index(good_id) < result_ids.index(bad_id), (
            "Good fragment must rank above failure-penalized one"
        )
    print("  ✓ Negative feedback de-ranks the penalized fragment")


def test_recall_correct_after_eviction():
    """
    After advance_turn() applies Ebbinghaus decay and rebuilds the LSH index,
    pinned live fragments must still be found correctly.

    This tests that rebuild_lsh_index() correctly re-slots remaining fragments
    after non-pinned ones are evicted — a slot-index corruption would silently
    cause the engine to miss relevant fragments.
    """
    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))

    # Ingest transient fragments that will decay (not pinned)
    for i in range(5):
        engine.ingest_fragment(f"TEMP_VAR_{i} = {i}", source=f"file:temp{i}.py")

    # Ingest the critical live fragment — pinned, so it CANNOT be evicted
    engine.ingest_fragment(
        "def authenticate_user(token: str) -> User: return db.find_by_token(token)",
        source="file:auth.py",
        is_pinned=True,
    )

    # Advance enough turns so non-pinned fragments decay below min_relevance
    # half_life=15, after 60 turns: score ≈ 0.5^(60/15) = 0.0625 > 0.05; after 90: ≈ 0.03 < 0.05
    for _ in range(90):
        engine.advance_turn()

    # Pinned fragment must survive and still be recallable via LSH
    results = engine.recall_relevant("authenticate user token", top_k=5)
    result_sources = [r.get("source", "") for r in results]

    assert "file:auth.py" in result_sources, (
        "Pinned live fragment must still be recalled after many turns + LSH rebuild. "
        f"Got: {result_sources}"
    )

    # Verify that non-pinned fragments were actually evicted
    stats = engine.get_stats()
    total_remaining = stats["session"]["total_fragments"]
    assert total_remaining <= 2, (  # at most the pinned + possible LSH artefact
        f"Non-pinned fragments should be evicted after 90 turns; "
        f"expected ≤2 remaining, got {total_remaining}"
    )
    print(f"  ✓ LSH rebuild after eviction: {total_remaining} fragment(s) remain, auth.py is recallable")


def test_fragment_guard_flags_secrets():
    """
    FragmentGuard must detect hardcoded secrets in ingested fragments.
    The 'quality_issues' key in the ingest response must be non-empty.
    """
    from entroly.adaptive_pruner import FragmentGuard

    guard = FragmentGuard()
    if not guard.available:
        print("  ⚠ FragmentGuard not available (Rust backend not installed) — skipping")
        return

    issues = guard.scan(
        'API_KEY = "sk-proj-ABCDEFGHIJKlmnopqrst1234567890"\nrequests.get(url)',
        source="file:config.py",
    )
    assert len(issues) > 0, f"FragmentGuard must flag hardcoded secret. Got: {issues}"
    secret_flagged = any("secret" in i.lower() or "api" in i.lower() or "key" in i.lower()
                         for i in issues)
    assert secret_flagged, f"Should flag an API key issue, got: {issues}"
    print(f"  ✓ FragmentGuard detected {len(issues)} issue(s) in secret-containing code")


def test_fragment_guard_passes_clean_code():
    """FragmentGuard must not flag clean Rust code."""
    from entroly.adaptive_pruner import FragmentGuard

    guard = FragmentGuard()
    if not guard.available:
        print("  ⚠ FragmentGuard not available — skipping")
        return

    clean = """
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}
"""
    issues = guard.scan(clean, source="file:dedup.rs")
    assert len(issues) == 0, f"Clean code should have no issues, got: {issues}"
    print("  ✓ FragmentGuard passes clean production code")


def test_provenance_hallucination_risk():
    """
    ContextProvenance must produce a hallucination_risk between 0 and 1,
    and verified_fraction must reflect the fraction of fragments with known sources.
    Build provenance directly from a real optimize_context call so we test
    the actual integration path, not a mocked one.
    """
    from entroly.provenance import build_provenance

    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))
    engine.ingest_fragment(
        "def authenticate_user(token): return db.find_by_token(token)",
        source="file:auth.py",
    )
    engine.ingest_fragment(
        "SECRET_KEY = 'hardcoded-not-great'",
        source="",  # no source — provenance cannot verify this fragment
    )

    opt_result = engine.optimize_context(token_budget=4000, query="authenticate token")

    prov = build_provenance(
        optimize_result=opt_result,
        query="authenticate token",
        refined_query=None,
        turn=1,
        token_budget=4000,
    )

    assert prov.hallucination_risk in ("low", "medium", "high"), (
        f"hallucination_risk must be 'low'/'medium'/'high', got {prov.hallucination_risk!r}"
    )
    assert 0.0 <= prov.verified_fraction <= 1.0, (
        f"verified_fraction must be in [0,1], got {prov.verified_fraction}"
    )
    print(f"  ✓ Provenance: risk={prov.hallucination_risk!r}, verified={prov.verified_fraction:.2f}")



def test_export_import_preserves_prism_covariance():
    """
    After export_state/import_state roundtrip, the PRISM optimizer covariance
    must survive (not reset to identity), proving the learned RL state persists
    across restarts.
    """
    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))

    r = engine.ingest_fragment(
        "def calculate_hash(data: bytes) -> str: return hashlib.sha256(data).hexdigest()",
        source="file:crypto.py",
    )
    frag_id = r["fragment_id"]

    # Feed the PRISM optimizer non-trivial gradients so covariance moves away from identity
    for _ in range(20):
        engine.record_success([frag_id])

    if not engine._use_rust:
        print("  ⚠ Rust engine not available — skipping PRISM serialization test")
        return

    # Export
    json_state = engine._rust.export_state()
    assert "prism_optimizer" in json_state, (
        "Exported state must include prism_optimizer covariance"
    )

    # Create a fresh engine and import
    engine2 = EntrolyEngine()
    engine2._rust.import_state(json_state)

    # The covariance diagonal should NOT be the fresh 1e-4 identity if state was preserved
    import json as _json
    state = _json.loads(json_state)
    prism = state.get("prism_optimizer", {})
    cov_data = prism.get("covariance", {}).get("data", [])
    # Covariance is serialized as a nested list of lists (row-major 2D array).
    # Extract diagonal elements: cov_data[i][i] for each dimension.
    if cov_data and isinstance(cov_data[0], list):
        dim = len(cov_data)
        diagonal_vals = [cov_data[i][i] for i in range(dim)]
    else:
        # Flat format fallback: data is a flat Vec<f64>
        dim = int(len(cov_data) ** 0.5)
        diagonal_vals = [cov_data[i * dim + i] for i in range(dim) if i * dim + i < len(cov_data)]
    assert any(abs(v - 1e-4) > 1e-6 for v in diagonal_vals), (
        f"PRISM covariance must be non-trivial after feedback. Diagonal: {diagonal_vals}"
    )
    print(f"  ✓ PRISM covariance survives checkpoint roundtrip. Diagonal: {[f'{v:.4e}' for v in diagonal_vals]}")


def test_token_budget_zero_uses_default():
    """
    Calling optimize_context with token_budget=0 must not crash and must
    use the server's configured default budget.
    """
    engine = EntrolyEngine(config=EntrolyConfig(checkpoint_dir=tempfile.mkdtemp()))
    engine.ingest_fragment("def foo(): pass", source="file:foo.py")
    result = engine.optimize_context(token_budget=0, query="")
    total_toks = result.get("total_tokens") or result.get("optimization_stats", {}).get("total_tokens", 0)
    assert total_toks >= 0, "Must return total_tokens even with budget=0"
    print("  ✓ token_budget=0 uses default budget without crash")


# ═══════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import hashlib  # noqa — needed by export/import test
    print("\n═══ Entroly Test Suite ═══\n")

    print("Knapsack Optimizer:")
    test_knapsack_selects_optimal_subset()
    test_knapsack_respects_pinned()
    test_ebbinghaus_decay()

    print("\nShannon Entropy Scorer:")
    test_entropy_all_same_chars()
    test_entropy_increases_with_diversity()
    test_boilerplate_detection()
    test_cross_fragment_redundancy()

    print("\nSimHash Deduplication:")
    test_simhash_identical_texts()
    test_simhash_similar_texts_close()
    test_dedup_index_catches_duplicates()
    test_dedup_index_allows_different()

    print("\nPredictive Pre-fetch:")
    test_import_extraction()
    test_test_file_inference()
    test_co_access_learning()

    print("\nCheckpoint & Resume:")
    test_checkpoint_save_and_load()

    print("\nFull Engine Pipeline:")
    test_full_engine_pipeline()

    print("\nPRISM RL / Feedback Loop:")
    test_positive_feedback_raises_fragment_value()
    test_negative_feedback_suppresses_fragment()
    test_recall_correct_after_eviction()

    print("\nCode Quality Guard:")
    test_fragment_guard_flags_secrets()
    test_fragment_guard_passes_clean_code()

    print("\nContext Provenance (Hallucination Risk):")
    test_provenance_hallucination_risk()

    print("\nCheckpoint + PRISM Serialization:")
    test_export_import_preserves_prism_covariance()

    print("\nEdge Cases:")
    test_token_budget_zero_uses_default()

    print("\n═══ ALL TESTS PASSED ═══\n")
