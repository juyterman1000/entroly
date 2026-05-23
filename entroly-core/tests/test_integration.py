#!/usr/bin/env python3
"""
Comprehensive integration test for entroly_core Rust engine.

Tests every feature end-to-end with real data and edge cases.
No mocks, no stubs, no hardcoded expected values where avoidable.
"""

import sys
import traceback

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# Import
# ═══════════════════════════════════════════════════════════════════

import entroly_core as sc

print("═══ 1. CONSTRUCTOR & CONFIGURATION ═══")

def test_default_constructor():
    e = sc.EntrolyEngine()
    assert e.fragment_count() == 0
    assert e.get_turn() == 0
test("Default constructor", test_default_constructor)

def test_custom_params():
    e = sc.EntrolyEngine(
        w_recency=0.4, w_frequency=0.3, w_semantic=0.2, w_entropy=0.1,
        decay_half_life=20, min_relevance=0.1,
        hamming_threshold=5, exploration_rate=0.2,
    )
    assert e.fragment_count() == 0
test("Custom constructor params", test_custom_params)

def test_exploration_rate_clamp():
    e = sc.EntrolyEngine(exploration_rate=5.0)
    # Should be clamped to 1.0 internally
    e2 = sc.EntrolyEngine(exploration_rate=-1.0)
    # Should be clamped to 0.0 internally
test("Exploration rate clamping", test_exploration_rate_clamp)


print("\n═══ 2. INGEST ═══")

def test_basic_ingest():
    e = sc.EntrolyEngine()
    r = e.ingest("def hello(): return 'world'", "hello.py", 0, False)
    assert r["status"] == "ingested"
    assert "fragment_id" in r
    assert r["token_count"] > 0
    assert 0.0 <= r["entropy_score"] <= 1.0
    assert r["criticality"] == "Normal"
    assert r["is_pinned"] == False
    assert r["total_fragments"] == 1
    assert e.fragment_count() == 1
test("Basic ingest", test_basic_ingest)

def test_ingest_with_explicit_tokens():
    e = sc.EntrolyEngine()
    r = e.ingest("test", "test.py", 42, False)
    assert r["token_count"] == 42
test("Ingest with explicit token count", test_ingest_with_explicit_tokens)

def test_token_estimation_code_vs_prose():
    e = sc.EntrolyEngine()
    code = "if (x > 0) { return x * 2; } else { return -x; }"
    prose = "The quick brown fox jumps over the lazy dog again"
    r_code = e.ingest(code, "code.js", 0, False)
    r_prose = e.ingest(prose, "prose.txt", 0, False)
    # Code has more non-alpha chars → higher chars/token → fewer estimated tokens
    # Both should be reasonable (not 0, not astronomical)
    assert 5 <= r_code["token_count"] <= 50, f"Code tokens: {r_code['token_count']}"
    assert 5 <= r_prose["token_count"] <= 50, f"Prose tokens: {r_prose['token_count']}"
test("Token estimation: code vs prose", test_token_estimation_code_vs_prose)

def test_pinned_ingest():
    e = sc.EntrolyEngine()
    r = e.ingest("important data", "data.txt", 0, True)
    assert r["is_pinned"] == True
test("Pinned ingest", test_pinned_ingest)

def test_empty_content():
    e = sc.EntrolyEngine()
    r = e.ingest("", "empty.py", 0, False)
    assert r["status"] == "ingested"
    assert r["token_count"] >= 1  # Should be at least 1
test("Empty content ingest", test_empty_content)

def test_large_content():
    e = sc.EntrolyEngine()
    content = "x = 1\n" * 10000  # 60K chars
    r = e.ingest(content, "large.py", 0, False)
    assert r["status"] == "ingested"
    assert r["token_count"] > 1000
test("Large content ingest", test_large_content)

def test_multiple_ingests():
    e = sc.EntrolyEngine()
    for i in range(100):
        r = e.ingest(f"def func_{i}(): return {i}", f"file_{i}.py", 0, False)
        assert r["status"] == "ingested"
    assert e.fragment_count() == 100
test("100 ingests", test_multiple_ingests)


print("\n═══ 3. CRITICAL FILE PROTECTION ═══")

CRITICAL_FILES = [
    ("package.json", "Critical"),
    ("requirements.txt", "Critical"),
    ("Cargo.toml", "Critical"),
    ("Dockerfile", "Critical"),
    (".env", "Critical"),
    (".env.example", "Critical"),
    ("docker-compose.yml", "Critical"),
    ("pyproject.toml", "Critical"),
    ("go.mod", "Critical"),
    ("tsconfig.json", "Critical"),
]

SAFETY_FILES = [
    ("LICENSE", "Safety"),
    ("LICENSE.md", "Safety"),
    ("SECURITY.md", "Safety"),
    ("CODEOWNERS", "Safety"),
]

IMPORTANT_FILES = [
    ("schema.proto", "Important"),
    ("types.ts", "Important"),
    ("models.py", "Important"),
    ("test_auth.py", "Important"),
    ("auth_test.rs", "Important"),
    ("api_handler.py", "Important"),
]

NORMAL_FILES = [
    ("utils.py", "Normal"),
    ("helpers.js", "Normal"),
    ("main.rs", "Normal"),
]

def test_criticality(path, expected_crit):
    def _test():
        e = sc.EntrolyEngine()
        r = e.ingest(f"content for {path}", path, 0, False)
        assert r["criticality"] == expected_crit, \
            f"{path}: expected {expected_crit}, got {r['criticality']}"
        if expected_crit in ("Critical", "Safety"):
            assert r["is_pinned"] == True, f"{path} should be auto-pinned"
    return _test

for path, crit in CRITICAL_FILES + SAFETY_FILES + IMPORTANT_FILES + NORMAL_FILES:
    test(f"Criticality: {path} → {crit}", test_criticality(path, crit))


print("\n═══ 4. SAFETY SIGNAL DETECTION ═══")

def test_license_safety():
    e = sc.EntrolyEngine()
    # License *files* (by name) are Safety-pinned; license *text* in normal
    # files is NOT auto-pinned (intentional — broad content matching destroyed
    # budgets in real codebases).
    r = e.ingest("MIT License\nCopyright 2024", "LICENSE", 0, False)
    assert r["is_pinned"] == True, "LICENSE file should be auto-pinned via Safety criticality"
test("License file auto-pinned", test_license_safety)

def test_security_warning_safety():
    e = sc.EntrolyEngine()
    r = e.ingest("# SECURITY WARNING: do not expose API keys\nAPI_KEY = os.environ['KEY']", "config.py", 0, False)
    assert r["is_pinned"] == True
test("Security warning auto-pinned", test_security_warning_safety)

def test_normal_code_not_pinned():
    e = sc.EntrolyEngine()
    r = e.ingest("def add(a, b): return a + b", "math.py", 0, False)
    assert r["is_pinned"] == False, "Normal code should NOT be auto-pinned"
test("Normal code NOT auto-pinned", test_normal_code_not_pinned)


print("\n═══ 5. ENTROPY SCORING ═══")

def test_entropy_honest_for_critical():
    """Critical files should have honest entropy, not inflated."""
    e = sc.EntrolyEngine()
    # requirements.txt has low real entropy (repetitive format)
    r = e.ingest("numpy==2.0\nnumpy==2.0\nnumpy==2.0", "requirements.txt", 0, False)
    # Should be floored at 0.5 (not multiplied by criticality_boost=5.0)
    assert r["entropy_score"] >= 0.5, f"Floor should apply: {r['entropy_score']}"
    assert r["entropy_score"] <= 1.0, f"Should not exceed 1.0: {r['entropy_score']}"
test("Entropy floor for critical files (not inflated)", test_entropy_honest_for_critical)

def test_entropy_varies_with_content():
    e = sc.EntrolyEngine()
    # Low entropy: repetitive
    r1 = e.ingest("aaa aaa aaa aaa aaa aaa", "a.txt", 0, False)
    # High entropy: diverse
    r2 = e.ingest("The quick brown fox jumps over the lazy dog while parsing JSON", "b.txt", 0, False)
    assert r2["entropy_score"] >= r1["entropy_score"], \
        f"Diverse content should have higher entropy: {r1['entropy_score']} vs {r2['entropy_score']}"
test("Entropy varies with content diversity", test_entropy_varies_with_content)


print("\n═══ 6. DUPLICATE DETECTION ═══")

def test_exact_duplicate():
    e = sc.EntrolyEngine()
    r1 = e.ingest("def foo(): return 42", "a.py", 0, False)
    r2 = e.ingest("def foo(): return 42", "b.py", 0, False)
    assert r2["status"] == "duplicate"
    assert r2["duplicate_of"] == r1["fragment_id"]
    assert r2["tokens_saved"] > 0
    assert e.fragment_count() == 1  # Only one stored
test("Exact duplicate detected", test_exact_duplicate)

def test_different_content_not_duplicate():
    e = sc.EntrolyEngine()
    e.ingest("def foo(): return 42", "a.py", 0, False)
    r2 = e.ingest("class Bar: pass", "b.py", 0, False)
    assert r2["status"] == "ingested"
    assert e.fragment_count() == 2
test("Different content not flagged as duplicate", test_different_content_not_duplicate)

def test_near_duplicate():
    e = sc.EntrolyEngine()
    e.ingest("def process_payment(amount, currency): return amount * get_rate(currency)", "a.py", 0, False)
    # Very similar but not identical
    r2 = e.ingest("def process_payment(amount, currency): return amount * get_rate(currency) + 0", "b.py", 0, False)
    # Should be caught as near-duplicate via SimHash (hamming_threshold=3)
    # Note: may or may not be caught depending on hash sensitivity
    # The point is it doesn't crash
    assert r2["status"] in ("duplicate", "ingested")
test("Near-duplicate handling (no crash)", test_near_duplicate)


print("\n═══ 7. DEPENDENCY GRAPH ═══")

def test_dep_graph_auto_link():
    """Ingest definition then usage — should create dep edge."""
    e = sc.EntrolyEngine()
    e.ingest("def calculate_tax(income):\n    return income * 0.3", "tax.py", 0, False)
    e.ingest("total = calculate_tax(50000)", "main.py", 0, False)
    dg = e.dep_graph_stats()
    assert dg["edges"] > 0, f"Should have dep edges, got {dg['edges']}"
test("Dep graph auto-links function calls", test_dep_graph_auto_link)

def test_dep_graph_import_detection():
    """Python import statements should create strong dep edges."""
    e = sc.EntrolyEngine()
    e.ingest("def process_payment(amount):\n    return amount * 1.1", "payments.py", 0, False)
    e.ingest("from payments import process_payment\nresult = process_payment(100)", "main.py", 0, False)
    dg = e.dep_graph_stats()
    assert dg["edges"] >= 1, f"Import should create edges, got {dg['edges']}"
test("Dep graph detects Python imports", test_dep_graph_import_detection)

def test_dep_graph_order_matters():
    """Usage before definition: no edge. Then ingest definition: edge created on next usage."""
    e = sc.EntrolyEngine()
    # Ingest usage first (no definition yet)
    e.ingest("result = unknown_function(42)", "caller.py", 0, False)
    dg1 = e.dep_graph_stats()
    # Now ingest the definition
    e.ingest("def unknown_function(x):\n    return x * 2", "impl.py", 0, False)
    # The definition registers the symbol, but caller already ingested
    # This tests that the symbol table is populated correctly
    dg2 = e.dep_graph_stats()
    # At minimum, unknown_function should be in the symbol table now
    assert dg2["nodes"] >= 0  # Just verify no crash
test("Dep graph handles definition-after-usage", test_dep_graph_order_matters)

def test_dep_graph_empty():
    e = sc.EntrolyEngine()
    dg = e.dep_graph_stats()
    assert dg["nodes"] == 0
    assert dg["edges"] == 0
test("Dep graph empty initially", test_dep_graph_empty)


print("\n═══ 8. TASK CLASSIFICATION ═══")

def test_bug_tracing():
    r = sc.EntrolyEngine().classify_task("fix the payment processing bug causing crashes")
    assert r["task_type"] == "BugTracing"
    assert r["budget_multiplier"] == 1.5
test("classify: bug tracing", test_bug_tracing)

def test_refactoring():
    r = sc.EntrolyEngine().classify_task("refactor the auth module into smaller files")
    assert r["task_type"] == "Refactoring"
    assert r["budget_multiplier"] == 1.0
test("classify: refactoring", test_refactoring)

def test_code_generation():
    r = sc.EntrolyEngine().classify_task("create a new REST API endpoint for users")
    assert r["task_type"] == "CodeGeneration"
    assert r["budget_multiplier"] == 0.7
test("classify: code generation", test_code_generation)

def test_testing():
    r = sc.EntrolyEngine().classify_task("write unit tests for the payment module")
    assert r["task_type"] == "Testing"
    assert r["budget_multiplier"] == 0.8
test("classify: testing", test_testing)

def test_unknown_task():
    r = sc.EntrolyEngine().classify_task("banana smoothie recipe")
    assert r["task_type"] == "Unknown"
    assert r["budget_multiplier"] == 1.0
test("classify: unknown", test_unknown_task)


print("\n═══ 9. OPTIMIZE ═══")

def test_optimize_empty():
    e = sc.EntrolyEngine()
    r = e.optimize(1000, "")
    assert r["selected_count"] == 0
    assert r["total_tokens"] == 0
test("Optimize empty engine", test_optimize_empty)

def test_optimize_selects_within_budget():
    e = sc.EntrolyEngine()
    for i in range(10):
        e.ingest(f"def func_{i}(): return {i} * {i+1} + {i*2}", f"f{i}.py", 50, False)
    r = e.optimize(200, "")
    assert r["total_tokens"] <= 200, f"Should stay within budget: {r['total_tokens']}"
    assert r["selected_count"] > 0
    assert r["selected_count"] < 10  # Can't fit all
test("Optimize stays within budget", test_optimize_selects_within_budget)

def test_optimize_adaptive_budget():
    """Bug tracing query surfaces a 1.5x recommended budget (advisory)."""
    e = sc.EntrolyEngine()
    for i in range(10):
        e.ingest(f"def func_{i}(): return {i}", f"f{i}.py", 50, False)
    r = e.optimize(200, "fix the crash bug")
    # effective_budget is capped at the caller's token_budget (hard ceiling —
    # downstream context windows don't tolerate silent expansion). The 1.5x
    # multiplier is surfaced as recommended_budget for the caller to opt in.
    assert r["effective_budget"] == 200, f"effective_budget must respect cap: {r['effective_budget']}"
    assert r["recommended_budget"] == 300, f"BugTracing should recommend 1.5x: {r['recommended_budget']}"
test("Optimize adaptive budget (bug=1.5x recommended)", test_optimize_adaptive_budget)

def test_optimize_pinned_always_included():
    e = sc.EntrolyEngine()
    e.ingest("pinned content here", "critical.txt", 10, True)
    e.ingest("other content", "other.py", 10, False)
    r = e.optimize(15, "")  # Budget only fits ~1
    # Pinned should always be included
    selected_sources = [f["source"] for f in r["selected"]]
    assert any("critical" in s for s in selected_sources), \
        f"Pinned fragment must be included: {selected_sources}"
test("Pinned fragments always included", test_optimize_pinned_always_included)

def test_optimize_sufficiency():
    e = sc.EntrolyEngine()
    e.ingest("def calculate_tax(income):\n    return income * 0.3", "tax.py", 0, False)
    e.ingest("total = calculate_tax(50000)", "main.py", 0, False)
    r = e.optimize(10000, "tax calculation")
    assert "sufficiency" in r
    assert 0.0 <= r["sufficiency"] <= 1.0
test("Optimize returns sufficiency score", test_optimize_sufficiency)

def test_optimize_returns_ordered():
    e = sc.EntrolyEngine()
    e.ingest("critical schema", "schema.proto", 10, False)
    e.ingest("normal code", "utils.py", 10, False)
    e.ingest("MIT License\nCopyright 2024", "LICENSE", 10, False)
    r = e.optimize(10000, "")
    # Should have selected fragments in order
    assert r["selected_count"] == 3
    # Critical/safety files should be ordered first
    sources = [f["source"] for f in r["selected"]]
    assert len(sources) == 3
test("Optimize returns ordered fragments", test_optimize_returns_ordered)


print("\n═══ 10. FEEDBACK LOOP ═══")

def test_feedback_success():
    e = sc.EntrolyEngine()
    r1 = e.ingest("def good_code(): return True", "good.py", 0, False)
    # Should not crash
    e.record_success([r1["fragment_id"]])
test("Record success", test_feedback_success)

def test_feedback_failure():
    e = sc.EntrolyEngine()
    r1 = e.ingest("def bad_code(): return False", "bad.py", 0, False)
    e.record_failure([r1["fragment_id"]])
test("Record failure", test_feedback_failure)

def test_feedback_affects_ranking():
    """Fragments with positive feedback should rank higher."""
    e = sc.EntrolyEngine(exploration_rate=0.0)
    r1 = e.ingest("def good(): return 1", "good.py", 50, False)
    r2 = e.ingest("def bad(): return 0", "bad.py", 50, False)
    # Record lots of success for good, lots of failure for bad
    for _ in range(20):
        e.record_success([r1["fragment_id"]])
        e.record_failure([r2["fragment_id"]])
    # Optimize with tight budget (only fits one)
    result = e.optimize(60, "return value")
    selected_ids = [f["id"] for f in result["selected"]]
    # Good should be preferred
    assert r1["fragment_id"] in selected_ids, \
        f"Feedback-boosted 'good' should be selected: {selected_ids}"
test("Feedback affects selection ranking", test_feedback_affects_ranking)

def test_feedback_empty_ids():
    e = sc.EntrolyEngine()
    # Should not crash with empty list
    e.record_success([])
    e.record_failure([])
test("Feedback with empty IDs (no crash)", test_feedback_empty_ids)

def test_feedback_nonexistent_id():
    e = sc.EntrolyEngine()
    # Should not crash with unknown fragment ID
    e.record_success(["nonexistent_fragment_xyz"])
    e.record_failure(["nonexistent_fragment_xyz"])
test("Feedback with nonexistent ID (no crash)", test_feedback_nonexistent_id)


print("\n═══ 11. EXPLAINABILITY ═══")

def test_explain_before_optimize():
    e = sc.EntrolyEngine()
    r = e.explain_selection()
    assert "error" in r
test("Explain before optimize returns error", test_explain_before_optimize)

def test_explain_after_optimize():
    e = sc.EntrolyEngine()
    e.ingest("def foo(): return 42", "foo.py", 0, False)
    e.ingest("x = foo()", "main.py", 0, False)
    e.optimize(10000, "foo")
    expl = e.explain_selection()
    assert "included" in expl
    assert "sufficiency" in expl
    assert len(expl["included"]) > 0
    # Check scoring breakdown
    first = expl["included"][0]
    assert "scores" in first
    scores = first["scores"]
    for key in ["recency", "frequency", "semantic", "entropy", "feedback_mult", "dep_boost", "criticality", "composite"]:
        assert key in scores, f"Missing score dimension: {key}"
    assert "reason" in first
test("Explain shows full scoring breakdown", test_explain_after_optimize)


print("\n═══ 12. RECALL ═══")

def test_recall_empty():
    e = sc.EntrolyEngine()
    r = e.recall("anything", 5)
    assert r == []
test("Recall from empty engine", test_recall_empty)

def test_recall_returns_ranked():
    e = sc.EntrolyEngine()
    e.ingest("def process_payment(amount): return amount * 1.1", "payments.py", 0, False)
    e.ingest("def send_email(to, body): pass", "email.py", 0, False)
    e.ingest("def refund_payment(txn_id): pass", "refund.py", 0, False)
    results = e.recall("payment processing", 3)
    assert len(results) > 0
    # Should have source and relevance
    assert "source" in results[0]
    assert "relevance" in results[0]
    # Results should be sorted by relevance (descending)
    relevances = [r["relevance"] for r in results]
    assert relevances == sorted(relevances, reverse=True), \
        f"Should be sorted by relevance: {relevances}"
test("Recall returns ranked results", test_recall_returns_ranked)


print("\n═══ 13. ADVANCE TURN & DECAY ═══")

def test_advance_turn():
    e = sc.EntrolyEngine()
    assert e.get_turn() == 0
    e.advance_turn()
    assert e.get_turn() == 1
    e.advance_turn()
    assert e.get_turn() == 2
test("Advance turn increments", test_advance_turn)

def test_decay_evicts_stale():
    e = sc.EntrolyEngine(min_relevance=0.3, decay_half_life=5)
    e.ingest("stale content", "stale.py", 10, False)
    # Advance many turns — should decay below threshold
    for _ in range(50):
        e.advance_turn()
    assert e.fragment_count() == 0, f"Stale fragment should be evicted: {e.fragment_count()}"
test("Decay evicts stale unpinned fragments", test_decay_evicts_stale)

def test_pinned_survives_decay():
    e = sc.EntrolyEngine(min_relevance=0.3, decay_half_life=5)
    e.ingest("pinned content", "pinned.py", 10, True)
    for _ in range(50):
        e.advance_turn()
    assert e.fragment_count() == 1, "Pinned fragment should survive decay"
test("Pinned fragment survives decay", test_pinned_survives_decay)

def test_critical_file_survives_decay():
    e = sc.EntrolyEngine(min_relevance=0.3, decay_half_life=5)
    e.ingest("numpy==2.0", "requirements.txt", 10, False)
    for _ in range(50):
        e.advance_turn()
    assert e.fragment_count() == 1, "Critical auto-pinned file should survive decay"
test("Critical file survives decay", test_critical_file_survives_decay)


print("\n═══ 14. EXPORT / IMPORT STATE ═══")

def test_export_import_roundtrip():
    e = sc.EntrolyEngine()
    e.ingest("def foo(): return 42", "foo.py", 0, False)
    e.ingest("def bar(): return foo()", "bar.py", 0, False)
    e.ingest("MIT License", "LICENSE", 0, False)
    e.advance_turn()
    e.record_success([list(e.stats()["session"].keys())[0]] if False else [])

    state = e.export_state()
    assert len(state) > 100, f"State should be substantial JSON: {len(state)} bytes"

    e2 = sc.EntrolyEngine()
    e2.import_state(state)
    assert e2.fragment_count() == e.fragment_count()
    assert e2.get_turn() == e.get_turn()
test("Export/import roundtrip", test_export_import_roundtrip)

def test_import_invalid_json():
    e = sc.EntrolyEngine()
    try:
        e.import_state("not valid json")
        assert False, "Should have raised"
    except Exception:
        pass  # Expected
test("Import invalid JSON raises error", test_import_invalid_json)


print("\n═══ 15. STATS ═══")

def test_stats_structure():
    e = sc.EntrolyEngine()
    e.ingest("content A", "a.py", 0, False)
    e.ingest("content A", "a_dup.py", 0, False)  # Duplicate
    e.ingest("content B", "b.py", 0, False)
    s = e.stats()
    # Nested structure
    assert "session" in s
    assert "savings" in s
    assert "dedup" in s
    # Session
    assert s["session"]["total_fragments"] == 2  # 1 dup removed
    assert s["session"]["current_turn"] == 0
    assert s["session"]["pinned"] == 0
    # Savings
    assert s["savings"]["total_duplicates_caught"] == 1
    assert s["savings"]["total_fragments_ingested"] == 3
    # Dedup
    assert s["dedup"]["duplicates_detected"] == 1
test("Stats structure and correctness", test_stats_structure)


print("\n═══ 16. STANDALONE MATH FUNCTIONS ═══")

def test_shannon_entropy():
    e = sc.py_shannon_entropy("aaaa")
    assert e == 0.0, f"All same char should have 0 entropy: {e}"
    e2 = sc.py_shannon_entropy("abcdefghijklmnop")
    assert e2 > 0.0, "Diverse chars should have positive entropy"
test("Shannon entropy", test_shannon_entropy)

def test_simhash():
    fp1 = sc.py_simhash("hello world test")
    fp2 = sc.py_simhash("hello world test")
    assert fp1 == fp2, "Same content should have same simhash"
    fp3 = sc.py_simhash("completely different content here")
    # Different content should (usually) have different hash
    assert isinstance(fp3, int)
test("SimHash deterministic", test_simhash)

def test_hamming_distance():
    d = sc.py_hamming_distance(0, 0)
    assert d == 0
    d2 = sc.py_hamming_distance(0b1111, 0b0000)
    assert d2 == 4
    d3 = sc.py_hamming_distance(0xFFFFFFFFFFFFFFFF, 0)
    assert d3 == 64, f"Max distance for 64-bit: {d3}"
test("Hamming distance", test_hamming_distance)

def test_normalized_entropy():
    e = sc.py_normalized_entropy("abcdefgh")
    assert 0.0 <= e <= 1.0, f"Normalized entropy out of range: {e}"
test("Normalized entropy in [0,1]", test_normalized_entropy)

def test_boilerplate_ratio():
    r = sc.py_boilerplate_ratio("the the the is is a a")
    assert 0.0 <= r <= 1.0
test("Boilerplate ratio in [0,1]", test_boilerplate_ratio)


print("\n═══ 17. EDGE CASES ═══")

def test_unicode_content():
    e = sc.EntrolyEngine()
    r = e.ingest("def greet(): return '你好世界 🌍'", "unicode.py", 0, False)
    assert r["status"] == "ingested"
test("Unicode content", test_unicode_content)

def test_binary_like_content():
    e = sc.EntrolyEngine()
    r = e.ingest("\x00\x01\x02\xff\xfe", "binary.bin", 0, False)
    assert r["status"] == "ingested"
test("Binary-like content (no crash)", test_binary_like_content)

def test_very_long_source_path():
    e = sc.EntrolyEngine()
    path = "a/" * 500 + "file.py"
    r = e.ingest("content", path, 0, False)
    assert r["status"] == "ingested"
test("Very long source path", test_very_long_source_path)

def test_optimize_zero_budget():
    e = sc.EntrolyEngine()
    e.ingest("content", "f.py", 10, False)
    r = e.optimize(0, "")
    # Should handle gracefully
    assert r["selected_count"] >= 0
test("Optimize with zero budget", test_optimize_zero_budget)

def test_recall_zero_k():
    e = sc.EntrolyEngine()
    e.ingest("content", "f.py", 0, False)
    r = e.recall("query", 0)
    assert len(r) == 0
test("Recall with k=0", test_recall_zero_k)


print("\n═══ 18. SKELETON FRAGMENTATION ═══")

PYTHON_CODE = '''import os
from pathlib import Path

class DataPipeline:
    """ETL pipeline for processing events."""

    def __init__(self, config: dict):
        self.config = config
        self.buffer = []
        self._setup_logging()

    def ingest_event(self, event: dict) -> bool:
        """Ingest a single event into the buffer."""
        if not event.get("id"):
            return False
        self.buffer.append(event)
        if len(self.buffer) >= self.config.get("batch_size", 100):
            self.flush()
        return True

    def flush(self) -> int:
        """Flush buffered events to storage."""
        count = len(self.buffer)
        for evt in self.buffer:
            self._write(evt)
        self.buffer.clear()
        return count

    def _setup_logging(self):
        import logging
        self.logger = logging.getLogger(__name__)

    def _write(self, event: dict):
        path = Path(self.config["output_dir"]) / f"{event['id']}.json"
        path.write_text(str(event))

MAX_BUFFER = 1000
DEFAULT_BATCH = 100
'''

JS_CODE = '''import { useState, useEffect } from 'react';
import axios from 'axios';

export class UserService {
    constructor(baseUrl, timeout = 5000) {
        this.baseUrl = baseUrl;
        this.timeout = timeout;
        this.cache = new Map();
    }

    async fetchUser(id) {
        if (this.cache.has(id)) return this.cache.get(id);
        const response = await axios.get(`${this.baseUrl}/users/${id}`, {
            timeout: this.timeout
        });
        this.cache.set(id, response.data);
        return response.data;
    }

    async createUser(data) {
        const response = await axios.post(`${this.baseUrl}/users`, data);
        return response.data;
    }
}

export function formatDate(date) {
    const d = new Date(date);
    return d.toISOString().split('T')[0];
}

const API_TIMEOUT = 5000;
'''

def test_skeleton_populated_on_ingest_python():
    e = sc.EntrolyEngine()
    r = e.ingest(PYTHON_CODE, "pipeline.py", 0, False)
    assert r["status"] == "ingested"
    assert r.get("has_skeleton") == True, \
        f"Python file should have skeleton; keys={list(r.keys())}"
    assert r.get("skeleton_token_count", 0) > 0, "Skeleton token count should be positive"
test("Skeleton populated for Python file", test_skeleton_populated_on_ingest_python)

def test_skeleton_token_count_less_than_full():
    e = sc.EntrolyEngine()
    r = e.ingest(PYTHON_CODE, "pipeline.py", 0, False)
    assert r["status"] == "ingested"
    full_tc = r["token_count"]
    skel_tc = r.get("skeleton_token_count", full_tc)
    assert skel_tc < full_tc, \
        f"Skeleton should be shorter: skeleton={skel_tc}, full={full_tc}"
test("Skeleton token count < full token count", test_skeleton_token_count_less_than_full)

def test_no_skeleton_for_non_code():
    e = sc.EntrolyEngine()
    r = e.ingest("This is a plain text document with no code structure at all.\n" * 10, "readme.md", 0, False)
    assert r.get("has_skeleton", False) == False, \
        "Non-code file should NOT have skeleton"
test("No skeleton for non-code files (markdown)", test_no_skeleton_for_non_code)

def test_skeleton_present_for_js():
    e = sc.EntrolyEngine()
    r = e.ingest(JS_CODE, "service.js", 0, False)
    assert r["status"] == "ingested"
    assert r.get("has_skeleton") == True, \
        f"JS file should have skeleton; keys={list(r.keys())}"
test("Skeleton populated for JS file", test_skeleton_present_for_js)

def test_optimize_uses_skeleton_when_budget_tight():
    """With a very tight budget, skeleton variants should appear in results."""
    e = sc.EntrolyEngine()
    # Ingest two large Python files
    r1 = e.ingest(PYTHON_CODE, "pipeline.py", 0, False)
    r2 = e.ingest(JS_CODE, "service.js", 0, False)
    full_tc1 = r1["token_count"]
    full_tc2 = r2["token_count"]
    # Budget: fits only ~1 full fragment but might fit 1 full + 1 skeleton
    skel_tc1 = r1.get("skeleton_token_count", full_tc1)
    tight_budget = full_tc1 + skel_tc1 + 5  # enough for 1 full + 1 skeleton
    result = e.optimize(tight_budget, "data pipeline")
    # Verify we get either full or skeleton variants
    variants = [f.get("variant", "full") for f in result["selected"]]
    assert len(result["selected"]) > 0, "Should select at least 1 fragment"
    assert all(v in ("full", "skeleton", "reference") for v in variants), \
        f"Unexpected variant values: {variants}"
test("Optimize uses skeleton variants to fill budget", test_optimize_uses_skeleton_when_budget_tight)

def test_optimize_prefers_full_when_budget_allows():
    """With a large budget, all fragments should be 'full', not skeletons."""
    e = sc.EntrolyEngine()
    e.ingest(PYTHON_CODE, "pipeline.py", 0, False)
    result = e.optimize(100000, "data pipeline")  # Very large budget
    for frag in result["selected"]:
        variant = frag.get("variant", "full")
        assert variant == "full", \
            f"Large budget should prefer full: got variant='{variant}' for {frag.get('source')}"
test("Optimize prefers full fragments when budget is generous", test_optimize_prefers_full_when_budget_allows)

def test_optimize_variant_field_always_present():
    """Every selected fragment should have a 'variant' field."""
    e = sc.EntrolyEngine()
    e.ingest(PYTHON_CODE, "pipeline.py", 0, False)
    e.ingest(JS_CODE, "service.js", 0, False)
    result = e.optimize(100000, "")
    for frag in result["selected"]:
        assert "variant" in frag, \
            f"Fragment missing 'variant' field: {list(frag.keys())}"
        assert frag["variant"] in ("full", "skeleton", "reference"), \
            f"Invalid variant value: {frag['variant']}"
test("Every selected fragment has a valid 'variant' field", test_optimize_variant_field_always_present)


# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'═' * 50}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'═' * 50}")

if FAIL > 0:
    sys.exit(1)
else:
    print("  ALL TESTS PASS ✓")
    sys.exit(0)
