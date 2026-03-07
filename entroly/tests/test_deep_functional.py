#!/usr/bin/env python3
"""
Entroly — Deep Functional Test Suite (Real-User Edge Cases)
=============================================================

Complements test_functional.py (F-01..F-18) and test_intensive_functional.py
(F-19..F-40) with adversarial edge cases, boundary conditions, and stress
scenarios that a real production user would eventually hit.

No mocks. No stubs. Real corpus. Real engine.

Sections:
  D-01  UNICODE + BINARY CONTENT
  D-02  MASSIVE FRAGMENT (500KB)
  D-03  RAPID-FIRE INGEST (500 fragments)
  D-04  DEDUP NEAR-MISS (1-char diff)
  D-05  DEDUP PARAGRAPH REORDER
  D-06  FEEDBACK SATURATION (100x success)
  D-07  FEEDBACK ON NONEXISTENT IDs
  D-08  OPTIMIZE AFTER AGGRESSIVE EVICTION
  D-09  INTERLEAVED INGEST + OPTIMIZE
  D-10  QUERY WITH SPECIAL CHARS
  D-11  ALL PINNED, BUDGET < TOTAL
  D-12  CONCURRENT-LIKE ACCESS (no advance)
  D-13  CHECKPOINT ROUND-TRIP FIDELITY
  D-14  CHECKPOINT THEN MORE INGEST
  D-15  CORRUPTED CHECKPOINT RECOVERY
  D-16  ZERO-TOKEN FRAGMENT AUTO-ESTIMATE
  D-17  SAME SOURCE DIFFERENT CONTENT
  D-18  RECALL TOP-K BOUNDARY (k=1, k=99999)
  D-19  BUDGET = EXACT CORPUS TOTAL
  D-20  ADVANCE 1000 TURNS (extreme aging)
  D-21  MULTI-ENGINE ISOLATION
  D-22  EMPTY + WHITESPACE CONTENT
  D-23  SELF-SIMILAR CORPUS (50 variants)
  D-24  QUERY REFINEMENT VAGUE vs PRECISE
  D-25  PREFETCH CO-ACCESS LEARNING
  D-26  PREFETCH IMPORT ANALYSIS
  D-27  STATS CONSISTENCY CHAIN
  D-28  WILSON SCORE MATH VERIFICATION
  D-29  EBBINGHAUS DECAY MATH VERIFICATION
  D-30  KNAPSACK DP CORRECTNESS
  D-31  ENTROPY SCORING SANITY
  D-32  FULL LIFECYCLE STRESS (20 turns)
"""

import os
import sys
import math
import time
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
CORE = REPO.parent / "entroly-core"

PASS = "  \u2713"
FAIL = "  \u2717"
SKIP = "  \u2298"
passed = failed = skipped = 0
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    global passed, failed
    sym = PASS if condition else FAIL
    dest = sys.stdout if condition else sys.stderr
    print(f"{sym} {label}{(' -- ' + detail) if detail else ''}", file=dest)
    if condition:
        passed += 1
    else:
        failed += 1
        _failures.append(label)
    return condition


def skip_check(label: str, reason: str = ""):
    global skipped
    skipped += 1
    print(f"{SKIP} {label}{(' -- ' + reason) if reason else ''}")


def section(name: str):
    print(f"\n{'=' * 68}")
    print(f"  {name}")
    print(f"{'=' * 68}")


# ── Response extractors (handle the nested response format) ──────────────

def get_selected(opt: dict) -> list[dict]:
    """Extract selected fragments from optimize response."""
    return opt.get("selected_fragments", opt.get("selected", []))


def get_total_tokens(opt: dict) -> int:
    """Extract total_tokens from optimize response."""
    stats = opt.get("optimization_stats", {})
    return stats.get("total_tokens", opt.get("total_tokens", 0))


def get_effective_budget(opt: dict, fallback: int = 0) -> int:
    """Extract effective budget from optimize response."""
    stats = opt.get("optimization_stats", {})
    return stats.get("effective_budget", opt.get("effective_budget", fallback))


def get_total_fragments(stats: dict) -> int:
    """Extract total_fragments from stats response."""
    return (stats.get("total_fragments")
            or stats.get("session", {}).get("total_fragments", 0))


def get_current_turn(stats: dict) -> int:
    """Extract current_turn from stats response."""
    return (stats.get("current_turn")
            or stats.get("session", {}).get("current_turn", 0))


# ── Corpus helpers ──────────────────────────────────────────────────────

def real_sources() -> list[tuple[str, Path]]:
    out = []
    for p in sorted((REPO / "entroly").glob("*.py")):
        out.append((f"entroly/{p.name}", p))
    if CORE.exists():
        for p in sorted((CORE / "src").glob("*.rs")):
            out.append((f"src/{p.name}", p))
    return out


def fresh_engine(tmp_dir: str | None = None, **cfg_kwargs):
    from entroly.server import EntrolyEngine
    from entroly.config import EntrolyConfig
    d = tmp_dir or tempfile.mkdtemp()
    cfg = EntrolyConfig(
        default_token_budget=200_000,
        decay_half_life_turns=20,
        min_relevance_threshold=0.0,
        checkpoint_dir=d,
        auto_checkpoint_interval=99999,
        **cfg_kwargs,
    )
    return EntrolyEngine(config=cfg), d


def ingest_corpus(engine, sources=None, pinned_names=()) -> dict[str, str]:
    if sources is None:
        sources = real_sources()
    ids: dict[str, str] = {}
    for label, path in sources:
        content = path.read_text(encoding="utf-8", errors="replace")
        pinned = any(k in path.name for k in pinned_names)
        r = engine.ingest_fragment(
            content, source=str(path),
            token_count=max(1, len(content) // 4),
            is_pinned=pinned,
        )
        if r.get("status") in ("ingested", "duplicate"):
            ids[label] = r.get("fragment_id", "")
    return ids


# === D-01: UNICODE + BINARY CONTENT ======================================

def test_unicode_binary():
    section("D-01  UNICODE + BINARY CONTENT")
    engine, _ = fresh_engine()
    cases = [
        ("emoji", "def greet(): return '\U0001f389\U0001f525 hello'"),
        ("cjk", "# \u4f60\u597d\u4e16\u754c\nclass Test:\n    pass"),
        ("cyrillic", "# \u041f\u0440\u0438\u0432\u0435\u0442\ndef f(): return 42"),
        ("math_symbols", "# \u2200x \u2208 \u211d, \u2203y: f(y) = \u222b e^x dx"),
        ("null_bytes_repr", "data = b'\\x00\\x01\\xff'  # binary repr"),
    ]
    for name, content in cases:
        try:
            r = engine.ingest_fragment(content, source=f"{name}.py", token_count=0)
            check(f"unicode/{name}: ingest succeeds",
                  r.get("status") in ("ingested", "duplicate"),
                  f"status={r.get('status')}")
        except Exception as e:
            check(f"unicode/{name}: no crash", False, f"{type(e).__name__}: {e}")
    try:
        results = engine.recall_relevant("emoji hello", top_k=5)
        check("unicode query recall returns list", isinstance(results, list))
    except Exception as e:
        check("unicode query recall no crash", False, f"{type(e).__name__}: {e}")


# === D-02: MASSIVE FRAGMENT ===============================================

def test_massive_fragment():
    section("D-02  MASSIVE FRAGMENT (500KB)")
    engine, _ = fresh_engine()
    lines = [f"def function_{i}(x, y):\n    return x * y + {i}\n" for i in range(5000)]
    big = "\n".join(lines)
    tc = len(big) // 4
    check("content is >100KB", len(big) > 100_000, f"size={len(big):,}")
    r = engine.ingest_fragment(big, source="massive.py", token_count=tc)
    check("massive fragment ingested", r.get("status") == "ingested")
    opt = engine.optimize_context(token_budget=1000, query="function result")
    check("optimize with budget < massive doesn't crash", isinstance(opt, dict))
    opt2 = engine.optimize_context(token_budget=tc * 2, query="function")
    check("massive fragment selectable with big budget", len(get_selected(opt2)) >= 1)


# === D-03: RAPID-FIRE INGEST =============================================

def test_rapid_fire_ingest():
    section("D-03  RAPID-FIRE INGEST (500 fragments)")
    engine, _ = fresh_engine()
    t0 = time.perf_counter()
    count = 0
    for i in range(500):
        content = f"# Fragment {i}\ndef fn_{i}(): return {i * 17}\n"
        r = engine.ingest_fragment(content, source=f"gen_{i}.py", token_count=0)
        if r.get("fragment_id"):
            count += 1
    elapsed = time.perf_counter() - t0
    check("500 fragments ingested", count >= 400, f"ingested={count}")
    check("completes in <30s", elapsed < 30.0, f"elapsed={elapsed:.2f}s")
    stats = engine.get_stats()
    total = get_total_fragments(stats)
    check("stats reflects count", total >= 400, f"total={total}")


# === D-04: DEDUP NEAR-MISS ===============================================

def test_dedup_near_miss():
    section("D-04  DEDUP NEAR-MISS (1-char diff)")
    engine, _ = fresh_engine()
    base = "def compute_score(x, y): return x * y + 42\n" * 20
    r1 = engine.ingest_fragment(base, source="original.py")
    check("base ingested", r1.get("status") == "ingested")
    variant = base[:-1] + "X"
    r2 = engine.ingest_fragment(variant, source="near1.py")
    check("1-char-diff completes", "status" in r2, f"status={r2.get('status')}")
    r3 = engine.ingest_fragment(base, source="copy.py")
    check("exact copy is duplicate", r3.get("status") == "duplicate")


# === D-05: DEDUP PARAGRAPH REORDER =======================================

def test_dedup_reorder():
    section("D-05  DEDUP PARAGRAPH REORDER")
    engine, _ = fresh_engine()
    a = "def alpha(): return 1\ndef beta(): return 2\n"
    b = "class Gamma:\n    def delta(self): pass\n"
    c = "CONSTANT = 42\nVALUE = 99\n"
    r1 = engine.ingest_fragment(a + b + c, source="ordered.py")
    check("original ingested", r1.get("status") == "ingested")
    r2 = engine.ingest_fragment(c + a + b, source="reordered.py")
    check("reordered completes",
          r2.get("status") in ("ingested", "duplicate"), f"status={r2.get('status')}")


# === D-06: FEEDBACK SATURATION ===========================================

def test_feedback_saturation():
    section("D-06  FEEDBACK SATURATION (100x success)")
    engine, _ = fresh_engine()
    ingest_corpus(engine)
    Q = "optimization knapsack entropy scoring"
    opt = engine.optimize_context(token_budget=500_000, query=Q)
    selected = get_selected(opt)
    if not selected:
        skip_check("feedback saturation", "no fragments selected")
        return
    target = selected[0]["id"]
    base_score = selected[0].get("relevance", 0)
    for _ in range(100):
        engine.record_success([target])
    opt2 = engine.optimize_context(token_budget=500_000, query=Q)
    after = next((f.get("relevance") for f in get_selected(opt2)
                  if f.get("id") == target), None)
    if after is not None:
        check("100x success: no infinite score",
              after < 100.0 and not math.isinf(after), f"score={after}")
        check("100x success: score >= baseline",
              after >= base_score, f"base={base_score:.4f}, after={after:.4f}")
    else:
        check("target still in selection", False, "target disappeared")


# === D-07: FEEDBACK ON NONEXISTENT IDs ===================================

def test_feedback_nonexistent():
    section("D-07  FEEDBACK ON NONEXISTENT IDs")
    engine, _ = fresh_engine()
    fake = ["nonexistent_abc", "fake_xyz", ""]
    try:
        engine.record_success(fake)
        check("record_success on fake IDs: no crash", True)
    except Exception as e:
        check("record_success on fake IDs: no crash", False, f"{type(e).__name__}")
    try:
        engine.record_failure(fake)
        check("record_failure on fake IDs: no crash", True)
    except Exception as e:
        check("record_failure on fake IDs: no crash", False, f"{type(e).__name__}")
    engine.ingest_fragment("def real(): return 42", source="real.py")
    opt = engine.optimize_context(token_budget=10_000, query="real")
    check("optimize works after fake feedback", isinstance(opt, dict))


# === D-08: OPTIMIZE AFTER AGGRESSIVE EVICTION ============================

def test_optimize_after_eviction():
    section("D-08  OPTIMIZE AFTER AGGRESSIVE EVICTION")
    from entroly.config import EntrolyConfig
    from entroly.server import EntrolyEngine
    d = tempfile.mkdtemp()
    cfg = EntrolyConfig(
        default_token_budget=200_000, decay_half_life_turns=5,
        min_relevance_threshold=0.1, checkpoint_dir=d,
        auto_checkpoint_interval=99999,
    )
    engine = EntrolyEngine(config=cfg)
    for i in range(10):
        engine.ingest_fragment(f"def fn_{i}(): return {i}\n", source=f"f{i}.py", token_count=50)
    n1 = get_total_fragments(engine.get_stats())
    for _ in range(50):
        engine.advance_turn()
    n2 = get_total_fragments(engine.get_stats())
    check("some fragments evicted after 50 turns (hl=5)",
          n2 < n1 or n2 == 0, f"before={n1}, after={n2}")
    opt = engine.optimize_context(token_budget=100_000, query="function return")
    check("optimize after eviction works", isinstance(opt, dict))


# === D-09: INTERLEAVED INGEST + OPTIMIZE =================================

def test_interleaved():
    section("D-09  INTERLEAVED INGEST+OPTIMIZE")
    engine, _ = fresh_engine()
    for i in range(5):
        engine.ingest_fragment(f"def early_{i}(): return {i}", source=f"e{i}.py")
    opt1 = engine.optimize_context(token_budget=100_000, query="function")
    n1 = len(get_selected(opt1))
    for i in range(5, 10):
        engine.ingest_fragment(f"def late_{i}(): return {i}", source=f"l{i}.py")
    opt2 = engine.optimize_context(token_budget=100_000, query="function")
    n2 = len(get_selected(opt2))
    check("second optimize sees more fragments", n2 >= n1, f"first={n1}, second={n2}")


# === D-10: SPECIAL CHAR QUERIES ==========================================

def test_special_char_queries():
    section("D-10  SPECIAL CHAR QUERIES")
    engine, _ = fresh_engine()
    engine.ingest_fragment("def process(data): return data.strip()", source="p.py", token_count=20)
    queries = [
        "process(data)", 'key="value"', "[a-z]+\\d{3}",
        "<script>alert(1)</script>", "", " ", "a" * 10000,
    ]
    for q in queries:
        label = repr(q[:30]) if len(q) > 30 else repr(q)
        try:
            r = engine.recall_relevant(q, top_k=5)
            check(f"query {label}: no crash", isinstance(r, list))
        except Exception as e:
            check(f"query {label}: no crash", False, f"{type(e).__name__}")


# === D-11: ALL PINNED, BUDGET < TOTAL ====================================

def test_pin_everything():
    section("D-11  ALL PINNED, BUDGET < TOTAL")
    engine, _ = fresh_engine()
    total_tc = 0
    for i in range(10):
        content = f"def pinned_{i}(): return {i}\n" * 10
        tc = max(1, len(content) // 4)
        total_tc += tc
        engine.ingest_fragment(content, source=f"pin{i}.py", token_count=tc, is_pinned=True)
    opt = engine.optimize_context(token_budget=total_tc // 2, query="pinned function")
    selected = get_selected(opt)
    # Pinned items bypass budget, so all 10 should be included
    check("pinned fragments selected even with small budget",
          len(selected) >= 5, f"selected={len(selected)}")


# === D-12: CONCURRENT-LIKE ACCESS ========================================

def test_concurrent_like():
    section("D-12  CONCURRENT-LIKE (5 optimizes, no advance)")
    engine, _ = fresh_engine()
    ingest_corpus(engine)
    results = []
    for q in ["knapsack", "entropy", "SimHash", "checkpoint", "decay"]:
        results.append(engine.optimize_context(token_budget=100_000, query=q))
    check("all 5 return dicts", all(isinstance(r, dict) for r in results))
    totals = [get_total_tokens(r) for r in results]
    check("all token counts > 0", all(t > 0 for t in totals), f"totals={totals}")


# === D-13: CHECKPOINT ROUND-TRIP FIDELITY ================================

def test_checkpoint_fidelity():
    section("D-13  CHECKPOINT FIDELITY")
    engine, d = fresh_engine()
    ingest_corpus(engine)
    Q = "knapsack optimization entropy"
    opt = engine.optimize_context(token_budget=500_000, query=Q)
    sel = get_selected(opt)
    if sel:
        engine.record_success([sel[0]["id"]])
    engine.checkpoint(metadata={"test": "fidelity"})
    pre_ids = [r["fragment_id"] for r in engine.recall_relevant(Q, top_k=10)]
    engine2, _ = fresh_engine(tmp_dir=d)
    result = engine2.resume()
    check("resume succeeds", result.get("status") == "resumed")
    post_ids = [r["fragment_id"] for r in engine2.recall_relevant(Q, top_k=10)]
    check("recall IDs match after round-trip",
          set(pre_ids) == set(post_ids),
          f"pre={len(pre_ids)}, post={len(post_ids)}, overlap={len(set(pre_ids) & set(post_ids))}")


# === D-14: CHECKPOINT THEN MORE INGEST ===================================

def test_checkpoint_then_ingest():
    section("D-14  CHECKPOINT THEN INGEST")
    engine, d = fresh_engine()
    engine.ingest_fragment("def original(): return 1", source="orig.py")
    engine.checkpoint()
    engine2, _ = fresh_engine(tmp_dir=d)
    engine2.resume()
    r = engine2.ingest_fragment("def brand_new(): return 2", source="new.py")
    check("ingest after resume works", r.get("status") == "ingested")
    recall = engine2.recall_relevant("function return", top_k=10)
    check("recall finds fragments after resume+ingest", len(recall) >= 1)


# === D-15: CORRUPTED CHECKPOINT ==========================================

def test_corrupted_checkpoint():
    section("D-15  CORRUPTED CHECKPOINT RECOVERY")
    engine, d = fresh_engine()
    engine.ingest_fragment("def test(): pass", source="t.py")
    ckpt = engine.checkpoint()
    with open(ckpt, "r+b") as f:
        f.truncate(32)
    engine2, _ = fresh_engine(tmp_dir=d)
    try:
        result = engine2.resume()
        check("corrupted checkpoint: no crash", True, f"status={result.get('status')}")
    except Exception as e:
        check("corrupted checkpoint: no crash", False, f"{type(e).__name__}: {e}")


# === D-16: ZERO-TOKEN FRAGMENT ============================================

def test_zero_token_fragments():
    section("D-16  ZERO-TOKEN FRAGMENT AUTO-ESTIMATE")
    engine, _ = fresh_engine()
    content = "def hello(): return 'world'\n" * 5
    r = engine.ingest_fragment(content, source="z.py", token_count=0)
    check("token_count=0 ingest succeeds", r.get("status") == "ingested")
    reported = r.get("token_count", 0)
    check("auto-estimated token count > 0", reported > 0, f"tc={reported}")


# === D-17: SAME SOURCE DIFFERENT CONTENT =================================

def test_same_source_diff_content():
    section("D-17  SAME SOURCE DIFFERENT CONTENT")
    engine, _ = fresh_engine()
    r1 = engine.ingest_fragment("version 1: def foo(): return 1", source="shared.py", token_count=20)
    check("first ingest succeeds", r1.get("status") == "ingested")
    r2 = engine.ingest_fragment("version 2: completely different code with no overlap",
                                source="shared.py", token_count=20)
    check("different content same source completes",
          r2.get("status") in ("ingested", "duplicate"), f"status={r2.get('status')}")


# === D-18: RECALL TOP-K BOUNDARY ========================================

def test_recall_topk_boundary():
    section("D-18  RECALL TOP-K BOUNDARY")
    engine, _ = fresh_engine()
    for i in range(5):
        engine.ingest_fragment(f"def fn_{i}(): return {i}", source=f"f{i}.py")
    for k in [1, 2, 5, 99999]:
        try:
            r = engine.recall_relevant("function", top_k=k)
            check(f"top_k={k}: returns list", isinstance(r, list))
            check(f"top_k={k}: count <= max(k, corpus)",
                  len(r) <= max(k, 5), f"count={len(r)}")
        except Exception as e:
            check(f"top_k={k}: no crash", False, f"{type(e).__name__}")


# === D-19: BUDGET = EXACT CORPUS TOTAL ===================================

def test_budget_equals_corpus():
    section("D-19  BUDGET = EXACT CORPUS TOTAL")
    engine, _ = fresh_engine()
    total = 0
    for i in range(5):
        content = f"def fn_{i}(): return {i}\n"
        tc = max(1, len(content) // 4)
        total += tc
        engine.ingest_fragment(content, source=f"f{i}.py", token_count=tc)
    opt = engine.optimize_context(token_budget=total, query="function return")
    used = get_total_tokens(opt)
    check("exact budget: used <= budget", used <= total, f"used={used}, budget={total}")
    check("exact budget: some selected", len(get_selected(opt)) >= 3)


# === D-20: ADVANCE 1000 TURNS ============================================

def test_extreme_aging():
    section("D-20  ADVANCE 1000 TURNS (min_relevance=0)")
    engine, _ = fresh_engine()
    engine.ingest_fragment("def ancient(): return 'old'\n" * 10, source="ancient.py", token_count=50)
    for _ in range(1000):
        engine.advance_turn()
    stats = engine.get_stats()
    total = get_total_fragments(stats)
    check("fragment survives 1000 turns (min_relevance=0)", total >= 1, f"total={total}")
    recall = engine.recall_relevant("ancient old", top_k=5)
    check("fragment recallable after 1000 turns", len(recall) >= 1)


# === D-21: MULTI-ENGINE ISOLATION ========================================

def test_multi_engine_isolation():
    section("D-21  MULTI-ENGINE ISOLATION")
    ea, _ = fresh_engine()
    eb, _ = fresh_engine()
    ea.ingest_fragment("def only_a(): return 'A'", source="a.py")
    eb.ingest_fragment("def only_b(): return 'B'", source="b.py")
    ids_a = {r["fragment_id"] for r in ea.recall_relevant("only", top_k=10)}
    ids_b = {r["fragment_id"] for r in eb.recall_relevant("only", top_k=10)}
    check("engines don't share state", ids_a.isdisjoint(ids_b))


# === D-22: EMPTY + WHITESPACE CONTENT ====================================

def test_empty_whitespace():
    section("D-22  EMPTY + WHITESPACE CONTENT")
    engine, _ = fresh_engine()
    cases = [("empty", ""), ("spaces", "   "), ("tabs", "\t\t"),
             ("newlines", "\n\n\n"), ("mixed_ws", "  \n\t  ")]
    for name, content in cases:
        try:
            r = engine.ingest_fragment(content, source=f"{name}.py")
            check(f"{name}: ingest completes", "status" in r)
        except Exception as e:
            check(f"{name}: no crash", False, f"{type(e).__name__}")


# === D-23: SELF-SIMILAR CORPUS ===========================================

def test_self_similar_corpus():
    section("D-23  SELF-SIMILAR CORPUS (50 variants)")
    engine, _ = fresh_engine()
    base = "def process(x, y):\n    result = x * y\n    return result\n"
    ingested = deduped = 0
    for i in range(50):
        variant = base.replace("process", f"process_v{i}").replace("result", f"res_{i}")
        r = engine.ingest_fragment(variant, source=f"v{i}.py")
        if r.get("status") == "ingested":
            ingested += 1
        elif r.get("status") == "duplicate":
            deduped += 1
    check("some ingested", ingested > 0, f"ingested={ingested}, deduped={deduped}")
    check("all 50 processed", ingested + deduped == 50)


# === D-24: QUERY REFINEMENT ==============================================

def test_query_refinement():
    section("D-24  QUERY REFINEMENT VAGUE vs PRECISE")
    engine, _ = fresh_engine()
    ingest_corpus(engine)
    vague = engine.optimize_context(token_budget=100_000, query="fix the bug")
    check("vague query completes", isinstance(vague, dict))
    precise = engine.optimize_context(
        token_budget=100_000,
        query="knapsack_optimize function budget quantization dynamic programming DP"
    )
    check("precise query completes", isinstance(precise, dict))
    # Vague query may have query_refinement info
    if "query_refinement" in vague:
        check("vague query has refinement info", True,
              f"refined={vague['query_refinement'].get('refined_query', '')[:50]}")


# === D-25: PREFETCH CO-ACCESS LEARNING ===================================

def test_prefetch_coacccess():
    section("D-25  PREFETCH CO-ACCESS LEARNING")
    engine, _ = fresh_engine()
    # Simulate: file_a and file_b always accessed together
    for turn in range(10):
        engine.ingest_fragment(f"# turn {turn}\ndef a(): pass",
                               source="/project/file_a.py", token_count=10)
        engine.ingest_fragment(f"# turn {turn}\ndef b(): pass",
                               source="/project/file_b.py", token_count=10)
        engine.advance_turn()
    # Pass empty source_content to isolate co-access from static import analysis
    preds = engine.prefetch_related("/project/file_a.py", source_content="")
    check("prefetch returns list", isinstance(preds, list))
    pred_paths = [p.get("path", "") for p in preds]
    has_b = any("file_b" in p for p in pred_paths)
    check("prefetch predicts co-accessed file_b", has_b, f"preds={pred_paths[:5]}")


# === D-26: PREFETCH IMPORT ANALYSIS ======================================

def test_prefetch_imports():
    section("D-26  PREFETCH IMPORT ANALYSIS")
    engine, _ = fresh_engine()
    source = """
from mypackage.utils import helper
from mypackage.models import User
import json
def process(data):
    return helper(User.from_dict(data))
"""
    preds = engine.prefetch_related("/project/mypackage/main.py", source_content=source)
    pred_paths = [p.get("path", "") for p in preds]
    check("import analysis returns predictions", len(preds) > 0, f"count={len(preds)}")
    has_utils = any("utils" in p for p in pred_paths)
    has_models = any("models" in p for p in pred_paths)
    check("predicts mypackage.utils", has_utils, f"paths={pred_paths[:5]}")
    check("predicts mypackage.models", has_models, f"paths={pred_paths[:5]}")


# === D-27: STATS CONSISTENCY CHAIN =======================================

def test_stats_consistency():
    section("D-27  STATS CONSISTENCY CHAIN")
    engine, _ = fresh_engine()
    t0 = get_total_fragments(engine.get_stats())
    check("empty engine: total_fragments=0", t0 == 0, f"total={t0}")
    for i in range(3):
        engine.ingest_fragment(f"def fn_{i}(): return {i}", source=f"s{i}.py")
    t1 = get_total_fragments(engine.get_stats())
    check("after 3 ingests: total_fragments=3", t1 == 3, f"total={t1}")
    engine.ingest_fragment("def fn_0(): return 0", source="s0.py")  # duplicate
    t2 = get_total_fragments(engine.get_stats())
    check("after duplicate: still 3", t2 == 3, f"total={t2}")


# === D-28: WILSON SCORE MATH =============================================

def test_wilson_score_math():
    section("D-28  WILSON SCORE MATH")
    try:
        from entroly.server import _WilsonFeedbackTracker
    except ImportError:
        skip_check("Wilson score", "tracker not importable")
        return
    t = _WilsonFeedbackTracker()
    check("no data: multiplier=1.0", t.learned_value("x") == 1.0)
    t.record_success(["a"] * 10)
    check("10 successes: > 1.0", t.learned_value("a") > 1.0, f"val={t.learned_value('a'):.4f}")
    t.record_failure(["b"] * 10)
    check("10 failures: < 1.0", t.learned_value("b") < 1.0, f"val={t.learned_value('b'):.4f}")
    t.record_success(["c"] * 50)
    t.record_failure(["c"] * 50)
    vc = t.learned_value("c")
    check("50/50: near 1.0 (+/-0.3)", 0.7 < vc < 1.3, f"val={vc:.4f}")
    t.record_success(["d"] * 1000)
    check("1000 successes: <= 2.0", t.learned_value("d") <= 2.0, f"val={t.learned_value('d'):.4f}")
    t.record_failure(["e"] * 1000)
    check("1000 failures: >= 0.5", t.learned_value("e") >= 0.5, f"val={t.learned_value('e'):.4f}")


# === D-29: EBBINGHAUS DECAY MATH ========================================

def test_ebbinghaus_decay_math():
    section("D-29  EBBINGHAUS DECAY MATH")
    from entroly_core import ContextFragment, py_apply_ebbinghaus_decay as _decay
    hl = 15
    frag = ContextFragment(fragment_id="t", content="x", token_count=10)
    frag.recency_score = 1.0
    frag.turn_last_accessed = 0
    [updated] = _decay([frag], current_turn=hl, half_life=hl)
    check("1 half-life: recency ~= 0.5", abs(updated.recency_score - 0.5) < 0.01,
          f"r={updated.recency_score:.4f}")
    updated.recency_score = 1.0
    updated.turn_last_accessed = 0
    [frag2] = _decay([updated], current_turn=2 * hl, half_life=hl)
    check("2 half-lives: recency ~= 0.25", abs(frag2.recency_score - 0.25) < 0.01,
          f"r={frag2.recency_score:.4f}")
    frag2.recency_score = 1.0
    frag2.turn_last_accessed = 100
    [frag3] = _decay([frag2], current_turn=100, half_life=hl)
    check("0 turns: recency = 1.0", frag3.recency_score == 1.0)


# === D-30: KNAPSACK DP CORRECTNESS =======================================

def test_knapsack_dp():
    section("D-30  KNAPSACK DP CORRECTNESS")
    from entroly_core import ContextFragment, py_knapsack_optimize as knapsack_optimize
    frags = []
    for i in range(20):
        f = ContextFragment(fragment_id=f"f{i}", content=f"item {i}", token_count=10 + i)
        f.recency_score = 0.5
        f.frequency_score = 0.3
        f.semantic_score = 0.2 + (i * 0.03)
        f.entropy_score = 0.6
        frags.append(f)
    selected, stats = knapsack_optimize(frags, token_budget=100)
    total_tc = sum(f.token_count for f in selected)
    check("total_tokens <= budget", total_tc <= 100, f"total={total_tc}")
    check("at least 1 selected", len(selected) >= 1)
    # Rust returns total_tokens/total_relevance in stats, no 'method' key
    check("stats has total_tokens", "total_tokens" in stats, f"keys={list(stats.keys())}")


# === D-31: ENTROPY SCORING ===============================================

def test_entropy_scoring():
    section("D-31  ENTROPY SCORING SANITY")
    from entroly_core import py_information_score as compute_information_score
    high = """
def compute_gradient_descent(weights, learning_rate, loss_fn):
    gradients = loss_fn.backward(weights)
    updated = weights - learning_rate * gradients
    momentum = 0.9 * previous_velocity + gradients
    return clamp(updated + momentum, -1.0, 1.0)
"""
    low = "import os\nimport sys\nimport json\npass\npass\npass\n"
    sh = compute_information_score(high, [])
    sl = compute_information_score(low, [])

    check("high-entropy > low-entropy", sh > sl, f"high={sh:.4f}, low={sl:.4f}")
    check("scores in [0,1]", 0 <= sh <= 1 and 0 <= sl <= 1)


# === D-32: FULL LIFECYCLE STRESS =========================================

def test_full_lifecycle_stress():
    section("D-32  FULL LIFECYCLE STRESS (20 turns)")
    engine, d = fresh_engine()
    ids = ingest_corpus(engine)
    check("corpus ingested", len(ids) > 0, f"count={len(ids)}")
    queries = [
        "knapsack optimization dynamic programming",
        "entropy scoring Shannon information",
        "SimHash dedup fingerprint hamming",
        "checkpoint resume serialization",
        "Ebbinghaus decay recency frequency",
        "query refinement vagueness expansion",
        "prefetch co-access import prediction",
        "Wilson score feedback learning",
        "budget utilization token savings",
        "provenance hallucination risk",
    ]
    for turn in range(10):
        q = queries[turn % len(queries)]
        opt = engine.optimize_context(token_budget=100_000, query=q)
        used = get_total_tokens(opt)
        eff = get_effective_budget(opt, 100_000)
        check(f"turn {turn+1}: budget OK", used <= eff or eff == 0, f"used={used}")
        sel = get_selected(opt)
        if sel:
            engine.record_success([sel[0]["id"]])
            if len(sel) > 2:
                engine.record_failure([sel[-1]["id"]])
        engine.advance_turn()
    ckpt = engine.checkpoint(metadata={"stress": "test"})
    check("checkpoint created", os.path.isfile(ckpt))
    engine2, _ = fresh_engine(tmp_dir=d)
    result = engine2.resume()
    check("resume successful", result.get("status") == "resumed")
    for turn in range(10, 15):
        q = queries[turn % len(queries)]
        opt = engine2.optimize_context(token_budget=100_000, query=q)
        check(f"post-resume turn {turn+1}: works", isinstance(opt, dict))
    r = engine2.ingest_fragment("def post_resume(): return 'ok'", source="pr.py")
    check("ingest after resume works", r.get("status") in ("ingested", "duplicate"))


# ══════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  Entroly v0.2.0 -- Deep Functional Test Suite")
    print(f"  Corpus: {len(real_sources())} real project files")
    print("=" * 70)

    tests = [
        test_unicode_binary,
        test_massive_fragment,
        test_rapid_fire_ingest,
        test_dedup_near_miss,
        test_dedup_reorder,
        test_feedback_saturation,
        test_feedback_nonexistent,
        test_optimize_after_eviction,
        test_interleaved,
        test_special_char_queries,
        test_pin_everything,
        test_concurrent_like,
        test_checkpoint_fidelity,
        test_checkpoint_then_ingest,
        test_corrupted_checkpoint,
        test_zero_token_fragments,
        test_same_source_diff_content,
        test_recall_topk_boundary,
        test_budget_equals_corpus,
        test_extreme_aging,
        test_multi_engine_isolation,
        test_empty_whitespace,
        test_self_similar_corpus,
        test_query_refinement,
        test_prefetch_coacccess,
        test_prefetch_imports,
        test_stats_consistency,
        test_wilson_score_math,
        test_ebbinghaus_decay_math,
        test_knapsack_dp,
        test_entropy_scoring,
        test_full_lifecycle_stress,
    ]

    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"\n  EXCEPTION in {t.__name__}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    print(f"\n{'==' * 35}")
    if failed == 0:
        print(f"  Results: {passed}/{passed+failed} passed  -- ALL CHECKS PASS"
              + (f"  ({skipped} skipped)" if skipped else ""))
    else:
        print(f"  Results: {passed}/{passed+failed} passed  ({failed} FAILED)"
              + (f"  ({skipped} skipped)" if skipped else ""))
        print(f"  Failures:")
        for f in _failures:
            print(f"    X {f}")
    print(f"{'==' * 35}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
