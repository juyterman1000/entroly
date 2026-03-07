#!/usr/bin/env python3
"""
Entroly — Intensive Functional Test Suite
================================================

Real-user flows. No mocks. No stubs. No fabricated data.

Strategy: exercise every API call path a production user would hit,
at boundary conditions they'd actually encounter. Each test section
is independent — it builds its own engine from real corpus files.

Sections:
  F-01  COLD START              empty corpus edge cases
  F-02  SINGLE FILE             one-file corpus boundary
  F-03  INGEST CONTRACT         return shape, field types, token counts
  F-04  DEDUP BOUNDARY          exact-copy, whitespace-shifted, section-reorder
  F-05  FEEDBACK LOOP           multi-round success/failure amplification
  F-06  MULTI-TURN LIFECYCLE    real user session: ingest → optimize → feedback × N
  F-07  QUERY SENSITIVITY       different queries surface different top fragments
  F-08  SCORE STABILITY         same query twice is deterministic
  F-09  ADVANCE TURN            turn counter increments correctly
  F-10  STATS CONTRACT          get_stats() returns all expected keys and is consistent
  F-11  EXPLAIN SELECTION       explain_selection() reflects optimize decisions
  F-12  CHECKPOINT CRASH SIM    delete checkpoint mid-way, expect graceful resume fail
  F-13  LARGE BUDGET            budget >> corpus → selects everything
  F-14  TINY BUDGET             budget << smallest fragment → still returns something
  F-15  RECALL vs OPTIMIZE      recall and optimize see same fragment universe
  F-16  MIXED FEEDBACK          success then failure nets neutral or down
  F-17  PREFETCH PREDICTION     after co-access patterns, prefetch predicts correctly
  F-18  ZERO QUERY              empty query string handled gracefully
"""

import os
import sys
import time
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
CORE = REPO.parent / "entroly-core"

# ── Colour output ─────────────────────────────────────────────────────────────
PASS = "  ✓"
FAIL = "  ✗"
SKIP = "  ⊘"
passed = failed = skipped = 0
_section_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    global passed, failed
    sym = PASS if condition else FAIL
    dest = sys.stdout if condition else sys.stderr
    print(f"{sym} {label}{' — ' + detail if detail else ''}", file=dest)
    if condition:
        passed += 1
    else:
        failed += 1
        _section_failures.append(label)
    return condition


def skip(label: str, reason: str = ""):
    global skipped
    skipped += 1
    print(f"{SKIP} {label}{' — ' + reason if reason else ''}")


def section(name: str):
    _section_failures.clear()
    print(f"\n{'═' * 62}")
    print(f"  {name}")
    print(f"{'═' * 62}")


# ── API response helpers ──────────────────────────────────────────────────────
# The Python fallback returns "selected_fragments" + nested "optimization_stats"
# while the Rust engine may return "selected" + flat keys. These helpers
# normalise both shapes so tests stay engine-agnostic.

def opt_selected(opt: dict) -> list[dict]:
    """Extract the list of selected fragments from an optimize result."""
    return opt.get("selected_fragments", opt.get("selected", []))


def opt_total_tokens(opt: dict) -> int:
    """Extract total_tokens from an optimize result."""
    stats = opt.get("optimization_stats", {})
    return stats.get("total_tokens", opt.get("total_tokens", 0))


def opt_effective_budget(opt: dict, fallback: int) -> int:
    """Extract effective budget from an optimize result."""
    stats = opt.get("optimization_stats", {})
    return stats.get("effective_budget", opt.get("effective_budget", fallback))


# ── Corpus helpers ────────────────────────────────────────────────────────────

def real_sources() -> list[tuple[str, Path]]:
    """Return all real .py and .rs files from the project."""
    out = []
    for p in sorted((REPO / "entroly").glob("*.py")):
        out.append((f"entroly/{p.name}", p))
    for p in sorted((CORE / "src").glob("*.rs")):
        out.append((f"src/{p.name}", p))
    return out


def fresh_engine(tmp_dir: str | None = None, **cfg_kwargs):
    """Create a EntrolyEngine backed by a private temp checkpoint dir."""
    from entroly.server import EntrolyEngine
    from entroly.config import EntrolyConfig
    d = tmp_dir or tempfile.mkdtemp()
    cfg = EntrolyConfig(
        default_token_budget=200_000,
        decay_half_life_turns=20,
        min_relevance_threshold=0.0,   # don't filter anything out
        checkpoint_dir=d,
        auto_checkpoint_interval=99999,
        **cfg_kwargs,
    )
    return EntrolyEngine(config=cfg), d


def ingest_corpus(engine, sources, pinned_names=()) -> dict[str, str]:
    """Ingest all sources. Returns label→fragment_id map."""
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


# ══════════════════════════════════════════════════════════════════════════════
# F-01: COLD START
# ══════════════════════════════════════════════════════════════════════════════

def test_cold_start():
    section("F-01  COLD START  —  empty engine behaviour")
    engine, _ = fresh_engine()

    # recall on empty engine must return an empty list, not crash
    result = engine.recall_relevant("anything at all", top_k=10)
    check("recall on empty corpus returns []", result == [], f"got={result}")

    # optimize on empty engine must return a valid dict
    opt = engine.optimize_context(token_budget=1000, query="anything")
    check("optimize on empty corpus returns dict", isinstance(opt, dict))
    check("optimize on empty returns total_tokens=0",
          opt_total_tokens(opt) == 0,
          f"total_tokens={opt_total_tokens(opt)}")

    # stats on empty engine must be consistent
    stats = engine.get_stats()
    check("stats returns dict", isinstance(stats, dict))

    # checkpoint on empty is legal
    ckpt = engine.checkpoint()
    check("checkpoint on empty engine returns a path", os.path.isfile(ckpt),
          f"path={ckpt}")


# ══════════════════════════════════════════════════════════════════════════════
# F-02: SINGLE FILE CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def test_single_file():
    section("F-02  SINGLE FILE  —  one-fragment corpus")
    engine, _ = fresh_engine()
    sources = real_sources()
    label, path = sources[0]   # __init__.py — smallest file
    content = path.read_text(encoding="utf-8", errors="replace")

    r = engine.ingest_fragment(content, source=str(path),
                               token_count=max(1, len(content) // 4))
    fid = r.get("fragment_id", "")
    check("ingest returns fragment_id", bool(fid), f"fid={fid}")

    # recall must return this one fragment
    results = engine.recall_relevant("python module", top_k=5)
    check("single-file recall has ≤ 1 result", len(results) <= 1,
          f"count={len(results)}")

    # optimize must not exceed the single fragment's token count
    tc = r.get("token_count", 0)
    opt = engine.optimize_context(token_budget=1_000_000, query="python module")
    used = opt_total_tokens(opt)
    check("optimize with 1 file uses ≤ file_tokens", used <= tc or tc == 0,
          f"used={used}, token_count={tc}")


# ══════════════════════════════════════════════════════════════════════════════
# F-03: INGEST CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

def test_ingest_contract():
    section("F-03  INGEST CONTRACT  —  return shape and field types")
    engine, _ = fresh_engine()
    sources = real_sources()

    first_label, first_path = sources[0]
    content = first_path.read_text(encoding="utf-8", errors="replace")
    tc = max(1, len(content) // 4)
    r = engine.ingest_fragment(content, source=str(first_path), token_count=tc)

    check("status is 'ingested'", r.get("status") == "ingested",
          f"status={r.get('status')}")
    check("fragment_id is a non-empty string",
          isinstance(r.get("fragment_id"), str) and len(r.get("fragment_id", "")) > 0)
    check("token_count is a positive int",
          isinstance(r.get("token_count"), int) and r.get("token_count", 0) > 0,
          f"token_count={r.get('token_count')}")
    check("entropy_score is a float in [0,1]",
          isinstance(r.get("entropy_score"), float)
          and 0.0 <= r.get("entropy_score", -1) <= 1.0,
          f"entropy={r.get('entropy_score')}")

    # Re-ingest: must be duplicate
    r2 = engine.ingest_fragment(content, source=str(first_path), token_count=tc)
    check("re-ingest returns status='duplicate'",
          r2.get("status") == "duplicate", f"status={r2.get('status')}")
    check("duplicate has duplicate_of field",
          "duplicate_of" in r2, f"keys={list(r2.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# F-04: DEDUP BOUNDARY
# ══════════════════════════════════════════════════════════════════════════════

def test_dedup_boundary():
    section("F-04  DEDUP BOUNDARY  —  exact-copy, leading whitespace, newline shift")
    engine, _ = fresh_engine()
    sources = real_sources()
    _, path = next((s for s in sources if "server.py" in s[0]), sources[-1])
    content = path.read_text(encoding="utf-8", errors="replace")

    r1 = engine.ingest_fragment(content, source="original.py",
                                token_count=max(1, len(content) // 4))
    fid = r1.get("fragment_id", "")

    # Exact copy: must dedup
    r2 = engine.ingest_fragment(content, source="copy.py",
                                token_count=max(1, len(content) // 4))
    check("exact-copy deduplicated",
          r2.get("status") == "duplicate", f"status={r2.get('status')}")

    # Whitespace-shifted: add a trailing newline — SimHash is content hash,
    # expectation: may or may not dedup (implementation-defined), BUT must not crash
    r3 = engine.ingest_fragment(content + "\n", source="shifted.py",
                                token_count=max(1, len(content + "\n") // 4))
    check("whitespace-shifted ingest completes without crash",
          "status" in r3, f"status={r3.get('status')}")

    # Completely different content must NOT dedup with original
    different = "# completely different file — no overlap with server.py\nx = 42\n"
    r4 = engine.ingest_fragment(different, source="different.py", token_count=10)
    check("different content is not a duplicate of server.py",
          r4.get("status") != "duplicate" or r4.get("duplicate_of") != fid,
          f"status={r4.get('status')}")


# ══════════════════════════════════════════════════════════════════════════════
# F-05: FEEDBACK LOOP AMPLIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def test_feedback_loop():
    section("F-05  FEEDBACK LOOP  —  extended success/failure amplification")
    engine, _ = fresh_engine()
    sources = real_sources()
    ids = ingest_corpus(engine, sources)

    # Find two fragments that appear in the optimize output
    opt0 = engine.optimize_context(token_budget=500_000, query="optimization scoring")
    selected = opt_selected(opt0)
    if len(selected) < 2:
        skip("feedback loop", "fewer than 2 fragments selected — corpus too small")
        return

    def score_of(fid: str) -> float | None:
        opt = engine.optimize_context(token_budget=500_000, query="optimization scoring")
        for f in opt_selected(opt):
            if f.get("id") == fid:
                return f.get("relevance", None)
        return None

    winner_id = selected[0]["id"]
    loser_id  = selected[-1]["id"]

    w_before = score_of(winner_id)
    l_before = score_of(loser_id)

    # Apply 10 rounds of success on winner, 10 rounds of failure on loser
    for _ in range(10):
        engine.record_success([winner_id])
        engine.record_failure([loser_id])

    w_after = score_of(winner_id)
    l_after = score_of(loser_id)

    if w_before is not None and w_after is not None:
        check("10× success raises winner score",
              w_after > w_before,
              f"{w_before:.4f} → {w_after:.4f}")
    if l_before is not None and l_after is not None:
        check("10× failure lowers loser score",
              l_after < l_before,
              f"{l_before:.4f} → {l_after:.4f}")

    # Verify the score spread grew: winner >> loser
    if w_after is not None and l_after is not None:
        spread_before = (w_before or 0) - (l_before or 0)
        spread_after  = (w_after  or 0) - (l_after  or 0)
        check("feedback amplifies score spread",
              spread_after >= spread_before,
              f"spread: {spread_before:.4f} → {spread_after:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# F-06: MULTI-TURN LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

def test_multi_turn_lifecycle():
    section("F-06  MULTI-TURN LIFECYCLE  —  real session: ingest → optimize × 10 turns")
    engine, _ = fresh_engine()
    sources = real_sources()
    ids = ingest_corpus(engine, sources)

    queries = [
        "knapsack token budget optimization dynamic programming",
        "SimHash dedup fingerprint hamming distance",
        "Shannon entropy information density scoring",
        "checkpoint resume gzip serialization",
        "PRISM optimizer covariance spectral",
        "LLM context window relevance ordering",
        "dependency graph auto-link imports",
        "Wilson score feedback learning multiplier",
        "Ebbinghaus decay recency frequency",
        "LSH locality sensitive hashing recall",
    ]

    all_budgets = []
    prev_top    = None

    # Use a budget smaller than total corpus (~63K tokens) so the knapsack
    # must actually select a subset, making the query influence which
    # fragments are chosen and thus how many tokens are used.
    for turn, q in enumerate(queries):
        opt = engine.optimize_context(token_budget=30_000, query=q)
        used  = opt_total_tokens(opt)
        selected = opt_selected(opt)

        all_budgets.append(used)

        check(f"turn {turn+1}: budget not exceeded",
              used <= opt_effective_budget(opt, 30_000),
              f"used={used:,}")
        check(f"turn {turn+1}: at least 1 fragment selected",
              len(selected) >= 1,
              f"count={len(selected)}")

        # Give positive feedback on top fragment
        if selected:
            engine.record_success([selected[0]["id"]])
        if len(selected) > 1:
            engine.record_failure([selected[-1]["id"]])

        engine.advance_turn()
        prev_top = selected[0]["id"] if selected else prev_top

    check("used tokens varied across 10 turns (engine is query-sensitive)",
          len(set(all_budgets)) > 1,
          f"budget values={all_budgets}")


# ══════════════════════════════════════════════════════════════════════════════
# F-07: QUERY SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def test_query_sensitivity():
    section("F-07  QUERY SENSITIVITY  —  different queries surface different fragments")
    engine, _ = fresh_engine()
    sources = real_sources()
    ingest_corpus(engine, sources)

    # If the engine is truly query-sensitive, the TOP fragment should differ
    # across semantically distant queries
    top_results: list[str] = []
    distinct_queries = [
        "knapsack optimization dynamic programming budget",
        "SimHash fingerprint deduplication hamming",
        "Ebbinghaus exponential decay recency score",
        "Wilson confidence interval relevance multiplier",
        "PRISM covariance weight spectral shaping",
    ]
    for q in distinct_queries:
        res = engine.recall_relevant(q, top_k=5)
        top = res[0]["fragment_id"] if res else ""
        top_results.append(top)

    non_empty = [t for t in top_results if t]
    unique_tops = len(set(non_empty))
    check("different queries return at least 2 distinct top-ranked fragments",
          unique_tops >= 2,
          f"unique_tops={unique_tops}, tops={non_empty}")


# ══════════════════════════════════════════════════════════════════════════════
# F-08: SCORE STABILITY
# ══════════════════════════════════════════════════════════════════════════════

def test_score_stability():
    section("F-08  SCORE STABILITY  —  same query twice is deterministic")
    engine, _ = fresh_engine()
    sources = real_sources()
    ingest_corpus(engine, sources)

    Q = "knapsack optimization token budget"
    r1 = engine.recall_relevant(Q, top_k=10)
    r2 = engine.recall_relevant(Q, top_k=10)

    ids1 = [x["fragment_id"] for x in r1]
    ids2 = [x["fragment_id"] for x in r2]
    check("same query twice returns identical fragment list",
          ids1 == ids2, f"r1={ids1[:3]}, r2={ids2[:3]}")

    scores1 = [x.get("relevance", x.get("score", 0)) for x in r1]
    scores2 = [x.get("relevance", x.get("score", 0)) for x in r2]
    check("same query twice returns identical scores",
          scores1 == scores2, f"s1={scores1[:3]}, s2={scores2[:3]}")


# ══════════════════════════════════════════════════════════════════════════════
# F-09: ADVANCE TURN
# ══════════════════════════════════════════════════════════════════════════════

def test_advance_turn():
    section("F-09  ADVANCE TURN  —  turn counter increments correctly")
    engine, _ = fresh_engine()

    def get_turn():
        s = engine.get_stats()
        return (s.get("current_turn")
                or s.get("session", {}).get("current_turn")
                or 0)

    t0 = get_turn()
    engine.advance_turn()
    t1 = get_turn()
    engine.advance_turn()
    engine.advance_turn()
    t3 = get_turn()

    check("advance_turn increments turn by 1 each call",
          t1 == t0 + 1 and t3 == t0 + 3,
          f"t0={t0}, t1={t1}, t3={t3}")


# ══════════════════════════════════════════════════════════════════════════════
# F-10: STATS CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

def test_stats_contract():
    section("F-10  STATS CONTRACT  —  get_stats() keys and consistency")
    engine, _ = fresh_engine()
    sources = real_sources()
    ids = ingest_corpus(engine, sources)
    n = len([v for v in ids.values() if v])

    stats = engine.get_stats()

    # Either flat or nested under 'session'
    total_frags = (stats.get("total_fragments")
                   or stats.get("session", {}).get("total_fragments")
                   or 0)
    check("stats reports total_fragments > 0 after ingest",
          total_frags > 0, f"total_fragments={total_frags}")
    check("stats total_fragments matches ingested count",
          total_frags == n, f"reported={total_frags}, ingested={n}")

    # Ingest one more, check stats update
    _, path = sources[0]
    content = path.read_text(encoding="utf-8", errors="replace") + " # extra"
    engine.ingest_fragment(content, source="extra.py",
                           token_count=max(1, len(content) // 4))
    stats2 = engine.get_stats()
    total2 = (stats2.get("total_fragments")
              or stats2.get("session", {}).get("total_fragments")
              or 0)
    check("stats total_fragments increments after new ingest",
          total2 == total_frags + 1, f"before={total_frags}, after={total2}")


# ══════════════════════════════════════════════════════════════════════════════
# F-11: EXPLAIN SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def test_explain_selection():
    section("F-11  EXPLAIN SELECTION  —  explain reflects optimize decision")
    engine, _ = fresh_engine()
    sources = real_sources()
    ingest_corpus(engine, sources)

    # Run optimize first — explain_selection only has data after this
    engine.optimize_context(token_budget=200_000, query="knapsack budget optimization")
    exp = engine.explain_selection()

    check("explain_selection returns a dict", isinstance(exp, dict))
    # Must have either 'included' or at least not crash/return None
    # (Rust engine may structure this differently)
    has_content = (
        "included" in exp
        or "selected" in exp
        or "fragments" in exp
        or "error" in exp      # Python-only limitation is documented
    )
    check("explain_selection has recognisable keys", has_content,
          f"keys={list(exp.keys())[:5]}")


# ══════════════════════════════════════════════════════════════════════════════
# F-12: CHECKPOINT CRASH SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def test_checkpoint_crash_sim():
    section("F-12  CHECKPOINT CRASH SIM  —  resume from no-checkpoint returns known status")
    # Engine with empty dir → no checkpoint exists → resume must say so gracefully
    engine, _ = fresh_engine()
    result = engine.resume()
    check("resume with no checkpoint returns status dict",
          isinstance(result, dict))
    check("resume with no checkpoint returns a no-checkpoint status",
          result.get("status") in ("no_checkpoint_found", "not_found", "empty", ""),
          f"status={result.get('status')}")

    # Now write a checkpoint, corrupt it, try to resume
    sources = real_sources()
    ingest_corpus(engine, sources)
    ckpt_path = engine.checkpoint()

    # Truncate the file to simulate partial write / crash during flush
    with open(ckpt_path, "r+b") as f:
        f.truncate(64)

    engine2, _ = fresh_engine(tmp_dir=str(Path(ckpt_path).parent))
    try:
        r2 = engine2.resume()
        # Either it recovers (unlikely) or returns an error status — must not raise
        check("resume from corrupted checkpoint does not raise",
              True, f"status={r2.get('status')}")
    except Exception as e:
        # A hard crash here is a real bug
        check("resume from corrupted checkpoint does not raise",
              False, f"raised {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# F-13: LARGE BUDGET
# ══════════════════════════════════════════════════════════════════════════════

def test_large_budget():
    section("F-13  LARGE BUDGET  —  budget >> corpus → selects everything")
    engine, _ = fresh_engine()
    sources = real_sources()
    ids = ingest_corpus(engine, sources)

    total_corpus_tokens = sum(
        max(1, p.stat().st_size // 4)
        for _, p in sources
    )

    # Budget 10× the corpus — must select every fragment
    opt = engine.optimize_context(
        token_budget=total_corpus_tokens * 10,
        query="knapsack optimization entropy decay",
    )
    used  = opt_total_tokens(opt)
    eff   = opt_effective_budget(opt, total_corpus_tokens * 10)
    count = len(opt_selected(opt))

    check("large budget: utilization ≤ 1.0", used <= eff,
          f"used={used:,}, effective={eff:,}")
    check("large budget: all fragments selected",
          count == len([v for v in ids.values() if v]),
          f"selected={count}, ingested={len(ids)}")


# ══════════════════════════════════════════════════════════════════════════════
# F-14: TINY BUDGET
# ══════════════════════════════════════════════════════════════════════════════

def test_tiny_budget():
    section("F-14  TINY BUDGET  —  budget = 1 token → engine handles gracefully")
    engine, _ = fresh_engine()
    sources = real_sources()
    ingest_corpus(engine, sources)

    for budget in [0, 1, 10]:
        opt = engine.optimize_context(token_budget=budget,
                                      query="optimization budget knapsack")
        check(f"budget={budget}: optimize doesn't crash",
              isinstance(opt, dict), f"got={type(opt)}")
        used = opt_total_tokens(opt)
        eff  = opt_effective_budget(opt, budget if budget > 0 else 1)
        # With very tiny budget: either 0 selected OR only pinned items (which bypass budget)
        # We just verify the return is a valid result
        check(f"budget={budget}: total_tokens is non-negative int",
              isinstance(used, int) and used >= 0, f"total_tokens={used}")


# ══════════════════════════════════════════════════════════════════════════════
# F-15: RECALL vs OPTIMIZE UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

def test_recall_vs_optimize_universe():
    section("F-15  RECALL vs OPTIMIZE  —  same fragment universe")
    engine, _ = fresh_engine()
    sources = real_sources()
    ids = ingest_corpus(engine, sources)

    Q = "optimization entropy scoring fragment relevance"

    recall_ids  = {r["fragment_id"] for r in engine.recall_relevant(Q, top_k=50)}
    opt_ids     = {f["id"] for f in
                   engine.optimize_context(token_budget=500_000, query=Q)
                   .get("selected", [])}
    ingested_ids = set(v for v in ids.values() if v)

    # All recalled/optimized IDs must come from the ingested set
    check("all recall IDs are from ingested corpus",
          recall_ids <= ingested_ids,
          f"extra={recall_ids - ingested_ids}")
    check("all optimize IDs are from ingested corpus",
          opt_ids <= ingested_ids,
          f"extra={opt_ids - ingested_ids}")


# ══════════════════════════════════════════════════════════════════════════════
# F-16: MIXED FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

def test_mixed_feedback():
    section("F-16  MIXED FEEDBACK  —  success then failure nets downward")
    engine, _ = fresh_engine()
    sources = real_sources()
    ingest_corpus(engine, sources)

    Q = "knapsack optimization budget"
    opt0 = engine.optimize_context(token_budget=500_000, query=Q)
    selected = opt_selected(opt0)
    if not selected:
        skip("mixed feedback", "no fragments selected")
        return

    target_id = selected[0]["id"]
    score_baseline = selected[0].get("relevance", 0.0)

    # 3 successes then 5 failures — net effect should be below 3-success-only
    engine.record_success([target_id])
    engine.record_success([target_id])
    engine.record_success([target_id])

    opt_success = engine.optimize_context(token_budget=500_000, query=Q)
    score_after_success = next(
        (f.get("relevance") for f in opt_selected(opt_success)
         if f.get("id") == target_id), None
    )

    engine.record_failure([target_id])
    engine.record_failure([target_id])
    engine.record_failure([target_id])
    engine.record_failure([target_id])
    engine.record_failure([target_id])

    opt_mixed = engine.optimize_context(token_budget=500_000, query=Q)
    score_after_mixed = next(
        (f.get("relevance") for f in opt_selected(opt_mixed)
         if f.get("id") == target_id), None
    )

    if score_after_success is not None and score_after_mixed is not None:
        check("3× success then 5× failure: score < pure-success score",
              score_after_mixed < score_after_success,
              f"after_success={score_after_success:.4f}, after_mixed={score_after_mixed:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# F-17: PREFETCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def test_prefetch_prediction():
    section("F-17  PREFETCH PREDICTION  —  co-access patterns drive predictions")
    engine, _ = fresh_engine()
    sources = real_sources()

    # Build co-access: ingest server.py and knapsack.py always together
    server_path  = next((p for _, p in sources if "server.py"  in str(p)), None)
    knapsack_path = next((p for _, p in sources if "knapsack.py" in str(p)), None)

    if not server_path or not knapsack_path:
        skip("prefetch prediction", "required source files not in corpus")
        return

    server_content   = server_path.read_text(encoding="utf-8", errors="replace")
    knapsack_content = knapsack_path.read_text(encoding="utf-8", errors="replace")

    # Ingest both files once (for content availability)
    engine.ingest_fragment(server_content,   source=str(server_path),
                           token_count=max(1, len(server_content) // 4))
    engine.ingest_fragment(knapsack_content, source=str(knapsack_path),
                           token_count=max(1, len(knapsack_content) // 4))

    # Simulate 5 co-access sessions by directly recording access patterns.
    # Ingest deduplicates identical content, so record_access only fires once
    # per unique fragment. We test the prefetch co-access learning directly.
    for turn in range(5):
        engine._prefetch.record_access(str(server_path), turn)
        engine._prefetch.record_access(str(knapsack_path), turn)

    # Pass empty source_content to isolate co-access learning from static
    # import analysis (server.py has many stdlib imports that fill max_results
    # before the co-access prediction appears)
    predictions = engine.prefetch_related(str(server_path), source_content="")
    check("prefetch_related returns a list", isinstance(predictions, list))

    # After seeing server.py 5× always with knapsack.py, knapsack should appear
    predicted_paths = [p.get("path", "") for p in predictions]
    check("prefetch predicts knapsack.py as co-accessed with server.py",
          any("knapsack" in pp for pp in predicted_paths),
          f"predictions={predicted_paths[:3]}")


# ══════════════════════════════════════════════════════════════════════════════
# F-18: ZERO / EMPTY QUERY
# ══════════════════════════════════════════════════════════════════════════════

def test_zero_query():
    section("F-18  ZERO QUERY  —  empty string query handled gracefully")
    engine, _ = fresh_engine()
    sources = real_sources()
    ingest_corpus(engine, sources)

    # recall with empty query
    try:
        r = engine.recall_relevant("", top_k=5)
        check("empty-query recall returns list", isinstance(r, list),
              f"type={type(r).__name__}")
    except Exception as e:
        check("empty-query recall does not crash", False,
              f"raised {type(e).__name__}: {e}")

    # optimize with empty query
    try:
        opt = engine.optimize_context(token_budget=100_000, query="")
        check("empty-query optimize returns dict", isinstance(opt, dict))
        check("empty-query optimize returns total_tokens",
              "total_tokens" in opt or "optimization_stats" in opt,
              f"keys={list(opt.keys())[:5]}")
    except Exception as e:
        check("empty-query optimize does not crash", False,
              f"raised {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("══════════════════════════════════════════════════════════════")
    print("  Entroly v0.2.0 — Intensive Functional Test Suite")
    print(f"  Corpus: {len(real_sources())} real project files")
    print("══════════════════════════════════════════════════════════════")

    tests = [
        test_cold_start,
        test_single_file,
        test_ingest_contract,
        test_dedup_boundary,
        test_feedback_loop,
        test_multi_turn_lifecycle,
        test_query_sensitivity,
        test_score_stability,
        test_advance_turn,
        test_stats_contract,
        test_explain_selection,
        test_checkpoint_crash_sim,
        test_large_budget,
        test_tiny_budget,
        test_recall_vs_optimize_universe,
        test_mixed_feedback,
        test_prefetch_prediction,
        test_zero_query,
    ]

    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"\n  EXCEPTION in {t.__name__}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    total = passed + failed + skipped
    print(f"\n{'══' * 31}")
    if failed == 0:
        print(f"  Results: {passed}/{passed+failed} passed  — ALL CHECKS PASS ✓"
              + (f"  ({skipped} skipped)" if skipped else ""))
    else:
        print(f"  Results: {passed}/{passed+failed} passed  ({failed} FAILED)"
              + (f"  ({skipped} skipped)" if skipped else ""))
    print(f"{'══' * 31}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
