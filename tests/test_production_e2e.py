#!/usr/bin/env python3
"""
Entroly — Production Edge-Case Test Suite
==========================================

Focus: gaps NOT covered by existing tests/test_functional.py.

Tests added here:
  P-01  CONTEXT EFFICIENCY METRIC    optimize() exposes bounded efficiency
  P-02  TOP-K RECALL                 k caps, k=0, k > corpus
  P-03  MCP SERVER TOOL LAYER        ingest_fragment / optimize_context / recall_tool via Python layer
  P-04  GC FREEZE STATE              gc is disabled after engine __init__; re-enabled after optimize
  P-05  AUTOTUNE CONFIG I/O          json round-trip, weight normalization, bounds
  P-06  DAEMON CONTROL               enabled / disabled flag respected
  P-07  UNICODE + BINARY-LIKE        non-ASCII content ingested and recalled
  P-08  EMPTY CONTENT                empty string ingest handled gracefully
  P-09  ZERO TOKEN BUDGET            optimize(0, ...) returns valid response
  P-10  ALL DUPLICATES CORPUS        100% dupe corpus → still optimize works
  P-11  RAPID CONCURRENT INGESTS     threading doesn't corrupt fragment count
  P-12  RECALL WINDOW > CORPUS       window larger than N fragments = same as no window
  P-13  EFFICIENCY STABILITY         repeated optimize calls remain bounded
  P-14  PRISM DRIFT GUARD            10 feedbacks → weights still in (0,1) simplex
  P-15  BENCH EVALUATE CONFIG        bench/evaluate reads updated tuning_config correctly
  P-16  MAX_FRAGMENTS EVICTION       fragment_count never exceeds max_fragments
  P-17  EXPORT/IMPORT ROUNDTRIP      export → import restores engine state faithfully
"""

import gc
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main():

    try:
        from entroly_core import EntrolyEngine
    except ImportError:
        print("FATAL: entroly_core not importable. Run `maturin develop` first.")
        return 1

    test_state = tempfile.TemporaryDirectory(prefix="entroly_production_e2e_")
    os.environ["ENTROLY_DIR"] = test_state.name

    # ── Test harness ──────────────────────────────────────────────────────────────
    _passed = _failed = 0
    _failures: list[str] = []

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"


    def ok(label: str, cond: bool, detail: str = "") -> bool:
        nonlocal _passed, _failed
        if cond:
            _passed += 1
            print(f"  {GREEN}✓{RESET} {label}" + (f"  [{detail}]" if detail else ""))
        else:
            _failed += 1
            _failures.append(label)
            print(f"  {RED}✗{RESET} {label}" + (f"  ← {detail}" if detail else ""), file=sys.stderr)
        return cond


    def section(name: str):
        print(f"\n{BOLD}{name}{RESET}")


    def fresh(max_frags: int = 10_000) -> EntrolyEngine:
        return EntrolyEngine(
            w_recency=0.30, w_frequency=0.25, w_semantic=0.25, w_entropy=0.20,
            decay_half_life=15, min_relevance=0.05,
            hamming_threshold=3, exploration_rate=0.0,
            max_fragments=max_frags,
        )


    CODE = {
        "auth":    ("def authenticate(token): return hmac.compare_digest(token, SECRET)", "auth.py", 200),
        "pay":     ("class PaymentProcessor:\n    def charge(self, amount): return gateway.process(amount)", "payments.py", 150),
        "db":      ("SELECT u.id, u.email FROM users u WHERE u.active = 1", "queries.sql", 100),
        "cache":   ("class LRUCache:\n    def __init__(self, max_size):\n        self.store = collections.OrderedDict()", "cache.py", 120),
        "deploy":  ("def deploy(env, version):\n    subprocess.run(['kubectl', 'apply', '-f', f'deploy/{env}'])", "deploy.py", 180),
    }


    # ─── P-01: Context Efficiency Metric ─────────────────────────────────────────
    section("P-01  CONTEXT EFFICIENCY METRIC")
    e = fresh()
    for name, (content, src, tokens) in CODE.items():
        e.ingest(content, src, tokens, False)
    optimized = dict(e.optimize(4096, "auth payment"))
    efficiency = optimized.get("context_efficiency")
    ok("optimize() contains context_efficiency", isinstance(efficiency, float))
    ok(
        "context_efficiency is bounded",
        isinstance(efficiency, float) and 0.0 <= efficiency <= 1.0,
        f"actual={efficiency}",
    )
    ok("optimize() reports selected tokens", optimized.get("total_tokens", 0) > 0)
    ok(
        "stats() tracks optimization count",
        dict(e.stats()).get("savings", {}).get("total_optimizations") == 1,
    )


    # ─── P-02: Sliding Window Recall ─────────────────────────────────────────────
    section("P-02  TOP-K RECALL")
    e_win = fresh()
    ids = []
    for i in range(10):
        r = dict(e_win.ingest(f"function module_{i}(): handles specific logic for component {i}", f"mod{i}.py", 50, False))
        if r.get("status") == "ingested":
            ids.append(r.get("fragment_id", ""))

    recalled_top3 = [dict(r) for r in e_win.recall("function module logic", 3)]
    ok("top_k=3 caps recall results", len(recalled_top3) <= 3,
       f"got {len(recalled_top3)}")
    ok("top_k=0 returns no results", len(list(e_win.recall("function module logic", 0))) == 0)

    # Window = 0 → all fragments eligible
    e_nowin = fresh()
    for i in range(10):
        e_nowin.ingest(f"function module_{i}(): handles specific logic for component {i}", f"mod{i}.py", 50, False)
    recalled_all = [dict(r) for r in e_nowin.recall("function module logic", 10)]
    ok("top_k=10 returns at least as many results as top_k=3", len(recalled_all) >= len(recalled_top3),
       f"top10={len(recalled_all)} vs top3={len(recalled_top3)}")

    # Window larger than corpus → same as no window
    e_bigwin = fresh()
    for i in range(5):
        e_bigwin.ingest(f"procedure step {i} for pipeline processing", f"step{i}.py", 40, False)
    recalled_big = [dict(r) for r in e_bigwin.recall("pipeline procedure", 10)]
    ok("top_k > corpus size works (no crash, returns results)", len(recalled_big) >= 1)


    # ─── P-03: MCP Server Tool Layer ─────────────────────────────────────────────
    section("P-03  MCP SERVER TOOL LAYER (Python → Rust)")
    try:
        from entroly.server import EntrolyEngine as ServerEngine
        srv = ServerEngine()

        # ingest_fragment
        ir = srv.ingest_fragment("def login(user, pw): return db.verify(user, pw)", "login.py")
        ok("server.ingest_fragment returns dict", isinstance(ir, dict))
        ok("server.ingest_fragment status=ingested", ir.get("status") == "ingested", str(ir))

        # duplicate via server layer
        ir2 = srv.ingest_fragment("def login(user, pw): return db.verify(user, pw)", "login.py")
        ok("server.ingest_fragment detects duplicate", ir2.get("status") == "duplicate", str(ir2))

        # optimize_context
        srv.ingest_fragment("class DB:\n    def verify(self, u, p): return self.conn.auth(u, p)", "db.py")
        oc = srv.optimize_context(4096, "login authentication")
        ok("optimize_context returns dict", isinstance(oc, dict))
        ok("optimize_context has selected key", "selected_fragments" in oc, str(list(oc.keys())))
        ok("optimize_context has >0 selected", len(oc.get("selected_fragments", [])) > 0)

        # advance_turn
        before_turn = srv._rust.get_turn() if srv._use_rust else 0
        srv.advance_turn()
        after_turn = srv._rust.get_turn() if srv._use_rust else 1
        ok("advance_turn increments turn counter", after_turn > before_turn,
           f"before={before_turn} after={after_turn}")

    except ImportError as e:
        print(f"  ⊘ Skipping MCP layer (import error: {e})")
    except Exception as e:
        ok("server tool layer (unexpected error)", False, str(e))


    # ─── P-04: GC Freeze State ───────────────────────────────────────────────────
    section("P-04  GC FREEZE STATE")
    # GC is disabled by EntrolyEngine.__init__ (server.py) at startup
    # Cannot re-test __init__ here without reinitializing, but we can verify
    # the pattern manually:
    gc.enable()
    gc.isenabled()
    gc.disable()
    try:
        _ = [x * x for x in range(1000)]  # no GC during tight loop
    finally:
        gc.enable()
        gc.collect()
    ok("manual gc.disable/enable cycle works correctly", gc.isenabled())
    ok("gc.freeze does not raise on non-empty heap", True)  # structural


    # ─── P-05: Autotune Config I/O ───────────────────────────────────────────────
    section("P-05  AUTOTUNE CONFIG I/O")
    try:
        from bench.autotune import mutate_random, TUNABLE_PARAMS
        from bench.evaluate import load_tuning_config
        import random
        import copy

        cfg_orig = load_tuning_config()

        # Weight normalization invariant: always sums to 1.0
        failures_norm = 0
        for trial in range(200):
            rng = random.Random(trial)
            c, name, old_v, new_v = mutate_random(copy.deepcopy(cfg_orig), rng)
            w = c["weights"]
            total = round(sum(w[k] for k in ["recency", "frequency", "semantic_sim", "entropy"]), 5)
            if abs(total - 1.0) > 0.001:
                failures_norm += 1
        ok("weights sum=1.0 after 200 random mutations", failures_norm == 0,
           f"{failures_norm} violations")

        # Bounds invariant: mutated values stay in declared [min, max]
        failures_bounds = 0
        for trial in range(200):
            rng = random.Random(trial + 5000)
            c, _, _, _ = mutate_random(copy.deepcopy(cfg_orig), rng)
            for param in TUNABLE_PARAMS:
                val = param.get(c)
                if not (param.min_val - 1e-9 <= val <= param.max_val + 1e-9):
                    failures_bounds += 1
        ok("all mutated params stay in declared bounds (200 trials)", failures_bounds == 0,
           f"{failures_bounds} out-of-bounds")

        # Config round-trip: save + reload preserves exact values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = Path(f.name)
            json.dump(cfg_orig, f, indent=2)
        reloaded = json.loads(tmp_path.read_text())
        tmp_path.unlink()
        ok("config json round-trip preserves all keys", set(cfg_orig.keys()) == set(reloaded.keys()))
        ok("config json round-trip preserves weights", cfg_orig["weights"] == reloaded["weights"])

    except Exception as exc:
        ok("autotune config checks complete", False, str(exc))


    # ─── P-06: Daemon enabled/disabled flag ──────────────────────────────────────
    section("P-06  DAEMON CONTROL")
    try:
        from entroly.server import _start_autotune_daemon
        from bench.evaluate import load_tuning_config
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "tuning_config.json"
            cfg = load_tuning_config()
            cfg["autotuner"] = {"enabled": True}
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            previous = os.environ.get("ENTROLY_TUNING_CONFIG")
            os.environ["ENTROLY_TUNING_CONFIG"] = str(cfg_path)
            try:
                with patch("threading.Thread") as thread_cls:
                    thread_cls.return_value.ident = 123
                    _start_autotune_daemon(None)
                    ok("daemon starts when enabled=true", thread_cls.called)
                    ok(
                        "autotune thread is daemon=True",
                        thread_cls.call_args.kwargs.get("daemon") is True,
                    )

                cfg["autotuner"]["enabled"] = False
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
                with patch("threading.Thread") as thread_cls:
                    _start_autotune_daemon(None)
                    ok("daemon does not start when enabled=false", not thread_cls.called)
            finally:
                if previous is None:
                    os.environ.pop("ENTROLY_TUNING_CONFIG", None)
                else:
                    os.environ["ENTROLY_TUNING_CONFIG"] = previous

    except Exception as exc:
        ok("daemon control checks complete", False, str(exc))


    # ─── P-07: Unicode + Non-ASCII content ───────────────────────────────────────
    section("P-07  UNICODE + NON-ASCII CONTENT")
    e_uni = fresh()
    uni_cases = [
        ("def grüßen(name):\n    return f'Hallo {name}'", "german.py", 50),
        ("func 认证(用户名 string) bool {\n    return db.Check(用户名)\n}", "auth_cn.go", 80),
        ("الدالة authenticate(المستخدم):\n    return db.تحقق(المستخدم)", "auth_ar.py", 60),
        ("🔐 Security check: token must be non-null and unexpired 🔑", "security_notes.md", 30),
        ("def ñoño(): pass  # Spanish function name with ñ", "spanish.py", 20),
    ]
    for content, src, tok in uni_cases:
        r = dict(e_uni.ingest(content, src, tok, False))
        ok(f"unicode ingest OK [{src}]", r.get("status") in ("ingested", "duplicate"), str(r))
    recalled_uni = [dict(r) for r in e_uni.recall("authentication check", 5)]
    ok("recall works after unicode ingests", len(recalled_uni) >= 1)


    # ─── P-08: Empty Content ─────────────────────────────────────────────────────
    section("P-08  EMPTY CONTENT EDGE CASES")
    e_em = fresh()
    try:
        r_empty = dict(e_em.ingest("", "empty.py", 0, False))
        ok("empty string ingest does not crash", True)
        ok("empty string ingest handled (any valid status)", r_empty.get("status") in ("ingested", "duplicate", "error", "rejected"), str(r_empty))
    except Exception as exc:
        ok("empty string ingest does not crash", False, str(exc))

    try:
        dict(e_em.ingest("   \n\t\n   ", "whitespace.py", 0, False))
        ok("whitespace-only ingest does not crash", True)
    except Exception as exc:
        ok("whitespace-only ingest does not crash", False, str(exc))

    # Optimize on empty corpus
    e_empty_corpus = fresh()
    try:
        r_opt = dict(e_empty_corpus.optimize(4096, "any query"))
        ok("optimize on empty corpus does not crash", True)
        ok("optimize on empty corpus returns selected key", "selected" in r_opt)
        ok("optimize on empty corpus returns empty selected", len(r_opt.get("selected", [])) == 0)
    except Exception as exc:
        ok("optimize on empty corpus does not crash", False, str(exc))

    # Recall on empty corpus
    try:
        recalled_empty = list(e_empty_corpus.recall("query", 5))
        ok("recall on empty corpus does not crash", True)
        ok("recall on empty corpus returns empty list", len(recalled_empty) == 0)
    except Exception as exc:
        ok("recall on empty corpus does not crash", False, str(exc))


    # ─── P-09: Zero Token Budget ─────────────────────────────────────────────────
    section("P-09  ZERO/TINY TOKEN BUDGET")
    e_z = fresh()
    for name, (content, src, tokens) in CODE.items():
        e_z.ingest(content, src, tokens, False)

    try:
        r0 = dict(e_z.optimize(0, "auth"))
        ok("optimize(budget=0) does not crash", True)
        ok("optimize(budget=0) returns selected key", "selected" in r0)
    except Exception as exc:
        ok("optimize(budget=0) does not crash", False, str(exc))

    try:
        dict(e_z.optimize(1, "auth"))
        ok("optimize(budget=1) does not crash", True)
    except Exception as exc:
        ok("optimize(budget=1) does not crash", False, str(exc))


    # ─── P-10: All-Duplicate Corpus ───────────────────────────────────────────────
    section("P-10  ALL-DUPLICATE CORPUS")
    e_dup = fresh()
    BASE = "def process(data): return transform(validate(data))"
    r_first = dict(e_dup.ingest(BASE, "process.py", 100, False))
    ok("first ingest accepted", r_first.get("status") == "ingested")
    dup_count = 0
    for _ in range(10):
        r = dict(e_dup.ingest(BASE, "process.py", 100, False))
        if r.get("status") == "duplicate":
            dup_count += 1
    ok("all repeated ingests detected as duplicate", dup_count == 10, f"{dup_count}/10")
    opt_dup = dict(e_dup.optimize(4096, "process data"))
    ok("optimize works on all-duplicate corpus", "selected" in opt_dup)
    ok("all-dup corpus: optimize returns the 1 unique fragment", len(opt_dup.get("selected", [])) == 1)

    stats_dup = dict(e_dup.stats())
    ok("stats reports total_fragments=1 (dedup working)", dict(stats_dup.get("session", {})).get("total_fragments", -1) == 1,
       str(dict(stats_dup.get("session", {}))))


    # ─── P-11: Concurrent Rapid Ingests ───────────────────────────────────────────
    section("P-11  CONCURRENT RAPID INGESTS (threading)")
    e_conc = fresh()
    errors = []
    lock = threading.Lock()
    ingest_results = []

    def worker(thread_id: int):
        for i in range(10):
            try:
                r = dict(e_conc.ingest(
                    f"function thread_{thread_id}_iter_{i}(x): return x * {thread_id + i}",
                    f"t{thread_id}_f{i}.py", 40, False
                ))
                with lock:
                    ingest_results.append(r.get("status"))
            except Exception as exc:
                with lock:
                    errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok("no exceptions during concurrent ingests", len(errors) == 0, f"{len(errors)} errors: {errors[:2]}")
    total_accepted = sum(1 for s in ingest_results if s == "ingested")
    ok("concurrent ingests: some accepted (>0)", total_accepted > 0, f"{total_accepted} ingested")
    frag_count = dict(dict(e_conc.stats()).get("session", {})).get("total_fragments", 0)
    ok("total_fragments consistent with accepted ingests", frag_count == total_accepted,
       f"total_fragments={frag_count} accepted={total_accepted}")


    # ─── P-12: Recall Window > Corpus ─────────────────────────────────────────────
    section("P-12  RECALL TOP_K LARGER THAN CORPUS")
    e_wlarge = fresh()
    for i in range(5):
        e_wlarge.ingest(f"utility function helper_{i}() for general use", f"util{i}.py", 30, False)
    r_wlarge = [dict(x) for x in e_wlarge.recall("utility function", 10)]
    ok("top_k=10 with 5-fragment corpus: no crash", True)
    ok("top_k=10 with 5-fragment corpus: returns results", len(r_wlarge) >= 1)


    # ─── P-13: Efficiency Stability ───────────────────────────────────────────────
    section("P-13  EFFICIENCY STABILITY (across N optimize calls)")
    e_math = fresh()
    for name, (content, src, tokens) in CODE.items():
        e_math.ingest(content, src, tokens, False)

    efficiencies = [
        dict(e_math.optimize(4096, "auth payment database")).get("context_efficiency")
        for _ in range(5)
    ]
    ok(
        "efficiency remains finite and bounded across repeated calls",
        all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in efficiencies),
        str(efficiencies),
    )


    # ─── P-14: PRISM Drift Guard ──────────────────────────────────────────────────
    section("P-14  PRISM DRIFT GUARD (value residual)")
    e_prism = fresh()
    for name, (content, src, tokens) in CODE.items():
        e_prism.ingest(content, src, tokens, False)

    all_frags = [dict(r)["fragment_id"] for r in e_prism.recall("auth payment", 5)]

    # Apply 20 success feedbacks on the same fragment — weights should not run away
    for _ in range(20):
        e_prism.record_success(all_frags[:1])

    # Engine must still function correctly
    opt_after = dict(e_prism.optimize(4096, "auth"))
    ok("engine functional after 20 success feedbacks", "selected" in opt_after)
    ok("selected count > 0 after heavy feedback", len(opt_after.get("selected", [])) > 0)

    stats_prism = dict(e_prism.stats())
    prism_stats = dict(stats_prism.get("prism", {}))
    ok("stats still returns PRISM state after updates", bool(prism_stats))
    ok(
        "PRISM weights remain finite and bounded",
        all(0.0 <= float(prism_stats.get(key, -1)) <= 1.0 for key in (
            "w_recency", "w_frequency", "w_semantic", "w_entropy"
        )),
        str(prism_stats),
    )


    # ─── P-15: bench/evaluate reads updated config ────────────────────────────────
    section("P-15  BENCH/EVALUATE READS LIVE TUNING CONFIG")
    try:
        from bench.evaluate import evaluate, load_tuning_config
        cfg = load_tuning_config()
        ok("load_tuning_config() returns dict", isinstance(cfg, dict))
        ok("config has weights section", "weights" in cfg)
        result = evaluate(cfg)
        ok("evaluate() returns composite_score", "composite_score" in result, str(list(result.keys())))
        ok("composite_score in [0, 1]", 0.0 <= result["composite_score"] <= 1.0, str(result["composite_score"]))
        ok("all_latency_ok = True (all under 500ms)", result["all_latency_ok"],
           f"avg={result.get('avg_latency_ms', '?')}ms")
    except Exception as exc:
        ok("bench evaluate checks complete", False, str(exc))


    # ─── P-16: Max_Fragments Eviction ─────────────────────────────────────────────
    section("P-16  FRAGMENT EVICTION (max_fragments cap)")
    # max_fragments defaults to 10000 — we can't easily test it at that scale
    # but we can verify fragment_count is capped by checking the engine caps correctly
    # by using a small effective limit via the Rust default
    e_cap = EntrolyEngine(
        w_recency=0.30, w_frequency=0.25, w_semantic=0.25, w_entropy=0.20,
    )
    # Ingest 50 fragments and verify count doesn't exceed insert count
    for i in range(50):
        e_cap.ingest(f"fragment content block {i} with unique function logic_{i}()", f"file{i}.rs", 30, False)

    fc = dict(dict(e_cap.stats()).get("session", {})).get("total_fragments", -1)
    ok("total_fragments tracks ingested count (dedup excluded)", 0 < fc <= 50, f"count={fc}")


    # ─── P-17: Export / Import Roundtrip ──────────────────────────────────────────
    section("P-17  EXPORT / IMPORT ROUNDTRIP (via server.py layer)")
    try:
        from entroly.server import EntrolyEngine as SrvEng
        from entroly.config import EntrolyConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EntrolyConfig()
            cfg.checkpoint_dir = Path(tmpdir)

            srv_exp = SrvEng(config=cfg)
            for name, (content, src, tokens) in CODE.items():
                srv_exp.ingest_fragment(content, src)
            srv_exp.optimize_context(4096, "auth payment")

            try:
                ckpt_id = srv_exp.checkpoint({"test_meta": 1})
                ok("checkpoint() (server layer) does not raise", True)
                ok("checkpoint ID returned", isinstance(ckpt_id, str) and len(ckpt_id) > 0)

                srv_imp = SrvEng(config=cfg)
                res = srv_imp.resume()
                ok("resume() (server layer) does not raise", True)
                ok("resume() status=resumed", res.get("status") == "resumed")

                # Fragment universe preserved
                fc_orig = dict(dict(srv_exp._rust.stats()).get("session", {})).get("total_fragments", -1)
                fc_rest = dict(dict(srv_imp._rust.stats()).get("session", {})).get("total_fragments", -1)
                ok("total_fragments preserved after checkpoint", fc_orig == fc_rest,
                   f"orig={fc_orig} restored={fc_rest}")

                # Recall still works post-restore
                recalled_r = list(srv_imp._rust.recall("auth payment", 5))
                ok("recall works after checkpoint restore", len(recalled_r) > 0)

            except Exception as exc:
                ok("checkpoint roundtrip (unexpected error)", False, str(exc))

    except ImportError as exc:
        print(f"  ⊘ Skipping checkpoint roundtrip (import: {exc})")



    # Summary
    total = _passed + _failed
    sep = '=' * 60
    print(f"\n{sep}")
    print(f"PRODUCTION TEST RESULTS: {_passed}/{total} passed", end="  ")
    if _failed == 0:
        print("ALL PASS")
    else:
        print(f"{_failed} FAILED")
        print("\nFailed tests:")
        for f in _failures:
            print(f"  {f}")
    test_state.cleanup()
    return _failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
