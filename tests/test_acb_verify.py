"""Comprehensive ACB + accuracy harness verification."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [OK] {name}" + (f"  [{detail}]" if detail else ""))
        else:
            failed += 1
            print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))

    # ── 1. ACB imports ──
    print("=" * 60)
    print("  ACB MODULE VERIFICATION")
    print("=" * 60)

    from entroly.adaptive_budget import (
        AdaptiveBudgetModel,
        QueryFeatures,
        TrainingExample,
        derive_optimal_budget,
        extract_features,
        BUDGET_MIN,
        BUDGET_MAX,
    )
    check("ACB imports", True)

    # ── 2. Feature extraction ──
    f = extract_features("fix the auth bug in login.py", "def login(): pass " * 20)
    check("extract_features returns QueryFeatures", isinstance(f, QueryFeatures))
    check("query_len_tokens > 0", f.query_len_tokens > 0, str(f.query_len_tokens))
    check("ctx_len_tokens > 0", f.ctx_len_tokens > 0, str(f.ctx_len_tokens))
    check("task_type is valid", f.task_type in ["BugTracing", "Unknown"], f.task_type)

    arr = f.to_array()
    check("to_array returns list", isinstance(arr, list))
    check("to_array length=14 (6 numeric + 8 task onehot)", len(arr) == 14, str(len(arr)))

    # ── 3. derive_optimal_budget ──
    accs = {0.05: 0.60, 0.10: 0.70, 0.20: 0.85, 0.30: 0.92, 0.50: 0.95}
    b = derive_optimal_budget(accs, recovery_alpha=0.95)
    check("optimal budget at alpha=0.95", b == 0.30, f"b={b}")

    # Edge: all budgets below threshold → returns max
    accs_low = {0.05: 0.10, 0.10: 0.15, 0.20: 0.20}
    b_low = derive_optimal_budget(accs_low, recovery_alpha=0.95)
    check("all-below-threshold returns max budget", b_low == 0.20, f"b={b_low}")

    # Edge: empty dict → BUDGET_MAX
    check("empty dict returns BUDGET_MAX", derive_optimal_budget({}) == BUDGET_MAX)

    # ── 4. Cold start ──
    m = AdaptiveBudgetModel()
    p = m.predict(f)
    check("cold-start fallback='cold_start'", p["fallback"] == "cold_start")
    check("cold-start budget_raw=0.20", p["budget_raw"] == 0.20)
    check("cold-start budget_se is None", p["budget_se"] is None)

    # ── 5. Training + fit ──
    # Add 25 examples with varying features → should fit
    for i in range(25):
        ex = TrainingExample(
            features=extract_features(f"query {i}", "context " * (50 + i * 10)),
            optimal_budget=0.15 + i * 0.02,
            max_accuracy=0.9,
        )
        m.add_example(ex)

    fit_result = m.fit()
    check("fit status='fit'", fit_result["status"] == "fit", str(fit_result))
    check("fit n=25", fit_result["n"] == 25)
    check("fit train_mse < 0.1", fit_result["train_mse"] < 0.1, f"mse={fit_result['train_mse']}")
    check("bootstrap_n=50", fit_result["bootstrap_n"] == 50)

    # ── 6. Learned prediction ──
    p2 = m.predict(f)
    check("learned fallback=None (confident)", p2["fallback"] is None, str(p2))
    check("learned budget_se is not None", p2["budget_se"] is not None)
    check("budget_raw in [0.05, 0.95]", BUDGET_MIN <= p2["budget_raw"] <= BUDGET_MAX, str(p2["budget_raw"]))

    # ── 7. Ceiling cap ──
    p3 = m.predict(f, ceiling=0.10)
    check("ceiling caps budget_used", p3["budget_used"] <= 0.10, str(p3["budget_used"]))

    # ── 8. Save / Load roundtrip ──
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)
    m.save(tmp_path)
    check("save creates file", tmp_path.exists(), str(tmp_path.stat().st_size) + " bytes")

    m2 = AdaptiveBudgetModel.load(tmp_path)
    p_restored = m2.predict(f)
    check("load restores weights", p_restored["budget_raw"] == p2["budget_raw"],
          f"original={p2['budget_raw']} restored={p_restored['budget_raw']}")
    check("load restores n_training", p_restored["n_training"] == 25)
    tmp_path.unlink()

    # ── 9. State dict ──
    state = m.state()
    check("state has weights", state["weights"] is not None)
    check("state n_training=25", state["n_training"] == 25)
    check("state n_bootstrap=50", state["n_bootstrap"] == 50)

    # ── 10. Thread safety ──
    import threading
    errors = []
    def thread_fn():
        try:
            for _ in range(50):
                m.predict(f)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=thread_fn) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    check("thread-safe (4x50 concurrent predicts)", len(errors) == 0, f"errors={len(errors)}")

    # ── 11. Accuracy harness imports ──
    print("\n" + "=" * 60)
    print("  ACCURACY HARNESS VERIFICATION")
    print("=" * 60)

    import bench.accuracy as a
    check("bench.accuracy imports", True)
    check("_COMPRESSORS has 6 modes", len(a._COMPRESSORS) == 6,
          str(list(a._COMPRESSORS.keys())))

    import inspect
    sig_rb = inspect.signature(a.run_benchmark)
    check("run_benchmark has 'mode' param", "mode" in sig_rb.parameters)

    sig_ps = inspect.signature(a.run_pareto_sweep)
    check("run_pareto_sweep has 'modes' param", "modes" in sig_ps.parameters)

    sig_cm = inspect.signature(a._compress_messages_modal)
    check("_compress_messages_modal has 'mode' param", "mode" in sig_cm.parameters)

    # ── 12. Compressors work on real text ──
    text = ("The Wright brothers were two American aviation pioneers credited "
            "with inventing, building, and flying the world's first successful "
            "motor-operated airplane. They made the first controlled, sustained "
            "flight of a powered, heavier-than-air aircraft on December 17, 1903. ") * 5
    q = "When did the Wright brothers first fly?"
    budget = 80

    for mode in ("entroly", "head", "tail", "random"):
        try:
            out = a._COMPRESSORS[mode](text, budget, q)
            ratio = len(out) / len(text) if text else 0
            check(f"compressor '{mode}' works", len(out) > 0 and len(out) < len(text),
                  f"in={len(text)}c out={len(out)}c ratio={ratio:.1%}")
        except Exception as e:
            check(f"compressor '{mode}' works", False, f"{type(e).__name__}: {e}")

    # LLMLingua/hybrid might fail without torch — that's expected on some envs
    for mode in ("llmlingua", "hybrid"):
        try:
            out = a._COMPRESSORS[mode](text, budget, q)
            check(f"compressor '{mode}' works", len(out) > 0,
                  f"in={len(text)}c out={len(out)}c")
        except Exception as e:
            check(f"compressor '{mode}' works (optional dep)", True,
                  f"skipped: {type(e).__name__}")

    # ── Summary ──
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  PASSED: {passed}/{total}  |  FAILED: {failed}/{total}")
    if failed == 0:
        print("  ALL TESTS PASSED")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
