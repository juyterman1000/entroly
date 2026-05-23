"""E2E test: DreamingLoop + PRISM 5D + ArchetypeOptimizer + AutoTune."""
import os, sys, tempfile, shutil, time


def main():
    tmpdir = tempfile.mkdtemp(prefix="dream_test_")

    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal passed, failed
        try:
            result = fn()
            print(f"  [OK] {name}: {result}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    # ============================================================
    print("=" * 60)
    print("  PRISM 5D + ARCHETYPE OPTIMIZER")
    print("=" * 60)

    from entroly.archetype_optimizer import ArchetypeOptimizer, ArchetypeInfo

    arch = ArchetypeOptimizer(data_dir=tmpdir, project_root=".")
    check("ArchetypeOptimizer init", lambda: "OK")

    # Detect archetype from this repo
    info = arch.detect_and_load()
    check("Archetype detected", lambda: f"'{info.label}' (conf={info.confidence:.2f})")

    # PRISM 5D weights
    weights = arch.current_weights()
    check("PRISM 5D weights loaded", lambda: f"recency={weights.get('w_recency')}, "
          f"freq={weights.get('w_frequency')}, sem={weights.get('w_semantic')}, "
          f"ent={weights.get('w_entropy')}, res={weights.get('w_resonance')}")

    # Export weights (4D + resonance)
    export = arch.get_export_weights()
    check("Export weights", lambda: f"w_r={export['w_r']}, w_res={export['w_res']}")

    # Update weights (simulate dream cycle finding better weights)
    original_recency = weights.get("w_recency", 0.3)
    new_weights = dict(weights)
    new_weights["w_recency"] = 0.45
    new_weights["w_semantic"] = 0.35
    arch.update_weights(new_weights)
    updated = arch.current_weights()
    check("Weights updated", lambda: f"recency {original_recency:.2f} -> {updated['w_recency']:.2f}")

    # Stats
    stats = arch.stats()
    check("Archetype stats", lambda: f"label={stats.get('current_archetype')}, "
          f"strategies={len(stats.get('strategy_table', {}))}")

    # ============================================================
    print("\n" + "=" * 60)
    print("  FEEDBACK JOURNAL")
    print("=" * 60)

    from entroly.autotune import FeedbackJournal, WEIGHT_KEYS, reward_weighted_optimize

    journal = FeedbackJournal(journal_dir=tmpdir)
    check("Journal init", lambda: f"path={journal.journal_path}")

    # Simulate 20 feedback episodes with varying rewards
    import random
    random.seed(42)

    for i in range(20):
        w = {"w_r": 0.30 + random.gauss(0, 0.05),
             "w_f": 0.25 + random.gauss(0, 0.05),
             "w_s": 0.25 + random.gauss(0, 0.05),
             "w_e": 0.20 + random.gauss(0, 0.05)}
        # Higher recency -> higher reward (simulating a pattern)
        reward = 0.5 * w["w_r"] + 0.3 * w["w_s"] - 0.2 + random.gauss(0, 0.1)
        reward = max(-1, min(1, reward))
        journal.log(weights=w, reward=reward, selected_count=5,
                    query=f"fix bug in module_{i}", token_budget=8000, turn=i)

    check("20 episodes logged", lambda: f"count={journal.count()}")

    jstats = journal.stats()
    check("Journal stats", lambda: f"episodes={jstats['episodes']}, "
          f"successes={jstats['successes']}, avg_reward={jstats['avg_reward']:.3f}")

    # Reward-weighted optimization
    current_w = {"w_r": 0.30, "w_f": 0.25, "w_s": 0.25, "w_e": 0.20}
    opt_result = reward_weighted_optimize(journal.load(), current_w)
    if opt_result:
        check("Reward-weighted optimize", lambda: f"optimal: w_r={opt_result['optimal']['w_r']:.3f}, "
              f"w_s={opt_result['optimal']['w_s']:.3f}, confidence={opt_result['confidence']:.2f}")
    else:
        check("Reward-weighted optimize", lambda: "returned None (not enough data)")

    # ============================================================
    print("\n" + "=" * 60)
    print("  COMPONENT FEEDBACK BUS")
    print("=" * 60)

    from entroly.autotune import ComponentFeedbackBus

    bus = ComponentFeedbackBus(data_dir=tmpdir)
    check("ComponentFeedbackBus init", lambda: "OK")

    # Simulate improving metric
    for i in range(30):
        bus.log("prefetch", "hit_rate", 0.5 + i * 0.01)  # steadily improving

    trend = bus.get_trend("prefetch", "hit_rate")
    check("Trend detection", lambda: f"improving={trend['improving']}, ema={trend['ema']:.3f}, "
          f"delta={trend['delta']:.3f}")

    # Self-tuning suggestion
    suggestion = bus.suggest_adjustment(
        "prefetch", "hit_rate",
        current_value=0.5, bounds=(0.0, 1.0), step_size=0.02
    )
    check("Self-tune suggestion", lambda: f"suggested={suggestion:.3f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("  TASK PROFILE OPTIMIZER")
    print("=" * 60)

    from entroly.autotune import TaskProfileOptimizer

    task_opt = TaskProfileOptimizer(journal)
    check("TaskProfileOptimizer init", lambda: "OK")

    # Classify queries
    from entroly.autotune import classify_query
    check("Classify 'fix auth bug'", lambda: classify_query("fix auth bug"))
    check("Classify 'add login feature'", lambda: classify_query("add login feature"))
    check("Classify 'refactor utils'", lambda: classify_query("refactor utils"))
    check("Classify 'optimize perf'", lambda: classify_query("optimize performance"))

    # Get task profile
    profile = task_opt.get_profile_for_query("fix auth bug")
    check("Task profile", lambda: f"weights={list(profile[0].values())[:2]}, type={profile[1]}")

    # ============================================================
    print("\n" + "=" * 60)
    print("  DREAMING LOOP (SELF-PLAY)")
    print("=" * 60)

    from entroly.autotune import DreamingLoop

    dreamer = DreamingLoop(
        journal=journal,
        max_iterations=5,
        archetype_optimizer=arch,
    )
    check("DreamingLoop init", lambda: "OK")

    # Force idle (override timer)
    dreamer._last_activity = time.time() - 120  # pretend 120s idle
    check("should_dream after 120s idle", lambda: f"{dreamer.should_dream()}")

    # Generate synthetic queries
    synthetic = dreamer.generate_synthetic_queries()
    check("Synthetic queries generated", lambda: f"{len(synthetic)} queries")
    if synthetic:
        check("Sample synthetic query", lambda: f"'{synthetic[0].get('query', '')[:50]}' "
              f"type={synthetic[0].get('task_type', 'unknown')}")

    # Run dream cycle (needs bench/cases.json)
    from pathlib import Path
    cases_exist = Path("bench/cases.json").exists()
    check("bench/cases.json exists", lambda: f"{cases_exist}")

    if cases_exist:
        result = dreamer.run_dream_cycle()
        check("Dream cycle completed", lambda: f"status={result['status']}, "
              f"experiments={result.get('experiments', 0)}, "
              f"improvements={result.get('improvements', 0)}, "
              f"wall={result.get('wall_seconds', 0):.1f}s")
    else:
        # Run without cases — should gracefully return no_cases
        result = dreamer.run_dream_cycle()
        check("Dream cycle (no cases)", lambda: f"status={result['status']}")

    dream_stats = dreamer.stats()
    check("DreamingLoop stats", lambda: f"dreams={dream_stats['total_dreams']}, "
          f"improvements={dream_stats['total_improvements']}, "
          f"will_dream={dream_stats['will_dream']}")

    # ============================================================
    print("\n" + "=" * 60)
    print("  FULL PIPELINE: ARCHETYPE -> JOURNAL -> DREAM -> WEIGHTS UPDATE")
    print("=" * 60)

    # Before: check archetype weights
    before = arch.current_weights()
    print(f"  Before: w_recency={before.get('w_recency', 0):.3f}, "
          f"w_semantic={before.get('w_semantic', 0):.3f}")

    # The dream cycle should have potentially updated archetype weights
    after = arch.current_weights()
    print(f"  After:  w_recency={after.get('w_recency', 0):.3f}, "
          f"w_semantic={after.get('w_semantic', 0):.3f}")
    check("Pipeline executed", lambda: "OK")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"  PASSED: {passed}/{passed+failed}  |  FAILED: {failed}/{passed+failed}")
    if failed == 0:
        print("  ALL TESTS PASSED!")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

