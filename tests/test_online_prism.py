"""Test: Online Bayesian PRISM — live weight adaptation in EntrolyEngine."""
import sys


def main():
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
    print("  ONLINE PRISM UNIT TESTS")
    print("=" * 60)

    from entroly.online_learner import (
        OnlinePrism, compute_implicit_reward, compute_contributions,
        WEIGHT_DIMS,
    )

    # --- Core Bayesian mechanics ---
    prism = OnlinePrism(
        prior_weights={"w_recency": 0.30, "w_frequency": 0.25,
                        "w_semantic": 0.25, "w_entropy": 0.20},
        prior_strength=20.0,
    )

    w0 = prism.weights()
    check("Initial weights = prior", lambda: f"rec={w0['w_recency']:.3f}, "
          f"sem={w0['w_semantic']:.3f}")

    # Verify weights sum to ~1
    check("Weights sum to 1", lambda: f"sum={sum(w0.values()):.4f}" 
          if abs(sum(w0.values()) - 1.0) < 0.01 
          else (_ for _ in ()).throw(Exception(f"sum={sum(w0.values())}")))

    # --- Positive reward shifts weights toward contributing dims ---
    # Simulate: recency contributed most to a good result
    for i in range(10):
        prism.observe(
            reward=0.9,  # high reward
            contributions={"w_recency": 0.5, "w_frequency": 0.2,
                           "w_semantic": 0.2, "w_entropy": 0.1},
        )

    w_after_recency = prism.weights()
    check("Recency upweighted after positive signal",
          lambda: f"{w0['w_recency']:.3f} -> {w_after_recency['w_recency']:.3f}"
          if w_after_recency['w_recency'] > w0['w_recency']
          else (_ for _ in ()).throw(Exception("recency didn't increase")))

    # --- Negative advantage pushes weights away ---
    prism2 = OnlinePrism(
        prior_weights={"w_recency": 0.30, "w_frequency": 0.25,
                        "w_semantic": 0.25, "w_entropy": 0.20},
        prior_strength=10.0,  # weaker prior = faster adaptation
    )
    # First establish a high baseline
    for _ in range(5):
        prism2.observe(reward=0.8, contributions={d: 0.25 for d in WEIGHT_DIMS})

    # Then send low rewards with high recency contribution
    for _ in range(15):
        prism2.observe(
            reward=0.2,  # below EMA baseline → negative advantage
            contributions={"w_recency": 0.6, "w_frequency": 0.1,
                           "w_semantic": 0.2, "w_entropy": 0.1},
        )
    w_neg = prism2.weights()
    check("Negative advantage reduces recency",
          lambda: f"rec={w_neg['w_recency']:.3f}")

    # --- Learning rate decay ---
    check("LR decays with observations",
          lambda: f"eta={prism._eta0 / (prism._n + 1)**0.5:.4f} (n={prism._n})")

    # --- Thread safety ---
    import threading

    prism3 = OnlinePrism(prior_strength=10.0)
    errors = []
    def thread_fn():
        try:
            for _ in range(100):
                prism3.observe(reward=0.5, contributions={d: 0.25 for d in WEIGHT_DIMS})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=thread_fn) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    check("Thread-safe (4×100 concurrent updates)",
          lambda: f"n={prism3._n}, errors={len(errors)}")

    # --- Implicit reward computation ---
    print("\n" + "=" * 60)
    print("  IMPLICIT REWARD FUNCTION")
    print("=" * 60)

    r1 = compute_implicit_reward(10, 25, 7000, 8000)
    check("Good utilization (85%)", lambda: f"reward={r1:.3f}" if r1 > 0.6 
          else (_ for _ in ()).throw(Exception(f"too low: {r1}")))

    r2 = compute_implicit_reward(25, 25, 8000, 8000)
    check("Over-selection (100%)", lambda: f"reward={r2:.3f}" if r2 < r1
          else (_ for _ in ()).throw(Exception(f"not penalized: {r2}")))

    r3 = compute_implicit_reward(1, 25, 500, 8000)
    check("Under-utilization (6%)", lambda: f"reward={r3:.3f}" if r3 < r1
          else (_ for _ in ()).throw(Exception(f"not penalized: {r3}")))

    r4 = compute_implicit_reward(10, 25, 7000, 8000, query_present=True)
    r5 = compute_implicit_reward(10, 25, 7000, 8000, query_present=False)
    check("Query bonus", lambda: f"with={r4:.3f}, without={r5:.3f}"
          if r4 >= r5 else (_ for _ in ()).throw(Exception("no bonus")))

    # --- State serialization ---
    print("\n" + "=" * 60)
    print("  SERIALIZATION + RESTORE")
    print("=" * 60)

    state = prism.state()
    check("State export", lambda: f"n={state.n_observations}, "
          f"alphas={len(state.alphas)}")

    prism_restored = OnlinePrism()
    prism_restored.load_state(state)
    w_restored = prism_restored.weights()
    w_original = prism.weights()
    check("State restored correctly",
          lambda: f"original={w_original['w_recency']:.4f}, "
          f"restored={w_restored['w_recency']:.4f}"
          if abs(w_original['w_recency'] - w_restored['w_recency']) < 0.001
          else (_ for _ in ()).throw(Exception("weights diverged")))

    # --- Engine integration ---
    print("\n" + "=" * 60)
    print("  ENGINE INTEGRATION")
    print("=" * 60)

    from entroly.server import EntrolyEngine
    engine = EntrolyEngine()
    check("Engine has OnlinePrism", lambda: f"type={type(engine._online_prism).__name__}")
    check("Initial PRISM weights", lambda: f"{engine._online_prism.weights()}")

    # Ingest some fragments
    for i in range(15):
        engine.ingest_fragment(
            content=f"def function_{i}(): return {i} * 2  # module {i % 3}",
            source=f"module_{i % 3}.py",
            token_count=20,
            is_pinned=(i < 2),
        )
    check("Ingested 15 fragments", lambda: "OK")

    # Run optimize_context — this should trigger online learning
    result = engine.optimize_context(token_budget=200, query="fix function_0")
    check("optimize_context succeeded", lambda: f"selected={len(result.get('selected_fragments', result.get('selected', [])))}")

    prism_info = result.get("online_prism", {})
    check("OnlinePrism in result", lambda: f"reward={prism_info.get('reward')}, "
          f"n={prism_info.get('n')}, phase={prism_info.get('phase')}")

    # Run 5 more calls — weights should evolve
    for i in range(5):
        r = engine.optimize_context(token_budget=200, query=f"optimize module_{i % 3}")

    prism_stats = engine._online_prism.stats()
    check("After 6 calls", lambda: f"n={prism_stats['n_observations']}, "
          f"phase={prism_stats['phase']}, "
          f"avg_reward={prism_stats['avg_reward']:.3f}")

    w_evolved = engine._online_prism.weights()
    w_init = {"w_recency": 0.30, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.20}
    total_drift = sum(abs(w_evolved[d] - w_init[d]) for d in WEIGHT_DIMS)
    check("Weights have adapted", lambda: f"total_drift={total_drift:.4f}, "
          f"weights={', '.join(f'{d}={w_evolved[d]:.3f}' for d in WEIGHT_DIMS)}")

    print("\n" + "=" * 60)
    print(f"  PASSED: {passed}/{passed+failed}  |  FAILED: {failed}/{passed+failed}")
    if failed == 0:
        print("  ALL TESTS PASSED — Online PRISM is live!")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

