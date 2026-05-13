"""
Smoke test for entroly.benchmark_harness.

This module exposes `run_benchmark()` and is documented (in its own
module docstring) as being callable for offline benchmarking. The
function had no callers in the codebase prior to this test — verifying
it works keeps the module from drifting into a stale dead artifact.

The test runs in <1 second and uses an isolated, ephemeral engine so it
does not depend on a warm-start index or contaminate one.
"""
from __future__ import annotations

from entroly.benchmark_harness import (
    BENCHMARK_BUDGET,
    BENCHMARK_QUERY,
    run_benchmark,
)
from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine


def test_constants_exposed():
    """The module-level constants are part of its documented API."""
    assert isinstance(BENCHMARK_QUERY, str)
    assert len(BENCHMARK_QUERY) > 0
    assert isinstance(BENCHMARK_BUDGET, int)
    assert BENCHMARK_BUDGET > 0


def test_run_benchmark_returns_documented_shape(tmp_path):
    """run_benchmark() must return a dict with the keys callers expect.

    Ingests a few fragments first so the benchmark has something to
    select against; otherwise the optimizer returns empty and the
    metrics are degenerate.
    """
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=tmp_path / "ckpt",
    ))
    # Seed with enough content for a meaningful selection
    seed = [
        ("def authenticate_user(token): return verify(token)", "auth.py"),
        ("def process_payment(amount): return charge(amount)", "payments.py"),
        ("class RateLimiter:\n    def allow(self, key): return True", "ratelimit.py"),
        ("def verify(token): return token == 'valid'", "verify.py"),
        ("def charge(amount): return {'status': 'ok'}", "charge.py"),
    ]
    for content, source in seed:
        engine.ingest_fragment(content=content, source=source, token_count=20)

    result = run_benchmark(engine, budget_seconds=2.0)
    assert isinstance(result, dict)
    # Pin the documented return shape. If the harness changes its
    # contract, this test will surface the change — better than silent
    # drift in a module no other caller touches.
    expected_keys = {
        "context_efficiency",
        "dedup_tokens_avoided",
        "num_fragments_selected",
        "wall_seconds",
        "timed_out",
    }
    missing = expected_keys - set(result.keys())
    assert not missing, f"run_benchmark missing keys: {missing} (got {list(result.keys())})"
    # Sanity: we seeded 5 fragments, so selection should be non-empty
    assert result["num_fragments_selected"] > 0, (
        f"benchmark selected 0 fragments: {result}"
    )
    assert result["wall_seconds"] >= 0
    assert isinstance(result["timed_out"], bool)
