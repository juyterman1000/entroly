import logging
import sys
import tempfile
from pathlib import Path

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("functional_test")

# Import the engine and configuration from *this* checkout.
#
# This inserted `Path(__file__).parent` -- the tests/ directory -- which does
# nothing for `import entroly`, so resolution fell through to site-packages.
# Running `python tests/functional_test.py` puts tests/ on sys.path[0] and
# never the repository root, so on a machine carrying an editable install of
# entroly pointing somewhere else, this script silently exercised that other
# checkout and reported its result as this one's. That happened here: a stale
# `.pth` from an old worktree made this script pass locally while CI failed on
# all five Python versions.
#
# `parents[1]` is the repository root, the same anchor `base_dir` below already
# uses for the fixtures. A test that cannot say which code it ran is not
# evidence, so this is pinned rather than left to the environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import entroly  # noqa: E402

# Fail loudly rather than reporting another checkout's result as this one's.
# The path insert above is enough on its own; this is here because the failure
# it prevents is silent by nature -- the script ran, printed numbers, and
# passed, and nothing in the output said which tree produced them.
_imported_from = Path(entroly.__file__).resolve()
if _REPO_ROOT not in _imported_from.parents:
    raise SystemExit(
        f"functional test imported entroly from {_imported_from}, which is "
        f"outside this checkout ({_REPO_ROOT}). An editable install or "
        f"PYTHONPATH entry is shadowing the package; the result would describe "
        f"code that is not under test."
    )

from entroly.server import EntrolyEngine  # noqa: E402
from entroly.config import EntrolyConfig  # noqa: E402
from entroly.engine import naive_context_baseline  # noqa: E402


def run_functional_test() -> None:
    with tempfile.TemporaryDirectory(prefix="entroly_functional_") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        logger.info("=== Starting Functional E2E Test ===")
        logger.info(f"Using temp directory for checkpoints: {temp_dir}")

        config = EntrolyConfig(checkpoint_dir=temp_dir)
        engine = EntrolyEngine(config=config)

        # This CI surface is specifically the native end-to-end contract.
        assert engine._use_rust, "functional test requires the Rust core"
        logger.info(f"Engine instantiated. Rust Core Active: {engine._use_rust}")

        # Real repository files used as ingestion fixtures. This is a hardcoded
        # path contract: `prism.rs` moved to entroly-engine when the shared
        # compute crate was extracted so the PyO3 and WebAssembly builds stop
        # carrying separate copies of the algorithms.
        #
        # Note this script is NOT collected by `pytest tests/` -- it runs
        # standalone (`python tests/functional_test.py`) and in CI -- so a stale
        # path here survives a fully green local test run.
        files_to_ingest = [
            "entroly-core/src/lib.rs",
            "entroly-engine/src/prism.rs",
            "entroly/server.py",
            "README.md",
            "pyproject.toml",
        ]

        logger.info("\n--- Phase 1: Ingesting Context ---")
        base_dir = Path(__file__).resolve().parents[1]

        total_tokens = 0
        for filename in files_to_ingest:
            path = base_dir / filename
            assert path.exists(), f"required functional fixture is missing: {filename}"

            content = path.read_text(encoding="utf-8")
            result = engine.ingest_fragment(
                content=content,
                source=f"file:{filename}",
            )
            assert result.get("status") == "ingested", result
            t_count = result.get("token_count", 0)
            entropy = result.get("entropy_score", 0)
            assert t_count > 0
            total_tokens += t_count

            logger.info(f"Ingested {filename}: {t_count} tokens | Entropy: {entropy:.4f}")

        assert total_tokens > 0
        logger.info(f"Total tokens ingested: {total_tokens}")

        logger.info("\n--- Phase 2: Context Optimization ---")
        query = "jacobi eigendecomposition algorithm python server mcp"
        budget = 25000
        logger.info(f"Query: '{query}'")
        logger.info(f"Budget: {budget} tokens")

        recalled = engine.recall_relevant(query, top_k=5)
        assert recalled, "recall returned no candidates"
        logger.info("\nRaw Recall Scores (before Knapsack):")
        for r in recalled:
            logger.info(f" - {r['source']} (Relevance: {r['relevance']:.4f})")

        opt_result = engine.optimize_context(token_budget=budget, query=query)

        selected = opt_result.get("selected_fragments", [])
        stats = opt_result.get("optimization_stats", {})
        tokens_used = opt_result.get("total_tokens", stats.get("total_tokens", 0))
        tokens_saved = opt_result.get(
            "tokens_saved", opt_result.get("tokens_saved_this_call", 0)
        )
        assert opt_result.get("selector") == "qccr", opt_result
        assert selected, "optimization returned no context"
        assert 0 < tokens_used <= budget
        # A saving is measured against a prompt someone could have sent, not
        # against the whole ingested corpus. This asserted
        # `total_tokens - tokens_used`, which credited the engine with the
        # entire corpus: on this fixture that is 107,047 tokens "saved" against
        # 11,442 actually sent, and it grows with the corpus rather than with
        # anything the engine did. Nobody pastes 118k tokens into a model.
        # `naive_context_baseline` is the shared ceiling; asserting against the
        # function rather than a literal keeps this in step with the engine.
        assert tokens_saved == naive_context_baseline(total_tokens) - tokens_used, (
            f"expected {naive_context_baseline(total_tokens) - tokens_used}, "
            f"got {tokens_saved} (corpus {total_tokens}, used {tokens_used})"
        )
        assert 0 <= tokens_saved <= naive_context_baseline(total_tokens)
        assert opt_result.get("selected_count") == len(selected)

        logger.info("\nOptimization Result:")
        logger.info(f"Selected Fragments: {len(selected)}")
        for f in selected:
            logger.info(f" - {f['source']} (Relevance: {f['relevance']:.4f}, Tokens: {f['token_count']})")

        logger.info(f"Total tokens used: {tokens_used} / {budget}")
        logger.info(f"Tokens saved by engine: {tokens_saved}")

        logger.info("\n--- Phase 3: Metric Validation ---")
        logger.info(
            f"Budget utilization: {opt_result.get('budget_utilization', 0):.4f}"
        )

        logger.info("\n--- Phase 4: PRISM RL Update ---")
        selected_ids = [f["id"] for f in selected]
        assert selected_ids
        engine.record_success(selected_ids)
        logger.info(f"Recorded positive RL feedback for {len(selected_ids)} fragments.")

        logger.info("\n--- Phase 5: Stats & Checkpoints ---")
        engine_stats = engine.get_stats()
        assert isinstance(engine_stats, dict)

        cf_eff = engine_stats.get("context_efficiency", {}).get(
            "context_efficiency", "N/A"
        )
        logger.info(f"Global Context Efficiency Metric: {cf_eff}")

        ckpt = engine.checkpoint({"test": "e2e"})
        assert ckpt is not None
        logger.info("Saved Checkpoint Successfully: True")

        logger.info("\n=== Functional Test Complete ===")


if __name__ == "__main__":
    run_functional_test()
