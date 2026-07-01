import logging
import sys
import tempfile
from pathlib import Path

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("functional_test")

# Import the engine and configuration
sys.path.insert(0, str(Path(__file__).parent))
from entroly.server import EntrolyEngine  # noqa: E402
from entroly.config import EntrolyConfig  # noqa: E402


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

        files_to_ingest = [
            "entroly-core/src/lib.rs",
            "entroly-core/src/prism.rs",
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
        assert tokens_saved == total_tokens - tokens_used
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
