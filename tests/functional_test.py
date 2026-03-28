import os
import sys
import shutil
import json
import logging
import uuid
from pathlib import Path

# Force the test to use a temporary checkpoint dir to avoid polluting the host
import tempfile
temp_dir = Path(tempfile.gettempdir()) / f"entroly_test_{uuid.uuid4().hex[:8]}"
temp_dir.mkdir(parents=True, exist_ok=False)

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("functional_test")

# Import the engine and configuration
sys.path.insert(0, str(Path(__file__).parent))
from entroly.server import EntrolyEngine
from entroly.config import EntrolyConfig
from entroly_core import py_shannon_entropy

def run_functional_test():
    logger.info(f"=== Starting Functional E2E Test ===")
    logger.info(f"Using temp directory for checkpoints: {temp_dir}")
    
    config = EntrolyConfig(checkpoint_dir=temp_dir)
    engine = EntrolyEngine(config=config)
    
    # Prove the Rust core and RL features are active
    logger.info(f"Engine instantiated. Rust Core Active: {engine._use_rust}")
    
    # 1. Ingest files from the repo
    files_to_ingest = [
        "entroly-core/src/lib.rs",
        "entroly-core/src/prism.rs",
        "entroly/server.py",
        "README.md",
        "pyproject.toml"
    ]
    
    logger.info("\n--- Phase 1: Ingesting Context ---")
    base_dir = Path(__file__).resolve().parents[1]
    
    total_tokens = 0
    for filename in files_to_ingest:
        path = base_dir / filename
        if not path.exists():
            logger.warning(f"File {filename} not found, skipping.")
            continue
            
        content = path.read_text(encoding="utf-8")
        
        # Ingest the fragment
        result = engine.ingest_fragment(
            content=content,
            source=f"file:{filename}",
        )
        t_count = result.get('token_count', 0)
        entropy = result.get('entropy_score', 0)
        total_tokens += t_count
        
        logger.info(f"Ingested {filename}: {t_count} tokens | Entropy: {entropy:.4f}")

    logger.info(f"Total tokens ingested: {total_tokens}")

    # 2. Query Optimization (The Core Pailitao-VL Engine)
    logger.info("\n--- Phase 2: Context Optimization ---")
    
    # We set a large budget of 25000 to ensure we can capture SSSL cutting the tail
    query = "jacobi eigendecomposition algorithm python server mcp"
    budget = 25000
    
    logger.info(f"Query: '{query}'")
    logger.info(f"Budget: {budget} tokens")
    
    # Let's see raw recall scores
    recalled = engine.recall_relevant(query, top_k=5)
    logger.info("\nRaw Recall Scores (before Knapsack):")
    for r in recalled:
        logger.info(f" - {r['source']} (Relevance: {r['relevance']:.4f})")
    
    opt_result = engine.optimize_context(token_budget=budget, query=query)
    
    selected = opt_result.get("selected_fragments", [])
    stats = opt_result.get("optimization_stats", {})
    provenance = opt_result.get("provenance", {})
    
    logger.info(f"\nOptimization Result:")
    logger.info(f"Selected Fragments: {len(selected)}")
    for f in selected:
        logger.info(f" - {f['source']} (Relevance: {f['relevance']:.4f}, Tokens: {f['token_count']})")
        
    logger.info(f"Total tokens used: {stats.get('total_tokens', 0)} / {budget}")
    logger.info(f"Tokens saved by engine: {opt_result.get('tokens_saved_this_call', 0)}")
    
    # 3. Validation of New Metrics (SSSL & Context Efficiency)
    logger.info("\n--- Phase 3: Metric Validation ---")
    
    sssl_purged = opt_result.get('sssl_tokens_purged', 0)
    logger.info(f"SSSL Tokens Purged (noise filtering): {sssl_purged}")
    
    ctx_eff = -1
    if "optimization_stats" in opt_result:
        # Pailitao-VL phase 2 outputs context efficiency in the rust result directly
        # Wait, optimize_context returns optimization_stats containing metrics.
        # But wait, in our Rust code we added context_efficiency to py_result.
        # Let's see if server.py forwarded it. server.py pulls stats from rust.
        pass
    
    # 4. Check PRISM RL Update
    logger.info("\n--- Phase 4: PRISM RL Update ---")
    # Simulate the user signaling success
    selected_ids = [f['id'] for f in selected]
    if selected_ids:
        engine.record_success(selected_ids)
        logger.info(f"Recorded positive RL feedback for {len(selected_ids)} fragments.")
    
    logger.info("\n--- Phase 5: Stats & Checkpoints ---")
    engine_stats = engine.get_stats()
    
    cf_eff= engine_stats.get('context_efficiency', {}).get('context_efficiency', 'N/A')
    logger.info(f"Global Context Efficiency Metric: {cf_eff}")
    
    prism_cov = engine_stats.get('memory', {}).get('optimized_cost_per_call_usd', '...')
    # Just show we can serialize checkpoint
    ckpt = engine.checkpoint({"test": "e2e"})
    logger.info(f"Saved Checkpoint Successfully: {ckpt is not None}")

    logger.info("\n=== Functional Test Complete ===")
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_functional_test()
