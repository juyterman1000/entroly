"""
Entroly Configuration
==========================

Central configuration for the context optimization engine.
All tunable parameters live here — no magic numbers buried in code.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class EntrolyConfig:
    """Configuration for the Entroly MCP server."""

    # ── Token Budget ────────────────────────────────────────────────────
    default_token_budget: int = 128_000
    """Default max tokens for context optimization (matches GPT-4 Turbo)."""

    max_fragments: int = 10_000
    """Maximum context fragments tracked per session."""

    # ── Knapsack Optimizer Weights ──────────────────────────────────────
    weight_recency: float = 0.30
    """How much to weight recency (turns since last access)."""

    weight_frequency: float = 0.25
    """How much to weight access frequency."""

    weight_semantic_sim: float = 0.25
    """How much to weight semantic similarity to current query."""

    weight_entropy: float = 0.20
    """How much to weight information density (Shannon entropy)."""

    # ── Ebbinghaus Decay ────────────────────────────────────────────────
    decay_half_life_turns: int = 15
    """Number of turns for a fragment's relevance to halve."""

    min_relevance_threshold: float = 0.05
    """Fragments below this relevance get evicted entirely."""

    # ── Deduplication ───────────────────────────────────────────────────
    dedup_similarity_threshold: float = 0.92
    """SimHash Jaccard threshold above which fragments are considered duplicates."""

    # ── Predictive Pre-fetch ────────────────────────────────────────────
    prefetch_depth: int = 2
    """How many hops in the call graph to pre-fetch."""

    max_prefetch_fragments: int = 10
    """Maximum fragments to pre-fetch per symbol lookup."""

    # ── Checkpoint ──────────────────────────────────────────────────────
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ENTROLY_DIR",
                os.path.expanduser("~/.entroly/checkpoints"),
            )
        )
    )
    """Directory for persisting checkpoint state."""

    auto_checkpoint_interval: int = 5
    """Auto-checkpoint every N tool calls."""

    # ── Server ──────────────────────────────────────────────────────────
    server_name: str = "entroly"
    server_version: str = field(
        default_factory=lambda: __import__("entroly", fromlist=["__version__"]).__version__
    )
