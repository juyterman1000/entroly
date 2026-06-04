"""
Entroly Configuration
==========================

Central configuration for the context optimization engine.
All tunable parameters live here — no magic numbers buried in code.
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _project_checkpoint_dir() -> Path:
    """Return a project-isolated checkpoint directory.

    Uses ENTROLY_DIR if set, otherwise hashes the cwd to create
    ~/.entroly/checkpoints/{project_hash}/ so multiple projects
    don't bleed fragments into each other.
    """
    explicit = os.environ.get("ENTROLY_DIR")
    if explicit:
        return Path(explicit)
    cwd = os.getcwd()
    project_hash = hashlib.sha256(cwd.encode()).hexdigest()[:12]
    default_dir = Path(os.path.expanduser(f"~/.entroly/checkpoints/{project_hash}"))
    try:
        default_dir.mkdir(parents=True, exist_ok=True)
        probe = default_dir / ".entroly_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return default_dir
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "entroly" / "checkpoints" / project_hash
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


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
        default_factory=lambda: _project_checkpoint_dir()
    )
    """Directory for persisting checkpoint state (project-isolated)."""

    auto_checkpoint_interval: int = 5
    """Auto-checkpoint every N tool calls."""

    use_persistent_index: bool = True
    """Controls the shared warm-start index at `<checkpoint_dir>/index.json.gz`.

    When True (production default), EntrolyEngine loads the index on construction
    and writes back on every auto-checkpoint so subsequent sessions warm-start
    instantly.

    When False, the engine neither reads nor writes the shared index — it is
    fully ephemeral. Use this for isolated engines (tests, probes, multi-tenant
    workers, anything where state from another session would contaminate
    results, or where saving could corrupt another caller's state).
    """

    # ── Server ──────────────────────────────────────────────────────────
    server_name: str = "entroly"
    server_version: str = field(
        default_factory=lambda: __import__("entroly", fromlist=["__version__"]).__version__
    )


def resolve_tuning_kwargs(cfg: dict) -> dict:
    """Map a ``tuning_config`` dict to ``EntrolyConfig`` keyword arguments.

    `entroly autotune` writes the **nested** schema produced by
    ``bench/evaluate.py`` (``{"weights": {"recency": ...}, "decay": {...}}``).
    Earlier builds of the runtime loader read **flat** keys
    (``weight_recency``), so autotuned values were silently dropped and the
    engine always fell back to defaults. This bridges both: nested wins, flat
    is honoured for back-compat, and any missing / non-numeric / out-of-range
    value falls back to the ``EntrolyConfig`` default (never raises).

    Only the six fields the runtime engine actually consumes are mapped (the
    four knapsack weights + the two Ebbinghaus-decay params). ``knapsack`` /
    ``ios`` / ``dedup`` tuning keys have no corresponding runtime-engine wiring
    yet, so they are intentionally ignored here rather than mis-mapped.
    """
    if not isinstance(cfg, dict):
        cfg = {}
    # Read field defaults without instantiating EntrolyConfig (avoids the
    # checkpoint-dir mkdir side effect — this is a pure dict→dict mapper).
    _f = EntrolyConfig.__dataclass_fields__

    class _D:
        weight_recency = _f["weight_recency"].default
        weight_frequency = _f["weight_frequency"].default
        weight_semantic_sim = _f["weight_semantic_sim"].default
        weight_entropy = _f["weight_entropy"].default
        decay_half_life_turns = _f["decay_half_life_turns"].default
        min_relevance_threshold = _f["min_relevance_threshold"].default
    defaults = _D
    weights = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
    decay = cfg.get("decay") if isinstance(cfg.get("decay"), dict) else {}

    def pick(nested, flat_key, default, cast, lo, hi):
        # Prefer the nested autotune key, then the legacy flat key, then default.
        val = nested
        if val is None:
            val = cfg.get(flat_key)
        if val is None:
            return default
        try:
            val = cast(val)
        except (TypeError, ValueError):
            return default
        if val < lo or val > hi:
            return default
        return val

    return {
        "weight_recency": pick(
            weights.get("recency"), "weight_recency", defaults.weight_recency, float, 0.0, 1.0),
        "weight_frequency": pick(
            weights.get("frequency"), "weight_frequency", defaults.weight_frequency, float, 0.0, 1.0),
        "weight_semantic_sim": pick(
            weights.get("semantic_sim"), "weight_semantic_sim", defaults.weight_semantic_sim, float, 0.0, 1.0),
        "weight_entropy": pick(
            weights.get("entropy"), "weight_entropy", defaults.weight_entropy, float, 0.0, 1.0),
        "decay_half_life_turns": pick(
            decay.get("half_life_turns"), "decay_half_life_turns", defaults.decay_half_life_turns, int, 1, 1000),
        "min_relevance_threshold": pick(
            decay.get("min_relevance_threshold"), "min_relevance_threshold", defaults.min_relevance_threshold, float, 0.0, 1.0),
    }
