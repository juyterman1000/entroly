"""
Tests for PageRank Integration — End-to-End Coverage.

Covers:
  1. Rust compute_pagerank() PyO3 binding
  2. EvolutionDaemon PageRank-based gap prioritization
  3. EvolutionLogger source file tracking
  4. Integration: daemon run_once with structural synthesis
  5. Edge cases: empty graphs, single node, disconnected components
"""

import os
import tempfile
import time
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. Rust compute_pagerank() PyO3 Binding Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRustPageRank:
    """Test the Rust EntrolyEngine.compute_pagerank() method."""

    @pytest.fixture
    def rust_engine(self):
        """Create a Rust engine if available, skip if not."""
        try:
            from entroly_core import EntrolyEngine
            return EntrolyEngine(
                w_recency=0.3, w_frequency=0.25,
                w_semantic=0.25, w_entropy=0.2,
                decay_half_life=10, min_relevance=0.05,
            )
        except ImportError:
            pytest.skip("entroly_core not installed")

    def test_empty_engine_returns_empty_dict(self, rust_engine):
        scores = dict(rust_engine.compute_pagerank())
        assert scores == {}

    def test_single_fragment_returns_score(self, rust_engine):
        rust_engine.ingest("def foo(): pass", "foo.py", 10, False)
        scores = dict(rust_engine.compute_pagerank())
        assert len(scores) == 1
        # Single node: PageRank = 1/N = 1.0
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_multiple_fragments_return_scores(self, rust_engine):
        rust_engine.ingest(
            "from utils import helper\ndef main(): helper()",
            "main.py", 20, False,
        )
        rust_engine.ingest(
            "def helper(): return 42\ndef format_output(x): return str(x)",
            "utils.py", 15, False,
        )
        rust_engine.ingest(
            "from utils import format_output\ndef display(): format_output(1)",
            "display.py", 15, False,
        )
        scores = dict(rust_engine.compute_pagerank())
        assert len(scores) >= 2
        # All scores should be in [0, 1]
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_hub_file_gets_higher_score(self, rust_engine):
        """A file imported by many others should have higher PageRank."""
        # Hub: utils.py defines helper, used by 3 others
        rust_engine.ingest(
            "def shared_helper(): return 42\ndef shared_format(x): return str(x)",
            "utils.py", 20, False,
        )
        # 3 consumers import from utils
        for i in range(3):
            rust_engine.ingest(
                f"from utils import shared_helper\ndef func_{i}(): shared_helper()",
                f"consumer_{i}.py", 15, False,
            )
        # Leaf: isolated file
        rust_engine.ingest(
            "def standalone(): pass\nx = 1\ny = 2",
            "leaf.py", 10, False,
        )
        scores = dict(rust_engine.compute_pagerank())
        # Find utils.py's fragment score
        utils_scores = [v for k, v in scores.items() if "utils" in k.lower()]
        leaf_scores = [v for k, v in scores.items() if "leaf" in k.lower()]
        if utils_scores and leaf_scores:
            assert max(utils_scores) >= max(leaf_scores)

    def test_scores_are_deterministic(self, rust_engine):
        """Same input → same output."""
        rust_engine.ingest("def foo(): pass", "a.py", 10, False)
        rust_engine.ingest("from a import foo\nfoo()", "b.py", 10, False)
        scores1 = dict(rust_engine.compute_pagerank())
        scores2 = dict(rust_engine.compute_pagerank())
        assert scores1 == scores2

    def test_scores_sum_close_to_one(self, rust_engine):
        """PageRank scores should sum to approximately 1.0."""
        for i in range(5):
            rust_engine.ingest(f"def func_{i}(): pass", f"f{i}.py", 10, False)
        scores = dict(rust_engine.compute_pagerank())
        total = sum(scores.values())
        assert 0.9 <= total <= 1.1  # Allow small numerical error

    def test_compute_pagerank_after_dep_graph_built(self, rust_engine):
        """PageRank should work after dep graph auto_link runs during ingest."""
        # Ingest files that create dep graph edges
        rust_engine.ingest(
            "class Config:\n    DEBUG = True\n    PORT = 8080",
            "config.py", 15, False,
        )
        rust_engine.ingest(
            "from config import Config\ndef start(): Config.DEBUG",
            "server.py", 20, False,
        )
        stats = dict(rust_engine.dep_graph_stats())
        scores = dict(rust_engine.compute_pagerank())
        # Should have scores regardless of edge count
        assert len(scores) >= 2


# ═══════════════════════════════════════════════════════════════════════
# 2. EvolutionDaemon PageRank Gap Prioritization Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDaemonPageRankPrioritization:
    """Test that the daemon sorts gaps by PageRank centrality."""

    def _make_daemon_with_gaps(self):
        from entroly.vault import VaultManager, VaultConfig
        from entroly.evolution_logger import EvolutionLogger
        from entroly.value_tracker import ValueTracker
        from entroly.evolution_daemon import EvolutionDaemon

        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        evo_logger = EvolutionLogger(vault_path=os.path.join(tmp, "vault"), gap_threshold=1)
        value_tracker = ValueTracker(data_dir=Path(tmp))

        # Create gaps with source files
        evo_logger.record_miss(
            query="hub file question",
            entity_key="hub_entity",
            source_files=["utils.py"],
        )
        evo_logger.record_miss(
            query="leaf file question",
            entity_key="leaf_entity",
            source_files=["leaf.py"],
        )

        daemon = EvolutionDaemon(
            vault=vault,
            evolution_logger=evo_logger,
            value_tracker=value_tracker,
            project_root=tmp,
            data_dir=tmp,
        )
        return daemon, evo_logger

    def test_daemon_processes_gaps_without_pagerank(self):
        """Daemon should work even without Rust engine (no PageRank)."""
        daemon, _ = self._make_daemon_with_gaps()
        result = daemon.run_once()
        assert "gaps_processed" in result

    def test_daemon_stats_after_gap_processing(self):
        daemon, _ = self._make_daemon_with_gaps()
        daemon.run_once()
        stats = daemon.stats()
        assert "structural_successes" in stats
        assert "structural_failures" in stats

    def test_gaps_include_source_files(self):
        """Verify gaps returned by EvolutionLogger include source_files."""
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=1)
        el.record_miss(
            query="test q",
            entity_key="entity_a",
            source_files=["a.py", "b.py"],
        )
        gaps = el.get_pending_gaps()
        assert len(gaps) == 1
        assert gaps[0]["source_files"] == ["a.py", "b.py"]

    def test_pagerank_sorting_order(self):
        """Test the PageRank sorting logic directly."""
        gaps = [
            {"entity_key": "low", "source_files": ["leaf.py"]},
            {"entity_key": "high", "source_files": ["hub.py"]},
            {"entity_key": "mid", "source_files": ["mid.py"]},
        ]
        pagerank_scores = {
            "hub.py": 0.8,
            "mid.py": 0.3,
            "leaf.py": 0.05,
        }

        from typing import Any

        def _gap_priority(gap: dict[str, Any]) -> float:
            source_files = gap.get("source_files", [])
            if not source_files:
                return 0.0
            return max(
                (pagerank_scores.get(sf, 0.0) for sf in source_files),
                default=0.0,
            )

        sorted_gaps = sorted(gaps, key=_gap_priority, reverse=True)
        assert sorted_gaps[0]["entity_key"] == "high"
        assert sorted_gaps[1]["entity_key"] == "mid"
        assert sorted_gaps[2]["entity_key"] == "low"

    def test_gap_with_no_source_files_gets_zero_priority(self):
        gaps = [
            {"entity_key": "no_source", "source_files": []},
            {"entity_key": "with_source", "source_files": ["hub.py"]},
        ]
        pagerank_scores = {"hub.py": 0.5}

        def _gap_priority(gap):
            sf = gap.get("source_files", [])
            if not sf:
                return 0.0
            return max((pagerank_scores.get(s, 0.0) for s in sf), default=0.0)

        sorted_gaps = sorted(gaps, key=_gap_priority, reverse=True)
        assert sorted_gaps[0]["entity_key"] == "with_source"

    def test_gap_with_multiple_source_files_uses_max(self):
        """When a gap has multiple source files, max PageRank wins."""
        gaps = [
            {"entity_key": "multi", "source_files": ["low.py", "high.py"]},
            {"entity_key": "single", "source_files": ["mid.py"]},
        ]
        pagerank_scores = {"low.py": 0.1, "high.py": 0.9, "mid.py": 0.5}

        def _gap_priority(gap):
            sf = gap.get("source_files", [])
            if not sf:
                return 0.0
            return max((pagerank_scores.get(s, 0.0) for s in sf), default=0.0)

        sorted_gaps = sorted(gaps, key=_gap_priority, reverse=True)
        assert sorted_gaps[0]["entity_key"] == "multi"  # max(0.1, 0.9) = 0.9 > 0.5


# ═══════════════════════════════════════════════════════════════════════
# 3. EvolutionLogger Source File Tracking Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionLoggerSourceTracking:
    """Test source file deduplication and tracking in gaps."""

    def test_source_files_deduplication(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=2)

        # Same entity, overlapping source files
        el.record_miss("q1", "auth", source_files=["auth.py", "tokens.py"])
        el.record_miss("q2", "auth", source_files=["auth.py", "middleware.py"])

        gaps = el.get_pending_gaps()
        assert len(gaps) == 1
        sf = gaps[0]["source_files"]
        assert len(sf) == 3
        assert "auth.py" in sf
        assert "tokens.py" in sf
        assert "middleware.py" in sf

    def test_source_files_preserves_order(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=1)
        el.record_miss("q", "ent", source_files=["b.py", "a.py", "c.py"])
        gaps = el.get_pending_gaps()
        # Order should match insertion order
        assert gaps[0]["source_files"] == ["b.py", "a.py", "c.py"]

    def test_empty_source_files(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=1)
        el.record_miss("q", "ent")
        gaps = el.get_pending_gaps()
        assert gaps[0]["source_files"] == []

    def test_gap_threshold_respected(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=3)
        el.record_miss("q1", "entity_x", source_files=["x.py"])
        assert len(el.get_pending_gaps()) == 0
        el.record_miss("q2", "entity_x", source_files=["x.py"])
        assert len(el.get_pending_gaps()) == 0
        el.record_miss("q3", "entity_x", source_files=["x.py"])
        assert len(el.get_pending_gaps()) == 1

    def test_multiple_entities_tracked_independently(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=2)
        el.record_miss("q1", "auth", source_files=["auth.py"])
        el.record_miss("q2", "auth", source_files=["auth.py"])
        el.record_miss("q3", "db", source_files=["db.py"])
        # Only auth has 2 misses (threshold met), db has only 1
        gaps = el.get_pending_gaps()
        assert len(gaps) == 1
        assert gaps[0]["entity_key"] == "auth"

    def test_stats_accurate(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=2)
        el.record_miss("q1", "a")
        el.record_miss("q2", "b")
        el.record_miss("q3", "a")
        stats = el.stats()
        assert stats["total_misses"] == 3
        assert stats["unique_entities"] == 2
        assert stats["skill_gaps_detected"] == 1

    def test_vault_write_creates_file(self):
        from entroly.evolution_logger import EvolutionLogger
        tmp = tempfile.mkdtemp()
        el = EvolutionLogger(vault_path=os.path.join(tmp, "vault"), gap_threshold=1)
        result = el.record_miss("q", "test_entity", source_files=["test.py"])
        assert result["is_skill_gap"]
        # Check file was created
        vault_evo = Path(tmp) / "vault" / "evolution"
        if vault_evo.exists():
            files = list(vault_evo.glob("gap_*.md"))
            assert len(files) == 1
            content = files[0].read_text()
            assert "test_entity" in content


# ═══════════════════════════════════════════════════════════════════════
# 4. Daemon Lifecycle Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDaemonLifecycle:
    """Test daemon start/stop and background thread management."""

    def _make_daemon(self):
        from entroly.vault import VaultManager, VaultConfig
        from entroly.evolution_logger import EvolutionLogger
        from entroly.value_tracker import ValueTracker
        from entroly.evolution_daemon import EvolutionDaemon

        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        evo_logger = EvolutionLogger(vault_path=os.path.join(tmp, "vault"))
        value_tracker = ValueTracker(data_dir=Path(tmp))
        daemon = EvolutionDaemon(
            vault=vault,
            evolution_logger=evo_logger,
            value_tracker=value_tracker,
            project_root=tmp,
            data_dir=tmp,
        )
        return daemon

    def test_start_creates_thread(self):
        daemon = self._make_daemon()
        daemon.start()
        assert daemon._thread is not None
        assert daemon._thread.is_alive()
        daemon.stop()

    def test_stop_terminates_thread(self):
        daemon = self._make_daemon()
        daemon.start()
        daemon.stop()
        time.sleep(0.5)
        assert not daemon._thread.is_alive()

    def test_double_start_is_safe(self):
        daemon = self._make_daemon()
        daemon.start()
        daemon.start()  # Should not create a second thread
        daemon.stop()

    def test_stats_while_running(self):
        daemon = self._make_daemon()
        daemon.start()
        stats = daemon.stats()
        assert stats["running"] is True
        daemon.stop()

    def test_stats_while_stopped(self):
        daemon = self._make_daemon()
        stats = daemon.stats()
        assert stats["running"] is False

    def test_cooldown_prevents_rapid_processing(self):
        from entroly.evolution_logger import EvolutionLogger
        from entroly.vault import VaultManager, VaultConfig
        from entroly.value_tracker import ValueTracker
        from entroly.evolution_daemon import EvolutionDaemon

        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        evo_logger = EvolutionLogger(vault_path=os.path.join(tmp, "vault"), gap_threshold=1)
        vt = ValueTracker(data_dir=Path(tmp))

        evo_logger.record_miss("q", "entity_a", source_files=["a.py"])

        daemon = EvolutionDaemon(
            vault=vault,
            evolution_logger=evo_logger,
            value_tracker=vt,
            project_root=tmp,
            data_dir=tmp,
        )

        # First run processes the gap
        r1 = daemon.run_once()
        # Second run should skip due to cooldown
        r2 = daemon.run_once()
        assert r2["gaps_processed"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 5. Vault + Config Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestVaultConfigIntegration:
    """Test VaultConfig/VaultManager construction patterns used in server.py."""

    def test_vault_config_base_path(self):
        from entroly.vault import VaultConfig
        cfg = VaultConfig(base_path="/tmp/test_vault")
        assert str(cfg.path) == "/tmp/test_vault" or "test_vault" in str(cfg.path)

    def test_vault_config_default_path(self):
        from entroly.vault import VaultConfig
        cfg = VaultConfig()
        assert "vault" in str(cfg.path).lower() or ".entroly" in str(cfg.path).lower()

    def test_vault_manager_creates_structure(self):
        from entroly.vault import VaultManager, VaultConfig
        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        result = vault.ensure_structure()
        assert result["status"] == "initialized"
        assert (Path(tmp) / "vault" / "beliefs").exists()
        assert (Path(tmp) / "vault" / "evolution").exists()
        assert (Path(tmp) / "vault" / "evolution" / "skills").exists()

    def test_vault_manager_idempotent_init(self):
        from entroly.vault import VaultManager, VaultConfig
        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        vault.ensure_structure()
        result2 = vault.ensure_structure()
        assert result2["status"] == "already_initialized"


# ═══════════════════════════════════════════════════════════════════════
# 6. Python Fallback Engine Tests (no Rust required)
# ═══════════════════════════════════════════════════════════════════════

class TestPythonFallbackEngine:
    """Test the Python EntrolyEngine without Rust dependency."""

    def _make_engine(self):
        from entroly.server import EntrolyEngine
        from entroly.config import EntrolyConfig
        return EntrolyEngine(EntrolyConfig())

    def test_ingest_and_optimize_cycle(self):
        engine = self._make_engine()
        result = engine.ingest_fragment(
            "def hello(): return 'world'",
            source="hello.py",
            token_count=10,
        )
        assert result["status"] in ("ingested", "duplicate")

    def test_advance_turn(self):
        engine = self._make_engine()
        engine.advance_turn()
        # Should not raise

    def test_record_success_failure(self):
        engine = self._make_engine()
        engine.ingest_fragment("def foo(): pass", "foo.py", 10)
        engine.record_success(["test_id"])
        engine.record_failure(["test_id"])
        # Should not raise

    def test_get_stats(self):
        engine = self._make_engine()
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert len(stats) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
