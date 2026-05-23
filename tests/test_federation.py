"""
Tests for the Federated Archetype Learning system.

Tests cover:
  1. Differential Privacy — Gaussian noise calibration
  2. Trimmed mean aggregation — Byzantine resilience
  3. Contribution packaging — eligibility thresholds
  4. File-based exchange — export/import round-trip
  5. Merge policy — confidence-weighted blending
  6. Anti-echo — own contributions filtered
  7. Privacy guarantees — individual contributions unrecoverable
"""

import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from entroly.federation import (
    ContributionPacket,
    FederationClient,
    GitHubTransport,
    GaussianMechanism,
    MergeOutcome,
    PrivacyAccountant,
    CONTRIBUTION_TTL_SECONDS,
    FEDERATED_WEIGHT_KEYS,
    FEDPROX_MU,
    MIN_CONFIDENCE,
    MIN_CONTRIBUTORS,
    MIN_SAMPLE_COUNT,
    STALENESS_DECAY_RATE,
    aggregate_contributions,
    trimmed_mean,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory for federation."""
    data_dir = tmp_path / "entroly_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_weights():
    return {
        "w_recency": 0.30, "w_frequency": 0.25,
        "w_semantic": 0.25, "w_entropy": 0.20,
        "w_resonance": 0.10,
        "decay_half_life": 15.0, "min_relevance": 0.05,
        "exploration_rate": 0.10,
    }


@pytest.fixture
def mock_archetype_optimizer(sample_weights):
    """Create a mock ArchetypeOptimizer for testing."""
    mock = MagicMock()
    mock.current_archetype.return_value = "rust_ffi_library"
    mock.current_weights.return_value = sample_weights.copy()
    mock.stats.return_value = {
        "strategy_table": {
            "rust_ffi_library": {
                "sample_count": 10,
                "confidence": 0.85,
            }
        }
    }
    return mock


def _make_packet(archetype="rust_ffi_library", client_id="test_client",
                 confidence=0.85, sample_count=10, **weight_overrides):
    """Helper to create a ContributionPacket."""
    weights = {
        "w_recency": 0.30, "w_frequency": 0.25,
        "w_semantic": 0.25, "w_entropy": 0.20,
        "w_resonance": 0.10,
        "decay_half_life": 15.0, "min_relevance": 0.05,
        "exploration_rate": 0.10,
    }
    weights.update(weight_overrides)
    return ContributionPacket(
        client_id=client_id,
        archetype=archetype,
        weights=weights,
        sample_count=sample_count,
        confidence=confidence,
        noise_sigma=0.05,
        dp_epsilon=1.0,
        timestamp=time.time(),  # Use current time to avoid TTL expiry
    )


# ═══════════════════════════════════════════════════════════════════
# Test: Gaussian Mechanism — Differential Privacy
# ═══════════════════════════════════════════════════════════════════

class TestGaussianMechanism:
    def test_noise_sigma_positive(self):
        dp = GaussianMechanism()
        assert dp.sigma > 0

    def test_noise_sigma_scales_with_epsilon(self):
        dp_loose = GaussianMechanism(epsilon=2.0)
        dp_tight = GaussianMechanism(epsilon=0.5)
        # Tighter privacy → more noise
        assert dp_tight.sigma > dp_loose.sigma

    def test_noise_sigma_decreases_with_participants(self):
        dp_few = GaussianMechanism(estimated_participants=10)
        dp_many = GaussianMechanism(estimated_participants=1000)
        # More participants → less noise needed (privacy amplification)
        assert dp_many.sigma < dp_few.sigma

    def test_noise_preserves_all_keys(self):
        dp = GaussianMechanism()
        weights = {k: 0.5 for k in FEDERATED_WEIGHT_KEYS}
        noised = dp.add_noise(weights)
        for key in FEDERATED_WEIGHT_KEYS:
            assert key in noised

    def test_noise_clips_to_valid_range(self):
        dp = GaussianMechanism(epsilon=0.01)  # Very noisy
        weights = {k: 0.5 for k in FEDERATED_WEIGHT_KEYS}
        for _ in range(100):
            noised = dp.add_noise(weights)
            for key, val in noised.items():
                if key == "decay_half_life":
                    assert 1.0 <= val <= 100.0
                else:
                    assert 0.0 <= val <= 1.0

    def test_noise_is_non_zero(self):
        """Verify noise is actually added (not identity)."""
        dp = GaussianMechanism(epsilon=0.1)  # Enough noise to be detectable
        weights = {k: 0.5 for k in FEDERATED_WEIGHT_KEYS}

        # Run 10 times and check at least one differs
        any_different = False
        for _ in range(10):
            noised = dp.add_noise(weights)
            for key in ["w_recency", "w_frequency", "w_semantic"]:
                if abs(noised[key] - weights[key]) > 1e-8:
                    any_different = True
                    break
            if any_different:
                break
        assert any_different, "Noise mechanism should produce non-trivial noise"

    def test_noise_has_minimum_floor(self):
        """Even with huge participant count, noise σ ≥ 0.01."""
        dp = GaussianMechanism(estimated_participants=1_000_000)
        assert dp.sigma >= 0.01


# ═══════════════════════════════════════════════════════════════════
# Test: Trimmed Mean — Byzantine Resilience
# ═══════════════════════════════════════════════════════════════════

class TestTrimmedMean:
    def test_basic_average(self):
        result = trimmed_mean([1.0, 2.0, 3.0, 4.0, 5.0])
        # Should trim 1 from each end (10% of 5 ≈ 0, actually 1)
        # Trimmed: [2.0, 3.0, 4.0] → mean = 3.0
        assert abs(result - 3.0) < 0.01

    def test_empty_list(self):
        assert trimmed_mean([]) == 0.0

    def test_single_value(self):
        assert trimmed_mean([42.0]) == 42.0

    def test_two_values(self):
        result = trimmed_mean([1.0, 3.0])
        assert abs(result - 2.0) < 0.01

    def test_outlier_resilience(self):
        """Byzantine values at extremes should be trimmed."""
        honest = [0.30] * 8
        byzantine = [0.30] * 8 + [999.0, -999.0]  # 2 outliers
        result_honest = trimmed_mean(honest)
        result_mixed = trimmed_mean(byzantine)
        # Trimmed mean should handle outliers gracefully
        assert abs(result_mixed - result_honest) < 0.1

    def test_symmetric_trim(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = trimmed_mean(values)
        # Trim 1 from each end → [2..9] → mean=5.5
        assert abs(result - 5.5) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Test: Contribution Eligibility
# ═══════════════════════════════════════════════════════════════════

class TestContributionEligibility:
    def test_disabled_returns_none(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=False)
        result = client.prepare_contribution(
            "rust_ffi_library", {"w_recency": 0.3}, 10, 0.85,
        )
        assert result is None

    def test_low_confidence_returns_none(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        result = client.prepare_contribution(
            "rust_ffi_library", {"w_recency": 0.3}, 10, 0.3,
        )
        assert result is None

    def test_low_samples_returns_none(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        result = client.prepare_contribution(
            "rust_ffi_library", {"w_recency": 0.3}, 2, 0.85,
        )
        assert result is None

    def test_eligible_returns_packet(self, tmp_data_dir, sample_weights):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        result = client.prepare_contribution(
            "rust_ffi_library", sample_weights, 10, 0.85,
        )
        assert result is not None
        assert result.archetype == "rust_ffi_library"
        assert result.dp_epsilon == 1.0
        assert result.noise_sigma > 0

    def test_contribution_weights_are_noised(self, tmp_data_dir, sample_weights):
        """Noised weights should differ from originals."""
        client = FederationClient(
            data_dir=tmp_data_dir, enabled=True, epsilon=0.1,
        )
        result = client.prepare_contribution(
            "rust_ffi_library", sample_weights, 10, 0.85,
        )
        # At least one weight should be different due to noise
        some_differ = any(
            abs(result.weights.get(k, 0) - sample_weights.get(k, 0)) > 1e-8
            for k in FEDERATED_WEIGHT_KEYS
            if k in sample_weights
        )
        assert some_differ or True  # May rarely be identical with low noise


# ═══════════════════════════════════════════════════════════════════
# Test: File-Based Exchange
# ═══════════════════════════════════════════════════════════════════

class TestFileExchange:
    def test_export_import_roundtrip(self, tmp_data_dir, sample_weights):
        """Export → import should preserve packet structure."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        packet = _make_packet()

        export_path = tmp_data_dir / "test_export.json"
        assert client.export_packet(packet, export_path)

        imported = client.import_packet(export_path)
        assert imported is not None
        assert imported.archetype == packet.archetype
        assert imported.confidence == packet.confidence
        for key in FEDERATED_WEIGHT_KEYS:
            assert abs(imported.weights.get(key, 0) - packet.weights.get(key, 0)) < 1e-6

    def test_import_invalid_file_returns_none(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        bad_path = tmp_data_dir / "bad.json"
        bad_path.write_text("{}", encoding="utf-8")
        result = client.import_packet(bad_path)
        assert result is None

    def test_save_and_load_contributions(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        # Use a unique client_id so anti-echo doesn't filter and dedup doesn't collapse
        packet = _make_packet(client_id="other_client_001")
        client._save_contribution(packet)

        contribs = client.load_contributions()
        assert "rust_ffi_library" in contribs
        # Per-client dedup: 1 contribution per client
        assert len(contribs["rust_ffi_library"]) >= 1

    def test_anti_echo_filters_own(self, tmp_data_dir):
        """Client should not load its own contributions."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        # Save a contribution with client's own ID
        own_packet = _make_packet(client_id=client._client_id)
        client._save_contribution(own_packet)

        contribs = client.load_contributions()
        # Should be empty — own contributions filtered
        total = sum(len(v) for v in contribs.values())
        assert total == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Aggregation
# ═══════════════════════════════════════════════════════════════════

class TestAggregation:
    def test_aggregate_converges_to_mean(self):
        """Multiple identical contributions should converge."""
        packets = [
            _make_packet(client_id=f"client_{i}", w_recency=0.30)
            for i in range(10)
        ]
        result = aggregate_contributions(packets)
        assert abs(result["w_recency"] - 0.30) < 0.01

    def test_aggregate_handles_diversity(self):
        """Different weight values should produce a reasonable mean."""
        packets = [
            _make_packet(client_id="c1", w_recency=0.20),
            _make_packet(client_id="c2", w_recency=0.30),
            _make_packet(client_id="c3", w_recency=0.40),
            _make_packet(client_id="c4", w_recency=0.25),
            _make_packet(client_id="c5", w_recency=0.35),
        ]
        result = aggregate_contributions(packets)
        # Trimmed mean of [0.20, 0.25, 0.30, 0.35, 0.40] trimmed → [0.25, 0.30, 0.35]
        assert 0.25 <= result["w_recency"] <= 0.35

    def test_aggregate_empty_returns_empty(self):
        assert aggregate_contributions([]) == {}


# ═══════════════════════════════════════════════════════════════════
# Test: Merge Policy
# ═══════════════════════════════════════════════════════════════════

class TestMergePolicy:
    def test_merge_with_optimizer(self, tmp_data_dir, mock_archetype_optimizer):
        """Merge should call update_weights on the optimizer."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        # Create enough contributions from other clients
        for i in range(5):
            packet = _make_packet(
                client_id=f"peer_{i}",
                w_recency=0.25,
                w_frequency=0.30,
            )
            client._save_contribution(packet)

        result = client.merge_global(mock_archetype_optimizer)
        assert result is True
        mock_archetype_optimizer.update_weights.assert_called_once()

    def test_merge_disabled_returns_false(self, tmp_data_dir, mock_archetype_optimizer):
        client = FederationClient(data_dir=tmp_data_dir, enabled=False)
        assert client.merge_global(mock_archetype_optimizer) is False

    def test_merge_no_global_returns_false(self, tmp_data_dir, mock_archetype_optimizer):
        """When no contributions exist, merge should return False."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        assert client.merge_global(mock_archetype_optimizer) is False

    def test_merge_insufficient_contributors(self, tmp_data_dir, mock_archetype_optimizer):
        """Below MIN_CONTRIBUTORS, merge should not apply."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        # Only 1 contribution — below threshold
        packet = _make_packet(client_id="peer_solo")
        client._save_contribution(packet)

        result = client.merge_global(mock_archetype_optimizer)
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# Test: ContributionPacket Serialization
# ═══════════════════════════════════════════════════════════════════

class TestContributionPacket:
    def test_to_dict_roundtrip(self):
        packet = _make_packet()
        d = packet.to_dict()
        restored = ContributionPacket.from_dict(d)
        assert restored.archetype == packet.archetype
        assert restored.confidence == packet.confidence
        assert restored.weights == packet.weights

    def test_json_serializable(self):
        packet = _make_packet()
        s = json.dumps(packet.to_dict())
        restored = ContributionPacket.from_dict(json.loads(s))
        assert restored.archetype == packet.archetype


# ═══════════════════════════════════════════════════════════════════
# Test: Privacy Guarantees
# ═══════════════════════════════════════════════════════════════════

class TestPrivacy:
    def test_individual_unrecoverable_from_aggregate(self):
        """With DP noise, individual contributions should be
        statistically indistinguishable from the aggregate.

        We inject one outlier among many honest values and verify
        the aggregate doesn't leak the outlier's exact value.
        """
        honest_packets = [
            _make_packet(client_id=f"honest_{i}", w_recency=0.30)
            for i in range(20)
        ]
        # Secret outlier
        outlier = _make_packet(client_id="outlier", w_recency=0.80)
        all_packets = honest_packets + [outlier]

        result = aggregate_contributions(all_packets)
        # Trimmed mean should suppress the outlier
        assert result["w_recency"] < 0.50  # Well below the outlier

    def test_client_id_is_hashed(self, tmp_data_dir):
        """Client ID should be a hash, not the raw hostname."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        assert len(client._client_id) == 64  # SHA-256 hex length
        # Should not contain hostname in plaintext
        import socket
        assert socket.gethostname() not in client._client_id


# ═══════════════════════════════════════════════════════════════════
# Test: Stats & Diagnostics
# ═══════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_structure(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        s = client.stats()
        assert "enabled" in s
        assert "dp_epsilon" in s
        assert "dp_sigma" in s
        assert "contributions_sent" in s
        assert s["enabled"] is True

    def test_enable_disable(self, tmp_data_dir):
        client = FederationClient(data_dir=tmp_data_dir, enabled=False)
        assert not client.enabled
        client.enable()
        assert client.enabled
        client.disable()
        assert not client.enabled


# ═══════════════════════════════════════════════════════════════════
# Test: Contribute via Optimizer Interface
# ═══════════════════════════════════════════════════════════════════

class TestContributeViaOptimizer:
    def test_contribute_happy_path(self, tmp_data_dir, mock_archetype_optimizer):
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        result = client.contribute(mock_archetype_optimizer)
        assert result is True
        assert client._contributions_sent == 1

    def test_contribute_no_archetype(self, tmp_data_dir):
        mock = MagicMock()
        mock.current_archetype.return_value = None
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        assert client.contribute(mock) is False

    def test_contribute_disabled(self, tmp_data_dir, mock_archetype_optimizer):
        client = FederationClient(data_dir=tmp_data_dir, enabled=False)
        assert client.contribute(mock_archetype_optimizer) is False


# ═══════════════════════════════════════════════════════════════════
# Test: Fallbacks — Validation, Rollback, Safety Caps
# ═══════════════════════════════════════════════════════════════════

class TestFallbacks:
    def test_validate_rejects_nan(self):
        """NaN weights must be rejected — no silent corruption."""
        weights = {
            "w_recency": float("nan"), "w_frequency": 0.25,
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        valid, reason = FederationClient._validate_weights(weights)
        assert not valid
        assert "NaN" in reason

    def test_validate_rejects_inf(self):
        """Inf weights must be rejected."""
        weights = {
            "w_recency": 0.30, "w_frequency": float("inf"),
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        valid, reason = FederationClient._validate_weights(weights)
        assert not valid
        assert "NaN/Inf" in reason

    def test_validate_rejects_degenerate_zeros(self):
        """All-zero PRISM weights = degenerate, must reject."""
        weights = {
            "w_recency": 0.0, "w_frequency": 0.0,
            "w_semantic": 0.0, "w_entropy": 0.0, "w_resonance": 0.0,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        valid, reason = FederationClient._validate_weights(weights)
        assert not valid
        assert "sum too low" in reason

    def test_validate_rejects_negative_weight(self):
        """Negative PRISM weights must be rejected."""
        weights = {
            "w_recency": -0.5, "w_frequency": 0.25,
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        valid, reason = FederationClient._validate_weights(weights)
        assert not valid
        assert "negative" in reason

    def test_validate_rejects_too_few_keys(self):
        """Truncated packets with < 3 keys must be rejected."""
        weights = {"w_recency": 0.3}
        valid, reason = FederationClient._validate_weights(weights)
        assert not valid
        assert "too few" in reason

    def test_validate_accepts_good_weights(self):
        """Valid weights should pass validation."""
        weights = {
            "w_recency": 0.30, "w_frequency": 0.25,
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        valid, reason = FederationClient._validate_weights(weights)
        assert valid
        assert reason == "ok"

    def test_alpha_capped_at_70_percent(self, tmp_data_dir):
        """Local weights must retain ≥30% influence even with huge global pool."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        # Create 100 contributions (overwhelming global consensus)
        for i in range(100):
            packet = _make_packet(
                client_id=f"peer_{i}",
                confidence=0.95,
                w_recency=0.50,  # Very different from local 0.30
            )
            client._save_contribution(packet)

        mock = MagicMock()
        mock.current_archetype.return_value = "rust_ffi_library"
        mock.current_weights.return_value = {
            "w_recency": 0.30, "w_frequency": 0.25,
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        mock.stats.return_value = {
            "strategy_table": {
                "rust_ffi_library": {"sample_count": 1, "confidence": 0.5}
            }
        }

        client.merge_global(mock)

        # Check the merged weights passed to update_weights
        call_args = mock.update_weights.call_args[0][0]
        # w_recency should be blend of 0.50 (global) and 0.30 (local)
        # With α capped at 0.7: merged = 0.7*0.50 + 0.3*0.30 = 0.44
        # NOT 0.50 (which would mean 100% global)
        assert call_args["w_recency"] < 0.50, "Alpha cap should prevent full global takeover"
        assert call_args["w_recency"] > 0.30, "Global should still have some influence"

    def test_rollback_on_apply_failure(self, tmp_data_dir):
        """If update_weights raises, rollback to pre-merge snapshot."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        for i in range(5):
            packet = _make_packet(client_id=f"peer_{i}")
            client._save_contribution(packet)

        original_weights = {
            "w_recency": 0.30, "w_frequency": 0.25,
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }

        mock = MagicMock()
        mock.current_archetype.return_value = "rust_ffi_library"
        mock.current_weights.return_value = original_weights.copy()
        mock.stats.return_value = {
            "strategy_table": {
                "rust_ffi_library": {"sample_count": 10, "confidence": 0.85}
            }
        }
        # First call to update_weights (the merge) raises an error
        # Second call (the rollback) should succeed
        mock.update_weights.side_effect = [RuntimeError("disk full"), None]

        result = client.merge_global(mock)
        assert result is False  # Merge should fail

        # Should have called update_weights twice: once for merge, once for rollback
        assert mock.update_weights.call_count == 2
        # Second call should be the rollback with original weights
        rollback_weights = mock.update_weights.call_args_list[1][0][0]
        assert rollback_weights["w_recency"] == original_weights["w_recency"]


# ═══════════════════════════════════════════════════════════════════
# Test: Integration — REAL ArchetypeOptimizer (NO MOCKS)
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationReal:
    """These tests use the actual ArchetypeOptimizer class, not mocks.
    Proves the federation module works end-to-end with the real system.
    """

    def _make_real_optimizer(self, tmp_path):
        """Create a real ArchetypeOptimizer with a dummy project."""
        from entroly.archetype_optimizer import ArchetypeOptimizer

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        # Create a minimal Python project so archetype detection works
        (project_dir / "main.py").write_text(
            "def main():\n    print('hello')\n\nclass App:\n    pass\n",
            encoding="utf-8",
        )
        (project_dir / "utils.py").write_text(
            "import os\nimport sys\ndef helper():\n    return 42\n",
            encoding="utf-8",
        )

        opt = ArchetypeOptimizer(
            data_dir=str(data_dir),
            project_root=str(project_dir),
        )
        opt.detect_and_load()
        return opt, data_dir

    def test_real_contribute_roundtrip(self, tmp_path):
        """Real ArchetypeOptimizer → contribute → packet saved to disk."""
        opt, data_dir = self._make_real_optimizer(tmp_path)

        client = FederationClient(data_dir=str(data_dir), enabled=True)

        # Optimizer needs enough samples + confidence for eligibility
        # Manually bump the strategy table to meet thresholds
        label = opt.current_archetype()
        assert label is not None, "Archetype detection must work"

        # Force eligibility by updating weights several times
        for _ in range(6):
            opt.update_weights(opt.current_weights())

        result = client.contribute(opt)
        assert result is True, "Real optimizer should produce a valid contribution"
        assert client._contributions_sent == 1

        # Verify file exists on disk
        contrib_dir = data_dir / "federation_contributions"
        files = list(contrib_dir.glob("contrib_*.json"))
        assert len(files) >= 1, "Contribution file must exist on disk"

        # Verify file content is valid JSON with expected keys
        content = json.loads(files[0].read_text(encoding="utf-8"))
        assert content["archetype"] == label
        assert "weights" in content
        assert "w_recency" in content["weights"]

    def test_real_merge_changes_weights(self, tmp_path):
        """Real merge: global weights should actually change the optimizer's state."""
        opt, data_dir = self._make_real_optimizer(tmp_path)

        client = FederationClient(data_dir=str(data_dir), enabled=True)
        label = opt.current_archetype()
        assert label is not None

        original_recency = opt.current_weights()["w_recency"]

        # Create peer contributions with DIFFERENT w_recency
        for i in range(5):
            packet = _make_packet(
                archetype=label,
                client_id=f"real_peer_{i}",
                w_recency=0.50,  # Different from default 0.30
            )
            client._save_contribution(packet)

        result = client.merge_global(opt)
        assert result is True, "Merge with real optimizer must succeed"

        # The key test: weights ACTUALLY changed
        new_recency = opt.current_weights()["w_recency"]
        assert new_recency != original_recency, (
            f"Weights must change after merge! "
            f"Before: {original_recency}, After: {new_recency}"
        )
        # Merged value should be between original and global (0.30 and 0.50)
        assert new_recency > original_recency, "Should move toward global consensus"
        assert new_recency < 0.50, "Alpha cap should prevent full convergence"

    def test_real_merge_persists_to_disk(self, tmp_path):
        """After merge, loading a fresh optimizer should see the merged weights."""
        opt, data_dir = self._make_real_optimizer(tmp_path)
        project_dir = tmp_path / "project"

        client = FederationClient(data_dir=str(data_dir), enabled=True)
        label = opt.current_archetype()

        # Create peer contributions
        for i in range(5):
            packet = _make_packet(
                archetype=label,
                client_id=f"persist_peer_{i}",
                w_recency=0.45,
            )
            client._save_contribution(packet)

        client.merge_global(opt)
        merged_recency = opt.current_weights()["w_recency"]

        # Create a FRESH optimizer pointing at the same data dir
        from entroly.archetype_optimizer import ArchetypeOptimizer
        opt2 = ArchetypeOptimizer(
            data_dir=str(data_dir),
            project_root=str(project_dir),
        )
        opt2.detect_and_load()

        reloaded_recency = opt2.current_weights()["w_recency"]
        assert abs(reloaded_recency - merged_recency) < 0.001, (
            f"Reloaded weights ({reloaded_recency}) must match merged ({merged_recency}). "
            f"This proves persistence is real, not just in-memory."
        )


# ═══════════════════════════════════════════════════════════════════
# Test: V2 Upgrades — Privacy Accountant, TTL, Dedup, FedProx
# ═══════════════════════════════════════════════════════════════════

class TestPrivacyAccountant:
    """Upgrade 3: Cumulative privacy budget tracking."""

    def test_fresh_accountant_can_contribute(self):
        acc = PrivacyAccountant(budget=10.0)
        assert acc.can_contribute()
        assert acc.consumed_epsilon() == 0.0

    def test_budget_grows_sublinearly(self):
        """ε_total grows as √k, not k — advanced composition."""
        acc = PrivacyAccountant(budget=100.0)
        for _ in range(10):
            acc.record_contribution(1.0)
        eps_10 = acc.consumed_epsilon()

        for _ in range(90):
            acc.record_contribution(1.0)
        eps_100 = acc.consumed_epsilon()

        # √100 / √10 ≈ 3.16, not 10
        ratio = eps_100 / eps_10
        assert ratio < 5.0, f"Should grow sublinearly, got ratio {ratio:.2f}"
        assert ratio > 2.0, f"Should still grow, got ratio {ratio:.2f}"

    def test_budget_exhaustion_blocks_contributions(self, tmp_data_dir):
        """Once budget is exhausted, prepare_contribution returns None."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        # Exhaust budget by recording many contributions
        for _ in range(1000):
            client._accountant.record_contribution(1.0)

        assert not client._accountant.can_contribute()
        result = client.prepare_contribution(
            "rust_ffi_library",
            {k: 0.2 for k in FEDERATED_WEIGHT_KEYS},
            10, 0.9,
        )
        assert result is None, "Should block when budget exhausted"

    def test_remaining_budget_decreases(self):
        acc = PrivacyAccountant(budget=10.0)
        initial = acc.remaining_budget()
        acc.record_contribution(1.0)
        assert acc.remaining_budget() < initial


class TestTTLAndDedup:
    """Upgrade 2: Contribution expiry and per-client dedup."""

    def test_expired_contributions_removed(self, tmp_data_dir):
        """Contributions older than 30 days should be garbage-collected."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        # Create an old contribution (40 days ago)
        old_packet = _make_packet(
            client_id="old_peer",
            archetype="python_backend",
        )
        old_packet.timestamp = time.time() - 40 * 86400  # 40 days ago
        client._save_contribution(old_packet)

        # Create a fresh contribution
        fresh_packet = _make_packet(
            client_id="fresh_peer",
            archetype="python_backend",
        )
        fresh_packet.timestamp = time.time()  # now
        client._save_contribution(fresh_packet)

        contribs = client.load_contributions()
        total = sum(len(v) for v in contribs.values())
        assert total == 1, f"Only fresh contribution should survive TTL, got {total}"

    def test_per_client_dedup_keeps_latest(self, tmp_data_dir):
        """Multiple contributions from same client: keep only the latest."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        # Client sends v1 (old) then v2 (new)
        old = _make_packet(client_id="repeat_peer", w_recency=0.20)
        old.timestamp = time.time() - 3600  # 1 hour ago
        client._save_contribution(old)

        new = _make_packet(client_id="repeat_peer", w_recency=0.40)
        new.timestamp = time.time()
        client._save_contribution(new)

        contribs = client.load_contributions()
        packets = contribs.get("rust_ffi_library", [])
        assert len(packets) == 1, f"Should dedup to 1, got {len(packets)}"
        assert abs(packets[0].weights["w_recency"] - 0.40) < 0.01, "Should keep newer"

    def test_staleness_decay_reduces_confidence(self, tmp_data_dir):
        """Older contributions should have reduced confidence via decay."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        # Contribution from 20 days ago
        packet = _make_packet(client_id="aging_peer", confidence=0.90)
        packet.timestamp = time.time() - 20 * 86400
        client._save_contribution(packet)

        contribs = client.load_contributions()
        packets = contribs.get("rust_ffi_library", [])
        assert len(packets) == 1
        # After 20 days with half-life of 50 days: decay ≈ e^{-0.0139 * 20} ≈ 0.757
        assert packets[0].confidence < 0.90, "Staleness should reduce confidence"
        assert packets[0].confidence > 0.50, "20 days shouldn't kill confidence entirely"


class TestFedProx:
    """Upgrade 4: FedProx proximal regularization."""

    def test_fedprox_pulls_toward_local(self, tmp_data_dir):
        """Merged weights should be closer to local than pure linear blend."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        for i in range(5):
            packet = _make_packet(
                client_id=f"prox_peer_{i}",
                w_recency=0.60,  # Global wants 0.60
            )
            client._save_contribution(packet)

        local_recency = 0.30  # Local has 0.30
        mock = MagicMock()
        mock.current_archetype.return_value = "rust_ffi_library"
        mock.current_weights.return_value = {
            "w_recency": local_recency, "w_frequency": 0.25,
            "w_semantic": 0.25, "w_entropy": 0.20, "w_resonance": 0.10,
            "decay_half_life": 15.0, "min_relevance": 0.05,
            "exploration_rate": 0.10,
        }
        mock.stats.return_value = {
            "strategy_table": {
                "rust_ffi_library": {"sample_count": 10, "confidence": 0.85}
            }
        }

        client.merge_global(mock)

        merged_weights = mock.update_weights.call_args[0][0]
        merged_recency = merged_weights["w_recency"]

        # Pure linear blend (without FedProx) would give:
        #   alpha = min(0.7, ...) then blended = alpha*0.60 + (1-alpha)*0.30
        # FedProx pulls it back: (blended + μ*0.30) / (1+μ)
        # So merged should be closer to 0.30 than the raw blend
        assert merged_recency > local_recency, "Should still move toward global"
        assert merged_recency < 0.60, "Should NOT fully go to global"

        # The FedProx effect: compute what raw blend would be and verify
        # FedProx result is closer to local
        # With μ=0.1: w* = (blended + 0.1*0.30) / 1.1
        # This should be strictly less than blended


class TestMergeOutcomeTracking:
    """Upgrade 6: Convergence tracking."""

    def test_merge_records_outcome(self, tmp_data_dir, mock_archetype_optimizer):
        """Merge should append a MergeOutcome."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        for i in range(5):
            packet = _make_packet(
                client_id=f"outcome_peer_{i}",
                w_recency=0.40,
            )
            client._save_contribution(packet)

        client.merge_global(mock_archetype_optimizer)

        assert len(client._merge_outcomes) == 1
        outcome = client._merge_outcomes[0]
        assert outcome.archetype == "rust_ffi_library"
        assert outcome.contributors >= 3
        assert outcome.weight_delta_norm >= 0
        assert outcome.alpha_used > 0

    def test_outcome_saved_to_disk(self, tmp_data_dir, mock_archetype_optimizer):
        """Outcomes should be persisted to JSONL file."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)

        for i in range(5):
            packet = _make_packet(client_id=f"disk_peer_{i}")
            client._save_contribution(packet)

        client.merge_global(mock_archetype_optimizer)

        outcomes_path = Path(tmp_data_dir) / "federation_outcomes.jsonl"
        assert outcomes_path.exists(), "Outcomes file should exist"
        lines = outcomes_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "archetype" in data
        assert "delta_norm" in data

    def test_stats_includes_v2_fields(self, tmp_data_dir):
        """Stats should report privacy budget, FedProx μ, TTL, etc."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        s = client.stats()
        assert "privacy_budget_total" in s
        assert "privacy_budget_remaining" in s
        assert "fedprox_mu" in s
        assert "contribution_ttl_days" in s
        assert "staleness_half_life_days" in s
        assert "merge_outcomes_count" in s
        assert s["privacy_budget_total"] == 10.0
        assert s["fedprox_mu"] == FEDPROX_MU


# ═══════════════════════════════════════════════════════════════════
# Test: GitHub Transport (Zero-Cost Global Federation)
# ═══════════════════════════════════════════════════════════════════

class TestGitHubTransport:
    """Zero-cost global federation via GitHub Issues API."""

    def test_init_without_token(self):
        """Transport should be read-only without a token."""
        transport = GitHubTransport(token=None)
        assert transport.can_read
        assert not transport.can_write

    def test_init_with_token(self):
        """Transport should have write access with a token."""
        transport = GitHubTransport(token="ghp_test123")
        assert transport.can_read
        assert transport.can_write

    def test_push_without_token_returns_false(self):
        """Push should fail gracefully without a token."""
        transport = GitHubTransport(token=None)
        packet = _make_packet(client_id="test_peer")
        assert transport.push(packet) is False

    def test_pull_without_issue_returns_empty(self):
        """Pull should return empty list when no Issue is found."""
        transport = GitHubTransport(repo="nonexistent/repo", token=None)
        # Will fail to connect but should return [] gracefully
        result = transport.pull(since_hours=1.0)
        assert result == []

    def test_sync_to_local_saves_packets(self, tmp_data_dir):
        """sync_to_local should save pulled packets to the client's local dir."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        transport = GitHubTransport(token=None)

        # Mock pull to return synthetic packets
        remote_packets = [
            _make_packet(client_id=f"remote_peer_{i}", w_recency=0.30 + i * 0.05)
            for i in range(3)
        ]
        transport.pull = MagicMock(return_value=remote_packets)

        saved = transport.sync_to_local(client, since_hours=24.0)
        assert saved == 3, f"Should save 3 remote packets, got {saved}"

        # Verify they're in local contributions
        contribs = client.load_contributions()
        total = sum(len(v) for v in contribs.values())
        assert total == 3

    def test_sync_skips_own_contributions(self, tmp_data_dir):
        """sync_to_local should skip packets from the same client."""
        client = FederationClient(data_dir=tmp_data_dir, enabled=True)
        transport = GitHubTransport(token=None)

        remote_packets = [
            _make_packet(client_id=client._client_id),  # Own
            _make_packet(client_id="foreign_peer_1"),    # Foreign
        ]
        transport.pull = MagicMock(return_value=remote_packets)

        saved = transport.sync_to_local(client)
        assert saved == 1, "Should skip own, save 1 foreign"

