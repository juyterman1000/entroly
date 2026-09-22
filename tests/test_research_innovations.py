"""Tests for frontier context-control primitives.

Each test asserts the mechanism actually fires — a test that passes
when the mechanism is absent certifies a broken implementation.
"""
from __future__ import annotations

import pytest


# ── Entroly obligation-budget witness ─────────────────────────────────

class TestObligationBudgetWitness:
    def test_budget_sufficient_when_obligations_covered(self):
        from entroly.sufficiency import Candidate, build_obligation_budget_witness

        candidates = [
            Candidate("auth_handler.py", utility=0.9, cost=100, selected=True),
            Candidate("database_connector.py", utility=0.8, cost=80, selected=True),
        ]
        cert = build_obligation_budget_witness(
            candidates,
            obligations=["auth", "database"],
            budget=200,
        )
        assert cert.sufficient is True
        assert cert.deficit == 0
        assert cert.uncovered_obligations == ()

    def test_budget_insufficient_emits_deficit(self):
        from entroly.sufficiency import Candidate, build_obligation_budget_witness

        candidates = [
            Candidate("auth.py", utility=0.9, cost=100, selected=True),
            Candidate("database.py", utility=0.8, cost=150, selected=False),
            Candidate("cache.py", utility=0.7, cost=120, selected=False),
        ]
        cert = build_obligation_budget_witness(
            candidates,
            obligations=["auth", "database", "cache"],
            budget=100,
        )
        assert cert.sufficient is False
        assert cert.deficit > 0
        assert cert.minimum_cover_cost > cert.budget

    def test_uncovered_obligation_named(self):
        from entroly.sufficiency import Candidate, build_obligation_budget_witness

        candidates = [
            Candidate("auth.py", utility=0.9, cost=50, selected=True),
        ]
        cert = build_obligation_budget_witness(
            candidates,
            obligations=["auth", "quantum_encryption_protocol"],
            budget=500,
        )
        assert cert.sufficient is False
        assert "quantum_encryption_protocol" in cert.uncovered_obligations

    def test_empty_obligations_always_sufficient(self):
        from entroly.sufficiency import Candidate, build_obligation_budget_witness

        cert = build_obligation_budget_witness([], obligations=[], budget=0)
        assert cert.sufficient is True
        assert cert.deficit == 0

    def test_deficit_ratio_computed(self):
        from entroly.sufficiency import Candidate, build_obligation_budget_witness

        candidates = [
            Candidate("big.py", utility=0.5, cost=300, selected=False),
        ]
        cert = build_obligation_budget_witness(
            candidates,
            obligations=["big"],
            budget=100,
        )
        assert cert.sufficient is False
        assert cert.deficit_ratio == pytest.approx(2.0)
        d = cert.to_dict()
        assert "deficit_ratio" in d


# ── Entroly context-drift receipt ─────────────────────────────────────

class TestContextDriftReceipt:
    def test_update_requires_reverification_even_with_identical_prior_evidence(self):
        from entroly.relate.compression_residual import measure_context_drift

        fragment = "The rate limit is 100 requests per minute."
        retained = "The rate limit is 100 requests per minute."
        update = "Added logging to the dashboard."

        cert = measure_context_drift(fragment, retained, update)
        assert cert.stable is None
        assert cert.requires_reverification is True

    def test_unstable_omission_detected_after_conflicting_update(self):
        from entroly.relate.compression_residual import measure_context_drift

        fragment = "Deploy target is us-east-1 with fallback to eu-west-1."
        retained = "The service runs on AWS infrastructure."
        update = (
            "Migration: all deployments moving from us-east-1 to "
            "ap-southeast-1. Update fallback regions accordingly. "
            "The old target us-east-1 is being decommissioned. "
            "Fallback eu-west-1 is no longer available."
        )

        cert = measure_context_drift(fragment, retained, update)
        # The update introduced information that makes the omitted
        # fragment's region-specific details newly relevant.
        d = cert.to_dict()
        assert "pre_update_residual" in d
        assert "post_update_residual" in d
        assert cert.stable is None
        assert cert.requires_reverification is True
        assert cert.reason == "requires_reverification"

    def test_collapse_detected_for_indistinguishable_variants(self):
        from entroly.relate.compression_residual import measure_context_drift

        fragment = "Version A of the config"
        retained = "Version A of the config"
        update = "Version A of the config"

        cert = measure_context_drift(fragment, retained, update)
        d = cert.to_dict()
        assert d["stable"] is None
        assert d["requires_reverification"] is True

    def test_empty_fragment_is_stable(self):
        from entroly.relate.compression_residual import measure_context_drift

        cert = measure_context_drift("", "some retained text", "some update")
        assert cert.stable is True

    def test_per_compressor_reported(self):
        from entroly.relate.compression_residual import measure_context_drift

        cert = measure_context_drift("fragment", "retained", "update")
        d = cert.to_dict()
        assert "per_compressor" in d
        assert "zlib" in d["per_compressor"]


# ── Entroly evidence-boundary audit ───────────────────────────────────

class TestEvidenceBoundaryReceipt:
    def test_honest_when_all_entities_sourced(self):
        from entroly.relate.coverage_verification import audit_evidence_boundary

        answer = "The `AuthManager` class in auth/manager.py handles tokens."
        selected = [
            {"source_path": "auth/manager.py", "text": "class AuthManager: ..."},
        ]
        omitted = [
            {"source_path": "db/models.py", "text": "class UserModel: ..."},
        ]

        verdict = audit_evidence_boundary(answer, selected, omitted)
        assert verdict.honest is True
        assert verdict.entities_unsourced == 0

    def test_unsourced_detected_for_omitted_entity(self):
        from entroly.relate.coverage_verification import audit_evidence_boundary

        answer = (
            "The `AuthManager` handles auth, and `UserModel` "
            "stores user data in db/models.py."
        )
        selected = [
            {"source_path": "auth/manager.py", "text": "class AuthManager: ..."},
        ]
        omitted = [
            {"source_path": "db/models.py", "text": "class UserModel: ..."},
        ]

        verdict = audit_evidence_boundary(answer, selected, omitted)
        assert verdict.honest is False
        assert verdict.entities_unsourced >= 1
        unsourced_entities = [ua.entity for ua in verdict.unsourced]
        assert any("UserModel" in e for e in unsourced_entities)

    def test_no_entities_abstains(self):
        from entroly.relate.coverage_verification import audit_evidence_boundary

        verdict = audit_evidence_boundary("Everything is fine.", [], [])
        assert verdict.honest is None
        assert verdict.coverage_ratio == 0.0

    def test_to_dict_includes_all_fields(self):
        from entroly.relate.coverage_verification import audit_evidence_boundary

        verdict = audit_evidence_boundary(
            "The `Foo` class.",
            [{"source_path": "foo.py", "text": "class Foo: pass"}],
            [],
        )
        d = verdict.to_dict()
        assert "honest" in d
        assert "coverage_ratio" in d
        assert "unsourced" in d


# ── Entroly source-boundary propagation ───────────────────────────────

class TestSourceBoundaryPolicy:
    def test_source_policy_requires_existing_file_and_explicit_root(self, tmp_path):
        from entroly.vault import classify_source_boundary
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "sdk.py").write_text("pass")
        assert classify_source_boundary("src/sdk.py:12", tmp_path) == "trusted"
        assert classify_source_boundary("src/sdk.py") == "unknown"
        assert classify_source_boundary("src/missing.py", tmp_path) == "unknown"

    def test_untrusted_source_classified(self, tmp_path):
        from entroly.vault import classify_source_boundary
        assert classify_source_boundary("../outside.py", tmp_path) == "untrusted"

    def test_belief_trust_derivation(self, tmp_path):
        from entroly.vault import derive_belief_source_class
        (tmp_path / "sdk.py").write_text("pass")
        assert derive_belief_source_class(["sdk.py"], tmp_path) == "trusted"
        assert derive_belief_source_class([]) == "unknown"
        assert derive_belief_source_class(["sdk.py", "missing.py"], tmp_path) == "mixed"
        assert derive_belief_source_class(["sdk.py", "../outside.py"], tmp_path) == "mixed"

    def test_untrusted_belief_barred_from_selection(self):
        from entroly.vault import BeliefArtifact

        belief = BeliefArtifact(
            entity="malicious_function",
            trust_class="untrusted",
        )
        assert belief.selection_eligible is False

    def test_trusted_belief_eligible_for_selection(self):
        from entroly.vault import BeliefArtifact

        belief = BeliefArtifact(
            entity="safe_function",
            trust_class="trusted",
        )
        assert belief.selection_eligible is True

    def test_trust_class_in_markdown(self):
        from entroly.vault import BeliefArtifact

        belief = BeliefArtifact(entity="test", trust_class="untrusted")
        md = belief.to_markdown()
        assert "trust_class: untrusted" in md

    def test_unknown_trust_omitted_from_markdown(self):
        from entroly.vault import BeliefArtifact

        belief = BeliefArtifact(entity="test", trust_class="unknown")
        md = belief.to_markdown()
        assert "trust_class" not in md


# ── Entroly benchmark binding ─────────────────────────────────────────

class TestBenchmarkBinding:
    def test_untrusted_benchmark_blocks_promotion(self):
        """A skill with untrusted benchmark origin cannot be promoted."""
        from unittest.mock import MagicMock, patch

        from entroly.skill_engine import SkillEngine

        mock_vault = MagicMock()
        mock_vault.config.path.return_value = "/tmp/vault"
        mock_benchmark = MagicMock()

        engine = SkillEngine.__new__(SkillEngine)
        engine._vault = mock_vault
        engine._benchmark = mock_benchmark
        engine.PROMOTION_THRESHOLD = 0.7
        engine.PRUNE_THRESHOLD = 0.3

        mock_spec = MagicMock()
        mock_spec.metrics = {
            "fitness_score": 0.95,
            "benchmark_runs": 5,
            "benchmark_contract_version": 1,
            "benchmark_origin": "external:untrusted-repo",
            "benchmark_evidence_class": "untrusted",
        }
        mock_spec.status = "testing"
        mock_spec.skill_id = "test-skill"

        with patch.object(engine, "_load_skill", return_value=mock_spec), \
             patch.object(engine, "_valid_skill_id", return_value=True), \
             patch.object(engine, "_skill_dir", return_value=MagicMock()), \
             patch.object(engine, "_update_registry"):
            # Mock the SKILL.md file
            skill_dir = engine._skill_dir("test-skill")
            skill_md = skill_dir.__truediv__.return_value
            skill_md.exists.return_value = False

            result = engine.promote_or_prune("test-skill")

        assert result["status"] == "kept"
        assert result.get("binding_blocked") is True
        assert "untrusted" in result.get("binding_reason", "")

    def test_missing_origin_blocks_promotion(self):
        """A skill with no benchmark_origin cannot be promoted."""
        from unittest.mock import MagicMock, patch

        from entroly.skill_engine import SkillEngine

        engine = SkillEngine.__new__(SkillEngine)
        engine._vault = MagicMock()
        engine._benchmark = MagicMock()
        engine.PROMOTION_THRESHOLD = 0.7
        engine.PRUNE_THRESHOLD = 0.3

        mock_spec = MagicMock()
        mock_spec.metrics = {
            "fitness_score": 0.95,
            "benchmark_runs": 3,
            "benchmark_contract_version": 1,
        }
        mock_spec.status = "testing"

        with patch.object(engine, "_load_skill", return_value=mock_spec), \
             patch.object(engine, "_valid_skill_id", return_value=True), \
             patch.object(engine, "_skill_dir", return_value=MagicMock()), \
             patch.object(engine, "_update_registry"):
            skill_dir = engine._skill_dir("test-skill")
            skill_md = skill_dir.__truediv__.return_value
            skill_md.exists.return_value = False

            result = engine.promote_or_prune("test-skill")

        assert result["status"] == "kept"
        assert result.get("binding_blocked") is True
