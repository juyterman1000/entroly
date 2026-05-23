"""
Property + invariant tests for the continuous-risk WITNESS stack.

Coverage targets:
  - witness_features: feature ranges, signed semantics, edge cases
  - witness_risk_model: boundedness, monotonicity, hard-gate dominance,
                        SGD-step convergence, idempotence
  - witness_atomic: boundedness, monotonicity, salience scaling
  - witness_calibration: threshold ordering, action map, conformal
                          quantile correctness, calibration round-trip
"""

from __future__ import annotations

import math
import random

import pytest

from entroly.witness_features import (
    ClaimFeatures,
    extract_features,
    feat_entity_precision,
    feat_idf_lex_overlap,
    feat_negation_polarity,
    feat_number_consistency,
    feat_qa_alignment,
    feat_quote_support,
    feat_reverse_entail,
    feat_forward_entail,
    named_entities,
)
from entroly.witness_risk_model import (
    DEFAULT_WEIGHTS,
    RiskModel,
    label_from_features_and_risk,
)
from entroly.witness_atomic import AtomClaim, aggregate, decompose
from entroly.witness_calibration import (
    Action,
    CalibrationSample,
    CalibrationStore,
    ThresholdSet,
    calibrate,
    conformal_quantile,
    default_thresholds,
)
from entroly.witness_training import WitnessTrainingStore, label_to_y


# ═══════════════════════════════════════════════════════════════════
# witness_features
# ═══════════════════════════════════════════════════════════════════


class TestFeatureRanges:
    """All feature outputs must respect their declared codomains."""

    def test_entity_precision_in_signed_unit(self):
        ctx = "Albert Einstein was a physicist."
        # Both entities present → +1
        assert feat_entity_precision("Albert Einstein invented things.", ctx) == 1.0
        # No entities → +1 (neutral)
        assert feat_entity_precision("things were done.", ctx) == 0.0
        # All entities missing → -1
        val = feat_entity_precision("Niels Bohr did stuff.", ctx)
        assert -1.0 <= val <= 1.0
        assert val < 0

    def test_number_consistency_in_signed_unit(self):
        ctx = "Einstein was born in 1879."
        assert feat_number_consistency("Einstein was born in 1879.", ctx) == 1.0
        assert feat_number_consistency("Einstein wrote 3 papers.", ctx) == -1.0
        assert feat_number_consistency("Einstein wrote papers.", ctx) == 0.0
        assert feat_number_consistency("Einstein was born in 1985.", ctx) == -1.0

    def test_idf_lex_overlap_in_unit_interval(self):
        ctx = "the cat sat on the mat"
        for claim in ["cat on mat", "dog on log", "", "different words entirely"]:
            v = feat_idf_lex_overlap(claim, ctx)
            assert 0.0 <= v <= 1.0

    def test_quote_support_in_unit_interval(self):
        ctx = "The quick brown fox jumps over the lazy dog"
        # Exact substring → 1.0
        assert feat_quote_support("brown fox jumps", ctx) > 0.9
        # Empty claim
        assert feat_quote_support("", ctx) == 0.0

    def test_negation_polarity_signed(self):
        # No shared content → neutral
        assert feat_negation_polarity("apples are red", "trains run on time") == 0.0
        # Same polarity → +1
        assert feat_negation_polarity("Einstein was a physicist", "Einstein was a physicist") == 1.0
        # Polarity flipped: claim negates what ctx affirms
        ctx = "Einstein was a physicist."
        v = feat_negation_polarity("Einstein was not a physicist.", ctx)
        assert v == -1.0


class TestNamedEntities:
    """Entity extraction must keep sentence-initial entities (the bug-class)."""

    def test_sentence_initial_entity_kept(self):
        ents = named_entities("Bohr won the Nobel Prize.")
        assert "bohr" in ents

    def test_common_sentence_starter_dropped(self):
        ents = named_entities("The cat sat on the mat.")
        # "The" must not be treated as an entity
        assert "the" not in ents

    def test_multiword_entity(self):
        ents = named_entities("Albert Einstein worked at Princeton.")
        # Either combined or component — verify at least one form is present
        assert any("einstein" in e for e in ents)


class TestQAAlignment:
    """φ₉ — signed feature with conservative active-mismatch trigger."""

    K = (
        "Albert Einstein developed general relativity in 1915. "
        "He won the Nobel Prize in 1921 for the photoelectric effect."
    )
    Q = "What year did Einstein develop general relativity?"

    def test_correct_answer_positive(self):
        assert feat_qa_alignment("1915", self.K, self.Q) > 0.5

    def test_misplaced_year_negative(self):
        # "1921" is in K but for a different fact
        v = feat_qa_alignment("1921", self.K, self.Q)
        assert v < 0, f"misplaced year should be negative, got {v}"

    def test_off_topic_answer_negative(self):
        v = feat_qa_alignment("Tuesday", self.K, self.Q)
        # Word answer, raw_score below 0.2, AND best sentence has ≥2 Q-keywords
        assert v < 0

    def test_no_question_neutral(self):
        assert feat_qa_alignment("1915", self.K, None) == 0.0
        assert feat_qa_alignment("1915", self.K, "") == 0.0

    def test_empty_answer_neutral(self):
        assert feat_qa_alignment("", self.K, self.Q) == 0.0

    # ── Extractive-binding margin gate (recombination / wrong-option) ──
    # The dominant HaluEval-QA miss class: the answer reuses the
    # knowledge's own vocabulary (so every bag-of-words feature is
    # mildly supportive) but recombines it into a false proposition or
    # picks the wrong option. Empirically, genuine QA answers are
    # near-verbatim contiguous spans of their evidence sentence; these
    # are not. The gate must fire on them WITHOUT demoting genuine
    # extractive answers (retention guard).
    K2 = (
        "Jonathan Stark won two Grand Slam doubles titles during his career. "
        "Henri Leconte reached the 1988 French Open singles final."
    )
    Q2 = "Which player won more Grand Slam titles, Henri Leconte or Jonathan Stark?"

    def test_wrong_option_recombination_negative(self):
        # Wrong option; tokens all present in K but no contiguous answer
        # span at the question's evidence sentence (the Stark sentence).
        v = feat_qa_alignment(
            "Henri Leconte won more Grand Slam titles.", self.K2, self.Q2
        )
        assert v < 0, f"wrong-option recombination should gate negative, got {v}"

    def test_genuine_extractive_answer_stays_positive(self):
        # The true answer IS a contiguous span of its evidence sentence —
        # bind_q high — so the gate must NOT fire (retention preserved).
        v = feat_qa_alignment(
            "Jonathan Stark won two Grand Slam doubles titles.",
            self.K2, self.Q2,
        )
        assert v > 0, f"genuine extractive answer must stay positive, got {v}"

    def test_binding_gate_needs_confident_locus(self):
        # No question keywords land a confident locus → gate must not
        # fire spuriously; feature stays non-negative.
        v = feat_qa_alignment(
            "Henri Leconte won more Grand Slam titles.",
            self.K2, "Tell me something interesting?",
        )
        assert v >= 0.0

    # ── Question-residual payload gate (wrong-slot factoid) ──────────
    # The answer parrots the question's frame and appends a filler. Only
    # the answer's question-residual carries truth value. Single-hop
    # guarantee: if the asserted filler is entirely absent from the
    # question's evidence sentence, gate; if present there, don't.
    K3 = (
        "Warren Bryant was the CEO of Longs Drugs, a chain "
        "headquartered in California."
    )
    Q3 = "In which state is the drugstore chain whose CEO was Warren Bryant?"

    def test_wrong_slot_filler_negative(self):
        # "Utah" is the asserted novelty and is absent from the evidence
        # sentence — wrong-slot hallucination.
        v = feat_qa_alignment(
            "The drugstore chain whose CEO was Warren Bryant is in Utah.",
            self.K3, self.Q3,
        )
        assert v < 0, f"wrong-slot filler should gate negative, got {v}"

    def test_correct_extractive_answer_retained(self):
        # The gate's actual guarantee: a correct answer phrased
        # *extractively* (mirrors the evidence sentence — which is how
        # HaluEval-QA safe answers empirically look: bind_q ≈ 0.94) is
        # retained. This is why measured Retention stays flat at 0.738.
        v = feat_qa_alignment(
            "Warren Bryant was the CEO of a chain headquartered in California.",
            self.K3, self.Q3,
        )
        assert v > 0, f"correct extractive answer must stay positive, got {v}"

    @pytest.mark.xfail(
        reason="KNOWN LIMITATION: a correct but heavily reordered / "
        "non-extractive paraphrase can be false-gated (bind_q low). "
        "Out-of-distribution for HaluEval-QA (safe answers are "
        "extractive), so empirical Retention is unaffected (0.738 flat). "
        "Documented in benchmarks/results/witness_v3_report.md. "
        "Lifting this needs a paraphrase-robust (e.g. NLI) verifier.",
        strict=True,
    )
    def test_reordered_paraphrase_limitation(self):
        v = feat_qa_alignment(
            "The drugstore chain whose CEO was Warren Bryant is in California.",
            self.K3, self.Q3,
        )
        assert v > 0  # currently fails by design — encodes the gap


# ═══════════════════════════════════════════════════════════════════
# witness_risk_model
# ═══════════════════════════════════════════════════════════════════


class TestRiskModelMath:
    """Property tests for ρ(c, C) = predict(features)."""

    def _features(self, **overrides) -> ClaimFeatures:
        """A feature vector defaulting to 'fully supportive' (ρ→0)."""
        base = dict(
            entity_precision=1.0,
            number_consistency=1.0,
            idf_lex_overlap=1.0,
            quote_support=1.0,
            forward_entail=1.0,
            reverse_entail=1.0,
            negation_polarity=1.0,
            adequacy=1.0,
            qa_alignment=1.0,
        )
        base.update(overrides)
        return ClaimFeatures(**base)

    def test_output_bounded(self):
        m = RiskModel()
        for _ in range(100):
            # Random valid feature vectors
            feats = ClaimFeatures(
                entity_precision=random.uniform(-1, 1),
                number_consistency=random.uniform(-1, 1),
                idf_lex_overlap=random.uniform(0, 1),
                quote_support=random.uniform(0, 1),
                forward_entail=random.uniform(0, 1),
                reverse_entail=random.uniform(0, 1),
                negation_polarity=random.uniform(-1, 1),
                adequacy=random.uniform(0, 1),
                qa_alignment=random.uniform(-1, 1),
            )
            rho = m.predict(feats)
            assert 0.0 <= rho <= 1.0

    def test_full_support_low_risk(self):
        m = RiskModel()
        rho = m.predict(self._features())
        assert rho < 0.05, f"full-support should be ρ→0, got {rho}"

    def test_number_contradiction_dominates(self):
        """Single hard-contradiction gate must drive ρ → 1 regardless of
        any number of supportive features."""
        m = RiskModel()
        rho = m.predict(self._features(number_consistency=-1.0))
        assert rho > 0.99, f"hard contradiction should push ρ→1, got {rho}"

    def test_entity_contradiction_dominates(self):
        m = RiskModel()
        rho = m.predict(self._features(entity_precision=-1.0))
        assert rho > 0.99

    def test_negation_flip_dominates(self):
        m = RiskModel()
        rho = m.predict(self._features(negation_polarity=-1.0))
        assert rho > 0.99

    def test_monotonic_in_supportive_features(self):
        """Reducing any single supportive feature must not DECREASE ρ."""
        m = RiskModel()
        base = m.predict(self._features())
        for feature_name in ["entity_precision", "idf_lex_overlap",
                             "quote_support", "forward_entail",
                             "reverse_entail", "adequacy", "qa_alignment"]:
            for val in (0.5, 0.0):
                modified = m.predict(self._features(**{feature_name: val}))
                assert modified >= base - 1e-9, (
                    f"reducing {feature_name}: {feature_name}={val} "
                    f"gave ρ={modified} but base ρ={base}"
                )

    def test_sgd_update_moves_toward_label(self):
        """One SGD step on a labeled example should move ρ toward y."""
        m = RiskModel()
        # Start with a feature vector that has high risk (multiple negatives)
        f = self._features(number_consistency=-1.0)
        pre = m.predict(f)
        # Tell the model "actually this is safe"
        post = m.update(f, y=0)
        assert post <= pre + 1e-9, "y=0 should not raise risk"
        # Inverse direction with full support
        f2 = self._features()
        pre2 = m.predict(f2)
        post2 = m.update(f2, y=1)
        assert post2 >= pre2 - 1e-9, "y=1 should not lower risk"

    def test_update_rejects_bad_label(self):
        m = RiskModel()
        with pytest.raises(ValueError):
            m.update(self._features(), y=2)

    def test_explain_decomposes_z(self):
        m = RiskModel()
        f = self._features()
        explanation = m.explain(f)
        # All feature names present
        for n in ClaimFeatures.feature_names():
            assert n in explanation
        assert "rho" in explanation
        assert "z" in explanation
        # Math invariant: rho == sigmoid of z when no hard gates fire
        # (all features non-negative here, so soft-only path)
        assert abs(explanation["rho"] - m.predict(f)) < 1e-9

    def test_weights_bounded_after_many_updates(self):
        """Online SGD must not let weights run away."""
        m = RiskModel(learning_rate=0.5)
        for _ in range(500):
            y = random.choice([0, 1])
            f = ClaimFeatures(
                entity_precision=random.uniform(-1, 1),
                number_consistency=random.uniform(-1, 1),
                idf_lex_overlap=random.uniform(0, 1),
                quote_support=random.uniform(0, 1),
                forward_entail=random.uniform(0, 1),
                reverse_entail=random.uniform(0, 1),
                negation_polarity=random.uniform(-1, 1),
                adequacy=random.uniform(0, 1),
                qa_alignment=random.uniform(-1, 1),
            )
            m.update(f, y=y)
        for w in m.weights.values():
            assert -10.0 <= w <= 10.0
        assert -10.0 <= m.bias <= 10.0


class TestLabelDerivation:
    def test_contradicted_on_signed_negative(self):
        f = ClaimFeatures(
            entity_precision=1.0, number_consistency=-1.0, idf_lex_overlap=1.0,
            quote_support=1.0, forward_entail=1.0, reverse_entail=1.0,
            negation_polarity=1.0, adequacy=1.0, qa_alignment=1.0,
        )
        # Even at low risk, signed-negative input must yield "contradicted"
        assert label_from_features_and_risk(f, risk=0.05) == "contradicted"

    def test_grounded_when_supported(self):
        f = ClaimFeatures(
            entity_precision=1.0, number_consistency=1.0, idf_lex_overlap=1.0,
            quote_support=1.0, forward_entail=1.0, reverse_entail=1.0,
            negation_polarity=1.0, adequacy=1.0, qa_alignment=1.0,
        )
        assert label_from_features_and_risk(f, risk=0.05) == "grounded"


# ═══════════════════════════════════════════════════════════════════
# witness_atomic
# ═══════════════════════════════════════════════════════════════════


class TestAtomic:
    def test_decompose_non_empty(self):
        out = decompose(
            "Einstein developed relativity in 1915. He won the Nobel Prize in 1921."
        )
        assert len(out) >= 2
        for atom in out:
            assert atom.salience >= 0
            assert len(atom.text) > 0

    def test_decompose_empty_input(self):
        assert decompose("") == []
        assert decompose("   \n\n  ") == []

    def test_decompose_salience_sums_to_n(self):
        out = decompose(
            "X is alpha. Y is beta. Z is gamma."
        )
        total = sum(a.salience for a in out)
        assert math.isclose(total, len(out), rel_tol=1e-6)

    def test_aggregate_bounded(self):
        for _ in range(50):
            atoms = decompose("Alpha is true. Beta is true. Gamma is true.")
            risks = [(a, random.random()) for a in atoms]
            agg = aggregate(risks)
            assert 0.0 <= agg <= 1.0

    def test_aggregate_empty(self):
        assert aggregate([]) == 0.0

    def test_aggregate_monotone_in_max(self):
        """Adding a higher-risk atom must not decrease aggregate."""
        a = AtomClaim(text="x"*30, salience=1.0)
        b = AtomClaim(text="y"*30, salience=1.0)
        low = aggregate([(a, 0.1), (b, 0.1)])
        high = aggregate([(a, 0.1), (b, 0.9)])
        assert high >= low

    def test_aggregate_single_atom_idempotent(self):
        """Single atom at salience 1 must give ρ_agg = ρ."""
        a = AtomClaim(text="x"*30, salience=1.0)
        for rho in (0.0, 0.3, 0.7, 1.0 - 1e-9):
            assert math.isclose(aggregate([(a, rho)]), rho, abs_tol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# witness_calibration
# ═══════════════════════════════════════════════════════════════════


class TestThresholdSet:
    def test_action_ordering(self):
        ts = ThresholdSet(tau_pass=0.3, tau_hedge=0.5, tau_warn=0.8)
        assert ts.action(0.0) == Action.PASS
        assert ts.action(0.29) == Action.PASS
        assert ts.action(0.30) == Action.HEDGE
        assert ts.action(0.50) == Action.WARN
        assert ts.action(0.80) == Action.SUPPRESS
        assert ts.action(1.0) == Action.SUPPRESS

    def test_threshold_set_auto_sorts(self):
        ts = ThresholdSet(tau_pass=0.8, tau_hedge=0.3, tau_warn=0.5)
        assert ts.tau_pass <= ts.tau_hedge <= ts.tau_warn

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            ThresholdSet(tau_pass=0.3, tau_hedge=0.5, tau_warn=1.5)

    def test_default_thresholds_per_profile(self):
        for profile in ("benchmark_qa", "qa", "rag", "code", "summary",
                        "dialogue", "chat", "auto"):
            ts = default_thresholds(profile)
            assert 0 <= ts.tau_pass <= ts.tau_hedge <= ts.tau_warn <= 1


class TestConformalQuantile:
    def test_empty_returns_one(self):
        assert conformal_quantile([], alpha=0.5) == 1.0

    def test_finite_sample_coverage_property(self):
        """For n iid samples, the conformal quantile τ at level α
        satisfies P[future ≤ τ] ≥ 1 − α by exchangeability."""
        rng = random.Random(42)
        n = 100
        alpha = 0.1
        # Generate 100 calibration scores from U[0,1]
        cal = [rng.random() for _ in range(n)]
        tau = conformal_quantile(cal, alpha)
        # Empirical coverage on the calibration set itself must be ≥ 1 - α
        coverage = sum(1 for x in cal if x <= tau) / n
        assert coverage >= 1 - alpha - 1e-6, (
            f"empirical coverage {coverage} < 1-α={1-alpha}"
        )

    def test_monotone_in_alpha(self):
        """Higher α → lower τ (less strict)."""
        rng = random.Random(1)
        cal = [rng.random() for _ in range(50)]
        t_strict = conformal_quantile(cal, alpha=0.1)
        t_loose = conformal_quantile(cal, alpha=0.5)
        assert t_loose <= t_strict


class TestCalibrate:
    def test_calibrate_basic(self):
        samples = (
            [CalibrationSample(rho=0.05, y=0) for _ in range(50)]
            + [CalibrationSample(rho=0.80, y=1) for _ in range(40)]
            + [CalibrationSample(rho=0.55, y=1) for _ in range(10)]
        )
        cr = calibrate(samples, profile="qa",
                       alpha_pass=0.80, alpha_hedge=0.50, alpha_warn=0.20)
        assert cr.tau_pass <= cr.tau_hedge <= cr.tau_warn
        assert cr.n_samples == 100
        assert cr.n_hallucinated == 50
        assert len(cr.dataset_crc32) == 8

    def test_calibrate_rejects_bad_alphas(self):
        samples = [CalibrationSample(rho=0.5, y=1) for _ in range(10)]
        with pytest.raises(ValueError):
            calibrate(samples, alpha_pass=0.2, alpha_hedge=0.5, alpha_warn=0.8)


class TestCalibrationStore:
    def test_persistence_roundtrip(self, tmp_path):
        path = tmp_path / "cal.json"
        store = CalibrationStore(path)
        samples = (
            [CalibrationSample(rho=0.05, y=0) for _ in range(40)]
            + [CalibrationSample(rho=0.80, y=1) for _ in range(40)]
        )
        result = calibrate(samples, profile="benchmark_qa")
        store.update(result)

        # Open a fresh store from the same path
        store2 = CalibrationStore(path)
        ts = store2.get_thresholds("benchmark_qa")
        assert math.isclose(ts.tau_pass, result.tau_pass)
        assert math.isclose(ts.tau_warn, result.tau_warn)

    def test_missing_profile_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "cal.json"
        store = CalibrationStore(path)
        ts = store.get_thresholds("benchmark_qa")
        defaults = default_thresholds("benchmark_qa")
        assert ts.tau_pass == defaults.tau_pass
        assert ts.tau_warn == defaults.tau_warn


# ═══════════════════════════════════════════════════════════════════
# end-to-end extract_features
# ═══════════════════════════════════════════════════════════════════


class TestExtractFeaturesEndToEnd:
    def test_qa_grounded(self):
        K = "Einstein developed relativity in 1915 at Princeton."
        Q = "When did Einstein develop relativity?"
        f = extract_features("1915", K, adequacy=0.85, question=Q)
        m = RiskModel()
        rho = m.predict(f)
        assert rho < 0.1

    def test_qa_misplaced_year_caught(self):
        K = ("Einstein developed relativity in 1915. "
             "He won the Nobel Prize in 1921 for the photoelectric effect.")
        Q = "When did Einstein develop relativity?"
        f = extract_features("1921", K, adequacy=0.85, question=Q)
        m = RiskModel()
        rho = m.predict(f)
        assert rho > 0.9, f"misplaced-year case missed; ρ={rho}"

    def test_no_question_neutral_qa_alignment(self):
        K = "Einstein was a physicist."
        f = extract_features("Bohr was a physicist.", K, adequacy=0.85)
        # No question -> qa_alignment = 0.0 (neutral/no signal)
        assert f.qa_alignment == 0.0


class TestOnlineTraining:
    def test_label_mapping(self):
        assert label_to_y("ci_passed") == 0
        assert label_to_y("test_failed") == 1
        with pytest.raises(ValueError):
            label_to_y("agent_said_probably")

    def test_training_record_persists_model(self, tmp_path):
        path = tmp_path / "risk_model.json"
        store = WitnessTrainingStore(path)
        record = store.record(
            context="Einstein developed relativity in 1915.",
            output="Einstein developed relativity in 1915.",
            label="safe",
            profile="benchmark_qa",
            source="pytest",
        )
        assert record.n_claims >= 1
        assert path.exists()
        store2 = WitnessTrainingStore(path)
        assert store2.model.updates >= record.model_updates

    def test_ravs_event_bridge_requires_honest_signal(self, tmp_path):
        store = WitnessTrainingStore(tmp_path / "risk_model.json")
        assert store.record_ravs_outcome({
            "final_label": "unknown",
            "context": "ctx",
            "output": "out",
        }) is None
        record = store.record_ravs_outcome({
            "final_label": "ci_failed",
            "context": "The module contains fetch_user().",
            "output": "Call delete_user().",
            "profile": "code",
        })
        assert record is not None
        assert record.y == 1
