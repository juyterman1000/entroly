"""Calibration must never cost the user an answer, and must stop paying.

The routing certificate could not bootstrap: it needed observations of a cheap
model, which needed the cheap model to have been used, which needed the
certificate. Shadow calibration breaks that by separating who answers the user
from what is learned -- the flagship answers, and the cheap model is called
only to be compared.

That trade is acceptable only while three things hold, so each is pinned here:
the user is never served a shadow result, sampling terminates once enough
observations exist, and no failure in any of it reaches the request.
"""

from __future__ import annotations

import random
from typing import Any

from entroly.ravs.shadow_calibration import (
    ShadowCalibrator,
    diverged,
    divergence_score,
    should_sample,
)


class TestDivergenceLabel:
    def test_identical_answers_do_not_diverge(self):
        text = "The auth module hashes passwords with a per-user salt."
        assert divergence_score(text, text) == 0.0
        assert diverged(text, text) is False

    def test_unrelated_answers_diverge(self):
        assert diverged(
            "The auth module hashes passwords with a per-user salt.",
            "Kubernetes schedules pods onto nodes using taints.",
        ) is True

    def test_two_empty_answers_agree(self):
        """Both produced nothing; a shared failure is not a routing fault."""
        assert divergence_score("", "") == 0.0
        assert diverged("", "") is False

    def test_one_empty_answer_diverges(self):
        assert divergence_score("a real answer here", "") == 1.0

    def test_the_score_is_bounded(self):
        for a, b in [("x", "y"), ("", "z"), ("a b c", "a b c d"), ("", "")]:
            assert 0.0 <= divergence_score(a, b) <= 1.0

    def test_wording_differences_are_tolerated(self):
        """Paraphrase must not read as divergence, or nothing ever routes."""
        assert diverged(
            "The function hashes the password using a salt value",
            "The function hashes the password using a salt",
        ) is False

    def test_the_label_is_symmetric(self):
        a, b = "alpha beta gamma", "beta gamma delta"
        assert divergence_score(a, b) == divergence_score(b, a)


class TestSamplingPolicy:
    def test_high_risk_requests_are_never_sampled(self):
        """Nothing is learned by paying for a request that will never route."""
        decision = should_sample(
            risk_level="high", observations=0, samples_needed=49, epsilon=1.0)
        assert decision.sample is False
        assert "never routed" in decision.reason

    def test_sampling_stops_once_the_certificate_permits_routing(self):
        """A bootstrap phase with an end, not a standing tax."""
        decision = should_sample(
            risk_level="low", observations=49, samples_needed=49,
            permits_routing=True, epsilon=1.0)
        assert decision.sample is False
        assert "certified and routing" in decision.reason

    def test_sampling_continues_when_the_minimum_is_met_but_nothing_certifies(self):
        """Finding 2: stopping at samples_needed deadlocked the feature.

        samples_needed is only the smallest n at which the route-nothing
        threshold becomes certifiable. Landing there with a few divergent
        observations left a valid certificate permitting nothing and no way to
        ever gather the evidence that would change it.
        """
        decision = should_sample(
            risk_level="low", observations=49, samples_needed=49,
            permits_routing=False, epsilon=1.0)
        assert decision.sample is True, (
            "routing would be permanently disabled with no recovery edge"
        )

    def test_sampling_gives_up_eventually(self):
        """A cheap model that never agrees must not be paid for forever."""
        decision = should_sample(
            risk_level="low", observations=5000, samples_needed=49,
            permits_routing=False, max_observations=5000, epsilon=1.0)
        assert decision.sample is False
        assert "exhausted" in decision.reason

    def test_unclassified_risk_is_refused_not_assumed_low(self):
        """Finding 5: the gate tested truthiness first and so failed open.

        Several RoutingDecision early-return paths leave risk_level as "",
        which skipped the check entirely and sampled traffic that can never be
        routed -- polluting the calibration set with a population the bound is
        not about.
        """
        for unknown in ("", "   ", "unknown"):
            decision = should_sample(
                risk_level=unknown, observations=0, samples_needed=49,
                epsilon=1.0)
            assert decision.sample is False, f"fails open on {unknown!r}"
            assert "never routed" in decision.reason

    def test_low_risk_and_uncalibrated_is_sampled(self):
        assert should_sample(
            risk_level="low", observations=0, samples_needed=49,
            epsilon=1.0).sample is True

    def test_a_zero_rate_disables_sampling(self):
        assert should_sample(
            risk_level="low", observations=0, samples_needed=49,
            epsilon=0.0).sample is False

    def test_the_rate_is_respected(self):
        """Spend is bounded by epsilon rather than left to chance."""
        rng = random.Random(11)
        sampled = sum(
            should_sample(risk_level="low", observations=0, samples_needed=49,
                          epsilon=0.10, rng=rng).sample
            for _ in range(4000)
        )
        assert 250 <= sampled <= 550, f"rate drifted from 10 percent: {sampled}/4000"

    def test_it_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_RAVS_SHADOW", "0")
        assert should_sample(
            risk_level="low", observations=0, samples_needed=49,
            epsilon=1.0).sample is False


class TestCalibratorNeverCostsTheRequest:
    def test_an_observation_is_recorded(self):
        recorded: list[tuple[float, bool]] = []
        ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "a completely unrelated reply about pods",
            record=lambda c, d: recorded.append((c, d)),
        ).observe(prompt="q", cheap_model="haiku",
                  flagship_output="salted password hashing", confidence=0.9)

        assert recorded == [(0.9, True)]

    def test_agreement_is_recorded_as_no_divergence(self):
        recorded: list[tuple[float, bool]] = []
        ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "salted password hashing",
            record=lambda c, d: recorded.append((c, d)),
        ).observe(prompt="q", cheap_model="haiku",
                  flagship_output="salted password hashing", confidence=0.8)

        assert recorded == [(0.8, False)]

    def test_a_failing_cheap_call_records_nothing_and_does_not_raise(self):
        recorded: list[Any] = []

        def explode(_m, _p):
            raise RuntimeError("provider down")

        ShadowCalibrator(
            invoke_cheap=explode, record=lambda c, d: recorded.append((c, d)),
        ).observe(prompt="q", cheap_model="haiku", flagship_output="x",
                  confidence=0.9)

        assert recorded == [], "a failed shadow call must not become an observation"

    def test_an_empty_cheap_response_is_not_an_observation(self):
        """No answer is not the same as a different answer."""
        recorded: list[Any] = []
        ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "",
            record=lambda c, d: recorded.append((c, d)),
        ).observe(prompt="q", cheap_model="haiku", flagship_output="x",
                  confidence=0.9)

        assert recorded == []

    def test_a_failing_recorder_does_not_raise(self):
        def explode(_c, _d):
            raise OSError("disk full")

        ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "different text entirely",
            record=explode,
        ).observe(prompt="q", cheap_model="haiku", flagship_output="x",
                  confidence=0.9)

    def test_background_observation_does_not_block(self):
        recorded: list[Any] = []
        thread = ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "some other answer about pods",
            record=lambda c, d: recorded.append((c, d)),
        ).observe_in_background(prompt="q", cheap_model="haiku",
                                flagship_output="salted hashing", confidence=0.7)
        thread.join(timeout=5)

        assert thread.daemon is True, "must not hold the process open"
        assert len(recorded) == 1


class TestItActuallyCertifies:
    def test_shadow_observations_can_earn_a_certificate(self, tmp_path):
        """The point of the whole design: this path must bootstrap.

        Before it, the controller had no producer that could run before routing
        was enabled, so the certificate was unreachable by construction.
        """
        from entroly.ravs.conformal import ConformalRoutingController

        controller = ConformalRoutingController(tmp_path / "c.json", alpha=0.05)
        calibrator = ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "salted password hashing",
            record=controller.record)

        assert controller.certificate().permits_routing is False
        for _ in range(40):
            calibrator.observe(prompt="q", cheap_model="haiku",
                               flagship_output="salted password hashing",
                               confidence=0.95)

        certificate = controller.certificate()
        assert certificate.permits_routing is True, (
            "shadow calibration must be able to bootstrap the certificate"
        )
        assert certificate.lambda_hat <= 0.95

    def test_a_divergent_cheap_model_never_earns_one(self, tmp_path):
        from entroly.ravs.conformal import ConformalRoutingController

        controller = ConformalRoutingController(tmp_path / "c.json", alpha=0.05)
        calibrator = ShadowCalibrator(
            invoke_cheap=lambda _m, _p: "entirely unrelated content about pods",
            record=controller.record)

        for _ in range(40):
            calibrator.observe(prompt="q", cheap_model="haiku",
                               flagship_output="salted password hashing",
                               confidence=0.95)

        assert controller.certificate().permits_routing is False, (
            "a cheap model that keeps diverging must never be authorised"
        )
