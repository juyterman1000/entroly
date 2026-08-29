"""The routing threshold must be earned, and the guarantee must be real.

Routing was gated behind an authorisation nobody could answer informedly: the
risk was unbounded and unmeasured, so "may I swap your model?" had no basis.
Conformal risk control replaces the guess with a threshold carrying a
distribution-free finite-sample bound on the divergence rate.

A guarantee asserted in a docstring is worth nothing, so the last test here
measures it: simulated traffic, repeated trials, realised divergence compared
against the promised alpha.
"""

from __future__ import annotations

import random

import pytest

from entroly.ravs.conformal import (
    NEVER,
    Certificate,
    ConformalRoutingController,
    certify,
    required_samples,
)


class TestSampleRequirement:
    def test_the_minimum_falls_out_of_the_bound(self):
        """n >= B/alpha - 1, not a tuning constant."""
        assert required_samples(0.05) == 19
        assert required_samples(0.02) == 49
        assert required_samples(0.10) == 9

    def test_too_few_samples_cannot_be_certified(self):
        """Below the minimum, no threshold satisfies (1) -- including 'route nothing'."""
        observations = [(0.99, False)] * 10
        result = certify(observations, alpha=0.02)

        assert result.has_enough_data is False
        assert result.permits_routing is False
        assert "insufficient calibration" in result.reason

    def test_no_observations_refuses(self):
        assert certify([], alpha=0.05).permits_routing is False


class TestCertification:
    def test_clean_history_permits_routing(self):
        observations = [(0.95, False)] * 60
        result = certify(observations, alpha=0.05)

        assert result.has_enough_data is True
        assert result.permits_routing is True
        assert result.lambda_hat <= 0.95

    def test_divergent_history_refuses_to_route(self):
        """Every routed request diverged; the only safe threshold routes nothing."""
        observations = [(0.95, True)] * 60
        result = certify(observations, alpha=0.05)

        assert result.has_enough_data is True, "enough data to conclude something"
        assert result.permits_routing is False, (
            "a history of pure divergence must not authorise routing"
        )
        assert result.lambda_hat == NEVER

    def test_the_threshold_excludes_the_divergent_region(self):
        """Low-confidence decisions diverge; the bar must rise above them."""
        observations = [(0.60, True)] * 30 + [(0.99, False)] * 70
        result = certify(observations, alpha=0.05)

        assert result.permits_routing is True
        assert result.lambda_hat > 0.60, (
            "the certified bar must sit above the confidences that diverged"
        )

    def test_a_stricter_alpha_is_never_more_permissive(self):
        observations = [(0.9, False)] * 90 + [(0.9, True)] * 10
        loose = certify(observations, alpha=0.20)
        strict = certify(observations, alpha=0.02)

        assert strict.lambda_hat >= loose.lambda_hat, (
            "demanding less risk must not permit more routing"
        )

    def test_certified_is_not_the_same_as_enabled(self):
        """A valid certificate can authorise nothing; the two must not be conflated."""
        result = certify([(0.95, True)] * 60, alpha=0.05)
        assert result.has_enough_data is True and result.permits_routing is False


class TestGuaranteeHolds:
    def test_realised_divergence_respects_alpha(self):
        """Measure the promise: E[L] <= alpha on held-out requests.

        Split traffic into calibration and test, certify on the former, apply
        the threshold to the latter, and count how often a routed request
        diverged. Averaged over trials this must not exceed alpha.
        """
        rng = random.Random(20260828)
        alpha = 0.10
        realised = []

        for _ in range(200):
            def draw():
                confidence = rng.uniform(0.5, 1.0)
                # Divergence is more likely when confidence is low.
                return confidence, rng.random() > confidence

            calibration = [draw() for _ in range(120)]
            holdout = [draw() for _ in range(120)]

            result = certify(calibration, alpha=alpha)
            routed = [
                diverged for confidence, diverged in holdout
                if result.permits_routing and confidence >= result.lambda_hat
            ]
            # Requests we declined to route carry no loss.
            realised.append(sum(routed) / len(holdout))

        mean_loss = sum(realised) / len(realised)
        assert mean_loss <= alpha, (
            f"guarantee violated: realised {mean_loss:.4f} > alpha {alpha}"
        )

    def test_the_bound_is_not_vacuous(self):
        """A method that never routes would trivially satisfy any alpha."""
        observations = [(0.98, False)] * 200
        result = certify(observations, alpha=0.10)

        assert result.permits_routing is True, (
            "on clean history the certificate must actually allow routing, "
            "or the guarantee is met by doing nothing"
        )


class TestControllerFailsClosed:
    @pytest.fixture
    def controller(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
        return ConformalRoutingController(tmp_path / "calib.json", alpha=0.05)

    def test_a_fresh_install_does_not_route(self, controller):
        assert controller.certificate().permits_routing is False

    def test_observations_persist(self, controller, tmp_path):
        for _ in range(60):
            controller.record(0.97, False)

        reopened = ConformalRoutingController(tmp_path / "calib.json", alpha=0.05)
        assert reopened.certificate().n == 60, (
            "calibration gathered from real traffic must survive a restart"
        )

    def test_a_corrupt_store_refuses_rather_than_routes(self, controller, tmp_path):
        (tmp_path / "calib.json").write_text("{not json", encoding="utf-8")
        assert controller.certificate().permits_routing is False

    def test_non_finite_confidence_is_ignored(self, controller):
        controller.record(float("nan"), False)
        controller.record(float("inf"), False)
        assert controller.certificate().n == 0

    def test_recording_never_raises(self, controller):
        controller.record("not a number", False)  # type: ignore[arg-type]

    def test_the_payload_states_its_method(self, controller):
        payload = controller.certificate().as_dict()
        assert "conformal risk control" in payload["method"]
        assert payload["permits_routing"] is False
        assert isinstance(payload["samples_needed"], int)


def test_certificate_is_immutable():
    """A guarantee that can be edited after issue is not a guarantee."""
    cert = Certificate(True, 0.9, 0.05, 100, 0.0, 19, "certified")
    with pytest.raises(Exception):
        cert.lambda_hat = 0.1  # type: ignore[misc]


class TestProxyGating:
    """Certification must gate the switch, and a fresh install must not route."""

    def test_the_proxy_consults_the_certificate(self):
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        assert "permits_routing" in source, (
            "the proxy must gate on the certificate, not merely compute one"
        )
        assert "ENTROLY_RAVS_AUTO" in source

    def test_an_uncertified_install_leaves_routing_off(self, tmp_path, monkeypatch):
        """The whole safety property in one assertion."""
        monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
        monkeypatch.delenv("ENTROLY_RAVS_ROUTER", raising=False)
        from entroly.ravs import conformal

        conformal.reset_for_tests()
        assert conformal.get_controller().certificate().permits_routing is False

    def test_the_explicit_switch_still_works(self, monkeypatch):
        """Certification adds a path to on; it must not remove the existing one."""
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        assert 'os.environ.get("ENTROLY_RAVS_ROUTER", "0") == "1"' in source


class TestTheLoopIsClosed:
    """A certificate with no producer can never be earned.

    The first version of this module shipped a consumer and no writer: nothing
    called record(), so n stayed 0 and the feature was inert. That is the same
    defect as a dashboard field nothing increments, and it is pinned here.
    """

    def test_the_proxy_records_calibration_observations(self):
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        assert "get_controller().record(" in source, (
            "no producer: the certificate could never be earned"
        )

    def test_the_observation_is_paired_with_its_confidence(self):
        """A label without the confidence it was observed at cannot calibrate."""
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        assert "confidence=_ravs_prev_conf" in source
        assert "_ravs_prev_conf = prev_routed" in source, (
            "confidence must be carried from the decision to its outcome"
        )

    def test_the_docstring_does_not_claim_a_verifier_label(self):
        """The label is behavioural feedback; claiming otherwise oversells it."""
        from entroly.ravs import conformal

        doc = conformal.__doc__ or ""
        assert "behavioural proxy" in doc
        assert "not a deterministic verifier" in doc

    def test_the_bootstrap_limitation_is_documented(self):
        from entroly.ravs import conformal

        doc = conformal.__doc__ or ""
        assert "circularity" in doc, (
            "a user must be told the certificate cannot self-start"
        )


class TestReviewFixes:
    """Five defects an extra-high-recall review found in this feature."""

    def test_an_unreadable_store_is_not_an_empty_one(self, tmp_path):
        """Finding 3: one torn read used to destroy the whole history."""
        from entroly.ravs.conformal import ConformalRoutingController

        path = tmp_path / "calib.json"
        controller = ConformalRoutingController(path, alpha=0.05)
        for _ in range(60):
            controller.record(0.97, False)
        assert controller.certificate().n == 60

        path.write_text('{"observations": [{"conf', encoding="utf-8")
        controller.record(0.97, False)

        assert path.read_text(encoding="utf-8").startswith('{"observations": [{"conf'), (
            "a store that could not be read must not be overwritten"
        )

    def test_an_unreadable_store_refuses_to_route(self, tmp_path):
        from entroly.ravs.conformal import ConformalRoutingController

        path = tmp_path / "calib.json"
        path.write_text("{ truncated", encoding="utf-8")
        certificate = ConformalRoutingController(path, alpha=0.05).certificate()

        assert certificate.permits_routing is False
        assert "unreadable" in certificate.reason

    def test_a_missing_store_is_still_a_clean_start(self, tmp_path):
        """Absent and corrupt must stay distinguishable."""
        from entroly.ravs.conformal import ConformalRoutingController

        controller = ConformalRoutingController(tmp_path / "none.json", alpha=0.05)
        controller.record(0.9, False)
        assert controller.certificate().n == 1

    def test_the_store_is_replaced_atomically(self, tmp_path):
        """Finding 4: write_text truncated before writing, so readers tore."""
        import inspect

        from entroly.ravs import conformal

        source = inspect.getsource(conformal.ConformalRoutingController._write)
        assert "os.replace" in source, "the swap must be atomic for concurrent readers"

    def test_no_temp_files_are_left_behind(self, tmp_path):
        from entroly.ravs.conformal import ConformalRoutingController

        controller = ConformalRoutingController(tmp_path / "calib.json", alpha=0.05)
        for _ in range(5):
            controller.record(0.9, False)

        assert list(tmp_path.glob("*.tmp")) == []


class TestCertifiedThresholdIsEnforced:
    """Finding 1: the bound was proven for a rule production did not apply."""

    def test_the_proxy_compares_confidence_against_lambda_hat(self):
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        assert "_conf < _cert.lambda_hat" in source, (
            "routing below the certified threshold is outside the proven region"
        )

    def test_lambda_hat_is_not_only_a_log_argument(self):
        """It was previously reachable only from a logger.info format string."""
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        before, _, after = source.partition("logger.info")
        assert "lambda_hat" in source
        assert "_cert.lambda_hat" in source.replace("logger.info", "", 1) or                "_conf < _cert.lambda_hat" in source
