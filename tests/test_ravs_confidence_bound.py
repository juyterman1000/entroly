"""RAVS must not claim more confidence than its posterior supports.

The router routes a request to a cheaper model when the lower end of a 95%
credible interval on that cell's success rate clears ``ci_threshold``. That
bound was a normal approximation, ``mean - 1.96 * std``. A Beta is not
symmetric, and the gate lives exactly where the approximation is worst: few
observations, mean near 1.

Measured against the exact quantile, the approximation overstates the bound in
every configuration tested, and in three it is the difference between holding
and routing:

    cell             mean-1.96s   exact 2.5%
    n=10, 10/0         0.8367       0.7828
    n=20, 19/1         0.8210       0.7892
    n=35, 32/3         0.8073       0.7886

The worst case is the *smallest cell that can ever qualify* -- ``min_samples``
observations with a perfect record -- so the overstatement is largest where the
evidence is thinnest. CLAUDE.md lists RAVS as fail-closed ("always routes to
Opus when uncertain; never sacrifice correctness for cost"); a bound that
overstates fails open.

``report.py`` had a second, independent version of the same problem: it kept the
empirical-Bayes prior that ``router.py`` had already removed, so the report an
operator reads to audit routing disagreed with the router that made the call.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entroly.ravs import report as ravs_report  # noqa: E402
from entroly.ravs.beta_bounds import beta_cdf, beta_lower_bound  # noqa: E402
from entroly.ravs.router import BayesianRouter, classify_archetype  # noqa: E402

FLAGSHIP = "claude-sonnet-4-20250514"


def _normal_lower(alpha: float, beta: float) -> float:
    """The approximation that was in use, for comparison."""
    ab = alpha + beta
    mean = alpha / ab
    var = (alpha * beta) / (ab * ab * (ab + 1))
    return max(0.0, mean - 1.96 * math.sqrt(var))


@pytest.mark.parametrize(
    "alpha,beta,expected",
    [
        (1.0, 1.0, 0.025),                       # Uniform
        (2.0, 1.0, math.sqrt(0.025)),            # CDF = x^2
        (1.0, 2.0, 1 - math.sqrt(0.975)),        # CDF = 1-(1-x)^2
        (3.0, 1.0, 0.025 ** (1 / 3)),            # CDF = x^3
    ],
)
def test_bound_matches_closed_form_quantiles(alpha, beta, expected):
    """Anchor the implementation on Betas whose quantiles are known exactly."""
    assert beta_lower_bound(alpha, beta) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "alpha,beta",
    [(0.5, 0.5), (1.5, 0.5), (10.5, 0.5), (20.5, 1.5), (40.5, 3.5), (200.5, 9.5)],
)
def test_the_bound_is_actually_the_quantile(alpha, beta):
    """It must invert the CDF, not approximate it."""
    x = beta_lower_bound(alpha, beta)
    assert beta_cdf(x, alpha, beta) == pytest.approx(0.025, abs=1e-9)
    assert 0.0 <= x <= 1.0


@pytest.mark.parametrize(
    "passes,failures",
    [(10, 0), (19, 1), (32, 3), (5, 0), (50, 2), (100, 7), (2, 0)],
)
def test_the_bound_is_never_more_optimistic_than_the_posterior(passes, failures):
    """The regression that matters: never report above the exact quantile.

    Failing closed means erring downward. The normal approximation errs upward,
    which is what let thin cells route.
    """
    alpha, beta = 0.5 + passes, 0.5 + failures
    exact = beta_lower_bound(alpha, beta)
    approx = _normal_lower(alpha, beta)
    assert exact <= approx + 1e-12, (
        f"{passes}/{failures}: exact bound {exact:.4f} exceeds the normal "
        f"approximation {approx:.4f}; the direction of the error has flipped"
    )


def test_the_smallest_qualifying_cell_does_not_route(monkeypatch):
    """A perfect record at exactly `min_samples` must not clear the gate.

    This is the concrete case the approximation got wrong: 10 passes, 0
    failures gives 0.8367 approximated and 0.7828 exactly, against a 0.80
    threshold.
    """
    router = BayesianRouter()
    router.enabled = True

    message = "rename this variable"
    archetype = classify_archetype(message)
    passes, failures = router._min_samples, 0
    cell = {
        "n": passes + failures,
        "passes": passes,
        "failures": failures,
        "alpha": 0.5 + passes,
        "beta": 0.5 + failures,
        "posterior_mean": (0.5 + passes) / (1.0 + passes + failures),
    }
    monkeypatch.setattr(router, "_get_cells", lambda: {archetype: cell})

    # The approximation would have cleared the threshold here.
    assert _normal_lower(cell["alpha"], cell["beta"]) > router._ci_threshold
    assert beta_lower_bound(cell["alpha"], cell["beta"]) < router._ci_threshold

    decision = router.route(FLAGSHIP, message, _ignore_enabled=True)
    assert decision.use_original, (
        f"routed to a cheaper model on {passes}/{failures}: {decision.reason} "
        f"(confidence={decision.confidence})"
    )
    assert "ci_low" in decision.reason, decision.reason


def test_enough_evidence_still_routes(monkeypatch):
    """Guard the other direction: the exact bound must not disable routing.

    A gate that never opens is not fail-closed, it is broken, and would be
    invisible except as a silent loss of every cost saving RAVS exists for.
    """
    router = BayesianRouter()
    router.enabled = True

    message = "rename this variable"
    archetype = classify_archetype(message)
    passes, failures = 200, 2
    cell = {
        "n": passes + failures,
        "passes": passes,
        "failures": failures,
        "alpha": 0.5 + passes,
        "beta": 0.5 + failures,
        "posterior_mean": (0.5 + passes) / (1.0 + passes + failures),
    }
    monkeypatch.setattr(router, "_get_cells", lambda: {archetype: cell})

    decision = router.route(FLAGSHIP, message, _ignore_enabled=True)
    assert not decision.use_original, (
        f"a 200/2 record did not route: {decision.reason}"
    )
    assert decision.recommended_model
    assert decision.confidence >= router._ci_threshold


def test_gate_status_does_not_report_its_own_threshold_as_a_measurement():
    """A metric field must hold a measurement or say it has none.

    ``compute_gate_status`` set ``executor_coverage=min_executor_coverage`` with
    the comment "# from shadow data", so the reported coverage was always
    exactly the gate's threshold (0.50) no matter what the fleet did, and read
    as measured. Nothing produces executor coverage, so the honest value is
    "not measured".
    """
    from entroly.ravs.router import compute_gate_status

    base = {
        "total_requests": 100,
        "decomposition_evidence_rate": 0.9,
        "success_rate": 0.9,
    }

    unmeasured = compute_gate_status(base)
    assert unmeasured.executor_coverage is None, (
        f"reported {unmeasured.executor_coverage} for a metric nothing "
        "measured; None means 'not measured'"
    )

    # Changing the threshold must not change the reported metric.
    shifted = compute_gate_status(base, min_executor_coverage=0.99)
    assert shifted.executor_coverage == unmeasured.executor_coverage, (
        "the reported coverage tracks the threshold, so it is the threshold"
    )

    # And a real value, if one ever appears, is passed through.
    measured = compute_gate_status({**base, "executor_coverage": 0.73})
    assert measured.executor_coverage == pytest.approx(0.73)


def test_report_and_router_score_a_cell_with_the_same_prior(tmp_path):
    """The audit surface must not contradict the decision it explains.

    ``report.py`` kept an empirical-Bayes prior fitted to the log it scored --
    the construction ``router.py`` removed and documents as failing open. On a
    log with no failures that prior is Beta(2.0, 0.1), a mean of 0.952 before
    any observation, and it displayed P(success)=0.976 for a 2-pass cell the
    router scores at 0.833.
    """
    import json

    log = tmp_path / "ravs.jsonl"
    lines = []
    for i in range(2):
        rid = f"r{i}"
        lines.append(
            json.dumps(
                {"kind": "trace", "request_id": rid, "timestamp": 1.0 + i,
                 "tool": "test", "type": "request"}
            )
        )
        lines.append(
            json.dumps(
                {"kind": "outcome", "request_id": rid, "timestamp": 2.0 + i,
                 "event_type": "test_result", "value": "passed", "tool": "test"}
            )
        )
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ravs_report.generate_report(log)
    cells = report.get("bayesian_cells", {})
    if not cells:
        pytest.skip("no bayesian cells derived from the synthetic log")

    for key, cell in cells.items():
        implied_alpha_0 = cell["alpha"] - cell["passes"]
        implied_beta_0 = cell["beta"] - cell["failures"]
        assert implied_alpha_0 == pytest.approx(0.5, abs=1e-6), (
            f"{key}: report prior alpha_0={implied_alpha_0}, router uses 0.5"
        )
        assert implied_beta_0 == pytest.approx(0.5, abs=1e-6), (
            f"{key}: report prior beta_0={implied_beta_0}, router uses 0.5"
        )
