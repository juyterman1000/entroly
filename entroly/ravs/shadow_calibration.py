"""Earn the routing certificate without ever serving a cheap answer.

The conformal controller in :mod:`entroly.ravs.conformal` can certify a routing
threshold, but only from observations of how a cheap model performed -- and
those cannot exist until the cheap model has been used. Routing therefore could
not bootstrap: the certificate required routing, and routing required the
certificate.

The way out is to separate *who answers the user* from *what we learn*. On a
small sample of requests the flagship model answers, exactly as it would have,
and the cheap model is called alongside purely to be compared. The user is
never served the cheap output during calibration, so the bootstrap costs money
rather than quality -- a few dozen cheap calls to earn a permanent threshold.

Two properties make that affordable:

*It stops.* Sampling is disabled once enough observations exist to certify at
the configured ``alpha``. Calibration is a bootstrap phase with a defined end,
not a permanent tax.

*It is narrow.* Only low-risk requests are sampled. A request the risk
classifier will never route is a request there is nothing to learn from.

The divergence label
--------------------
Two answers are compared by token-set Jaccard distance. This is deliberately a
measure of *difference*, not of *quality*: a cheap model that answers correctly
in different words is scored as divergent. That makes the label conservative --
it over-counts divergence -- and a conservative label biases the conformal
bound toward routing less than strictly necessary. Erring toward under-routing
is the correct direction for a guarantee about someone else's traffic, so the
imprecision is a feature here rather than a defect, but it is imprecision and
should not be described as a quality judgement.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("entroly.ravs.shadow_calibration")

_DEFAULT_EPSILON = 0.10
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Ceiling on how long a bootstrap may run before giving up. Sampling continues
# until a certificate permits routing rather than until the minimum feasible
# sample size, so without this a cheap model that never agrees would be paid
# for indefinitely. Matches the controller's retention window: observations
# beyond it displace older ones, so more spending cannot grow the sample.
_MAX_CALIBRATION = 5000

# Above this Jaccard distance the two answers are treated as divergent.
# Chosen so paraphrase-level differences in wording do not dominate, while
# genuinely different content does.
_DIVERGENCE_THRESHOLD = 0.50


def _epsilon_from_env() -> float:
    raw = os.environ.get("ENTROLY_RAVS_SHADOW_RATE", "").strip()
    if not raw:
        return _DEFAULT_EPSILON
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_EPSILON
    return value if 0.0 <= value <= 1.0 else _DEFAULT_EPSILON


def shadow_enabled() -> bool:
    """On unless disabled. Costs cheap-model calls, so it is opt-out, not free."""
    return os.environ.get("ENTROLY_RAVS_SHADOW", "1").strip() not in {
        "0", "false", "no",
    }


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def divergence_score(flagship: str, cheap: str) -> float:
    """Jaccard distance between the two answers, in ``[0, 1]``.

    Both empty is treated as agreement: two models that produced nothing
    produced the same nothing, and calling that a divergence would penalise
    routing for a failure the flagship shares.
    """
    a, b = _tokens(flagship), _tokens(cheap)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - len(a & b) / len(a | b)


def diverged(flagship: str, cheap: str,
             threshold: float = _DIVERGENCE_THRESHOLD) -> bool:
    """Whether the cheap answer differs enough to count against routing."""
    return divergence_score(flagship, cheap) > threshold


@dataclass(frozen=True)
class SampleDecision:
    """Why this request was or was not shadowed -- inspectable, not a bare bool."""

    sample: bool
    reason: str


def should_sample(
    *,
    risk_level: str,
    observations: int,
    samples_needed: int,
    permits_routing: bool = False,
    max_observations: int = _MAX_CALIBRATION,
    epsilon: float | None = None,
    rng: random.Random | None = None,
) -> SampleDecision:
    """Decide whether to spend a cheap-model call on this request.

    Pure apart from the draw, and the draw is injectable, so the policy can be
    tested without probability.
    """
    if not shadow_enabled():
        return SampleDecision(False, "shadow calibration disabled")
    # An unclassified request is not a low-risk one. Testing truthiness first
    # let risk_level="" -- which several RoutingDecision early-return paths
    # produce -- skip the gate entirely and be sampled as though it had been
    # classified, spending a call on traffic that can never be routed and
    # polluting the calibration set with a population the bound is not about.
    if risk_level.lower() != "low":
        return SampleDecision(
            False, f"risk={risk_level or 'unclassified'} is never routed")
    # Stopping at samples_needed was wrong: that is only the smallest n at
    # which the route-nothing threshold becomes certifiable, not the point at
    # which routing is justified. Landing there with a few divergent
    # observations left the certificate valid, permitting nothing, and
    # permanently unable to gather the evidence that would change it.
    # Calibration now continues until a certificate actually permits routing.
    if permits_routing:
        return SampleDecision(
            False, f"certified and routing ({observations} observations)")
    if observations >= max_observations:
        # A ceiling is still needed, or a cheap model that never agrees would
        # be paid for forever.
        return SampleDecision(
            False,
            f"calibration exhausted at {observations} observations without a "
            f"routing certificate; the cheap model does not agree often enough")
    rate = _epsilon_from_env() if epsilon is None else epsilon
    if rate <= 0.0:
        return SampleDecision(False, "sampling rate is zero")
    draw = (rng or random).random()
    if draw >= rate:
        return SampleDecision(False, "not selected by sampling")
    return SampleDecision(True, f"sampled at rate {rate}")


class ShadowCalibrator:
    """Runs the comparison off the request path and records the observation.

    The user's response has already been produced and returned by the time this
    does anything. It exists to spend a cheap call and write one row, so every
    failure is swallowed: calibration that breaks a request has cost more than
    it could ever earn.
    """

    def __init__(self, invoke_cheap: Callable[[str, str], str],
                 record: Callable[[float, bool], None]):
        self._invoke_cheap = invoke_cheap
        self._record = record

    def observe(self, *, prompt: str, cheap_model: str, flagship_output: str,
                confidence: float) -> None:
        """Compare the two answers and record the result. Never raises."""
        try:
            cheap_output = self._invoke_cheap(cheap_model, prompt)
            if not cheap_output:
                return
            self._record(confidence, diverged(flagship_output, cheap_output))
        except Exception as e:  # noqa: BLE001 - calibration is never worth a request
            logger.debug("shadow calibration observation skipped: %s", e)

    def observe_in_background(self, **kwargs: Any) -> threading.Thread:
        """Same, on a daemon thread, so nothing waits on a second model call."""
        thread = threading.Thread(
            target=self.observe, kwargs=kwargs,
            name="entroly-ravs-shadow", daemon=True)
        thread.start()
        return thread


__all__ = [
    "SampleDecision",
    "ShadowCalibrator",
    "divergence_score",
    "diverged",
    "shadow_enabled",
    "should_sample",
]
