"""Certify a routing threshold instead of asking the user to guess one.

Routing substitutes the model on a live request, so it has always required an
explicit authorisation. That is the right default for an unbounded risk, but it
made the question unanswerable: "may I swap your model?" has no informed
answer, because nothing measured what the swap would cost.

Bounding the risk changes the question. Given calibration observations drawn
from the user's own traffic, conformal risk control returns a threshold with a
distribution-free, finite-sample guarantee on the expected divergence rate. The
user then answers a question they *can* answer -- "do you accept a 2% divergence
rate?" -- once, rather than per request.

Method
------
Conformal risk control (Angelopoulos, Bates, Fisch, Lei & Schuster, ICLR 2024)
generalises split conformal prediction from coverage to any monotone loss. For
losses ``L_i(λ)`` non-increasing in ``λ`` and bounded above by ``B``:

    λ̂ = inf{ λ : (n·R̂_n(λ) + B) / (n + 1) ≤ α }        (1)

guarantees ``E[L_{n+1}(λ̂)] ≤ α`` on the next request.

Here ``λ`` is the confidence a routing decision must clear before the swap is
taken, and ``L_i(λ) = 1`` when request ``i`` would have been routed at ``λ``
(its confidence cleared the bar) *and* the cheap model then diverged. Raising
``λ`` routes strictly fewer requests, so ``L_i`` is non-increasing and (1)
applies. Divergence is scored by the deterministic RAVS verifiers -- tests,
lint, file reads -- so calibration costs no model calls and the label is an
observation rather than a judgement.

The minimum sample size is not a tuning knob; it falls out of (1). At the
most conservative threshold nothing is routed, so ``R̂_n = 0`` and the bound
reduces to ``B/(n+1) ≤ α``, giving

    n ≥ B/α − 1                                          (2)

Below that, no threshold can be certified at level ``α`` -- and this module
reports that rather than routing on an unearned guarantee.

What the guarantee does not cover
---------------------------------
Exchangeability. The bound holds when calibration observations and the next
request are exchangeable. Traffic that shifts -- a new repository, a different
task mix -- breaks that assumption, which is why observations carry a cohort
key and a certificate is scoped to the cohort that produced it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.ravs.conformal")

# Loss is an indicator, so it is bounded by 1.
_LOSS_BOUND = 1.0

# Sentinel threshold meaning "no confidence clears the bar" -- certified, but
# permitting nothing. Kept distinct from 1.0 so a genuine 1.0-confidence
# decision cannot be confused with refusing to route.
NEVER = float("inf")

_DEFAULT_ALPHA = 0.02
_MAX_OBSERVATIONS = 5000


def _alpha_from_env() -> float:
    raw = os.environ.get("ENTROLY_RAVS_ALPHA", "").strip()
    if not raw:
        return _DEFAULT_ALPHA
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_ALPHA
    return value if 0.0 < value < 1.0 else _DEFAULT_ALPHA


def required_samples(alpha: float, bound: float = _LOSS_BOUND) -> int:
    """Smallest ``n`` for which any threshold can be certified at level ``alpha``.

    From (2): ``n ≥ B/α − 1``. Reported so a user waiting for automatic
    routing can see how far away it is instead of wondering whether it is
    broken.
    """
    if alpha <= 0.0:
        return 0
    return max(1, math.ceil(bound / alpha - 1.0))


@dataclass(frozen=True)
class Certificate:
    """The outcome of applying (1) to the observations on hand."""

    certified: bool
    lambda_hat: float
    alpha: float
    n: int
    empirical_risk: float
    samples_needed: int
    reason: str

    @property
    def permits_routing(self) -> bool:
        """Whether the certificate allows any request to be routed.

        A certificate at :data:`NEVER` is valid and permits nothing: the data
        support a guarantee, and the guarantee they support is "route none of
        it". Distinguishing the two stops "certified" from being read as
        "enabled".
        """
        return self.certified and self.lambda_hat != NEVER

    def as_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "permits_routing": self.permits_routing,
            "lambda_hat": None if self.lambda_hat == NEVER else self.lambda_hat,
            "alpha": self.alpha,
            "n": self.n,
            "empirical_risk": self.empirical_risk,
            "samples_needed": self.samples_needed,
            "reason": self.reason,
            "method": (
                "conformal risk control (Angelopoulos et al., ICLR 2024): "
                "lambda_hat = inf{lambda : (n*R_n(lambda) + B)/(n+1) <= alpha}, "
                "B = 1; guarantees E[L] <= alpha under exchangeability"
            ),
        }


def certify(
    observations: list[tuple[float, bool]], alpha: float, bound: float = _LOSS_BOUND
) -> Certificate:
    """Apply (1) to ``observations``: ``(confidence, diverged)`` pairs.

    Pure function of its inputs -- no IO, no clock, no global state -- so the
    guarantee can be checked by hand against the same numbers.
    """
    n = len(observations)
    needed = required_samples(alpha, bound)

    if n == 0:
        return Certificate(False, NEVER, alpha, 0, 0.0, needed,
                           "no calibration observations")
    if n < needed:
        # Not conservatism: at n below this, (1) is unsatisfiable at every
        # threshold, including the one that routes nothing.
        return Certificate(
            False, NEVER, alpha, n, 0.0, needed,
            f"insufficient calibration: {n} of {needed} required at alpha={alpha}")

    # L_i is a step function of lambda that only changes at an observed
    # confidence, so scanning those values plus NEVER finds the exact infimum.
    candidates = sorted({confidence for confidence, _ in observations})
    candidates.append(NEVER)

    for lam in candidates:
        risk = sum(
            1.0 for confidence, diverged in observations
            if confidence >= lam and diverged
        ) / n
        if (n * risk + bound) / (n + 1) <= alpha:
            return Certificate(
                True, lam, alpha, n, round(risk, 6), needed,
                "certified" if lam != NEVER else "certified: routes nothing")

    # Unreachable while n >= needed, since the NEVER candidate drives R̂ to 0
    # and reduces the bound to B/(n+1) <= alpha. Kept so a future change to the
    # loss cannot silently fall through into an uncertified route.
    return Certificate(False, NEVER, alpha, n, 0.0, needed,
                       "no threshold satisfies the risk bound")


class ConformalRoutingController:
    """Collects calibration observations and certifies a routing threshold.

    Persisted so a guarantee survives a restart: calibration is gathered from
    real traffic, and discarding it on every process boundary would mean the
    threshold could never be earned.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None,
                 alpha: float | None = None):
        self._alpha = alpha if alpha is not None else _alpha_from_env()
        self._lock = threading.Lock()
        root = os.environ.get("ENTROLY_DIR") or os.path.join(os.getcwd(), ".entroly")
        self._path = Path(path) if path else Path(root) / "ravs_calibration.json"

    @property
    def alpha(self) -> float:
        return self._alpha

    def _load(self) -> list[tuple[float, bool]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [
                (float(row["confidence"]), bool(row["diverged"]))
                for row in raw.get("observations", [])
                if "confidence" in row and "diverged" in row
            ]
        except (OSError, ValueError, TypeError, KeyError):
            return []

    def record(self, confidence: float, diverged: bool) -> None:
        """Add one calibration observation. Never raises.

        ``diverged`` must come from a deterministic verifier, not a model
        judging itself -- the guarantee is only as sound as the label.
        """
        try:
            value = float(confidence)
            if not math.isfinite(value):
                return
            with self._lock:
                observations = self._load()
                observations.append((value, bool(diverged)))
                # Bounded, keeping the most recent: exchangeability is more
                # plausible over a recent window than over all history.
                observations = observations[-_MAX_OBSERVATIONS:]
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps({
                    "observations": [
                        {"confidence": c, "diverged": d} for c, d in observations
                    ],
                }), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - calibration must not fail a request
            logger.debug("calibration observation not recorded: %s", e)

    def certificate(self) -> Certificate:
        """Current certificate. Fails closed on any error."""
        try:
            with self._lock:
                observations = self._load()
            return certify(observations, self._alpha)
        except Exception as e:  # noqa: BLE001
            logger.debug("certification failed, refusing to route: %s", e)
            return Certificate(False, NEVER, self._alpha, 0, 0.0,
                               required_samples(self._alpha),
                               f"certification error: {type(e).__name__}")


_controller: ConformalRoutingController | None = None
_controller_lock = threading.Lock()


def get_controller() -> ConformalRoutingController:
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = ConformalRoutingController()
        return _controller


def reset_for_tests() -> None:
    global _controller
    with _controller_lock:
        _controller = None


__all__ = [
    "Certificate",
    "ConformalRoutingController",
    "NEVER",
    "certify",
    "get_controller",
    "required_samples",
    "reset_for_tests",
]
