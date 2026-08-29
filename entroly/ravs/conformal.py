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
(its confidence cleared the bar) *and* the outcome was recorded as a failure.
Raising ``λ`` routes strictly fewer requests, so ``L_i`` is non-increasing and
(1) applies.

What the label actually is
--------------------------
The bound is only as meaningful as the quantity being bounded, so it is worth
being exact. The label is not a deterministic verifier result. The only outcome
signal observable on the routing path is the proxy's implicit feedback: a
routed request followed by a near-duplicate query is read as a failure (the
user rephrased), a topic change as a success. That is a behavioural proxy,
recorded elsewhere at ``source_strength=0.4``.

So (1) bounds *the rate at which routed requests are followed by a rephrase*,
not the rate of true quality divergence. Those correlate, and the first is
observable while the second is not, but they are not the same and a reader
deciding whether to accept ``α`` should know which one they are accepting.

Bootstrapping
-------------
There is a circularity that no theorem removes: the label requires having
routed, and routing requires a certificate. A cheap model's divergence cannot
be estimated without ever running the cheap model. Calibration therefore
accrues only while routing is on, which means the honest shape of this feature
is *explicit opt-in to begin, automatic maintenance and revocation thereafter*
-- a user who enables routing gets a threshold that tightens as evidence
arrives and withdraws itself if outcomes degrade.

Escaping that would need either a consented exploration budget (routing a small
ε of low-risk traffic to gather evidence) or paying to run both models on a
sample. Both are real options; neither is free, and neither is implemented
here.

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

    has_enough_data: bool
    lambda_hat: float
    alpha: float
    n: int
    empirical_risk: float
    samples_needed: int
    reason: str

    @property
    def permits_routing(self) -> bool:
        """Whether the certificate allows any request to be routed.

        A certificate can hold enough data to conclude something and have
        that conclusion be "route none of it". The field was called
        ``certified``, which reads as "enabled" -- and ``if cert.certified:``
        type-checks, sounds right, and would authorise routing on a
        certificate whose only valid threshold is NEVER. This is the only
        field a caller should gate on.
        """
        return self.has_enough_data and self.lambda_hat != NEVER

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_enough_data": self.has_enough_data,
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

    # Unreachable while n >= needed: the NEVER candidate drives R̂ to 0, which
    # reduces the bound to B/(n+1) <= alpha, guaranteed by the guard above. A
    # bare return here would silently hand back an uncertified result if a
    # future change to the loss broke that; say what is assumed instead.
    raise AssertionError(
        f"no threshold satisfied the risk bound with n={n} >= needed={needed}; "
        "the loss is no longer monotone or the NEVER candidate was dropped")


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

    def _load(self) -> tuple[list[tuple[float, bool]], bool]:
        """Return ``(observations, intact)``.

        A file that is absent and a file that is unreadable are different
        situations and used to be conflated into an empty list. Because
        :meth:`record` writes back what it loads, one torn read turned a full
        calibration history into a single row -- reported afterwards as
        "insufficient calibration", which is indistinguishable from a fresh
        install. ``intact`` lets callers refuse to overwrite evidence they
        could not read.
        """
        if not self._path.exists():
            return [], True
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [
                (float(row["confidence"]), bool(row["diverged"]))
                for row in raw.get("observations", [])
                if "confidence" in row and "diverged" in row
            ], True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "calibration store at %s is unreadable (%s); refusing to "
                "overwrite it and declining to route until it is repaired or "
                "removed", self._path, exc)
            return [], False

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
                observations, intact = self._load()
                if not intact:
                    # Appending to what could not be read would persist a
                    # one-row file over an unknown amount of real evidence.
                    return
                observations.append((value, bool(diverged)))
                # Bounded, keeping the most recent: exchangeability is more
                # plausible over a recent window than over all history.
                observations = observations[-_MAX_OBSERVATIONS:]
                self._write(observations)
        except Exception as e:  # noqa: BLE001 - calibration must not fail a request
            logger.debug("calibration observation not recorded: %s", e)

    def _write(self, observations: list[tuple[float, bool]]) -> None:
        """Replace the store atomically.

        ``write_text`` truncates before writing, so a reader arriving mid-write
        -- or a process killed there -- sees a partial document. That matters
        more than usual here: this runs on daemon threads, which are killed
        without joining at interpreter exit, and the proxy, MCP server and
        dashboard are separate processes sharing one ENTROLY_DIR.

        Writing to a temporary file and renaming makes the swap atomic on both
        POSIX and Windows, so a concurrent reader sees either the old file or
        the new one. It does not make read-modify-write atomic across
        processes -- two writers can still interleave and lose an observation
        -- but a lost observation only shrinks the sample, whereas a torn file
        used to destroy the whole history.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "observations": [
                {"confidence": c, "diverged": d} for c, d in observations
            ],
        })
        temporary = self._path.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self._path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def certificate(self) -> Certificate:
        """Current certificate. Fails closed on any error."""
        try:
            with self._lock:
                observations, intact = self._load()
            if not intact:
                # An unreadable store is not an empty one. Certifying from it
                # would mean reasoning about evidence we could not read.
                return Certificate(
                    False, NEVER, self._alpha, 0, 0.0,
                    required_samples(self._alpha),
                    "calibration store unreadable; refusing to route")
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
