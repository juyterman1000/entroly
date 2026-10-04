"""Delayed outcome correction must be anchored to the request being corrected.

The defect: ``on_honest_outcome`` read ``prism._reward_ema`` at *correction*
time, so the advantage for request A depended on every unrelated request that
happened to arrive between A's observation and A's verified outcome. Measured on
an identical verified success (``command_exit``/``success``, reward 0.90) with an
identical observation (baseline 0.40, implicit advantage +0.20):

    EMA at outcome time   delta_advantage
    0.20                        +0.50
    0.40                        +0.30
    0.75                        -0.05      <- a passing command, applied as a penalty

All three describe the same request and the same ground truth, so all three must
produce the same correction. These tests pin that invariant.

The baseline is recovered from the cached pair rather than stored separately,
because ``implicit_advantage`` was defined as ``implicit_reward - baseline`` at
observation time; T2/T3 are what prove the recovery is actually independent of
later traffic rather than merely looking independent.
"""

from __future__ import annotations

import pytest

from entroly.online_learner import OnlinePrism
from entroly.ravs.outcome_bridge import OutcomeBridge

# Semantic-heavy attribution: a uniform vector would move every alpha by the
# same amount and hide a per-dimension sign error.
CONTRIBS = {
    "w_recency": 0.1,
    "w_frequency": 0.1,
    "w_semantic": 0.7,
    "w_entropy": 0.1,
}

# The observation under test, fixed across T1-T4 so the only varying quantity is
# the unrelated traffic that lands in between.
OBS_REWARD = 0.60
OBS_ADVANTAGE = 0.20          # => baseline at observation time was 0.40


def _prism() -> OnlinePrism:
    return OnlinePrism(
        prior_weights={
            "w_recency": 0.30,
            "w_frequency": 0.20,
            "w_semantic": 0.35,
            "w_entropy": 0.15,
        },
        prior_strength=10.0,
    )


def _drive_ema_to(prism: OnlinePrism, target: float) -> None:
    """Move the EMA near ``target`` using unrelated observations.

    This is the intervening traffic: other requests, nothing to do with the one
    whose outcome arrives later. ``_n`` also advances, which is realistic -- and
    it means eta changes too, so the tests below compare the *sign and
    magnitude of delta_advantage*, which is the quantity the fix governs, rather
    than the raw alpha deltas.
    """
    for _ in range(60):
        prism.observe(target, CONTRIBS)


def _run(ema_target: float | None) -> dict:
    """One observation, optional intervening traffic, then a verified outcome."""
    prism = _prism()
    bridge = OutcomeBridge(prism)
    bridge.cache_observation(
        request_id="A",
        implicit_reward=OBS_REWARD,
        implicit_advantage=OBS_ADVANTAGE,
        contributions=CONTRIBS,
        weights=prism.weights(),
    )
    if ema_target is not None:
        _drive_ema_to(prism, ema_target)
    result = bridge.on_honest_outcome("A", "command_exit", "success", "strong")
    assert result is not None, "a strong verified outcome must correct"
    return result


# ── T1: the reference case ─────────────────────────────────────────────

def test_t1_no_intervening_traffic():
    """Baseline untouched: delta is honest_advantage - implicit_advantage."""
    result = _run(None)
    # honest 0.90 - baseline 0.40 = +0.50 honest advantage; minus the +0.20
    # already applied leaves +0.30 to correct by.
    assert result["baseline_at_observation"] == pytest.approx(0.40)
    assert result["delta_advantage"] == pytest.approx(0.30)
    assert result["delta_advantage"] > 0, "verified success must reinforce"


# ── T2/T3: the invariant ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "ema_target",
    [0.95, 0.75, 0.05, 0.20],
    ids=["drift_up_hard", "drift_up", "drift_down_hard", "drift_down"],
)
def test_t2_t3_correction_is_invariant_to_intervening_ema_drift(ema_target):
    """Same request, same outcome, different unrelated traffic -> same credit."""
    reference = _run(None)
    drifted = _run(ema_target)

    assert drifted["baseline_at_observation"] == reference["baseline_at_observation"]
    assert drifted["delta_advantage"] == pytest.approx(
        reference["delta_advantage"]
    ), f"EMA drift to {ema_target} changed the credit for a fixed outcome"


# ── T4: the measured sign inversion ────────────────────────────────────

def test_t4_verified_success_is_never_applied_as_a_penalty():
    """Regression for the exact observed case: EMA 0.75 gave delta -0.05.

    Asserted on the applied alpha update, not only on the reported number, so a
    future refactor that keeps the diagnostic honest while mutating the wrong
    direction still fails.
    """
    prism = _prism()
    bridge = OutcomeBridge(prism)
    bridge.cache_observation(
        request_id="A",
        implicit_reward=OBS_REWARD,
        implicit_advantage=OBS_ADVANTAGE,
        contributions=CONTRIBS,
        weights=prism.weights(),
    )
    _drive_ema_to(prism, 0.75)

    before = dict(prism._alphas)  # noqa: SLF001
    result = bridge.on_honest_outcome("A", "command_exit", "success", "strong")
    after = dict(prism._alphas)  # noqa: SLF001

    assert result is not None
    assert result["delta_advantage"] > 0, (
        f"verified success scored {result['delta_advantage']:+.4f} -- "
        "the pre-fix value here was -0.05"
    )
    # The dimension that earned the outcome must move up, not down.
    assert after["w_semantic"] > before["w_semantic"]


# ── T5: the self-report authority boundary ─────────────────────────────

def test_t5_weak_self_report_never_mutates_prism():
    prism = _prism()
    bridge = OutcomeBridge(prism)
    bridge.cache_observation(
        request_id="A",
        implicit_reward=OBS_REWARD,
        implicit_advantage=OBS_ADVANTAGE,
        contributions=CONTRIBS,
        weights=prism.weights(),
    )
    before = dict(prism._alphas)  # noqa: SLF001

    assert bridge.on_honest_outcome("A", "agent_self_report", "success", "weak") is None
    # test_result is the strongest mapping there is; weak strength must still
    # veto it. Authority comes from the strength, not from the event name.
    assert bridge.on_honest_outcome("A", "test_result", "passed", "weak") is None

    assert dict(prism._alphas) == before  # noqa: SLF001


# ── T6: the event types production actually emits ──────────────────────

@pytest.mark.parametrize(
    ("event_type", "value", "strength"),
    [
        # Reachable from MCP record_test_result / record_command_exit /
        # record_ci_result / record_edit_outcome via server._record_honest.
        ("test_result", "passed", "strong"),
        ("test_result", "failed", "strong"),
        ("command_exit", "success", "strong"),
        ("command_exit", "failure", "strong"),
        ("ci_result", "passed", "strong"),
        ("edit_outcome", "accepted", "strong"),
        # Medium behavioural inference.
        ("topic_change", "success", "medium"),
        ("retry_event", "failure", "medium"),
    ],
)
def test_t6_every_live_event_type_is_ema_invariant(event_type, value, strength):
    """The invariant must hold for every mapping, not just the one measured."""

    def correct(ema_target: float | None) -> float:
        prism = _prism()
        bridge = OutcomeBridge(prism)
        bridge.cache_observation(
            request_id="A",
            implicit_reward=OBS_REWARD,
            implicit_advantage=OBS_ADVANTAGE,
            contributions=CONTRIBS,
            weights=prism.weights(),
        )
        if ema_target is not None:
            _drive_ema_to(prism, ema_target)
        res = bridge.on_honest_outcome("A", event_type, value, strength)
        assert res is not None, f"{event_type}/{value}/{strength} should map"
        return res["delta_advantage"]

    assert correct(0.95) == pytest.approx(correct(None))
    assert correct(0.05) == pytest.approx(correct(None))


# ── T7: unknown request ────────────────────────────────────────────────

def test_t7_unknown_request_does_not_mutate():
    prism = _prism()
    bridge = OutcomeBridge(prism)
    before = dict(prism._alphas)  # noqa: SLF001

    assert bridge.on_honest_outcome("never-seen", "test_result", "passed", "strong") is None

    assert dict(prism._alphas) == before  # noqa: SLF001
    assert bridge.stats()["cache_misses"] == 1


# ── T8: duplicate delivery ─────────────────────────────────────────────

def test_t8_duplicate_outcome_corrects_once():
    """Current semantics: the cache entry is popped, so a redelivery is a miss.

    Documented rather than changed. Unlike proxy belief attribution -- where a
    failed mutation had to stay retryable -- a lost PRISM correction is a
    recoverable statistical loss, while a double correction silently doubles the
    learning rate for one request.
    """
    prism = _prism()
    bridge = OutcomeBridge(prism)
    bridge.cache_observation(
        request_id="A",
        implicit_reward=OBS_REWARD,
        implicit_advantage=OBS_ADVANTAGE,
        contributions=CONTRIBS,
        weights=prism.weights(),
    )

    first = bridge.on_honest_outcome("A", "test_result", "passed", "strong")
    assert first is not None
    after_first = dict(prism._alphas)  # noqa: SLF001

    assert bridge.on_honest_outcome("A", "test_result", "passed", "strong") is None
    assert dict(prism._alphas) == after_first  # noqa: SLF001
    assert bridge.stats()["corrections_applied"] == 1
    assert bridge.stats()["cache_misses"] == 1


# ── The defect must not be reintroducible ──────────────────────────────

def test_correction_does_not_read_the_live_ema():
    """A direct read of the live EMA in the correction path is the bug itself."""
    import inspect

    source = inspect.getsource(OutcomeBridge.on_honest_outcome)
    live_reads = [
        line
        for line in source.splitlines()
        if "_reward_ema" in line and not line.strip().startswith("#")
    ]
    assert not live_reads, f"live EMA read in correction path: {live_reads}"
