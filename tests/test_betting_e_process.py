"""Mathematical property tests for ``BettingEProcess`` (Ville's inequality)."""

from __future__ import annotations

import random
import pytest

from entroly.e_value import BettingEProcess


def test_betting_e_process_validates_inputs():
    with pytest.raises(ValueError):
        BettingEProcess(null_risk=0.0)
    with pytest.raises(ValueError):
        BettingEProcess(delta=1.0)
    with pytest.raises(ValueError):
        BettingEProcess(prior_strength=0.0)
    with pytest.raises(ValueError):
        BettingEProcess(max_bet_fraction=1.0)

    proc = BettingEProcess(null_risk=0.1, delta=0.05)
    with pytest.raises(ValueError):
        proc.update(-0.01)
    with pytest.raises(ValueError):
        proc.update(1.01)


def test_betting_e_process_stays_nonnegative_and_supermartingale_under_null():
    # Under H_0: Bernoulli(null_risk=0.10), Ville's inequality bounds the
    # probability of EVER crossing 1/delta=20 across arbitrary stopping times
    # by <= delta = 0.05. Across 400 independent streams of 150 steps each,
    # empirical false-rejection rate should stay comfortably under 0.05.
    rng = random.Random(20261126)
    null_risk = 0.10
    delta = 0.05
    trials = 400
    horizon = 150
    false_rejections = 0

    for _ in range(trials):
        proc = BettingEProcess(null_risk=null_risk, delta=delta)
        for _ in range(horizon):
            loss = 1.0 if rng.random() < null_risk else 0.0
            rec = proc.update(loss)
            assert rec.wealth >= 0.0
            assert 0.0 <= rec.bet_lambda <= proc.max_bet
            assert 0.0 < rec.anytime_p <= 1.0
            if rec.rejected:
                false_rejections += 1
                break

    # Expected <= 0.05; allow finite-sample margin (<= 0.07) to stay deterministic.
    assert false_rejections / trials <= 0.07


def test_betting_e_process_rejects_sustained_shift_above_null_risk():
    rng = random.Random(42)
    proc = BettingEProcess(null_risk=0.10, delta=0.05)
    for _ in range(80):
        # True loss rate 0.45 >> null_risk 0.10.
        loss = 1.0 if rng.random() < 0.45 else 0.0
        proc.update(loss)

    summary = proc.summary()
    assert proc.rejected is True
    assert proc.max_wealth >= 20.0
    assert proc.anytime_p <= 0.05
    assert summary["method"] == "predictable-betting-test-supermartingale-ville"
    assert summary["steps"] == 80


def test_first_step_bet_is_zero_predictable():
    proc = BettingEProcess(null_risk=0.2, delta=0.05)
    assert proc.next_bet() == 0.0
    rec = proc.update(1.0)
    assert rec.bet_lambda == 0.0
    assert rec.wealth == 1.0
    # After observing loss=1.0 > null_risk=0.2, next predictable bet becomes positive.
    assert proc.next_bet() > 0.0
