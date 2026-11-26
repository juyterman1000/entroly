"""Mathematical property tests for ``BettingEProcess`` (ONS + Ville's inequality)."""

from __future__ import annotations

import math
import random
import pytest

from entroly.e_value import BettingEProcess


def test_betting_e_process_validates_inputs():
    with pytest.raises(ValueError):
        BettingEProcess(null_risk=0.0)
    with pytest.raises(ValueError):
        BettingEProcess(delta=1.0)
    with pytest.raises(ValueError):
        BettingEProcess(max_bet_fraction=1.0)

    proc = BettingEProcess(null_risk=0.1, delta=0.05)
    with pytest.raises(ValueError):
        proc.update(-0.01)
    with pytest.raises(ValueError):
        proc.update(1.01)


def test_ville_anytime_false_rejection_bound_under_continuous_optional_stopping():
    """Under H_0: E[loss_t | F_{t-1}] <= null_risk, Ville's inequality bounds
    P(exists t >= 1 : M_t >= 1/delta) <= delta across arbitrary stopping times."""
    rng = random.Random(20261126)
    null_risk = 0.10
    delta = 0.05
    trials = 500
    horizon = 200
    ever_crossed = 0

    for _ in range(trials):
        proc = BettingEProcess(null_risk=null_risk, delta=delta)
        for _ in range(horizon):
            loss = 1.0 if rng.random() < null_risk else 0.0
            rec = proc.update(loss)
            # Factor m_t >= 0.5 ensures wealth stays strictly positive.
            assert rec.multiplier >= 0.5 - 1e-12
            assert rec.wealth > 0.0
            assert 0.0 <= rec.bet_lambda <= proc.max_bet + 1e-12
            if rec.rejected:
                ever_crossed += 1
                break

    # Ville's upper bound is delta = 0.05 (typically much more conservative, ~0.01-0.03).
    assert ever_crossed / trials <= 0.065


def test_ons_exponential_wealth_growth_under_alternative():
    """When true loss rate p=0.45 > null_risk=0.10, ONS achieves positive
    log-wealth drift approaching the optimal Kelly rate and rejects rapidly."""
    rng = random.Random(42)
    proc = BettingEProcess(null_risk=0.10, delta=0.01)
    for _ in range(100):
        loss = 1.0 if rng.random() < 0.45 else 0.0
        proc.update(loss)

    summary = proc.summary()
    assert proc.rejected is True
    assert proc.max_wealth >= 100.0
    assert proc.anytime_p <= 0.01
    assert math.log(proc.wealth) > 4.6
    assert summary["method"] == "ons-predictable-betting-supermartingale-ville"


def test_predictability_no_lookahead_at_step_one():
    proc = BettingEProcess(null_risk=0.20, delta=0.05)
    assert proc.next_bet() == 0.0
    # Even an adversarial loss=1.0 on step 1 cannot change wealth at step 1,
    # proving lambda_1 was fixed strictly prior to seeing loss_1.
    rec1 = proc.update(1.0)
    assert rec1.bet_lambda == 0.0
    assert rec1.wealth == 1.0
    assert proc.next_bet() > 0.0
