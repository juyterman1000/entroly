from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "benchmarks" / "context-arena" / "phase2.py"
spec = importlib.util.spec_from_file_location("context_arena_phase2", MODULE)
assert spec and spec.loader
phase2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2)


class Good:
    name = "good"
    def prepare_context(self, task, token_budget):
        return "useful evidence"


class Blocked:
    name = "blocked"
    def prepare_context(self, task, token_budget):
        raise ModuleNotFoundError("optional competitor unavailable")


class Oversize:
    name = "oversize"
    def prepare_context(self, task, token_budget):
        return "x" * 100


def test_valid_arm_is_claimable():
    row = phase2.prepare_arm(Good(), {"task_id": "t1"}, 20)
    assert row.status is phase2.ArmStatus.OK
    assert row.valid_measurement
    assert phase2.comparison_is_claimable([row])


def test_missing_competitor_is_blocked_not_a_loss():
    row = phase2.prepare_arm(Blocked(), {"task_id": "t1"}, 20)
    assert row.status is phase2.ArmStatus.BLOCKED
    assert not row.valid_measurement
    assert not phase2.comparison_is_claimable([row])


def test_budget_violation_invalidates_comparison():
    good = phase2.prepare_arm(Good(), {"task_id": "t1"}, 20)
    oversize = phase2.prepare_arm(Oversize(), {"task_id": "t1"}, 5)
    assert oversize.status is phase2.ArmStatus.BUDGET_VIOLATION
    assert not phase2.comparison_is_claimable([good, oversize])


def test_comparison_requires_every_arm_to_be_valid():
    good = phase2.prepare_arm(Good(), {"task_id": "t1"}, 20)
    blocked = phase2.prepare_arm(Blocked(), {"task_id": "t1"}, 20)
    assert phase2.comparison_is_claimable([good, blocked]) is False


def test_empty_comparison_is_not_claimable():
    assert phase2.comparison_is_claimable([]) is False
