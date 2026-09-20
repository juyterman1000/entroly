from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .types import CollisionReport, RelationVector


class Decision(str, Enum):
    SELECT_LEFT = "select_left"
    SELECT_RIGHT = "select_right"
    RETAIN_BOTH = "retain_both"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class SemanticDecision:
    decision: Decision
    reason: str


def decide_precalibration(left: RelationVector, right: RelationVector, collision: CollisionReport) -> SemanticDecision:
    """Fail-closed policy before calibration exists.

    Uncalibrated semantic signals may trigger caution but never destructive pruning.
    """
    if collision.escalated:
        return SemanticDecision(Decision.RETAIN_BOTH, "semantic collision without calibrated override")
    return SemanticDecision(Decision.ABSTAIN, "no calibrated semantic override available")
