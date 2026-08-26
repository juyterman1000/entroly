"""Three surfaces presented unverified state as verified.

Each was found by driving the code rather than reading it, and each fails in
the direction that lets weaker evidence pass for stronger.
"""

from __future__ import annotations

import json

from entroly.ravs.router import _load_cells_from_log
from entroly.vault import BeliefArtifact


def _log(tmp_path, events):
    p = tmp_path / "events.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return str(p)


def test_self_reported_outcomes_do_not_move_the_posterior(tmp_path) -> None:
    """Invariant 1 of `ravs/events.py`, enforced where it actually matters.

    `include_in_default_training=False` marks an unverified self-report. The
    cell loader that builds the posterior never read the flag, so twelve
    self-reported passes carried a cell to ci_lo 0.951 and authorised routing
    to the cheapest model -- an agent could mint the evidence that downgraded
    its own review. `derive_label` already returned "unknown" for the same
    event, so the guard existed and was simply bypassed.
    """
    events = [
        {
            "kind": "outcome",
            "event_type": "test",
            "tool": "pytest",
            "value": "pass",
            "strength": "weak",
            "source": "agent_self_report",
            "include_in_default_training": False,
        }
        for _ in range(12)
    ]
    assert _load_cells_from_log(_log(tmp_path, events)) == {}


def test_events_without_the_flag_are_still_counted(tmp_path) -> None:
    """Only an explicit False excludes; older events predate the field."""
    events = [
        {"kind": "outcome", "event_type": "test", "tool": "pytest", "value": "pass"}
        for _ in range(4)
    ]
    cells = _load_cells_from_log(_log(tmp_path, events))
    assert cells["test/pytest"]["passes"] == 4


def test_prior_is_not_fitted_to_the_evidence_it_scores(tmp_path) -> None:
    """The prior was estimated from the same log it then scored.

    With no failures anywhere `global_mean` was 1.0, so beta collapsed to a 0.1
    floor against alpha 2.0 -- a prior mean of 0.952 before any observation, and
    two passes were enough to clear a 0.80 confidence threshold. That made the
    threshold decorative. Jeffreys Beta(1/2, 1/2) is independent of the data.
    """
    events = [
        {"kind": "outcome", "event_type": "test", "tool": "pytest", "value": "pass"}
        for _ in range(10)
    ]
    cell = _load_cells_from_log(_log(tmp_path, events))["test/pytest"]

    # alpha = prior + passes, beta = prior + failures.
    assert cell["alpha"] == 0.5 + 10
    assert cell["beta"] == 0.5, "a run with no failures must not collapse the prior"


def test_unsourced_belief_is_not_rendered_as_though_it_had_a_source() -> None:
    """An empty source list rendered as `- unknown`.

    A reader cannot tell that apart from a real citation, so a belief carrying
    `status: verified` and `confidence: 0.99` was stored looking sourced when
    nothing backed it.
    """
    unsourced = BeliefArtifact(
        entity="auth.verify_token", status="verified", confidence=0.99, sources=[]
    ).to_markdown()

    assert "- unknown" not in unsourced, "an absent source must not be invented"
    assert "sources:" in unsourced

    sourced = BeliefArtifact(entity="auth", sources=["src/auth.py:10"]).to_markdown()
    assert "  - src/auth.py:10" in sourced, "real citations must still render"
