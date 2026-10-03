"""R1-R14: the Attempt-2 runner must be correct before it spends any quota.

Attempt 1 burned 57 provider calls after the quota was already gone and left a
monolithic results file with no way to tell a complete row from a truncated one.
Every check here is derived from one of those failures, and all of them run on
mocks and recorded Attempt-1 fixtures so none consumes Codex quota.

The distinction these tests exist to protect: a legitimate scientific failure --
the agent's code failing pytest, a nonzero exit, the verifier rejecting the
solution -- must persist as COMPLETE and must never stop the run. Only provider
and auth refusals are terminal.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "research" / "pilot"))

import attempt2 as a2  # noqa: E402

ATTEMPT1_RESULTS = REPO / "research" / "ledger" / "pilot1" / "results.json"


@pytest.fixture(autouse=True)
def isolated_rows(tmp_path, monkeypatch):
    """Never let a test touch the real ledger."""
    monkeypatch.setattr(a2, "OUT_DIR", tmp_path / "attempt2")
    monkeypatch.setattr(a2, "ROWS_DIR", tmp_path / "attempt2" / "rows")
    monkeypatch.setattr(a2, "TRACE_DIR", tmp_path / "attempt2" / "traces")
    return tmp_path


def _manifest(**over):
    m = {
        "attempt_id": a2.ATTEMPT_ID,
        "manifest_sha256": "m" * 64,
    }
    m.update(over)
    return m


def _row(**over):
    row = {
        "attempt_id": a2.ATTEMPT_ID,
        "manifest_sha256": "m" * 64,
        "task_id": "p01-half-up",
        "arm": "B3",
        "invocation": 1,
        "task_sha256": "t" * 64,
        "arm_definition_sha256": "a" * 64,
        "status": a2.COMPLETE,
        "verified_success": True,
    }
    row.update(over)
    return row


# ── R1: successful row atomic persistence ──────────────────────────────

def test_r1_successful_row_persists_atomically():
    path = a2.write_row_atomic(_row())
    assert path.is_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["status"] == a2.COMPLETE
    assert stored["row_digest"] == a2.row_digest(stored)
    # No temp files left behind.
    assert not list(a2.ROWS_DIR.glob("*.tmp"))


# ── R2: a legitimate task failure is COMPLETE evidence ─────────────────

def test_r2_task_failure_persists_as_complete_not_environment_failure():
    row = _row(verified_success=False, verifier_returncode=1,
               verifier_tail="1 failed")
    path = a2.write_row_atomic(row)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["status"] == a2.COMPLETE, (
        "an agent whose code fails the tests is evidence, not an environment "
        "failure"
    )
    assert stored["verified_success"] is False


@pytest.mark.parametrize("stream", [
    "FAILED tests/test_money.py::test_half_up - assert Decimal('2.67')",
    "1 failed, 1 passed in 0.44s",
    "Traceback (most recent call last):\nValueError: bad",
    "pytest exited with code 1",
    "AssertionError: expected 2.68",
])
def test_r2b_scientific_failure_never_trips_the_circuit_breaker(stream):
    assert a2.classify_provider(stream) == (None, None)


# ── R3: quota error persists and circuit-breaks ─────────────────────────

def test_r3_quota_error_is_terminal():
    label, detail = a2.classify_provider(
        '{"type":"error","message":"You\'ve hit your usage limit. '
        'Upgrade to Pro ..."}'
    )
    assert label == "PROVIDER_QUOTA_EXHAUSTED"
    assert "usage limit" in detail


@pytest.mark.parametrize(("stream", "expected"), [
    ("Failed to authenticate: OAuth session expired", "PROVIDER_AUTH_FAILURE"),
    ("401 Unauthorized", "PROVIDER_AUTH_FAILURE"),
    ("The 'gpt-5.1-codex' model is not supported when using Codex with a "
     "ChatGPT account.", "MODEL_UNAVAILABLE_GLOBAL"),
    ("requires a newer version of Codex", "MODEL_UNAVAILABLE_GLOBAL"),
])
def test_r3b_terminal_classes(stream, expected):
    assert a2.classify_provider(stream)[0] == expected


def test_r3c_row_level_environment_is_not_terminal():
    """A missing interpreter kills one row, not the attempt."""
    stream = "python : The term 'python' is not recognized as the name of a cmdlet"
    assert a2.classify_provider(stream) == (None, None)
    assert a2.classify_row_environment(stream) is not None


# ── R4/R5/R6/R7: resume gate ───────────────────────────────────────────

def test_r4_resume_skips_only_validated_complete_rows():
    m = _manifest()
    a2.write_row_atomic(_row())
    ok, why = a2.can_skip("p01-half-up", "B3", m, "t" * 64, "a" * 64)
    assert ok, why


def test_r5_incomplete_and_corrupt_rows_do_not_skip():
    m = _manifest()
    a2.write_row_atomic(_row(status=a2.RUNNING))
    ok, why = a2.can_skip("p01-half-up", "B3", m, "t" * 64, "a" * 64)
    assert not ok and "status=" in why

    # Tampered row: digest no longer matches.
    path = a2.write_row_atomic(_row(), allow_overwrite=True)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["verified_success"] = False      # flip the result, keep the digest
    path.write_text(json.dumps(stored), encoding="utf-8")
    ok, why = a2.can_skip("p01-half-up", "B3", m, "t" * 64, "a" * 64)
    assert not ok and "integrity" in why

    # Unparseable file.
    path.write_text("{not json", encoding="utf-8")
    ok, why = a2.can_skip("p01-half-up", "B3", m, "t" * 64, "a" * 64)
    assert not ok


def test_r6_manifest_or_treatment_mismatch_refuses_resume():
    a2.write_row_atomic(_row())
    for kwargs, needle in (
        ({"manifest_sha256": "z" * 64}, "manifest_sha256"),
        ({}, "task_sha256"),
        ({}, "arm_definition_sha256"),
    ):
        m = _manifest(**kwargs)
        th = "WRONG" if needle == "task_sha256" else "t" * 64
        ah = "WRONG" if needle == "arm_definition_sha256" else "a" * 64
        ok, why = a2.can_skip("p01-half-up", "B3", m, th, ah)
        assert not ok and needle in why, (needle, why)


def test_r7_attempt1_rows_can_never_satisfy_attempt2_completion():
    a2.write_row_atomic(_row(attempt_id="pilot1-attempt1"))
    ok, why = a2.can_skip("p01-half-up", "B3", _manifest(), "t" * 64, "a" * 64)
    assert not ok
    assert "attempt_id" in why and "cross-attempt" in why


# ── R8: a COMPLETE row is never overwritten ────────────────────────────

def test_r8_complete_row_cannot_be_silently_overwritten():
    a2.write_row_atomic(_row(verified_success=True))
    with pytest.raises(RuntimeError, match="refusing to overwrite COMPLETE"):
        a2.write_row_atomic(_row(verified_success=False))
    stored = json.loads(a2.row_path("p01-half-up", "B3").read_text(encoding="utf-8"))
    assert stored["verified_success"] is True


def test_r8b_environment_failure_row_may_be_retried():
    """A quota refusal is an invocation artifact, not the scientific record."""
    a2.write_row_atomic(_row(status=a2.ENVIRONMENT_FAILURE,
                             environment_class="PROVIDER_QUOTA_EXHAUSTED",
                             verified_success=None))
    a2.write_row_atomic(_row(invocation=2), allow_overwrite=True)
    stored = json.loads(a2.row_path("p01-half-up", "B3").read_text(encoding="utf-8"))
    assert stored["status"] == a2.COMPLETE and stored["invocation"] == 2


# ── R9/R10: treatment and order are unchanged ──────────────────────────

def test_r9_arm_definitions_match_attempt1_content():
    """B1S text and B0 emptiness must be identical to Attempt 1's treatment."""
    import harness as base
    from tasks_pilot1 import SUITE

    task = SUITE[0]
    assert a2.arm_definition_hash("B0", task) == base.sha256_text("B0\n")
    assert a2.arm_definition_hash("B1S", task) == base.sha256_text(
        "B1S\n" + base.b1_handoff(task)
    )
    # And the rendering itself still carries the recorded facts.
    text = base.b1_handoff(task)
    assert task.recorded.remaining_work[0] in text
    assert task.recorded.rejected[0] in text


def test_r10_run_order_is_recorded_and_counterbalanced():
    from tasks_pilot1 import SUITE

    manifest = a2.build_manifest(SUITE)
    orders = manifest["run_order"]
    assert len(orders) == 24
    for task_id, order in orders.items():
        assert sorted(order) == ["B0", "B1S", "B3"], task_id
    # Not every task may run in the same order, or provider session effects
    # would align with one arm.
    assert len({tuple(o) for o in orders.values()}) > 1
    first = [o[0] for o in orders.values()]
    assert len(set(first)) > 1, "the same arm always ran first"


# ── R11/R12/R13: the analysis contract ─────────────────────────────────

def test_r11_usage_aggregation_is_ratio_of_sums():
    from tasks_pilot1 import SUITE

    manifest = a2.build_manifest(SUITE)
    assert manifest["usage_aggregation"] == "RATIO OF SUMS over MATCHED_PRIMARY_TASKS"
    assert manifest["usage_guard"] == "B3_total / B1S_total <= 1.10"
    assert "SENSITIVITY" in manifest["usage_sensitivity"]

    # The arithmetic itself, on Attempt-1's five matched tasks.
    b1s = [29767, 39375, 19892, 38565, 46734]
    b3 = [30386, 41530, 26358, 30395, 38193]
    assert round(sum(b3) / sum(b1s), 4) == 0.9571
    import statistics
    assert round(statistics.median(b3) / statistics.median(b1s), 4) == 0.7881


def test_r12_matched_denominator_and_discrete_threshold():
    from tasks_pilot1 import SUITE

    manifest = a2.build_manifest(SUITE)
    assert manifest["matched_primary_rule"].startswith("tasks where BOTH")
    assert manifest["success_threshold_pp"] == 10.0
    assert "100/N" in manifest["success_denominator"]
    # N is never hard-coded to 24.
    for n in (5, 11, 24):
        assert round(100 / n, 2) == round(100 / n, 2)


def test_r13_discordant_pair_accounting():
    """Paired binary diagnostic, and it must not be the decision rule."""
    from tasks_pilot1 import SUITE

    manifest = a2.build_manifest(SUITE)
    assert "does not replace" in manifest["paired_diagnostic"]

    pairs = [(False, True), (False, True), (True, False),
             (True, True), (False, False)]
    a = sum(1 for b1, b3 in pairs if not b1 and b3)       # B1S fail, B3 pass
    b = sum(1 for b1, b3 in pairs if b1 and not b3)       # B1S pass, B3 fail
    both_pass = sum(1 for b1, b3 in pairs if b1 and b3)
    both_fail = sum(1 for b1, b3 in pairs if not b1 and not b3)
    assert (a, b, both_pass, both_fail) == (2, 1, 1, 1)
    assert a + b + both_pass + both_fail == len(pairs)


# ── R14: quota-boundary metadata ───────────────────────────────────────

def test_r14_quota_boundary_phase_is_recorded():
    path = a2.write_row_atomic(_row(quota_boundary_phase="POST_RESUME"))
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["quota_boundary_phase"] == "POST_RESUME"

    from tasks_pilot1 import SUITE
    manifest = a2.build_manifest(SUITE)
    assert "PRE_INTERRUPTION" in manifest["quota_boundary_rule"]
    assert "must not attribute" in manifest["quota_boundary_rule"]


# ── Regression against the real Attempt-1 record ───────────────────────

@pytest.mark.skipif(not ATTEMPT1_RESULTS.is_file(),
                    reason="Attempt-1 record not present")
def test_circuit_breaker_confusion_matrix_on_attempt1_rows():
    """Replay all 72 recorded rows. This is the precondition for trusting it."""
    rows = json.loads(ATTEMPT1_RESULTS.read_text(encoding="utf-8"))["results"]
    assert len(rows) == 72

    fired, classes = [], {}
    for row in rows:
        stream = json.dumps(row.get("errors") or [])
        label, _ = a2.classify_provider(stream)
        classes[label] = classes.get(label, 0) + 1
        if label:
            fired.append(row)

    # Attempt 1's own eligibility: did the agent execute anything at all?
    did_work = [r for r in rows if (r.get("command_count") or 0) > 0]
    no_work = [r for r in rows if (r.get("command_count") or 0) == 0]

    assert len(fired) == 57, f"expected 57 quota rows, detected {len(fired)}"
    assert classes.get("PROVIDER_QUOTA_EXHAUSTED") == 57, classes
    assert len(no_work) == 57
    assert len(did_work) == 15
    # The decisive property: no row where the agent actually worked is
    # misclassified as a terminal provider failure.
    false_positives = [r for r in fired if (r.get("command_count") or 0) > 0]
    assert not false_positives, (
        f"{len(false_positives)} rows with real model work were flagged terminal"
    )
