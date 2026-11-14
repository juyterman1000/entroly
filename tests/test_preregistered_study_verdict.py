"""A small/model-free pilot must never certify the 400-task study."""
from __future__ import annotations

import pytest

from benchmarks.preregistered_study_verdict import inspect, nonnegative_decimal


def study(count: int = 8, *, invoice: str = "NOT_TESTED") -> dict:
    rows = []
    for i in range(count):
        split = "dev" if i < count // 2 else "holdout"
        repo = f"{split}-repository-{i % 2}"
        for arm in ("raw", "compress", "closed_loop"):
            rows.append({
                "repo": repo, "task_id": f"task-{i}", "split": split,
                "arm": arm, "success": True, "oracle_status": "verified_fail_to_pass",
                "input_tokens": 500 if arm != "raw" else 1000,
                "output_tokens": 50, "seed": 17, "model_digest": "known-model-digest",
                "billed_usd": "0.001",
            })
    return {
        "schema": "entroly.agentic-study.v1",
        "rows": rows, "preregistration_sha256": "frozen-protocol",
        "invoice_status": invoice, "actual_spend_usd": "0", "budget_usd": "100",
    }


def test_empty_dataset_is_unverified():
    result = inspect({"schema": "entroly.agentic-study.v1", "rows": []})
    assert result["status"] == "not_certified"
    assert result["effective_tasks"] == 0


def test_eight_task_pilot_cannot_pass_sample_size_or_invoice_gate():
    result = inspect(study())
    assert result["status"] == "not_certified"
    assert result["effective_tasks"] == 8
    assert any("400" in reason for reason in result["reasons"])
    assert any("invoices" in reason for reason in result["reasons"])


def test_fake_400_task_data_still_requires_real_invoice():
    result = inspect(study(400))
    assert result["status"] == "not_certified"
    assert result["holdout_tasks"] == 200
    assert result["holdout_median_input_reduction_pct"] == 50
    assert result["holdout_cluster_bootstrap_delta_ci95"] == [0.0, 0.0]
    assert any("invoices" in reason for reason in result["reasons"])


def test_even_self_reported_400_task_invoices_do_not_auto_certify():
    result = inspect(study(400, invoice="attested_per_request_reconciled"))
    assert result["status"] == "external_review_required"
    assert result["limitations"]


def test_repository_leakage_rejected():
    payload = study()
    payload["rows"][0]["split"] = "holdout"
    with pytest.raises(ValueError, match="repository|split"):
        inspect(payload)


def test_repeated_task_or_seed_cannot_inflate_independent_n():
    payload = study()
    payload["rows"].append(payload["rows"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        inspect(payload)
    payload = study()
    payload["rows"][1]["seed"] = 44
    with pytest.raises(ValueError, match="seed"):
        inspect(payload)


def test_missing_arm_or_different_model_rejected():
    payload = study()
    payload["rows"].pop()
    with pytest.raises(ValueError, match="three paired arms"):
        inspect(payload)
    payload = study()
    payload["rows"][0]["model_digest"] = "other-model"
    with pytest.raises(ValueError, match="model digest"):
        inspect(payload)


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-Infinity", "-1", "money"])
def test_invalid_invoice_costs_fail_closed(invalid):
    with pytest.raises(ValueError, match="nonnegative finite"):
        nonnegative_decimal(invalid)


def test_raw_quality_shift_falsifies_noninferiority_claim():
    payload = study(400)
    for row in payload["rows"]:
        if row["split"] == "holdout" and row["arm"] == "raw":
            row["success"] = False
    result = inspect(payload)
    assert any("F1" in reason for reason in result["reasons"])


def test_failed_noninferiority_cannot_pass_even_with_positive_token_savings():
    payload = study(400)
    for row in payload["rows"]:
        if row["split"] == "holdout" and row["arm"] == "compress":
            row["success"] = False
    result = inspect(payload)
    assert any("N1" in reason and "confidence" not in reason for reason in result["reasons"])


def test_budget_overrun_is_not_hidden():
    payload = study(400)
    payload["actual_spend_usd"] = "101"
    result = inspect(payload)
    assert any("B1" in reason for reason in result["reasons"])
