#!/usr/bin/env python3
"""Falsifiable, fail-closed decision on the 400-task cross-repository protocol.

This is a *gate*, not an experiment. No task success, real invoice or source
authenticity is inferred or fabricated. Passing all arithmetic criteria is
still a CANDIDATE for external audit; automatic production certification is
never issued based on user-provided JSON.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from random import Random
from statistics import median

ARMS = ("raw", "compress", "closed_loop")


def nonnegative_decimal(x: object) -> Decimal:
    try:
        value = Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid nonnegative finite dollar amount") from exc
    if not value.is_finite() or value < 0:
        raise ValueError("invalid nonnegative finite dollar amount")
    return value


def clustered_ci95(tasks: list[dict], *, seed: int = 17, n: int = 2000) -> list[float]:
    groups = defaultdict(list)
    for task in tasks:
        groups[task["repo"]].append(task)
    repos = sorted(groups)
    if len(repos) < 2:
        raise ValueError("independent repository clusters required")
    rng = Random(seed)
    deltas = []
    for _ in range(n):
        sample = []
        for _ in repos:
            pool = groups[repos[rng.randrange(len(repos))]]
            sample.extend(pool[rng.randrange(len(pool))] for _ in pool)
        deltas.append(sum(int(row["compress"]["success"]) - int(row["raw"]["success"])
                          for row in sample) / len(sample))
    deltas.sort()
    return [deltas[int((n-1)*0.025)], deltas[int((n-1)*0.975)]]


def inspect(study: dict) -> dict:
    if study.get("schema") != "entroly.agentic-study.v1":
        raise ValueError("unsupported study schema")
    rows = study.get("rows")
    if not isinstance(rows, list):
        raise ValueError("explicit rows are required")
    by_task = {}
    repo_split = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed study row")
        key = (row.get("repo"), row.get("task_id"))
        if not all(isinstance(x, str) and x.strip() for x in key):
            raise ValueError("immutable repo and task IDs required")
        arm = row.get("arm")
        if arm not in ARMS or type(row.get("success")) is not bool:
            raise ValueError("invalid arm or nonboolean test result")
        split = row.get("split")
        if split not in ("dev", "holdout"):
            raise ValueError("invalid repository split")
        if row.get("oracle_status") != "verified_fail_to_pass":
            raise ValueError("task lacks independently verified fail-to-pass oracle")
        for field in ("input_tokens", "output_tokens"):
            token_count = row.get(field)
            if type(token_count) is not int or token_count < (1 if field == "input_tokens" else 0):
                raise ValueError("actual model-reported token counts are mandatory")
        if type(row.get("seed")) is not int or not row.get("model_digest"):
            raise ValueError("paired seed and pinned model digest required")
        if key[0] in repo_split and repo_split[key[0]] != split:
            raise ValueError("same repository cannot appear in both dev and holdout")
        repo_split[key[0]] = split
        task = by_task.setdefault(key, {"repo": key[0], "split": split})
        if task["split"] != split or arm in task:
            raise ValueError("duplicate task/arm or changed split")
        task[arm] = row
    if not by_task:
        return {"schema": "entroly.agentic-study-verdict.v1", "status": "not_certified",
                "effective_tasks": 0, "reasons": ["no independently tested tasks"]}
    for task in by_task.values():
        if any(arm not in task for arm in ARMS):
            raise ValueError("every task requires all three paired arms")
        if len({task[arm]["seed"] for arm in ARMS}) != 1:
            raise ValueError("paired arms have inconsistent random seed")
        if len({task[arm]["model_digest"] for arm in ARMS}) != 1:
            raise ValueError("paired arms have inconsistent model digest")
    tasks = list(by_task.values())
    dev = [t for t in tasks if t["split"] == "dev"]
    hold = [t for t in tasks if t["split"] == "holdout"]
    reasons = []
    if len(tasks) < 400 or len(dev) < 200 or len(hold) < 200:
        reasons.append("N1: fewer than 400 distinct tasks and 200 per partition")
    dr, hr = {t["repo"] for t in dev}, {t["repo"] for t in hold}
    if len(dr) < 2 or len(hr) < 2:
        reasons.append("N1: insufficient independent repo clusters")
    if not study.get("preregistration_sha256"):
        reasons.append("B1: no frozen preregistration digest")
    raw_dev = (sum(t["raw"]["success"] for t in dev) / len(dev)) if dev else None
    raw_hold = (sum(t["raw"]["success"] for t in hold) / len(hold)) if hold else None
    if raw_dev is None or raw_hold is None or abs(raw_dev - raw_hold) > 0.10:
        reasons.append("F1: baseline dev/holdout quality shift exceeds 10pp or is unknown")
    reductions = [100 * (1 - t["compress"]["input_tokens"] / t["raw"]["input_tokens"])
                  for t in hold]
    median_reduction = median(reductions) if reductions else None
    if median_reduction is None or median_reduction < 30:
        reasons.append("N1: <30% median measured input-token reduction")
    ci = clustered_ci95(hold) if len(hr) >= 2 else None
    if ci is None or ci[0] < -0.03:
        reasons.append("N1: holdout cluster-bootstrap lower bound fails -3pp noninferiority")
    if study.get("invoice_status") != "attested_per_request_reconciled":
        reasons.append("D1: provider-authenticated, per-request invoices unavailable")
    try:
        spend, budget = nonnegative_decimal(study["actual_spend_usd"]), nonnegative_decimal(study["budget_usd"])
        if spend > budget:
            reasons.append("B1: paid-provider budget exceeded")
    except (ValueError, KeyError):
        reasons.append("B1: no independently measured spend and locked budget")
    if not reasons:
        try:
            costs = {arm: sum((nonnegative_decimal(t[arm]["billed_usd"]) for t in hold),
                              Decimal(0)) for arm in ARMS}
            successes = {arm: sum(t[arm]["success"] for t in hold) for arm in ARMS}
        except (KeyError, ValueError):
            reasons.append("D1: per-arm invoice-backed billed cost missing")
        else:
            if not successes["compress"] or not successes["closed_loop"]:
                reasons.append("D1: no successful tasks for cost/success calculation")
            elif not (
                (successes["closed_loop"] >= successes["compress"] and
                 costs["closed_loop"] / successes["closed_loop"] < costs["compress"] / successes["compress"])
                or (successes["closed_loop"] > successes["compress"] and
                    costs["closed_loop"] <= costs["compress"])
            ):
                reasons.append("D1: closed-loop did not dominate cost/success frontier")
    return {
        "schema": "entroly.agentic-study-verdict.v1",
        "status": "not_certified" if reasons else "external_review_required",
        "reasons": reasons,
        "effective_tasks": len(tasks), "dev_tasks": len(dev), "holdout_tasks": len(hold),
        "dev_repo_count": len(dr), "holdout_repo_count": len(hr),
        "raw_dev_success": raw_dev, "raw_holdout_success": raw_hold,
        "holdout_median_input_reduction_pct": median_reduction,
        "holdout_cluster_bootstrap_delta_ci95": ci,
        "limitations": [
            "Even a passing metric report cannot authenticate invoices or source code",
            "Repeated seeds are not counted as independent tasks",
            "External audit and task-level provenance review remain mandatory",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    opts = parser.parse_args()
    result = inspect(json.loads(opts.study.read_text(encoding="utf-8")))
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "tasks": result["effective_tasks"],
                      "reasons": result["reasons"]}))
    return 0 if result["status"] == "external_review_required" else 1


if __name__ == "__main__":
    raise SystemExit(main())
