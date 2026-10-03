"""How much externally grounded outcome evidence does Entroly actually receive?

This decides whether verified-only crystallization (Arm B) is even feasible, so
it must be measured on a real event log rather than estimated. An earlier
estimate of this rate was written without evidence and removed; this replaces it.

Denominator discipline, and a correction to the first version of this script.
The intended denominator is "optimize_context calls a cached PRISM observation
could exist for". **That denominator is not recorded anywhere.** `optimize_context`
writes no `request` record; the only thing it can write to this log is a `trace`
event, and only when `_shadow_plan.decomposed_nodes > 0` (`server.py:1525`) --
1 such record in 9,738 lines.

The `request` records that do exist come from `source: "hook"` (4,756) and
`source: "real_e2e"` (42), neither of which exists in this tree, and 4,788 of
4,788 outcomes were written within 10 ms of their matching request. They are
synchronous tool-execution captures, not delayed outcomes for an optimization.

So the ratio reported below is computed against *tool executions*, and is
labelled as such. The honest headline is not a coverage percentage: it is that
of 4,814 recorded strong outcomes, the number OutcomeBridge can act on is
measured directly.

Honest outcome means a RAVS-strong or RAVS-medium externally produced signal.
Agent self-report (`agent_self_report`, and anything with strength `weak`) is
NOT counted, per the OutcomeBridge contract
`_STRENGTH_CONFIDENCE["weak"] = 0.0` / "never correct from self-report".

A second, sharper question is answered at the same time: does the vocabulary in
the log actually match the keys the bridge can act on? A strong outcome whose
`(event_type, value)` pair is absent from `_OUTCOME_REWARD` produces no learning
at all, and would be invisible to a coverage number computed from `strength`
alone.

Run: python research/experiments/honest_outcome_coverage.py [path/to/events.jsonl]
Writes research/ledger/honest_outcome_coverage.json
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "honest_outcome_coverage.json"
DEFAULT_LOG = pathlib.Path.home() / ".entroly" / "ravs" / "events.jsonl"


def main(argv: list[str]) -> int:
    log = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    if not log.is_file():
        print(f"no event log at {log}")
        return 1

    from entroly.ravs.outcome_bridge import _OUTCOME_REWARD, honest_reward

    requests: dict[str, float] = {}       # request_id -> timestamp
    outcomes: list[dict] = []
    malformed = 0
    no_request_id = 0

    with log.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            kind = rec.get("type") or rec.get("kind")
            rid = rec.get("request_id")
            if kind == "request":
                if rid:
                    requests.setdefault(str(rid), float(rec.get("timestamp") or 0.0))
                else:
                    no_request_id += 1
            elif kind == "outcome":
                if rid:
                    outcomes.append(rec)
                else:
                    no_request_id += 1

    # ── Classify every outcome ────────────────────────────────────────
    by_strength: collections.Counter[str] = collections.Counter()
    by_pair: collections.Counter[tuple[str, str]] = collections.Counter()
    actionable_pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    unmapped_strong: collections.Counter[tuple[str, str]] = collections.Counter()

    strong_rids: set[str] = set()
    medium_rids: set[str] = set()
    weak_rids: set[str] = set()
    actionable_rids: set[str] = set()
    delays: list[float] = []
    orphan_outcomes = 0

    for rec in outcomes:
        rid = str(rec["request_id"])
        etype = str(rec.get("event_type") or "")
        value = str(rec.get("value") or "")
        strength = str(rec.get("strength") or "")
        by_strength[strength or "<missing>"] += 1
        by_pair[(etype, value)] += 1

        if strength == "strong":
            strong_rids.add(rid)
        elif strength == "medium":
            medium_rids.add(rid)
        else:
            weak_rids.add(rid)

        # The decisive test: would the bridge actually act on this?
        mapped = honest_reward(etype, value, strength)
        if mapped is not None:
            actionable_rids.add(rid)
            actionable_pairs[(etype, value)] += 1
        elif strength in ("strong", "medium"):
            unmapped_strong[(etype, value)] += 1

        if rid in requests:
            d = float(rec.get("timestamp") or 0.0) - requests[rid]
            if d >= 0:
                delays.append(d)
        else:
            orphan_outcomes += 1

    eligible = len(requests)
    any_outcome = len({str(r["request_id"]) for r in outcomes} & set(requests))
    graded_rids = (strong_rids | medium_rids) & set(requests)
    actionable_in_scope = actionable_rids & set(requests)

    def pct(num: int, den: int) -> float | None:
        return None if den == 0 else round(num / den, 4)

    def quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        s = sorted(values)
        return round(s[min(len(s) - 1, int(q * len(s)))], 3)

    report = {
        "log": str(log),
        "log_bytes": log.stat().st_size,
        "baseline_commit": "9ba8d410",
        "parse": {
            "malformed_lines": malformed,
            "records_without_request_id": no_request_id,
        },
        "denominator": {
            "recorded_request_records": eligible,
            "what_these_actually_are": (
                "tool executions captured by an external hook producer, NOT "
                "optimize_context calls. 4788/4788 outcomes land within 10ms of "
                "their request, so none is a delayed outcome."
            ),
            "optimize_context_calls_recorded": (
                "UNMEASURABLE -- optimize_context writes no request record; it "
                "can only write a trace event when decomposed_nodes > 0"
            ),
        },
        "coverage": {
            "requests_with_any_outcome": any_outcome,
            "requests_with_strong_or_medium": len(graded_rids),
            "requests_with_strong": len(strong_rids & set(requests)),
            "requests_with_medium": len(medium_rids & set(requests)),
            "requests_weak_only": len(
                (weak_rids & set(requests)) - strong_rids - medium_rids
            ),
            "requests_unresolved": eligible - any_outcome,
            "orphan_outcomes_no_matching_request": orphan_outcomes,
            # By strength label alone, against the tool-execution denominator.
            "coverage_by_strength_vs_tool_executions": pct(
                len(graded_rids), eligible
            ),
            # The number that actually matters for learning: the (event_type,
            # value) pair must ALSO be a key of _OUTCOME_REWARD, or
            # honest_reward() returns None and the bridge never touches PRISM.
            "actionable_outcomes_absolute": len(actionable_rids),
            "coverage_actionable_vs_tool_executions": pct(
                len(actionable_in_scope), eligible
            ),
        },
        "delay_seconds": {
            "n": len(delays),
            "median": quantile(delays, 0.50),
            "p95": quantile(delays, 0.95),
            "max": round(max(delays), 3) if delays else None,
            "mean": round(statistics.fmean(delays), 3) if delays else None,
        },
        "strength_distribution": dict(by_strength.most_common()),
        "event_pair_distribution": {
            f"{a}/{b}": n for (a, b), n in by_pair.most_common(25)
        },
        "actionable_pairs": {
            f"{a}/{b}": n for (a, b), n in actionable_pairs.most_common()
        },
        "strong_or_medium_pairs_the_bridge_cannot_act_on": {
            f"{a}/{b}": n for (a, b), n in unmapped_strong.most_common(25)
        },
        "bridge_reward_table_keys": sorted(
            f"{a}/{b}" for a, b in _OUTCOME_REWARD
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "bridge_reward_table_keys"}, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
