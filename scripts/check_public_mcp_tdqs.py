"""Fail on TDQS public-catalog warnings, apart from one verified cost-only candidate.

TDQS 0.2.0's shadow prefilter compares invocation costs alone. It flags the
review submission beside the file reader even though one writes a review record
and the other only reads source text. Keep that exact candidate visible while
rejecting every other warning or error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def check_report(report: dict[str, Any]) -> list[str]:
    findings = report.get("findings")
    candidates = report.get("shadowCandidates")
    tools = report.get("tools")
    server = report.get("server")
    if not isinstance(findings, list) or not isinstance(candidates, list) or not isinstance(tools, list):
        return ["TDQS report is missing findings, shadowCandidates, or tools"]
    if (
        not isinstance(server, dict)
        or server.get("name") != "entroly"
        or server.get("toolCount") != len(tools)
        or not tools
    ):
        return ["TDQS report does not contain the expected Entroly tool catalog"]

    by_name = {tool.get("name"): tool for tool in tools if isinstance(tool, dict)}
    review = by_name.get("work_record_review", {}).get("contextSignals", {})
    reader = by_name.get("read_source_file", {}).get("contextSignals", {})
    review_annotations = review.get("annotationValues") if isinstance(review, dict) else None
    reader_annotations = reader.get("annotationValues") if isinstance(reader, dict) else None
    matching_candidates = [
        candidate for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("tool") == "work_record_review"
        and candidate.get("cheaperSibling") == "read_source_file"
    ]
    disjoint = (
        isinstance(review_annotations, dict)
        and isinstance(reader_annotations, dict)
        and review_annotations.get("readOnly") is False
        and reader_annotations.get("readOnly") is True
        and len(matching_candidates) == 1
    )

    failures: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            failures.append("TDQS report contains a malformed finding")
            continue
        if finding.get("severity") not in {"info", "warning", "error"}:
            failures.append(f"TDQS report has an unknown severity: {finding.get('severity')}")
            continue
        if finding["severity"] == "info":
            continue
        known_cost_candidate = (
            disjoint
            and finding.get("rule") == "shadow-candidate"
            and finding.get("tool") == "work_record_review"
            and "may be shadowed by read_source_file" in str(finding.get("message", ""))
        )
        if not known_cost_candidate:
            failures.append(str(finding.get("message", finding)))
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_public_mcp_tdqs.py REPORT.json", file=sys.stderr)
        return 2
    try:
        report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("TDQS report must be an object")
        failures = check_report(report)
    except (OSError, ValueError) as exc:
        print(f"Cannot check TDQS report: {exc}", file=sys.stderr)
        return 2
    if failures:
        for failure in failures:
            print(f"TDQS: {failure}", file=sys.stderr)
        return 1
    if any(finding.get("rule") == "shadow-candidate" for finding in report["findings"]):
        print("TDQS: reviewed cost-only candidate: work_record_review -> read_source_file")
    print("TDQS: no actionable public tool-schema warnings or errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
