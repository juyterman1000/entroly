from __future__ import annotations

import json

from entroly import verify_claims


def test_verify_claims_reports_exact_recovery(tmp_path):
    report_path = tmp_path / "verification.json"

    result = verify_claims.run(output=str(report_path), max_files=3)

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result_ids = {item["id"] for item in report["results"]}
    assert {"CCR-1", "CCR-2", "CCR-3"}.issubset(result_ids)
    assert report["recovery"]["retrieval_handles"] >= 1
    assert report["recovery"]["retrieved_exact"] is True
    assert report["recovery"]["slice_tokens"] <= report["recovery"]["slice_budget"]
