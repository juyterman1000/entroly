from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.verify_public_trust import (
    PROMINENT_PUBLIC_FILES,
    _collect_prism_r_public_failures,
    collect_offline_failures,
)


ROOT = Path(__file__).resolve().parents[1]


def _prominent_text() -> dict[str, str]:
    return {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in PROMINENT_PUBLIC_FILES
    }


def _query_shift_report() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/neural_query_shift.json").read_text(
            encoding="utf-8"
        )
    )


def test_prominent_public_trust_contracts() -> None:
    assert collect_offline_failures() == []


def test_prism_r_public_claim_is_bound_to_verified_non_headline_artifact() -> None:
    assert (
        _collect_prism_r_public_failures(_prominent_text(), _query_shift_report()) == []
    )

    tampered = copy.deepcopy(_query_shift_report())
    tampered["headline_eligible"] = True
    failures = _collect_prism_r_public_failures(_prominent_text(), tampered)
    assert any("headline_eligible=false" in failure for failure in failures)


def test_prism_r_public_claim_rejects_unscoped_reuse() -> None:
    text = _prominent_text()
    text["PYPI_README.md"] += "\n87.0%\n"
    failures = _collect_prism_r_public_failures(text, _query_shift_report())
    assert any("unscoped public claim '87.0%'" in failure for failure in failures)
