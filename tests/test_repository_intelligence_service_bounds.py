from __future__ import annotations

from pathlib import Path

import pytest

from entroly.repository_intelligence import (
    InvalidChangedPaths,
    RepositoryIntelligenceService,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "pkg/source.py", "def execute():\n    return 1\n")
    _write(root, "tests/test_source.py", "def test_execute():\n    assert True\n")


@pytest.mark.parametrize("changed", [[], "pkg/source.py"])
def test_service_rejects_empty_and_scalar_changed_paths(
    tmp_path: Path,
    changed,
) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)

    with pytest.raises(InvalidChangedPaths):
        service.impact(changed)


def test_service_caps_direct_sdk_limits(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    observed: dict[str, int] = {}

    class Report:
        def to_dict(self) -> dict[str, object]:
            return {"impacted_paths": []}

    def fake_impact(index, changed, *, max_depth, max_impacted_paths):
        observed["depth"] = max_depth
        observed["impact_limit"] = max_impacted_paths
        return Report()

    def fake_tests(index, changed, *, limit):
        observed["test_limit"] = limit
        return []

    monkeypatch.setattr(
        "entroly.repository_intelligence.service_impl.analyze_change_impact",
        fake_impact,
    )
    monkeypatch.setattr(
        "entroly.repository_intelligence.service_impl.localize_tests",
        fake_tests,
    )

    service.impact(["pkg/source.py"], max_depth=999, limit=999_999)
    service.tests(["pkg/source.py"], limit=999_999)

    assert observed == {
        "depth": 12,
        "impact_limit": 5_000,
        "test_limit": 100,
    }
