from __future__ import annotations

from pathlib import Path
import warnings

import pytest

import entroly.parser_compatibility as compat
import entroly.repository_intelligence as repository_intelligence
import entroly.runtime_doctor as doctor_module
from entroly.parser_compatibility import ParserRegistryStatus
from entroly.runtime_doctor import runtime_doctor


def _status(version: str, *, compatible: bool) -> ParserRegistryStatus:
    return ParserRegistryStatus(
        installed=True,
        version=version,
        minimum_version=compat.TREE_SITTER_LANGUAGE_PACK_MIN_VERSION,
        compatible=compatible,
        detail="compatible" if compatible else "below_supported_floor",
    )


def test_declared_parser_floor_matches_runtime_guard() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    requirement = (
        "tree-sitter-language-pack>="
        f"{compat.TREE_SITTER_LANGUAGE_PACK_MIN_VERSION},<2"
    )
    assert text.count(requirement) >= 2


def test_below_floor_is_detected_and_warned_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat.metadata,
        "version",
        lambda _distribution: "1.13.9",
    )
    compat._WARNED_INCOMPATIBLE_VERSIONS.clear()

    status = compat.language_pack_status()
    assert status.installed is True
    assert status.compatible is False
    assert status.detail == "below_supported_floor"

    with pytest.warns(RuntimeWarning, match=r"below the supported >=1\.14\.3 floor"):
        warned = compat.warn_if_incompatible_language_pack()
    assert warned == status

    # Repeated repository operations must not flood logs with the same warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compat.warn_if_incompatible_language_pack()
    assert caught == []


def test_floor_release_is_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compat.metadata,
        "version",
        lambda _distribution: compat.TREE_SITTER_LANGUAGE_PACK_MIN_VERSION,
    )
    status = compat.language_pack_status()
    assert status.compatible is True
    assert status.detail == "compatible"


def test_prerelease_of_floor_is_not_accepted() -> None:
    assert compat._version_at_least("1.14.3rc1", "1.14.3") is False
    assert compat._version_at_least("1.14.3.post1", "1.14.3") is True
    assert compat._version_at_least("1.14.4", "1.14.3") is True


def test_runtime_doctor_surfaces_incompatible_parser_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor_module,
        "language_pack_status",
        lambda: _status("1.13.9", compatible=False),
    )
    report = runtime_doctor(
        data_dir=tmp_path / "state",
        capability_factory=lambda: {
            "schema_version": "entroly.runtime-capabilities.v1",
            "engine": {"native": {"available": True}},
        },
        status_factory=lambda *, port: {
            "schema_version": "entroly.runtime-status.v1",
            "healthy": True,
        },
    )

    assert report["healthy"] is True
    assert {
        "name": "code_intelligence_registry",
        "status": "warning",
        "detail": "incompatible:1.13.9;requires>=1.14.3",
    } in report["checks"]


def test_repository_index_records_parser_coverage_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sample.py").write_text(
        "def answer():\n    return 42\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        repository_intelligence,
        "warn_if_incompatible_language_pack",
        lambda: _status("1.13.9", compatible=False),
    )

    index = repository_intelligence.build_repository_index(tmp_path)
    assert any(
        "parser registry degraded" in diagnostic
        and "1.13.9" in diagnostic
        and ">=1.14.3" in diagnostic
        for diagnostic in index.diagnostics
    )
