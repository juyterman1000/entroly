"""The build stamp compares the serving version to the one installed on disk.

A long-lived server binds ``entroly.__version__`` once, at first import, and
then serves that value for its whole lifetime -- a repeat ``import`` is a
``sys.modules`` cache hit and never looks at the filesystem again. A running
server was observed reporting ``1.0.87`` while ``1.0.90`` was the package on
disk on the same machine in the same minute. The stamp reported that one number
and nothing anywhere compared it to what was installed, so an upgrade left the
user being served months-old behaviour with no signal at all.

These tests pin the comparison, and pin that an unreadable on-disk version
reports ``unknown`` rather than passing itself off as agreement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entroly.engine import (
    classify_version_skew,
    read_declared_version_from_disk,
)
from entroly.server import EntrolyEngine


class _FakeEngine:
    _use_rust = False
    _build_stamp = EntrolyEngine._build_stamp


def _stamp() -> dict:
    return EntrolyEngine._build_stamp(_FakeEngine())  # type: ignore[arg-type]


def _write_package(tmp_path: Path, version_line: str) -> Path:
    """Write a stand-in package __init__ carrying the given declaration."""
    init = tmp_path / "__init__.py"
    init.write_text(
        '"""Stand-in package source."""\n' + version_line + "\n",
        encoding="utf-8",
    )
    return init


# --- the skew comparison itself -------------------------------------------


def test_serving_version_differing_from_installed_reports_skew() -> None:
    assert classify_version_skew("1.0.87", "1.0.90") == "skew"


def test_matching_versions_report_no_skew() -> None:
    assert classify_version_skew("1.0.90", "1.0.90") == "none"


@pytest.mark.parametrize(
    "serving, installed",
    [
        ("unknown", "1.0.90"),
        ("1.0.87", "unknown"),
        ("", "1.0.90"),
        ("1.0.87", ""),
        (None, "1.0.90"),
        ("1.0.87", None),
    ],
)
def test_unreadable_version_never_reports_agreement(serving, installed) -> None:
    """The failure mode this exists to prevent: an unchecked version that
    presents itself as a match. ``unknown`` must never collapse into
    ``none``."""
    level = classify_version_skew(serving, installed)
    assert level == "unknown"
    assert level != "none"


def test_skew_level_is_truthy_for_both_skew_and_unknown() -> None:
    """A caller doing a plain truthiness test must not read "I could not
    check" as "they agree"."""
    assert bool(classify_version_skew("1.0.87", "1.0.90")) is True
    assert bool(classify_version_skew("1.0.87", "unknown")) is True
    assert classify_version_skew("1.0.90", "1.0.90") == "none"


# --- reading the installed version off disk --------------------------------


def test_disk_read_bypasses_the_import_cache(tmp_path: Path) -> None:
    init = _write_package(tmp_path, '__version__ = "1.0.90"')
    assert read_declared_version_from_disk(init) == "1.0.90"


def test_disk_read_sees_an_in_place_upgrade(tmp_path: Path) -> None:
    """Simulates the real scenario without installing anything: the file is
    rewritten under a process that already loaded the older value."""
    init = _write_package(tmp_path, '__version__ = "1.0.87"')
    assert read_declared_version_from_disk(init) == "1.0.87"
    _write_package(tmp_path, '__version__ = "1.0.90"')
    assert read_declared_version_from_disk(init) == "1.0.90"
    assert classify_version_skew("1.0.87", read_declared_version_from_disk(init)) == (
        "skew"
    )


@pytest.mark.parametrize(
    "version_line",
    [
        "__version__ = '1.0.90'",
        '__version__="1.0.90"',
        '__version__  =  "1.0.90"',
    ],
)
def test_disk_read_accepts_the_usual_declaration_spellings(
    tmp_path: Path, version_line: str
) -> None:
    init = _write_package(tmp_path, version_line)
    assert read_declared_version_from_disk(init) == "1.0.90"


@pytest.mark.parametrize(
    "init_path_factory",
    [
        lambda tmp_path: None,
        lambda tmp_path: "",
        lambda tmp_path: tmp_path / "does_not_exist.py",
        lambda tmp_path: tmp_path,  # a directory, not a file
    ],
)
def test_unreadable_source_reports_unknown(tmp_path: Path, init_path_factory) -> None:
    assert read_declared_version_from_disk(init_path_factory(tmp_path)) == "unknown"


def test_source_without_a_declaration_reports_unknown(tmp_path: Path) -> None:
    init = _write_package(tmp_path, "VERSION_NOTE = 'no declaration here'")
    assert read_declared_version_from_disk(init) == "unknown"


# --- the stamp surface -----------------------------------------------------


def test_stamp_reports_skew_and_both_versions_when_they_differ(monkeypatch) -> None:
    """A stale long-lived server: the module attribute is pinned to an older
    release while the package on disk declares the current one."""
    import entroly

    monkeypatch.setattr(entroly, "__version__", "1.0.87", raising=False)
    on_disk = read_declared_version_from_disk(entroly.__file__)
    assert on_disk != "1.0.87", "fixture needs a disk version that is not 1.0.87"

    stamp = _stamp()
    assert stamp["version_skew"] == "skew"
    assert stamp["entroly_version_serving"] == "1.0.87"
    assert stamp["entroly_version_installed"] == on_disk
    assert "1.0.87" in stamp["version_skew_detail"]
    assert on_disk in stamp["version_skew_detail"]
    # the pre-existing field keeps its meaning for callers already reading it
    assert stamp["entroly_version"] == "1.0.87"


def test_stamp_reports_no_skew_when_versions_agree(monkeypatch) -> None:
    import entroly

    on_disk = read_declared_version_from_disk(entroly.__file__)
    assert on_disk != "unknown", "fixture needs a readable on-disk version"
    monkeypatch.setattr(entroly, "__version__", on_disk, raising=False)

    stamp = _stamp()
    assert stamp["version_skew"] == "none"
    assert stamp["entroly_version_serving"] == on_disk
    assert stamp["entroly_version_installed"] == on_disk
    assert "version_skew_detail" not in stamp


def test_stamp_reports_unknown_when_disk_version_is_unreadable(
    monkeypatch, tmp_path: Path
) -> None:
    """The boundary must distinguish "I could not check" from "they match"."""
    import entroly

    monkeypatch.setattr(entroly, "__version__", "1.0.90", raising=False)
    monkeypatch.setattr(
        entroly, "__file__", str(tmp_path / "gone" / "__init__.py"), raising=False
    )

    stamp = _stamp()
    assert stamp["version_skew"] == "unknown"
    assert stamp["version_skew"] != "none"
    assert stamp["entroly_version_serving"] == "1.0.90"
    assert stamp["entroly_version_installed"] == "unknown"
    assert "version_skew_detail" in stamp


def test_stamp_always_carries_the_comparison_fields() -> None:
    """Regression guard: the stamp claimed to let a caller tell whether a
    shipped fix is live, but published no comparison to tell it with."""
    stamp = _stamp()
    for field in (
        "entroly_version",
        "entroly_version_serving",
        "entroly_version_installed",
        "version_skew",
    ):
        assert field in stamp, f"build stamp is missing {field}"
    assert stamp["version_skew"] in {"none", "skew", "unknown"}
