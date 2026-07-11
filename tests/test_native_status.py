from __future__ import annotations

import sys
from types import ModuleType

from entroly import native_status as ns


def test_native_status_reports_missing_symbols_and_stale_version(monkeypatch):
    fake = ModuleType("entroly_core")
    fake.py_qccr_expand_query = lambda query: [query]
    fake.__file__ = "C:/fake/entroly_core.pyd"
    monkeypatch.setitem(sys.modules, "entroly_core", fake)
    monkeypatch.setattr(ns.metadata, "version", lambda _name: "1.0.18")

    status = ns.native_status(ns.QCCR_SYMBOLS)

    assert status.available is True
    assert status.version == "1.0.18"
    assert status.version_ok is False
    assert status.missing_symbols == ("py_qccr_rank_files", "py_qccr_select")
    message = ns.native_status_message(status, feature="QCCR")
    assert "loaded version 1.0.18" in message
    assert "missing symbols: py_qccr_rank_files, py_qccr_select" in message


def test_native_status_accepts_complete_current_module(monkeypatch):
    fake = ModuleType("entroly_core")
    fake.py_qccr_expand_query = lambda query: [query]
    fake.py_qccr_rank_files = lambda *_args: []
    fake.py_qccr_select = lambda *_args: "[]"
    monkeypatch.setitem(sys.modules, "entroly_core", fake)
    monkeypatch.setattr(ns.metadata, "version", lambda _name: ns.MIN_ENTROLY_CORE_VERSION)

    status = ns.native_status(ns.QCCR_SYMBOLS)

    assert status.ok is True
    assert status.missing_symbols == ()


def test_native_status_distinguishes_missing_symbols_from_stale_version():
    status = ns.NativeStatus(
        available=True,
        module=ModuleType("entroly_core"),
        version=ns.MIN_ENTROLY_CORE_VERSION,
        path="C:/fake/entroly_core.pyd",
        missing_symbols=("py_qccr_select",),
        version_ok=True,
    )

    message = ns.native_status_message(status, feature="QCCR")

    assert "required symbols are missing" in message
    assert "requires a newer" not in message


def test_native_status_accepts_local_build_metadata(monkeypatch):
    fake = ModuleType("entroly_core")
    fake.py_qccr_expand_query = lambda query: [query]
    fake.py_qccr_rank_files = lambda *_args: []
    fake.py_qccr_select = lambda *_args: "[]"
    monkeypatch.setitem(sys.modules, "entroly_core", fake)
    monkeypatch.setattr(ns.metadata, "version", lambda _name: ns.MIN_ENTROLY_CORE_VERSION + "+local.1")

    status = ns.native_status(ns.QCCR_SYMBOLS)

    assert status.version_ok is True


def test_native_status_rejects_prerelease_of_minimum(monkeypatch):
    fake = ModuleType("entroly_core")
    fake.py_qccr_expand_query = lambda query: [query]
    fake.py_qccr_rank_files = lambda *_args: []
    fake.py_qccr_select = lambda *_args: "[]"
    monkeypatch.setitem(sys.modules, "entroly_core", fake)
    monkeypatch.setattr(ns.metadata, "version", lambda _name: ns.MIN_ENTROLY_CORE_VERSION + "-rc.1")

    status = ns.native_status(ns.QCCR_SYMBOLS)

    assert status.version_ok is False


def test_native_status_accepts_prerelease_above_minimum():
    major, minor, patch = map(int, ns.MIN_ENTROLY_CORE_VERSION.split("."))
    newer_prerelease = f"{major}.{minor}.{patch + 1}-rc.1"

    assert ns._version_at_least(newer_prerelease, ns.MIN_ENTROLY_CORE_VERSION) is True


def test_native_status_leaves_unknown_versions_indeterminate():
    assert ns._version_at_least("not-a-version", ns.MIN_ENTROLY_CORE_VERSION) is None
