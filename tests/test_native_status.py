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
