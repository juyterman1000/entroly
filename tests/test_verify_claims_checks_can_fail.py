"""Every counted `entroly verify-claims` check must be able to fail.

On main, 3 of the 12 "checks" were constants: IDX-2 asserted `index_s >= 0`,
ENG-1 and LOCAL-1 asserted `True`. "12/12 checks passed" therefore counted
three claims nothing had verified -- including "all operations ran locally".
"""

from __future__ import annotations

import json
import socket

import entroly.value_tracker as value_tracker
from entroly import verify_claims
from entroly.value_tracker import ValueTracker


def _project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text(
        "def main_module_structure():\n    return 'explain the main module'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(value_tracker, "_tracker", ValueTracker(tmp_path / "state"))
    return root


def _results(path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in report["results"]}


def test_local_check_fails_when_the_run_opens_a_network_connection(
    tmp_path, monkeypatch, capsys
):
    _project(tmp_path, monkeypatch)
    real_auto_index = verify_claims._auto_index.auto_index

    def auto_index_that_phones_home(engine, *args, **kwargs):
        try:  # TEST-NET-1: never routable; only the attempt matters
            socket.create_connection(("192.0.2.1", 9), timeout=0.05).close()
        except OSError:
            pass
        return real_auto_index(engine, *args, **kwargs)

    monkeypatch.setattr(verify_claims._auto_index, "auto_index", auto_index_that_phones_home)
    out_path = tmp_path / "report.json"

    rc = verify_claims.run(output=str(out_path))
    capsys.readouterr()

    local = _results(out_path)["LOCAL-1"]
    assert local["status"] == "FAIL", local
    assert "192.0.2.1" in local["detail"]
    assert rc == 1


def test_local_check_passes_and_says_what_it_measured(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    out_path = tmp_path / "report.json"

    verify_claims.run(output=str(out_path))
    capsys.readouterr()

    local = _results(out_path)["LOCAL-1"]
    assert local["status"] == "PASS"
    assert "0 outbound connections" in local["detail"]


def test_indexing_check_fails_when_indexing_errors(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        verify_claims._auto_index,
        "auto_index",
        lambda *_a, **_k: {"status": "error", "files_indexed": 0, "total_tokens": 0},
    )
    out_path = tmp_path / "report.json"

    verify_claims.run(output=str(out_path))
    capsys.readouterr()

    assert _results(out_path)["IDX-2"]["status"] == "FAIL"


def test_engine_mode_is_reported_not_counted_as_a_verified_check(
    tmp_path, monkeypatch, capsys
):
    _project(tmp_path, monkeypatch)
    out_path = tmp_path / "report.json"

    verify_claims.run(output=str(out_path))
    out = capsys.readouterr().out

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert "ENG-1" not in _results(out_path)
    assert report["engine"] in {"rust", "python"}
    assert "[INFO] Engine:" in out
