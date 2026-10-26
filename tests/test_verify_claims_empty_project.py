"""`entroly verify-claims` run outside a project must say why it failed.

Reproduced on a fresh HOME in an empty directory: verify-claims exited 1 with
"[FAIL] Files indexed > 0 / 0 files" and "RESULTS: 11/12 checks passed
(1 failed)" -- read as a broken install -- while `entroly simulate` in the
same directory says "No files found to index. Run this inside a project
directory."
"""

from __future__ import annotations

import entroly.value_tracker as value_tracker
from entroly import verify_claims
from entroly.value_tracker import ValueTracker


def test_empty_directory_gets_an_actionable_hint(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(value_tracker, "_tracker", ValueTracker(tmp_path / "state"))

    rc = verify_claims.run(output=str(tmp_path / "report.json"))
    out = capsys.readouterr().out

    assert rc == 1  # the indexing claim is still unverified here
    assert "[FAIL] Files indexed > 0" in out
    assert "inside a project directory" in out, out
