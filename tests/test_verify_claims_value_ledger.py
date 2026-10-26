"""`entroly verify-claims` must not book its own probe as the user's value.

Reproduced on a fresh HOME: the first documented command,
`entroly verify-claims`, compressed a fixed probe string through the public
SDK, and the SDK recorded that as a local operation. The next
`entroly value --json` then reported "local_operations": {"operations": 1,
"tokens_reduced": 3, ...} and an energy estimate for 3 tokens -- value the
user never produced, on the receipt the first-run guide says to read.
"""

from __future__ import annotations

import entroly.value_tracker as value_tracker
from entroly import verify_claims
from entroly.value_tracker import ValueTracker


def test_verify_claims_leaves_the_value_ledger_empty(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(
        "def main_module_structure():\n    return 'explain the main module'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    tracker = ValueTracker(tmp_path / "state")
    monkeypatch.setattr(value_tracker, "_tracker", tracker)

    verify_claims.run(output=str(tmp_path / "report.json"))
    capsys.readouterr()

    lifetime = tracker.get_lifetime()
    assert lifetime["local_operations"] == 0, lifetime
    assert lifetime["tokens_saved"] == 0, lifetime
    assert lifetime["requests_total"] == 0, lifetime
