"""Without --query, `entroly simulate` must say it measured sample questions.

The built-in questions ask about an authentication flow and SQL injection. On
a click-derived project, which has neither, they headlined "Average reduction:
98.3%" with nothing on screen saying the questions were not the user's.
"""

from __future__ import annotations

from types import SimpleNamespace

from entroly import cli


class _FakeEngine:
    def advance_turn(self):
        pass

    def optimize_context(self, *, token_budget, query):
        return {"selected_fragments": [{"token_count": 20, "source": "a.py",
                                        "relevance": 1.0}]}


def _report(monkeypatch, query):
    monkeypatch.setattr(
        cli, "_load_local_simulation_engine", lambda _: (_FakeEngine(), 1, 100, "ok")
    )
    monkeypatch.setattr(cli, "_query_conditioned_selection_active", lambda _: True)
    return cli._run_local_simulation(
        SimpleNamespace(max_files=1, budget=50, baseline=100, query=query)
    )


def test_sample_questions_are_flagged_in_json_and_text(monkeypatch, capsys):
    report = _report(monkeypatch, None)

    assert report["sample_queries"] is True
    assert any("--query" in line and "sample" in line for line in report["limitations"])

    cli._print_local_simulation(report, title="Entroly Simulate", include_perf=False)
    out = capsys.readouterr().out
    assert "Sample questions (no --query given)" in out


def test_a_supplied_query_is_not_flagged(monkeypatch, capsys):
    report = _report(monkeypatch, ["How are shelf rows reconciled?"])

    assert report["sample_queries"] is False
    cli._print_local_simulation(report, title="Entroly Simulate", include_perf=False)
    assert "Sample questions" not in capsys.readouterr().out
