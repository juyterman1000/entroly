from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks import evidence_retention


def _run_benchmark(monkeypatch, tmp_path, gold, fragments, *, scoped=(), selections=None):
    selections = selections or {}

    class FixtureEngine:
        def __init__(self, config):
            self._rust = SimpleNamespace(export_fragments=lambda: fragments)

        def _ensure_index_loaded(self):
            pass

        def optimize_context(self, budget, query):
            selected = selections.get(query, [fragments[0]])
            return {"selected_fragments": selected, "tokens_used": 20, "method": "fixture"}

    output = tmp_path / "result.json"
    monkeypatch.setattr(evidence_retention, "GOLD", gold)
    monkeypatch.setattr(evidence_retention, "SCOPED_EVIDENCE", scoped)
    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["evidence_retention", "--json", str(output)])
    monkeypatch.setitem(
        sys.modules, "entroly.server", SimpleNamespace(EntrolyEngine=FixtureEngine, EntrolyConfig=object)
    )
    status = evidence_retention.main()
    return status, json.loads(output.read_text())


def test_misses_get_zero_credit_against_the_same_baseline(monkeypatch, tmp_path):
    gold = [("first query", ("hit.py",)), ("second query", ("miss.py",))]
    fragments = [
        {"source": "file:hit.py#2", "content": "first query", "token_count": 60},
        {"source": "file:miss.py", "content": "", "token_count": 40},
    ]

    status, report = _run_benchmark(monkeypatch, tmp_path, gold, fragments)

    assert status == 1
    assert report["recall"] == 0.5
    assert report["credited_reduction"] == 0.4
    assert report["rows"][0]["selected_sources"] == ["hit.py"]
    assert report["rows"][0]["selector"] == "fixture"


def test_alternative_for_one_query_does_not_cover_another_query(monkeypatch, tmp_path):
    gold = [("first query", ("missing.py", "present.py")), ("second query", ("missing.py",))]
    fragments = [{"source": "file:present.py", "token_count": 100}]

    with pytest.raises(SystemExit, match="gold files absent.*missing.py"):
        _run_benchmark(monkeypatch, tmp_path, gold, fragments)


@pytest.mark.parametrize("complete", [False, True], ids=["wrong-excerpt", "complete-evidence"])
def test_scoped_evidence_checks_content_independently_of_source_label(monkeypatch, tmp_path, complete):
    from entroly.auto_index import _run_git
    from entroly.source_span import SourceSpan

    scoped = evidence_retention.SCOPED_EVIDENCE
    gold = [("first query", ("hit.py",))]
    fragments = [{"source": "file:hit.py", "content": "first query", "token_count": 60}]
    selections = {}
    for (query, path, _), function in zip(scoped, (_run_git, SourceSpan.verify), strict=True):
        content = inspect.getsource(function)
        indexed = {"source": f"file:{path}#2", "content": content, "token_count": 20}
        fragments.append(indexed)
        selections[query] = [indexed] if complete else [
            {**indexed, "content": "unrelated excerpt"},
            {**indexed, "source": "file:other.py"},
        ]

    status, report = _run_benchmark(
        monkeypatch, tmp_path, gold, fragments, scoped=scoped, selections=selections,
    )

    assert status == 0
    assert report["recall"] == 1.0
    assert report["credited_reduction"] == 0.8
    assert [row["query"] for row in report["rows"]] == ["first query"]
    excerpts = report["scoped_excerpt_retention"]
    assert excerpts["total"] == 2
    assert excerpts["hits"] == (2 if complete else 0)
    assert excerpts["recall"] == (1.0 if complete else 0.0)
    for row, (_, _, required) in zip(excerpts["rows"], scoped, strict=True):
        assert row["source_retained"]
        assert row["retained"] == complete
        assert row["matched_evidence"] == (list(required) if complete else [])
        assert row["missing_evidence"] == ([] if complete else list(required))
        assert row["missing_from_index"] == []
