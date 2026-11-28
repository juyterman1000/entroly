from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks import evidence_retention


def _run_benchmark(monkeypatch, tmp_path, gold, fragments):
    class FixtureEngine:
        def __init__(self, config):
            self._rust = SimpleNamespace(export_fragments=lambda: fragments)

        def _ensure_index_loaded(self):
            pass

        def optimize_context(self, budget, query):
            return {"selected_fragments": [fragments[0]], "tokens_used": 20, "method": "fixture"}

    output = tmp_path / "result.json"
    monkeypatch.setattr(evidence_retention, "GOLD", gold)
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
