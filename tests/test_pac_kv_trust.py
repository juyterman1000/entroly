"""PAC-KV must not claim savings it did not make, nor write at startup.

Driven with real objects. `tests/test_proxy_pac_kv.py` mocks the aligner,
which is how attribute errors and an unearned VERIFIED saving shipped.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from entroly.cache_aligner import CacheAligner
from entroly.memory_graph import MemoryGraph
from entroly.optimization_ledger import OptimizationLedger
from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig


def _conversation(turn: str) -> dict:
    system = "You are a careful coding assistant. Follow the repository rules.\n" * 400
    return {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Summarise the failing test."},
            {"role": "assistant", "content": "test_flag_default fails on the default."},
            {"role": "user", "content": turn},
        ],
    }


def _ledger_rows(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        return sum(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        )


class _BudgetedAligner(CacheAligner):
    """A real aligner given a token ceiling, so warm turns compress the tail.

    The proxy does not pass a budget today, which leaves the booking branch
    unreachable; a budget is the change that would make it fire.
    """

    def align_prompt(self, *args, **kwargs):
        kwargs.setdefault("target_budget", 4000)
        return super().align_prompt(*args, **kwargs)


LONG_TURN = "Rerun the suite and paste every line here:\n" + (
    "tests/test_x.py::test_case PASSED\n" * 300
)


def test_alignment_hook_books_no_saving_for_an_unchanged_request(tmp_path):
    ledger_path = tmp_path / "optimization.sqlite"
    proxy = PromptCompilerProxy(engine=None, config=ProxyConfig())
    proxy._cache_aligner = _BudgetedAligner(default_provider="anthropic")
    proxy._optimization_ledger = OptimizationLedger(ledger_path)
    baseline_rows = _ledger_rows(ledger_path)

    headers = {}
    for turn in ("Fix it.\n" + LONG_TURN, LONG_TURN):
        body = _conversation(turn)
        sent = repr(body)
        out, headers = proxy._apply_pac_kv_alignment(
            conversation_id="conv-trust", provider="anthropic", body=body
        )
        assert repr(out) == sent  # the hook does not rewrite the request

    # The second turn is a warm, cache-locked decision whose receipt shows
    # raw > aligned: the case that used to book the difference as VERIFIED.
    assert headers.get("X-Entroly-PAC-Warm") == "true", headers
    assert headers.get("X-Entroly-PAC-Decision") == "cache_locked_tail_compressed"
    assert "X-Entroly-PAC-Status" not in headers, headers
    assert _ledger_rows(ledger_path) == baseline_rows


def test_aligner_stats_name_what_they_measure():
    aligner = CacheAligner(default_provider="anthropic")
    prefix = "System: Core guidelines and tool schemas.\n" * 300
    aligner.align_prompt("s", prefix, "User: step 1", provider="anthropic")
    aligner.align_prompt("s", prefix, "User: step 2", provider="anthropic")

    stats = aligner.stats()

    assert stats["match_policy"] == "exact_sha256"
    assert "total_dollar_savings" not in stats
    assert stats["projected_dollar_savings"] > 0.0


def test_memory_graph_construction_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    home_is_a_file = tmp_path / "home-is-a-file"
    home_is_a_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_is_a_file))

    graph = MemoryGraph()  # must not raise NotADirectoryError

    assert graph.stats()["total_nodes"] == 0

    root = tmp_path / "state"
    graph = MemoryGraph(root)
    assert not root.exists()
    graph.index_memory("mem_1", ["Alice", "BackendTeam"], "Alice manages it")
    graph.save()
    assert (root / "memory_graph" / "nodes.json").is_file()
