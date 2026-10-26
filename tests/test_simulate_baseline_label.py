"""Every savings figure on the onboarding path must name the baseline it uses.

`entroly simulate` measures each query against a fixed baseline -- the smaller
of the indexed tokens and 32,000 -- so the percentage rises as the selection
shrinks, whatever the selection contains. On click, an off-topic question
("the best recipe for banana bread ...") that matched one common word scored
"97.6% fewer; 31,241 tokens saved" while the relevant question scored 87.3%,
with nothing on screen to say why. `entroly value` described the same MCP
figures as "measured".
"""

from __future__ import annotations

from entroly.cli import _print_local_simulation
from entroly.value_tracker import ValueTracker


def _report() -> dict:
    rows = [
        ("How does click parse command line options?", 12, 4074, 27926, 87.27),
        ("What is the best recipe for banana bread?", 8, 759, 31241, 97.63),
    ]
    return {
        "files_indexed": 94,
        "repo_tokens_indexed": 182_799,
        "baseline_tokens_per_query": 32_000,
        "budget": 4096,
        "budget_narrowed_to_demonstrate": False,
        "query_conditioned_selection": True,
        "queries": [
            {
                "query": query,
                "selected_fragments": fragments,
                "selected_tokens": tokens,
                "baseline_tokens": 32_000,
                "tokens_saved": saved,
                "reduction_pct": pct,
                "latency_ms": 1.0,
                "top_sources": ["file:src/click/core.py"],
            }
            for query, fragments, tokens, saved, pct in rows
        ],
        "average_reduction_pct": 92.45,
        "average_budget_difference_pct": 92.45,
        "latency_ms": {"min": 1.0, "p95": 1.0, "max": 1.0},
        "limitations": [],
    }


def test_each_figure_names_the_fixed_baseline(capsys):
    _print_local_simulation(_report(), title="Entroly Simulate", include_perf=False)
    out = capsys.readouterr().out

    assert "31,241 tokens saved vs the 32,000-token baseline" in out
    assert "27,926 tokens saved vs the 32,000-token baseline" in out
    assert "vs a fixed 32,000-token baseline" in out
    assert "scores higher whether or not it answers the question" in out


def test_value_receipt_does_not_call_baseline_arithmetic_measured(tmp_path):
    evidence = ValueTracker(tmp_path).get_value_receipt()["local_operations"]["evidence"]

    assert "The token counts are measured" not in evidence
    assert "fixed baseline of at most 32,000 tokens per call" in evidence
