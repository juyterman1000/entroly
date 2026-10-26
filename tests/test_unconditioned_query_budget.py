"""A query with no searchable term must still get a budgeted selection.

``qccr.select`` returns its input unchanged when the query expands to no
terms -- only stopwords ("What is this?"), or a script its tokenizer does not
read. ``EntrolyEngine.optimize_context`` and ``entroly optimize`` treated that
output as a budgeted selection, so a 4,000-token request on a copy of click
returned the whole repository: 173 fragments / 379,372 tokens from the CLI and
87 fragments / 250,906 tokens from the SDK. The no-match contract could not
catch it, because a selection containing every candidate is exempt from it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from entroly import optimize

ROOT = Path(__file__).resolve().parents[1]

BUDGET = 600
QUERIES_WITHOUT_TERMS = [
    "What is this?",
    "what does this do",
    "Как это работает?",
]


def _module_source(index: int) -> str:
    lines = [f'"""Ledger module {index} for warehouse partition {index}."""', ""]
    for fn in range(6):
        name = f"reconcile_partition_{index}_step_{fn}"
        lines += [
            f"def {name}(rows, threshold_{fn}={fn + index}):",
            f'    """Reconcile shelf rows for partition {index}, step {fn}."""',
            f"    kept = [row for row in rows if row.weight_{fn} > threshold_{fn}]",
            f"    return sorted(kept, key=lambda row: row.bin_{index}_{fn})",
            "",
        ]
    return "\n".join(lines)


def _fragments() -> list[dict]:
    return [
        {"content": _module_source(i), "source": f"pkg/ledger_{i}.py"}
        for i in range(12)
    ]


@pytest.mark.parametrize("query", QUERIES_WITHOUT_TERMS)
def test_sdk_optimize_stays_within_budget_for_a_query_without_terms(query):
    fragments = _fragments()
    corpus_tokens = sum(len(f["content"]) // 4 for f in fragments)
    assert corpus_tokens > 4 * BUDGET  # the budget has to bind

    result = optimize(fragments, budget=BUDGET, query=query)

    assert result["total_tokens"] <= BUDGET, (
        f"{query!r} returned {result['fragments_selected']}/{len(fragments)} "
        f"fragments, {result['total_tokens']} tokens, for a {BUDGET}-token budget"
    )
    delivered = sum(len(f.get("content") or "") for f in result["selected"]) // 4
    assert delivered <= BUDGET + len(result["selected"]), delivered


def _qccr_available() -> bool:
    from entroly.qccr import _HAS_RUST

    return bool(_HAS_RUST)


@pytest.mark.skipif(
    not _qccr_available(),
    reason="query-conditioned selection (QCCR) requires the native engine",
)
def test_sdk_optimize_still_conditions_on_a_query_with_terms():
    result = optimize(
        _fragments(), budget=BUDGET, query="reconcile_partition_7_step_3 threshold"
    )

    assert result["total_tokens"] <= BUDGET
    assert "reconcile_partition_7_step_3" in result["context_text"]


def test_cli_optimize_snapshot_stays_within_budget_for_a_query_without_terms(
    tmp_path: Path,
):
    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    for fragment in _fragments():
        (project / fragment["source"]).write_text(fragment["content"], encoding="utf-8")

    pythonpath = os.pathsep.join(
        part for part in (str(ROOT), os.environ.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "entroly", "optimize",
            "--task", "What is this?", "--budget", str(BUDGET),
            "--format", "json", "--quiet",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
            "ENTROLY_NO_DOCKER": "1",
            "PYTHONPATH": pythonpath,
        },
    )
    assert result.returncode == 0, result.stderr[-2000:]
    snapshot = json.loads(result.stdout[result.stdout.find("{"):])

    corpus = sum(len(f["content"]) // 4 for f in _fragments())
    delivered = sum(int(f.get("token_count") or 0) for f in snapshot["fragments"])
    assert snapshot["tokens_used"] <= BUDGET, snapshot["tokens_used"]
    # Not the repository. (The native optimizer can still overshoot a small
    # budget when a materialised skeleton is larger than its ingest-time
    # estimate; that is a separate accounting defect, so this bound is loose.)
    assert delivered < corpus / 2, (delivered, corpus)
