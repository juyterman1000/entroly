"""`entroly simulate` must say how much of the repository it indexed.

It indexes at most ``--max-files`` files (default 100). On a 2,018-file clone
of this repository it printed "Indexed 100 files" and headlined "88.6% fewer"
for a function in a file it never indexed, without saying a cap applied.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(project: Path, home: Path, *extra: str) -> str:
    pythonpath = os.pathsep.join(
        part for part in (str(ROOT), os.environ.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-m", "entroly", "simulate", "--max-files", "2",
         "--query", "How are shelf rows reconciled?", *extra],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "HOME": str(home),
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
            "ENTROLY_NO_DOCKER": "1",
            "NO_COLOR": "1",
            "PYTHONPATH": pythonpath,
        },
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_a_capped_index_is_reported_as_a_subset(tmp_path: Path):
    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    # Genuinely different files. Five copies that differ by one digit are
    # near-duplicates, and which two `--max-files 2` picks follows directory
    # order: on CI's filesystem it picked a pair the engine deduplicates, so
    # this test measured deduplication instead of cap reporting.
    verbs = ("reconcile", "audit", "restock", "relabel", "archive")
    fields = ("bin", "sku", "aisle", "batch", "owner")
    for i, (verb, field) in enumerate(zip(verbs, fields)):
        (project / "pkg" / f"shelf_{i}.py").write_text(
            "".join(
                f"def {verb}_shelf_{i}_{n}(rows):\n"
                f"    return [row for row in rows if row.{field} == {n * (i + 3)}]\n\n"
                for n in range(20)
            ),
            encoding="utf-8",
        )

    out = _run(project, tmp_path / "home", "--json")
    report = json.loads(out[out.find("{"):])
    assert report["files_indexed"] == 2
    assert report["files_indexable"] == 5
    assert report["coverage_capped"] is True
    assert (report["file_cap"], report["file_cap_source"]) == (2, "--max-files")
    assert any("Only 2 of 5 indexable files" in line for line in report["limitations"])

    text = _run(project, tmp_path / "home")
    assert "Indexed 2 of 5 indexable files" in text
    assert "Capped by --max-files 2" in text
    if report["query_conditioned_selection"]:
        # The no-native fallback prints its own "unearned" average instead.
        assert "over 2 of 5 files" in text


def test_a_deduplicated_file_is_not_counted_as_indexed(tmp_path: Path):
    """Cold and warm runs must report the same coverage.

    These two files differ by one digit, and the native engine keeps only one
    of them. The cold run counted both ("Indexed 2 files") beside one file's
    tokens; the warm run, which counts the files the loaded index holds, said
    1. The pure-Python engine already excluded duplicates; only the native
    batch path counted every file it was handed.
    """
    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    for i in (0, 2):
        (project / "pkg" / f"shelf_{i}.py").write_text(
            "".join(
                f"def reconcile_shelf_{i}_{n}(rows):\n"
                f"    return [row for row in rows if row.bin == {n}]\n\n"
                for n in range(20)
            ),
            encoding="utf-8",
        )

    cold = json.loads((out := _run(project, tmp_path / "home", "--json"))[out.find("{"):])
    warm = json.loads((out := _run(project, tmp_path / "home", "--json"))[out.find("{"):])

    assert cold["files_indexed"] == warm["files_indexed"], (cold, warm)
    assert cold["repo_tokens_indexed"] == warm["repo_tokens_indexed"], (cold, warm)
    assert 1 <= cold["files_indexed"] <= 2
