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
    for i in range(5):
        (project / "pkg" / f"shelf_{i}.py").write_text(
            "".join(
                f"def reconcile_shelf_{i}_{n}(rows):\n"
                f"    return [row for row in rows if row.bin == {n}]\n\n"
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
