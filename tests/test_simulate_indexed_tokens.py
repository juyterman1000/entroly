"""`entroly simulate` must count the tokens the project holds, not selection weights.

auto_index stores config and doc fragments with an inflated ``token_count`` --
x3 for .yaml/.yml/.json/.toml and x2 for .md/.rst -- so the knapsack prefers
source code. Simulate summed those weights as "estimated tokens" and used the
sum as its savings baseline. Measured on a 10-file project holding 20,426
estimated tokens: "Indexed 10 files (33,675 estimated tokens)", a baseline at the
32,000 cap, and 27,940 "tokens saved" (87.3%) where the project could yield
at most 16,366 (80.1%).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "src/ledger.py": "".join(
        f"def reconcile_shelf_{i}(rows):\n"
        f"    return [row for row in rows if row.bin == {i}]\n\n"
        for i in range(40)
    ),
    "README.md": "".join(
        f"## Shelf {i}\n\nShelf {i} holds the ledger rows for warehouse bin {i}.\n\n"
        for i in range(120)
    ),
    "pyproject.toml": "[project]\nname = \"ledger\"\n" + "".join(
        f"option_{i} = \"value-{i}\"\n" for i in range(60)
    ),
}


def test_indexed_tokens_and_baseline_count_content_not_selection_weights(tmp_path: Path):
    project = tmp_path / "project"
    for rel, text in FILES.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    expected = sum(max(1, len(text) // 4) for text in FILES.values())

    pythonpath = os.pathsep.join(
        part for part in (str(ROOT), os.environ.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-m", "entroly", "simulate", "--json",
         "--query", "How are shelf rows reconciled?"],
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
    report = json.loads(result.stdout[result.stdout.find("{"):])

    assert report["files_indexed"] == len(FILES)
    assert report["repo_tokens_indexed"] == expected, (
        f"project holds {expected} estimated tokens; simulate reported "
        f"{report['repo_tokens_indexed']}"
    )
    assert report["baseline_tokens_per_query"] == expected
