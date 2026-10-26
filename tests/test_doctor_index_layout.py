"""`entroly doctor` must find the index that `simulate`/`go` actually wrote.

Reproduced on a fresh HOME: `entroly simulate` wrote
~/.entroly/checkpoints/<project-hash>/index.json.gz, then `entroly doctor`
printed "No index found (will be created on first run)". The index lives in a
per-project subdirectory (entroly/config.py:_project_checkpoint_dir) but the
doctor only globbed the top level of ~/.entroly/checkpoints.
"""

from __future__ import annotations

import argparse
import gzip

import entroly.cli as cli
from entroly.config import _project_checkpoint_dir


def test_doctor_sees_the_per_project_index(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.chdir(project)

    # The same path the engine persists its warm-start index to.
    checkpoint = _project_checkpoint_dir()
    assert checkpoint.parent == home / ".entroly" / "checkpoints"
    with gzip.open(checkpoint / "index.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    cli.cmd_doctor(argparse.Namespace(json_output=False, port=21099))
    out = capsys.readouterr().out

    assert "No index found" not in out, out
    assert "Index fresh" in out, out
