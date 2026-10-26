"""`entroly health` must name the most vulnerable *file*, not an internal id.

Reproduced on a fresh HOME (click, and a two-file project with an
`os.system` + `eval` sink): the security summary ended with
"Most vulnerable: f89025cc1_000001" -- an engine fragment id the user cannot
map to anything. The engine's own report carries the fragment's source.
"""

from __future__ import annotations

import argparse

import pytest

import entroly.cli as cli
from entroly.native_status import QCCR_SYMBOLS, native_status

pytestmark = pytest.mark.skipif(
    not native_status(QCCR_SYMBOLS).ok, reason="security report needs the native engine"
)


def test_health_names_the_vulnerable_file(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(
        "import os\n\ndef run(user_cmd):\n"
        "    os.system('sh -c ' + user_cmd)\n    return eval(user_cmd)\n",
        encoding="utf-8",
    )
    (project / "util.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.chdir(project)

    cli.cmd_health(argparse.Namespace(json_output=False, verbose=False))
    out = capsys.readouterr().out

    line = next(
        (text for text in out.splitlines() if "Most vulnerable:" in text), None
    )
    assert line is not None, out
    assert "app.py" in line, line
