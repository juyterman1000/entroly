"""The dashboard's inline JavaScript must actually parse.

`dashboard.py` is a Python file whose payload is a large HTML document with one
inline <script>. `python -c "import ast; ast.parse(...)"` therefore proves
nothing about the part users run: a stray quote inside the JS is valid Python
and a fatal SyntaxError in the browser.

That is not hypothetical. An inline `onchange="...setItem('key',...)"` was added
inside a single-quoted JS string, which terminated it. Python parsed happily,
the server started happily, the HTML contained every expected substring -- and
every panel rendered blank, because the whole script died before defining a
single function. The page looked like a server outage.

This test parses the extracted script with Node, which is the only check that
speaks the right language. It skips when Node is unavailable rather than
failing, so it never blocks a contributor without it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "entroly" / "dashboard.py"


def _inline_scripts() -> list[str]:
    source = DASHBOARD.read_text(encoding="utf-8")
    return [s for s in re.findall(r"<script>(.*?)</script>", source, re.S) if s.strip()]


def test_dashboard_has_an_inline_script_to_check() -> None:
    """Guard the guard: if the markup is restructured, this test must be updated."""
    scripts = _inline_scripts()
    assert scripts, "no inline <script> found in dashboard.py — has the template changed?"
    assert max(len(s) for s in scripts) > 5000, "inline script is suspiciously small"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_dashboard_javascript_parses(tmp_path: Path) -> None:
    for index, script in enumerate(_inline_scripts()):
        target = tmp_path / f"dashboard_{index}.js"
        target.write_text(script, encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(target)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "dashboard inline JavaScript does not parse -- every panel will "
            "render blank in the browser while Python, the server and any "
            f"substring assertion all pass:\n{result.stderr}"
        )
