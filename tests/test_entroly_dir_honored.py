"""Commands that read or delete state must honour ``ENTROLY_DIR``.

``_resolve_entroly_dir`` is the canonical resolver and honours the variable.
Eight call sites bypassed it with a hardcoded ``Path.home() / ".entroly"``.

The dangerous one is ``entroly clean``. With ``ENTROLY_DIR`` pointed at an empty
sandbox it still enumerated the real home directory -- 8,094 checkpoints and 302
index files -- so ``entroly clean -y`` would have deleted global state a caller
had explicitly sandboxed away from. The same bypass sent ``ravs report`` and
``migrate`` to the wrong tree.
"""

from __future__ import annotations

import re
from pathlib import Path

from entroly.cli import _resolve_entroly_dir

CLI_SOURCE = Path(__file__).resolve().parents[1] / "entroly" / "cli.py"


def test_resolver_honours_the_environment_variable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "sandbox"))
    assert _resolve_entroly_dir() == (tmp_path / "sandbox")


def test_only_the_resolver_itself_may_hardcode_the_home_path() -> None:
    """One occurrence is legitimate: the resolver's own fallback candidate.

    Any other is a command bypassing the resolver, which is how `clean` came to
    target a directory the caller had redirected away from.
    """
    text = CLI_SOURCE.read_text(encoding="utf-8")
    hits = [
        (i + 1, line.strip())
        for i, line in enumerate(text.splitlines())
        if re.search(r'Path\.home\(\)\s*/\s*"\.entroly"', line)
    ]
    assert len(hits) == 1, (
        "only `_resolve_entroly_dir` may name the home directory directly; "
        f"found {len(hits)}: {hits}"
    )
    # And it must be the fallback inside the resolver, not a command.
    assert hits[0][0] < 130, f"unexpected location for the fallback: {hits[0]}"
