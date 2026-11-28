from __future__ import annotations

import json
import subprocess
import sys

import pytest

from entroly.native_status import (
    CORE_SYMBOLS,
    MIN_ENTROLY_CORE_VERSION,
    QCCR_SYMBOLS,
    WORK_GRAPH_SYMBOLS,
)
from scripts.ci_install_exact_head import _verify_core


@pytest.mark.parametrize(
    "missing_symbol",
    [None, "EntrolyEngine", "py_qccr_select", "recovery_handle_verify_bytes"],
)
def test_exact_head_bootstrap_checks_native_capabilities(
    tmp_path, monkeypatch, capfd, missing_symbol
) -> None:
    version = MIN_ENTROLY_CORE_VERSION
    dist_info = tmp_path / f"entroly_core-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Name: entroly-core\nVersion: {version}\n", encoding="utf-8"
    )
    symbols = CORE_SYMBOLS + QCCR_SYMBOLS + WORK_GRAPH_SYMBOLS
    (tmp_path / "entroly_core.py").write_text(
        "".join(f"{name} = True\n" for name in symbols if name != missing_symbol),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    if missing_symbol is None:
        _verify_core(sys.executable, version)
        assert json.loads(capfd.readouterr().out.splitlines()[-1]) == {
            "entroly_core": version,
            "work_graph": True,
        }
    else:
        with pytest.raises(subprocess.CalledProcessError):
            _verify_core(sys.executable, version)
        assert json.loads(capfd.readouterr().err) == {
            "expected": version,
            "actual": version,
            "missing": [missing_symbol],
        }
