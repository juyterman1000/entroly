"""Release smoke gate for packaged WITNESS.

Run from the repository root before tagging a release:

    python scripts/verify_witness_release.py

The script installs the current checkout into a clean virtualenv, then verifies
that the installed package exposes the Rust WITNESS functions and that the
documented CLI command works outside the source tree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def main() -> None:
    temp = Path(tempfile.mkdtemp(prefix="entroly-witness-release-"))
    try:
        venv = temp / "venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        py = _venv_python(venv)
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", str(ROOT / "entroly-core")])
        _run([str(py), "-m", "pip", "install", str(ROOT)])

        check_code = """
import json
import entroly
import entroly_core
assert hasattr(entroly_core, 'py_witness_analyze'), 'missing py_witness_analyze'
payload = json.loads(entroly_core.py_witness_analyze(
    'The Oberoi Group has its head office in Delhi.',
    'The Oberoi Group has its head office in Delhi. Paris is the capital of Germany.',
    'strict',
    'rag',
))
assert payload['policy']['changed'] is True
assert 'Paris is the capital of Germany' not in payload['output']
print(json.dumps({'ok': True, 'version': entroly.__version__}))
"""
        _run([str(py), "-c", check_code])

        proc = _run([
            str(py),
            "-m",
            "entroly.cli",
            "witness",
            "--context",
            "The context mentions Berlin.",
            "--output",
            "Paris is the capital of Germany.",
            "--mode",
            "strict",
            "--profile",
            "rag",
            "--json",
        ])
        payload = json.loads(proc.stdout[proc.stdout.find("{") :])
        assert payload["policy"]["changed"] is True
        assert "Paris is the capital of Germany" not in payload["output"]
        print("WITNESS release smoke gate passed.")
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
