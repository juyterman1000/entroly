"""Test a built wheel in an isolated new virtualenv outside the source tree.

Every command runs with PYTHONPATH cleared, so an editable source checkout
cannot mask a broken released artifact. Only opt-in package installation uses
network; the installed Entroly runtime is air-gapped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def verify(wheel: Path, report: dict) -> None:
    assert wheel.is_file(), f"wheel missing: {wheel}"
    report.update({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "checks": {},
    })
    with tempfile.TemporaryDirectory(prefix="entroly-fresh-wheel-") as location:
        root = Path(location)
        subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True, timeout=120)
        folder = "Scripts" if os.name == "nt" else "bin"
        suffix = ".exe" if os.name == "nt" else ""
        python = root / "venv" / folder / f"python{suffix}"
        cli = root / "venv" / folder / f"entroly{suffix}"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(ENTROLY_AIR_GAP="1", ENTROLY_NO_SELF_HEAL="1",
                           ENTROLY_DISABLE_TELEMETRY="1", ENTROLY_DISABLE_UPDATE_CHECK="1")

        def command(name: str, parts: list[str], seconds: int = 120) -> None:
            completed = subprocess.run(parts, cwd=root, env=environment, timeout=seconds,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            report["checks"][name] = {"exit_code": completed.returncode,
                                      "output_tail": completed.stdout[-500:]}
            if completed.returncode != 0:
                raise RuntimeError(f"fresh install {name} failed: {completed.stdout[-500:]}")

        command("install", [str(python), "-m", "pip", "install", "--no-cache-dir", str(wheel)], 420)
        command("pip_check", [str(python), "-m", "pip", "check"])
        command("import_and_native", [str(python), "-c", (
            "import json, entroly; from entroly.native_status import native_status; "
            "n=native_status(); print(json.dumps({'file':entroly.__file__,'ok':n.ok,'version':n.version})); "
            "assert n.ok, 'native backend not active'")], 60)
        command("cli_version", [str(cli), "--version"], 45)
        command("cli_help", [str(cli), "--help"], 45)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"status": "not_started"}
    try:
        matches = list(args.wheel_dir.glob("entroly-*.whl"))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one Entroly wheel: {[x.name for x in matches]}")
        verify(matches[0].resolve(), report)
        report["status"] = "passed"
        return_code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"[-500:]
        return_code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": list(report.get("checks", {}))}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
