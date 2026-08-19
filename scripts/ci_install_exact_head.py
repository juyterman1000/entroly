#!/usr/bin/env python3
"""Install an Entroly source checkout without consulting PyPI for its native core.

Release candidates intentionally raise the root package's minimum ``entroly-core``
version before that core exists on PyPI. PR/source CI must therefore build the
native dependency from the same checkout first; otherwise pip either fails on the
future version or, worse, tests an older published core.

This helper is cross-platform and supports both editable source installs and an
isolated root-wheel install. It is CI orchestration only: no product semantics live
here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _run(argv: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in argv), flush=True)
    subprocess.run(argv, cwd=str(cwd or ROOT), check=True)


def _output(argv: list[str]) -> str:
    return subprocess.check_output(argv, cwd=str(ROOT), text=True).strip()


def _target_scripts(python: str) -> Path:
    value = _output(
        [
            python,
            "-c",
            "import sysconfig; print(sysconfig.get_path('scripts'))",
        ]
    )
    return Path(value)


def _maturin_executable(python: str) -> str:
    scripts = _target_scripts(python)
    names = ("maturin.exe", "maturin") if os.name == "nt" else ("maturin", "maturin.exe")
    for name in names:
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("maturin")
    if found:
        return found
    raise RuntimeError(f"maturin executable not found under {scripts}")


def _project_versions() -> tuple[str, str]:
    root_data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core_data = tomllib.loads((ROOT / "entroly-core" / "pyproject.toml").read_text(encoding="utf-8"))
    return str(root_data["project"]["version"]), str(core_data["project"]["version"])


def _pip(python: str, *args: str) -> None:
    _run([python, "-m", "pip", *args])


def _build_and_install_core(python: str, work: Path) -> Path:
    _pip(python, "install", "--upgrade", "pip")
    _pip(python, "install", "maturin>=1.8,<2")
    maturin = _maturin_executable(python)
    core_dist = work / "core"
    core_dist.mkdir(parents=True, exist_ok=True)
    _run(
        [
            maturin,
            "build",
            "--release",
            "--out",
            str(core_dist),
            "--manifest-path",
            str(ROOT / "entroly-core" / "Cargo.toml"),
            "--interpreter",
            python,
        ]
    )
    wheels = sorted(core_dist.glob("entroly_core-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one exact-head core wheel, found {wheels!r}")
    _pip(python, "install", "--force-reinstall", "--no-deps", str(wheels[0]))
    return wheels[0]


def _verify_core(python: str, expected: str) -> None:
    program = r'''
import importlib.metadata
import json
import entroly_core

expected = __import__("sys").argv[1]
actual = importlib.metadata.version("entroly-core")
missing = [name for name in ("WorkGraph",) if not hasattr(entroly_core, name)]
if actual != expected or missing:
    raise SystemExit(json.dumps({"expected": expected, "actual": actual, "missing": missing}))
print(json.dumps({"entroly_core": actual, "work_graph": True}, sort_keys=True))
'''
    _run([python, "-c", program, expected])


def _install_root_editable(python: str, extras: str, packages: list[str]) -> None:
    suffix = f"[{extras}]" if extras else ""
    _pip(python, "install", "-e", f".{suffix}", *packages)


def _install_root_wheel(python: str, work: Path, packages: list[str]) -> None:
    _pip(python, "install", "build>=1,<2")
    root_dist = work / "root"
    root_dist.mkdir(parents=True, exist_ok=True)
    _run([python, "-m", "build", "--wheel", "--outdir", str(root_dist)], cwd=ROOT)
    wheels = sorted(root_dist.glob("entroly-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one Entroly root wheel, found {wheels!r}")
    _pip(python, "install", "--no-cache-dir", str(wheels[0]), *packages)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Target Python interpreter")
    parser.add_argument("--extras", default="", help="Root editable extras, e.g. test,proxy")
    parser.add_argument(
        "--mode",
        choices=("editable", "wheel", "core-only"),
        default="editable",
        help="How to install the root package after installing exact-head entroly-core",
    )
    parser.add_argument(
        "--extra-package",
        action="append",
        default=[],
        help="Additional pip requirement; may be repeated",
    )
    args = parser.parse_args()

    python = str(Path(args.python).resolve()) if Path(args.python).exists() else args.python
    root_version, core_version = _project_versions()
    if root_version != core_version:
        raise SystemExit(
            f"release surfaces diverged before CI bootstrap: entroly={root_version}, core={core_version}"
        )

    with tempfile.TemporaryDirectory(prefix="entroly-exact-head-") as tmp:
        work = Path(tmp)
        wheel = _build_and_install_core(python, work)
        _verify_core(python, core_version)
        if args.mode == "editable":
            _install_root_editable(python, args.extras, list(args.extra_package))
        elif args.mode == "wheel":
            _install_root_wheel(python, work, list(args.extra_package))
        elif args.extra_package:
            _pip(python, "install", *args.extra_package)
        _verify_core(python, core_version)

        print(
            json.dumps(
                {
                    "root_version": root_version,
                    "core_version": core_version,
                    "core_wheel": wheel.name,
                    "root_mode": args.mode,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
