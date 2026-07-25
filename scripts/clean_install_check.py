"""Clean-install release gate — Journeys A and B.

Builds the wheel from the current tree, installs it into a throwaway virtual
environment, and runs the documented first commands from a directory that is NOT
the repository, so a source-tree import cannot masquerade as a working install.

    python scripts/clean_install_check.py [--keep]

Exit code 0 only if every documented first-run step succeeds. Any failure is
printed with the captured output; nothing is swallowed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None,
         timeout: int = 900) -> tuple[int, str]:
    """Returns (rc, combined output) for reporting."""
    rc, out, err = _run_split(cmd, cwd=cwd, env=env, timeout=timeout)
    return rc, out + err


def _run_split(cmd: list[str], cwd: str | None = None, env: dict | None = None,
               timeout: int = 900) -> tuple[int, str, str]:
    """Returns (rc, stdout, stderr) kept separate.

    Merging them is unsafe when a check parses program output: any warning a
    dependency prints to stderr would be appended after the real answer, so
    reading "the last line" silently yields the warning instead. That would make
    the source-tree gate below pass on a source-tree import — the exact failure
    this script exists to catch.
    """
    try:
        p = subprocess.run(
            cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, (p.stdout or ""), (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "", f"TIMEOUT after {timeout}s: {' '.join(cmd)}"
    except (OSError, ValueError) as exc:
        return 125, "", f"{type(exc).__name__}: {exc}"


def main() -> int:
    keep = "--keep" in sys.argv
    work = Path(tempfile.mkdtemp(prefix="entroly_clean_"))
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
        if not ok and detail:
            print("        " + detail.strip().replace("\n", "\n        ")[:1500], flush=True)

    try:
        print(f"\n  Clean-install check  (workdir: {work})\n", flush=True)

        dist = work / "dist"
        rc, out = _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
                       cwd=str(REPO))
        if rc != 0:
            rc, out = _run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                            "-w", str(dist), "."], cwd=str(REPO))
        wheels = sorted(dist.glob("entroly-*.whl")) if dist.exists() else []
        check("wheel builds from the current tree", bool(wheels), out)
        if not wheels:
            return 1

        venv = work / "venv"
        rc, out = _run([sys.executable, "-m", "venv", str(venv)])
        check("throwaway virtualenv created", rc == 0, out)
        if rc != 0:
            return 1

        scripts = venv / ("Scripts" if os.name == "nt" else "bin")
        py = scripts / ("python.exe" if os.name == "nt" else "python")

        rc, out = _run([str(py), "-m", "pip", "install", "-q", str(wheels[-1])])
        check("wheel installs into a clean environment", rc == 0, out)
        if rc != 0:
            return 1

        # Run from a neutral cwd so the repo cannot be imported implicitly.
        neutral = work / "elsewhere"
        neutral.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("ENTROLY_") and k != "PYTHONPATH"}

        # stdout only — a stderr warning must not be mistaken for the path.
        rc, out, err = _run_split(
            [str(py), "-c",
             "import entroly,os;print(os.path.dirname(entroly.__file__))"],
            cwd=str(neutral), env=env)
        imported_from = out.strip().splitlines()[-1] if rc == 0 and out.strip() else ""
        from_repo = str(REPO).lower() in imported_from.lower()
        check("import resolves to the installed wheel, not the source tree",
              rc == 0 and bool(imported_from) and not from_repo,
              f"imported from: {imported_from or '(no stdout)'} | stderr: {err.strip()[:300]}")

        entroly_bin = scripts / ("entroly.exe" if os.name == "nt" else "entroly")
        check("console script is installed", entroly_bin.exists(), str(entroly_bin))

        if entroly_bin.exists():
            rc, out = _run([str(entroly_bin), "--help"], cwd=str(neutral), env=env, timeout=180)
            check("documented first command `entroly --help` succeeds", rc == 0, out)

            rc, dout, derr = _run_split(
                [str(entroly_bin), "doctor"], cwd=str(neutral), env=env, timeout=300)
            # doctor now exits non-zero on real failures; on a clean box it must pass.
            check("`entroly doctor` succeeds on a clean install", rc == 0, dout + derr)
            # Match doctor's actual failure markers on STDOUT only. A bare
            # "failed" substring over merged streams was both blind (doctor's red
            # lines read "x Config error: ...", never "failed") and falsely
            # tripped by any unrelated stderr warning containing the word.
            red_lines = [ln for ln in dout.splitlines() if ln.lstrip().startswith("x ")]
            check("doctor reports no failed checks",
                  not red_lines and ", 0 failed" not in dout and " failed" not in dout,
                  "\n".join(red_lines) or dout)

        rc, out = _run([str(py), "-c",
                        "from entroly.sdk import compress; "
                        "print(len(compress('def a():\\n    return 1\\n', 50)))"],
                       cwd=str(neutral), env=env, timeout=300)
        check("Journey B: SDK works without the native core installed", rc == 0, out)

        print()
        failed = [n for n, ok, _ in results if not ok]
        print(f"  {len(results) - len(failed)}/{len(results)} checks passed"
              + (f", {len(failed)} failed" if failed else ""))
        # Written outside `work`, which the finally block deletes — otherwise the
        # result artifact is destroyed on every default run.
        summary = REPO / "benchmarks" / "results" / "clean_install_result.json"
        try:
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps(
                {"passed": len(results) - len(failed), "total": len(results),
                 "failed": failed}, indent=2), encoding="utf-8")
            print(f"  -> {summary}")
        except OSError as exc:
            print(f"  (could not write result artifact: {exc})")
        return 1 if failed else 0
    finally:
        if keep:
            print(f"\n  kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
