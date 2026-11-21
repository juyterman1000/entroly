"""`entroly verify-claims` must not report an unavailable engine as success.

Measured on an install where `entroly_core` could not be imported: the body of
the run said

    [INFO] Engine: Python fallback (...); relevance ranking is OFF
    selected=1 tokens=16 savings=0.0% ... (unearned: relevance ranking is OFF)

and then the last line a new user reads said

    RESULTS: 11/11 checks passed (0 failed)

Every counted check passed on exactly the install where selection never read
the query, so "the capability is unavailable" was reported as "the claim was
verified". Zero failures made it look like a clean pass, and the per-check
caveats two sections up did not change the headline.

So these tests assert on the *state*, not on the banner's wording: a run with
the engine dead must not be all-passed, and must carry a machine-readable
verdict a gate can read. The same three buckets, and the same reason for them,
as `entroly doctor` (`checks_passed` / `checks_failed` / `checks_warned`).

Both halves run in a subprocess with one variable between them -- whether
`entroly_core` can be imported -- because patching a flag in-process mixes the
native fragment types into the Python code path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_SHIM = """
{block}
from entroly import verify_claims
raise SystemExit(verify_claims.run(output={out!r}, max_files=8))
"""


def _run(tmp_path, *, native: bool) -> tuple[int, str, dict]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(
        "def main_module_structure():\n    return 'explain the main module'\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "report.json"
    home = tmp_path / "home"
    home.mkdir()

    script = _SHIM.format(
        # `None` in sys.modules makes `import entroly_core` raise, which is what
        # an install without a usable engine looks like from Python.
        block="" if native else "import sys; sys.modules['entroly_core'] = None",
        out=str(out_path),
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": str(REPO),
        "PYTHONIOENCODING": "utf-8",
        "ENTROLY_DIR": str(tmp_path / "state"),
        "ENTROLY_BELIEF_AUTOSEED": "0",
    }
    # Windows cannot start a Python process from a fully hand-built
    # environment: CPython seeds hash randomization from the OS CSPRNG, which
    # needs SystemRoot, and without it the child dies before running a line
    # with "_Py_HashRandomization_Init: failed to get random numbers". Carry
    # over only what the OS itself requires; the isolation that matters here
    # is ENTROLY_DIR, HOME, and whether entroly_core imports.
    for key in ("SystemRoot", "SystemDrive", "COMSPEC", "PATHEXT",
                "TEMP", "TMP", "WINDIR", "NUMBER_OF_PROCESSORS"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert out_path.exists(), proc.stdout + proc.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    return proc.returncode, proc.stdout, report


def _assert_buckets_are_exhaustive(report: dict) -> None:
    """Every counted check lands in exactly one bucket.

    Stated as a sum rather than trusting the counters: the pre-fix code derived
    failures as ``len(results) - passed``, which forced every non-pass into
    "failed" and is why a third state could not exist.
    """
    assert (
        report["passed"] + report["failed"] + report["degraded"]
        == report["checks_total"]
        == len(report["results"])
    ), report


def test_unavailable_engine_is_not_reported_as_all_passed(tmp_path):
    rc, out, report = _run(tmp_path, native=False)

    # The state the test meant to create, asserted before anything is concluded
    # from it -- otherwise a run that quietly found the engine would pass here.
    assert report["engine"] == "python", report
    assert report["relevance_ranking"] is False, report

    _assert_buckets_are_exhaustive(report)
    assert report["degraded"] >= 1, report
    assert report["verdict"] == "degraded", report
    assert report["passed"] < report["checks_total"], (
        "every counted check passed while selection ignored the query"
    )
    statuses = {item["id"]: item["status"] for item in report["results"]}
    assert "DEGRADED" in set(statuses.values()), statuses

    # The headline a new user actually reads must not claim an all-pass.
    total = report["checks_total"]
    banner = out[out.rindex("RESULTS:"):]
    assert f"{total}/{total} checks passed" not in banner, banner

    # Degraded is a third state, not a failure: the exit code still means "a
    # check failed", which is how `entroly doctor` treats a missing native
    # engine (counted warning, `return 1 if checks_failed else 0`) so it stays
    # usable as a CI gate. The not-a-clean-pass signal is `verdict`, which is
    # why the assertions above read it instead of the exit status.
    assert report["failed"] == 0, report
    assert rc == 0, out


def test_unavailable_engine_keeps_the_per_check_caveats(tmp_path):
    """The honest labelling that already existed must survive the headline fix."""
    _rc, out, _report = _run(tmp_path, native=False)

    savings_line = next(line for line in out.splitlines() if "savings=" in line)
    assert "unearned" in savings_line, savings_line
    # Needle lowercased too: the haystack is .lower()ed, so a mixed-case
    # needle could never match whatever the command printed.
    assert "relevance ranking is off" in out.lower()
    banner = out[out.rindex("RESULTS:"):]
    assert "Relevance ranking is OFF" in banner, banner
    assert "entroly-core" in banner, banner  # names the fix


def test_native_engine_still_reports_a_clean_pass(tmp_path):
    """No new caveat on a healthy install -- the verdict has to stay usable."""
    import entroly.engine as engine_module

    if not engine_module._RUST_AVAILABLE:
        pytest.skip("native engine not installed")

    rc, out, report = _run(tmp_path, native=True)

    assert report["engine"] == "rust", report
    assert report["relevance_ranking"] is True, report

    _assert_buckets_are_exhaustive(report)
    assert report["degraded"] == 0, report
    assert report["failed"] == 0, [
        item for item in report["results"] if item["status"] != "PASS"
    ]
    assert report["verdict"] == "pass", report
    assert report["passed"] == report["checks_total"], report
    assert rc == 0, out
    assert "DEGRADED" not in out, out
