"""Harness for benchmarks/HANDOFF_EVIDENCE_PREREGISTRATION.md (section 4 and 5).

Builds the synthetic ``invoice`` fixture, runs a predecessor that refactors,
verifies and hands off, applies each preregistered transition, and asks five
policies what the successor should do. Every Entroly call goes through the
installed ``entroly-work`` console script as a subprocess. The successor is a
separate process started with a minimal environment (``env -i`` equivalent).

Labels used in the output: ``fixture`` (synthetic repository), ``mock host``
(harness-generated Ed25519 key), ``real entry point`` (console scripts). No
model or provider is involved.

Usage::

    python benchmarks/handoff_evidence.py --work /tmp/he --entroly-bin <venv>/bin \
        --main-bin <venv-with-main>/bin --reps 3 --out benchmarks/results/handoff_evidence.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

PREREGISTRATION = Path(__file__).with_name("HANDOFF_EVIDENCE_PREREGISTRATION.md")
VERIFY = ["python", "-m", "pytest", "tests/test_pricing.py"]
PINS = ["pytest==9.2.0", "pytest-testmon==2.3.0", "coverage==7.17.3"]
PREDECESSOR = "claude-code:session-a"
SUCCESSOR = "codex:session-b"
ISSUER = "mock-host"
AUDIENCE = "entroly-work"

# ── ground truth, copied from the preregistration table (section 4.2) ──
CASES: dict[str, tuple[str, bool]] = {
    "A1": ("revalidate", True), "A2": ("revalidate", True), "A3": ("revalidate", False),
    "A4": ("revalidate", True), "A5": ("revalidate", True),
    "B1": ("revalidate", True), "B2": ("revalidate", True), "B3": ("revalidate", True),
    "B4": ("revalidate", True), "B5": ("revalidate", False),
    "C1": ("reuse", False), "C2": ("reuse", False), "C3": ("reuse", False),
    "C4": ("reuse", False), "C5": ("reuse", False), "C6": ("reuse", False),
    "D1": ("revalidate", False), "D2": ("revalidate", False), "D3": ("revalidate", False),
}
METHODS = ("M0", "M1", "M2", "M3", "M4")


# ── fixture ──────────────────────────────────────────────────────────

RATELIB = {
    "1.0.0": "ROUND_HALF_UP",
    "1.0.1": "ROUND_HALF_EVEN",
    "1.1.0": "ROUND_HALF_EVEN",
}

FILES = {
    ".gitignore": ".pytest_cache/\n__pycache__/\n.testmondata*\n",
    "pyproject.toml": (
        '[project]\nname = "invoice-fixture"\nversion = "0.1.0"\n'
        'dependencies = ["ratelib==1.0.0"]\n\n'
        '[tool.pytest.ini_options]\npythonpath = ["src"]\ntestpaths = ["tests"]\naddopts = "-q"\n'
    ),
    "src/invoice/__init__.py": "",
    "src/invoice/pricing.py": (
        "import warnings\nfrom decimal import Decimal\n\nfrom ratelib import round_money\n\n"
        "from . import tax\n\n\n"
        "def _legacy_discount(amount):\n"
        '    warnings.warn("legacy discount path", DeprecationWarning, stacklevel=2)\n'
        "    return amount\n\n\n"
        "def total(items, region):\n"
        "    subtotal = Decimal(0)\n"
        "    for item in items:\n"
        '        subtotal += Decimal(item["price"]) * item["qty"]\n'
        "    subtotal = _legacy_discount(subtotal)\n"
        "    return round_money(subtotal * (1 + Decimal(str(tax.rate(region)))))\n"
    ),
    "src/invoice/tax.py": (
        "import json\nfrom pathlib import Path\n\n"
        '_RATES = Path(__file__).with_name("data") / "rates.json"\n\n\n'
        "def rate(region):\n    return json.loads(_RATES.read_text())[region]\n"
    ),
    "src/invoice/data/rates.json": '{"EU": 0.35, "US": 0.07}\n',
    "src/invoice/report.py": "def line(name, amount):\n    return f\"{name}: {amount}\"\n",
    "tests/conftest.py": (
        "import pytest\n\n\n@pytest.fixture\ndef items():\n"
        '    return [{"price": "2.50", "qty": 3}]\n'
    ),
    "tests/test_pricing.py": (
        "from decimal import Decimal\n\nfrom invoice.pricing import total\n\n\n"
        'def test_eu_total(items):\n    assert total(items, "EU") == Decimal("10.13")\n\n\n'
        'def test_us_total(items):\n    assert total(items, "US") == Decimal("8.03")\n'
    ),
    "tests/test_report.py": (
        "from invoice.report import line\n\n\n"
        'def test_line():\n    assert line("a", 1) == "a: 1"\n'
    ),
    "docs/notes.md": "# Notes\n",
}

REFACTOR = (
    "def _subtotal(items):\n"
    "    subtotal = Decimal(0)\n"
    "    for item in items:\n"
    '        subtotal += Decimal(item["price"]) * item["qty"]\n'
    "    return subtotal\n\n\n"
    "def total(items, region):\n"
    "    subtotal = _legacy_discount(_subtotal(items))\n"
    "    return round_money(subtotal * (1 + Decimal(str(tax.rate(region)))))\n"
)


def _wheel(directory: Path, version: str) -> Path:
    files = {
        "ratelib/__init__.py": (
            "from decimal import Decimal, " + RATELIB[version] + "\n\n\n"
            "def round_money(value):\n"
            '    return Decimal(value).quantize(Decimal("0.01"), rounding=' + RATELIB[version] + ")\n"
        ),
        f"ratelib-{version}.dist-info/METADATA": f"Metadata-Version: 2.1\nName: ratelib\nVersion: {version}\n",
        f"ratelib-{version}.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: handoff-evidence\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = []
    for name, text in files.items():
        data = text.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(data)}")
    record.append(f"ratelib-{version}.dist-info/RECORD,,")
    files[f"ratelib-{version}.dist-info/RECORD"] = "\n".join(record) + "\n"
    target = directory / f"ratelib-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return target


def _run(cmd: list[str], cwd: Path, env: dict[str, str], check: bool = False, timeout: float = 600):
    started = time.perf_counter()
    done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if check and done.returncode != 0:
        raise RuntimeError(f"{cmd} failed ({done.returncode}): {done.stdout[-2000:]} {done.stderr[-2000:]}")
    return done, elapsed


def _venv(path: Path, python: str, wheels: Path) -> Path:
    if not (path / "bin" / "python").exists():
        subprocess.run([python, "-m", "venv", str(path)], check=True)
        subprocess.run([str(path / "bin" / "pip"), "install", "-q", "--disable-pip-version-check", *PINS],
                       check=True)
    _install_ratelib(path, wheels, "1.0.0")
    return path


def _install_ratelib(venv: Path, wheels: Path, version: str) -> None:
    subprocess.run(
        [str(venv / "bin" / "pip"), "install", "-q", "--disable-pip-version-check", "--force-reinstall",
         "--no-deps", str(wheels / f"ratelib-{version}-py3-none-any.whl")],
        check=True,
    )


def _environment(venv: Path, home: Path, **extra: str) -> dict[str, str]:
    env = {"PATH": f"{venv / 'bin'}:/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8"}
    env.update(extra)
    return env


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)


def _build_fixture(project: Path) -> None:
    if project.exists():
        shutil.rmtree(project)
    for name, text in FILES.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(project, "init", "-q", "-b", "main")
    _git(project, "config", "user.email", "fixture@example.com")
    _git(project, "config", "user.name", "Fixture")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "base")


def _json(done: subprocess.CompletedProcess) -> dict[str, Any]:
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {"status": "error", "detail": (done.stdout + done.stderr)[-2000:]}


def _tests_executed(output: str) -> int:
    counts = re.findall(r"(\d+) (passed|failed|error|errors)\b", output)
    return sum(int(n) for n, _ in counts)


# ── one case ─────────────────────────────────────────────────────────


class Context:
    def __init__(self, work: Path, args: argparse.Namespace) -> None:
        self.work = work
        self.project = work / "invoice-fixture"
        self.home = work / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.wheels = work / "wheels"
        self.wheels.mkdir(exist_ok=True)
        for version in RATELIB:
            _wheel(self.wheels, version)
        self.venv312 = _venv(work / "venv312", "/usr/bin/python3.12", self.wheels)
        self.venv313 = _venv(work / "venv313", "/usr/bin/python3.13", self.wheels)
        self.entroly = Path(args.entroly_bin) / "entroly-work"
        self.main = Path(args.main_bin) / "entroly-work"
        self.host = work / "host"
        if self.host.exists():
            shutil.rmtree(self.host)
        self.host.mkdir(mode=0o700)
        tool_env = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "LANG": "C.UTF-8"}
        key = _json(_run([str(self.entroly), "--json", "host-keygen", "--private-key-out",
                          str(self.host / "host.key")], work, tool_env, check=True)[0])
        self.trust = self.host / "trust.json"
        self.trust.write_text(json.dumps({
            "schema": "entroly.handoff-trust.v1", "audience": AUDIENCE,
            "issuers": {ISSUER: [key["public_key"]]},
        }), encoding="utf-8")

    def env(self, venv: Path | None = None, state: str = "branch", **extra: str) -> dict[str, str]:
        return _environment(venv or self.venv312, self.home,
                            ENTROLY_DIR=str(self.work / f"state-{state}"), **extra)


def _predecessor(ctx: Context, case: str) -> dict[str, Any]:
    for state in ("branch", "main"):
        shutil.rmtree(ctx.work / f"state-{state}", ignore_errors=True)
    _install_ratelib(ctx.venv312, ctx.wheels, "1.0.0")
    _install_ratelib(ctx.venv313, ctx.wheels, "1.0.0")
    _build_fixture(ctx.project)
    pricing = ctx.project / "src/invoice/pricing.py"
    text = pricing.read_text(encoding="utf-8")
    start = text.index("def total(")
    pricing.write_text(text[:start] + REFACTOR, encoding="utf-8")  # the benign refactor
    env = ctx.env()
    out: dict[str, Any] = {}
    if case != "D3":
        done, _ = _run([*VERIFY, "--testmon"], ctx.project, env)
        out["testmon_seed_passed"] = done.returncode == 0
    for tool, state in ((ctx.entroly, "branch"), (ctx.main, "main")):
        tenv = ctx.env(state=state)
        _run([str(tool), "--project", str(ctx.project), "--json", "claim", "--agent", PREDECESSOR,
              "--task", "extract _subtotal in pricing"], ctx.project, tenv, check=True)
        state_view = _json(_run([str(tool), "--project", str(ctx.project), "--json", "state"],
                                ctx.project, tenv, check=True)[0])
        out[f"workstream_{state}"] = state_view["unfinished"][0]["node_id"]
    benv = ctx.env()
    evidence: list[str] = []
    if case != "D3":
        run = _json(_run([str(ctx.entroly), "--project", str(ctx.project), "--json", "evidence", "run",
                          "--name", "pricing tests", "--agent", PREDECESSOR, "--", *VERIFY],
                         ctx.project, benv, check=True)[0])
        out["record"] = {k: run.get(k) for k in ("record_id", "record_digest", "verdict", "trace", "inputs")}
        evidence = ["--evidence", run["record_id"]]
    handoff = _json(_run([str(ctx.entroly), "--project", str(ctx.project), "--json", "handoff",
                          "--workstream", out["workstream_branch"], "--from-agent", PREDECESSOR,
                          "--to-agent", SUCCESSOR, *evidence], ctx.project, benv, check=True)[0])
    out["handoff_ref"] = handoff["handoff_ref"]
    out["offer_digest"] = handoff["offer_digest"]
    out["repo_id"] = handoff["offer"]["repo_id"]
    main_handoff = _json(_run([str(ctx.main), "--project", str(ctx.project), "--json", "handoff",
                               "--workstream", out["workstream_main"], "--from-agent", PREDECESSOR,
                               "--to-agent", SUCCESSOR], ctx.project, ctx.env(state="main"), check=True)[0])
    out["main_handoff_ref"] = main_handoff["handoff_ref"]
    assertion = ctx.host / f"{case}.assertion"
    assertion.unlink(missing_ok=True)
    _run([str(ctx.entroly), "--json", "issue-assertion", "--private-key-file", str(ctx.host / "host.key"),
          "--issuer", ISSUER, "--audience", AUDIENCE, "--subject", SUCCESSOR, "--workspace", out["repo_id"],
          "--task", out["workstream_branch"], "--handoff", out["offer_digest"], "--out", str(assertion)],
         ctx.work, ctx.env(), check=True)
    out["assertion"] = str(assertion)
    out["summary"] = "pricing tests pass after extracting _subtotal"  # what M0 trusts
    return out


def _edit(ctx: Context, name: str, old: str, new: str) -> None:
    path = ctx.project / name
    text = path.read_text(encoding="utf-8")
    assert old in text, (name, old)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _record_path(ctx: Context, record_id: str) -> Path:
    matches = list((ctx.work / "state-branch").rglob(f"evidence/{record_id.split(':', 1)[1]}.json"))
    assert len(matches) == 1, matches
    return matches[0]


def _alter_record(ctx: Context, record_id: str) -> None:
    """A self-consistent record that is not the one the offer bound (D2)."""
    from entroly import evidence_applicability as ea

    path = _record_path(ctx, record_id)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["name"] = "pricing tests (rewritten after handoff)"
    record.pop("verification_record", None)
    record["record_id"] = "evidence:" + ea._body_digest(record).split(":", 1)[1][:32]
    record["verification_record"] = ea._verification_record(record, "rewritten")
    assert ea.verify_record(record) == []
    path.write_text(json.dumps(record), encoding="utf-8")


def _transition(ctx: Context, case: str, pre: dict[str, Any]) -> tuple[Path, dict[str, str], dict[str, str]]:
    """Apply the case. Returns (successor venv, extra successor env, notes)."""
    venv, extra, notes = ctx.venv312, {}, {}
    if case == "A1":
        _edit(ctx, "src/invoice/tax.py", "[region]\n", "[region] + 0.01\n")
    elif case == "A2":
        _edit(ctx, "src/invoice/data/rates.json", '"EU": 0.35', '"EU": 0.36')
    elif case == "A3":
        _edit(ctx, "src/invoice/pricing.py", "def _subtotal(items):\n", "def _subtotal(items):  # sum lines\n")
    elif case == "A4":
        shadow = ctx.project / "src/invoice/tax"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("def rate(region):\n    return 0.5\n", encoding="utf-8")
    elif case == "A5":
        _edit(ctx, "tests/conftest.py", '"qty": 3', '"qty": 4')
    elif case == "B1":
        _edit(ctx, "pyproject.toml", 'addopts = "-q"', 'addopts = "-q -W error::DeprecationWarning"')
    elif case == "B2":
        extra["PYTEST_ADDOPTS"] = "-W error::DeprecationWarning"
    elif case == "B3":
        _install_ratelib(ctx.venv312, ctx.wheels, "1.0.1")
    elif case == "B4":
        _install_ratelib(ctx.venv312, ctx.wheels, "1.1.0")
    elif case == "B5":
        venv = ctx.venv313
    elif case == "C1":
        _edit(ctx, "docs/notes.md", "# Notes\n", "# Notes\n\nPricing refactor in progress.\n")
    elif case == "C2":
        _edit(ctx, "src/invoice/report.py", "return f", "return  f")
    elif case == "C3":
        _edit(ctx, "tests/test_report.py", '"a: 1"', "'a: 1'")
    elif case == "C4":
        (ctx.project / "scripts").mkdir()
        (ctx.project / "scripts/release.sh").write_text("#!/bin/sh\necho release\n", encoding="utf-8")
    elif case == "C5":
        _git(ctx.project, "add", "-A")
        _git(ctx.project, "commit", "-q", "-m", "extract _subtotal")
    elif case == "C6":
        extra["EDITOR"] = "vi"
    elif case == "D1":
        _record_path(ctx, pre["record"]["record_id"]).unlink()
        (ctx.project / ".testmondata").unlink(missing_ok=True)
        notes["M3"] = "its evidence (.testmondata) deleted"
    elif case == "D2":
        _alter_record(ctx, pre["record"]["record_id"])
        notes["M3"] = "n/a: testmon has no handed-over artifact to alter (no offer binding)"
    elif case == "D3":
        notes["M3"] = "no .testmondata: the predecessor recorded no evidence"
    return venv, extra, notes


def _decide(ctx: Context, case: str, pre: dict[str, Any], venv: Path, extra: dict[str, str],
            notes: dict[str, str]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    decisions["M0"] = {"decision": "reuse" if "pass" in pre["summary"] else "revalidate"}
    decisions["M1"] = {"decision": "revalidate"}

    main_env = ctx.env(venv, state="main", **extra)
    done, seconds = _run([str(ctx.main), "--project", str(ctx.project), "--json", "handoffs",
                          "--to-agent", SUCCESSOR], ctx.project, main_env)
    entry = next((h for h in _json(done).get("handoffs", []) if h.get("ref") == pre["main_handoff_ref"]), None)
    decisions["M2"] = {
        "decision": "reuse" if entry and entry.get("current") else "revalidate",
        "current": entry.get("current") if entry else None,
        "decision_seconds": round(seconds, 3),
    }

    done, seconds = _run([str(ctx.entroly), "--project", str(ctx.project), "--json", "accept",
                          "--handoff-ref", pre["handoff_ref"], "--assertion", pre["assertion"],
                          "--trust-file", str(ctx.trust)], ctx.project, ctx.env(venv, **extra))
    accepted = _json(done)
    if accepted.get("status") != "ok":
        decisions["M4"] = {"decision": "stop", "refusal": accepted.get("code"), "detail": accepted.get("detail")}
    else:
        evidence = accepted.get("evidence") or []
        raw = evidence[0]["decision"] if evidence else "no-evidence"
        details = next(iter((accepted.get("evidence_details") or {}).values()), {})
        decisions["M4"] = {
            "decision": "reuse" if raw == "reuse" else "revalidate",
            "applicability": raw,
            "reasons": evidence[0]["reasons"] if evidence else ["no evidence handed over"],
            "changed": [c.get("key") for c in details.get("changed", [])][:10],
            "epoch": accepted.get("epoch", {}).get("epoch"),
            "decision_seconds": round(seconds, 3),
        }

    if case == "D2":
        decisions["M3"] = {"decision": "n/a", "note": notes["M3"]}
    else:
        done, seconds = _run([*VERIFY, "--testmon"], ctx.project, ctx.env(venv, **extra))
        executed = _tests_executed(done.stdout + done.stderr)
        decisions["M3"] = {
            "decision": "reuse" if executed == 0 else "revalidate",
            "tests_executed": executed,
            "exit_code": done.returncode,
            "seconds": round(seconds, 3),
            **({"note": notes["M3"]} if "M3" in notes else {}),
        }
    return decisions


def run_case(ctx: Context, case: str) -> dict[str, Any]:
    required, consequential = CASES[case]
    result: dict[str, Any] = {"case": case, "required": required, "consequential": consequential}
    try:
        pre = _predecessor(ctx, case)
        result["predecessor"] = {k: pre.get(k) for k in ("record", "offer_digest", "testmon_seed_passed")}
        venv, extra, notes = _transition(ctx, case, pre)
        result["decisions"] = _decide(ctx, case, pre, venv, extra, notes)
        truth, seconds = _run([*VERIFY, "-p", "no:cacheprovider"], ctx.project, ctx.env(venv, **extra))
        result["rerun"] = {"passed": truth.returncode == 0, "exit_code": truth.returncode,
                           "tests": _tests_executed(truth.stdout + truth.stderr), "seconds": round(seconds, 3)}
        result["fixture_defect"] = consequential and truth.returncode == 0
    except Exception as exc:  # recorded, never dropped
        result["harness_error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        _install_ratelib(ctx.venv312, ctx.wheels, "1.0.0")
    return result


# ── scoring ──────────────────────────────────────────────────────────


def score(results: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    revalidate_cases = [r for r in results if r["required"] == "revalidate" and "decisions" in r]
    reuse_cases = [r for r in results if r["required"] == "reuse" and "decisions" in r]
    for method in METHODS:
        stale, missed, redundant, redundant_tests, false_stops, na = [], [], [], 0, [], []
        for r in results:
            d = r.get("decisions", {}).get(method)
            if d is None:
                continue
            if d["decision"] == "n/a":
                na.append(r["case"])
                continue
            if r["required"] == "revalidate" and d["decision"] == "reuse":
                stale.append(r["case"])
                if r["case"] in {"A2", "A4", "B1", "B2", "B3", "B4", "B5", "D1", "D2", "D3"}:
                    missed.append(r["case"])
            if r["required"] == "reuse":
                if d["decision"] == "revalidate":
                    redundant.append(r["case"])
                if d["decision"] == "stop":
                    false_stops.append(r["case"])
                if method == "M3":
                    redundant_tests += d.get("tests_executed", 0)
                elif d["decision"] == "revalidate":
                    redundant_tests += r["rerun"]["tests"]
        denominator = len([r for r in revalidate_cases if r["decisions"].get(method, {}).get("decision") != "n/a"])
        table[method] = {
            "stale_accepted": stale, "stale_count": len(stale), "revalidate_denominator": denominator,
            "stale_upper_95": _clopper_pearson_upper(len(stale), denominator),
            "missed_required_inputs_or_evidence": missed,
            "redundant_reruns": redundant, "redundant_count": len(redundant),
            "redundant_test_executions": redundant_tests, "reuse_allowed_denominator": len(reuse_cases),
            "false_stops": false_stops, "not_applicable": na,
            "correct_next_action": len(results) - len(stale) - len(false_stops) - len(na)
            - sum(1 for r in results if "harness_error" in r),
        }
    return table


def _clopper_pearson_upper(k: int, n: int) -> float | None:
    if n == 0:
        return None
    if k == 0:
        return round(1 - 0.05 ** (1 / n), 4)
    # one-sided upper bound by bisection on the binomial tail
    from math import comb

    def tail(p: float) -> float:
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > 0.05 else (lo, mid)
    return round(hi, 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True)
    parser.add_argument("--entroly-bin", required=True)
    parser.add_argument("--main-bin", required=True)
    parser.add_argument("--main-source", default="", help="checkout the --main-bin entroly was installed from")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ctx = Context(work, args)
    cases = [c for c in args.cases.split(",") if c]
    repetitions = []
    for rep in range(args.reps):
        repetitions.append([run_case(ctx, case) for case in cases])
        print(f"repetition {rep + 1}/{args.reps} done", file=sys.stderr)
    first = repetitions[0]
    stable = all(
        [r.get("decisions", {}).get(m, {}).get("decision") for m in METHODS]
        == [o.get("decisions", {}).get(m, {}).get("decision") for m in METHODS]
        for rep in repetitions[1:] for r, o in zip(rep, first)
    )
    timing = {}
    for method, key in (("M2", "decision_seconds"), ("M4", "decision_seconds"), ("M3", "seconds")):
        timing[method] = {
            case: statistics.median(
                rep[i]["decisions"][method].get(key, 0) for rep in repetitions
                if "decisions" in rep[i] and key in rep[i]["decisions"].get(method, {})
            ) if any(key in rep[i].get("decisions", {}).get(method, {}) for rep in repetitions) else None
            for i, case in enumerate(cases)
        }
    timing["rerun"] = {case: statistics.median(rep[i]["rerun"]["seconds"] for rep in repetitions if "rerun" in rep[i])
                       for i, case in enumerate(cases)}

    def git_head(path: Path) -> str:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    payload = {
        "benchmark_module": "benchmarks/handoff_evidence.py",
        "preregistration": str(PREREGISTRATION.name),
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "labels": {"repository": "fixture", "host": "mock host (harness-generated Ed25519 key)",
                   "entry_point": "real entry point (entroly-work console scripts)",
                   "provider": "not run: no model or provider credentials"},
        "implementation_head": git_head(Path(__file__).resolve().parents[1]),
        "comparator_main_head": git_head(Path(args.main_source)) if args.main_source else None,
        "python": platform.python_version(),
        "fixture_pins": PINS + ["ratelib (local wheels 1.0.0, 1.0.1, 1.1.0)"],
        "repetitions": args.reps,
        "decisions_stable_across_repetitions": stable,
        "cases": first,
        "score": score(first),
        "median_seconds": timing,
        "all_repetitions": repetitions[1:],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["score"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
