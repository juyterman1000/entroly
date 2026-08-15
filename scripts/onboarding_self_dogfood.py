#!/usr/bin/env python3
"""Black-box first-user dogfood against the Entroly repository itself."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 240) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return result


def _json_object(text: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label} did not emit clean JSON: {text[:2000]!r}") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"{label} JSON root must be an object")
    json.dumps(value, allow_nan=False)
    return value


def _nonnegative(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AssertionError(f"{label} must be finite and non-negative, got {value!r}")
    return number


def _validate_local_report(report: dict[str, Any], *, label: str) -> None:
    required = {
        "queries",
        "files_indexed",
        "repo_tokens_indexed",
        "baseline_tokens_per_query",
        "total_tokens_saved",
        "average_reduction_pct",
        "latency_ms",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise AssertionError(f"{label} missing required fields: {missing}")

    rows = report["queries"]
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"{label}.queries must be a non-empty list")

    for field in (
        "files_indexed",
        "repo_tokens_indexed",
        "baseline_tokens_per_query",
        "total_tokens_saved",
        "average_reduction_pct",
    ):
        _nonnegative(report[field], label=f"{label}.{field}")

    latency = report["latency_ms"]
    if not isinstance(latency, dict):
        raise AssertionError(f"{label}.latency_ms must be an object")
    required_latency = {"min", "p95", "max"}
    missing_latency = sorted(required_latency.difference(latency))
    if missing_latency:
        raise AssertionError(
            f"{label}.latency_ms missing required fields: {missing_latency}"
        )
    latency_values = {
        field: _nonnegative(latency[field], label=f"{label}.latency_ms.{field}")
        for field in sorted(required_latency)
    }
    if not (
        latency_values["min"]
        <= latency_values["p95"]
        <= latency_values["max"]
    ):
        raise AssertionError(
            f"{label}.latency_ms must satisfy min <= p95 <= max, got {latency!r}"
        )

    if int(report["files_indexed"]) <= 0 or int(report["repo_tokens_indexed"]) <= 0:
        raise AssertionError(f"{label} did not produce a usable repository index")
    if int(report["total_tokens_saved"]) <= 0 or float(report["average_reduction_pct"]) <= 0:
        raise AssertionError(f"{label} did not demonstrate positive local token reduction")

    for index, row in enumerate(rows):
        baseline = int(row.get("baseline_tokens", -1))
        selected = int(row.get("selected_tokens", -1))
        saved = int(row.get("tokens_saved", -1))
        if min(baseline, selected, saved) < 0:
            raise AssertionError(f"{label}.queries[{index}] contains negative accounting")
        expected = max(0, baseline - selected)
        if saved != expected:
            raise AssertionError(
                f"{label}.queries[{index}] does not reconcile: "
                f"baseline={baseline} selected={selected} saved={saved} expected={expected}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    executable = shutil.which("entroly")
    if not executable:
        raise AssertionError("installed entroly console entrypoint is not on PATH")

    with tempfile.TemporaryDirectory(prefix="entroly-onboarding-") as temp_dir:
        temp = Path(temp_dir)
        home = temp / "home"
        state = temp / "state"
        home.mkdir()
        state.mkdir()

        fake_secret = "dogfood-secret-must-never-appear"
        env = {
            **os.environ,
            "HOME": str(home),
            "USERPROFILE": str(home),
            "ENTROLY_DIR": str(state),
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
            "ENTROLY_AIR_GAP": "1",
            "ENTROLY_NO_DOCKER": "1",
            "PYTHONUTF8": "1",
            "NO_COLOR": "1",
            "OPENAI_API_KEY": fake_secret,
            "ANTHROPIC_API_KEY": fake_secret,
        }

        evidence: dict[str, Any] = {"repo": str(repo), "entrypoint": executable, "steps": {}}
        outputs: list[str] = []

        first = _run([executable, "status"], cwd=repo, env=env)
        outputs += [first["stdout"], first["stderr"]]
        first_text = first["stdout"] + first["stderr"]
        if "Welcome to Entroly" not in first_text:
            raise AssertionError("fresh-user first command did not show the one-time welcome")
        if "entroly simulate" not in first_text or "no API key needed" not in first_text:
            raise AssertionError("welcome did not clearly point to local no-key value proof")
        evidence["steps"]["first_status"] = {"elapsed_ms": first["elapsed_ms"]}

        second = _run([executable, "status"], cwd=repo, env=env)
        outputs += [second["stdout"], second["stderr"]]
        if "Welcome to Entroly" in second["stdout"] + second["stderr"]:
            raise AssertionError("one-time welcome repeated on second command")
        evidence["steps"]["second_status"] = {"elapsed_ms": second["elapsed_ms"]}

        capabilities = _run([executable, "capabilities"], cwd=repo, env=env)
        outputs += [capabilities["stdout"], capabilities["stderr"]]
        evidence["steps"]["capabilities"] = {"elapsed_ms": capabilities["elapsed_ms"]}

        init = _run([executable, "init", "--dry-run"], cwd=repo, env=env)
        outputs += [init["stdout"], init["stderr"]]
        if "preserves_unrelated_configuration" not in init["stdout"] + init["stderr"]:
            raise AssertionError("init --dry-run did not communicate non-destructive config merge")
        evidence["steps"]["init_dry_run"] = {"elapsed_ms": init["elapsed_ms"]}

        verification_path = temp / "verification.json"
        verify = _run(
            [executable, "verify-claims", "--max-files", "120", "--output", str(verification_path)],
            cwd=repo,
            env=env,
            timeout=300,
        )
        outputs += [verify["stdout"], verify["stderr"]]
        if not verification_path.exists():
            raise AssertionError("verify-claims did not create its evidence report")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        json.dumps(verification, allow_nan=False)
        evidence["steps"]["verify_claims"] = {
            "elapsed_ms": verify["elapsed_ms"],
            "report": verification,
        }

        common = ["--json", "--budget", "4096", "--max-files", "120"]
        simulate_run = _run([executable, "simulate", *common], cwd=repo, env=env, timeout=300)
        outputs += [simulate_run["stdout"], simulate_run["stderr"]]
        simulate = _json_object(simulate_run["stdout"], label="simulate")
        _validate_local_report(simulate, label="simulate")
        evidence["steps"]["simulate"] = {"elapsed_ms": simulate_run["elapsed_ms"], "report": simulate}

        perf_run = _run([executable, "perf", *common], cwd=repo, env=env, timeout=300)
        outputs += [perf_run["stdout"], perf_run["stderr"]]
        perf = _json_object(perf_run["stdout"], label="perf")
        _validate_local_report(perf, label="perf")
        evidence["steps"]["perf"] = {"elapsed_ms": perf_run["elapsed_ms"], "report": perf}

        for field in (
            "files_indexed",
            "repo_tokens_indexed",
            "baseline_tokens_per_query",
            "total_tokens_saved",
            "average_reduction_pct",
        ):
            if simulate.get(field) != perf.get(field):
                raise AssertionError(
                    f"simulate/perf disagree on canonical field {field}: "
                    f"simulate={simulate.get(field)!r} perf={perf.get(field)!r}"
                )

        value_run = _run([executable, "value", "--json"], cwd=repo, env=env)
        outputs += [value_run["stdout"], value_run["stderr"]]
        value = _json_object(value_run["stdout"], label="value")
        if "provider_path" not in value or "local_operations" not in value:
            raise AssertionError("value receipt lost provider/local evidence separation")
        evidence["steps"]["value"] = {"elapsed_ms": value_run["elapsed_ms"], "report": value}

        if fake_secret in "\n".join(outputs):
            raise AssertionError("provider credential leaked into onboarding output")

        evidence["summary"] = {
            "total_tokens_saved": int(simulate["total_tokens_saved"]),
            "average_reduction_pct": float(simulate["average_reduction_pct"]),
            "baseline_tokens_per_query": int(simulate["baseline_tokens_per_query"]),
            "repo_tokens_indexed": int(simulate["repo_tokens_indexed"]),
            "files_indexed": int(simulate["files_indexed"]),
            "queries": len(simulate["queries"]),
            "simulate_elapsed_ms": simulate_run["elapsed_ms"],
            "perf_elapsed_ms": perf_run["elapsed_ms"],
            "air_gapped": True,
            "credential_output_clean": True,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence["summary"], sort_keys=True, allow_nan=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
