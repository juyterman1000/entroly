#!/usr/bin/env python3
"""Black-box first-user dogfood against the Entroly repository itself.

The user journey uses only the installed ``entroly`` console entrypoint. A small
post-simulation diagnostic probe additionally inspects the index persistence
boundary so a cold/warm accounting regression can identify its exact source.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int = 240,
    require_success: bool = True,
) -> dict[str, Any]:
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
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_ms": elapsed_ms,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if require_success and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return result


def _strict_json(text: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label} did not emit clean JSON: {text[:2000]!r}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} JSON root must be an object")
    json.dumps(payload, allow_nan=False)
    return payload


def _assert_finite_nonnegative(value: Any, *, label: str) -> None:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AssertionError(f"{label} must be finite and non-negative, got {value!r}")


def _validate_simulation(payload: dict[str, Any], *, label: str) -> None:
    required = {
        "queries",
        "baseline_tokens_per_query",
        "total_tokens_saved",
        "average_reduction_pct",
        "latency_ms",
        "files_indexed",
        "repo_tokens_indexed",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise AssertionError(f"{label} missing required fields: {missing}")

    rows = payload["queries"]
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"{label} must contain at least one query result")

    _assert_finite_nonnegative(payload["baseline_tokens_per_query"], label=f"{label}.baseline")
    _assert_finite_nonnegative(payload["total_tokens_saved"], label=f"{label}.tokens_saved")
    _assert_finite_nonnegative(payload["average_reduction_pct"], label=f"{label}.reduction_pct")
    _assert_finite_nonnegative(payload["files_indexed"], label=f"{label}.files_indexed")
    _assert_finite_nonnegative(payload["repo_tokens_indexed"], label=f"{label}.repo_tokens_indexed")

    if int(payload["total_tokens_saved"]) <= 0:
        raise AssertionError(
            f"{label} found zero token reduction on Entroly's own repository; "
            "the first-user value demonstration has no wow moment"
        )
    if float(payload["average_reduction_pct"]) <= 0.0:
        raise AssertionError(f"{label} reported a non-positive average reduction")
    if int(payload["files_indexed"]) <= 0 or int(payload["repo_tokens_indexed"]) <= 0:
        raise AssertionError(f"{label} did not produce a usable repository index")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AssertionError(f"{label}.queries[{index}] must be an object")
        baseline = int(row.get("baseline_tokens", -1))
        selected = int(row.get("selected_tokens", -1))
        saved = int(row.get("tokens_saved", -1))
        if baseline < 0 or selected < 0 or saved < 0:
            raise AssertionError(f"{label}.queries[{index}] contains negative token accounting")
        expected = max(0, baseline - selected)
        if saved != expected:
            raise AssertionError(
                f"{label}.queries[{index}] does not reconcile: "
                f"baseline={baseline} selected={selected} saved={saved} expected={expected}"
            )


def _persisted_index_summary(state_dir: Path) -> dict[str, Any]:
    paths = sorted(state_dir.rglob("index.json.gz"))
    if len(paths) != 1:
        return {"status": "missing" if not paths else "ambiguous", "paths": len(paths)}
    path = paths[0]
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    fragments = payload.get("fragments", []) if isinstance(payload, dict) else []
    if isinstance(fragments, dict):
        rows = list(fragments.values())
    elif isinstance(fragments, list):
        rows = fragments
    else:
        rows = []
    return {
        "status": "ok",
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "fragments": len(rows),
        "files": len({str(row.get("source") or "") for row in rows if row.get("source")}),
        "tokens": sum(max(0, int(row.get("token_count", 0) or 0)) for row in rows),
    }


def _reconciliation_probe(repo: Path, env: dict[str, str], max_files: int) -> dict[str, Any]:
    code = r'''
import json
import os
from entroly import auto_index as auto_index_module
from entroly.server import EntrolyEngine

auto_index_module.MAX_FILES = int(os.environ["ENTROLY_DOGFOOD_MAX_FILES"])
engine = EntrolyEngine()
engine.wait_until_warm()

def summary():
    if engine._use_rust:
        rows = [dict(fragment) for fragment in engine._rust.export_fragments()]
    else:
        rows = [
            {"source": fragment.source, "token_count": fragment.token_count}
            for fragment in engine._fragments.values()
        ]
    return {
        "engine": "rust" if engine._use_rust else "python",
        "fragments": len(rows),
        "files": len({str(row.get("source") or "") for row in rows if row.get("source")}),
        "tokens": sum(max(0, int(row.get("token_count", 0) or 0)) for row in rows),
    }

before = summary()
index_result = auto_index_module.auto_index(engine, os.getcwd())
after = summary()
print(json.dumps({"before": before, "after": after, "index_result": index_result}, sort_keys=True))
'''
    probe_env = {**env, "ENTROLY_DOGFOOD_MAX_FILES": str(max_files)}
    result = _run([sys.executable, "-c", code], cwd=repo, env=probe_env, timeout=300)
    return _strict_json(result["stdout"], label="reconciliation probe")


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

        evidence: dict[str, Any] = {
            "repo": str(repo),
            "entrypoint": executable,
            "steps": {},
        }

        first = _run([executable, "status"], cwd=repo, env=env)
        evidence["steps"]["first_status"] = {
            "returncode": first["returncode"],
            "elapsed_ms": first["elapsed_ms"],
        }
        combined_first = first["stdout"] + first["stderr"]
        if "Welcome to Entroly" not in combined_first:
            raise AssertionError("fresh-user first command did not show the one-time welcome")
        if "entroly simulate" not in combined_first or "no API key needed" not in combined_first:
            raise AssertionError("fresh-user welcome does not clearly point to local value proof")

        second = _run([executable, "status"], cwd=repo, env=env)
        evidence["steps"]["second_status"] = {
            "returncode": second["returncode"],
            "elapsed_ms": second["elapsed_ms"],
        }
        if "Welcome to Entroly" in second["stdout"] + second["stderr"]:
            raise AssertionError("one-time welcome repeated on the second command")

        capabilities = _run([executable, "capabilities"], cwd=repo, env=env)
        evidence["steps"]["capabilities"] = {
            "returncode": capabilities["returncode"],
            "elapsed_ms": capabilities["elapsed_ms"],
        }

        init = _run([executable, "init", "--dry-run"], cwd=repo, env=env)
        init_text = init["stdout"] + init["stderr"]
        if "preserves_unrelated_configuration" not in init_text:
            raise AssertionError("init --dry-run did not communicate non-destructive config merge")
        evidence["steps"]["init_dry_run"] = {
            "returncode": init["returncode"],
            "elapsed_ms": init["elapsed_ms"],
        }

        verification_path = temp / "verification.json"
        verify_claims = _run(
            [
                executable,
                "verify-claims",
                "--max-files",
                "120",
                "--output",
                str(verification_path),
            ],
            cwd=repo,
            env=env,
            timeout=300,
        )
        if not verification_path.exists():
            raise AssertionError("verify-claims did not create its machine-readable evidence report")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        json.dumps(verification, allow_nan=False)
        evidence["steps"]["verify_claims"] = {
            "returncode": verify_claims["returncode"],
            "elapsed_ms": verify_claims["elapsed_ms"],
            "report": verification,
        }

        common = ["--json", "--budget", "4096", "--max-files", "120"]
        simulate_result = _run([executable, "simulate", *common], cwd=repo, env=env, timeout=300)
        simulate = _strict_json(simulate_result["stdout"], label="simulate")
        _validate_simulation(simulate, label="simulate")
        persisted_before_warm = _persisted_index_summary(state)
        evidence["steps"]["simulate"] = {
            "returncode": simulate_result["returncode"],
            "elapsed_ms": simulate_result["elapsed_ms"],
            "report": simulate,
            "persisted_index": persisted_before_warm,
        }

        probe = _reconciliation_probe(repo, env, max_files=120)
        evidence["steps"]["reconciliation_probe"] = probe
        print("INDEX_PERSISTENCE_PROBE=" + json.dumps(probe, sort_keys=True), file=sys.stderr)

        perf_result = _run([executable, "perf", *common], cwd=repo, env=env, timeout=300)
        perf = _strict_json(perf_result["stdout"], label="perf")
        _validate_simulation(perf, label="perf")
        persisted_after_warm = _persisted_index_summary(state)
        evidence["steps"]["perf"] = {
            "returncode": perf_result["returncode"],
            "elapsed_ms": perf_result["elapsed_ms"],
            "report": perf,
            "persisted_index": persisted_after_warm,
        }

        for key in ("baseline_tokens_per_query", "total_tokens_saved", "average_reduction_pct"):
            if simulate.get(key) != perf.get(key):
                raise AssertionError(
                    f"simulate/perf disagree on canonical savings field {key}: "
                    f"simulate={simulate.get(key)!r} perf={perf.get(key)!r}"
                )

        if persisted_before_warm.get("status") == "ok":
            if int(persisted_before_warm.get("tokens", -1)) != int(simulate["repo_tokens_indexed"]):
                raise AssertionError(
                    "fresh live index and immediately persisted index disagree on token count: "
                    f"live={simulate['repo_tokens_indexed']} persisted={persisted_before_warm.get('tokens')}"
                )

        for field in ("files_indexed", "repo_tokens_indexed"):
            if simulate.get(field) != perf.get(field):
                raise AssertionError(
                    f"cold/warm index accounting disagrees on {field}: "
                    f"cold={simulate.get(field)!r} warm={perf.get(field)!r}"
                )

        value_result = _run([executable, "value", "--json"], cwd=repo, env=env)
        value = _strict_json(value_result["stdout"], label="value")
        if "provider_path" not in value or "local_operations" not in value:
            raise AssertionError("value receipt lost provider/local evidence separation")
        json.dumps(value, allow_nan=False)
        evidence["steps"]["value"] = {
            "returncode": value_result["returncode"],
            "elapsed_ms": value_result["elapsed_ms"],
            "report": value,
        }

        all_output = "\n".join(
            step["stdout"] + step["stderr"]
            for step in (first, second, capabilities, init, verify_claims, simulate_result, perf_result, value_result)
        )
        if fake_secret in all_output:
            raise AssertionError("a provider credential value leaked into onboarding output")

        evidence["summary"] = {
            "total_tokens_saved": int(simulate["total_tokens_saved"]),
            "average_reduction_pct": float(simulate["average_reduction_pct"]),
            "baseline_tokens_per_query": int(simulate["baseline_tokens_per_query"]),
            "queries": len(simulate["queries"]),
            "simulate_elapsed_ms": simulate_result["elapsed_ms"],
            "perf_elapsed_ms": perf_result["elapsed_ms"],
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