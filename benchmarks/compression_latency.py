"""Evidence-gated warm and cold compression latency comparison.

The suite uses the same byte-identical fixtures and public compressor entry
points as the compression gauntlet. Latency is eligible for a scoped claim only
when both participants complete the matrix deterministically, retain every
preregistered evidence needle, and never inflate tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from benchmarks.compression_gauntlet import Scenario, _messages, build_scenarios


SCHEMA_VERSION = "entroly.compression-latency.v1"
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "compression_latency_protocol.json"
_ENV_ALLOWLIST = (
    "APPDATA",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
)


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def _distribution_record_sha256(package: str) -> str | None:
    try:
        record = importlib.metadata.distribution(package).read_text("RECORD")
    except importlib.metadata.PackageNotFoundError:
        return None
    return _sha256(record) if record else None


def _versions(packages: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _phase_config(protocol: dict[str, Any], phase: str) -> dict[str, int]:
    raw = protocol["phases"][phase]
    return {
        "warmups": int(raw["warmups"]),
        "warm_runs_per_fixture": int(raw["warm_runs_per_fixture"]),
        "cold_runs_per_fixture": int(raw["cold_runs_per_fixture"]),
        "bootstrap_iterations": int(raw["bootstrap_iterations"]),
        "seed": int(raw["seed"]),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [round(float(value), 6) for value in values]
    return {
        "samples_ms": samples,
        "p50_ms": round(statistics.median(samples), 6) if samples else 0.0,
        "p95_ms": round(_percentile(samples, 0.95), 6),
    }


def _subprocess_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ROOT),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def _make_compressor(
    system: str, payload: dict[str, Any]
) -> tuple[Callable[[Scenario], tuple[str, dict[str, Any]]], dict[str, Any]]:
    if system == "entroly":
        from entroly import __version__
        from entroly.compression_proxy import compress_proxy_payload

        budget = int(payload["entroly_budget_tokens"])

        def compress(scenario: Scenario) -> tuple[str, dict[str, Any]]:
            result = compress_proxy_payload(
                {"model": payload["model"], "messages": _messages(scenario)},
                query=scenario.query,
                budget_tokens=budget,
                include_receipt_header=False,
            )
            return str(result.body["messages"][-1]["content"]), {
                "changed": result.changed,
                "compressed_blocks": result.receipt.compressed_blocks,
            }

        participant = {
            "package": "entroly",
            "version": __version__,
            "entry_point": "entroly.compression_proxy.compress_proxy_payload",
            "configuration": {
                "budget_tokens": budget,
                "receipt_header": False,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": _versions(("tiktoken",)),
                "implementation_sha256": _source_sha256(
                    (
                        Path(__file__).resolve(),
                        ROOT / "benchmarks" / "compression_gauntlet.py",
                        ROOT / "entroly" / "compression_proxy.py",
                        ROOT / "entroly" / "evidence_locked_compression.py",
                    )
                ),
            },
        }
        return compress, participant

    if system == "headroom":
        from headroom import compress as headroom_compress

        def compress(scenario: Scenario) -> tuple[str, dict[str, Any]]:
            messages = json.loads(json.dumps(_messages(scenario)))
            result = headroom_compress(
                messages,
                model=str(payload["model"]),
                protect_recent=0,
                savings_profile=str(payload["headroom_savings_profile"]),
            )
            return str(result.messages[-1]["content"]), {
                "tokens_before": int(result.tokens_before),
                "tokens_after": int(result.tokens_after),
                "transforms": list(result.transforms_applied),
            }

        participant = {
            "package": "headroom-ai",
            "version": importlib.metadata.version("headroom-ai"),
            "entry_point": "headroom.compress",
            "configuration": {
                "model": payload["model"],
                "protect_recent": 0,
                "savings_profile": payload["headroom_savings_profile"],
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": _versions(
                    ("headroom-ai", "litellm", "onnxruntime", "tiktoken")
                ),
                "distribution_record_sha256": _distribution_record_sha256(
                    "headroom-ai"
                ),
            },
        }
        return compress, participant

    raise ValueError(f"unsupported system: {system}")


def _scenario_from_payload(payload: dict[str, Any]) -> Scenario:
    raw = payload["scenario"]
    return Scenario(
        scenario_id=str(raw["scenario_id"]),
        category=str(raw["category"]),
        query=str(raw["query"]),
        content=str(raw["content"]),
        evidence_needles=tuple(str(value) for value in raw["evidence_needles"]),
    )


def _adapter(args: argparse.Namespace) -> int:
    payload = json.load(sys.stdin)
    scenario = _scenario_from_payload(payload)
    import_started = time.perf_counter()
    compress, participant = _make_compressor(args.system, payload)
    import_ms = (time.perf_counter() - import_started) * 1_000

    for _ in range(int(payload["warmups"])):
        compress(scenario)

    samples: list[float] = []
    output_hashes: list[str] = []
    outputs: dict[str, str] = {}
    native_metrics: list[dict[str, Any]] = []
    for _ in range(int(payload["runs"])):
        started = time.perf_counter()
        output, metrics = compress(scenario)
        samples.append((time.perf_counter() - started) * 1_000)
        output_hash = _sha256(output)
        output_hashes.append(output_hash)
        outputs.setdefault(output_hash, output)
        native_metrics.append(metrics)

    print(
        json.dumps(
            {
                "system": args.system,
                "scenario_id": scenario.scenario_id,
                "participant": participant,
                "import_ms": round(import_ms, 6),
                "call_latency_ms": [round(value, 6) for value in samples],
                "cold_latency_ms": round(import_ms + samples[0], 6)
                if int(payload["warmups"]) == 0 and len(samples) == 1
                else None,
                "output_hashes": output_hashes,
                "outputs": outputs,
                "native_metrics": native_metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _invoke_adapter(
    *,
    python: str,
    system: str,
    scenario: Scenario,
    protocol: dict[str, Any],
    warmups: int,
    runs: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": protocol["model"],
        "entroly_budget_tokens": protocol["entroly_budget_tokens"],
        "headroom_savings_profile": protocol["headroom_savings_profile"],
        "warmups": warmups,
        "runs": runs,
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "category": scenario.category,
            "query": scenario.query,
            "content": scenario.content,
            "evidence_needles": list(scenario.evidence_needles),
        },
    }
    command = [
        str(Path(python).resolve()),
        "-m",
        "benchmarks.compression_latency",
        "adapter",
        "--system",
        system,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_subprocess_env(),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "system": system,
            "scenario_id": scenario.scenario_id,
            "errors": [{"type": "TimeoutExpired", "message": str(error)[:1000]}],
            "process_wall_ms": round((time.perf_counter() - started) * 1_000, 6),
        }
    wall_ms = (time.perf_counter() - started) * 1_000
    if completed.returncode != 0:
        return {
            "system": system,
            "scenario_id": scenario.scenario_id,
            "errors": [
                {
                    "type": "AdapterExit",
                    "message": (completed.stderr or completed.stdout)[-2000:],
                    "exit_code": completed.returncode,
                }
            ],
            "process_wall_ms": round(wall_ms, 6),
            "stderr_sha256": _sha256(completed.stderr),
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "system": system,
            "scenario_id": scenario.scenario_id,
            "errors": [{"type": "JSONDecodeError", "message": str(error)[:1000]}],
            "process_wall_ms": round(wall_ms, 6),
            "stdout_sha256": _sha256(completed.stdout),
            "stderr_sha256": _sha256(completed.stderr),
        }
    result["errors"] = []
    result["process_wall_ms"] = round(wall_ms, 6)
    result["stderr_sha256"] = _sha256(completed.stderr)
    return result


def _token_counter() -> tuple[Callable[[str], int], str]:
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")

    def count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    return count, importlib.metadata.version("tiktoken")


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(statistics.mean(math.log(value) for value in values))


def _bootstrap_speedup(
    samples: dict[str, dict[str, list[float]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    scenario_ids = sorted(samples["entroly"])

    def statistic(source: dict[str, dict[str, list[float]]]) -> float:
        ratios = []
        for scenario_id in scenario_ids:
            entroly = statistics.median(source["entroly"][scenario_id])
            headroom = statistics.median(source["headroom"][scenario_id])
            if entroly <= 0 or headroom <= 0:
                return 0.0
            ratios.append(headroom / entroly)
        return _geometric_mean(ratios)

    observed = statistic(samples)
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(iterations):
        resampled: dict[str, dict[str, list[float]]] = {
            "entroly": {},
            "headroom": {},
        }
        for system in ("entroly", "headroom"):
            for scenario_id in scenario_ids:
                values = samples[system][scenario_id]
                resampled[system][scenario_id] = [
                    values[rng.randrange(len(values))] for _ in values
                ]
        bootstrapped.append(statistic(resampled))
    return {
        "geometric_mean_speedup": round(observed, 6),
        "ci95_lower": round(_percentile(bootstrapped, 0.025), 6),
        "ci95_upper": round(_percentile(bootstrapped, 0.975), 6),
    }


def analyze(
    *,
    protocol: dict[str, Any],
    phase: str,
    warm_cells: Sequence[dict[str, Any]],
    cold_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    config = _phase_config(protocol, phase)
    scenarios = build_scenarios()
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    expected_pairs = {
        (system, scenario.scenario_id)
        for system in ("entroly", "headroom")
        for scenario in scenarios
    }
    observed_warm = {
        (str(cell["system"]), str(cell["scenario_id"])) for cell in warm_cells
    }
    if observed_warm != expected_pairs or len(warm_cells) != len(expected_pairs):
        raise ValueError("warm adapter matrix is incomplete or duplicated")

    expected_cold_count = len(expected_pairs) * config["cold_runs_per_fixture"]
    if len(cold_cells) != expected_cold_count:
        raise ValueError("cold adapter matrix has the wrong number of runs")
    cold_counts: dict[tuple[str, str], int] = {}
    cold_replicates: dict[tuple[str, str], set[int]] = {}
    for cell in cold_cells:
        key = (str(cell["system"]), str(cell["scenario_id"]))
        cold_counts[key] = cold_counts.get(key, 0) + 1
        cold_replicates.setdefault(key, set()).add(int(cell.get("replicate", -1)))
    if set(cold_counts) != expected_pairs or any(
        count != config["cold_runs_per_fixture"] for count in cold_counts.values()
    ):
        raise ValueError("cold adapter matrix is incomplete or unbalanced")
    expected_replicates = set(range(config["cold_runs_per_fixture"]))
    if any(values != expected_replicates for values in cold_replicates.values()):
        raise ValueError("cold adapter replicate ids are missing or duplicated")

    count_tokens, tokenizer_version = _token_counter()
    participants: dict[str, dict[str, Any]] = {}
    grouped_warm = {
        (str(cell["system"]), str(cell["scenario_id"])): cell
        for cell in warm_cells
    }
    grouped_cold: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cold_cells:
        grouped_cold.setdefault(
            (str(cell["system"]), str(cell["scenario_id"])), []
        ).append(cell)

    measurements: list[dict[str, Any]] = []
    mode_samples: dict[str, dict[str, dict[str, list[float]]]] = {
        "warm": {"entroly": {}, "headroom": {}},
        "cold": {"entroly": {}, "headroom": {}},
    }
    for system, scenario_id in sorted(expected_pairs):
        scenario = scenario_by_id[scenario_id]
        warm = grouped_warm[(system, scenario_id)]
        cold = grouped_cold[(system, scenario_id)]
        all_cells = [warm, *cold]
        errors = [error for cell in all_cells for error in cell.get("errors", [])]
        participant_records = [
            cell.get("participant")
            for cell in all_cells
            if isinstance(cell.get("participant"), dict)
        ]
        if participant_records:
            canonical = _canonical_sha256(participant_records[0])
            if any(_canonical_sha256(item) != canonical for item in participant_records):
                errors.append(
                    {
                        "type": "ParticipantDrift",
                        "message": "participant metadata changed between runs",
                    }
                )
            prior = participants.get(system)
            if prior is not None and _canonical_sha256(prior) != canonical:
                errors.append(
                    {
                        "type": "ParticipantDrift",
                        "message": "participant metadata changed between fixtures",
                    }
                )
            participants.setdefault(system, participant_records[0])

        output_hashes: list[str] = []
        outputs: dict[str, str] = {}
        for cell in all_cells:
            output_hashes.extend(str(value) for value in cell.get("output_hashes", []))
            for output_hash, output in cell.get("outputs", {}).items():
                if _sha256(str(output)) != str(output_hash):
                    errors.append(
                        {
                            "type": "OutputHashMismatch",
                            "message": f"output {output_hash} failed its SHA-256",
                        }
                    )
                outputs[str(output_hash)] = str(output)

        evidence_missing = sorted(
            {
                needle
                for output in outputs.values()
                for needle in scenario.evidence_needles
                if needle not in output
            }
        )
        input_tokens = count_tokens(scenario.content)
        output_token_counts = {
            output_hash: count_tokens(output) for output_hash, output in outputs.items()
        }
        deterministic = bool(output_hashes) and len(set(output_hashes)) == 1
        no_inflation = bool(output_token_counts) and all(
            tokens <= input_tokens for tokens in output_token_counts.values()
        )
        warm_samples = [float(value) for value in warm.get("call_latency_ms", [])]
        cold_samples = [
            float(cell["cold_latency_ms"])
            for cell in cold
            if cell.get("cold_latency_ms") is not None
        ]
        complete = (
            not errors
            and len(warm_samples) == config["warm_runs_per_fixture"]
            and len(cold_samples) == config["cold_runs_per_fixture"]
            and len(output_hashes)
            == config["warm_runs_per_fixture"] + config["cold_runs_per_fixture"]
        )
        valid = (
            complete
            and deterministic
            and not evidence_missing
            and no_inflation
        )
        mode_samples["warm"][system][scenario_id] = warm_samples
        mode_samples["cold"][system][scenario_id] = cold_samples
        measurements.append(
            {
                "system": system,
                "scenario_id": scenario_id,
                "category": scenario.category,
                "input_sha256": _sha256(scenario.content),
                "input_tokens": input_tokens,
                "output_hashes": sorted(set(output_hashes)),
                "outputs": outputs,
                "output_token_counts": output_token_counts,
                "evidence_missing": evidence_missing,
                "deterministic": deterministic,
                "no_token_inflation": no_inflation,
                "complete": complete,
                "valid": valid,
                "errors": errors,
                "warm_latency": _latency_summary(warm_samples),
                "cold_latency": _latency_summary(cold_samples),
                "cold_import_latency": _latency_summary(
                    [float(cell["import_ms"]) for cell in cold if "import_ms" in cell]
                ),
                "cold_first_call_latency": _latency_summary(
                    [
                        float(cell["call_latency_ms"][0])
                        for cell in cold
                        if cell.get("call_latency_ms")
                    ]
                ),
            }
        )

    expected_versions = {
        system: str(record["version"])
        for system, record in protocol["participants"].items()
    }
    version_gates = {
        system: str(participants.get(system, {}).get("version", ""))
        == expected_version
        for system, expected_version in expected_versions.items()
    }

    quality_passed = {
        system: version_gates[system]
        and all(row["valid"] for row in measurements if row["system"] == system)
        for system in ("entroly", "headroom")
    }
    participant_gates = {
        system: {
            "expected_version": expected_versions[system],
            "observed_version": str(
                participants.get(system, {}).get("version", "")
            ),
            "passed": version_gates[system],
        }
        for system in ("entroly", "headroom")
    }
    mode_analysis: dict[str, Any] = {}
    for mode_index, mode in enumerate(("warm", "cold")):
        if all(quality_passed.values()):
            speedup = _bootstrap_speedup(
                mode_samples[mode],
                iterations=config["bootstrap_iterations"],
                seed=config["seed"] + mode_index,
            )
        else:
            speedup = {
                "geometric_mean_speedup": 0.0,
                "ci95_lower": 0.0,
                "ci95_upper": 0.0,
            }
        allowed = (
            phase == "holdout"
            and all(quality_passed.values())
            and speedup["ci95_lower"]
            > float(protocol["claim_policy"]["minimum_bootstrap_lower_speedup"])
        )
        mode_analysis[mode] = {
            **speedup,
            "entroly_faster_claim_allowed": allowed,
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "environment": {
            "controller_python": platform.python_version(),
            "controller_platform": platform.platform(),
            "shared_tokenizer": f"tiktoken=={tokenizer_version}/o200k_base",
        },
        "participants": participants,
        "fixtures": [scenario.public_record() for scenario in scenarios],
        "raw": {"warm_cells": list(warm_cells), "cold_cells": list(cold_cells)},
        "measurements": measurements,
        "participant_gates": participant_gates,
        "quality_gates": quality_passed,
        "mode_analysis": mode_analysis,
        "claim_gate": {
            "any_latency_claim_allowed": any(
                record["entroly_faster_claim_allowed"]
                for record in mode_analysis.values()
            ),
            "aggregate_product_score_allowed": False,
            "universal_superiority_claim_allowed": False,
        },
    }
    report["payload_sha256"] = _canonical_sha256(report)
    return report


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    stored_hash = report.get("payload_sha256")
    unhashed = {key: value for key, value in report.items() if key != "payload_sha256"}
    if stored_hash != _canonical_sha256(unhashed):
        raise ValueError("payload_sha256 mismatch")
    protocol = report["protocol"]
    if report.get("protocol_sha256") != _canonical_sha256(protocol):
        raise ValueError("protocol_sha256 mismatch")
    if protocol != _protocol():
        raise ValueError("artifact protocol does not match frozen protocol")
    recomputed = analyze(
        protocol=protocol,
        phase=str(report["phase"]),
        warm_cells=report["raw"]["warm_cells"],
        cold_cells=report["raw"]["cold_cells"],
    )
    for field in (
        "participants",
        "fixtures",
        "measurements",
        "participant_gates",
        "quality_gates",
        "mode_analysis",
        "claim_gate",
    ):
        if report[field] != recomputed[field]:
            raise ValueError(f"{field} mismatch")


def _run(args: argparse.Namespace) -> int:
    protocol = _protocol()
    config = _phase_config(protocol, args.phase)
    scenarios = build_scenarios()
    python_by_system = {
        "entroly": sys.executable,
        "headroom": args.headroom_python,
    }
    order = [
        (system, scenario)
        for system in ("entroly", "headroom")
        for scenario in scenarios
    ]
    rng = random.Random(config["seed"])
    rng.shuffle(order)
    warm_cells = [
        _invoke_adapter(
            python=python_by_system[system],
            system=system,
            scenario=scenario,
            protocol=protocol,
            warmups=config["warmups"],
            runs=config["warm_runs_per_fixture"],
            timeout=args.timeout,
        )
        for system, scenario in order
    ]

    cold_order = [
        (system, scenario, replicate)
        for replicate in range(config["cold_runs_per_fixture"])
        for system in ("entroly", "headroom")
        for scenario in scenarios
    ]
    rng.shuffle(cold_order)
    cold_cells = [
        {
            **_invoke_adapter(
                python=python_by_system[system],
                system=system,
                scenario=scenario,
                protocol=protocol,
                warmups=0,
                runs=1,
                timeout=args.timeout,
            ),
            "replicate": replicate,
        }
        for system, scenario, replicate in cold_order
    ]
    report = analyze(
        protocol=protocol,
        phase=args.phase,
        warm_cells=warm_cells,
        cold_cells=cold_cells,
    )
    verify_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "phase": args.phase,
                "quality_gates": report["quality_gates"],
                "mode_analysis": report["mode_analysis"],
                "claim_gate": report["claim_gate"],
                "payload_sha256": report["payload_sha256"],
            },
            indent=2,
        )
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    report = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(report)
    print(
        f"verified phase={report['phase']} payload_sha256={report['payload_sha256']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--phase", choices=("development", "holdout"), required=True)
    run.add_argument("--headroom-python", required=True)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=_run)

    adapter = subparsers.add_parser("adapter", help=argparse.SUPPRESS)
    adapter.add_argument("--system", choices=("entroly", "headroom"), required=True)
    adapter.set_defaults(func=_adapter)

    verify = subparsers.add_parser("verify")
    verify.add_argument("input", type=Path)
    verify.set_defaults(func=_verify)

    args = parser.parse_args()
    if hasattr(args, "timeout") and float(args.timeout) <= 0:
        parser.error("--timeout must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
