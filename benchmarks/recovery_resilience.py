"""Preregistered cross-process recovery-store resilience benchmark.

This suite evaluates the local source-of-truth behind reversible compression.
It does not score token savings or model quality. Independent worker processes
write unique omitted payloads, exit, and a fresh process attempts exact recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA_VERSION = "entroly.recovery-resilience.v1"
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "competitive_evidence_protocol.json"
SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _canonical_source_sha256(paths: Sequence[Path]) -> str:
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


def _protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _phase_config(protocol: dict[str, Any], phase: str) -> dict[str, int]:
    raw = protocol["suites"]["recovery_resilience"][phase]
    return {
        "workers": int(raw["workers"]),
        "entries_per_worker": int(raw["entries_per_worker"]),
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


def _payload(seed: int, worker_id: int, entry_index: int) -> str:
    identity = f"{seed}:{worker_id}:{entry_index}"
    nonce = _sha256(identity)[:24]
    return (
        f"recovery-entry worker={worker_id} index={entry_index} nonce={nonce}\n"
        f"exact-payload-{nonce}-" + (f"segment-{worker_id}-{entry_index} " * 24).strip()
    )


def _store_paths(system: str, state_dir: Path) -> list[Path]:
    if system == "entroly":
        return [state_dir / "recovery.json"]
    if system == "headroom":
        return [state_dir / "ccr.db"]
    raise ValueError(f"unsupported system: {system}")


def _open_store(system: str, state_dir: Path) -> tuple[Any, dict[str, Any]]:
    if system == "entroly":
        from entroly.compression_retrieval_store import CompressionRetrievalStore

        path = state_dir / "recovery.json"
        return CompressionRetrievalStore(path), {
            "backend": "entroly atomic JSON recovery store",
            "path": path.name,
        }
    if system == "headroom":
        from headroom.cache.backends.sqlite import SQLiteBackend
        from headroom.cache.compression_store import CompressionStore

        path = state_dir / "ccr.db"
        return CompressionStore(backend=SQLiteBackend(path)), {
            "backend": "headroom SQLite CCR store",
            "path": path.name,
        }
    raise ValueError(f"unsupported system: {system}")


def _store_payload(system: str, store: Any, payload: str) -> dict[str, str]:
    compressed = "[omitted; recover by emitted reference]"
    if system == "entroly":
        stored = store.put(
            original_text=payload,
            compressed_text=compressed,
            receipt={
                "original_tokens": max(1, len(payload.encode("utf-8")) // 4),
                "compressed_tokens": max(1, len(compressed) // 4),
                "omitted_spans": [
                    {"start_line": 1, "end_line": 2, "reason": "benchmark"}
                ],
            },
            metadata={"suite": SCHEMA_VERSION},
        )
        if len(stored.spans) != 1:
            raise RuntimeError("Entroly did not persist exactly one omitted span")
        return {
            "receipt_id": stored.receipt_id,
            "span_id": stored.spans[0].span_id,
        }
    if system == "headroom":
        hash_key = store.store(
            original=payload,
            compressed=compressed,
            original_tokens=max(1, len(payload.encode("utf-8")) // 4),
            compressed_tokens=max(1, len(compressed) // 4),
            original_item_count=1,
            compressed_item_count=0,
            tool_name="recovery_resilience",
            compression_strategy="preregistered_exact_recovery",
        )
        return {"hash": str(hash_key)}
    raise ValueError(f"unsupported system: {system}")


def _retrieve_payload(system: str, store: Any, reference: dict[str, str]) -> str | None:
    if system == "entroly":
        span = store.get_span(reference["receipt_id"], reference["span_id"])
        return None if span is None else str(span.content)
    if system == "headroom":
        entry = store.retrieve(reference["hash"])
        return None if entry is None else str(entry.original_content)
    raise ValueError(f"unsupported system: {system}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _worker(args: argparse.Namespace) -> int:
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    result: dict[str, Any] = {
        "worker_id": args.worker_id,
        "entries": [],
        "errors": [],
    }
    try:
        store, _metadata = _open_store(args.system, state_dir)
        (state_dir / f"ready-{args.worker_id}").write_text("ready\n", encoding="utf-8")
        deadline = time.monotonic() + args.timeout
        start_flag = state_dir / "start.flag"
        while not start_flag.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for coordinated start")
            time.sleep(0.005)
        for entry_index in range(args.entries):
            payload = _payload(args.seed, args.worker_id, entry_index)
            started = time.perf_counter()
            reference = _store_payload(args.system, store, payload)
            latency_ms = (time.perf_counter() - started) * 1_000
            result["entries"].append(
                {
                    "worker_id": args.worker_id,
                    "entry_index": entry_index,
                    "payload_sha256": _sha256(payload),
                    "reference": reference,
                    "store_latency_ms": round(latency_ms, 6),
                }
            )
    except Exception as error:  # benchmark errors remain in the matrix
        result["errors"].append(
            {"type": type(error).__name__, "message": str(error)[:1000]}
        )
    _write_json(output, result)
    return 0 if not result["errors"] else 1


def _subprocess_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in SECRET_MARKERS)
    }
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    return env


def _adapter(args: argparse.Namespace) -> int:
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    if any(state_dir.iterdir()):
        raise ValueError("adapter state directory must be empty")

    workers: list[tuple[int, subprocess.Popen[str], Path]] = []
    env = _subprocess_env()
    for worker_id in range(args.workers):
        output = state_dir / f"worker-{worker_id}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.recovery_resilience",
            "worker",
            "--system",
            args.system,
            "--state-dir",
            str(state_dir),
            "--worker-id",
            str(worker_id),
            "--entries",
            str(args.entries),
            "--seed",
            str(args.seed),
            "--timeout",
            str(args.timeout),
            "--output",
            str(output),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        workers.append((worker_id, process, output))

    ready_deadline = time.monotonic() + args.timeout
    while len(list(state_dir.glob("ready-*"))) < args.workers:
        if time.monotonic() >= ready_deadline:
            break
        if any(process.poll() is not None for _, process, _ in workers):
            break
        time.sleep(0.01)
    (state_dir / "start.flag").write_text("start\n", encoding="utf-8")

    worker_runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for worker_id, process, output in workers:
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
            exit_code: int | None = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            exit_code = None
        parsed: dict[str, Any] = {
            "worker_id": worker_id,
            "entries": [],
            "errors": [
                {"type": "MissingWorkerResult", "message": "worker output missing"}
            ],
        }
        if output.exists():
            try:
                parsed = json.loads(output.read_text(encoding="utf-8"))
            except Exception as error:
                parsed["errors"] = [
                    {"type": type(error).__name__, "message": str(error)[:1000]}
                ]
        worker_runs.append(
            {
                "worker_id": worker_id,
                "exit_code": exit_code,
                "errors": list(parsed.get("errors", [])),
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
                "stderr_tail": stderr[-1000:],
            }
        )
        rows.extend(dict(row) for row in parsed.get("entries", []))

    recovery_open_error: dict[str, str] | None = None
    try:
        recovery_store, store_metadata = _open_store(args.system, state_dir)
    except Exception as error:
        recovery_store = None
        store_metadata = {"backend": "failed to reopen", "path": "unknown"}
        recovery_open_error = {
            "type": type(error).__name__,
            "message": str(error)[:1000],
        }

    for row in rows:
        recovered: str | None = None
        retrieval_error: dict[str, str] | None = None
        started = time.perf_counter()
        if recovery_store is not None:
            try:
                recovered = _retrieve_payload(
                    args.system, recovery_store, dict(row["reference"])
                )
            except Exception as error:
                retrieval_error = {
                    "type": type(error).__name__,
                    "message": str(error)[:1000],
                }
        retrieval_ms = (time.perf_counter() - started) * 1_000
        recovered_sha = _sha256(recovered) if recovered is not None else None
        row.update(
            {
                "system": args.system,
                "recovered_sha256": recovered_sha,
                "exact": recovered_sha == row["payload_sha256"],
                "retrieve_latency_ms": round(retrieval_ms, 6),
                "retrieval_error": retrieval_error,
            }
        )

    state_files: list[dict[str, Any]] = []
    for base in _store_paths(args.system, state_dir):
        for path in sorted(base.parent.glob(base.name + "*")):
            if path.is_file():
                state_files.append(
                    {"name": path.name, "bytes": path.stat().st_size}
                )

    if args.system == "entroly":
        from entroly import __version__

        participant = {
            "package": "entroly",
            "version": __version__,
            "release_status": "merged source checkout",
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "implementation_sha256": _canonical_source_sha256(
                    (
                        Path(__file__).resolve(),
                        ROOT / "entroly" / "compression_retrieval_store.py",
                    )
                ),
            },
        }
    else:
        participant = {
            "package": "headroom-ai",
            "version": importlib.metadata.version("headroom-ai"),
            "release_status": "published PyPI release",
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "distribution_record_sha256": _distribution_record_sha256(
                    "headroom-ai"
                ),
            },
        }

    result = {
        "system": args.system,
        "participant": participant,
        "configuration": {
            "workers": args.workers,
            "entries_per_worker": args.entries,
            "seed": args.seed,
            **store_metadata,
        },
        "worker_runs": worker_runs,
        "recovery_open_error": recovery_open_error,
        "rows": sorted(rows, key=lambda row: (row["worker_id"], row["entry_index"])),
        "state_files": state_files,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _aggregate(adapter: dict[str, Any]) -> dict[str, Any]:
    config = adapter["configuration"]
    expected = int(config["workers"]) * int(config["entries_per_worker"])
    rows = adapter["rows"]
    worker_runs = adapter["worker_runs"]
    worker_successes = sum(
        run.get("exit_code") == 0 and not run.get("errors") for run in worker_runs
    )
    recovered = sum(row.get("recovered_sha256") is not None for row in rows)
    exact = sum(bool(row.get("exact")) for row in rows)
    incorrect = sum(
        row.get("recovered_sha256") is not None and not bool(row.get("exact"))
        for row in rows
    )
    worker_errors = sum(len(run.get("errors", [])) for run in worker_runs)
    retrieval_errors = sum(row.get("retrieval_error") is not None for row in rows)
    store_latencies = [float(row["store_latency_ms"]) for row in rows]
    retrieve_latencies = [float(row["retrieve_latency_ms"]) for row in rows]
    gates = {
        "worker_exit_success_rate": round(
            worker_successes / int(config["workers"]), 6
        ),
        "write_success_rate": round(len(rows) / expected, 6),
        "restart_recovery_rate": round(recovered / expected, 6),
        "exact_byte_recovery_rate": round(exact / expected, 6),
        "incorrect_payloads": incorrect,
    }
    passed = (
        gates["worker_exit_success_rate"] == 1.0
        and gates["write_success_rate"] == 1.0
        and gates["restart_recovery_rate"] == 1.0
        and gates["exact_byte_recovery_rate"] == 1.0
        and incorrect == 0
        and worker_errors == 0
        and retrieval_errors == 0
        and adapter.get("recovery_open_error") is None
    )
    return {
        "expected_entries": expected,
        "written_entries": len(rows),
        "recovered_entries": recovered,
        "exact_entries": exact,
        "incorrect_payloads": incorrect,
        "worker_errors": worker_errors,
        "retrieval_errors": retrieval_errors,
        "store_latency": _latency_summary(store_latencies),
        "retrieve_latency": _latency_summary(retrieve_latencies),
        "state_bytes": sum(int(item["bytes"]) for item in adapter["state_files"]),
        "gates": gates,
        "passed": passed,
    }


def analyze(
    *,
    protocol: dict[str, Any],
    phase: str,
    adapters: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected_config = _phase_config(protocol, phase)
    systems = sorted(str(adapter["system"]) for adapter in adapters)
    if systems != ["entroly", "headroom"]:
        raise ValueError("recovery comparison requires Entroly and Headroom")
    by_system = {str(adapter["system"]): adapter for adapter in adapters}
    expected_versions = {
        "entroly": str(protocol["comparison"]["entroly"]).split()[0],
        "headroom": str(protocol["comparison"]["headroom"]).split()[1],
    }
    for system, adapter in by_system.items():
        observed_version = str(adapter["participant"]["version"])
        if observed_version != expected_versions[system]:
            raise ValueError(
                f"{system} participant version {observed_version!r} does not match "
                f"frozen version {expected_versions[system]!r}"
            )
        observed = {
            "workers": int(adapter["configuration"]["workers"]),
            "entries_per_worker": int(
                adapter["configuration"]["entries_per_worker"]
            ),
            "seed": int(adapter["configuration"]["seed"]),
        }
        if observed != expected_config:
            raise ValueError(f"{system} adapter configuration drifted from protocol")

    aggregates = {system: _aggregate(by_system[system]) for system in systems}
    if phase == "development":
        label = "Development evidence only; no public leadership claim"
    elif aggregates["entroly"]["passed"] and aggregates["headroom"]["passed"]:
        label = "Both systems satisfy the frozen recovery-integrity gate"
    elif aggregates["entroly"]["passed"]:
        label = "Entroly alone satisfies the frozen recovery-integrity gate"
    else:
        label = "Entroly recovery leadership is not established"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "participants": {
            system: by_system[system]["participant"] for system in systems
        },
        "adapters": by_system,
        "aggregates": aggregates,
        "claim_gate": {
            "label": label,
            "public_leadership_claim_allowed": phase == "holdout"
            and aggregates["entroly"]["passed"]
            and not aggregates["headroom"]["passed"],
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
        raise ValueError("artifact protocol does not match frozen protocol file")
    phase = str(report["phase"])
    adapters = [report["adapters"][system] for system in ("entroly", "headroom")]
    recomputed = analyze(protocol=protocol, phase=phase, adapters=adapters)
    for field in ("participants", "aggregates", "claim_gate"):
        if report[field] != recomputed[field]:
            raise ValueError(f"{field} mismatch")


def _invoke_adapter(
    python: str,
    system: str,
    config: dict[str, int],
    *,
    timeout: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"entroly-{system}-recovery-") as temp:
        command = [
            str(Path(python).resolve()),
            "-m",
            "benchmarks.recovery_resilience",
            "adapter",
            "--system",
            system,
            "--state-dir",
            temp,
            "--workers",
            str(config["workers"]),
            "--entries",
            str(config["entries_per_worker"]),
            "--seed",
            str(config["seed"]),
            "--timeout",
            str(timeout),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout * (config["workers"] + 2),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{system} adapter failed with {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{system} adapter returned invalid JSON: {completed.stdout[-1000:]}"
        ) from error
    result["adapter_stderr_sha256"] = _sha256(completed.stderr)
    result["adapter_stderr_tail"] = completed.stderr[-1000:]
    return result


def _run(args: argparse.Namespace) -> int:
    protocol = _protocol()
    config = _phase_config(protocol, args.phase)
    adapters = [
        _invoke_adapter(sys.executable, "entroly", config, timeout=args.timeout),
        _invoke_adapter(
            args.headroom_python, "headroom", config, timeout=args.timeout
        ),
    ]
    report = analyze(protocol=protocol, phase=args.phase, adapters=adapters)
    verify_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "phase": args.phase,
                "aggregates": report["aggregates"],
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
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=_run)

    adapter = subparsers.add_parser("adapter", help=argparse.SUPPRESS)
    adapter.add_argument("--system", choices=("entroly", "headroom"), required=True)
    adapter.add_argument("--state-dir", type=Path, required=True)
    adapter.add_argument("--workers", type=int, required=True)
    adapter.add_argument("--entries", type=int, required=True)
    adapter.add_argument("--seed", type=int, required=True)
    adapter.add_argument("--timeout", type=float, required=True)
    adapter.set_defaults(func=_adapter)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--system", choices=("entroly", "headroom"), required=True)
    worker.add_argument("--state-dir", type=Path, required=True)
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--entries", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--timeout", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.set_defaults(func=_worker)

    verify = subparsers.add_parser("verify")
    verify.add_argument("input", type=Path)
    verify.set_defaults(func=_verify)

    args = parser.parse_args()
    for field in ("workers", "entries"):
        if hasattr(args, field) and int(getattr(args, field)) < 1:
            parser.error(f"--{field} must be positive")
    if hasattr(args, "timeout") and float(args.timeout) <= 0:
        parser.error("--timeout must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
