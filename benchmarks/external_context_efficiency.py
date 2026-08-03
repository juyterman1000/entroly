"""Fail-closed tooling for external context-efficiency experiments.

This module deliberately imports no Entroly production package. The `A_full`
arm must execute in a process where Entroly is absent, not merely disabled by a
label in the result file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROTOCOL_VERSION = "external-context-efficiency-v1"
SCHEMA_PATH = Path(__file__).with_name("external_context_efficiency.schema.json")

_IDENTITY_FIELDS = (
    "benchmark",
    "benchmark_version",
    "benchmark_task_id",
    "pair_id",
    "model_id",
    "agent_id",
    "provider_id",
    "harness_commit",
    "environment_digest",
    "task_input_digest",
    "seed",
)
_PROXY_OVERRIDE_KEYS = {
    "ANTHROPIC_BASE_URL",
    "AZURE_OPENAI_ENDPOINT",
    "GOOGLE_GEMINI_BASE_URL",
    "LITELLM_PROXY_API_BASE",
    "LITELLM_PROXY_URL",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
}
_SECRET_NAME_RE = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|token)$",
    re.IGNORECASE,
)
_SECRET_ARGUMENT_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{8,}|api[_-]?key\s*[:=]|authorization\s*[:=])",
    re.IGNORECASE,
)


class BenchmarkContractError(ValueError):
    """A benchmark artifact or execution violates the frozen contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_digest", None)
    return sha256_bytes(_canonical_bytes(unsigned))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _semantic_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    arm = record.get("arm")

    if arm == "A_full":
        forbidden_values = {
            "entroly_commit": record.get("entroly_commit"),
            "entroly_config_digest": record.get("entroly_config_digest"),
            "sufficiency_verdict": record.get("sufficiency_verdict"),
            "calibration_policy_id": record.get("calibration_policy_id"),
            "false_sufficient": record.get("false_sufficient"),
        }
        for field, value in forbidden_values.items():
            if value is not None:
                errors.append(f"A_full requires {field}=null")
        for field in ("compression_time_ms", "recovery_tokens", "recovery_calls"):
            if record.get(field) not in (None, 0, 0.0):
                errors.append(f"A_full requires {field} to be zero or null")
        before = record.get("context_tokens_before")
        after = record.get("context_tokens_after")
        if before is not None and after is not None and before != after:
            errors.append("A_full requires context_tokens_after == context_tokens_before")

    if arm == "E_entroly_no_recovery":
        for field in ("recovery_tokens", "recovery_calls"):
            if record.get(field) not in (None, 0):
                errors.append(f"E_entroly_no_recovery requires {field}=0 or null")

    verdict = record.get("sufficiency_verdict")
    evidence_present = record.get("required_evidence_present")
    false_sufficient = record.get("false_sufficient")
    if false_sufficient is True and verdict != "sufficient_calibrated":
        errors.append(
            "false_sufficient=true requires sufficiency_verdict=sufficient_calibrated"
        )
    if verdict == "sufficient_calibrated" and evidence_present is False:
        if false_sufficient is not True:
            errors.append(
                "calibrated sufficiency with missing evidence requires false_sufficient=true"
            )

    if record.get("excluded") is False and record.get("exclusion_reason") is not None:
        errors.append("non-excluded records require exclusion_reason=null")

    return errors


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate one row against JSON Schema plus semantic invariants."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError("jsonschema is required to validate benchmark rows") from exc

    schema_errors = sorted(
        Draft202012Validator(_schema()).iter_errors(dict(record)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in schema_errors
    ]
    errors.extend(_semantic_errors(record))
    if errors:
        raise BenchmarkContractError("; ".join(errors))


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BenchmarkContractError(
                f"line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise BenchmarkContractError(f"line {line_number}: row must be an object")
        records.append(value)
    if not records:
        raise BenchmarkContractError("result JSONL contains no records")
    return records


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    required_arms: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate rows, pairing, treatment identity, and task-input equality."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        try:
            validate_record(record)
        except BenchmarkContractError as exc:
            raise BenchmarkContractError(f"record {index}: {exc}") from exc
        groups[str(record["pair_id"])].append(record)

    required = set(required_arms)
    arm_counts: Counter[str] = Counter()
    excluded_pairs = 0
    for pair_id, pair in sorted(groups.items()):
        by_arm: dict[str, Mapping[str, Any]] = {}
        for record in pair:
            arm = str(record["arm"])
            if arm in by_arm:
                raise BenchmarkContractError(
                    f"pair {pair_id}: duplicate result for arm {arm}"
                )
            by_arm[arm] = record
            arm_counts[arm] += 1

        if required - set(by_arm):
            missing = ", ".join(sorted(required - set(by_arm)))
            raise BenchmarkContractError(f"pair {pair_id}: missing required arms: {missing}")
        if len(by_arm) > 1 and "A_full" not in by_arm:
            raise BenchmarkContractError(
                f"pair {pair_id}: compared treatment arms require A_full"
            )

        reference = by_arm.get("A_full") or next(iter(by_arm.values()))
        for arm, record in sorted(by_arm.items()):
            for field in _IDENTITY_FIELDS:
                if record.get(field) != reference.get(field):
                    raise BenchmarkContractError(
                        f"pair {pair_id}: {field} differs in arm {arm}"
                    )
            baseline_tokens = reference.get("context_tokens_before")
            treatment_tokens = record.get("context_tokens_before")
            if (
                baseline_tokens is not None
                and treatment_tokens is not None
                and baseline_tokens != treatment_tokens
            ):
                raise BenchmarkContractError(
                    f"pair {pair_id}: context_tokens_before differs in arm {arm}"
                )

        if all(bool(record.get("excluded")) for record in pair):
            excluded_pairs += 1

    canonical = [dict(record) for record in records]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "records": len(records),
        "pairs": len(groups),
        "excluded_pairs": excluded_pairs,
        "arms": dict(sorted(arm_counts.items())),
        "artifact_digest": sha256_bytes(_canonical_bytes(canonical)),
    }


def validate_jsonl(
    path: str | os.PathLike[str],
    *,
    required_arms: Iterable[str] = (),
) -> dict[str, Any]:
    return validate_records(load_jsonl(path), required_arms=required_arms)


def _is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME_RE.search(name))


def sanitize_baseline_environment(
    environment: Mapping[str, str] | None = None,
    *,
    preserve_provider_base_urls: bool = False,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Remove Entroly and local-proxy hooks from a child environment."""
    clean = dict(os.environ if environment is None else environment)
    removed: list[str] = []
    for key in sorted(tuple(clean)):
        if key.startswith("ENTROLY_") or (
            not preserve_provider_base_urls and key in _PROXY_OVERRIDE_KEYS
        ):
            removed.append(key)
            clean.pop(key, None)
    return clean, tuple(removed)


def _environment_digest(environment: Mapping[str, str]) -> str:
    safe_values = {
        key: "<set>" if _is_secret_name(key) else value
        for key, value in sorted(environment.items())
    }
    return sha256_bytes(_canonical_bytes(safe_values))


def _assert_baseline_command(command: Sequence[str]) -> None:
    if not command:
        raise BenchmarkContractError("baseline command is empty")
    for argument in command:
        if "entroly" in argument.casefold():
            raise BenchmarkContractError(
                "A_full command must not reference Entroly executables or modules"
            )
        if _SECRET_ARGUMENT_RE.search(argument):
            raise BenchmarkContractError(
                "credentials must be passed through environment variables, not arguments"
            )


def _assert_no_entroly_modules(module_names: Iterable[str] | None = None) -> None:
    names = sys.modules if module_names is None else module_names
    loaded = sorted(
        name for name in names if name == "entroly" or name.startswith("entroly.")
    )
    if loaded:
        preview = ", ".join(loaded[:5])
        raise BenchmarkContractError(
            f"A_full process already loaded Entroly modules: {preview}"
        )


def build_baseline_manifest(
    *,
    benchmark: str,
    benchmark_version: str,
    run_id: str,
    model_id: str,
    agent_id: str,
    provider_id: str | None,
    harness_commit: str | None,
    task_set: str | os.PathLike[str],
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
    preserve_provider_base_urls: bool = False,
    module_names: Iterable[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a secret-safe manifest proving the raw arm configuration."""
    _assert_baseline_command(command)
    _assert_no_entroly_modules(module_names)
    task_path = Path(task_set).expanduser().resolve(strict=True)
    clean_environment, removed = sanitize_baseline_environment(
        environment,
        preserve_provider_base_urls=preserve_provider_base_urls,
    )
    if any(key.startswith("ENTROLY_") for key in clean_environment):
        raise BenchmarkContractError("failed to remove every ENTROLY_* variable")

    credential_names = sorted(
        key for key, value in clean_environment.items() if value and _is_secret_name(key)
    )
    manifest: dict[str, Any] = {
        "schema_version": "entroly.external-baseline-manifest.v1",
        "protocol_version": PROTOCOL_VERSION,
        "arm": "A_full",
        "treatment": "none",
        "status": "planned",
        "benchmark": benchmark,
        "benchmark_version": benchmark_version,
        "run_id": run_id,
        "model_id": model_id,
        "agent_id": agent_id,
        "provider_id": provider_id,
        "harness_commit": harness_commit,
        "task_set_digest": sha256_file(task_path),
        "task_set_name": task_path.name,
        "command": list(command),
        "command_digest": sha256_bytes(_canonical_bytes(list(command))),
        "environment_digest": _environment_digest(clean_environment),
        "credential_environment_names": credential_names,
        "removed_environment_keys": list(removed),
        "provider_base_url_overrides_preserved": preserve_provider_base_urls,
        "entroly_modules_loaded": [],
        "entroly_environment_keys": [],
        "working_directory": ".",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "created_unix_ms": int(time.time() * 1_000),
    }
    manifest["manifest_digest"] = _manifest_digest(manifest)
    return manifest, clean_environment


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
    else:  # pragma: no cover - platform dependent
        process.terminate()
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()


def run_baseline(
    *,
    output_directory: str | os.PathLike[str],
    execute: bool,
    timeout_seconds: float,
    **manifest_arguments: Any,
) -> dict[str, Any]:
    """Write a manifest and optionally execute the raw external harness."""
    manifest, environment = build_baseline_manifest(**manifest_arguments)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "a_full_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    if not execute:
        return manifest

    stdout_path = output / "a_full.stdout"
    stderr_path = output / "a_full.stderr"
    started = time.perf_counter()
    process: subprocess.Popen[bytes] | None = None
    status = "failed"
    error_class: str | None = None
    exit_code: int | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(manifest_arguments["command"]),
                cwd=Path.cwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=(os.name == "posix"),
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
                status = "completed" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
                error_class = "TimeoutExpired"
                _terminate(process)
                exit_code = process.returncode
    except BaseException as exc:
        error_class = type(exc).__name__
        if process is not None:
            _terminate(process)
        raise
    finally:
        finished = dict(manifest)
        finished.update(
            {
                "status": status,
                "exit_code": exit_code,
                "error_class": error_class,
                "wall_time_ms": round((time.perf_counter() - started) * 1_000, 3),
                "stdout_digest": (
                    sha256_file(stdout_path) if stdout_path.exists() else None
                ),
                "stderr_digest": (
                    sha256_file(stderr_path) if stderr_path.exists() else None
                ),
                "stdout_bytes": (
                    stdout_path.stat().st_size if stdout_path.exists() else 0
                ),
                "stderr_bytes": (
                    stderr_path.stat().st_size if stderr_path.exists() else 0
                ),
            }
        )
        finished["manifest_digest"] = _manifest_digest(finished)
        _atomic_write_json(manifest_path, finished)
        manifest = finished
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate paired external results or prepare a raw A_full run."
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    validate = subparsers.add_parser("validate", help="validate task-level JSONL")
    validate.add_argument("results", type=Path)
    validate.add_argument("--require-arm", action="append", default=[])

    baseline = subparsers.add_parser(
        "baseline",
        help="write a no-Entroly manifest and optionally execute a harness",
    )
    baseline.add_argument("--benchmark", required=True)
    baseline.add_argument("--benchmark-version", required=True)
    baseline.add_argument("--run-id", required=True)
    baseline.add_argument("--model-id", required=True)
    baseline.add_argument("--agent-id", required=True)
    baseline.add_argument("--provider-id")
    baseline.add_argument("--harness-commit")
    baseline.add_argument("--task-set", required=True, type=Path)
    baseline.add_argument("--output-directory", required=True, type=Path)
    baseline.add_argument("--timeout-seconds", type=float, default=3_600.0)
    baseline.add_argument("--preserve-provider-base-urls", action="store_true")
    baseline.add_argument("--execute", action="store_true")
    baseline.add_argument("harness_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command_name == "validate":
            summary = validate_jsonl(
                args.results,
                required_arms=args.require_arm,
            )
        else:
            command = list(args.harness_command)
            if command and command[0] == "--":
                command = command[1:]
            summary = run_baseline(
                output_directory=args.output_directory,
                execute=bool(args.execute),
                timeout_seconds=float(args.timeout_seconds),
                benchmark=args.benchmark,
                benchmark_version=args.benchmark_version,
                run_id=args.run_id,
                model_id=args.model_id,
                agent_id=args.agent_id,
                provider_id=args.provider_id,
                harness_commit=args.harness_commit,
                task_set=args.task_set,
                command=command,
                preserve_provider_base_urls=bool(
                    args.preserve_provider_base_urls
                ),
            )
    except (BenchmarkContractError, OSError, RuntimeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"ok": True, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
