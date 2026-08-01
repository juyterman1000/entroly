"""Neutral, command-driven competitor gauntlet with immutable run identities.

Every system receives the same canonical JSONL input and experiment contract.
The harness records executable/artifact hashes, version output, environment and
output fingerprints, latency, commands, and raw artifacts. It never silently
resolves ``latest``, normalizes away failures, or treats incomparable runs as
competitive evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_MAX_ARTIFACT_FILES = 20_000
_IDENTITY_SCHEMA = "entroly-neutral-gauntlet-identity-v1"


@dataclass(frozen=True)
class RunnerSpec:
    name: str
    version_command: tuple[str, ...]
    run_command: tuple[str, ...]
    timeout_seconds: float = 120.0
    expected_version_pattern: str | None = None
    expected_executable_sha256: str | None = None
    artifact_paths: tuple[str, ...] = ()
    expected_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("runner name is required")
        if not self.version_command or not self.run_command:
            raise ValueError("version_command and run_command are required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for field_name, value in (
            ("expected_executable_sha256", self.expected_executable_sha256),
            ("expected_artifact_sha256", self.expected_artifact_sha256),
        ):
            if value is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if self.expected_version_pattern is not None:
            re.compile(self.expected_version_pattern)


@dataclass(frozen=True)
class RunnerIdentity:
    version: str
    executable_path: str
    executable_sha256: str
    artifact_sha256: str
    expected_version_pattern: str | None
    expected_executable_sha256: str | None
    expected_artifact_sha256: str | None
    version_matches: bool
    executable_matches: bool
    artifact_matches: bool
    pinned: bool

    @property
    def verified(self) -> bool:
        return self.version_matches and self.executable_matches and self.artifact_matches

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verified"] = self.verified
        return payload


@dataclass(frozen=True)
class RunnerResult:
    name: str
    identity: RunnerIdentity
    returncode: int | None
    latency_ms: float
    stdout: str
    stderr: str
    timed_out: bool
    command: tuple[str, ...]
    version_command: tuple[str, ...]
    input_sha256: str
    output_sha256: str
    experiment_sha256: str
    environment_sha256: str
    platform_sha256: str
    working_directory_policy: str = "isolated-temporary-directory"

    @property
    def version(self) -> str:
        """Compatibility accessor for older result consumers."""
        return self.identity.version

    @property
    def claim_ready(self) -> bool:
        return (
            self.identity.pinned
            and self.identity.verified
            and not self.timed_out
            and self.returncode == 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity"] = self.identity.to_dict()
        payload["command"] = list(self.command)
        payload["version_command"] = list(self.version_command)
        payload["claim_ready"] = self.claim_ready
        return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "\n".join(canonical_json(dict(row)) for row in rows) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest(paths: Sequence[str]) -> str:
    """Hash explicit files/directories with path names and file contents."""
    digest = hashlib.sha256()
    digest.update((_IDENTITY_SCHEMA + "\0").encode())
    file_count = 0
    for raw in sorted(paths):
        path = Path(raw).expanduser().resolve(strict=True)
        candidates = (
            [path]
            if path.is_file()
            else sorted(item for item in path.rglob("*") if item.is_file())
        )
        for candidate in candidates:
            file_count += 1
            if file_count > _MAX_ARTIFACT_FILES:
                raise ValueError(
                    f"artifact identity exceeds {_MAX_ARTIFACT_FILES:,} files"
                )
            relative = (
                candidate.name
                if path.is_file()
                else candidate.relative_to(path).as_posix()
            )
            digest.update(str(path).encode("utf-8", "surrogatepass"))
            digest.update(b"\0")
            digest.update(relative.encode("utf-8", "surrogatepass"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _safe_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "HOME",
            "USERPROFILE",
            "SYSTEMROOT",
            "WINDIR",
            "TMP",
            "TEMP",
        }
    }
    if extra:
        allowed.update({str(key): str(value) for key, value in extra.items()})
    return allowed


def _environment_digest(env: Mapping[str, str]) -> str:
    return _sha256_bytes(
        canonical_json(dict(sorted(env.items()))).encode("utf-8")
    )


def _platform_digest() -> str:
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "implementation": platform.python_implementation(),
    }
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _resolved_executable(
    command: Sequence[str], env: Mapping[str, str]
) -> Path | None:
    executable = command[0]
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        try:
            return candidate.resolve(strict=True)
        except OSError:
            return None
    resolved = shutil.which(executable, path=env.get("PATH"))
    return Path(resolved).resolve(strict=True) if resolved else None


def _version(spec: RunnerSpec, env: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(
            spec.version_command,
            text=True,
            capture_output=True,
            timeout=min(spec.timeout_seconds, 30.0),
            check=False,
            env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return (result.stdout or result.stderr).strip()[:1000]


def resolve_identity(
    spec: RunnerSpec, env: Mapping[str, str]
) -> RunnerIdentity:
    version = _version(spec, env)
    executable = _resolved_executable(spec.run_command, env)
    executable_path = str(executable) if executable else ""
    executable_sha = (
        _sha256_file(executable)
        if executable and executable.is_file()
        else ""
    )
    try:
        artifact_sha = (
            _artifact_digest(spec.artifact_paths) if spec.artifact_paths else ""
        )
    except (OSError, ValueError) as exc:
        artifact_sha = f"unavailable:{type(exc).__name__}:{exc}"

    version_matches = (
        True
        if spec.expected_version_pattern is None
        else re.search(spec.expected_version_pattern, version) is not None
    )
    executable_matches = (
        True
        if spec.expected_executable_sha256 is None
        else executable_sha.lower() == spec.expected_executable_sha256.lower()
    )
    artifact_matches = (
        True
        if spec.expected_artifact_sha256 is None
        else artifact_sha.lower() == spec.expected_artifact_sha256.lower()
    )
    pinned = any(
        value is not None
        for value in (
            spec.expected_version_pattern,
            spec.expected_executable_sha256,
            spec.expected_artifact_sha256,
        )
    )
    return RunnerIdentity(
        version=version,
        executable_path=executable_path,
        executable_sha256=executable_sha,
        artifact_sha256=artifact_sha,
        expected_version_pattern=spec.expected_version_pattern,
        expected_executable_sha256=spec.expected_executable_sha256,
        expected_artifact_sha256=spec.expected_artifact_sha256,
        version_matches=version_matches,
        executable_matches=executable_matches,
        artifact_matches=artifact_matches,
        pinned=pinned,
    )


def run_one(
    spec: RunnerSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_contract: Mapping[str, Any] | str | None = None,
    extra_env: Mapping[str, str] | None = None,
    require_pinned_identity: bool = False,
) -> RunnerResult:
    body = canonical_jsonl(rows)
    input_digest = _sha256_bytes(body.encode("utf-8"))
    contract = experiment_contract if experiment_contract is not None else {}
    contract_body = (
        contract if isinstance(contract, str) else canonical_json(contract)
    )
    experiment_digest = _sha256_bytes(contract_body.encode("utf-8"))
    env = _safe_environment(extra_env)
    env_digest = _environment_digest(env)
    identity = resolve_identity(spec, env)
    platform_digest = _platform_digest()

    identity_failure = not identity.verified or (
        require_pinned_identity and not identity.pinned
    )
    if identity_failure:
        reasons = []
        if require_pinned_identity and not identity.pinned:
            reasons.append("runner identity is not pinned")
        if not identity.version_matches:
            reasons.append("version output does not match expected pattern")
        if not identity.executable_matches:
            reasons.append("executable SHA-256 does not match")
        if not identity.artifact_matches:
            reasons.append("artifact SHA-256 does not match")
        error = "; ".join(reasons) or "runner identity verification failed"
        return RunnerResult(
            spec.name,
            identity,
            None,
            0.0,
            "",
            error,
            False,
            spec.run_command,
            spec.version_command,
            input_digest,
            _sha256_bytes(b""),
            experiment_digest,
            env_digest,
            platform_digest,
        )

    with tempfile.TemporaryDirectory(prefix="entroly-gauntlet-") as directory:
        input_path = Path(directory) / "input.jsonl"
        output_path = Path(directory) / "output.jsonl"
        input_path.write_text(body, encoding="utf-8")
        command = tuple(
            arg.replace("{input}", str(input_path)).replace(
                "{output}", str(output_path)
            )
            for arg in spec.run_command
        )
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=spec.timeout_seconds,
                check=False,
                cwd=directory,
                env=env,
            )
            latency = (time.perf_counter() - started) * 1000
            stdout = result.stdout
            if output_path.exists():
                stdout = output_path.read_text(encoding="utf-8")
            return RunnerResult(
                spec.name,
                identity,
                result.returncode,
                latency,
                stdout,
                result.stderr,
                False,
                command,
                spec.version_command,
                input_digest,
                _sha256_bytes(stdout.encode("utf-8")),
                experiment_digest,
                env_digest,
                platform_digest,
            )
        except subprocess.TimeoutExpired as exc:
            latency = (time.perf_counter() - started) * 1000
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            return RunnerResult(
                spec.name,
                identity,
                None,
                latency,
                stdout,
                stderr,
                True,
                command,
                spec.version_command,
                input_digest,
                _sha256_bytes(stdout.encode("utf-8")),
                experiment_digest,
                env_digest,
                platform_digest,
            )
        except OSError as exc:
            latency = (time.perf_counter() - started) * 1000
            return RunnerResult(
                spec.name,
                identity,
                None,
                latency,
                "",
                str(exc),
                False,
                command,
                spec.version_command,
                input_digest,
                _sha256_bytes(b""),
                experiment_digest,
                env_digest,
                platform_digest,
            )


def run_gauntlet(
    specs: Sequence[RunnerSpec],
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_contract: Mapping[str, Any] | str | None = None,
    extra_env: Mapping[str, str] | None = None,
    require_pinned_identity: bool = False,
) -> list[RunnerResult]:
    return [
        run_one(
            spec,
            rows,
            experiment_contract=experiment_contract,
            extra_env=extra_env,
            require_pinned_identity=require_pinned_identity,
        )
        for spec in specs
    ]


def assert_comparable(
    results: Sequence[RunnerResult],
    *,
    require_claim_ready: bool = True,
    require_same_environment: bool = True,
    require_same_platform: bool = True,
) -> None:
    """Refuse evidence aggregation when the experiment contract differs."""
    if not results:
        raise ValueError("at least one runner result is required")
    checks = {
        "input SHA-256": {result.input_sha256 for result in results},
        "experiment SHA-256": {
            result.experiment_sha256 for result in results
        },
    }
    if require_same_environment:
        checks["environment SHA-256"] = {
            result.environment_sha256 for result in results
        }
    if require_same_platform:
        checks["platform SHA-256"] = {
            result.platform_sha256 for result in results
        }
    mismatched = [
        name for name, values in checks.items() if len(values) != 1
    ]
    if mismatched:
        raise ValueError(
            "incomparable runs: mismatched " + ", ".join(mismatched)
        )
    if require_claim_ready:
        failures = [result.name for result in results if not result.claim_ready]
        if failures:
            raise ValueError(
                "incomparable runs: runners are not claim-ready: "
                + ", ".join(failures)
            )


def pareto_dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> bool:
    no_worse = all(left[key] >= right[key] for key in maximize) and all(
        left[key] <= right[key] for key in minimize
    )
    strictly_better = any(
        left[key] > right[key] for key in maximize
    ) or any(left[key] < right[key] for key in minimize)
    return no_worse and strictly_better
