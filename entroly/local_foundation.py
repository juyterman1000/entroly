"""Strict-local foundational model support for Entroly.

This module intentionally has no cloud-provider integration. It validates a
local GGUF model, builds a llama.cpp server command that uses only a local file,
and emits a sanitized environment for child processes.

The one-time model download is handled by ``scripts/setup_qwen_strict_local.ps1``.
At runtime, no Hugging Face, OpenRouter, Alibaba, OpenAI, Anthropic, or Gemini
endpoint is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


DEFAULT_MODEL_ALIAS = "qwen3-30b-a3b-thinking-2507-local"
DEFAULT_PORT = 9378
DEFAULT_CONTEXT_SIZE = 12_288
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
STRICT_LOCAL_MODE = "strict-local"

_CLOUD_PROVIDER_ENV_VARS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GEMINI_BASE_URL",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "HF_INFERENCE_ENDPOINT",
    "HF_ENDPOINT",
    "HUGGING_FACE_HUB_TOKEN",
    "ENTROLY_ANTHROPIC_BASE",
    "ENTROLY_GEMINI_BASE",
    "ENTROLY_CLOUD_FALLBACK",
}

_ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LocalFoundationError(RuntimeError):
    """Raised when strict-local foundational model policy is violated."""


def entroly_dir() -> Path:
    """Return the local Entroly state directory without creating it."""

    explicit = os.environ.get("ENTROLY_DIR")
    return Path(explicit).expanduser() if explicit else Path.home() / ".entroly"


def default_config_path() -> Path:
    return entroly_dir() / "local_foundation.json"


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_loopback_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise LocalFoundationError(
            f"strict-local endpoint must use http on loopback, got {url!r}"
        )
    if parsed.hostname not in _ALLOWED_LOOPBACK_HOSTS:
        raise LocalFoundationError(
            f"strict-local mode rejected non-loopback endpoint: {url}"
        )
    if parsed.username or parsed.password:
        raise LocalFoundationError("credentials are not allowed in a local endpoint URL")
    if parsed.query or parsed.fragment:
        raise LocalFoundationError("query strings and fragments are not allowed in endpoint URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise LocalFoundationError(f"invalid local endpoint port in {url!r}") from exc
    if port is None:
        raise LocalFoundationError("strict-local endpoint must include an explicit port")


@dataclass(frozen=True)
class LocalFoundationConfig:
    """Configuration for a local GGUF model served by llama.cpp."""

    schema_version: int
    mode: str
    provider: str
    model_alias: str
    model_path: str
    model_sha256: str
    server_executable: str
    base_url: str
    host: str
    port: int
    context_size: int
    max_output_tokens: int
    gpu_layers: int
    threads: int
    cloud_fallback: bool
    allow_non_loopback_endpoint: bool
    remote_embeddings: bool
    remote_reranking: bool
    telemetry: bool
    runtime_model_downloads: bool

    @classmethod
    def create(
        cls,
        *,
        model_path: str | Path,
        server_executable: str | Path,
        port: int = DEFAULT_PORT,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        gpu_layers: int = 0,
        threads: int = 0,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        model_sha256: str = "",
    ) -> "LocalFoundationConfig":
        model = _normalize_path(model_path)
        server = _resolve_server_executable(server_executable)
        digest = model_sha256.strip().lower() or sha256_file(model)
        return cls(
            schema_version=1,
            mode=STRICT_LOCAL_MODE,
            provider="llama.cpp",
            model_alias=model_alias,
            model_path=str(model),
            model_sha256=digest,
            server_executable=str(server),
            base_url=f"http://127.0.0.1:{port}/v1",
            host="127.0.0.1",
            port=port,
            context_size=context_size,
            max_output_tokens=max_output_tokens,
            gpu_layers=gpu_layers,
            threads=threads,
            cloud_fallback=False,
            allow_non_loopback_endpoint=False,
            remote_embeddings=False,
            remote_reranking=False,
            telemetry=False,
            runtime_model_downloads=False,
        )

    def validate(self, *, verify_hash: bool = False) -> None:
        errors: list[str] = []

        if self.schema_version != 1:
            errors.append(f"unsupported schema_version={self.schema_version}")
        if self.mode != STRICT_LOCAL_MODE:
            errors.append(f"mode must be {STRICT_LOCAL_MODE!r}")
        if self.provider != "llama.cpp":
            errors.append("provider must be 'llama.cpp'")
        if self.host != "127.0.0.1":
            errors.append("host must be the literal loopback address 127.0.0.1")
        try:
            _require_loopback_url(self.base_url)
        except LocalFoundationError as exc:
            errors.append(str(exc))

        parsed = urlparse(self.base_url)
        try:
            parsed_port = parsed.port
        except ValueError:
            parsed_port = None
        if parsed_port != self.port:
            errors.append("base_url port does not match configured port")

        if not (1024 <= self.port <= 65535):
            errors.append("port must be between 1024 and 65535")
        if not (2_048 <= self.context_size <= 65_536):
            errors.append("context_size must be between 2048 and 65536")
        if not (256 <= self.max_output_tokens <= 32_768):
            errors.append("max_output_tokens must be between 256 and 32768")
        if self.gpu_layers < 0:
            errors.append("gpu_layers must be zero or greater")
        if self.threads < 0:
            errors.append("threads must be zero or greater")

        forbidden_true = {
            "cloud_fallback": self.cloud_fallback,
            "allow_non_loopback_endpoint": self.allow_non_loopback_endpoint,
            "remote_embeddings": self.remote_embeddings,
            "remote_reranking": self.remote_reranking,
            "telemetry": self.telemetry,
            "runtime_model_downloads": self.runtime_model_downloads,
        }
        errors.extend(
            f"{name} must be false in strict-local mode"
            for name, enabled in forbidden_true.items()
            if enabled
        )

        model = Path(self.model_path)
        if not model.is_file():
            errors.append(f"local GGUF model not found: {model}")
        elif model.suffix.lower() != ".gguf":
            errors.append(f"model must be a local .gguf file: {model}")

        server = Path(self.server_executable)
        if not server.is_file():
            errors.append(f"llama.cpp server executable not found: {server}")

        digest = self.model_sha256.strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            errors.append("model_sha256 must be a 64-character lowercase hexadecimal digest")
        elif verify_hash and model.is_file() and digest:
            actual = sha256_file(model)
            if actual != digest:
                errors.append(
                    f"model SHA256 mismatch: expected {digest}, observed {actual}"
                )

        if errors:
            raise LocalFoundationError("; ".join(errors))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: dict) -> "LocalFoundationConfig":
        if not isinstance(payload, dict):
            raise LocalFoundationError("local foundation config must be a JSON object")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise LocalFoundationError(f"invalid local foundation config: {exc}") from exc


def _resolve_server_executable(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    located = shutil.which(str(value))
    if located:
        return Path(located).resolve()

    raise LocalFoundationError(f"llama.cpp server executable not found: {value}")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_config(
    config: LocalFoundationConfig,
    path: str | Path | None = None,
    *,
    verify_hash: bool = False,
) -> Path:
    config.validate(verify_hash=verify_hash)
    target = _normalize_path(path or default_config_path())
    target.parent.mkdir(parents=True, exist_ok=True)

    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(config.to_json(), encoding="utf-8")
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    temp.replace(target)
    return target


def load_config(
    path: str | Path | None = None,
    *,
    verify_hash: bool = False,
) -> LocalFoundationConfig:
    source = _normalize_path(path or default_config_path())
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LocalFoundationError(f"local foundation config not found: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalFoundationError(f"unable to read local foundation config: {exc}") from exc

    config = LocalFoundationConfig.from_dict(payload)
    config.validate(verify_hash=verify_hash)
    return config


def build_llama_server_command(config: LocalFoundationConfig) -> list[str]:
    """Build a command that can only load an already-downloaded local model."""

    config.validate()
    command = [
        config.server_executable,
        "-m",
        config.model_path,
        "--alias",
        config.model_alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "-c",
        str(config.context_size),
        "-np",
        "1",
    ]
    if config.gpu_layers:
        command.extend(["--n-gpu-layers", str(config.gpu_layers)])
    if config.threads:
        command.extend(["--threads", str(config.threads)])
    return command


def strict_local_environment(
    config: LocalFoundationConfig,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that disables cloud-provider and runtime downloads."""

    config.validate()
    env = dict(os.environ if base is None else base)
    for key in _CLOUD_PROVIDER_ENV_VARS:
        env.pop(key, None)

    env.update(
        {
            "ENTROLY_LOCAL_ONLY": "1",
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
            "ENTROLY_OPENAI_BASE": config.base_url.removesuffix("/v1"),
            "ENTROLY_CLOUD_FALLBACK": "0",
            "ENTROLY_REMOTE_EMBEDDINGS": "0",
            "ENTROLY_REMOTE_RERANKING": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
        }
    )
    return env


def health_url(config: LocalFoundationConfig) -> str:
    return f"http://127.0.0.1:{config.port}/health"


def check_health(config: LocalFoundationConfig, *, timeout: float = 2.0) -> tuple[bool, str]:
    config.validate()
    request = urllib.request.Request(health_url(config), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if response.status == 200 and payload.get("status") == "ok":
            return True, "local llama.cpp server is ready"
        return False, f"unexpected health response: HTTP {response.status} {payload!r}"
    except urllib.error.URLError as exc:
        return False, f"local llama.cpp server is not reachable: {exc}"
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid local llama.cpp health response: {exc}"


def doctor(
    config: LocalFoundationConfig,
    *,
    verify_hash: bool = False,
    check_server: bool = True,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []

    try:
        config.validate(verify_hash=verify_hash)
        checks.append((True, "strict-local configuration is valid"))
    except LocalFoundationError as exc:
        return [(False, str(exc))]

    command = build_llama_server_command(config)
    checks.append((True, "runtime command uses a local -m GGUF path"))
    checks.append((all(arg != "-hf" for arg in command), "runtime command contains no Hugging Face downloader"))
    checks.append((_require_no_remote_url(command), "runtime command contains no remote URL"))
    checks.append((not config.cloud_fallback, "cloud fallback is disabled"))
    checks.append((not config.remote_embeddings, "remote embeddings are disabled"))
    checks.append((not config.remote_reranking, "remote reranking is disabled"))
    checks.append((not config.telemetry, "telemetry is disabled"))

    if check_server:
        checks.append(check_health(config))
    return checks


def _require_no_remote_url(values: Iterable[str]) -> bool:
    for value in values:
        if value.startswith(("http://", "https://")):
            parsed = urlparse(value)
            if parsed.hostname not in _ALLOWED_LOOPBACK_HOSTS:
                return False
    return True


def serve(config: LocalFoundationConfig, *, dry_run: bool = False) -> int:
    command = build_llama_server_command(config)
    if dry_run:
        print(json.dumps(command))
        return 0
    completed = subprocess.run(
        command,
        env=strict_local_environment(config),
        check=False,
    )
    return int(completed.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and run Entroly's strict-local Qwen foundation model."
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to local_foundation.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser("configure", help="Write strict-local config")
    configure.add_argument("--model-path", required=True)
    configure.add_argument("--server-executable", default="llama-server")
    configure.add_argument("--model-sha256", default="")
    configure.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    configure.add_argument("--port", type=int, default=DEFAULT_PORT)
    configure.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    configure.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    configure.add_argument("--gpu-layers", type=int, default=0)
    configure.add_argument("--threads", type=int, default=0)
    configure.add_argument("--verify-hash", action="store_true")

    diagnose = subparsers.add_parser("doctor", help="Audit strict-local setup")
    diagnose.add_argument("--verify-hash", action="store_true")
    diagnose.add_argument("--no-server-check", action="store_true")

    run = subparsers.add_parser("serve", help="Start local llama.cpp server")
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)

    try:
        if args.command == "configure":
            config = LocalFoundationConfig.create(
                model_path=args.model_path,
                server_executable=args.server_executable,
                port=args.port,
                context_size=args.context_size,
                max_output_tokens=args.max_output_tokens,
                gpu_layers=args.gpu_layers,
                threads=args.threads,
                model_alias=args.model_alias,
                model_sha256=args.model_sha256,
            )
            saved = save_config(
                config,
                config_path,
                verify_hash=args.verify_hash,
            )
            print(saved)
            return 0

        config = load_config(
            config_path,
            verify_hash=getattr(args, "verify_hash", False),
        )
        if args.command == "doctor":
            checks = doctor(
                config,
                verify_hash=args.verify_hash,
                check_server=not args.no_server_check,
            )
            for ok, message in checks:
                print(f"{'PASS' if ok else 'FAIL'} {message}")
            return 0 if all(ok for ok, _ in checks) else 2

        if args.command == "serve":
            return serve(config, dry_run=args.dry_run)

    except LocalFoundationError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
