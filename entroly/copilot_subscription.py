"""Opt-in preparation for GitHub Copilot CLI subscription proxy mode.

This module does not add another router, provider adapter, or proxy. It only
prepares the existing ``entroly wrap copilot`` path so the current Copilot CLI
custom-provider seam traverses Entroly's hardened OpenAI-compatible proxy.
Subscription authentication is acquired and refreshed inside that proxy by
``copilot_subscription_transport``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

_DEFAULT_API_ORIGIN = "https://api.githubcopilot.com"
_PUBLIC_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_ALLOWED_WIRE_APIS = frozenset({"completions", "responses"})
_LOCAL_PROVIDER_BEARER = "entroly-local-provider-route"


class CopilotSubscriptionError(ValueError):
    """Safe, actionable subscription-route preparation failure."""


@dataclass(frozen=True, slots=True)
class CopilotSubscriptionPlan:
    cleaned_argv: tuple[str, ...]
    upstream_origin: str
    wire_api: str
    model: str
    proxy_port: int

    def public_summary(self) -> dict[str, object]:
        return {
            "client": "github-copilot-cli",
            "mode": "subscription-provider-route",
            "provider_bound": True,
            "wire_provider": "openai",
            "upstream_origin": self.upstream_origin,
            "wire_api": self.wire_api,
            "model": self.model,
            "proxy_port": self.proxy_port,
            "secrets_persisted": False,
            "existing_proxy_reused": False,
        }


def is_subscription_wrap(argv: Sequence[str]) -> bool:
    """Return True only for wrapper-owned ``--subscription`` before ``--``."""
    values = tuple(str(value) for value in argv)
    if len(values) < 3 or values[:2] != ("wrap", "copilot"):
        return False
    boundary = values.index("--") if "--" in values else len(values)
    return "--subscription" in values[2:boundary]


def prepare_subscription_wrap(
    argv: Sequence[str],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> CopilotSubscriptionPlan:
    """Prepare the existing wrapper/proxy lifecycle for Copilot subscription traffic."""
    values = [str(value) for value in argv]
    if not is_subscription_wrap(values):
        raise CopilotSubscriptionError(
            "expected `entroly wrap copilot --subscription`"
        )

    env = os.environ if environ is None else environ
    cleaned, wire_from_cli, origin_from_cli = _strip_wrapper_options(values)
    model = _resolve_model(cleaned, env)
    wire_api = _resolve_wire_api(wire_from_cli, env)
    origin = validate_copilot_api_origin(
        origin_from_cli
        or env.get("ENTROLY_COPILOT_API_URL")
        or env.get("GITHUB_COPILOT_API_URL")
        or _DEFAULT_API_ORIGIN
    )

    cleaned, port = _ensure_dedicated_port(cleaned)
    env["ENTROLY_OPENAI_BASE"] = origin
    env["ENTROLY_COPILOT_SUBSCRIPTION"] = "1"
    env["ENTROLY_CLIENT_ROUTE"] = "github-copilot-subscription"
    env["COPILOT_PROVIDER_WIRE_API"] = wire_api
    env["COPILOT_PROVIDER_BEARER_TOKEN"] = _LOCAL_PROVIDER_BEARER
    env.pop("COPILOT_PROVIDER_API_KEY", None)
    env["COPILOT_MODEL"] = model
    _ensure_loopback_no_proxy(env)

    return CopilotSubscriptionPlan(
        cleaned_argv=tuple(cleaned),
        upstream_origin=origin,
        wire_api=wire_api,
        model=model,
        proxy_port=port,
    )


def start_subscription_proxy(
    plan: CopilotSubscriptionPlan,
    *,
    environ: MutableMapping[str, str] | None = None,
    timeout_s: float = 15.0,
) -> Path:
    """Start the existing hardened proxy on the plan's dedicated loopback port.

    The proxy is started before the existing ``cmd_wrap`` lifecycle runs. That
    lifecycle then observes the healthy Entroly proxy and reuses it instead of
    starting its legacy direct proxy path. No provider credential is synthesized
    or copied into the parent process here.
    """
    env = dict(os.environ if environ is None else environ)
    port = _validate_port(plan.proxy_port)
    if _port_is_occupied(port):
        raise CopilotSubscriptionError(
            f"subscription mode requires a dedicated proxy port; {port} is already in use"
        )

    env["ENTROLY_PROXY_HOST"] = "127.0.0.1"
    env["ENTROLY_PROXY_PORT"] = str(port)
    env["ENTROLY_PROXY_DASHBOARD"] = "1"
    env["ENTROLY_OPENAI_BASE"] = plan.upstream_origin
    env["ENTROLY_COPILOT_SUBSCRIPTION"] = "1"

    runtime = _runtime_dir(env)
    log_path = runtime / f"copilot-subscription-proxy-{port}.log"
    command = [sys.executable, "-m", "entroly.container_proxy"]
    try:
        handle = log_path.open("ab")
    except OSError as exc:
        raise CopilotSubscriptionError(
            "unable to create the local Copilot subscription proxy log"
        ) from exc

    try:
        handle.write(
            (
                "\n--- Entroly Copilot subscription proxy start "
                f"port={port} ---\n"
            ).encode("utf-8")
        )
        handle.flush()
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise CopilotSubscriptionError(
                "unable to start the hardened Entroly proxy process"
            ) from exc
    finally:
        handle.close()

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = _tail_log(log_path)
            raise CopilotSubscriptionError(
                "Copilot subscription proxy exited during startup"
                + (f": {detail}" if detail else "")
            )
        if _healthy_proxy(health_url):
            return log_path
        time.sleep(0.1)

    _stop_process(process)
    detail = _tail_log(log_path)
    raise CopilotSubscriptionError(
        "timed out waiting for the Copilot subscription proxy"
        + (f": {detail}" if detail else "")
    )


def validate_copilot_api_origin(raw: object) -> str:
    """Validate a GitHub-operated Copilot CAPI origin without performing I/O."""
    value = _safe_text(raw, 2048)
    if not value:
        raise CopilotSubscriptionError("Copilot API origin is empty")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise CopilotSubscriptionError("Copilot API origin is malformed") from exc
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port not in {None, 443}
    ):
        raise CopilotSubscriptionError(
            "Copilot API URL must be an HTTPS origin without credentials, "
            "query, fragment, or non-root path"
        )

    host = parsed.hostname.casefold().rstrip(".")
    if _is_public_copilot_host(host) or _ghe_tenant_from_capi_host(host):
        return f"https://{host}"
    raise CopilotSubscriptionError(
        "subscription mode accepts only GitHub-operated Copilot API origins"
    )


def token_exchange_url_for_origin(origin: str) -> str:
    """Derive the GitHub token-exchange endpoint from a validated CAPI origin."""
    normalized = validate_copilot_api_origin(origin)
    host = urlsplit(normalized).hostname or ""
    if _is_public_copilot_host(host):
        return _PUBLIC_TOKEN_EXCHANGE_URL
    tenant = _ghe_tenant_from_capi_host(host)
    if tenant:
        return f"https://api.{tenant}.ghe.com/copilot_internal/v2/token"
    raise CopilotSubscriptionError("unable to derive a trusted Copilot token endpoint")


def _is_public_copilot_host(host: str) -> bool:
    host = host.casefold().rstrip(".")
    return host == "api.githubcopilot.com" or (
        host.startswith("api.") and host.endswith(".githubcopilot.com")
    )


def _ghe_tenant_from_capi_host(host: str) -> str:
    host = host.casefold().rstrip(".")
    prefix = "copilot-api."
    suffix = ".ghe.com"
    if not (host.startswith(prefix) and host.endswith(suffix)):
        return ""
    tenant = host[len(prefix) : -len(suffix)]
    labels = tenant.split(".")
    if not labels:
        return ""
    for label in labels:
        if (
            not label
            or len(label) > 63
            or label[0] == "-"
            or label[-1] == "-"
            or any(not (char.isascii() and (char.isalnum() or char == "-")) for char in label)
        ):
            return ""
    return tenant


def _strip_wrapper_options(
    argv: list[str],
) -> tuple[list[str], str | None, str | None]:
    boundary = argv.index("--") if "--" in argv else len(argv)
    prefix, suffix = argv[:boundary], argv[boundary:]
    out: list[str] = []
    wire_api: str | None = None
    origin: str | None = None

    index = 0
    while index < len(prefix):
        value = prefix[index]
        if value == "--subscription":
            index += 1
            continue
        if value == "--wire-api":
            if index + 1 >= len(prefix):
                raise CopilotSubscriptionError("--wire-api requires a value")
            wire_api = prefix[index + 1]
            index += 2
            continue
        if value.startswith("--wire-api="):
            wire_api = value.split("=", 1)[1]
            index += 1
            continue
        if value == "--copilot-api-url":
            if index + 1 >= len(prefix):
                raise CopilotSubscriptionError(
                    "--copilot-api-url requires an HTTPS origin"
                )
            origin = prefix[index + 1]
            index += 2
            continue
        if value.startswith("--copilot-api-url="):
            origin = value.split("=", 1)[1]
            index += 1
            continue
        out.append(value)
        index += 1
    return out + suffix, wire_api, origin


def _resolve_model(
    argv: Sequence[str],
    environ: MutableMapping[str, str],
) -> str:
    configured = _safe_text(environ.get("COPILOT_MODEL"), 256)
    if configured:
        return configured
    values = tuple(argv)
    boundary = values.index("--") + 1 if "--" in values else 0
    for index in range(boundary, len(values)):
        value = values[index]
        if value == "--model" and index + 1 < len(values):
            model = _safe_text(values[index + 1], 256)
            if model:
                return model
        if value.startswith("--model="):
            model = _safe_text(value.split("=", 1)[1], 256)
            if model:
                return model
    raise CopilotSubscriptionError(
        "subscription mode needs an explicit Copilot model; set COPILOT_MODEL "
        "or pass `-- --model MODEL`"
    )


def _resolve_wire_api(
    cli_value: str | None,
    environ: MutableMapping[str, str],
) -> str:
    raw = cli_value or environ.get("COPILOT_PROVIDER_WIRE_API") or "completions"
    value = _safe_text(raw, 32).casefold()
    if value not in _ALLOWED_WIRE_APIS:
        raise CopilotSubscriptionError(
            "Copilot wire API must be `completions` or `responses`"
        )
    return value


def _ensure_dedicated_port(argv: list[str]) -> tuple[list[str], int]:
    boundary = argv.index("--") if "--" in argv else len(argv)
    prefix, suffix = argv[:boundary], argv[boundary:]
    for index, value in enumerate(prefix):
        if value == "--port" and index + 1 < len(prefix):
            return argv, _validate_port(prefix[index + 1])
        if value.startswith("--port="):
            return argv, _validate_port(value.split("=", 1)[1])

    port = _reserve_loopback_port()
    return prefix + ["--port", str(port)] + suffix, port


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _validate_port(value: object) -> int:
    if isinstance(value, bool):
        raise CopilotSubscriptionError("proxy port must be 1..65535")
    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CopilotSubscriptionError("proxy port must be 1..65535") from exc
    if not 1 <= port <= 65535:
        raise CopilotSubscriptionError("proxy port must be 1..65535")
    return port


def _ensure_loopback_no_proxy(environ: MutableMapping[str, str]) -> None:
    required = ("localhost", "127.0.0.1", "::1")
    for name in ("NO_PROXY", "no_proxy"):
        current = [
            item.strip()
            for item in str(environ.get(name, "")).split(",")
            if item.strip()
        ]
        seen = {item.casefold() for item in current}
        current.extend(item for item in required if item.casefold() not in seen)
        environ[name] = ",".join(current)


def _runtime_dir(environ: MutableMapping[str, str] | dict[str, str]) -> Path:
    explicit = environ.get("ENTROLY_DIR")
    candidates = [Path(explicit).expanduser()] if explicit else [
        Path.home() / ".entroly",
        Path(tempfile.gettempdir()) / "entroly",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".copilot-write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / "entroly"


def _port_is_occupied(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.2)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def _healthy_proxy(url: str) -> bool:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=0.4) as response:
            payload = json.loads(response.read())
        return (
            response.status == 200
            and payload.get("status") == "ok"
            and payload.get("service") == "entroly-proxy"
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return False


def _tail_log(path: Path, limit: int = 1800) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    text = data[-max(1, limit) :].decode("utf-8", errors="replace")
    return " ".join(text.split())[-limit:]


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _safe_text(value: object, limit: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > limit or any(ord(char) < 32 or ord(char) == 127 for char in text):
        return ""
    return text


__all__ = [
    "CopilotSubscriptionError",
    "CopilotSubscriptionPlan",
    "is_subscription_wrap",
    "prepare_subscription_wrap",
    "start_subscription_proxy",
    "token_exchange_url_for_origin",
    "validate_copilot_api_origin",
]
