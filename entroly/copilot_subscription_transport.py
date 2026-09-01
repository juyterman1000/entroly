"""GitHub Copilot subscription authentication for Entroly's existing proxy.

This module is deliberately narrow. It does not route requests, transform model
payloads, retry provider calls, or create a second receipt/accounting layer. In
explicit Copilot subscription mode it supplies GitHub's short-lived Copilot API
credential to the already-hardened OpenAI-compatible Entroly transport.

Security properties:
* GitHub credentials are read only from documented environment variables or
  ``gh auth token`` and are never persisted by Entroly;
* token exchange is permitted only to a GitHub-operated endpoint derived from a
  validated Copilot API origin;
* redirects are rejected and exchange responses are bounded before JSON parsing;
* the Copilot API origin advertised by the first successful token exchange is
  pinned for the lifetime of the proxy process;
* the local dummy provider bearer supplied to Copilot CLI is always replaced
  before an upstream request leaves Entroly;
* refresh happens in a daemon thread so the request event loop normally performs
  no authentication I/O.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .copilot_subscription import (
    CopilotSubscriptionError,
    token_exchange_url_for_origin,
    validate_copilot_api_origin,
)

_MAX_TOKEN_CHARS = 16_384
_MAX_EXCHANGE_RESPONSE_BYTES = 256 * 1024
_DEFAULT_REFRESH_SKEW = 60.0
_DEFAULT_INTEGRATION_ID = "copilot-cli-chat"
_GITHUB_TOKEN_ENV_VARS = (
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)


class CopilotSubscriptionAuthError(RuntimeError):
    """Authentication failure whose message is safe to show to a user."""


@dataclass(frozen=True, slots=True)
class CopilotAPIToken:
    token: str
    api_origin: str
    expires_at: float
    refresh_at: float


class CopilotTokenManager:
    """In-memory, process-local Copilot API token manager."""

    def __init__(
        self,
        *,
        api_origin: str,
        environ: Mapping[str, str] | None = None,
        integration_id: str | None = None,
        clock: Callable[[], float] = time.time,
        exchange: Callable[[str, str, str], Mapping[str, Any]] | None = None,
        credential_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._requested_origin = validate_copilot_api_origin(api_origin)
        self._exchange_url = token_exchange_url_for_origin(self._requested_origin)
        self._integration_id = _validated_integration_id(
            integration_id
            or self._environ.get("ENTROLY_COPILOT_INTEGRATION_ID")
            or _DEFAULT_INTEGRATION_ID
        )
        self._clock = clock
        self._exchange = exchange or _exchange_token_payload
        self._credential_resolver = credential_resolver or (
            lambda: _resolve_github_credential(self._environ)
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        self._current: CopilotAPIToken | None = None
        self._pinned_origin: str | None = None
        self._last_refresh_error = ""

    @property
    def integration_id(self) -> str:
        return self._integration_id

    @property
    def api_origin(self) -> str:
        with self._lock:
            return self._pinned_origin or self._requested_origin

    def prime(self) -> CopilotAPIToken:
        """Acquire the first API token before accepting provider traffic."""
        token = self._refresh(force=True)
        self._start_background_refresh()
        return token

    def current_token(self) -> str:
        """Return a valid cached token, synchronously recovering only if necessary."""
        now = self._clock()
        with self._lock:
            current = self._current
            if current is not None and now < current.expires_at - 5.0:
                return current.token
        # Background refresh should make this path exceptional. If it did not,
        # refresh under the manager lock rather than forwarding an expired token.
        return self._refresh(force=True).token

    def stop(self) -> None:
        self._stop.set()

    def public_summary(self) -> dict[str, object]:
        with self._lock:
            current = self._current
            return {
                "mode": "github-copilot-subscription",
                "api_origin": self._pinned_origin or self._requested_origin,
                "integration_id": self._integration_id,
                "token_cached": current is not None,
                "token_persisted": False,
                "credential_persisted": False,
                "refresh_background": self._refresh_thread is not None,
                "last_refresh_error": self._last_refresh_error[:160],
            }

    def _refresh(self, *, force: bool) -> CopilotAPIToken:
        with self._lock:
            now = self._clock()
            current = self._current
            if (
                not force
                and current is not None
                and now < current.refresh_at
                and now < current.expires_at - 5.0
            ):
                return current

            try:
                credential = self._credential_resolver()
                payload = self._exchange(
                    self._exchange_url,
                    credential,
                    self._integration_id,
                )
                token = _token_from_exchange_payload(
                    payload,
                    requested_origin=self._requested_origin,
                    now=now,
                )
                if self._pinned_origin is None:
                    self._pinned_origin = token.api_origin
                elif token.api_origin != self._pinned_origin:
                    raise CopilotSubscriptionAuthError(
                        "GitHub changed the Copilot API origin during this session; "
                        "restart Entroly to re-establish the trust boundary"
                    )
                self._current = token
                self._last_refresh_error = ""
                return token
            except CopilotSubscriptionAuthError as exc:
                self._last_refresh_error = str(exc)
                raise
            except Exception as exc:
                self._last_refresh_error = type(exc).__name__
                raise CopilotSubscriptionAuthError(
                    "unable to refresh the GitHub Copilot subscription credential"
                ) from exc

    def _start_background_refresh(self) -> None:
        with self._lock:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._refresh_loop,
                name="entroly-copilot-token-refresh",
                daemon=True,
            )
            self._refresh_thread = thread
            thread.start()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                current = self._current
                now = self._clock()
                delay = 5.0 if current is None else max(1.0, current.refresh_at - now)
            if self._stop.wait(min(delay, 300.0)):
                return
            with self._lock:
                current = self._current
                if current is not None and self._clock() < current.refresh_at:
                    continue
            try:
                self._refresh(force=True)
            except CopilotSubscriptionAuthError:
                # Keep a still-valid token and retry soon. The request path will
                # fail closed once no valid token remains.
                self._stop.wait(15.0)


def _resolve_github_credential(environ: Mapping[str, str]) -> str:
    for name in _GITHUB_TOKEN_ENV_VARS:
        token = _validated_secret(environ.get(name))
        if token:
            return token

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
            env=dict(environ),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise CopilotSubscriptionAuthError(
            "no GitHub credential is available; authenticate `gh` or set "
            "COPILOT_GITHUB_TOKEN"
        ) from exc

    token = _validated_secret(result.stdout if result.returncode == 0 else "")
    if not token:
        raise CopilotSubscriptionAuthError(
            "no GitHub credential is available; authenticate `gh` or set "
            "COPILOT_GITHUB_TOKEN"
        )
    return token


def _exchange_token_payload(
    exchange_url: str,
    github_credential: str,
    integration_id: str,
) -> Mapping[str, Any]:
    """Exchange a GitHub credential for a bounded short-lived Copilot API token."""
    # Re-derive the trusted URL from the configured CAPI origin elsewhere; this
    # function still rejects anything outside the two allowed GitHub API forms.
    _validate_exchange_url(exchange_url)
    credential = _validated_secret(github_credential)
    if not credential:
        raise CopilotSubscriptionAuthError("GitHub credential is empty or malformed")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {credential}",
        "User-Agent": "GithubCopilot/1.0",
        "Editor-Version": "vscode/1.0",
        "Editor-Plugin-Version": "copilot/1.0",
        "Copilot-Integration-Id": _validated_integration_id(integration_id),
    }

    # Reuse Entroly's existing CA/proxy-trust policy without importing another
    # transport stack. Import lazily so this module remains unit-testable alone.
    try:
        from .proxy_transport_final import _safe_http_client_kwargs

        kwargs = dict(_safe_http_client_kwargs())
    except Exception:
        kwargs = {"follow_redirects": False, "trust_env": False}
    kwargs["timeout"] = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    kwargs["follow_redirects"] = False

    try:
        with httpx.Client(**kwargs) as client:
            with client.stream("GET", exchange_url, headers=headers) as response:
                if 300 <= response.status_code < 400:
                    raise CopilotSubscriptionAuthError(
                        "GitHub Copilot token exchange redirected; refusing to forward credentials"
                    )
                if response.status_code in {401, 403}:
                    raise CopilotSubscriptionAuthError(
                        "GitHub rejected the credential for Copilot subscription access"
                    )
                if response.status_code >= 400:
                    raise CopilotSubscriptionAuthError(
                        f"GitHub Copilot token exchange failed with HTTP {response.status_code}"
                    )
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if len(raw) + len(chunk) > _MAX_EXCHANGE_RESPONSE_BYTES:
                        raise CopilotSubscriptionAuthError(
                            "GitHub Copilot token response exceeded the safety limit"
                        )
                    raw.extend(chunk)
    except CopilotSubscriptionAuthError:
        raise
    except (httpx.TimeoutException, httpx.TransportError, OSError) as exc:
        raise CopilotSubscriptionAuthError(
            "unable to reach GitHub's Copilot token exchange endpoint"
        ) from exc

    try:
        payload = json.loads(bytes(raw))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot token exchange returned invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot token exchange returned an invalid payload"
        )
    return payload


def _token_from_exchange_payload(
    payload: Mapping[str, Any],
    *,
    requested_origin: str,
    now: float,
) -> CopilotAPIToken:
    token = _validated_secret(payload.get("token"))
    if not token:
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot token response did not contain a usable token"
        )

    api_origin = requested_origin
    endpoints = payload.get("endpoints")
    if isinstance(endpoints, Mapping) and endpoints.get("api"):
        try:
            api_origin = validate_copilot_api_origin(endpoints.get("api"))
        except CopilotSubscriptionError as exc:
            raise CopilotSubscriptionAuthError(
                "GitHub Copilot token response advertised an untrusted API origin"
            ) from exc
        if not _same_trust_partition(requested_origin, api_origin):
            raise CopilotSubscriptionAuthError(
                "GitHub Copilot token response crossed the configured tenant boundary"
            )

    expires_at = _finite_timestamp(payload.get("expires_at"))
    refresh_in = _positive_seconds(payload.get("refresh_in"))
    if expires_at is None:
        # Missing expiry is treated as a short cache, never an hour-long guess.
        expires_at = now + (refresh_in if refresh_in is not None else 60.0)
    if expires_at <= now + 5.0:
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot token response was already expired"
        )

    refresh_at = expires_at - _DEFAULT_REFRESH_SKEW
    if refresh_in is not None:
        refresh_at = min(refresh_at, now + refresh_in)
    refresh_at = max(now + 1.0, min(refresh_at, expires_at - 5.0))
    return CopilotAPIToken(
        token=token,
        api_origin=api_origin,
        expires_at=expires_at,
        refresh_at=refresh_at,
    )


def _same_trust_partition(requested_origin: str, advertised_origin: str) -> bool:
    from urllib.parse import urlsplit

    requested_host = (urlsplit(requested_origin).hostname or "").casefold()
    advertised_host = (urlsplit(advertised_origin).hostname or "").casefold()
    requested_public = requested_host == "api.githubcopilot.com" or (
        requested_host.startswith("api.")
        and requested_host.endswith(".githubcopilot.com")
    )
    advertised_public = advertised_host == "api.githubcopilot.com" or (
        advertised_host.startswith("api.")
        and advertised_host.endswith(".githubcopilot.com")
    )
    if requested_public or advertised_public:
        return requested_public and advertised_public

    prefix = "copilot-api."
    suffix = ".ghe.com"
    if not (
        requested_host.startswith(prefix)
        and requested_host.endswith(suffix)
        and advertised_host.startswith(prefix)
        and advertised_host.endswith(suffix)
    ):
        return False
    return requested_host[len(prefix) : -len(suffix)] == advertised_host[
        len(prefix) : -len(suffix)
    ]


def _validated_secret(value: object) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    if (
        not token
        or len(token) > _MAX_TOKEN_CHARS
        or any(ord(char) < 33 or ord(char) == 127 for char in token)
    ):
        return ""
    return token


def _validated_integration_id(value: object) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 128
        or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in text)
    ):
        raise CopilotSubscriptionAuthError(
            "Copilot integration ID must contain only ASCII letters, digits, '.', '_', or '-'"
        )
    return text


def _finite_timestamp(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0.0 < parsed < 1e16):
        return None
    if parsed > 1e12:
        parsed /= 1000.0
    return parsed


def _positive_seconds(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0.0 < parsed <= 86_400.0):
        return None
    return parsed


def _validate_exchange_url(url: str) -> None:
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise CopilotSubscriptionAuthError("Copilot token endpoint is malformed") from exc
    host = (parsed.hostname or "").casefold().rstrip(".")
    expected_path = "/copilot_internal/v2/token"
    public = host == "api.github.com"
    ghe = host.startswith("api.") and host.endswith(".ghe.com")
    if (
        parsed.scheme.casefold() != "https"
        or not (public or ghe)
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise CopilotSubscriptionAuthError(
            "Copilot token exchange is restricted to GitHub-operated HTTPS endpoints"
        )


def _env_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("ENTROLY_COPILOT_SUBSCRIPTION", "")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_copilot_subscription_transport() -> bool:
    """Install the narrow auth seam on Entroly's existing hardened proxy."""
    if not _env_enabled():
        return False

    from . import proxy as proxy_module
    from .proxy_config import ProxyConfig

    current_create = proxy_module.create_proxy_app
    current_headers = proxy_module.PromptCompilerProxy._build_headers
    current_shutdown = proxy_module.PromptCompilerProxy.shutdown

    if getattr(current_create, "__entroly_copilot_subscription__", False):
        return True

    def create_proxy_app(
        engine: Any,
        config: ProxyConfig | None = None,
        start_dashboard: bool = True,
        start_autotune: bool | None = None,
    ):
        selected = config if config is not None else ProxyConfig.from_env()
        requested_origin = validate_copilot_api_origin(
            os.environ.get("ENTROLY_OPENAI_BASE", selected.openai_base_url)
        )
        manager = CopilotTokenManager(api_origin=requested_origin)
        token = manager.prime()
        selected.openai_base_url = token.api_origin
        try:
            app = current_create(
                engine,
                selected,
                start_dashboard=start_dashboard,
                start_autotune=start_autotune,
            )
        except BaseException:
            manager.stop()
            raise
        app.state.proxy._copilot_subscription_token_manager = manager
        app.state.copilot_subscription = manager.public_summary()
        return app

    def build_headers(
        self: Any,
        original: dict[str, str],
        provider: str,
    ) -> dict[str, str]:
        headers = current_headers(self, original, provider)
        if provider != "openai":
            return headers
        manager = getattr(self, "_copilot_subscription_token_manager", None)
        if not isinstance(manager, CopilotTokenManager):
            return headers

        token = manager.current_token()
        # Remove any local/custom-provider credential before adding the GitHub
        # Copilot API credential. Fixed header names avoid arbitrary injection.
        for name in list(headers):
            if name.casefold() in {"authorization", "api-key", "x-api-key"}:
                headers.pop(name, None)
        headers["Authorization"] = f"Bearer {token}"
        headers["Copilot-Integration-Id"] = manager.integration_id
        headers.setdefault("Editor-Version", "vscode/1.0")
        headers.setdefault("Editor-Plugin-Version", "copilot/1.0")
        headers.setdefault("User-Agent", "GithubCopilot/1.0")
        return headers

    async def shutdown(self: Any) -> None:
        manager = getattr(self, "_copilot_subscription_token_manager", None)
        if isinstance(manager, CopilotTokenManager):
            manager.stop()
        await current_shutdown(self)

    create_proxy_app.__entroly_copilot_subscription__ = True
    create_proxy_app.__entroly_copilot_subscription_original__ = current_create
    build_headers.__entroly_copilot_subscription__ = True
    build_headers.__entroly_copilot_subscription_original__ = current_headers
    shutdown.__entroly_copilot_subscription__ = True
    shutdown.__entroly_copilot_subscription_original__ = current_shutdown

    proxy_module.create_proxy_app = create_proxy_app
    proxy_module.PromptCompilerProxy._build_headers = build_headers
    proxy_module.PromptCompilerProxy.shutdown = shutdown
    return True


__all__ = [
    "CopilotAPIToken",
    "CopilotSubscriptionAuthError",
    "CopilotTokenManager",
    "install_copilot_subscription_transport",
]
