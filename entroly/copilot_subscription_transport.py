"""GitHub Copilot subscription authentication for Entroly's existing proxy.

This module is deliberately narrow. It does not route requests, transform model
payloads, retry provider calls, or create a second receipt/accounting layer.
Explicit Copilot subscription mode follows GitHub's current runtime contract:

* a supported GitHub user/runtime credential is the CAPI bearer;
* ``/copilot_internal/user`` is bounded entitlement/endpoint discovery, not a
  token-minting service;
* credentials remain process-local and are never persisted by Entroly;
* a generic public CAPI bootstrap may adopt GitHub's advertised SKU host, while
  an explicitly selected SKU or GHE tenant cannot silently move elsewhere;
* redirects, malformed responses, untrusted origins, and tenant changes fail
  closed;
* the local dummy provider key supplied to Copilot CLI is replaced before an
  upstream request leaves Entroly;
* Entroly never retries a provider request merely because authentication fails.

GitHub's current runtime exposes OAuth-authenticated CAPI sessions with the
resolved GitHub bearer as the provider API key. A separate direct Copilot API
credential mode exists upstream, but this subscription wrapper intentionally
does not conflate those credential classes.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from .copilot_cli_provider_contract import (
    CopilotCLIProviderContractError,
    configure_copilot_integration_identity,
)
from .copilot_subscription import (
    CopilotSubscriptionError,
    user_info_url_for_origin,
    validate_copilot_api_origin,
)

_MAX_TOKEN_CHARS = 16_384
_MAX_USER_RESPONSE_BYTES = 256 * 1024
_DEFAULT_INTEGRATION_ID = "copilot-developer-cli"
_PUBLIC_CAPI_BOOTSTRAP_HOST = "api.githubcopilot.com"
_USER_API_VERSION = "2025-04-01"
_SUPPORTED_GITHUB_PREFIXES = ("gho_", "ghu_", "github_pat_", "ghs_")
_CLASSIC_PAT_PREFIX = "ghp_"
_GITHUB_TOKEN_ENV_VARS = (
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)


class CopilotSubscriptionAuthError(RuntimeError):
    """Authentication failure whose message is safe to show to a user."""


@dataclass(frozen=True, slots=True)
class CopilotProviderCredential:
    """Process-local provider credential plus its pinned Copilot API origin."""

    token: str
    api_origin: str


class CopilotTokenManager:
    """Resolve one GitHub-backed CAPI credential for one proxy process.

    The name reflects ownership of the GitHub token used by CAPI. The manager
    deliberately does not invent token expiry or run a credential-refresh
    timer: the current GitHub runtime does not expose a trustworthy expiry at
    this boundary. Provider auth failures therefore propagate instead of
    triggering replay of a model request.
    """

    def __init__(
        self,
        *,
        api_origin: str,
        environ: Mapping[str, str] | None = None,
        integration_id: str | None = None,
        user_info_fetch: Callable[[str, str], Mapping[str, Any]] | None = None,
        credential_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._requested_origin = validate_copilot_api_origin(api_origin)
        self._user_info_url = user_info_url_for_origin(self._requested_origin)
        self._integration_id = _resolved_integration_id(
            self._environ,
            explicit=integration_id,
        )
        self._fetch_user_info = user_info_fetch or _fetch_user_info_payload
        self._credential_resolver = credential_resolver or (
            lambda: _resolve_github_credential(self._environ)
        )
        self._lock = threading.RLock()
        self._current: CopilotProviderCredential | None = None
        self._pinned_origin: str | None = None
        self._last_auth_error = ""

    @property
    def integration_id(self) -> str:
        return self._integration_id

    @property
    def api_origin(self) -> str:
        with self._lock:
            return self._pinned_origin or self._requested_origin

    def prime(self) -> CopilotProviderCredential:
        """Validate entitlement and resolve the provider boundary before startup."""
        return self._refresh(force=True)

    def current_token(self) -> str:
        """Return the process-local GitHub bearer without hidden network I/O."""
        with self._lock:
            current = self._current
        if current is None:
            raise CopilotSubscriptionAuthError(
                "Copilot subscription credential was not primed before provider traffic"
            )
        return current.token

    def stop(self) -> None:
        """Lifecycle compatibility hook; no background credential worker exists."""
        return None

    def public_summary(self) -> dict[str, object]:
        with self._lock:
            current = self._current
            return {
                "mode": "github-copilot-subscription",
                "auth_semantics": "github-bearer",
                "api_origin": self._pinned_origin or self._requested_origin,
                "integration_id": self._integration_id,
                "credential_cached": current is not None,
                "credential_persisted": False,
                "user_preflight": current is not None,
                "background_refresh": False,
                "automatic_auth_replay": False,
                "last_auth_error": self._last_auth_error[:160],
            }

    def _refresh(self, *, force: bool) -> CopilotProviderCredential:
        """Re-run endpoint/entitlement discovery without replaying model traffic."""
        with self._lock:
            if not force and self._current is not None:
                return self._current
            try:
                credential = _validated_runtime_credential(self._credential_resolver())
                payload = self._fetch_user_info(self._user_info_url, credential)
                resolved = _credential_from_user_payload(
                    credential,
                    payload,
                    requested_origin=self._requested_origin,
                )
                if self._pinned_origin is None:
                    self._pinned_origin = resolved.api_origin
                elif resolved.api_origin != self._pinned_origin:
                    raise CopilotSubscriptionAuthError(
                        "GitHub changed the Copilot API origin during this session; "
                        "restart Entroly to re-establish the trust boundary"
                    )
                self._current = resolved
                self._last_auth_error = ""
                return resolved
            except CopilotSubscriptionAuthError as exc:
                self._last_auth_error = str(exc)
                raise
            except Exception as exc:
                self._last_auth_error = type(exc).__name__
                raise CopilotSubscriptionAuthError(
                    "unable to validate the GitHub Copilot subscription credential"
                ) from exc


def _resolved_integration_id(
    environ: Mapping[str, str],
    *,
    explicit: str | None,
) -> str:
    identity_env = dict(environ)
    explicit_text = str(explicit or "").strip()
    if explicit_text and not identity_env.get("ENTROLY_COPILOT_INTEGRATION_ID"):
        identity_env["ENTROLY_COPILOT_INTEGRATION_ID"] = explicit_text
    try:
        selected = configure_copilot_integration_identity(identity_env)
    except CopilotCLIProviderContractError as exc:
        raise CopilotSubscriptionAuthError(str(exc)) from exc
    if explicit_text and explicit_text != selected:
        raise CopilotSubscriptionAuthError(
            "explicit Copilot integration ID conflicts with the configured runtime identity"
        )
    return selected or _DEFAULT_INTEGRATION_ID


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


def _validated_runtime_credential(value: object) -> str:
    token = _validated_secret(value)
    if not token:
        raise CopilotSubscriptionAuthError("GitHub credential is empty or malformed")
    if token.startswith(_CLASSIC_PAT_PREFIX):
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot CLI does not support classic `ghp_` personal access tokens; "
            "use an OAuth token or a fine-grained PAT with Copilot Requests permission"
        )
    if not token.startswith(_SUPPORTED_GITHUB_PREFIXES):
        raise CopilotSubscriptionAuthError(
            "unsupported GitHub credential type for Copilot subscription routing"
        )
    return token


def _transport_client_kwargs() -> dict[str, Any]:
    """Load Entroly's outbound trust policy without silently bypassing failures."""
    try:
        from .proxy_transport_final import _safe_http_client_kwargs
    except (ImportError, AttributeError):
        # Standalone unit-test environments may not have installed the hardened
        # proxy layer. The fallback still refuses ambient proxy inheritance.
        return {"follow_redirects": False, "trust_env": False}

    try:
        return dict(_safe_http_client_kwargs())
    except Exception as exc:
        raise CopilotSubscriptionAuthError(
            "unable to apply Entroly's outbound transport trust policy"
        ) from exc


def _fetch_user_info_payload(
    user_info_url: str,
    github_credential: str,
) -> Mapping[str, Any]:
    """Fetch bounded Copilot user/entitlement metadata using the GitHub bearer."""
    _validate_user_info_url(user_info_url)
    credential = _validated_runtime_credential(github_credential)
    headers = {
        "Authorization": f"Bearer {credential}",
        "Accept": "application/json",
        "X-GitHub-Api-Version": _USER_API_VERSION,
    }

    kwargs = _transport_client_kwargs()
    kwargs["timeout"] = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    kwargs["follow_redirects"] = False

    try:
        with httpx.Client(**kwargs) as client:
            with client.stream("GET", user_info_url, headers=headers) as response:
                if 300 <= response.status_code < 400:
                    raise CopilotSubscriptionAuthError(
                        "GitHub Copilot user preflight redirected; refusing to forward credentials"
                    )
                if response.status_code in {401, 403}:
                    raise CopilotSubscriptionAuthError(
                        "GitHub rejected the credential for Copilot subscription access"
                    )
                if response.status_code >= 400:
                    raise CopilotSubscriptionAuthError(
                        f"GitHub Copilot user preflight failed with HTTP {response.status_code}"
                    )
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if len(raw) + len(chunk) > _MAX_USER_RESPONSE_BYTES:
                        raise CopilotSubscriptionAuthError(
                            "GitHub Copilot user response exceeded the safety limit"
                        )
                    raw.extend(chunk)
    except CopilotSubscriptionAuthError:
        raise
    except (httpx.TimeoutException, httpx.TransportError, OSError) as exc:
        raise CopilotSubscriptionAuthError(
            "unable to reach GitHub's Copilot user endpoint"
        ) from exc

    try:
        payload = json.loads(bytes(raw))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot user preflight returned invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CopilotSubscriptionAuthError(
            "GitHub Copilot user preflight returned an invalid payload"
        )
    return payload


def _credential_from_user_payload(
    credential: str,
    payload: Mapping[str, Any],
    *,
    requested_origin: str,
) -> CopilotProviderCredential:
    token = _validated_runtime_credential(credential)
    if payload.get("chat_enabled") is False:
        raise CopilotSubscriptionAuthError(
            "GitHub reports that Copilot chat is not enabled for this credential"
        )

    api_origin = requested_origin
    endpoints = payload.get("endpoints")
    if isinstance(endpoints, Mapping) and endpoints.get("api"):
        try:
            api_origin = validate_copilot_api_origin(endpoints.get("api"))
        except CopilotSubscriptionError as exc:
            raise CopilotSubscriptionAuthError(
                "GitHub Copilot user response advertised an untrusted API origin"
            ) from exc
        if not _same_trust_partition(requested_origin, api_origin):
            raise CopilotSubscriptionAuthError(
                "GitHub Copilot user response crossed the configured tenant boundary"
            )

    return CopilotProviderCredential(token=token, api_origin=api_origin)


def _is_public_capi_host(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    return normalized == _PUBLIC_CAPI_BOOTSTRAP_HOST or (
        normalized.startswith("api.")
        and normalized.endswith(".githubcopilot.com")
    )


def _same_trust_partition(requested_origin: str, advertised_origin: str) -> bool:
    """Return whether endpoint discovery may replace the requested CAPI origin."""
    requested_host = (urlsplit(requested_origin).hostname or "").casefold().rstrip(".")
    advertised_host = (urlsplit(advertised_origin).hostname or "").casefold().rstrip(".")
    requested_public = _is_public_capi_host(requested_host)
    advertised_public = _is_public_capi_host(advertised_host)

    if requested_public or advertised_public:
        if not (requested_public and advertised_public):
            return False
        if requested_host == _PUBLIC_CAPI_BOOTSTRAP_HOST:
            return True
        return requested_host == advertised_host

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


def _validate_user_info_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise CopilotSubscriptionAuthError("Copilot user endpoint is malformed") from exc
    host = (parsed.hostname or "").casefold().rstrip(".")
    expected_path = "/copilot_internal/user"
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
            "Copilot user preflight is restricted to GitHub-operated HTTPS endpoints"
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
        credential = manager.prime()
        selected.openai_base_url = credential.api_origin
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
        for name in list(headers):
            if name.casefold() in {"authorization", "api-key", "x-api-key"}:
                headers.pop(name, None)
        headers["Authorization"] = f"Bearer {token}"
        headers["Copilot-Integration-Id"] = manager.integration_id
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
    "CopilotProviderCredential",
    "CopilotSubscriptionAuthError",
    "CopilotTokenManager",
    "install_copilot_subscription_transport",
]
