"""Receipt-first Python client and SDK wrappers for the Entroly sidecar."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping


class GatewayError(RuntimeError):
    """The compression sidecar rejected or could not process a request."""


@dataclass(frozen=True)
class GatewayReceipt:
    count: int
    recovery: str
    compression: str
    headers: dict[str, str]


@dataclass(frozen=True)
class GatewayCompression:
    payload: dict[str, Any]
    receipt: GatewayReceipt


class CompressionGatewayClient:
    """Call the local recoverable ``/v1/compress`` contract.

    Non-loopback gateways require an explicit opt-in so an integration cannot
    silently send prompts to a new remote destination.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9377",
        *,
        provider: str = "openai",
        budget_tokens: int = 32_000,
        access_token: str = "",
        sidecar_token: str = "",
        timeout: float = 10.0,
        allow_remote: bool = False,
        opener: Any = urllib.request.urlopen,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("gateway URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("gateway URL must not contain credentials")
        if parsed.hostname.casefold() not in {"localhost", "127.0.0.1", "::1"} and not allow_remote:
            raise ValueError("remote Entroly gateways require allow_remote=True")
        if provider not in {"openai", "anthropic", "gemini"}:
            raise ValueError(f"unsupported provider: {provider}")
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.budget_tokens = int(budget_tokens)
        self.access_token = access_token
        self.sidecar_token = sidecar_token
        self.timeout = float(timeout)
        self._opener = opener

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.access_token:
            headers["X-Entroly-Access-Token"] = self.access_token
        if self.sidecar_token:
            headers["X-Entroly-Sidecar-Token"] = self.sidecar_token
        return headers

    def _request(self, request: urllib.request.Request) -> tuple[dict[str, Any], dict[str, str]]:
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
                headers = {key.casefold(): value for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise GatewayError(f"gateway rejected request with HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise GatewayError(f"gateway request failed: {exc}") from exc
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayError("gateway returned a non-JSON response") from exc
        if not isinstance(body, dict):
            raise GatewayError("gateway returned a non-object JSON response")
        return body, headers

    def compress_payload(
        self,
        payload: Mapping[str, Any],
        *,
        provider: str | None = None,
        budget_tokens: int | None = None,
    ) -> GatewayCompression:
        selected_provider = provider or self.provider
        if selected_provider not in {"openai", "anthropic", "gemini"}:
            raise ValueError(f"unsupported provider: {selected_provider}")
        selected_budget = self.budget_tokens if budget_tokens is None else int(budget_tokens)
        if selected_budget <= 0:
            raise ValueError("budget_tokens must be positive")
        query = urllib.parse.urlencode(
            {"provider": selected_provider, "budget_tokens": selected_budget}
        )
        request = urllib.request.Request(
            f"{self.base_url}/v1/compress?{query}",
            data=json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        body, headers = self._request(request)
        entroly_headers = {
            key: value for key, value in headers.items() if key.startswith("x-entroly-")
        }
        receipt = GatewayReceipt(
            count=int(entroly_headers.get("x-entroly-receipt-count", "0")),
            recovery=entroly_headers.get("x-entroly-recovery", "unknown"),
            compression=entroly_headers.get("x-entroly-compression", "unknown"),
            headers=entroly_headers,
        )
        return GatewayCompression(body, receipt)

    async def async_compress_payload(self, payload: Mapping[str, Any], **kwargs: Any) -> GatewayCompression:
        return await asyncio.to_thread(self.compress_payload, payload, **kwargs)

    def retrieve(
        self,
        *,
        receipt_id: str,
        span_id: str,
        retrieval_id: str = "",
    ) -> dict[str, Any]:
        if not receipt_id or not span_id:
            raise ValueError("receipt_id and span_id are required")
        query = {"receipt_id": receipt_id, "span_id": span_id}
        if retrieval_id:
            query["retrieval_id"] = retrieval_id
        request = urllib.request.Request(
            f"{self.base_url}/retrieve?{urllib.parse.urlencode(query)}",
            headers=self._headers(),
            method="GET",
        )
        body, _ = self._request(request)
        return body

    def retrieve_image(self, receipt_id: str) -> dict[str, Any]:
        if not receipt_id:
            raise ValueError("receipt_id is required")
        request = urllib.request.Request(
            f"{self.base_url}/retrieve-image?{urllib.parse.urlencode({'receipt_id': receipt_id})}",
            headers=self._headers(),
            method="GET",
        )
        body, _ = self._request(request)
        return body


def wrap_openai(client: Any, *, gateway: CompressionGatewayClient | None = None) -> Any:
    gateway = gateway or CompressionGatewayClient(provider="openai")

    class _Create:
        def __init__(self, create: Any) -> None:
            self._create = create

        def create(self, **params: Any) -> Any:
            return self._create(**gateway.compress_payload(params, provider="openai").payload)

    wrapped = SimpleNamespace(
        raw=client,
        entroly_gateway=gateway,
        chat=SimpleNamespace(completions=_Create(client.chat.completions.create)),
    )
    if getattr(client, "responses", None) is not None:
        wrapped.responses = _Create(client.responses.create)
    return wrapped


def wrap_anthropic(client: Any, *, gateway: CompressionGatewayClient | None = None) -> Any:
    gateway = gateway or CompressionGatewayClient(provider="anthropic")

    class _Messages:
        def create(self, **params: Any) -> Any:
            return client.messages.create(
                **gateway.compress_payload(params, provider="anthropic").payload
            )

    return SimpleNamespace(raw=client, entroly_gateway=gateway, messages=_Messages())


__all__ = [
    "CompressionGatewayClient",
    "GatewayCompression",
    "GatewayError",
    "GatewayReceipt",
    "wrap_anthropic",
    "wrap_openai",
]
