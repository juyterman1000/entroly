"""Final correctness guard for live Traffic Receipts.

Traffic Receipts are buyer-visible evidence, so attribution must fail closed.
This module is intentionally narrow and installs after ``proxy_traffic_receipt``:

* establishes one request id at admission when the client did not provide one,
  so the receipt and provider usage ledger correlate on the same id;
* keeps recursive exact-recovery attempts from finalizing the outer receipt;
* withholds shared proxy coverage state until a request-local coverage source is
  wired, preventing concurrent requests from borrowing each other's evidence;
* tightens the UI wording so SHA-256 payload integrity is not presented as a
  cryptographic signature and recovery evidence is not mislabeled as recovery
  availability.

It does not change routing, compression, retry, provider, or recovery policy.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

from . import proxy as _proxy
from . import proxy_traffic_receipt as _receipt


def _ensure_request_id(request: Request) -> str:
    """Ensure downstream proxy and receipt code observe the same request id."""
    scope = request.scope
    raw_headers = list(scope.get("headers") or [])
    for key, value in raw_headers:
        if bytes(key).lower() == b"x-request-id":
            try:
                return bytes(value).decode("latin-1")
            except Exception:
                return ""

    request_id = uuid.uuid4().hex[:16]
    raw_headers.append((b"x-request-id", request_id.encode("ascii")))
    scope["headers"] = raw_headers
    # Starlette caches Headers on first access. Invalidate only that derived
    # view; body admission/cache state is untouched.
    if hasattr(request, "_headers"):
        try:
            delattr(request, "_headers")
        except Exception:
            pass
    return request_id


def _nested_recovery_depth(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Extract PromptCompilerProxy._forward_response recovery_depth safely."""
    raw = kwargs.get("recovery_depth")
    if raw is None and len(args) >= 6:
        # Positional fields after body are:
        # selected_frag_ids, witness_context, provider, recoverable_fragments,
        # request_id, recovery_depth, ...
        raw = args[5]
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _request_local_coverage_unavailable(_proxy_instance: Any) -> tuple[None, str, str]:
    """Do not publish shared mutable coverage as per-request evidence."""
    return None, "withheld_shared_state", "UNKNOWN"


def _tighten_receipt_html() -> None:
    html = _receipt._TRAFFIC_HTML
    html = html.replace(
        '<div class="label">Recoverable</div>',
        '<div class="label">Recovery evidence</div>',
    )
    html = html.replace(
        '<div class="label">Traffic Receipt</div><div class="value good">✓ VERIFIED</div>',
        '<div class="label">Receipt integrity</div><div class="value good">✓ SHA-256 OK</div>',
    )
    _receipt._TRAFFIC_HTML = html


def install_receipt_final_guard() -> None:
    """Install attribution/correlation guards exactly once."""
    current_run = _receipt._run_traffic_handle_proxy
    if not hasattr(current_run, "__entroly_receipt_final_original__"):
        original_run = current_run

        async def run_with_request_id(
            proxy: Any,
            request: Request,
            original: Callable[[Any, Request], Awaitable[Response]],
        ) -> Response:
            _ensure_request_id(request)
            return await original_run(proxy, request, original)

        run_with_request_id.__entroly_receipt_final_original__ = original_run
        _receipt._run_traffic_handle_proxy = run_with_request_id

    current_forward = _proxy.PromptCompilerProxy._forward_response
    if not hasattr(current_forward, "__entroly_receipt_final_original__"):
        receipt_forward = current_forward
        core_forward = _receipt._ORIGINAL_FORWARD_RESPONSE

        async def depth_safe_forward(
            self: Any,
            url: str,
            headers: dict[str, str],
            body: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> Response:
            if _nested_recovery_depth(args, kwargs) > 0:
                # Nested exact-recovery attempts contribute to the final outer
                # response but must never own/finalize the request receipt.
                return await core_forward(self, url, headers, body, *args, **kwargs)
            return await receipt_forward(self, url, headers, body, *args, **kwargs)

        depth_safe_forward.__entroly_receipt_final_original__ = receipt_forward
        _proxy.PromptCompilerProxy._forward_response = depth_safe_forward

    # The current proxy exposes coverage through shared mutable last-* fields.
    # Until the optimizer supplies a ContextVar/request-local value, publishing
    # that as request evidence is worse than showing it as unavailable.
    _receipt._coverage_snapshot = _request_local_coverage_unavailable
    _tighten_receipt_html()


install_receipt_final_guard()


__all__ = [
    "_ensure_request_id",
    "_nested_recovery_depth",
    "install_receipt_final_guard",
]
