"""Final correctness guard for live Traffic Receipts.

This layer owns request correlation, nested-recovery isolation, request-local
coverage policy, and final value-attribution lifecycle. It does not change
routing, compression, retry, provider, or recovery policy.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from . import proxy as _proxy
from . import proxy_traffic_receipt as _receipt
from . import proxy_traffic_session as _value_state


def _ensure_request_id(request: Request) -> str:
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
    if hasattr(request, "_headers"):
        try:
            delattr(request, "_headers")
        except Exception:
            pass
    return request_id


def _nested_recovery_depth(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    raw = kwargs.get("recovery_depth")
    if raw is None and len(args) >= 6:
        raw = args[5]
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _request_local_coverage_unavailable(_proxy_instance: Any) -> tuple[None, str, str]:
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
    if "Value attribution" not in html:
        html = html.replace(
            "function render(r){",
            "function attribution(r){const rows=(r.value_contributions||[]).filter(x=>Number(x.tokens||0)!==0||Number(x.micro_usd||0)!==0);if(!rows.length)return '—';return rows.slice(0,6).map(x=>esc(x.source).replaceAll('_',' ')+' · '+esc(String(x.tier||'').toUpperCase())+(x.tokens?' · '+(x.tokens<0?'-':'')+fmt(Math.abs(x.tokens))+' tok':'')).join('<br>');}\nfunction render(r){",
        )
        marker = '<div class="row sep"><div class="label">Tokens avoided</div><div class="value big good">${fmt(r.tokens_avoided)}</div></div>'
        html = html.replace(
            marker,
            marker + '<div class="row"><div class="label">Value attribution</div><div class="value reason">${attribution(r)}</div></div>',
        )
    _receipt._TRAFFIC_HTML = html


def _state_for(context: Any) -> _value_state.AttributionState | None:
    return _value_state.CURRENT_ATTRIBUTION.get() or _value_state.active_state(
        str(getattr(context, "request_id", ""))
    )


def _add_receipt_evidence(state: _value_state.AttributionState, built: Any) -> None:
    cache_benefit = getattr(built, "cache_benefit_micro_usd", None)
    if cache_benefit is not None and int(cache_benefit) > 0:
        _value_state.record_internal(
            "provider_cache",
            tier=_value_state.ValueTier.MEASURED,
            role=_value_state.AccountingRole.PROTECTED,
            tokens=max(0, int(getattr(built, "cache_read_tokens", 0) or 0)),
            micro_usd=int(cache_benefit),
            evidence_source=_value_state.EvidenceSource.PROVIDER_USAGE,
            state=state,
        )
    warm = max(0, int(getattr(built, "warm_prefix_protected_tokens", 0) or 0))
    if warm:
        _value_state.record_internal(
            "warm_prefix_protection",
            tier=_value_state.ValueTier.MEASURED,
            role=_value_state.AccountingRole.PROTECTED,
            tokens=warm,
            state=state,
        )
    recovered = max(0, int(getattr(built, "recovery_receipts", 0) or 0))
    if recovered:
        _value_state.record_internal(
            "recovery_evidence",
            tier=_value_state.ValueTier.MEASURED,
            role=_value_state.AccountingRole.EXPLANATORY,
            details={"receipts": recovered},
            state=state,
        )


def _install_value_lifecycle() -> None:
    current_payload = _receipt.TrafficReceipt.payload
    if not hasattr(current_payload, "__entroly_value_state_original__"):
        original_payload = current_payload

        def payload_with_value(self: Any) -> dict[str, Any]:
            value = original_payload(self)
            value.update(_value_state.receipt_meta(str(getattr(self, "receipt_id", ""))))
            return value

        payload_with_value.__entroly_value_state_original__ = original_payload
        _receipt.TrafficReceipt.payload = payload_with_value

    current_snapshot = _receipt.TrafficReceiptLedger.snapshot
    if not hasattr(current_snapshot, "__entroly_value_state_original__"):
        original_snapshot = current_snapshot

        def snapshot_with_value(self: Any, *, limit: int | None = None) -> dict[str, Any]:
            snapshot = original_snapshot(self, limit=limit)
            for row in [snapshot.get("latest"), *(snapshot.get("recent") or [])]:
                if isinstance(row, dict):
                    row.update(_value_state.receipt_meta(str(row.get("receipt_id") or "")))
            snapshot["attribution_schema_version"] = _value_state.ATTRIBUTION_SCHEMA
            return snapshot

        snapshot_with_value.__entroly_value_state_original__ = original_snapshot
        _receipt.TrafficReceiptLedger.snapshot = snapshot_with_value

    current_build = _receipt._build_receipt
    if not hasattr(current_build, "__entroly_value_state_original__"):
        original_build = current_build

        def build_with_value(context: Any, response: Response):
            state = _state_for(context)
            if state is not None and state.lifecycle == "admitted":
                state.lifecycle = (
                    "buffered_completed"
                    if int(getattr(response, "status_code", 0) or 0) < 500
                    else "buffered_error_response"
                )
            built = original_build(context, response)
            if state is None:
                return built
            _value_state.set_canonical_context_delta(
                state, int(getattr(built, "tokens_avoided", 0) or 0)
            )
            _add_receipt_evidence(state, built)
            rows = _value_state.aggregate_contributions(state.contributions)
            headline = sum(
                int(row.get("tokens", 0) or 0)
                for row in rows
                if bool(row.get("headline_included", False))
            )
            extra_cost = (
                int(state.extra_provider_cost_micro_usd)
                if state.extra_provider_priced_calls > 0
                else None
            )
            _value_state.remember_receipt_meta(
                built.receipt_id,
                {
                    "attribution_schema_version": _value_state.ATTRIBUTION_SCHEMA,
                    "value_contributions": rows,
                    "attribution_reconciled": headline
                    == int(getattr(built, "tokens_avoided", 0) or 0),
                    "extra_provider_calls": int(state.extra_provider_calls),
                    "extra_provider_tokens": int(state.extra_provider_tokens),
                    "extra_provider_cost_micro_usd": extra_cost,
                    "lifecycle_outcome": state.lifecycle,
                },
            )
            object.__setattr__(
                built,
                "receipt_digest",
                hashlib.sha256(_receipt._canonical_json(built.payload())).hexdigest(),
            )
            state.finalized = True
            return built

        build_with_value.__entroly_value_state_original__ = original_build
        _receipt._build_receipt = build_with_value

    current_iterator = _receipt._finalizing_iterator
    if not hasattr(current_iterator, "__entroly_value_state_original__"):

        async def iterator_with_value(iterator: Any, *, context: Any, response: Response):
            state = _value_state.active_state(str(getattr(context, "request_id", "")))
            token = (
                _value_state.CURRENT_ATTRIBUTION.set(state)
                if state is not None
                else None
            )
            try:
                async for chunk in iterator:
                    yield chunk
            except BaseException:
                if state is not None:
                    state.lifecycle = "stream_error"
                raise
            else:
                if state is not None:
                    state.lifecycle = "stream_completed"
            finally:
                try:
                    _receipt._complete_context(context, response)
                finally:
                    if token is not None:
                        _value_state.CURRENT_ATTRIBUTION.reset(token)
                    _value_state.forget_active(str(getattr(context, "request_id", "")))

        iterator_with_value.__entroly_value_state_original__ = current_iterator
        _receipt._finalizing_iterator = iterator_with_value

    current_run = _receipt._run_traffic_handle_proxy
    if not hasattr(current_run, "__entroly_value_state_original__"):
        original_run = current_run

        async def run_with_value(
            proxy: Any,
            request: Request,
            original: Callable[[Any, Request], Awaitable[Response]],
        ) -> Response:
            request_id = _ensure_request_id(request)
            state = _value_state.AttributionState(request_id=request_id)
            _value_state.remember_active(state)
            token = _value_state.CURRENT_ATTRIBUTION.set(state)
            response: Response | None = None
            try:
                response = await original_run(proxy, request, original)
                if isinstance(response, StreamingResponse):
                    state.lifecycle = "stream_open"
                return response
            except BaseException:
                state.lifecycle = "request_error"
                raise
            finally:
                _value_state.CURRENT_ATTRIBUTION.reset(token)
                if not isinstance(response, StreamingResponse):
                    _value_state.forget_active(request_id)

        run_with_value.__entroly_value_state_original__ = original_run
        _receipt._run_traffic_handle_proxy = run_with_value


def install_receipt_final_guard() -> None:
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
                return await core_forward(self, url, headers, body, *args, **kwargs)
            return await receipt_forward(self, url, headers, body, *args, **kwargs)

        depth_safe_forward.__entroly_receipt_final_original__ = receipt_forward
        _proxy.PromptCompilerProxy._forward_response = depth_safe_forward

    _receipt._coverage_snapshot = _request_local_coverage_unavailable
    _tighten_receipt_html()
    _install_value_lifecycle()


install_receipt_final_guard()


__all__ = [
    "_ensure_request_id",
    "_nested_recovery_depth",
    "_request_local_coverage_unavailable",
    "install_receipt_final_guard",
]
