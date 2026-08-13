"""Live, content-blind AI traffic receipts for the Entroly proxy.

This module turns the proxy's existing context, cache, routing, usage and
verification signals into one bounded per-request product surface. It does not
introduce another router, retry loop, optimizer or pricing model.

The design is deliberately evidence conservative:

* prompt/message content and credentials are never stored in receipts;
* token counts are deterministic local estimates unless provider usage exists;
* cache hits come from provider-reported usage (or remain unknown);
* monetary values require provider usage plus an explicit pricing catalog;
* "net measured saving" remains unavailable until a measured counterfactual is
  linked, rather than presenting modeled compression savings as invoice truth;
* the receipt is SHA-256 self-verifying.

The installer wraps the already-hardened live proxy seams. The request wrapper
is placed *inside* the bounded transport admission path, so reading the cached
request body here cannot bypass the proxy's request-size limit.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import proxy as _proxy
from . import proxy_transport_safe as _transport
from .provider_adapters import canonical_request_from_provider_body
from .proxy_transform import detect_provider, extract_model
from .usage_ledger import TokenUsage, parse_provider_usage

_SCHEMA_VERSION = "entroly.traffic-receipt.v1"
_DEFAULT_MAX_RECORDS = 100
_MAX_RECORDS_LIMIT = 1_000
_CORRELATION_SALT = os.urandom(32)
_CURRENT_CONTEXT: contextvars.ContextVar["_TrafficRequestContext | None"] = (
    contextvars.ContextVar("entroly_traffic_receipt_context", default=None)
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        parsed = default if raw is None else int(raw)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bounded(value: object, *, limit: int = 180) -> str:
    return (
        str(value or "")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()[:limit]
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _correlation_digest(value: str) -> str:
    return hashlib.sha256(
        _CORRELATION_SALT + value.encode("utf-8", errors="replace")
    ).hexdigest()[:16]


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return default


def _lower_headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {str(key).casefold(): str(value) for key, value in headers.items()}


def _classify_client(headers: Mapping[str, str]) -> str:
    """Return a bounded product label without retaining the raw user-agent."""
    tool = _bounded(headers.get("x-entroly-tool", ""), limit=64)
    if tool:
        return tool
    user_agent = str(headers.get("user-agent", "")).casefold()
    markers = (
        ("claude-code", "Claude Code"),
        ("claude code", "Claude Code"),
        ("codex", "Codex"),
        ("cursor", "Cursor"),
        ("openclaw", "OpenClaw"),
        ("opencode", "OpenCode"),
        ("cline", "Cline"),
        ("continue", "Continue"),
        ("aider", "Aider"),
    )
    for marker, label in markers:
        if marker in user_agent:
            return label
    return "AI client"


def _estimate_context_tokens(
    provider: str,
    body: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    path: str = "",
) -> int:
    """Use the canonical gateway estimator; fall back to bounded JSON bytes."""
    try:
        adapter = canonical_request_from_provider_body(
            provider,
            body,
            headers=headers,
            path=path,
        )
        return max(
            0,
            int(adapter.prefix_tokens_estimate)
            + int(adapter.new_input_tokens_estimate),
        )
    except Exception:
        try:
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            return max(1, len(encoded) // 4)
        except Exception:
            return 0


def _response_json(response: Response) -> Mapping[str, Any] | None:
    payload = getattr(response, "body", None)
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return None
    try:
        decoded = json.loads(bytes(payload))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _usage_payload_present(payload: Mapping[str, Any]) -> bool:
    if any(key in payload for key in ("usage", "usage_metadata", "usageMetadata")):
        return True
    for key in ("message", "response"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and "usage" in nested:
            return True
    return False


def _provider_usage(response: Response, provider: str) -> TokenUsage | None:
    payload = _response_json(response)
    if payload is None or not _usage_payload_present(payload):
        return None
    try:
        return parse_provider_usage(provider, payload)
    except Exception:
        return None


def _input_cost_micro_usd(usage: TokenUsage, pricing: Any) -> int:
    """Price input/cache categories only, using integer microdollars."""
    raw = (
        Decimal(usage.uncached_input_tokens) * pricing.input_per_million
        + Decimal(usage.cache_read_tokens) * pricing.cache_read_per_million
        + Decimal(usage.cache_write_tokens) * pricing.cache_write_rate
    )
    return int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _cache_benefit_micro_usd(usage: TokenUsage, pricing: Any) -> int:
    no_cache = Decimal(usage.cache_read_tokens + usage.cache_write_tokens) * (
        pricing.input_per_million
    )
    actual = (
        Decimal(usage.cache_read_tokens) * pricing.cache_read_per_million
        + Decimal(usage.cache_write_tokens) * pricing.cache_write_rate
    )
    return max(
        0,
        int((no_cache - actual).quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
    )


def _money_for_request(
    proxy: Any,
    *,
    provider: str,
    model: str,
    usage: TokenUsage | None,
) -> tuple[int | None, int | None, str]:
    if usage is None:
        return None, None, "provider_usage_unavailable"
    catalog = getattr(proxy, "_pricing_catalog", None)
    pricing = catalog.resolve(provider, model) if catalog is not None else None
    if pricing is None:
        return None, None, "auditable_pricing_unavailable"
    return (
        _input_cost_micro_usd(usage, pricing),
        _cache_benefit_micro_usd(usage, pricing),
        _bounded(getattr(pricing, "source", "explicit"), limit=160),
    )


def _coverage_snapshot(proxy: Any) -> tuple[float | None, str, str]:
    raw = getattr(proxy, "_last_coverage", None)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        value = -1.0
    if 0.0 <= value <= 1.0:
        percent: float | None = round(value * 100.0, 2)
        source = "context_coverage_estimate"
    else:
        percent = None
        source = "unavailable"
    risk = _bounded(getattr(proxy, "_last_coverage_risk", "unknown"), limit=48)
    return percent, source, (risk.upper() if risk else "UNKNOWN")


def _verification(response: Response) -> str:
    headers = _lower_headers(getattr(response, "headers", {}))
    witness = headers.get("x-entroly-witness", "").casefold()
    eicv = headers.get("x-entroly-eicv", "").casefold()
    if witness == "flagged" or eicv in {"hallucinated", "flagged"}:
        return "FAIL"
    if witness == "pass" or eicv == "clean":
        return "PASS"
    if witness == "error" or eicv == "error":
        return "ERROR"
    return "NOT_REPORTED"


def _recovery_state(
    request_headers: Mapping[str, str],
    response: Response,
) -> tuple[bool, int]:
    merged = dict(_lower_headers(request_headers))
    merged.update(_lower_headers(getattr(response, "headers", {})))
    keys = (
        "x-entroly-session-recovery-receipts",
        "x-entroly-recovery-fragments",
        "x-entroly-recovered-fragments",
    )
    receipts = max((_safe_int(merged.get(key)) for key in keys), default=0)
    recovered = merged.get("x-entroly-recovered", "").casefold() == "true"
    return bool(receipts > 0 or recovered), receipts


def _prefix_protection(
    request_headers: Mapping[str, str],
    response: Response,
) -> int:
    merged = dict(_lower_headers(request_headers))
    merged.update(_lower_headers(getattr(response, "headers", {})))
    if merged.get("x-entroly-prefix-guard", "") != "preserve_warm_prefix":
        return 0
    return _safe_int(merged.get("x-entroly-prefix-tokens-at-risk"))


def _routing_decision(
    requested_model: str,
    executed_model: str,
    response: Response,
) -> tuple[str, str]:
    headers = _lower_headers(getattr(response, "headers", {}))
    explicit = _bounded(headers.get("x-entroly-routing-decision", ""), limit=64)
    reason = _bounded(headers.get("x-entroly-routing-reason", ""), limit=180)
    if explicit:
        normalized = explicit.upper().replace("_", "-")
        if normalized in {"EXECUTED", "WOULD-SWITCH", "SWITCH"}:
            decision = "SWITCH"
        elif normalized in {"STAY", "DENIED", "NO-PROPOSAL"}:
            decision = "STAY"
        else:
            decision = normalized
        return decision, reason or "routing authority decision"
    if requested_model and executed_model and requested_model != executed_model:
        return "SWITCH", "authorized model rewrite"
    return "STAY", "requested model preserved"


@dataclass(frozen=True, slots=True)
class TrafficReceipt:
    schema_version: str
    receipt_id: str
    request_correlation: str
    client: str
    provider: str
    requested_model: str
    executed_model: str
    original_context_tokens: int
    entroly_context_tokens: int
    tokens_avoided: int
    evidence_retained_pct: float | None
    evidence_retained_source: str
    recoverable: bool
    recovery_receipts: int
    warm_prefix_protected_tokens: int
    cache_hit: bool | None
    cache_read_tokens: int
    routing_decision: str
    routing_reason: str
    input_cost_micro_usd: int | None
    cache_benefit_micro_usd: int | None
    net_measured_saving_micro_usd: int | None
    money_source: str
    context_risk: str
    verification: str
    response_status: int | None
    streaming: bool
    latency_ms: float
    observed_at: float
    receipt_digest: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("receipt_digest", None)
        return value

    def verify(self) -> bool:
        return (
            hashlib.sha256(_canonical_json(self.payload())).hexdigest()
            == self.receipt_digest
        )


class TrafficReceiptLedger:
    """Bounded in-memory receipt ledger containing no prompt or credentials."""

    def __init__(self, *, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self.max_records = max(1, min(int(max_records), _MAX_RECORDS_LIMIT))
        self._records: deque[TrafficReceipt] = deque(maxlen=self.max_records)
        self._lock = threading.RLock()
        self._requests = 0
        self._completed = 0
        self._failures = 0

    def register_request(self) -> None:
        with self._lock:
            self._requests += 1

    def fail(self) -> None:
        with self._lock:
            self._failures += 1

    def append(self, receipt: TrafficReceipt) -> None:
        if not receipt.verify():
            raise ValueError("traffic receipt digest verification failed")
        with self._lock:
            self._records.append(receipt)
            self._completed += 1

    def snapshot(self, *, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            requested = self.max_records if limit is None else max(1, int(limit))
            selected = records[-min(requested, self.max_records) :]
            return {
                "schema_version": _SCHEMA_VERSION,
                "records_contain_prompt_content": False,
                "records_contain_credentials": False,
                "money_policy": (
                    "provider usage + explicit pricing only; net measured saving "
                    "requires a linked measured counterfactual"
                ),
                "requests": self._requests,
                "completed": self._completed,
                "failures": self._failures,
                "max_records": self.max_records,
                "latest": asdict(records[-1]) if records else None,
                "recent": [asdict(record) for record in reversed(selected)],
            }


@dataclass(slots=True)
class _TrafficRequestContext:
    proxy: Any
    request_id: str
    request_correlation: str
    client: str
    provider: str
    path: str
    headers: Mapping[str, str]
    requested_model: str
    original_context_tokens: int
    started_at: float = field(default_factory=time.perf_counter)
    conversation_id: str = ""
    cache_hits_before: int = 0
    cache_misses_before: int = 0
    executed_model: str = ""
    entroly_context_tokens: int = 0
    evidence_retained_pct: float | None = None
    evidence_retained_source: str = "unavailable"
    context_risk: str = "UNKNOWN"
    outbound_headers: dict[str, str] = field(default_factory=dict)
    streaming: bool = False
    completed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


def _ledger_for_proxy(proxy: Any) -> TrafficReceiptLedger:
    ledger = getattr(proxy, "_traffic_receipt_ledger", None)
    if isinstance(ledger, TrafficReceiptLedger):
        return ledger
    ledger = TrafficReceiptLedger(
        max_records=_env_int(
            "ENTROLY_TRAFFIC_RECEIPT_MAX_RECORDS",
            _DEFAULT_MAX_RECORDS,
            minimum=1,
            maximum=_MAX_RECORDS_LIMIT,
        )
    )
    proxy._traffic_receipt_ledger = ledger
    return ledger


def _usage_for_context(
    context: _TrafficRequestContext,
    response: Response,
) -> TokenUsage | None:
    usage = _provider_usage(response, context.provider)
    if usage is not None:
        return usage
    ledger = getattr(context.proxy, "_usage_ledger", None)
    if ledger is not None and context.request_id:
        try:
            event = ledger.get(context.request_id)
        except Exception:
            event = None
        if event is not None:
            return event.usage
    return None


def _cache_observation(
    context: _TrafficRequestContext,
    usage: TokenUsage | None,
) -> tuple[bool | None, int]:
    if usage is not None:
        return usage.cache_read_tokens > 0, usage.cache_read_tokens
    if not context.conversation_id:
        return None, 0
    router = getattr(context.proxy, "_cache_router", None)
    if router is None:
        return None, 0
    try:
        lease = router.lease_snapshot(context.conversation_id)
    except Exception:
        return None, 0
    if lease is None:
        return None, 0
    if int(lease.hits) > context.cache_hits_before:
        return True, max(0, int(lease.cached_prefix_tokens))
    if int(lease.misses) > context.cache_misses_before:
        return False, 0
    return None, 0


def _capture_outbound_state(
    context: _TrafficRequestContext,
    body: Mapping[str, Any],
    *,
    extra_headers: Mapping[str, Any] | None,
    streaming: bool,
) -> None:
    context.executed_model = _bounded(
        str(body.get("model") or context.requested_model), limit=128
    )
    context.entroly_context_tokens = _estimate_context_tokens(
        context.provider,
        body,
        headers=context.headers,
        path=context.path,
    )
    context.outbound_headers.update(_lower_headers(extra_headers))
    context.streaming = streaming
    (
        context.evidence_retained_pct,
        context.evidence_retained_source,
        context.context_risk,
    ) = _coverage_snapshot(context.proxy)


def _build_receipt(
    context: _TrafficRequestContext,
    response: Response,
) -> TrafficReceipt:
    usage = _usage_for_context(context, response)
    executed_model = context.executed_model or context.requested_model
    response_headers = _lower_headers(getattr(response, "headers", {}))
    escalated_to = _bounded(response_headers.get("x-entroly-escalated-to", ""), limit=128)
    if escalated_to:
        executed_model = escalated_to

    input_cost, cache_benefit, money_source = _money_for_request(
        context.proxy,
        provider=context.provider,
        model=executed_model,
        usage=usage,
    )
    recoverable, recovery_receipts = _recovery_state(
        context.outbound_headers,
        response,
    )
    protected_tokens = _prefix_protection(context.outbound_headers, response)
    routing_decision, routing_reason = _routing_decision(
        context.requested_model,
        executed_model,
        response,
    )
    optimized = context.entroly_context_tokens or context.original_context_tokens
    tokens_avoided = max(0, context.original_context_tokens - optimized)
    cache_hit, cache_read_tokens = _cache_observation(context, usage)
    unsigned: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "receipt_id": f"tr_{uuid.uuid4().hex[:16]}",
        "request_correlation": context.request_correlation,
        "client": context.client,
        "provider": _bounded(context.provider, limit=48),
        "requested_model": _bounded(context.requested_model, limit=128),
        "executed_model": _bounded(executed_model, limit=128),
        "original_context_tokens": max(0, context.original_context_tokens),
        "entroly_context_tokens": max(0, optimized),
        "tokens_avoided": tokens_avoided,
        "evidence_retained_pct": context.evidence_retained_pct,
        "evidence_retained_source": context.evidence_retained_source,
        "recoverable": recoverable,
        "recovery_receipts": recovery_receipts,
        "warm_prefix_protected_tokens": protected_tokens,
        "cache_hit": cache_hit,
        "cache_read_tokens": cache_read_tokens,
        "routing_decision": routing_decision,
        "routing_reason": routing_reason,
        "input_cost_micro_usd": input_cost,
        "cache_benefit_micro_usd": cache_benefit,
        "net_measured_saving_micro_usd": None,
        "money_source": money_source,
        "context_risk": context.context_risk,
        "verification": _verification(response),
        "response_status": int(getattr(response, "status_code", 0) or 0) or None,
        "streaming": context.streaming,
        "latency_ms": round((time.perf_counter() - context.started_at) * 1000.0, 3),
        "observed_at": time.time(),
    }
    digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return TrafficReceipt(receipt_digest=digest, **unsigned)


def _complete_context(context: _TrafficRequestContext, response: Response) -> None:
    with context.lock:
        if context.completed:
            return
        context.completed = True
    try:
        receipt = _build_receipt(context, response)
        _ledger_for_proxy(context.proxy).append(receipt)
    except Exception:
        _ledger_for_proxy(context.proxy).fail()
        _proxy.logger.warning("Traffic receipt finalization failed", exc_info=True)


async def _finalizing_iterator(
    iterator: AsyncIterator[bytes | str],
    *,
    context: _TrafficRequestContext,
    response: Response,
) -> AsyncIterator[bytes | str]:
    try:
        async for chunk in iterator:
            yield chunk
    finally:
        _complete_context(context, response)


async def _run_traffic_handle_proxy(
    proxy: Any,
    request: Request,
    original: Callable[[Any, Request], Awaitable[Response]],
) -> Response:
    """Start a receipt after bounded transport has admitted the request body."""
    body: Mapping[str, Any] = {}
    try:
        decoded = json.loads(await request.body())
        if isinstance(decoded, Mapping):
            body = decoded
    except Exception:
        body = {}

    headers = {str(key).casefold(): str(value) for key, value in request.headers.items()}
    path = request.url.path
    try:
        provider = detect_provider(path, dict(headers), dict(body))
    except Exception:
        provider = "unknown"
    requested_model = ""
    if body:
        try:
            requested_model = extract_model(dict(body), path) or str(body.get("model") or "")
        except Exception:
            requested_model = str(body.get("model") or "")
    request_id = headers.get("x-request-id") or uuid.uuid4().hex[:16]
    conversation_id = ""
    cache_hits_before = 0
    cache_misses_before = 0
    if body:
        try:
            conversation_id = proxy._routing_conversation_id(dict(body), provider)
        except Exception:
            conversation_id = ""
    if conversation_id:
        try:
            lease = proxy._cache_router.lease_snapshot(conversation_id)
        except Exception:
            lease = None
        if lease is not None:
            cache_hits_before = int(lease.hits)
            cache_misses_before = int(lease.misses)

    context = _TrafficRequestContext(
        proxy=proxy,
        request_id=request_id,
        request_correlation=_correlation_digest(request_id),
        client=_classify_client(headers),
        provider=provider,
        path=path,
        headers=headers,
        requested_model=_bounded(requested_model, limit=128),
        original_context_tokens=(
            _estimate_context_tokens(provider, body, headers=headers, path=path)
            if body
            else 0
        ),
        conversation_id=conversation_id,
        cache_hits_before=cache_hits_before,
        cache_misses_before=cache_misses_before,
    )
    _ledger_for_proxy(proxy).register_request()
    token = _CURRENT_CONTEXT.set(context)
    response: Response | None = None
    try:
        response = await original(proxy, request)
        if not context.completed and not isinstance(response, StreamingResponse):
            if context.evidence_retained_source == "unavailable":
                (
                    context.evidence_retained_pct,
                    context.evidence_retained_source,
                    context.context_risk,
                ) = _coverage_snapshot(proxy)
            _complete_context(context, response)
        return response
    except Exception:
        _ledger_for_proxy(proxy).fail()
        raise
    finally:
        _CURRENT_CONTEXT.reset(token)


async def _traffic_forward_response(
    self: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Response:
    context = _CURRENT_CONTEXT.get()
    if context is not None:
        _capture_outbound_state(
            context,
            body,
            extra_headers=kwargs.get("extra_headers"),
            streaming=False,
        )
    response = await _ORIGINAL_FORWARD_RESPONSE(self, url, headers, body, *args, **kwargs)
    if context is not None:
        _complete_context(context, response)
    return response


async def _traffic_stream_response(
    self: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Response:
    context = _CURRENT_CONTEXT.get()
    if context is not None:
        _capture_outbound_state(
            context,
            body,
            extra_headers=kwargs.get("extra_headers"),
            streaming=True,
        )
    response = await _ORIGINAL_STREAM_RESPONSE(self, url, headers, body, *args, **kwargs)
    if context is None:
        return response
    if isinstance(response, StreamingResponse):
        response.body_iterator = _finalizing_iterator(
            response.body_iterator,
            context=context,
            response=response,
        )
    else:
        _complete_context(context, response)
    return response


_TRAFFIC_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly Traffic Receipt</title>
<style>
:root{color-scheme:dark;--bg:#07090d;--card:#0d1118;--line:#202938;--text:#edf2f7;--dim:#8b98aa;--good:#5ee6a8;--blue:#79b8ff;--warn:#ffd166;--bad:#ff7b8b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#172036 0,transparent 42%),var(--bg);font:14px/1.45 Inter,ui-sans-serif,system-ui;color:var(--text)}
main{max-width:900px;margin:0 auto;padding:32px 20px 64px}.top{display:flex;align-items:end;justify-content:space-between;margin-bottom:18px}.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--good);font-weight:800}.title{font-size:28px;font-weight:850;letter-spacing:-.03em}.live{font-size:12px;color:var(--dim)}
.card{background:linear-gradient(180deg,rgba(18,24,34,.96),rgba(11,15,22,.96));border:1px solid var(--line);border-radius:18px;box-shadow:0 24px 80px rgba(0,0,0,.35);overflow:hidden}.cardhead{display:flex;justify-content:space-between;align-items:center;padding:18px 22px;border-bottom:1px solid var(--line)}.client{font-size:17px;font-weight:800}.receipt{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim)}
.rows{padding:12px 22px 18px}.row{display:grid;grid-template-columns:minmax(220px,1fr) auto;gap:24px;align-items:center;padding:9px 0}.row.sep{border-top:1px solid var(--line);margin-top:8px;padding-top:16px}.label{color:var(--dim)}.value{font:700 14px ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right}.big{font-size:17px}.good{color:var(--good)}.blue{color:var(--blue)}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--dim)}
.reason{font-family:Inter,ui-sans-serif,system-ui;font-size:12px;max-width:430px}.foot{display:flex;justify-content:space-between;gap:20px;padding:13px 22px;border-top:1px solid var(--line);color:var(--dim);font-size:11px}.empty{padding:44px 22px;text-align:center;color:var(--dim)}
.note{margin-top:14px;padding:12px 15px;border:1px solid var(--line);border-radius:12px;color:var(--dim);font-size:12px}.note b{color:var(--text)}
@media(max-width:620px){.row{grid-template-columns:1fr}.value{text-align:left}.top{align-items:start;gap:10px;flex-direction:column}.foot{flex-direction:column}}
</style>
</head>
<body><main>
<div class="top"><div><div class="eyebrow">AI Traffic Assurance</div><div class="title">Traffic Receipt</div></div><div class="live" id="status">waiting for live traffic…</div></div>
<div class="card" id="card"><div class="empty">Send an AI request through Entroly to generate a receipt.</div></div>
<div class="note"><b>Truth policy:</b> monetary values appear only when provider-reported usage and an explicit pricing catalog are available. Net measured saving stays blank until a request-correlated measured counterfactual exists.</div>
</main>
<script>
const fmt=n=>Number(n||0).toLocaleString();
const usd=m=>m==null?'—':'$'+(Number(m)/1e6).toFixed(6).replace(/0+$/,'').replace(/\.$/,'');
const yn=v=>v===true?'YES':v===false?'NO':'—';
const hit=v=>v===true?'YES':v===false?'NO':'—';
const cls=(v,good)=>v===good?'good':v==='FAIL'||v==='ERROR'?'bad':'muted';
function render(r){
 if(!r){document.getElementById('card').innerHTML='<div class="empty">Send an AI request through Entroly to generate a receipt.</div>';return;}
 const evidence=r.evidence_retained_pct==null?'—':Number(r.evidence_retained_pct).toFixed(1)+'%';
 const decision=r.routing_decision||'—', verify=r.verification||'NOT_REPORTED';
 document.getElementById('card').innerHTML=`
 <div class="cardhead"><div class="client">${esc(r.client||'AI client')} request</div><div class="receipt">${esc(r.receipt_id||'')}</div></div>
 <div class="rows">
  <div class="row"><div class="label">Original context</div><div class="value big">${fmt(r.original_context_tokens)} tokens</div></div>
  <div class="row"><div class="label">Entroly context</div><div class="value big blue">${fmt(r.entroly_context_tokens)} tokens</div></div>
  <div class="row sep"><div class="label">Tokens avoided</div><div class="value big good">${fmt(r.tokens_avoided)}</div></div>
  <div class="row sep"><div class="label">Evidence retained <span class="muted">(${esc(r.evidence_retained_source||'unavailable')})</span></div><div class="value">${evidence}</div></div>
  <div class="row"><div class="label">Recoverable</div><div class="value ${r.recoverable?'good':'muted'}">${yn(r.recoverable)}</div></div>
  <div class="row"><div class="label">Warm prefix protected</div><div class="value">${fmt(r.warm_prefix_protected_tokens)} tokens</div></div>
  <div class="row"><div class="label">Cache hit</div><div class="value ${r.cache_hit===true?'good':'muted'}">${hit(r.cache_hit)}</div></div>
  <div class="row sep"><div class="label">Requested model</div><div class="value">${esc(r.requested_model||'—')}</div></div>
  <div class="row"><div class="label">Executed model</div><div class="value">${esc(r.executed_model||'—')}</div></div>
  <div class="row"><div class="label">Routing decision</div><div class="value ${decision==='STAY'?'good':'blue'}">${esc(decision)}</div></div>
  <div class="row"><div class="label">Reason</div><div class="value reason">${esc(r.routing_reason||'—')}</div></div>
  <div class="row sep"><div class="label">Input cost</div><div class="value">${usd(r.input_cost_micro_usd)}</div></div>
  <div class="row"><div class="label">Cache benefit</div><div class="value good">${usd(r.cache_benefit_micro_usd)}</div></div>
  <div class="row"><div class="label">Net measured saving</div><div class="value">${usd(r.net_measured_saving_micro_usd)}</div></div>
  <div class="row sep"><div class="label">Context risk</div><div class="value ${r.context_risk==='LOW'?'good':r.context_risk==='HIGH'?'bad':'warn'}">${esc(r.context_risk||'UNKNOWN')}</div></div>
  <div class="row"><div class="label">Verification</div><div class="value ${cls(verify,'PASS')}">${esc(verify)}</div></div>
  <div class="row"><div class="label">Traffic Receipt</div><div class="value good">✓ VERIFIED</div></div>
 </div>
 <div class="foot"><span>${esc(r.provider||'unknown')} · ${r.streaming?'streaming':'buffered'} · ${Number(r.latency_ms||0).toFixed(1)} ms</span><span>digest ${esc((r.receipt_digest||'').slice(0,16))}…</span></div>`;
 document.getElementById('status').textContent='live · '+new Date((r.observed_at||0)*1000).toLocaleTimeString();
}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function refresh(){try{const x=await fetch('/traffic-receipts?limit=1',{cache:'no-store'});if(!x.ok)throw Error(x.status);const d=await x.json();render(d.latest);}catch(e){document.getElementById('status').textContent='receipt API unavailable';}}
refresh();setInterval(refresh,2000);
</script></body></html>"""


async def _traffic_receipts_endpoint(request: Request) -> Response:
    proxy = request.app.state.proxy
    raw_limit = request.query_params.get("limit", "20")
    try:
        limit = max(1, min(int(raw_limit), _MAX_RECORDS_LIMIT))
    except (TypeError, ValueError, OverflowError):
        limit = 20
    return JSONResponse(
        _ledger_for_proxy(proxy).snapshot(limit=limit),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def _traffic_page_endpoint(_request: Request) -> Response:
    return Response(
        _TRAFFIC_HTML,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; "
                "img-src 'none'; frame-ancestors 'none'; base-uri 'none'"
            ),
        },
    )


def _install_route(app: Any) -> None:
    routes = getattr(getattr(app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return
    existing = {getattr(route, "path", None) for route in routes}
    page = _proxy._sidecar_guard(_traffic_page_endpoint)
    api = _proxy._sidecar_guard(_traffic_receipts_endpoint)
    if "/traffic" not in existing:
        routes.insert(
            0,
            Route("/traffic", endpoint=page, methods=["GET"], name="traffic-receipt-page"),
        )
    if "/traffic-receipts" not in existing:
        routes.insert(
            0,
            Route(
                "/traffic-receipts",
                endpoint=api,
                methods=["GET"],
                name="traffic-receipts",
            ),
        )


def install_traffic_receipts() -> None:
    """Install request/forwarding observers without changing execution policy."""
    global _ORIGINAL_FORWARD_RESPONSE, _ORIGINAL_STREAM_RESPONSE

    current_core = _transport._ORIGINAL_HANDLE_PROXY
    if not hasattr(current_core, "__entroly_traffic_receipt_original__"):
        original_core = current_core

        async def traffic_core_handle(self: Any, request: Request) -> Response:
            return await _run_traffic_handle_proxy(self, request, original_core)

        traffic_core_handle.__entroly_traffic_receipt_original__ = original_core
        # Forward boundary-contract markers from all prior wrappers so each
        # module's install test remains introspectable after this one runs.
        for _marker in (
            "__entroly_gateway_shadow_original__",
            "__entroly_routing_authority_original__",
        ):
            _val = getattr(original_core, _marker, None)
            if _val is not None:
                setattr(traffic_core_handle, _marker, _val)
        _transport._ORIGINAL_HANDLE_PROXY = traffic_core_handle

    current_forward = _proxy.PromptCompilerProxy._forward_response
    if not hasattr(current_forward, "__entroly_traffic_receipt_original__"):
        _ORIGINAL_FORWARD_RESPONSE = current_forward
        _traffic_forward_response.__entroly_traffic_receipt_original__ = current_forward
        _proxy.PromptCompilerProxy._forward_response = _traffic_forward_response

    # For _stream_response: proxy_transport_final installs _bounded_stream_response
    # as the class method and calls its own _ORIGINAL_STREAM_RESPONSE module global
    # at invocation time. Inject traffic observability into that global so the
    # bounded wrapper stays as the class method (preserving the stream-bounds
    # contract) while still routing through the traffic receipt layer.
    from . import proxy_transport_final as _transport_final

    current_stream_inner = _transport_final._ORIGINAL_STREAM_RESPONSE
    if not hasattr(current_stream_inner, "__entroly_traffic_receipt_original__"):
        _ORIGINAL_STREAM_RESPONSE = current_stream_inner
        _traffic_stream_response.__entroly_traffic_receipt_original__ = current_stream_inner
        _transport_final._ORIGINAL_STREAM_RESPONSE = _traffic_stream_response


_current_forward = _proxy.PromptCompilerProxy._forward_response
_current_stream = _proxy.PromptCompilerProxy._stream_response
_ORIGINAL_FORWARD_RESPONSE: Callable[..., Awaitable[Response]] = getattr(
    _current_forward,
    "__entroly_traffic_receipt_original__",
    _current_forward,
)
_ORIGINAL_STREAM_RESPONSE: Callable[..., Awaitable[Response]] = getattr(
    _current_stream,
    "__entroly_traffic_receipt_original__",
    _current_stream,
)

install_traffic_receipts()


__all__ = [
    "TrafficReceipt",
    "TrafficReceiptLedger",
    "_TRAFFIC_HTML",
    "_canonical_json",
    "_classify_client",
    "_install_route",
    "_prefix_protection",
    "_routing_decision",
    "_verification",
    "install_traffic_receipts",
]
