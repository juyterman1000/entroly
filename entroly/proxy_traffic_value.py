"""Executive AI Traffic Value dashboard for Entroly.

This module turns Traffic Receipts into durable executive rollups inside the
existing ValueTracker file. It does not create a second accounting database.

The surface is deliberately evidence-classified:
- token reduction and context reduction come from locally observed pre/post
  request estimates;
- estimated value avoided comes from observed avoided tokens and an explicit
  model price in Entroly's pricing table;
- cache benefit and provider input spend are shown only when the Traffic Receipt
  has provider usage plus auditable pricing;
- verification/recovery percentages are derived from request-level evidence;
- lifetime totals keep accumulating across proxy restarts.

Rolling windows are Today / 7D / 30D / 60D / 90D. The selected default adapts
to how long Traffic Value has actually been collecting data, while All Time is
always visible beside the selected period.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import proxy as _proxy
from . import proxy_traffic_receipt as _traffic_receipt
from .value_tracker import (
    _has_priced_model,
    estimate_cost,
    get_tracker,
    pricing_provenance,
)

_SCHEMA_VERSION = "entroly.traffic-value.v2"
_TRAFFIC_KEY = "traffic_assurance"
_MAX_DAILY = 90
_MAX_SEEN_RECEIPTS = 2048

_COUNTER_FIELDS = (
    "requests_observed",
    "requests_optimized",
    "tokens_received",
    "tokens_sent",
    "tokens_avoided",
    "verified_requests",
    "verification_passed",
    "recovery_invoked",
    "recovery_succeeded",
    "warm_cache_protected_tokens",
    "cache_observed",
    "cache_hits",
    "provider_priced_requests",
)
_MONEY_FIELDS = (
    "estimated_value_avoided_usd",
    "measured_cache_benefit_usd",
    "provider_input_spend_usd",
)


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _nonnegative_float(value: object) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if parsed < 0.0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return 0.0
    return parsed


def _empty_metrics() -> dict[str, int | float]:
    return {
        **{field: 0 for field in _COUNTER_FIELDS},
        **{field: 0.0 for field in _MONEY_FIELDS},
    }


def _ensure_metrics(row: dict[str, Any]) -> dict[str, Any]:
    defaults = _empty_metrics()
    for key, value in defaults.items():
        row.setdefault(key, value)
    return row


def _accumulate(target: dict[str, Any], row: Mapping[str, Any]) -> None:
    for field in _COUNTER_FIELDS:
        target[field] = _nonnegative_int(target.get(field)) + _nonnegative_int(
            row.get(field)
        )
    for field in _MONEY_FIELDS:
        target[field] = round(
            _nonnegative_float(target.get(field))
            + _nonnegative_float(row.get(field)),
            6,
        )


def _finalize_metrics(row: dict[str, Any]) -> dict[str, Any]:
    received = _nonnegative_int(row.get("tokens_received"))
    avoided = _nonnegative_int(row.get("tokens_avoided"))
    requests = _nonnegative_int(row.get("requests_observed"))
    verified = _nonnegative_int(row.get("verified_requests"))
    passed = _nonnegative_int(row.get("verification_passed"))
    recovery_invoked = _nonnegative_int(row.get("recovery_invoked"))
    recovery_succeeded = _nonnegative_int(row.get("recovery_succeeded"))
    cache_observed = _nonnegative_int(row.get("cache_observed"))
    cache_hits = _nonnegative_int(row.get("cache_hits"))

    row["context_reduction_pct"] = round(100.0 * avoided / max(1, received), 2)
    row["requests_verified_pct"] = round(100.0 * verified / max(1, requests), 2)
    row["verification_pass_pct"] = (
        round(100.0 * passed / verified, 2) if verified else None
    )
    row["recovery_invoked_pct"] = round(
        100.0 * recovery_invoked / max(1, requests), 2
    )
    row["recovery_succeeded_pct"] = (
        round(100.0 * recovery_succeeded / recovery_invoked, 2)
        if recovery_invoked
        else None
    )
    row["cache_hit_request_pct"] = (
        round(100.0 * cache_hits / cache_observed, 2)
        if cache_observed
        else None
    )
    row["total_ai_value_protected_usd"] = round(
        _nonnegative_float(row.get("estimated_value_avoided_usd"))
        + _nonnegative_float(row.get("measured_cache_benefit_usd")),
        6,
    )
    return row


def _receipt_day(receipt: Any) -> str:
    ts = _nonnegative_float(getattr(receipt, "observed_at", 0.0)) or time.time()
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).date().isoformat()


def _estimated_value_for_receipt(receipt: Any) -> float:
    model = str(getattr(receipt, "executed_model", "") or "")
    tokens = _nonnegative_int(getattr(receipt, "tokens_avoided", 0))
    if not tokens or not _has_priced_model(model):
        return 0.0
    return round(max(0.0, float(estimate_cost(tokens, model, kind="input"))), 6)


def _receipt_delta(receipt: Any) -> dict[str, int | float]:
    original = _nonnegative_int(getattr(receipt, "original_context_tokens", 0))
    sent = _nonnegative_int(getattr(receipt, "entroly_context_tokens", 0))
    avoided = _nonnegative_int(getattr(receipt, "tokens_avoided", 0))
    verification = str(getattr(receipt, "verification", "") or "").upper()
    recovery_receipts = _nonnegative_int(getattr(receipt, "recovery_receipts", 0))
    recovery_invoked = recovery_receipts > 0
    # A recovery is counted successful only when the final surfaced response has
    # explicit PASS evidence. This avoids equating "attempted" with "succeeded".
    recovery_succeeded = recovery_invoked and verification == "PASS"
    cache_hit = getattr(receipt, "cache_hit", None)
    input_cost_micro = getattr(receipt, "input_cost_micro_usd", None)
    cache_benefit_micro = getattr(receipt, "cache_benefit_micro_usd", None)

    return {
        "requests_observed": 1,
        "requests_optimized": int(avoided > 0),
        "tokens_received": original,
        "tokens_sent": sent,
        "tokens_avoided": avoided,
        "estimated_value_avoided_usd": _estimated_value_for_receipt(receipt),
        "measured_cache_benefit_usd": (
            round(_nonnegative_int(cache_benefit_micro) / 1_000_000.0, 6)
            if cache_benefit_micro is not None
            else 0.0
        ),
        # Current Traffic Receipt v1 prices provider input/cache categories only.
        # The UI labels this precisely as provider input spend, not full invoice.
        "provider_input_spend_usd": (
            round(_nonnegative_int(input_cost_micro) / 1_000_000.0, 6)
            if input_cost_micro is not None
            else 0.0
        ),
        "provider_priced_requests": int(input_cost_micro is not None),
        "verified_requests": int(verification in {"PASS", "FAIL"}),
        "verification_passed": int(verification == "PASS"),
        "recovery_invoked": int(recovery_invoked),
        "recovery_succeeded": int(recovery_succeeded),
        "warm_cache_protected_tokens": _nonnegative_int(
            getattr(receipt, "warm_prefix_protected_tokens", 0)
        ),
        "cache_observed": int(cache_hit is not None),
        "cache_hits": int(cache_hit is True),
    }


def record_traffic_value_receipt(
    receipt: Any,
    *,
    tracker: Any | None = None,
) -> bool:
    """Persist one Traffic Receipt into the existing ValueTracker data file.

    The bounded ``seen_receipts`` list makes recent retries/idempotent replays
    harmless without retaining request content or an unbounded identifier set.
    """

    if not getattr(receipt, "verify", lambda: False)():
        return False
    value_tracker = tracker or get_tracker()
    receipt_id = str(getattr(receipt, "receipt_id", "") or "")
    if not receipt_id:
        return False
    day = _receipt_day(receipt)
    delta = _receipt_delta(receipt)
    observed_at = _nonnegative_float(getattr(receipt, "observed_at", 0.0)) or time.time()

    try:
        with value_tracker._mutation():  # same process+file lock as ValueTracker
            root = value_tracker._data.setdefault(
                _TRAFFIC_KEY,
                {
                    "started_at": observed_at,
                    "lifetime": _empty_metrics(),
                    "daily": {},
                    "seen_receipts": [],
                },
            )
            started = _nonnegative_float(root.get("started_at"))
            if not started or observed_at < started:
                root["started_at"] = observed_at

            seen = [str(x) for x in root.get("seen_receipts", []) if x]
            if receipt_id in seen:
                return False
            seen.append(receipt_id)
            root["seen_receipts"] = seen[-_MAX_SEEN_RECEIPTS:]

            lifetime = _ensure_metrics(root.setdefault("lifetime", {}))
            daily = root.setdefault("daily", {})
            day_row = _ensure_metrics(daily.setdefault(day, {}))
            _accumulate(lifetime, delta)
            _accumulate(day_row, delta)

            # Keep exactly the rolling-history horizon needed by this view.
            for old_day in sorted(daily)[:-_MAX_DAILY]:
                daily.pop(old_day, None)

            value_tracker._save()
        return True
    except Exception as exc:
        _proxy.logger.debug("Traffic value persistence skipped: %s", exc, exc_info=True)
        return False


def _traffic_state(tracker: Any) -> Mapping[str, Any]:
    data = getattr(tracker, "_data", {})
    state = data.get(_TRAFFIC_KEY, {}) if isinstance(data, Mapping) else {}
    return state if isinstance(state, Mapping) else {}


def _rolling_window(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    days: int,
    key: str,
    label: str,
    today: _dt.date,
) -> dict[str, Any]:
    result: dict[str, Any] = {"key": key, "label": label, **_empty_metrics()}
    cutoff = today - _dt.timedelta(days=max(1, days) - 1)
    for raw_day, row in rows.items():
        try:
            day = _dt.date.fromisoformat(str(raw_day))
        except (TypeError, ValueError):
            continue
        if cutoff <= day <= today and isinstance(row, Mapping):
            _accumulate(result, row)
    result["window_start"] = cutoff.isoformat()
    result["window_end"] = today.isoformat()
    return _finalize_metrics(result)


def _lifetime_rollup(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "key": "lifetime",
        "label": "All time",
        **_empty_metrics(),
    }
    lifetime = state.get("lifetime", {})
    if isinstance(lifetime, Mapping):
        _accumulate(result, lifetime)
    return _finalize_metrics(result)


def _default_window(age_days: int) -> str:
    # Product rule:
    # 3 days installed  -> 7D
    # 45 days installed -> 30D
    # 60-99 days        -> 60D
    # 100+ days         -> 90D
    if age_days < 30:
        return "7d"
    if age_days < 60:
        return "30d"
    if age_days < 100:
        return "60d"
    return "90d"


def build_traffic_value_snapshot(
    tracker: Any | None = None,
    *,
    today: _dt.date | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    value_tracker = tracker or get_tracker()
    try:
        value_tracker.reload_if_changed()
    except Exception:
        pass

    now_ts = time.time() if now is None else float(now)
    current_day = today or _dt.datetime.fromtimestamp(
        now_ts, tz=_dt.timezone.utc
    ).date()
    state = _traffic_state(value_tracker)
    daily = state.get("daily", {})
    if not isinstance(daily, Mapping):
        daily = {}

    windows = [
        _rolling_window(daily, days=1, key="today", label="Today", today=current_day),
        _rolling_window(daily, days=7, key="7d", label="7 days", today=current_day),
        _rolling_window(daily, days=30, key="30d", label="30 days", today=current_day),
        _rolling_window(daily, days=60, key="60d", label="60 days", today=current_day),
        _rolling_window(daily, days=90, key="90d", label="90 days", today=current_day),
        _lifetime_rollup(state),
    ]

    started_at = _nonnegative_float(state.get("started_at"))
    age_days = (
        max(0, int((now_ts - started_at) // 86400))
        if started_at
        else 0
    )
    default_window = _default_window(age_days)
    pricing = pricing_provenance()

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_unix": round(now_ts, 3),
        "collection_started_at": started_at or None,
        "installed_days": age_days,
        "default_window": default_window,
        "always_show_lifetime": True,
        "windows": {window["key"]: window for window in windows},
        "window_order": [window["key"] for window in windows],
        "pricing": pricing,
        "truth": {
            "tokens_received_sent": (
                "Local deterministic pre/post context estimates for provider-bound "
                "requests observed by Traffic Receipts."
            ),
            "estimated_value": (
                "Observed avoided input tokens multiplied by the configured model "
                "input rate. It is modeled value avoided, not invoice savings."
            ),
            "measured_cache_benefit": (
                "Shown only when provider usage and auditable pricing were present "
                "on the Traffic Receipt."
            ),
            "provider_input_spend": (
                "Provider-reported input/cache token categories priced with the "
                "auditable catalog. It is not labeled full provider invoice spend."
            ),
            "verification": (
                "Requests verified counts only explicit PASS/FAIL evidence. "
                "NOT_REPORTED and ERROR do not inflate the percentage."
            ),
            "recovery": (
                "Recovery invoked requires recovery evidence; recovery succeeded "
                "requires final explicit PASS verification."
            ),
            "total_ai_value": (
                "Estimated value on tokens not sent plus measured cache benefit on "
                "tokens that were sent. These are displayed as separate evidence "
                "classes and summed as Total AI value protected."
            ),
            "lifetime": (
                "All Time persists in the existing ValueTracker file across proxy "
                "restarts and accumulates until telemetry is reset/deleted."
            ),
        },
    }


_TRAFFIC_VALUE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly AI Traffic Value</title>
<style>
:root{color-scheme:dark;--bg:#07090d;--card:#0d1118;--line:#202938;--text:#edf2f7;--dim:#8b98aa;--good:#5ee6a8;--blue:#79b8ff;--warn:#ffd166;--bad:#ff7b8b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#172036 0,transparent 42%),var(--bg);font:14px/1.45 Inter,ui-sans-serif,system-ui;color:var(--text)}
main{max-width:1080px;margin:0 auto;padding:32px 20px 64px}.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--good);font-weight:800}.title{font-size:32px;font-weight:900;letter-spacing:-.04em;margin-top:4px}.sub{color:var(--dim);margin-top:5px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 16px}.tab{border:1px solid var(--line);background:#0b1018;color:var(--dim);padding:8px 12px;border-radius:999px;cursor:pointer;font:750 12px Inter,system-ui}.tab.active{color:var(--text);border-color:#3b506d;background:#152033}
.card{background:linear-gradient(180deg,rgba(18,24,34,.97),rgba(11,15,22,.97));border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 24px 80px rgba(0,0,0,.32)}.period{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.12em;font-weight:850}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.metric{border:1px solid var(--line);border-radius:14px;padding:15px;background:rgba(8,12,18,.58)}.mval{font:800 21px ui-monospace,SFMono-Regular,Menlo,monospace}.mlabel{color:var(--dim);font-size:12px;margin-top:5px}
.moneygrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.moneybox{border:1px solid var(--line);border-radius:15px;padding:17px;background:rgba(8,12,18,.7)}.money{font-size:30px;font-weight:900;letter-spacing:-.04em}.moneylabel{color:var(--dim);font-size:12px;margin-top:4px}
.alltime{margin-top:18px;border-color:#30435e}.alltitle{display:flex;justify-content:space-between;gap:20px;align-items:center}.badge{font-size:11px;color:var(--dim);border:1px solid var(--line);padding:5px 9px;border-radius:999px}
.good{color:var(--good)}.blue{color:var(--blue)}.warn{color:var(--warn)}.bad{color:var(--bad)}.dim{color:var(--dim)}
.note{margin-top:14px;padding:13px 15px;border:1px solid var(--line);border-radius:12px;color:var(--dim);font-size:12px}.note b{color:var(--text)}.meta{display:flex;justify-content:space-between;gap:18px;margin-top:14px;color:var(--dim);font-size:11px}
@media(max-width:760px){.grid,.moneygrid{grid-template-columns:1fr}.alltitle,.meta{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body><main>
<div class="eyebrow">AI Traffic Assurance</div>
<div class="title">AI Traffic Value</div>
<div class="sub">The value Entroly has created and protected across your AI traffic.</div>
<div class="tabs" id="tabs"></div>
<div class="card" id="period"><div class="period">Loading live value…</div></div>
<div class="card alltime" id="lifetime"><div class="period">Loading All Time…</div></div>
<div class="note"><b>Evidence policy:</b> avoided-token dollars are estimates from observed token reduction and configured model prices. Cache benefit is measured only when provider usage is available. Provider input spend is not presented as a full invoice.</div>
</main>
<script>
const fmt=n=>Number(n||0).toLocaleString();
const compact=n=>{n=Number(n||0);if(Math.abs(n)>=1e9)return(n/1e9).toFixed(n>=1e10?1:2).replace(/\.0+$/,'')+'B';if(Math.abs(n)>=1e6)return(n/1e6).toFixed(n>=1e7?1:2).replace(/\.0+$/,'')+'M';if(Math.abs(n)>=1e3)return(n/1e3).toFixed(n>=1e4?1:2).replace(/\.0+$/,'')+'K';return fmt(n)};
const usd=v=>'$'+Number(v||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const pct=v=>v==null?'—':Number(v).toFixed(1)+'%';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let DATA=null,ACTIVE=null,USER_SELECTED=false;
function moneyBox(value,label,cls='good'){return `<div class="moneybox"><div class="money ${cls}">${usd(value)}</div><div class="moneylabel">${esc(label)}</div></div>`}
function metric(value,label,cls=''){return `<div class="metric"><div class="mval ${cls}">${value}</div><div class="mlabel">${esc(label)}</div></div>`}
function renderWindow(w){
 const measured=w.provider_priced_requests>0;
 document.getElementById('period').innerHTML=`
 <div class="period">AI TRAFFIC VALUE — ${esc(w.label).toUpperCase()}</div>
 <div class="grid">
  ${metric(fmt(w.requests_optimized),'Requests optimized','blue')}
  ${metric(compact(w.tokens_received)+' tokens','Tokens received')}
  ${metric(compact(w.tokens_sent)+' tokens','Tokens sent')}
  ${metric(compact(w.tokens_avoided)+' tokens','Tokens avoided','good')}
  ${metric(pct(w.context_reduction_pct),'Context reduction','good')}
  ${metric(compact(w.warm_cache_protected_tokens)+' tokens','Warm cache protected','blue')}
 </div>
 <div class="moneygrid">
  ${moneyBox(w.estimated_value_avoided_usd,'Estimated value avoided')}
  ${moneyBox(w.measured_cache_benefit_usd,'Measured cache benefit',measured?'good':'dim')}
  ${moneyBox(w.provider_input_spend_usd,'Provider input spend',measured?'':'dim')}
 </div>
 <div class="grid">
  ${metric(pct(w.requests_verified_pct),'Requests verified')}
  ${metric(pct(w.recovery_invoked_pct),'Recovery invoked')}
  ${metric(pct(w.recovery_succeeded_pct),'Recovery succeeded')}
  ${metric(pct(w.verification_pass_pct),'Verification pass rate')}
  ${metric(pct(w.cache_hit_request_pct),'Cache-hit requests')}
  ${metric(usd(w.total_ai_value_protected_usd),'Total AI value protected','good')}
 </div>
 <div class="meta"><span>${w.window_start?esc(w.window_start)+' → '+esc(w.window_end):''}</span><span>${measured?'provider-priced traffic observed':'measured provider $ appears when usage + pricing exist'}</span></div>`;
}
function renderLifetime(w){
 document.getElementById('lifetime').innerHTML=`
 <div class="alltitle"><div><div class="period">ALL TIME</div><div class="dim">Keeps accumulating across proxy restarts.</div></div><div class="badge">${fmt(DATA.installed_days||0)} days collecting Traffic Value</div></div>
 <div class="moneygrid">
  ${moneyBox(w.estimated_value_avoided_usd,'Estimated value avoided')}
  ${moneyBox(w.measured_cache_benefit_usd,'Measured cache benefit','blue')}
  ${moneyBox(w.total_ai_value_protected_usd,'Total AI value protected','good')}
 </div>
 <div class="grid">
  ${metric(compact(w.tokens_avoided),'Tokens avoided')}
  ${metric(fmt(w.requests_optimized),'Requests optimized')}
  ${metric(compact(w.warm_cache_protected_tokens),'Warm-cache tokens protected')}
 </div>`;
}
function render(){
 if(!DATA)return;
 if(!ACTIVE)ACTIVE=DATA.default_window||'7d';
 document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.key===ACTIVE));
 renderWindow(DATA.windows[ACTIVE]||DATA.windows['7d']);
 renderLifetime(DATA.windows.lifetime);
}
async function load(){
 try{
  const r=await fetch('/traffic-value.json',{cache:'no-store'});if(!r.ok)throw Error(r.status);
  const next=await r.json();DATA=next;
  if(!USER_SELECTED)ACTIVE=DATA.default_window||'7d';
  const tabs=document.getElementById('tabs');tabs.innerHTML='';
  for(const key of DATA.window_order||[]){
   if(key==='lifetime')continue;
   const b=document.createElement('button');b.className='tab';b.dataset.key=key;b.textContent=DATA.windows[key].label;
   b.onclick=()=>{USER_SELECTED=true;ACTIVE=key;render()};tabs.appendChild(b);
  }
  render();
 }catch(e){
  document.getElementById('period').innerHTML='<div class="period">Traffic Value API unavailable</div>';
 }
}
load();setInterval(load,10000);
</script></body></html>"""


async def _traffic_value_json(_request: Request) -> Response:
    return JSONResponse(
        build_traffic_value_snapshot(),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def _traffic_value_page(_request: Request) -> Response:
    return Response(
        _TRAFFIC_VALUE_HTML,
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
    page = _proxy._sidecar_guard(_traffic_value_page)
    api = _proxy._sidecar_guard(_traffic_value_json)
    if "/traffic-value" not in existing:
        routes.insert(
            0,
            Route(
                "/traffic-value",
                endpoint=page,
                methods=["GET"],
                name="traffic-value-page",
            ),
        )
    if "/traffic-value.json" not in existing:
        routes.insert(
            0,
            Route(
                "/traffic-value.json",
                endpoint=api,
                methods=["GET"],
                name="traffic-value-json",
            ),
        )


def _install_receipt_bridge() -> None:
    """Persist every admitted Traffic Receipt into the existing ValueTracker."""
    current = _traffic_receipt.TrafficReceiptLedger.append
    if hasattr(current, "__entroly_traffic_value_original__"):
        return
    original = current

    def append_with_value(self: Any, receipt: Any) -> None:
        original(self, receipt)
        record_traffic_value_receipt(receipt)

    append_with_value.__entroly_traffic_value_original__ = original
    _traffic_receipt.TrafficReceiptLedger.append = append_with_value


_install_receipt_bridge()


__all__ = [
    "_TRAFFIC_VALUE_HTML",
    "_default_window",
    "_install_route",
    "build_traffic_value_snapshot",
    "record_traffic_value_receipt",
]
