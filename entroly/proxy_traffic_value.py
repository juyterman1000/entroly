"""Rolling executive value dashboard for Entroly AI traffic.

This module exposes durable, evidence-classified value accumulated by the
existing :mod:`entroly.value_tracker`. It deliberately does not create a second
accounting system. Provider-bound token reductions are shown with modeled input
value using Entroly's auditable pricing provenance; local-only reductions remain
token-only.

The dashboard uses rolling windows (today, 7d, 30d, 60d, 90d) plus lifetime so
value compounds visibly without resetting at arbitrary calendar boundaries.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import proxy as _proxy
from .value_tracker import get_tracker, pricing_provenance

_SCHEMA_VERSION = "entroly.traffic-value.v1"


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


def _empty_rollup(*, key: str, label: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "provider_tokens_avoided": 0,
        "estimated_input_value_avoided_usd": 0.0,
        "provider_requests_observed": 0,
        "provider_requests_optimized": 0,
        "provider_unpriced_tokens": 0,
        "provider_unpriced_requests": 0,
        "local_tokens_reduced": 0,
        "local_operations": 0,
    }


def _accumulate(target: dict[str, Any], row: Mapping[str, Any]) -> None:
    target["provider_tokens_avoided"] += _nonnegative_int(
        row.get("provider_tokens_saved")
    )
    target["estimated_input_value_avoided_usd"] = round(
        float(target["estimated_input_value_avoided_usd"])
        + _nonnegative_float(row.get("provider_cost_avoided_usd")),
        6,
    )
    target["provider_requests_observed"] += _nonnegative_int(
        row.get("provider_requests")
    )
    target["provider_requests_optimized"] += _nonnegative_int(
        row.get("provider_requests_optimized")
    )
    target["provider_unpriced_tokens"] += _nonnegative_int(
        row.get("provider_unpriced_tokens")
    )
    target["provider_unpriced_requests"] += _nonnegative_int(
        row.get("provider_unpriced_requests")
    )
    target["local_tokens_reduced"] += _nonnegative_int(
        row.get("local_tokens_reduced")
    )
    target["local_operations"] += _nonnegative_int(row.get("local_operations"))


def _rolling_window(
    rows: list[Mapping[str, Any]],
    *,
    days: int,
    key: str,
    label: str,
    today: _dt.date,
) -> dict[str, Any]:
    result = _empty_rollup(key=key, label=label)
    cutoff = today - _dt.timedelta(days=max(1, days) - 1)
    for row in rows:
        try:
            day = _dt.date.fromisoformat(str(row.get("date") or ""))
        except (TypeError, ValueError):
            continue
        if cutoff <= day <= today:
            _accumulate(result, row)
    result["window_start"] = cutoff.isoformat()
    result["window_end"] = today.isoformat()
    return result


def _lifetime_rollup(lifetime: Mapping[str, Any]) -> dict[str, Any]:
    result = _empty_rollup(key="lifetime", label="All time")
    result.update(
        {
            "provider_tokens_avoided": _nonnegative_int(
                lifetime.get("provider_tokens_saved")
            ),
            "estimated_input_value_avoided_usd": round(
                _nonnegative_float(lifetime.get("provider_cost_avoided_usd")), 6
            ),
            "provider_requests_observed": _nonnegative_int(
                lifetime.get("provider_requests")
            ),
            "provider_requests_optimized": _nonnegative_int(
                lifetime.get("provider_requests_optimized")
            ),
            "provider_unpriced_tokens": _nonnegative_int(
                lifetime.get("provider_unpriced_tokens")
            ),
            "provider_unpriced_requests": _nonnegative_int(
                lifetime.get("provider_unpriced_requests")
            ),
            "local_tokens_reduced": _nonnegative_int(
                lifetime.get("local_tokens_reduced")
            ),
            "local_operations": _nonnegative_int(
                lifetime.get("local_operations")
            ),
        }
    )
    return result


def build_traffic_value_snapshot(
    tracker: Any | None = None,
    *,
    today: _dt.date | None = None,
) -> dict[str, Any]:
    """Build rolling provider-value windows from the durable ValueTracker.

    ``estimated_input_value_avoided_usd`` is intentionally evidence-labelled:
    it is observed provider-bound token reduction multiplied by the configured
    input rate, not an invoice claim. Local SDK/MCP/npm reductions never enter
    that dollar number.
    """

    value_tracker = tracker or get_tracker()
    try:
        value_tracker.reload_if_changed()
    except Exception:
        pass
    current_day = today or _dt.datetime.now(_dt.timezone.utc).date()
    daily = list(value_tracker.get_daily(90))
    lifetime = value_tracker.get_lifetime()
    windows = [
        _rolling_window(daily, days=1, key="today", label="Today", today=current_day),
        _rolling_window(daily, days=7, key="7d", label="7 days", today=current_day),
        _rolling_window(daily, days=30, key="30d", label="30 days", today=current_day),
        _rolling_window(daily, days=60, key="60d", label="60 days", today=current_day),
        _rolling_window(daily, days=90, key="90d", label="90 days", today=current_day),
        _lifetime_rollup(lifetime),
    ]
    pricing = pricing_provenance()
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_unix": round(time.time(), 3),
        "default_window": "lifetime",
        "windows": {window["key"]: window for window in windows},
        "window_order": [window["key"] for window in windows],
        "pricing": pricing,
        "truth": {
            "provider_tokens": (
                "Observed provider-bound token reduction recorded by Entroly."
            ),
            "estimated_usd": (
                "Observed provider-bound token reduction multiplied by the "
                "configured model input rate. This is modeled value avoided, "
                "not a provider invoice or counterfactual billing measurement."
            ),
            "local_tokens": (
                "Local SDK/MCP/npm token reductions are shown separately and "
                "never converted to dollars without provider-bound evidence."
            ),
            "lifetime": (
                "Lifetime counters persist across proxy restarts and keep "
                "accumulating until the user deletes or resets local telemetry."
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
:root{color-scheme:dark;--bg:#07090d;--card:#0d1118;--line:#202938;--text:#edf2f7;--dim:#8b98aa;--good:#5ee6a8;--blue:#79b8ff;--warn:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#172036 0,transparent 42%),var(--bg);font:14px/1.45 Inter,ui-sans-serif,system-ui;color:var(--text)}
main{max-width:980px;margin:0 auto;padding:32px 20px 64px}.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--good);font-weight:800}.title{font-size:30px;font-weight:850;letter-spacing:-.035em;margin-top:4px}.sub{color:var(--dim);margin-top:4px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 16px}.tab{border:1px solid var(--line);background:#0b1018;color:var(--dim);padding:8px 12px;border-radius:999px;cursor:pointer;font:700 12px Inter,system-ui}.tab.active{color:var(--text);border-color:#3b506d;background:#152033}
.hero{background:linear-gradient(180deg,rgba(18,24,34,.97),rgba(11,15,22,.97));border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 24px 80px rgba(0,0,0,.35)}.period{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.12em;font-weight:800}.money{font-size:52px;font-weight:900;letter-spacing:-.055em;margin-top:8px;color:var(--good)}.moneylabel{font-size:13px;color:var(--dim)}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}.metric{border:1px solid var(--line);border-radius:14px;padding:16px;background:rgba(8,12,18,.6)}.mval{font:800 22px ui-monospace,SFMono-Regular,Menlo,monospace}.mlabel{color:var(--dim);font-size:12px;margin-top:5px}.blue{color:var(--blue)}.good{color:var(--good)}.warn{color:var(--warn)}
.note{margin-top:14px;padding:13px 15px;border:1px solid var(--line);border-radius:12px;color:var(--dim);font-size:12px}.note b{color:var(--text)}.meta{display:flex;justify-content:space-between;gap:18px;margin-top:14px;color:var(--dim);font-size:11px}
@media(max-width:720px){.grid{grid-template-columns:1fr}.money{font-size:42px}.meta{flex-direction:column}}
</style>
</head>
<body><main>
<div class="eyebrow">AI Traffic Assurance</div><div class="title">Value that keeps adding up</div><div class="sub">Provider-bound token reduction and its evidence-labelled economic value.</div>
<div class="tabs" id="tabs"></div>
<div class="hero" id="hero"><div class="period">Loading…</div></div>
<div class="note"><b>Truth policy:</b> the dollar hero is estimated input value avoided from observed provider-bound token reduction and configured model pricing. It is not presented as invoice-verified savings. Local-only token reduction is kept separate.</div>
</main>
<script>
const fmt=n=>Number(n||0).toLocaleString();
const usd=v=>'$'+Number(v||0).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let DATA=null,ACTIVE='lifetime';
function render(){if(!DATA)return;const w=DATA.windows[ACTIVE]||DATA.windows.lifetime;const p=DATA.pricing||{};document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.key===ACTIVE));document.getElementById('hero').innerHTML=`
<div class="period">${esc(w.label)}</div><div class="money">${usd(w.estimated_input_value_avoided_usd)}</div><div class="moneylabel">estimated provider input value avoided</div>
<div class="grid"><div class="metric"><div class="mval good">${fmt(w.provider_tokens_avoided)}</div><div class="mlabel">provider-bound tokens avoided</div></div><div class="metric"><div class="mval blue">${fmt(w.provider_requests_optimized)}</div><div class="mlabel">requests optimized</div></div><div class="metric"><div class="mval">${fmt(w.local_tokens_reduced)}</div><div class="mlabel">local-only tokens reduced · no $ claim</div></div></div>
<div class="meta"><span>pricing ${esc(p.as_of||'unknown')} · ${esc(p.source||'unknown')}</span><span>${w.window_start?esc(w.window_start)+' → '+esc(w.window_end):'persistent lifetime total'}</span></div>`;}
async function load(){try{const r=await fetch('/traffic-value.json',{cache:'no-store'});if(!r.ok)throw Error(r.status);DATA=await r.json();ACTIVE=DATA.default_window||'lifetime';const tabs=document.getElementById('tabs');tabs.innerHTML='';for(const key of DATA.window_order||[]){const b=document.createElement('button');b.className='tab';b.dataset.key=key;b.textContent=DATA.windows[key].label;b.onclick=()=>{ACTIVE=key;render()};tabs.appendChild(b)}render()}catch(e){document.getElementById('hero').innerHTML='<div class="period">Value API unavailable</div>';}}
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


__all__ = [
    "_TRAFFIC_VALUE_HTML",
    "_install_route",
    "build_traffic_value_snapshot",
]
