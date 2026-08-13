"""Executive AI Traffic Value dashboard for Entroly.

Traffic Receipts feed two views through one accounting path:

* ``This session`` is an in-process, content-blind rollup for immediate proof.
  It resets when the proxy process restarts.
* Today / 7D / 30D / 60D / 90D / All Time are durable rollups stored in the
  existing ValueTracker file.

No second savings database is introduced. Dollar fields keep their evidence
class visible: avoided-token value is modeled from Entroly's configured model
price, while cache benefit and provider input spend require provider-observed
usage carried by the Traffic Receipt.
"""

from __future__ import annotations

import datetime as _dt
import threading
import time
from collections import deque
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

_SCHEMA_VERSION = "entroly.traffic-value.v3"
_TRAFFIC_KEY = "traffic_assurance"
_MAX_DAILY = 90
_MAX_SEEN_RECEIPTS = 2048
_MAX_SESSION_SEEN = 2048

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
    "estimated_priced_requests",
    "cache_benefit_priced_requests",
    "provider_priced_requests",
)
_MONEY_FIELDS = (
    "estimated_value_avoided_usd",
    "measured_cache_benefit_usd",
    "provider_input_spend_usd",
)

_SESSION_LOCK = threading.RLock()
_SESSION_STARTED_AT = time.time()
_SESSION_METRICS: dict[str, Any] = {}
_SESSION_SEEN: deque[str] = deque()
_SESSION_SEEN_SET: set[str] = set()


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


# Initialize after _empty_metrics exists.
_SESSION_METRICS = _empty_metrics()


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


def _receipt_delta(receipt: Any) -> dict[str, int | float]:
    original = _nonnegative_int(getattr(receipt, "original_context_tokens", 0))
    sent = _nonnegative_int(getattr(receipt, "entroly_context_tokens", 0))
    avoided = _nonnegative_int(getattr(receipt, "tokens_avoided", 0))
    model = str(getattr(receipt, "executed_model", "") or "")
    model_priced = bool(model and _has_priced_model(model))
    verification = str(getattr(receipt, "verification", "") or "").upper()

    recovery_receipts = _nonnegative_int(getattr(receipt, "recovery_receipts", 0))
    recovery_invoked = recovery_receipts > 0
    # Receipt v1 currently exposes recovery evidence, not a dedicated invocation
    # event. Keep this conservative and require explicit final PASS for success.
    recovery_succeeded = recovery_invoked and verification == "PASS"

    cache_hit = getattr(receipt, "cache_hit", None)
    input_cost_micro = getattr(receipt, "input_cost_micro_usd", None)
    cache_benefit_micro = getattr(receipt, "cache_benefit_micro_usd", None)

    estimated_value = 0.0
    if avoided and model_priced:
        try:
            estimated_value = round(
                max(0.0, float(estimate_cost(avoided, model, kind="input"))),
                6,
            )
        except Exception:
            model_priced = False
            estimated_value = 0.0

    return {
        "requests_observed": 1,
        "requests_optimized": int(avoided > 0),
        "tokens_received": original,
        "tokens_sent": sent,
        "tokens_avoided": avoided,
        "estimated_value_avoided_usd": estimated_value,
        "measured_cache_benefit_usd": (
            round(_nonnegative_int(cache_benefit_micro) / 1_000_000.0, 6)
            if cache_benefit_micro is not None
            else 0.0
        ),
        # Traffic Receipt v1 prices provider input/cache categories only.
        "provider_input_spend_usd": (
            round(_nonnegative_int(input_cost_micro) / 1_000_000.0, 6)
            if input_cost_micro is not None
            else 0.0
        ),
        "estimated_priced_requests": int(model_priced),
        "cache_benefit_priced_requests": int(cache_benefit_micro is not None),
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


def _record_session_receipt(receipt: Any) -> bool:
    """Accumulate one verified receipt into the current proxy process session."""
    if not getattr(receipt, "verify", lambda: False)():
        return False
    receipt_id = str(getattr(receipt, "receipt_id", "") or "")
    if not receipt_id:
        return False
    delta = _receipt_delta(receipt)

    with _SESSION_LOCK:
        if receipt_id in _SESSION_SEEN_SET:
            return False
        if len(_SESSION_SEEN) >= _MAX_SESSION_SEEN:
            old = _SESSION_SEEN.popleft()
            _SESSION_SEEN_SET.discard(old)
        _SESSION_SEEN.append(receipt_id)
        _SESSION_SEEN_SET.add(receipt_id)
        _accumulate(_SESSION_METRICS, delta)
    return True


def _session_status(row: Mapping[str, Any]) -> tuple[str, str]:
    requests = _nonnegative_int(row.get("requests_observed"))
    if requests == 0:
        return (
            "waiting",
            "Send traffic through Entroly to see immediate value proof here.",
        )
    if (
        _nonnegative_int(row.get("tokens_avoided")) > 0
        or _nonnegative_int(row.get("warm_cache_protected_tokens")) > 0
        or _nonnegative_float(row.get("measured_cache_benefit_usd")) > 0
    ):
        return (
            "measurable",
            "Entroly is creating measurable value in this session.",
        )
    return (
        "no_measurable_value",
        "Traffic is flowing; no measurable context reduction or cache benefit "
        "in this session yet.",
    )


def _session_rollup(*, now: float | None = None) -> dict[str, Any]:
    now_ts = time.time() if now is None else float(now)
    with _SESSION_LOCK:
        started_at = _SESSION_STARTED_AT
        result: dict[str, Any] = {
            "key": "session",
            "label": "This session",
            **_empty_metrics(),
        }
        _accumulate(result, _SESSION_METRICS)

    result["session_started_at"] = started_at
    result["session_elapsed_seconds"] = max(0, int(now_ts - started_at))
    result["window_start"] = _dt.datetime.fromtimestamp(
        started_at, tz=_dt.timezone.utc
    ).isoformat().replace("+00:00", "Z")
    result["window_end"] = "now"
    result["durability"] = "process-local"
    result["reset_event"] = "proxy process restart"
    status, message = _session_status(result)
    result["session_status"] = status
    result["session_status_message"] = message
    return _finalize_metrics(result)


def _reset_session_state_for_tests(*, started_at: float | None = None) -> None:
    """Reset process-local session state. Internal test seam only."""
    global _SESSION_STARTED_AT, _SESSION_METRICS
    with _SESSION_LOCK:
        _SESSION_STARTED_AT = time.time() if started_at is None else float(started_at)
        _SESSION_METRICS = _empty_metrics()
        _SESSION_SEEN.clear()
        _SESSION_SEEN_SET.clear()


def record_traffic_value_receipt(
    receipt: Any,
    *,
    tracker: Any | None = None,
) -> bool:
    """Record one Traffic Receipt into session and durable executive rollups.

    Session aggregation is deliberately independent of durable persistence so a
    temporary telemetry-file problem does not erase immediate operator feedback.
    The return value retains the historical contract: ``True`` means the durable
    rollup accepted and persisted the receipt.
    """
    if not getattr(receipt, "verify", lambda: False)():
        return False

    receipt_id = str(getattr(receipt, "receipt_id", "") or "")
    if not receipt_id:
        return False

    value_tracker = tracker or get_tracker()
    day = _receipt_day(receipt)
    delta = _receipt_delta(receipt)
    observed_at = (
        _nonnegative_float(getattr(receipt, "observed_at", 0.0)) or time.time()
    )

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

            for old_day in sorted(daily)[:-_MAX_DAILY]:
                daily.pop(old_day, None)

            value_tracker._save()

        # Only live receipts accepted by the durable idempotency gate enter the
        # process session. If persistence itself is unavailable, the exception
        # path below still preserves immediate process-local proof.
        _record_session_receipt(receipt)
        return True
    except Exception as exc:
        _record_session_receipt(receipt)
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

    session = _session_rollup(now=now_ts)
    windows = [
        session,
        _rolling_window(daily, days=1, key="today", label="Today", today=current_day),
        _rolling_window(daily, days=7, key="7d", label="7 days", today=current_day),
        _rolling_window(daily, days=30, key="30d", label="30 days", today=current_day),
        _rolling_window(daily, days=60, key="60d", label="60 days", today=current_day),
        _rolling_window(daily, days=90, key="90d", label="90 days", today=current_day),
        _lifetime_rollup(state),
    ]

    started_at = _nonnegative_float(state.get("started_at"))
    age_days = max(0, int((now_ts - started_at) // 86400)) if started_at else 0
    default_window = _default_window(age_days)
    if session["requests_observed"] > 0 and age_days == 0:
        default_window = "session"

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_unix": round(now_ts, 3),
        "collection_started_at": started_at or None,
        # Kept for backward API compatibility. This is days collecting Traffic Value.
        "installed_days": age_days,
        "default_window": default_window,
        "always_show_lifetime": True,
        "session_started_at": session["session_started_at"],
        "session_elapsed_seconds": session["session_elapsed_seconds"],
        "windows": {window["key"]: window for window in windows},
        "window_order": [window["key"] for window in windows],
        "pricing": pricing_provenance(),
        "truth": {
            "session": (
                "This session is a process-local rollup of verified Traffic "
                "Receipts. It resets on proxy restart and is not added again to "
                "All Time."
            ),
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
                "Recovery invoked currently follows receipt recovery evidence; "
                "recovery succeeded additionally requires final explicit PASS "
                "verification."
            ),
            "total_ai_value": (
                "Estimated value on tokens not sent plus measured cache benefit on "
                "tokens that were sent. The components keep separate evidence labels."
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
.sessionstatus{display:flex;justify-content:space-between;gap:16px;margin-top:14px;padding:12px 14px;border:1px solid var(--line);border-radius:12px;background:rgba(8,12,18,.6)}.sessionstatus.measurable b{color:var(--good)}.sessionstatus.waiting b{color:var(--blue)}.sessionstatus.no_measurable_value b{color:var(--warn)}.sessionstatus span{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.metric{border:1px solid var(--line);border-radius:14px;padding:15px;background:rgba(8,12,18,.58)}.mval{font:800 21px ui-monospace,SFMono-Regular,Menlo,monospace}.mlabel{color:var(--dim);font-size:12px;margin-top:5px}
.moneygrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.moneybox{border:1px solid var(--line);border-radius:15px;padding:17px;background:rgba(8,12,18,.7)}.money{font-size:30px;font-weight:900;letter-spacing:-.04em}.moneylabel{color:var(--dim);font-size:12px;margin-top:4px}
.alltime{margin-top:18px;border-color:#30435e}.alltitle{display:flex;justify-content:space-between;gap:20px;align-items:center}.badge{font-size:11px;color:var(--dim);border:1px solid var(--line);padding:5px 9px;border-radius:999px}
.good{color:var(--good)}.blue{color:var(--blue)}.warn{color:var(--warn)}.bad{color:var(--bad)}.dim{color:var(--dim)}
.note{margin-top:14px;padding:13px 15px;border:1px solid var(--line);border-radius:12px;color:var(--dim);font-size:12px}.note b{color:var(--text)}.meta{display:flex;justify-content:space-between;gap:18px;margin-top:14px;color:var(--dim);font-size:11px}
@media(max-width:760px){.grid,.moneygrid{grid-template-columns:1fr}.alltitle,.meta,.sessionstatus{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body><main>
<div class="eyebrow">AI Traffic Assurance</div>
<div class="title">AI Traffic Value</div>
<div class="sub">Immediate session proof, recent operating value, and durable All Time impact.</div>
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
const duration=s=>{s=Math.max(0,Number(s||0));const h=Math.floor(s/3600),m=Math.floor((s%3600)/60);return h?`${h}h ${m}m`:`${m}m`};
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let DATA=null,ACTIVE=null,USER_SELECTED=false;
function moneyBox(value,label,cls='good'){return `<div class="moneybox"><div class="money ${cls}">${usd(value)}</div><div class="moneylabel">${esc(label)}</div></div>`}
function moneyMaybe(value,label,available,cls='good'){return `<div class="moneybox"><div class="money ${available?cls:'dim'}">${available?usd(value):'—'}</div><div class="moneylabel">${esc(label)}${available?'':' · not measured'}</div></div>`}
function metric(value,label,cls=''){return `<div class="metric"><div class="mval ${cls}">${value}</div><div class="mlabel">${esc(label)}</div></div>`}
function renderWindow(w){
 const estimateAvailable=w.estimated_priced_requests>0;
 const cacheBenefitAvailable=w.cache_benefit_priced_requests>0;
 const spendAvailable=w.provider_priced_requests>0;
 const totalAvailable=(Number(w.tokens_avoided||0)===0||estimateAvailable)&&(Number(w.cache_hits||0)===0||cacheBenefitAvailable);
 const sessionBanner=w.key==='session'?`<div class="sessionstatus ${esc(w.session_status)}"><b>${esc(w.session_status_message)}</b><span>Session duration ${duration(w.session_elapsed_seconds)} · resets on proxy restart</span></div>`:'';
 document.getElementById('period').innerHTML=`
 <div class="period">AI TRAFFIC VALUE — ${esc(w.label).toUpperCase()}</div>
 ${sessionBanner}
 <div class="grid">
  ${metric(fmt(w.requests_optimized),'Requests optimized','blue')}
  ${metric(compact(w.tokens_received)+' tokens','Tokens received by Entroly')}
  ${metric(compact(w.tokens_sent)+' tokens','Tokens sent to provider')}
  ${metric(compact(w.tokens_avoided)+' tokens','Tokens avoided','good')}
  ${metric(pct(w.context_reduction_pct),'Context reduction','good')}
  ${metric(compact(w.warm_cache_protected_tokens)+' tokens','Warm cache protected','blue')}
 </div>
 <div class="moneygrid">
  ${moneyMaybe(w.estimated_value_avoided_usd,'Estimated value avoided',estimateAvailable)}
  ${moneyMaybe(w.measured_cache_benefit_usd,'Measured cache benefit',cacheBenefitAvailable)}
  ${moneyMaybe(w.provider_input_spend_usd,'Provider input spend observed',spendAvailable,'')}
 </div>
 <div class="grid">
  ${metric(pct(w.requests_verified_pct),'Requests verified')}
  ${metric(pct(w.recovery_invoked_pct),'Recovery evidence observed')}
  ${metric(pct(w.recovery_succeeded_pct),'Recovery evidence + PASS')}
  ${metric(pct(w.verification_pass_pct),'Verification pass rate')}
  ${metric(pct(w.cache_hit_request_pct),'Cache-hit requests')}
  ${metric(totalAvailable?usd(w.total_ai_value_protected_usd):'—','Total AI value protected',totalAvailable?'good':'dim')}
 </div>
 <div class="meta"><span>${w.key==='session'?'Process-local session':(w.window_start?esc(w.window_start)+' → '+esc(w.window_end):'')}</span><span>${spendAvailable?'provider-priced input usage observed':'provider $ appears when usage + pricing exist'}</span></div>`;
}
function renderLifetime(w){
 const estimateAvailable=w.estimated_priced_requests>0;
 const cacheBenefitAvailable=w.cache_benefit_priced_requests>0;
 const totalAvailable=(Number(w.tokens_avoided||0)===0||estimateAvailable)&&(Number(w.cache_hits||0)===0||cacheBenefitAvailable);
 document.getElementById('lifetime').innerHTML=`
 <div class="alltitle"><div><div class="period">ALL TIME</div><div class="dim">Keeps accumulating across proxy restarts.</div></div><div class="badge">${fmt(DATA.installed_days||0)} days collecting Traffic Value</div></div>
 <div class="moneygrid">
  ${moneyMaybe(w.estimated_value_avoided_usd,'Estimated value avoided',estimateAvailable)}
  ${moneyMaybe(w.measured_cache_benefit_usd,'Measured cache benefit',cacheBenefitAvailable,'blue')}
  ${moneyMaybe(w.total_ai_value_protected_usd,'Total AI value protected',totalAvailable)}
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
  DATA=await r.json();
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
    """Persist each admitted Traffic Receipt through the single value recorder."""
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
    "_reset_session_state_for_tests",
    "_session_rollup",
    "build_traffic_value_snapshot",
    "record_traffic_value_receipt",
]
