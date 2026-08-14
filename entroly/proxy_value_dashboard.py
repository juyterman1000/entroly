"""Traffic Value dashboard projection for canonical attribution rows."""

from __future__ import annotations

from . import proxy_traffic_value as _value


def install_value_dashboard() -> None:
    html = _value._TRAFFIC_VALUE_HTML
    if "Value attribution by source" in html:
        return
    html = html.replace(
        "function renderWindow(w){",
        "function valueRows(w){const rows=(w.value_by_source||[]).filter(x=>Number(x.tokens||0)!==0||Number(x.micro_usd||0)!==0);if(!rows.length)return '<div class=\"dim\">No attributed contribution evidence yet.</div>';return '<div class=\"grid\">'+rows.slice(0,9).map(x=>metric((x.tokens?((x.tokens<0?'-':'')+compact(Math.abs(x.tokens))+' tok'):'$'+(Number(x.micro_usd||0)/1e6).toFixed(4)),x.source.replaceAll('_',' ')+' · '+String(x.tier||'').toUpperCase()+' · '+String(x.role||''),x.tokens<0?'warn':x.role==='protected'?'blue':'good')).join('')+'</div>';}\nfunction renderWindow(w){",
    )
    marker = '<div class="meta"><span>${w.key===\'session\'?\'Process-local session\':(w.window_start?esc(w.window_start)+\' → \'+esc(w.window_end):\'\')}</span>'
    html = html.replace(
        marker,
        '<div class="period" style="margin-top:18px">Value attribution by source</div>${valueRows(w)}'
        + marker,
    )
    html = html.replace(
        "<b>Evidence policy:</b>",
        "<b>Evidence policy:</b> attributed rows explain the headline and are not added twice; observed extra provider cost is a debit when auditable. ",
    )
    _value._TRAFFIC_VALUE_HTML = html


install_value_dashboard()

__all__ = ["install_value_dashboard"]
