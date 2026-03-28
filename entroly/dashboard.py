"""
Entroly Live Dashboard — Real-time AI value metrics at localhost:9378
=====================================================================

Shows developers exactly what Entroly's Rust engine is doing for them,
pulling REAL data from all engine subsystems:

  Engine Stats:       tokens saved, cost saved, dedup hits, turn count
  PRISM RL Weights:   learned scoring weights (recency/frequency/semantic/entropy)
  Health Analysis:    code health grade A–F, clones, dead symbols, god files
  SAST Security:      vulnerability findings with CWE categories
  Knapsack Decisions: which fragments were included/excluded and why
  Dep Graph:          symbol definitions, edges, coupling stats

Starts alongside the proxy and auto-refreshes every 3 seconds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional

logger = logging.getLogger("entroly.dashboard")

# ── Engine reference (set by start_dashboard) ─────────────────────────────────
_engine: Optional[Any] = None
_lock = threading.Lock()

# Per-request tracking (populated by proxy integration)
_request_log: list[dict] = []
_MAX_LOG = 50


def record_request(entry: dict):
    """Record a proxy request's metrics (called from proxy.py)."""
    with _lock:
        _request_log.append(entry)
        if len(_request_log) > _MAX_LOG:
            del _request_log[: len(_request_log) - _MAX_LOG]


def _safe_json(obj: Any) -> Any:
    """Recursively convert to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return 0.0
        return round(obj, 6)
    return obj


def _get_full_snapshot() -> dict:
    """Pull ALL real data from the engine subsystems."""
    snap: dict[str, Any] = {
        "ts": time.time(),
        "engine_available": _engine is not None,
    }

    # Persistent value tracker (independent of engine — always available)
    try:
        from entroly.value_tracker import get_tracker
        tracker = get_tracker()
        snap["value_trends"] = tracker.get_trends()
        snap["value_confidence"] = tracker.get_confidence()
    except Exception:
        snap["value_trends"] = None
        snap["value_confidence"] = None

    if _engine is None:
        return snap

    try:
        # 1. Core stats — tokens saved, cost, dedup, turns
        stats = _engine.stats() if hasattr(_engine, "stats") else {}
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            stats = _engine._rust.stats()
            stats = dict(stats)
            for k, v in stats.items():
                if hasattr(v, "items"):
                    stats[k] = dict(v)
        snap["stats"] = _safe_json(stats)
    except Exception as e:
        snap["stats"] = {"error": str(e)}

    try:
        # 2. PRISM RL weights — the learned scoring weights
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            rust = _engine._rust
            snap["prism_weights"] = {
                "recency": round(getattr(rust, "w_recency", 0.3), 4),
                "frequency": round(getattr(rust, "w_frequency", 0.25), 4),
                "semantic": round(getattr(rust, "w_semantic", 0.25), 4),
                "entropy": round(getattr(rust, "w_entropy", 0.2), 4),
            }
    except Exception:
        snap["prism_weights"] = None

    try:
        # 3. Health analysis — code health grade
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            health_json = _engine._rust.analyze_health()
            snap["health"] = _safe_json(json.loads(health_json))
    except Exception:
        snap["health"] = None

    try:
        # 4. SAST security report
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            sec_json = _engine._rust.security_report()
            snap["security"] = _safe_json(json.loads(sec_json))
    except Exception:
        snap["security"] = None

    try:
        # 5. Knapsack explainability — last optimization decisions
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            explain = _engine._rust.explain_selection()
            snap["explain"] = _safe_json(dict(explain))
    except Exception:
        snap["explain"] = None

    try:
        # 6. Dependency graph stats
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            dg = _engine._rust.dep_graph_stats()
            snap["dep_graph"] = _safe_json(dict(dg))
    except Exception:
        snap["dep_graph"] = None

    try:
        # 7. Cache intelligence — live EGSC + RAVEN-UCB observability
        stats = snap.get("stats", {}) if isinstance(snap.get("stats"), dict) else {}
        cache = stats.get("cache", {}) if isinstance(stats.get("cache"), dict) else {}
        policy = stats.get("policy", {}) if isinstance(stats.get("policy"), dict) else {}
        snap["cache_intelligence"] = _safe_json(
            {
                "entries": cache.get("entries", 0),
                "lookups": cache.get("lookups", 0),
                "exact_hits": cache.get("exact_hits", 0),
                "semantic_hits": cache.get("semantic_hits", 0),
                "tokens_saved": cache.get("tokens_saved", 0),
                "hit_rate": cache.get("hit_rate", 0.0),
                "hit_rate_ema": cache.get("hit_rate_ema", cache.get("hit_rate", 0.0)),
                "thompson_alpha": cache.get("thompson_alpha", 0.0),
                "thompson_beta": cache.get("thompson_beta", 0.0),
                "adaptive_alpha": cache.get("adaptive_alpha", 0.0),
                "admissions": cache.get("admissions", 0),
                "rejections": cache.get("rejections", 0),
                "invalidations": cache.get("invalidations", 0),
                "total_resets": cache.get("total_resets", 0),
                "exploration_rate": policy.get("adaptive_exploration_rate", 0.0),
                "configured_exploration_rate": policy.get("configured_exploration_rate", 0.0),
                "feedback_observations": policy.get("feedback_observations", 0),
                "total_explorations": policy.get("total_explorations", 0),
                "total_exploitations": policy.get("total_exploitations", 0),
                "explore_ratio": policy.get("explore_ratio", 0.0),
                "exploit_ratio": policy.get("exploit_ratio", 0.0),
            }
        )
    except Exception:
        snap["cache_intelligence"] = None

    # 8. Recent proxy requests
    with _lock:
        snap["recent_requests"] = list(_request_log)

    return snap


# ── HTML Dashboard ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly — Intelligence Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #050508; --bg2: #0a0b10; --card: rgba(14,17,24,0.85);
  --glass: rgba(255,255,255,0.03); --glass2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.06); --border2: rgba(255,255,255,0.12);
  --text: #e8ecf4; --dim: #6b7280; --dim2: #3b4252;
  --emerald: #34d399; --emerald-glow: rgba(52,211,153,0.15);
  --blue: #60a5fa; --blue-glow: rgba(96,165,250,0.12);
  --violet: #a78bfa; --violet-glow: rgba(167,139,250,0.12);
  --amber: #fbbf24; --amber-glow: rgba(251,191,36,0.10);
  --rose: #fb7185; --rose-glow: rgba(251,113,133,0.10);
  --cyan: #22d3ee; --cyan-glow: rgba(34,211,238,0.10);
  --grad1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --grad2: linear-gradient(135deg, #34d399 0%, #06b6d4 100%);
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;top:-50%;left:-50%;width:200%;height:200%;
  background:radial-gradient(circle at 30% 20%,rgba(102,126,234,0.04),transparent 50%),
  radial-gradient(circle at 70% 80%,rgba(118,75,162,0.03),transparent 50%);z-index:0;pointer-events:none;}
.topbar{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
  padding:14px 32px;background:rgba(5,5,8,0.8);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);}
.brand{display:flex;align-items:center;gap:14px;}
.brand h1{font-size:24px;font-weight:900;letter-spacing:-0.5px;
  background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.brand .btag{font-size:11px;padding:3px 10px;border-radius:20px;background:var(--emerald-glow);
  color:var(--emerald);font-weight:600;letter-spacing:0.5px;}
.live{display:flex;align-items:center;gap:8px;color:var(--emerald);font-size:12px;font-weight:500;}
.live .dot{width:7px;height:7px;border-radius:50%;background:var(--emerald);
  box-shadow:0 0 12px var(--emerald);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.main{position:relative;z-index:1;padding:24px 32px;max-width:1440px;margin:0 auto;}
/* Onboarding whisper */
.whisper{padding:12px 20px;margin-bottom:20px;background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);
  border-radius:12px;font-size:13px;color:var(--blue);display:flex;align-items:center;gap:10px;}
.whisper code{background:rgba(96,165,250,0.15);padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:12px;}
.whisper .dismiss{margin-left:auto;cursor:pointer;opacity:0.5;font-size:16px;}
.whisper .dismiss:hover{opacity:1;}
/* Hero */
.hero{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:20px;}
.hero-card{padding:20px 24px;background:var(--card);border:1px solid var(--border);border-radius:16px;position:relative;overflow:hidden;}
.hero-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.hc-files::before{background:var(--grad2);}
.hc-health::before{background:linear-gradient(90deg,var(--blue),var(--violet));}
.hc-sast::before{background:linear-gradient(90deg,var(--rose),var(--amber));}
.hc-savings::before{background:linear-gradient(90deg,var(--emerald),var(--cyan));}
.hero-icon{font-size:28px;margin-bottom:8px;}
.hero-val{font-size:36px;font-weight:900;letter-spacing:-2px;font-feature-settings:'tnum';}
.hero-label{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;}
.hero-sub{font-size:12px;color:var(--dim);margin-top:8px;}
.hv-green{color:var(--emerald);}
.hv-blue{color:var(--blue);}
.hv-rose{color:var(--rose);}
.hv-amber{color:var(--amber);}
/* Before/After */
.ba-panel{display:grid;grid-template-columns:1fr auto 1fr;gap:0;margin-bottom:20px;
  background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;}
.ba-side{padding:24px 28px;}
.ba-left{border-right:1px solid var(--border);}
.ba-center{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 24px;
  background:rgba(52,211,153,0.04);}
.ba-title{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);margin-bottom:8px;}
.ba-val{font-size:28px;font-weight:800;font-feature-settings:'tnum';}
.ba-detail{font-size:12px;color:var(--dim);margin-top:4px;}
.ba-arrow{font-size:24px;color:var(--dim2);margin:8px 0;}
.ba-pct{font-size:36px;font-weight:900;letter-spacing:-2px;
  background:var(--grad2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.ba-pct-label{font-size:11px;color:var(--emerald);font-weight:600;margin-top:4px;}
/* Grid */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px;}
/* Panel */
.panel{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;
  backdrop-filter:blur(10px);transition:border-color 0.3s,box-shadow 0.3s;}
.panel:hover{border-color:var(--border2);box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.ph{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border);}
.ph h2{font-size:14px;font-weight:700;}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
.b-green{background:var(--emerald-glow);color:var(--emerald);}
.b-blue{background:var(--blue-glow);color:var(--blue);}
.b-violet{background:var(--violet-glow);color:var(--violet);}
.b-amber{background:var(--amber-glow);color:var(--amber);}
.b-rose{background:var(--rose-glow);color:var(--rose);}
.b-cyan{background:var(--cyan-glow);color:var(--cyan);}
.pb{padding:20px;}
/* Radar */
.radar-wrap{display:flex;align-items:center;justify-content:center;padding:12px 0;}
.radar-canvas{width:180px;height:180px;}
.radar-legend{list-style:none;margin-left:20px;}
.radar-legend li{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:13px;color:var(--dim);}
.radar-legend .rdot{width:8px;height:8px;border-radius:50%;}
.radar-legend .rval{font-weight:700;color:var(--text);font-feature-settings:'tnum';min-width:36px;}
.prism-insight{padding:10px 16px;margin-top:8px;background:rgba(167,139,250,0.06);border-radius:10px;
  font-size:12px;color:var(--violet);line-height:1.5;}
/* Health Ring */
.health-ring-wrap{display:flex;align-items:center;justify-content:center;gap:24px;padding:16px 0;}
.health-ring{position:relative;width:110px;height:110px;}
.health-ring canvas{width:100%;height:100%;}
.health-ring .grade{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:40px;font-weight:900;}
.health-stats{list-style:none;}
.health-stats li{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:13px;color:var(--dim);}
.health-stats .hv{font-weight:700;color:var(--text);min-width:20px;text-align:right;}
.health-rec{padding:10px;background:rgba(251,191,36,0.06);border-radius:10px;font-size:12px;color:var(--amber);margin-top:8px;line-height:1.4;}
.cache-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:14px;}
.cache-kpi{padding:14px;border:1px solid var(--border);background:var(--glass);border-radius:12px;}
.cache-kpi-label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1.1px;margin-bottom:6px;}
.cache-kpi-val{font-size:24px;font-weight:800;font-feature-settings:'tnum';}
.cache-kpi-sub{font-size:11px;color:var(--dim);margin-top:4px;}
.cache-split{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.cache-block{padding:14px;border:1px solid var(--border);background:var(--glass);border-radius:12px;}
.cache-block h3{font-size:12px;font-weight:700;margin-bottom:8px;}
.cache-pair{display:flex;justify-content:space-between;gap:12px;padding:4px 0;font-size:12px;color:var(--dim);}
.cache-pair strong{color:var(--text);font-weight:700;}
.cache-meter{height:10px;display:flex;overflow:hidden;border-radius:999px;background:rgba(255,255,255,0.06);margin-top:10px;}
.cache-meter-exploit{background:linear-gradient(90deg,var(--emerald),var(--cyan));}
.cache-meter-explore{background:linear-gradient(90deg,var(--amber),var(--rose));}
.cache-note{padding:10px 12px;margin-top:12px;background:rgba(34,211,238,0.06);border-radius:10px;font-size:12px;color:var(--cyan);line-height:1.5;}
/* Tables */
table{width:100%;border-collapse:collapse;}
th{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim2);padding:10px 14px;
  text-align:left;background:rgba(255,255,255,0.02);font-weight:600;}
td{padding:8px 14px;border-top:1px solid var(--border);font-size:13px;}
td.mono{font-family:'JetBrains Mono',monospace;font-size:12px;}
tr:hover td{background:rgba(255,255,255,0.015);}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;}
.t-green{background:var(--emerald-glow);color:var(--emerald);}
.t-rose{background:var(--rose-glow);color:var(--rose);}
.t-amber{background:var(--amber-glow);color:var(--amber);}
.t-violet{background:var(--violet-glow);color:var(--violet);}
.empty{text-align:center;padding:28px;color:var(--dim2);font-size:13px;}
/* Sparkline */
.sparkline{display:flex;align-items:flex-end;gap:2px;height:40px;margin-top:8px;}
.sparkline .bar{flex:1;background:var(--grad2);border-radius:2px 2px 0 0;min-width:3px;
  transition:height 0.4s cubic-bezier(0.16,1,0.3,1);opacity:0.7;}
.sparkline .bar:hover{opacity:1;}
/* Finding row */
.finding{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);}
.finding:last-child{border-bottom:none;}
.finding-sev{font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;flex-shrink:0;margin-top:2px;}
.finding-sev.crit{background:var(--rose-glow);color:var(--rose);}
.finding-sev.high{background:var(--amber-glow);color:var(--amber);}
.finding-file{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text);}
.finding-desc{font-size:11px;color:var(--dim);margin-top:2px;}
/* Value Trends */
.trends-panel{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-bottom:20px;}
.trends-header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border);}
.trends-header h2{font-size:14px;font-weight:700;}
.trends-body{padding:20px;}
.trends-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px;}
.trends-kpi{padding:14px;border:1px solid var(--border);background:var(--glass);border-radius:12px;}
.trends-kpi-label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1.1px;margin-bottom:6px;}
.trends-kpi-val{font-size:24px;font-weight:800;font-feature-settings:'tnum';}
.trends-kpi-sub{font-size:11px;color:var(--dim);margin-top:4px;}
.trends-chart{height:80px;display:flex;align-items:flex-end;gap:3px;padding:8px 0;}
.trends-chart .tbar{flex:1;border-radius:3px 3px 0 0;min-width:6px;transition:height 0.4s cubic-bezier(0.16,1,0.3,1);position:relative;}
.trends-chart .tbar:hover::after{content:attr(data-tip);position:absolute;bottom:100%;left:50%;transform:translateX(-50%);
  background:var(--card);border:1px solid var(--border);padding:4px 8px;border-radius:6px;font-size:10px;white-space:nowrap;color:var(--text);z-index:10;}
.trends-chart .tbar.cost{background:linear-gradient(180deg,var(--emerald),var(--cyan));}
.trends-chart .tbar.tokens{background:linear-gradient(180deg,var(--blue),var(--violet));}
.trends-tabs{display:flex;gap:8px;margin-bottom:12px;}
.trends-tab{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;
  background:var(--glass);border:1px solid var(--border);color:var(--dim);transition:all 0.2s;}
.trends-tab.active{background:var(--emerald-glow);color:var(--emerald);border-color:rgba(52,211,153,0.3);}
.trends-tab:hover{border-color:var(--border2);}
@media(max-width:1100px){.hero{grid-template-columns:1fr 1fr;}.grid3{grid-template-columns:1fr 1fr;}.ba-panel{grid-template-columns:1fr;}.cache-kpis{grid-template-columns:1fr 1fr;}.cache-split{grid-template-columns:1fr;}}
@media(max-width:768px){.hero,.grid2,.grid3,.cache-kpis{grid-template-columns:1fr;}.main{padding:16px;}}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><h1>⚡ Entroly</h1><span class="btag">INTELLIGENCE DASHBOARD</span></div>
  <div class="live"><div class="dot"></div>Live · 3s refresh</div>
</div>
<div class="main">
  <div class="whisper" id="whisper">
    💡 Point your AI tool's API base URL to <code>http://localhost:9377/v1</code> to start optimizing every LLM call.
    <span class="dismiss" onclick="this.parentElement.style.display='none'">✕</span>
  </div>
  <div class="hero" id="hero"></div>
  <div id="valueTrends"></div>
  <div id="ba"></div>
  <div class="grid2">
    <div class="panel">
      <div class="ph"><h2>🧠 PRISM Intelligence</h2><span class="badge b-violet">RL-Learned</span></div>
      <div class="pb" id="prism"></div>
    </div>
    <div class="panel">
      <div class="ph"><h2>🏥 Code Health</h2><span id="hb" class="badge b-green">—</span></div>
      <div class="pb" id="health"></div>
    </div>
  </div>
  <div class="panel" style="margin-bottom:20px;">
    <div class="ph"><h2>Cache Intelligence</h2><span id="cb" class="badge b-blue">—</span></div>
    <div class="pb" id="cacheintel"></div>
  </div>
  <div id="grid3wrap"></div>
  <div class="panel" style="margin-bottom:28px;">
    <div class="ph"><h2>📡 Request Flow</h2><span id="rb" class="badge b-cyan">—</span></div>
    <div id="sparkarea" style="padding:12px 20px 0;"></div>
    <div style="overflow-x:auto;">
      <table><thead><tr><th>Time</th><th>Model</th><th>Tokens In</th><th>Saved</th><th>Dedup</th><th>SAST</th><th>Query</th></tr></thead>
      <tbody id="reqs"></tbody></table>
    </div>
  </div>
</div>
<script>
const fmt=n=>{if(n==null)return'—';return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(n)};
const money=n=>'$'+(n||0).toFixed(2);
const pct=n=>Math.round((n||0)*100)+'%';
const ago=ts=>{const s=Math.floor(Date.now()/1000-ts);return s<60?s+'s ago':s<3600?Math.floor(s/60)+'m ago':Math.floor(s/3600)+'h ago';};
let hasRequests=false;

function renderHero(d){
  const s=d.stats||{},sv=s.savings||{},ss=s.session||{},dd=s.dedup||{};
  const frags=ss.total_fragments||0;
  const secTotal=(d.security&&!d.security.error)?((d.security.critical_total||0)+(d.security.high_total||0)):0;
  const h=d.health||{};
  const grade=h.health_grade||'?';
  const score=h.code_health_score||0;
  const gc={'A':'var(--emerald)','B':'var(--blue)','C':'var(--amber)','D':'#e3872d','F':'var(--rose)'}[grade]||'var(--dim)';
  const tokensSaved=sv.total_tokens_saved||0;
  const costSaved=sv.estimated_cost_saved_usd||0;
  const reqs=d.recent_requests||[];
  if(reqs.length>0)hasRequests=true;
  // Hide whisper after first request
  const w=document.getElementById('whisper');
  if(w&&hasRequests)w.style.display='none';

  document.getElementById('hero').innerHTML=`
    <div class="hero-card hc-files"><div class="hero-icon">📁</div>
      <div class="hero-val hv-blue">${frags}</div>
      <div class="hero-label">Files Indexed</div>
      <div class="hero-sub">${fmt(ss.total_tokens_ingested||ss.total_token_count||0)} tokens scanned</div></div>
    <div class="hero-card hc-health"><div class="hero-icon">🏥</div>
      <div class="hero-val" style="color:${gc}">${grade}</div>
      <div class="hero-label">Code Health</div>
      <div class="hero-sub">${score}/100 · ${(h.god_files||[]).length} god files</div></div>
    <div class="hero-card hc-sast"><div class="hero-icon">🛡️</div>
      <div class="hero-val ${secTotal>0?'hv-rose':'hv-green'}">${secTotal>0?secTotal:'✓'}</div>
      <div class="hero-label">${secTotal>0?'Security Findings':'All Clear'}</div>
      <div class="hero-sub">${secTotal>0?(d.security.critical_total||0)+' critical · '+(d.security.high_total||0)+' high':'No vulnerabilities found'}</div></div>
    <div class="hero-card hc-savings"><div class="hero-icon">💰</div>
      <div class="hero-val hv-green">${tokensSaved>0?fmt(tokensSaved):(costSaved>0?money(costSaved):'—')}</div>
      <div class="hero-label">${tokensSaved>0?'Tokens Saved':'Savings'}</div>
      <div class="hero-sub">${tokensSaved>0?money(costSaved)+' saved · '+(dd.duplicates_detected||sv.total_duplicates_caught||0)+' dedup hits':'Starts counting with proxy requests'}</div></div>`;
}

function renderBA(d){
  const s=d.stats||{},ss=s.session||{},sv=s.savings||{};
  const ex=d.explain||{};
  const totalTokens=ss.total_token_count||ss.total_tokens_ingested||0;
  const selected=ex.included||[];
  const excluded=ex.excluded||[];
  const selTokens=selected.reduce((a,f)=>a+(f.tokens||f.token_count||0),0);
  const totalFrag=selected.length+excluded.length;
  if(totalTokens===0&&selected.length===0){document.getElementById('ba').innerHTML='';return;}
  const ratio=totalTokens>0&&selTokens>0?Math.round((1-selTokens/totalTokens)*100):0;
  const coverPct=totalFrag>0?Math.round(selected.length/totalFrag*100):0;

  document.getElementById('ba').innerHTML=`<div class="ba-panel">
    <div class="ba-side ba-left">
      <div class="ba-title">Without Entroly (Raw)</div>
      <div class="ba-val" style="color:var(--rose);">${fmt(totalTokens)} tokens</div>
      <div class="ba-detail">${totalFrag} fragments · no dedup · no scoring</div>
      <div class="ba-detail" style="margin-top:4px;">Top-K: ~5 files visible to LLM</div>
    </div>
    <div class="ba-center">
      <div class="ba-arrow">→</div>
      <div class="ba-pct">${ratio>0?ratio+'%':'—'}</div>
      <div class="ba-pct-label">${ratio>0?'reduction':'awaiting optimize'}</div>
    </div>
    <div class="ba-side">
      <div class="ba-title">With Entroly (Optimized)</div>
      <div class="ba-val" style="color:var(--emerald);">${selTokens>0?fmt(selTokens)+' tokens':'—'}</div>
      <div class="ba-detail">${selected.length} fragments · knapsack-optimal · deduped</div>
      <div class="ba-detail" style="margin-top:4px;">Coverage: ${coverPct}% of codebase at variable resolution</div>
    </div>
  </div>`;
}

function drawRadar(ctx,w,vals,colors){
  const cx=w/2,cy=w/2,r=w/2-18,n=vals.length;
  ctx.clearRect(0,0,w,w);
  for(let i=1;i<=4;i++){ctx.beginPath();for(let j=0;j<=n;j++){const a=Math.PI*2*j/n-Math.PI/2;const rr=r*i/4;j===0?ctx.moveTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a)):ctx.lineTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a));}ctx.strokeStyle='rgba(255,255,255,0.06)';ctx.stroke();}
  ctx.beginPath();vals.forEach((v,i)=>{const a=Math.PI*2*i/n-Math.PI/2;const rr=r*Math.min(v/0.5,1);i===0?ctx.moveTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a)):ctx.lineTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a));});
  ctx.closePath();ctx.fillStyle='rgba(167,139,250,0.15)';ctx.fill();ctx.strokeStyle='rgba(167,139,250,0.8)';ctx.lineWidth=2;ctx.stroke();
  vals.forEach((v,i)=>{const a=Math.PI*2*i/n-Math.PI/2;const rr=r*Math.min(v/0.5,1);ctx.beginPath();ctx.arc(cx+rr*Math.cos(a),cy+rr*Math.sin(a),4,0,Math.PI*2);ctx.fillStyle=colors[i];ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();});
}

function renderPrism(d){
  const w=d.prism_weights;
  if(!w){document.getElementById('prism').innerHTML='<div class="empty">Engine not initialized</div>';return;}
  const names=['Recency','Frequency','Semantic','Entropy'];
  const vals=[w.recency,w.frequency,w.semantic,w.entropy];
  const colors=['#667eea','#f5576c','#4facfe','#43e97b'];
  const maxIdx=vals.indexOf(Math.max(...vals));
  const insight=`Your codebase responds best to <b>${names[maxIdx].toLowerCase()}</b> — ${names[maxIdx]==='Recency'?'recent edits are most predictive of what the LLM needs next':names[maxIdx]==='Frequency'?'frequently accessed files are the best context signal':names[maxIdx]==='Semantic'?'semantic similarity to the query drives the best results':'information-dense files contribute most to LLM accuracy'}.`;
  document.getElementById('prism').innerHTML=`<div class="radar-wrap">
    <canvas class="radar-canvas" id="radarC" width="180" height="180"></canvas>
    <ul class="radar-legend">${names.map((n,i)=>`<li><span class="rdot" style="background:${colors[i]}"></span>${n}<span class="rval">${pct(vals[i])}</span></li>`).join('')}</ul>
  </div><div class="prism-insight">💡 ${insight}</div>`;
  const c=document.getElementById('radarC');if(c)drawRadar(c.getContext('2d'),180,vals,colors);
}

function renderHealth(d){
  const h=d.health,el=document.getElementById('health'),b=document.getElementById('hb');
  if(!h||h.error){el.innerHTML='<div class="empty">Ingest code to see health</div>';return;}
  const g=h.health_grade||'?',sc=h.code_health_score||0;
  const gc={'A':'var(--emerald)','B':'var(--blue)','C':'var(--amber)','D':'#e3872d','F':'var(--rose)'}[g]||'var(--dim)';
  b.textContent=g+' · '+sc+'/100';b.className='badge '+(g<='B'?'b-green':g==='C'?'b-amber':'b-rose');
  el.innerHTML=`<div class="health-ring-wrap">
    <div class="health-ring"><canvas id="hring" width="110" height="110"></canvas><div class="grade" style="color:${gc}">${g}</div></div>
    <ul class="health-stats">
      <li><span class="hv">${(h.clone_pairs||[]).length}</span>clone pairs</li>
      <li><span class="hv">${(h.dead_symbols||[]).length}</span>dead symbols</li>
      <li><span class="hv">${(h.god_files||[]).length}</span>god files</li>
      <li><span class="hv">${(h.arch_violations||[]).length}</span>arch violations</li>
      <li><span class="hv">${(h.naming_issues||[]).length}</span>naming issues</li>
    </ul></div>
    ${h.top_recommendation?'<div class="health-rec">💡 '+h.top_recommendation+'</div>':''}`;
  const c=document.getElementById('hring');
  if(c){const ctx=c.getContext('2d'),cx=55,cy=55,r=46,p=sc/100;
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.strokeStyle='rgba(255,255,255,0.05)';ctx.lineWidth=8;ctx.stroke();
    ctx.beginPath();ctx.arc(cx,cy,r,-Math.PI/2,-Math.PI/2+Math.PI*2*p);ctx.strokeStyle=gc;ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();}
}

function renderCache(d){
  const c=d.cache_intelligence||{},el=document.getElementById('cacheintel'),b=document.getElementById('cb');
  if(!el||!b){return;}
  const entries=c.entries||0,lookups=c.lookups||0,exact=c.exact_hits||0,semantic=c.semantic_hits||0;
  const hitRateEma=c.hit_rate_ema||0,exploreRatio=c.explore_ratio||0,exploitRatio=c.exploit_ratio||0;
  const warmState=entries===0?'Cold':hitRateEma>=0.30?'Warm':lookups>0?'Warming':'Idle';
  b.textContent=warmState+' · '+pct(hitRateEma);
  b.className='badge '+(hitRateEma>=0.30?'b-green':entries>0?'b-amber':'b-blue');
  el.innerHTML=`<div class="cache-kpis">
    <div class="cache-kpi"><div class="cache-kpi-label">Hit Rate EMA</div><div class="cache-kpi-val hv-green">${pct(hitRateEma)}</div><div class="cache-kpi-sub">${exact} exact · ${semantic} semantic</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Entries</div><div class="cache-kpi-val hv-blue">${fmt(entries)}</div><div class="cache-kpi-sub">${fmt(lookups)} lookups processed</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Tokens Saved</div><div class="cache-kpi-val hv-amber">${fmt(c.tokens_saved||0)}</div><div class="cache-kpi-sub">${c.invalidations||0} invalidations · ${c.total_resets||0} resets</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Explore / Exploit</div><div class="cache-kpi-val hv-rose">${pct(exploreRatio)} / ${pct(exploitRatio)}</div><div class="cache-kpi-sub">${c.total_explorations||0} explore · ${c.total_exploitations||0} exploit</div></div>
  </div>
  <div class="cache-split">
    <div class="cache-block">
      <h3>Thompson Gate</h3>
      <div class="cache-pair"><span>Posterior α / β</span><strong>${(c.thompson_alpha||0).toFixed(2)} / ${(c.thompson_beta||0).toFixed(2)}</strong></div>
      <div class="cache-pair"><span>Adaptive Rényi α</span><strong>${(c.adaptive_alpha||0).toFixed(4)}</strong></div>
      <div class="cache-pair"><span>Admissions / Rejections</span><strong>${fmt(c.admissions||0)} / ${fmt(c.rejections||0)}</strong></div>
    </div>
    <div class="cache-block">
      <h3>RAVEN-UCB</h3>
      <div class="cache-pair"><span>Current exploration_rate</span><strong>${(c.exploration_rate||0).toFixed(4)}</strong></div>
      <div class="cache-pair"><span>Configured cap</span><strong>${(c.configured_exploration_rate||0).toFixed(4)}</strong></div>
      <div class="cache-pair"><span>Feedback observations</span><strong>${fmt(c.feedback_observations||0)}</strong></div>
      <div class="cache-meter"><div class="cache-meter-exploit" style="width:${Math.max(0,Math.min(100,exploitRatio*100))}%;"></div><div class="cache-meter-explore" style="width:${Math.max(0,Math.min(100,exploreRatio*100))}%;"></div></div>
    </div>
  </div>
  <div class="cache-note">Cache reuse is tracked by hit-rate EMA while exploration_rate is the annealed RAVEN-UCB coefficient. As feedback accumulates, the exploration side should shrink and the warm cache hit-rate should rise.</div>`;
}

function renderSecAndKnapsack(d){
  const sec=d.security,ex=d.explain,dg=d.dep_graph;
  let panels='<div class="grid3" style="margin-bottom:20px;">';

  // Security panel with top findings
  panels+='<div class="panel"><div class="ph"><h2>🛡️ Security</h2><span id="sb" class="badge '+(sec&&(sec.critical_total||0)+(sec.high_total||0)>0?'b-rose':'b-green')+'">';
  if(sec&&!sec.error){
    const tot=(sec.critical_total||0)+(sec.high_total||0);
    panels+=tot>0?tot+' findings':'✓ Clean';
    panels+='</span></div><div class="pb">';
    if(tot===0){panels+='<div style="text-align:center;padding:16px;"><div style="font-size:40px;filter:drop-shadow(0 0 16px rgba(52,211,153,0.4));">🛡️</div><div style="color:var(--emerald);font-weight:700;margin-top:8px;">No vulnerabilities</div><div style="font-size:12px;color:var(--dim);margin-top:4px;">'+(sec.fragments_scanned||0)+' fragments scanned</div></div>';}
    else{
      const findings=sec.findings||sec.top_findings||[];
      const cats=sec.findings_by_category||{};
      if(findings.length>0){
        panels+=findings.slice(0,4).map(f=>`<div class="finding"><span class="finding-sev ${(f.severity||'').toLowerCase()==='critical'?'crit':'high'}">${(f.severity||'?')[0]}</span><div><div class="finding-file">${f.file||f.source||'unknown'}${f.line?':'+f.line:''}</div><div class="finding-desc">${f.message||f.category||''}</div></div></div>`).join('');
      } else {
        panels+=Object.entries(cats).map(([k,v])=>`<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;"><span style="color:var(--dim);">${k}</span><span class="tag t-rose">${v}</span></div>`).join('');
      }
    }
  } else {panels+='—</span></div><div class="pb"><div class="empty">No scan yet</div>';}
  panels+='</div></div>';

  // Dep graph - only show if data exists
  if(dg&&(dg.total_symbols||dg.symbol_count||0)>0){
    const sym=dg.total_symbols||dg.symbol_count||0,edg=dg.total_edges||dg.edge_count||0;
    panels+='<div class="panel"><div class="ph"><h2>🕸️ Dep Graph</h2><span class="badge b-cyan">'+sym+' symbols</span></div><div class="pb">';
    panels+='<div style="display:flex;gap:20px;justify-content:center;padding:12px 0;"><div style="text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--cyan);">'+sym+'</div><div style="font-size:10px;color:var(--dim);margin-top:4px;">SYMBOLS</div></div>';
    panels+='<div style="text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--blue);">'+edg+'</div><div style="font-size:10px;color:var(--dim);margin-top:4px;">EDGES</div></div></div>';
    panels+='</div></div>';
  }

  // Knapsack with included AND excluded
  if(ex&&!ex.error){
    const inc=ex.included||[],exc=ex.excluded||[];
    panels+='<div class="panel"><div class="ph"><h2>🎯 Knapsack</h2><span class="badge b-violet">'+inc.length+' selected · '+pct(ex.sufficiency)+' suff.</span></div>';
    panels+='<div class="pb" style="max-height:320px;overflow-y:auto;">';
    let rows=inc.slice(0,5).map(f=>{const sc=f.scores||{};return`<tr><td class="mono" style="color:var(--emerald);">✓ ${(f.source||f.id||'').split(/[\\/]/).pop()}</td><td class="mono">${pct(sc.composite)}</td><td style="font-size:11px;color:var(--dim);">${(f.reason||'').slice(0,30)}</td></tr>`;}).join('');
    rows+=exc.slice(0,3).map(f=>{const sc=f.scores||{};return`<tr style="opacity:0.5;"><td class="mono" style="color:var(--rose);">✗ ${(f.source||f.id||'').split(/[\\/]/).pop()}</td><td class="mono">${pct(sc.composite)}</td><td style="font-size:11px;color:var(--rose);">${(f.reason||'below threshold').slice(0,30)}</td></tr>`;}).join('');
    panels+='<table><thead><tr><th>Fragment</th><th>Score</th><th>Reason</th></tr></thead><tbody>'+rows+'</tbody></table>';
    panels+='</div></div>';
  }
  panels+='</div>';
  document.getElementById('grid3wrap').innerHTML=panels;
}

let sparkData=[];
function renderRequests(d){
  const reqs=d.recent_requests||[],tbody=document.getElementById('reqs'),b=document.getElementById('rb');
  b.textContent=reqs.length+' recent';
  if(reqs.length>0){reqs.forEach(r=>{if(sparkData.length>=30)sparkData.shift();sparkData.push(r.tokens_saved||0);});
    const mx=Math.max(...sparkData,1);
    document.getElementById('sparkarea').innerHTML='<div class="sparkline">'+sparkData.map(v=>'<div class="bar" style="height:'+Math.max(2,v/mx*40)+'px;"></div>').join('')+'</div>';}
  if(reqs.length===0){tbody.innerHTML='<tr><td colspan="7" class="empty">No requests yet — route LLM calls through proxy on :9377</td></tr>';return;}
  tbody.innerHTML=reqs.slice().reverse().slice(0,15).map(r=>`<tr>
    <td>${ago(r.time||0)}</td><td>${r.model||'—'}</td><td class="mono">${fmt(r.tokens_in||0)}</td>
    <td><span class="tag t-green">−${fmt(r.tokens_saved||0)}</span></td>
    <td>${(r.dedup_hits||0)>0?'<span class="tag t-amber">'+r.dedup_hits+'</span>':'<span style="color:var(--dim2)">0</span>'}</td>
    <td>${(r.sast_findings||0)>0?'<span class="tag t-rose">'+r.sast_findings+'</span>':'<span style="color:var(--dim2)">0</span>'}</td>
    <td style="max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dim);">${r.query||'—'}</td></tr>`).join('');
}

let trendsView='daily';
function renderValueTrends(d){
  const vt=d.value_trends,vc=d.value_confidence,el=document.getElementById('valueTrends');
  if(!el)return;
  if(!vt||!vc||(!vt.lifetime.tokens_saved&&!vc.session.tokens_saved)){el.innerHTML='';return;}
  const lt=vt.lifetime||{},sess=vc.session||{},today=vc.today||{};
  const status=vc.status||'idle';
  const statusColor=status==='active'?'var(--emerald)':'var(--dim)';
  const days_active=Math.max(1,Math.round((lt.last_seen-lt.first_seen)/86400));
  const daily_avg=lt.tokens_saved>0?(lt.cost_saved_usd/days_active):0;

  // Select data based on current view
  const chartData=trendsView==='daily'?vt.daily:trendsView==='weekly'?vt.weekly:vt.monthly;
  const maxTokens=Math.max(...chartData.map(d=>d.tokens_saved||0),1);
  const bars=chartData.map(d=>{
    const h=Math.max(2,((d.tokens_saved||0)/maxTokens)*72);
    const label=d.date||d.week||d.month||'';
    return '<div class="tbar cost" style="height:'+h+'px;" data-tip="'+label+': '+fmt(d.tokens_saved||0)+' tokens / $'+(d.cost_saved||0).toFixed(4)+'"></div>';
  }).join('');

  el.innerHTML='<div class="trends-panel"><div class="trends-header"><h2>Lifetime Value</h2>'+
    '<span class="badge" style="background:rgba(52,211,153,0.1);color:'+statusColor+';"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:'+statusColor+';margin-right:6px;'+(status==='active'?'box-shadow:0 0 8px var(--emerald);':'')+'"></span>'+status+'</span></div>'+
    '<div class="trends-body">'+
    '<div class="trends-kpis">'+
    '<div class="trends-kpi"><div class="trends-kpi-label">Lifetime Saved</div><div class="trends-kpi-val hv-green">$'+(lt.cost_saved_usd||0).toFixed(2)+'</div><div class="trends-kpi-sub">'+fmt(lt.tokens_saved||0)+' tokens across '+days_active+' day'+(days_active!==1?'s':'')+'</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">Today</div><div class="trends-kpi-val hv-blue">$'+(today.cost_saved_usd||0).toFixed(4)+'</div><div class="trends-kpi-sub">'+fmt(today.tokens_saved||0)+' tokens · '+(today.requests||0)+' reqs</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">This Session</div><div class="trends-kpi-val hv-amber">'+fmt(sess.tokens_saved||0)+'</div><div class="trends-kpi-sub">$'+(sess.cost_saved_usd||0).toFixed(4)+' · '+(sess.requests||0)+' reqs</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">Daily Average</div><div class="trends-kpi-val hv-green">$'+daily_avg.toFixed(4)+'</div><div class="trends-kpi-sub">'+(lt.requests_optimized||0)+' reqs optimized · '+(lt.duplicates_caught||0)+' dedup</div></div>'+
    '</div>'+
    '<div class="trends-tabs">'+
    '<div class="trends-tab'+(trendsView==='daily'?' active':'')+'" onclick="trendsView=\'daily\'">Daily</div>'+
    '<div class="trends-tab'+(trendsView==='weekly'?' active':'')+'" onclick="trendsView=\'weekly\'">Weekly</div>'+
    '<div class="trends-tab'+(trendsView==='monthly'?' active':'')+'" onclick="trendsView=\'monthly\'">Monthly</div></div>'+
    '<div class="trends-chart">'+bars+'</div>'+
    '</div></div>';
}

async function refresh(){
  try{const r=await fetch('/api/metrics');const d=await r.json();
    renderHero(d);renderValueTrends(d);renderBA(d);renderPrism(d);renderHealth(d);renderCache(d);renderSecAndKnapsack(d);renderRequests(d);
  }catch(e){console.error('Refresh:',e);}
}
refresh();setInterval(refresh,3000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress access logs

    def _send_security_headers(self):
        """Add security headers to prevent clickjacking and MIME sniffing."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.send_header("Referrer-Policy", "no-referrer")

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "http://localhost:9378")
            self.send_header("Cache-Control", "no-cache")
            self._send_security_headers()
            self.end_headers()
            snap = _get_full_snapshot()
            self.wfile.write(json.dumps(snap, default=str).encode())
        elif self.path == "/api/trends":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "http://localhost:9378")
            self.send_header("Cache-Control", "no-cache")
            self._send_security_headers()
            self.end_headers()
            try:
                from entroly.value_tracker import get_tracker
                data = get_tracker().get_trends()
            except Exception:
                data = {}
            self.wfile.write(json.dumps(data, default=str).encode())
        elif self.path == "/api/confidence":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "http://localhost:9378")
            self.send_header("Cache-Control", "no-cache")
            self._send_security_headers()
            self.end_headers()
            try:
                from entroly.value_tracker import get_tracker
                data = get_tracker().get_confidence()
            except Exception:
                data = {}
            self.wfile.write(json.dumps(data, default=str).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def start_dashboard(engine: Any = None, port: int = 9378, daemon: bool = True):
    """
    Start the dashboard HTTP server in a background thread.

    Args:
        engine: The EntrolyEngine instance to pull real data from.
        port: Port to serve on (default: 9378).
        daemon: Run as daemon thread (dies with main process).

    Returns:
        The HTTPServer instance.
    """
    global _engine
    _engine = engine

    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(f"Dashboard live at http://localhost:{port}")
    return server
