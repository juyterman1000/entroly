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

    # 7. Recent proxy requests
    with _lock:
        snap["recent_requests"] = list(_request_log)

    return snap


# ── HTML Dashboard ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly — Value Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #08090d; --bg2: #0d1117; --card: #131920; --card2: #161b22;
    --border: #21262d; --border2: #30363d;
    --text: #e6edf3; --dim: #7d8590; --dim2: #484f58;
    --accent: #58a6ff; --accent2: #388bfd;
    --green: #3fb950; --green-bg: rgba(63,185,80,0.08);
    --yellow: #d29922; --yellow-bg: rgba(210,153,34,0.08);
    --red: #f85149; --red-bg: rgba(248,81,73,0.08);
    --purple: #bc8cff; --purple-bg: rgba(188,140,255,0.08);
    --cyan: #39d2c0; --cyan-bg: rgba(57,210,192,0.08);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; overflow-x: hidden;
  }

  /* ── Top Bar ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 32px; border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }
  .topbar .brand {
    display: flex; align-items: center; gap: 12px;
  }
  .topbar .brand h1 {
    font-size: 22px; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .topbar .live {
    display: flex; align-items: center; gap: 8px;
    color: var(--green); font-size: 13px; font-weight: 500;
  }
  .topbar .live .dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* ── Layout ── */
  .main { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }

  /* ── Metric Cards ── */
  .hero-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 16px; margin-bottom: 28px;
  }
  .hero-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 24px; position: relative;
    overflow: hidden; transition: all 0.25s;
  }
  .hero-card:hover {
    border-color: var(--border2); transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
  }
  .hero-card .label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--dim); margin-bottom: 10px;
  }
  .hero-card .value {
    font-size: 36px; font-weight: 800; letter-spacing: -1px;
    font-feature-settings: 'tnum';
  }
  .hero-card .sub { font-size: 12px; color: var(--dim2); margin-top: 6px; }
  .hero-card::after {
    content: ''; position: absolute; top: 0; right: 0;
    width: 80px; height: 80px; border-radius: 0 14px 0 80px;
    opacity: 0.04;
  }
  .hero-card.green .value { color: var(--green); }
  .hero-card.green::after { background: var(--green); }
  .hero-card.accent .value { color: var(--accent); }
  .hero-card.accent::after { background: var(--accent); }
  .hero-card.purple .value { color: var(--purple); }
  .hero-card.purple::after { background: var(--purple); }
  .hero-card.yellow .value { color: var(--yellow); }
  .hero-card.yellow::after { background: var(--yellow); }
  .hero-card.red .value { color: var(--red); }
  .hero-card.red::after { background: var(--red); }

  /* ── Panel Grid ── */
  .panel-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px;
  }
  .panel-grid.three { grid-template-columns: 1fr 1fr 1fr; }
  .panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden;
  }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; border-bottom: 1px solid var(--border);
  }
  .panel-header h2 {
    font-size: 14px; font-weight: 700; letter-spacing: -0.2px;
  }
  .panel-header .badge {
    padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
  }
  .badge-green { background: var(--green-bg); color: var(--green); }
  .badge-yellow { background: var(--yellow-bg); color: var(--yellow); }
  .badge-red { background: var(--red-bg); color: var(--red); }
  .badge-purple { background: var(--purple-bg); color: var(--purple); }
  .badge-cyan { background: var(--cyan-bg); color: var(--cyan); }
  .panel-body { padding: 16px 20px; }

  /* ── PRISM Weight Bars ── */
  .weight-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 14px;
  }
  .weight-row:last-child { margin-bottom: 0; }
  .weight-label {
    width: 90px; font-size: 13px; font-weight: 500; color: var(--dim);
  }
  .weight-bar-bg {
    flex: 1; height: 28px; background: var(--bg2); border-radius: 6px;
    position: relative; overflow: hidden;
  }
  .weight-bar {
    height: 100%; border-radius: 6px; transition: width 0.6s ease;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 8px; font-size: 12px; font-weight: 700;
    color: rgba(255,255,255,0.9); min-width: 40px;
  }
  .w-recency { background: linear-gradient(90deg, #667eea, #764ba2); }
  .w-frequency { background: linear-gradient(90deg, #f093fb, #f5576c); }
  .w-semantic { background: linear-gradient(90deg, #4facfe, #00f2fe); }
  .w-entropy { background: linear-gradient(90deg, #43e97b, #38f9d7); }

  /* ── Health Gauge ── */
  .health-gauge {
    display: flex; align-items: center; justify-content: center;
    gap: 24px; padding: 16px 0;
  }
  .health-grade {
    width: 100px; height: 100px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 48px; font-weight: 800;
    border: 4px solid var(--border);
  }
  .grade-A { border-color: var(--green); color: var(--green); }
  .grade-B { border-color: var(--accent); color: var(--accent); }
  .grade-C { border-color: var(--yellow); color: var(--yellow); }
  .grade-D { border-color: #e3872d; color: #e3872d; }
  .grade-F { border-color: var(--red); color: var(--red); }
  .health-details { list-style: none; }
  .health-details li {
    font-size: 13px; color: var(--dim); padding: 4px 0;
    display: flex; align-items: center; gap: 8px;
  }
  .health-details li span { font-weight: 600; color: var(--text); min-width: 24px; text-align: right; }

  /* ── Tables ── */
  table { width: 100%; border-collapse: collapse; }
  th {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--dim2); padding: 10px 14px; text-align: left;
    background: var(--bg2); font-weight: 600;
  }
  td {
    padding: 8px 14px; border-top: 1px solid var(--border);
    font-size: 13px;
  }
  td.mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
  tr:hover td { background: rgba(255,255,255,0.02); }

  /* ── Tags ── */
  .tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
  }
  .tag-green { background: var(--green-bg); color: var(--green); }
  .tag-red { background: var(--red-bg); color: var(--red); }
  .tag-yellow { background: var(--yellow-bg); color: var(--yellow); }
  .tag-purple { background: var(--purple-bg); color: var(--purple); }
  .tag-cyan { background: var(--cyan-bg); color: var(--cyan); }

  /* ── Empty state ── */
  .empty {
    text-align: center; padding: 32px; color: var(--dim2); font-size: 13px;
  }

  /* ── Responsive ── */
  @media (max-width: 1100px) {
    .hero-grid { grid-template-columns: repeat(3, 1fr); }
    .panel-grid.three { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 768px) {
    .hero-grid { grid-template-columns: 1fr 1fr; }
    .panel-grid, .panel-grid.three { grid-template-columns: 1fr; }
    .main { padding: 16px; }
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <h1>⚡ Entroly</h1>
    <span style="color: var(--dim); font-size: 13px;">Value Dashboard</span>
  </div>
  <div class="live"><div class="dot"></div> Live · auto-refresh 3s</div>
</div>

<div class="main">
  <!-- Hero metrics -->
  <div class="hero-grid" id="hero"></div>

  <!-- Row: PRISM Weights + Health -->
  <div class="panel-grid">
    <div class="panel">
      <div class="panel-header">
        <h2>🧠 PRISM RL Weights</h2>
        <span class="badge badge-purple">Learned</span>
      </div>
      <div class="panel-body" id="prism"></div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <h2>🏥 Code Health</h2>
        <span id="health-badge" class="badge badge-green">—</span>
      </div>
      <div class="panel-body" id="health"></div>
    </div>
  </div>

  <!-- Row: Security + Dep Graph + Knapsack -->
  <div class="panel-grid three">
    <div class="panel">
      <div class="panel-header">
        <h2>🛡️ Security Scan</h2>
        <span id="sec-badge" class="badge badge-green">Clean</span>
      </div>
      <div class="panel-body" id="security"></div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <h2>🕸️ Dependency Graph</h2>
        <span id="dep-badge" class="badge badge-cyan">—</span>
      </div>
      <div class="panel-body" id="depgraph"></div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <h2>🎯 Knapsack Decisions</h2>
        <span id="knapsack-badge" class="badge badge-purple">—</span>
      </div>
      <div class="panel-body" id="knapsack" style="max-height:320px;overflow-y:auto;"></div>
    </div>
  </div>

  <!-- Recent Requests -->
  <div class="panel" style="margin-bottom: 28px;">
    <div class="panel-header">
      <h2>📡 Recent Proxy Requests</h2>
      <span id="req-badge" class="badge badge-cyan">—</span>
    </div>
    <div style="overflow-x: auto;">
      <table>
        <thead><tr>
          <th>Time</th><th>Model</th><th>Tokens In</th><th>Saved</th>
          <th>Dedup</th><th>SAST</th><th>Query</th>
        </tr></thead>
        <tbody id="requests"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
function fmt(n) { if(n==null) return '—'; return n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n); }
function money(n) { return '$' + (n||0).toFixed(2); }
function pct(n) { return Math.round((n||0)*100) + '%'; }
function ago(ts) {
  const s = Math.floor(Date.now()/1000 - ts);
  if(s<60) return s+'s ago';
  if(s<3600) return Math.floor(s/60)+'m ago';
  return Math.floor(s/3600)+'h ago';
}

function renderHero(d) {
  const s = d.stats || {};
  const savings = s.savings || {};
  const session = s.session || {};
  const dedup = s.dedup || {};

  const tokensSaved = savings.total_tokens_saved || 0;
  const costSaved = savings.estimated_cost_saved_usd || 0;
  const totalFrags = session.total_fragments || 0;
  const dupsCaught = savings.total_duplicates_caught || dedup.duplicates_detected || 0;
  const avgEntropy = session.avg_entropy || 0;

  document.getElementById('hero').innerHTML = `
    <div class="hero-card green">
      <div class="label">Tokens Saved</div>
      <div class="value">${fmt(tokensSaved)}</div>
      <div class="sub">${fmt(savings.total_optimizations||0)} optimizations run</div>
    </div>
    <div class="hero-card green">
      <div class="label">Cost Saved</div>
      <div class="value">${money(costSaved)}</div>
      <div class="sub">at $3/1M context tokens</div>
    </div>
    <div class="hero-card accent">
      <div class="label">Fragments Indexed</div>
      <div class="value">${fmt(totalFrags)}</div>
      <div class="sub">${fmt(session.total_tokens_tracked||0)} tokens tracked</div>
    </div>
    <div class="hero-card yellow">
      <div class="label">Duplicates Caught</div>
      <div class="value">${dupsCaught}</div>
      <div class="sub">SimHash dedup engine</div>
    </div>
    <div class="hero-card purple">
      <div class="label">Avg Entropy</div>
      <div class="value">${(avgEntropy||0).toFixed(3)}</div>
      <div class="sub">information density score</div>
    </div>
  `;
}

function renderPrism(d) {
  const w = d.prism_weights;
  if(!w) { document.getElementById('prism').innerHTML = '<div class="empty">Engine not initialized</div>'; return; }
  const max = Math.max(w.recency, w.frequency, w.semantic, w.entropy, 0.01);
  const bar = (cls, val) => `<div class="weight-bar ${cls}" style="width:${(val/0.8)*100}%">${pct(val)}</div>`;

  document.getElementById('prism').innerHTML = `
    <div class="weight-row">
      <span class="weight-label">Recency</span>
      <div class="weight-bar-bg">${bar('w-recency', w.recency)}</div>
    </div>
    <div class="weight-row">
      <span class="weight-label">Frequency</span>
      <div class="weight-bar-bg">${bar('w-frequency', w.frequency)}</div>
    </div>
    <div class="weight-row">
      <span class="weight-label">Semantic</span>
      <div class="weight-bar-bg">${bar('w-semantic', w.semantic)}</div>
    </div>
    <div class="weight-row">
      <span class="weight-label">Entropy</span>
      <div class="weight-bar-bg">${bar('w-entropy', w.entropy)}</div>
    </div>
    <div style="margin-top:12px;font-size:12px;color:var(--dim);">
      Weights learn via PRISM spectral RL — updated after each success/failure signal.
    </div>
  `;
}

function renderHealth(d) {
  const h = d.health;
  const el = document.getElementById('health');
  const badge = document.getElementById('health-badge');
  if(!h || h.error) {
    el.innerHTML = '<div class="empty">Ingest code to see health analysis</div>';
    return;
  }
  const grade = h.health_grade || '?';
  const score = h.code_health_score || 0;
  const gc = 'grade-' + grade;

  badge.textContent = grade + ' · ' + score + '/100';
  badge.className = 'badge ' + (grade==='A'?'badge-green':grade==='B'?'badge-green':grade==='C'?'badge-yellow':'badge-red');

  el.innerHTML = `
    <div class="health-gauge">
      <div class="health-grade ${gc}">${grade}</div>
      <ul class="health-details">
        <li><span>${(h.clone_pairs||[]).length}</span> clone pairs detected</li>
        <li><span>${(h.dead_symbols||[]).length}</span> dead symbols</li>
        <li><span>${(h.god_files||[]).length}</span> god files (over-coupled)</li>
        <li><span>${(h.arch_violations||[]).length}</span> architecture violations</li>
        <li><span>${(h.naming_issues||[]).length}</span> naming convention issues</li>
      </ul>
    </div>
    ${h.top_recommendation ? '<div style="margin-top:8px;padding:10px;background:var(--bg2);border-radius:8px;font-size:12px;color:var(--yellow);">💡 '+h.top_recommendation+'</div>' : ''}
  `;
}

function renderSecurity(d) {
  const s = d.security;
  const el = document.getElementById('security');
  const badge = document.getElementById('sec-badge');
  if(!s || s.error) {
    el.innerHTML = '<div class="empty">No fragments scanned yet</div>';
    return;
  }

  const total = (s.critical_total||0) + (s.high_total||0);
  const cats = s.findings_by_category || {};

  if(total === 0 && Object.keys(cats).length === 0) {
    badge.textContent = '✓ Clean';
    badge.className = 'badge badge-green';
    el.innerHTML = `
      <div style="text-align:center;padding:20px;">
        <div style="font-size:40px;margin-bottom:8px;">🛡️</div>
        <div style="color:var(--green);font-weight:600;">No vulnerabilities found</div>
        <div style="color:var(--dim);font-size:12px;margin-top:4px;">${s.fragments_scanned||0} fragments scanned</div>
      </div>
    `;
    return;
  }

  badge.textContent = total + ' findings';
  badge.className = 'badge ' + (s.critical_total > 0 ? 'badge-red' : 'badge-yellow');

  let catHtml = Object.entries(cats).map(([k,v]) =>
    `<li style="display:flex;justify-content:space-between;padding:4px 0;">
      <span style="color:var(--dim);font-size:12px;">${k}</span>
      <span class="tag tag-red">${v}</span>
    </li>`
  ).join('');

  el.innerHTML = `
    <div style="display:flex;gap:20px;margin-bottom:12px;">
      <div style="text-align:center;flex:1;">
        <div style="font-size:28px;font-weight:800;color:var(--red);">${s.critical_total||0}</div>
        <div style="font-size:11px;color:var(--dim);">Critical</div>
      </div>
      <div style="text-align:center;flex:1;">
        <div style="font-size:28px;font-weight:800;color:var(--yellow);">${s.high_total||0}</div>
        <div style="font-size:11px;color:var(--dim);">High</div>
      </div>
      <div style="text-align:center;flex:1;">
        <div style="font-size:28px;font-weight:800;color:var(--accent);">${s.fragments_with_findings||0}</div>
        <div style="font-size:11px;color:var(--dim);">Files</div>
      </div>
    </div>
    <ul style="list-style:none;">${catHtml}</ul>
    ${s.most_vulnerable_fragment ? '<div style="margin-top:8px;font-size:11px;color:var(--dim);">🔴 Most vulnerable: <span style="color:var(--red);">'+s.most_vulnerable_fragment+'</span></div>' : ''}
  `;
}

function renderDepGraph(d) {
  const dg = d.dep_graph;
  const el = document.getElementById('depgraph');
  const badge = document.getElementById('dep-badge');
  if(!dg) {
    el.innerHTML = '<div class="empty">Run optimize to build dep graph</div>';
    return;
  }

  const symbols = dg.total_symbols || dg.symbol_count || 0;
  const edges = dg.total_edges || dg.edge_count || 0;
  const files = dg.total_files || dg.file_count || 0;

  badge.textContent = symbols + ' symbols';
  badge.className = 'badge badge-cyan';

  el.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div style="text-align:center;padding:16px 0;">
        <div style="font-size:28px;font-weight:800;color:var(--cyan);">${symbols}</div>
        <div style="font-size:11px;color:var(--dim);">Symbols</div>
      </div>
      <div style="text-align:center;padding:16px 0;">
        <div style="font-size:28px;font-weight:800;color:var(--accent);">${edges}</div>
        <div style="font-size:11px;color:var(--dim);">Edges</div>
      </div>
    </div>
    <div style="font-size:12px;color:var(--dim);text-align:center;margin-top:8px;">
      Cross-file dependency tracking for knapsack coherence
    </div>
  `;
}

function renderKnapsack(d) {
  const ex = d.explain;
  const el = document.getElementById('knapsack');
  const badge = document.getElementById('knapsack-badge');
  if(!ex || ex.error) {
    el.innerHTML = '<div class="empty">Run optimize to see knapsack decisions</div>';
    return;
  }

  const inc = ex.included || [];
  const exc = ex.excluded || [];
  const suff = ex.sufficiency;
  badge.textContent = inc.length + ' selected · ' + pct(suff) + ' sufficiency';
  badge.className = 'badge badge-purple';

  let rows = inc.slice(0, 8).map(f => {
    const s = f.scores || {};
    return `<tr>
      <td class="mono" style="color:var(--green);">✓ ${(f.source||f.id||'').split('/').pop()}</td>
      <td class="mono">${pct(s.composite)}</td>
      <td class="mono" style="color:var(--dim);">${s.criticality||'—'}</td>
      <td style="font-size:11px;color:var(--dim);">${(f.reason||'').slice(0,40)}</td>
    </tr>`;
  }).join('');

  rows += exc.slice(0, 4).map(f => {
    const s = f.scores || {};
    return `<tr style="opacity:0.5;">
      <td class="mono" style="color:var(--red);">✗ ${(f.source||f.id||'').split('/').pop()}</td>
      <td class="mono">${pct(s.composite)}</td>
      <td class="mono" style="color:var(--dim);">${s.criticality||'—'}</td>
      <td style="font-size:11px;color:var(--dim);">${(f.reason||'').slice(0,40)}</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <table>
      <thead><tr><th>Fragment</th><th>Score</th><th>Crit</th><th>Reason</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4" class="empty">No fragments yet</td></tr>'}</tbody>
    </table>
  `;
}

function renderRequests(d) {
  const reqs = d.recent_requests || [];
  const tbody = document.getElementById('requests');
  const badge = document.getElementById('req-badge');
  badge.textContent = reqs.length + ' recent';

  if(reqs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No requests yet — start using your IDE with Entroly proxy on :9377</td></tr>';
    return;
  }

  tbody.innerHTML = reqs.slice().reverse().map(r => `
    <tr>
      <td>${ago(r.time||0)}</td>
      <td>${r.model||'—'}</td>
      <td class="mono">${fmt(r.tokens_in||0)}</td>
      <td><span class="tag tag-green">−${fmt(r.tokens_saved||0)}</span></td>
      <td>${(r.dedup_hits||0)>0 ? '<span class="tag tag-yellow">'+r.dedup_hits+'</span>' : '<span style="color:var(--dim2)">0</span>'}</td>
      <td>${(r.sast_findings||0)>0 ? '<span class="tag tag-red">'+r.sast_findings+'</span>' : '<span style="color:var(--dim2)">0</span>'}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dim);">${r.query||'—'}</td>
    </tr>
  `).join('');
}

async function refresh() {
  try {
    const r = await fetch('/api/metrics');
    const d = await r.json();
    renderHero(d);
    renderPrism(d);
    renderHealth(d);
    renderSecurity(d);
    renderDepGraph(d);
    renderKnapsack(d);
    renderRequests(d);
  } catch(e) { console.error('Refresh failed:', e); }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress access logs

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            snap = _get_full_snapshot()
            self.wfile.write(json.dumps(snap, default=str).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
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

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(f"Dashboard live at http://localhost:{port}")
    return server
