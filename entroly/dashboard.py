"""
Entroly Live Dashboard â€” Real-time AI value metrics at localhost:9378
=====================================================================

Shows developers exactly what Entroly's Rust engine is doing for them,
pulling REAL data from all engine subsystems:

  Engine Stats:       tokens saved, cost saved, dedup hits, turn count
  PRISM RL Weights:   learned scoring weights (recency/frequency/semantic/entropy)
  Health Analysis:    code health grade Aâ€“F, clones, dead symbols, god files
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
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger("entroly.dashboard")

# â”€â”€ Engine reference (set by start_dashboard) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_engine: Any | None = None
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

def _control_learning_snapshot(daemon: Any) -> dict:
    """Control API payload for /api/control/learning."""
    if hasattr(daemon, "get_learning_stats"):
        try:
            return _safe_json(daemon.get_learning_stats())
        except Exception:
            pass
    state = getattr(daemon, "state", None)
    return _safe_json({
        "local_enabled": getattr(state, "learning_enabled", True),
        "autotune_enabled": getattr(state, "autotune_enabled", True),
        "weights": daemon.get_learning_weights() if hasattr(daemon, "get_learning_weights") else {},
    })

def _record_section_error(snap: dict, section: str, exc: BaseException) -> None:
    """Append a structured error so the dashboard JS can render a banner.
    Per-section error fields (`snap[section] = {"error": str(e)}` or `None`)
    are kept for backward-compat, but they're invisible to the user â€” the
    `errors` array is what gets surfaced in the UI."""
    snap.setdefault("errors", []).append({
        "section": section,
        "type": type(exc).__name__,
        "message": str(exc) or repr(exc),
    })


def _cogops_unavailable_snapshot(reason: str) -> dict:
    """Dashboard-safe CogOps payload when the optional native module is absent."""
    return _safe_json({
        "total_beliefs": 0,
        "verified": 0,
        "stale": 0,
        "doc_beliefs": 0,
        "avg_confidence": 0.0,
        "freshness_pct": 100.0,
        "entity_count": 0,
        "engine": "unavailable",
        "status": "native_module_missing",
        "hint": (
            "CogOps is degraded: the native engine 'entroly_core' isn't "
            "installed. Everything else works. To enable it, reinstall "
            "with the native extra â€” pip: `pip install 'entroly[full]'`; "
            "uv: `uv tool install 'entroly[full]'` (or `uv tool install "
            "entroly --with entroly-core`). If no prebuilt wheel exists "
            "for your platform, a Rust toolchain is required to build it."
        ),
        "reason": reason,
    })


def _get_full_snapshot() -> dict:
    """Pull ALL real data from the engine subsystems."""
    snap: dict[str, Any] = {
        "ts": time.time(),
        "engine_available": _engine is not None,
        "errors": [],
    }

    # Persistent value tracker (independent of engine â€” always available).
    # reload_if_changed() is THE cross-process fix: a standalone
    # `entroly dashboard` is a different process from the proxy/MCP/npm
    # writer, so without this its in-memory copy froze at startup and the
    # page looked permanently blank/stale. One cheap stat() per poll.
    try:
        from entroly.value_tracker import get_tracker
        tracker = get_tracker()
        tracker.reload_if_changed()
        snap["value_trends"] = tracker.get_trends()
        snap["value_confidence"] = tracker.get_confidence()
        # Persistent, cross-process activity feed. Merge any same-process
        # proxy in-memory rows (richer) ahead of the disk feed, dedup by
        # (ts, summary) so we never double-count an event.
        disk_activity = snap["value_trends"].get("activity", []) \
            if isinstance(snap.get("value_trends"), dict) else []
        with _lock:
            mem = list(reversed(_request_log))
        seen = set()
        merged = []
        for row in (mem + disk_activity):
            key = (round(float(row.get("ts", 0)), 1),
                   str(row.get("summary", row.get("model", ""))))
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        snap["activity"] = _safe_json(merged[:50])
        snap["data_source"] = "disk+memory (live)"
        # Map the cross-process feed into the shape the existing
        # renderRequests()/hero sparkline already consume, so EVERY
        # install mode (npm/SDK/MCP/proxy) lights up the live panel
        # with zero JS changes. The engine path may override this with
        # richer same-process proxy rows below (only if it has any).
        snap["recent_requests"] = _safe_json([
            {
                "time": row.get("ts", row.get("time", 0)),
                "model": row.get("model", "") or row.get("kind", ""),
                "tokens_in": row.get("tokens_in", 0),
                "tokens_saved": row.get("tokens_saved", 0),
                "dedup_hits": row.get("duplicates", row.get("dedup_hits", 0)),
                "sast_findings": row.get("sast_findings", 0),
                "query": row.get("summary", row.get("query", "")),
            }
            for row in merged[:50]
        ])
    except Exception as e:
        snap["value_trends"] = None
        snap["value_confidence"] = None
        snap["activity"] = []
        _record_section_error(snap, "value_tracker", e)

    if _engine is None:
        # No Rust engine (npm/SDK-only users, or dashboard launched
        # standalone). This is NOT an error and must NOT render blank:
        # the value tiles + activity above are the whole point for these
        # users. Signal a friendly zero/degraded state instead.
        lt = (snap.get("value_confidence") or {}).get("lifetime", {}) \
            if isinstance(snap.get("value_confidence"), dict) else {}
        snap["engine_state"] = {
            "available": False,
            "reason": "native engine not running in this process",
            "has_value_data": bool(
                lt.get("tokens_saved") or lt.get("hallucinations_blocked")
            ),
            "hint": ("Value metrics below are live from the shared "
                     "telemetry file written by your MCP server / proxy / "
                     "npm runtime. Engine-only panels (health, SAST, "
                     "knapsack) need the in-process Rust engine."),
        }
        return snap

    try:
        # 1. Core stats â€” engine telemetry only. NEVER prices "$ saved" from
        # these numbers; the only source of truth for money saved is
        # value_trends.lifetime (proxy-only). The Rust struct historically
        # exposed a `savings` block with a fabricated `estimated_cost_saved_usd`
        # field that conflated CLI dedup with real LLM savings â€” we normalize
        # that block to `engine` here and strip the misleading cost field.
        if hasattr(_engine, "_rust") and _engine._rust is not None:
            stats = _engine._rust.stats()
            stats = dict(stats)
            for k, v in stats.items():
                if hasattr(v, "items"):
                    stats[k] = dict(v)
        elif hasattr(_engine, "stats"):
            stats = _engine.stats()
        else:
            stats = {}
        if isinstance(stats, dict) and "savings" in stats:
            sv = stats.pop("savings")
            if isinstance(sv, dict):
                stats["engine"] = {
                    "dedup_tokens_avoided": sv.get("total_tokens_saved", 0),
                    "duplicates_caught": sv.get("total_duplicates_caught", 0),
                    "optimize_calls": sv.get("total_optimizations", 0),
                    "fragments_ingested": sv.get("total_fragments_ingested", 0),
                }
        snap["stats"] = _safe_json(stats)
    except Exception as e:
        snap["stats"] = {"error": str(e)}
        _record_section_error(snap, "stats", e)

    try:
        # 2. PRISM RL weights â€” the learned scoring weights
        if hasattr(_engine, "_rust") and _engine._rust is not None:
            rust = _engine._rust
            snap["prism_weights"] = {
                "recency": round(getattr(rust, "w_recency", 0.3), 4),
                "frequency": round(getattr(rust, "w_frequency", 0.25), 4),
                "semantic": round(getattr(rust, "w_semantic", 0.25), 4),
                "entropy": round(getattr(rust, "w_entropy", 0.2), 4),
            }
    except Exception as e:
        snap["prism_weights"] = None
        _record_section_error(snap, "prism_weights", e)

    try:
        # 3. Health analysis â€” code health grade
        if hasattr(_engine, "_rust") and _engine._rust is not None:
            health_json = _engine._rust.analyze_health()
            snap["health"] = _safe_json(json.loads(health_json))
    except Exception as e:
        snap["health"] = None
        _record_section_error(snap, "health", e)

    try:
        # 4. SAST security report
        if hasattr(_engine, "_rust") and _engine._rust is not None:
            sec_json = _engine._rust.security_report()
            snap["security"] = _safe_json(json.loads(sec_json))
    except Exception as e:
        snap["security"] = None
        _record_section_error(snap, "security", e)

    try:
        # 5. Knapsack explainability â€” last optimization decisions
        if hasattr(_engine, "_rust") and _engine._rust is not None:
            explain = _engine._rust.explain_selection()
            snap["explain"] = _safe_json(dict(explain))
    except Exception as e:
        snap["explain"] = None
        _record_section_error(snap, "explain", e)

    try:
        # 6. Dependency graph stats
        if hasattr(_engine, "_rust") and _engine._rust is not None:
            dg = _engine._rust.dep_graph_stats()
            snap["dep_graph"] = _safe_json(dict(dg))
    except Exception as e:
        snap["dep_graph"] = None
        _record_section_error(snap, "dep_graph", e)

    try:
        # 7. Cache intelligence â€” live EGSC + RAVEN-UCB observability
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
    except Exception as e:
        snap["cache_intelligence"] = None
        _record_section_error(snap, "cache_intelligence", e)

    try:
        # 8. Context Resonance + Coverage + Consolidation stats
        stats = snap.get("stats", {}) if isinstance(snap.get("stats"), dict) else {}
        resonance = stats.get("resonance", {}) if isinstance(stats.get("resonance"), dict) else {}
        consolidation = stats.get("consolidation", {}) if isinstance(stats.get("consolidation"), dict) else {}
        snap["resonance"] = _safe_json({
            "tracked_pairs": resonance.get("tracked_pairs", 0),
            "mean_strength": resonance.get("mean_strength", 0.0),
            "w_resonance": resonance.get("w_resonance", 0.0),
            "resonance_energy_fraction": resonance.get("resonance_energy_fraction", 0.0),
            "is_calibrated": resonance.get("is_calibrated", False),
            "condition_number_5d": resonance.get("condition_number_5d", 1.0),
        })
        snap["consolidation"] = _safe_json({
            "total_consolidations": consolidation.get("total_consolidations", 0),
            "tokens_saved": consolidation.get("tokens_saved", 0),
        })
        # Causal Context Graph stats
        causal = stats.get("causal", {}) if isinstance(stats.get("causal"), dict) else {}
        snap["causal"] = _safe_json({
            "total_traces": causal.get("total_traces", 0),
            "tracked_fragments": causal.get("tracked_fragments", 0),
            "interventional_fragments": causal.get("interventional_fragments", 0),
            "temporal_links": causal.get("temporal_links", 0),
            "gravity_sources": causal.get("gravity_sources", 0),
            "mean_causal_mass": causal.get("mean_causal_mass", 0.0),
            "base_rate": causal.get("base_rate", 0.0),
        })
    except Exception as e:
        snap["resonance"] = None
        snap["consolidation"] = None
        snap["causal"] = None
        _record_section_error(snap, "resonance/consolidation/causal", e)

    # 9. Recent proxy requests â€” only override the activity-derived feed
    # when this process actually has richer same-process proxy rows;
    # otherwise keep the cross-process feed set above (don't blank it).
    with _lock:
        if _request_log:
            snap["recent_requests"] = list(_request_log)

    # 9b. WITNESS sidecar certificates from the live proxy, when available.
    snap["witness"] = _fetch_witness_snapshot()

    # 10. CogOps Epistemic Engine stats
    try:
        import os

        from entroly_core import CogOpsEngine
        vault_base = os.environ.get(
            "ENTROLY_VAULT",
            os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
        )
        _cogops = CogOpsEngine(vault_base)  # noqa: F841 â€” side-effect: initializes vault

        # Read all beliefs for summary stats
        from pathlib import Path
        beliefs_dir = Path(vault_base) / "beliefs"
        total_beliefs = 0
        verified = 0
        stale = 0
        doc_beliefs = 0
        avg_confidence = 0.0
        entities = []
        if beliefs_dir.exists():
            for md in beliefs_dir.rglob("*.md"):
                try:
                    content = md.read_text(encoding="utf-8", errors="replace")
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = parts[1]
                        total_beliefs += 1
                        entity = ""
                        conf = 0.5
                        status = "inferred"
                        for line in fm.splitlines():
                            t = line.strip()
                            if t.startswith("entity:"):
                                entity = t[7:].strip()
                            elif t.startswith("confidence:"):
                                try:
                                    conf = float(t[11:].strip())
                                except ValueError:
                                    pass
                            elif t.startswith("status:"):
                                status = t[7:].strip()
                        avg_confidence += conf
                        if status == "verified":
                            verified += 1
                        elif status == "stale":
                            stale += 1
                        if entity.startswith("doc/"):
                            doc_beliefs += 1
                        entities.append(entity)
                except Exception:
                    pass
        if total_beliefs > 0:
            avg_confidence /= total_beliefs

        snap["cogops"] = _safe_json({
            "total_beliefs": total_beliefs,
            "verified": verified,
            "stale": stale,
            "doc_beliefs": doc_beliefs,
            "avg_confidence": avg_confidence,
            "freshness_pct": round((1 - stale / max(total_beliefs, 1)) * 100, 1),
            "entity_count": len(set(entities)),
            "engine": "rust",
        })
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") == "entroly_core":
            snap["cogops"] = _cogops_unavailable_snapshot(str(e))
        else:
            snap["cogops"] = None
            _record_section_error(snap, "cogops", e)
    except ImportError as e:
        if "entroly_core" in str(e):
            snap["cogops"] = _cogops_unavailable_snapshot(str(e))
        else:
            snap["cogops"] = None
            _record_section_error(snap, "cogops", e)
    except Exception as e:
        snap["cogops"] = None
        _record_section_error(snap, "cogops", e)

    return snap


def _fetch_witness_snapshot() -> dict[str, Any]:
    """Read WITNESS certificate UX state from the local proxy.

    The dashboard runs on its own port, so it queries the proxy over localhost
    with a short timeout and degrades to an idle state when the proxy is off.
    """
    import os
    import urllib.error
    import urllib.request

    ports = [os.environ.get("ENTROLY_PROXY_PORT", "9377"), "9399"]
    seen: set[str] = set()
    for port in ports:
        if port in seen:
            continue
        seen.add(port)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/witness?limit=8", timeout=0.25) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                payload["proxy_port"] = int(port)
                payload["available"] = True
                return payload
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
            continue
    return {"available": False, "count": 0, "items": [], "feedback": {}}


# â”€â”€ HTML Dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly â€” Intelligence Dashboard</title>
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
/* Hero â€” Impact-first design */
.hero{margin-bottom:20px;}
.hero-impact{text-align:center;padding:32px 24px 24px;background:var(--card);border:1px solid var(--border);
  border-radius:20px;position:relative;overflow:hidden;margin-bottom:16px;}
.hero-impact::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--grad2);}
.hero-impact::after{content:'';position:absolute;top:50%;left:50%;width:400px;height:400px;transform:translate(-50%,-50%);
  background:radial-gradient(circle,rgba(52,211,153,0.06),transparent 70%);pointer-events:none;}
.hero-big{font-size:64px;font-weight:900;letter-spacing:-3px;font-feature-settings:'tnum';
  background:var(--grad2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 30px rgba(52,211,153,0.2));line-height:1.1;}
.hero-big-label{font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:2px;margin-top:6px;font-weight:600;}
.hero-subtitle{font-size:13px;color:var(--dim);margin-top:12px;}
.hero-subtitle b{color:var(--text);font-weight:700;}
.hero-spark{height:48px;display:flex;align-items:flex-end;gap:2px;margin:16px auto 0;max-width:500px;}
.hero-spark .hbar{flex:1;background:var(--grad2);border-radius:3px 3px 0 0;min-width:4px;
  transition:height 0.6s cubic-bezier(0.16,1,0.3,1);opacity:0.6;}
.hero-spark .hbar:last-child{opacity:1;}
.hero-spark .hbar:hover{opacity:1;}
.hero-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
.hero-metric{padding:16px 20px;background:var(--card);border:1px solid var(--border);border-radius:14px;
  position:relative;overflow:hidden;transition:border-color 0.3s,transform 0.2s;}
.hero-metric:hover{border-color:var(--border2);transform:translateY(-2px);}
.hero-metric::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.hm-files::before{background:var(--grad2);}
.hm-health::before{background:linear-gradient(90deg,var(--blue),var(--violet));}
.hm-sast::before{background:linear-gradient(90deg,var(--rose),var(--amber));}
.hm-reqs::before{background:linear-gradient(90deg,var(--cyan),var(--blue));}
.hm-icon{font-size:20px;margin-bottom:6px;}
.hm-val{font-size:28px;font-weight:900;letter-spacing:-1px;font-feature-settings:'tnum';}
.hm-label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1.2px;margin-top:2px;}
.hm-sub{font-size:11px;color:var(--dim);margin-top:6px;}
.hv-green{color:var(--emerald);}
.hv-blue{color:var(--blue);}
.hv-rose{color:var(--rose);}
.hv-amber{color:var(--amber);}
.hv-cyan{color:var(--cyan);}
/* Empty state */
.empty-hero{text-align:center;padding:48px 24px;background:var(--card);border:1px solid var(--border);
  border-radius:20px;margin-bottom:20px;}
.empty-icon{font-size:48px;margin-bottom:12px;filter:grayscale(0.5);opacity:0.7;}
.empty-title{font-size:18px;font-weight:700;color:var(--text);margin-bottom:8px;}
.empty-desc{font-size:13px;color:var(--dim);max-width:400px;margin:0 auto;line-height:1.6;}
.empty-desc code{background:rgba(96,165,250,0.15);padding:2px 8px;border-radius:4px;
  font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--blue);}
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
@media(max-width:1100px){.hero-metrics{grid-template-columns:1fr 1fr;}.grid3{grid-template-columns:1fr 1fr;}.ba-panel{grid-template-columns:1fr;}.cache-kpis{grid-template-columns:1fr 1fr;}.cache-split{grid-template-columns:1fr;}}
@media(max-width:768px){.hero-metrics,.grid2,.grid3,.cache-kpis{grid-template-columns:1fr;}.main{padding:16px;}.hero-big{font-size:48px;}}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand"><h1>âš¡ Entroly</h1><span class="btag">INTELLIGENCE DASHBOARD</span></div>
  <div style="display:flex;align-items:center;gap:16px;">
    <a href="/" style="color:var(--emerald);text-decoration:none;font-size:13px;font-weight:600;">Dashboard</a>
    <a href="/controls" style="color:var(--dim);text-decoration:none;font-size:13px;font-weight:500;transition:color .2s;" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--dim)'">âš™ï¸ Controls</a>
    <div class="live"><div class="dot"></div>Live Â· 3s refresh</div>
  </div>
</div>
<div class="main">
  <div class="whisper" id="whisper">
    ðŸ’¡ Point your AI tool's API base URL to <code>http://localhost:9377/v1</code> to start optimizing every LLM call.
    <span class="dismiss" onclick="this.parentElement.style.display='none'">âœ•</span>
  </div>
  <div id="errBanner" style="display:none;margin-bottom:16px;padding:12px 16px;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.4);border-radius:10px;color:#fecaca;font-size:13px;line-height:1.5;"></div>
  <div class="hero" id="hero"></div>
  <div id="valueTrends"></div>
  <div id="ba"></div>
  <div class="grid2">
    <div class="panel">
      <div class="ph"><h2>ðŸ§  PRISM Intelligence</h2><span id="pb" class="badge b-amber">Default</span></div>
      <div class="pb" id="prism"></div>
    </div>
    <div class="panel">
      <div class="ph"><h2>ðŸ¥ Code Health</h2><span id="hb" class="badge b-green">â€”</span></div>
      <div class="pb" id="health"></div>
    </div>
  </div>
  <div class="panel" style="margin-bottom:20px;">
    <div class="ph"><h2>Cache Intelligence</h2><span id="cb" class="badge b-blue">â€”</span></div>
    <div class="pb" id="cacheintel"></div>
  </div>
  <div class="panel" style="margin-bottom:20px;">
    <div class="ph"><h2>ðŸ§  Epistemic Engine</h2><span id="cogb" class="badge b-violet">CogOps</span></div>
    <div class="pb" id="cogops"></div>
  </div>
  <div class="panel" style="margin-bottom:20px;">
    <div class="ph"><h2>WITNESS Certificates</h2><span id="wb" class="badge b-blue">--</span></div>
    <div class="pb" id="witness"></div>
  </div>
  <div id="grid3wrap"></div>
  <div class="panel" style="margin-bottom:28px;">
    <div class="ph"><h2>ðŸ“¡ Request Flow</h2><span id="rb" class="badge b-cyan">â€”</span></div>
    <div id="sparkarea" style="padding:12px 20px 0;"></div>
    <div style="overflow-x:auto;">
      <table><thead><tr><th>Time</th><th>Model</th><th>Tokens In</th><th>Saved</th><th>Dedup</th><th>SAST</th><th>Query</th></tr></thead>
      <tbody id="reqs"></tbody></table>
    </div>
  </div>
</div>
<script>
const fmt=n=>{if(n==null)return'â€”';return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(n)};
const money=n=>'$'+(n||0).toFixed(2);
const pct=n=>Math.round((n||0)*100)+'%';
const ago=ts=>{const s=Math.floor(Date.now()/1000-ts);return s<60?s+'s ago':s<3600?Math.floor(s/60)+'m ago':Math.floor(s/3600)+'h ago';};
let hasRequests=false;

let heroSparkData=[];
function renderHero(d){
  // Single source of truth for "saved": value_trends.lifetime (proxy-only).
  // stats.engine is internal efficiency telemetry â€” never priced as $$.
  const s=d.stats||{},eng=s.engine||s.savings||{},ss=s.session||{},dd=s.dedup||{};
  const frags=ss.total_fragments||0;
  const secTotal=(d.security&&!d.security.error)?((d.security.critical_total||0)+(d.security.high_total||0)):0;
  const h=d.health||{};
  const grade=h.health_grade||'?';
  const score=h.code_health_score||0;
  const gc={'A':'var(--emerald)','B':'var(--blue)','C':'var(--amber)','D':'#e3872d','F':'var(--rose)'}[grade]||'var(--dim)';
  const lt=(d.value_trends&&d.value_trends.lifetime)||{};
  const realCost=lt.cost_saved_usd||0;
  const realTokens=lt.tokens_saved||0;
  const realReqs=lt.requests_optimized||0;
  const dedupCount=dd.duplicates_detected||eng.duplicates_caught||eng.total_duplicates_caught||0;
  const optCalls=eng.optimize_calls||eng.total_optimizations||0;
  const reqs=d.recent_requests||[];
  if(realReqs>0)hasRequests=true;
  const w=document.getElementById('whisper');
  if(w&&hasRequests)w.style.display='none';

  // Track sparkline data
  if(reqs.length>0){reqs.forEach(r=>{if(heroSparkData.length>=40)heroSparkData.shift();heroSparkData.push(r.tokens_saved||0);});}

  // Empty state â€” nothing indexed AND no value from any path yet.
  const anyVal=realTokens>0||realReqs>0||(lt.hallucinations_blocked||0)>0||(lt.routing_saved_usd||0)>0||(d.activity||[]).length>0;
  if(frags===0&&!hasRequests&&!anyVal){
    document.getElementById('hero').innerHTML=`<div class="empty-hero">
      <div class="empty-icon">ðŸš€</div>
      <div class="empty-title">Ready to optimize</div>
      <div class="empty-desc">Entroly will track value automatically through any path you use â€” the <b>proxy</b> (<code>http://localhost:9377/v1</code>), the <b>MCP server</b> (pip or npm â€” call <code>optimize_context</code>), or the <b>SDK</b> (<code>from entroly import compress</code>).</div>
    </div>`;return;
  }

  // Impact sparkline
  const mx=Math.max(...heroSparkData,1);
  const sparkBars=heroSparkData.map(v=>'<div class="hbar" style="height:'+Math.max(3,v/mx*44)+'px;"></div>').join('');
  const sparkHTML=heroSparkData.length>0?'<div class="hero-spark">'+sparkBars+'</div>':'';

  // Headline = REAL savings only. Pre-traffic state shows what's READY,
  // not a fake projected dollar amount.
  const bigNum=realReqs>0?'$'+realCost.toFixed(2):fmt(frags);
  const bigLabel=realReqs>0?'COST SAVED':'FRAGMENTS READY';
  const subtitle=realReqs>0
    ?`<b>${fmt(realTokens)}</b> tokens saved across <b>${realReqs}</b> real LLM request${realReqs!==1?'s':''}`
    :`Indexed and ready Â· <span style="opacity:.7">point your AI tool to <code>http://localhost:9377/v1</code> to start saving on real requests</span>`;

  document.getElementById('hero').innerHTML=`
    <div class="hero-impact">
      <div class="hero-big">${bigNum}</div>
      <div class="hero-big-label">${bigLabel}</div>
      <div class="hero-subtitle">${subtitle}</div>
      ${sparkHTML}
    </div>
    <div class="hero-metrics">
      <div class="hero-metric hm-files"><div class="hm-icon">ðŸ“</div>
        <div class="hm-val hv-blue">${frags}</div>
        <div class="hm-label">Files Indexed</div>
        <div class="hm-sub">${fmt(ss.total_tokens_tracked||ss.total_tokens_ingested||ss.total_token_count||0)} tokens</div></div>
      <div class="hero-metric hm-health"><div class="hm-icon">ðŸ¥</div>
        <div class="hm-val" style="color:${gc}">${grade}</div>
        <div class="hm-label">Code Health</div>
        <div class="hm-sub">${score}/100 Â· ${(h.god_files||[]).length} god files</div></div>
      <div class="hero-metric hm-sast"><div class="hm-icon">ðŸ›¡ï¸</div>
        <div class="hm-val ${secTotal>0?'hv-rose':'hv-green'}">${secTotal>0?secTotal:'âœ“'}</div>
        <div class="hm-label">${secTotal>0?'Findings':'All Clear'}</div>
        <div class="hm-sub">${secTotal>0?(d.security.critical_total||0)+' crit Â· '+(d.security.high_total||0)+' high':'No vulnerabilities'}</div></div>
      <div class="hero-metric hm-reqs"><div class="hm-icon">âš¡</div>
        <div class="hm-val hv-cyan">${realReqs}</div>
        <div class="hm-label">Requests</div>
        <div class="hm-sub">${dedupCount} fragments deduped</div></div>
    </div>`;
}

function renderBA(d){
  const s=d.stats||{},ss=s.session||{};
  const ex=d.explain||{};
  const totalTokens=ss.total_tokens_tracked||ss.total_token_count||ss.total_tokens_ingested||0;
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
      <div class="ba-detail">${totalFrag} fragments Â· no dedup Â· no scoring</div>
      <div class="ba-detail" style="margin-top:4px;">Top-K: ~5 files visible to LLM</div>
    </div>
    <div class="ba-center">
      <div class="ba-arrow">â†’</div>
      <div class="ba-pct">${ratio>0?ratio+'%':'â€”'}</div>
      <div class="ba-pct-label">${ratio>0?'reduction':'awaiting optimize'}</div>
    </div>
    <div class="ba-side">
      <div class="ba-title">With Entroly (Optimized)</div>
      <div class="ba-val" style="color:var(--emerald);">${selTokens>0?fmt(selTokens)+' tokens':'â€”'}</div>
      <div class="ba-detail">${selected.length} fragments Â· knapsack-optimal Â· deduped</div>
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
  const w=d.prism_weights,pb=document.getElementById('pb');
  if(!w){document.getElementById('prism').innerHTML='<div class="empty">Engine not initialized</div>';return;}
  const names=['Recency','Frequency','Semantic','Entropy'];
  const vals=[w.recency,w.frequency,w.semantic,w.entropy];
  const defaults=[0.3,0.25,0.25,0.2];
  // L-infinity norm: if max absolute deviation from defaults < 0.01, weights are default
  const maxDev=Math.max(...vals.map((v,i)=>Math.abs(v-defaults[i])));
  const isLearned=maxDev>=0.01;
  if(pb){pb.textContent=isLearned?'RL-Learned':'Default';pb.className='badge '+(isLearned?'b-violet':'b-amber');}
  const colors=['#667eea','#f5576c','#4facfe','#43e97b'];
  const maxIdx=vals.indexOf(Math.max(...vals));
  const insight=`Your codebase responds best to <b>${names[maxIdx].toLowerCase()}</b> â€” ${names[maxIdx]==='Recency'?'recent edits are most predictive of what the LLM needs next':names[maxIdx]==='Frequency'?'frequently accessed files are the best context signal':names[maxIdx]==='Semantic'?'semantic similarity to the query drives the best results':'information-dense files contribute most to LLM accuracy'}.`;
  document.getElementById('prism').innerHTML=`<div class="radar-wrap">
    <canvas class="radar-canvas" id="radarC" width="180" height="180"></canvas>
    <ul class="radar-legend">${names.map((n,i)=>`<li><span class="rdot" style="background:${colors[i]}"></span>${n}<span class="rval">${pct(vals[i])}</span></li>`).join('')}</ul>
  </div><div class="prism-insight">ðŸ’¡ ${insight}</div>`;
  const c=document.getElementById('radarC');if(c)drawRadar(c.getContext('2d'),180,vals,colors);
}

function renderHealth(d){
  const h=d.health,el=document.getElementById('health'),b=document.getElementById('hb');
  if(!h||h.error){el.innerHTML='<div class="empty">Ingest code to see health</div>';return;}
  const g=h.health_grade||'?',sc=h.code_health_score||0;
  const gc={'A':'var(--emerald)','B':'var(--blue)','C':'var(--amber)','D':'#e3872d','F':'var(--rose)'}[g]||'var(--dim)';
  b.textContent=g+' Â· '+sc+'/100';b.className='badge '+(g<='B'?'b-green':g==='C'?'b-amber':'b-rose');
  el.innerHTML=`<div class="health-ring-wrap">
    <div class="health-ring"><canvas id="hring" width="110" height="110"></canvas><div class="grade" style="color:${gc}">${g}</div></div>
    <ul class="health-stats">
      <li><span class="hv">${(h.clone_pairs||[]).length}</span>clone pairs</li>
      <li><span class="hv">${(h.dead_symbols||[]).length}</span>dead symbols</li>
      <li><span class="hv">${(h.god_files||[]).length}</span>god files</li>
      <li><span class="hv">${(h.arch_violations||[]).length}</span>arch violations</li>
      <li><span class="hv">${(h.naming_issues||[]).length}</span>naming issues</li>
    </ul></div>
    ${h.top_recommendation?'<div class="health-rec">ðŸ’¡ '+h.top_recommendation+'</div>':''}`;
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
  // Use actual hit rate (hits/lookups), not the EMA prior which starts at ~0.485
  const realHits=exact+semantic;
  const realHitRate=lookups>0?realHits/lookups:0;
  const warmState=entries===0?'Cold':realHits>0?'Warm':lookups>0?'Warming':'Idle';
  b.textContent=warmState+' Â· '+pct(realHitRate);
  b.className='badge '+(realHits>0?'b-green':entries>0?'b-amber':'b-blue');
  el.innerHTML=`<div class="cache-kpis">
    <div class="cache-kpi"><div class="cache-kpi-label">Hit Rate EMA</div><div class="cache-kpi-val hv-green">${pct(hitRateEma)}</div><div class="cache-kpi-sub">${exact} exact Â· ${semantic} semantic</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Entries</div><div class="cache-kpi-val hv-blue">${fmt(entries)}</div><div class="cache-kpi-sub">${fmt(lookups)} lookups processed</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Tokens Saved</div><div class="cache-kpi-val hv-amber">${fmt(c.tokens_saved||0)}</div><div class="cache-kpi-sub">${c.invalidations||0} invalidations Â· ${c.total_resets||0} resets</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Explore / Exploit</div><div class="cache-kpi-val hv-rose">${pct(exploreRatio)} / ${pct(exploitRatio)}</div><div class="cache-kpi-sub">${c.total_explorations||0} explore Â· ${c.total_exploitations||0} exploit</div></div>
  </div>
  <div class="cache-split">
    <div class="cache-block">
      <h3>Thompson Gate</h3>
      <div class="cache-pair"><span>Posterior Î± / Î²</span><strong>${(c.thompson_alpha||0).toFixed(2)} / ${(c.thompson_beta||0).toFixed(2)}</strong></div>
      <div class="cache-pair"><span>Adaptive RÃ©nyi Î±</span><strong>${(c.adaptive_alpha||0).toFixed(4)}</strong></div>
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
  panels+='<div class="panel"><div class="ph"><h2>ðŸ›¡ï¸ Security</h2><span id="sb" class="badge '+(sec&&(sec.critical_total||0)+(sec.high_total||0)>0?'b-rose':'b-green')+'">';
  if(sec&&!sec.error){
    const tot=(sec.critical_total||0)+(sec.high_total||0);
    panels+=tot>0?tot+' findings':'âœ“ Clean';
    panels+='</span></div><div class="pb">';
    if(tot===0){panels+='<div style="text-align:center;padding:16px;"><div style="font-size:40px;filter:drop-shadow(0 0 16px rgba(52,211,153,0.4));">ðŸ›¡ï¸</div><div style="color:var(--emerald);font-weight:700;margin-top:8px;">No vulnerabilities</div><div style="font-size:12px;color:var(--dim);margin-top:4px;">'+(sec.fragments_scanned||0)+' fragments scanned</div></div>';}
    else{
      const findings=sec.findings||sec.top_findings||[];
      const cats=sec.findings_by_category||{};
      if(findings.length>0){
        panels+=findings.slice(0,4).map(f=>`<div class="finding"><span class="finding-sev ${(f.severity||'').toLowerCase()==='critical'?'crit':'high'}">${(f.severity||'?')[0]}</span><div><div class="finding-file">${f.file||f.source||'unknown'}${f.line?':'+f.line:''}</div><div class="finding-desc">${f.message||f.category||''}</div></div></div>`).join('');
      } else {
        panels+=Object.entries(cats).map(([k,v])=>`<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;"><span style="color:var(--dim);">${k}</span><span class="tag t-rose">${v}</span></div>`).join('');
      }
    }
  } else {panels+='â€”</span></div><div class="pb"><div class="empty">No scan yet</div>';}
  panels+='</div></div>';

  // Dep graph - only show if data exists
  if(dg&&(dg.total_symbols||dg.symbol_count||0)>0){
    const sym=dg.total_symbols||dg.symbol_count||0,edg=dg.total_edges||dg.edge_count||0;
    panels+='<div class="panel"><div class="ph"><h2>ðŸ•¸ï¸ Dep Graph</h2><span class="badge b-cyan">'+sym+' symbols</span></div><div class="pb">';
    panels+='<div style="display:flex;gap:20px;justify-content:center;padding:12px 0;"><div style="text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--cyan);">'+sym+'</div><div style="font-size:10px;color:var(--dim);margin-top:4px;">SYMBOLS</div></div>';
    panels+='<div style="text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--blue);">'+edg+'</div><div style="font-size:10px;color:var(--dim);margin-top:4px;">EDGES</div></div></div>';
    panels+='</div></div>';
  }

  // Knapsack with included AND excluded
  if(ex&&!ex.error){
    const inc=ex.included||[],exc=ex.excluded||[];
    panels+='<div class="panel"><div class="ph"><h2>ðŸŽ¯ Knapsack</h2><span class="badge b-violet">'+inc.length+' selected Â· '+pct(ex.sufficiency)+' suff.</span></div>';
    panels+='<div class="pb" style="max-height:320px;overflow-y:auto;">';
    let rows=inc.slice(0,5).map(f=>{const sc=f.scores||{};return`<tr><td class="mono" style="color:var(--emerald);">âœ“ ${(f.source||f.id||'').split(/[\\/]/).pop()}</td><td class="mono">${pct(sc.composite)}</td><td style="font-size:11px;color:var(--dim);">${(f.reason||'').slice(0,30)}</td></tr>`;}).join('');
    rows+=exc.slice(0,3).map(f=>{const sc=f.scores||{};return`<tr style="opacity:0.5;"><td class="mono" style="color:var(--rose);">âœ— ${(f.source||f.id||'').split(/[\\/]/).pop()}</td><td class="mono">${pct(sc.composite)}</td><td style="font-size:11px;color:var(--rose);">${(f.reason||'below threshold').slice(0,30)}</td></tr>`;}).join('');
    panels+='<table><thead><tr><th>Fragment</th><th>Score</th><th>Reason</th></tr></thead><tbody>'+rows+'</tbody></table>';
    panels+='</div></div>';
  }
  panels+='</div>';
  document.getElementById('grid3wrap').innerHTML=panels;
}

let sparkData=[];
function escHtml(s){
  return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderWitness(d){
  const w=d.witness||{},el=document.getElementById('witness'),b=document.getElementById('wb');
  if(!el||!b)return;
  if(!w.available){
    b.textContent='Proxy idle';b.className='badge b-blue';
    el.innerHTML='<div class="empty">Start the proxy with <code>entroly proxy --witness audit</code> or <code>--witness strict</code> to see proof certificates.</div>';
    return;
  }
  const items=w.items||[],feedback=w.feedback||{};
  const suppressed=items.reduce((a,x)=>a+(((x.policy||{}).suppressed_count)||0),0);
  const flagged=items.reduce((a,x)=>a+((x.n_contradicted||0)+(x.n_unsupported||0)+(x.n_unknown||0)),0);
  b.textContent=items.length+' recent Â· '+flagged+' flagged';
  b.className='badge '+(flagged>0?'b-rose':'b-green');
  const kpis=`<div class="cache-kpis">
    <div class="cache-kpi"><div class="cache-kpi-label">Flagged Claims</div><div class="cache-kpi-val hv-rose">${flagged}</div><div class="cache-kpi-sub">recent sidecar certificates</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Suppressed</div><div class="cache-kpi-val hv-amber">${suppressed}</div><div class="cache-kpi-sub">strict-mode removals</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Feedback</div><div class="cache-kpi-val hv-blue">${feedback.false_positive||0}</div><div class="cache-kpi-sub">false-positive reports</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Proxy</div><div class="cache-kpi-val hv-cyan">${w.proxy_port||'--'}</div><div class="cache-kpi-sub">/witness sidecar API</div></div>
  </div>`;
  if(items.length===0){el.innerHTML=kpis+'<div class="empty">No WITNESS certificates yet.</div>';return;}
  const rows=items.slice(0,6).map(item=>{
    const claims=item.flagged_claims||[];
    const first=claims[0]||{};
    const proof=(first.proof_path||[]).map(p=>escHtml((p.operator||'proof')+': '+(p.evidence||''))).join('<br>');
    return `<tr>
      <td class="mono">${escHtml(item.id||'--')}</td>
      <td><span class="tag ${claims.length?'t-rose':'t-green'}">${claims.length?'flagged':'pass'}</span></td>
      <td>${escHtml(first.claim_text||'--')}</td>
      <td style="font-size:11px;color:var(--dim);max-width:360px;">${proof||'--'}</td>
    </tr>`;
  }).join('');
  el.innerHTML=kpis+`<div style="overflow-x:auto;"><table><thead><tr><th>ID</th><th>Status</th><th>Flagged claim</th><th>Proof / evidence</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function renderRequests(d){
  const reqs=d.recent_requests||[],tbody=document.getElementById('reqs'),b=document.getElementById('rb');
  b.textContent=reqs.length+' recent';
  if(reqs.length>0){reqs.forEach(r=>{if(sparkData.length>=30)sparkData.shift();sparkData.push(r.tokens_saved||0);});
    const mx=Math.max(...sparkData,1);
    document.getElementById('sparkarea').innerHTML='<div class="sparkline">'+sparkData.map(v=>'<div class="bar" style="height:'+Math.max(2,v/mx*40)+'px;"></div>').join('')+'</div>';}
  if(reqs.length===0){tbody.innerHTML='<tr><td colspan="7" class="empty">No activity yet â€” flows in live from proxy, MCP (pip/npm), or the SDK on first use</td></tr>';return;}
  tbody.innerHTML=reqs.slice().reverse().slice(0,15).map(r=>`<tr>
    <td>${ago(r.time||0)}</td><td>${r.model||'â€”'}</td><td class="mono">${fmt(r.tokens_in||0)}</td>
    <td><span class="tag t-green">âˆ’${fmt(r.tokens_saved||0)}</span></td>
    <td>${(r.dedup_hits||0)>0?'<span class="tag t-amber">'+r.dedup_hits+'</span>':'<span style="color:var(--dim2)">0</span>'}</td>
    <td>${(r.sast_findings||0)>0?'<span class="tag t-rose">'+r.sast_findings+'</span>':'<span style="color:var(--dim2)">0</span>'}</td>
    <td style="max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dim);">${r.query||'â€”'}</td></tr>`).join('');
}

let trendsView='daily';
function renderValueTrends(d){
  const vt=d.value_trends,vc=d.value_confidence,el=document.getElementById('valueTrends');
  if(!el)return;
  const _lt0=(vt&&vt.lifetime)||{};
  const anyValue=vt&&vc&&(vt.lifetime.tokens_saved||vc.session.tokens_saved||
    _lt0.hallucinations_blocked||_lt0.routing_saved_usd);
  if(!anyValue){
    // Friendly zero-state instead of a blank panel. Works for every
    // install mode â€” proxy, MCP (pip), SDK, and npm.
    el.innerHTML='<div class="trends-panel"><div class="trends-header">'+
      '<h2>Lifetime Value</h2><span class="badge" style="background:'+
      'rgba(148,163,184,.12);color:var(--dim)">waiting for first request</span>'+
      '</div><div class="empty" style="padding:22px 16px;line-height:1.6">'+
      'No value recorded yet. Entroly starts counting tokens saved, '+
      'hallucinations blocked, and model-routing savings the moment your '+
      'first request flows through <b>any</b> path:<br>'+
      '&nbsp;&nbsp;â€¢ <b>Proxy</b>: point your AI tool at '+
      '<code>http://localhost:9377/v1</code><br>'+
      '&nbsp;&nbsp;â€¢ <b>MCP</b> (pip/npm): call the <code>optimize_context</code> tool<br>'+
      '&nbsp;&nbsp;â€¢ <b>SDK</b>: <code>from entroly import compress</code><br>'+
      'This panel is live from the shared telemetry file and refreshes '+
      'automatically.</div></div>';
    return;
  }
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
    '<div class="trends-kpi"><div class="trends-kpi-label">Today</div><div class="trends-kpi-val hv-blue">$'+(today.cost_saved_usd||0).toFixed(4)+'</div><div class="trends-kpi-sub">'+fmt(today.tokens_saved||0)+' tokens Â· '+(today.requests||0)+' reqs</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">This Session</div><div class="trends-kpi-val hv-amber">'+fmt(sess.tokens_saved||0)+'</div><div class="trends-kpi-sub">$'+(sess.cost_saved_usd||0).toFixed(4)+' Â· '+(sess.requests||0)+' reqs</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">Daily Average</div><div class="trends-kpi-val hv-green">$'+daily_avg.toFixed(4)+'</div><div class="trends-kpi-sub">'+(lt.requests_optimized||0)+' reqs optimized Â· '+(lt.duplicates_caught||0)+' dedup</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">Hallucinations Blocked</div><div class="trends-kpi-val hv-rose">'+fmt(lt.hallucinations_blocked||0)+'</div><div class="trends-kpi-sub">unsupported claims stopped by WITNESS</div></div>'+
    '<div class="trends-kpi"><div class="trends-kpi-label">Model-Routing Saved</div><div class="trends-kpi-val hv-violet">$'+(lt.routing_saved_usd||0).toFixed(2)+'</div><div class="trends-kpi-sub">'+(lt.routing_decisions||0)+' RAVS routing decisions</div></div>'+
    '</div>'+
    '<div class="trends-tabs">'+
    '<div class="trends-tab'+(trendsView==='daily'?' active':'')+'" onclick="trendsView=\'daily\'">Daily</div>'+
    '<div class="trends-tab'+(trendsView==='weekly'?' active':'')+'" onclick="trendsView=\'weekly\'">Weekly</div>'+
    '<div class="trends-tab'+(trendsView==='monthly'?' active':'')+'" onclick="trendsView=\'monthly\'">Monthly</div></div>'+
    '<div class="trends-chart">'+bars+'</div>'+
    '</div></div>';
}

function renderCogops(d){
  const c=d.cogops,el=document.getElementById('cogops'),b=document.getElementById('cogb');
  if(!el)return;
  if(!c){el.innerHTML='<div class="empty">Epistemic engine not initialized â€” vault missing or unreadable</div>';return;}
  const tb=c.total_beliefs||0,ver=c.verified||0,st=c.stale||0,db=c.doc_beliefs||0;
  const conf=c.avg_confidence||0,fresh=c.freshness_pct||0,ents=c.entity_count||0;
  if(b){b.textContent=(c.engine||'cogops')+' Â· '+tb+' beliefs';b.className='badge '+(tb>0?'b-violet':'b-blue');}
  if(tb===0){el.innerHTML='<div class="empty">No beliefs yet â€” run <code>compile_beliefs</code> to seed the vault</div>';return;}
  el.innerHTML=`<div class="cache-kpis">
    <div class="cache-kpi"><div class="cache-kpi-label">Total Beliefs</div><div class="cache-kpi-val hv-blue">${fmt(tb)}</div><div class="cache-kpi-sub">${ents} distinct entities Â· ${db} doc-linked</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Avg Confidence</div><div class="cache-kpi-val hv-green">${(conf*100).toFixed(1)}%</div><div class="cache-kpi-sub">${ver} verified Â· ${tb-ver-st} inferred</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Freshness</div><div class="cache-kpi-val hv-amber">${fresh.toFixed(0)}%</div><div class="cache-kpi-sub">${st} stale beliefs flagged</div></div>
    <div class="cache-kpi"><div class="cache-kpi-label">Engine</div><div class="cache-kpi-val hv-violet">${c.engine||'â€”'}</div><div class="cache-kpi-sub">5-layer epistemic topology</div></div>
  </div>`;
}

function renderErrors(items){
  const el=document.getElementById('errBanner');
  if(!el){return;}
  if(!items||items.length===0){el.style.display='none';el.innerHTML='';return;}
  const head=`<b>${items.length} subsystem error${items.length!==1?'s':''}</b> â€” dashboard data may be incomplete.`;
  const list=items.slice(0,5).map(x=>`<div style="margin-top:6px;opacity:.85"><code style="background:rgba(239,68,68,.12);padding:1px 6px;border-radius:4px">${x.section||'?'}</code> ${x.type||'Error'}: ${(x.message||'').substring(0,200)}</div>`).join('');
  const more=items.length>5?`<div style="margin-top:6px;opacity:.6">â€¦and ${items.length-5} more</div>`:'';
  el.innerHTML=head+list+more;el.style.display='block';
}

async function refresh(){
  try{
    const r=await fetch('/api/metrics');
    if(!r.ok){renderErrors([{section:'http',type:'HTTPError',message:'/api/metrics returned '+r.status}]);return;}
    const d=await r.json();
    renderErrors(d.errors||[]);
    renderHero(d);renderValueTrends(d);renderBA(d);renderPrism(d);renderHealth(d);renderCache(d);renderCogops(d);renderWitness(d);renderSecAndKnapsack(d);renderRequests(d);
  }catch(e){
    console.error('Refresh:',e);
    renderErrors([{section:'fetch',type:e.name||'Error',message:e.message||String(e)}]);
  }
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
        self.send_header("Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "img-src 'self' data:"
        )
        self.send_header("Referrer-Policy", "no-referrer")

    # Maps URL path â†’ daemon-state key for the read-only control GETs.
    _CONTROL_GET_ROUTES = {
        "/api/control/status": "status",
        "/api/control/repos": "repos",
        "/api/control/learning": "learning",
        "/api/control/privacy": "privacy",
        "/api/control/federation": "federation",
        "/api/control/context/last": "context_last",
        "/api/control/logs": "logs",
    }

    def do_GET(self):
        # Static HTML
        if self.path in ("/", "/dashboard"):
            self._send_html(DASHBOARD_HTML)
        elif self.path == "/controls":
            from entroly.controls_html import CONTROLS_HTML
            self._send_html(CONTROLS_HTML)
        # JSON read APIs (CORS so other localhost dashboards can query)
        elif self.path == "/api/metrics":
            self._send_json(200, _get_full_snapshot())
        elif self.path == "/api/trends":
            self._send_json(200, self._safe_tracker_call("get_trends"))
        elif self.path == "/api/confidence":
            self._send_json(200, self._safe_tracker_call("get_confidence"))
        elif self.path == "/health":
            # Health probe â€” no CORS, kept tight for internal liveness checks.
            self._respond(200, "application/json", b'{"status":"ok"}')
        # Control API reads
        elif self.path in self._CONTROL_GET_ROUTES:
            self._handle_control_get(self._CONTROL_GET_ROUTES[self.path])
        else:
            self._send_json(404, {"error": "not found", "path": self.path})

    @staticmethod
    def _safe_tracker_call(method: str) -> dict:
        """Call a value_tracker method, returning {} on any failure rather
        than letting the exception bubble into a 500. The caller decides
        what HTTP status to send."""
        try:
            from entroly.value_tracker import get_tracker
            return getattr(get_tracker(), method)()
        except Exception:
            return {}

    def do_POST(self):
        """Handle Control API POST requests."""
        from entroly.daemon import get_daemon
        daemon = get_daemon()

        # Read POST body. Reject malformed JSON loudly â€” silent fallback to
        # `{}` lets bad payloads enable features by accident (every handler
        # below uses `body.get("enabled", True)` style defaults).
        content_length = int(self.headers.get("Content-Length", 0))
        body: dict = {}
        if content_length > 0:
            raw = self.rfile.read(content_length)
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json(400, {
                    "ok": False,
                    "error": "invalid JSON body",
                    "detail": str(e),
                })
                return
            if not isinstance(parsed, dict):
                self._send_json(400, {
                    "ok": False,
                    "error": "JSON body must be an object",
                    "got": type(parsed).__name__,
                })
                return
            body = parsed

        result = {"ok": False, "error": "unknown route"}
        status_code = 404

        if daemon is None:
            # Daemon not running â€” control API unavailable
            result = {"ok": False, "error": "daemon not running (use `entroly daemon` to start)"}
            status_code = 503
        elif self.path == "/api/control/optimization/enable":
            daemon.set_optimization(True)
            result = {"ok": True, "optimization": True}
            status_code = 200
        elif self.path == "/api/control/optimization/pause":
            daemon.set_optimization(False)
            result = {"ok": True, "optimization": False}
            status_code = 200
        elif self.path == "/api/control/bypass":
            enabled = body.get("enabled", True)
            daemon.set_bypass(enabled)
            result = {"ok": True, "bypass": enabled}
            status_code = 200
        elif self.path == "/api/control/quality":
            mode = body.get("mode", "balanced")
            try:
                daemon.set_quality(mode)
                result = {"ok": True, "quality": mode}
                status_code = 200
            except ValueError as e:
                result = {"ok": False, "error": str(e)}
                status_code = 400
        elif self.path == "/api/control/repos/reindex":
            path = body.get("path")
            daemon.reindex_repo(path)
            result = {"ok": True, "reindexed": path or "all"}
            status_code = 200
        elif self.path == "/api/control/learning/enable":
            enabled = body.get("enabled", True)
            daemon.state.learning_enabled = enabled
            result = {"ok": True, "learning_enabled": enabled}
            status_code = 200
        elif self.path == "/api/control/learning/reset":
            daemon.reset_learning()
            result = {"ok": True, "weights_reset": True}
            status_code = 200
        elif self.path == "/api/control/learning/autotune":
            daemon.state.autotune_enabled = True
            result = {"ok": True, "autotune": "triggered"}
            status_code = 200
        elif self.path == "/api/control/federation/enable":
            daemon.state.federation_enabled = True
            daemon.state.federation_mode = body.get("mode", "anonymous")
            result = {"ok": True, "federation": daemon.state.federation_mode}
            status_code = 200
        elif self.path == "/api/control/federation/disable":
            daemon.state.federation_enabled = False
            daemon.state.federation_mode = "off"
            result = {"ok": True, "federation": "off"}
            status_code = 200
        elif self.path == "/api/control/stop":
            result = {"ok": True, "status": "stopping"}
            status_code = 200
            # Schedule stop after response
            import threading
            threading.Thread(
                target=lambda: (
                    __import__("time").sleep(0.5), daemon.stop()
                ),
                daemon=True,
            ).start()

        self._send_json(status_code, result)

    def _respond(
        self,
        status: int,
        content_type: str,
        body: bytes,
        *,
        no_cache: bool = False,
        cors_origin: str | None = None,
    ) -> None:
        """Single response writer for *all* routes (HTML, JSON, health).
        Centralizing here is the only way headers (CORS, CSP, cache) stay
        in sync across handlers â€” every previous drift bug came from
        copy-pasted send_header blocks falling out of step."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if no_cache:
            self.send_header("Cache-Control", "no-cache")
        if cors_origin:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
        self._send_security_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict, *, cors: bool = True) -> None:
        self._respond(
            status,
            "application/json",
            json.dumps(payload, default=str).encode(),
            no_cache=True,
            cors_origin="http://localhost:9378" if cors else None,
        )

    def _send_html(self, body: str) -> None:
        self._respond(
            200,
            "text/html; charset=utf-8",
            body.encode(),
            no_cache=True,
        )

    def do_OPTIONS(self):
        """Handle CORS preflight for control API."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:9378")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_control_get(self, key: str):
        """Handle a control API GET request."""
        from entroly.daemon import get_daemon
        daemon = get_daemon()

        if daemon is None:
            data = {"error": "daemon not running", "hint": "use `entroly daemon` to start"}
        elif key == "status":
            data = daemon.state.to_dict()
        elif key == "repos":
            data = {"repos": [
                {"path": r.path, "watching": r.watching,
                 "indexed_files": r.indexed_files,
                 "total_tokens": r.total_tokens,
                 "last_sync": r.last_sync}
                for r in daemon.state.repos
            ]}
        elif key == "learning":
            data = _control_learning_snapshot(daemon)
        elif key == "privacy":
            try:
                import os
                from pathlib import Path

                from entroly.privacy import file_info, retention_days

                dirs: list[Path] = []
                env_dir = os.environ.get("ENTROLY_DIR")
                if env_dir:
                    dirs.append(Path(env_dir))
                # Supervisor default (when ENTROLY_DIR not set)
                dirs.append(Path(os.getcwd()) / ".entroly")
                # Shared telemetry default (~/.entroly)
                dirs.append(Path.home() / ".entroly")

                seen = set()
                roots: list[Path] = []
                for d in dirs:
                    dp = d.resolve()
                    if str(dp) in seen:
                        continue
                    seen.add(str(dp))
                    roots.append(dp)

                files = []
                for root in roots:
                    files.extend([
                        file_info(root / "value_tracker.json"),
                        file_info(root / "activity.jsonl"),
                        file_info(root / "feedback_journal.jsonl"),
                    ])

                data = {
                    "retention_days": retention_days(),
                    "telemetry_roots": [str(r) for r in roots],
                    "files": files,
                    "notes": [
                        "Local-only: data stays on this machine unless federation is enabled.",
                        "Retention applies to JSONL telemetry; set ENTROLY_RETENTION_DAYS=0 to disable pruning.",
                    ],
                }
            except Exception as e:
                data = {"error": str(e)}
        elif key == "federation":
            data = {
                "enabled": daemon.state.federation_enabled,
                "mode": daemon.state.federation_mode,
            }
        elif key == "context_last":
            data = daemon.get_last_context()
        elif key == "logs":
            data = {"lines": daemon.get_logs(100)}
        else:
            data = {"error": f"unknown key: {key}"}

        # Daemon-down should be 503 so callers can branch on status; key-down
        # is a server-side bug surfaced as 200+error for the existing UI.
        status = 503 if daemon is None else 200
        self._send_json(status, data)


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

    # Auto-register a lightweight daemon state so the control API
    # works even when called from `entroly go` / `entroly proxy`
    # (not just `entroly daemon`).
    from entroly.daemon import get_daemon, _register_control_api, EntrolyDaemon, _install_log_buffer
    _install_log_buffer()  # Ensure live logs work from any entry point
    if get_daemon() is None:
        _lite = EntrolyDaemon.__new__(EntrolyDaemon)
        _lite.state = __import__("entroly.daemon", fromlist=["EntrolyDaemonState"]).EntrolyDaemonState()
        _lite.state.status = "running"
        _lite.state.started_at = __import__("time").time()
        _lite.state.dashboard.running = True
        _lite.state.dashboard.port = port
        _lite.state.proxy.running = True  # proxy is live via entroly go
        _lite.state.proxy.port = 9377
        _lite._engine = engine
        _lite._proxy_server = None
        _lite._dashboard_server = None
        _lite._workers = {}
        _lite._shutdown = __import__("threading").Event()
        _lite._lock = __import__("threading").Lock()
        _lite._host = "127.0.0.1"
        _lite._enable_proxy = True
        _lite._enable_mcp = False
        _lite._repo_paths = [__import__("os").getcwd()]
        _lite._proxy_config = None  # will be set if proxy is running

        # Populate repo state from engine
        try:
            stats = engine._rust.stats() if hasattr(engine, "_rust") else {}
            sess = stats.get("session", {})
            _lite.state.repos.append(
                __import__("entroly.daemon", fromlist=["RepoState"]).RepoState(
                    path=__import__("os").getcwd(),
                    watching=True,
                    indexed_files=sess.get("total_fragments", 0),
                    total_tokens=sess.get("total_tokens_tracked", 0),
                    last_sync=__import__("time").time(),
                )
            )
        except Exception:
            pass

        _register_control_api(_lite)

    class _ReuseAddrHTTPServer(HTTPServer):
        allow_reuse_address = True

        def process_request(self, request, client_address):
            """Handle each request in a new thread to avoid blocking."""
            t = threading.Thread(target=self.finish_request, args=(request, client_address), daemon=True)
            t.start()

    server = _ReuseAddrHTTPServer(("127.0.0.1", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(f"Dashboard live at http://localhost:{port}")
    return server

