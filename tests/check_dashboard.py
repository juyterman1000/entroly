"""Check dashboard data accuracy."""
import sys, json, time, urllib.request
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

from entroly.dashboard import start_dashboard
from entroly.server import EntrolyEngine
from entroly.auto_index import auto_index

engine = EntrolyEngine()
auto_index(engine)
engine.advance_turn()
opt = engine.optimize_context(token_budget=32000, query="How does the server work?")

start_dashboard(engine=engine, port=9379, daemon=True)
time.sleep(1)

r = urllib.request.urlopen("http://127.0.0.1:9379/api/metrics")
m = json.loads(r.read())

print("=== RAW API RESPONSE KEYS ===")
for k, v in m.items():
    if isinstance(v, dict):
        print(f"  {k}: {list(v.keys())[:8]}")
    elif v is None:
        print(f"  {k}: None")
    else:
        print(f"  {k}: {str(v)[:80]}")

print("\n=== DASHBOARD DATA AUDIT ===")
issues = []

# Stats (nested under stats.session)
stats = m.get("stats", {})
session = stats.get("session", {}) if isinstance(stats, dict) else {}
savings = stats.get("savings", {}) if isinstance(stats, dict) else {}

frags = session.get("total_fragments", 0)
tokens = session.get("total_tokens_tracked", 0)
tok_saved = savings.get("total_tokens_saved", 0)

print(f"  stats.session.total_fragments: {frags}")
print(f"  stats.session.total_tokens_tracked: {tokens}")
print(f"  stats.savings.total_tokens_saved: {tok_saved}")

if frags == 0: issues.append("total_fragments is 0")
if tokens == 0: issues.append("total_tokens is 0")

# Health
health = m.get("health")
if health and not isinstance(health, dict):
    health = None
grade = health.get("health_grade", "?") if health else "missing"
print(f"  health.health_grade: {grade}")
if grade in ("?", "missing"): issues.append("health_grade missing")

# PRISM
prism = m.get("prism_weights", {})
print(f"  prism_weights: {json.dumps(prism)}")
if not prism: issues.append("prism_weights empty")

# Explain (knapsack)
explain = m.get("explain")
if explain and isinstance(explain, dict):
    inc = explain.get("included", [])
    exc = explain.get("excluded", [])
    print(f"  explain: {len(inc)} included, {len(exc)} excluded")
else:
    print(f"  explain: None")
    issues.append("explain is None")

# Security
sec = m.get("security")
if sec and isinstance(sec, dict):
    print(f"  security: {sec.get('critical_total', 0)} critical, {sec.get('high_total', 0)} high")
else:
    print(f"  security: None")
    issues.append("security is None")

# CogOps
cogops = m.get("cogops", {})
print(f"  cogops: {cogops.get('total_beliefs', 0)} beliefs, engine={cogops.get('engine', '?')}")

if issues:
    print(f"\nISSUES: {issues}")
else:
    print("\nAll dashboard data correct!")
