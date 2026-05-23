"""Check exact field names in explain_selection output."""
import sys, json
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

from entroly.server import EntrolyEngine
from entroly.auto_index import auto_index

engine = EntrolyEngine()
auto_index(engine)
engine.advance_turn()
engine.optimize_context(token_budget=32000, query="How does the server work?")

explain = engine._rust.explain_selection()
explain = dict(explain)

# Show first included fragment's keys
inc = explain.get("included", [])
if inc:
    f = dict(inc[0]) if hasattr(inc[0], "items") else inc[0]
    print("=== FIRST INCLUDED FRAGMENT KEYS ===")
    for k, v in (f.items() if isinstance(f, dict) else []):
        print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")
else:
    print("No included fragments")

# Show repo state
print("\n=== ENGINE STATS SESSION ===")
stats = dict(engine._rust.stats())
sess = dict(stats.get("session", {}))
print(f"  total_fragments: {sess.get('total_fragments', 'MISSING')}")
print(f"  total_tokens_tracked: {sess.get('total_tokens_tracked', 'MISSING')}")
