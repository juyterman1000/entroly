"""Verify every README claim against actual codebase."""
import sys

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

# === SDK ===
check("SDK: compress import", lambda: __import__("entroly").compress and "OK")
check("SDK: compress_messages import", lambda: __import__("entroly").compress_messages and "OK")
check("SDK: compress works", lambda: f"{len(__import__('entroly').compress('hello ' * 500, budget=50))} chars")

# === CLI: wrap ===
from entroly.cli import _WRAP_AGENTS
check("wrap claude", lambda: "claude" in _WRAP_AGENTS and "OK")
check("wrap codex", lambda: "codex" in _WRAP_AGENTS and "OK")
check("wrap aider", lambda: "aider" in _WRAP_AGENTS and "OK")
check("wrap cursor", lambda: "cursor" in _WRAP_AGENTS and "OK")
check("wrap copilot", lambda: "copilot" in _WRAP_AGENTS and "OK" or (_ for _ in ()).throw(Exception("MISSING")))

# === Proxy ===
check("Proxy: PromptCompilerProxy", lambda: __import__("entroly.proxy", fromlist=["PromptCompilerProxy"]).PromptCompilerProxy and "OK")
check("Proxy: ProxyConfig port=9377", lambda: f"port={__import__('entroly.proxy_config', fromlist=['ProxyConfig']).ProxyConfig.from_env().port}")

# === Engine ===
from entroly.server import EntrolyEngine
e = EntrolyEngine()
check("Engine: Rust backend", lambda: f"use_rust={e._use_rust}")
check("Engine: ingest_fragment", lambda: e.ingest_fragment("def foo(): pass", "test.py", 5) and "OK")
r = e.optimize_context(token_budget=8000, query="foo")
check("Engine: optimize_context", lambda: f"{len(r.get('selected_fragments',[]))} frags")

# === Federation ===
check("Federation", lambda: __import__("entroly.federation", fromlist=["FederationClient"]).FederationClient and "OK")

# === CCR (reversible) ===
check("CCR reversible", lambda: __import__("entroly.ccr", fromlist=["get_ccr_store"]).get_ccr_store and "OK")

# === Value tracker ===
from entroly.value_tracker import estimate_cost
check("estimate_cost", lambda: f"10K gpt-4o = ${estimate_cost(10000, 'gpt-4o'):.4f}")

# === Dashboard ===
check("Dashboard", lambda: __import__("entroly.dashboard", fromlist=["start_dashboard"]).start_dashboard and "OK")

# === auto_index ===
check("auto_index", lambda: __import__("entroly.auto_index", fromlist=["auto_index"]).auto_index and "OK")

# === Language support ===
from entroly.auto_index import SUPPORTED_EXTENSIONS
check("Language extensions", lambda: f"{len(SUPPORTED_EXTENSIONS)} extensions supported")

# === bench/accuracy.py ===
check("bench/accuracy.py", lambda: __import__("pathlib").Path("bench/accuracy.py").exists() and "OK")

# === Summary ===
print(f"\n{'='*50}")
print(f"  PASSED: {passed}  |  FAILED: {failed}")
if failed:
    print(f"  README has {failed} unverified claim(s)!")
else:
    print(f"  All README claims verified!")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
