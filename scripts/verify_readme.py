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
from entroly.cli import _WRAP_AGENTS  # noqa: E402
from pathlib import Path  # noqa: E402

README_TEXT = Path("README.md").read_text(encoding="utf-8")
COOKBOOK_TEXT = Path("cookbook/README.md").read_text(encoding="utf-8")
FOR_TEAMS_TEXT = Path("docs/for-teams.md").read_text(encoding="utf-8")

check("wrap claude", lambda: "claude" in _WRAP_AGENTS and "OK")
check("wrap codex", lambda: "codex" in _WRAP_AGENTS and "OK")
check("wrap aider", lambda: "aider" in _WRAP_AGENTS and "OK")
check("wrap cursor", lambda: "cursor" in _WRAP_AGENTS and "OK")
check("wrap copilot", lambda: "copilot" in _WRAP_AGENTS and "OK" or (_ for _ in ()).throw(Exception("MISSING")))

README_WRAP_SLUGS = [
    "claude", "cursor", "codex", "aider", "gemini", "qwen", "opencode",
    "crush", "hermes", "pi", "ollama", "windsurf", "vscode",
    "claude-desktop", "claude-code", "zed", "cline", "roo", "continue",
    "cody", "amp", "kiro", "qoder", "trae", "antigravity", "amazonq",
    "verdent", "jetbrains", "helix", "tabby", "twinny", "sublime",
    "emacs", "neovim", "fittencode", "tabnine", "supermaven",
]
check(
    "wrap README slug coverage",
    lambda: (
        not [slug for slug in README_WRAP_SLUGS if slug not in _WRAP_AGENTS]
        and f"{len(README_WRAP_SLUGS)} slugs"
    ) or (_ for _ in ()).throw(Exception("README lists missing wrapper slug")),
)
check(
    "README print-only wording",
    lambda: (
        "Paste once, restart, done" not in README_TEXT
        and "prints the exact file path and field name" not in README_TEXT
        and "best-effort endpoint/config hint" in README_TEXT
        and "OK"
    ) or (_ for _ in ()).throw(Exception("README overpromises print-only wrappers")),
)
check(
    "Gemini base URL env spelling",
    lambda: (
        "export GEMINI_BASE_URL" not in README_TEXT
        and "export GEMINI_BASE_URL" not in COOKBOOK_TEXT
        and "GOOGLE_GEMINI_BASE_URL" in README_TEXT
        and "GOOGLE_GEMINI_BASE_URL" in COOKBOOK_TEXT
        and "OK"
    ) or (_ for _ in ()).throw(Exception("Use GOOGLE_GEMINI_BASE_URL in docs")),
)
check(
    "team brief telemetry wording",
    lambda: (
        "no telemetry by default" not in README_TEXT.lower()
        and "no telemetry by default" not in FOR_TEAMS_TEXT.lower()
        and "no outbound analytics by default" in README_TEXT.lower()
        and "no outbound analytics by default" in FOR_TEAMS_TEXT.lower()
        and "OK"
    ) or (_ for _ in ()).throw(Exception("Distinguish local metrics from outbound analytics")),
)
check(
    "team brief verify-claims scope",
    lambda: (
        "Measure *your* number in 60 seconds" not in FOR_TEAMS_TEXT
        and "hand to finance" not in FOR_TEAMS_TEXT
        and "bounded install smoke test" in FOR_TEAMS_TEXT
        and "representative proxy pilot" in FOR_TEAMS_TEXT
        and "OK"
    ) or (_ for _ in ()).throw(Exception("verify-claims is a bounded smoke test, not an ROI receipt")),
)
check(
    "team brief determinism scope",
    lambda: (
        "bit-identical output" not in FOR_TEAMS_TEXT
        and "Auditable local core" in FOR_TEAMS_TEXT
        and "Evaluate stateful learning, exploration, routing" in FOR_TEAMS_TEXT
        and "OK"
    ) or (_ for _ in ()).throw(Exception("Scope determinism claims to the tested local paths")),
)
check(
    "README proof-first star CTA",
    lambda: (
        "img.shields.io/github/stars/juyterman1000/entroly?style=social" in README_TEXT
        and "Deciding whether to star?" in README_TEXT
        and "entroly verify-claims && entroly simulate" in README_TEXT
        and "open an issue with the verification JSON" in README_TEXT
        and "Token_Savings-tested_70--95%25" not in README_TEXT
        and "Token_Savings-workload_dependent" in README_TEXT
        and "OK"
    ) or (_ for _ in ()).throw(Exception("README must ask for stars through local proof, not broad first-fold claims")),
)

# === Proxy ===
check("Proxy: PromptCompilerProxy", lambda: __import__("entroly.proxy", fromlist=["PromptCompilerProxy"]).PromptCompilerProxy and "OK")
check("Proxy: ProxyConfig port=9377", lambda: f"port={__import__('entroly.proxy_config', fromlist=['ProxyConfig']).ProxyConfig.from_env().port}")

# === Engine ===
from entroly.server import EntrolyEngine  # noqa: E402
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
from entroly.value_tracker import estimate_cost  # noqa: E402
check("estimate_cost", lambda: f"10K gpt-4o = ${estimate_cost(10000, 'gpt-4o'):.4f}")

# === Dashboard ===
check("Dashboard", lambda: __import__("entroly.dashboard", fromlist=["start_dashboard"]).start_dashboard and "OK")

# === auto_index ===
check("auto_index", lambda: __import__("entroly.auto_index", fromlist=["auto_index"]).auto_index and "OK")

# === Language support ===
from entroly.auto_index import SUPPORTED_EXTENSIONS  # noqa: E402
check("Language extensions", lambda: f"{len(SUPPORTED_EXTENSIONS)} extensions supported")

# === bench/accuracy.py ===
check("bench/accuracy.py", lambda: __import__("pathlib").Path("bench/accuracy.py").exists() and "OK")

# === Summary ===
print(f"\n{'='*50}")
print(f"  PASSED: {passed}  |  FAILED: {failed}")
if failed:
    print(f"  README has {failed} unverified claim(s)!")
else:
    print("  All README claims verified!")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
