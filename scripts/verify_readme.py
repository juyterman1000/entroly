"""Verify Entroly's public README profiles against actual product contracts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        passed += 1
    except Exception as exc:  # noqa: BLE001 - verifier reports every failure
        print(f"  [FAIL] {name}: {exc}")
        failed += 1


def require(condition: bool, message: str) -> str:
    if not condition:
        raise AssertionError(message)
    return "OK"


from entroly.cli import _WRAP_AGENTS  # noqa: E402
from verify_public_trust import collect_offline_failures  # noqa: E402

README_TEXT = Path("README.md").read_text(encoding="utf-8")
PYPI_TEXT = Path("PYPI_README.md").read_text(encoding="utf-8")
PUBLIC_TEXT = README_TEXT + "\n" + PYPI_TEXT
COOKBOOK_TEXT = Path("cookbook/README.md").read_text(encoding="utf-8")
FOR_TEAMS_TEXT = Path("docs/for-teams.md").read_text(encoding="utf-8")
DOCS_DISCORD_TEXT = Path("docs/discord.html").read_text(encoding="utf-8")
INSTALL_TEXT = Path("scripts/install.sh").read_text(encoding="utf-8")
OPENCLAW_TEXT = Path("integrations/openclaw/README.md").read_text(encoding="utf-8")

# === Intentional public product profiles ===
check(
    "README developer Context OS identity",
    lambda: require(
        all(
            phrase in README_TEXT
            for phrase in (
                "Entroly — The Open-Source Context OS for AI Agents",
                "Keep your agent. Give it a Context OS.",
                "recoverable compression",
                "Context Receipts",
                "local-first",
            )
        ),
        "README no longer states the developer and AI-agent Context OS contract",
    ),
)
check(
    "PyPI Context Assurance identity",
    lambda: require(
        all(
            phrase in PYPI_TEXT
            for phrase in (
                "Entroly — Context Assurance That Helps Lower AI Costs",
                "small one-time setup",
                "no agent-architecture rewrite",
                "Context Receipts",
                "content-addressed recovery handles",
            )
        ),
        "PyPI first fold drifted from the Context Assurance identity",
    ),
)

# === Honest claim boundaries ===
check(
    "public cost and quality scope",
    lambda: require(
        all(
            phrase in PUBLIC_TEXT
            for phrase in (
                "does not promise a universal compression percentage",
                "guaranteed bill reduction",
                "subscription price may not change",
                "not a provider invoice",
                "does not establish universal truth",
                "not shipped yet",
            )
        ),
        "README/PyPI is missing a required cost, quality, or Simple Mode caveat",
    ),
)
check(
    "public copy rejects universal promises",
    lambda: require(
        all(
            phrase.lower() not in PUBLIC_TEXT.lower()
            for phrase in (
                "we guarantee savings",
                "zero setup required",
                "works with every ai app",
                "70–95% savings",
                "same accuracy",
            )
        ),
        "README/PyPI contains an unverified universal promise",
    ),
)

# === Trust and discovery links ===
check(
    "canonical trust links across public profiles",
    lambda: require(
        all(
            link in PUBLIC_TEXT
            for link in (
                "docs/ai-cost-optimization.html",
                "docs/public-evidence.md",
                "docs/limitations.md",
                "docs/benchmarks/neural-evidence-frontier.md",
                "docs/benchmarks/model-triggered-recovery.md",
                "benchmarks/results/context_commit_conformance.json",
            )
        ),
        "README/PyPI must link the cost guide, evidence policy, limitations, and artifacts",
    ),
)
check(
    "supported integration names",
    lambda: require(
        all(
            client in PUBLIC_TEXT
            for client in (
                "Claude Code",
                "Codex",
                "OpenClaw",
                "Hermes Agent",
                "OpenCode",
                "GitHub Copilot",
                "local models",
            )
        ),
        "README/PyPI is missing a supported integration named in the product surface",
    ),
)

# === Runtime and command contracts ===
for agent in ("claude", "codex", "aider", "cursor", "copilot", "opencode", "hermes"):
    check(
        f"wrap registry: {agent}",
        lambda agent=agent: require(agent in _WRAP_AGENTS, f"missing wrapper {agent}"),
    )
check(
    "Gemini base URL spelling",
    lambda: require(
        "export GEMINI_BASE_URL" not in README_TEXT + COOKBOOK_TEXT
        and "GOOGLE_GEMINI_BASE_URL" in README_TEXT
        and "GOOGLE_GEMINI_BASE_URL" in COOKBOOK_TEXT,
        "Use GOOGLE_GEMINI_BASE_URL in public setup instructions",
    ),
)
check(
    "MCP launch distinction",
    lambda: require(
        "For an MCP client, register the installed `entroly` command with no arguments"
        in PYPI_TEXT
        and "uvx --from entroly entroly" in PYPI_TEXT
        and "npx -y entroly-mcp" in PYPI_TEXT
        and "uvx --from entroly entroly serve" not in PYPI_TEXT,
        "PyPI MCP instructions no longer match the installed stdio launcher",
    ),
)
check(
    "OpenClaw public install path",
    lambda: require(
        "openclaw plugins install npm:entroly-openclaw" in OPENCLAW_TEXT
        and "openclaw plugins enable entroly" in OPENCLAW_TEXT
        and "openclaw gateway restart" in OPENCLAW_TEXT
        and "openclaw plugins install clawhub:entroly-openclaw" not in OPENCLAW_TEXT,
        "OpenClaw docs must default to the public npm package",
    ),
)
check(
    "outbound analytics wording",
    lambda: require(
        "no telemetry by default" not in README_TEXT.lower()
        and "no telemetry by default" not in FOR_TEAMS_TEXT.lower()
        and "no outbound analytics by default" in README_TEXT.lower()
        and "no outbound analytics by default" in FOR_TEAMS_TEXT.lower(),
        "Distinguish local metrics from outbound analytics",
    ),
)
check(
    "community link integrity",
    lambda: require(
        "discord.gg/Xp7VwWnJNY" not in README_TEXT + DOCS_DISCORD_TEXT + INSTALL_TEXT
        and "discord.gg/entroly" not in README_TEXT + DOCS_DISCORD_TEXT + INSTALL_TEXT
        and "https://discord.gg/G833X5c7R6" in DOCS_DISCORD_TEXT
        and "https://juyterman1000.github.io/entroly/docs/discord.html" in README_TEXT
        and "https://juyterman1000.github.io/entroly/docs/discord.html" in INSTALL_TEXT,
        "Public community links must not route to expired invites",
    ),
)

# === Shared public-trust gate ===
check(
    "public trust contracts",
    lambda: require(
        not (failures := collect_offline_failures()),
        "; ".join(failures),
    ),
)

# === SDK and runtime smoke checks ===
check("SDK: compress import", lambda: __import__("entroly").compress and "OK")
check(
    "SDK: compress_messages import",
    lambda: __import__("entroly").compress_messages and "OK",
)
check(
    "SDK: compress works",
    lambda: f"{len(__import__('entroly').compress('hello ' * 500, budget=50))} chars",
)
check(
    "Proxy: PromptCompilerProxy",
    lambda: __import__("entroly.proxy", fromlist=["PromptCompilerProxy"]).PromptCompilerProxy
    and "OK",
)
check(
    "Proxy: ProxyConfig port=9377",
    lambda: f"port={__import__('entroly.proxy_config', fromlist=['ProxyConfig']).ProxyConfig.from_env().port}",
)

from entroly.server import EntrolyEngine  # noqa: E402

engine = EntrolyEngine()
check("Engine: backend", lambda: f"use_rust={engine._use_rust}")
check(
    "Engine: ingest_fragment",
    lambda: engine.ingest_fragment("def foo(): pass", "test.py", 5) and "OK",
)
result = engine.optimize_context(token_budget=8000, query="foo")
check(
    "Engine: optimize_context",
    lambda: f"{len(result.get('selected_fragments', []))} frags",
)
check(
    "Federation",
    lambda: __import__("entroly.federation", fromlist=["FederationClient"]).FederationClient
    and "OK",
)
check(
    "CCR reversible",
    lambda: __import__("entroly.ccr", fromlist=["get_ccr_store"]).get_ccr_store
    and "OK",
)

from entroly.value_tracker import estimate_cost  # noqa: E402

check("estimate_cost", lambda: f"10K gpt-4o = ${estimate_cost(10000, 'gpt-4o'):.4f}")
check(
    "Dashboard",
    lambda: __import__("entroly.dashboard", fromlist=["start_dashboard"]).start_dashboard
    and "OK",
)
check(
    "auto_index",
    lambda: __import__("entroly.auto_index", fromlist=["auto_index"]).auto_index
    and "OK",
)

from entroly.auto_index import SUPPORTED_EXTENSIONS  # noqa: E402

check("Language extensions", lambda: f"{len(SUPPORTED_EXTENSIONS)} extensions supported")
check(
    "bench/accuracy.py",
    lambda: Path("bench/accuracy.py").exists() and "OK",
)

print(f"\n{'=' * 50}")
print(f"  PASSED: {passed}  |  FAILED: {failed}")
if failed:
    print(f"  Public README profiles have {failed} failed product or trust contract(s).")
else:
    print("  Developer README, PyPI profile, and runtime contracts passed.")
    print("  This does not certify every product or benchmark claim.")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)
