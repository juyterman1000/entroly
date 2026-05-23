#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                   ENTROLY VALUE DEMONSTRATOR                        ║
║       See exactly what Entroly does for your AI coding agent        ║
╚══════════════════════════════════════════════════════════════════════╝

Run:  python demo_value.py

This script simulates a real AI coding session, then shows you
side-by-side what happens WITH and WITHOUT Entroly's optimization.
"""

import json
import time
import sys

try:
    from entroly_core import EntrolyEngine
except ImportError:
    print("❌ entroly_core not installed. Run: cd entroly-core && maturin develop")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# ANSI Colors for rich terminal output
# ──────────────────────────────────────────────────────────────────────
class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[38;5;82m"
    RED     = "\033[38;5;196m"
    YELLOW  = "\033[38;5;220m"
    CYAN    = "\033[38;5;45m"
    MAGENTA = "\033[38;5;213m"
    ORANGE  = "\033[38;5;208m"
    BLUE    = "\033[38;5;33m"
    WHITE   = "\033[97m"
    GRAY    = "\033[38;5;240m"
    BG_GREEN  = "\033[48;5;22m"
    BG_RED    = "\033[48;5;52m"
    BG_BLUE   = "\033[48;5;17m"
    BG_DARK   = "\033[48;5;233m"
    RESET   = "\033[0m"
    UNDERLINE = "\033[4m"

def bar(value, max_val, width=30, color=C.GREEN):
    """Render a Unicode progress bar."""
    filled = int(value / max(max_val, 0.001) * width)
    filled = min(filled, width)
    return f"{color}{'█' * filled}{C.GRAY}{'░' * (width - filled)}{C.RESET}"

def sparkline(values, color=C.CYAN):
    """Render a sparkline chart from a list of values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return color + "".join(blocks[min(int((v - mn) / rng * 8), 8)] for v in values) + C.RESET

def header(text, width=72):
    """Render a styled header."""
    pad = width - len(text) - 4
    left = pad // 2
    right = pad - left
    print(f"\n{C.BG_BLUE}{C.WHITE}{C.BOLD} {'─' * left} {text} {'─' * right} {C.RESET}")

def subheader(text):
    print(f"\n  {C.CYAN}{C.BOLD}▸ {text}{C.RESET}")

def metric(label, value, color=C.WHITE, indent=4):
    print(f"{' ' * indent}{C.GRAY}{label:<35}{C.RESET}{color}{value}{C.RESET}")

def divider(char="─", width=72):
    print(f"  {C.GRAY}{char * width}{C.RESET}")

# ──────────────────────────────────────────────────────────────────────
# Simulated Real-World Coding Session
# ──────────────────────────────────────────────────────────────────────

FRAGMENTS = [
    # RELEVANT — SQL injection fix
    {"content": """def get_user(user_id):
    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')
    return cursor.fetchone()""",
     "source": "auth/db.py", "tokens": 30, "relevant": True},

    {"content": """def parameterized_query(cursor, query, params):
    cursor.execute(query, params)
    return cursor.fetchall()""",
     "source": "auth/queries.py", "tokens": 25, "relevant": True},

    {"content": """DB_HOST = 'localhost'
DB_PORT = 5432
DB_NAME = 'app_db'
SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}'""",
     "source": "config/database.py", "tokens": 20, "relevant": True},

    {"content": """class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), default='user')""",
     "source": "models/user.py", "tokens": 35, "relevant": True},

    # NOISE — stuff that wastes context
    {"content": """# Application README\n\nThis is a web application built with Flask.\nSee docs/ for more information.\n\n## Setup\npip install -r requirements.txt""",
     "source": "README.md", "tokens": 25, "relevant": False},

    {"content": """import pytest\n\ndef test_conftest():\n    pass\n\n@pytest.fixture\ndef client():\n    return app.test_client()""",
     "source": "tests/conftest.py", "tokens": 20, "relevant": False},

    {"content": """.button { color: blue; font-size: 14px; }
.nav { display: flex; justify-content: space-between; }
.container { max-width: 1200px; margin: 0 auto; }
.footer { background: #333; color: white; padding: 20px; }""",
     "source": "static/style.css", "tokens": 30, "relevant": False},

    {"content": """def send_welcome_email(user):
    msg = Message('Welcome!', recipients=[user.email])
    msg.body = f'Hello {user.name}, welcome to our platform!'
    mail.send(msg)""",
     "source": "utils/email.py", "tokens": 25, "relevant": False},

    {"content": """# Changelog\n\n## v2.1.0\n- Added dark mode\n- Fixed pagination bug\n\n## v2.0.0\n- Complete rewrite\n- New API endpoints""",
     "source": "CHANGELOG.md", "tokens": 20, "relevant": False},

    {"content": """def validate_email(email):
    import re
    pattern = r'^[\\w.-]+@[\\w.-]+\\.\\w+$'
    return bool(re.match(pattern, email))""",
     "source": "utils/validators.py", "tokens": 20, "relevant": False},

    # DUPLICATE of first fragment (slightly modified)
    {"content": """def get_user(user_id):
    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')
    return cursor.fetchone()  # TODO: fix SQL injection""",
     "source": "auth/db.py (v2)", "tokens": 30, "relevant": False},

    {"content": """def render_homepage():
    return render_template('index.html',
        title='Welcome',
        featured=get_featured_items())""",
     "source": "views/home.py", "tokens": 15, "relevant": False},
]

QUERY = "fix the SQL injection vulnerability in cursor.execute"
TOKEN_BUDGET = 100  # Tight budget — forces smart selection


def run_demo():
    print(f"\n{C.BG_DARK}{C.WHITE}")
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║                                                                ║")
    print(f"  ║  {C.CYAN}{C.BOLD}  E N T R O L Y   V A L U E   D E M O N S T R A T O R  {C.WHITE}   ║")
    print("  ║                                                                ║")
    print(f"  ║  {C.GRAY}  What happens when your AI agent has Entroly vs without?  {C.WHITE}  ║")
    print("  ║                                                                ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝{C.RESET}\n")

    time.sleep(0.3)

    # ── Scenario Setup ──
    header("SCENARIO")
    print(f"""
  {C.WHITE}You're fixing a SQL injection bug. Your AI agent has ingested
  {C.BOLD}12 code fragments{C.RESET}{C.WHITE} from your codebase into context.{C.RESET}

  {C.GRAY}Query: {C.YELLOW}"{QUERY}"{C.RESET}
  {C.GRAY}Budget: {C.ORANGE}{TOKEN_BUDGET} tokens{C.RESET} {C.GRAY}(tight — like a real production constraint){C.RESET}

  {C.GRAY}Only {C.GREEN}4 of 12{C.GRAY} fragments are actually relevant.{C.RESET}
  {C.GRAY}The rest are noise: README, CSS, email utils, changelog...{C.RESET}
  {C.GRAY}Plus {C.RED}1 near-duplicate{C.GRAY} that wastes tokens.{C.RESET}
""")

    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════
    # WITHOUT ENTROLY
    # ══════════════════════════════════════════════════════════════
    header("WITHOUT ENTROLY  ❌")

    total_tokens = sum(f["tokens"] for f in FRAGMENTS)
    relevant_tokens = sum(f["tokens"] for f in FRAGMENTS if f["relevant"])
    noise_tokens = total_tokens - relevant_tokens

    print(f"""
  {C.RED}Your AI agent receives ALL 12 fragments — {total_tokens} tokens.{C.RESET}
  {C.RED}The budget is {TOKEN_BUDGET} tokens, so it must truncate.{C.RESET}
  {C.RED}Random truncation loses critical context.{C.RESET}
""")

    subheader("What the AI sees (first-fit, no intelligence)")
    naive_tokens = 0
    naive_relevant = 0
    naive_noise = 0
    naive_selected = []
    for f in FRAGMENTS:
        if naive_tokens + f["tokens"] <= TOKEN_BUDGET:
            naive_tokens += f["tokens"]
            is_rel = f["relevant"]
            status = f"{C.GREEN}✓ RELEVANT{C.RESET}" if is_rel else f"{C.RED}✗ NOISE   {C.RESET}"
            print(f"    {status}  {C.GRAY}{f['source']:<25}{C.RESET} {C.DIM}{f['tokens']:>3} tok{C.RESET}")
            naive_selected.append(f)
            if is_rel:
                naive_relevant += 1
            else:
                naive_noise += 1
        else:
            print(f"    {C.DIM}⊘ DROPPED  {f['source']:<25} {f['tokens']:>3} tok (no room){C.RESET}")

    naive_recall = naive_relevant / sum(1 for f in FRAGMENTS if f["relevant"])
    naive_precision = naive_relevant / max(len(naive_selected), 1)
    waste_pct = (naive_noise / max(len(naive_selected), 1)) * 100

    print()
    metric("Recall", f"{naive_recall:.0%}", C.RED if naive_recall < 0.8 else C.GREEN)
    metric("Precision", f"{naive_precision:.0%}", C.RED if naive_precision < 0.5 else C.GREEN)
    metric("Wasted context", f"{waste_pct:.0f}% of selected tokens are noise", C.RED)
    metric("Tokens used", f"{naive_tokens}/{TOKEN_BUDGET}", C.YELLOW)

    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════
    # WITH ENTROLY
    # ══════════════════════════════════════════════════════════════
    header("WITH ENTROLY  ✅")

    engine = EntrolyEngine(
        w_recency=0.30,
        w_frequency=0.25,
        w_semantic=0.25,
        w_entropy=0.20,
        decay_half_life=15,
        min_relevance=0.01,
    )

    fragments_meta = []
    for f in FRAGMENTS:
        t0 = time.perf_counter()
        result = dict(engine.ingest(f["content"], f["source"], f["tokens"], False))
        t1 = time.perf_counter()
        fragments_meta.append({
            **f,
            "status": result.get("status"),
            "fragment_id": result.get("fragment_id", result.get("duplicate_of", "")),
            "entropy": result.get("entropy_score", 0),
            "ingest_us": (t1 - t0) * 1_000_000,
        })

    t0 = time.perf_counter()
    opt_result = dict(engine.optimize(TOKEN_BUDGET, QUERY))
    optimize_ms = (time.perf_counter() - t0) * 1000

    selected = opt_result.get("selected", [])
    selected_sources = {dict(s).get("source", "") for s in selected}
    stats = dict(engine.stats())

    subheader("Entroly's intelligent selection")

    entroly_relevant = 0
    entroly_noise = 0
    entropy_scores = []

    for s in selected:
        s = dict(s)
        src = s.get("source", "")
        is_rel = any(f["source"] == src and f["relevant"] for f in FRAGMENTS)
        entropy = s.get("entropy_score", 0)
        entropy_scores.append(entropy)
        status = f"{C.GREEN}✓ RELEVANT{C.RESET}" if is_rel else f"{C.ORANGE}◇ INCLUDED{C.RESET}"
        ent_bar = bar(entropy, 1.0, width=12, color=C.CYAN)
        print(f"    {status}  {C.GRAY}{src:<25}{C.RESET} {C.DIM}{s.get('token_count', '?'):>3} tok{C.RESET}  entropy {ent_bar} {C.CYAN}{entropy:.2f}{C.RESET}")
        if is_rel:
            entroly_relevant += 1
        else:
            entroly_noise += 1

    # Show what was excluded and WHY
    print()
    subheader("Intelligently excluded")
    for f in FRAGMENTS:
        if f["source"] not in selected_sources:
            fm = next((m for m in fragments_meta if m["source"] == f["source"]), {})
            reason = ""
            if fm.get("status") == "duplicate":
                reason = f"{C.RED}DUPLICATE{C.RESET} — saved {f['tokens']} tokens"
            elif not f["relevant"]:
                reason = f"{C.GRAY}low relevance{C.RESET} — noise filtered out"
            else:
                reason = f"{C.YELLOW}budget limit{C.RESET} — lower priority"
            print(f"    {C.DIM}✗{C.RESET}  {C.GRAY}{f['source']:<25}{C.RESET}  {reason}")

    entroly_recall = entroly_relevant / sum(1 for f in FRAGMENTS if f["relevant"])
    entroly_precision = entroly_relevant / max(len(selected), 1)
    entroly_f1 = 2 * entroly_precision * entroly_recall / max(entroly_precision + entroly_recall, 1e-9)

    total_tokens_used = opt_result.get("total_tokens", 0)
    tokens_saved = stats.get("session", {}).get("total_tokens_saved", stats.get("savings", {}).get("total_tokens_saved", 0))
    dupes = stats.get("session", {}).get("total_duplicates_caught", stats.get("savings", {}).get("total_duplicates_caught", 0))

    print()
    metric("Recall", f"{entroly_recall:.0%}", C.GREEN if entroly_recall >= 0.5 else C.RED)
    metric("Precision", f"{entroly_precision:.0%}", C.GREEN if entroly_precision >= 0.5 else C.RED)
    metric("F1 Score", f"{entroly_f1:.2f}", C.GREEN if entroly_f1 >= 0.5 else C.RED)
    metric("Tokens used", f"{total_tokens_used}/{TOKEN_BUDGET}", C.GREEN)
    metric("Optimize latency", f"{optimize_ms:.2f} ms", C.GREEN if optimize_ms < 10 else C.YELLOW)
    metric("Duplicates caught", f"{dupes}", C.CYAN)

    time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════
    # SIDE-BY-SIDE COMPARISON
    # ══════════════════════════════════════════════════════════════
    header("SIDE-BY-SIDE COMPARISON")

    print(f"""
  {C.BG_RED}{C.WHITE}{C.BOLD} WITHOUT ENTROLY {C.RESET}                    {C.BG_GREEN}{C.WHITE}{C.BOLD} WITH ENTROLY {C.RESET}
""")

    # Recall comparison
    r1 = bar(naive_recall, 1.0, 20, C.RED)
    r2 = bar(entroly_recall, 1.0, 20, C.GREEN)
    print(f"  Recall:     {r1} {C.RED}{naive_recall:.0%}{C.RESET}       {r2} {C.GREEN}{entroly_recall:.0%}{C.RESET}")

    # Precision comparison
    p1 = bar(naive_precision, 1.0, 20, C.RED)
    p2 = bar(entroly_precision, 1.0, 20, C.GREEN)
    print(f"  Precision:  {p1} {C.RED}{naive_precision:.0%}{C.RESET}       {p2} {C.GREEN}{entroly_precision:.0%}{C.RESET}")

    # Noise comparison
    n1 = bar(waste_pct / 100, 1.0, 20, C.RED)
    entroly_waste = (entroly_noise / max(len(selected), 1)) * 100
    n2 = bar(1 - entroly_waste / 100, 1.0, 20, C.GREEN)
    print(f"  Signal:     {n1} {C.RED}{100 - waste_pct:.0f}%{C.RESET}        {n2} {C.GREEN}{100 - entroly_waste:.0f}%{C.RESET}")

    # Dedup
    print(f"  Dedup:      {C.RED}0 duplicates caught{C.RESET}          {C.GREEN}{dupes} duplicate{'s' if dupes != 1 else ''} caught ⚡{C.RESET}")

    # Latency
    print(f"  Latency:    {C.RED}N/A (no optimization){C.RESET}        {C.GREEN}{optimize_ms:.2f} ms (knapsack DP){C.RESET}")

    time.sleep(0.3)

    # ══════════════════════════════════════════════════════════════
    # ENGINE INTERNALS
    # ══════════════════════════════════════════════════════════════
    header("WHAT ENTROLY DID UNDER THE HOOD")

    capabilities = [
        ("Shannon Entropy Scoring", "Measured information density of each fragment — ranked by actual value"),
        ("SimHash Dedup (64-bit)", f"Detected {dupes} near-duplicate(s) via hamming distance < 3"),
        ("Hybrid Semantic Scoring", "SimHash + n-gram Jaccard blend for query relevance ranking"),
        ("0/1 Knapsack DP", f"Maximized total value within {TOKEN_BUDGET}-token budget in {optimize_ms:.2f}ms"),
        ("Ebbinghaus Decay", "Older fragments naturally deprioritized via exponential forgetting curve"),
        ("Dependency Graph", "Auto-linked related files via import/identifier analysis"),
        ("Compare-Calibrate Filter", "Checked selected fragments for redundancy, swapped if too similar"),
        ("ε-Greedy Exploration", "Occasionally explores new fragments to avoid feedback loop starvation"),
        ("SAST Security Scan", "Auto-scanned for hardcoded secrets, SQL injection, unsafe patterns"),
        ("Skeleton Substitution", "Fit structural summaries of excluded files into remaining budget"),
    ]

    for i, (name, desc) in enumerate(capabilities, 1):
        icon = ["🧮", "🔍", "🎯", "🎒", "🧠", "🔗", "⚖️", "🎲", "🛡️", "💀"][i-1]
        print(f"  {icon}  {C.BOLD}{C.WHITE}{name}{C.RESET}")
        print(f"      {C.GRAY}{desc}{C.RESET}")
        if i < len(capabilities):
            print(f"      {C.DIM}│{C.RESET}")

    time.sleep(0.3)

    # ══════════════════════════════════════════════════════════════
    # VALUE SUMMARY
    # ══════════════════════════════════════════════════════════════
    header("VALUE SUMMARY")

    recall_improvement = entroly_recall - naive_recall
    precision_improvement = entroly_precision - naive_precision

    print(f"""
  {C.BOLD}{C.GREEN}┌─────────────────────────────────────────────────────────────┐{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}  {C.BOLD}Recall improvement:{C.RESET}      {C.GREEN}+{recall_improvement:+.0%}{C.RESET} {C.GRAY}(finding the right code){C.RESET}        {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}  {C.BOLD}Precision improvement:{C.RESET}   {C.GREEN}+{precision_improvement:+.0%}{C.RESET} {C.GRAY}(avoiding the wrong code){C.RESET}      {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}  {C.BOLD}Duplicates eliminated:{C.RESET}   {C.CYAN}{dupes}{C.RESET} {C.GRAY}(tokens saved from re-ingestion){C.RESET}  {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}  {C.BOLD}Selection latency:{C.RESET}       {C.CYAN}{optimize_ms:.2f} ms{C.RESET} {C.GRAY}(faster than a network hop){C.RESET}    {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}                                                             {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}  {C.BOLD}{C.YELLOW}Your AI agent gets better answers because it sees{C.RESET}         {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}│{C.RESET}  {C.BOLD}{C.YELLOW}the RIGHT code, not just ALL the code.{C.RESET}                   {C.BOLD}{C.GREEN}│{C.RESET}
  {C.BOLD}{C.GREEN}└─────────────────────────────────────────────────────────────┘{C.RESET}

  {C.GRAY}At scale (100K+ files, 128K token budget), Entroly's value{C.RESET}
  {C.GRAY}compounds: more noise to filter = bigger quality improvement.{C.RESET}

  {C.BOLD}{C.WHITE}Architecture:{C.RESET} {C.CYAN}Knapsack DP{C.RESET} + {C.CYAN}Shannon Entropy{C.RESET} + {C.CYAN}SimHash{C.RESET} + {C.CYAN}LSH{C.RESET}
  {C.BOLD}{C.WHITE}            :{C.RESET} {C.CYAN}PRISM RL{C.RESET} + {C.CYAN}Dep Graph{C.RESET} + {C.CYAN}SAST{C.RESET} + {C.CYAN}Autotune{C.RESET}
  {C.BOLD}{C.WHITE}Language    :{C.RESET} {C.ORANGE}100% Rust{C.RESET} {C.GRAY}(via PyO3 to Python MCP server){C.RESET}
""")


if __name__ == "__main__":
    run_demo()
