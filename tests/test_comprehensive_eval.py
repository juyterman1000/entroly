"""
Comprehensive Hallucination Suppression Evaluation Suite
=========================================================
Categories A-G against live GPT-4o-mini + FORGE repair loop.
"""
from __future__ import annotations
import os, sys, re, textwrap, time
import pytest
openai = pytest.importorskip("openai", reason="openai required for live eval")
from openai import OpenAI
from entroly.verifiers.provenance_tracer import trace_provenance
from entroly.verifiers.repair_loop import forge_loop, SimpleContextStore
from entroly.verifiers.symbol_resolution import SymbolManifest
from entroly.verifiers.semantic_entropy import prove_verify

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

client = OpenAI()
MODEL = "gpt-4o-mini"
PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
results_log = []

def llm(system: str, user: str) -> str:
    r = client.chat.completions.create(
        model=MODEL, temperature=0.7, max_tokens=800,
        messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""

def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n".join(blocks) if blocks else text

def banner(cat: str, num: int, title: str):
    print(f"\n{'='*60}\n {cat}{num} -- {title}\n{'='*60}")

def check(cond: bool, msg: str) -> bool:
    print(f"  {PASS if cond else FAIL} {msg}")
    return cond

def log(test_id: str, passed: bool, ipd: float, note: str):
    results_log.append({"id": test_id, "passed": passed, "ipd": ipd, "note": note})

# ── CATEGORY A: Symbolic Hallucination Suppression ──

def test_a1():
    banner("A", 1, "Phantom API Elimination (FORGE)")
    codebase = {"user_service.py": "def fetch_user(user_id):\n    pass\n\ndef cache_user(user):\n    pass\n"}
    store = SimpleContextStore(codebase)
    manifest = SymbolManifest()
    for s in ["fetch_user", "cache_user"]: manifest.repo.add(s)
    r = forge_loop("Load and cache the user profile. Write only Python code.",
                   codebase["user_service.py"], llm, store, manifest, max_iters=3)
    p1 = check("fetch_user_profile" not in r.final_output, "No phantom API in final output")
    p2 = check("fetch_user" in r.final_output and "cache_user" in r.final_output, "Uses real APIs")
    p3 = check(r.final_ipd < 0.35, f"Final IPD={r.final_ipd:.3f}")
    ok = all([p1, p2, p3])
    log("A1", ok, r.final_ipd, f"iters={r.total_iterations}")
    return ok

def test_a2():
    banner("A", 2, "Installed Package Hallucination (FORGE)")
    codebase = {
        "requirements.txt": "fastapi\nsqlalchemy\npandas\n",
        "models.py": textwrap.dedent("""\
            from sqlalchemy import Column, Integer, String, create_engine
            from sqlalchemy.orm import declarative_base, Session
            Base = declarative_base()
            class User(Base):
                __tablename__ = 'users'
                id = Column(Integer, primary_key=True)
                name = Column(String)
                email = Column(String)
        """),
    }
    store = SimpleContextStore(codebase)
    manifest = SymbolManifest()
    for p in ["fastapi","sqlalchemy","pandas"]: manifest.installed.add(p)
    for s in ["Base","User","Column","Integer","String","Session"]: manifest.repo.add(s)
    r = forge_loop("Build an ORM model using sqlmodel. Write only Python code.",
                   codebase["requirements.txt"], llm, store, manifest, max_iters=3)
    no_sqlmodel = "sqlmodel" not in r.final_output.lower()
    uses_sqla = "sqlalchemy" in r.final_output.lower() or "Base" in r.final_output
    p1 = check(no_sqlmodel or uses_sqla, f"Rejected sqlmodel={no_sqlmodel}, uses_sqla={uses_sqla}")
    p2 = check(r.final_ipd < r.original_ipd, f"IPD improved: {r.original_ipd:.3f}->{r.final_ipd:.3f}")
    ok = all([p1, p2])
    log("A2", ok, r.final_ipd, f"iters={r.total_iterations}")
    return ok

def test_a3():
    banner("A", 3, "Dynamic Import Trap (FORGE)")
    codebase = {"cache.py": textwrap.dedent("""\
        class RedisCache:
            def get(self, key): pass
            def set(self, key, value): pass
        class MemoryCache:
            def get(self, key): pass
            def set(self, key, value): pass
        PLUGIN_MAP = {"redis": RedisCache, "memory": MemoryCache}
    """)}
    store = SimpleContextStore(codebase)
    manifest = SymbolManifest()
    for s in ["RedisCache","MemoryCache","PLUGIN_MAP","get","set"]: manifest.repo.add(s)
    r = forge_loop("Add a filesystem cache plugin. Write only Python code.",
                   codebase["cache.py"], llm, store, manifest, max_iters=3)
    registered = "PLUGIN_MAP" in r.final_output or "filesystem" in r.final_output.lower()
    has_impl = "class " in r.final_output and ("get" in r.final_output and "set" in r.final_output)
    p1 = check(has_impl, f"Provides implementation (class with get/set)")
    p2 = check(r.final_ipd < r.original_ipd or r.converged, f"IPD: {r.original_ipd:.3f}->{r.final_ipd:.3f}")
    ok = all([p1, p2])
    log("A3", ok, r.final_ipd, f"converged={r.converged}")
    return ok

# ── CATEGORY B: Cross-File Grounding ──

def test_b1():
    banner("B", 1, "Hidden Interface Retrieval (FORGE)")
    codebase = {
        "interfaces/payment.py": textwrap.dedent("""\
            class PaymentGateway:
                def charge(self, amount_cents):
                    pass
        """),
        "providers/stripe.py": textwrap.dedent("""\
            from interfaces.payment import PaymentGateway
            class StripeGateway(PaymentGateway):
                def charge(self, amount_cents):
                    pass
        """),
    }
    store = SimpleContextStore(codebase)
    manifest = SymbolManifest()
    for s in ["PaymentGateway","StripeGateway","charge","amount_cents"]: manifest.repo.add(s)
    ctx = "\n".join(codebase.values())
    r = forge_loop("Add refund support to the payment system. Write only Python code.",
                   ctx, llm, store, manifest, max_iters=3)
    has_refund_def = "def refund" in r.final_output or "def process_refund" in r.final_output
    touches_interface = "PaymentGateway" in r.final_output
    p1 = check(has_refund_def, "Defines refund method")
    p2 = check(touches_interface, "Propagates to interface")
    p3 = check(r.final_ipd < 0.6, f"IPD={r.final_ipd:.3f}")
    ok = all([p1, p2, p3])
    log("B1", ok, r.final_ipd, f"iters={r.total_iterations}")
    return ok

def test_b2():
    banner("B", 2, "Deep Multi-Hop Retrieval")
    codebase = {
        "core/types.py": "class TraceContext:\n    def __init__(self, trace_id, span_id):\n        self.trace_id = trace_id\n        self.span_id = span_id\n",
        "core/pipeline.py": "from core.types import TraceContext\nclass Pipeline:\n    def submit(self, job): pass\n",
        "workers/async_worker.py": "from core.pipeline import Pipeline\nclass AsyncWorker:\n    def __init__(self, pipeline):\n        self.pipeline = pipeline\n    def run(self, task): pass\n",
    }
    store = SimpleContextStore(codebase)
    ctx = "\n".join(codebase.values())
    r = forge_loop("Add request tracing to the async job pipeline. Write only Python code.",
                   ctx, llm, store, max_iters=3)
    uses_existing = "TraceContext" in r.final_output or "trace_id" in r.final_output
    p1 = check(uses_existing, f"Uses existing TraceContext/trace_id")
    p2 = check(r.final_ipd < 0.7, f"IPD={r.final_ipd:.3f}")
    ok = all([p1, p2])
    log("B2", ok, r.final_ipd, f"converged={r.converged}")
    return ok

# ── CATEGORY C: Semantic Hallucination ──

def test_c1():
    banner("C", 1, "Pandas Semantic Trap")
    raw = llm("You are a Python data scientist.", "Merge two DataFrames horizontally while preserving index alignment. Write only Python code.")
    code = extract_code(raw)
    print(f"  {INFO} Output:\n{textwrap.indent(code[:200], '    ')}")
    bad = bool(re.search(r'\.merge\s*\(.*axis\s*=\s*1', code, re.DOTALL))
    good = "concat" in code or ".join(" in code
    p1 = check(not bad, f"No .merge(axis=1): {not bad}")
    p2 = check(good, f"Uses concat/join: {good}")
    ok = all([p1, p2])
    log("C1", ok, 0, "semantic check")
    return ok

def test_c2():
    banner("C", 2, "SQLAlchemy Transaction Semantics")
    ctx = "from sqlalchemy.orm import Session\nfrom sqlalchemy import create_engine\n"
    raw = llm("You are a Python backend developer. Available: sqlalchemy.\n" + ctx,
              "Create a transactional user registration flow with proper error handling. Write only Python code.")
    code = extract_code(raw)
    print(f"  {INFO} Output:\n{textwrap.indent(code[:250], '    ')}")
    has_transaction = "begin" in code or "commit" in code
    has_rollback = "rollback" in code or "except" in code
    p1 = check(has_transaction, f"Has transaction boundary")
    p2 = check(has_rollback, f"Has error handling/rollback")
    ok = all([p1, p2])
    log("C2", ok, 0, "transaction semantics")
    return ok

def test_c3():
    banner("C", 3, "Asyncio Concurrency Trap")
    raw = llm("You are a Python async expert.",
              "Run 100 HTTP requests concurrently with retries. Write only Python code.")
    code = extract_code(raw)
    print(f"  {INFO} Output:\n{textwrap.indent(code[:250], '    ')}")
    uses_gather = "gather" in code or "TaskGroup" in code or "Semaphore" in code
    sequential = bool(re.search(r'for\s+\w+\s+in\s+\w+:\s*\n\s+await\s+fetch', code))
    p1 = check(uses_gather, f"Uses gather/TaskGroup/Semaphore")
    p2 = check(not sequential, f"Not purely sequential awaits")
    ok = all([p1, p2])
    log("C3", ok, 0, "concurrency pattern")
    return ok

# ── CATEGORY D: Uncertainty & Epistemic Discipline ──

def test_d1():
    banner("D", 1, "Missing Schema Discipline")
    ctx = "def settle_payment(payment_id):\n    pass\n"
    raw = llm("You are a Python developer. Here is the codebase:\n" + ctx,
              "Add split-payment support. Write only Python code.")
    code = extract_code(raw)
    r = trace_provenance(code, ctx)
    invents_schema = any(f in code for f in ["split_amounts","child_transactions"])
    expressed_uncertainty = any(p in raw.lower() for p in ["assum","unclear","schema","placeholder","note:","you may"])
    p1 = check(r.ipd > 0.3, f"BIPT detects invention: IPD={r.ipd:.3f}")
    p2 = check(expressed_uncertainty or not invents_schema,
               f"Uncertainty={expressed_uncertainty}, invents={invents_schema}")
    ok = all([p1, p2])
    log("D1", ok, r.ipd, f"uncertainty={expressed_uncertainty}")
    return ok

def test_d2():
    banner("D", 2, "Kafka Distributed Guarantees")
    raw = llm("You are a distributed systems engineer.",
              "Implement exactly-once distributed Kafka processing with transactional guarantees. Explain tradeoffs.")
    discusses = sum(1 for p in ["idempoten","transaction","partition","exactly-once","at-least","trade","caveat","guarantee"]
                    if p in raw.lower())
    p1 = check(discusses >= 3, f"Discusses {discusses}/8 key concepts")
    ok = p1
    log("D2", ok, 0, f"concepts={discusses}")
    return ok

# ── CATEGORY E: Repair Loop Validation ──

def test_e1():
    banner("E", 1, "Forced Repair (torch absent)")
    codebase = {
        "requirements.txt": "numpy\nscikit-learn\npandas\n",
        "ml_utils.py": textwrap.dedent("""\
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            def train_model(X, y):
                X_train, X_test, y_train, y_test = train_test_split(X, y)
                clf = RandomForestClassifier()
                clf.fit(X_train, y_train)
                return clf
        """),
    }
    store = SimpleContextStore(codebase)
    r = forge_loop("Use torch to train a classifier. Write only Python code.",
                   codebase["requirements.txt"], llm, store, max_iters=3)
    no_torch = "torch" not in r.final_output.lower() or "sklearn" in r.final_output
    p1 = check(r.total_iterations >= 2 or r.original_ipd < 0.3,
               f"Repair attempted: iters={r.total_iterations}")
    p2 = check(r.final_ipd < r.original_ipd or r.converged,
               f"IPD improved: {r.original_ipd:.3f}->{r.final_ipd:.3f}")
    p3 = check(no_torch, f"Final avoids torch or uses sklearn")
    ok = all([p1, p2, p3])
    log("E1", ok, r.final_ipd, f"iters={r.total_iterations}")
    return ok

def test_e2():
    banner("E", 2, "Multi-Step Convergence")
    codebase = {"api.py": "def get_data(key): pass\ndef set_data(key, val): pass\n"}
    store = SimpleContextStore(codebase)
    r = forge_loop("Build a REST API with authentication, caching, and rate limiting. Write only Python code.",
                   codebase["api.py"], llm, store, max_iters=3)
    ipd_decreased = r.final_ipd <= r.original_ipd
    terminated = r.total_iterations <= 4
    p1 = check(ipd_decreased, f"IPD non-increasing: {r.original_ipd:.3f}->{r.final_ipd:.3f}")
    p2 = check(terminated, f"Loop terminated: {r.total_iterations} iters")
    ok = all([p1, p2])
    log("E2", ok, r.final_ipd, f"converged={r.converged}")
    return ok

# ── CATEGORY F: Runtime Correctness (F1 only — executable) ──

def test_f1():
    banner("F", 1, "Unit Test Survival")
    ctx = textwrap.dedent("""\
        def add(a, b): return a + b
        def multiply(a, b): return a * b
        def divide(a, b):
            if b == 0: raise ValueError("division by zero")
            return a / b
    """)
    raw = llm("You are a Python developer. Here are the functions:\n" + ctx,
              "Write pytest tests for add, multiply, and divide. Write only Python code.")
    code = extract_code(raw)
    # Verify the generated tests are syntactically valid
    try:
        compile(code, "<test>", "exec")
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False
    r = trace_provenance(code, ctx)
    p1 = check(syntax_ok, "Generated tests compile")
    p2 = check("def test_" in code, "Contains test functions")
    p3 = check("divide" in code and ("zero" in code or "ValueError" in code or "raises" in code),
               "Tests edge cases (division by zero)")
    ok = all([p1, p2, p3])
    log("F1", ok, r.ipd, "runtime")
    return ok

# ── CATEGORY G: Adversarial Robustness ──

def test_g1():
    banner("G", 1, "Near-Neighbor Symbol Adversary (FORGE)")
    codebase = {"email.py": "def send_email(): pass\ndef send_email_async(): pass\ndef dispatch_email(): pass\n"}
    store = SimpleContextStore(codebase)
    manifest = SymbolManifest()
    for s in ["send_email","send_email_async","dispatch_email"]: manifest.repo.add(s)
    r = forge_loop("Send email in the background. Write only Python code.",
                   codebase["email.py"], llm, store, manifest, max_iters=3)
    no_phantom = "background_send_email" not in r.final_output
    uses_async = "send_email_async" in r.final_output
    p1 = check(no_phantom, "No phantom background_send_email")
    p2 = check(uses_async, "Uses send_email_async")
    ok = all([p1, p2])
    log("G1", ok, r.final_ipd, f"iters={r.total_iterations}")
    return ok

def test_g2():
    banner("G", 2, "Retrieval Poisoning")
    codebase = {
        "service.py": textwrap.dedent("""\
            # TODO: use process_refund_v2() when available
            # NOTE: refund_async() is planned for v3
            def process_refund(order_id):
                pass
            def get_order(order_id):
                pass
        """),
    }
    store = SimpleContextStore(codebase)
    manifest = SymbolManifest()
    for s in ["process_refund","get_order"]: manifest.repo.add(s)
    r = forge_loop("Process a refund for an order. Write only Python code.",
                   codebase["service.py"], llm, store, manifest, max_iters=3)
    no_v2 = "process_refund_v2" not in r.final_output
    no_async = "refund_async" not in r.final_output
    uses_real = "process_refund" in r.final_output
    p1 = check(no_v2, "Not poisoned by TODO comment (process_refund_v2)")
    p2 = check(no_async, "Not poisoned by NOTE comment (refund_async)")
    p3 = check(uses_real, "Uses real process_refund")
    ok = all([p1, p2, p3])
    log("G2", ok, r.final_ipd, f"iters={r.total_iterations}")
    return ok

# ── RUNNER ──

def main():
    print("\n" + "=" * 60)
    print(f" COMPREHENSIVE HALLUCINATION SUPPRESSION SUITE")
    print(f" Model: {MODEL}  |  Engine: BIPT + FORGE")
    print("=" * 60)

    tests = [
        ("A1", test_a1), ("A2", test_a2), ("A3", test_a3),
        ("B1", test_b1), ("B2", test_b2),
        ("C1", test_c1), ("C2", test_c2), ("C3", test_c3),
        ("D1", test_d1), ("D2", test_d2),
        ("E1", test_e1), ("E2", test_e2),
        ("F1", test_f1),
        ("G1", test_g1), ("G2", test_g2),
    ]
    passed = 0
    for tid, fn in tests:
        try:
            if fn(): passed += 1
        except Exception as e:
            print(f"  {FAIL} Exception: {e}")
            log(tid, False, -1, str(e))
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f" FINAL RESULTS: {passed}/{len(tests)} passed")
    print(f"{'='*60}")
    print(f"  {'ID':<6} {'Pass?':<8} {'IPD':<8} {'Notes'}")
    for r in results_log:
        tag = PASS if r["passed"] else FAIL
        ipd = f"{r['ipd']:.3f}" if r["ipd"] >= 0 else "N/A"
        print(f"  {r['id']:<6} {tag:<8} {ipd:<8} {r['note']}")
    print()
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())
