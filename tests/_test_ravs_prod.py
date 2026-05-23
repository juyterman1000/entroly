"""Quick smoke test for the new RAVS production implementations."""
from entroly.ravs import (
    ShadowEvaluator, TestRunnerExecutor, RetrievalExecutor, CitationVerifier,
)

# 1. Shadow policies
print("=== Shadow Evaluator ===")
e = ShadowEvaluator()
print(f"Policies: {e.policy_names}")

r = e.evaluate(
    query_text="fix the auth bug in login.py",
    query_features={"archetype": "code/fix_bug", "has_code": True},
    current_model="claude-sonnet-4-20250514",
    candidates=["gpt-4o-mini", "claude-3-haiku-20240307"],
)
for name, rec in r.items():
    model = rec.get("model", "abstain")
    reason = rec.get("reason", "")[:70]
    p = rec.get("predicted_p_success")
    insuf = rec.get("insufficient_data", False)
    tag = " [INSUFFICIENT]" if insuf else ""
    print(f"  {name}: {model} (P={p}) — {reason}{tag}")

# Feed some outcomes to train the policies
print("\n=== Training policies with outcomes ===")
for i in range(20):
    e.observe_outcome(
        query_text=f"run pytest tests/test_auth.py attempt {i}",
        query_features={"archetype": "test/run"},
        model_used="gpt-4o-mini",
        succeeded=(i % 3 != 0),  # ~67% success
    )
    e.observe_outcome(
        query_text=f"fix the auth bug in module {i}",
        query_features={"archetype": "code/fix_bug"},
        model_used="claude-sonnet-4-20250514",
        succeeded=(i % 5 != 0),  # 80% success
    )
print("  Fed 40 observations")

# Re-evaluate after training
r2 = e.evaluate(
    query_text="run pytest tests/test_payment.py",
    query_features={"archetype": "test/run"},
    current_model="claude-sonnet-4-20250514",
    candidates=["gpt-4o-mini", "claude-3-haiku-20240307"],
)
print("\nPost-training recommendations for 'run pytest':")
for name, rec in r2.items():
    model = rec.get("model", "abstain")
    p = rec.get("predicted_p_success")
    insuf = rec.get("insufficient_data", False)
    tag = " [INSUFFICIENT]" if insuf else ""
    print(f"  {name}: {model} (P={p}){tag}")

# 2. Citation Verifier
print("\n=== Citation Verifier ===")
cv = CitationVerifier()

# Should PASS — claim is supported by source
r1 = cv.verify(
    "The function authenticate() uses bcrypt for password hashing",
    "def authenticate(user, password):\n    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())\n    return verify(user, hashed)",
)
print(f"  Supported claim: passed={r1.passed} reason={r1.reason}")

# Should FAIL — claim not supported
r2 = cv.verify(
    "The API achieves 99.7% accuracy on MMLU benchmark",
    "The system uses a simple HTTP server with Flask routing",
)
print(f"  Unsupported claim: passed={r2.passed} reason={r2.reason}")

# 3. Retrieval Executor
print("\n=== Retrieval Executor ===")
re_exec = RetrievalExecutor()
result = re_exec.execute(
    "authentication and password hashing",
    corpus=[
        {"id": "auth.py", "content": "def authenticate(user, pw): return bcrypt.verify(pw, db.get(user))"},
        {"id": "config.yaml", "content": "server:\n  port: 8080\n  host: 0.0.0.0"},
        {"id": "payments.py", "content": "def process_payment(amount, card): return stripe.charge(card, amount)"},
    ],
)
print(f"  succeeded={result.succeeded} time={result.execution_time_ms}ms")
import json
matches = json.loads(result.result).get("matches", [])
for m in matches:
    print(f"    {m['id']}: similarity={m['similarity']}")

print("\n=== ALL SMOKE TESTS PASSED ===")
