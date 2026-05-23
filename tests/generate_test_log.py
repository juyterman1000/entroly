"""
Generate a realistic RAVS event log for end-to-end CLI testing.
Simulates 20 requests across a typical coding session.
"""
import json, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from entroly.ravs.events import AppendOnlyEventLog, TraceEvent, OutcomeEvent
from entroly.ravs.shadow_runner import ShadowRunner

log_dir = os.path.join(os.path.dirname(__file__), "..", ".test_ravs_log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "events.jsonl")

# Wipe old
if os.path.exists(log_path):
    os.remove(log_path)

log = AppendOnlyEventLog(log_path)
runner = ShadowRunner()

# Simulated coding session
session = [
    # (query, model, cost_usd, latency_ms, outcome_type, outcome_value, strength)
    ("calculate the factorial of 12", "gpt-4o-mini", 0.0003, 120, "test_result", "passed", "strong"),
    ("list all functions in auth.py", "gpt-4o-mini", 0.0004, 95, "user_acceptance", "accepted", "strong"),
    ("refactor the login handler to use async/await", "gpt-4o", 0.008, 2100, "ci_result", "passed", "strong"),
    ("what is 2^32 - 1", "gpt-4o-mini", 0.0002, 65, "test_result", "passed", "strong"),
    ("explain why we use dependency injection here", "gpt-4o", 0.006, 1800, "agent_self_report", "success", "weak"),
    ("run pytest on tests/test_auth.py", "gpt-4o-mini", 0.0003, 110, "command_exit", "success", "strong"),
    ("find the signature of validate_token", "gpt-4o-mini", 0.0003, 88, "user_acceptance", "accepted", "strong"),
    ("compute the average response time from these logs: 120, 340, 89, 201, 456", "gpt-4o-mini", 0.0004, 130, "test_result", "passed", "strong"),
    ("write a migration script for the users table", "gpt-4o", 0.012, 3200, "ci_result", "failed", "strong"),
    ("debug the race condition in the cache module", "gpt-4o", 0.015, 4100, "test_result", "failed", "strong"),
    ("what does the API docs say about rate limiting", "gpt-4o-mini", 0.0003, 95, "user_acceptance", "accepted", "strong"),
    ("simplify the expression (a+b)^2 - a^2 - 2*a*b", "gpt-4o-mini", 0.0002, 70, "test_result", "passed", "strong"),
    ("add error handling to the payment processor", "gpt-4o", 0.009, 2500, "ci_result", "passed", "strong"),
    ("how many classes are in models.py", "gpt-4o-mini", 0.0002, 60, "user_acceptance", "accepted", "strong"),
    ("is it true that the config supports hot reloading", "gpt-4o-mini", 0.0003, 85, "agent_self_report", "success", "weak"),
    ("solve 3x + 7 = 22", "gpt-4o-mini", 0.0002, 55, "test_result", "passed", "strong"),
    ("run the full test suite", "gpt-4o-mini", 0.0004, 140, "command_exit", "success", "strong"),
    ("review the error handling in api/routes.py", "gpt-4o", 0.007, 1900, "user_acceptance", "rejected", "strong"),
    ("compute 15% of 2847.50", "gpt-4o-mini", 0.0002, 50, "test_result", "passed", "strong"),
    ("write comprehensive tests for the auth module", "gpt-4o", 0.011, 2800, "ci_result", "passed", "strong"),
]

base_ts = time.time() - 3600  # start 1 hour ago

for i, (query, model, cost, latency, etype, evalue, strength) in enumerate(session):
    rid = f"req-{i+1:03d}"
    ts = base_ts + i * 180  # 3 minutes apart

    # Shadow compile
    plan = runner.compile_and_run(query, request_id=rid, model=model, model_cost_usd=cost)
    decomp = [
        {"kind": n.kind, "source": "shadow_compiler",
         "executor": n.executor, "confidence": round(n.confidence, 2)}
        for n in plan.nodes if n.kind != "model_bound"
    ]

    # Shadow policy stubs
    shadow_recs = {
        "cost_optimizer": {
            "insufficient_data": (i < 5),
            "model": "gpt-4o-mini" if model == "gpt-4o" else model,
            "predicted_p_success": round(random.uniform(0.6, 0.95), 2) if i >= 5 else None,
            "policy_name": "cost_optimizer",
            "reason": "shadow estimate" if i >= 5 else "",
        },
    }

    log.write_trace(TraceEvent(
        request_id=rid, model=model, cost_usd=cost,
        latency_ms=latency, context_size_tokens=random.randint(3000, 12000),
        timestamp=ts, retrieved_fragments=[],
        decomposition_evidence=decomp,
        shadow_recommendations=shadow_recs,
    ))

    log.write_outcome(OutcomeEvent(
        request_id=rid, event_type=etype, value=evalue,
        strength=strength, source="e2e_simulation",
        include_in_default_training=(strength != "weak"),
        timestamp=ts + random.uniform(5, 60),
    ))

print(f"Generated {len(session)} requests to: {log_path}")
print(f"Log size: {os.path.getsize(log_path)} bytes")
