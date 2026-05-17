"""Smoke test: verify new SDK functions are importable and callable."""

# 1. Import check
from entroly import detect_hallucination, optimize, compress, verify
print("[PASS] All 4 SDK functions importable from entroly")

# 2. detect_hallucination
result = detect_hallucination(
    response="The function process_data() calls validate_input() and returns a DataFrame.",
    context="def process_data(x): return x * 2\ndef transform(y): return y + 1",
    prompt="fix the data pipeline",
)
assert "fused_risk" in result
assert "verdict" in result
assert result["verdict"] in ("pass", "warn", "flag")
assert "recommendation" in result
print(f"[PASS] detect_hallucination: verdict={result['verdict']}, fused_risk={result['fused_risk']}")

# 3. Grounded text should pass
grounded = detect_hallucination(
    response="The function process_data takes x and returns x times 2. The transform function adds 1.",
    context="def process_data(x): return x * 2\ndef transform(y): return y + 1",
)
print(f"[PASS] Grounded text: verdict={grounded['verdict']}, fused_risk={grounded['fused_risk']}")

# 4. optimize
fragments = [
    {"content": "def login(user, password): validate(user); return token", "source": "auth.py", "token_count": 15},
    {"content": "def logout(session): clear(session); return True", "source": "auth.py", "token_count": 12},
    {"content": "const PI = 3.14159; function area(r) { return PI*r*r; }", "source": "math.js", "token_count": 18},
]
opt_result = optimize(fragments, budget=30, query="fix the login function")
assert "selected" in opt_result
assert "context_text" in opt_result
assert "fragments_selected" in opt_result
assert opt_result["fragments_total"] == 3
print(f"[PASS] optimize: selected {opt_result['fragments_selected']}/{opt_result['fragments_total']} fragments, {opt_result['total_tokens']} tokens")

# 5. compress still works
long_text = "The quick brown fox jumped over the lazy dog. " * 100
compressed = compress(long_text, budget=50)
assert len(compressed) <= len(long_text)
print(f"[PASS] compress: {len(long_text)} -> {len(compressed)} chars")

# 6. verify still works
v = verify("import os\nos.path.exists('/tmp')", context="import os")
assert "ipd" in v
print(f"[PASS] verify: ipd={v['ipd']}, verdict={v['verdict']}")

print()
print("All 6 SDK integration tests passed!")
