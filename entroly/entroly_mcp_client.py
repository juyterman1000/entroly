# Example: Python client for Entroly MCP server
import httpx

# Replace with your actual Entroly MCP server URL
ENTROLY_URL = "http://localhost:8000"

# Example: remember_fragment
fragment = {
    "content": "def process_payment(...): ...",
    "source": "payments.py",
    "token_count": 45
}
response = httpx.post(f"{ENTROLY_URL}/remember_fragment", json=fragment)
print("remember_fragment:", response.json())

# Example: optimize_context
optimize_payload = {
    "token_budget": 128000,
    "query": "fix payment bug"
}
response = httpx.post(f"{ENTROLY_URL}/optimize_context", json=optimize_payload)
print("optimize_context:", response.json())

# Add more tool calls as needed (recall_relevant, record_outcome, etc.)
