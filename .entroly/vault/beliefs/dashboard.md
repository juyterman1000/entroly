---
claim_id: 64b65ffa-8a12-4075-b411-204d2e850f46
entity: dashboard
status: inferred
confidence: 0.75
sources:
  - entroly\dashboard.py:776
  - entroly\dashboard.py:38
  - entroly\dashboard.py:779
  - entroly\dashboard.py:789
  - entroly\dashboard.py:843
  - entroly\dashboard.py:280
last_checked: 2026-04-23T03:07:07.802384+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: dashboard

**Language:** python
**Lines of code:** 866

## Types
- `class DashboardHandler(BaseHTTPRequestHandler)` — HTTP handler for the dashboard.

## Functions
- `def record_request(entry: dict)` — Record a proxy request's metrics (called from proxy.py).
- `def log_message(self, format, *args)`
- `def do_GET(self)`
- `def start_dashboard(engine: Any = None, port: int = 9378, daemon: bool = True)` — Start the dashboard HTTP server in a background thread. Args: engine: The EntrolyEngine instance to pull real data from. port: Port to serve on (default: 9378). daemon: Run as daemon thread (dies with

## Dependencies
- `BaseHTTPRequestHandler`
- `__future__`
- `http.server`
- `json`
- `logging`
- `threading`
- `time`
- `typing`
