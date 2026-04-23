---
claim_id: f0dac78d-e384-4f6c-8478-20cfbe178e96
entity: slack_gateway
status: inferred
confidence: 0.75
sources:
  - entroly\integrations\slack_gateway.py:36
  - entroly\integrations\slack_gateway.py:37
  - entroly\integrations\slack_gateway.py:50
  - entroly\integrations\slack_gateway.py:53
  - entroly\integrations\slack_gateway.py:63
  - entroly\integrations\slack_gateway.py:68
last_checked: 2026-04-23T03:07:07.830007+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: slack_gateway

**Language:** python
**Lines of code:** 154

## Types
- `class SlackGateway()`

## Functions
- `def __init__(
        self,
        webhook_url: str,
        poll_interval_s: float = 30.0,
    )`
- `def attach(self, daemon: Any) -> None`
- `def start(self) -> None`
- `def stop(self) -> None`
- `def send(self, text: str) -> dict[str, Any]`

## Dependencies
- `__future__`
- `json`
- `logging`
- `os`
- `threading`
- `time`
- `typing`
- `urllib.request`
