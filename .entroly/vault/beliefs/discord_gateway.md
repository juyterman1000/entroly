---
claim_id: d02ed3a7-b2ee-4916-8b93-557a5aa55eb6
entity: discord_gateway
status: inferred
confidence: 0.75
sources:
  - entroly\integrations\discord_gateway.py:39
  - entroly\integrations\discord_gateway.py:40
  - entroly\integrations\discord_gateway.py:55
  - entroly\integrations\discord_gateway.py:58
  - entroly\integrations\discord_gateway.py:68
  - entroly\integrations\discord_gateway.py:73
last_checked: 2026-04-23T03:07:07.828288+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: discord_gateway

**Language:** python
**Lines of code:** 161

## Types
- `class DiscordGateway()`

## Functions
- `def __init__(
        self,
        webhook_url: str,
        poll_interval_s: float = 30.0,
        username: str = "Entroly",
    )`
- `def attach(self, daemon: Any) -> None`
- `def start(self) -> None`
- `def stop(self) -> None`
- `def send(self, content: str) -> dict[str, Any]`

## Dependencies
- `__future__`
- `json`
- `logging`
- `os`
- `threading`
- `time`
- `typing`
- `urllib.request`
