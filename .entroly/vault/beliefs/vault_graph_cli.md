---
claim_id: adb2e318-8c8c-4e52-a858-c5caf4d38a2f
entity: vault_graph_cli
status: inferred
confidence: 0.75
sources:
  - scripts\vault_graph_cli.py:14
  - scripts\vault_graph_cli.py:15
  - scripts\vault_graph_cli.py:92
  - scripts\vault_graph_cli.py:114
  - scripts\vault_graph_cli.py:142
  - scripts\vault_graph_cli.py:164
  - scripts\vault_graph_cli.py:174
  - scripts\vault_graph_cli.py:192
last_checked: 2026-04-23T03:07:07.919206+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: vault_graph_cli

**Language:** python
**Lines of code:** 247

## Types
- `class VaultGraphParser()`

## Functions
- `def __init__(self, vault_path: str)`
- `def show_file_info(self, filename: str)` — Show detailed info about a file
- `def show_graph(self, filename: str = None, depth: int = 2, direction: str = 'forward')` — Show graph relationship tree
- `def find_path(self, start: str, end: str) -> Optional[List[str]]` — Find path between two files using BFS
- `def list_files(self, pattern: str = None)` — List all files, optionally filtered by pattern
- `def show_stats(self)` — Show vault statistics
- `def main()`

## Dependencies
- `collections`
- `json`
- `os`
- `pathlib`
- `re`
- `sys`
- `typing`
