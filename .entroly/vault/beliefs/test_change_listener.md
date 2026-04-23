---
claim_id: 3ec7ea40-3d95-4548-8253-3a42ddb0a8a6
entity: test_change_listener
status: inferred
confidence: 0.75
sources:
  - tests\test_change_listener.py:8
  - tests\test_change_listener.py:14
  - tests\test_change_listener.py:34
  - tests\test_change_listener.py:37
last_checked: 2026-04-23T03:07:07.924229+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_change_listener

**Language:** python
**Lines of code:** 48


## Functions
- `def test_workspace_change_listener_syncs_changed_file(tmp_path)`
- `def verify_token(self, token: str) -> bool` — , encoding="utf-8", ) vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault"))) compiler = BeliefCompiler(vault) verifier = VerificationEngine(vault) change_pipe = ChangePipeline(vault, ver
- `def verify_token(self, token: str) -> bool`
- `def rotate_keys(self) -> None` — , encoding="utf-8", ) second = listener.scan_once() assert second.status == "synced" assert "auth.py" in second.changed_files assert second.beliefs_written >= 1 assert second.verification_summary["tot

## Dependencies
- `entroly.belief_compiler`
- `entroly.change_listener`
- `entroly.change_pipeline`
- `entroly.vault`
- `entroly.verification_engine`
