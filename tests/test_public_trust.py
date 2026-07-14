from __future__ import annotations

from scripts.verify_public_trust import collect_offline_failures


def test_prominent_public_trust_contracts() -> None:
    assert collect_offline_failures() == []
