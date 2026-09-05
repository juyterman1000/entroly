from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly import response_contract


def test_response_contract_is_atomic_reversible_and_does_not_claim_savings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(response_contract, "_state_root", lambda _scope: tmp_path)

    first = response_contract.set_contract("concise")
    second = response_contract.set_contract("evidence")

    assert first["previous_digest"] is None
    assert second["backup"] is not None
    assert Path(second["backup"]).is_file()
    current = response_contract.load_contract(fall_back_to_user=False)
    assert current["name"] == "evidence"
    assert "not measured token savings" in second["claim_boundary"]
    assert json.loads(Path(first["path"]).read_text(encoding="utf-8"))["name"] == "evidence"


def test_unknown_response_contract_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(response_contract, "_state_root", lambda _scope: tmp_path)
    with pytest.raises(ValueError, match="unknown response contract"):
        response_contract.set_contract("telepathic")
