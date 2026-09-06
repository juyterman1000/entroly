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


def test_environment_contract_survives_an_unresolvable_path(monkeypatch):
    """An optional pointer must not fail `entroly wrap`.

    `Path.home()` picks its flavour from ``os.name``. Any caller that has
    swapped it — the wrap tests do, to exercise the Windows shim path — makes
    `Path` construction raise on POSIX: `pathlib.UnsupportedOperation` on 3.13+,
    plain `NotImplementedError` earlier, both reported as "cannot instantiate
    'WindowsPath' on your system".

    Before this guard that turned an optional environment pointer into a hard
    failure of the wrap command, on every Linux wheel build. Returning `{}` is
    the same answer the function already gives when no contract exists.
    """
    from entroly import response_contract

    def explode(_scope: str = "project"):
        raise NotImplementedError("cannot instantiate 'WindowsPath' on your system")

    monkeypatch.setattr(response_contract, "contract_path", explode)
    assert response_contract.environment_contract() == {}
