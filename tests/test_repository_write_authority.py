from __future__ import annotations

from pathlib import Path

import pytest

from entroly.repository_intelligence import service as service_module
from entroly.repository_intelligence.write_authority import (
    REPOSITORY_WRITES_ENV,
    RepositoryWriteAuthorizationError,
)


def test_service_snapshots_disabled_authority_at_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REPOSITORY_WRITES_ENV, raising=False)
    service = service_module.RepositoryIntelligenceService(tmp_path)
    assert service.write_authority_status()["enabled"] is False

    # A tool/model changing process state after service creation cannot grant
    # authority to the already-running service instance.
    monkeypatch.setenv(REPOSITORY_WRITES_ENV, "1")
    with pytest.raises(RepositoryWriteAuthorizationError, match="operator must set"):
        service.rename_apply(
            {},
            expected_plan_sha256="irrelevant",
            acknowledge_incomplete=True,
        )


def test_acknowledgement_boolean_never_grants_write_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REPOSITORY_WRITES_ENV, "0")
    service = service_module.RepositoryIntelligenceService(tmp_path)
    for method_name in ("rename_apply", "safe_delete_apply", "file_move_apply"):
        method = getattr(service, method_name)
        with pytest.raises(RepositoryWriteAuthorizationError):
            method(
                {},
                expected_plan_sha256="irrelevant",
                acknowledge_incomplete=True,
            )


def test_pre_authorized_service_does_not_depend_on_later_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REPOSITORY_WRITES_ENV, "1")
    service = service_module.RepositoryIntelligenceService(tmp_path)
    assert service.write_authority_status()["enabled"] is True
    monkeypatch.setenv(REPOSITORY_WRITES_ENV, "0")

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_apply(self, *args, **kwargs):
        calls.append((args, kwargs))
        return {"applied": True}

    monkeypatch.setattr(
        service_module._BaseRepositoryIntelligenceService,
        "rename_apply",
        fake_apply,
    )
    result = service.rename_apply(
        {"plan": "value"},
        expected_plan_sha256="sha",
        acknowledge_incomplete=True,
    )
    assert result == {"applied": True}
    assert len(calls) == 1


def test_authority_status_contains_no_secret_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REPOSITORY_WRITES_ENV, "1")
    service = service_module.RepositoryIntelligenceService(tmp_path)
    assert service.write_authority_status() == {
        "schema_version": "entroly.repository-write-authority.v1",
        "enabled": True,
        "source": "process-start-environment",
        "mutable_by_tool_arguments": False,
    }
