from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import entroly.cli as cli
import entroly.runtime_doctor as doctor_module
from entroly.runtime_doctor import SCHEMA_VERSION, runtime_doctor


def _capabilities(native: bool = False) -> dict:
    return {
        "schema_version": "entroly.runtime-capabilities.v1",
        "engine": {"native": {"available": native}},
    }


def _status(healthy: bool = False) -> dict:
    return {
        "schema_version": "entroly.runtime-status.v1",
        "healthy": healthy,
    }


def test_runtime_doctor_reports_warnings_without_claiming_failure(tmp_path: Path) -> None:
    report = runtime_doctor(
        data_dir=tmp_path / "state",
        capability_factory=lambda: _capabilities(False),
        status_factory=lambda *, port: _status(False),
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["healthy"] is True
    assert report["summary"]["errors"] == 0
    assert report["summary"]["warnings"] == 2
    assert report["claims"]["provider_connectivity_verified"] is False
    assert report["claims"]["production_readiness_implied"] is False
    rendered = json.dumps(report)
    assert str(tmp_path) not in rendered


def test_runtime_doctor_fails_closed_on_invalid_configuration(tmp_path: Path) -> None:
    data_dir = tmp_path / "state"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{not-json", encoding="utf-8")

    report = runtime_doctor(
        data_dir=data_dir,
        capability_factory=lambda: _capabilities(True),
        status_factory=lambda *, port: _status(True),
    )

    assert report["healthy"] is False
    assert report["summary"]["errors"] == 1
    assert {
        "name": "runtime_config",
        "status": "error",
        "detail": "invalid_json",
    } in report["checks"]


def test_cmd_doctor_json_uses_machine_readable_exit_contract(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_ENTROLY_DIR", tmp_path)
    monkeypatch.setattr(
        doctor_module,
        "runtime_doctor",
        lambda **_kwargs: {
            "schema_version": SCHEMA_VERSION,
            "healthy": False,
            "summary": {"passed": 1, "warnings": 0, "errors": 1},
            "checks": [],
            "capabilities_schema": "entroly.runtime-capabilities.v1",
            "status_schema": "entroly.runtime-status.v1",
            "claims": {
                "provider_connectivity_verified": False,
                "production_readiness_implied": False,
            },
        },
    )

    result = cli.cmd_doctor(
        SimpleNamespace(port=9377, privacy=False, json_output=True)
    )

    assert result == 1
    assert json.loads(capsys.readouterr().out)["healthy"] is False
