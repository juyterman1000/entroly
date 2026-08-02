from __future__ import annotations

import json
from argparse import Namespace

from entroly import cli
from entroly.native_status import NativeStatus
from entroly.runtime_capabilities import (
    SCHEMA_VERSION,
    build_runtime_capabilities,
    capability_exit_code,
)


def test_runtime_capabilities_are_stable_and_conservative(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.runtime_capabilities.native_status",
        lambda _: NativeStatus(
            available=False,
            module=None,
            version=None,
            path=None,
            missing_symbols=("py_qccr_select",),
            version_ok=None,
            error="not installed",
        ),
    )
    monkeypatch.setattr(
        "entroly.runtime_capabilities._module_available",
        lambda name: name in {"httpx", "starlette", "uvicorn"},
    )

    report = build_runtime_capabilities()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["engine"]["active_mode"] == "python"
    assert report["engine"]["pure_python_fallback"] is True
    assert report["engine"]["native"]["module_file"] is None
    assert report["engine"]["native"]["unavailable_reason"] == "import_failed"
    assert "path" not in report["engine"]["native"]
    assert "error" not in report["engine"]["native"]
    assert report["surfaces"]["proxy"]["available"] is True
    assert report["surfaces"]["mcp"]["available"] is False
    assert report["surfaces"]["secure_recovery"]["default"] is True
    assert [item["name"] for item in report["providers"]] == [
        "anthropic",
        "gemini",
        "openai",
    ]
    assert all(item["connectivity_verified"] is False for item in report["providers"])
    assert report["claims"]["benchmark_leadership_implied"] is False
    json.dumps(report, sort_keys=True)


def test_capability_exit_code_requires_secure_base_runtime() -> None:
    report = {
        "surfaces": {
            "python_sdk": {"available": True},
            "secure_recovery": {"available": True},
        }
    }
    assert capability_exit_code(report) == 0
    report["surfaces"]["secure_recovery"]["available"] = False
    assert capability_exit_code(report) == 1


def test_cli_capabilities_json_is_plain_machine_readable(monkeypatch, capsys) -> None:
    report = {
        "schema_version": SCHEMA_VERSION,
        "package": {"version": "1.0.70"},
        "engine": {"active_mode": "python"},
        "surfaces": {
            "python_sdk": {"available": True},
            "secure_recovery": {"available": True},
        },
        "providers": [],
    }
    monkeypatch.setattr(
        "entroly.runtime_capabilities.build_runtime_capabilities", lambda: report
    )

    rc = cli.cmd_capabilities(Namespace(json_output=True))
    output = capsys.readouterr().out

    assert rc == 0
    assert "\x1b[" not in output
    assert json.loads(output) == report
