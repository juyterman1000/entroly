from __future__ import annotations

from argparse import Namespace

from entroly import cli
from entroly.native_status import NativeStatus


def test_doctor_treats_absent_optional_native_engine_as_healthy(
    monkeypatch,
    capsys,
) -> None:
    status = NativeStatus(
        available=False,
        module=None,
        version=None,
        path=None,
        missing_symbols=("py_qccr_select",),
        version_ok=None,
        error="not installed",
    )
    monkeypatch.setattr("entroly.native_status.native_status", lambda _: status)

    cli.cmd_doctor(Namespace(port=9377, privacy=False))

    output = capsys.readouterr().out
    assert "Optional Rust acceleration not installed" in output
    assert "pure-Python engine is active" in output
    assert "8/8 checks passed" in output


def test_doctor_flags_an_installed_but_incomplete_native_engine(
    monkeypatch,
    capsys,
) -> None:
    status = NativeStatus(
        available=True,
        module=None,
        version="1.0.1",
        path="/tmp/entroly_core.so",
        missing_symbols=("py_qccr_select",),
        version_ok=False,
    )
    monkeypatch.setattr("entroly.native_status.native_status", lambda _: status)

    cli.cmd_doctor(Namespace(port=9377, privacy=False))

    output = capsys.readouterr().out
    assert "Installed Rust engine is stale or incomplete" in output
    assert "Loaded version: 1.0.1" in output
    assert "7/8 checks passed" in output
