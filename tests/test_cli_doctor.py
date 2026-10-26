from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from entroly import cli
from entroly.native_status import NativeStatus


@pytest.mark.parametrize("uploads", [False, True])
def test_privacy_checks_disclose_telemetry_and_do_not_certify_no_egress(
    uploads, tuning_config, capsys, monkeypatch,
):
    from entroly import product_telemetry

    monkeypatch.setattr(product_telemetry, "status", lambda: {"upload_configured": uploads})
    cli.cmd_doctor(Namespace(port=9377, privacy=True))
    output = capsys.readouterr().out
    assert f"Product telemetry uploads configured: {uploads}" in output
    assert "not a network audit or privacy certification" in output
    assert "PRIVACY VERIFIED" not in output
    assert "No Entroly-owned phone-home path detected" not in output
    assert "other configured stores" in output or "configured stores may still contain data" in output


@pytest.fixture(autouse=True)
def _empty_entroly_home(monkeypatch, tmp_path_factory):
    """Doctor reads ~/.entroly (index freshness); keep the developer's out.

    These tests assert exact check counts. Once doctor could see per-project
    indexes, a >24h-old index in the real HOME turned "8/8" into "7/8".
    """
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)


@pytest.fixture
def tuning_config(monkeypatch, tmp_path: Path):
    """Point doctor's weights file at a temp path and write arbitrary content."""

    path = tmp_path / "tuning_config.json"

    def _write(content: str) -> Path:
        path.write_text(content, encoding="utf-8")
        return path

    # cmd_doctor resolves the file as Path(cli.__file__).parent / "tuning_config.json"
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))
    return _write


def test_doctor_warns_that_relevance_ranking_is_off_without_native_engine(
    monkeypatch,
    capsys,
) -> None:
    """README sends users to doctor when simulate says "Relevance ranking is
    OFF". Without the native engine doctor called it "Optional Rust
    acceleration" and counted a pass ("8/8 checks passed"), so the one
    diagnostic for that state reported all-clear. It is a counted warning,
    matching `doctor --json` (native_engine: warning, pure_python_fallback):
    the pure-Python fallback still works, so the exit code stays 0.
    """
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

    rc = cli.cmd_doctor(Namespace(port=9377, privacy=False))

    output = capsys.readouterr().out
    assert "Relevance ranking is OFF" in output
    assert "python -m pip install -U entroly-core" in output
    assert "Optional Rust acceleration" not in output
    assert "8/8 checks passed" not in output
    assert "7/8 checks passed" in output
    assert "warning(s)" in output
    assert rc in (None, 0)


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


def test_heavy_weight_drift_warns_without_failing_the_command(tuning_config, capsys) -> None:
    # Heavy drift used to be counted as a PASS ("8/8 checks passed"), hiding it.
    # It must be surfaced — but it is produced by the supported `entroly
    # autotune`, and the message only suggests a rollback, so it must not exit
    # non-zero either or every autotuned machine fails the CLI smoke scripts.
    tuning_config(
        json.dumps(
            {"weights": {"recency": 0.95, "frequency": 0.02, "semantic": 0.02, "entropy": 0.01}}
        )
    )

    rc = cli.cmd_doctor(Namespace(port=9377, privacy=False))
    output = capsys.readouterr().out

    assert "Weights heavily drifted" in output
    assert "8/8 checks passed" not in output, "drift must not be reported as a pass"
    assert "warning" in output.lower()
    assert rc == 0, "an advisory must not break `entroly doctor` as a CI gate"


def test_doctor_reports_an_unreadable_weights_file_instead_of_passing_silently(
    tuning_config, capsys
) -> None:
    # An unparseable config previously hit a bare `except: checks_passed += 1`
    # that printed nothing at all — the check vanished from the output while
    # still counting as a pass.
    tuning_config("{invalid json")

    rc = cli.cmd_doctor(Namespace(port=9377, privacy=False))
    output = capsys.readouterr().out

    assert "Weight drift unreadable" in output
    assert "Config error" in output
    assert "2 failed" in output
    assert rc == 1


def test_doctor_passes_cleanly_when_nothing_is_wrong(
    tuning_config, capsys, monkeypatch
) -> None:
    # Pin the native status like the sibling tests do. Otherwise a stale-but-
    # loadable core — which the code itself calls "the normal state in a source
    # checkout between a Rust edit and maturin develop" — makes doctor report
    # "7/8 checks passed, 1 warning(s)" and fails this test for a reason
    # unrelated to what it checks.
    monkeypatch.setattr(
        "entroly.native_status.native_status",
        lambda _: NativeStatus(
            available=True, module=None, version="1.0.66", path=None,
            missing_symbols=(), version_ok=True,
        ),
    )
    # Defaults (no tuning_config.json present) must still be a clean 8/8 exit 0;
    # informational lines (proxy/docker/index absent) are not failures.
    rc = cli.cmd_doctor(Namespace(port=9377, privacy=False))
    output = capsys.readouterr().out

    assert "8/8 checks passed" in output
    assert "failed" not in output
    assert rc == 0


def test_a_shadowed_installation_is_counted_as_a_warning(monkeypatch, capsys) -> None:
    """The summary must not contradict the line above it.

    `entroly` on PATH resolving to a different installation than the imported
    package means every command runs code the user is not editing. Doctor
    printed that warning and incremented nothing, so the run still ended
    "8/8 checks passed" with no warning count -- the same false-green the split
    counters exist to prevent, on the one warning least able to afford being
    missed.
    """
    monkeypatch.setattr(
        cli.shutil, "which", lambda _name: "/somewhere/else/bin/entroly"
    )

    cli.cmd_doctor(Namespace(port=9377, privacy=False))

    output = capsys.readouterr().out
    assert "different installation than the imported package" in output
    assert "warning(s)" in output, (
        "a printed warning that the summary does not count reads as all-clear"
    )
