"""Startup must never install packages without operator consent."""
from argparse import Namespace
from unittest.mock import Mock

import pytest

from entroly import self_heal


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    for name in (self_heal.ENV_ENABLE, self_heal.ENV_DISABLE,
                 self_heal.ENV_AIR_GAP, self_heal.ENV_GUARD):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(self_heal, "native_engine_ready", lambda: False)
    monkeypatch.setattr(self_heal, "_attempted_in_this_process", False)


@pytest.mark.parametrize("entrypoint", ["server", "proxy", "measurement", "repair", "install"])
def test_default_startup_does_not_invoke_installer(entrypoint, monkeypatch, capsys):
    installer = Mock(side_effect=AssertionError("unconsented installer"))
    monkeypatch.setattr(self_heal, "_installer_command", installer)
    if entrypoint == "server":
        from entroly.server import _repair_native_engine_at_startup
        _repair_native_engine_at_startup()
    elif entrypoint == "proxy":
        from entroly.cli import _auto_repair_for_service
        _auto_repair_for_service("proxy")
    elif entrypoint == "measurement":
        from entroly.cli import _auto_repair_before_measuring
        _auto_repair_before_measuring(Namespace(json=True))
    elif entrypoint == "repair":
        outcome = self_heal.repair_native(force=True)
        assert outcome.blocked and not outcome.attempted
    else:
        ok, reason = self_heal.install_native_engine()
        assert not ok and "consent" in reason
    installer.assert_not_called()
    output = capsys.readouterr()
    assert output.out == ""
    if entrypoint in ("server", "proxy"):
        assert "WARNING" in output.err
        assert self_heal.ENV_ENABLE in output.err


@pytest.mark.parametrize("flag", [self_heal.ENV_DISABLE, self_heal.ENV_AIR_GAP])
def test_hard_off_overrides_opt_in_and_explicit_sdk_repair(flag, monkeypatch):
    monkeypatch.setenv(flag, "1")
    monkeypatch.setenv(self_heal.ENV_ENABLE, "1")
    installer = Mock()
    monkeypatch.setattr(self_heal, "_installer_command", installer)
    assert not self_heal.automatic_allowed()
    assert flag in self_heal.repair().blocked_reason
    ok, reason = self_heal.install_native_engine(authorized=True)
    assert not ok and flag in reason
    installer.assert_not_called()


@pytest.mark.parametrize("consent", ["environment", "sdk"])
def test_consent_allows_one_bounded_install(consent, monkeypatch):
    command = ["fake-installer", "install", "entroly-core"]
    monkeypatch.setattr(self_heal, "_installer_command", lambda: command)
    runner = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(self_heal.subprocess, "run", runner)
    if consent == "environment":
        monkeypatch.setenv(self_heal.ENV_ENABLE, "1")
        result = self_heal.repair_native()
    else:
        import entroly
        result = entroly.repair()
    assert result.attempted and result.needs_reexec
    runner.assert_called_once_with(command, capture_output=True, text=True,
                                  timeout=180, check=False)
    assert not self_heal.repair_native().attempted


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_opt_in_requires_documented_value(value, monkeypatch):
    monkeypatch.setenv(self_heal.ENV_ENABLE, value)
    assert not self_heal.automatic_allowed()


def test_pep668_marker_is_inside_stdlib_and_venvs_remain_supported(tmp_path, monkeypatch):
    stdlib = tmp_path / "lib" / "python3.12"
    stdlib.mkdir(parents=True)
    (stdlib / "EXTERNALLY-MANAGED").touch()
    monkeypatch.setattr(self_heal.sysconfig, "get_path", lambda _: str(stdlib))
    monkeypatch.setattr(self_heal.sys, "prefix", "system")
    monkeypatch.setattr(self_heal.sys, "base_prefix", "system")
    assert self_heal._externally_managed()
    assert self_heal._installer_command() is None
    monkeypatch.setattr(self_heal.sys, "prefix", "venv")
    assert not self_heal._externally_managed()
