from __future__ import annotations

import json
import subprocess

import pytest

from entroly.install_manager import (
    InstallError,
    InstallSpec,
    PersistentInstallManager,
)


class RecordingRunner:
    def __init__(self, returncode: int = 0):
        self.calls: list[tuple[str, ...]] = []
        self.returncode = returncode

    def __call__(self, args, *, check=True, capture_output=True):
        command = tuple(str(value) for value in args)
        self.calls.append(command)
        return subprocess.CompletedProcess(command, self.returncode, "", "")


def test_linux_apply_is_user_scoped_and_reversible(tmp_path):
    runner = RecordingRunner()
    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="linux",
        python_executable=tmp_path / "venv" / "bin" / "python",
        runner=runner,
    )

    result = manager.apply(InstallSpec(proxy_port=9444))

    unit = manager.artifact_path.read_text(encoding="utf-8")
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    assert result.installed is True
    assert "--proxy-port\" \"9444" in unit
    assert "PYTHONSAFEPATH=1" in unit
    assert manifest["artifact_path"] == str(manager.artifact_path)
    assert runner.calls == [
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "enable", "--now", "entroly.service"),
    ]

    manager.remove()
    assert not manager.artifact_path.exists()
    assert not manager.manifest_path.exists()
    assert ("systemctl", "--user", "disable", "--now", "entroly.service") in runner.calls


def test_dry_run_does_not_write_or_execute(tmp_path):
    runner = RecordingRunner()
    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="linux",
        runner=runner,
    )

    result = manager.apply(InstallSpec(), dry_run=True)

    assert result.dry_run is True
    assert "[Service]" in result.detail
    assert not manager.artifact_path.exists()
    assert not manager.manifest_path.exists()
    assert runner.calls == []


def test_remove_refuses_modified_definition(tmp_path):
    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="linux",
        runner=RecordingRunner(),
    )
    manager.apply(InstallSpec())
    manager.artifact_path.write_text("user changed this", encoding="utf-8")

    with pytest.raises(InstallError, match="refusing to remove modified"):
        manager.remove()


def test_windows_plan_uses_least_privilege_task_and_argument_vector(tmp_path):
    runner = RecordingRunner()
    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="win32",
        python_executable=tmp_path / "Python" / "python.exe",
        runner=runner,
    )

    result = manager.apply(InstallSpec(no_mcp=True), dry_run=True)

    assert "<RunLevel>LeastPrivilege</RunLevel>" in result.detail
    assert "--no-mcp" in result.detail
    assert result.commands[0][0] == "schtasks.exe"
    assert runner.calls == []


def test_macos_stop_disables_keepalive_before_terminating(tmp_path):
    runner = RecordingRunner()
    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="darwin",
        user_id=501,
        runner=runner,
    )
    manager.apply(InstallSpec())
    runner.calls.clear()

    manager.lifecycle("stop")

    assert runner.calls == [
        ("launchctl", "disable", "gui/501/io.entroly.daemon"),
        ("launchctl", "kill", "SIGTERM", "gui/501/io.entroly.daemon"),
    ]


def test_windows_restart_tolerates_already_stopped_task(tmp_path):
    checks = []

    def runner(args, *, check=True, capture_output=True):
        checks.append((tuple(args), check))
        return subprocess.CompletedProcess(args, 0, "", "")

    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="win32",
        runner=runner,
    )
    manager.apply(InstallSpec())
    checks.clear()

    manager.lifecycle("restart")

    assert checks[0] == (("schtasks.exe", "/End", "/TN", "Entroly"), False)
    assert checks[1] == (("schtasks.exe", "/Run", "/TN", "Entroly"), True)


def test_status_distinguishes_installed_from_running(tmp_path):
    runner = RecordingRunner(returncode=3)
    manager = PersistentInstallManager(
        state_dir=tmp_path / "state",
        home=tmp_path / "home",
        platform_name="linux",
        runner=runner,
    )
    manager.apply(InstallSpec())
    runner.returncode = 3

    status = manager.status()

    assert status.installed is True
    assert status.running is False
    assert status.detail == "installed but not running"
