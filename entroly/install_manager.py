"""Reversible, user-scoped persistent Entroly service installation.

The installer deliberately avoids system-wide privileges.  It writes one
owned service definition and one manifest below the user's Entroly state, then
delegates lifecycle operations to the operating system's user-service manager.
Every external command is an argument vector (never a shell string), and
removal refuses to delete a service definition whose digest no longer matches
the manifest Entroly wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.sax.saxutils import escape as xml_escape


SCHEMA_VERSION = "entroly.persistent-install.v1"
SERVICE_ID = "entroly"
LAUNCHD_LABEL = "io.entroly.daemon"


class InstallError(RuntimeError):
    """Persistent installation could not be completed safely."""


@dataclass(frozen=True)
class InstallSpec:
    proxy_port: int = 9377
    dashboard_port: int = 9378
    mcp_port: int = 9379
    host: str = "127.0.0.1"
    no_proxy: bool = False
    no_mcp: bool = False
    quality: str = "balanced"

    def daemon_args(self) -> list[str]:
        args = [
            "daemon",
            "--proxy-port",
            str(self.proxy_port),
            "--dashboard-port",
            str(self.dashboard_port),
            "--mcp-port",
            str(self.mcp_port),
            "--host",
            self.host,
            "--quality",
            self.quality,
        ]
        if self.no_proxy:
            args.append("--no-proxy")
        if self.no_mcp:
            args.append("--no-mcp")
        return args


@dataclass(frozen=True)
class InstallResult:
    action: str
    platform: str
    installed: bool
    running: bool | None
    artifact_path: str
    commands: tuple[tuple[str, ...], ...]
    dry_run: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commands"] = [list(command) for command in self.commands]
        return payload


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _default_runner(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=30,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _systemd_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class PersistentInstallManager:
    """Create and operate one Entroly user service on Windows/macOS/Linux."""

    def __init__(
        self,
        *,
        state_dir: Path,
        home: Path | None = None,
        platform_name: str | None = None,
        python_executable: str | None = None,
        user_id: int | None = None,
        runner: Runner = _default_runner,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.home = Path(home or Path.home()).expanduser().resolve()
        self.platform = self._normalize_platform(platform_name or sys.platform)
        self.python_executable = str(Path(python_executable or sys.executable).resolve())
        self.user_id = (
            int(user_id)
            if user_id is not None
            else int(getattr(os, "getuid", lambda: 0)())
        )
        self.runner = runner
        self.manifest_path = self.state_dir / "manifest.json"

    @staticmethod
    def _normalize_platform(value: str) -> str:
        lowered = value.casefold()
        if lowered.startswith("win"):
            return "windows"
        if lowered == "darwin":
            return "macos"
        if lowered.startswith("linux"):
            return "linux"
        raise InstallError(f"persistent install is unsupported on platform {value!r}")

    @property
    def artifact_path(self) -> Path:
        if self.platform == "windows":
            return self.state_dir / "entroly-task.xml"
        if self.platform == "macos":
            return self.home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
        return self.home / ".config" / "systemd" / "user" / "entroly.service"

    def command(self, spec: InstallSpec) -> list[str]:
        return [self.python_executable, "-m", "entroly.cli", *spec.daemon_args()]

    def plan(self, spec: InstallSpec) -> tuple[bytes, tuple[tuple[str, ...], ...]]:
        command = self.command(spec)
        if self.platform == "linux":
            exec_start = " ".join(_systemd_quote(part) for part in command)
            artifact = (
                "# Managed by Entroly; remove with `entroly install remove`.\n"
                "[Unit]\nDescription=Entroly context assurance daemon\n"
                "After=network.target\n\n[Service]\nType=simple\n"
                f"ExecStart={exec_start}\n"
                f"WorkingDirectory={_systemd_quote(str(self.state_dir))}\n"
                "Environment=PYTHONSAFEPATH=1\nRestart=on-failure\nRestartSec=5\n\n"
                "[Install]\nWantedBy=default.target\n"
            ).encode("utf-8")
            commands = (
                ("systemctl", "--user", "daemon-reload"),
                ("systemctl", "--user", "enable", "--now", "entroly.service"),
            )
        elif self.platform == "macos":
            artifact = plistlib.dumps(
                {
                    "Label": LAUNCHD_LABEL,
                    "ProgramArguments": command,
                    "WorkingDirectory": str(self.state_dir),
                    "EnvironmentVariables": {"PYTHONSAFEPATH": "1"},
                    "RunAtLoad": True,
                    "KeepAlive": {"SuccessfulExit": False},
                    "ProcessType": "Background",
                    "StandardOutPath": str(self.state_dir / "daemon.stdout.log"),
                    "StandardErrorPath": str(self.state_dir / "daemon.stderr.log"),
                    "EntrolyManaged": True,
                },
                sort_keys=True,
            )
            domain = f"gui/{self.user_id}"
            commands = (
                ("launchctl", "bootstrap", domain, str(self.artifact_path)),
                ("launchctl", "enable", f"{domain}/{LAUNCHD_LABEL}"),
            )
        else:
            arguments = subprocess.list2cmdline(command[1:])
            artifact = self._windows_task_xml(command[0], arguments).encode("utf-16")
            commands = (
                (
                    "schtasks.exe",
                    "/Create",
                    "/TN",
                    "Entroly",
                    "/XML",
                    str(self.artifact_path),
                    "/F",
                ),
                ("schtasks.exe", "/Run", "/TN", "Entroly"),
            )
        return artifact, commands

    def _windows_task_xml(self, executable: str, arguments: str) -> str:
        return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Entroly context assurance daemon</Description><URI>\\Entroly</URI></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>true</StartWhenAvailable><Enabled>true</Enabled><Hidden>false</Hidden><ExecutionTimeLimit>PT0S</ExecutionTimeLimit></Settings>
  <Actions Context="Author"><Exec><Command>{xml_escape(executable)}</Command><Arguments>{xml_escape(arguments)}</Arguments><WorkingDirectory>{xml_escape(str(self.state_dir))}</WorkingDirectory></Exec></Actions>
</Task>
'''

    def apply(self, spec: InstallSpec, *, dry_run: bool = False) -> InstallResult:
        artifact, commands = self.plan(spec)
        if dry_run:
            return InstallResult(
                "apply", self.platform, False, None, str(self.artifact_path), commands,
                dry_run=True, detail=artifact.decode("utf-16" if self.platform == "windows" else "utf-8"),
            )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.artifact_path, artifact)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "platform": self.platform,
            "service_id": SERVICE_ID,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": _sha256(artifact),
            "command": self.command(spec),
            "spec": asdict(spec),
        }
        _atomic_write(
            self.manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        try:
            for command in commands:
                self.runner(command, check=True, capture_output=True)
        except Exception as exc:
            raise InstallError(
                f"service definition was written to {self.artifact_path}, but activation failed: {exc}"
            ) from exc
        return InstallResult(
            "apply", self.platform, True, True, str(self.artifact_path), commands,
            detail="user-scoped service installed and started",
        )

    def _lifecycle_commands(self, action: str) -> tuple[tuple[str, ...], ...]:
        if self.platform == "linux":
            verb = {"start": "start", "stop": "stop", "restart": "restart"}[action]
            return (("systemctl", "--user", verb, "entroly.service"),)
        if self.platform == "macos":
            domain = f"gui/{self.user_id}"
            target = f"{domain}/{LAUNCHD_LABEL}"
            if action == "stop":
                return (
                    ("launchctl", "disable", target),
                    ("launchctl", "kill", "SIGTERM", target),
                )
            if action == "start":
                return (
                    ("launchctl", "enable", target),
                    ("launchctl", "kickstart", "-p", target),
                )
            return (
                ("launchctl", "enable", target),
                ("launchctl", "kickstart", "-k", target),
            )
        verb = {"start": "/Run", "stop": "/End", "restart": "/Run"}[action]
        commands: list[tuple[str, ...]] = []
        if action == "restart":
            commands.append(("schtasks.exe", "/End", "/TN", "Entroly"))
        commands.append(("schtasks.exe", verb, "/TN", "Entroly"))
        return tuple(commands)

    def lifecycle(self, action: str, *, dry_run: bool = False) -> InstallResult:
        if action not in {"start", "stop", "restart"}:
            raise ValueError(f"unsupported lifecycle action: {action}")
        if not self.manifest_path.exists():
            raise InstallError("Entroly is not persistently installed for this user")
        commands = self._lifecycle_commands(action)
        if not dry_run:
            for index, command in enumerate(commands):
                tolerate_stopped_windows_task = (
                    self.platform == "windows" and action == "restart" and index == 0
                )
                self.runner(
                    command,
                    check=not tolerate_stopped_windows_task,
                    capture_output=True,
                )
        return InstallResult(
            action,
            self.platform,
            True,
            action != "stop" if not dry_run else None,
            str(self.artifact_path),
            commands,
            dry_run=dry_run,
        )

    def status(self) -> InstallResult:
        installed = self.manifest_path.exists() and self.artifact_path.exists()
        commands: tuple[tuple[str, ...], ...]
        if self.platform == "linux":
            commands = (("systemctl", "--user", "is-active", "entroly.service"),)
        elif self.platform == "macos":
            commands = (("launchctl", "print", f"gui/{self.user_id}/{LAUNCHD_LABEL}"),)
        else:
            commands = (("schtasks.exe", "/Query", "/TN", "Entroly"),)
        running: bool | None = False
        detail = "not installed"
        if installed:
            result = self.runner(commands[0], check=False, capture_output=True)
            running = result.returncode == 0
            detail = "running" if running else "installed but not running"
        return InstallResult(
            "status", self.platform, installed, running, str(self.artifact_path), commands,
            detail=detail,
        )

    def remove(self, *, dry_run: bool = False) -> InstallResult:
        manifest = self._read_manifest()
        commands = self._remove_commands()
        self._verify_owned_artifact(manifest)
        if not dry_run:
            for command in commands:
                self.runner(command, check=False, capture_output=True)
            self.artifact_path.unlink(missing_ok=True)
            self.manifest_path.unlink(missing_ok=True)
            if self.platform == "linux":
                self.runner(
                    ("systemctl", "--user", "daemon-reload"),
                    check=False,
                    capture_output=True,
                )
        return InstallResult(
            "remove", self.platform, False if not dry_run else True, False,
            str(self.artifact_path), commands, dry_run=dry_run,
            detail="owned user-service definition removed" if not dry_run else "removal preview",
        )

    def _read_manifest(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise InstallError("Entroly is not persistently installed for this user") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise InstallError(f"cannot read persistent install manifest: {exc}") from exc
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise InstallError("persistent install manifest has an unsupported schema")
        if Path(payload.get("artifact_path", "")).resolve() != self.artifact_path.resolve():
            raise InstallError("persistent install manifest points outside the expected service path")
        return payload

    def _verify_owned_artifact(self, manifest: dict[str, Any]) -> None:
        try:
            current = self.artifact_path.read_bytes()
        except FileNotFoundError:
            return
        if _sha256(current) != manifest.get("artifact_sha256"):
            raise InstallError(
                f"refusing to remove modified service definition at {self.artifact_path}"
            )

    def _remove_commands(self) -> tuple[tuple[str, ...], ...]:
        if self.platform == "linux":
            return (("systemctl", "--user", "disable", "--now", "entroly.service"),)
        if self.platform == "macos":
            return (("launchctl", "bootout", f"gui/{self.user_id}", str(self.artifact_path)),)
        return (("schtasks.exe", "/Delete", "/TN", "Entroly", "/F"),)


__all__ = [
    "InstallError",
    "InstallResult",
    "InstallSpec",
    "PersistentInstallManager",
]
