from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from entroly.session_attach import AttachmentStore, install_attachment, uninstall_attachment

REPORT_PATH = Path("kimi-mcp-e2e.json")
EXPECTED_TOOLS = (
    "get_stats",
    "optimize_context",
    "entroly_retrieve",
    "repo_file_map",
)


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: int = 120,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    report: dict[str, Any] = {
        "schema": "entroly.kimi-mcp-real-client-e2e.v1",
        "passed": False,
    }

    with tempfile.TemporaryDirectory(prefix="entroly-kimi-mcp-e2e-") as raw:
        temp = Path(raw)
        project = temp / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "entroly-kimi-e2e"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        (project / "sample.py").write_text(
            "def answer(): return 'E2E_OK'\n",
            encoding="utf-8",
        )

        home = temp / "home"
        home.mkdir()
        state = temp / "state"
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "ENTROLY_DISABLE_UPDATE_CHECK": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )

        version = _run(["kimi", "--version"], env=env, cwd=project, timeout=30)
        report["client_version"] = version

        store = AttachmentStore(state)
        issued = store.create(
            client="kimi",
            project_root=project,
            scopes=("observe", "context"),
            ttl_seconds=600,
        )
        name = f"entroly-{issued.grant.grant_id}"
        report["grant_id"] = issued.grant.grant_id
        report["server_name"] = name
        report["install_commands"] = [list(command) for command in issued.install_commands]

        old_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(env)
        try:
            installed = install_attachment(issued, store=store)
            report["install"] = [
                {
                    "returncode": item.returncode,
                    "stdout": item.stdout,
                    "stderr": item.stderr,
                }
                for item in installed
            ]

            inspection = _run(
                ["kimi", "mcp", "test", name],
                env=env,
                cwd=project,
            )
            report["inspection"] = inspection
            combined = f"{inspection['stdout']}\n{inspection['stderr']}"
            report["tools_observed"] = [
                tool for tool in EXPECTED_TOOLS if tool in combined
            ]
            report["grant_use_count"] = store.get(issued.grant.grant_id).use_count
            report["passed"] = bool(
                inspection["returncode"] == 0
                and set(report["tools_observed"]) == set(EXPECTED_TOOLS)
                and report["grant_use_count"] > 0
            )
        except Exception as exc:  # noqa: BLE001 - report exact E2E boundary
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                removed = uninstall_attachment(issued.grant)
                report["remove"] = [item.returncode for item in removed]
            except Exception as exc:  # noqa: BLE001
                report["remove_error"] = f"{type(exc).__name__}: {exc}"
            try:
                store.revoke(issued.grant.grant_id)
            except Exception as exc:  # noqa: BLE001
                report["revoke_error"] = f"{type(exc).__name__}: {exc}"
            os.environ.clear()
            os.environ.update(old_env)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report.get("passed"),
                "tools_observed": report.get("tools_observed"),
                "grant_use_count": report.get("grant_use_count"),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
