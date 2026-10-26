"""Installing the capture hook must not duplicate one a project provides.

The host fans a single tool call out to *every* matching ``PostToolUse``
registration. ``install_claude_code`` checked only the scope it was about to
write, so a repository that ships its own capture hook got a second, broader
one at user scope -- and both matched Bash.

Measured consequence on a real session: 12 byte-identical duplicate lines out
of 84 in the session journal, with one call recorded four times, and Bash
double-counted in the tool-use counters that exist to justify a schema
allowlist. The journal now collapses redelivery at read time, but a counter
that has already been incremented twice cannot be un-incremented -- so the
registration itself has to stay single.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import entroly.cli as cli_module
import entroly.ravs.hooks as hooks_module
from entroly.ravs.hooks import install_claude_code


@pytest.fixture()
def scopes(tmp_path, monkeypatch):
    """Isolate both scopes: a fake home and a fake project cwd."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".claude").mkdir(parents=True)
    (project / ".claude").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.chdir(project)
    return home, project


def _write_project_hook(project, filename="settings.json", matcher="Bash|Read|Edit"):
    (project / ".claude" / filename).write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": matcher,
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "entroly ravs capture --stdin --quiet",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def test_a_project_hook_suppresses_a_second_user_scope_hook(scopes) -> None:
    """The defect: two registrations, both matching Bash, one call recorded twice."""
    home, project = scopes
    _write_project_hook(project)

    result = install_claude_code()

    assert result["action"] == "already_present_in_project"
    assert "settings.json" in result["project_path"]
    assert not (home / ".claude" / "settings.json").exists(), (
        "nothing may be written when the project already covers these calls"
    )


def test_settings_local_counts_as_a_project_registration(scopes) -> None:
    home, project = scopes
    _write_project_hook(project, filename="settings.local.json")

    assert install_claude_code()["action"] == "already_present_in_project"
    assert not (home / ".claude" / "settings.json").exists()


def test_dry_run_also_reports_the_project_hook(scopes) -> None:
    """A dry run that claims it would install is a wrong answer too."""
    _, project = scopes
    _write_project_hook(project)

    assert install_claude_code(dry_run=True)["action"] == (
        "already_present_in_project"
    )


def test_installation_still_happens_with_no_project_hook(scopes) -> None:
    """The guard must not block the case it was built for."""
    home, project = scopes

    result = install_claude_code()

    assert result["action"] == "installed"
    written = json.loads(
        (home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    entries = written["hooks"]["PostToolUse"]
    assert sum("ravs capture" in str(entry) for entry in entries) == 1


def test_an_unrelated_project_hook_does_not_suppress_installation(scopes) -> None:
    """Only Entroly's own capture hook counts as coverage."""
    home, project = scopes
    (project / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "some-other-linter"}
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert install_claude_code()["action"] == "installed"


def test_unreadable_project_settings_do_not_break_installation(scopes) -> None:
    """A malformed config must not become a failed install."""
    home, project = scopes
    (project / ".claude" / "settings.json").write_text("{not json", encoding="utf-8")

    assert install_claude_code()["action"] == "installed"


def test_every_action_the_installer_returns_has_a_cli_branch() -> None:
    """A new action must not reach the user as "Unknown action".

    Adding ``already_present_in_project`` without updating its only consumer
    broke three CI jobs: the CLI fell through to its error branch, so
    ``entroly ravs hook install --claude-code --dry-run`` reported an internal
    action name instead of saying what happened. The installer's contract and
    the CLI's handling of it are in different files, so this pins them
    together.
    """
    import inspect
    import re

    # Only this function's own actions. uninstall_claude_code returns its own
    # set ("uninstalled", "not_found", "parse_error") to a different command.
    installer_source = inspect.getsource(hooks_module.install_claude_code)
    returned = set(re.findall(r'"action":\s*"([a-z_]+)"', installer_source))
    assert returned >= {"installed", "already_present"}, (
        "the installer's known actions must be discoverable from its source"
    )

    cli_source = (
        pathlib.Path(cli_module.__file__).read_text(encoding="utf-8")
    )
    # The hook-install branch, not the whole 7000-line CLI.
    start = cli_source.index("install_claude_code(dry_run=")
    branch = cli_source[start : start + 1800]
    handled = set(re.findall(r'action == "([a-z_]+)"', branch))

    assert returned <= handled, (
        f"actions with no CLI branch: {sorted(returned - handled)}; each would "
        "print 'Unknown action' to the user"
    )


def test_a_user_scope_hook_is_still_detected_first(scopes) -> None:
    """The pre-existing same-scope check keeps working."""
    home, _ = scopes
    (home / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "entroly ravs capture --stdin",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert install_claude_code()["action"] == "already_present"


def _registered_commands(home) -> list[str]:
    settings = json.loads(
        (home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    return [
        hook["command"]
        for entry in settings.get("hooks", {}).get("PostToolUse", [])
        for hook in entry["hooks"]
    ]


def test_reinstalling_without_entroly_on_path_keeps_one_registration(
    scopes, monkeypatch
) -> None:
    """The ``python -m entroly.cli`` fallback must count as already installed.

    With no ``entroly`` executable on PATH the installer writes
    ``<python> -m entroly.cli ravs capture``. The presence check looked for the
    literal ``entroly ravs capture``, so every re-run appended another copy and
    the host recorded each Bash call once per copy.
    """
    home, _ = scopes
    empty_bin = home / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    first = install_claude_code()
    second = install_claude_code()

    commands = _registered_commands(home)
    assert first["action"] == "installed"
    assert "-m entroly.cli ravs capture" in commands[0]
    assert second["action"] == "already_present"
    assert len(commands) == 1


def test_uninstall_removes_the_fallback_registration(scopes, monkeypatch) -> None:
    home, _ = scopes
    empty_bin = home / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    install_claude_code()

    assert hooks_module.uninstall_claude_code()["action"] == "uninstalled"
    assert _registered_commands(home) == []
