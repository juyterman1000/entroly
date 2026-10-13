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

import pytest

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
