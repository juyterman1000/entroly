"""Kiro installer: a user-tuned Entroly hook is installed, and --force undoes.

Two rough edges in `entroly activation install|uninstall --host kiro`:

* After the user tuned Entroly's own hook (say, a longer timeout), running
  install again reported "conflict" and exited 2, as if a foreign file were
  in the way.
* `install --force` over the user's own file at that path moved it to a backup,
  and `uninstall` only disabled Entroly's hook: the user's file stayed in a
  backup Kiro never loads.
"""

from __future__ import annotations

import json
from pathlib import Path

from entroly.agent_activation import configure_kiro_hook


def _target(project: Path) -> Path:
    return project / ".kiro" / "hooks" / "entroly-activation.json"


def test_reinstalling_over_a_user_tuned_entroly_hook_keeps_it(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    configure_kiro_hook(project)
    target = _target(project)
    document = json.loads(target.read_text(encoding="utf-8"))
    document["hooks"][0]["timeout"] = 60
    target.write_text(json.dumps(document), encoding="utf-8")
    tuned = target.read_bytes()

    report = configure_kiro_hook(project)

    assert report["status"] == "already_installed"
    assert target.read_bytes() == tuned


def test_uninstall_after_force_gives_the_user_file_back(tmp_path) -> None:
    project = tmp_path / "project"
    target = _target(project)
    target.parent.mkdir(parents=True)
    original = b'{"version":"v1","hooks":[{"name":"customer-hook","trigger":"PromptSubmit"}]}'
    target.write_bytes(original)

    assert configure_kiro_hook(project, force=True)["status"] == "installed"
    report = configure_kiro_hook(project, uninstall=True)

    assert report["status"] == "disabled"
    assert target.read_bytes() == original
    assert Path(report["recoverable_at"]).is_file()
    assert not list(target.parent.glob("entroly-activation.json.entroly-backup-*"))
