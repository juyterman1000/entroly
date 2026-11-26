"""Regression tests for three live defects surfaced during deep dogfooding:

1. Workspace plugin subdirectory source-root trap: MCP servers started with
   ``cwd`` inside ``<repo>/.agents/plugins/entroly`` (without ``ENTROLY_SOURCE``)
   previously indexed the 4-file plugin directory and rejected ``work_state``
   requests for ``<repo>``.
2. Prohibitive-verb false contradiction in ESG/EICV: ``"are blocked from entering"``
   against ``"can never enter"`` previously scored ``contradiction_fraction=1.0``
   (``decision="hallucinated"``), and words ending in ``"no"`` (``"piano "``)
   falsely triggered substring negation.
3. Windows pip ``.exe`` console-script wrapper identity in ``entroly doctor``:
   ``_launcher_package_dir`` previously read only text line-0 ``#!`` shebangs,
   returning ``None`` on Windows PE wrappers and firing a false shadowed-install
   warning even on healthy editable/venv installs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from entroly import cli
from entroly.eicv import EICVAnalyzer
from entroly.esg import _contradiction_score, _has_negation_before
from entroly.server_source_root import resolved_source_root
from entroly.work_graph_mcp import _project_path


def test_plugin_subdirectory_resolves_to_enclosing_workspace_root(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    (repo / ".git").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    plugin_dir = repo / ".agents" / "plugins" / "entroly"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    monkeypatch.chdir(plugin_dir)

    assert Path(resolved_source_root()) == repo.resolve()
    # work_graph_mcp path guard now accepts the enclosing workspace root.
    assert _project_path(str(repo)) == repo.resolve()


def test_unrelated_non_plugin_directory_is_not_walked_up(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    app_bundle = outer / "Contents" / "Resources"
    app_bundle.mkdir(parents=True)

    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    monkeypatch.chdir(app_bundle)

    assert Path(resolved_source_root()) == app_bundle.resolve()


def test_prohibitive_verb_matches_grammatical_negation_without_false_contradiction():
    evidence = (
        "External and untrusted observations can never enter or be promoted to "
        "permanent, non-decaying semantic memory during consolidation."
    )
    claim_synonym = (
        "Untrusted external observations are blocked from entering permanent "
        "semantic memory."
    )
    claim_opposite = (
        "Untrusted external observations are allowed to enter permanent "
        "semantic memory."
    )

    cert_same = EICVAnalyzer(profile="fact_check").verify(evidence, claim_synonym)
    assert cert_same.contradiction_fraction == 0.0
    assert cert_same.decision != "hallucinated"

    # Opposite polarity ("can enter" vs "blocked from entering") must still fire!
    assert _contradiction_score(
        "Untrusted observations can enter permanent semantic memory.",
        claim_synonym,
    ) == 1.0
    assert _contradiction_score(evidence, claim_opposite) == 1.0


def test_word_ending_in_no_does_not_trigger_substring_negation():
    assert _has_negation_before("the classical piano recital began", "recital") is False
    assert _has_negation_before("there was no recital today", "recital") is True


def test_windows_pe_shim_embedded_shebang_resolves_package_dir(tmp_path):
    # Simulate a 120 KB Windows distlib PE .exe wrapper embedding `#!"<python.exe>"`
    # past the initial 4 KB header, followed by zip payload bytes.
    fake_exe = tmp_path / "Scripts" / "entroly.exe"
    fake_exe.parent.mkdir(parents=True)
    py_exe = Path(sys.executable)
    # Only exercise execution when sys.executable ends with .exe or create a .exe symlink/shim check:
    if os.name == "nt":
        pe_stub = b"MZ" + (b"\x00" * 80_000)
        shebang = f'#!"{py_exe}"\r\nPK\x03\x04'.encode("utf-8")
        fake_exe.write_bytes(pe_stub + shebang)
        resolved = cli._launcher_package_dir(str(fake_exe))
        expected = Path(cli.__file__).resolve().parent
        assert resolved == expected
    else:
        # On POSIX, verify binary regex extraction doesn't crash on non-.exe binary stubs.
        fake_exe.write_bytes(b"MZ" + (b"\x00" * 10_000))
        assert cli._launcher_package_dir(str(fake_exe)) is None
