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

from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_nested_plugin_layout_uses_nearest_workspace_boundary(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    workspace = outer / ".cache" / "plugins" / "checkout"
    (workspace / ".git").mkdir(parents=True)
    plugin = workspace / ".agents" / "plugins" / "entroly"
    plugin.mkdir(parents=True)
    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    monkeypatch.chdir(plugin)

    assert Path(resolved_source_root()) == workspace.resolve()
    assert _project_path(str(workspace)) == workspace.resolve()
    with pytest.raises(ValueError):
        _project_path(str(outer))

    # An operator's explicit choice still overrides automatic discovery.
    monkeypatch.setenv("ENTROLY_SOURCE", str(outer))
    assert Path(resolved_source_root()) == outer.resolve()


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


@pytest.mark.parametrize("contraction,expanded", [
    ("won't", "will not"), ("can't", "cannot"), ("shan't", "shall not"),
    ("won’t", "will not"), ("can’t", "cannot"), ("isn't", "is not"),
    ("doesn't", "does not"), ("wouldn't", "would not"),
])
def test_contractions_preserve_polarity(contraction, expanded):
    negative = f"Agents {contraction} expose private keys."
    equivalent = f"Agents {expanded} expose private keys."
    positive = "Agents expose private keys."
    assert _has_negation_before(negative.lower(), "expose") is True
    assert _contradiction_score(negative, positive) == 1.0
    assert _contradiction_score(positive, negative) == 1.0
    assert _contradiction_score(negative, equivalent) == 0.0


def test_windows_pe_shim_embedded_shebang_resolves_package_dir(tmp_path, monkeypatch):
    # Exercise PE parsing on every platform, without depending on whichever
    # editable install the test interpreter finds outside this checkout.
    fake_exe = tmp_path / "Scripts" / "entroly.exe"
    fake_exe.parent.mkdir(parents=True)
    py_exe = tmp_path / "Python with spaces" / "python.exe"
    expected = tmp_path / "other-install" / "site-packages" / "entroly"
    fake_exe.write_bytes(b"MZ" + b"\x00" * 80_000 + f'#!"{py_exe}"\r\nPK\x03\x04'.encode())
    calls = []

    def probe(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] == [str(py_exe), "-c"]
        assert kwargs["cwd"] == str(fake_exe.parent)
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=str(expected) + "\n")

    monkeypatch.setattr(cli.subprocess, "run", probe)
    assert cli._launcher_package_dir(str(fake_exe)) == expected.resolve()
    assert len(calls) == 1


def test_binary_without_embedded_interpreter_is_not_executed(tmp_path, monkeypatch):
    launcher = tmp_path / "entroly.exe"
    launcher.write_bytes(b"MZ" + b"\x00" * 10_000)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected execution"))
    assert cli._launcher_package_dir(str(launcher)) is None
