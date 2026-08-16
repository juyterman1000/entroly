"""Guards for the traps that made the vault hard to use correctly.

Each of these cost real time before it was a test: a search answering from a
store the user never filled, a compile that had to be run per directory, and
beliefs whose provenance had to be guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from entroly.belief_compiler import BeliefCompiler
from entroly.context_receipts.ingest import _is_skipped_directory
from entroly.vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
    _parse_frontmatter,
    vault_readiness,
)


def test_empty_vault_with_a_document_index_names_the_right_command(tmp_path):
    """`ingest` and `search` use different stores and ingest says "success".

    Without this the two look interchangeable, and a search after an ingest
    answers from whatever the belief vault last held.
    """

    project = tmp_path / "p"
    (project / ".entroly" / "receipts").mkdir(parents=True)
    (project / ".entroly" / "receipts" / "index.json").write_text(
        json.dumps({"documents": [], "chunks": []}), encoding="utf-8"
    )
    vault_base = project / ".entroly" / "vault"
    VaultManager(VaultConfig(base_path=str(vault_base))).ensure_structure()

    report = vault_readiness(vault_base, project)

    assert report["ready"] is False
    assert report["has_document_index"] is True
    assert any("entroly compile" in reason for reason in report["reasons"])
    assert any("does not read" in reason for reason in report["reasons"])


def test_beliefs_older_than_source_are_reported_as_behind(tmp_path):
    project = tmp_path / "p"
    (project / "src").mkdir(parents=True)
    vault = VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    vault.write_belief(
        BeliefArtifact(entity="m", title="m", body="b", sources=["src/m.py:1"])
    )

    source = project / "src" / "m.py"
    source.write_text("x = 1\n", encoding="utf-8")
    import os
    import time

    os.utime(source, (time.time() + 3600, time.time() + 3600))

    report = vault_readiness(project / ".entroly" / "vault", project)

    assert report["ready"] is False
    assert any("source is newer" in reason for reason in report["reasons"])


def test_a_current_vault_reports_ready(tmp_path):
    project = tmp_path / "p"
    (project / "src").mkdir(parents=True)
    (project / "src" / "m.py").write_text("x = 1\n", encoding="utf-8")
    vault = VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    vault.write_belief(
        BeliefArtifact(entity="m", title="m", body="b", sources=["src/m.py:1"])
    )

    assert vault_readiness(project / ".entroly" / "vault", project)["ready"] is True


def test_source_root_is_backfilled_only_when_unambiguous(tmp_path):
    """A guess would look authoritative while pointing at the wrong file."""

    project = tmp_path / "p"
    (project / "scripts").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "scripts" / "solo.py").write_text("x = 1\n", encoding="utf-8")
    (project / "scripts" / "shared.py").write_text("x = 1\n", encoding="utf-8")
    (project / "tests" / "shared.py").write_text("x = 1\n", encoding="utf-8")

    vault = VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    vault.write_belief(
        BeliefArtifact(entity="solo", title="s", body="b", sources=["solo.py:1"])
    )
    vault.write_belief(
        BeliefArtifact(entity="shared", title="sh", body="b", sources=["shared.py:1"])
    )

    result = vault.backfill_source_roots([str(project)])

    assert result["backfilled_entities"] == ["solo"]
    assert result["ambiguous_entities"] == ["shared"]

    beliefs = project / ".entroly" / "vault" / "beliefs"
    solo = _parse_frontmatter((beliefs / "solo.md").read_text(encoding="utf-8")) or {}
    shared = _parse_frontmatter((beliefs / "shared.md").read_text(encoding="utf-8")) or {}
    assert solo["source_root"] == "scripts"
    assert "source_root" not in shared


def test_scratch_trees_are_pruned_by_both_walkers(tmp_path):
    """One unpruned `.tmp` made three separate commands impractical.

    It timed out the external-name scan, produced a 528 MB document index, and
    put 24,831 of 25,819 files in front of `entroly compile .`.
    """

    for name in (".tmp", "tmp", "node_modules", ".venv"):
        assert _is_skipped_directory(name), f"ingest must prune {name}"
        assert name in BeliefCompiler.SKIP_DIRS, f"compiler must prune {name}"


def test_compile_walks_a_whole_repository(tmp_path):
    """`entroly compile .` is recursive, so one command should cover the tree."""

    project = tmp_path / "p"
    for sub in ("pkg", "scripts", "nested/deep"):
        (project / sub).mkdir(parents=True)
        (project / sub / "mod.py").write_text(
            "def f():\n    \"\"\"D.\"\"\"\n    return 1\n", encoding="utf-8"
        )
    (project / ".tmp").mkdir()
    (project / ".tmp" / "scratch.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    compiler = BeliefCompiler(
        VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    )
    found = compiler._walk_source_files(Path(project), 0)
    names = {p.parent.name for p in found}

    assert {"pkg", "scripts", "deep"} <= names
    assert ".tmp" not in names
