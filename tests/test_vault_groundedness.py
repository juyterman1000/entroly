"""Retraction of beliefs whose source file is gone (entroly/vault.py).

Compilation is additive: it writes a belief when it sees a file and never
revisits one whose file was deleted or moved, so the belief keeps its original
confidence forever and retrieval returns it beside live beliefs with nothing to
tell them apart.
"""

from __future__ import annotations

from entroly.vault import BeliefArtifact, VaultConfig, VaultManager


def _vault(tmp_path):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return project, VaultManager(
        VaultConfig(base_path=str(project / ".entroly" / "vault"))
    )


def _status_and_confidence(vault, entity: str) -> tuple[str, float]:
    record = vault.read_belief(entity)
    assert record is not None, f"belief {entity!r} disappeared"
    frontmatter = record["frontmatter"]
    return str(frontmatter.get("status", "")), float(frontmatter.get("confidence", -1.0))


def test_belief_with_a_missing_source_is_retracted(tmp_path):
    project, vault = _vault(tmp_path)
    (project / "src").mkdir()
    (project / "src" / "live.py").write_text("x = 1\n", encoding="utf-8")

    vault.write_belief(
        BeliefArtifact(entity="live", title="live", body="Real module.",
                       sources=["src/live.py:1"])
    )
    vault.write_belief(
        BeliefArtifact(entity="removed", title="removed", body="Deleted module.",
                       sources=["src/removed.py:1"])
    )

    result = vault.mark_beliefs_ungrounded([str(project)])

    assert result["retracted_entities"] == ["removed"]
    assert _status_and_confidence(vault, "removed") == ("ungrounded", 0.0)
    # The live belief keeps both its status and the confidence retrieval ranks on.
    status, confidence = _status_and_confidence(vault, "live")
    assert status != "ungrounded"
    assert confidence > 0.0


def test_source_relative_to_the_compiled_dir_is_not_retracted(tmp_path):
    """`entroly compile scripts` records `helper.py`, not `scripts/helper.py`.

    The belief never records which directory was compiled, so resolving a
    source against the project root alone marks live beliefs dead. Doing that
    retracted 275 of 715 real beliefs on the entroly repository.
    """

    project, vault = _vault(tmp_path)
    (project / "scripts").mkdir()
    (project / "scripts" / "helper.py").write_text("y = 2\n", encoding="utf-8")

    vault.write_belief(
        BeliefArtifact(entity="helper", title="helper", body="From scripts/.",
                       sources=["helper.py:1"])
    )

    result = vault.mark_beliefs_ungrounded([str(project)])

    assert result["retracted_entities"] == []


def test_belief_without_sources_is_never_retracted(tmp_path):
    """No sources means nothing to verify against, which is not evidence of death."""

    project, vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="sourceless", title="sourceless", body="No sources.",
                       sources=[])
    )

    result = vault.mark_beliefs_ungrounded([str(project)])

    assert result["retracted_entities"] == []


def test_retraction_is_idempotent(tmp_path):
    """A second pass reports the belief as already retracted rather than again."""

    project, vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="gone", title="gone", body="Deleted module.",
                       sources=["src/gone.py:1"])
    )

    first = vault.mark_beliefs_ungrounded([str(project)])
    second = vault.mark_beliefs_ungrounded([str(project)])

    assert first["retracted_entities"] == ["gone"]
    assert second["retracted_entities"] == []
    assert second["already_ungrounded"] == ["gone"]


def test_a_restored_file_lets_compilation_bring_the_belief_back(tmp_path):
    """Retraction marks, never deletes, so the belief stays auditable."""

    project, vault = _vault(tmp_path)
    vault.write_belief(
        BeliefArtifact(entity="returning", title="returning", body="Module.",
                       sources=["src/returning.py:1"])
    )
    vault.mark_beliefs_ungrounded([str(project)])
    assert _status_and_confidence(vault, "returning") == ("ungrounded", 0.0)

    (project / "src").mkdir()
    (project / "src" / "returning.py").write_text("z = 3\n", encoding="utf-8")
    vault.write_belief(
        BeliefArtifact(entity="returning", title="returning", body="Module.",
                       sources=["src/returning.py:1"])
    )

    status, confidence = _status_and_confidence(vault, "returning")
    assert status != "ungrounded"
    assert confidence > 0.0
