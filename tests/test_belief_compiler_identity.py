from __future__ import annotations

from entroly.belief_compiler import (
    BeliefCompiler,
    CodeEntity,
    _module_entity,
)
from entroly.vault import VaultConfig, VaultManager


def test_module_identity_is_path_stable_and_cross_language_unique() -> None:
    assert _module_entity("auth.py", "auth") == "auth"
    assert _module_entity("ledger\\postings.py", "postings") == "ledger/postings"
    assert _module_entity("ledger/postings.ts", "postings") == "ledger/postings.ts"
    assert _module_entity("./ledger/postings.rs", "postings") == "ledger/postings.rs"

    python_entity = CodeEntity("save", "function", "ledger/postings.py")
    typescript_entity = CodeEntity("save", "function", "ledger/postings.ts")
    sibling_entity = CodeEntity("save", "function", "store/postings.py")

    assert {
        python_entity.qualified_name,
        typescript_entity.qualified_name,
        sibling_entity.qualified_name,
    } == {
        "ledger/postings::save",
        "ledger/postings.ts::save",
        "store/postings::save",
    }


def test_compile_directory_preserves_same_basename_modules(tmp_path) -> None:
    project = tmp_path / "project"
    (project / "ledger").mkdir(parents=True)
    (project / "store").mkdir(parents=True)
    (project / "ledger" / "postings.py").write_text(
        "def record_ledger():\n    return 'ledger'\n",
        encoding="utf-8",
    )
    (project / "store" / "postings.py").write_text(
        "def record_store():\n    return 'store'\n",
        encoding="utf-8",
    )
    (project / "store" / "postings.ts").write_text(
        "export function recordStoreTs() { return 'store-ts'; }\n",
        encoding="utf-8",
    )
    (project / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )

    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    result = BeliefCompiler(vault).compile_directory(str(project))

    assert result.errors == []
    assert result.files_processed == 4
    assert result.modules_mapped == 4
    assert result.beliefs_written == 5  # four modules plus architecture

    expected_entities = {
        "auth",
        "ledger/postings",
        "store/postings",
        "store/postings.ts",
    }
    stored_entities = {item["entity"] for item in vault.list_beliefs()}
    assert expected_entities <= stored_entities

    expected_bodies = {
        "ledger/postings": "record_ledger",
        "store/postings": "record_store",
        "store/postings.ts": "recordStoreTs",
    }
    for entity, symbol in expected_bodies.items():
        belief = vault.read_belief(entity)
        assert belief is not None
        assert symbol in belief["body"]

    architecture = vault.read_belief(f"architecture::{project.name}")
    assert architecture is not None
    for entity in expected_entities:
        assert f"**{entity}**" in architecture["body"]

    diagram = (vault.config.path / "media" / f"modules_{project.name}.md").read_text(
        encoding="utf-8"
    )
    for entity in expected_entities:
        assert f'["{entity}' in diagram
