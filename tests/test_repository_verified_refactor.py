from __future__ import annotations

import copy
from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_refactor import (
    apply_verified_refactor_plan,
    apply_verified_rename_plan,
    build_verified_rename_plan,
    build_verified_safe_delete_plan,
    verify_refactor_plan_commitment,
)
from entroly.tree_sitter_support import extract_structural_spans


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "source.py", "def execute(value):\n    return value + 1\n")
    _write(
        root,
        "caller.py",
        "from source import execute\ndef run():\n    return execute(1)\n",
    )


def _plan(root: Path, new_name: str = "perform"):
    service = RepositoryIntelligenceService(root)
    summary = service.summary()
    index = service._index
    assert index is not None
    plan = build_verified_rename_plan(
        root,
        index,
        "execute",
        new_name,
        index_digest=str(summary["index_digest"]),
    )
    return index, str(summary["index_digest"]), plan


def test_preview_is_exact_tamper_evident_and_honest_about_completeness(tmp_path: Path) -> None:
    _project(tmp_path)
    _write(tmp_path, "dynamic.py", "def run_dynamic(obj):\n    return obj.execute()\n")
    index, _digest, plan = _plan(tmp_path)

    assert plan["resolution"] == "resolved"
    assert {(item["path"], item["kind"]) for item in plan["changes"]} == {
        ("source.py", "definition"),
        ("caller.py", "resolved-call"),
        ("caller.py", "python-import-binding"),
    }
    assert plan["risk"]["reference_completeness"] == "not-proven"
    assert plan["risk"]["unresolved_same_name_calls"] == 0
    assert plan["risk"]["unindexed_lexical_occurrences"] >= 1
    assert plan["receipt"]["writes_performed"] == 0
    assert verify_refactor_plan_commitment(plan)
    assert any(
        item["source_sha256"] == index.files["source.py"].sha256
        for item in plan["changes"]
    )

    tampered = copy.deepcopy(plan)
    tampered["changes"][0]["new_identifier"] = "forged"
    assert not verify_refactor_plan_commitment(tampered)


def test_ambiguous_symbol_produces_no_writable_plan(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def execute():\n    return 1\n")
    _write(tmp_path, "b.py", "def execute():\n    return 2\n")
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_rename_plan(
        tmp_path,
        service._index,
        "execute",
        "perform",
        index_digest=str(summary["index_digest"]),
    )
    assert plan["resolution"] == "ambiguous"
    assert len(plan["candidates"]) == 2
    assert plan["changes"] == []
    assert verify_refactor_plan_commitment(plan)


def test_apply_requires_plan_hash_and_explicit_incomplete_acknowledgement(tmp_path: Path) -> None:
    _project(tmp_path)
    index, digest, plan = _plan(tmp_path)
    plan_sha = plan["receipt"]["plan_sha256"]

    with pytest.raises(ValueError, match="explicit acknowledgement"):
        apply_verified_rename_plan(
            tmp_path, index, plan,
            index_digest=digest,
            expected_plan_sha256=plan_sha,
        )
    with pytest.raises(ValueError, match="does not match"):
        apply_verified_rename_plan(
            tmp_path, index, plan,
            index_digest=digest,
            expected_plan_sha256="0" * 64,
            acknowledge_incomplete=True,
        )
    assert "execute" in (tmp_path / "source.py").read_text(encoding="utf-8")


def test_apply_rechecks_preimages_and_performs_two_file_rename(tmp_path: Path) -> None:
    _project(tmp_path)
    index, digest, plan = _plan(tmp_path)
    result = apply_verified_rename_plan(
        tmp_path,
        index,
        plan,
        index_digest=digest,
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )

    assert result["change_count"] == 3
    assert result["file_count"] == 2
    assert result["rollback_performed"] is False
    assert "def perform(" in (tmp_path / "source.py").read_text(encoding="utf-8")
    assert "perform(1)" in (tmp_path / "caller.py").read_text(encoding="utf-8")
    rebuilt = build_repository_index(tmp_path)
    assert any(symbol.name == "perform" for symbol in rebuilt.symbols.values())


def test_apply_refuses_source_changed_after_preview(tmp_path: Path) -> None:
    _project(tmp_path)
    index, digest, plan = _plan(tmp_path)
    _write(tmp_path, "caller.py", "from source import execute\ndef run():\n    return execute(2)\n")

    with pytest.raises(ValueError, match="stale-index"):
        apply_verified_rename_plan(
            tmp_path,
            index,
            plan,
            index_digest=digest,
            expected_plan_sha256=plan["receipt"]["plan_sha256"],
            acknowledge_incomplete=True,
        )
    assert "def execute(" in (tmp_path / "source.py").read_text(encoding="utf-8")


def test_filesystem_failure_rolls_back_completed_replacements(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    index, digest, plan = _plan(tmp_path)
    originals = {path: (tmp_path / path).read_bytes() for path in ("caller.py", "source.py")}
    real_replace = Path.replace
    stages = 0

    def fail_second_stage(self: Path, target: Path):
        nonlocal stages
        if "entroly-stage" in self.name:
            stages += 1
            if stages == 2:
                raise OSError("injected replacement failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_second_stage)
    with pytest.raises(ValueError, match="rollback attempted"):
        apply_verified_rename_plan(
            tmp_path,
            index,
            plan,
            index_digest=digest,
            expected_plan_sha256=plan["receipt"]["plan_sha256"],
            acknowledge_incomplete=True,
        )
    assert {(path, (tmp_path / path).read_bytes()) for path in originals} == {
        (path, raw) for path, raw in originals.items()
    }


def test_verified_external_semantic_reference_augments_static_call_plan(tmp_path: Path) -> None:
    _project(tmp_path)
    _write(
        tmp_path,
        "callback.py",
        "from source import execute\ncallback = execute\n",
    )
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_rename_plan(
        tmp_path,
        service._index,
        "execute",
        "perform",
        index_digest=str(summary["index_digest"]),
        provider="test-lsp",
        semantic_relationships=[{
            "kind": "reference",
            "source": {
                "path": "callback.py", "line": 1,
                "start_character": 11, "end_character": 18,
            },
            "target": {
                "path": "source.py", "line": 0,
                "start_character": 4, "end_character": 11,
            },
        }],
    )
    assert any(
        item["path"] == "callback.py"
        and item["kind"] == "external-semantic-reference"
        for item in plan["changes"]
    )
    assert plan["risk"]["non_call_references_indexed"] is True
    assert plan["semantic_overlay_receipt"]["accepted_relationship_count"] == 1


def test_headless_safe_delete_blocks_known_references(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_safe_delete_plan(
        tmp_path,
        service._index,
        "execute",
        index_digest=str(summary["index_digest"]),
    )
    assert plan["operation"] == "safe-delete"
    assert plan["safe_to_apply"] is False
    assert plan["changes"] == []
    assert {item["kind"] for item in plan["blockers"]} >= {
        "resolved-call", "unclassified-lexical-reference"
    }
    assert verify_refactor_plan_commitment(plan)


def test_headless_safe_delete_is_two_phase_and_source_verified(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "module.py",
        "def keep():\n    return 1\n\n@staticmethod\ndef unused():\n    return 2\n",
    )
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_safe_delete_plan(
        tmp_path,
        service._index,
        "unused",
        index_digest=str(summary["index_digest"]),
    )
    assert plan["safe_to_apply"] is True
    assert plan["receipt"]["blocker_count_before_output_limit"] == 0
    assert "@staticmethod" in plan["changes"][0]["old_identifier"]
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        apply_verified_refactor_plan(
            tmp_path,
            service._index,
            plan,
            index_digest=str(summary["index_digest"]),
            expected_plan_sha256=plan["receipt"]["plan_sha256"],
        )
    applied = apply_verified_refactor_plan(
        tmp_path,
        service._index,
        plan,
        index_digest=str(summary["index_digest"]),
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )
    assert applied["operation"] == "safe-delete"
    assert applied["files"][0]["syntax_validation"] == "verified-python-ast"
    text = (tmp_path / "module.py").read_text(encoding="utf-8")
    assert "unused" not in text
    assert "@staticmethod" not in text
    assert "def keep" in text


@pytest.mark.parametrize(
    ("path", "source"),
    [
        ("src/lib.rs", "fn execute(x: i32) -> i32 { x + 1 }\nfn run() -> i32 { execute(1) }\n"),
        ("web/app.ts", "function execute(x: number) { return x + 1; }\nfunction run() { return execute(1); }\n"),
    ],
)
def test_parser_backed_non_python_rename_preserves_syntax(
    tmp_path: Path,
    path: str,
    source: str,
) -> None:
    if extract_structural_spans(source, path) is None:
        pytest.skip("optional local parser grammar unavailable")
    _write(tmp_path, path, source)
    index, digest, plan = _plan(tmp_path)
    result = apply_verified_rename_plan(
        tmp_path,
        index,
        plan,
        index_digest=digest,
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )
    assert result["files"][0]["syntax_validation"] == "verified-tree-sitter"
    assert "perform" in (tmp_path / path).read_text(encoding="utf-8")
