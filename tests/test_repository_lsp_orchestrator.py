from __future__ import annotations

import sys
from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.lsp_orchestrator import collect_lsp_references
from entroly.repository_intelligence.lsp_orchestrator import (
    verify_lsp_rename_preview_commitment,
)
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.semantic_overlay import (
    build_verified_semantic_overlay,
    verify_semantic_overlay_commitment,
)


FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_lsp_server.py"


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


def test_configured_lsp_process_is_bounded_and_ranges_are_source_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _project(tmp_path)
    monkeypatch.setenv("ENTROLY_TEST_SECRET", "must-not-reach-child")
    index = build_repository_index(tmp_path)
    result = collect_lsp_references(
        tmp_path,
        index,
        "execute",
        command=[sys.executable, str(FAKE_SERVER)],
        language_id="python",
        timeout_seconds=5,
    )

    assert result["schema_version"] == "entroly.lsp-orchestration.v1"
    assert result["process"]["exit_code"] == 0
    assert result["process"]["network_control"] == "not-enforced-external-process"
    assert result["process"]["ignored_messages"] == 0
    assert len(result["relationships"]) == 2
    assert result["omissions_by_reason"] == {
        "invalid-or-outside-workspace-location": 1,
    }

    overlay = build_verified_semantic_overlay(
        tmp_path,
        index,
        result["relationships"],
        index_digest="sha256:test",
        provider=result["provider"],
    )
    assert overlay["receipt"]["accepted_relationship_count"] == 2
    assert verify_semantic_overlay_commitment(overlay)


def test_lsp_timeout_kills_hung_process(tmp_path: Path) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)
    with pytest.raises(ValueError, match="timed out"):
        collect_lsp_references(
            tmp_path,
            index,
            "execute",
            command=[sys.executable, str(FAKE_SERVER), "--hang"],
            language_id="python",
            timeout_seconds=1,
        )


def test_lsp_output_and_message_bounds_fail_closed(tmp_path: Path) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)
    with pytest.raises(ValueError, match="output limit"):
        collect_lsp_references(
            tmp_path,
            index,
            "execute",
            command=[sys.executable, str(FAKE_SERVER), "--oversized-output"],
            language_id="python",
            timeout_seconds=5,
            max_output_bytes=1024,
        )
    with pytest.raises(ValueError, match="message limit"):
        collect_lsp_references(
            tmp_path,
            index,
            "execute",
            command=[sys.executable, str(FAKE_SERVER), "--many-messages"],
            language_id="python",
            timeout_seconds=5,
            max_messages=10,
        )


def test_lsp_deadline_also_bounds_blocked_protocol_writes(tmp_path: Path) -> None:
    _project(tmp_path)
    source = tmp_path / "source.py"
    source.write_text(
        source.read_text(encoding="utf-8") + ("# backpressure\n" * 70_000),
        encoding="utf-8",
    )
    index = build_repository_index(tmp_path)
    with pytest.raises(ValueError, match="timed out while writing"):
        collect_lsp_references(
            tmp_path,
            index,
            "execute",
            command=[
                sys.executable,
                str(FAKE_SERVER),
                "--stop-reading-after-init",
            ],
            language_id="python",
            timeout_seconds=1,
        )


def test_lsp_rejects_non_utf16_and_truncates_relationships_visibly(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)
    with pytest.raises(ValueError, match="UTF-16"):
        collect_lsp_references(
            tmp_path,
            index,
            "execute",
            command=[sys.executable, str(FAKE_SERVER), "--bad-encoding"],
            language_id="python",
            timeout_seconds=5,
        )
    result = collect_lsp_references(
        tmp_path,
        index,
        "execute",
        command=[sys.executable, str(FAKE_SERVER)],
        language_id="python",
        timeout_seconds=5,
        max_relationships=1,
    )
    assert len(result["relationships"]) == 1
    assert result["omissions_by_reason"] == {"relationship-limit": 1}


def test_lsp_command_is_argument_array_and_executable_must_exist(tmp_path: Path) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)
    with pytest.raises(ValueError, match="argument array"):
        collect_lsp_references(
            tmp_path, index, "execute",
            command="python fake.py",
            language_id="python",
        )
    with pytest.raises(ValueError, match="not found"):
        collect_lsp_references(
            tmp_path, index, "execute",
            command=["definitely-absent-entroly-lsp"],
            language_id="python",
        )


def test_service_builds_committed_lsp_augmented_rename_preview(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    payload = service.lsp_rename_preview(
        "execute",
        "perform",
        command=[sys.executable, str(FAKE_SERVER)],
        language_id="python",
        timeout_seconds=5,
    )
    assert payload["schema_version"] == "entroly.lsp-rename-preview.v1"
    assert payload["receipt"]["writes_performed"] == 0
    assert payload["plan"]["resolution"] == "resolved"
    assert payload["plan"]["risk"]["non_call_references_indexed"] is True
    assert verify_lsp_rename_preview_commitment(payload)
