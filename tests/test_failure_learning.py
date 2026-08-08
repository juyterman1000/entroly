from __future__ import annotations

import json

import pytest

from entroly.failure_learning import (
    LearningError,
    apply_learning_proposal,
    build_learning_proposal,
    write_learning_proposal,
)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_proposal_requires_observed_same_operation_failure_then_success(tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "tool_name": "shell_command",
                "command": "pytest tests/test_auth.py",
                "exit_code": 1,
                "stderr": "AssertionError: token=super-secret-value",
            },
            {"role": "assistant", "content": "I will inspect the fixture."},
            {
                "tool_name": "shell_command",
                "command": "pytest tests/test_auth.py -q",
                "exit_code": 0,
                "stdout": "4 passed",
            },
        ],
    )

    proposal = build_learning_proposal([transcript])

    assert proposal["mode"] == "dry_run_proposal_only"
    assert len(proposal["corrections"]) == 1
    correction = proposal["corrections"][0]
    assert correction["operation"] == "shell_command:pytest"
    assert correction["failure"]["line_number"] == 1
    assert correction["success"]["line_number"] == 3
    serialized = json.dumps(proposal)
    assert "super-secret-value" not in serialized
    assert "[REDACTED]" in serialized


def test_unresolved_failure_does_not_create_advice(tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_jsonl(
        transcript,
        [{"tool": "shell", "command": "ruff check .", "exit_code": 1, "stderr": "failed"}],
    )

    proposal = build_learning_proposal([transcript])

    assert proposal["corrections"] == []


def test_apply_reverifies_sources_backs_up_and_marks_target(tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_jsonl(
        transcript,
        [
            {"tool": "shell", "command": "pytest x.py", "exit_code": 1, "stderr": "failed"},
            {"tool": "shell", "command": "pytest x.py -q", "exit_code": 0, "stdout": "passed"},
        ],
    )
    proposal_path = write_learning_proposal(
        build_learning_proposal([transcript]), tmp_path / "proposal.json"
    )
    target = tmp_path / "AGENTS.md"
    target.write_text("# Instructions\n", encoding="utf-8")

    result = apply_learning_proposal(proposal_path, target)

    updated = target.read_text(encoding="utf-8")
    assert "## Entroly verified learnings" in updated
    assert "Evidence: failure" in updated
    assert result["backup"]
    assert (tmp_path / result["backup"].split("\\")[-1].split("/")[-1]).exists()


def test_apply_refuses_changed_transcript(tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_jsonl(
        transcript,
        [
            {"tool": "shell", "command": "pytest x.py", "exit_code": 1},
            {"tool": "shell", "command": "pytest x.py", "exit_code": 0},
        ],
    )
    proposal_path = write_learning_proposal(
        build_learning_proposal([transcript]), tmp_path / "proposal.json"
    )
    transcript.write_text("changed\n", encoding="utf-8")
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Instructions\n", encoding="utf-8")

    with pytest.raises(LearningError, match="changed after proposal"):
        apply_learning_proposal(proposal_path, target)


def test_secret_token_is_fully_redacted_and_files_are_not_cross_correlated(tmp_path):
    failed = tmp_path / "failed.jsonl"
    succeeded = tmp_path / "succeeded.jsonl"
    _write_jsonl(
        failed,
        [{"tool": "shell", "command": "pytest x.py", "exit_code": 1, "stderr": "sk-abcdefghijklmnopqrstuvwxyz"}],
    )
    _write_jsonl(
        succeeded,
        [{"tool": "shell", "command": "pytest x.py", "exit_code": 0, "stdout": "passed"}],
    )

    proposal = build_learning_proposal([failed, succeeded])

    serialized = json.dumps(proposal)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert proposal["corrections"] == []
